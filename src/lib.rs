use petgraph::graph::NodeIndex;
use std::collections::{HashMap, HashSet};

type DomainIndex = usize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum Sharding {
    None,
    By(NodeIndex, usize),
}

impl Default for Sharding {
    fn default() -> Self {
        Sharding::None
    }
}

impl From<(NodeIndex, usize)> for Sharding {
    fn from((ni, col): (NodeIndex, usize)) -> Self {
        Sharding::By(ni, col)
    }
}

#[derive(Clone, Debug, Default)]
struct Tmp;

impl Tmp {
    fn lookup_in(&self) -> impl Iterator<Item = (NodeIndex, usize)> {
        Vec::new().into_iter()
    }
    fn source_of(&self, _col: usize) -> impl Iterator<Item = (NodeIndex, usize)> {
        Vec::new().into_iter()
    }
    fn mirror(&self) -> Self {
        self.clone()
    }
}

type Node = Tmp;
type Edge = Sharding;

#[derive(Default, Debug)]
pub struct State {
    graph: petgraph::Graph<Node, Edge>,
    in_domain: HashMap<NodeIndex, DomainIndex>,
    sharding: HashMap<DomainIndex, Sharding>,
    assigned_domain: HashMap<NodeIndex, DomainIndex>,
    assigned_sharding: HashMap<NodeIndex, (NodeIndex, usize)>,
}

impl State {
    pub fn migrate(&mut self) -> Migration {
        Migration {
            graph: self.graph.clone(),
            added: Default::default(),
            assigned_domain: self.assigned_domain.clone(),
            assigned_sharding: self.assigned_sharding.clone(),
            state: self,
        }
    }
}

pub struct Migration<'a> {
    graph: petgraph::Graph<Node, Edge>,
    added: HashSet<NodeIndex>,
    state: &'a mut State,
    assigned_domain: HashMap<NodeIndex, DomainIndex>,
    assigned_sharding: HashMap<NodeIndex, (NodeIndex, usize)>,
}

impl<'a> Migration<'a> {
    fn resolve(&self, mut ni: NodeIndex, mut column: usize) -> (NodeIndex, usize) {
        loop {
            // canonicalize by always choosing smaller node index
            match self.graph[ni].source_of(column).min_by_key(|(pi, _)| *pi) {
                Some((pi, pc)) => {
                    ni = pi;
                    column = pc;
                }
                None => {
                    return (ni, column);
                }
            }
        }
    }

    pub fn commit(mut self) {
        // first, find all _required_ shardings
        let mut desired_sharding = HashMap::new();
        for &ni in &self.added {
            // if a node does lookups into a particular state, it must itself be sharded by that
            // state. note that we go towards the min same as with resolve. this is so that an
            // operator that does lookups into the output of a join by the join key is considered
            // sharded the same way as the join itself.
            if let Some((neighbor, column)) = self.graph[ni].lookup_in().min_by_key(|(i, _)| *i) {
                self.assigned_sharding
                    .insert(ni, self.resolve(neighbor, column));
            }

            // we'll also register a desire to have the lookup targets sharded by the key we look
            // up by.
            for (neighbor, column) in self.graph[ni].lookup_in() {
                if self.added.contains(&neighbor) && neighbor != ni {
                    let wants = self.resolve(neighbor, column);
                    desired_sharding
                        .entry(neighbor)
                        .or_insert_with(HashSet::new)
                        .insert(wants);

                    // remember that there's a sharding requirement along this edge
                    self.graph.update_edge(neighbor, ni, Sharding::from(wants));
                }
            }
        }

        // nodes that should be sharded, but we have to decide how
        let mut arbitrary_sharding = HashMap::new();

        // edges where we may have to inject a shuffle
        let mut maybe_shuffle = Vec::new();

        // next, we try to figure out how to shard nodes that have lookups into them.
        // we only assign shardings to the ones where there's no conflict for the time being.
        for (ni, shardings) in desired_sharding {
            use std::collections::hash_map::Entry;
            match self.assigned_sharding.entry(ni) {
                Entry::Occupied(s) => {
                    if shardings.len() == 1 && shardings.contains(s.get()) {
                        // no conflict here!
                    } else {
                        // lookup target is sharded one way, and at least one child requires a
                        // different sharding. we can pick any of them, and would have to shuffle
                        // for the others.
                        arbitrary_sharding.insert(ni, shardings);

                        // mark all outgoing edges as "may require resharding"
                        for child in self.graph.edges_directed(ni, petgraph::Direction::Outgoing) {
                            let (i, col) = *s.get();
                            match *child.weight() {
                                Sharding::By(ci, ccol) if ci == i && ccol == col => {}
                                _ => {
                                    use petgraph::visit::EdgeRef;
                                    maybe_shuffle.push(child.id());
                                }
                            }
                        }
                    }
                }
                Entry::Vacant(e) => {
                    if shardings.len() == 1 {
                        // no conflicting sharding, so we can just go ahead and shard
                        e.insert(shardings.into_iter().next().unwrap());
                    } else {
                        // multiple children who do lookups based on different columns
                        // we'll have to pick one, and then do shuffles for the others
                        arbitrary_sharding.insert(ni, shardings);

                        // mark all outgoing edges as "may require resharding"
                        for child in self.graph.edges_directed(ni, petgraph::Direction::Outgoing) {
                            if let Sharding::By(..) = *child.weight() {
                                use petgraph::visit::EdgeRef;
                                maybe_shuffle.push(child.id());
                            }
                        }
                    }
                }
            }
        }

        // at this point, we've taken note of all the uncontested sharding desires (though keep in
        // mind that we don't _have_ to shard any of them -- _not_ sharding is always an option).
        // all remaining sharding decisions are "arbitrary", in the sense that there are multiple
        // desired shardings, and we'll need to pick one. for any one we pick, there'll be at least
        // one shuffle. because of that, we now step into materializations, because decisions there
        // may affect how we want to shard those nodes.

        // first of all, we need to determine all the things that are going to be materialized,
        // and what they'll be keyed with. note that this may introduce materializations on
        // _existing_ nodes in the data-flow!
        let mut materializations = HashMap::new();
        for &ni in &self.added {
            // if a node does lookups into a particular state, there must be a materialization on
            // that state (TODO: relax this for query_through).
            for (neighbor, column) in self.graph[ni].lookup_in() {
                if !self.added.contains(&neighbor) {
                    // TODO: keep track of the fact that we need to add the appropriate index
                }

                materializations
                    .entry(neighbor)
                    .or_insert_with(HashSet::new)
                    .insert(column);
            }
        }

        // we may end up with nodes that are sharded, but also have multiple keys into their
        // materializations. this won't work -- we will have to keep a second, re-sharded copy of
        // the materialization for each other sharding.
        let mut resharded_copy = Vec::new();
        // TODO: only look at new?
        for (&ni, columns) in &mut materializations {
            if let Some(sharding) = self.assigned_sharding.get(&ni) {
                // self is sharded -- check for any incompatible indices
                columns.retain(|&column| {
                    let want = self.resolve(ni, column);
                    if *sharding != want {
                        // we'll need re-sharded copy of this state
                        resharded_copy.push((ni, column, want));
                        false
                    } else {
                        true
                    }
                });
            }
        }
        for (ni, column, want) in resharded_copy {
            // create a materialized identity node of ni that is sharded by the lookup key (column)
            // TODO: make sure to also unmark existing node as changed if applicable
            let mirror = self.graph[ni].mirror();
            let mni = self.graph.add_node(mirror);
            materializations
                .entry(mni)
                .or_insert_with(HashSet::new)
                .insert(column);
            let ei = self.graph.add_edge(ni, mni, Sharding::from(want));
            maybe_shuffle.push(ei);

            // rewire any outgoing edges from ni that required sharding by column
            // so that they instead link to the re-sharded mirror
            let mut rewire = Vec::new();
            for child in self.graph.edges_directed(ni, petgraph::Direction::Outgoing) {
                if let Sharding::By(root_ni, root_col) = child.weight() {
                    if (*root_ni, *root_col) == want {
                        use petgraph::visit::EdgeRef;
                        rewire.push(child.target());
                    }
                }
            }
            for child in rewire {
                let ei = self.graph.find_edge(ni, child).unwrap();
                let e = self.graph.remove_edge(ei).unwrap();
                self.graph.add_edge(mni, child, e);
            }
            // TODO: maybe_shuffle is invalidated by the remove_edges above
        }

        // we now have all the materializations we want in place.
        // next step now is to figure out which of them we can make partial.
        // in the process, we also want to resolve all the sharding decisions that are in
        // "arbitrary sharding".
        //
        // TODO: we probably have to re-do the work of splitting materializations once all
        // arbitrary shardings have been resolved :/ can we structure this in a better way?

        // TODO

        let arbitrary_sharding = arbitrary_sharding;
        let maybe_shuffle = maybe_shuffle;
    }
}
