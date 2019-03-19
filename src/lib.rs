use petgraph::graph::NodeIndex;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

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
    fn source_of(&self, _col: usize) -> impl Iterator<Item = (NodeIndex, Option<usize>)> {
        Vec::new().into_iter()
    }
    fn mirror(&self) -> Self {
        self.clone()
    }
    fn rewire(&mut self, _ancestor: NodeIndex, _to: NodeIndex) {}
    fn is_base(&self) -> bool {
        false
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

    // we want to preserve add order so that we can cheaply iterate in topological order
    added: HashSet<NodeIndex>,

    state: &'a mut State,
    assigned_domain: HashMap<NodeIndex, DomainIndex>,
    assigned_sharding: HashMap<NodeIndex, (NodeIndex, usize)>,
}

impl<'a> Migration<'a> {
    fn resolve_for_lookup(&self, mut ni: NodeIndex, mut column: usize) -> (NodeIndex, usize) {
        loop {
            // canonicalize by always choosing smaller node index
            match self.graph[ni]
                .source_of(column)
                .filter_map(|(pi, col)| col.map(move |col| (pi, col)))
                .min_by_key(|(pi, _)| *pi)
            {
                Some((pi, pc)) => {
                    ni = pi;
                    column = pc;
                }
                None if self.graph[ni].is_base() => {
                    return (ni, column);
                }
                None => unreachable!("looking up into column that does not exist"),
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
                    .insert(ni, self.resolve_for_lookup(neighbor, column));
            }

            // we'll also register a desire to have the lookup targets sharded by the key we look
            // up by.
            for (neighbor, column) in self.graph[ni].lookup_in() {
                if self.added.contains(&neighbor) && neighbor != ni {
                    let wants = self.resolve_for_lookup(neighbor, column);
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
                    .or_insert_with(Materialization::default)
                    .keys
                    .insert(column);
            }
        }

        // we may end up with nodes that are sharded, but also have multiple keys into their
        // materializations. this won't work -- we will have to keep a second, re-sharded copy of
        // the materialization for each other sharding.
        let mut resharded_copy = Vec::new();
        // TODO: only look at new?
        for (&ni, mat) in &mut materializations {
            if let Some(sharding) = self.assigned_sharding.get(&ni) {
                // self is sharded -- check for any incompatible indices
                mat.keys.retain(|&column| {
                    let want = self.resolve_for_lookup(ni, column);
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
        // it's time to insert the identity nodes.
        // one thing to keep in mind though is that we'll want to do topological iterations over
        // the new nodes further down. normally, we could just iterate over the new nodes in order
        // of their node id (since a child must be added after its parent, and would thus get a
        // larger id), but that becomes tricky when we inject nodes into the _middle_ of the graph.
        // to fix this, we construct a heap that initially holds every node index with a weight
        // equal to its node id, and whenever we add an identity node, we insert its node index
        // with a weight slightly larger than the node index of the node it is copying! that way,
        // topological traversal is simply a matter of traversing the heap in order by weight,
        // which heaps are great at.
        let mut topo = BinaryHeap::new();
        for &ni in &self.added {
            topo.push(VisitNode {
                logical: (ni, 0),
                index: ni,
            });
        }
        for (ni, column, want) in resharded_copy {
            // create a materialized identity node of ni that is sharded by the lookup key (column)
            // TODO: make sure to also unmark existing node as changed if applicable
            let mirror = self.graph[ni].mirror();
            let mni = self.graph.add_node(mirror);
            // NOTE: we don't need to remove from materializations[ni], b/c of retain above.
            materializations
                .entry(mni)
                .or_insert_with(Materialization::default)
                .keys
                .insert(column);
            let ei = self.graph.add_edge(ni, mni, Sharding::from(want));
            maybe_shuffle.push(ei);
            self.added.insert(mni);

            // also keep track of it in the heap so we'll iterate over it.
            // weight should be half-way between that of the node we're copying and the next node.
            // NOTE: in this particular case, we know that there are no other copies of ni below
            // mni, so it's fine to just use half the space!
            topo.push(VisitNode {
                logical: (ni, usize::max_value() >> 1),
                index: mni,
            });

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
                self.graph[child].rewire(ni, mni);
            }
            // TODO: maybe_shuffle is invalidated by the remove_edges above
        }

        // we now have all the materializations we want in place.
        // next step now is to figure out which of them we can make partial.
        // in the process, we also want to resolve all the sharding decisions that are in
        // "arbitrary sharding".

        // we have to walk the graph from the leaves and up, since we are not allowed to create
        // partial materializations "above" (closer to the base tables than) full materializations.
        // the heap we've maintained will give us reverse topological order (i.e., leaves-up),
        // which is exactly what we want.
        let mut candidates = VecDeque::new();
        while let Some(VisitNode {
            logical: _,
            index: ni,
        }) = topo.pop()
        {
            if let Some(materialization) = materializations.get_mut(&ni) {
                if let Some(is_full) = materialization.is_full {
                    // materialization state has already been determined!
                    // that can only be the case if we have already determined it needs to be full.
                    assert!(is_full);
                    continue;
                }

                // there needs to be a replay path to each of the materialization's columns. for
                // that to be the case for a column, the column needs to exist in at least one
                // upstream materialization (all through a union, one through a join). if that is
                // not the case, all upstream materializations need to be marked as full!
                for &column in &materialization.keys {
                    // we're going to keep track of all the upquery paths we're considering.
                    // each candidate is really a _set_ of paths, since a union can cause an
                    // upquery to have to branch through _multiple_ paths. if _any_ path in a
                    // candidate can't resolve the upquery column, that candidate is removed.
                    // if _all_ the paths in a candidate terminate in a materialization, then that
                    // candidate is viable. we continue resolving columns until all candidates are
                    // viable.
                    assert!(candidates.is_empty());
                    candidates.push_back(VecDeque::from(vec![vec![(ni, column)]]));
                    'candidate: while let Some(mut candidate) = candidates.pop_front() {
                        // check the next path in this candidate
                        let path = candidate.pop_front().expect("candidate had no paths?");

                        // we want to extend the path to get to a materialization, which we do
                        // by resolving the last (node, column) pair we got to.
                        let (lni, lcol) = path.last().cloned().unwrap();
                        if materializations.contains_key(&lni) {
                            // this path already terminates
                            // TODO: how do we detect termination?
                            // TODO: only stop at _full_ materializations to aid in adding indices?
                            candidate.push_back(path);
                            candidates.push_back(candidate);
                            continue;
                        } else {
                            let required_ancestors: Option<HashSet<NodeIndex>> = None;
                            if let Some(required_ancestors) = required_ancestors {
                                // lni is a join -- add all resolve paths _as_ candidates
                                // because the column can resolve in any of them.
                                for (pni, pcol) in self.graph[lni].source_of(lcol) {
                                    if !required_ancestors.contains(&pni) {
                                        continue;
                                    }

                                    if let Some(pcol) = pcol {
                                        let mut npath = path.clone();
                                        npath.push((pni, pcol));
                                        let mut ncandidate = candidate.clone();
                                        ncandidate.push_back(npath);
                                        candidates.push_back(ncandidate);
                                    }
                                }
                            } else {
                                // lni is a union -- add all resolve paths _to_ candidate
                                // because the column needs to resolve in all of them.
                                for (pni, pcol) in self.graph[lni].source_of(lcol) {
                                    if let Some(pcol) = pcol {
                                        let mut npath = path.clone();
                                        npath.push((pni, pcol));
                                        candidate.push_back(npath);
                                    } else {
                                        // we can't resolve the column any further in this
                                        // ancestor, so this candidate isn't viable.
                                        continue 'candidate;
                                    }
                                }
                                candidates.push_back(candidate);
                            }
                        }
                    }

                    // once the loop above has terminated, we should end up with a set of candidate
                    // upquery paths in `candidates`. if `candidates` is empty, then this
                    // materialization must be full, and so must any of its ancestors.
                    if candidates.is_empty() {
                        // mark as full, and mark all ancestors as full
                        // TODO
                    } else {
                        // we can pick any of the sets of paths in `candidates`.
                        // let's choose whichever one requires fewer upqueries.
                        // TODO
                        // TODO: also resolve arbitrary_sharding
                        // TODO: also add any additional indices we now need (recursively)
                        // TODO: can we detect join eviction cases here too?
                    }
                }
            }
        }

        let arbitrary_sharding = arbitrary_sharding;
        let maybe_shuffle = maybe_shuffle;
    }
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, PartialEq, Eq)]
struct VisitNode {
    logical: (NodeIndex, usize),
    index: NodeIndex,
}

#[derive(Default, Debug, Clone)]
struct Materialization {
    is_full: Option<bool>,
    keys: HashSet<usize>,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn visitnode_order() {
        let a = VisitNode {
            logical: (NodeIndex::from(1), 0),
            index: NodeIndex::from(1),
        };
        let b = VisitNode {
            logical: (NodeIndex::from(2), 0),
            index: NodeIndex::from(2),
        };
        // insert x between a and b
        let x = VisitNode {
            logical: (NodeIndex::from(1), usize::max_value() / 2),
            index: NodeIndex::from(3),
        };
        // insert y between a and x
        let y = VisitNode {
            logical: (NodeIndex::from(1), usize::max_value() / 2 / 2),
            index: NodeIndex::from(4),
        };
        // insert z between x and b
        let z = VisitNode {
            logical: (NodeIndex::from(1), 3 * (usize::max_value() / 2 / 2)),
            index: NodeIndex::from(5),
        };

        // we want to do a reverse topological search using a heap
        let mut heap = BinaryHeap::new();
        heap.push(a);
        heap.push(b);
        heap.push(x);
        heap.push(y);
        heap.push(z);
        let mut got = Vec::new();
        while let Some(o) = heap.pop() {
            got.push(o);
        }
        let topo = vec![b, z, x, y, a];
        assert_eq!(got, topo);
    }
}
