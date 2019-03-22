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
    fn egress(&self, _shuffle_to: Sharding) -> Self {
        self.clone()
    }
    fn rewire(&mut self, _ancestor: NodeIndex, _to: NodeIndex) {}
    fn is_base(&self) -> bool {
        false
    }
    fn good_new_domain_boundary_above(&self) -> bool {
        // true for aggregations and readers
        false
    }
    fn is_sharder(&self) -> bool {
        false
    }
    fn is_mirror(&self) -> bool {
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
    ndomains: usize,
}

impl State {
    pub fn migrate(&mut self) -> Migration {
        Migration {
            graph: self.graph.clone(),
            added: Default::default(),
            assigned_domain: self.assigned_domain.clone(),
            assigned_sharding: self.assigned_sharding.clone(),
            ndomains: self.ndomains,
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
    ndomains: usize,
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

        // edges where we have to inject a shuffle
        let mut shuffles = Vec::new();

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
                        // different sharding, so we need to shuffle.
                        for child in self.graph.edges_directed(ni, petgraph::Direction::Outgoing) {
                            let (i, col) = *s.get();
                            match *child.weight() {
                                Sharding::By(ci, ccol) if ci == i && ccol == col => {}
                                _ => {
                                    use petgraph::visit::EdgeRef;
                                    shuffles.push((ni, child.target()));
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
                        // multiple children who do lookups based on different columns.
                        // we'll have to pick one, and then do shuffles for the others.
                        // TODO: choose more intelligently?
                        let sharding = shardings
                            .into_iter()
                            .min()
                            .expect("no shardings needed, why is the entry there?");
                        e.insert(sharding);

                        for child in self.graph.edges_directed(ni, petgraph::Direction::Outgoing) {
                            let (i, col) = sharding;
                            match *child.weight() {
                                Sharding::By(ci, ccol) if ci == i && ccol == col => {}
                                _ => {
                                    use petgraph::visit::EdgeRef;
                                    shuffles.push((ni, child.target()));
                                }
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
            self.graph.add_edge(ni, mni, Sharding::from(want));
            shuffles.push((ni, mni));
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
        }

        // we now have all the materializations we want in place.
        // next step now is to figure out which of them we can make partial.
        let shuffles = shuffles; // no more shuffles can be added

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
            if let Some(mut materialization) = materializations.remove(&ni) {
                if let Some(ref plan) = materialization.plan {
                    // materialization state has already been determined!
                    // that can only be the case if we have already determined it needs to be full.
                    assert!(plan.is_full());
                    materializations.insert(ni, materialization);
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

                        // we want to extend the path to get to a full materialization, which we do
                        // by continuously resolving the last (node, column) pair we got to.
                        let (lni, lcol) = path.last().cloned().unwrap();
                        if let Some(mat) = materializations.get(&lni) {
                            if let Some(true) = mat.is_full() {
                                // this path is already complete
                                // TODO: how do we detect termination?
                                candidate.push_back(path);
                                candidates.push_back(candidate);
                                continue;
                            }
                        }

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

                    // once the loop above has terminated, we should end up with a set of candidate
                    // upquery paths in `candidates`. if `candidates` is empty, then this
                    // materialization must be full, and so must any of its ancestors.
                    if candidates.is_empty() {
                        // mark as full, and mark all ancestor materializations as full
                        let mut visit = vec![ni];
                        while let Some(ni) = visit.pop() {
                            if let Some(mat) = materializations.get_mut(&ni) {
                                if !self.added.contains(&ni)
                                    && !mat
                                        .is_full()
                                        .expect("existing materialization without a plan")
                                {
                                    unimplemented!(
                                        "forced to turn existing partial materialization full"
                                    );
                                }

                                mat.plan = Some(MaterializationPlan::Full);
                            }

                            visit.extend(
                                self.graph
                                    .neighbors_directed(ni, petgraph::Direction::Incoming),
                            );
                        }

                        assert_eq!(materialization.plan, None);
                        materialization.plan = Some(MaterializationPlan::Full);
                    } else {
                        // we can pick any of the sets of paths in `candidates`.
                        let mut chosen = None;
                        for candidate in candidates.drain(..) {
                            // how expensive is this candidate?
                            // compute based on
                            //
                            //   a) # shard crossings (fan-out is expensive)
                            //   b) # of paths (more upqueries)
                            //   c) # of materialization on paths (more recursive upqueries)
                            //
                            // TODO
                            chosen = Some(candidate);
                        }

                        assert_eq!(materialization.plan, None);
                        materialization.plan = Some(MaterializationPlan::Partial {
                            paths: chosen.expect("!candidates.is_empty()"),
                        });

                        // TODO: when we announce this plan, we have to announce it in segments!
                    }
                }
                materializations.insert(ni, materialization);
            }
        }

        // for each new partial materialization, we need to recursively add any new indices that
        // path requires. as part of that, we should also truncate replay paths to the nearest full
        // materialization. while the candidate-finding code does do that already, _new_
        // materializations along a chosen replay path may also have been made full _after_ the
        // path was chosen.
        for &ni in &self.added {
            if let Some(Materialization {
                plan: Some(MaterializationPlan::Partial { mut paths }),
                keys,
            }) = materializations.remove(&ni)
            {
                for path in &mut paths {
                    // each path _starts_ with the materialization at ni, and ends in _some_ full
                    // materialization. in other words, it starts towards the leaves of the graph,
                    // and ends closer to the root.
                    //
                    // we _first_ want to trim the suffix from the first full materialization:
                    let first_full = path
                        .iter()
                        .position(|(ni, _)| {
                            if let Some(Materialization {
                                plan: Some(MaterializationPlan::Full),
                                ..
                            }) = materializations.get(&ni)
                            {
                                true
                            } else {
                                false
                            }
                        })
                        .expect("no full materialization on chosen replay path");
                    path.truncate(first_full + 1);

                    // next, we have to add any indices mandated by the new replay path
                    for &mut (ni, col) in path {
                        if let Some(mat) = materializations.get_mut(&ni) {
                            if mat.keys.insert(col) && self.assigned_sharding.contains_key(&ni) {
                                // I _think_ this is the join eviction case?
                                // TODO
                            }
                        }
                    }
                }
                materializations.insert(
                    ni,
                    Materialization {
                        plan: Some(MaterializationPlan::Partial { paths }),
                        keys,
                    },
                );
            }
        }

        // we still have some operators that we haven't decided on the sharding of. these are
        // likely operators like filters, projections, and the like, which don't care how they're
        // sharded. we still have to assign them a sharding.
        let mut unassigned: Vec<_> = self
            .added
            .iter()
            .filter(|ni| self.assigned_sharding.contains_key(ni))
            .cloned()
            .collect();
        while !unassigned.is_empty() {
            let n = unassigned.len();
            unassigned.retain(|&ni| {
                // let's, for the time being, simply assign the sharding of a random parent.
                let psharding = self
                    .graph
                    .neighbors_directed(ni, petgraph::Direction::Incoming)
                    .filter_map(|pni| self.assigned_sharding.get(&pni))
                    .cloned()
                    .next();
                if let Some(psharding) = psharding {
                    self.assigned_sharding.insert(ni, psharding);
                    false
                } else {
                    true
                }
            });

            assert_ne!(unassigned.len(), n, "made no progress on sharding");
        }

        // insert sharding egress nodes as appropriate at each shuffle.
        //
        // TODO: maybe this is where we decide _not_ to shard?
        // NOTE: if we un-shard, we can also suddenly keep more than one key in one mat.
        // NOTE: in theory we could de-duplicate sharder nodes if there were multiple children
        // that required the same sharding by placing an identity in the target sharding.
        for (src, dst) in shuffles {
            let ei = self.graph.find_edge(src, dst).unwrap();
            let src_sharding = Sharding::from(self.assigned_sharding[&src]);
            let dst_sharding = self.graph[ei];
            assert_ne!(src_sharding, dst_sharding);

            // add the sharder egress node
            let shuffle = self.graph[src].egress(dst_sharding);
            let sni = self.graph.add_node(shuffle);
            self.graph.remove_edge(ei);
            self.graph.add_edge(src, sni, src_sharding);
            self.graph.add_edge(sni, dst, dst_sharding);
            self.added.insert(sni);
        }

        // assign domains to all new nodes.
        //
        // there are only a few _requirements_ for domain assignment:
        //
        //  - nodes that are sharded differently must be in different domains
        //  - joins must have their inputs materialized in their own domain
        //
        // we can also observe that it is relatively cheap to add a domain boundary above
        // readers and aggregations, so we'd like to do that too.
        //
        // the way we're going to go about this is to first give every new node its own new domain.
        // then we're going to repeatedly iterate over the nodes and adopt the domain of one of
        // their ancestors unless doing so would violate one of the rules above. we'll also favor
        // adopting existing domains rather than creating new ones.
        let mut next = self.ndomains;
        for &ni in &self.added {
            self.assigned_domain.insert(ni, next);
            next += 1;
        }
        let mut modified = true;
        while modified {
            for &ni in &self.added {
                for nbri in self.graph.neighbors_undirected(ni) {
                    // check whether this is a viable domain for this node:
                    // don't merge domains across shuffles
                    if self.assigned_sharding.get(&ni) != self.assigned_sharding.get(&nbri) {
                        continue;
                    }
                    // don't merge domains good domain boundaries
                    if self.graph.find_edge(ni, nbri).is_some() {
                        // neighbor is a child
                        if self.graph[nbri].good_new_domain_boundary_above() {
                            continue;
                        }
                    } else {
                        // neighbor is a parent
                        if self.graph[ni].good_new_domain_boundary_above() {
                            continue;
                        }
                    }

                    // this is a candidate!
                    // prefer it to our current assignment if it is smaller than ours.
                    // that will prefer existing domains over new ones, and will also minimize the
                    // number of new domains.
                    let ours = self.assigned_domain[&ni];
                    let theirs = self.assigned_domain[&nbri];
                    if theirs < ours {
                        self.assigned_domain.insert(ni, theirs);
                        modified = true;
                    }
                }
            }
        }
        // let's avoid sparse domain assignments:
        let mut consec = HashMap::new();
        let mut next = self.ndomains;
        for &ni in &self.added {
            let d = self.assigned_domain[&ni];
            if d >= self.ndomains {
                consec.entry(d).or_insert_with(|| {
                    next += 1;
                    next - 1
                });
            }
        }
        for &ni in &self.added {
            let d = self.assigned_domain[&ni];
            if d >= self.ndomains {
                self.assigned_domain.insert(ni, consec[&d]);
            }
        }
        // add ingress/egress nodes where necessary
        let mut extra = Vec::new();
        let mut disconnected = Vec::new();
        'merge: for &ni in &self.added {
            let d = self.assigned_domain[&ni];
            if d < self.ndomains {
                // added to existing domain, so no need for egress/ingress
                continue;
            }
            for pi in self
                .graph
                .neighbors_directed(ni, petgraph::Direction::Incoming)
            {
                let pd = self.assigned_domain[&pi];
                if d == pd {
                    // parent is in same domain
                    continue;
                }

                if self.assigned_sharding.get(&ni) != self.assigned_sharding.get(&pi) {
                    // there's a shuffle above us, so there's a sharder egress there
                    assert!(self.graph[pi].is_sharder());

                    // if we originally had `a -> b` and `a -> c`, and b and c are in the same
                    // domain and sharded differently from a, then they should share a sharder.
                    let mut ppis = self
                        .graph
                        .neighbors_directed(pi, petgraph::Direction::Incoming);
                    let ppi = ppis.next().expect("sharder had no parent");
                    assert_eq!(ppis.count(), 0, "sharder had more than one parent");
                    for ppci in self
                        .graph
                        .neighbors_directed(ppi, petgraph::Direction::Outgoing)
                    {
                        if !self.graph[ppci].is_sharder() || ppci == pi {
                            // not a sharder we might merge with pi
                            continue;
                        }

                        // this is a sharder; check if it also goes to ni's domain
                        // remember that a sharder can only ever have _one_ child
                        let mut ppccis = self
                            .graph
                            .neighbors_directed(ppci, petgraph::Direction::Outgoing);
                        let ppcci = ppccis.next().expect("sharder had no children");
                        assert_eq!(ppccis.count(), 0, "sharder had more than one child");
                        if self.assigned_domain[&ppcci] == d {
                            // that sharder _does_ also lead to this domain!
                            // let's merge them by having a single ingress in this domain.
                            // so, we have this:
                            //
                            //              [ ppi ]
                            //                 |
                            //           +-----+------+
                            //           v            v
                            //         { pi }      { ppci }
                            //           v            v
                            //         [ ni ]     [ ppcci ]
                            //
                            // where pi and ppci are both {sharders}. we want to end up with:
                            //
                            //              [ ppi ]
                            //                 v
                            //               { ? }
                            //                 v
                            //               ( ? )
                            //                 |
                            //           +-----+------+
                            //           v            v
                            //         [ ni ]       [ ? ]
                            //
                            // where ( ? ) is an ingress node (a mirror of ppi). we don't really
                            // care which of { pi } and { ppci } to use as the final sharded. We do
                            // have an interesting case when ppcci is already an ingress node
                            // though, since it may already have other merged children, and we may
                            // as well re-use that ingress.
                            //
                            // so, we want to end up with one of these two rewirings:
                            //
                            //          [ ppi ]                     [ ppi ]
                            //             |                           |
                            //             +------+                    +------+
                            //                    v                           v
                            //     ( pi ) <--- { ppci }        { pi }      { ppci }
                            //       |                                        |
                            //       +------------+                           |
                            //       |            |                           |
                            //       v            v                           v
                            //     [ ni ]     [ ppcci ]        [ ni ] <-- ( ppcci )
                            //
                            // first, remove our connection to our grandparent.
                            // TODO: in theory, ni could already be a state copy mirror...
                            let ei = self.graph.find_edge(ppi, pi).unwrap();
                            let e = self.graph.remove_edge(ei).unwrap();
                            // TODO: call node.rewire

                            // next, figure out what we have to do to get the ingress in place.
                            if self.graph[ppcci].is_mirror() {
                                // we already have an ingress. let's just re-use it!
                                // this is just a matter of replacing (pi, ni) with (ppcci, ni).
                                //
                                // this is the _right_ diagram from above.
                                let ei = self.graph.find_edge(pi, ni).unwrap();
                                let e = self.graph.remove_edge(ei).unwrap();
                                self.graph.add_edge(ppci, ni, e);
                                // TODO: call node.rewire

                                // stash away pi in case we need some new nodes later
                                disconnected.push(pi);
                            } else {
                                // we have to make an ingress node to merge these. specifically, we
                                // turn _our_ sharder into an ingress, and keep the _other_ sharder
                                // as a sharder, and then just do re-wiring. we do this to avoid
                                // removing nodes (which would lead to unstable node indices).
                                //
                                // this is the _left_ diagram from above.

                                // so, first we change "our" sharder to an ingress
                                let ingress = self.graph[pi].mirror();
                                *self.graph.node_weight_mut(pi).unwrap() = ingress;

                                // we make it a child of the "other" sharder.
                                // sharding is the same as the connection from sharder to ni.
                                let e = self.graph[self.graph.find_edge(pi, ni).unwrap()];
                                self.graph.add_edge(ppci, pi, e);
                                // TODO: call node.rewire

                                // then, we move the other child to be under this ingress
                                let ei = self.graph.find_edge(ppci, ppcci).unwrap();
                                let e = self.graph.remove_edge(ei).unwrap();
                                self.graph.add_edge(pi, ppcci, e);
                                // TODO: call node.rewire
                            }

                            // it's okay to break here since we can never merge with _more_ than
                            // one sharder (one of them should already have merged with the other).
                            continue 'merge;
                        }
                    }
                }

                // we're sharded the same as this ancestor, but are in a different domain.
                // egress/ingress time!
                // TODO
            }
        }
        self.added.extend(extra);
        if !disconnected.is_empty() {
            unimplemented!("gc disconnected nodes");
        }
        self.ndomains += next;

        // TODO: fix up replay paths now that we have sharder/egress/ingress nodes.
    }
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, PartialEq, Eq)]
struct VisitNode {
    logical: (NodeIndex, usize),
    index: NodeIndex,
}

#[derive(Default, Debug, Clone, Eq, PartialEq)]
struct Materialization {
    keys: HashSet<usize>,
    plan: Option<MaterializationPlan>,
}

impl Materialization {
    fn is_full(&self) -> Option<bool> {
        self.plan.as_ref().map(|p| p.is_full())
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum MaterializationPlan {
    Full,
    Partial {
        paths: VecDeque<Vec<(NodeIndex, usize)>>,
    },
}

impl MaterializationPlan {
    fn is_full(&self) -> bool {
        if let MaterializationPlan::Full = *self {
            true
        } else {
            false
        }
    }
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
