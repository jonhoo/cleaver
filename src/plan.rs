use petgraph::graph::NodeIndex;
use std::collections::HashMap;

pub struct Modification;
pub enum Connect {
    Egress,
    Sharder,
}
pub struct StateDescription;
#[derive(Clone, Copy)]
pub struct Tag;

pub enum Step {
    Boot {
        index: super::DomainIndex,
        nodes: Vec<NodeIndex>,
    },
    Append {
        domain: super::DomainIndex,
        node: NodeIndex,
    },
    Modify {
        domain: super::DomainIndex,
        node: NodeIndex,
        m: Modification,
    },
    Connect {
        domain: super::DomainIndex,
        node: NodeIndex,
        c: Connect,
    },
    PrepareState {
        domain: super::DomainIndex,
        node: NodeIndex,
        s: StateDescription,
    },
    RegisterPath {
        tag: Tag,
        domain: super::DomainIndex,
        path: Vec<(NodeIndex, usize)>,
    },
    FullReplay {
        tag: Tag,
        domain: super::DomainIndex,
    },
    Ready {
        domain: super::DomainIndex,
        node: NodeIndex,
    },
}

pub(crate) fn from_staged<O: super::DataflowOperator>(staged: &super::Stage<O>) -> Vec<Step> {
    // we now have to construct a plan for getting from the original Dataflow to the Dataflow
    // we just designed. while we _could_ keep track of the changes we make as we make them
    // above, that'd make the above code much trickier to read. instead, we'll just construct
    // the delta here, even if that is a bit more costly.
    //
    // the ultimate plan has several stages:
    //
    //  1. boot all new domains with their nodes
    //  2. for each new _node_ send that node to its domain (inc. anc. information)
    //  3. tell all old base nodes about column changes
    //    3.1. also inform downstream ingress nodes about added columns
    //  4. for each new ingress, find parent egress/sharder, and tell it about new tx
    //    NOTE: egress nodes should only be told about corresponding shard!
    //  5. for each new materialization (could be on existing node), in topo order:
    //    5.1. send "preparestate"
    //    5.2. set up replay path(s)
    //    5.3. (if full) initiate and wait for replay.
    let mut steps = Vec::new();

    // Step 1: boot new domains
    let mut new_domains = HashMap::new();
    for &ni in &staged.added {
        if staged.assigned_domain[&ni] < staged.ndomains - staged.new_domains {
            continue;
        }

        // TODO: assign local index
        new_domains
            .entry(staged.assigned_domain[&ni])
            .or_insert_with(Vec::new)
            .push(ni);
    }
    for (di, nodes) in new_domains {
        // TODO: nodes should be in topo order
        steps.push(Step::Boot { index: di, nodes });
    }

    // Step 2: augment existing domains
    // TODO: topological order
    for &ni in &staged.added {
        let di = staged.assigned_domain[&ni];
        if di >= staged.ndomains - staged.new_domains {
            continue;
        }

        // TODO: assign local index
        steps.push(Step::Append {
            domain: di,
            node: ni,
        });
    }

    // Step 3: base table column changes
    // TODO

    // Step 4: connect egress/sharder to ingress nodes
    for &ni in &staged.added {
        if !staged.graph[ni].is_ingress() {
            continue;
        }

        for pi in staged
            .graph
            .neighbors_directed(ni, petgraph::Direction::Incoming)
        {
            if staged.graph[pi].is_egress() {
                steps.push(Step::Connect {
                    domain: staged.assigned_domain[&pi],
                    node: ni,
                    c: Connect::Egress,
                });
            } else if staged.graph[pi].is_sharder() {
                steps.push(Step::Connect {
                    domain: staged.assigned_domain[&pi],
                    node: ni,
                    c: Connect::Sharder,
                });
            }
        }
    }

    // Step 5: set up all materializations
    // TODO: in topo order
    for &ni in staged.added.union(&staged.touched) {
        if let Some(m) = staged.materializations.get(&ni) {
            let new = staged.added.contains(&ni);
            if !new {
                // TODO: how do we know whether a materialization was added, or just a key?
            } else {
                steps.push(Step::PrepareState {
                    domain: staged.assigned_domain[&ni],
                    node: ni,
                    s: StateDescription,
                });

                match m
                    .plan
                    .as_ref()
                    .expect("no materialization plan established")
                {
                    super::MaterializationPlan::Full => {
                        // TODO: do replay
                        // TODO: how do we know/inform about paths?
                    }
                    super::MaterializationPlan::Partial { ref paths } => {
                        // every path here is an upquery path
                        for path in paths {
                            let tag = Tag; // TODO

                            // we need to tell each domain on the path about its segment of the
                            // upquery path.
                            // TODO: should we just tell each domain about the full path?
                            // TODO: does it also need to know about the node _after_
                            //       egress/sharder so it doesn't send to all children?
                            let mut segment = Vec::new();
                            for (i, &(ni, col)) in path.iter().enumerate() {
                                segment.push((ni, col));

                                let d = staged.assigned_domain[&ni];
                                if i == path.len() - 1
                                    || d != staged.assigned_domain[&path[i + 1].0]
                                {
                                    // TODO: do we need to reverse the path here? probably.
                                    steps.push(Step::RegisterPath {
                                        tag,
                                        domain: d,
                                        path: segment.split_off(0),
                                    });
                                }
                            }
                            assert!(segment.is_empty());
                        }
                    }
                }
            }
        }
    }

    // mark nodes as ready
    // TODO: also mark other materialization?
    // TODO: what about updates between full replay and ready?
    for &ni in &staged.added {
        steps.push(Step::Ready {
            domain: staged.assigned_domain[&ni],
            node: ni,
        });
    }

    steps
}
