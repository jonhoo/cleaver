use petgraph::graph::NodeIndex;
use std::collections::HashMap;

pub struct Modification;
pub enum Connect {
    Egress,
    Sharder,
}
pub struct StateDescription;
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
        path: Vec<Vec<(NodeIndex, usize)>>,
    },
    FullReplay {
        tag: Tag,
        domain: super::DomainIndex,
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
    let mut changed_domains = HashMap::new();
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
    // TODO

    steps
}
