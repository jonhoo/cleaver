use petgraph::graph::NodeIndex;

pub struct Modification;
pub enum Connect {
    Egress,
    Sharder,
}
pub struct StateDescription;
pub struct Tag;

pub enum Step<O> {
    Boot {
        index: super::DomainIndex,
        nodes: Vec<O>,
    },
    Append {
        domain: super::DomainIndex,
        node: O,
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

pub(crate) fn from_staged<O>(staged: super::Stage<O>) -> Vec<Step<O>> {
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
    unimplemented!()
}
