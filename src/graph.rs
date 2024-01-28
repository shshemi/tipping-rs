use std::{collections::BTreeSet, hash::Hash};

use itertools::Itertools;
use petgraph::algo::kosaraju_scc;
use petgraph::matrix_graph::MatrixGraph;

pub fn build_graph<T: Clone + Eq + Hash, Iter: Iterator<Item = T>>(
    token_iter: Iter,
    is_connected: impl Fn(&T, &T) -> bool,
) -> MatrixGraph<T, ()> {
    let tokens = token_iter.collect::<Vec<_>>();
    let mut graph = MatrixGraph::with_capacity(tokens.len());
    let nodes = tokens
        .iter()
        .unique()
        .cloned()
        .map(|tok| graph.add_node(tok))
        .collect::<Vec<_>>();
    nodes.iter().tuple_combinations().for_each(|(n1, n2)| {
        if is_connected(graph.node_weight(*n1), graph.node_weight(*n2)) {
            graph.add_edge(*n1, *n2, ());
        }

        if is_connected(graph.node_weight(*n2), graph.node_weight(*n1)) {
            graph.add_edge(*n2, *n1, ());
        }
    });
    graph
}
pub fn key_node_values<T: Clone + Eq + Hash + Ord>(
    g: MatrixGraph<T, ()>,
) -> BTreeSet<T> {
    let scc = kosaraju_scc(&g);
    let key_nodes = scc
        .iter()
        .enumerate()
        .max_by_key(|(_, cc)| cc.len())
        .map(|(lcc_idx, _)| {
            let temp_toks = scc[..=lcc_idx]
                .iter()
                .flat_map(|v| v.iter())
                .map(|n| g.node_weight(*n).clone())
                .collect::<BTreeSet<_>>();
            temp_toks
        })
        .unwrap_or_default();
    key_nodes
}

