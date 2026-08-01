//! Edge aggregation for the semantic-LOD cut.
//!
//! Collapsing a subtree hides its nodes, and the spring pass skips any edge
//! with a hidden endpoint. Left there, that silently *deletes* a collapsed
//! module's cross-cutting dependency attractions instead of transferring them
//! to the proxy standing in for it — a collapsed module exerts no pull on the
//! modules it imports, so the collapsed layout differs structurally from the
//! expanded one.
//!
//! This module transfers them. One pass over the edge array maps each endpoint
//! to its *lowest visible ancestor* through the containment `parent` column and
//! sorts every edge into one of three outcomes:
//!
//! - **internal** — both endpoints resolve to the same ancestor. The edge is
//!   entirely inside one collapsed subtree and has nothing left to pull on.
//! - **live** — both endpoints are themselves visible. The edge is simulated
//!   and drawn exactly as it always was, named by its own index.
//! - **crossing** — the endpoints resolve to two different ancestors, at least
//!   one of which is a proxy. These are deduplicated into weighted bundles.
//!
//! # No source mutation
//!
//! The source edge arrays are read, never written. Re-targeting edges in place
//! would mean rebuilding the CSR on every band transition and keeping enough
//! bookkeeping to undo it on expand; an aggregated list next to the originals
//! is reversed by discarding it.
//!
//! # Bundle identity
//!
//! A bundle is keyed on the *unordered* endpoint pair: `A→B` and `B→A` land in
//! the same bundle. Spring attraction is symmetric — the source gets `+F` and
//! the target `−F` — so two bundles differing only in direction would be one
//! force split in two, and would draw as two overlapping lines.

use super::bubble::HIERARCHY_ROOT;

/// Ancestor value for a slot with no visible ancestor at all.
///
/// Reachable when a cut leaves a whole tree of the containment forest hidden.
/// Edges touching such a slot are dropped: there is nothing on screen for them
/// to pull on.
pub const NO_VISIBLE_ANCESTOR: u32 = u32::MAX;

/// The aggregated edge set for one cut.
///
/// `live_edges` is ascending by edge index and the bundle columns are ascending
/// by `(source, target)`, so the whole result is a pure function of the inputs —
/// the same cut aggregates to the same bytes, which is what lets a caller skip
/// an upload by comparing.
pub struct EdgeAggregation {
    /// Indices of the source edges whose endpoints are both visible.
    pub live_edges: Vec<u32>,
    /// Lower endpoint of each bundle.
    pub bundle_sources: Vec<u32>,
    /// Upper endpoint of each bundle; always greater than the lower one.
    pub bundle_targets: Vec<u32>,
    /// How many source edges each bundle stands for. Never zero.
    pub bundle_weights: Vec<u32>,
}

/// Resolve every slot to the lowest ancestor of itself that is visible.
///
/// A visible slot resolves to itself; a hidden one to the nearest visible slot
/// on its root path, or [`NO_VISIBLE_ANCESTOR`] when its whole tree is hidden.
///
/// One walk per unresolved chain with the whole chain memoised on the way back
/// down, so the total work is O(node_count) whatever the tree depth. Iterative:
/// a recursive walk overflows the stack on a deep chain and a stack overflow in
/// WASM is an uncatchable trap.
///
/// `node_count` is `min(parent.len(), visible.len())`; slots beyond it, and
/// parents pointing past it, are treated as roots.
pub fn lowest_visible_ancestors(parent: &[u32], visible: &[u8]) -> Vec<u32> {
    let node_count = parent.len().min(visible.len());
    let mut ancestor = vec![NO_VISIBLE_ANCESTOR; node_count];
    let mut resolved = vec![false; node_count];

    for slot in 0..node_count {
        if visible[slot] != 0 {
            ancestor[slot] = slot as u32;
            resolved[slot] = true;
        }
    }

    let mut chain: Vec<u32> = Vec::new();
    for slot in 0..node_count {
        if resolved[slot] {
            continue;
        }
        let mut cursor = slot as u32;
        let answer = loop {
            let index = cursor as usize;
            if resolved[index] {
                break ancestor[index];
            }
            chain.push(cursor);
            let up = parent[index];
            if up == HIERARCHY_ROOT || up as usize >= node_count {
                break NO_VISIBLE_ANCESTOR;
            }
            // A validated containment forest carries no cycles, so a chain can
            // never exceed the slot count. The bound makes a corrupt column
            // terminate instead of hang.
            if chain.len() > node_count {
                break NO_VISIBLE_ANCESTOR;
            }
            cursor = up;
        };
        for &link in chain.iter() {
            ancestor[link as usize] = answer;
            resolved[link as usize] = true;
        }
        chain.clear();
    }

    ancestor
}

/// Sort `(a, b)` pairs ascending by `a` then `b`, in place.
///
/// Two stable counting-sort passes over the dense slot space — O(pairs +
/// node_count) rather than the O(pairs log pairs) of a comparison sort, which
/// is what keeps a 253 000-edge transition inside the frame budget.
fn sort_pairs(pair_a: &mut [u32], pair_b: &mut [u32], node_count: usize) {
    let count = pair_a.len();
    if count < 2 {
        return;
    }

    let mut counts = vec![0u32; node_count + 1];
    let mut out_a = vec![0u32; count];
    let mut out_b = vec![0u32; count];

    // Least-significant key first: sorting by `b` and then stably by `a`
    // leaves the pairs ordered by `(a, b)`.
    for pass in 0..2 {
        let key: &[u32] = if pass == 0 { pair_b } else { pair_a };
        counts.fill(0);
        for &k in key.iter() {
            counts[k as usize + 1] += 1;
        }
        for i in 0..node_count {
            counts[i + 1] += counts[i];
        }
        for i in 0..count {
            let slot = &mut counts[key[i] as usize];
            out_a[*slot as usize] = pair_a[i];
            out_b[*slot as usize] = pair_b[i];
            *slot += 1;
        }
        pair_a.copy_from_slice(&out_a);
        pair_b.copy_from_slice(&out_b);
    }
}

/// Aggregate the edge set against a visible cut.
///
/// # Arguments
///
/// * `edge_sources`, `edge_targets` - The source edge arrays, read only. Pairs
///   past the shorter of the two are ignored.
/// * `parent` - Containment parent per slot, [`HIERARCHY_ROOT`] for a root.
/// * `visible` - `1` for a slot in the cut, `0` for one the cut hides. Must
///   describe the same slot space as `parent`; the shorter of the two bounds
///   the slot space and endpoints outside it are dropped.
///
/// Every source edge lands in exactly one of: dropped (internal, or touching a
/// slot with no visible ancestor), `live_edges`, or one bundle's weight — so
/// the bundle weights sum to the number of edges crossing a collapse boundary.
pub fn aggregate_edges(
    edge_sources: &[u32],
    edge_targets: &[u32],
    parent: &[u32],
    visible: &[u8],
) -> EdgeAggregation {
    let node_count = parent.len().min(visible.len());
    let edge_count = edge_sources.len().min(edge_targets.len());
    let ancestor = lowest_visible_ancestors(parent, visible);

    let mut live_edges: Vec<u32> = Vec::new();
    let mut pair_a: Vec<u32> = Vec::new();
    let mut pair_b: Vec<u32> = Vec::new();

    for edge in 0..edge_count {
        let source = edge_sources[edge] as usize;
        let target = edge_targets[edge] as usize;
        if source >= node_count || target >= node_count {
            continue;
        }
        if visible[source] != 0 && visible[target] != 0 {
            live_edges.push(edge as u32);
            continue;
        }
        let a = ancestor[source];
        let b = ancestor[target];
        if a == NO_VISIBLE_ANCESTOR || b == NO_VISIBLE_ANCESTOR || a == b {
            continue;
        }
        pair_a.push(a.min(b));
        pair_b.push(a.max(b));
    }

    sort_pairs(&mut pair_a, &mut pair_b, node_count);

    let mut bundle_sources: Vec<u32> = Vec::new();
    let mut bundle_targets: Vec<u32> = Vec::new();
    let mut bundle_weights: Vec<u32> = Vec::new();
    for i in 0..pair_a.len() {
        if let Some(&last_a) = bundle_sources.last()
            && last_a == pair_a[i]
            && *bundle_targets.last().unwrap() == pair_b[i]
        {
            *bundle_weights.last_mut().unwrap() += 1;
            continue;
        }
        bundle_sources.push(pair_a[i]);
        bundle_targets.push(pair_b[i]);
        bundle_weights.push(1);
    }

    EdgeAggregation {
        live_edges,
        bundle_sources,
        bundle_targets,
        bundle_weights,
    }
}

/// Number of `u32`s [`aggregate_edge_data`] puts before the first list.
pub const EDGE_AGGREGATION_HEADER: usize = 2;

/// `u32`s per bundle in the [`aggregate_edge_data`] encoding.
pub const EDGE_BUNDLE_STRIDE: usize = 3;

/// Flat `u32` encoding of [`aggregate_edges`] for the JS boundary.
///
/// `[live_count, bundle_count, live_edges…, (source, target, weight)…]` — one
/// allocation and one wasm-bindgen copy for the whole result.
///
/// The bundles are interleaved rather than kept as columns because that is the
/// layout the spring shader reads: the caller uploads the tail of this array to
/// the GPU as it stands, with no repacking pass on the transition path.
pub fn aggregate_edge_data(
    edge_sources: &[u32],
    edge_targets: &[u32],
    parent: &[u32],
    visible: &[u8],
) -> Vec<u32> {
    let aggregation = aggregate_edges(edge_sources, edge_targets, parent, visible);
    let live_count = aggregation.live_edges.len();
    let bundle_count = aggregation.bundle_sources.len();

    let mut result = Vec::with_capacity(
        EDGE_AGGREGATION_HEADER + live_count + bundle_count * EDGE_BUNDLE_STRIDE,
    );
    result.push(live_count as u32);
    result.push(bundle_count as u32);
    result.extend_from_slice(&aggregation.live_edges);
    for k in 0..bundle_count {
        result.push(aggregation.bundle_sources[k]);
        result.push(aggregation.bundle_targets[k]);
        result.push(aggregation.bundle_weights[k]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::bubble::{BubbleConfig, compute_bubble_hierarchy};

    /// A three-level tree: root 0, modules 1 and 2, leaves 3..=6 under 1 and
    /// 7..=10 under 2.
    fn module_tree() -> Vec<u32> {
        let mut edges = vec![0, 1, 0, 2];
        for leaf in 3..=6u32 {
            edges.extend_from_slice(&[1, leaf]);
        }
        for leaf in 7..=10u32 {
            edges.extend_from_slice(&[2, leaf]);
        }
        edges
    }

    fn module_parents() -> Vec<u32> {
        compute_bubble_hierarchy(&module_tree(), 11, None, &BubbleConfig::default()).parent
    }

    /// Cut with modules 1 and 2 collapsed: root and both modules on screen,
    /// every leaf hidden.
    fn collapsed_modules() -> Vec<u8> {
        let mut visible = vec![0u8; 11];
        visible[0] = 1;
        visible[1] = 1;
        visible[2] = 1;
        visible
    }

    #[test]
    fn ancestors_resolve_to_the_lowest_visible_slot() {
        let parent = module_parents();
        let ancestor = lowest_visible_ancestors(&parent, &collapsed_modules());

        assert_eq!(ancestor[0], 0);
        assert_eq!(ancestor[1], 1);
        assert_eq!(ancestor[2], 2);
        assert_eq!(&ancestor[3..=6], &[1, 1, 1, 1], "module 1's leaves");
        assert_eq!(&ancestor[7..=10], &[2, 2, 2, 2], "module 2's leaves");
    }

    #[test]
    fn a_wholly_hidden_tree_has_no_visible_ancestor() {
        // 0 -> 1 -> 2, nothing visible.
        let parent = compute_bubble_hierarchy(&[0, 1, 1, 2], 3, None, &BubbleConfig::default())
            .parent;
        let ancestor = lowest_visible_ancestors(&parent, &[0, 0, 0]);
        assert_eq!(ancestor, vec![NO_VISIBLE_ANCESTOR; 3]);
    }

    #[test]
    fn deep_chains_resolve_without_recursion() {
        // A 100 000-link chain: only the root is visible.
        let depth = 100_000u32;
        let mut edges = Vec::new();
        for i in 0..depth {
            edges.extend_from_slice(&[i, i + 1]);
        }
        let count = depth as usize + 1;
        let parent =
            compute_bubble_hierarchy(&edges, count, None, &BubbleConfig::default()).parent;
        let mut visible = vec![0u8; count];
        visible[0] = 1;

        let ancestor = lowest_visible_ancestors(&parent, &visible);
        assert_eq!(ancestor, vec![0u32; count]);
    }

    #[test]
    fn internal_edges_produce_no_bundle() {
        let parent = module_parents();
        let visible = collapsed_modules();
        // Two leaves of module 1 referencing each other, and one leaf
        // referencing its own collapsed parent.
        let sources = [3u32, 4, 5];
        let targets = [4u32, 6, 1];

        let aggregation = aggregate_edges(&sources, &targets, &parent, &visible);
        assert!(aggregation.bundle_sources.is_empty());
        assert!(aggregation.live_edges.is_empty());
    }

    #[test]
    fn crossing_edges_bundle_onto_the_visible_ancestors() {
        let parent = module_parents();
        let visible = collapsed_modules();
        // Five imports from module 1's leaves into module 2's, one of them
        // written backwards, plus one leaf-to-visible-root edge.
        let sources = [3u32, 4, 5, 6, 8, 3];
        let targets = [7u32, 8, 9, 10, 3, 0];

        let aggregation = aggregate_edges(&sources, &targets, &parent, &visible);

        assert_eq!(aggregation.bundle_sources, vec![0, 1]);
        assert_eq!(aggregation.bundle_targets, vec![1, 2]);
        assert_eq!(aggregation.bundle_weights, vec![1, 5]);
        assert!(aggregation.live_edges.is_empty());

        // Every bundle endpoint is the lowest visible ancestor of an original
        // endpoint, and every bundle endpoint is itself visible.
        let ancestor = lowest_visible_ancestors(&parent, &visible);
        for k in 0..aggregation.bundle_sources.len() {
            let a = aggregation.bundle_sources[k];
            let b = aggregation.bundle_targets[k];
            assert_eq!(visible[a as usize], 1);
            assert_eq!(visible[b as usize], 1);
            assert!(a < b, "bundle endpoints are ordered");
            let found = sources.iter().zip(targets.iter()).any(|(&s, &t)| {
                let ea = ancestor[s as usize];
                let eb = ancestor[t as usize];
                (ea.min(eb), ea.max(eb)) == (a, b)
            });
            assert!(found, "bundle ({a}, {b}) matches no source edge");
        }
    }

    #[test]
    fn bundle_weights_sum_to_the_crossing_edge_count() {
        let parent = module_parents();
        let visible = collapsed_modules();
        let sources = [3u32, 4, 5, 6, 8, 3, 4, 0];
        let targets = [7u32, 8, 9, 10, 3, 4, 5, 1];

        let aggregation = aggregate_edges(&sources, &targets, &parent, &visible);
        let ancestor = lowest_visible_ancestors(&parent, &visible);

        let mut crossing = 0u32;
        let mut live = 0u32;
        for e in 0..sources.len() {
            let source = sources[e] as usize;
            let target = targets[e] as usize;
            if visible[source] == 1 && visible[target] == 1 {
                live += 1;
                continue;
            }
            let a = ancestor[source];
            let b = ancestor[target];
            if a != b && a != NO_VISIBLE_ANCESTOR && b != NO_VISIBLE_ANCESTOR {
                crossing += 1;
            }
        }

        assert_eq!(aggregation.bundle_weights.iter().sum::<u32>(), crossing);
        assert_eq!(aggregation.live_edges.len() as u32, live);
        assert!(crossing > 0 && live > 0, "the fixture must exercise both");
    }

    #[test]
    fn expanding_reverses_to_the_original_edge_set() {
        let parent = module_parents();
        let sources = [3u32, 4, 5, 6, 8];
        let targets = [7u32, 8, 9, 10, 3];
        let sources_before = sources;
        let targets_before = targets;

        let collapsed = aggregate_edges(&sources, &targets, &parent, &collapsed_modules());
        assert!(!collapsed.bundle_sources.is_empty());

        let expanded = aggregate_edges(&sources, &targets, &parent, &[1u8; 11]);
        assert!(expanded.bundle_sources.is_empty(), "nothing is hidden, nothing bundles");
        assert_eq!(expanded.live_edges, vec![0, 1, 2, 3, 4]);

        // The source arrays are inputs, never a workspace: aggregation is
        // undone by discarding its output.
        assert_eq!(sources, sources_before);
        assert_eq!(targets, targets_before);
    }

    #[test]
    fn self_loops_and_out_of_range_endpoints_are_dropped() {
        let parent = module_parents();
        let visible = collapsed_modules();
        let sources = [3u32, 99, 3];
        let targets = [3u32, 4, 99];

        let aggregation = aggregate_edges(&sources, &targets, &parent, &visible);
        assert!(aggregation.bundle_sources.is_empty());
        assert!(aggregation.live_edges.is_empty());
    }

    #[test]
    fn nested_collapses_fold_into_the_outer_proxy() {
        // Root 0 visible, module 1 collapsed, sub-module 3 inside it also
        // marked collapsed: everything under 1 must resolve to 1.
        let parent = module_parents();
        let mut visible = vec![0u8; 11];
        visible[0] = 1;
        visible[2] = 1;
        let sources = [4u32, 5];
        let targets = [7u32, 8];

        let aggregation = aggregate_edges(&sources, &targets, &parent, &visible);
        assert_eq!(aggregation.bundle_sources, vec![0]);
        assert_eq!(aggregation.bundle_targets, vec![2]);
        assert_eq!(aggregation.bundle_weights, vec![2]);
    }

    #[test]
    fn bundles_come_out_sorted_and_deduplicated_at_scale() {
        // 64 modules under a root, 16 leaves each, with a pseudo-random import
        // graph between leaves of different modules.
        let modules = 64u32;
        let leaves = 16u32;
        let mut edges = Vec::new();
        for m in 0..modules {
            edges.extend_from_slice(&[0, m + 1]);
            for l in 0..leaves {
                edges.extend_from_slice(&[m + 1, modules + 1 + m * leaves + l]);
            }
        }
        let node_count = (modules + 1 + modules * leaves) as usize;
        let parent =
            compute_bubble_hierarchy(&edges, node_count, None, &BubbleConfig::default()).parent;

        let mut visible = vec![0u8; node_count];
        visible[0] = 1;
        for m in 0..modules {
            visible[(m + 1) as usize] = 1;
        }

        let mut sources = Vec::new();
        let mut targets = Vec::new();
        let mut state = 12345u32;
        for _ in 0..20_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let a = modules + 1 + (state >> 8) % (modules * leaves);
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let b = modules + 1 + (state >> 8) % (modules * leaves);
            sources.push(a);
            targets.push(b);
        }

        let aggregation = aggregate_edges(&sources, &targets, &parent, &visible);

        // Strictly ascending by (source, target) — so deduplication is total.
        for k in 1..aggregation.bundle_sources.len() {
            let previous = (aggregation.bundle_sources[k - 1], aggregation.bundle_targets[k - 1]);
            let current = (aggregation.bundle_sources[k], aggregation.bundle_targets[k]);
            assert!(previous < current, "bundles out of order at {k}");
        }

        let ancestor = lowest_visible_ancestors(&parent, &visible);
        let mut crossing = 0u32;
        for e in 0..sources.len() {
            let a = ancestor[sources[e] as usize];
            let b = ancestor[targets[e] as usize];
            if a != b {
                crossing += 1;
            }
        }
        assert_eq!(aggregation.bundle_weights.iter().sum::<u32>(), crossing);
        assert!(aggregation.bundle_sources.len() < crossing as usize, "bundling must compress");
    }

    #[test]
    fn flat_encoding_carries_every_column() {
        let parent = module_parents();
        let mut visible = collapsed_modules();
        visible[3] = 1;
        let sources = [3u32, 4, 5];
        let targets = [1u32, 8, 9];

        let aggregation = aggregate_edges(&sources, &targets, &parent, &visible);
        let data = aggregate_edge_data(&sources, &targets, &parent, &visible);

        let live_count = data[0] as usize;
        let bundle_count = data[1] as usize;
        assert_eq!(live_count, aggregation.live_edges.len());
        assert_eq!(bundle_count, aggregation.bundle_sources.len());
        assert!(live_count > 0 && bundle_count > 0, "the fixture must exercise both");
        assert_eq!(
            data.len(),
            EDGE_AGGREGATION_HEADER + live_count + bundle_count * EDGE_BUNDLE_STRIDE
        );

        let base = EDGE_AGGREGATION_HEADER;
        assert_eq!(&data[base..base + live_count], &aggregation.live_edges[..]);
        let bundles = &data[base + live_count..];
        for k in 0..bundle_count {
            let triple = &bundles[k * EDGE_BUNDLE_STRIDE..(k + 1) * EDGE_BUNDLE_STRIDE];
            assert_eq!(triple[0], aggregation.bundle_sources[k]);
            assert_eq!(triple[1], aggregation.bundle_targets[k]);
            assert_eq!(triple[2], aggregation.bundle_weights[k]);
        }
    }

    #[test]
    fn an_empty_slot_space_aggregates_to_nothing() {
        let data = aggregate_edge_data(&[], &[], &[], &[]);
        assert_eq!(data, vec![0, 0]);
    }
}
