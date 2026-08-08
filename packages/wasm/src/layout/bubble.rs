//! Containment hierarchy derivation for nested bubble layout and semantic LOD.
//!
//! Derives four per-slot columns from the graph's containment (parent→child)
//! edges:
//!
//! - **parent**: the slot's containment parent, or [`HIERARCHY_ROOT`] for a
//!   root. This is what makes "walk up to the lowest visible ancestor"
//!   possible and what makes a *forest* representable.
//! - **depth**: distance from the slot's own root (roots are depth 0).
//! - **wellRadius** (bubble size): bottom-up from the subtree — leaves get a
//!   base radius, internal nodes get whichever is larger of the summed-area
//!   estimate `sqrt(sum_child_areas / (pi * density(k)))` and the two widest
//!   children `r1 + r2`, plus padding. The radius has to be one the children
//!   can actually fit inside; see [`ENCLOSING_RATIO`] for why a single packing
//!   constant is not.
//! - **subtreeSize**: node count of the subtree rooted at the slot, itself
//!   included. Drives collapse priority and edge-bundle weight.
//!
//! # Forest, not tree
//!
//! Every slot in `0..node_count` belongs to exactly one tree. A slot with no
//! incoming containment edge is the root of its own tree; a repo-root plus a
//! handful of orphan config files is exactly that shape, and an implementation
//! that walks only the largest component leaves the orphans invisible to any
//! tree-walk consumer. Slots reachable only through a containment cycle are
//! rooted at their lowest slot index, so the output is a spanning forest of the
//! whole dense slot space under every input.
//!
//! # Consumers
//!
//! The columns feed the Relativity Atlas bubble mode on the GPU (depth-decaying
//! gravity, wellRadius phantom zones, scaled orbit springs) and the CPU-side
//! `HierarchyColumns` the LOD cut walks. [`compute_bubble_hierarchy`] carries no
//! `wasm_bindgen` in its signature, so a native indexer can precompute the same
//! columns offline and ship them with the graph.

/// Parent value marking a forest root: the slot has no containment parent.
pub const HIERARCHY_ROOT: u32 = u32::MAX;

/// Configuration for bubble radius computation.
pub struct BubbleConfig {
    /// Base radius for leaf nodes (default: 10.0).
    pub base_radius: f32,
    /// Padding added to internal node radii (default: 5.0).
    pub padding: f32,
    /// Packing efficiency for wide fan-out, where the boundary loss has washed
    /// out and one constant is a fair estimate (default: 0.82).
    ///
    /// Only consulted above [`ENCLOSING_RATIO`]'s last entry. Small fan-out is
    /// the regime where a single asymptotic constant is not merely imprecise
    /// but *unsatisfiable*, so it is read from the table instead — see
    /// [`packing_density`].
    pub packing_efficiency: f32,
}

/// Radius of the smallest circle enclosing `k` disjoint unit circles, indexed
/// by `k`. Entry 0 is unused.
///
/// This is the classical "circles in a circle" packing problem. Entries 1..=9
/// are exact closed forms evaluated to `f32`; 10..=12 are the published
/// numerical optima, which have no closed form.
///
/// The table exists because the density it implies — `k / ratio^2` — is *not*
/// monotonic and does not resemble its own asymptote. It runs 1.0, 0.50, 0.65,
/// 0.69, 0.69, 0.67, 0.78, 0.73, 0.69, 0.69, 0.71, 0.74 against a large-`k`
/// limit of ~0.9. Two children pack at density 0.5, not 0.82: a parent sized
/// from the asymptote is 28% too small to hold them, and no arrangement of the
/// subtree can fix that, because the radius itself is the thing that is wrong.
/// A code tree branches two and three ways constantly, so this is the common
/// case rather than a corner of it.
const ENCLOSING_RATIO: [f32; 13] = [
    0.0,         // unused
    1.0,         // k=1
    2.0,         // k=2: side by side
    2.154_700_5, // k=3: 1 + 2/sqrt(3)
    2.414_213_5, // k=4: 1 + sqrt(2)
    2.701_302,   // k=5: 1 + sqrt(2 + 2/sqrt(5))
    3.0,         // k=6: ring of 6, no centre
    3.0,         // k=7: centre + ring of 6 (hexagonal)
    3.304_765_6, // k=8: 1 + 1/sin(pi/7)
    3.613_125_9, // k=9: 1 + sqrt(2 * (2 + sqrt(2)))
    3.813_026_2, // k=10
    3.923_804_4, // k=11
    4.029_309_3, // k=12
];

/// Achievable packing density for `k` circles: `k / ENCLOSING_RATIO[k]^2` where
/// the table reaches, and `fallback` beyond it.
///
/// Returning a density *below* what is achievable only wastes space. Returning
/// one above it is unsatisfiable — the caller sizes a circle that its contents
/// cannot fit inside — so the table is the conservative direction everywhere it
/// applies.
fn packing_density(k: u32, fallback: f32) -> f32 {
    match ENCLOSING_RATIO.get(k as usize) {
        Some(&ratio) if k > 0 => k as f32 / (ratio * ratio),
        _ => fallback,
    }
}

impl Default for BubbleConfig {
    fn default() -> Self {
        Self {
            base_radius: 10.0,
            padding: 5.0,
            packing_efficiency: 0.82,
        }
    }
}

/// Per-slot containment hierarchy columns over a dense slot space.
///
/// Every vector has exactly `node_count` entries, indexed by slot.
pub struct BubbleHierarchy {
    /// Bubble radius per slot (bottom-up over the subtree).
    pub well_radius: Vec<f32>,
    /// Depth from the slot's own root; roots are 0.
    pub depth: Vec<u32>,
    /// Containment parent slot, or [`HIERARCHY_ROOT`].
    pub parent: Vec<u32>,
    /// Subtree node count including the slot itself; leaves are 1.
    pub subtree_size: Vec<u32>,
}

/// Parent→children CSR over the dense slot space.
///
/// `offsets` has `node_count + 1` entries; the children of slot `n` are
/// `children[offsets[n]..offsets[n + 1]]`. A `Vec`-indexed CSR replaces a
/// `HashMap<u32, Vec<u32>>` here: the slot space is dense and known, so the
/// hashing (and one heap allocation per parent) buys nothing.
struct ChildrenCsr {
    offsets: Vec<u32>,
    children: Vec<u32>,
}

impl ChildrenCsr {
    /// Build from flat `[parent0, child0, parent1, child1, ...]` pairs.
    ///
    /// Pairs referencing a slot outside `0..node_count`, and self-edges, are
    /// dropped. A trailing unpaired element is ignored. `has_parent[c]` is set
    /// for every child of a retained pair.
    fn build(containment_edges: &[u32], node_count: usize, has_parent: &mut [bool]) -> Self {
        let mut offsets = vec![0u32; node_count + 1];
        let pair_count = containment_edges.len() / 2;

        for i in 0..pair_count {
            let parent = containment_edges[i * 2] as usize;
            let child = containment_edges[i * 2 + 1] as usize;
            if parent >= node_count || child >= node_count || parent == child {
                continue;
            }
            offsets[parent + 1] += 1;
            has_parent[child] = true;
        }

        for i in 0..node_count {
            offsets[i + 1] += offsets[i];
        }

        let mut children = vec![0u32; offsets[node_count] as usize];
        let mut cursor = offsets[..node_count].to_vec();
        for i in 0..pair_count {
            let parent = containment_edges[i * 2] as usize;
            let child = containment_edges[i * 2 + 1] as usize;
            if parent >= node_count || child >= node_count || parent == child {
                continue;
            }
            children[cursor[parent] as usize] = child as u32;
            cursor[parent] += 1;
        }

        Self { offsets, children }
    }
}

/// Traversal scratch shared by every root walk.
struct Walk {
    parent: Vec<u32>,
    depth: Vec<u32>,
    visited: Vec<bool>,
    /// Discovery order: a topological order of the spanning forest, so a
    /// reverse sweep visits every node after all of its tree children.
    order: Vec<u32>,
    stack: Vec<u32>,
}

impl Walk {
    /// Claim `slot` for the forest under `parent` at `depth` and stage it for
    /// expansion. Returns `false` when the slot was already claimed — by a
    /// second containment parent (the first one wins) or by a cycle folding
    /// back into this tree.
    fn claim(&mut self, slot: u32, parent: u32, depth: u32) -> bool {
        let idx = slot as usize;
        if self.visited[idx] {
            return false;
        }
        self.visited[idx] = true;
        self.parent[idx] = parent;
        self.depth[idx] = depth;
        self.order.push(slot);
        self.stack.push(slot);
        true
    }

    /// Depth-first walk rooting a tree at `root`, recording parents and depths.
    ///
    /// A no-op if `root` was already claimed by an earlier walk. Iterative by
    /// construction: a recursive walk overflows the stack on a deep chain, and
    /// a stack overflow in WASM is an uncatchable trap.
    ///
    /// A child is claimed when it is *pushed*, not when it is popped, so a slot
    /// enters the stack at most once and the stack is bounded by `node_count`
    /// rather than by the edge count. Children are pushed in reverse CSR order
    /// so the first-listed child's subtree is explored first — the conventional
    /// depth-first order, and the one that decides which parent wins when a
    /// slot has more than one incoming containment edge.
    fn root_tree_at(&mut self, root: u32, csr: &ChildrenCsr) {
        if !self.claim(root, HIERARCHY_ROOT, 0) {
            return;
        }

        while let Some(node) = self.stack.pop() {
            let node_idx = node as usize;
            let child_depth = self.depth[node_idx] + 1;
            let lo = csr.offsets[node_idx] as usize;
            let hi = csr.offsets[node_idx + 1] as usize;

            for i in (lo..hi).rev() {
                self.claim(csr.children[i], node, child_depth);
            }
        }
    }
}

/// Derive the containment hierarchy columns from containment-only edges.
///
/// # Arguments
///
/// * `containment_edges` - Flat `[parent0, child0, parent1, child1, ...]`.
///   Must be containment edges only: a cross-cutting dependency edge (an
///   import, a test reference) reparents its target and skews every column.
/// * `node_count` - Total number of node slots (node_bound).
/// * `root_id` - Optional slot to root first. It becomes a root even if it has
///   a containment parent; the rest of the forest is derived around it. `None`
///   (and an out-of-range id) leaves root selection to the edges alone.
/// * `config` - Bubble radius configuration.
pub fn compute_bubble_hierarchy(
    containment_edges: &[u32],
    node_count: usize,
    root_id: Option<u32>,
    config: &BubbleConfig,
) -> BubbleHierarchy {
    if node_count == 0 {
        return BubbleHierarchy {
            well_radius: Vec::new(),
            depth: Vec::new(),
            parent: Vec::new(),
            subtree_size: Vec::new(),
        };
    }

    let mut has_parent = vec![false; node_count];
    let csr = ChildrenCsr::build(containment_edges, node_count, &mut has_parent);

    let mut walk = Walk {
        parent: vec![HIERARCHY_ROOT; node_count],
        depth: vec![0u32; node_count],
        visited: vec![false; node_count],
        order: Vec::with_capacity(node_count),
        stack: Vec::new(),
    };

    // 1. The caller's explicit root, so it outranks the edge-derived ones.
    if let Some(root) = root_id
        && (root as usize) < node_count
    {
        walk.root_tree_at(root, &csr);
    }

    // 2. Every slot with no containment parent roots its own tree. This is what
    //    makes the output a forest: orphan config files, a second repo root and
    //    isolated slots all get real trees instead of a flat depth-0 fallback.
    for (slot, &parented) in has_parent.iter().enumerate() {
        if !parented {
            walk.root_tree_at(slot as u32, &csr);
        }
    }

    // 3. Anything still unclaimed sits in a containment cycle with no entry
    //    point. Root it at its lowest slot so the forest spans the slot space.
    for slot in 0..node_count {
        if !walk.visited[slot] {
            walk.root_tree_at(slot as u32, &csr);
        }
    }

    // Bottom-up sweep. Reverse discovery order visits every node after all of
    // its tree children, so one linear pass computes radii and subtree sizes
    // and pushes both into the parent's accumulators. Contributions are keyed
    // on the node (not on the edge), so duplicate containment edges and
    // second parents cannot double-count.
    let packing = config.packing_efficiency.max(f32::MIN_POSITIVE);
    let mut well_radius = vec![config.base_radius; node_count];
    let mut subtree_size = vec![1u32; node_count];
    let mut child_area = vec![0f32; node_count];
    let mut child_count = vec![0u32; node_count];
    // The two widest children, which set a floor the area alone cannot see.
    let mut widest = vec![0f32; node_count];
    let mut next_widest = vec![0f32; node_count];

    for &node in walk.order.iter().rev() {
        let idx = node as usize;
        let radius = if child_count[idx] == 0 {
            config.base_radius
        } else {
            let density = packing_density(child_count[idx], packing);
            let by_area = (child_area[idx] / (std::f32::consts::PI * density)).sqrt();
            // Two disjoint circles of radii a and b inside a circle of radius R
            // put their far points 2R >= (a + b) + |c1 - c2| >= 2(a + b) apart,
            // so R >= a + b whatever the rest of the subtree looks like. Summed
            // area cannot express that: one wide child among many narrow ones
            // contributes little area and still needs the room.
            let by_pair = widest[idx] + next_widest[idx];
            by_area.max(by_pair).max(config.base_radius) + config.padding
        };
        well_radius[idx] = radius;

        let parent = walk.parent[idx];
        if parent != HIERARCHY_ROOT {
            let parent_idx = parent as usize;
            child_area[parent_idx] += std::f32::consts::PI * radius * radius;
            subtree_size[parent_idx] += subtree_size[idx];
            child_count[parent_idx] += 1;
            if radius > widest[parent_idx] {
                next_widest[parent_idx] = widest[parent_idx];
                widest[parent_idx] = radius;
            } else if radius > next_widest[parent_idx] {
                next_widest[parent_idx] = radius;
            }
        }
    }

    BubbleHierarchy {
        well_radius,
        depth: walk.depth,
        parent: walk.parent,
        subtree_size,
    }
}

/// Flat `f32` encoding of [`compute_bubble_hierarchy`] for the JS boundary.
///
/// # Returns
///
/// `Vec<f32>` of length `4 * node_count`, four concatenated columns:
/// `[wellRadius…, depth…, parent…, subtreeSize…]`.
///
/// `parent` is `-1.0` for a forest root and the parent's slot index otherwise.
/// Slot indices and subtree sizes round-trip exactly through `f32` up to
/// 2^24 (16 777 216) — three orders of magnitude above the 220K target, and
/// the same bound the existing depth column already carried.
pub fn compute_bubble_data(
    containment_edges: &[u32],
    node_count: usize,
    root_id: Option<u32>,
    config: &BubbleConfig,
) -> Vec<f32> {
    let hierarchy = compute_bubble_hierarchy(containment_edges, node_count, root_id, config);

    let mut result = Vec::with_capacity(node_count * 4);
    result.extend_from_slice(&hierarchy.well_radius);
    result.extend(hierarchy.depth.iter().map(|&d| d as f32));
    result.extend(hierarchy.parent.iter().map(
        |&p| {
            if p == HIERARCHY_ROOT { -1.0 } else { p as f32 }
        },
    ));
    result.extend(hierarchy.subtree_size.iter().map(|&s| s as f32));
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    /// Column views over the flat `compute_bubble_data` result.
    struct Columns<'a> {
        radii: &'a [f32],
        depths: &'a [f32],
        parents: &'a [f32],
        subtree: &'a [f32],
    }

    fn columns(data: &[f32], n: usize) -> Columns<'_> {
        assert_eq!(data.len(), n * 4);
        Columns {
            radii: &data[0..n],
            depths: &data[n..n * 2],
            parents: &data[n * 2..n * 3],
            subtree: &data[n * 3..],
        }
    }

    #[test]
    fn test_empty_graph() {
        let result = compute_bubble_data(&[], 0, None, &BubbleConfig::default());
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_node_no_edges() {
        let result = compute_bubble_data(&[], 1, None, &BubbleConfig::default());
        assert_eq!(result.len(), 4);
        let c = columns(&result, 1);
        assert_eq!(c.radii[0], 10.0); // base_radius
        assert_eq!(c.depths[0], 0.0);
        assert_eq!(c.parents[0], -1.0); // root sentinel
        assert_eq!(c.subtree[0], 1.0);
    }

    #[test]
    fn test_simple_parent_child() {
        // Node 0 -> Node 1
        let edges = [0u32, 1];
        let result = compute_bubble_data(&edges, 2, None, &BubbleConfig::default());
        let c = columns(&result, 2);

        // Parent (0) should have larger radius than child (1)
        assert!(c.radii[0] > c.radii[1]);
        // Child is leaf, gets base_radius
        assert_eq!(c.radii[1], 10.0);
        // Root depth = 0, child depth = 1
        assert_eq!(c.depths[0], 0.0);
        assert_eq!(c.depths[1], 1.0);
        assert_eq!(c.parents[0], -1.0);
        assert_eq!(c.parents[1], 0.0);
        assert_eq!(c.subtree[0], 2.0);
        assert_eq!(c.subtree[1], 1.0);
    }

    #[test]
    fn test_wide_tree() {
        // Node 0 -> [1, 2, 3, 4, 5] (root with 5 children)
        let edges = [0u32, 1, 0, 2, 0, 3, 0, 4, 0, 5];
        let result = compute_bubble_data(&edges, 6, None, &BubbleConfig::default());
        let c = columns(&result, 6);

        // Root should have much larger radius (encloses 5 children)
        assert!(c.radii[0] > c.radii[1]);
        for i in 1..6 {
            assert_eq!(c.radii[i], 10.0);
            assert_eq!(c.depths[i], 1.0);
            assert_eq!(c.parents[i], 0.0);
            assert_eq!(c.subtree[i], 1.0);
        }
        assert_eq!(c.depths[0], 0.0);
        assert_eq!(c.subtree[0], 6.0);
    }

    #[test]
    fn test_deep_chain() {
        // 0 -> 1 -> 2 -> 3 -> 4
        let edges = [0u32, 1, 1, 2, 2, 3, 3, 4];
        let config = BubbleConfig {
            base_radius: 5.0,
            padding: 2.0,
            ..Default::default()
        };
        let result = compute_bubble_data(&edges, 5, None, &config);
        let c = columns(&result, 5);

        // Radii should decrease as we go deeper (less subtree)
        assert!(c.radii[0] > c.radii[1]);
        assert!(c.radii[1] > c.radii[2]);
        assert!(c.radii[2] > c.radii[3]);
        // Node 4 is leaf
        assert_eq!(c.radii[4], 5.0);

        for (i, &depth) in c.depths.iter().enumerate() {
            assert_eq!(depth, i as f32);
            assert_eq!(c.subtree[i], (5 - i) as f32);
        }
    }

    #[test]
    fn test_cycle_handling() {
        // 0 -> 1 -> 2 -> 0 (cycle, no node without a parent)
        let edges = [0u32, 1, 1, 2, 2, 0];
        let result = compute_bubble_data(&edges, 3, None, &BubbleConfig::default());
        let c = columns(&result, 3);

        // The cycle is broken at the lowest slot, deterministically.
        assert_eq!(c.parents[0], -1.0);
        assert_eq!(c.depths[0], 0.0);
        assert_eq!(c.parents[1], 0.0);
        assert_eq!(c.depths[1], 1.0);
        assert_eq!(c.parents[2], 1.0);
        assert_eq!(c.depths[2], 2.0);
        assert_eq!(c.subtree[0], 3.0);
    }

    #[test]
    fn test_forest_of_components() {
        // Two real trees plus two isolated slots — a repo root, an orphan
        // config directory, and two loose files. Before forest handling the
        // orphan component was flattened to depth 0 / base radius, which made
        // it invisible to every tree-walk consumer.
        //   Component A: 0 -> {1, 2}, 1 -> {3, 4}
        //   Component B: 5 -> {6, 7}
        //   Isolated:    8, 9
        let edges = [0u32, 1, 0, 2, 1, 3, 1, 4, 5, 6, 5, 7];
        let n = 10;
        let result = compute_bubble_data(&edges, n, None, &BubbleConfig::default());
        let c = columns(&result, n);

        // Component A
        assert_eq!(c.parents[0], -1.0);
        assert_eq!(c.depths[3], 2.0);
        assert_eq!(c.parents[3], 1.0);
        assert_eq!(c.subtree[0], 5.0);
        assert_eq!(c.subtree[1], 3.0);

        // Component B is a real tree, not a flattened orphan
        assert_eq!(c.parents[5], -1.0);
        assert_eq!(c.depths[6], 1.0);
        assert_eq!(c.depths[7], 1.0);
        assert_eq!(c.parents[6], 5.0);
        assert_eq!(c.subtree[5], 3.0);
        assert!(
            c.radii[5] > 10.0,
            "orphan component root must get a real bubble radius, got {}",
            c.radii[5]
        );

        // Isolated slots are singleton roots
        for slot in [8usize, 9] {
            assert_eq!(c.parents[slot], -1.0);
            assert_eq!(c.depths[slot], 0.0);
            assert_eq!(c.subtree[slot], 1.0);
            assert_eq!(c.radii[slot], 10.0);
        }

        // Subtree sizes partition the slot space exactly once per root.
        let root_total: f32 = (0..n)
            .filter(|&i| c.parents[i] < 0.0)
            .map(|i| c.subtree[i])
            .sum();
        assert_eq!(root_total, n as f32);
    }

    #[test]
    fn test_component_is_rooted_at_its_parentless_slot_not_its_lowest_slot() {
        // Containment pointing "backwards" in slot order: 3 → {1, 2}. Slot 3 is
        // the only parentless slot in the component, so it must root it — even
        // though slot 1 is the lowest slot there. Rooting components by lowest
        // slot instead would make 1 and 2 depth-0 roots and leave 3 childless,
        // which is the same invisible-orphan failure in a different disguise.
        let edges = [3u32, 1, 3, 2];
        let result = compute_bubble_data(&edges, 4, None, &BubbleConfig::default());
        let c = columns(&result, 4);

        assert_eq!(c.parents[3], -1.0);
        assert_eq!(c.parents[1], 3.0);
        assert_eq!(c.parents[2], 3.0);
        assert_eq!(c.depths[1], 1.0);
        assert_eq!(c.depths[2], 1.0);
        assert_eq!(c.subtree[3], 3.0);
        assert!(c.radii[3] > 10.0);

        // Slot 0 is in no component at all: a singleton root.
        assert_eq!(c.parents[0], -1.0);
        assert_eq!(c.subtree[0], 1.0);
    }

    #[test]
    fn test_forest_root_radii_match_isolated_computation() {
        // A component's columns must not depend on what else is in the graph.
        let solo = compute_bubble_data(&[5u32, 6, 5, 7], 8, None, &BubbleConfig::default());
        let mixed = compute_bubble_data(
            &[0u32, 1, 0, 2, 1, 3, 1, 4, 5, 6, 5, 7],
            8,
            None,
            &BubbleConfig::default(),
        );
        let solo_c = columns(&solo, 8);
        let mixed_c = columns(&mixed, 8);
        for slot in [5usize, 6, 7] {
            assert_eq!(solo_c.radii[slot], mixed_c.radii[slot]);
            assert_eq!(solo_c.depths[slot], mixed_c.depths[slot]);
            assert_eq!(solo_c.subtree[slot], mixed_c.subtree[slot]);
        }
    }

    #[test]
    fn test_disconnected_nodes() {
        // Only 0 -> 1, nodes 2 and 3 are disconnected
        let edges = [0u32, 1];
        let result = compute_bubble_data(&edges, 4, None, &BubbleConfig::default());
        let c = columns(&result, 4);

        // Disconnected nodes get base_radius, depth 0 and root parents
        for slot in [2usize, 3] {
            assert_eq!(c.radii[slot], 10.0);
            assert_eq!(c.depths[slot], 0.0);
            assert_eq!(c.parents[slot], -1.0);
            assert_eq!(c.subtree[slot], 1.0);
        }
    }

    #[test]
    fn test_explicit_root() {
        // 0 -> 1, 0 -> 2, root pinned to 0
        let edges = [0u32, 1, 0, 2];
        let result = compute_bubble_data(&edges, 3, Some(0), &BubbleConfig::default());
        let c = columns(&result, 3);
        assert_eq!(c.depths[0], 0.0);
        assert_eq!(c.depths[1], 1.0);
        assert_eq!(c.depths[2], 1.0);
    }

    #[test]
    fn test_explicit_root_overrides_containment_parent() {
        // 0 -> 1 -> 2, but the caller pins 1 as the root: 1 becomes a root and
        // 0 roots its own (now childless-of-1) tree.
        let edges = [0u32, 1, 1, 2];
        let result = compute_bubble_data(&edges, 3, Some(1), &BubbleConfig::default());
        let c = columns(&result, 3);
        assert_eq!(c.parents[1], -1.0);
        assert_eq!(c.depths[1], 0.0);
        assert_eq!(c.depths[2], 1.0);
        assert_eq!(c.parents[0], -1.0);
        assert_eq!(c.subtree[0], 1.0);
        assert_eq!(c.subtree[1], 2.0);
    }

    #[test]
    fn test_out_of_range_and_self_edges_are_dropped() {
        let edges = [0u32, 1, 0, 99, 42, 1, 2, 2];
        let result = compute_bubble_data(&edges, 3, None, &BubbleConfig::default());
        let c = columns(&result, 3);
        assert_eq!(c.parents[1], 0.0);
        assert_eq!(c.parents[2], -1.0);
        assert_eq!(c.subtree[0], 2.0);
    }

    #[test]
    fn test_duplicate_containment_edges_do_not_double_count() {
        let once = compute_bubble_data(&[0u32, 1, 0, 2], 3, None, &BubbleConfig::default());
        let twice = compute_bubble_data(
            &[0u32, 1, 0, 2, 0, 1, 0, 2],
            3,
            None,
            &BubbleConfig::default(),
        );
        assert_eq!(once, twice);
    }

    #[test]
    fn test_deep_chain_no_stack_overflow() {
        // 100_000-node linked chain: recursive tree walks would overflow the
        // stack (fatal trap in WASM); all traversals must be iterative.
        let n: u32 = 100_000;
        let mut edges = Vec::with_capacity((n as usize - 1) * 2);
        for i in 0..n - 1 {
            edges.push(i);
            edges.push(i + 1);
        }

        let result = compute_bubble_data(&edges, n as usize, Some(0), &BubbleConfig::default());
        let c = columns(&result, n as usize);
        assert_eq!(c.depths[0], 0.0);
        assert_eq!(c.depths[n as usize - 1], (n - 1) as f32);
        assert_eq!(c.radii[n as usize - 1], 10.0);
        assert_eq!(c.subtree[0], n as f32);
        assert_eq!(c.subtree[n as usize - 1], 1.0);
    }

    #[test]
    fn test_dependency_edges_corrupt_radii_without_filtering() {
        // Containment: 0 → {1, 2}, 1 → {3, 4}, 2 → {5, 6}
        let containment = [0u32, 1, 0, 2, 1, 3, 1, 4, 2, 5, 2, 6];
        let clean = compute_bubble_data(&containment, 7, Some(0), &BubbleConfig::default());

        // Same hierarchy plus a dependency edge 3 → 5 (e.g. an import). If it
        // is fed into the tree build, node 5 attaches under 3 instead of its
        // real parent 2, skewing depths and well radii.
        let mut mixed = containment.to_vec();
        mixed.extend_from_slice(&[3, 5]);
        let corrupted = compute_bubble_data(&mixed, 7, Some(0), &BubbleConfig::default());

        let clean_c = columns(&clean, 7);
        let corrupted_c = columns(&corrupted, 7);
        assert_eq!(clean_c.depths[5], 2.0);
        // Documents why callers must pass containment-only edges: the walk
        // attaches 5 under whichever neighbor reaches it first.
        assert_eq!(corrupted_c.depths[5], 3.0);
        // Node 3 is a leaf in the clean tree but an internal node when the
        // dependency edge is included, inflating its well radius.
        assert!(corrupted_c.radii[3] > clean_c.radii[3]);
    }

    #[test]
    fn test_first_listed_child_subtree_wins_a_contested_slot() {
        // 0 → {1, 2}, and both 1 and 2 claim 3. Depth-first order means 1's
        // whole subtree is walked before 2 is reached, so 1 claims 3. Pinning
        // this makes the "first containment parent wins" rule reproducible
        // instead of an accident of stack order.
        let edges = [0u32, 1, 0, 2, 1, 3, 2, 3];
        let result = compute_bubble_data(&edges, 4, None, &BubbleConfig::default());
        let c = columns(&result, 4);
        assert_eq!(c.parents[3], 1.0);
        assert_eq!(c.depths[3], 2.0);
        assert_eq!(c.subtree[1], 2.0);
        assert_eq!(c.subtree[2], 1.0);
    }

    #[test]
    fn test_large_hierarchy() {
        // Build a 3-level tree: root -> 10 dirs -> 5 files each = 61 nodes
        let mut edges = Vec::new();
        for dir in 1..=10 {
            edges.push(0u32);
            edges.push(dir);
            for file in 0..5 {
                let file_id = 11 + (dir - 1) * 5 + file;
                edges.push(dir);
                edges.push(file_id);
            }
        }
        let node_count = 61;
        let result = compute_bubble_data(&edges, node_count, Some(0), &BubbleConfig::default());
        let c = columns(&result, node_count);

        // Root has the largest radius
        let max_radius = c.radii.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(c.radii[0], max_radius);
        assert_eq!(c.subtree[0], node_count as f32);

        // All leaf files get base_radius
        for file in 11..61 {
            assert_eq!(c.radii[file], 10.0);
            assert_eq!(c.depths[file], 2.0);
            assert_eq!(c.subtree[file], 1.0);
        }

        // Dirs are at depth 1
        for dir in 1..=10 {
            assert_eq!(c.depths[dir as usize], 1.0);
            assert!(c.radii[dir as usize] > 10.0);
            assert_eq!(c.subtree[dir as usize], 6.0);
        }
    }

    // =========================================================================
    // CSR parity against the HashMap adjacency it replaced
    // =========================================================================

    /// Seeded LCG — a fixed tree shape without pulling in a dependency.
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, bound: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as usize) % bound
        }
    }

    /// Random forest: every slot above 0 attaches to an earlier slot, except
    /// one in eight which roots its own component.
    fn seeded_forest(n: usize, seed: u64) -> Vec<u32> {
        let mut rng = Lcg(seed);
        let mut edges = Vec::with_capacity(n * 2);
        for child in 1..n {
            if rng.below(8) == 0 {
                continue; // orphan root
            }
            let parent = rng.below(child);
            edges.push(parent as u32);
            edges.push(child as u32);
        }
        edges
    }

    /// The adjacency the CSR replaced: a SipHash `HashMap<u32, Vec<u32>>` with
    /// one heap allocation per parent, built from the same pair list.
    fn hashmap_children(containment_edges: &[u32], node_count: usize) -> HashMap<u32, Vec<u32>> {
        let mut map: HashMap<u32, Vec<u32>> = HashMap::new();
        for i in 0..containment_edges.len() / 2 {
            let parent = containment_edges[i * 2];
            let child = containment_edges[i * 2 + 1];
            if parent as usize >= node_count || child as usize >= node_count || parent == child {
                continue;
            }
            map.entry(parent).or_default().push(child);
        }
        map
    }

    #[test]
    fn test_csr_matches_hashmap_adjacency_on_seeded_forest() {
        let n = 5_000;
        let edges = seeded_forest(n, 0xC0FFEE);

        let mut has_parent = vec![false; n];
        let csr = ChildrenCsr::build(&edges, n, &mut has_parent);
        let map = hashmap_children(&edges, n);

        // Same child multiset per parent, same has_parent set.
        let mut expected_has_parent = vec![false; n];
        for children in map.values() {
            for &c in children {
                expected_has_parent[c as usize] = true;
            }
        }
        assert_eq!(has_parent, expected_has_parent);

        for slot in 0..n {
            let lo = csr.offsets[slot] as usize;
            let hi = csr.offsets[slot + 1] as usize;
            let from_csr = &csr.children[lo..hi];
            let empty: Vec<u32> = Vec::new();
            let from_map = map.get(&(slot as u32)).unwrap_or(&empty);
            assert_eq!(
                from_csr,
                &from_map[..],
                "children of slot {slot} diverge between CSR and HashMap"
            );
        }
        assert_eq!(
            csr.children.len(),
            map.values().map(Vec::len).sum::<usize>()
        );
    }

    #[test]
    fn test_seeded_forest_columns_are_self_consistent() {
        let n = 5_000;
        let edges = seeded_forest(n, 0x5EED);
        let h = compute_bubble_hierarchy(&edges, n, None, &BubbleConfig::default());

        assert_eq!(h.parent.len(), n);
        assert_eq!(h.depth.len(), n);
        assert_eq!(h.well_radius.len(), n);
        assert_eq!(h.subtree_size.len(), n);

        // Every slot is in exactly one tree: depth[c] == depth[parent] + 1,
        // and subtree sizes sum to n over the roots.
        let mut root_total = 0u32;
        let mut seen: HashSet<u32> = HashSet::new();
        for slot in 0..n {
            let parent = h.parent[slot];
            if parent == HIERARCHY_ROOT {
                assert_eq!(h.depth[slot], 0);
                root_total += h.subtree_size[slot];
            } else {
                assert_eq!(h.depth[slot], h.depth[parent as usize] + 1);
                assert!(seen.insert(slot as u32));
            }
            assert!(h.subtree_size[slot] >= 1);
        }
        assert_eq!(root_total, n as u32);

        // subtree_size[p] == 1 + sum of children's subtree sizes.
        let mut summed = vec![1u32; n];
        for slot in 0..n {
            let parent = h.parent[slot];
            if parent != HIERARCHY_ROOT {
                summed[parent as usize] += h.subtree_size[slot];
            }
        }
        assert_eq!(summed, h.subtree_size);
    }

    /// Children of `parent` in a derived hierarchy, widest first.
    fn child_radii(h: &BubbleHierarchy, parent: u32) -> Vec<f32> {
        let mut radii: Vec<f32> = (0..h.parent.len())
            .filter(|&slot| h.parent[slot] == parent)
            .map(|slot| h.well_radius[slot])
            .collect();
        radii.sort_by(|a, b| b.partial_cmp(a).unwrap());
        radii
    }

    /// A star: slot 0 contains `k` leaves.
    fn star(k: u32) -> Vec<u32> {
        (1..=k).flat_map(|child| [0u32, child]).collect()
    }

    /// A tree of `levels` generations rooted at slot 0, where fan-out varies
    /// 2..=6 by position so both the small-`k` table and its tail apply.
    /// Returns the containment edges and the slot count.
    fn mixed_fan_out_tree(levels: usize) -> (Vec<u32>, usize) {
        let mut edges = Vec::new();
        let mut next = 1u32;
        let mut frontier = vec![0u32];
        for level in 0..levels {
            let mut children = Vec::new();
            for (i, &parent) in frontier.iter().enumerate() {
                let fan_out = 2 + (i + level) % 5;
                let first = next;
                next += fan_out as u32;
                children.extend(first..next);
                edges.extend((first..next).flat_map(|child| [parent, child]));
            }
            frontier = children;
        }
        (edges, next as usize)
    }

    #[test]
    fn two_children_need_a_bubble_twice_their_radius() {
        // The case a single asymptotic constant gets worst, and the one a code
        // tree hits constantly. Two disjoint circles of radius r inside a
        // circle of radius R force R >= 2r; sizing from 0.82 yields 1.56r, and
        // no arrangement of the two children can rescue a radius that small.
        let config = BubbleConfig {
            base_radius: 10.0,
            padding: 0.0,
            packing_efficiency: 0.82,
        };
        let h = compute_bubble_hierarchy(&star(2), 3, None, &config);
        assert!(
            h.well_radius[0] >= 20.0,
            "two radius-10 children need an enclosing radius of at least 20, got {}",
            h.well_radius[0],
        );
    }

    #[test]
    fn equal_children_get_the_room_their_own_geometry_demands() {
        // These two bounds are derived here rather than read from
        // `ENCLOSING_RATIO`, so the table cannot satisfy them by restating
        // itself. For k equal circles of radius r inside a circle of radius R,
        // the centres are pairwise at least 2r apart and all lie within R - r
        // of the middle. So R - r is at least the circumradius of the tightest
        // arrangement of k points with separation 2r:
        //
        //   k=3  equilateral triangle of side 2r  ->  2r/sqrt(3)
        //   k=6  regular hexagon of side 2r       ->  2r
        //
        // Both exceed the two-widest-children bound of 2r, which is what makes
        // this test bite where that one cannot: the summed-area estimate has to
        // be right for the fan-out, not merely right in the limit.
        let config = BubbleConfig {
            base_radius: 10.0,
            padding: 0.0,
            packing_efficiency: 0.82,
        };

        let three = compute_bubble_hierarchy(&star(3), 4, None, &config);
        let three_floor = 10.0 + 20.0 / 3.0f32.sqrt();
        assert!(
            three.well_radius[0] >= three_floor,
            "three radius-10 children need at least {three_floor}, got {}",
            three.well_radius[0],
        );

        let six = compute_bubble_hierarchy(&star(6), 7, None, &config);
        assert!(
            six.well_radius[0] >= 30.0,
            "six radius-10 children need at least 30, got {}",
            six.well_radius[0],
        );
    }

    #[test]
    fn two_wide_children_set_a_floor_summed_area_cannot_see() {
        // A directory holding two large packages and one loose file. The two
        // packages have to sit side by side whatever else is in there, so the
        // radius is theirs to set — but they are only two terms in the area sum,
        // which at this fan-out lands ~10% short of what they need.
        //
        // The shape matters: one wide child among many narrow ones does *not*
        // exercise this, because the narrow ones add enough area to cover the
        // pair anyway. It takes two.
        let mut edges = Vec::new();
        let mut next = 3u32;
        for package in [1u32, 2u32] {
            edges.extend([0, package]);
            for _ in 0..12 {
                edges.extend([package, next]);
                next += 1;
            }
        }
        edges.extend([0, next]); // the loose file
        let n = next as usize + 1;

        let config = BubbleConfig::default();
        let h = compute_bubble_hierarchy(&edges, n, None, &config);

        let radii = child_radii(&h, 0);
        assert_eq!(radii.len(), 3);
        let floor = radii[0] + radii[1];
        assert!(
            h.well_radius[0] >= floor + config.padding,
            "the two widest children ({} + {}) must fit side by side inside {}",
            radii[0],
            radii[1],
            h.well_radius[0],
        );
    }

    #[test]
    fn every_parent_can_hold_its_two_widest_children() {
        // The invariant behind both tests above, over a shape with mixed fan-out
        // at every level. Derived from geometry rather than from the packing
        // table, so it stays a real check even if the table is re-tuned: two
        // disjoint circles of radii a and b inside a circle of radius R put
        // their far points (a + b) + |c1 - c2| >= 2(a + b) apart, and no two
        // points of that circle are more than 2R apart.
        let (edges, n) = mixed_fan_out_tree(4);
        let config = BubbleConfig::default();
        let h = compute_bubble_hierarchy(&edges, n, None, &config);

        let mut checked = 0;
        for parent in 0..n as u32 {
            let radii = child_radii(&h, parent);
            if radii.len() < 2 {
                continue;
            }
            checked += 1;
            assert!(
                h.well_radius[parent as usize] >= radii[0] + radii[1],
                "parent {} has radius {} but its two widest children are {} and {}",
                parent,
                h.well_radius[parent as usize],
                radii[0],
                radii[1],
            );
        }
        assert!(
            checked > 30,
            "expected a broad tree to check, saw {checked} parents"
        );
    }

    #[test]
    fn enclosing_ratios_never_shrink_as_circles_are_added() {
        // A guard on the table itself: k + 1 circles cannot fit in a smaller
        // circle than k of them do, so a mistyped entry shows up here.
        for k in 2..ENCLOSING_RATIO.len() {
            assert!(
                ENCLOSING_RATIO[k] >= ENCLOSING_RATIO[k - 1],
                "ratio for k={k} is below k={}",
                k - 1,
            );
        }
        // And the densities they imply stay physically possible: no packing of
        // two or more equal circles beats the hexagonal limit. (One circle in a
        // circle is the degenerate exception — it is 100% dense.)
        for k in 2..ENCLOSING_RATIO.len() as u32 {
            let density = packing_density(k, 0.82);
            assert!(
                density <= 0.907,
                "k={k} implies density {density}, above the hexagonal limit",
            );
        }
    }

    #[test]
    fn wide_fan_out_still_uses_the_configured_efficiency() {
        // The table stops at 12 by design; beyond it the boundary loss has
        // washed out and the configured constant governs, so a caller tuning
        // `packing_efficiency` still has an effect where it is meaningful.
        let loose = BubbleConfig {
            base_radius: 10.0,
            padding: 0.0,
            packing_efficiency: 0.5,
        };
        let tight = BubbleConfig {
            base_radius: 10.0,
            padding: 0.0,
            packing_efficiency: 0.9,
        };
        let edges = star(40);
        let loose_h = compute_bubble_hierarchy(&edges, 41, None, &loose);
        let tight_h = compute_bubble_hierarchy(&edges, 41, None, &tight);
        assert!(
            loose_h.well_radius[0] > tight_h.well_radius[0],
            "a lower efficiency must widen a 40-child bubble: {} vs {}",
            loose_h.well_radius[0],
            tight_h.well_radius[0],
        );
    }
}
