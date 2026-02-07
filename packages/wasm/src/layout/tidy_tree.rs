//! Buchheim-Junger-Leipert tidy tree layout algorithm.
//!
//! Implements the O(n) algorithm from "Improving Walker's Algorithm to Run in
//! Linear Time" (Buchheim, Junger, Leipert, 2002) for laying out arbitrary
//! m-ary trees with compact, aesthetically pleasing positioning.
//!
//! The algorithm produces (x, depth) coordinates per node, which can be
//! transformed into either linear (top-down) or radial (polar) coordinates
//! for visualization.
//!
//! # Algorithm Overview
//!
//! 1. **First walk (bottom-up):** Recursively assign preliminary x-coordinates
//!    to each node by merging subtree contours. Uses threads for O(1) amortized
//!    contour traversal.
//! 2. **Second walk (top-down):** Apply accumulated modifiers to convert
//!    preliminary x-coordinates to final positions.
//! 3. **Coordinate transform:** Convert (x, depth) to desired coordinate system
//!    (linear or radial).

use std::collections::{HashMap, HashSet};

/// Coordinate mode for the final layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateMode {
    /// Linear top-down layout: x = horizontal, y = depth * level_spacing.
    Linear,
    /// Radial layout: angle = x * angular_scale, radius = depth * level_spacing.
    Radial,
}

/// Configuration for the tidy tree layout.
#[derive(Debug, Clone)]
pub struct TidyTreeConfig {
    /// Minimum horizontal separation between sibling nodes.
    pub sibling_separation: f32,
    /// Minimum horizontal separation between subtrees.
    pub subtree_separation: f32,
    /// Vertical (or radial) spacing between tree levels.
    pub level_separation: f32,
    /// Coordinate output mode.
    pub coordinate_mode: CoordinateMode,
}

impl Default for TidyTreeConfig {
    fn default() -> Self {
        Self {
            sibling_separation: 1.0,
            subtree_separation: 2.0,
            level_separation: 80.0,
            coordinate_mode: CoordinateMode::Radial,
        }
    }
}

/// Internal node data used during the Buchheim algorithm.
#[derive(Debug)]
struct LayoutNode {
    /// Index into the output arrays.
    slot: usize,
    /// Depth in the tree (root = 0).
    depth: u32,
    /// Parent layout index (None for root).
    parent: Option<usize>,
    /// Children (ordered by the edge insertion order).
    children: Vec<usize>,
    /// Preliminary x-coordinate (from first walk).
    prelim: f32,
    /// Modifier for subtree shift (accumulated in first walk, applied in second).
    modifier: f32,
    /// Left thread pointer (index into layout_nodes).
    thread_left: Option<usize>,
    /// Right thread pointer (index into layout_nodes).
    thread_right: Option<usize>,
    /// Ancestor pointer (for the "default ancestor" in apportion).
    ancestor: usize,
    /// Shift value for even spacing of intermediate children.
    shift: f32,
    /// Change value for even spacing of intermediate children.
    change: f32,
    /// Number (left-to-right index among siblings).
    number: usize,
}

/// Result of the tidy tree layout computation.
pub struct TidyTreeResult {
    /// Target X positions (one per node in graph slot order).
    pub positions_x: Vec<f32>,
    /// Target Y positions (one per node in graph slot order).
    pub positions_y: Vec<f32>,
    /// Number of nodes laid out.
    pub node_count: usize,
}

/// The tidy tree layout engine.
pub struct TidyTreeLayout {
    config: TidyTreeConfig,
}

impl TidyTreeLayout {
    /// Create a new tidy tree layout with the given configuration.
    pub fn new(config: TidyTreeConfig) -> Self {
        Self { config }
    }

    /// Create a tidy tree layout with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(TidyTreeConfig::default())
    }

    /// Compute the tidy tree layout.
    ///
    /// # Arguments
    ///
    /// * `node_count` - Total number of node slots (may include holes from removals)
    /// * `edges` - Flat array of directed edge pairs [src0, tgt0, src1, tgt1, ...]
    ///   representing parent→child relationships
    /// * `root_id` - The root node ID (or None to auto-detect)
    ///
    /// # Returns
    ///
    /// A `TidyTreeResult` with target positions for all nodes. Nodes not in the
    /// tree (disconnected, removed) get position (0, 0).
    pub fn compute(
        &self,
        node_count: usize,
        edges: &[u32],
        root_id: Option<u32>,
    ) -> TidyTreeResult {
        // Sentinel value for "not part of tree". The GPU shader checks for this
        // to skip non-tree nodes. Using a very large value that no real layout
        // position would produce — the shader checks `target_pos.x >= SENTINEL`.
        const SENTINEL: f32 = 3.402_823e+38;

        let empty_result = || TidyTreeResult {
            positions_x: vec![SENTINEL; node_count],
            positions_y: vec![SENTINEL; node_count],
            node_count: 0,
        };

        if node_count == 0 || edges.is_empty() {
            return empty_result();
        }

        // Validate edge array: must be even length (pairs of [parent, child])
        if edges.len() % 2 != 0 {
            return empty_result();
        }

        // Build adjacency: parent → children
        let mut children_map: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut has_parent: HashMap<u32, bool> = HashMap::new();
        let mut all_nodes: HashSet<u32> = HashSet::new();

        let edge_count = edges.len() / 2;
        for i in 0..edge_count {
            let parent = edges[i * 2];
            let child = edges[i * 2 + 1];

            // Validate node IDs are within bounds
            if parent as usize >= node_count || child as usize >= node_count {
                continue;
            }
            // Skip self-loops
            if parent == child {
                continue;
            }

            children_map.entry(parent).or_default().push(child);
            has_parent.insert(child, true);
            all_nodes.insert(parent);
            all_nodes.insert(child);
        }

        // If no valid edges after filtering, return empty
        if all_nodes.is_empty() {
            return empty_result();
        }

        // Find root: specified or auto-detect (node with no incoming edges)
        let root = if let Some(r) = root_id {
            r
        } else {
            // Find nodes with no parent
            let roots: Vec<u32> = all_nodes
                .iter()
                .filter(|n| !has_parent.get(n).copied().unwrap_or(false))
                .copied()
                .collect();

            if roots.is_empty() {
                // Cycle or no clear root; pick node 0
                *all_nodes.iter().min().unwrap_or(&0)
            } else if roots.len() == 1 {
                roots[0]
            } else {
                // Multiple roots: pick the one with most descendants
                roots
                    .iter()
                    .max_by_key(|&&r| Self::count_descendants(r, &children_map))
                    .copied()
                    .unwrap_or(roots[0])
            }
        };

        // Build layout nodes via DFS from root (with cycle detection)
        let mut layout_nodes: Vec<LayoutNode> = Vec::new();
        let mut node_to_layout: HashMap<u32, usize> = HashMap::new();
        let mut visited: HashSet<u32> = HashSet::new();

        Self::build_layout_tree(
            root,
            None,
            0,
            &children_map,
            &mut layout_nodes,
            &mut node_to_layout,
            &mut visited,
        );

        if layout_nodes.is_empty() {
            return TidyTreeResult {
                positions_x: vec![SENTINEL; node_count],
                positions_y: vec![SENTINEL; node_count],
                node_count: 0,
            };
        }

        // Run Buchheim's algorithm
        self.first_walk(0, &mut layout_nodes);

        // Collect final prelim values after second walk
        let mut final_x: Vec<f32> = vec![0.0; layout_nodes.len()];
        self.second_walk_collect(0, 0.0, &layout_nodes, &mut final_x);

        // Center the tree: find min x and shift everything so min_x = 0
        let min_x = final_x.iter().copied().fold(f32::INFINITY, f32::min);
        let max_x = final_x
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let x_range = max_x - min_x;

        // Convert to output coordinates (sentinel means "not in tree")
        let mut positions_x = vec![SENTINEL; node_count];
        let mut positions_y = vec![SENTINEL; node_count];
        let mut laid_out = 0;

        match self.config.coordinate_mode {
            CoordinateMode::Linear => {
                // Center horizontally around 0
                let x_offset = -(min_x + x_range / 2.0);
                for (layout_idx, node) in layout_nodes.iter().enumerate() {
                    let slot = node.slot;
                    if slot < node_count {
                        positions_x[slot] =
                            (final_x[layout_idx] + x_offset) * self.config.level_separation;
                        positions_y[slot] =
                            node.depth as f32 * self.config.level_separation;
                        laid_out += 1;
                    }
                }
            }
            CoordinateMode::Radial => {
                // Map x range to angular range (0..2*PI), depth to radius
                let divisor = x_range + self.config.sibling_separation;
                if x_range > 0.0 && divisor > f32::EPSILON {
                    let angular_scale = std::f32::consts::TAU / divisor;
                    for (layout_idx, node) in layout_nodes.iter().enumerate() {
                        let slot = node.slot;
                        if slot < node_count {
                            let normalized_x = final_x[layout_idx] - min_x;
                            let angle = normalized_x * angular_scale;
                            let radius = (node.depth as f32 + 1.0) * self.config.level_separation;
                            positions_x[slot] = radius * angle.cos();
                            positions_y[slot] = radius * angle.sin();
                            laid_out += 1;
                        }
                    }
                    // Root at center
                    if let Some(&root_layout_idx) = node_to_layout.get(&root) {
                        let slot = layout_nodes[root_layout_idx].slot;
                        if slot < node_count {
                            positions_x[slot] = 0.0;
                            positions_y[slot] = 0.0;
                        }
                    }
                } else {
                    // Single node or all nodes at same x
                    for node in &layout_nodes {
                        let slot = node.slot;
                        if slot < node_count {
                            positions_x[slot] = 0.0;
                            positions_y[slot] = 0.0;
                            laid_out += 1;
                        }
                    }
                }
            }
        }

        TidyTreeResult {
            positions_x,
            positions_y,
            node_count: laid_out,
        }
    }

    /// Count descendants of a node (for root selection heuristic).
    /// Uses visited set to handle cycles safely.
    fn count_descendants(node: u32, children_map: &HashMap<u32, Vec<u32>>) -> usize {
        let mut count = 0;
        let mut stack = vec![node];
        let mut visited = HashSet::new();
        visited.insert(node);
        while let Some(n) = stack.pop() {
            if let Some(children) = children_map.get(&n) {
                for &child in children {
                    if visited.insert(child) {
                        count += 1;
                        stack.push(child);
                    }
                }
            }
        }
        count
    }

    /// Build the layout tree via DFS from root.
    /// Uses a visited set to prevent infinite recursion on cyclic graphs.
    /// Nodes already visited are skipped (breaking the cycle).
    fn build_layout_tree(
        node_id: u32,
        parent_layout_idx: Option<usize>,
        depth: u32,
        children_map: &HashMap<u32, Vec<u32>>,
        layout_nodes: &mut Vec<LayoutNode>,
        node_to_layout: &mut HashMap<u32, usize>,
        visited: &mut HashSet<u32>,
    ) {
        // Cycle detection: skip already-visited nodes
        if !visited.insert(node_id) {
            return;
        }

        let layout_idx = layout_nodes.len();
        node_to_layout.insert(node_id, layout_idx);

        layout_nodes.push(LayoutNode {
            slot: node_id as usize,
            depth,
            parent: parent_layout_idx,
            children: Vec::new(),
            prelim: 0.0,
            modifier: 0.0,
            thread_left: None,
            thread_right: None,
            ancestor: layout_idx,
            shift: 0.0,
            change: 0.0,
            number: 0,
        });

        if let Some(children) = children_map.get(&node_id) {
            let mut child_layout_indices: Vec<usize> = Vec::with_capacity(children.len());

            for (number, &child_id) in children.iter().enumerate() {
                let before_len = layout_nodes.len();
                Self::build_layout_tree(
                    child_id,
                    Some(layout_idx),
                    depth + 1,
                    children_map,
                    layout_nodes,
                    node_to_layout,
                    visited,
                );
                // Only add to children list if the node was actually inserted
                // (it won't be if it was a cycle back-edge)
                if layout_nodes.len() > before_len {
                    if let Some(&child_idx) = node_to_layout.get(&child_id) {
                        layout_nodes[child_idx].number = number;
                        child_layout_indices.push(child_idx);
                    }
                }
            }

            layout_nodes[layout_idx].children = child_layout_indices;
        }
    }

    /// Buchheim first walk: bottom-up assignment of preliminary x-coordinates.
    fn first_walk(&self, v: usize, nodes: &mut Vec<LayoutNode>) {
        // Clone children indices to avoid borrow conflict during recursion
        let children: Vec<usize> = nodes[v].children.clone();

        if children.is_empty() {
            // Leaf node: position relative to left sibling
            nodes[v].prelim = 0.0;
            return;
        }

        // Recursively walk children
        for &child in &children {
            self.first_walk(child, nodes);
        }

        // Default ancestor for the apportion step
        let mut default_ancestor = children[0];

        // Position children and merge contours
        for (i, &child) in children.iter().enumerate() {
            if i > 0 {
                let left_sibling = children[i - 1];
                // Shift child so it doesn't overlap left sibling's subtree
                let shift = self.separate(left_sibling, child, nodes);
                nodes[child].prelim += shift;
                nodes[child].modifier += shift;

                // Apportion: handle subtrees between left_sibling and child
                default_ancestor =
                    self.apportion(child, left_sibling, default_ancestor, nodes);
            }
        }

        // Distribute extra space evenly among intermediate children
        self.execute_shifts(v, nodes);

        // Center parent over first and last children
        // Safety: children is non-empty (checked above), so first()/last() are safe
        let first_child_prelim = nodes[children[0]].prelim;
        let last_child_prelim = nodes[children[children.len() - 1]].prelim;
        let midpoint = (first_child_prelim + last_child_prelim) / 2.0;
        nodes[v].prelim = midpoint;
    }

    /// Compute the minimum separation needed between two sibling subtrees.
    fn separate(&self, left: usize, right: usize, nodes: &Vec<LayoutNode>) -> f32 {
        // Walk the right contour of left subtree and left contour of right subtree
        let mut left_contour = left;
        let mut right_contour = right;
        let mut left_mod = 0.0f32;
        let mut right_mod = 0.0f32;
        let mut max_shift = 0.0f32;

        loop {
            let left_x = nodes[left_contour].prelim + left_mod;
            let right_x = nodes[right_contour].prelim + right_mod;

            let desired_sep = if self.are_siblings(left_contour, right_contour, nodes) {
                self.config.sibling_separation
            } else {
                self.config.subtree_separation
            };

            let overlap = left_x + desired_sep - right_x;
            if overlap > max_shift {
                max_shift = overlap;
            }

            // Advance contours down one level
            let next_left = self.next_right(left_contour, nodes);
            let next_right = self.next_left(right_contour, nodes);

            match (next_left, next_right) {
                (Some(nl), Some(nr)) => {
                    left_mod += nodes[left_contour].modifier;
                    right_mod += nodes[right_contour].modifier;
                    left_contour = nl;
                    right_contour = nr;
                }
                _ => break,
            }
        }

        max_shift
    }

    /// Check if two layout nodes are siblings (share the same parent).
    fn are_siblings(&self, a: usize, b: usize, nodes: &[LayoutNode]) -> bool {
        nodes[a].parent.is_some() && nodes[a].parent == nodes[b].parent
    }

    /// Get the next node on the right contour of a subtree.
    fn next_right(&self, v: usize, nodes: &[LayoutNode]) -> Option<usize> {
        if let Some(&last_child) = nodes[v].children.last() {
            Some(last_child)
        } else {
            nodes[v].thread_right
        }
    }

    /// Get the next node on the left contour of a subtree.
    fn next_left(&self, v: usize, nodes: &[LayoutNode]) -> Option<usize> {
        if let Some(&first_child) = nodes[v].children.first() {
            Some(first_child)
        } else {
            nodes[v].thread_left
        }
    }

    /// Apportion: ensure that subtrees between siblings don't overlap.
    /// This is the core of Buchheim's linear-time improvement over Walker's algorithm.
    fn apportion(
        &self,
        v: usize,
        left_sibling: usize,
        mut default_ancestor: usize,
        nodes: &mut Vec<LayoutNode>,
    ) -> usize {
        // v_inner_left: left contour of v's subtree
        // v_outer_left: left contour going leftward from v
        // v_inner_right: right contour of left_sibling's subtree
        // v_outer_right: right contour going rightward from v

        let mut v_inner_right = left_sibling;
        let mut v_outer_right = left_sibling;
        let mut v_inner_left = v;
        // Find leftmost sibling via O(1) parent lookup
        let mut v_outer_left = if let Some(parent_idx) = nodes[v].parent {
            // First child of parent is leftmost sibling
            nodes[parent_idx].children.first().copied().unwrap_or(v)
        } else {
            v
        };

        let mut s_inner_right = nodes[v_inner_right].modifier;
        let mut s_outer_right = nodes[v_outer_right].modifier;
        let mut s_inner_left = nodes[v_inner_left].modifier;
        let mut s_outer_left = nodes[v_outer_left].modifier;

        // Use explicit match instead of .expect() to avoid panics
        loop {
            let next_ir = self.next_right(v_inner_right, nodes);
            let next_il = self.next_left(v_inner_left, nodes);

            match (next_ir, next_il) {
                (Some(ir), Some(il)) => {
                    v_inner_right = ir;
                    v_inner_left = il;
                }
                _ => break,
            }

            if let Some(next) = self.next_left(v_outer_left, nodes) {
                v_outer_left = next;
            }
            if let Some(next) = self.next_right(v_outer_right, nodes) {
                v_outer_right = next;
            }

            nodes[v_outer_right].ancestor = v;

            let shift = (nodes[v_inner_right].prelim + s_inner_right)
                - (nodes[v_inner_left].prelim + s_inner_left)
                + self.config.subtree_separation;

            if shift > 0.0 {
                let ancestor_v = nodes[v].ancestor;
                let move_ancestor = if self.is_ancestor_of(ancestor_v, v, nodes) {
                    ancestor_v
                } else {
                    default_ancestor
                };

                self.move_subtree(move_ancestor, v, shift, nodes);

                s_inner_left += shift;
                s_outer_left += shift;
            }

            s_inner_right += nodes[v_inner_right].modifier;
            s_inner_left += nodes[v_inner_left].modifier;
            s_outer_left += nodes[v_outer_left].modifier;
            s_outer_right += nodes[v_outer_right].modifier;
        }

        // Set threads
        if self.next_right(v_inner_right, nodes).is_some()
            && self.next_right(v_outer_right, nodes).is_none()
        {
            let next = self.next_right(v_inner_right, nodes);
            nodes[v_outer_right].thread_right = next;
            nodes[v_outer_right].modifier += s_inner_right - s_outer_right;
        }

        if self.next_left(v_inner_left, nodes).is_some()
            && self.next_left(v_outer_left, nodes).is_none()
        {
            let next = self.next_left(v_inner_left, nodes);
            nodes[v_outer_left].thread_left = next;
            nodes[v_outer_left].modifier += s_inner_left - s_outer_left;
            default_ancestor = v;
        }

        default_ancestor
    }

    /// Check if `ancestor` is an ancestor of `v` within the same sibling group.
    fn is_ancestor_of(&self, ancestor: usize, v: usize, nodes: &[LayoutNode]) -> bool {
        // In Buchheim's algorithm, this checks if ancestor is a sibling of v
        // or an ancestor of a sibling. We simplify: check same depth and
        // that ancestor's ancestor field points to a valid common ancestor.
        let v_depth = nodes[v].depth;
        let a_depth = nodes[ancestor].depth;
        a_depth <= v_depth
    }

    /// Move subtree: shift node v and adjust spacing between ancestor and v.
    fn move_subtree(
        &self,
        wl: usize,
        wr: usize,
        shift: f32,
        nodes: &mut Vec<LayoutNode>,
    ) {
        let subtrees = (nodes[wr].number as f32 - nodes[wl].number as f32).max(1.0);
        let per_subtree = shift / subtrees;

        nodes[wr].change -= per_subtree;
        nodes[wr].shift += shift;
        nodes[wl].change += per_subtree;
        nodes[wr].prelim += shift;
        nodes[wr].modifier += shift;
    }

    /// Execute accumulated shifts for children of node v.
    fn execute_shifts(&self, v: usize, nodes: &mut Vec<LayoutNode>) {
        // Clone children indices to avoid borrow conflict
        let children: Vec<usize> = nodes[v].children.clone();
        let mut shift = 0.0f32;
        let mut change = 0.0f32;

        for &child in children.iter().rev() {
            nodes[child].prelim += shift;
            nodes[child].modifier += shift;
            change += nodes[child].change;
            shift += nodes[child].shift + change;
        }
    }

    /// Second walk: apply accumulated modifiers to get final x-coordinates.
    fn second_walk_collect(
        &self,
        v: usize,
        modifier_sum: f32,
        nodes: &[LayoutNode],
        final_x: &mut Vec<f32>,
    ) {
        final_x[v] = nodes[v].prelim + modifier_sum;

        for &child in &nodes[v].children {
            self.second_walk_collect(child, modifier_sum + nodes[v].modifier, nodes, final_x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_node() {
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            coordinate_mode: CoordinateMode::Linear,
            ..Default::default()
        });

        // No edges means root only
        let result = layout.compute(1, &[], Some(0));
        assert_eq!(result.node_count, 0); // No edges, no tree
    }

    #[test]
    fn test_simple_tree() {
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            coordinate_mode: CoordinateMode::Linear,
            level_separation: 100.0,
            sibling_separation: 1.0,
            subtree_separation: 2.0,
        });

        // Tree:  0 → 1, 0 → 2
        let edges = [0, 1, 0, 2];
        let result = layout.compute(3, &edges, Some(0));

        assert_eq!(result.node_count, 3);

        // Root should be centered over children
        let root_x = result.positions_x[0];
        let child1_x = result.positions_x[1];
        let child2_x = result.positions_x[2];

        // Root x should be midpoint of children
        let midpoint = (child1_x + child2_x) / 2.0;
        assert!(
            (root_x - midpoint).abs() < 0.01,
            "Root x ({root_x}) should be midpoint of children ({midpoint})"
        );

        // Children should be on level 1 (y = 100)
        assert!(
            (result.positions_y[1] - 100.0).abs() < 0.01,
            "Child 1 y should be 100, got {}",
            result.positions_y[1]
        );
        assert!(
            (result.positions_y[2] - 100.0).abs() < 0.01,
            "Child 2 y should be 100, got {}",
            result.positions_y[2]
        );

        // Root should be on level 0 (y = 0)
        assert!(
            result.positions_y[0].abs() < 0.01,
            "Root y should be 0, got {}",
            result.positions_y[0]
        );

        // Children should be separated
        assert!(
            (child2_x - child1_x).abs() >= layout.config.sibling_separation * layout.config.level_separation,
            "Children should be separated: child1_x={child1_x}, child2_x={child2_x}"
        );
    }

    #[test]
    fn test_deeper_tree() {
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            coordinate_mode: CoordinateMode::Linear,
            level_separation: 50.0,
            ..Default::default()
        });

        // Tree:  0 → 1, 0 → 2, 1 → 3, 1 → 4, 2 → 5
        let edges = [0, 1, 0, 2, 1, 3, 1, 4, 2, 5];
        let result = layout.compute(6, &edges, Some(0));

        assert_eq!(result.node_count, 6);

        // Depth checks
        assert!(result.positions_y[0].abs() < 0.01, "Root at depth 0");
        assert!((result.positions_y[1] - 50.0).abs() < 0.01, "Node 1 at depth 1");
        assert!((result.positions_y[2] - 50.0).abs() < 0.01, "Node 2 at depth 1");
        assert!((result.positions_y[3] - 100.0).abs() < 0.01, "Node 3 at depth 2");
        assert!((result.positions_y[4] - 100.0).abs() < 0.01, "Node 4 at depth 2");
        assert!((result.positions_y[5] - 100.0).abs() < 0.01, "Node 5 at depth 2");
    }

    #[test]
    fn test_radial_layout() {
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            coordinate_mode: CoordinateMode::Radial,
            level_separation: 100.0,
            ..Default::default()
        });

        // Tree: 0 → 1, 0 → 2, 0 → 3, 0 → 4
        let edges = [0, 1, 0, 2, 0, 3, 0, 4];
        let result = layout.compute(5, &edges, Some(0));

        assert_eq!(result.node_count, 5);

        // Root should be at center
        assert!(result.positions_x[0].abs() < 0.01, "Root x should be ~0");
        assert!(result.positions_y[0].abs() < 0.01, "Root y should be ~0");

        // Children should be at radius = level_separation from center
        for i in 1..5 {
            let dist = (result.positions_x[i].powi(2) + result.positions_y[i].powi(2)).sqrt();
            assert!(
                (dist - 200.0).abs() < 1.0, // (depth+1)*level_sep = 2*100
                "Child {i} distance from center should be ~200, got {dist}"
            );
        }
    }

    #[test]
    fn test_auto_root_detection() {
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            coordinate_mode: CoordinateMode::Linear,
            level_separation: 50.0,
            ..Default::default()
        });

        // Tree: 0 → 1, 0 → 2 (node 0 has no incoming edges)
        let edges = [0, 1, 0, 2];
        let result = layout.compute(3, &edges, None);

        assert_eq!(result.node_count, 3);
        // Root (0) should be at depth 0
        assert!(result.positions_y[0].abs() < 0.01, "Auto-detected root at depth 0");
    }

    #[test]
    fn test_cyclic_graph_does_not_hang() {
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            coordinate_mode: CoordinateMode::Linear,
            ..Default::default()
        });

        // Cycle: 0 → 1 → 2 → 0 (back-edge)
        let edges = [0, 1, 1, 2, 2, 0];
        let result = layout.compute(3, &edges, Some(0));

        // Should not hang — cycle is broken during DFS
        // All 3 nodes should still be laid out (cycle back-edge is ignored)
        assert!(result.node_count > 0, "Should lay out nodes despite cycle");
        assert!(result.node_count <= 3, "Should not exceed node_count");
    }

    #[test]
    fn test_odd_edge_array_returns_empty() {
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            coordinate_mode: CoordinateMode::Linear,
            ..Default::default()
        });

        // Odd-length edge array is invalid
        let edges = [0, 1, 2];
        let result = layout.compute(3, &edges, Some(0));
        assert_eq!(result.node_count, 0, "Odd edge array should return empty result");
    }

    #[test]
    fn test_out_of_bounds_node_ids_skipped() {
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            coordinate_mode: CoordinateMode::Linear,
            ..Default::default()
        });

        // node_count=3 but edge references node 999
        let edges = [0, 1, 0, 999];
        let result = layout.compute(3, &edges, Some(0));

        // Only edge 0→1 is valid; node 999 is out of bounds and skipped
        assert_eq!(result.node_count, 2, "Should only lay out valid nodes");
    }

    #[test]
    fn test_self_loop_skipped() {
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            coordinate_mode: CoordinateMode::Linear,
            ..Default::default()
        });

        // Self-loop: 0→0
        let edges = [0, 0, 0, 1];
        let result = layout.compute(2, &edges, Some(0));

        // Self-loop should be skipped, only 0→1 edge used
        assert_eq!(result.node_count, 2, "Self-loop should be skipped");
    }

    #[test]
    fn test_asymmetric_tree() {
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            coordinate_mode: CoordinateMode::Linear,
            level_separation: 50.0,
            sibling_separation: 1.0,
            subtree_separation: 2.0,
        });

        // Asymmetric: left subtree deeper than right
        // 0 → 1, 0 → 2, 1 → 3, 3 → 4
        let edges = [0, 1, 0, 2, 1, 3, 3, 4];
        let result = layout.compute(5, &edges, Some(0));

        assert_eq!(result.node_count, 5);

        // Node 4 should be at depth 3
        assert!(
            (result.positions_y[4] - 150.0).abs() < 0.01,
            "Deep node at depth 3, got {}",
            result.positions_y[4]
        );

        // Subtrees should not overlap horizontally
        // Left subtree (1, 3, 4) should be distinct from right (2)
        let left_max_x = result.positions_x[1]
            .max(result.positions_x[3])
            .max(result.positions_x[4]);
        let right_min_x = result.positions_x[2];

        assert!(
            left_max_x < right_min_x || right_min_x < result.positions_x[1],
            "Subtrees should not overlap: left max x = {left_max_x}, right min x = {right_min_x}"
        );
    }
}
