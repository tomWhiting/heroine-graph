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
    /// Left-to-right index into the parent's `children` vec.
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
        self.second_walk_collect(0, &layout_nodes, &mut final_x);

        // Center the tree: find min x and shift everything so min_x = 0
        let min_x = final_x.iter().copied().fold(f32::INFINITY, f32::min);
        let max_x = final_x
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let x_range = max_x - min_x;

        // Convert to output coordinates (sentinel means "not in tree")
        let mut result = TidyTreeResult {
            positions_x: vec![SENTINEL; node_count],
            positions_y: vec![SENTINEL; node_count],
            node_count: 0,
        };

        match self.config.coordinate_mode {
            CoordinateMode::Linear => {
                // Center horizontally around 0
                let x_offset = -(min_x + x_range / 2.0);
                self.emit_linear(&layout_nodes, &final_x, x_offset, &mut result);
            }
            CoordinateMode::Radial => {
                let root_layout_idx = node_to_layout.get(&root).copied();
                self.emit_radial(&layout_nodes, &final_x, min_x, x_range, root_layout_idx, &mut result);
            }
        }

        result
    }

    /// Write linear (top-down) coordinates into `out`, counting laid-out nodes.
    fn emit_linear(
        &self,
        layout_nodes: &[LayoutNode],
        final_x: &[f32],
        x_offset: f32,
        out: &mut TidyTreeResult,
    ) {
        let node_count = out.positions_x.len();
        for (layout_idx, node) in layout_nodes.iter().enumerate() {
            let slot = node.slot;
            if slot < node_count {
                out.positions_x[slot] =
                    (final_x[layout_idx] + x_offset) * self.config.level_separation;
                out.positions_y[slot] = node.depth as f32 * self.config.level_separation;
                out.node_count += 1;
            }
        }
    }

    /// Write radial coordinates into `out`: x range maps to angular range
    /// (0..2*PI), depth to radius, root pinned at the center.
    fn emit_radial(
        &self,
        layout_nodes: &[LayoutNode],
        final_x: &[f32],
        min_x: f32,
        x_range: f32,
        root_layout_idx: Option<usize>,
        out: &mut TidyTreeResult,
    ) {
        let node_count = out.positions_x.len();
        let divisor = x_range + self.config.sibling_separation;

        if x_range <= 0.0 || divisor <= f32::EPSILON {
            // Single node or all nodes at same x
            for node in layout_nodes.iter().filter(|n| n.slot < node_count) {
                out.positions_x[node.slot] = 0.0;
                out.positions_y[node.slot] = 0.0;
                out.node_count += 1;
            }
            return;
        }

        let angular_scale = std::f32::consts::TAU / divisor;
        // Arc-length preservation: grow the innermost ring so one
        // Buchheim x-unit maps to at least one world unit of arc.
        // With a fixed radius, per-node arc spacing shrinks as 1/N
        // and the leaf-heavy outer levels merge into a solid ring
        // at 1K+ nodes.
        let ring_base = (divisor / std::f32::consts::TAU).max(self.config.level_separation);
        for (layout_idx, node) in layout_nodes.iter().enumerate() {
            let slot = node.slot;
            if slot < node_count {
                let normalized_x = final_x[layout_idx] - min_x;
                let angle = normalized_x * angular_scale;
                let radius = ring_base + node.depth as f32 * self.config.level_separation;
                out.positions_x[slot] = radius * angle.cos();
                out.positions_y[slot] = radius * angle.sin();
                out.node_count += 1;
            }
        }

        // Root at center
        if let Some(root_idx) = root_layout_idx {
            let slot = layout_nodes[root_idx].slot;
            if slot < node_count {
                out.positions_x[slot] = 0.0;
                out.positions_y[slot] = 0.0;
            }
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
            let Some(children) = children_map.get(&n) else {
                continue;
            };
            // filter marks nodes visited as a side effect (insert returns
            // false for already-visited children, i.e. cycle back-edges)
            for &child in children.iter().filter(|&&c| visited.insert(c)) {
                count += 1;
                stack.push(child);
            }
        }
        count
    }

    /// Build the layout tree via DFS from root.
    /// Uses an explicit stack so tree depth cannot overflow the call stack,
    /// and a visited set to break cycles (back-edges are skipped).
    fn build_layout_tree(
        root: u32,
        children_map: &HashMap<u32, Vec<u32>>,
        layout_nodes: &mut Vec<LayoutNode>,
        node_to_layout: &mut HashMap<u32, usize>,
        visited: &mut HashSet<u32>,
    ) {
        // (node_id, parent layout index, depth)
        let mut stack: Vec<(u32, Option<usize>, u32)> = vec![(root, None, 0)];

        while let Some((node_id, parent_layout_idx, depth)) = stack.pop() {
            // Cycle detection: skip already-visited nodes
            if !visited.insert(node_id) {
                continue;
            }

            let layout_idx = layout_nodes.len();
            node_to_layout.insert(node_id, layout_idx);

            // Number = index into the parent's accepted-children list (needed
            // for left-sibling lookup and move_subtree spacing). Cycle
            // back-edges never claim an index, so numbers stay contiguous.
            let number = parent_layout_idx
                .map(|p| layout_nodes[p].children.len())
                .unwrap_or(0);

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
                number,
            });

            if let Some(parent_idx) = parent_layout_idx {
                layout_nodes[parent_idx].children.push(layout_idx);
            }

            // Push in reverse so children pop (and get numbered) in order.
            for &child_id in children_map.get(&node_id).into_iter().flatten().rev() {
                stack.push((child_id, Some(layout_idx), depth + 1));
            }
        }
    }

    /// Buchheim first walk: bottom-up assignment of preliminary x-coordinates.
    ///
    /// Iterative post-order traversal (explicit stack) so tree depth cannot
    /// overflow the call stack. Follows the published algorithm: each child
    /// subtree is apportioned against its left-sibling forest as soon as it
    /// finishes, then the node is placed relative to its left sibling (or
    /// centered over its children if it has none).
    fn first_walk(&self, root: usize, nodes: &mut [LayoutNode]) {
        // Per-node traversal state: which child to descend into next, and the
        // running default ancestor for apportion.
        struct Frame {
            v: usize,
            next_child: usize,
            default_ancestor: usize,
        }

        let root_default = nodes[root].children.first().copied().unwrap_or(root);
        let mut stack = vec![Frame {
            v: root,
            next_child: 0,
            default_ancestor: root_default,
        }];

        while !stack.is_empty() {
            let top = stack.len() - 1;
            let v = stack[top].v;
            let cursor = stack[top].next_child;
            let child_count = nodes[v].children.len();

            // A child subtree just finished: apportion it against the forest
            // of its left siblings (canonical order: firstWalk(w); apportion(w)).
            if cursor > 0 {
                let w = nodes[v].children[cursor - 1];
                let da = stack[top].default_ancestor;
                stack[top].default_ancestor = self.apportion(w, da, nodes);
            }

            if cursor < child_count {
                stack[top].next_child += 1;
                let w = nodes[v].children[cursor];
                let w_default = nodes[w].children.first().copied().unwrap_or(w);
                stack.push(Frame {
                    v: w,
                    next_child: 0,
                    default_ancestor: w_default,
                });
                continue;
            }

            // All children processed: assign v's preliminary position.
            self.assign_prelim(v, nodes);
            stack.pop();
        }
    }

    /// Assign v's preliminary position once all its children are placed:
    /// a leaf sits next to its left sibling; an internal node is centered
    /// over its children, shifted right of the left sibling if one exists.
    fn assign_prelim(&self, v: usize, nodes: &mut [LayoutNode]) {
        let child_count = nodes[v].children.len();
        if child_count == 0 {
            nodes[v].prelim = match self.left_sibling(v, nodes) {
                Some(w) => nodes[w].prelim + self.config.sibling_separation,
                None => 0.0,
            };
            return;
        }

        // Distribute extra space evenly among intermediate children
        self.execute_shifts(v, nodes);

        let first_child_prelim = nodes[nodes[v].children[0]].prelim;
        let last_child_prelim = nodes[nodes[v].children[child_count - 1]].prelim;
        let midpoint = (first_child_prelim + last_child_prelim) / 2.0;

        if let Some(w) = self.left_sibling(v, nodes) {
            nodes[v].prelim = nodes[w].prelim + self.config.sibling_separation;
            nodes[v].modifier = nodes[v].prelim - midpoint;
        } else {
            nodes[v].prelim = midpoint;
        }
    }

    /// The sibling immediately to the left of v, if any.
    fn left_sibling(&self, v: usize, nodes: &[LayoutNode]) -> Option<usize> {
        let parent = nodes[v].parent?;
        let number = nodes[v].number;
        if number > 0 {
            Some(nodes[parent].children[number - 1])
        } else {
            None
        }
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

    /// Apportion: resolve overlaps between v's subtree and the forest of its
    /// left siblings by walking the two facing contours level by level.
    /// This is the core of Buchheim's linear-time improvement over Walker's algorithm.
    fn apportion(
        &self,
        v: usize,
        mut default_ancestor: usize,
        nodes: &mut [LayoutNode],
    ) -> usize {
        let Some(left_sibling) = self.left_sibling(v, nodes) else {
            return default_ancestor;
        };

        // Contour walkers (Buchheim's notation in parentheses):
        // v_inner_right: right contour of the left-sibling forest (v_i⁻)
        // v_outer_right: right contour of v's own subtree (v_o⁺)
        // v_inner_left:  left contour of v's own subtree (v_i⁺)
        // v_outer_left:  left contour of the leftmost sibling (v_o⁻)
        let mut v_inner_right = left_sibling;
        let mut v_outer_right = v;
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

            // Record which sibling's subtree the right contour belongs to, so
            // later apportion calls can shift the correct ancestor sibling.
            nodes[v_outer_right].ancestor = v;

            let shift = (nodes[v_inner_right].prelim + s_inner_right)
                - (nodes[v_inner_left].prelim + s_inner_left)
                + self.config.subtree_separation;

            if shift > 0.0 {
                let move_ancestor = self.select_ancestor(v_inner_right, v, default_ancestor, nodes);
                self.move_subtree(move_ancestor, v, shift, nodes);

                s_inner_left += shift;
                s_outer_right += shift;
            }

            s_inner_right += nodes[v_inner_right].modifier;
            s_inner_left += nodes[v_inner_left].modifier;
            s_outer_left += nodes[v_outer_left].modifier;
            s_outer_right += nodes[v_outer_right].modifier;
        }

        // Set threads so later contour walks continue past the shallower
        // subtree's bottom into the deeper one.
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

    /// Buchheim's Ancestor function: use the recorded ancestor of the contour
    /// node if it is a sibling of v (so move_subtree shifts against a real
    /// left sibling); otherwise fall back to the default ancestor.
    fn select_ancestor(
        &self,
        contour_node: usize,
        v: usize,
        default_ancestor: usize,
        nodes: &[LayoutNode],
    ) -> usize {
        let candidate = nodes[contour_node].ancestor;
        if nodes[candidate].parent.is_some() && nodes[candidate].parent == nodes[v].parent {
            candidate
        } else {
            default_ancestor
        }
    }

    /// Move subtree: shift node v and adjust spacing between ancestor and v.
    fn move_subtree(
        &self,
        wl: usize,
        wr: usize,
        shift: f32,
        nodes: &mut [LayoutNode],
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
    fn execute_shifts(&self, v: usize, nodes: &mut [LayoutNode]) {
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
    /// Iterative (explicit stack) so tree depth cannot overflow the call stack.
    fn second_walk_collect(&self, root: usize, nodes: &[LayoutNode], final_x: &mut [f32]) {
        let mut stack: Vec<(usize, f32)> = vec![(root, 0.0)];

        while let Some((v, modifier_sum)) = stack.pop() {
            final_x[v] = nodes[v].prelim + modifier_sum;

            for &child in &nodes[v].children {
                stack.push((child, modifier_sum + nodes[v].modifier));
            }
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
    fn test_cousin_subtrees_do_not_overlap() {
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            coordinate_mode: CoordinateMode::Linear,
            level_separation: 1.0, // Keep output in raw Buchheim x-units
            sibling_separation: 1.0,
            subtree_separation: 2.0,
        });

        // Deep/shallow/deep: root(0) → A(1), B(2 leaf), C(3).
        // A and C both have depth-2 leaves; B has none, so the A/C conflict
        // is only visible via contour threading across the shallow sibling.
        // A(1) → 4, 5, 6; C(3) → 7, 8, 9
        let edges = [0u32, 1, 0, 2, 0, 3, 1, 4, 1, 5, 1, 6, 3, 7, 3, 8, 3, 9];
        let result = layout.compute(10, &edges, Some(0));
        assert_eq!(result.node_count, 10);

        // All depth-2 leaves (A's and C's) must respect the separation floor.
        let depth2 = [4usize, 5, 6, 7, 8, 9];
        for (i, &a) in depth2.iter().enumerate() {
            for &b in &depth2[i + 1..] {
                let gap = (result.positions_x[a] - result.positions_x[b]).abs();
                assert!(
                    gap >= layout.config.sibling_separation - 1e-3,
                    "Depth-2 nodes {a} and {b} too close: gap {gap} \
                     (x = {}, {})",
                    result.positions_x[a],
                    result.positions_x[b]
                );
            }
        }
    }

    #[test]
    fn test_radial_large_tree_outer_ring_spacing() {
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            coordinate_mode: CoordinateMode::Radial,
            level_separation: 80.0,
            sibling_separation: 1.0,
            subtree_separation: 2.0,
        });

        // Root → 40 dirs → 50 leaves each = 2000 leaves at the outer depth.
        let mut edges: Vec<u32> = Vec::new();
        let mut leaf_ids: Vec<usize> = Vec::new();
        let mut next_id = 1u32;
        for _ in 0..40 {
            let dir = next_id;
            next_id += 1;
            edges.push(0);
            edges.push(dir);
            for _ in 0..50 {
                let leaf = next_id;
                next_id += 1;
                edges.push(dir);
                edges.push(leaf);
                leaf_ids.push(leaf as usize);
            }
        }
        let node_count = next_id as usize;

        let result = layout.compute(node_count, &edges, Some(0));
        assert_eq!(result.node_count, node_count);

        // Arc-length preservation: min pairwise world distance among the 2000
        // outer-depth leaves must stay above a floor near sibling_separation,
        // instead of shrinking as 1/N with a fixed-radius ring.
        let mut min_dist_sq = f32::INFINITY;
        for (i, &a) in leaf_ids.iter().enumerate() {
            for &b in &leaf_ids[i + 1..] {
                let dx = result.positions_x[a] - result.positions_x[b];
                let dy = result.positions_y[a] - result.positions_y[b];
                let d = dx * dx + dy * dy;
                min_dist_sq = min_dist_sq.min(d);
            }
        }
        let min_dist = min_dist_sq.sqrt();
        let floor = layout.config.sibling_separation * 0.9;
        assert!(
            min_dist >= floor,
            "Outer-ring leaves too close: min distance {min_dist} < floor {floor}"
        );
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

        let layout = TidyTreeLayout::new(TidyTreeConfig {
            coordinate_mode: CoordinateMode::Linear,
            ..Default::default()
        });
        let result = layout.compute(n as usize, &edges, Some(0));
        assert_eq!(result.node_count, n as usize);
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
