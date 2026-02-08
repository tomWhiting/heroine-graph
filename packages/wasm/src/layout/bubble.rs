//! Bubble radius and depth computation for nested bubble layout mode.
//!
//! Computes two per-node values from the graph's containment hierarchy:
//! - **Well radius** (bubble size): Bottom-up from subtree, leaves get a base
//!   radius, internal nodes get `sqrt(sum_child_areas / (pi * packing_eff)) + padding`.
//! - **Depth**: BFS distance from the auto-detected root.
//!
//! These values are uploaded to GPU buffers and used by the Relativity Atlas
//! algorithm's bubble mode for depth-decaying gravity, wellRadius-based phantom
//! zones, and scaled orbit springs.

use std::collections::{HashMap, HashSet, VecDeque};

/// Configuration for bubble radius computation.
pub struct BubbleConfig {
    /// Base radius for leaf nodes (default: 10.0).
    pub base_radius: f32,
    /// Padding added to internal node radii (default: 5.0).
    pub padding: f32,
    /// Packing efficiency for circle packing (default: 0.82).
    pub packing_efficiency: f32,
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

/// Internal node for tree traversal.
struct TreeNode {
    /// Original graph slot index.
    slot: usize,
    /// Children indices (into tree_nodes vec).
    children: Vec<usize>,
    /// Computed bubble radius.
    radius: f32,
    /// Tree depth (0 = root).
    depth: u32,
}

/// Compute bubble data (well radii + depths) from containment hierarchy.
///
/// # Arguments
///
/// * `containment_edges` - Flat `[parent0, child0, parent1, child1, ...]`
/// * `node_count` - Total number of node slots (node_bound)
/// * `root_id` - Optional root node ID (None = auto-detect)
/// * `config` - Bubble configuration
///
/// # Returns
///
/// `Vec<f32>` of length `2 * node_count`:
/// `[wellRadius_0, ..., wellRadius_{n-1}, depth_0_as_f32, ..., depth_{n-1}_as_f32]`
pub fn compute_bubble_data(
    containment_edges: &[u32],
    node_count: usize,
    root_id: Option<u32>,
    config: &BubbleConfig,
) -> Vec<f32> {
    if node_count == 0 {
        return Vec::new();
    }

    // Default values: base_radius for wellRadius, 0.0 for depth
    let mut well_radii = vec![config.base_radius; node_count];
    let mut depths = vec![0.0_f32; node_count];

    if containment_edges.len() < 2 || containment_edges.len() % 2 != 0 {
        let mut result = well_radii;
        result.extend_from_slice(&depths);
        return result;
    }

    // Build parent→children adjacency
    let mut children_map: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut has_parent: HashSet<u32> = HashSet::new();
    let mut all_nodes: HashSet<u32> = HashSet::new();

    let edge_count = containment_edges.len() / 2;
    for i in 0..edge_count {
        let parent = containment_edges[i * 2];
        let child = containment_edges[i * 2 + 1];

        if parent as usize >= node_count || child as usize >= node_count {
            continue;
        }
        if parent == child {
            continue;
        }

        children_map.entry(parent).or_default().push(child);
        has_parent.insert(child);
        all_nodes.insert(parent);
        all_nodes.insert(child);
    }

    if all_nodes.is_empty() {
        let mut result = well_radii;
        result.extend_from_slice(&depths);
        return result;
    }

    // Find root (same heuristic as codebase.rs)
    let root = if let Some(r) = root_id {
        r
    } else {
        let roots: Vec<u32> = all_nodes
            .iter()
            .filter(|n| !has_parent.contains(n))
            .copied()
            .collect();

        if roots.is_empty() {
            *all_nodes.iter().min().unwrap_or(&0)
        } else if roots.len() == 1 {
            roots[0]
        } else {
            roots
                .iter()
                .max_by_key(|&&r| count_descendants(r, &children_map))
                .copied()
                .unwrap_or(roots[0])
        }
    };

    // Build tree via DFS with cycle detection
    let mut tree_nodes: Vec<TreeNode> = Vec::new();
    let mut slot_to_tree: HashMap<u32, usize> = HashMap::new();
    let mut visited: HashSet<u32> = HashSet::new();

    build_tree(
        root,
        node_count,
        &children_map,
        &mut tree_nodes,
        &mut slot_to_tree,
        &mut visited,
    );

    if tree_nodes.is_empty() {
        let mut result = well_radii;
        result.extend_from_slice(&depths);
        return result;
    }

    // Compute depths via BFS from root (index 0 in tree_nodes)
    compute_depths(&mut tree_nodes);

    // Bottom-up radius computation
    compute_radii(0, &mut tree_nodes, config);

    // Write results back to per-slot arrays
    for node in &tree_nodes {
        if node.slot < node_count {
            well_radii[node.slot] = node.radius;
            depths[node.slot] = node.depth as f32;
        }
    }

    // Concatenate: [wellRadii..., depths...]
    let mut result = well_radii;
    result.extend_from_slice(&depths);
    result
}

/// Count descendants for root selection heuristic.
fn count_descendants(node: u32, children_map: &HashMap<u32, Vec<u32>>) -> usize {
    let mut count = 0;
    let mut stack = vec![node];
    let mut visited = HashSet::new();
    visited.insert(node);
    while let Some(n) = stack.pop() {
        let Some(children) = children_map.get(&n) else {
            continue;
        };
        for &child in children {
            if visited.insert(child) {
                count += 1;
                stack.push(child);
            }
        }
    }
    count
}

/// Build tree nodes via DFS with cycle detection.
fn build_tree(
    node_id: u32,
    node_count: usize,
    children_map: &HashMap<u32, Vec<u32>>,
    tree_nodes: &mut Vec<TreeNode>,
    slot_to_tree: &mut HashMap<u32, usize>,
    visited: &mut HashSet<u32>,
) {
    if !visited.insert(node_id) {
        return;
    }

    let slot = node_id as usize;
    if slot >= node_count {
        return;
    }

    let tree_idx = tree_nodes.len();
    slot_to_tree.insert(node_id, tree_idx);

    tree_nodes.push(TreeNode {
        slot,
        children: Vec::new(),
        radius: 0.0,
        depth: 0,
    });

    if let Some(children) = children_map.get(&node_id) {
        let mut child_tree_indices: Vec<usize> = Vec::with_capacity(children.len());

        for &child_id in children {
            let before_len = tree_nodes.len();
            build_tree(
                child_id,
                node_count,
                children_map,
                tree_nodes,
                slot_to_tree,
                visited,
            );
            if tree_nodes.len() > before_len && slot_to_tree.contains_key(&child_id) {
                child_tree_indices.push(slot_to_tree[&child_id]);
            }
        }

        tree_nodes[tree_idx].children = child_tree_indices;
    }
}

/// Compute depths via BFS from root (tree_nodes[0]).
fn compute_depths(tree_nodes: &mut [TreeNode]) {
    if tree_nodes.is_empty() {
        return;
    }

    let mut queue = VecDeque::new();
    tree_nodes[0].depth = 0;
    queue.push_back(0_usize);

    while let Some(idx) = queue.pop_front() {
        let depth = tree_nodes[idx].depth;
        let children: Vec<usize> = tree_nodes[idx].children.clone();
        for child_idx in children {
            tree_nodes[child_idx].depth = depth + 1;
            queue.push_back(child_idx);
        }
    }
}

/// Bottom-up radius computation.
///
/// Leaf nodes get `base_radius`. Internal nodes get a radius that encloses
/// all children circles: `sqrt(sum_areas / (pi * packing_eff)) + padding`.
fn compute_radii(idx: usize, nodes: &mut Vec<TreeNode>, config: &BubbleConfig) {
    let children: Vec<usize> = nodes[idx].children.clone();
    for &child_idx in &children {
        compute_radii(child_idx, nodes, config);
    }

    if children.is_empty() {
        nodes[idx].radius = config.base_radius;
    } else {
        let total_area: f32 = children
            .iter()
            .map(|&c| {
                let r = nodes[c].radius;
                std::f32::consts::PI * r * r
            })
            .sum();

        let enclosing_radius =
            (total_area / (std::f32::consts::PI * config.packing_efficiency)).sqrt();

        nodes[idx].radius = enclosing_radius.max(config.base_radius) + config.padding;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let result = compute_bubble_data(&[], 0, None, &BubbleConfig::default());
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_node_no_edges() {
        let result = compute_bubble_data(&[], 1, None, &BubbleConfig::default());
        assert_eq!(result.len(), 2); // [wellRadius, depth]
        assert_eq!(result[0], 10.0); // base_radius
        assert_eq!(result[1], 0.0); // depth
    }

    #[test]
    fn test_simple_parent_child() {
        // Node 0 -> Node 1
        let edges = [0u32, 1];
        let result = compute_bubble_data(&edges, 2, None, &BubbleConfig::default());
        assert_eq!(result.len(), 4); // 2 radii + 2 depths

        let radii = &result[0..2];
        let depths = &result[2..4];

        // Parent (0) should have larger radius than child (1)
        assert!(radii[0] > radii[1]);
        // Child is leaf, gets base_radius
        assert_eq!(radii[1], 10.0);
        // Root depth = 0, child depth = 1
        assert_eq!(depths[0], 0.0);
        assert_eq!(depths[1], 1.0);
    }

    #[test]
    fn test_wide_tree() {
        // Node 0 -> [1, 2, 3, 4, 5] (root with 5 children)
        let edges = [0u32, 1, 0, 2, 0, 3, 0, 4, 0, 5];
        let result = compute_bubble_data(&edges, 6, None, &BubbleConfig::default());
        assert_eq!(result.len(), 12);

        let radii = &result[0..6];
        let depths = &result[6..12];

        // Root should have much larger radius (encloses 5 children)
        assert!(radii[0] > radii[1]);
        // All children are leaves
        for i in 1..6 {
            assert_eq!(radii[i], 10.0);
            assert_eq!(depths[i], 1.0);
        }
        assert_eq!(depths[0], 0.0);
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
        assert_eq!(result.len(), 10);

        let radii = &result[0..5];
        let depths = &result[5..10];

        // Radii should decrease as we go deeper (less subtree)
        assert!(radii[0] > radii[1]);
        assert!(radii[1] > radii[2]);
        assert!(radii[2] > radii[3]);
        // Node 4 is leaf
        assert_eq!(radii[4], 5.0);

        // Depths should increase
        for i in 0..5 {
            assert_eq!(depths[i], i as f32);
        }
    }

    #[test]
    fn test_cycle_handling() {
        // 0 -> 1 -> 2 -> 0 (cycle)
        let edges = [0u32, 1, 1, 2, 2, 0];
        let result = compute_bubble_data(&edges, 3, None, &BubbleConfig::default());
        assert_eq!(result.len(), 6);
        // Should not panic or infinite loop
    }

    #[test]
    fn test_disconnected_nodes() {
        // Only 0 -> 1, nodes 2 and 3 are disconnected
        let edges = [0u32, 1];
        let result = compute_bubble_data(&edges, 4, None, &BubbleConfig::default());
        assert_eq!(result.len(), 8);

        let radii = &result[0..4];
        let depths = &result[4..8];

        // Disconnected nodes get base_radius and depth 0
        assert_eq!(radii[2], 10.0);
        assert_eq!(radii[3], 10.0);
        assert_eq!(depths[2], 0.0);
        assert_eq!(depths[3], 0.0);
    }

    #[test]
    fn test_explicit_root() {
        // 0 -> 1, 0 -> 2, but we specify root as 1
        let edges = [0u32, 1, 0, 2];
        let result = compute_bubble_data(&edges, 3, Some(0), &BubbleConfig::default());
        assert_eq!(result.len(), 6);

        let depths = &result[3..6];
        assert_eq!(depths[0], 0.0); // root
        assert_eq!(depths[1], 1.0);
        assert_eq!(depths[2], 1.0);
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
        assert_eq!(result.len(), node_count * 2);

        let radii = &result[0..node_count];
        let depths = &result[node_count..];

        // Root has the largest radius
        let max_radius = radii.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(radii[0], max_radius);

        // All leaf files get base_radius
        for file in 11..61 {
            assert_eq!(radii[file], 10.0);
            assert_eq!(depths[file], 2.0);
        }

        // Dirs are at depth 1
        for dir in 1..=10 {
            assert_eq!(depths[dir as usize], 1.0);
            // Dir radius > leaf radius (has 5 children)
            assert!(radii[dir as usize] > 10.0);
        }
    }
}
