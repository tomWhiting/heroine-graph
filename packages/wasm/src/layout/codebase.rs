//! Codebase circle packing layout algorithm.
//!
//! Produces a nested circle-packing layout for hierarchical codebase graphs.
//! The containment hierarchy (repository→directory→file→symbol) is used to
//! create nested circles where each parent node's circle encloses all its
//! children.
//!
//! # Algorithm
//!
//! 1. **Build hierarchy tree** from containment edges (parent→child).
//! 2. **Bottom-up radius computation**: Leaves get a base radius from their
//!    node category. Internal nodes get a radius computed by packing their
//!    children circles.
//! 3. **Top-down position assignment**: Starting from the root at (0,0),
//!    recursively place children within each parent's circle.
//!
//! # Layout Strategy
//!
//! Children within a parent are arranged using a sunflower spiral, which
//! provides approximately uniform density. The parent's radius is computed
//! as the minimum enclosing circle of all packed children plus padding.

use std::collections::{HashMap, HashSet};

/// Node type categories for layout sizing.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeCategory {
    /// Repository root — largest radius.
    Repository = 0,
    /// Directory — large radius.
    Directory = 1,
    /// File — medium radius.
    File = 2,
    /// Symbol (function, class, method, etc.) — smallest radius.
    Symbol = 3,
    /// Other/unknown type.
    Other = 4,
}

impl From<u8> for NodeCategory {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Repository,
            1 => Self::Directory,
            2 => Self::File,
            3 => Self::Symbol,
            _ => Self::Other,
        }
    }
}

/// Configuration for the codebase circle packing layout.
pub struct CodebaseLayoutConfig {
    /// Padding within directory circles (space between boundary and children).
    pub directory_padding: f32,
    /// Padding within file circles.
    pub file_padding: f32,
    /// Base radius for symbol nodes (smallest elements).
    pub symbol_radius: f32,
    /// Minimum radius for file nodes.
    pub file_radius: f32,
    /// Minimum radius for directory nodes.
    pub directory_radius: f32,
    /// Global scale multiplier applied to all positions.
    pub spread_factor: f32,
}

impl Default for CodebaseLayoutConfig {
    fn default() -> Self {
        Self {
            directory_padding: 15.0,
            file_padding: 8.0,
            symbol_radius: 5.0,
            file_radius: 12.0,
            directory_radius: 25.0,
            spread_factor: 1.5,
        }
    }
}

/// Internal node used during layout computation.
struct LayoutNode {
    /// Original node index (slot in the graph).
    slot: usize,
    /// Node category (affects base radius).
    category: NodeCategory,
    /// Children indices (into the layout_nodes vec).
    children: Vec<usize>,
    /// Computed radius (from bottom-up pass).
    radius: f32,
    /// Final X position (from top-down pass).
    x: f32,
    /// Final Y position (from top-down pass).
    y: f32,
}

/// Compute codebase layout from containment hierarchy.
///
/// # Arguments
///
/// * `containment_edges` - Flat array of [parent0, child0, parent1, child1, ...] pairs
/// * `node_categories` - One u8 per node (0=repo, 1=dir, 2=file, 3=symbol, 4=other)
/// * `node_count` - Total number of node slots
/// * `root_id` - Optional root node ID (None = auto-detect)
/// * `config` - Layout configuration
///
/// # Returns
///
/// A `Vec<f32>` of interleaved target positions [x0, y0, x1, y1, ...].
/// Nodes not in the tree get sentinel values (f32::MAX).
pub fn compute_codebase_layout(
    containment_edges: &[u32],
    node_categories: &[u8],
    node_count: usize,
    root_id: Option<u32>,
    config: &CodebaseLayoutConfig,
) -> Vec<f32> {
    const SENTINEL: f32 = 3.402_823e+38;

    if node_count == 0 {
        return Vec::new();
    }

    let mut positions = vec![SENTINEL; node_count * 2];

    // Validate edge array
    if containment_edges.len() % 2 != 0 {
        return positions;
    }

    // Build parent→children adjacency
    let mut children_map: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut has_parent: HashSet<u32> = HashSet::new();
    let mut all_nodes: HashSet<u32> = HashSet::new();

    let edge_count = containment_edges.len() / 2;
    for i in 0..edge_count {
        let parent = containment_edges[i * 2];
        let child = containment_edges[i * 2 + 1];

        // Validate bounds
        if parent as usize >= node_count || child as usize >= node_count {
            continue;
        }
        // Skip self-loops
        if parent == child {
            continue;
        }

        children_map.entry(parent).or_default().push(child);
        has_parent.insert(child);
        all_nodes.insert(parent);
        all_nodes.insert(child);
    }

    if all_nodes.is_empty() {
        return positions;
    }

    // Find root
    let root = if let Some(r) = root_id {
        r
    } else {
        // Auto-detect: node with no parent and most descendants
        let roots: Vec<u32> = all_nodes
            .iter()
            .filter(|n| !has_parent.contains(n))
            .copied()
            .collect();

        if roots.is_empty() {
            // Cycle — pick lowest ID
            *all_nodes.iter().min().unwrap_or(&0)
        } else if roots.len() == 1 {
            roots[0]
        } else {
            // Multiple roots: pick the one with most descendants
            roots
                .iter()
                .max_by_key(|&&r| count_descendants(r, &children_map))
                .copied()
                .unwrap_or(roots[0])
        }
    };

    // Build layout tree via DFS (with cycle detection)
    let mut layout_nodes: Vec<LayoutNode> = Vec::new();
    let mut node_to_layout: HashMap<u32, usize> = HashMap::new();
    let mut visited: HashSet<u32> = HashSet::new();

    build_layout_tree(
        root,
        node_categories,
        node_count,
        &children_map,
        &mut layout_nodes,
        &mut node_to_layout,
        &mut visited,
    );

    if layout_nodes.is_empty() {
        return positions;
    }

    // Bottom-up pass: compute radii
    compute_radii(0, &mut layout_nodes, config);

    // Top-down pass: assign positions (root at origin)
    layout_nodes[0].x = 0.0;
    layout_nodes[0].y = 0.0;
    assign_positions(0, &mut layout_nodes, config);

    // Write positions to output
    for node in &layout_nodes {
        let idx = node.slot * 2;
        if idx + 1 < positions.len() {
            positions[idx] = node.x * config.spread_factor;
            positions[idx + 1] = node.y * config.spread_factor;
        }
    }

    positions
}

/// Count descendants of a node (for root selection heuristic).
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

/// Build layout tree via DFS with cycle detection.
fn build_layout_tree(
    node_id: u32,
    node_categories: &[u8],
    node_count: usize,
    children_map: &HashMap<u32, Vec<u32>>,
    layout_nodes: &mut Vec<LayoutNode>,
    node_to_layout: &mut HashMap<u32, usize>,
    visited: &mut HashSet<u32>,
) {
    if !visited.insert(node_id) {
        return; // Cycle detected
    }

    let slot = node_id as usize;
    let category = if slot < node_categories.len() {
        NodeCategory::from(node_categories[slot])
    } else if slot < node_count {
        NodeCategory::Other
    } else {
        return; // Out of bounds
    };

    let layout_idx = layout_nodes.len();
    node_to_layout.insert(node_id, layout_idx);

    layout_nodes.push(LayoutNode {
        slot,
        category,
        children: Vec::new(),
        radius: 0.0,
        x: 0.0,
        y: 0.0,
    });

    if let Some(children) = children_map.get(&node_id) {
        let mut child_layout_indices: Vec<usize> = Vec::with_capacity(children.len());

        for &child_id in children {
            let before_len = layout_nodes.len();
            build_layout_tree(
                child_id,
                node_categories,
                node_count,
                children_map,
                layout_nodes,
                node_to_layout,
                visited,
            );
            if layout_nodes.len() > before_len {
                if let Some(&child_idx) = node_to_layout.get(&child_id) {
                    child_layout_indices.push(child_idx);
                }
            }
        }

        layout_nodes[layout_idx].children = child_layout_indices;
    }
}

/// Bottom-up radius computation.
///
/// Leaf nodes get a base radius from their category.
/// Internal nodes get a radius that encloses all children circles.
fn compute_radii(idx: usize, nodes: &mut Vec<LayoutNode>, config: &CodebaseLayoutConfig) {
    // First, recursively compute children's radii
    let children: Vec<usize> = nodes[idx].children.clone();
    for &child_idx in &children {
        compute_radii(child_idx, nodes, config);
    }

    if children.is_empty() {
        // Leaf node: base radius from category
        nodes[idx].radius = base_radius(nodes[idx].category, config);
    } else {
        // Internal node: compute enclosing radius for all children
        // Sum of children circle areas determines minimum enclosing radius
        let total_area: f32 = children.iter()
            .map(|&c| {
                let r = nodes[c].radius;
                std::f32::consts::PI * r * r
            })
            .sum();

        // Enclosing circle radius from total area: A = π * R² → R = √(A/π)
        // Apply a packing efficiency factor (~0.9 for circles)
        let packing_efficiency = 0.82; // Typical for random circle packing
        let enclosing_radius = (total_area / (std::f32::consts::PI * packing_efficiency)).sqrt();

        // Add padding based on node category
        let padding = match nodes[idx].category {
            NodeCategory::Repository | NodeCategory::Directory => config.directory_padding,
            NodeCategory::File => config.file_padding,
            _ => config.file_padding,
        };

        // Ensure minimum radius for the category
        let min_radius = base_radius(nodes[idx].category, config);
        nodes[idx].radius = enclosing_radius.max(min_radius) + padding;
    }
}

/// Get base radius for a node category.
fn base_radius(category: NodeCategory, config: &CodebaseLayoutConfig) -> f32 {
    match category {
        NodeCategory::Repository => config.directory_radius * 2.0,
        NodeCategory::Directory => config.directory_radius,
        NodeCategory::File => config.file_radius,
        NodeCategory::Symbol => config.symbol_radius,
        NodeCategory::Other => config.symbol_radius,
    }
}

/// Top-down position assignment using sunflower spiral within each parent.
fn assign_positions(idx: usize, nodes: &mut Vec<LayoutNode>, config: &CodebaseLayoutConfig) {
    let children: Vec<usize> = nodes[idx].children.clone();

    if children.is_empty() {
        return;
    }

    let parent_x = nodes[idx].x;
    let parent_y = nodes[idx].y;
    let parent_radius = nodes[idx].radius;

    // Determine padding to use
    let padding = match nodes[idx].category {
        NodeCategory::Repository | NodeCategory::Directory => config.directory_padding,
        NodeCategory::File => config.file_padding,
        _ => config.file_padding,
    };

    // Available radius for placing children (subtract padding)
    let available_radius = (parent_radius - padding).max(0.0);

    let n = children.len();

    if n == 1 {
        // Single child: place at parent center
        let child_idx = children[0];
        nodes[child_idx].x = parent_x;
        nodes[child_idx].y = parent_y;
        assign_positions(child_idx, nodes, config);
        return;
    }

    // Sort children by radius (largest first) for better packing
    let mut sorted_children: Vec<(usize, f32)> = children.iter()
        .map(|&c| (c, nodes[c].radius))
        .collect();
    sorted_children.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Use sunflower spiral placement
    // The golden angle ensures approximately uniform distribution
    let golden_angle = std::f32::consts::TAU * (1.0 - 1.0 / ((1.0 + 5.0f32.sqrt()) / 2.0));

    // Compute placement radius: scale based on child sizes relative to parent
    let max_child_radius = sorted_children.iter()
        .map(|&(_, r)| r)
        .fold(0.0f32, f32::max);

    for (i, &(child_idx, child_radius)) in sorted_children.iter().enumerate() {
        // Spiral parameter: how far from center to place this child
        let t = if n <= 2 {
            // For 1-2 children, place at specific positions
            (i as f32 + 1.0) / (n as f32 + 1.0)
        } else {
            (i as f32 + 0.5) / n as f32
        };

        // Distance from parent center, scaled so children don't overlap parent boundary
        let placement_radius = (available_radius - child_radius).max(0.0) * t.sqrt();

        // Angle: golden angle spiral
        let angle = i as f32 * golden_angle;

        nodes[child_idx].x = parent_x + placement_radius * angle.cos();
        nodes[child_idx].y = parent_y + placement_radius * angle.sin();

        // If child overlaps parent boundary, clamp it
        let dist_from_parent = placement_radius + child_radius;
        if dist_from_parent > parent_radius - padding * 0.5 && placement_radius > f32::EPSILON {
            let clamped_dist = (parent_radius - padding * 0.5 - child_radius).max(0.0);
            let scale = clamped_dist / placement_radius;
            nodes[child_idx].x = parent_x + placement_radius * scale * angle.cos();
            nodes[child_idx].y = parent_y + placement_radius * scale * angle.sin();
        }

        // Recurse into child
        assign_positions(child_idx, nodes, config);
    }

    // Avoid overlaps between siblings by checking pairwise distances
    // and pushing apart if needed (single pass relaxation)
    resolve_overlaps(&sorted_children, nodes, parent_x, parent_y, available_radius, max_child_radius);
}

/// Single-pass overlap resolution for sibling circles.
/// Pushes overlapping children apart radially from the parent center.
fn resolve_overlaps(
    children: &[(usize, f32)],
    nodes: &mut Vec<LayoutNode>,
    parent_x: f32,
    parent_y: f32,
    available_radius: f32,
    _max_child_radius: f32,
) {
    let n = children.len();
    if n <= 1 {
        return;
    }

    // Run a few relaxation iterations for better results
    for _ in 0..3 {
        for i in 0..n {
            let (ci, ri) = children[i];
            for j in (i + 1)..n {
                let (cj, rj) = children[j];

                let dx = nodes[cj].x - nodes[ci].x;
                let dy = nodes[cj].y - nodes[ci].y;
                let dist_sq = dx * dx + dy * dy;
                let min_dist = ri + rj;
                let min_dist_sq = min_dist * min_dist;

                if dist_sq < min_dist_sq && dist_sq > f32::EPSILON {
                    let dist = dist_sq.sqrt();
                    let overlap = min_dist - dist;
                    let push = overlap * 0.5;

                    // Push apart along the line connecting their centers
                    let nx = dx / dist;
                    let ny = dy / dist;

                    nodes[ci].x -= nx * push;
                    nodes[ci].y -= ny * push;
                    nodes[cj].x += nx * push;
                    nodes[cj].y += ny * push;

                    // Clamp to stay within parent
                    clamp_to_parent(ci, ri, parent_x, parent_y, available_radius, nodes);
                    clamp_to_parent(cj, rj, parent_x, parent_y, available_radius, nodes);
                }
            }
        }
    }
}

/// Clamp a child's position so it stays within the parent's available radius.
fn clamp_to_parent(
    child_idx: usize,
    child_radius: f32,
    parent_x: f32,
    parent_y: f32,
    available_radius: f32,
    nodes: &mut [LayoutNode],
) {
    let dx = nodes[child_idx].x - parent_x;
    let dy = nodes[child_idx].y - parent_y;
    let dist = (dx * dx + dy * dy).sqrt();
    let max_dist = (available_radius - child_radius).max(0.0);

    if dist > max_dist && dist > f32::EPSILON {
        let scale = max_dist / dist;
        nodes[child_idx].x = parent_x + dx * scale;
        nodes[child_idx].y = parent_y + dy * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let positions = compute_codebase_layout(&[], &[], 0, None, &CodebaseLayoutConfig::default());
        assert!(positions.is_empty());
    }

    #[test]
    fn test_single_root() {
        let edges: [u32; 0] = [];
        let categories = [0u8]; // Repository
        let positions = compute_codebase_layout(&edges, &categories, 1, Some(0), &CodebaseLayoutConfig::default());
        // No edges, so no tree — returns sentinel
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn test_simple_hierarchy() {
        // repo(0) → dir(1) → file(2) → symbol(3)
        let edges = [0u32, 1, 1, 2, 2, 3];
        let categories = [0u8, 1, 2, 3]; // repo, dir, file, symbol
        let config = CodebaseLayoutConfig::default();

        let positions = compute_codebase_layout(&edges, &categories, 4, Some(0), &config);
        assert_eq!(positions.len(), 8);

        let sentinel = 3.402_823e+38_f32;
        // All 4 nodes should have non-sentinel positions
        for i in 0..4 {
            assert!(
                positions[i * 2] < sentinel,
                "Node {i} should have valid x position, got {}",
                positions[i * 2]
            );
            assert!(positions[i * 2].is_finite(), "Node {i} x should be finite");
            assert!(positions[i * 2 + 1].is_finite(), "Node {i} y should be finite");
        }

        // Root should be at origin (approximately, after spread_factor)
        assert!(
            positions[0].abs() < 0.01,
            "Root x should be ~0, got {}",
            positions[0]
        );
        assert!(
            positions[1].abs() < 0.01,
            "Root y should be ~0, got {}",
            positions[1]
        );
    }

    #[test]
    fn test_directory_with_multiple_files() {
        // repo(0) → dir(1) → file(2), file(3), file(4), file(5)
        let edges = [0u32, 1, 1, 2, 1, 3, 1, 4, 1, 5];
        let categories = [0u8, 1, 2, 2, 2, 2];
        let config = CodebaseLayoutConfig::default();

        let positions = compute_codebase_layout(&edges, &categories, 6, Some(0), &config);
        assert_eq!(positions.len(), 12);

        let sentinel = 3.402_823e+38_f32;
        for i in 0..6 {
            assert!(
                positions[i * 2] < sentinel,
                "Node {i} should have valid position"
            );
        }

        // Files should be near their parent directory, not at the exact same point
        let dir_x = positions[2];
        let dir_y = positions[3];
        let mut all_same = true;
        for i in 2..6 {
            let fx = positions[i * 2];
            let fy = positions[i * 2 + 1];
            if (fx - dir_x).abs() > 0.01 || (fy - dir_y).abs() > 0.01 {
                all_same = false;
                break;
            }
        }
        // With 4 files, they shouldn't all be at the exact same position
        assert!(!all_same, "Files should be distributed, not all at same position");
    }

    #[test]
    fn test_auto_root_detection() {
        // 0→1, 0→2 — node 0 has no parent
        let edges = [0u32, 1, 0, 2];
        let categories = [1u8, 2, 2]; // dir, file, file
        let config = CodebaseLayoutConfig::default();

        let positions = compute_codebase_layout(&edges, &categories, 3, None, &config);
        assert_eq!(positions.len(), 6);

        // Root (0) should be at origin
        assert!(positions[0].abs() < 0.01, "Auto-detected root at origin");
    }

    #[test]
    fn test_cycle_does_not_hang() {
        // 0→1→2→0 (cycle)
        let edges = [0u32, 1, 1, 2, 2, 0];
        let categories = [1u8, 2, 3];
        let config = CodebaseLayoutConfig::default();

        // Should not hang
        let positions = compute_codebase_layout(&edges, &categories, 3, Some(0), &config);
        assert_eq!(positions.len(), 6);
    }

    #[test]
    fn test_self_loop_skipped() {
        let edges = [0u32, 0, 0, 1]; // self-loop + valid edge
        let categories = [1u8, 2];
        let config = CodebaseLayoutConfig::default();

        let positions = compute_codebase_layout(&edges, &categories, 2, Some(0), &config);
        let sentinel = 3.402_823e+38_f32;
        for i in 0..2 {
            assert!(positions[i * 2] < sentinel, "Node {i} should have valid position");
        }
    }

    #[test]
    fn test_out_of_bounds_edges_ignored() {
        let edges = [0u32, 1, 0, 999]; // node 999 doesn't exist
        let categories = [1u8, 2];
        let config = CodebaseLayoutConfig::default();

        let positions = compute_codebase_layout(&edges, &categories, 2, Some(0), &config);
        let sentinel = 3.402_823e+38_f32;
        // Node 0 and 1 should have valid positions
        assert!(positions[0] < sentinel);
        assert!(positions[2] < sentinel);
    }

    #[test]
    fn test_odd_edge_array() {
        let edges = [0u32, 1, 2]; // Odd length = invalid
        let categories = [1u8, 2, 3];
        let config = CodebaseLayoutConfig::default();

        let positions = compute_codebase_layout(&edges, &categories, 3, None, &config);
        // Should return sentinel positions (no valid edges)
        let sentinel = 3.402_823e+38_f32;
        assert!(positions[0] >= sentinel, "Invalid edge array should produce sentinel");
    }

    #[test]
    fn test_large_codebase_hierarchy() {
        // Simulate: 1 repo → 10 dirs → 10 files each → 5 symbols each = 1 + 10 + 100 + 500 = 611 nodes
        let n = 611;
        let mut edges = Vec::new();
        let mut categories = vec![0u8; n];

        // Root = repo
        categories[0] = 0;
        let mut next_id = 1u32;

        // 10 directories
        for _ in 0..10 {
            let dir_id = next_id;
            categories[dir_id as usize] = 1;
            edges.push(0);
            edges.push(dir_id);
            next_id += 1;

            // 10 files per directory
            for _ in 0..10 {
                let file_id = next_id;
                categories[file_id as usize] = 2;
                edges.push(dir_id);
                edges.push(file_id);
                next_id += 1;

                // 5 symbols per file
                for _ in 0..5 {
                    let sym_id = next_id;
                    categories[sym_id as usize] = 3;
                    edges.push(file_id);
                    edges.push(sym_id);
                    next_id += 1;
                }
            }
        }

        assert_eq!(next_id as usize, n);

        let config = CodebaseLayoutConfig::default();
        let positions = compute_codebase_layout(&edges, &categories, n, Some(0), &config);
        assert_eq!(positions.len(), n * 2);

        let sentinel = 3.402_823e+38_f32;
        let valid_count = (0..n)
            .filter(|&i| positions[i * 2] < sentinel)
            .count();
        assert_eq!(valid_count, n, "All {} nodes should have valid positions", n);

        // Verify all positions are finite
        for i in 0..n {
            assert!(
                positions[i * 2].is_finite() && positions[i * 2 + 1].is_finite(),
                "Node {} should have finite positions",
                i
            );
        }
    }

    #[test]
    fn test_children_within_parent_radius() {
        // Simple test: 1 dir with 3 files
        let edges = [0u32, 1, 0, 2, 0, 3];
        let categories = [1u8, 2, 2, 2];
        let config = CodebaseLayoutConfig {
            spread_factor: 1.0, // No scaling for easier testing
            ..Default::default()
        };

        let positions = compute_codebase_layout(&edges, &categories, 4, Some(0), &config);

        let px = positions[0];
        let py = positions[1];

        // Each child should be within reasonable distance of parent
        for i in 1..4 {
            let cx = positions[i * 2];
            let cy = positions[i * 2 + 1];
            let dist = ((cx - px).powi(2) + (cy - py).powi(2)).sqrt();
            assert!(
                dist < 200.0,
                "Child {} should be near parent (dist={}), positions: ({}, {}) vs ({}, {})",
                i, dist, cx, cy, px, py
            );
        }
    }
}
