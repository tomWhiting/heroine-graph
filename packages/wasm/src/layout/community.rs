//! Multi-level Louvain community detection and circular community layout.
//!
//! Implements the full multi-level Louvain modularity optimization algorithm
//! for community detection, then places communities in a circular arrangement
//! with nodes distributed within each community's region.
//!
//! # Algorithm Overview
//!
//! **Multi-Level Louvain:**
//! 1. **Phase 1 (Local Moving):** Each node starts in its own community.
//!    For each node, compute modularity gain of moving to each neighbor's
//!    community. Move to best positive-gain community. Repeat until convergence.
//! 2. **Phase 2 (Aggregation):** Collapse each community into a super-node.
//!    Edge weights between super-nodes = sum of inter-community edge weights.
//!    Self-loops = sum of intra-community edge weights.
//! 3. Repeat from Phase 1 on the coarsened graph until no further reduction.
//! 4. Map multi-level assignments back to original node IDs.
//!
//! This multi-level approach is critical for tree-structured graphs where
//! single-level Louvain produces thousands of tiny communities. The coarsening
//! step iteratively merges these into meaningful larger clusters.
//!
//! **Layout:**
//! 1. Arrange community centers on a circle, sized proportional to member count.
//! 2. Place nodes within each community using a spiral layout for even spacing.
//!
//! # References
//!
//! - Blondel et al., "Fast unfolding of communities in large networks" (2008)

use std::collections::HashMap;

/// Result of community detection.
pub struct CommunityResult {
    /// Community assignment per node (indexed by node slot).
    /// Value is the community ID (0-based, contiguous after compaction).
    pub assignments: Vec<u32>,
    /// Number of distinct communities found.
    pub community_count: u32,
    /// Final modularity score (Q ∈ [-0.5, 1.0]).
    pub modularity: f64,
}

/// Configuration for community layout.
pub struct CommunityLayoutConfig {
    /// Louvain resolution parameter (default: 1.0).
    /// Higher values produce more, smaller communities.
    pub resolution: f32,
    /// Maximum Louvain iterations (default: 100).
    pub max_iterations: u32,
    /// Convergence threshold for modularity gain (default: 0.0001).
    pub min_modularity_gain: f64,
    /// Spacing between community cluster centers (default: 50.0).
    pub community_spacing: f32,
    /// Spacing between nodes within a community (default: 10.0).
    pub node_spacing: f32,
    /// Global scale multiplier (default: 1.5).
    pub spread_factor: f32,
}

impl Default for CommunityLayoutConfig {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            max_iterations: 100,
            min_modularity_gain: 0.0001,
            community_spacing: 50.0,
            node_spacing: 10.0,
            spread_factor: 1.5,
        }
    }
}

/// Adjacency representation for Louvain: CSR-like per-node neighbor lists.
/// Stores both outgoing and incoming edges as undirected for modularity.
struct AdjacencyList {
    /// For each node: list of (neighbor_id, edge_weight) pairs.
    neighbors: Vec<Vec<(usize, f64)>>,
    /// Total edge weight (sum of all edge weights, counting each directed edge once).
    total_weight: f64,
    /// Degree (weighted) of each node: sum of edge weights incident to this node.
    degree: Vec<f64>,
}

impl AdjacencyList {
    /// Build adjacency from CSR data.
    ///
    /// CSR format: [offsets...(node_count+1 elements), targets...]
    /// Treats the directed graph as undirected for modularity computation:
    /// each directed edge A→B contributes weight to both A and B.
    fn from_csr(csr: &[u32], node_count: usize) -> Self {
        if csr.len() <= node_count + 1 {
            return Self {
                neighbors: vec![Vec::new(); node_count],
                total_weight: 0.0,
                degree: vec![0.0; node_count],
            };
        }

        let offsets = &csr[..node_count + 1];
        let targets = &csr[node_count + 1..];

        let mut neighbors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); node_count];
        let mut degree = vec![0.0f64; node_count];
        let mut total_weight = 0.0f64;

        // Build undirected adjacency from directed edges.
        // For modularity, we treat A→B as an undirected edge with weight 1.0.
        // If both A→B and B→A exist, that's weight 2.0 between them.
        for src in 0..node_count {
            let start = offsets[src] as usize;
            let end = offsets[src + 1] as usize;
            for i in start..end.min(targets.len()) {
                let tgt = targets[i] as usize;
                if tgt >= node_count {
                    continue;
                }
                let w = 1.0f64; // All edges have weight 1.0 in our graph

                // Add forward edge A→B
                neighbors[src].push((tgt, w));
                degree[src] += w;

                // Add reverse edge B→A (undirected treatment)
                neighbors[tgt].push((src, w));
                degree[tgt] += w;

                total_weight += w;
            }
        }

        Self {
            neighbors,
            total_weight,
            degree,
        }
    }
}

/// Build a coarsened graph from community assignments.
///
/// Each community becomes a super-node. Edge weight between super-nodes is
/// the sum of edge weights between their member nodes. Internal edges become
/// self-loops (which contribute to sigma_in for the next Louvain pass).
fn coarsen_graph(
    adj: &AdjacencyList,
    community: &[usize],
    num_communities: usize,
) -> AdjacencyList {
    // Accumulate weighted edges between communities
    // Using HashMap<(src_comm, tgt_comm), weight>
    let mut inter_edges: HashMap<(usize, usize), f64> = HashMap::new();
    let node_count = community.len();

    for src in 0..node_count {
        let src_comm = community[src];
        for &(tgt, w) in &adj.neighbors[src] {
            let tgt_comm = community[tgt];
            // Only count each directed edge once (the adjacency stores both directions)
            // We add the full weight; since both (src→tgt) and (tgt→src) are in adj,
            // inter_edges will accumulate both directions automatically.
            *inter_edges.entry((src_comm, tgt_comm)).or_insert(0.0) += w;
        }
    }

    let mut neighbors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_communities];
    let mut degree = vec![0.0f64; num_communities];
    let mut total_weight = 0.0f64;

    for (&(src_comm, tgt_comm), &w) in &inter_edges {
        if src_comm == tgt_comm {
            // Self-loop: internal edges. Count only once for total_weight.
            // The degree contribution is already handled by both directed edges.
            // For the coarsened graph, self-loops contribute to degree but not neighbors.
            degree[src_comm] += w;
            // Self-loops count as half for total_weight (they're double-counted in the sum)
            total_weight += w / 2.0;
        } else {
            neighbors[src_comm].push((tgt_comm, w));
            degree[src_comm] += w;
            // Each inter-community edge pair (A→B, B→A) contributes once to total_weight.
            // Since we iterate all (src_comm, tgt_comm) pairs including both directions,
            // count each direction as half.
            total_weight += w / 2.0;
        }
    }

    AdjacencyList {
        neighbors,
        total_weight,
        degree,
    }
}

/// Run Phase 1 of Louvain: local moving optimization.
///
/// Returns the community assignment for each node (0-indexed, NOT compacted).
fn louvain_local_moving(
    adj: &AdjacencyList,
    node_count: usize,
    resolution: f64,
    max_iterations: u32,
    min_modularity_gain: f64,
) -> Vec<usize> {
    if adj.total_weight < f64::EPSILON {
        return (0..node_count).collect();
    }

    let m2 = 2.0 * adj.total_weight;

    // Initialize: each node in its own community
    let mut community: Vec<usize> = (0..node_count).collect();
    let mut sigma_tot: Vec<f64> = adj.degree.clone();
    let mut sigma_in: Vec<f64> = vec![0.0; node_count];

    // For the coarsened graph, self-loops represent internal edges from previous level.
    // Initialize sigma_in from self-loops in adjacency.
    for node in 0..node_count {
        for &(neighbor, weight) in &adj.neighbors[node] {
            if neighbor == node {
                sigma_in[node] += weight;
            }
        }
    }

    let mut improved = true;
    let mut iteration = 0u32;

    while improved && iteration < max_iterations {
        improved = false;
        iteration += 1;
        let mut total_gain = 0.0f64;

        for node in 0..node_count {
            let node_comm = community[node];
            let k_i = adj.degree[node];

            if k_i < f64::EPSILON {
                continue;
            }

            // Compute edge weights to each neighboring community
            let mut comm_weights: HashMap<usize, f64> = HashMap::new();
            for &(neighbor, weight) in &adj.neighbors[node] {
                let neighbor_comm = community[neighbor];
                *comm_weights.entry(neighbor_comm).or_insert(0.0) += weight;
            }

            let k_i_in = comm_weights.get(&node_comm).copied().unwrap_or(0.0);

            // Remove node from its current community
            sigma_tot[node_comm] -= k_i;
            sigma_in[node_comm] -= 2.0 * k_i_in;

            // Find the best community to move to
            let mut best_comm = node_comm;
            let mut best_gain = 0.0f64;

            for (&target_comm, &k_i_to_c) in &comm_weights {
                let delta_q = k_i_to_c / m2
                    - resolution * sigma_tot[target_comm] * k_i / (m2 * m2);
                let delta_q_back = k_i_in / m2
                    - resolution * sigma_tot[node_comm] * k_i / (m2 * m2);
                let net_gain = delta_q - delta_q_back;

                if net_gain > best_gain {
                    best_gain = net_gain;
                    best_comm = target_comm;
                }
            }

            // Move node to best community
            community[node] = best_comm;
            let k_i_to_best = comm_weights.get(&best_comm).copied().unwrap_or(0.0);
            sigma_tot[best_comm] += k_i;
            sigma_in[best_comm] += 2.0 * k_i_to_best;

            if best_comm != node_comm {
                improved = true;
                total_gain += best_gain;
            }
        }

        if total_gain < min_modularity_gain {
            break;
        }
    }

    community
}

/// Compact community IDs to be contiguous (0, 1, 2, ...).
///
/// Returns (compacted_assignments, num_communities).
fn compact_communities(community: &[usize]) -> (Vec<usize>, usize) {
    let mut id_map: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0usize;

    let compacted: Vec<usize> = community
        .iter()
        .map(|&comm| {
            *id_map.entry(comm).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect();

    (compacted, next_id)
}

/// Map multi-level community assignments back to original node IDs.
///
/// Given a stack of level mappings, traces each original node through
/// all levels to find its final community.
fn map_levels_to_original(levels: &[Vec<usize>], node_count: usize) -> Vec<u32> {
    let mut assignments = vec![0u32; node_count];
    for node in 0..node_count {
        let mut comm = node;
        for level in levels {
            comm = level[comm];
        }
        assignments[node] = comm as u32;
    }

    // Re-compact to contiguous IDs
    let mut id_map: HashMap<u32, u32> = HashMap::new();
    let mut next_id = 0u32;
    for a in &mut assignments {
        let compacted = *id_map.entry(*a).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        *a = compacted;
    }
    assignments
}

/// Detect communities using the full multi-level Louvain algorithm.
///
/// Iteratively applies Phase 1 (local moving) then Phase 2 (aggregation),
/// tracking modularity at each level to select the best partition. This
/// prevents over-merging on tree-structured graphs where Louvain would
/// otherwise collapse everything into a single community.
///
/// # Arguments
///
/// * `csr` - Graph edges in CSR format: [offsets..., targets...]
/// * `node_count` - Number of nodes in the graph
/// * `resolution` - Resolution parameter (1.0 = standard modularity)
/// * `max_iterations` - Maximum number of Louvain iterations per level
/// * `min_modularity_gain` - Convergence threshold
///
/// # Returns
///
/// A `CommunityResult` with community assignments and metadata.
pub fn detect_communities(
    csr: &[u32],
    node_count: usize,
    resolution: f32,
    max_iterations: u32,
    min_modularity_gain: f64,
) -> CommunityResult {
    if node_count == 0 {
        return CommunityResult {
            assignments: Vec::new(),
            community_count: 0,
            modularity: 0.0,
        };
    }

    let orig_adj = AdjacencyList::from_csr(csr, node_count);

    // Handle degenerate case: no edges
    if orig_adj.total_weight < f64::EPSILON {
        let assignments: Vec<u32> = (0..node_count as u32).collect();
        return CommunityResult {
            assignments,
            community_count: node_count as u32,
            modularity: 0.0,
        };
    }

    let resolution_f64 = resolution as f64;

    // Multi-level Louvain with modularity tracking.
    // We evaluate modularity on the original graph at each level and keep
    // the partition that maximizes it. This prevents over-coarsening on
    // tree-structured graphs where unchecked merging collapses to 1 community.
    let mut levels: Vec<Vec<usize>> = Vec::new();
    let mut current_adj = AdjacencyList::from_csr(csr, node_count);
    let mut current_node_count = node_count;
    let max_levels = 20;

    // Track the best result across all levels
    let mut best_assignments: Vec<u32> = (0..node_count as u32).collect();
    let mut best_community_count = node_count as u32;
    let mut best_modularity = -1.0f64;

    for _level in 0..max_levels {
        // Phase 1: Local moving
        let community = louvain_local_moving(
            &current_adj,
            current_node_count,
            resolution_f64,
            max_iterations,
            min_modularity_gain,
        );

        // Compact community IDs
        let (compacted, num_communities) = compact_communities(&community);

        // If no reduction happened, we're done
        if num_communities >= current_node_count {
            break;
        }

        levels.push(compacted.clone());

        // Map current levels back to original nodes and evaluate modularity
        let candidate = map_levels_to_original(&levels, node_count);
        let candidate_count = *candidate.iter().max().unwrap_or(&0) + 1;
        let candidate_mod = compute_modularity(
            &candidate,
            candidate_count,
            &orig_adj,
            resolution_f64,
        );

        // Keep this level if it improves modularity (and has more than 1 community)
        if candidate_mod > best_modularity && candidate_count > 1 {
            best_assignments = candidate;
            best_community_count = candidate_count;
            best_modularity = candidate_mod;
        }

        // If modularity dropped significantly, further coarsening is harmful — stop
        if candidate_mod < best_modularity - 0.01 {
            break;
        }

        // Phase 2: Coarsen the graph
        let new_adj = coarsen_graph(&current_adj, &compacted, num_communities);
        current_adj = new_adj;
        current_node_count = num_communities;
    }

    CommunityResult {
        assignments: best_assignments,
        community_count: best_community_count,
        modularity: best_modularity,
    }
}

/// Compute modularity Q for a given community assignment.
///
/// Q = (1/2m) * Σ_ij [A_ij - resolution * k_i * k_j / (2m)] * δ(c_i, c_j)
fn compute_modularity(
    assignments: &[u32],
    community_count: u32,
    adj: &AdjacencyList,
    resolution: f64,
) -> f64 {
    if adj.total_weight < f64::EPSILON {
        return 0.0;
    }

    let m2 = 2.0 * adj.total_weight;

    // Accumulate per-community: internal weight and total degree
    let mut internal_weight = vec![0.0f64; community_count as usize];
    let mut community_degree = vec![0.0f64; community_count as usize];

    let node_count = assignments.len();
    for node in 0..node_count {
        let c = assignments[node] as usize;
        community_degree[c] += adj.degree[node];

        for &(neighbor, weight) in &adj.neighbors[node] {
            if assignments[neighbor] == assignments[node] {
                internal_weight[c] += weight;
            }
        }
    }

    // Each internal edge is counted twice (once from each endpoint)
    // so internal_weight[c] is already 2 * actual internal weight.

    let mut q = 0.0f64;
    for c in 0..community_count as usize {
        let l_c = internal_weight[c] / 2.0; // Actual internal weight
        let d_c = community_degree[c];
        q += l_c / adj.total_weight - resolution * (d_c / m2).powi(2);
    }

    q
}

/// Compute layout positions from community assignments.
///
/// Communities are arranged in a circle with radius proportional to total
/// node count. Nodes within each community are placed in a spiral pattern
/// for even distribution.
///
/// # Arguments
///
/// * `assignments` - Community ID per node (from `detect_communities`)
/// * `community_count` - Number of distinct communities
/// * `node_count` - Total number of nodes
/// * `config` - Layout configuration parameters
///
/// # Returns
///
/// A `Vec<f32>` of interleaved target positions [x0, y0, x1, y1, ...].
/// Sentinel value (f32::MAX) is used for invalid/absent nodes.
pub fn compute_community_layout(
    assignments: &[u32],
    community_count: u32,
    node_count: usize,
    config: &CommunityLayoutConfig,
) -> Vec<f32> {
    const SENTINEL: f32 = 3.402_823e+38;

    if node_count == 0 || community_count == 0 {
        return Vec::new();
    }

    let mut positions = vec![SENTINEL; node_count * 2];

    // Gather nodes per community
    let mut community_members: Vec<Vec<usize>> = vec![Vec::new(); community_count as usize];
    for (node, &comm) in assignments.iter().enumerate() {
        if node < node_count && (comm as usize) < community_members.len() {
            community_members[comm as usize].push(node);
        }
    }

    // Compute community center positions on a circle.
    // The circle radius scales with the total number of nodes and spacing.
    let base_radius = if community_count <= 1 {
        0.0
    } else {
        // Circumference should be large enough to space communities apart.
        // Each community gets an arc proportional to its member count.
        let total_arc = community_members.iter()
            .map(|m| {
                let r = community_inner_radius(m.len(), config.node_spacing);
                r * 2.0 + config.community_spacing
            })
            .sum::<f32>();
        total_arc / std::f32::consts::TAU
    };

    let outer_radius = base_radius * config.spread_factor;

    // Place community centers along the circle
    let mut angle = 0.0f32;
    let total_weighted_count: f32 = community_members.iter()
        .map(|m| m.len() as f32)
        .sum();

    // Prevent division by zero for empty graphs
    let total_weighted_count = if total_weighted_count < 1.0 { 1.0 } else { total_weighted_count };

    for comm_id in 0..community_count as usize {
        let members = &community_members[comm_id];
        if members.is_empty() {
            continue;
        }

        // Fraction of circle this community occupies (weighted by size)
        let fraction = members.len() as f32 / total_weighted_count;
        let center_angle = angle + fraction * std::f32::consts::TAU / 2.0;

        let cx = outer_radius * center_angle.cos();
        let cy = outer_radius * center_angle.sin();

        // Place nodes within the community using spiral layout
        let inner_radius = community_inner_radius(members.len(), config.node_spacing);
        place_nodes_in_community(members, cx, cy, inner_radius, config, &mut positions);

        angle += fraction * std::f32::consts::TAU;
    }

    // Normalize positions to a target bounding radius.
    // Without normalization, the outer radius grows linearly with community count,
    // which causes layouts to spread far beyond the viewport for graphs with many
    // communities (common in trees). Target radius scales as sqrt(N) * spacing * spread.
    normalize_positions(&mut positions, node_count, config);

    positions
}

/// Normalize all non-sentinel positions so the layout fits within a target radius.
///
/// Target radius = `node_spacing * sqrt(node_count) * spread_factor`.
/// This ensures the layout scales predictably regardless of the number of communities.
fn normalize_positions(positions: &mut [f32], node_count: usize, config: &CommunityLayoutConfig) {
    const SENTINEL: f32 = 3.402_823e+38;

    if node_count == 0 {
        return;
    }

    // Find the maximum distance from origin across all placed nodes
    let mut max_dist_sq: f32 = 0.0;
    for i in 0..node_count {
        let idx = i * 2;
        if idx + 1 >= positions.len() {
            break;
        }
        let x = positions[idx];
        let y = positions[idx + 1];
        if x >= SENTINEL * 0.5 || y >= SENTINEL * 0.5 {
            continue; // Skip unplaced sentinel nodes
        }
        let d = x * x + y * y;
        if d > max_dist_sq {
            max_dist_sq = d;
        }
    }

    let max_dist = max_dist_sq.sqrt();
    if max_dist < 1.0 {
        return; // Layout already tiny or single-community, no normalization needed
    }

    // Target radius: proportional to sqrt(N), giving a visually balanced density
    let target_radius = config.node_spacing * (node_count as f32).sqrt() * config.spread_factor;
    let scale = target_radius / max_dist;

    // Only normalize if layout is significantly larger than target (avoid shrinking compact layouts)
    if scale >= 1.0 {
        return;
    }

    for i in 0..node_count {
        let idx = i * 2;
        if idx + 1 >= positions.len() {
            break;
        }
        if positions[idx] >= SENTINEL * 0.5 {
            continue;
        }
        positions[idx] *= scale;
        positions[idx + 1] *= scale;
    }
}

/// Compute the inner radius needed to fit `n` nodes with given spacing.
/// Uses the area of a circle: A = π * r² = n * spacing² → r = spacing * √(n/π)
fn community_inner_radius(n: usize, node_spacing: f32) -> f32 {
    if n <= 1 {
        return 0.0;
    }
    node_spacing * (n as f32 / std::f32::consts::PI).sqrt()
}

/// Place nodes within a community using a sunflower spiral.
///
/// The sunflower spiral (Fermat's spiral with golden angle) provides
/// approximately uniform density distribution within a circle.
fn place_nodes_in_community(
    members: &[usize],
    cx: f32,
    cy: f32,
    radius: f32,
    config: &CommunityLayoutConfig,
    positions: &mut [f32],
) {
    let n = members.len();

    if n == 0 {
        return;
    }

    // Single node: place at center
    if n == 1 {
        let node = members[0];
        let idx = node * 2;
        if idx + 1 < positions.len() {
            positions[idx] = cx;
            positions[idx + 1] = cy;
        }
        return;
    }

    // Sunflower spiral: angle = i * golden_angle, r = sqrt(i/n) * max_radius
    let golden_angle = std::f32::consts::TAU / (1.0 + 5.0f32.sqrt()); // ~2.3999...
    let scaled_radius = radius * config.spread_factor;

    for (i, &node) in members.iter().enumerate() {
        let idx = node * 2;
        if idx + 1 >= positions.len() {
            continue;
        }

        let t = (i as f32 + 0.5) / n as f32; // 0..1, offset by 0.5 for better distribution
        let r = scaled_radius * t.sqrt();
        let theta = i as f32 * golden_angle;

        positions[idx] = cx + r * theta.cos();
        positions[idx + 1] = cy + r * theta.sin();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple CSR from edge pairs for testing.
    fn build_csr(node_count: usize, edges: &[(u32, u32)]) -> Vec<u32> {
        let mut offsets = vec![0u32; node_count + 1];
        let mut targets = Vec::new();

        // Count edges per source
        for &(src, _) in edges {
            if (src as usize) < node_count {
                offsets[src as usize + 1] += 1;
            }
        }

        // Prefix sum
        for i in 1..=node_count {
            offsets[i] += offsets[i - 1];
        }

        // Build targets
        targets.resize(edges.len(), 0u32);
        let mut current = offsets[..node_count].to_vec();
        for &(src, tgt) in edges {
            let s = src as usize;
            if s < node_count {
                let offset = current[s] as usize;
                if offset < targets.len() {
                    targets[offset] = tgt;
                    current[s] += 1;
                }
            }
        }

        let mut result = Vec::with_capacity(offsets.len() + targets.len());
        result.extend(offsets);
        result.extend(targets);
        result
    }

    #[test]
    fn test_empty_graph() {
        let result = detect_communities(&[], 0, 1.0, 100, 0.0001);
        assert_eq!(result.community_count, 0);
        assert!(result.assignments.is_empty());
    }

    #[test]
    fn test_single_node_no_edges() {
        let csr = build_csr(1, &[]);
        let result = detect_communities(&csr, 1, 1.0, 100, 0.0001);
        assert_eq!(result.community_count, 1);
        assert_eq!(result.assignments.len(), 1);
    }

    #[test]
    fn test_two_disconnected_components() {
        // Two cliques: {0,1,2} and {3,4,5}, fully connected within each
        let edges = [
            (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
            (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4),
        ];
        let csr = build_csr(6, &edges);
        let result = detect_communities(&csr, 6, 1.0, 100, 0.0001);

        // Should detect 2 communities
        assert_eq!(result.community_count, 2, "Expected 2 communities, got {}", result.community_count);

        // Nodes 0,1,2 should be in same community
        assert_eq!(result.assignments[0], result.assignments[1]);
        assert_eq!(result.assignments[1], result.assignments[2]);

        // Nodes 3,4,5 should be in same community
        assert_eq!(result.assignments[3], result.assignments[4]);
        assert_eq!(result.assignments[4], result.assignments[5]);

        // The two groups should be in different communities
        assert_ne!(result.assignments[0], result.assignments[3]);

        // Modularity should be positive for well-separated communities
        assert!(result.modularity > 0.0, "Modularity should be positive, got {}", result.modularity);
    }

    #[test]
    fn test_fully_connected() {
        // Fully connected 4-node graph (K4): modularity is 0 for any partition.
        // The Louvain algorithm correctly recognizes there is no community structure
        // in a perfectly symmetric graph. Each node may remain in its own community
        // or merge — both are valid since ΔQ ≈ 0 in either case.
        let edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 3),
            (3, 0), (3, 1), (3, 2),
        ];
        let csr = build_csr(4, &edges);
        let result = detect_communities(&csr, 4, 1.0, 100, 0.0001);

        // Should produce valid assignments (every node has a community)
        assert_eq!(result.assignments.len(), 4);
        assert!(result.community_count >= 1);
        assert!(result.community_count <= 4);
    }

    #[test]
    fn test_linear_chain() {
        // Chain: 0→1→2→3→4
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4)];
        let csr = build_csr(5, &edges);
        let result = detect_communities(&csr, 5, 1.0, 100, 0.0001);

        // Should converge without error
        assert_eq!(result.assignments.len(), 5);
        assert!(result.community_count >= 1);
        assert!(result.community_count <= 5);
    }

    #[test]
    fn test_resolution_affects_community_count() {
        // Two loosely connected cliques with a bridge edge
        let edges = [
            (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
            (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4),
            (2, 3), // bridge
        ];
        let csr = build_csr(6, &edges);

        let low_res = detect_communities(&csr, 6, 0.5, 100, 0.0001);
        let high_res = detect_communities(&csr, 6, 2.0, 100, 0.0001);

        // Higher resolution should tend to produce more communities
        assert!(
            high_res.community_count >= low_res.community_count,
            "Higher resolution should produce >= communities: low={}, high={}",
            low_res.community_count,
            high_res.community_count,
        );
    }

    #[test]
    fn test_community_layout_produces_valid_positions() {
        let assignments = vec![0, 0, 0, 1, 1, 1];
        let config = CommunityLayoutConfig::default();
        let positions = compute_community_layout(&assignments, 2, 6, &config);

        assert_eq!(positions.len(), 12); // 6 nodes * 2 coords

        let sentinel = 3.402_823e+38_f32;
        for i in 0..6 {
            let x = positions[i * 2];
            let y = positions[i * 2 + 1];
            assert!(x < sentinel, "Node {i} should have valid x position");
            assert!(y < sentinel, "Node {i} should have valid y position");
            assert!(x.is_finite(), "Node {i} x should be finite");
            assert!(y.is_finite(), "Node {i} y should be finite");
        }
    }

    #[test]
    fn test_community_layout_separates_clusters() {
        let assignments = vec![0, 0, 0, 1, 1, 1];
        let config = CommunityLayoutConfig {
            community_spacing: 100.0,
            ..Default::default()
        };
        let positions = compute_community_layout(&assignments, 2, 6, &config);

        // Compute centroid of each community
        let (mut cx0, mut cy0, mut cx1, mut cy1) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for i in 0..3 {
            cx0 += positions[i * 2];
            cy0 += positions[i * 2 + 1];
        }
        for i in 3..6 {
            cx1 += positions[i * 2];
            cy1 += positions[i * 2 + 1];
        }
        cx0 /= 3.0; cy0 /= 3.0;
        cx1 /= 3.0; cy1 /= 3.0;

        let dist = ((cx1 - cx0).powi(2) + (cy1 - cy0).powi(2)).sqrt();
        assert!(dist > 10.0, "Community centroids should be well-separated, got distance {dist}");
    }

    #[test]
    fn test_layout_single_community() {
        let assignments = vec![0, 0, 0, 0];
        let config = CommunityLayoutConfig::default();
        let positions = compute_community_layout(&assignments, 1, 4, &config);

        assert_eq!(positions.len(), 8);
        // All positions should be near the origin (single community at center)
        for i in 0..4 {
            let x = positions[i * 2];
            let y = positions[i * 2 + 1];
            assert!(x.is_finite(), "Node {i} x should be finite");
            assert!(y.is_finite(), "Node {i} y should be finite");
        }
    }

    #[test]
    fn test_large_graph_performance() {
        // 10000 nodes, 5 clear communities connected in a ring
        let n = 10000;
        let communities = 5;
        let per_comm = n / communities;

        let mut edges = Vec::new();

        // Dense edges within each community
        for c in 0..communities {
            let base = c * per_comm;
            for i in 0..per_comm {
                let src = (base + i) as u32;
                // Connect to ~10 random neighbors within community
                for j in 1..=10.min(per_comm - 1) {
                    let tgt = (base + (i + j) % per_comm) as u32;
                    edges.push((src, tgt));
                }
            }
        }

        // Sparse edges between communities (bridge edges)
        for c in 0..communities {
            let next_c = (c + 1) % communities;
            let src = (c * per_comm) as u32;
            let tgt = (next_c * per_comm) as u32;
            edges.push((src, tgt));
        }

        let csr = build_csr(n, &edges);
        let result = detect_communities(&csr, n, 1.0, 50, 0.001);

        // Should detect roughly 5 communities (may merge some due to bridge edges)
        assert!(result.community_count >= 2, "Should detect multiple communities, got {}", result.community_count);
        assert!(result.community_count <= 20, "Should not over-segment, got {} communities", result.community_count);
        assert!(result.modularity > 0.0, "Modularity should be positive for clustered graph");

        // Layout should produce valid positions
        let config = CommunityLayoutConfig::default();
        let positions = compute_community_layout(&result.assignments, result.community_count, n, &config);
        assert_eq!(positions.len(), n * 2);

        let sentinel = 3.402_823e+38_f32;
        let valid_count = (0..n)
            .filter(|&i| positions[i * 2] < sentinel && positions[i * 2 + 1] < sentinel)
            .count();
        assert_eq!(valid_count, n, "All nodes should have valid positions");
    }
}
