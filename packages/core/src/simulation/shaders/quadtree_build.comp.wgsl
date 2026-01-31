// Quadtree Construction Compute Shader for Barnes-Hut
// Builds a proper implicit quadtree with consistent indexing
//
// Tree structure (implicit heap-like indexing):
// - Node i has children at 4*i + 1, 4*i + 2, 4*i + 3, 4*i + 4
// - Node i has parent at (i - 1) / 4
// - Level L has 4^L nodes, starting at index (4^L - 1) / 3
//
// For a tree of depth D:
// - Leaves are at level D-1
// - Total nodes = (4^D - 1) / 3

struct QuadtreeUniforms {
    node_count: u32,
    max_depth: u32,          // Tree depth (e.g., 8 for 65536 leaf cells)
    bounds_min_x: f32,
    bounds_min_y: f32,
    bounds_max_x: f32,
    bounds_max_y: f32,
    root_size: f32,          // bounds_max - bounds_min (assuming square)
    current_level: u32,      // For level-by-level internal node building
}

@group(0) @binding(0) var<uniform> uniforms: QuadtreeUniforms;

// Particle positions (original order)
@group(0) @binding(1) var<storage, read> positions_x: array<f32>;
@group(0) @binding(2) var<storage, read> positions_y: array<f32>;

// Quadtree node data
@group(0) @binding(3) var<storage, read_write> tree_com_x: array<f32>;     // Center of mass X
@group(0) @binding(4) var<storage, read_write> tree_com_y: array<f32>;     // Center of mass Y
@group(0) @binding(5) var<storage, read_write> tree_mass: array<f32>;      // Total mass in cell
@group(0) @binding(6) var<storage, read_write> tree_sizes: array<f32>;     // Cell size at each node

// Atomic counters for aggregation
@group(0) @binding(7) var<storage, read_write> tree_count: array<atomic<u32>>; // Particle count per cell

const WORKGROUP_SIZE: u32 = 256u;

// Get the level of a node given its global index
fn get_level(node_idx: u32) -> u32 {
    // Level 0: nodes 0
    // Level 1: nodes 1-4
    // Level 2: nodes 5-20
    // Level L starts at (4^L - 1) / 3
    var level = 0u;
    var level_start = 0u;
    var level_size = 1u;

    while (node_idx >= level_start + level_size) {
        level_start += level_size;
        level_size *= 4u;
        level += 1u;
    }
    return level;
}

// Get the first node index at a given level
fn level_start_index(level: u32) -> u32 {
    // Sum of 4^0 + 4^1 + ... + 4^(level-1) = (4^level - 1) / 3
    if (level == 0u) {
        return 0u;
    }
    var sum = 0u;
    var power = 1u;
    for (var l = 0u; l < level; l++) {
        sum += power;
        power *= 4u;
    }
    return sum;
}

// Get child index (0-3) within parent's children
fn get_quadrant(pos_x: f32, pos_y: f32, cell_center_x: f32, cell_center_y: f32) -> u32 {
    var q = 0u;
    if (pos_x >= cell_center_x) { q |= 1u; }
    if (pos_y >= cell_center_y) { q |= 2u; }
    return q;
}

// Phase 1: Clear tree (run once with enough threads to cover all tree nodes)
@compute @workgroup_size(256)
fn clear_tree(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    // Calculate max tree nodes: (4^max_depth - 1) / 3
    // For depth 8: (65536 - 1) / 3 = 21845 nodes
    let max_nodes = 262144u; // Safe upper bound

    if (node_idx >= max_nodes) {
        return;
    }

    tree_com_x[node_idx] = 0.0;
    tree_com_y[node_idx] = 0.0;
    tree_mass[node_idx] = 0.0;
    atomicStore(&tree_count[node_idx], 0u);

    // Compute cell size based on level
    let level = get_level(node_idx);
    let size = uniforms.root_size / f32(1u << level);
    tree_sizes[node_idx] = size;
}

// Phase 2: Insert particles into leaf cells
// Each particle finds its leaf cell and atomically adds itself
@compute @workgroup_size(256)
fn insert_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;

    if (particle_idx >= uniforms.node_count) {
        return;
    }

    let pos_x = positions_x[particle_idx];
    let pos_y = positions_y[particle_idx];

    // Traverse tree from root to find leaf cell
    var node_idx = 0u;
    var cell_min_x = uniforms.bounds_min_x;
    var cell_min_y = uniforms.bounds_min_y;
    var cell_size = uniforms.root_size;

    for (var level = 0u; level < uniforms.max_depth - 1u; level++) {
        let half_size = cell_size * 0.5;
        let center_x = cell_min_x + half_size;
        let center_y = cell_min_y + half_size;

        let quadrant = get_quadrant(pos_x, pos_y, center_x, center_y);

        // Move to child
        node_idx = 4u * node_idx + 1u + quadrant;

        // Update cell bounds
        if ((quadrant & 1u) != 0u) { cell_min_x = center_x; }
        if ((quadrant & 2u) != 0u) { cell_min_y = center_y; }
        cell_size = half_size;
    }

    // Add particle to leaf cell using atomics
    // We accumulate weighted position and count
    atomicAdd(&tree_count[node_idx], 1u);

    // For center of mass, we need to use atomics on floats.
    // WGSL lacks atomic float add, so we mark cell occupancy here
    // and compute proper center of mass in Phase 3 (compute_leaf_com).
}

// Phase 3: Compute leaf centers of mass.
// Each leaf cell iterates through all particles to find those within its bounds,
// then computes the center of mass from matching particles.
@compute @workgroup_size(256)
fn compute_leaf_com(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let leaf_local_idx = global_id.x;

    // Calculate leaf level start and count
    let leaf_level = uniforms.max_depth - 1u;
    let leaf_start = level_start_index(leaf_level);
    let num_leaves = 1u << (2u * leaf_level); // 4^leaf_level

    if (leaf_local_idx >= num_leaves) {
        return;
    }

    let node_idx = leaf_start + leaf_local_idx;

    // Compute cell bounds for this leaf
    var cell_min_x = uniforms.bounds_min_x;
    var cell_min_y = uniforms.bounds_min_y;
    var cell_size = uniforms.root_size;

    // Trace path from root to this leaf
    var temp_idx = leaf_local_idx;
    var path: array<u32, 16>;
    for (var l = 0u; l < leaf_level; l++) {
        path[leaf_level - 1u - l] = temp_idx & 3u;
        temp_idx = temp_idx >> 2u;
    }

    for (var l = 0u; l < leaf_level; l++) {
        let half_size = cell_size * 0.5;
        let quadrant = path[l];
        if ((quadrant & 1u) != 0u) { cell_min_x += half_size; }
        if ((quadrant & 2u) != 0u) { cell_min_y += half_size; }
        cell_size = half_size;
    }

    let cell_max_x = cell_min_x + cell_size;
    let cell_max_y = cell_min_y + cell_size;

    // Sum particles in this cell
    var sum_x = 0.0;
    var sum_y = 0.0;
    var count = 0.0;

    for (var i = 0u; i < uniforms.node_count; i++) {
        let px = positions_x[i];
        let py = positions_y[i];

        if (px >= cell_min_x && px < cell_max_x && py >= cell_min_y && py < cell_max_y) {
            sum_x += px;
            sum_y += py;
            count += 1.0;
        }
    }

    if (count > 0.0) {
        tree_com_x[node_idx] = sum_x / count;
        tree_com_y[node_idx] = sum_y / count;
        tree_mass[node_idx] = count;
    }
}

// Phase 4: Build internal nodes bottom-up
// Must be called for each level from (max_depth-2) down to 0
// Level is passed via num_workgroups.z (dispatch with (workgroups, 1, level+1))
@compute @workgroup_size(256)
fn build_level(@builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(num_workgroups) num_wg: vec3<u32>) {
    let local_idx = global_id.x;
    let level = num_wg.z - 1u;  // Level encoded in z dispatch count

    // Get the range of node indices for this level
    let level_start = level_start_index(level);
    let level_size = 1u << (2u * level);  // 4^level nodes at this level

    if (local_idx >= level_size) {
        return;
    }

    let node_idx = level_start + local_idx;
    let child_base = 4u * node_idx + 1u;

    // Aggregate children (children are at level+1, already computed)
    var sum_x = 0.0;
    var sum_y = 0.0;
    var total_mass = 0.0;

    for (var q = 0u; q < 4u; q++) {
        let child_idx = child_base + q;
        let child_mass = tree_mass[child_idx];

        if (child_mass > 0.0) {
            sum_x += tree_com_x[child_idx] * child_mass;
            sum_y += tree_com_y[child_idx] * child_mass;
            total_mass += child_mass;
        }
    }

    if (total_mass > 0.0) {
        tree_com_x[node_idx] = sum_x / total_mass;
        tree_com_y[node_idx] = sum_y / total_mass;
        tree_mass[node_idx] = total_mass;
    }
}
