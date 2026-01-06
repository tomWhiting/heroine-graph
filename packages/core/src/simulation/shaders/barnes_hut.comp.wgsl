// Barnes-Hut Force Traversal Compute Shader
// Computes repulsive forces using quadtree approximation
//
// The Barnes-Hut algorithm approximates long-range forces by treating
// distant groups of nodes as single massive bodies. The theta parameter
// controls the accuracy/speed tradeoff.

struct ForceUniforms {
    node_count: u32,
    repulsion_strength: f32,   // Repulsion force multiplier
    theta: f32,                // Opening angle (0.5-1.5 typical)
    min_distance: f32,         // Minimum distance to prevent singularities
    bounds_min: vec2<f32>,     // Bounding box for quadtree
    bounds_max: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: ForceUniforms;

// Node positions
@group(0) @binding(1) var<storage, read> positions_x: array<f32>;
@group(0) @binding(2) var<storage, read> positions_y: array<f32>;

// Output forces (accumulated)
@group(0) @binding(3) var<storage, read_write> forces_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> forces_y: array<f32>;

// Quadtree data
@group(0) @binding(5) var<storage, read> tree_nodes_x: array<f32>;    // Center of mass X
@group(0) @binding(6) var<storage, read> tree_nodes_y: array<f32>;    // Center of mass Y
@group(0) @binding(7) var<storage, read> tree_nodes_mass: array<f32>; // Total mass
@group(0) @binding(8) var<storage, read> tree_sizes: array<f32>;      // Cell sizes

// Tree structure
const MAX_DEPTH: u32 = 16u;
const MAX_TREE_SIZE: u32 = 262144u;

// Stack for iterative tree traversal
struct TraversalStack {
    nodes: array<u32, 64>,  // Max depth * 4 children
    count: u32,
}

// Compute repulsive force between two bodies
fn repulsive_force(dx: f32, dy: f32, mass: f32) -> vec2<f32> {
    let dist_sq = dx * dx + dy * dy;
    let min_dist_sq = uniforms.min_distance * uniforms.min_distance;
    let safe_dist_sq = max(dist_sq, min_dist_sq);
    let dist = sqrt(safe_dist_sq);

    // Coulomb-like repulsion: F = k * m1 * m2 / r^2
    // Direction: pointing away from the other body
    let force_magnitude = uniforms.repulsion_strength * mass / safe_dist_sq;

    // Normalize direction and apply magnitude
    return vec2<f32>(dx, dy) * (force_magnitude / dist);
}

// Check if cell should be opened (Barnes-Hut criterion)
fn should_open_cell(node_pos: vec2<f32>, cell_center: vec2<f32>, cell_size: f32) -> bool {
    let dx = node_pos.x - cell_center.x;
    let dy = node_pos.y - cell_center.y;
    let dist_sq = dx * dx + dy * dy;

    // Open if size/distance > theta
    // Equivalent to: size^2 > theta^2 * dist^2
    let theta_sq = uniforms.theta * uniforms.theta;
    return cell_size * cell_size > theta_sq * dist_sq;
}

// Get child cell index for a position
fn get_quadrant(pos: vec2<f32>, cell_center: vec2<f32>) -> u32 {
    var quadrant = 0u;
    if (pos.x >= cell_center.x) {
        quadrant |= 1u;
    }
    if (pos.y >= cell_center.y) {
        quadrant |= 2u;
    }
    return quadrant;
}

// Main force computation kernel
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let node_pos = vec2<f32>(positions_x[node_idx], positions_y[node_idx]);
    var total_force = vec2<f32>(0.0, 0.0);

    // Iterative tree traversal using stack
    // Start with root node (index 0)
    var stack: array<u32, 64>;
    var stack_ptr = 0;
    stack[0] = 0u;
    stack_ptr = 1;

    while (stack_ptr > 0) {
        stack_ptr--;
        let cell_idx = stack[stack_ptr];

        if (cell_idx >= MAX_TREE_SIZE) {
            continue;
        }

        let cell_mass = tree_nodes_mass[cell_idx];

        // Skip empty cells
        if (cell_mass <= 0.0) {
            continue;
        }

        let cell_center = vec2<f32>(tree_nodes_x[cell_idx], tree_nodes_y[cell_idx]);
        let cell_size = tree_sizes[cell_idx];

        let dx = node_pos.x - cell_center.x;
        let dy = node_pos.y - cell_center.y;
        let dist_sq = dx * dx + dy * dy;

        // Check if this is the node itself (or very close)
        if (dist_sq < 0.0001) {
            continue;
        }

        // Barnes-Hut criterion: use cell if it's far enough or is a leaf
        if (!should_open_cell(node_pos, cell_center, cell_size) || cell_size <= 0.0) {
            // Treat cell as single body
            total_force += repulsive_force(dx, dy, cell_mass);
        } else {
            // Open cell and push children onto stack
            let child_base = cell_idx * 4u + 1u;  // Quad tree indexing

            // Push all 4 children (they'll be skipped if empty)
            if (stack_ptr + 4 <= 64) {
                stack[stack_ptr] = child_base;
                stack[stack_ptr + 1] = child_base + 1u;
                stack[stack_ptr + 2] = child_base + 2u;
                stack[stack_ptr + 3] = child_base + 3u;
                stack_ptr += 4;
            }
        }
    }

    // Add force to output
    forces_x[node_idx] += total_force.x;
    forces_y[node_idx] += total_force.y;
}

// Simplified N^2 kernel for small node counts (fallback)
@compute @workgroup_size(256)
fn direct_n2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let node_pos = vec2<f32>(positions_x[node_idx], positions_y[node_idx]);
    var total_force = vec2<f32>(0.0, 0.0);

    // Direct N^2 summation
    for (var i = 0u; i < uniforms.node_count; i++) {
        if (i == node_idx) {
            continue;
        }

        let other_pos = vec2<f32>(positions_x[i], positions_y[i]);
        let dx = node_pos.x - other_pos.x;
        let dy = node_pos.y - other_pos.y;

        total_force += repulsive_force(dx, dy, 1.0);
    }

    forces_x[node_idx] += total_force.x;
    forces_y[node_idx] += total_force.y;
}
