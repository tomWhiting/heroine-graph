// Barnes-Hut Force Traversal Compute Shader
// Computes repulsive forces using quadtree approximation
//
// Tree structure (implicit heap-like indexing):
// - Node i has children at 4*i + 1, 4*i + 2, 4*i + 3, 4*i + 4
// - Node i has parent at (i - 1) / 4
//
// The Barnes-Hut algorithm approximates long-range forces by treating
// distant groups of nodes as single massive bodies. The theta parameter
// controls the accuracy/speed tradeoff.
//
// Uses vec2<f32> layout for consolidated position/force data.

struct ForceUniforms {
    node_count: u32,
    repulsion_strength: f32,   // Repulsion force multiplier
    theta: f32,                // Opening angle (0.5-1.5 typical)
    min_distance: f32,         // Minimum distance to prevent singularities
    min_cell_size: f32,        // Minimum cell size (leaf level) - below this, always use approximation
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> uniforms: ForceUniforms;

// Node positions - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Output forces (accumulated) - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Quadtree data (from build phase)
@group(0) @binding(3) var<storage, read> tree_com_x: array<f32>;    // Center of mass X
@group(0) @binding(4) var<storage, read> tree_com_y: array<f32>;    // Center of mass Y
@group(0) @binding(5) var<storage, read> tree_mass: array<f32>;     // Total mass in cell
@group(0) @binding(6) var<storage, read> tree_sizes: array<f32>;    // Cell sizes

const MAX_TREE_SIZE: u32 = 262144u;
const MAX_STACK_DEPTH: u32 = 64u;

// Compute repulsive force between a node and a cell/body
fn compute_repulsion(delta: vec2<f32>, mass: f32) -> vec2<f32> {
    let dist_sq = dot(delta, delta);
    let min_dist_sq = uniforms.min_distance * uniforms.min_distance;
    let safe_dist_sq = max(dist_sq, min_dist_sq);

    // Coulomb-like repulsion: F = k * m / r^2
    // Force points from other body toward this node (repulsive)
    let force_magnitude = uniforms.repulsion_strength * mass / safe_dist_sq;

    // Normalize direction and apply magnitude
    let dist = sqrt(safe_dist_sq);
    return delta * (force_magnitude / dist);
}

// Main force computation using tree traversal
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let pos = positions[node_idx];
    var total_force = vec2<f32>(0.0, 0.0);

    // Iterative tree traversal using explicit stack
    var stack: array<u32, 64>;
    var stack_ptr = 1u;
    stack[0] = 0u;  // Start with root node

    let theta_sq = uniforms.theta * uniforms.theta;

    while (stack_ptr > 0u) {
        stack_ptr -= 1u;
        let cell_idx = stack[stack_ptr];

        if (cell_idx >= MAX_TREE_SIZE) {
            continue;
        }

        let cell_mass = tree_mass[cell_idx];

        // Skip empty cells
        if (cell_mass <= 0.0) {
            continue;
        }

        let cell_com = vec2<f32>(tree_com_x[cell_idx], tree_com_y[cell_idx]);
        let cell_size = tree_sizes[cell_idx];

        // Distance from node to cell center of mass
        let delta = pos - cell_com;
        let dist_sq = dot(delta, delta);

        // Skip if this is essentially the same position (likely the node itself in a leaf)
        if (dist_sq < 0.0001) {
            continue;
        }

        // Barnes-Hut criterion: size/distance < theta
        // Equivalent to: size^2 < theta^2 * dist^2
        let size_sq = cell_size * cell_size;

        // Check if this is a leaf cell (at minimum size) - leaves cannot be opened further
        let is_leaf = cell_size <= uniforms.min_cell_size * 1.1;  // Small tolerance

        if (size_sq < theta_sq * dist_sq || cell_size <= 0.0 || is_leaf) {
            // Cell is far enough OR is a leaf - treat as single body
            total_force += compute_repulsion(delta, cell_mass);
        } else {
            // Cell is too close and can be subdivided - examine children
            let child_base = 4u * cell_idx + 1u;

            // Push all 4 children onto stack (they'll be skipped if empty)
            if (stack_ptr + 4u <= MAX_STACK_DEPTH) {
                stack[stack_ptr] = child_base;
                stack[stack_ptr + 1u] = child_base + 1u;
                stack[stack_ptr + 2u] = child_base + 2u;
                stack[stack_ptr + 3u] = child_base + 3u;
                stack_ptr += 4u;
            }
        }
    }

    // Accumulate force
    forces[node_idx] += total_force;
}
