// Barnes-Hut Force Traversal for Binary Radix Tree
// Computes repulsive forces using Karras binary tree approximation
//
// Tree structure (Karras binary radix tree):
// - N-1 internal nodes at indices 0..N-2
// - N leaf nodes at indices N-1..2N-2
// - left_child/right_child: negative values indicate leaf index (-(leaf_idx + 1))
// - Binary tree: each internal node has exactly 2 children
//
// The Barnes-Hut algorithm approximates long-range forces by treating
// distant groups of nodes as single massive bodies. The theta parameter
// controls the accuracy/speed tradeoff.

struct ForceUniforms {
    particle_count: u32,       // Number of particles
    repulsion_strength: f32,   // Repulsion force multiplier
    theta: f32,                // Opening angle (0.5-1.5 typical)
    min_distance: f32,         // Minimum distance to prevent singularities
    leaf_size: f32,            // Approximate size of leaf nodes
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> uniforms: ForceUniforms;

// Particle positions (original order, not sorted)
@group(0) @binding(1) var<storage, read> positions_x: array<f32>;
@group(0) @binding(2) var<storage, read> positions_y: array<f32>;

// Output forces (accumulated)
@group(0) @binding(3) var<storage, read_write> forces_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> forces_y: array<f32>;

// Tree structure (from Karras build)
@group(0) @binding(5) var<storage, read> left_child: array<i32>;
@group(0) @binding(6) var<storage, read> right_child: array<i32>;

// Node properties (2N-1 total: internal + leaves)
@group(0) @binding(7) var<storage, read> node_com_x: array<f32>;
@group(0) @binding(8) var<storage, read> node_com_y: array<f32>;
@group(0) @binding(9) var<storage, read> node_mass: array<f32>;
@group(0) @binding(10) var<storage, read> node_size: array<f32>;

const MAX_STACK_DEPTH: u32 = 64u;
const WORKGROUP_SIZE: u32 = 256u;

// Compute repulsive force between a particle and a cell/body
fn compute_repulsion(dx: f32, dy: f32, mass: f32) -> vec2<f32> {
    let dist_sq = dx * dx + dy * dy;
    let min_dist_sq = uniforms.min_distance * uniforms.min_distance;
    let safe_dist_sq = max(dist_sq, min_dist_sq);

    // Coulomb-like repulsion: F = k * m / r^2
    // Force points from other body toward this particle (repulsive)
    let force_magnitude = uniforms.repulsion_strength * mass / safe_dist_sq;

    // Normalize direction and apply magnitude
    let dist = sqrt(safe_dist_sq);
    return vec2<f32>(dx, dy) * (force_magnitude / dist);
}

// Convert child reference to node index
// Negative values are leaves: -(leaf_idx + 1) → node_idx = N - 1 + leaf_idx
fn child_to_node_idx(child: i32, n: u32) -> u32 {
    if (child < 0) {
        // Leaf: convert negative reference to node index
        let leaf_idx = u32(-(child + 1));
        return n - 1u + leaf_idx;
    } else {
        // Internal node: direct index
        return u32(child);
    }
}

// Check if child reference is a leaf
fn is_leaf_child(child: i32) -> bool {
    return child < 0;
}

// Main force computation using binary tree traversal
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    let n = uniforms.particle_count;

    if (particle_idx >= n) {
        return;
    }

    let pos_x = positions_x[particle_idx];
    let pos_y = positions_y[particle_idx];
    var total_force = vec2<f32>(0.0, 0.0);

    // Handle degenerate cases
    if (n == 0u) {
        return;
    }

    if (n == 1u) {
        // Only one particle, no forces
        return;
    }

    // Iterative tree traversal using explicit stack
    // Stack stores (node_index, is_internal_node) packed as u32
    // High bit: 0 = internal, 1 = leaf node index
    var stack: array<u32, 64>;
    var stack_ptr = 1u;
    stack[0] = 0u;  // Start with root (internal node 0)

    let theta_sq = uniforms.theta * uniforms.theta;
    let num_internal = n - 1u;

    while (stack_ptr > 0u) {
        stack_ptr -= 1u;
        let node_idx = stack[stack_ptr];

        // Get node properties
        let cell_mass = node_mass[node_idx];

        // Skip empty nodes
        if (cell_mass <= 0.0) {
            continue;
        }

        let cell_com_x = node_com_x[node_idx];
        let cell_com_y = node_com_y[node_idx];
        let cell_size = node_size[node_idx];

        // Distance from particle to cell center of mass
        let dx = pos_x - cell_com_x;
        let dy = pos_y - cell_com_y;
        let dist_sq = dx * dx + dy * dy;

        // Skip if this is essentially the same position
        // This handles self-interaction (particle in its own leaf)
        if (dist_sq < 0.0001) {
            continue;
        }

        // Check if this is a leaf node (stored at index >= N-1)
        let is_leaf = node_idx >= num_internal;

        // Barnes-Hut criterion: size/distance < theta
        // Equivalent to: size^2 < theta^2 * dist^2
        let size_sq = cell_size * cell_size;
        let use_approximation = (size_sq < theta_sq * dist_sq) || cell_size <= 0.0 || is_leaf;

        if (use_approximation) {
            // Cell is far enough OR is a leaf - treat as single body
            total_force += compute_repulsion(dx, dy, cell_mass);
        } else {
            // Cell is too close - examine children (binary tree: 2 children)
            let left = left_child[node_idx];
            let right = right_child[node_idx];

            // Push children onto stack
            if (stack_ptr + 2u <= MAX_STACK_DEPTH) {
                // Left child
                let left_node = child_to_node_idx(left, n);
                stack[stack_ptr] = left_node;
                stack_ptr += 1u;

                // Right child
                let right_node = child_to_node_idx(right, n);
                stack[stack_ptr] = right_node;
                stack_ptr += 1u;
            }
        }
    }

    // Accumulate force to output
    forces_x[particle_idx] += total_force.x;
    forces_y[particle_idx] += total_force.y;
}

// Alternative entry point for sorted particle order.
// When particles are processed in Morton-sorted order, this provides
// the same computation with improved memory access locality.
@compute @workgroup_size(256)
fn main_sorted(@builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(local_invocation_id) local_id: vec3<u32>) {
    main(global_id);
}
