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
//
// Uses vec2<f32> layout for consolidated position/force data.

struct ForceUniforms {
    particle_count: u32,       // Number of particles
    repulsion_strength: f32,   // Repulsion force multiplier
    theta: f32,                // Opening angle (0.5-1.5 typical)
    min_distance: f32,         // Minimum distance to prevent singularities
    leaf_size: f32,            // Approximate size of leaf nodes
    max_distance: f32,         // Maximum repulsion distance (0 = no limit)
    // Entries of the active-index list this pass dispatches over. The tree
    // itself is still built over every particle — hidden particles enter it
    // with the maximum Morton code and zero leaf mass, exactly as dead slots
    // do — so the gather shortens only the traversal, which is where the cost
    // is.
    active_count: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ForceUniforms;

// Particle positions (original order, not sorted) - vec2<f32> per particle
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Output forces (accumulated) - vec2<f32> per particle
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Tree structure (from Karras build)
@group(0) @binding(3) var<storage, read> left_child: array<i32>;
@group(0) @binding(4) var<storage, read> right_child: array<i32>;

// Node properties (2N-1 total: internal + leaves)
@group(0) @binding(5) var<storage, read> node_com: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> node_mass: array<f32>;
@group(0) @binding(7) var<storage, read> node_size: array<f32>;

// Node state flags (bit 0 = dead slot, bit 2 = hidden by LOD) — see pipeline.ts
@group(0) @binding(8) var<storage, read> node_flags: array<u32>;

// Per-particle simulation mass (f32 per slot), the same buffer init_leaves
// seeds the tree from. Read here only to know how much of the coincident mass
// is this particle's own leaf.
@group(0) @binding(9) var<storage, read> particle_mass: array<f32>;

// Active-index list: the first active_count entries are the slots taking part
// in this tick, ascending. `active` is a reserved WGSL identifier, hence the
// name.
@group(0) @binding(10) var<storage, read> live_idx: array<u32>;

// Stack depth for tree traversal. For a balanced binary tree with N nodes the
// maximum stack depth needed is log₂(N) — 23 levels at the 8,388,480-node
// ceiling (see MAX_BARNES_HUT_NODES in algorithms/barnes-hut.ts). However,
// Karras trees can be unbalanced. We use 128 to handle extreme cases,
// supporting trees up to 128 levels deep with massive safety margin.
const MAX_STACK_DEPTH: u32 = 128u;
const WORKGROUP_SIZE: u32 = 256u;
const NODE_FLAG_DEAD: u32 = 1u;
const NODE_FLAG_HIDDEN_LOD: u32 = 4u;
// A slot carrying either bit neither exerts nor receives force.
const NODE_FLAG_INERT: u32 = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD;

// Bodies closer than sqrt(this) are treated as coincident: the repulsion
// direction is degenerate, so they get a deterministic golden-angle
// separation impulse instead (mirrors collision.comp.wgsl's tiebreaker).
const COINCIDENT_DIST_SQ: f32 = 0.0001;

// Compute repulsive force between a particle and a cell/body
fn compute_repulsion(delta: vec2<f32>, mass: f32) -> vec2<f32> {
    let dist_sq = dot(delta, delta);
    let min_dist_sq = uniforms.min_distance * uniforms.min_distance;
    let safe_dist_sq = max(dist_sq, min_dist_sq);

    // Coulomb-like repulsion: F = k * m / r^2
    // Force points from other body toward this particle (repulsive)
    let force_magnitude = uniforms.repulsion_strength * mass / safe_dist_sq;

    // Normalize direction and apply magnitude
    let dist = sqrt(safe_dist_sq);
    return delta * (force_magnitude / dist);
}

// Convert child reference to node index
// Negative values are leaves: -(leaf_idx + 1) -> node_idx = N - 1 + leaf_idx
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

// Main force computation using binary tree traversal.
//
// One thread per ENTRY of the active-index list rather than per particle: a
// traversal is the expensive part of this algorithm, and a hidden particle's
// traversal produces nothing (every inert leaf carries zero mass), so under a
// cut those traversals are work that is simply not done.
//
// The flag mask is kept on top of the gather. A stale list can only
// over-include, and masking makes that a wasted traversal rather than a wrong
// force, so a host whose list was never refreshed (the identity list the
// fallback writes) stays correct, just slower — and with the identity list the
// per-particle results are unchanged, because the traversal reads the tree and
// never the list.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let entry = global_id.x;
    let n = uniforms.particle_count;

    if (entry >= uniforms.active_count) {
        return;
    }

    let particle_idx = live_idx[entry];
    if (particle_idx >= n) {
        return;
    }

    // Inert slots — dead or LOD-hidden — neither receive forces here nor
    // exert any (their leaves carry zero mass — see init_leaves)
    if ((node_flags[particle_idx] & NODE_FLAG_INERT) != 0u) {
        return;
    }

    let pos = positions[particle_idx];
    let own_mass = particle_mass[particle_idx];
    var total_force = vec2<f32>(0.0, 0.0);
    // Mass found at (essentially) this particle's own position, including its
    // own leaf. Any excess is other bodies stacked on this one.
    var coincident_mass = 0.0;

    // Handle degenerate cases
    if (n == 0u) {
        return;
    }

    if (n == 1u) {
        // Only one particle, no forces
        return;
    }

    // Iterative tree traversal using explicit stack
    // Stack stores node indices to visit
    var stack: array<u32, 128>;
    var stack_ptr = 1u;
    stack[0] = 0u;  // Start with root (internal node 0)

    let theta_sq = uniforms.theta * uniforms.theta;
    let num_internal = n - 1u;

    // SAFETY: Limit iterations to prevent infinite loops from corrupted tree structure
    // For N nodes, max iterations is O(N) when theta=0, plus stack overhead
    // 100K iterations handles graphs up to ~50K nodes with aggressive theta
    var iterations = 0u;
    let max_iterations = max(n * 4u, 1000u);

    while (stack_ptr > 0u && iterations < max_iterations) {
        iterations += 1u;
        stack_ptr -= 1u;
        let node_idx = stack[stack_ptr];

        // Get node properties
        let cell_mass = node_mass[node_idx];

        // Skip empty nodes
        if (cell_mass <= 0.0) {
            continue;
        }

        let cell_com = node_com[node_idx];
        let cell_size = node_size[node_idx];

        // Distance from particle to cell center of mass
        let delta = pos - cell_com;
        let dist_sq = dot(delta, delta);

        // Check if this is a leaf node (stored at index >= N-1)
        let is_leaf = node_idx >= num_internal;

        // Skip nodes beyond max_distance (0 = no limit)
        if (uniforms.max_distance > 0.0 && dist_sq > uniforms.max_distance * uniforms.max_distance) {
            continue;
        }

        // Barnes-Hut criterion: size/distance < theta
        // Equivalent to: size^2 < theta^2 * dist^2
        let size_sq = cell_size * cell_size;
        let use_approximation = (size_sq < theta_sq * dist_sq) || cell_size <= 0.0 || is_leaf;

        if (use_approximation) {
            if (dist_sq < COINCIDENT_DIST_SQ) {
                // Same position: repulsion direction is degenerate. This is
                // either the particle's own leaf (always encountered, carrying
                // own_mass) or a distinct body stacked on it. Tally the mass;
                // the own contribution is subtracted after the walk and the
                // excess gets a deterministic separation impulse.
                coincident_mass += cell_mass;
            } else {
                // Cell is far enough OR is a leaf - treat as single body
                total_force += compute_repulsion(delta, cell_mass);
            }
        } else {
            // Cell is too close - examine children (binary tree: 2 children)
            let left = left_child[node_idx];
            let right = right_child[node_idx];

            // Push children onto stack if space available
            if (stack_ptr + 2u <= MAX_STACK_DEPTH) {
                // Left child
                let left_node = child_to_node_idx(left, n);
                stack[stack_ptr] = left_node;
                stack_ptr += 1u;

                // Right child
                let right_node = child_to_node_idx(right, n);
                stack[stack_ptr] = right_node;
                stack_ptr += 1u;
            } else {
                // Stack overflow: cannot descend further into this subtree.
                // Fall back to treating the current node as an approximation.
                // This is the mathematically correct fallback - we use the
                // aggregate mass/center-of-mass of the entire subtree rather
                // than computing individual contributions.
                //
                // With MAX_STACK_DEPTH=128, this should never occur in practice
                // (would require a tree deeper than 128 levels, far beyond the
                // ~23 a balanced tree needs at the 8,388,480-node ceiling).
                total_force += compute_repulsion(delta, cell_mass);
            }
        }
    }

    // Coincident mass beyond this particle's own leaf belongs to genuinely
    // distinct bodies at the same position. Without this they would never
    // repel and stay permanently fused (springs also vanish at zero distance).
    // Push along a per-index golden-angle direction — the same deterministic
    // tiebreaker collision.comp.wgsl uses — at min_distance repulsion
    // strength. Subtracting own_mass rather than 1.0 is what stops a collapsed
    // proxy shoving itself sideways by its own aggregate mass.
    let extra_coincident = coincident_mass - own_mass;
    if (extra_coincident > 0.0) {
        let angle = f32(particle_idx) * 0.618033988749895 * 6.28318530718;  // Golden ratio
        let dir = vec2<f32>(cos(angle), sin(angle));
        let min_dist_sq = uniforms.min_distance * uniforms.min_distance;
        total_force += dir * (uniforms.repulsion_strength * extra_coincident / min_dist_sq);
    }

    // Accumulate force to output
    forces[particle_idx] += total_force;
}

