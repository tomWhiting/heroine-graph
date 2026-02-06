// Relativity Atlas: Sibling Repulsion Shader
// Computes repulsion forces only between siblings (nodes sharing a parent).
//
// This is the key innovation of Relativity Atlas:
// - O(N + E) instead of O(N^2) for repulsion
// - Natural hierarchical structure preservation
// - Siblings spread around their shared parent
//
// Uses vec2<f32> layout for consolidated position/force data.

struct SiblingUniforms {
    node_count: u32,
    edge_count: u32,
    repulsion_strength: f32,
    min_distance: f32,
    max_siblings: u32,              // Cap on siblings to check (perf limit for high-degree nodes)
    parent_child_multiplier: f32,   // Weaker repulsion for connected pairs (default: 0.3)
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: SiblingUniforms;

// Node positions - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Force accumulators - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Inverse CSR: incoming edges (parents)
// For node i, csr_inverse_offsets[i]..csr_inverse_offsets[i+1] are parent indices
@group(0) @binding(3) var<storage, read> csr_inverse_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> csr_inverse_sources: array<u32>;

// Forward CSR: outgoing edges (children)
// For parent p, csr_offsets[p]..csr_offsets[p+1] are child indices (siblings of each other)
@group(0) @binding(5) var<storage, read> csr_offsets: array<u32>;
@group(0) @binding(6) var<storage, read> csr_targets: array<u32>;

// Node masses (for mass-weighted repulsion)
@group(0) @binding(7) var<storage, read> node_mass: array<f32>;

const WORKGROUP_SIZE: u32 = 256u;
const EPSILON: f32 = 0.0001;

// Compute repulsive force between two nodes
fn compute_repulsion(delta: vec2<f32>, mass_i: f32, mass_j: f32) -> vec2<f32> {
    let dist_sq = dot(delta, delta);
    let min_dist_sq = uniforms.min_distance * uniforms.min_distance;
    let safe_dist_sq = max(dist_sq, min_dist_sq);

    // Mass-weighted Coulomb-like repulsion: F = k * m_i * m_j / r^2
    let force_magnitude = uniforms.repulsion_strength * mass_i * mass_j / safe_dist_sq;

    let dist = sqrt(safe_dist_sq);
    return delta * (force_magnitude / dist);
}

// Main sibling repulsion computation
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let pos = positions[node_idx];
    let mass_i = max(node_mass[node_idx], 1.0);

    var force = vec2<f32>(0.0, 0.0);

    // Get this node's parents (incoming edges)
    let parent_start = csr_inverse_offsets[node_idx];
    let parent_end = csr_inverse_offsets[node_idx + 1u];

    // For each parent, find siblings (other children of that parent)
    for (var p = parent_start; p < parent_end; p++) {
        let parent_idx = csr_inverse_sources[p];

        if (parent_idx >= uniforms.node_count) {
            continue;
        }

        // Get siblings (children of this parent)
        let sibling_start = csr_offsets[parent_idx];
        let sibling_end = csr_offsets[parent_idx + 1u];

        // CRITICAL: Limit loop iterations to prevent infinite loops
        // If sibling_end is corrupted (very large), or if max_siblings skipped entries
        // never increment `checked`, the loop could run forever.
        // Use loop counter (s - sibling_start) as the primary limit, not checked count.
        let max_iterations = min(sibling_end - sibling_start, uniforms.max_siblings);
        var checked = 0u;

        for (var s = sibling_start; s < sibling_start + max_iterations; s++) {
            let sibling_idx = csr_targets[s];

            // Skip self and invalid indices
            if (sibling_idx == node_idx || sibling_idx >= uniforms.node_count) {
                continue;
            }

            checked++;

            let sib_pos = positions[sibling_idx];
            let mass_j = max(node_mass[sibling_idx], 1.0);

            let delta = pos - sib_pos;

            // Skip if too close (will be handled by collision detection)
            let dist_sq = dot(delta, delta);
            if (dist_sq < EPSILON) {
                continue;
            }

            force += compute_repulsion(delta, mass_i, mass_j);
        }
    }

    // Also repel against direct neighbors (bidirectional links)
    // This handles cases where nodes are both parent and child of each other
    let child_start = csr_offsets[node_idx];
    let child_end = csr_offsets[node_idx + 1u];

    // SAFETY: Limit iterations in case CSR data is corrupted
    let max_children = min(child_end - child_start, uniforms.max_siblings);
    for (var c = child_start; c < child_start + max_children; c++) {
        let child_idx = csr_targets[c];

        if (child_idx >= uniforms.node_count) {
            continue;
        }

        let child_pos = positions[child_idx];
        let mass_j = max(node_mass[child_idx], 1.0);

        let delta = pos - child_pos;

        let dist_sq = dot(delta, delta);
        if (dist_sq < EPSILON) {
            continue;
        }

        // Weaker repulsion for parent-child (they're connected by spring)
        force += compute_repulsion(delta, mass_i, mass_j) * uniforms.parent_child_multiplier;
    }

    // Accumulate forces
    forces[node_idx] += force;
}

