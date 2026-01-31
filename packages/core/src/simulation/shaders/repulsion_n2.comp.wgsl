// N^2 Repulsion Force Compute Shader
// Simple all-pairs repulsion calculation for small graphs
//
// Uses vec2<f32> layout for consolidated position/force data.

struct RepulsionUniforms {
    node_count: u32,
    repulsion_strength: f32,
    min_distance: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: RepulsionUniforms;

// Node positions (read only) - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Force accumulators (read-write) - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let node_pos = positions[node_idx];
    var force = vec2<f32>(0.0, 0.0);

    // Direct N^2 summation
    for (var i = 0u; i < uniforms.node_count; i++) {
        if (i == node_idx) {
            continue;
        }

        let other_pos = positions[i];
        let delta = node_pos - other_pos;

        let dist_sq = dot(delta, delta);
        let min_dist_sq = uniforms.min_distance * uniforms.min_distance;
        let safe_dist_sq = max(dist_sq, min_dist_sq);
        let dist = sqrt(safe_dist_sq);

        // Coulomb-like repulsion: F = k / r^2
        let force_magnitude = uniforms.repulsion_strength / safe_dist_sq;

        force += delta * (force_magnitude / dist);
    }

    forces[node_idx] += force;
}
