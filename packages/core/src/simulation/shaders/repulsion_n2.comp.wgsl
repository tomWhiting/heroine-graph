// N^2 Repulsion Force Compute Shader
// Simple all-pairs repulsion calculation for small graphs

struct RepulsionUniforms {
    node_count: u32,
    repulsion_strength: f32,
    min_distance: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: RepulsionUniforms;

// Node positions (read only)
@group(0) @binding(1) var<storage, read> positions_x: array<f32>;
@group(0) @binding(2) var<storage, read> positions_y: array<f32>;

// Force accumulators (read-write)
@group(0) @binding(3) var<storage, read_write> forces_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> forces_y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let node_x = positions_x[node_idx];
    let node_y = positions_y[node_idx];
    var force_x = 0.0;
    var force_y = 0.0;

    // Direct N^2 summation
    for (var i = 0u; i < uniforms.node_count; i++) {
        if (i == node_idx) {
            continue;
        }

        let other_x = positions_x[i];
        let other_y = positions_y[i];
        let dx = node_x - other_x;
        let dy = node_y - other_y;

        let dist_sq = dx * dx + dy * dy;
        let min_dist_sq = uniforms.min_distance * uniforms.min_distance;
        let safe_dist_sq = max(dist_sq, min_dist_sq);
        let dist = sqrt(safe_dist_sq);

        // Coulomb-like repulsion: F = k / r^2
        let force_magnitude = uniforms.repulsion_strength / safe_dist_sq;

        force_x += dx * (force_magnitude / dist);
        force_y += dy * (force_magnitude / dist);
    }

    forces_x[node_idx] += force_x;
    forces_y[node_idx] += force_y;
}
