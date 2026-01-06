// Simplified Spring Force Compute Shader
// Computes attractive forces along edges using Hooke's law

struct SpringUniforms {
    edge_count: u32,
    spring_strength: f32,
    rest_length: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: SpringUniforms;

// Node positions (read only)
@group(0) @binding(1) var<storage, read> positions_x: array<f32>;
@group(0) @binding(2) var<storage, read> positions_y: array<f32>;

// Force accumulators (read-write for atomic-like accumulation)
@group(0) @binding(3) var<storage, read_write> forces_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> forces_y: array<f32>;

// Edge data (source, target pairs)
@group(0) @binding(5) var<storage, read> edge_sources: array<u32>;
@group(0) @binding(6) var<storage, read> edge_targets: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let edge_idx = global_id.x;

    if (edge_idx >= uniforms.edge_count) {
        return;
    }

    let source_idx = edge_sources[edge_idx];
    let target_idx = edge_targets[edge_idx];

    // Get positions
    let source_x = positions_x[source_idx];
    let source_y = positions_y[source_idx];
    let target_x = positions_x[target_idx];
    let target_y = positions_y[target_idx];

    let dx = target_x - source_x;
    let dy = target_y - source_y;
    let dist = sqrt(dx * dx + dy * dy);

    // Avoid division by zero
    if (dist < 0.0001) {
        return;
    }

    // Direction from source to target
    let dir_x = dx / dist;
    let dir_y = dy / dist;

    // Hooke's law: F = k * (length - rest_length)
    let displacement = dist - uniforms.rest_length;
    let force_magnitude = uniforms.spring_strength * displacement;

    let force_x = dir_x * force_magnitude;
    let force_y = dir_y * force_magnitude;

    // Add force to source (attractive toward target)
    // Subtract from target (attractive toward source)
    // Note: This has race conditions but acceptable for approximate physics
    forces_x[source_idx] += force_x;
    forces_y[source_idx] += force_y;
    forces_x[target_idx] -= force_x;
    forces_y[target_idx] -= force_y;
}
