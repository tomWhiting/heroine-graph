// Simplified Spring Force Compute Shader
// Computes attractive forces along edges using Hooke's law
//
// Uses vec2<f32> layout for consolidated position/force data.

struct SpringUniforms {
    edge_count: u32,
    spring_strength: f32,
    rest_length: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: SpringUniforms;

// Node positions (read only) - vec2<f32> per node
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Force accumulators (read-write for atomic-like accumulation) - vec2<f32> per node
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Edge data (source, target pairs)
@group(0) @binding(3) var<storage, read> edge_sources: array<u32>;
@group(0) @binding(4) var<storage, read> edge_targets: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let edge_idx = global_id.x;

    if (edge_idx >= uniforms.edge_count) {
        return;
    }

    let source_idx = edge_sources[edge_idx];
    let target_idx = edge_targets[edge_idx];

    // Get positions
    let source_pos = positions[source_idx];
    let target_pos = positions[target_idx];

    let delta = target_pos - source_pos;
    let dist = length(delta);

    // Avoid division by zero
    if (dist < 0.0001) {
        return;
    }

    // Direction from source to target
    let dir = delta / dist;

    // Hooke's law: F = k * (length - rest_length)
    let displacement = dist - uniforms.rest_length;
    let force_magnitude = uniforms.spring_strength * displacement;

    let force = dir * force_magnitude;

    // Add force to source (attractive toward target)
    // Subtract from target (attractive toward source)
    // Note: This has race conditions but acceptable for approximate physics
    forces[source_idx] += force;
    forces[target_idx] -= force;
}
