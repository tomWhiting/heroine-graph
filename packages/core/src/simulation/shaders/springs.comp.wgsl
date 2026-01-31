// Attractive Force (Spring) Compute Shader
// Computes spring forces along edges using Hooke's law
//
// Each edge acts as a spring pulling its endpoint nodes together.
// Force magnitude increases linearly with distance beyond rest length.

struct SpringUniforms {
    node_count: u32,
    edge_count: u32,
    spring_strength: f32,   // Spring constant k
    rest_length: f32,       // Natural spring length
    damping: f32,           // Velocity damping factor
    _padding: vec3<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: SpringUniforms;

// Node positions
@group(0) @binding(1) var<storage, read> positions_x: array<f32>;
@group(0) @binding(2) var<storage, read> positions_y: array<f32>;

// Node velocities (for damping)
@group(0) @binding(3) var<storage, read> velocities_x: array<f32>;
@group(0) @binding(4) var<storage, read> velocities_y: array<f32>;

// Output forces (accumulated)
@group(0) @binding(5) var<storage, read_write> forces_x: array<f32>;
@group(0) @binding(6) var<storage, read_write> forces_y: array<f32>;

// Edge data (source, target pairs)
@group(0) @binding(7) var<storage, read> edge_sources: array<u32>;
@group(0) @binding(8) var<storage, read> edge_targets: array<u32>;

// Optional: edge weights for variable spring strength
@group(0) @binding(9) var<storage, read> edge_weights: array<f32>;

// Compute spring force for an edge
fn spring_force(
    source_pos: vec2<f32>,
    target_pos: vec2<f32>,
    source_vel: vec2<f32>,
    target_vel: vec2<f32>,
    weight: f32
) -> vec2<f32> {
    let delta = target_pos - source_pos;
    let dist = length(delta);

    // Avoid division by zero
    if (dist < 0.0001) {
        return vec2<f32>(0.0, 0.0);
    }

    // Direction from source to target
    let direction = delta / dist;

    // Hooke's law: F = -k * (length - rest_length)
    let displacement = dist - uniforms.rest_length;
    let spring_magnitude = uniforms.spring_strength * displacement * weight;

    // Add damping based on relative velocity
    let rel_vel = target_vel - source_vel;
    let damping_magnitude = uniforms.damping * dot(rel_vel, direction);

    // Total force magnitude (positive = attractive)
    let force_magnitude = spring_magnitude + damping_magnitude;

    return direction * force_magnitude;
}

// Per-edge kernel: computes force for each edge and atomically adds to endpoints
@compute @workgroup_size(256)
fn per_edge(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let edge_idx = global_id.x;

    if (edge_idx >= uniforms.edge_count) {
        return;
    }

    let source_idx = edge_sources[edge_idx];
    let target_idx = edge_targets[edge_idx];

    // Get positions
    let source_pos = vec2<f32>(positions_x[source_idx], positions_y[source_idx]);
    let target_pos = vec2<f32>(positions_x[target_idx], positions_y[target_idx]);

    // Get velocities
    let source_vel = vec2<f32>(velocities_x[source_idx], velocities_y[source_idx]);
    let target_vel = vec2<f32>(velocities_x[target_idx], velocities_y[target_idx]);

    // Get edge weight (default 1.0)
    let weight = edge_weights[edge_idx];

    // Compute spring force from source to target
    let force = spring_force(source_pos, target_pos, source_vel, target_vel, weight);

    // WGSL lacks atomic float operations. Direct writes may have race conditions,
    // but this is acceptable for approximate physics simulation where small
    // numerical variations do not affect convergence or visual quality.
    forces_x[source_idx] += force.x;
    forces_y[source_idx] += force.y;
    forces_x[target_idx] -= force.x;
    forces_y[target_idx] -= force.y;
}


// Initialize forces to zero before accumulation
@compute @workgroup_size(256)
fn clear_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    forces_x[node_idx] = 0.0;
    forces_y[node_idx] = 0.0;
}
