// Density Field Repulsion Compute Shader
// Applies repulsive forces based on density gradient
//
// Each node samples the gradient texture at its position and receives
// a force in the opposite direction (away from high density).
// This provides O(n) repulsion computation.

struct DensityRepulsionUniforms {
    node_count: u32,
    width: u32,
    height: u32,
    repulsion_strength: f32,
    bounds_min_x: f32,
    bounds_min_y: f32,
    bounds_max_x: f32,
    bounds_max_y: f32,
}

@group(0) @binding(0) var<uniform> uniforms: DensityRepulsionUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;
@group(0) @binding(3) var gradient_texture: texture_2d<f32>;
@group(0) @binding(4) var gradient_sampler: sampler;

// Convert world position to texture UV
fn world_to_uv(pos: vec2<f32>) -> vec2<f32> {
    return (pos - vec2<f32>(uniforms.bounds_min_x, uniforms.bounds_min_y)) /
           (vec2<f32>(uniforms.bounds_max_x, uniforms.bounds_max_y) -
            vec2<f32>(uniforms.bounds_min_x, uniforms.bounds_min_y));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    let pos = positions[node_idx];
    let uv = world_to_uv(pos);

    // Clamp UV to valid range
    let clamped_uv = clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999));

    // Sample gradient at node position
    // Gradient points toward higher density
    let gradient = textureSampleLevel(gradient_texture, gradient_sampler, clamped_uv, 0.0).xy;

    // Force is opposite to gradient (move away from high density)
    // Scale by repulsion strength
    let force = -gradient * uniforms.repulsion_strength;

    // Accumulate force
    forces[node_idx] += force;
}
