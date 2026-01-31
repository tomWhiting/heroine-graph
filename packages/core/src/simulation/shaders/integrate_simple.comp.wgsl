// Minimal Integration Shader (vec2 buffer version)

struct Uniforms {
    node_count: u32,
    dt: f32,
    damping: f32,
    alpha: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> positions_in: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> positions_out: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> velocities_in: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> velocities_out: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> forces: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.node_count) { return; }

    let vel = velocities_in[i] * uniforms.damping + forces[i] * uniforms.alpha * uniforms.dt;

    positions_out[i] = positions_in[i] + vel * uniforms.dt;
    velocities_out[i] = vel;
}
