// Minimal Integration Shader

struct Uniforms {
    node_count: u32,
    dt: f32,
    damping: f32,
    alpha: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> pos_x_in: array<f32>;
@group(0) @binding(2) var<storage, read> pos_y_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> pos_x_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> pos_y_out: array<f32>;
@group(0) @binding(5) var<storage, read> vel_x_in: array<f32>;
@group(0) @binding(6) var<storage, read> vel_y_in: array<f32>;
@group(0) @binding(7) var<storage, read_write> vel_x_out: array<f32>;
@group(0) @binding(8) var<storage, read_write> vel_y_out: array<f32>;
@group(0) @binding(9) var<storage, read> force_x: array<f32>;
@group(0) @binding(10) var<storage, read> force_y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.node_count) { return; }

    var vx = vel_x_in[i] * uniforms.damping + force_x[i] * uniforms.alpha * uniforms.dt;
    var vy = vel_y_in[i] * uniforms.damping + force_y[i] * uniforms.alpha * uniforms.dt;

    pos_x_out[i] = pos_x_in[i] + vx * uniforms.dt;
    pos_y_out[i] = pos_y_in[i] + vy * uniforms.dt;
    vel_x_out[i] = vx;
    vel_y_out[i] = vy;
}
