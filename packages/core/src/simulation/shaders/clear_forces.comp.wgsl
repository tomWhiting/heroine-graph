// Clear Forces Compute Shader
// Resets force accumulators to zero before each simulation step

struct ClearUniforms {
    node_count: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ClearUniforms;

// Force accumulators
@group(0) @binding(1) var<storage, read_write> forces_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> forces_y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    forces_x[node_idx] = 0.0;
    forces_y[node_idx] = 0.0;
}
