// Clear Forces Compute Shader
// Resets force accumulators to zero before each simulation step
//
// Uses vec2<f32> layout for consolidated X/Y force data.

struct ClearUniforms {
    node_count: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ClearUniforms;

// Force accumulators - vec2<f32> per node
@group(0) @binding(1) var<storage, read_write> forces: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= uniforms.node_count) {
        return;
    }

    forces[node_idx] = vec2<f32>(0.0, 0.0);
}
