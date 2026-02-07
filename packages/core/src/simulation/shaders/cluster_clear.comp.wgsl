// Cluster Force: Clear centroid accumulators
//
// Runs once per frame before accumulation to zero out the
// per-community centroid sum and count buffers.

struct ClearUniforms {
    community_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ClearUniforms;
@group(0) @binding(1) var<storage, read_write> centroid_sum_x: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> centroid_sum_y: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> centroid_count: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.community_count) {
        return;
    }
    atomicStore(&centroid_sum_x[idx], 0);
    atomicStore(&centroid_sum_y[idx], 0);
    atomicStore(&centroid_count[idx], 0u);
}
