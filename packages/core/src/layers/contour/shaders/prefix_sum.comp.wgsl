// Contour Parallel Prefix Sum Compute Shader
//
// Second stage of marching squares: computes exclusive prefix sum
// of segment counts so each cell knows where to write its vertices.
//
// Uses Blelloch scan algorithm for work-efficient parallel prefix sum.

struct PrefixSumUniforms {
    // Total number of elements
    element_count: u32,
    // Current step offset
    step_offset: u32,
    // Padding
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: PrefixSumUniforms;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;

// Workgroup shared memory for efficient prefix sum
var<workgroup> shared_data: array<u32, 512>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load two elements per thread (Blelloch optimization)
    let ai = tid;
    let bi = tid + 256u;

    let global_ai = workgroup_id.x * 512u + ai;
    let global_bi = workgroup_id.x * 512u + bi;

    // Load into shared memory
    if (global_ai < uniforms.element_count) {
        shared_data[ai] = data[global_ai];
    } else {
        shared_data[ai] = 0u;
    }

    if (global_bi < uniforms.element_count) {
        shared_data[bi] = data[global_bi];
    } else {
        shared_data[bi] = 0u;
    }

    // Up-sweep (reduce) phase
    var offset = 1u;
    for (var d = 256u; d > 0u; d >>= 1u) {
        workgroupBarrier();

        if (tid < d) {
            let ai_idx = offset * (2u * tid + 1u) - 1u;
            let bi_idx = offset * (2u * tid + 2u) - 1u;
            shared_data[bi_idx] += shared_data[ai_idx];
        }
        offset *= 2u;
    }

    // Clear the last element (for exclusive scan)
    if (tid == 0u) {
        shared_data[511u] = 0u;
    }

    // Down-sweep phase
    for (var d = 1u; d < 512u; d *= 2u) {
        offset >>= 1u;
        workgroupBarrier();

        if (tid < d) {
            let ai_idx = offset * (2u * tid + 1u) - 1u;
            let bi_idx = offset * (2u * tid + 2u) - 1u;
            let t = shared_data[ai_idx];
            shared_data[ai_idx] = shared_data[bi_idx];
            shared_data[bi_idx] += t;
        }
    }

    workgroupBarrier();

    // Write results back
    if (global_ai < uniforms.element_count) {
        data[global_ai] = shared_data[ai];
    }

    if (global_bi < uniforms.element_count) {
        data[global_bi] = shared_data[bi];
    }
}
