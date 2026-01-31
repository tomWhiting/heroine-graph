// Bounding Box Calculation Compute Shader
// Computes the axis-aligned bounding box of all node positions
//
// This is used to normalize coordinates for Morton code generation.
// Uses parallel reduction to find min/max in O(log n) time.

struct BoundsUniforms {
    node_count: u32,
    _padding: vec3<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: BoundsUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;

// Output bounds
@group(0) @binding(2) var<storage, read_write> bounds_min_x: array<f32>;
@group(0) @binding(3) var<storage, read_write> bounds_min_y: array<f32>;
@group(0) @binding(4) var<storage, read_write> bounds_max_x: array<f32>;
@group(0) @binding(5) var<storage, read_write> bounds_max_y: array<f32>;

const WORKGROUP_SIZE: u32 = 256u;
const F32_MAX: f32 = 3.402823466e+38;
const F32_MIN: f32 = -3.402823466e+38;

// Shared memory for reduction
var<workgroup> shared_min_x: array<f32, 256>;
var<workgroup> shared_min_y: array<f32, 256>;
var<workgroup> shared_max_x: array<f32, 256>;
var<workgroup> shared_max_y: array<f32, 256>;

// First pass: reduce within each workgroup
@compute @workgroup_size(256)
fn reduce_bounds(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>,
                 @builtin(workgroup_id) group_id: vec3<u32>) {
    let idx = global_id.x;
    let lid = local_id.x;

    // Load with identity values for out-of-bounds
    var local_min_x = F32_MAX;
    var local_min_y = F32_MAX;
    var local_max_x = F32_MIN;
    var local_max_y = F32_MIN;

    if (idx < uniforms.node_count) {
        let pos = positions[idx];
        local_min_x = pos.x;
        local_min_y = pos.y;
        local_max_x = pos.x;
        local_max_y = pos.y;
    }

    // Store in shared memory
    shared_min_x[lid] = local_min_x;
    shared_min_y[lid] = local_min_y;
    shared_max_x[lid] = local_max_x;
    shared_max_y[lid] = local_max_y;
    workgroupBarrier();

    // Parallel reduction within workgroup
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (lid < stride) {
            shared_min_x[lid] = min(shared_min_x[lid], shared_min_x[lid + stride]);
            shared_min_y[lid] = min(shared_min_y[lid], shared_min_y[lid + stride]);
            shared_max_x[lid] = max(shared_max_x[lid], shared_max_x[lid + stride]);
            shared_max_y[lid] = max(shared_max_y[lid], shared_max_y[lid + stride]);
        }
        workgroupBarrier();
    }

    // First thread writes result
    if (lid == 0u) {
        bounds_min_x[group_id.x] = shared_min_x[0];
        bounds_min_y[group_id.x] = shared_min_y[0];
        bounds_max_x[group_id.x] = shared_max_x[0];
        bounds_max_y[group_id.x] = shared_max_y[0];
    }
}

// Second pass: reduce workgroup results to final bounds
// This assumes the number of workgroups fits in a single workgroup
@compute @workgroup_size(256)
fn finalize_bounds(@builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(num_workgroups) num_groups: vec3<u32>) {
    let lid = local_id.x;
    let num_groups_total = num_groups.x;

    // Load workgroup results
    var local_min_x = F32_MAX;
    var local_min_y = F32_MAX;
    var local_max_x = F32_MIN;
    var local_max_y = F32_MIN;

    if (lid < num_groups_total) {
        local_min_x = bounds_min_x[lid];
        local_min_y = bounds_min_y[lid];
        local_max_x = bounds_max_x[lid];
        local_max_y = bounds_max_y[lid];
    }

    shared_min_x[lid] = local_min_x;
    shared_min_y[lid] = local_min_y;
    shared_max_x[lid] = local_max_x;
    shared_max_y[lid] = local_max_y;
    workgroupBarrier();

    // Reduce
    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (lid < stride) {
            shared_min_x[lid] = min(shared_min_x[lid], shared_min_x[lid + stride]);
            shared_min_y[lid] = min(shared_min_y[lid], shared_min_y[lid + stride]);
            shared_max_x[lid] = max(shared_max_x[lid], shared_max_x[lid + stride]);
            shared_max_y[lid] = max(shared_max_y[lid], shared_max_y[lid + stride]);
        }
        workgroupBarrier();
    }

    // Write final result to first element
    if (lid == 0u) {
        // Add small epsilon to ensure bounds are non-degenerate
        let epsilon = 1.0;
        bounds_min_x[0] = shared_min_x[0] - epsilon;
        bounds_min_y[0] = shared_min_y[0] - epsilon;
        bounds_max_x[0] = shared_max_x[0] + epsilon;
        bounds_max_y[0] = shared_max_y[0] + epsilon;
    }
}

// Combined single-pass bounds for small node counts
@compute @workgroup_size(256)
fn bounds_small(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>) {
    let lid = local_id.x;

    var local_min_x = F32_MAX;
    var local_min_y = F32_MAX;
    var local_max_x = F32_MIN;
    var local_max_y = F32_MIN;

    // Process multiple elements per thread for small arrays
    var idx = lid;
    while (idx < uniforms.node_count) {
        let pos = positions[idx];
        local_min_x = min(local_min_x, pos.x);
        local_min_y = min(local_min_y, pos.y);
        local_max_x = max(local_max_x, pos.x);
        local_max_y = max(local_max_y, pos.y);
        idx += WORKGROUP_SIZE;
    }

    shared_min_x[lid] = local_min_x;
    shared_min_y[lid] = local_min_y;
    shared_max_x[lid] = local_max_x;
    shared_max_y[lid] = local_max_y;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride /= 2u) {
        if (lid < stride) {
            shared_min_x[lid] = min(shared_min_x[lid], shared_min_x[lid + stride]);
            shared_min_y[lid] = min(shared_min_y[lid], shared_min_y[lid + stride]);
            shared_max_x[lid] = max(shared_max_x[lid], shared_max_x[lid + stride]);
            shared_max_y[lid] = max(shared_max_y[lid], shared_max_y[lid + stride]);
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        let epsilon = 1.0;
        bounds_min_x[0] = shared_min_x[0] - epsilon;
        bounds_min_y[0] = shared_min_y[0] - epsilon;
        bounds_max_x[0] = shared_max_x[0] + epsilon;
        bounds_max_y[0] = shared_max_y[0] + epsilon;
    }
}
