// Parallel Radix Sort Compute Shader
// Sorts Morton codes and their associated node indices using GPU-parallel radix sort
//
// This implements a 4-bit radix sort with parallel prefix sum (scan) for histogram
// computation. The sort operates in 8 passes (32 bits / 4 bits per pass).

struct SortUniforms {
    node_count: u32,
    pass_number: u32,     // Current radix pass (0-7)
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: SortUniforms;

// Input/output buffers (ping-pong between passes)
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read> values_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> values_out: array<u32>;

// Histogram for counting digits
@group(0) @binding(5) var<storage, read_write> histogram: array<atomic<u32>>;

// Prefix sums for scattering
@group(0) @binding(6) var<storage, read> prefix_sums: array<u32>;

const RADIX_BITS: u32 = 4u;
const RADIX_SIZE: u32 = 16u;  // 2^4 = 16 buckets
const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> local_histogram: array<atomic<u32>, 16>;
var<workgroup> local_prefix: array<u32, 16>;

// Extract 4-bit digit at given pass
fn get_digit(key: u32, pass: u32) -> u32 {
    let shift = pass * RADIX_BITS;
    return (key >> shift) & 0xFu;
}

// Phase 1: Count histogram of digits
@compute @workgroup_size(256)
fn histogram_count(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) group_id: vec3<u32>) {
    let idx = global_id.x;

    // Initialize local histogram
    if (local_id.x < RADIX_SIZE) {
        atomicStore(&local_histogram[local_id.x], 0u);
    }
    workgroupBarrier();

    // Count digits in this workgroup
    if (idx < uniforms.node_count) {
        let key = keys_in[idx];
        let digit = get_digit(key, uniforms.pass_number);
        atomicAdd(&local_histogram[digit], 1u);
    }
    workgroupBarrier();

    // Add local histogram to global histogram
    if (local_id.x < RADIX_SIZE) {
        let count = atomicLoad(&local_histogram[local_id.x]);
        if (count > 0u) {
            let global_bucket = group_id.x * RADIX_SIZE + local_id.x;
            atomicAdd(&histogram[global_bucket], count);
        }
    }
}

// Phase 2: Compute prefix sum (scan) on histogram
// This is done separately with a single workgroup scan shader

// Phase 3: Scatter elements to sorted positions
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) global_id: vec3<u32>,
           @builtin(local_invocation_id) local_id: vec3<u32>,
           @builtin(workgroup_id) group_id: vec3<u32>) {
    let idx = global_id.x;

    // Load prefix sums for this workgroup's buckets
    if (local_id.x < RADIX_SIZE) {
        let global_bucket = group_id.x * RADIX_SIZE + local_id.x;
        local_prefix[local_id.x] = prefix_sums[global_bucket];
    }
    workgroupBarrier();

    if (idx >= uniforms.node_count) {
        return;
    }

    let key = keys_in[idx];
    let value = values_in[idx];
    let digit = get_digit(key, uniforms.pass_number);

    // Compute position within bucket
    // This requires counting how many elements with same digit came before
    // in this workgroup (simplified - full implementation needs local scan)
    let base_pos = local_prefix[digit];

    // Atomic increment to get unique position
    // Note: This is a simplified version. Full parallel radix sort uses
    // more sophisticated techniques to avoid atomic contention.
    var<workgroup> bucket_offsets: array<atomic<u32>, 16>;
    if (local_id.x < RADIX_SIZE) {
        atomicStore(&bucket_offsets[local_id.x], 0u);
    }
    workgroupBarrier();

    let local_offset = atomicAdd(&bucket_offsets[digit], 1u);
    let output_pos = base_pos + local_offset;

    // Write to output
    if (output_pos < uniforms.node_count) {
        keys_out[output_pos] = key;
        values_out[output_pos] = value;
    }
}

// Simple single-pass sort for small arrays (fallback)
@compute @workgroup_size(256)
fn simple_sort(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= uniforms.node_count) {
        return;
    }

    let key = keys_in[idx];
    let value = values_in[idx];

    // Count how many keys are smaller (gives sorted position)
    var pos = 0u;
    for (var i = 0u; i < uniforms.node_count; i++) {
        let other_key = keys_in[i];
        if (other_key < key || (other_key == key && i < idx)) {
            pos++;
        }
    }

    keys_out[pos] = key;
    values_out[pos] = value;
}
