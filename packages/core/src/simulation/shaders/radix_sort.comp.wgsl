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
var<workgroup> bucket_offsets: array<atomic<u32>, 16>;

// Extract 4-bit digit at given radix pass
fn get_digit(key: u32, pass_idx: u32) -> u32 {
    let shift = pass_idx * RADIX_BITS;
    return (key >> shift) & 0xFu;
}

// Phase 1: Count histogram of digits
@compute @workgroup_size(256)
fn histogram_count(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) group_id: vec3<u32>) {
    let idx = global_id.x;
    let is_valid = idx < uniforms.node_count;

    // Initialize local histogram - all threads participate
    if (local_id.x < RADIX_SIZE) {
        atomicStore(&local_histogram[local_id.x], 0u);
    }
    workgroupBarrier();

    // Count digits in this workgroup - only valid threads contribute
    if (is_valid) {
        let key = keys_in[idx];
        let digit = get_digit(key, uniforms.pass_number);
        atomicAdd(&local_histogram[digit], 1u);
    }
    workgroupBarrier();

    // Add local histogram to global histogram - all threads participate in sync
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
    let is_valid = idx < uniforms.node_count;

    // Load prefix sums for this workgroup's buckets - all threads participate
    if (local_id.x < RADIX_SIZE) {
        let global_bucket = group_id.x * RADIX_SIZE + local_id.x;
        local_prefix[local_id.x] = prefix_sums[global_bucket];
    }
    workgroupBarrier();

    // Initialize bucket offsets - all threads participate
    if (local_id.x < RADIX_SIZE) {
        atomicStore(&bucket_offsets[local_id.x], 0u);
    }
    workgroupBarrier();

    // Load data for valid threads
    var key = 0u;
    var value = 0u;
    var digit = 0u;
    if (is_valid) {
        key = keys_in[idx];
        value = values_in[idx];
        digit = get_digit(key, uniforms.pass_number);
    }

    // Compute position within bucket using atomic increment
    // All threads compute, but only valid ones will write
    var output_pos = 0u;
    if (is_valid) {
        let base_pos = local_prefix[digit];
        let local_offset = atomicAdd(&bucket_offsets[digit], 1u);
        output_pos = base_pos + local_offset;
    }

    // Write to output - only valid threads with valid positions
    if (is_valid && output_pos < uniforms.node_count) {
        keys_out[output_pos] = key;
        values_out[output_pos] = value;
    }
}

// Clear histogram buffer before each radix pass
// Each thread clears multiple histogram entries to cover all workgroup buckets
// Must be dispatched with enough threads to cover workgroup_count * RADIX_SIZE elements
@compute @workgroup_size(256)
fn clear_histogram(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    // Use node_count to derive the maximum histogram index we need to clear
    // Histogram size = ceil(node_count / WORKGROUP_SIZE) * RADIX_SIZE
    // We over-clear slightly to ensure all buckets are zeroed
    let workgroup_count = (uniforms.node_count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    let max_histogram_size = workgroup_count * RADIX_SIZE;

    if (idx < max_histogram_size) {
        atomicStore(&histogram[idx], 0u);
    }
}

// Simple counting sort for small arrays or single-pass fallback
// Each thread computes its sorted position by counting smaller keys
// O(n²) but GPU-parallel and works correctly with WGSL uniform control flow
@compute @workgroup_size(256)
fn simple_sort(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let is_valid = idx < uniforms.node_count;

    // Touch histogram and prefix_sums to ensure they're included in bind group layout
    // (needed because we share the same bind group as histogram_count/scatter)
    _ = atomicLoad(&histogram[0]);
    _ = prefix_sums[0];

    // Load key and value (use 0 for out-of-bounds threads)
    var key = 0u;
    var value = 0u;
    if (is_valid) {
        key = keys_in[idx];
        value = values_in[idx];
    }

    // Count how many keys are smaller (gives sorted position)
    var pos = 0u;
    if (is_valid) {
        for (var i = 0u; i < uniforms.node_count; i++) {
            let other_key = keys_in[i];
            // Sort stable: use index as tiebreaker
            if (other_key < key || (other_key == key && i < idx)) {
                pos++;
            }
        }
    }

    // Write to sorted position
    if (is_valid) {
        keys_out[pos] = key;
        values_out[pos] = value;
    }
}
