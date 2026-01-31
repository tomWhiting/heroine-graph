// Parallel Prefix Sum (Scan) Compute Shader
// Implements Blelloch-style work-efficient exclusive scan
//
// Used for radix sort histogram to compute scatter offsets.
// This shader operates on a single workgroup and handles arrays up to 2048 elements.
// For larger arrays, a multi-level scan would be needed.

struct ScanUniforms {
    element_count: u32,     // Number of elements to scan
    pass_number: u32,       // Radix pass (for multi-pass sort)
    workgroup_count: u32,   // Number of workgroups in the sort
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ScanUniforms;

// Input: per-workgroup histograms from radix sort
// Layout: [wg0_bucket0, wg0_bucket1, ..., wg0_bucket15, wg1_bucket0, ...]
@group(0) @binding(1) var<storage, read> histogram: array<u32>;

// Output: prefix sums for scatter phase
@group(0) @binding(2) var<storage, read_write> prefix_sums: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;
const RADIX_SIZE: u32 = 16u;  // 2^4 buckets

// Shared memory for scan
var<workgroup> scan_data: array<u32, 512>;  // Double buffer for work-efficient scan

// Single-workgroup exclusive prefix sum
// Handles up to 512 elements (workgroup_count * RADIX_SIZE)
@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;

    // Total elements = workgroup_count * RADIX_SIZE
    // We need to compute exclusive prefix sum across all histogram buckets
    // organized as: [wg0_b0, wg0_b1, ..., wg0_b15, wg1_b0, ...]

    let total_buckets = uniforms.workgroup_count * RADIX_SIZE;

    // Load data into shared memory (two elements per thread)
    let idx1 = tid * 2u;
    let idx2 = tid * 2u + 1u;

    if (idx1 < total_buckets) {
        scan_data[idx1] = histogram[idx1];
    } else {
        scan_data[idx1] = 0u;
    }

    if (idx2 < total_buckets) {
        scan_data[idx2] = histogram[idx2];
    } else {
        scan_data[idx2] = 0u;
    }

    workgroupBarrier();

    // Up-sweep (reduce) phase
    var offset = 1u;
    var n = 512u;  // Array size (power of 2)

    // Build sum tree
    for (var d = n >> 1u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            scan_data[bi] += scan_data[ai];
        }
        offset *= 2u;
    }

    // Clear the last element (for exclusive scan)
    if (tid == 0u) {
        scan_data[n - 1u] = 0u;
    }

    // Down-sweep phase
    for (var d = 1u; d < n; d *= 2u) {
        offset >>= 1u;
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            let t = scan_data[ai];
            scan_data[ai] = scan_data[bi];
            scan_data[bi] += t;
        }
    }

    workgroupBarrier();

    // Write results back (exclusive prefix sum)
    if (idx1 < total_buckets) {
        prefix_sums[idx1] = scan_data[idx1];
    }
    if (idx2 < total_buckets) {
        prefix_sums[idx2] = scan_data[idx2];
    }
}

// Alternative: Simple sequential scan for small arrays
// Used when array is small enough that a single thread can handle it efficiently
@compute @workgroup_size(1)
fn scan_sequential(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_buckets = uniforms.workgroup_count * RADIX_SIZE;

    var running_sum = 0u;
    for (var i = 0u; i < total_buckets; i++) {
        let val = histogram[i];
        prefix_sums[i] = running_sum;
        running_sum += val;
    }
}

// Rearrange histogram from per-workgroup to per-bucket layout
// Input: [wg0_b0, wg0_b1, ..., wg0_b15, wg1_b0, ...]
// Output: [b0_wg0, b0_wg1, ..., b0_wgN, b1_wg0, ...]
// This makes the prefix sum give us the correct global offsets
@compute @workgroup_size(256)
fn transpose_histogram(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_buckets = uniforms.workgroup_count * RADIX_SIZE;

    if (idx >= total_buckets) {
        return;
    }

    // Current position: workgroup * RADIX_SIZE + bucket
    let wg = idx / RADIX_SIZE;
    let bucket = idx % RADIX_SIZE;

    // New position: bucket * workgroup_count + workgroup
    let new_idx = bucket * uniforms.workgroup_count + wg;

    prefix_sums[new_idx] = histogram[idx];
}
