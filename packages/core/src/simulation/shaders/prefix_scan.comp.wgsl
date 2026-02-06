// Parallel Prefix Sum (Scan) Compute Shader - Reduce-then-Scan Implementation
//
// Implements a three-phase parallel scan for radix sort histogram processing:
// - Phase 1 (reduce): Each workgroup computes sum of its histogram chunk
// - Phase 2 (scan_workgroup_sums): Single workgroup scans the per-workgroup totals
// - Phase 3 (propagate): Each thread adds its workgroup's prefix to local values
//
// This approach handles arbitrary histogram sizes efficiently without sequential
// bottlenecks that would cause GPU watchdog timeouts on large datasets (100K+ nodes).
//
// Histogram layout: [wg0_b0, wg0_b1, ..., wg0_b15, wg1_b0, ..., wg1_b15, ...]
// Each workgroup has RADIX_SIZE (16) buckets.

struct ScanUniforms {
    element_count: u32,     // Total number of histogram elements
    pass_number: u32,       // Radix pass (unused, kept for compatibility)
    workgroup_count: u32,   // Number of workgroups in the radix sort
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ScanUniforms;

// Input: per-workgroup histograms from radix sort
// Layout: [wg0_bucket0, wg0_bucket1, ..., wg0_bucket15, wg1_bucket0, ...]
@group(0) @binding(1) var<storage, read> histogram: array<u32>;

// Output: prefix sums for scatter phase
// Also used as intermediate storage for workgroup sums during multi-phase scan
@group(0) @binding(2) var<storage, read_write> prefix_sums: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;
const RADIX_SIZE: u32 = 16u;  // 2^4 buckets per radix digit
const MAX_WORKGROUP_SUMS: u32 = 512u;  // Max workgroups we can scan in phase 2
const WORKGROUP_OFFSET_SLOT: u32 = 16u;  // Shared memory slot for workgroup offset (first unused after RADIX_SIZE)

// Shared memory for parallel scan operations
var<workgroup> scan_data: array<u32, 512>;

// =============================================================================
// PHASE 1: Reduce - Compute per-workgroup sums
// =============================================================================
// Each workgroup computes the sum of its RADIX_SIZE (16) histogram values.
// Output stored at prefix_sums[element_count + wg_id] to separate workgroup sums
// from per-bucket prefix sums (which occupy indices 0..element_count-1).
// Race condition safety between phases is guaranteed by WebGPU compute pass ordering.

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) global_id: vec3<u32>,
          @builtin(local_invocation_id) local_id: vec3<u32>,
          @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let wg_id = workgroup_id.x;
    let tid = local_id.x;

    // Each sort workgroup has RADIX_SIZE (16) histogram buckets
    // We dispatch one reduce workgroup per sort workgroup
    if (wg_id >= uniforms.workgroup_count) {
        return;
    }

    let base_idx = wg_id * RADIX_SIZE;

    // First 16 threads load the histogram values for this workgroup
    var value = 0u;
    if (tid < RADIX_SIZE && base_idx + tid < uniforms.element_count) {
        value = histogram[base_idx + tid];
    }

    // Store in shared memory
    if (tid < RADIX_SIZE) {
        scan_data[tid] = value;
    }
    workgroupBarrier();

    // Parallel reduction to compute sum of 16 values
    // Tree reduction: 16 -> 8 -> 4 -> 2 -> 1
    if (tid < 8u) {
        scan_data[tid] += scan_data[tid + 8u];
    }
    workgroupBarrier();

    if (tid < 4u) {
        scan_data[tid] += scan_data[tid + 4u];
    }
    workgroupBarrier();

    if (tid < 2u) {
        scan_data[tid] += scan_data[tid + 2u];
    }
    workgroupBarrier();

    if (tid == 0u) {
        let total = scan_data[0] + scan_data[1];
        // Store workgroup sum after the per-bucket prefix sums region.
        // This separates workgroup totals from the final output.
        prefix_sums[uniforms.element_count + wg_id] = total;
    }
}

// =============================================================================
// PHASE 2: Scan Workgroup Sums - Exclusive prefix sum of per-workgroup totals
// =============================================================================
// Single workgroup scans up to MAX_WORKGROUP_SUMS (512) workgroup totals.
// This gives each workgroup its starting offset in the global output.
// Uses Blelloch-style work-efficient parallel scan.
// Reads from and writes to: prefix_sums[element_count + 0..workgroup_count-1]

@compute @workgroup_size(256)
fn scan_workgroup_sums(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let n = uniforms.workgroup_count;
    let base_offset = uniforms.element_count;

    // Clamp to maximum supported workgroups
    // If workgroup_count > 512, this algorithm won't work correctly.
    // For 100K nodes with 256 threads/wg, we have ~391 workgroups, well under 512.
    // For 131K nodes we'd have 512 workgroups - exactly at limit.
    // For safety, we cap at 512. Larger counts would need hierarchical scan.
    let safe_n = min(n, MAX_WORKGROUP_SUMS);

    // Load workgroup sums into shared memory (2 elements per thread)
    // Read from prefix_sums[element_count + idx]
    let idx1 = tid * 2u;
    let idx2 = tid * 2u + 1u;

    if (idx1 < safe_n) {
        scan_data[idx1] = prefix_sums[base_offset + idx1];
    } else {
        scan_data[idx1] = 0u;
    }

    if (idx2 < safe_n) {
        scan_data[idx2] = prefix_sums[base_offset + idx2];
    } else {
        scan_data[idx2] = 0u;
    }

    workgroupBarrier();

    // Up-sweep (reduce) phase - build sum tree
    // The Blelloch scan operates on the full 512-element array regardless of safe_n.
    // For non-power-of-2 workgroup counts, unused elements are zero-padded (lines 128, 134).
    // Zeros contribute nothing to partial sums, ensuring correct exclusive scan for [0..safe_n).
    var offset = 1u;
    let array_size = 512u;  // Power of 2 required for Blelloch scan

    for (var d = array_size >> 1u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            scan_data[bi] += scan_data[ai];
        }
        offset *= 2u;
    }

    // Clear the last element for exclusive scan
    if (tid == 0u) {
        scan_data[array_size - 1u] = 0u;
    }

    // Down-sweep phase - traverse tree to build scan
    for (var d = 1u; d < array_size; d *= 2u) {
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

    // Write scanned workgroup sums back (exclusive prefix sums)
    // These are the base offsets for each workgroup
    // Write to prefix_sums[element_count + idx]
    if (idx1 < safe_n) {
        prefix_sums[base_offset + idx1] = scan_data[idx1];
    }
    if (idx2 < safe_n) {
        prefix_sums[base_offset + idx2] = scan_data[idx2];
    }
}

// =============================================================================
// PHASE 3: Propagate - Compute final prefix sums for all histogram elements
// =============================================================================
// Each workgroup computes the local exclusive scan within its 16 buckets,
// then adds the workgroup's base offset (from Phase 2) to get global prefix sums.
// Reads workgroup offset from: prefix_sums[element_count + wg_id]
// Writes final sums to: prefix_sums[wg_id * RADIX_SIZE + 0..15]

@compute @workgroup_size(256)
fn propagate(@builtin(global_invocation_id) global_id: vec3<u32>,
             @builtin(local_invocation_id) local_id: vec3<u32>,
             @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let wg_id = workgroup_id.x;
    let tid = local_id.x;

    if (wg_id >= uniforms.workgroup_count) {
        return;
    }

    let base_idx = wg_id * RADIX_SIZE;

    // Load workgroup base offset (computed in Phase 2)
    // Stored at prefix_sums[element_count + wg_id], separate from per-bucket sums.
    // WebGPU compute pass ordering guarantees Phase 2 completes before Phase 3 starts.
    var workgroup_offset = 0u;
    if (tid == 0u) {
        workgroup_offset = prefix_sums[uniforms.element_count + wg_id];
        scan_data[WORKGROUP_OFFSET_SLOT] = workgroup_offset;  // Broadcast to other threads via shared memory
    }
    workgroupBarrier();
    workgroup_offset = scan_data[WORKGROUP_OFFSET_SLOT];

    // First 16 threads load histogram values
    if (tid < RADIX_SIZE && base_idx + tid < uniforms.element_count) {
        scan_data[tid] = histogram[base_idx + tid];
    } else if (tid < RADIX_SIZE) {
        scan_data[tid] = 0u;
    }
    workgroupBarrier();

    // Compute local exclusive prefix sum for 16 elements
    // Simple sequential scan since it's only 16 elements
    if (tid == 0u) {
        var running_sum = 0u;
        for (var i = 0u; i < RADIX_SIZE; i++) {
            let val = scan_data[i];
            scan_data[i] = running_sum + workgroup_offset;
            running_sum += val;
        }
    }
    workgroupBarrier();

    // Write final prefix sums to the beginning of the buffer
    if (tid < RADIX_SIZE && base_idx + tid < uniforms.element_count) {
        prefix_sums[base_idx + tid] = scan_data[tid];
    }
}

// =============================================================================
// SINGLE-PASS SCAN - For small arrays (up to 512 elements)
// =============================================================================
// Used when total histogram size fits in single workgroup shared memory.
// More efficient than multi-phase approach for small datasets.

@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;

    // Total elements = workgroup_count * RADIX_SIZE
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
    let n = 512u;  // Array size (power of 2)

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

