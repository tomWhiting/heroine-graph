// Parallel Prefix Sum (Scan) Compute Shader - Reduce-then-Scan Implementation
//
// Implements a reduce / scan-the-sums / propagate parallel scan for radix sort
// histogram processing:
// - Phase 1 (reduce): each workgroup computes the sum of its histogram chunk
// - Phase 2: exclusive scan of those per-workgroup totals, EITHER as a single
//   512-wide Blelloch pass (scan_workgroup_sums, when W <= SCAN_BLOCK_SIZE) or
//   as a three-dispatch hierarchy (scan_sums_blocks / scan_block_sums /
//   propagate_block_offsets) when there are more workgroup sums than one
//   Blelloch pass covers
// - Phase 3 (propagate): each thread adds its workgroup's prefix to local values
//
// This approach handles arbitrary histogram sizes efficiently without sequential
// bottlenecks that would cause GPU watchdog timeouts on large datasets (100K+ nodes).
//
// Histogram layout is DIGIT-MAJOR: [d0_wg0, d0_wg1, ..., d0_wgN, d1_wg0, ...]
// (see radix_sort.comp.wgsl). The scan itself is a plain linear exclusive scan
// over element_count = workgroup_count * RADIX_SIZE values, so with the
// digit-major input each output entry is the global scatter base for its
// (digit, workgroup) pair. The 16-element "chunks" the phases operate on are an
// arbitrary partition of the linear array, not per-digit groups.
//
// PREFIX_SUMS REGION MAP (E = element_count, W = workgroup_count,
// B = ceil(W / SCAN_BLOCK_SIZE)); the host sizes the buffer for all three:
//   [0, E)          final per-bucket exclusive prefix sums (phase 3 output;
//                   what radix_sort.comp.wgsl's scatter reads)
//   [E, E + W)      per-workgroup totals: written by phase 1, replaced in
//                   place by their exclusive scan in phase 2, read by phase 3
//   [E+W, E+W+B)    level-2 block totals: written by scan_sums_blocks,
//                   replaced in place by their exclusive scan in
//                   scan_block_sums, read by propagate_block_offsets.
//                   Untouched on the single-dispatch phase-2 path.
//
// RACE-SAFETY: within one dispatch WGSL guarantees no cross-workgroup
// visibility for plain storage stores, so every phase above that consumes what
// another workgroup wrote is its OWN compute pass. WebGPU orders compute passes
// within a command buffer and makes each pass's storage writes visible to the
// next, which is the only ordering guarantee these phases rely on. Inside a
// workgroup, ordering is workgroupBarrier() as usual.

struct ScanUniforms {
    element_count: u32,     // Total number of histogram elements
    pass_number: u32,       // Radix pass (unused, kept for compatibility)
    workgroup_count: u32,   // Number of workgroups in the radix sort
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ScanUniforms;

// Input: digit-major histogram from radix sort
// Layout: [digit0_wg0, digit0_wg1, ..., digit1_wg0, ...]
@group(0) @binding(1) var<storage, read> histogram: array<u32>;

// Output: prefix sums for scatter phase. Also carries the workgroup-sum and
// block-total scratch regions — see the region map in the header.
@group(0) @binding(2) var<storage, read_write> prefix_sums: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;
const RADIX_SIZE: u32 = 16u;  // 2^4 buckets per radix digit

// Elements one Blelloch pass covers. The scan needs a power-of-two extent and
// shares the 512-u32 `scan_data` array below, so every level of the phase-2
// hierarchy works in blocks of exactly this size (two elements per thread).
const SCAN_BLOCK_SIZE: u32 = 512u;

const WORKGROUP_OFFSET_SLOT: u32 = 16u;  // Shared memory slot for workgroup offset (first unused after RADIX_SIZE)

// Shared memory for parallel scan operations
var<workgroup> scan_data: array<u32, 512>;

// Blelloch work-efficient exclusive scan of the whole `scan_data` array.
//
// Callers load SCAN_BLOCK_SIZE values (zero-padding any tail) and barrier
// first; on return `scan_data` holds their exclusive prefix sums. Must be
// called from uniform control flow — it barriers internally.
//
// Returns the block's INCLUSIVE total, i.e. the sum of everything loaded. The
// up-sweep leaves that value in the last slot, and the exclusive scan then
// destroys it by clearing that slot, so it is read (and broadcast to every
// thread by the barrier pair around it) in between.
fn blelloch_exclusive_scan(tid: u32) -> u32 {
    // Up-sweep (reduce): build the sum tree in place. Unused tail elements are
    // zero-padded by the caller and contribute nothing to any partial sum, so
    // non-power-of-two inputs scan correctly over their live prefix.
    var offset = 1u;
    for (var d = SCAN_BLOCK_SIZE >> 1u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            scan_data[bi] += scan_data[ai];
        }
        offset *= 2u;
    }

    // Capture the total before the exclusive-scan clear overwrites it. The
    // barrier before makes thread 0's final up-sweep write visible to all
    // threads; the barrier after keeps the clear from racing those reads.
    workgroupBarrier();
    let total = scan_data[SCAN_BLOCK_SIZE - 1u];
    workgroupBarrier();
    if (tid == 0u) {
        scan_data[SCAN_BLOCK_SIZE - 1u] = 0u;
    }

    // Down-sweep: traverse the tree back down to turn it into an exclusive scan
    for (var d = 1u; d < SCAN_BLOCK_SIZE; d *= 2u) {
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
    return total;
}

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
// PHASE 2 (flat): Scan Workgroup Sums - up to SCAN_BLOCK_SIZE totals
// =============================================================================
// Single workgroup scans W <= SCAN_BLOCK_SIZE (512) workgroup totals in one
// Blelloch pass. This gives each workgroup its starting offset in the global
// output, and is the whole of phase 2 for element counts up to
// 512 * 256 = 131,072. Above that the host dispatches the hierarchy below
// instead; the two produce identical contents in [E, E + W).
// Reads from and writes to: prefix_sums[element_count + 0..workgroup_count-1]

@compute @workgroup_size(256)
fn scan_workgroup_sums(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let n = uniforms.workgroup_count;
    let base_offset = uniforms.element_count;

    // The host only dispatches this entry point when W fits one block; clamp
    // anyway so an over-large W degrades to a wrong-but-bounded scan rather
    // than reading past the workgroup-sums region.
    let safe_n = min(n, SCAN_BLOCK_SIZE);

    // Load workgroup sums into shared memory (2 elements per thread),
    // zero-padding the tail so the power-of-two scan is still correct.
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

    // The grand total is discarded: with a single block there is no higher
    // level to carry it to.
    _ = blelloch_exclusive_scan(tid);

    // Write scanned workgroup sums back (exclusive prefix sums)
    // These are the base offsets for each workgroup
    if (idx1 < safe_n) {
        prefix_sums[base_offset + idx1] = scan_data[idx1];
    }
    if (idx2 < safe_n) {
        prefix_sums[base_offset + idx2] = scan_data[idx2];
    }
}

// =============================================================================
// PHASE 2a (hierarchical): Scan each block of SCAN_BLOCK_SIZE workgroup sums
// =============================================================================
// Dispatched with B = ceil(W / SCAN_BLOCK_SIZE) workgroups when W exceeds one
// block. Workgroup b exclusive-scans the sums in [b*512, min((b+1)*512, W)) in
// place and publishes that block's total to the level-2 region at
// prefix_sums[E + W + b]. The per-block scans are independent — no workgroup
// reads another's output — which is exactly why this level can be one dispatch.

@compute @workgroup_size(256)
fn scan_sums_blocks(@builtin(local_invocation_id) local_id: vec3<u32>,
                    @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let block = workgroup_id.x;
    let n = uniforms.workgroup_count;
    let sums_base = uniforms.element_count;
    let totals_base = sums_base + n;
    let block_base = block * SCAN_BLOCK_SIZE;

    let idx1 = tid * 2u;
    let idx2 = tid * 2u + 1u;

    if (block_base + idx1 < n) {
        scan_data[idx1] = prefix_sums[sums_base + block_base + idx1];
    } else {
        scan_data[idx1] = 0u;
    }

    if (block_base + idx2 < n) {
        scan_data[idx2] = prefix_sums[sums_base + block_base + idx2];
    } else {
        scan_data[idx2] = 0u;
    }

    workgroupBarrier();

    let block_total = blelloch_exclusive_scan(tid);

    // Level-2 input: the sum of every workgroup sum in this block. Writing it
    // from one thread keeps the store single-writer; nothing in this dispatch
    // reads the level-2 region.
    if (tid == 0u) {
        prefix_sums[totals_base + block] = block_total;
    }

    // Block-local exclusive scan, still missing the offset of all EARLIER
    // blocks — propagate_block_offsets adds that after scan_block_sums runs.
    if (block_base + idx1 < n) {
        prefix_sums[sums_base + block_base + idx1] = scan_data[idx1];
    }
    if (block_base + idx2 < n) {
        prefix_sums[sums_base + block_base + idx2] = scan_data[idx2];
    }
}

// =============================================================================
// PHASE 2b (hierarchical): Scan the block totals
// =============================================================================
// Single workgroup, one Blelloch pass over the B block totals written by
// scan_sums_blocks. B <= SCAN_BLOCK_SIZE is what caps the whole sort at
// 512 blocks * 512 sums * 256 threads = 67,108,864 elements; the host refuses
// to record a larger sort rather than letting this silently truncate.
// Reads from and writes to: prefix_sums[E + W .. E + W + B).

@compute @workgroup_size(256)
fn scan_block_sums(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let n = uniforms.workgroup_count;
    let totals_base = uniforms.element_count + n;
    // Derived, not a uniform: the host already publishes workgroup_count and
    // the block size is a shader constant.
    let block_count = min(
        (n + SCAN_BLOCK_SIZE - 1u) / SCAN_BLOCK_SIZE,
        SCAN_BLOCK_SIZE,
    );

    let idx1 = tid * 2u;
    let idx2 = tid * 2u + 1u;

    if (idx1 < block_count) {
        scan_data[idx1] = prefix_sums[totals_base + idx1];
    } else {
        scan_data[idx1] = 0u;
    }

    if (idx2 < block_count) {
        scan_data[idx2] = prefix_sums[totals_base + idx2];
    } else {
        scan_data[idx2] = 0u;
    }

    workgroupBarrier();

    // Grand total discarded: this is the top level of the hierarchy.
    _ = blelloch_exclusive_scan(tid);

    if (idx1 < block_count) {
        prefix_sums[totals_base + idx1] = scan_data[idx1];
    }
    if (idx2 < block_count) {
        prefix_sums[totals_base + idx2] = scan_data[idx2];
    }
}

// =============================================================================
// PHASE 2c (hierarchical): Add each block's offset to its workgroup sums
// =============================================================================
// Dispatched with the same B workgroups as scan_sums_blocks. Workgroup b adds
// the scanned offset of block b to every workgroup sum inside block b, which
// turns the per-block scans into the single global exclusive scan of all W
// workgroup sums — byte-for-byte what scan_workgroup_sums leaves in
// [E, E + W) for W <= SCAN_BLOCK_SIZE, so phase 3 is unchanged either way.

@compute @workgroup_size(256)
fn propagate_block_offsets(@builtin(local_invocation_id) local_id: vec3<u32>,
                           @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let block = workgroup_id.x;
    let n = uniforms.workgroup_count;
    let sums_base = uniforms.element_count;
    let totals_base = sums_base + n;
    let block_base = block * SCAN_BLOCK_SIZE;

    // Read directly rather than broadcasting through shared memory: the
    // level-2 region was written by an earlier dispatch and nothing in this
    // one writes it, so every thread reading the same address is a plain
    // uniform load with no ordering requirement.
    let block_offset = prefix_sums[totals_base + block];

    // Each element is touched by exactly one thread of exactly one workgroup,
    // so the read-modify-write below needs no atomics.
    let idx1 = tid * 2u;
    let idx2 = tid * 2u + 1u;

    if (block_base + idx1 < n) {
        prefix_sums[sums_base + block_base + idx1] += block_offset;
    }
    if (block_base + idx2 < n) {
        prefix_sums[sums_base + block_base + idx2] += block_offset;
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
// SINGLE-PASS SCAN - For small arrays (up to SCAN_BLOCK_SIZE elements)
// =============================================================================
// Used when total histogram size fits in single workgroup shared memory.
// More efficient than the multi-phase approach for small datasets: it scans the
// histogram itself, so phases 1-3 are skipped entirely.

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

    // Grand total discarded: one block covers the whole histogram here.
    _ = blelloch_exclusive_scan(tid);

    // Write results back (exclusive prefix sum)
    if (idx1 < total_buckets) {
        prefix_sums[idx1] = scan_data[idx1];
    }
    if (idx2 < total_buckets) {
        prefix_sums[idx2] = scan_data[idx2];
    }
}
