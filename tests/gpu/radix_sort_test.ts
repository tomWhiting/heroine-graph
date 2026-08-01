/**
 * GPU tests for the shared LSD radix sort (findings #8 and #11):
 *
 * - #8: the histogram/scan layout was workgroup-major, so the scatter never
 *   moved elements across 256-element workgroup boundaries — the buffer was
 *   never globally sorted for >= 1024 elements. The fixed shader uses a
 *   digit-major layout whose linear exclusive scan yields true global
 *   scatter offsets.
 * - #11: within-workgroup ranks were assigned by atomicAdd arrival order,
 *   destroying the per-pass stability LSD radix sort requires. The fixed
 *   shader counts equal-digit elements at lower buffer index.
 *
 * Both are asserted end-to-end: keys must come out fully sorted AND stable
 * (payload values carry the original index; equal keys must preserve index
 * order), across duplicate-heavy and full-range distributions, on both the
 * single-pass-scan (<= 512 histogram buckets) and three-phase-scan paths,
 * plus the small-N simple-sort path.
 *
 * The large cases below cover the hierarchical phase-2 scan (task #8). Phase 2
 * used to be a single 512-wide Blelloch pass over the per-workgroup sums, so
 * recordRadixSort refused anything past 512 * 256 = 131,072 elements and
 * Barnes-Hut was simply unavailable above that. It is now three dispatches
 * (scan_sums_blocks / scan_block_sums / propagate_block_offsets) whenever there
 * are more workgroup sums than one pass covers, which is why the boundary
 * tests straddle 131,072 exactly.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  GPU_SKIP_MESSAGE,
  loadModuleInliningWgsl,
  probeAdapter,
  requestHarnessDevice,
} from "../helpers/gpu.ts";
import { mulberry32 } from "../fixtures/prng.ts";

const adapter = await probeAdapter();
if (!adapter) {
  console.warn(`[gpu] ${GPU_SKIP_MESSAGE}`);
}

function gpuTest(name: string, fn: (device: GPUDevice) => Promise<void>): void {
  Deno.test({
    name,
    ignore: adapter === null,
    sanitizeResources: false,
    sanitizeOps: false,
    async fn() {
      const device = await requestHarnessDevice(adapter!);
      try {
        await fn(device);
      } finally {
        device.destroy();
      }
    },
  });
}

/**
 * Structural subset of utils/radix_sort.ts (loaded with .wgsl inlined —
 * see tests/helpers/gpu.ts module doc). Buffer/pipeline handles are opaque.
 */
interface RadixSortModule {
  createRadixSortPipeline(device: GPUDevice, label?: string): unknown;
  createRadixSortBuffers(device: GPUDevice, maxElements: number, label?: string): {
    keysA: GPUBuffer;
    keysB: GPUBuffer;
    valuesA: GPUBuffer;
    valuesB: GPUBuffer;
    maxElements: number;
  };
  createRadixSortBindGroups(device: GPUDevice, pipeline: unknown, buffers: unknown): unknown;
  updateRadixSortUniforms(device: GPUDevice, buffers: unknown, elementCount: number): void;
  recordRadixSort(
    encoder: GPUCommandEncoder,
    pipeline: unknown,
    bindGroups: unknown,
    buffers: unknown,
    elementCount: number,
    label?: string,
  ): boolean;
  wasSimpleSort(elementCount: number): boolean;
  destroyRadixSortBuffers(buffers: unknown): void;
}

function loadRadixSortModule(): Promise<RadixSortModule> {
  return loadModuleInliningWgsl<RadixSortModule>(
    new URL("../../packages/core/src/simulation/utils/radix_sort.ts", import.meta.url),
  );
}

async function readbackU32(device: GPUDevice, src: GPUBuffer, count: number): Promise<Uint32Array> {
  const staging = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(src, 0, staging, 0, count * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}

/**
 * Runs the full radix sort over the given keys (payload = original index)
 * and returns the sorted keys/values read back from the GPU.
 */
async function runSort(
  device: GPUDevice,
  keys: Uint32Array<ArrayBuffer>,
): Promise<{ keys: Uint32Array; values: Uint32Array }> {
  const mod = await loadRadixSortModule();
  const n = keys.length;

  const pipeline = mod.createRadixSortPipeline(device, "Test");
  const buffers = mod.createRadixSortBuffers(device, n, "Test");
  const bindGroups = mod.createRadixSortBindGroups(device, pipeline, buffers);

  const values = new Uint32Array(n);
  for (let i = 0; i < n; i++) values[i] = i;
  device.queue.writeBuffer(buffers.keysA, 0, keys);
  device.queue.writeBuffer(buffers.valuesA, 0, values);

  mod.updateRadixSortUniforms(device, buffers, n);
  const encoder = device.createCommandEncoder();
  const recorded = mod.recordRadixSort(encoder, pipeline, bindGroups, buffers, n, "Test");
  assert(recorded, "recordRadixSort refused to record");
  device.queue.submit([encoder.finish()]);

  // Simple sort (1 pass) lands in B; full radix (8 passes) lands back in A
  const simple = mod.wasSimpleSort(n);
  const sortedKeys = await readbackU32(device, simple ? buffers.keysB : buffers.keysA, n);
  const sortedValues = await readbackU32(device, simple ? buffers.valuesB : buffers.valuesA, n);

  mod.destroyRadixSortBuffers(buffers);
  return { keys: sortedKeys, values: sortedValues };
}

/**
 * Asserts sortedness, stability (equal keys keep ascending original-index
 * payloads), and that the payloads form a permutation of 0..n-1 (no element
 * dropped or duplicated by the scatter).
 */
function assertSortedStablePermutation(
  input: Uint32Array,
  keys: Uint32Array,
  values: Uint32Array,
  label: string,
): void {
  const n = input.length;
  const seen = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const v = values[i];
    assert(v < n, `${label}: payload ${v} out of range at ${i}`);
    assert(seen[v] === 0, `${label}: payload ${v} duplicated (scatter overwrote a slot)`);
    seen[v] = 1;
    assertEquals(
      keys[i],
      input[v],
      `${label}: key/payload pair broken at ${i} (key ${keys[i]}, payload ${v})`,
    );
    if (i > 0) {
      assert(
        keys[i] >= keys[i - 1],
        `${label}: not sorted at ${i}: ${keys[i - 1]} > ${keys[i]}`,
      );
      if (keys[i] === keys[i - 1]) {
        assert(
          values[i] > values[i - 1],
          `${label}: stability broken at ${i}: equal keys ${keys[i]} with payloads ` +
            `${values[i - 1]} then ${values[i]}`,
        );
      }
    }
  }
}

/**
 * The stable CPU sort of `keys`, as the permutation of original indices the
 * GPU's payload column must equal.
 *
 * {@link assertSortedStablePermutation} already pins the result uniquely — a
 * non-decreasing permutation that preserves index order among equal keys IS
 * the stable sort — but the large cases compare against this explicitly, since
 * an independent reference is what makes a 200K-element result checkable
 * rather than merely self-consistent.
 */
function cpuStableSortIndices(keys: Uint32Array): Uint32Array {
  const order = new Uint32Array(keys.length);
  for (let i = 0; i < order.length; i++) order[i] = i;
  // Keys are full-range u32; the differences stay exact in f64.
  return order.sort((a, b) => (keys[a] - keys[b]) || (a - b));
}

/** Asserts the GPU result equals {@link cpuStableSortIndices} element for element. */
function assertMatchesCpuSort(
  input: Uint32Array,
  keys: Uint32Array,
  values: Uint32Array,
  label: string,
): void {
  const reference = cpuStableSortIndices(input);
  for (let i = 0; i < reference.length; i++) {
    assertEquals(values[i], reference[i], `${label}: payload differs from CPU sort at ${i}`);
    assertEquals(keys[i], input[reference[i]], `${label}: key differs from CPU sort at ${i}`);
  }
}

/** Seeded full-range u32 keys with a scattering of forced duplicates. */
function randomKeys(n: number, seed: number, duplicateRate = 0.02): Uint32Array<ArrayBuffer> {
  const rng = mulberry32(seed);
  const keys = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    keys[i] = rng() < duplicateRate ? 0xC0FFEE00 : (Math.floor(rng() * 0x100000000) >>> 0);
  }
  return keys;
}

gpuTest(
  "GPU radix sort: 12K duplicate-heavy keys come out sorted and stable",
  async (device) => {
    // 32 distinct key values over 12,288 elements: every workgroup holds
    // hundreds of equal-digit elements, the worst case for both the old
    // workgroup-major layout and the old arrival-order rank assignment.
    const n = 12288;
    const rng = mulberry32(101);
    const keys = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      keys[i] = Math.floor(rng() * 32) * 0x08040201; // duplicates with bits in every digit
    }
    const { keys: sorted, values } = await runSort(device, keys);
    assertSortedStablePermutation(keys, sorted, values, "duplicate-heavy");
  },
);

gpuTest(
  "GPU radix sort: 10K full-range random u32 keys come out sorted and stable",
  async (device) => {
    const n = 10240;
    const rng = mulberry32(202);
    const keys = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      // Full 32-bit range; ~2.4% forced duplicates to exercise stability
      keys[i] = rng() < 0.024 ? 0xDEADBEEF : (Math.floor(rng() * 0x100000000) >>> 0);
    }
    const { keys: sorted, values } = await runSort(device, keys);
    assertSortedStablePermutation(keys, sorted, values, "full-range");
  },
);

gpuTest(
  "GPU radix sort: 2K keys (single-pass scan path) sorted and stable",
  async (device) => {
    // 2048 elements -> 8 workgroups -> 128 histogram buckets <= 512:
    // exercises the single-pass Blelloch scan instead of the three-phase one.
    const n = 2048;
    const rng = mulberry32(303);
    const keys = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      keys[i] = Math.floor(rng() * 256) << 8; // duplicates, multi-digit keys
    }
    const { keys: sorted, values } = await runSort(device, keys);
    assertSortedStablePermutation(keys, sorted, values, "single-pass-scan");
  },
);

gpuTest(
  "GPU radix sort: small-N simple-sort path sorted and stable",
  async (device) => {
    const n = 500;
    const rng = mulberry32(404);
    const keys = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      keys[i] = Math.floor(rng() * 64);
    }
    const { keys: sorted, values } = await runSort(device, keys);
    assertSortedStablePermutation(keys, sorted, values, "simple-sort");
  },
);

gpuTest(
  "GPU radix sort: 200K keys above the old 131,072 cap sort correctly",
  async (device) => {
    // 200,000 elements -> 782 workgroups -> 2 phase-2 blocks (the second one
    // partial). Before the hierarchical scan this call refused to record at
    // all, so runSort's `recorded` assertion is half the test.
    const keys = randomKeys(200_000, 505);
    const { keys: sorted, values } = await runSort(device, keys);
    assertSortedStablePermutation(keys, sorted, values, "200K");
    assertMatchesCpuSort(keys, sorted, values, "200K");
  },
);

gpuTest(
  "GPU radix sort: exactly 131,072 keys (last single-dispatch phase 2)",
  async (device) => {
    // 512 workgroups exactly: the flat phase-2 path, at the element count that
    // used to be the hard ceiling. Must keep working unchanged.
    const keys = randomKeys(131_072, 606);
    const { keys: sorted, values } = await runSort(device, keys);
    assertSortedStablePermutation(keys, sorted, values, "131072");
    assertMatchesCpuSort(keys, sorted, values, "131072");
  },
);

gpuTest(
  "GPU radix sort: 131,073 keys (first hierarchical phase 2)",
  async (device) => {
    // One element past the old ceiling: 513 workgroups, so block 1 holds a
    // single workgroup sum. Catches an off-by-one in the block partition or in
    // the tail zero-padding that a round count would hide.
    const keys = randomKeys(131_073, 707);
    const { keys: sorted, values } = await runSort(device, keys);
    assertSortedStablePermutation(keys, sorted, values, "131073");
    assertMatchesCpuSort(keys, sorted, values, "131073");
  },
);

gpuTest(
  "GPU radix sort: exactly two full phase-2 blocks (262,144 keys)",
  async (device) => {
    // 512 * 256 * 2 elements -> 1024 workgroups -> exactly 2 full blocks, no
    // partial tail. The complement of the 131,073 case: here every block total
    // comes from a fully populated Blelloch pass, so a block total captured
    // from the wrong slot shows up as a misplaced run rather than as padding.
    const keys = randomKeys(262_144, 808);
    const { keys: sorted, values } = await runSort(device, keys);
    assertSortedStablePermutation(keys, sorted, values, "262144");
    assertMatchesCpuSort(keys, sorted, values, "262144");
  },
);

gpuTest(
  "GPU radix sort: 200K duplicate-heavy keys across the block hierarchy",
  async (device) => {
    // Only 64 distinct keys over 200,000 elements — roughly 3,000 equal keys
    // each, spanning many workgroups and both phase-2 blocks. This is the
    // shape Morton codes actually take at scale (a 32-bit code collides freely
    // once N passes ~2^16), and it is where a lost block offset stops looking
    // like noise: whole equal-key runs land on top of each other.
    const n = 200_000;
    const rng = mulberry32(909);
    const keys = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      keys[i] = Math.floor(rng() * 64) * 0x04101041; // bits in every radix digit
    }
    const { keys: sorted, values } = await runSort(device, keys);
    assertSortedStablePermutation(keys, sorted, values, "200K-duplicates");
    assertMatchesCpuSort(keys, sorted, values, "200K-duplicates");
  },
);
