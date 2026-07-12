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
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import { GPU_SKIP_MESSAGE, loadModuleInliningWgsl, probeAdapter } from "../helpers/gpu.ts";
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
      const device = await adapter!.requestDevice();
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
  keys: Uint32Array,
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
