/**
 * GPU tests for the stream-derived intensity upload cache.
 *
 * A heatmap or metaball layer bound to a value stream used to rebuild its
 * per-node intensity array — `new Float32Array(nodeCount)`, a getValue() call
 * per node, and a full buffer upload — on every frame, twice (once from the
 * ping-pong swap path and once from renderFrame). At 35K nodes that is two
 * 35K-iteration loops and 140KB of uploads per frame per layer, all of it
 * recomputing values that had not changed.
 *
 * StreamIntensityCache makes that work happen only when its inputs change.
 * These tests count the stub stream's getValue() calls to prove it.
 *
 * Requires a WebGPU adapter (Deno flag --unstable-webgpu, wired into
 * `deno task test`). When no adapter is available the tests are skipped.
 */

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import { GPU_SKIP_MESSAGE, probeAdapter } from "../helpers/gpu.ts";
import {
  type IntensityStreamSource,
  StreamIntensityCache,
} from "../../packages/core/src/layers/stream_intensity.ts";

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
 * Stream stub that records how many slots each recompute read. One recompute
 * over N slots is exactly N getValue() calls, so `reads` is the direct
 * measure of the work the cache is supposed to avoid.
 */
class CountingStream implements IntensityStreamSource {
  reads = 0;
  constructor(
    private values: Array<number | undefined>,
    private domain: [number, number] = [0, 10],
  ) {}

  getValue(nodeIndex: number): number | undefined {
    this.reads++;
    return this.values[nodeIndex];
  }

  getColorScale(): { domain: [number, number] } {
    return { domain: this.domain };
  }

  /** Mutates the underlying values; callers must also advance the version. */
  setValue(nodeIndex: number, value: number): void {
    this.values[nodeIndex] = value;
  }

  setDomain(domain: [number, number]): void {
    this.domain = domain;
  }
}

/** Reads a GPU storage buffer back as floats. */
async function readBuffer(
  device: GPUDevice,
  buffer: GPUBuffer,
  count: number,
): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, count * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}

gpuTest(
  "stream intensity cache: recomputes once, then not again until something changes",
  async (device) => {
    const nodeCount = 8;
    const stream = new CountingStream([0, 2.5, 5, 7.5, 10, undefined, 100, -100]);
    const cache = new StreamIntensityCache("test");

    try {
      // Frame N: first sync computes.
      const first = cache.sync(device, "errors", stream, 1, nodeCount);
      assertEquals(stream.reads, nodeCount, "first sync should read every slot exactly once");

      // Frames N+1..N+9: nothing changed, so nothing is recomputed and the
      // same buffer is handed back (no reallocation either).
      stream.reads = 0;
      for (let frame = 0; frame < 9; frame++) {
        assertEquals(
          cache.sync(device, "errors", stream, 1, nodeCount),
          first,
          "unchanged sync should return the same buffer",
        );
      }
      assertEquals(stream.reads, 0, "unchanged frames must not read the stream");

      // A mutation (version bump) triggers exactly one recompute.
      stream.setValue(0, 10);
      assertEquals(cache.sync(device, "errors", stream, 2, nodeCount), first);
      assertEquals(stream.reads, nodeCount, "a stream mutation should recompute exactly once");

      // ...and only one: the frames after it are free again.
      stream.reads = 0;
      cache.sync(device, "errors", stream, 2, nodeCount);
      cache.sync(device, "errors", stream, 2, nodeCount);
      assertEquals(stream.reads, 0, "frames after a mutation must not recompute");
    } finally {
      cache.destroy();
    }
  },
);

gpuTest(
  "stream intensity cache: rebinding, node count, and domain changes each force one recompute",
  async (device) => {
    const nodeCount = 8;
    const stream = new CountingStream([0, 2.5, 5, 7.5, 10, undefined, 100, -100]);
    const cache = new StreamIntensityCache("test");

    try {
      cache.sync(device, "errors", stream, 1, nodeCount);

      // (b) The layer was rebound to a different stream ID.
      stream.reads = 0;
      cache.sync(device, "latency", stream, 1, nodeCount);
      assertEquals(stream.reads, nodeCount, "rebinding to another stream must recompute");

      // (c) Node count changed (add/remove/load).
      stream.reads = 0;
      cache.sync(device, "latency", stream, 1, nodeCount - 3);
      assertEquals(stream.reads, nodeCount - 3, "a node count change must recompute");

      // (d) The colour scale's domain changed, which re-normalizes every value
      // without touching the values themselves.
      stream.reads = 0;
      stream.setDomain([0, 20]);
      cache.sync(device, "latency", stream, 1, nodeCount - 3);
      assertEquals(stream.reads, nodeCount - 3, "a domain change must recompute");
    } finally {
      cache.destroy();
    }
  },
);

gpuTest(
  "stream intensity cache: uploads domain-normalized, clamped values",
  async (device) => {
    const nodeCount = 6;
    // Domain [0, 10]: below/above clamp to 0/1, a missing value contributes 0.
    const stream = new CountingStream([0, 2.5, 10, undefined, -100, 100]);
    const cache = new StreamIntensityCache("test");

    try {
      const buffer = cache.sync(device, "errors", stream, 1, nodeCount);
      const values = await readBuffer(device, buffer, nodeCount);
      const expected = [0, 0.25, 1, 0, 0, 1];
      for (let i = 0; i < nodeCount; i++) {
        assertAlmostEquals(values[i], expected[i], 1e-6, `slot ${i}`);
      }
    } finally {
      cache.destroy();
    }
  },
);

gpuTest(
  "stream intensity cache: grows the scratch and buffer, and shrinking reuses them",
  async (device) => {
    const values = Array.from({ length: 64 }, (_, i) => i);
    const stream = new CountingStream(values, [0, 63]);
    const cache = new StreamIntensityCache("test");

    try {
      const small = cache.sync(device, "errors", stream, 1, 4);
      const grown = cache.sync(device, "errors", stream, 1, 64);
      assert(grown !== small, "growing past capacity must allocate a larger buffer");
      assertEquals(grown.size >= 64 * 4, true);

      // Shrinking keeps the (larger) buffer and the (larger) scratch array;
      // only the live prefix is uploaded, so stale tail values never leak in.
      const shrunk = cache.sync(device, "errors", stream, 1, 3);
      assertEquals(shrunk, grown, "shrinking must reuse the existing buffer");
      const readBack = await readBuffer(device, shrunk, 3);
      assertAlmostEquals(readBack[0], 0, 1e-6);
      assertAlmostEquals(readBack[1], 1 / 63, 1e-6);
      assertAlmostEquals(readBack[2], 2 / 63, 1e-6);
    } finally {
      cache.destroy();
    }
  },
);
