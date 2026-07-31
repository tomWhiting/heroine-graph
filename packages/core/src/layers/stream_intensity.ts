/**
 * Per-Node Stream Intensity Upload Cache
 *
 * Heatmap and metaball layers can be driven by a value stream instead of raw
 * node density: each node contributes an intensity equal to its stream value
 * normalized into the stream's colour-scale domain.
 *
 * Deriving that array is O(nodeCount) on the CPU plus a full buffer upload, so
 * it must not run per frame. The derived array is a pure function of
 *
 *   (stream identity, stream mutation version, node count, colour-scale domain)
 *
 * and this module recomputes only when one of those changes. The scratch array
 * and the GPU buffer are reused across recomputes (grow-only).
 *
 * @module
 */

/**
 * The read surface of a value stream this module needs. Structural on purpose:
 * it keeps the cache testable without a StreamManager and makes the exact
 * inputs to the derived array explicit.
 */
export interface IntensityStreamSource {
  /** Value for a node slot, or undefined when the slot has no value */
  getValue(nodeIndex: number): number | undefined;
  /** Colour scale whose `domain` defines the 0..1 normalization range */
  getColorScale(): { domain: [number, number] };
}

/** Inputs the derived intensity array depends on. */
interface IntensityKey {
  streamId: string;
  /** StreamManager mutation counter (see StreamManager.version) */
  version: number;
  nodeCount: number;
  domainMin: number;
  domainRange: number;
}

function keysEqual(a: IntensityKey, b: IntensityKey): boolean {
  return a.streamId === b.streamId &&
    a.version === b.version &&
    a.nodeCount === b.nodeCount &&
    a.domainMin === b.domainMin &&
    a.domainRange === b.domainRange;
}

/**
 * Caches one layer's stream-derived intensity buffer.
 *
 * One instance per layer that can be bound to a stream. `sync` returns the GPU
 * buffer to bind, recomputing and re-uploading only when the inputs changed.
 */
export class StreamIntensityCache {
  /** Grow-only scratch for the normalized values; only [0, nodeCount) is live. */
  private scratch = new Float32Array(0);
  private buffer: GPUBuffer | null = null;
  private key: IntensityKey | null = null;

  constructor(private readonly label: string) {}

  /**
   * Returns the GPU buffer holding one f32 intensity per node slot, computing
   * and uploading it only when the stream, its mutation version, the node
   * count, or the colour-scale domain changed since the last call.
   *
   * @param device - GPU device
   * @param streamId - Identifier of the bound stream (part of the cache key)
   * @param stream - The bound stream
   * @param version - StreamManager mutation counter at call time
   * @param nodeCount - Number of node slots to fill
   */
  sync(
    device: GPUDevice,
    streamId: string,
    stream: IntensityStreamSource,
    version: number,
    nodeCount: number,
  ): GPUBuffer {
    const domain = stream.getColorScale().domain;
    const domainMin = domain[0];
    const domainRange = domain[1] - domainMin;
    const key: IntensityKey = { streamId, version, nodeCount, domainMin, domainRange };

    if (this.buffer && this.key && keysEqual(this.key, key)) {
      return this.buffer;
    }

    if (this.scratch.length < nodeCount) {
      this.scratch = new Float32Array(nodeCount);
    }
    const intensities = this.scratch;
    for (let i = 0; i < nodeCount; i++) {
      const value = stream.getValue(i);
      // A zero-width domain cannot normalize; treat every node as
      // non-contributing rather than dividing by zero.
      intensities[i] = value !== undefined && domainRange > 0
        ? Math.max(0, Math.min(1, (value - domainMin) / domainRange))
        : 0;
    }

    // WebGPU rejects zero-sized buffers; an empty graph still needs a bindable
    // buffer for the layer's bind group. COPY_SRC is not needed by the layers
    // themselves — it exists so the contents can be read back and asserted on
    // (tests/gpu/stream_intensity_test.ts).
    const requiredSize = Math.max(nodeCount * 4, 4);
    if (!this.buffer || this.buffer.size < requiredSize) {
      this.buffer?.destroy();
      this.buffer = device.createBuffer({
        label: this.label,
        size: requiredSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    }

    // Upload only the live prefix: the scratch array may be longer than
    // nodeCount after the graph shrank. dataOffset/size are element counts
    // because the source is a TypedArray. A zero-length write is skipped —
    // Deno's WebGPU backend panics on it, and the freshly created buffer is
    // already zero-filled.
    if (nodeCount > 0) {
      device.queue.writeBuffer(this.buffer, 0, intensities, 0, nodeCount);
    }

    this.key = key;
    return this.buffer;
  }

  /** Releases the GPU buffer and forces a recompute on the next sync. */
  destroy(): void {
    this.buffer?.destroy();
    this.buffer = null;
    this.key = null;
    this.scratch = new Float32Array(0);
  }
}
