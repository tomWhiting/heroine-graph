/**
 * Edge Buffer Manager
 *
 * Manages GPU buffers for graph edges in Compressed Sparse Row (CSR) format.
 * CSR format is efficient for GPU access patterns in force simulation.
 */

import { ErrorCode, HeroineGraphError } from "../../errors.ts";
import { toArrayBuffer } from "../../webgpu/buffer_utils.ts";

/**
 * CSR format edge data.
 *
 * For a graph with N nodes and E edges:
 * - offsets: N+1 elements, offsets[i] is the start of node i's edges
 * - targets: E elements, the target node IDs for each edge
 * - weights: E elements (optional), the edge weights
 */
export interface CSREdgeData {
  /** Node edge offsets (N+1 elements) */
  readonly offsets: Uint32Array;
  /** Edge target node IDs (E elements) */
  readonly targets: Uint32Array;
  /** Edge weights (E elements, optional) */
  readonly weights?: Float32Array | undefined;
}

/**
 * Configuration for edge buffers.
 */
export interface EdgeBufferConfig {
  /** Initial node capacity */
  readonly initialNodeCapacity: number;
  /** Initial edge capacity */
  readonly initialEdgeCapacity: number;
  /** Growth factor when resizing */
  readonly growthFactor: number;
  /** Label for debugging */
  readonly label?: string;
}

/**
 * Default edge buffer configuration.
 */
export const DEFAULT_EDGE_BUFFER_CONFIG: EdgeBufferConfig = {
  initialNodeCapacity: 1024,
  initialEdgeCapacity: 4096,
  growthFactor: 1.5,
  label: "EdgeBuffer",
};

/**
 * Manages edge data in CSR format on the GPU.
 */
export class EdgeBufferManager {
  private readonly device: GPUDevice;
  private readonly config: EdgeBufferConfig;

  /** Offsets buffer (N+1 elements) */
  private offsetsBuffer: GPUBuffer;
  /** Targets buffer (E elements) */
  private targetsBuffer: GPUBuffer;
  /** Weights buffer (E elements) */
  private weightsBuffer: GPUBuffer;

  /** Current node capacity */
  private nodeCapacity: number;
  /** Current edge capacity */
  private edgeCapacity: number;
  /** Actual node count */
  private nodeCount: number;
  /** Actual edge count */
  private edgeCount: number;

  constructor(device: GPUDevice, config: Partial<EdgeBufferConfig> = {}) {
    this.device = device;
    this.config = { ...DEFAULT_EDGE_BUFFER_CONFIG, ...config };
    this.nodeCapacity = this.config.initialNodeCapacity;
    this.edgeCapacity = this.config.initialEdgeCapacity;
    this.nodeCount = 0;
    this.edgeCount = 0;

    // Create initial buffers
    this.offsetsBuffer = this.createOffsetsBuffer(this.nodeCapacity + 1);
    this.targetsBuffer = this.createEdgeBuffer(this.edgeCapacity, "targets");
    this.weightsBuffer = this.createEdgeBuffer(this.edgeCapacity, "weights");
  }

  /**
   * Create an offsets buffer.
   */
  private createOffsetsBuffer(count: number): GPUBuffer {
    return this.device.createBuffer({
      size: count * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      label: `${this.config.label}_offsets`,
    });
  }

  /**
   * Create an edge data buffer.
   */
  private createEdgeBuffer(count: number, suffix: string): GPUBuffer {
    return this.device.createBuffer({
      size: count * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
      label: `${this.config.label}_${suffix}`,
    });
  }

  /**
   * Ensure buffers have sufficient capacity.
   */
  private ensureCapacity(nodeCount: number, edgeCount: number): void {
    // Check node capacity
    if (nodeCount + 1 > this.nodeCapacity) {
      let newCapacity = this.nodeCapacity;
      while (newCapacity < nodeCount + 1) {
        newCapacity = Math.ceil(newCapacity * this.config.growthFactor);
      }
      const newBuffer = this.createOffsetsBuffer(newCapacity);
      this.offsetsBuffer.destroy();
      this.offsetsBuffer = newBuffer;
      this.nodeCapacity = newCapacity;
    }

    // Check edge capacity
    if (edgeCount > this.edgeCapacity) {
      let newCapacity = this.edgeCapacity;
      while (newCapacity < edgeCount) {
        newCapacity = Math.ceil(newCapacity * this.config.growthFactor);
      }
      const newTargets = this.createEdgeBuffer(newCapacity, "targets");
      const newWeights = this.createEdgeBuffer(newCapacity, "weights");
      this.targetsBuffer.destroy();
      this.weightsBuffer.destroy();
      this.targetsBuffer = newTargets;
      this.weightsBuffer = newWeights;
      this.edgeCapacity = newCapacity;
    }
  }

  /**
   * Upload CSR edge data.
   */
  upload(data: CSREdgeData): void {
    const nodeCount = data.offsets.length - 1;
    const edgeCount = data.targets.length;

    if (edgeCount !== (data.weights?.length ?? edgeCount)) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        `Edge targets and weights must have same length: targets=${edgeCount}, weights=${data.weights?.length}`,
      );
    }

    this.ensureCapacity(nodeCount, edgeCount);
    this.nodeCount = nodeCount;
    this.edgeCount = edgeCount;

    // Upload offsets
    this.device.queue.writeBuffer(this.offsetsBuffer, 0, toArrayBuffer(data.offsets));

    // Upload targets (convert to f32 for uniform buffer access)
    const targetsF32 = new Float32Array(data.targets);
    this.device.queue.writeBuffer(this.targetsBuffer, 0, toArrayBuffer(targetsF32));

    // Upload weights (default to 1.0 if not provided)
    if (data.weights) {
      this.device.queue.writeBuffer(this.weightsBuffer, 0, toArrayBuffer(data.weights));
    } else {
      const defaultWeights = new Float32Array(edgeCount).fill(1.0);
      this.device.queue.writeBuffer(this.weightsBuffer, 0, toArrayBuffer(defaultWeights));
    }
  }

  /**
   * Upload from a combined CSR array from WASM.
   *
   * The array format is [offsets..., targets...] as returned by get_edges_csr().
   */
  uploadFromWasm(csrArray: Uint32Array, nodeCount: number): void {
    const offsetsCount = nodeCount + 1;
    const offsets = csrArray.slice(0, offsetsCount);
    const targets = csrArray.slice(offsetsCount);

    this.upload({
      offsets,
      targets,
    });
  }

  /**
   * Get the offsets buffer.
   */
  getOffsetsBuffer(): GPUBuffer {
    return this.offsetsBuffer;
  }

  /**
   * Get the targets buffer.
   */
  getTargetsBuffer(): GPUBuffer {
    return this.targetsBuffer;
  }

  /**
   * Get the weights buffer.
   */
  getWeightsBuffer(): GPUBuffer {
    return this.weightsBuffer;
  }

  /**
   * Get current counts.
   */
  getCounts(): { nodeCount: number; edgeCount: number } {
    return {
      nodeCount: this.nodeCount,
      edgeCount: this.edgeCount,
    };
  }

  /**
   * Create bind group entries for edge buffers.
   */
  createBindGroupEntries(startBinding: number): GPUBindGroupEntry[] {
    return [
      { binding: startBinding, resource: { buffer: this.offsetsBuffer } },
      { binding: startBinding + 1, resource: { buffer: this.targetsBuffer } },
      { binding: startBinding + 2, resource: { buffer: this.weightsBuffer } },
    ];
  }

  /**
   * Create bind group layout entries.
   */
  static createLayoutEntries(startBinding: number): GPUBindGroupLayoutEntry[] {
    return [
      {
        binding: startBinding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: startBinding + 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: startBinding + 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
    ];
  }

  /**
   * Destroy all buffers.
   */
  destroy(): void {
    this.offsetsBuffer.destroy();
    this.targetsBuffer.destroy();
    this.weightsBuffer.destroy();
  }
}

/**
 * Convert edge pairs array to CSR format.
 *
 * @param nodeCount Number of nodes
 * @param edges Array of [source, target] pairs
 * @param weights Optional array of edge weights
 * @returns CSR format data
 */
export function edgePairsToCSR(
  nodeCount: number,
  edges: readonly [number, number][],
  weights?: readonly number[],
): CSREdgeData {
  const edgeCount = edges.length;

  // Count edges per node
  const counts = new Uint32Array(nodeCount);
  for (const [source] of edges) {
    if (source < nodeCount) {
      counts[source]++;
    }
  }

  // Build offsets (prefix sum)
  const offsets = new Uint32Array(nodeCount + 1);
  for (let i = 0; i < nodeCount; i++) {
    offsets[i + 1] = offsets[i] + counts[i];
  }

  // Build targets (and weights if provided)
  const targets = new Uint32Array(edgeCount);
  const edgeWeights = weights ? new Float32Array(edgeCount) : undefined;
  const currentOffset = new Uint32Array(nodeCount);

  for (let i = 0; i < edges.length; i++) {
    const [source, target] = edges[i];
    if (source < nodeCount) {
      const offset = offsets[source] + currentOffset[source];
      targets[offset] = target;
      if (edgeWeights && weights) {
        edgeWeights[offset] = weights[i];
      }
      currentOffset[source]++;
    }
  }

  return {
    offsets,
    targets,
    weights: edgeWeights,
  };
}
