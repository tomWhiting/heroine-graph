/**
 * Uniform Buffer Management
 *
 * Manages GPU uniform buffers for simulation parameters,
 * viewport state, and other per-frame constants.
 */

import type { ForceConfig, ViewportState } from "../../types.ts";
import { toArrayBuffer } from "../../webgpu/buffer_utils.ts";

/**
 * Simulation uniform buffer layout.
 *
 * Must match the WGSL struct layout exactly.
 * All values are f32 for simplicity (no padding issues).
 */
export interface SimulationUniforms {
  /** Many-body repulsion strength */
  repulsion: number;
  /** Link spring strength */
  attraction: number;
  /** Center gravity strength */
  gravity: number;
  /** Gravity center X */
  centerX: number;
  /** Gravity center Y */
  centerY: number;
  /** Ideal link distance */
  linkDistance: number;
  /** Barnes-Hut theta parameter */
  theta: number;
  /** Current simulation alpha */
  alpha: number;
  /** Velocity decay factor */
  velocityDecay: number;
  /** Number of nodes */
  nodeCount: number;
  /** Number of edges */
  edgeCount: number;
  /** Delta time for integration */
  dt: number;
  /** Index signature for generic compatibility */
  [key: string]: number;
}

/**
 * Default simulation uniforms.
 */
export const DEFAULT_SIMULATION_UNIFORMS: SimulationUniforms = {
  repulsion: -30.0,
  attraction: 1.0,
  gravity: 0.1,
  centerX: 0.0,
  centerY: 0.0,
  linkDistance: 30.0,
  theta: 0.9,
  alpha: 1.0,
  velocityDecay: 0.4,
  nodeCount: 0,
  edgeCount: 0,
  dt: 1.0,
};

/**
 * Size of simulation uniforms in bytes (12 x f32 = 48 bytes).
 * Aligned to 16 bytes as required by WebGPU.
 */
export const SIMULATION_UNIFORMS_SIZE = 48;

/**
 * Viewport uniform buffer layout.
 */
export interface ViewportUniforms {
  /** Pan offset X in graph units */
  panX: number;
  /** Pan offset Y in graph units */
  panY: number;
  /** Zoom scale factor */
  scale: number;
  /** Canvas width in pixels */
  width: number;
  /** Canvas height in pixels */
  height: number;
  /** Device pixel ratio */
  dpr: number;
  /** Inverse scale (1/scale) for optimization */
  invScale: number;
  /** Padding for 16-byte alignment */
  _padding: number;
  /** Index signature for generic compatibility */
  [key: string]: number;
}

/**
 * Size of viewport uniforms in bytes (8 x f32 = 32 bytes).
 */
export const VIEWPORT_UNIFORMS_SIZE = 32;

/**
 * Manages a GPU uniform buffer.
 */
export class UniformBuffer<T extends Record<string, number>> {
  private readonly device: GPUDevice;
  private readonly buffer: GPUBuffer;
  private readonly data: Float32Array;
  private readonly fieldOrder: (keyof T)[];
  private dirty: boolean;

  constructor(
    device: GPUDevice,
    size: number,
    fieldOrder: (keyof T)[],
    label: string
  ) {
    this.device = device;
    this.fieldOrder = fieldOrder;
    this.data = new Float32Array(size / Float32Array.BYTES_PER_ELEMENT);
    this.dirty = false;

    this.buffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label,
    });
  }

  /**
   * Update uniforms from an object.
   */
  update(values: Partial<T>): void {
    for (const [key, value] of Object.entries(values)) {
      const index = this.fieldOrder.indexOf(key as keyof T);
      if (index !== -1 && typeof value === "number") {
        this.data[index] = value;
        this.dirty = true;
      }
    }
  }

  /**
   * Upload data to GPU if changed.
   */
  upload(): void {
    if (this.dirty) {
      this.device.queue.writeBuffer(this.buffer, 0, toArrayBuffer(this.data));
      this.dirty = false;
    }
  }

  /**
   * Force upload data to GPU.
   */
  forceUpload(): void {
    this.device.queue.writeBuffer(this.buffer, 0, toArrayBuffer(this.data));
    this.dirty = false;
  }

  /**
   * Get the GPU buffer.
   */
  getBuffer(): GPUBuffer {
    return this.buffer;
  }

  /**
   * Create a bind group entry.
   */
  createBindGroupEntry(binding: number): GPUBindGroupEntry {
    return {
      binding,
      resource: { buffer: this.buffer },
    };
  }

  /**
   * Destroy the buffer.
   */
  destroy(): void {
    this.buffer.destroy();
  }
}

/**
 * Simulation uniforms field order (must match WGSL struct).
 */
const SIMULATION_FIELD_ORDER: (keyof SimulationUniforms)[] = [
  "repulsion",
  "attraction",
  "gravity",
  "centerX",
  "centerY",
  "linkDistance",
  "theta",
  "alpha",
  "velocityDecay",
  "nodeCount",
  "edgeCount",
  "dt",
];

/**
 * Create a simulation uniform buffer.
 */
export function createSimulationUniformBuffer(
  device: GPUDevice
): UniformBuffer<SimulationUniforms> {
  const buffer = new UniformBuffer<SimulationUniforms>(
    device,
    SIMULATION_UNIFORMS_SIZE,
    SIMULATION_FIELD_ORDER,
    "SimulationUniforms"
  );
  buffer.update(DEFAULT_SIMULATION_UNIFORMS);
  buffer.forceUpload();
  return buffer;
}

/**
 * Viewport uniforms field order.
 */
const VIEWPORT_FIELD_ORDER: (keyof ViewportUniforms)[] = [
  "panX",
  "panY",
  "scale",
  "width",
  "height",
  "dpr",
  "invScale",
  "_padding",
];

/**
 * Create a viewport uniform buffer.
 */
export function createViewportUniformBuffer(
  device: GPUDevice
): UniformBuffer<ViewportUniforms> {
  const buffer = new UniformBuffer<ViewportUniforms>(
    device,
    VIEWPORT_UNIFORMS_SIZE,
    VIEWPORT_FIELD_ORDER,
    "ViewportUniforms"
  );
  return buffer;
}

/**
 * Convert ForceConfig to SimulationUniforms.
 */
export function forceConfigToUniforms(
  config: ForceConfig,
  alpha: number,
  velocityDecay: number,
  nodeCount: number,
  edgeCount: number,
  dt: number = 1.0
): Partial<SimulationUniforms> {
  return {
    repulsion: config.repulsion,
    attraction: config.attraction,
    gravity: config.gravity,
    centerX: config.centerX,
    centerY: config.centerY,
    linkDistance: config.linkDistance,
    theta: config.theta,
    alpha,
    velocityDecay,
    nodeCount,
    edgeCount,
    dt,
  };
}

/**
 * Convert ViewportState to ViewportUniforms.
 */
export function viewportStateToUniforms(
  state: ViewportState,
  dpr: number = 1.0
): Partial<ViewportUniforms> {
  return {
    panX: state.x,
    panY: state.y,
    scale: state.scale,
    width: state.width,
    height: state.height,
    dpr,
    invScale: 1.0 / state.scale,
    _padding: 0,
  };
}
