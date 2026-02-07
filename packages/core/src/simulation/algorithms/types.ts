/**
 * Force Algorithm Type Definitions
 *
 * Defines the interface for pluggable force algorithms.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";
import type { FullForceConfig } from "../config.ts";

/**
 * Available force algorithm types
 */
export type ForceAlgorithmType = "n2" | "barnes-hut" | "force-atlas2" | "density" | "relativity-atlas" | "tidy-tree" | "linlog" | "t-fdp" | "community" | "codebase";

/**
 * Algorithm metadata for display
 */
export interface ForceAlgorithmInfo {
  /** Unique identifier */
  readonly id: ForceAlgorithmType;
  /** Display name */
  readonly name: string;
  /** Description of the algorithm */
  readonly description: string;
  /** Minimum recommended node count */
  readonly minNodes: number;
  /** Maximum recommended node count (-1 for unlimited) */
  readonly maxNodes: number;
  /** Time complexity description */
  readonly complexity: string;
}

/**
 * Buffers specific to an algorithm (beyond the standard simulation buffers)
 */
export interface AlgorithmBuffers {
  /** Dispose of all algorithm-specific buffers */
  destroy(): void;
}

/**
 * Empty algorithm buffers for algorithms that don't need extra buffers
 */
export class EmptyAlgorithmBuffers implements AlgorithmBuffers {
  destroy(): void {
    // No buffers to destroy
  }
}

/**
 * Bind groups for algorithm-specific compute passes
 */
export interface AlgorithmBindGroups {
  /** Repulsion pass bind group */
  repulsion: GPUBindGroup;
  /** Additional named bind groups for multi-phase algorithms */
  [key: string]: GPUBindGroup;
}

/**
 * Pipelines for algorithm-specific compute passes
 */
export interface AlgorithmPipelines {
  /** Repulsion compute pipeline */
  repulsion: GPUComputePipeline;
  /** Additional named pipelines for multi-phase algorithms */
  [key: string]: GPUComputePipeline;
}

/**
 * Context passed to algorithm for rendering
 *
 * All position and force buffers use vec2<f32> layout (8 bytes per node).
 */
export interface AlgorithmRenderContext {
  /** GPU device */
  device: GPUDevice;
  /** Position buffer - vec2<f32> per node (interleaved X,Y) */
  positions: GPUBuffer;
  /** Force accumulator buffer - vec2<f32> per node (interleaved X,Y) */
  forces: GPUBuffer;
  /** Number of nodes */
  nodeCount: number;
  /** Number of edges (required for Relativity Atlas CSR traversal) */
  edgeCount: number;
  /** Current force configuration */
  forceConfig: FullForceConfig;
  /** Viewport bounds (for spatial algorithms like Barnes-Hut and Density Field) */
  bounds?: {
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
  } | undefined;
  /** Edge source indices buffer (for algorithms that handle their own springs) */
  edgeSources?: GPUBuffer;
  /** Edge target indices buffer (for algorithms that handle their own springs) */
  edgeTargets?: GPUBuffer;
  /** CPU-side edge source indices (for degree computation without GPU readback) */
  edgeSourcesData?: Uint32Array | undefined;
  /** CPU-side edge target indices (for degree computation without GPU readback) */
  edgeTargetsData?: Uint32Array | undefined;
}

/**
 * Force algorithm interface
 *
 * Algorithms implement this interface to provide custom repulsion force calculations.
 * The standard spring forces and integration are handled by the main pipeline.
 */
export interface ForceAlgorithm {
  /** Algorithm info */
  readonly info: ForceAlgorithmInfo;

  /**
   * Whether this algorithm applies its own center gravity.
   * When true, the integration shader skips its built-in gravity
   * to avoid double-applying the center pull.
   */
  readonly handlesGravity: boolean;

  /**
   * Whether this algorithm provides its own edge/spring forces.
   * When true, the simulation skips the default edge spring pass
   * to avoid competing forces. Used by layout algorithms (e.g. tidy-tree)
   * that compute exact target positions rather than using force-directed springs.
   */
  readonly handlesSprings?: boolean;

  /**
   * Create GPU pipelines for this algorithm
   *
   * @param context - GPU context
   * @returns Algorithm-specific pipelines
   */
  createPipelines(context: GPUContext): AlgorithmPipelines;

  /**
   * Create algorithm-specific buffers
   *
   * @param device - GPU device
   * @param maxNodes - Maximum number of nodes to support
   * @returns Algorithm-specific buffers
   */
  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers;

  /**
   * Create bind groups for the repulsion pass
   *
   * @param device - GPU device
   * @param pipelines - Algorithm pipelines
   * @param context - Render context with buffers
   * @param algorithmBuffers - Algorithm-specific buffers
   * @returns Bind groups
   */
  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups;

  /**
   * Update algorithm-specific uniform buffers
   *
   * @param device - GPU device
   * @param algorithmBuffers - Algorithm-specific buffers
   * @param context - Render context
   */
  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void;

  /**
   * Record the repulsion compute pass
   *
   * @param encoder - Command encoder
   * @param pipelines - Algorithm pipelines
   * @param bindGroups - Algorithm bind groups
   * @param nodeCount - Number of nodes
   */
  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: AlgorithmPipelines,
    bindGroups: AlgorithmBindGroups,
    nodeCount: number,
  ): void;
}
