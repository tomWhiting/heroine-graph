/**
 * t-Distribution Force-Directed Placement (t-FDP) Algorithm
 *
 * Implements the full t-FDP model from "Force-directed graph layouts revisited:
 * a new force based on the t-Distribution" (Zhong, Xue, Zhang, Zhang, Ban,
 * Deussen, Wang).
 *
 * The model has three force components:
 *   1. Repulsion (all pairs): F_r = (1/alpha) * d / (1 + d^2)^gamma * dir
 *      Bounded at short range, 1/r-like at long range.
 *   2. Linear spring (edges): F_spring = alpha * d * dir
 *      Standard Hooke's law with rest length 0.
 *   3. Attractive t-force (edges): F_tforce = beta * d / (1 + d^2) * dir
 *      Short-range boost that pulls connected nodes together.
 *
 * The combination satisfies three design principles:
 *   P1: Connected nodes drawn close (spring)
 *   P2: Nodes not drawn too close (bounded repulsion)
 *   P3: Connected nodes closer than unconnected (attractive t-force)
 *
 * Constraint: alpha * (1 + beta) < 1 for proper force balance.
 * Paper defaults: alpha=0.1, beta=8, gamma=2.
 *
 * Handles its own springs (attraction = linear spring + attractive t-force).
 * Gravity delegated to shared integration shader.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";
import { calculateWorkgroups } from "../../renderer/commands.ts";
import type {
  AlgorithmBindGroups,
  AlgorithmBuffers,
  AlgorithmPipelines,
  AlgorithmRenderContext,
  ForceAlgorithm,
  ForceAlgorithmInfo,
} from "./types.ts";

// Import shader sources
import T_FDP_REPULSION_WGSL from "../shaders/t_fdp.comp.wgsl";
import T_FDP_ATTRACTION_WGSL from "../shaders/t_fdp_attraction.comp.wgsl";

/**
 * t-FDP algorithm info
 */
const T_FDP_ALGORITHM_INFO: ForceAlgorithmInfo = {
  id: "t-fdp",
  name: "t-FDP",
  description:
    "Bounded repulsion + attractive t-force via t-distribution kernel. " +
    "Preserves local neighborhoods while maintaining global structure.",
  minNodes: 0,
  maxNodes: 10000,
  complexity: "O(n²)",
};

/**
 * Extended pipeline type for t-FDP (repulsion + attraction passes)
 */
interface TFdpPipelines extends AlgorithmPipelines {
  attraction: GPUComputePipeline;
  attractionLayout: GPUBindGroupLayout;
}

/**
 * Extended bind group type for t-FDP
 */
interface TFdpBindGroups extends AlgorithmBindGroups {
  attraction: GPUBindGroup;
}

/**
 * t-FDP algorithm-specific buffers
 */
class TFdpBuffers implements AlgorithmBuffers {
  constructor(
    public repulsionUniformBuffer: GPUBuffer,
    public attractionUniformBuffer: GPUBuffer,
  ) {}

  destroy(): void {
    this.repulsionUniformBuffer.destroy();
    this.attractionUniformBuffer.destroy();
  }
}

/**
 * t-FDP force algorithm implementation
 */
export class TFdpAlgorithm implements ForceAlgorithm {
  readonly info = T_FDP_ALGORITHM_INFO;
  readonly handlesGravity = false;
  readonly handlesSprings = true;

  /** Cached edge count from last updateUniforms for dispatch sizing */
  private lastEdgeCount = 0;

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

    // Repulsion shader module (N^2 all-pairs)
    const repulsionShader = device.createShaderModule({
      label: "t-FDP Repulsion Shader",
      code: T_FDP_REPULSION_WGSL,
    });

    const repulsion = device.createComputePipeline({
      label: "t-FDP Repulsion Pipeline",
      layout: "auto",
      compute: {
        module: repulsionShader,
        entryPoint: "main",
      },
    });

    // Attraction shader module (per-edge)
    const attractionShader = device.createShaderModule({
      label: "t-FDP Attraction Shader",
      code: T_FDP_ATTRACTION_WGSL,
    });

    // Attraction pipeline: uniforms, positions, forces, edge_sources, edge_targets
    const attractionLayout = device.createBindGroupLayout({
      label: "t-FDP Attraction Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const attraction = device.createComputePipeline({
      label: "t-FDP Attraction Pipeline",
      layout: device.createPipelineLayout({ bindGroupLayouts: [attractionLayout] }),
      compute: {
        module: attractionShader,
        entryPoint: "main",
      },
    });

    const pipelines: TFdpPipelines = {
      repulsion,
      attraction,
      attractionLayout,
    };
    return pipelines;
  }

  createBuffers(device: GPUDevice, _maxNodes: number): AlgorithmBuffers {
    // Repulsion uniforms: { node_count: u32, gamma: f32, repulsion_scale: f32, _padding: u32 }
    const repulsionUniformBuffer = device.createBuffer({
      label: "t-FDP Repulsion Uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Attraction uniforms: { edge_count: u32, alpha: f32, beta: f32, _padding: u32 }
    const attractionUniformBuffer = device.createBuffer({
      label: "t-FDP Attraction Uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    return new TFdpBuffers(repulsionUniformBuffer, attractionUniformBuffer);
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    const buffers = algorithmBuffers as TFdpBuffers;
    const tfdpPipelines = pipelines as TFdpPipelines;

    // Repulsion bind group: uniforms, positions, forces
    const repulsion = device.createBindGroup({
      label: "t-FDP Repulsion Bind Group",
      layout: pipelines.repulsion.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.repulsionUniformBuffer } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
      ],
    });

    // Attraction bind group: uniforms, positions, forces, edge_sources, edge_targets
    if (!context.edgeSources || !context.edgeTargets) {
      throw new Error(
        "t-FDP requires edge source/target buffers in AlgorithmRenderContext. " +
        "Ensure graph.ts populates edgeSources and edgeTargets.",
      );
    }

    const attraction = device.createBindGroup({
      label: "t-FDP Attraction Bind Group",
      layout: tfdpPipelines.attractionLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.attractionUniformBuffer } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: context.edgeSources } },
        { binding: 4, resource: { buffer: context.edgeTargets } },
      ],
    });

    const bindGroups: TFdpBindGroups = { repulsion, attraction };
    return bindGroups;
  }

  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void {
    const buffers = algorithmBuffers as TFdpBuffers;
    const fc = context.forceConfig;

    // Cache edge count for dispatch sizing in recordRepulsionPass
    this.lastEdgeCount = context.edgeCount;

    // Repulsion uniforms: { node_count, gamma, repulsion_scale, _padding }
    // Paper: repulsion_scale = 1/alpha (default: 1/0.1 = 10.0), scaled by user multiplier
    const repulsionData = new ArrayBuffer(16);
    const repView = new DataView(repulsionData);
    repView.setUint32(0, context.nodeCount, true);
    repView.setFloat32(4, fc.tFdpGamma, true);
    repView.setFloat32(8, (1.0 / fc.tFdpAlpha) * fc.tFdpRepulsionScale, true);
    repView.setUint32(12, 0, true); // padding
    device.queue.writeBuffer(buffers.repulsionUniformBuffer, 0, repulsionData);

    // Attraction uniforms: { edge_count, alpha, beta, _padding }
    const attractionData = new ArrayBuffer(16);
    const attrView = new DataView(attractionData);
    attrView.setUint32(0, context.edgeCount, true);
    attrView.setFloat32(4, fc.tFdpAlpha, true);
    attrView.setFloat32(8, fc.tFdpBeta, true);
    attrView.setUint32(12, 0, true); // padding
    device.queue.writeBuffer(buffers.attractionUniformBuffer, 0, attractionData);
  }

  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: AlgorithmPipelines,
    bindGroups: AlgorithmBindGroups,
    nodeCount: number,
  ): void {
    const tfdpPipelines = pipelines as TFdpPipelines;
    const tfdpBindGroups = bindGroups as TFdpBindGroups;

    // Pass 1: Repulsion (N^2 over all node pairs)
    {
      const workgroups = calculateWorkgroups(nodeCount, 256);
      const pass = encoder.beginComputePass({ label: "t-FDP Repulsion" });
      pass.setPipeline(tfdpPipelines.repulsion);
      pass.setBindGroup(0, tfdpBindGroups.repulsion);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
    }

    // Pass 2: Attraction (per-edge: linear spring + attractive t-force)
    if (this.lastEdgeCount > 0) {
      const edgeWorkgroups = calculateWorkgroups(this.lastEdgeCount, 256);
      const pass = encoder.beginComputePass({ label: "t-FDP Attraction" });
      pass.setPipeline(tfdpPipelines.attraction);
      pass.setBindGroup(0, tfdpBindGroups.attraction);
      pass.dispatchWorkgroups(edgeWorkgroups);
      pass.end();
    }
  }

  destroy(): void {
    // Buffers are destroyed via AlgorithmBuffers.destroy()
    this.lastEdgeCount = 0;
  }
}

/**
 * Create t-FDP force algorithm instance
 */
export function createTFdpAlgorithm(): ForceAlgorithm {
  return new TFdpAlgorithm();
}
