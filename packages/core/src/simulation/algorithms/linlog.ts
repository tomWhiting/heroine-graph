/**
 * LinLog Force Algorithm
 *
 * Implements the LinLog energy model (Noack 2009) for optimal cluster separation.
 * Based on the ForceAtlas2 paper (Jacomy et al. 2014).
 *
 * Key differences from standard force-directed:
 * - Logarithmic attraction: F ~ log(1 + d) instead of Hooke's law F ~ d
 * - Degree-weighted repulsion (same as FA2)
 * - Degree-weighted gravity
 *
 * The LinLog energy model (attraction=0, repulsion=-1) produces layouts where
 * node distances reflect community density rather than path length, giving the
 * best cluster separation of any known force-directed energy model.
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

// Import shader sources (separate files due to different bind group layouts)
import LINLOG_REPULSION_WGSL from "../shaders/linlog.comp.wgsl";
import LINLOG_ATTRACTION_WGSL from "../shaders/linlog_attraction.comp.wgsl";

/**
 * LinLog algorithm info
 */
const LINLOG_ALGORITHM_INFO: ForceAlgorithmInfo = {
  id: "linlog",
  name: "LinLog",
  description:
    "Logarithmic attraction with degree-weighted repulsion. Optimal for community structure visualization.",
  minNodes: 0,
  maxNodes: 50000,
  complexity: "O(n²)",
};

/**
 * Extended pipeline type for LinLog (repulsion + attraction passes)
 */
interface LinLogPipelines extends AlgorithmPipelines {
  attraction: GPUComputePipeline;
  attractionLayout: GPUBindGroupLayout;
}

/**
 * Extended bind group type for LinLog
 */
interface LinLogBindGroups extends AlgorithmBindGroups {
  attraction: GPUBindGroup;
}

/**
 * LinLog algorithm-specific buffers
 */
class LinLogBuffers implements AlgorithmBuffers {
  constructor(
    public uniformBuffer: GPUBuffer,
    public degreesBuffer: GPUBuffer,
    public edgeWeightsBuffer: GPUBuffer,
    public maxNodes: number,
    public maxEdges: number,
  ) {}

  destroy(): void {
    this.uniformBuffer.destroy();
    this.degreesBuffer.destroy();
    this.edgeWeightsBuffer.destroy();
  }
}

/**
 * LinLog force algorithm implementation.
 *
 * handlesSprings = false: LinLog now delegates to standard Hooke's law springs
 * for attraction. The original LinLog logarithmic attraction created patterns
 * inconsistent with other algorithms. Standard springs produce better layouts.
 * The LinLog attraction shader is preserved but not dispatched.
 */
export class LinLogAlgorithm implements ForceAlgorithm {
  readonly info = LINLOG_ALGORITHM_INFO;
  readonly handlesGravity = true;
  readonly handlesSprings = false;

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

    // Separate shader modules — different bind group layouts require separate WGSL files
    const repulsionShader = device.createShaderModule({
      label: "LinLog Repulsion + Gravity Shader",
      code: LINLOG_REPULSION_WGSL,
    });

    const attractionShader = device.createShaderModule({
      label: "LinLog Attraction Shader",
      code: LINLOG_ATTRACTION_WGSL,
    });

    // Repulsion pipeline: uniforms, positions, forces, degrees
    const repulsionLayout = device.createBindGroupLayout({
      label: "LinLog Repulsion Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const repulsion = device.createComputePipeline({
      label: "LinLog Repulsion Pipeline",
      layout: device.createPipelineLayout({ bindGroupLayouts: [repulsionLayout] }),
      compute: {
        module: repulsionShader,
        entryPoint: "main",
      },
    });

    // Attraction pipeline: uniforms, positions, forces, edge_sources, edge_targets, edge_weights
    const attractionLayout = device.createBindGroupLayout({
      label: "LinLog Attraction Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const attraction = device.createComputePipeline({
      label: "LinLog Attraction Pipeline",
      layout: device.createPipelineLayout({ bindGroupLayouts: [attractionLayout] }),
      compute: {
        module: attractionShader,
        entryPoint: "main",
      },
    });

    const pipelines: LinLogPipelines = {
      repulsion,
      attraction,
      attractionLayout,
    };
    return pipelines;
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    // LinLogUniforms: 32 bytes
    // { node_count: u32, edge_count: u32, kr: f32, kg: f32,
    //   edge_weight_influence: f32, flags: u32, _padding: vec2<u32> }
    const uniformBuffer = device.createBuffer({
      label: "LinLog Uniforms",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Degrees buffer: total degree per node
    const degreesBuffer = device.createBuffer({
      label: "LinLog Degrees",
      size: Math.max(maxNodes * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Edge weights: f32 per edge (allocate for max edges = maxNodes * 4 as estimate)
    const maxEdges = maxNodes * 4;
    const edgeWeightsBuffer = device.createBuffer({
      label: "LinLog Edge Weights",
      size: Math.max(maxEdges * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new LinLogBuffers(uniformBuffer, degreesBuffer, edgeWeightsBuffer, maxNodes, maxEdges);
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    const buffers = algorithmBuffers as LinLogBuffers;
    const llPipelines = pipelines as LinLogPipelines;

    // Repulsion bind group: uniforms, positions, forces, degrees
    const repulsion = device.createBindGroup({
      label: "LinLog Repulsion Bind Group",
      layout: pipelines.repulsion.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.uniformBuffer } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: buffers.degreesBuffer } },
      ],
    });

    // Attraction bind group: uniforms, positions, forces, edge_sources, edge_targets, edge_weights
    if (!context.edgeSources || !context.edgeTargets) {
      throw new Error(
        "LinLog requires edge source/target buffers in AlgorithmRenderContext. " +
        "Ensure graph.ts populates edgeSources and edgeTargets.",
      );
    }

    const attraction = device.createBindGroup({
      label: "LinLog Attraction Bind Group",
      layout: llPipelines.attractionLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.uniformBuffer } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: context.edgeSources } },
        { binding: 4, resource: { buffer: context.edgeTargets } },
        { binding: 5, resource: { buffer: buffers.edgeWeightsBuffer } },
      ],
    });

    const bindGroups: LinLogBindGroups = { repulsion, attraction };
    return bindGroups;
  }

  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void {
    const buffers = algorithmBuffers as LinLogBuffers;

    if (context.nodeCount > buffers.maxNodes) {
      throw new Error(
        `LinLog buffer overflow: nodeCount (${context.nodeCount}) exceeds buffer capacity (${buffers.maxNodes}). ` +
        `Buffers must be recreated with createBuffers() when node count increases.`,
      );
    }

    const fc = context.forceConfig;

    // LinLog force model calibration:
    // linlogScaling (default 0.1) maps user-facing repulsionStrength to LinLog's
    // logarithmic repulsion model. The 10x gravity boost matches FA2's calibration —
    // LinLog's constant-magnitude gravity needs amplification to counterbalance
    // the degree-weighted repulsion.
    const kr = fc.linlogScaling * Math.abs(fc.repulsionStrength);
    const kg = fc.linlogGravity * fc.centerStrength * 10;
    const edgeWeightInfluence = fc.linlogEdgeWeightInfluence;
    const flags = fc.linlogStrongGravity ? 1 : 0;

    // Write uniform buffer (32 bytes)
    const data = new ArrayBuffer(32);
    const view = new DataView(data);
    view.setUint32(0, context.nodeCount, true);
    view.setUint32(4, context.edgeCount, true);
    view.setFloat32(8, kr, true);
    view.setFloat32(12, kg, true);
    view.setFloat32(16, edgeWeightInfluence, true);
    view.setUint32(20, flags, true);
    view.setUint32(24, 0, true); // padding
    view.setUint32(28, 0, true); // padding

    device.queue.writeBuffer(buffers.uniformBuffer, 0, data);

    // Compute actual node degrees from CPU-side edge arrays.
    // Degree-weighted repulsion is the core of FA2/LinLog — using real degrees
    // ensures hubs repel proportionally to their connectivity, which is essential
    // for proper cluster separation in the (0, -1) energy model.
    const degrees = new Uint32Array(context.nodeCount);
    if (context.edgeSourcesData && context.edgeTargetsData) {
      const edgeCount = Math.min(
        context.edgeCount,
        context.edgeSourcesData.length,
        context.edgeTargetsData.length,
      );
      for (let i = 0; i < edgeCount; i++) {
        const src = context.edgeSourcesData[i];
        const tgt = context.edgeTargetsData[i];
        if (src < context.nodeCount) {
          degrees[src]++;
        }
        if (tgt < context.nodeCount) {
          degrees[tgt]++;
        }
      }
    }
    // degrees[i] is now 0 for isolated nodes, which is correct:
    // mass = deg + 1 in the shader ensures even isolated nodes have mass 1.
    device.queue.writeBuffer(buffers.degreesBuffer, 0, degrees);

    // Upload edge weights (all 1.0 for unweighted graphs)
    if (context.edgeCount > 0 && context.edgeCount <= buffers.maxEdges) {
      const weights = new Float32Array(context.edgeCount);
      weights.fill(1.0);
      device.queue.writeBuffer(buffers.edgeWeightsBuffer, 0, weights);
    }
  }

  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: AlgorithmPipelines,
    bindGroups: AlgorithmBindGroups,
    nodeCount: number,
  ): void {
    const llPipelines = pipelines as LinLogPipelines;
    const llBindGroups = bindGroups as LinLogBindGroups;

    // Pass 1: Repulsion + Gravity (combined in shader, N² over nodes)
    {
      const workgroups = calculateWorkgroups(nodeCount, 256);
      const pass = encoder.beginComputePass({ label: "LinLog Repulsion + Gravity" });
      pass.setPipeline(llPipelines.repulsion);
      pass.setBindGroup(0, llBindGroups.repulsion);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
    }

    // LinLog attraction pass disabled — handlesSprings=false delegates to standard
    // Hooke's law springs for consistent behavior across algorithms.
    // The attraction shader and pipeline are preserved for potential future use.
  }
}

/**
 * Create LinLog force algorithm instance
 */
export function createLinLogAlgorithm(): ForceAlgorithm {
  return new LinLogAlgorithm();
}
