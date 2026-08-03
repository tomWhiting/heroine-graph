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
import {
  type AlgorithmBindGroups,
  type AlgorithmBuffers,
  type AlgorithmPipelines,
  type AlgorithmRenderContext,
  assertAlgorithmSupportedOnDevice,
  type ForceAlgorithm,
  type ForceAlgorithmInfo,
} from "./types.ts";
import {
  type AlgorithmLodFallbacks,
  createAlgorithmLodFallbacks,
  lodActiveCount,
  type LodEdgeDispatch,
  lodEdgeDispatch,
  lodEdgeSet,
  lodLiveIndices,
  lodNodeMass,
} from "./lod_bindings.ts";

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
  // The bundled attraction pass is the widest layout here and sits exactly on
  // the WebGPU default: the six storage buffers the un-cut pass binds
  // (positions, forces, edge sources, edge targets, edge weights, node flags)
  // plus the combined LOD edge set and per-node mass. The active-edge list and
  // the bundle table are one buffer with two regions precisely so this is 8
  // and not 9 — at 9 the layout is invalid on a default-limit adapter and the
  // algorithm is unselectable. Declared rather than omitted so a ninth binding
  // fails a test instead of silently costing the algorithm those adapters.
  minStorageBuffersPerShaderStage: 8,
};

/** Byte size of LinLogUniforms, shared by both LinLog shaders. */
const LINLOG_UNIFORM_BYTES = 48;

/**
 * Extended pipeline type for LinLog (repulsion + attraction passes)
 */
interface LinLogPipelines extends AlgorithmPipelines {
  attraction: GPUComputePipeline;
  attractionLayout: GPUBindGroupLayout;
  /**
   * Attraction over the LOD active-edge list plus the aggregated bundles.
   *
   * A second entry point rather than a branch in the first, so that with no cut
   * the pipeline, its bind group and its arithmetic are the ones that shipped
   * before edge aggregation existed.
   */
  attractionBundled: GPUComputePipeline;
  attractionBundledLayout: GPUBindGroupLayout;
}

/**
 * Extended bind group type for LinLog
 */
interface LinLogBindGroups extends AlgorithmBindGroups {
  attraction: GPUBindGroup;
  attractionBundled: GPUBindGroup;
}

/**
 * LinLog algorithm-specific buffers
 */
class LinLogBuffers implements AlgorithmBuffers {
  constructor(
    public uniformBuffer: GPUBuffer,
    public degreesBuffer: GPUBuffer,
    public edgeWeightsBuffer: GPUBuffer,
    /** Identity list / unit mass / all-live flags for a context supplying none */
    public fallbacks: AlgorithmLodFallbacks,
    public maxNodes: number,
    public maxEdges: number,
  ) {}

  destroy(): void {
    this.uniformBuffer.destroy();
    this.degreesBuffer.destroy();
    this.edgeWeightsBuffer.destroy();
    this.fallbacks.destroy();
  }
}

/**
 * LinLog force algorithm implementation.
 *
 * handlesSprings = true: LinLog dispatches its native logarithmic attraction
 * (F = w^delta * log(1 + d), the (0, -1) energy model). The repulsion
 * calibration (kr = linlogScaling * |repulsionStrength|) assumes that
 * attraction; an earlier substitution of the shared Hooke spring pass left
 * the 1/d repulsion unopposed and inflated every graph into a structureless
 * uniform disc. Per-node adaptive speed (prefersAdaptiveSpeed) keeps the
 * rest-length-free attraction from oscillating under fixed-step integration.
 */
export class LinLogAlgorithm implements ForceAlgorithm {
  readonly info = LINLOG_ALGORITHM_INFO;
  readonly handlesGravity = true;
  readonly handlesSprings = true;
  readonly prefersAdaptiveSpeed = true;

  /** Cached edge count from last updateUniforms for attraction dispatch sizing */
  private lastEdgeCount = 0;
  /** Entries of the active-index list the last updateUniforms sized the pass for. */
  private lastActiveCount: number | null = null;
  /** The aggregated edge set the last updateUniforms saw, or null for no cut. */
  private lastLodEdges: LodEdgeDispatch | null = null;

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;
    assertAlgorithmSupportedOnDevice(LINLOG_ALGORITHM_INFO, device);

    // Separate shader modules — different bind group layouts require separate WGSL files
    const repulsionShader = device.createShaderModule({
      label: "LinLog Repulsion + Gravity Shader",
      code: LINLOG_REPULSION_WGSL,
    });

    const attractionShader = device.createShaderModule({
      label: "LinLog Attraction Shader",
      code: LINLOG_ATTRACTION_WGSL,
    });

    // Repulsion pipeline: uniforms, positions, forces, degrees, node_flags,
    // active-index list, per-node mass
    const repulsionLayout = device.createBindGroupLayout({
      label: "LinLog Repulsion Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
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

    // Attraction pipeline: uniforms, positions, forces, edge_sources, edge_targets,
    // edge_weights, node_flags
    const attractionEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ];

    const attractionLayout = device.createBindGroupLayout({
      label: "LinLog Attraction Layout",
      entries: attractionEntries,
    });

    const attraction = device.createComputePipeline({
      label: "LinLog Attraction Pipeline",
      layout: device.createPipelineLayout({ bindGroupLayouts: [attractionLayout] }),
      compute: {
        module: attractionShader,
        entryPoint: "main",
      },
    });

    // Bundled attraction: the un-cut bindings plus the combined LOD edge set
    // and the per-node mass each arriving pull is divided by. Eight storage
    // buffers, the WebGPU default, exactly.
    const attractionBundledLayout = device.createBindGroupLayout({
      label: "LinLog Attraction Layout (LOD bundles)",
      entries: [
        ...attractionEntries,
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const attractionBundled = device.createComputePipeline({
      label: "LinLog Attraction Pipeline (LOD bundles)",
      layout: device.createPipelineLayout({ bindGroupLayouts: [attractionBundledLayout] }),
      compute: {
        module: attractionShader,
        entryPoint: "main_bundled",
      },
    });

    const pipelines: LinLogPipelines = {
      repulsion,
      attraction,
      attractionLayout,
      attractionBundled,
      attractionBundledLayout,
    };
    return pipelines;
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    // LinLogUniforms: LINLOG_UNIFORM_BYTES
    // { node_count: u32, edge_count: u32, kr: f32, kg: f32,
    //   edge_weight_influence: f32, flags: u32, active_count: u32,
    //   active_edge_count: u32, bundle_count: u32, 3 x padding }
    // One buffer, bound by both LinLog shaders, which declare it identically.
    const uniformBuffer = device.createBuffer({
      label: "LinLog Uniforms",
      size: LINLOG_UNIFORM_BYTES,
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

    return new LinLogBuffers(
      uniformBuffer,
      degreesBuffer,
      edgeWeightsBuffer,
      createAlgorithmLodFallbacks(device, maxNodes, "LinLog"),
      maxNodes,
      maxEdges,
    );
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    const buffers = algorithmBuffers as LinLogBuffers;
    const llPipelines = pipelines as LinLogPipelines;

    if (!context.nodeFlags) {
      throw new Error(
        "LinLog requires the nodeFlags buffer in AlgorithmRenderContext. " +
          "Ensure graph.ts populates nodeFlags.",
      );
    }

    // Repulsion bind group: uniforms, positions, forces, degrees, node_flags,
    // active-index list, per-node mass
    const repulsion = device.createBindGroup({
      label: "LinLog Repulsion Bind Group",
      layout: pipelines.repulsion.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.uniformBuffer } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: buffers.degreesBuffer } },
        { binding: 4, resource: { buffer: context.nodeFlags } },
        { binding: 5, resource: { buffer: lodLiveIndices(context, buffers.fallbacks) } },
        { binding: 6, resource: { buffer: lodNodeMass(context, buffers.fallbacks) } },
      ],
    });

    // Attraction bind group: uniforms, positions, forces, edge_sources, edge_targets,
    // edge_weights, node_flags
    if (!context.edgeSources || !context.edgeTargets) {
      throw new Error(
        "LinLog requires edge source/target buffers in AlgorithmRenderContext. " +
          "Ensure graph.ts populates edgeSources and edgeTargets.",
      );
    }

    const attractionEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: buffers.uniformBuffer } },
      { binding: 1, resource: { buffer: context.positions } },
      { binding: 2, resource: { buffer: context.forces } },
      { binding: 3, resource: { buffer: context.edgeSources } },
      { binding: 4, resource: { buffer: context.edgeTargets } },
      { binding: 5, resource: { buffer: buffers.edgeWeightsBuffer } },
      { binding: 6, resource: { buffer: context.nodeFlags } },
    ];

    const attraction = device.createBindGroup({
      label: "LinLog Attraction Bind Group",
      layout: llPipelines.attractionLayout,
      entries: attractionEntries,
    });

    // Built unconditionally, dispatched only under a cut: whether an
    // aggregation is live is per-frame state, and createBindGroups is
    // contractually forbidden from capturing any of that.
    const attractionBundled = device.createBindGroup({
      label: "LinLog Attraction Bind Group (LOD bundles)",
      layout: llPipelines.attractionBundledLayout,
      entries: [
        ...attractionEntries,
        { binding: 7, resource: { buffer: lodEdgeSet(context, buffers.fallbacks) } },
        { binding: 8, resource: { buffer: lodNodeMass(context, buffers.fallbacks) } },
      ],
    });

    const bindGroups: LinLogBindGroups = { repulsion, attraction, attractionBundled };
    return bindGroups;
  }

  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void {
    const buffers = algorithmBuffers as LinLogBuffers;

    // Cache edge count for attraction dispatch sizing in recordRepulsionPass
    this.lastEdgeCount = context.edgeCount;

    if (context.nodeCount > buffers.maxNodes) {
      throw new Error(
        `LinLog buffer overflow: nodeCount (${context.nodeCount}) exceeds buffer capacity (${buffers.maxNodes}). ` +
          `Buffers must be recreated with createBuffers() when node count increases.`,
      );
    }

    // Validate edge count doesn't exceed buffer capacity
    if (context.edgeCount > buffers.maxEdges) {
      throw new Error(
        `LinLog buffer overflow: edgeCount (${context.edgeCount}) exceeds buffer capacity (${buffers.maxEdges}). ` +
          `Buffers must be recreated with createBuffers() when edge count increases.`,
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

    // The repulsion dispatch and the shader's two loop bounds must all come
    // from this one number, or threads read past the end of the list.
    this.lastActiveCount = lodActiveCount(context);
    this.lastLodEdges = lodEdgeDispatch(context);

    // Write uniform buffer (LINLOG_UNIFORM_BYTES). The three LOD counts are
    // zero with no cut, which is what keeps `main` reading nothing new.
    const data = new ArrayBuffer(LINLOG_UNIFORM_BYTES);
    const view = new DataView(data);
    view.setUint32(0, context.nodeCount, true);
    view.setUint32(4, context.edgeCount, true);
    view.setFloat32(8, kr, true);
    view.setFloat32(12, kg, true);
    view.setFloat32(16, edgeWeightInfluence, true);
    view.setUint32(20, flags, true);
    view.setUint32(24, this.lastActiveCount, true);
    view.setUint32(28, this.lastLodEdges?.activeEdgeCount ?? 0, true);
    view.setUint32(32, this.lastLodEdges?.bundleCount ?? 0, true);
    view.setUint32(36, 0, true); // padding
    view.setUint32(40, 0, true); // padding
    view.setUint32(44, 0, true); // padding

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
    if (context.edgeCount > 0) {
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

    // Pass 1: Repulsion + Gravity (combined in shader, N² over the active set)
    {
      const workgroups = calculateWorkgroups(this.lastActiveCount ?? nodeCount, 256);
      const pass = encoder.beginComputePass({ label: "LinLog Repulsion + Gravity" });
      pass.setPipeline(llPipelines.repulsion);
      pass.setBindGroup(0, llBindGroups.repulsion);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
    }

    // Pass 2: Native LinLog logarithmic attraction (per-edge, F = w^delta * log(1 + d)).
    // Balances the FA2-calibrated 1/d repulsion — see class doc.
    //
    // Under a live LOD cut it runs over the aggregated edge set instead: the
    // visible source edges plus the bundles standing in for everything that
    // crosses a collapse boundary. Every source edge is covered by exactly one
    // of the two, so an edge is never pulled twice and a collapsed subtree's
    // cross-cutting attraction is transferred rather than dropped. Zero
    // threads is a legal aggregation — every edge inside a collapsed subtree —
    // and must dispatch nothing rather than fall back to the whole edge array.
    const lodEdges = this.lastLodEdges;
    if (lodEdges !== null) {
      if (lodEdges.total > 0) {
        const pass = encoder.beginComputePass({ label: "LinLog Attraction (LOD bundles)" });
        pass.setPipeline(llPipelines.attractionBundled);
        pass.setBindGroup(0, llBindGroups.attractionBundled);
        pass.dispatchWorkgroups(calculateWorkgroups(lodEdges.total, 256));
        pass.end();
      }
    } else if (this.lastEdgeCount > 0) {
      const edgeWorkgroups = calculateWorkgroups(this.lastEdgeCount, 256);
      const pass = encoder.beginComputePass({ label: "LinLog Attraction" });
      pass.setPipeline(llPipelines.attraction);
      pass.setBindGroup(0, llBindGroups.attraction);
      pass.dispatchWorkgroups(edgeWorkgroups);
      pass.end();
    }
  }

  destroy(): void {
    // Buffers are destroyed via AlgorithmBuffers.destroy()
    this.lastEdgeCount = 0;
    this.lastActiveCount = null;
    this.lastLodEdges = null;
  }
}

/**
 * Create LinLog force algorithm instance
 */
export function createLinLogAlgorithm(): ForceAlgorithm {
  return new LinLogAlgorithm();
}
