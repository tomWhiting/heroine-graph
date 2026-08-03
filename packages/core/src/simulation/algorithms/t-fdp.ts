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
 * All kernels operate on the normalized distance d = dist / distScale
 * (distScale = 2 x springLength), mapping world units onto the paper's
 * unit scale where the model equilibrates at d = O(1).
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
import {
  type AlgorithmLodFallbacks,
  createAlgorithmLodFallbacks,
  lodActiveCount,
  type LodEdgeDispatch,
  lodEdgeDispatch,
  lodEdgeSet,
  lodLiveIndices,
  lodNodeFlags,
  lodNodeMass,
} from "./lod_bindings.ts";

// Import shader sources
import T_FDP_REPULSION_WGSL from "../shaders/t_fdp.comp.wgsl";
import T_FDP_ATTRACTION_WGSL from "../shaders/t_fdp_attraction.comp.wgsl";

/**
 * t-FDP algorithm info
 */
const T_FDP_ALGORITHM_INFO: ForceAlgorithmInfo = {
  id: "t-fdp",
  name: "t-FDP",
  description: "Bounded repulsion + attractive t-force via t-distribution kernel. " +
    "Preserves local neighborhoods while maintaining global structure.",
  minNodes: 0,
  maxNodes: 10000,
  complexity: "O(n²)",
};

/** Byte size of TFdpUniforms in t_fdp.comp.wgsl. */
const T_FDP_UNIFORM_BYTES = 32;

/** Byte size of TFdpAttractionUniforms in t_fdp_attraction.comp.wgsl. */
const T_FDP_ATTRACTION_UNIFORM_BYTES = 32;

/**
 * Extended pipeline type for t-FDP (repulsion + attraction passes)
 */
interface TFdpPipelines extends AlgorithmPipelines {
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
 * Extended bind group type for t-FDP
 */
interface TFdpBindGroups extends AlgorithmBindGroups {
  attraction: GPUBindGroup;
  attractionBundled: GPUBindGroup;
}

/**
 * t-FDP algorithm-specific buffers
 */
class TFdpBuffers implements AlgorithmBuffers {
  constructor(
    public repulsionUniformBuffer: GPUBuffer,
    public attractionUniformBuffer: GPUBuffer,
    /** Identity list / unit mass / all-live flags for a context supplying none */
    public fallbacks: AlgorithmLodFallbacks,
  ) {}

  destroy(): void {
    this.repulsionUniformBuffer.destroy();
    this.attractionUniformBuffer.destroy();
    this.fallbacks.destroy();
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
  /** Entries of the active-index list the last updateUniforms sized the pass for. */
  private lastActiveCount: number | null = null;
  /** The aggregated edge set the last updateUniforms saw, or null for no cut. */
  private lastLodEdges: LodEdgeDispatch | null = null;

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

    // Repulsion shader module (N^2 all-pairs)
    const repulsionShader = device.createShaderModule({
      label: "t-FDP Repulsion Shader",
      code: T_FDP_REPULSION_WGSL,
    });

    // One entry point: dispatched over the active-index list, inert slots
    // masked. The unmasked variant that used to sit beside it was a hole — a
    // removed node's zeroed slot repelled as a phantom body at the origin —
    // and a context that supplies no flags now binds an all-live fallback
    // rather than switching to unmasked arithmetic.
    const repulsion = device.createComputePipeline({
      label: "t-FDP Repulsion Pipeline",
      layout: "auto",
      compute: {
        module: repulsionShader,
        entryPoint: "main_masked",
      },
    });

    // Attraction shader module (per-edge)
    const attractionShader = device.createShaderModule({
      label: "t-FDP Attraction Shader",
      code: T_FDP_ATTRACTION_WGSL,
    });

    // Attraction pipeline: uniforms, positions, forces, edge_sources,
    // edge_targets, node_flags
    const attractionEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ];

    const attractionLayout = device.createBindGroupLayout({
      label: "t-FDP Attraction Layout",
      entries: attractionEntries,
    });

    const attraction = device.createComputePipeline({
      label: "t-FDP Attraction Pipeline",
      layout: device.createPipelineLayout({ bindGroupLayouts: [attractionLayout] }),
      compute: {
        module: attractionShader,
        entryPoint: "main",
      },
    });

    // Bundled attraction: the un-cut bindings plus the combined LOD edge set
    // and the per-node mass each arriving pull is divided by. Seven storage
    // buffers, inside the WebGPU default of eight.
    const attractionBundledLayout = device.createBindGroupLayout({
      label: "t-FDP Attraction Layout (LOD bundles)",
      entries: [
        ...attractionEntries,
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const attractionBundled = device.createComputePipeline({
      label: "t-FDP Attraction Pipeline (LOD bundles)",
      layout: device.createPipelineLayout({ bindGroupLayouts: [attractionBundledLayout] }),
      compute: {
        module: attractionShader,
        entryPoint: "main_bundled",
      },
    });

    const pipelines: TFdpPipelines = {
      repulsion,
      attraction,
      attractionLayout,
      attractionBundled,
      attractionBundledLayout,
    };
    return pipelines;
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    // Repulsion uniforms: { node_count, gamma, repulsion_scale, dist_scale,
    //                       active_count, 3 x padding }
    const repulsionUniformBuffer = device.createBuffer({
      label: "t-FDP Repulsion Uniforms",
      size: T_FDP_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Attraction uniforms: { edge_count, alpha, beta, dist_scale,
    //                        active_edge_count, bundle_count, 2 x padding }
    const attractionUniformBuffer = device.createBuffer({
      label: "t-FDP Attraction Uniforms",
      size: T_FDP_ATTRACTION_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    return new TFdpBuffers(
      repulsionUniformBuffer,
      attractionUniformBuffer,
      createAlgorithmLodFallbacks(device, maxNodes, "t-FDP"),
    );
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    const buffers = algorithmBuffers as TFdpBuffers;
    const tfdpPipelines = pipelines as TFdpPipelines;

    // Repulsion bind group: uniforms, positions, forces, node_flags,
    // active-index list, per-node mass. A context that supplies neither gets the all-live
    // flags and the identity list, which is the same slot-order all-pairs sum
    // the unmasked entry point used to compute.
    const repulsion = device.createBindGroup({
      label: "t-FDP Repulsion Bind Group",
      layout: tfdpPipelines.repulsion.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.repulsionUniformBuffer } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: lodNodeFlags(context, buffers.fallbacks) } },
        { binding: 4, resource: { buffer: lodLiveIndices(context, buffers.fallbacks) } },
        { binding: 5, resource: { buffer: lodNodeMass(context, buffers.fallbacks) } },
      ],
    });

    // Attraction bind group: uniforms, positions, forces, edge_sources,
    // edge_targets, node_flags
    if (!context.edgeSources || !context.edgeTargets) {
      throw new Error(
        "t-FDP requires edge source/target buffers in AlgorithmRenderContext. " +
          "Ensure graph.ts populates edgeSources and edgeTargets.",
      );
    }

    const attractionEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: buffers.attractionUniformBuffer } },
      { binding: 1, resource: { buffer: context.positions } },
      { binding: 2, resource: { buffer: context.forces } },
      { binding: 3, resource: { buffer: context.edgeSources } },
      { binding: 4, resource: { buffer: context.edgeTargets } },
      { binding: 5, resource: { buffer: lodNodeFlags(context, buffers.fallbacks) } },
    ];

    const attraction = device.createBindGroup({
      label: "t-FDP Attraction Bind Group",
      layout: tfdpPipelines.attractionLayout,
      entries: attractionEntries,
    });

    // Built unconditionally, dispatched only under a cut: whether an
    // aggregation is live is per-frame state, and createBindGroups is
    // contractually forbidden from capturing any of that.
    const attractionBundled = device.createBindGroup({
      label: "t-FDP Attraction Bind Group (LOD bundles)",
      layout: tfdpPipelines.attractionBundledLayout,
      entries: [
        ...attractionEntries,
        { binding: 6, resource: { buffer: lodEdgeSet(context, buffers.fallbacks) } },
        { binding: 7, resource: { buffer: lodNodeMass(context, buffers.fallbacks) } },
      ],
    });

    const bindGroups: TFdpBindGroups = { repulsion, attraction, attractionBundled };
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

    // Distance normalization: the paper's kernels do all their work at
    // normalized d ≈ O(1) — with defaults (alpha=0.1, beta=8, gamma=2) a
    // connected pair equilibrates at d ≈ 0.48. Mapping one kernel unit to
    // 2 × springLength puts that equilibrium near the configured spring
    // length in world units; without this the kernel is ~0 at typical
    // world spacing and the layout collapses.
    const distScale = 2 * fc.springLength;

    // The repulsion dispatch and the shader's two loop bounds must all come
    // from this one number, or threads read past the end of the list.
    this.lastActiveCount = lodActiveCount(context);
    this.lastLodEdges = lodEdgeDispatch(context);

    // Repulsion uniforms: { node_count, gamma, repulsion_scale, dist_scale,
    //                       active_count, 3 x padding }
    // Paper: repulsion_scale = 1/alpha (default: 1/0.1 = 10.0), scaled by user multiplier
    const repulsionData = new ArrayBuffer(T_FDP_UNIFORM_BYTES);
    const repView = new DataView(repulsionData);
    repView.setUint32(0, context.nodeCount, true);
    repView.setFloat32(4, fc.tFdpGamma, true);
    repView.setFloat32(8, (1.0 / fc.tFdpAlpha) * fc.tFdpRepulsionScale, true);
    repView.setFloat32(12, distScale, true);
    repView.setUint32(16, this.lastActiveCount, true);
    device.queue.writeBuffer(buffers.repulsionUniformBuffer, 0, repulsionData);

    // Attraction uniforms: { edge_count, alpha, beta, dist_scale,
    //                        active_edge_count, bundle_count, 2 x padding }
    // The two LOD counts are zero with no cut, which is what keeps `main`
    // reading nothing new.
    const attractionData = new ArrayBuffer(T_FDP_ATTRACTION_UNIFORM_BYTES);
    const attrView = new DataView(attractionData);
    attrView.setUint32(0, context.edgeCount, true);
    attrView.setFloat32(4, fc.tFdpAlpha, true);
    attrView.setFloat32(8, fc.tFdpBeta, true);
    attrView.setFloat32(12, distScale, true);
    attrView.setUint32(16, this.lastLodEdges?.activeEdgeCount ?? 0, true);
    attrView.setUint32(20, this.lastLodEdges?.bundleCount ?? 0, true);
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

    // Pass 1: Repulsion (N^2 over the active set, inert slots masked)
    {
      const workgroups = calculateWorkgroups(this.lastActiveCount ?? nodeCount, 256);
      const pass = encoder.beginComputePass({ label: "t-FDP Repulsion" });
      pass.setPipeline(tfdpPipelines.repulsion);
      pass.setBindGroup(0, tfdpBindGroups.repulsion);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
    }

    // Pass 2: Attraction (per-edge: linear spring + attractive t-force).
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
        const pass = encoder.beginComputePass({ label: "t-FDP Attraction (LOD bundles)" });
        pass.setPipeline(tfdpPipelines.attractionBundled);
        pass.setBindGroup(0, tfdpBindGroups.attractionBundled);
        pass.dispatchWorkgroups(calculateWorkgroups(lodEdges.total, 256));
        pass.end();
      }
    } else if (this.lastEdgeCount > 0) {
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
    this.lastActiveCount = null;
    this.lastLodEdges = null;
  }
}

/**
 * Create t-FDP force algorithm instance
 */
export function createTFdpAlgorithm(): ForceAlgorithm {
  return new TFdpAlgorithm();
}
