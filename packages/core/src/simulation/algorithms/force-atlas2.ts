/**
 * ForceAtlas2 Force Algorithm
 *
 * A force-directed layout algorithm optimized for network visualization.
 * Designed for continuous layout with good cluster separation.
 *
 * Key features:
 * - Linear attraction (NOT Hooke's law — no rest length, no grid patterns)
 * - Degree-weighted repulsion
 * - LinLog mode for better cluster separation
 * - Strong gravity for disconnected components
 *
 * The FA2 attraction formula is F = d (always pulling, proportional to distance).
 * This is fundamentally different from Hooke's law F = k * (d - rest_length)
 * which creates equilibrium distances that produce lattice/grid patterns.
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
import FORCE_ATLAS2_WGSL from "../shaders/force_atlas2.comp.wgsl";
import FA2_ATTRACTION_WGSL from "../shaders/fa2_attraction.comp.wgsl";

/**
 * ForceAtlas2 algorithm info
 */
const FORCE_ATLAS2_ALGORITHM_INFO: ForceAlgorithmInfo = {
  id: "force-atlas2",
  name: "ForceAtlas2",
  description:
    "Optimized for network visualization with degree-weighted forces and optional LinLog mode.",
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

/** Byte size of ForceAtlas2Uniforms in force_atlas2.comp.wgsl. */
const FA2_UNIFORM_BYTES = 32;

/** Byte size of FA2AttractionUniforms in fa2_attraction.comp.wgsl. */
const FA2_ATTRACTION_UNIFORM_BYTES = 32;

/**
 * Extended pipeline type for FA2 (repulsion + attraction passes)
 */
interface FA2Pipelines extends AlgorithmPipelines {
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
 * Extended bind group type for FA2
 */
interface FA2BindGroups extends AlgorithmBindGroups {
  attraction: GPUBindGroup;
  attractionBundled: GPUBindGroup;
}

/**
 * ForceAtlas2 algorithm-specific buffers
 */
class ForceAtlas2Buffers implements AlgorithmBuffers {
  constructor(
    public uniformBuffer: GPUBuffer,
    public attractionUniforms: GPUBuffer,
    public degreesBuffer: GPUBuffer,
    public edgeWeightsBuffer: GPUBuffer,
    /** Identity list / unit mass / all-live flags for a context supplying none */
    public fallbacks: AlgorithmLodFallbacks,
    /** Maximum node count this buffer set supports */
    public maxNodes: number,
    /** Maximum edge count this buffer set supports */
    public maxEdges: number,
  ) {}

  destroy(): void {
    this.uniformBuffer.destroy();
    this.attractionUniforms.destroy();
    this.degreesBuffer.destroy();
    this.edgeWeightsBuffer.destroy();
    this.fallbacks.destroy();
  }
}

/**
 * ForceAtlas2 algorithm implementation.
 *
 * handlesSprings = true: FA2 dispatches its native linear attraction
 * (F = w^delta * d, per the published model). The repulsion calibration
 * (scaling = 0.1 * |repulsionStrength|) was tuned against F = d attraction;
 * an earlier substitution of the ~10x weaker Hooke spring pass left the 1/d
 * repulsion unopposed and inflated every graph into a structureless uniform
 * disc. The spoke/wheel oscillation that motivated that substitution came
 * from unbounded F = d attraction overshooting under fixed-step integration;
 * it is suppressed by per-node adaptive speed (prefersAdaptiveSpeed).
 */
export class ForceAtlas2Algorithm implements ForceAlgorithm {
  readonly info = FORCE_ATLAS2_ALGORITHM_INFO;
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
    assertAlgorithmSupportedOnDevice(FORCE_ATLAS2_ALGORITHM_INFO, device);

    // Repulsion + gravity shader
    const repulsionShader = device.createShaderModule({
      label: "ForceAtlas2 Repulsion Shader",
      code: FORCE_ATLAS2_WGSL,
    });

    // Linear attraction shader (separate file due to different bind group layout)
    const attractionShader = device.createShaderModule({
      label: "ForceAtlas2 Attraction Shader",
      code: FA2_ATTRACTION_WGSL,
    });

    // Repulsion pipeline: uniforms, positions, forces, degrees
    const repulsion = device.createComputePipeline({
      label: "ForceAtlas2 Repulsion Pipeline",
      layout: "auto",
      compute: {
        module: repulsionShader,
        entryPoint: "repulsion",
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
      label: "ForceAtlas2 Attraction Layout",
      entries: attractionEntries,
    });

    const attraction = device.createComputePipeline({
      label: "ForceAtlas2 Attraction Pipeline",
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
      label: "ForceAtlas2 Attraction Layout (LOD bundles)",
      entries: [
        ...attractionEntries,
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const attractionBundled = device.createComputePipeline({
      label: "ForceAtlas2 Attraction Pipeline (LOD bundles)",
      layout: device.createPipelineLayout({ bindGroupLayouts: [attractionBundledLayout] }),
      compute: {
        module: attractionShader,
        entryPoint: "main_bundled",
      },
    });

    const pipelines: FA2Pipelines = {
      repulsion,
      attraction,
      attractionLayout,
      attractionBundled,
      attractionBundledLayout,
    };
    return pipelines;
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    // ForceAtlas2Uniforms: 32 bytes
    // { node_count, scaling, gravity, edge_weight_influence, flags,
    //   active_count, 2 x padding }
    const uniformBuffer = device.createBuffer({
      label: "ForceAtlas2 Uniforms",
      size: FA2_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // FA2AttractionUniforms: 32 bytes
    // { edge_count, edge_weight_influence, flags, active_edge_count,
    //   bundle_count, 3 x padding }
    const attractionUniforms = device.createBuffer({
      label: "ForceAtlas2 Attraction Uniforms",
      size: FA2_ATTRACTION_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Degrees buffer: stores degree of each node for weighted repulsion
    const degreesBuffer = device.createBuffer({
      label: "ForceAtlas2 Degrees",
      size: maxNodes * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Edge weights: f32 per edge (allocate for max edges = maxNodes * 4 as estimate)
    const maxEdges = maxNodes * 4;
    const edgeWeightsBuffer = device.createBuffer({
      label: "ForceAtlas2 Edge Weights",
      size: Math.max(maxEdges * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new ForceAtlas2Buffers(
      uniformBuffer,
      attractionUniforms,
      degreesBuffer,
      edgeWeightsBuffer,
      createAlgorithmLodFallbacks(device, maxNodes, "ForceAtlas2"),
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
    const buffers = algorithmBuffers as ForceAtlas2Buffers;
    const fa2Pipelines = pipelines as FA2Pipelines;

    if (!context.nodeFlags) {
      throw new Error(
        "ForceAtlas2 requires the nodeFlags buffer in AlgorithmRenderContext. " +
          "Ensure graph.ts populates nodeFlags.",
      );
    }

    // Repulsion bind group: uniforms, positions, forces, degrees, node_flags,
    // active-index list, per-node mass
    const repulsion = device.createBindGroup({
      label: "ForceAtlas2 Repulsion Bind Group",
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
        "ForceAtlas2 requires edge source/target buffers in AlgorithmRenderContext. " +
          "Ensure graph.ts populates edgeSources and edgeTargets.",
      );
    }

    const attractionEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: buffers.attractionUniforms } },
      { binding: 1, resource: { buffer: context.positions } },
      { binding: 2, resource: { buffer: context.forces } },
      { binding: 3, resource: { buffer: context.edgeSources } },
      { binding: 4, resource: { buffer: context.edgeTargets } },
      { binding: 5, resource: { buffer: buffers.edgeWeightsBuffer } },
      { binding: 6, resource: { buffer: context.nodeFlags } },
    ];

    const attraction = device.createBindGroup({
      label: "ForceAtlas2 Attraction Bind Group",
      layout: fa2Pipelines.attractionLayout,
      entries: attractionEntries,
    });

    // Built unconditionally, dispatched only under a cut: whether an
    // aggregation is live is per-frame state, and createBindGroups is
    // contractually forbidden from capturing any of that.
    const attractionBundled = device.createBindGroup({
      label: "ForceAtlas2 Attraction Bind Group (LOD bundles)",
      layout: fa2Pipelines.attractionBundledLayout,
      entries: [
        ...attractionEntries,
        { binding: 7, resource: { buffer: lodEdgeSet(context, buffers.fallbacks) } },
        { binding: 8, resource: { buffer: lodNodeMass(context, buffers.fallbacks) } },
      ],
    });

    const bindGroups: FA2BindGroups = { repulsion, attraction, attractionBundled };
    return bindGroups;
  }

  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void {
    const buffers = algorithmBuffers as ForceAtlas2Buffers;

    // Cache edge count for attraction dispatch sizing in recordRepulsionPass
    this.lastEdgeCount = context.edgeCount;

    // CRITICAL: Validate node count doesn't exceed buffer capacity.
    if (context.nodeCount > buffers.maxNodes) {
      throw new Error(
        `ForceAtlas2 buffer overflow: nodeCount (${context.nodeCount}) exceeds buffer capacity (${buffers.maxNodes}). ` +
          `Buffers must be recreated with createBuffers() when node count increases.`,
      );
    }

    // Validate edge count doesn't exceed buffer capacity
    if (context.edgeCount > buffers.maxEdges) {
      throw new Error(
        `ForceAtlas2 buffer overflow: edgeCount (${context.edgeCount}) exceeds buffer capacity (${buffers.maxEdges}). ` +
          `Buffers must be recreated with createBuffers() when edge count increases.`,
      );
    }

    // FA2 force model calibration:
    // FA2 uses 1/d repulsion (not 1/d² like Coulomb) with degree-weighted mass,
    // producing inherently stronger forces. The 0.1 factor calibrates FA2's kr
    // so that user-facing repulsionStrength produces similar magnitudes to N².
    // FA2's constant-magnitude gravity (F = kg * mass, no distance scaling) needs
    // a 10x boost to counterbalance the degree-amplified repulsion.
    const scaling = Math.abs(context.forceConfig.repulsionStrength) * 0.1;
    const gravity = context.forceConfig.centerStrength * 10;

    // Flags: bit 0 = linlog, bit 1 = strong_gravity, bit 2 = prevent_overlap
    let flags = 0;
    if (context.forceConfig.linlogStrongGravity) {
      flags |= 1; // FLAG_LINLOG
      flags |= 2; // FLAG_STRONG_GRAVITY
    }

    // The repulsion dispatch below and the shader's two loop bounds must all
    // come from this one number, or threads read past the end of the list.
    this.lastActiveCount = lodActiveCount(context);
    this.lastLodEdges = lodEdgeDispatch(context);

    // Repulsion uniforms (FA2_UNIFORM_BYTES)
    const data = new ArrayBuffer(FA2_UNIFORM_BYTES);
    const view = new DataView(data);

    view.setUint32(0, context.nodeCount, true); // node_count
    view.setFloat32(4, scaling, true); // scaling (kr)
    view.setFloat32(8, gravity, true); // gravity (kg)
    view.setFloat32(12, 1.0, true); // edge_weight_influence
    view.setUint32(16, flags, true); // flags
    view.setUint32(20, this.lastActiveCount, true); // active_count
    view.setUint32(24, 0, true);
    view.setUint32(28, 0, true);

    device.queue.writeBuffer(buffers.uniformBuffer, 0, data);

    // Attraction uniforms (FA2_ATTRACTION_UNIFORM_BYTES). The two LOD counts
    // are zero with no cut, which is what keeps `main` reading nothing new.
    const attractData = new ArrayBuffer(FA2_ATTRACTION_UNIFORM_BYTES);
    const attractView = new DataView(attractData);
    attractView.setUint32(0, context.edgeCount, true);
    attractView.setFloat32(4, 1.0, true); // edge_weight_influence (delta)
    attractView.setUint32(8, flags & 1, true); // linlog flag only
    attractView.setUint32(12, this.lastLodEdges?.activeEdgeCount ?? 0, true);
    attractView.setUint32(16, this.lastLodEdges?.bundleCount ?? 0, true);
    attractView.setUint32(20, 0, true);
    attractView.setUint32(24, 0, true);
    attractView.setUint32(28, 0, true);
    device.queue.writeBuffer(buffers.attractionUniforms, 0, attractData);

    // Compute actual node degrees from CPU-side edge arrays.
    // Degree-weighted repulsion is the core of FA2 — using real degrees
    // ensures hubs repel proportionally to their connectivity.
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
    const fa2Pipelines = pipelines as FA2Pipelines;
    const fa2BindGroups = bindGroups as FA2BindGroups;

    // Pass 1: Repulsion + Gravity (combined in shader, N² over the active set)
    {
      const workgroups = calculateWorkgroups(this.lastActiveCount ?? nodeCount, 256);
      const pass = encoder.beginComputePass({ label: "ForceAtlas2 Repulsion + Gravity" });
      pass.setPipeline(fa2Pipelines.repulsion);
      pass.setBindGroup(0, fa2BindGroups.repulsion);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
    }

    // Pass 2: Native FA2 linear attraction (per-edge, F = w^delta * d).
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
        const pass = encoder.beginComputePass({
          label: "ForceAtlas2 Attraction (LOD bundles)",
        });
        pass.setPipeline(fa2Pipelines.attractionBundled);
        pass.setBindGroup(0, fa2BindGroups.attractionBundled);
        pass.dispatchWorkgroups(calculateWorkgroups(lodEdges.total, 256));
        pass.end();
      }
    } else if (this.lastEdgeCount > 0) {
      const edgeWorkgroups = calculateWorkgroups(this.lastEdgeCount, 256);
      const pass = encoder.beginComputePass({ label: "ForceAtlas2 Attraction" });
      pass.setPipeline(fa2Pipelines.attraction);
      pass.setBindGroup(0, fa2BindGroups.attraction);
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
 * Create ForceAtlas2 force algorithm instance
 */
export function createForceAtlas2Algorithm(): ForceAlgorithm {
  return new ForceAtlas2Algorithm();
}
