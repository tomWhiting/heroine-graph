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
import type {
  AlgorithmBindGroups,
  AlgorithmBuffers,
  AlgorithmPipelines,
  AlgorithmRenderContext,
  ForceAlgorithm,
  ForceAlgorithmInfo,
} from "./types.ts";

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
};

/**
 * Extended pipeline type for FA2 (repulsion + attraction passes)
 */
interface FA2Pipelines extends AlgorithmPipelines {
  attraction: GPUComputePipeline;
  attractionLayout: GPUBindGroupLayout;
}

/**
 * Extended bind group type for FA2
 */
interface FA2BindGroups extends AlgorithmBindGroups {
  attraction: GPUBindGroup;
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
  }
}

/**
 * ForceAtlas2 algorithm implementation.
 *
 * handlesSprings = false: FA2 now delegates to standard Hooke's law springs
 * for attraction. The original FA2 linear attraction (F = d, no rest length)
 * created spoke/wheel patterns and had no equilibrium distance. Standard
 * springs produce better layouts and are consistent with other algorithms.
 * The FA2 attraction shader is preserved but not dispatched.
 */
export class ForceAtlas2Algorithm implements ForceAlgorithm {
  readonly info = FORCE_ATLAS2_ALGORITHM_INFO;
  readonly handlesGravity = true;
  readonly handlesSprings = false;

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

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

    // Attraction pipeline: uniforms, positions, forces, edge_sources, edge_targets, edge_weights
    const attractionLayout = device.createBindGroupLayout({
      label: "ForceAtlas2 Attraction Layout",
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
      label: "ForceAtlas2 Attraction Pipeline",
      layout: device.createPipelineLayout({ bindGroupLayouts: [attractionLayout] }),
      compute: {
        module: attractionShader,
        entryPoint: "main",
      },
    });

    const pipelines: FA2Pipelines = {
      repulsion,
      attraction,
      attractionLayout,
    };
    return pipelines;
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    // ForceAtlas2Uniforms: 48 bytes (due to vec3 alignment)
    const uniformBuffer = device.createBuffer({
      label: "ForceAtlas2 Uniforms",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // FA2AttractionUniforms: 16 bytes { edge_count, edge_weight_influence, flags, _padding }
    const attractionUniforms = device.createBuffer({
      label: "ForceAtlas2 Attraction Uniforms",
      size: 16,
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

    // Repulsion bind group: uniforms, positions, forces, degrees
    const repulsion = device.createBindGroup({
      label: "ForceAtlas2 Repulsion Bind Group",
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
        "ForceAtlas2 requires edge source/target buffers in AlgorithmRenderContext. " +
        "Ensure graph.ts populates edgeSources and edgeTargets.",
      );
    }

    const attraction = device.createBindGroup({
      label: "ForceAtlas2 Attraction Bind Group",
      layout: fa2Pipelines.attractionLayout,
      entries: [
        { binding: 0, resource: { buffer: buffers.attractionUniforms } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: context.edgeSources } },
        { binding: 4, resource: { buffer: context.edgeTargets } },
        { binding: 5, resource: { buffer: buffers.edgeWeightsBuffer } },
      ],
    });

    const bindGroups: FA2BindGroups = { repulsion, attraction };
    return bindGroups;
  }

  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void {
    const buffers = algorithmBuffers as ForceAtlas2Buffers;

    // CRITICAL: Validate node count doesn't exceed buffer capacity.
    if (context.nodeCount > buffers.maxNodes) {
      throw new Error(
        `ForceAtlas2 buffer overflow: nodeCount (${context.nodeCount}) exceeds buffer capacity (${buffers.maxNodes}). ` +
        `Buffers must be recreated with createBuffers() when node count increases.`
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

    // Repulsion uniforms: 48 bytes due to vec3 alignment requirements in WGSL
    const data = new ArrayBuffer(48);
    const view = new DataView(data);

    view.setUint32(0, context.nodeCount, true); // node_count
    view.setFloat32(4, scaling, true); // scaling (kr)
    view.setFloat32(8, gravity, true); // gravity (kg)
    view.setFloat32(12, 1.0, true); // edge_weight_influence
    view.setUint32(16, flags, true); // flags
    view.setUint32(20, 0, true);
    view.setUint32(24, 0, true);
    view.setUint32(28, 0, true);
    view.setUint32(32, 0, true);
    view.setUint32(36, 0, true);
    view.setUint32(40, 0, true);
    view.setUint32(44, 0, true);

    device.queue.writeBuffer(buffers.uniformBuffer, 0, data);

    // Attraction uniforms: 16 bytes { edge_count, edge_weight_influence, flags, _padding }
    const attractData = new ArrayBuffer(16);
    const attractView = new DataView(attractData);
    attractView.setUint32(0, context.edgeCount, true);
    attractView.setFloat32(4, 1.0, true); // edge_weight_influence (delta)
    attractView.setUint32(8, flags & 1, true); // linlog flag only
    attractView.setUint32(12, 0, true);
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
    const fa2Pipelines = pipelines as FA2Pipelines;
    const fa2BindGroups = bindGroups as FA2BindGroups;

    // Pass 1: Repulsion + Gravity (combined in shader, N² over nodes)
    {
      const workgroups = calculateWorkgroups(nodeCount, 256);
      const pass = encoder.beginComputePass({ label: "ForceAtlas2 Repulsion + Gravity" });
      pass.setPipeline(fa2Pipelines.repulsion);
      pass.setBindGroup(0, fa2BindGroups.repulsion);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
    }

    // FA2 attraction pass disabled — handlesSprings=false delegates to standard
    // Hooke's law springs for consistent behavior across algorithms.
    // The attraction shader and pipeline are preserved for potential future use.
  }

  destroy(): void {
    // Buffers are destroyed via AlgorithmBuffers.destroy()
  }
}

/**
 * Create ForceAtlas2 force algorithm instance
 */
export function createForceAtlas2Algorithm(): ForceAlgorithm {
  return new ForceAtlas2Algorithm();
}
