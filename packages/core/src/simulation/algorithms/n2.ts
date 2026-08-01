/**
 * N² Force Algorithm
 *
 * Simple O(n²) all-pairs repulsion calculation.
 * Best for small graphs (< 5,000 nodes).
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
// (EmptyAlgorithmBuffers no longer used — N2AlgorithmBuffers replaces it)
import {
  type AlgorithmLodFallbacks,
  createAlgorithmLodFallbacks,
  lodActiveCount,
  lodLiveIndices,
  lodNodeFlags,
  lodNodeMass,
} from "./lod_bindings.ts";

// Import shader source
import REPULSION_N2_WGSL from "../shaders/repulsion_n2.comp.wgsl";

/**
 * N² algorithm info
 */
const N2_ALGORITHM_INFO: ForceAlgorithmInfo = {
  id: "n2",
  name: "N² (Direct)",
  description: "Simple all-pairs repulsion. Fast for small graphs, slow for large ones.",
  minNodes: 0,
  maxNodes: 10000,
  complexity: "O(n²)",
};

/**
 * N² algorithm-specific buffers
 */
class N2AlgorithmBuffers implements AlgorithmBuffers {
  constructor(
    public uniformBuffer: GPUBuffer,
    /** Identity list / unit mass / all-live flags for a context that supplies none */
    public fallbacks: AlgorithmLodFallbacks,
  ) {}

  destroy(): void {
    this.uniformBuffer.destroy();
    this.fallbacks.destroy();
  }
}

/**
 * N² repulsion algorithm implementation.
 *
 * Runs `main_masked`, the same entry point the core simulation pipeline uses:
 * one thread per entry of the active-index list, with the inner all-pairs sum
 * over the same list and an inert-slot mask on top.
 *
 * It used to run an unmasked `main` whose bind group carried no node_flags.
 * That was a hole. A removed node keeps its slot inside the high-water mark
 * with its position zeroed, so under `setForceAlgorithm("n2")` every hole
 * repelled as a phantom body at the origin — a graph that had shed nodes pushed
 * its survivors outward from a point where nothing was drawn — and an LOD-hidden
 * node went on repelling the proxy that replaced it on screen.
 */
export class N2ForceAlgorithm implements ForceAlgorithm {
  readonly info = N2_ALGORITHM_INFO;
  readonly handlesGravity = false;

  private uniformBuffer: GPUBuffer | null = null;
  private fallbacks: AlgorithmLodFallbacks | null = null;
  /** Entries of the active list the last updateUniforms sized the pass for. */
  private lastActiveCount: number | null = null;

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

    const shaderModule = device.createShaderModule({
      label: "N² Repulsion Shader",
      code: REPULSION_N2_WGSL,
    });

    const repulsion = device.createComputePipeline({
      label: "N² Repulsion Pipeline",
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "main_masked",
      },
    });

    return { repulsion };
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    this.uniformBuffer = device.createBuffer({
      label: "N² Repulsion Uniforms",
      // RepulsionUniforms is 32 bytes.
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.fallbacks = createAlgorithmLodFallbacks(device, maxNodes, "N²");

    return new N2AlgorithmBuffers(this.uniformBuffer, this.fallbacks);
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    _algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    if (!this.uniformBuffer || !this.fallbacks) {
      throw new Error("N² algorithm buffers not initialized");
    }
    const fallbacks = this.fallbacks;

    const repulsion = device.createBindGroup({
      label: "N² Repulsion Bind Group",
      layout: pipelines.repulsion.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: lodNodeFlags(context, fallbacks) } },
        { binding: 4, resource: { buffer: lodNodeMass(context, fallbacks) } },
        { binding: 5, resource: { buffer: lodLiveIndices(context, fallbacks) } },
      ],
    });

    return { repulsion };
  }

  updateUniforms(
    device: GPUDevice,
    _algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void {
    if (!this.uniformBuffer) return;

    // The dispatch below and the shader's loop bounds must both come from this
    // one number, or threads read past the end of the list.
    this.lastActiveCount = lodActiveCount(context);

    const data = new ArrayBuffer(32);
    const view = new DataView(data);

    // RepulsionUniforms: { node_count, repulsion_strength, min_distance,
    //                      max_distance, active_count, 3 x padding }
    view.setUint32(0, context.nodeCount, true);
    view.setFloat32(4, Math.abs(context.forceConfig.repulsionStrength), true);
    view.setFloat32(8, context.forceConfig.repulsionDistanceMin, true);
    view.setFloat32(12, context.forceConfig.repulsionDistanceMax, true);
    view.setUint32(16, this.lastActiveCount, true);

    device.queue.writeBuffer(this.uniformBuffer, 0, data);
  }

  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: AlgorithmPipelines,
    bindGroups: AlgorithmBindGroups,
    nodeCount: number,
  ): void {
    // One thread per active-list entry. Zero entries is a legal answer — every
    // slot hidden — and must dispatch nothing, so the fallback is reached only
    // when updateUniforms has never run.
    const workgroups = calculateWorkgroups(this.lastActiveCount ?? nodeCount, 256);

    const pass = encoder.beginComputePass({ label: "N² Repulsion" });
    pass.setPipeline(pipelines.repulsion);
    pass.setBindGroup(0, bindGroups.repulsion);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
  }

  /**
   * Destroy algorithm resources
   */
  destroy(): void {
    this.uniformBuffer?.destroy();
    this.uniformBuffer = null;
    this.fallbacks?.destroy();
    this.fallbacks = null;
    this.lastActiveCount = null;
  }
}

/**
 * Create N² force algorithm instance
 */
export function createN2Algorithm(): ForceAlgorithm {
  return new N2ForceAlgorithm();
}
