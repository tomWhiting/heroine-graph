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
import { NODE_MASS_UNIT } from "../../lod/mass.ts";

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
    /** Unit-filled masses for contexts that supply none (see createBuffers) */
    public fallbackNodeMass: GPUBuffer,
  ) {}

  destroy(): void {
    this.uniformBuffer.destroy();
    this.fallbackNodeMass.destroy();
  }
}

/**
 * N² repulsion algorithm implementation
 */
export class N2ForceAlgorithm implements ForceAlgorithm {
  readonly info = N2_ALGORITHM_INFO;
  readonly handlesGravity = false;

  private uniformBuffer: GPUBuffer | null = null;
  private fallbackNodeMass: GPUBuffer | null = null;

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
        entryPoint: "main",
      },
    });

    return { repulsion };
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    this.uniformBuffer = device.createBuffer({
      label: "N² Repulsion Uniforms",
      // RepulsionUniforms is 32 bytes: this entry point ignores the
      // active-index half of it, but a uniform binding smaller than the
      // declared struct is a validation error, and an invalid bind group
      // discards the whole frame's command buffer rather than erroring here.
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Unit masses for contexts that supply no nodeMass buffer. Unlike the
    // node-flags fallback this cannot rely on zero-init: a zeroed mass buffer
    // is a graph with no repulsion at all.
    const slots = Math.max(maxNodes, 1);
    this.fallbackNodeMass = device.createBuffer({
      label: "N² Fallback Node Mass",
      size: slots * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      this.fallbackNodeMass,
      0,
      new Float32Array(slots).fill(NODE_MASS_UNIT),
    );

    return new N2AlgorithmBuffers(this.uniformBuffer, this.fallbackNodeMass);
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    _algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    if (!this.uniformBuffer || !this.fallbackNodeMass) {
      throw new Error("N² algorithm buffers not initialized");
    }

    const repulsion = device.createBindGroup({
      label: "N² Repulsion Bind Group",
      layout: pipelines.repulsion.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 4, resource: { buffer: context.nodeMass ?? this.fallbackNodeMass } },
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

    const data = new ArrayBuffer(32);
    const view = new DataView(data);

    // RepulsionUniforms: { node_count, repulsion_strength, min_distance,
    //                      max_distance, active_count, 3 x padding }
    // This plugin runs the unmasked `main` entry point, which iterates every
    // slot and reads neither active_count nor the list; it is written anyway
    // so the buffer never disagrees with the struct it is bound as.
    view.setUint32(0, context.nodeCount, true);
    view.setFloat32(4, Math.abs(context.forceConfig.repulsionStrength), true);
    view.setFloat32(8, context.forceConfig.repulsionDistanceMin, true);
    view.setFloat32(12, context.forceConfig.repulsionDistanceMax, true);
    view.setUint32(16, context.nodeCount, true);

    device.queue.writeBuffer(this.uniformBuffer, 0, data);
  }

  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: AlgorithmPipelines,
    bindGroups: AlgorithmBindGroups,
    nodeCount: number,
  ): void {
    const workgroups = calculateWorkgroups(nodeCount, 256);

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
    this.fallbackNodeMass?.destroy();
    this.fallbackNodeMass = null;
  }
}

/**
 * Create N² force algorithm instance
 */
export function createN2Algorithm(): ForceAlgorithm {
  return new N2ForceAlgorithm();
}
