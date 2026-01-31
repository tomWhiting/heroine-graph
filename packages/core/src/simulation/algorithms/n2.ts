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
import { EmptyAlgorithmBuffers } from "./types.ts";

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
 * N² repulsion algorithm implementation
 */
export class N2ForceAlgorithm implements ForceAlgorithm {
  readonly info = N2_ALGORITHM_INFO;

  private uniformBuffer: GPUBuffer | null = null;

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

  createBuffers(device: GPUDevice, _maxNodes: number): AlgorithmBuffers {
    // N² algorithm only needs a uniform buffer
    this.uniformBuffer = device.createBuffer({
      label: "N² Repulsion Uniforms",
      size: 16, // 4 x u32/f32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    return new EmptyAlgorithmBuffers();
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    _algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    if (!this.uniformBuffer) {
      throw new Error("N² algorithm buffers not initialized");
    }

    const repulsion = device.createBindGroup({
      label: "N² Repulsion Bind Group",
      layout: pipelines.repulsion.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
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

    const data = new ArrayBuffer(16);
    const view = new DataView(data);

    // RepulsionUniforms: { node_count, repulsion_strength, min_distance, _padding }
    view.setUint32(0, context.nodeCount, true);
    view.setFloat32(4, Math.abs(context.forceConfig.repulsionStrength), true);
    view.setFloat32(8, context.forceConfig.repulsionDistanceMin, true);
    view.setUint32(12, 0, true); // padding

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
  }
}

/**
 * Create N² force algorithm instance
 */
export function createN2Algorithm(): ForceAlgorithm {
  return new N2ForceAlgorithm();
}
