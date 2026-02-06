/**
 * ForceAtlas2 Force Algorithm
 *
 * A force-directed layout algorithm optimized for network visualization.
 * Designed for continuous layout with good cluster separation.
 *
 * Key features:
 * - Linear attraction (not quadratic spring)
 * - Degree-weighted repulsion
 * - LinLog mode for better cluster separation
 * - Strong gravity for disconnected components
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

// Import shader source
import FORCE_ATLAS2_WGSL from "../shaders/force_atlas2.comp.wgsl";

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
 * ForceAtlas2 algorithm-specific buffers
 */
class ForceAtlas2Buffers implements AlgorithmBuffers {
  constructor(
    public uniformBuffer: GPUBuffer,
    public degreesBuffer: GPUBuffer,
    /** Maximum node count this buffer set supports */
    public maxNodes: number,
  ) {}

  destroy(): void {
    this.uniformBuffer.destroy();
    this.degreesBuffer.destroy();
  }
}

/**
 * ForceAtlas2 repulsion algorithm implementation
 */
export class ForceAtlas2Algorithm implements ForceAlgorithm {
  readonly info = FORCE_ATLAS2_ALGORITHM_INFO;
  readonly handlesGravity = true;

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

    const shaderModule = device.createShaderModule({
      label: "ForceAtlas2 Shader",
      code: FORCE_ATLAS2_WGSL,
    });

    const repulsion = device.createComputePipeline({
      label: "ForceAtlas2 Repulsion Pipeline",
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "repulsion",
      },
    });

    return { repulsion };
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    // ForceAtlas2Uniforms: 48 bytes (due to vec3 alignment)
    // { node_count: u32, scaling: f32, gravity: f32, edge_weight_influence: f32, flags: u32, _padding: vec3<u32> }
    // Note: vec3<u32> has 16-byte alignment, so struct is padded to 48 bytes
    const uniformBuffer = device.createBuffer({
      label: "ForceAtlas2 Uniforms",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Degrees buffer: stores degree of each node for weighted repulsion
    const degreesBuffer = device.createBuffer({
      label: "ForceAtlas2 Degrees",
      size: maxNodes * 4, // u32 per node
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new ForceAtlas2Buffers(uniformBuffer, degreesBuffer, maxNodes);
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    const buffers = algorithmBuffers as ForceAtlas2Buffers;

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

    return { repulsion };
  }

  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void {
    const buffers = algorithmBuffers as ForceAtlas2Buffers;

    // CRITICAL: Validate node count doesn't exceed buffer capacity.
    // Buffer overflow from undersized buffers is a security issue that can corrupt
    // GPU memory, cause crashes, or produce undefined behavior.
    if (context.nodeCount > buffers.maxNodes) {
      throw new Error(
        `ForceAtlas2 buffer overflow: nodeCount (${context.nodeCount}) exceeds buffer capacity (${buffers.maxNodes}). ` +
        `Buffers must be recreated with createBuffers() when node count increases.`
      );
    }

    // ForceAtlas2 uses different scaling than standard force-directed
    // The repulsion strength maps to FA2's "scaling" parameter (kr)
    const scaling = Math.abs(context.forceConfig.repulsionStrength) * 0.1;
    const gravity = context.forceConfig.centerStrength * 10;

    // Flags: bit 0 = linlog, bit 1 = strong_gravity, bit 2 = prevent_overlap
    // For now, use sensible defaults. Could be exposed via config later.
    const flags = 0; // Standard mode

    // 48 bytes due to vec3 alignment requirements in WGSL
    const data = new ArrayBuffer(48);
    const view = new DataView(data);

    view.setUint32(0, context.nodeCount, true); // node_count
    view.setFloat32(4, scaling, true); // scaling (kr)
    view.setFloat32(8, gravity, true); // gravity (kg)
    view.setFloat32(12, 1.0, true); // edge_weight_influence
    view.setUint32(16, flags, true); // flags
    // Padding to align vec3 at offset 32 (16-byte boundary)
    view.setUint32(20, 0, true);
    view.setUint32(24, 0, true);
    view.setUint32(28, 0, true);
    // _padding vec3<u32> at offset 32
    view.setUint32(32, 0, true);
    view.setUint32(36, 0, true);
    view.setUint32(40, 0, true);
    // Final padding to 48 bytes
    view.setUint32(44, 0, true);

    device.queue.writeBuffer(buffers.uniformBuffer, 0, data);

    // Update degrees buffer
    // For now, we use uniform degree of 1 for all nodes
    // In a full implementation, this would come from the graph structure
    const degrees = new Uint32Array(context.nodeCount);
    degrees.fill(2); // Default degree of 2 (typical for connected nodes)
    device.queue.writeBuffer(buffers.degreesBuffer, 0, degrees);
  }

  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: AlgorithmPipelines,
    bindGroups: AlgorithmBindGroups,
    nodeCount: number,
  ): void {
    const workgroups = calculateWorkgroups(nodeCount, 256);

    const pass = encoder.beginComputePass({ label: "ForceAtlas2 Repulsion" });
    pass.setPipeline(pipelines.repulsion);
    pass.setBindGroup(0, bindGroups.repulsion);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
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
