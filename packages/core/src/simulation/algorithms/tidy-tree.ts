/**
 * Tidy Tree Layout Algorithm
 *
 * Computes target positions using Buchheim's O(n) tidy tree algorithm
 * (via WASM), then applies GPU spring forces to animate nodes toward
 * those target positions.
 *
 * Unlike force-directed algorithms which converge iteratively, this
 * algorithm computes the layout analytically and uses the force system
 * only for smooth animated transitions.
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
import TIDY_TREE_WGSL from "../shaders/tidy_tree.comp.wgsl";

/**
 * Tidy tree algorithm info
 */
const TIDY_TREE_ALGORITHM_INFO: ForceAlgorithmInfo = {
  id: "tidy-tree",
  name: "Tidy Tree",
  description:
    "Tree/hierarchy layout using Buchheim's algorithm. Best for directory trees, org charts, and DAGs.",
  minNodes: 0,
  maxNodes: -1, // Unlimited (O(n) layout)
  complexity: "O(n)",
};

/**
 * Configuration for the tidy tree algorithm.
 * Parameters are read from FullForceConfig (tidyTree* fields).
 */
export interface TidyTreeConfig {
  /** Spacing between tree levels (default: 80) */
  levelSeparation: number;
  /** Minimum separation between sibling nodes (default: 1.0) */
  siblingSeparation: number;
  /** Minimum separation between subtrees (default: 2.0) */
  subtreeSeparation: number;
  /** Spring stiffness toward target positions (default: 0.3) */
  stiffness: number;
  /** Damping factor for approach to target (default: 0.5) */
  damping: number;
  /** Use radial coordinates (true) or linear top-down (false) */
  radial: boolean;
}

/**
 * Tidy Tree algorithm buffers
 */
class TidyTreeBuffers implements AlgorithmBuffers {
  constructor(public targetPositions: GPUBuffer) {}

  destroy(): void {
    this.targetPositions.destroy();
  }
}

/**
 * Tidy Tree layout algorithm implementation.
 *
 * This algorithm works in two phases:
 * 1. CPU/WASM phase: Compute target positions using Buchheim's O(n) algorithm
 * 2. GPU phase: Apply spring forces pulling nodes toward target positions
 *
 * The target positions buffer must be populated externally by calling
 * uploadTargetPositions() after computing the layout.
 */
export class TidyTreeAlgorithm implements ForceAlgorithm {
  readonly info = TIDY_TREE_ALGORITHM_INFO;
  readonly handlesGravity = true; // Tree layout provides its own centering
  readonly handlesSprings = true; // Tree layout uses target positions, not edge springs

  private uniformBuffer: GPUBuffer | null = null;
  private targetPositionsBuffer: GPUBuffer | null = null;
  private maxNodes = 0;

  /**
   * Upload target positions to the GPU buffer.
   *
   * @param device - GPU device
   * @param positions - Float32Array of interleaved [x0, y0, x1, y1, ...] target positions
   */
  uploadTargetPositions(device: GPUDevice, positions: Float32Array): void {
    if (!this.targetPositionsBuffer) {
      throw new Error("Tidy tree buffers not initialized");
    }

    const maxBytes = this.maxNodes * 2 * 4; // vec2<f32> per node
    const uploadBytes = Math.min(positions.byteLength, maxBytes);
    device.queue.writeBuffer(
      this.targetPositionsBuffer,
      0,
      positions.buffer,
      positions.byteOffset,
      uploadBytes,
    );
  }

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

    const shaderModule = device.createShaderModule({
      label: "Tidy Tree Layout Shader",
      code: TIDY_TREE_WGSL,
    });

    const repulsion = device.createComputePipeline({
      label: "Tidy Tree Layout Pipeline",
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    return { repulsion };
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    this.maxNodes = maxNodes;

    // Uniform buffer: TreeUniforms { node_count: u32, stiffness: f32, damping: f32, _padding: u32 }
    this.uniformBuffer = device.createBuffer({
      label: "Tidy Tree Uniforms",
      size: 16, // 4 x u32/f32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Target positions buffer: vec2<f32> per node
    this.targetPositionsBuffer = device.createBuffer({
      label: "Tidy Tree Target Positions",
      size: maxNodes * 2 * 4, // vec2<f32> per node
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new TidyTreeBuffers(this.targetPositionsBuffer);
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    _algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    if (!this.uniformBuffer || !this.targetPositionsBuffer) {
      throw new Error("Tidy tree buffers not initialized");
    }

    const repulsion = device.createBindGroup({
      label: "Tidy Tree Bind Group",
      layout: pipelines.repulsion.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: this.targetPositionsBuffer } },
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

    // TreeUniforms: { node_count: u32, stiffness: f32, damping: f32, _padding: u32 }
    view.setUint32(0, context.nodeCount, true);
    view.setFloat32(4, context.forceConfig.tidyTreeStiffness, true);
    view.setFloat32(8, context.forceConfig.tidyTreeDamping, true);
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

    const pass = encoder.beginComputePass({ label: "Tidy Tree Layout" });
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
    // targetPositionsBuffer is destroyed via TidyTreeBuffers.destroy()
    this.targetPositionsBuffer = null;
  }
}

/**
 * Create Tidy Tree force algorithm instance
 */
export function createTidyTreeAlgorithm(): TidyTreeAlgorithm {
  return new TidyTreeAlgorithm();
}
