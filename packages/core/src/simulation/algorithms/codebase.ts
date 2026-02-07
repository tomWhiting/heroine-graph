/**
 * Codebase Layout Algorithm — Hierarchy-Aware Force-Directed
 *
 * Uses containment hierarchy (directory->file->symbol) to define
 * community assignments, then runs a force-directed simulation with:
 *   - Community-modulated N² repulsion (reduced for same-community pairs)
 *   - Cluster centroid attraction (pulls nodes toward their community center)
 *
 * Same GPU pipeline as Community Layout, different community source.
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

// Shaders (shared with community layout)
import REPULSION_COMMUNITY_WGSL from "../shaders/repulsion_community.comp.wgsl";
import CLUSTER_CLEAR_WGSL from "../shaders/cluster_clear.comp.wgsl";
import CLUSTER_ACCUMULATE_WGSL from "../shaders/cluster_accumulate.comp.wgsl";
import CLUSTER_ATTRACT_WGSL from "../shaders/cluster_attract.comp.wgsl";

const CODEBASE_LAYOUT_INFO: ForceAlgorithmInfo = {
  id: "codebase" as "codebase",
  name: "Codebase Layout",
  description:
    "Hierarchy-aware force simulation using containment relationships. " +
    "Nodes in the same directory cluster together naturally.",
  minNodes: 0,
  maxNodes: -1,
  complexity: "O(n²)",
};

/**
 * Codebase layout algorithm buffers
 */
class CodebaseLayoutBuffers implements AlgorithmBuffers {
  constructor(
    public communityIds: GPUBuffer,
    public centroidSumX: GPUBuffer,
    public centroidSumY: GPUBuffer,
    public centroidCount: GPUBuffer,
    public degrees: GPUBuffer,
    public repulsionUniforms: GPUBuffer,
    public clearUniforms: GPUBuffer,
    public accumUniforms: GPUBuffer,
    public attractUniforms: GPUBuffer,
  ) {}

  destroy(): void {
    this.communityIds.destroy();
    this.centroidSumX.destroy();
    this.centroidSumY.destroy();
    this.centroidCount.destroy();
    this.degrees.destroy();
    this.repulsionUniforms.destroy();
    this.clearUniforms.destroy();
    this.accumUniforms.destroy();
    this.attractUniforms.destroy();
  }
}

/**
 * Codebase Layout: force-directed with hierarchy-based cluster attraction
 * and community-modulated repulsion.
 *
 * Call `uploadCommunityIds()` after mapping hierarchy to community IDs.
 * The GPU simulation then continuously applies modulated repulsion and
 * cluster attraction forces alongside edge springs.
 */
export class CodebaseLayoutAlgorithm implements ForceAlgorithm {
  readonly info = CODEBASE_LAYOUT_INFO;
  readonly handlesGravity = true; // Cluster attraction replaces center gravity
  readonly handlesSprings = false;

  private repulsionUniforms: GPUBuffer | null = null;
  private clearUniforms: GPUBuffer | null = null;
  private accumUniforms: GPUBuffer | null = null;
  private attractUniforms: GPUBuffer | null = null;

  private communityIdsBuffer: GPUBuffer | null = null;
  private centroidSumXBuffer: GPUBuffer | null = null;
  private centroidSumYBuffer: GPUBuffer | null = null;
  private centroidCountBuffer: GPUBuffer | null = null;
  private degreesBuffer: GPUBuffer | null = null;
  private maxNodes = 0;
  private communityCount = 1;

  /**
   * Upload hierarchy-based community assignments to the GPU.
   * Each node's community ID corresponds to its parent directory/container.
   */
  uploadCommunityIds(
    device: GPUDevice,
    assignments: Uint32Array,
    communityCount: number,
  ): void {
    if (!this.communityIdsBuffer) {
      throw new Error("Codebase layout buffers not initialized");
    }
    this.communityCount = Math.max(1, communityCount);
    const maxBytes = this.maxNodes * 4;
    const uploadBytes = Math.min(assignments.byteLength, maxBytes);
    device.queue.writeBuffer(
      this.communityIdsBuffer,
      0,
      assignments.buffer as ArrayBuffer,
      assignments.byteOffset,
      uploadBytes,
    );
  }

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

    const repulsionModule = device.createShaderModule({
      label: "Codebase: Modulated Repulsion Shader",
      code: REPULSION_COMMUNITY_WGSL,
    });
    const clearModule = device.createShaderModule({
      label: "Codebase: Clear Centroids Shader",
      code: CLUSTER_CLEAR_WGSL,
    });
    const accumModule = device.createShaderModule({
      label: "Codebase: Accumulate Centroids Shader",
      code: CLUSTER_ACCUMULATE_WGSL,
    });
    const attractModule = device.createShaderModule({
      label: "Codebase: Cluster Attract Shader",
      code: CLUSTER_ATTRACT_WGSL,
    });

    return {
      repulsion: device.createComputePipeline({
        label: "Codebase: Modulated Repulsion Pipeline",
        layout: "auto",
        compute: { module: repulsionModule, entryPoint: "main" },
      }),
      clearCentroids: device.createComputePipeline({
        label: "Codebase: Clear Centroids Pipeline",
        layout: "auto",
        compute: { module: clearModule, entryPoint: "main" },
      }),
      accumulate: device.createComputePipeline({
        label: "Codebase: Accumulate Centroids Pipeline",
        layout: "auto",
        compute: { module: accumModule, entryPoint: "main" },
      }),
      attract: device.createComputePipeline({
        label: "Codebase: Cluster Attract Pipeline",
        layout: "auto",
        compute: { module: attractModule, entryPoint: "main" },
      }),
    };
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    this.maxNodes = maxNodes;

    this.repulsionUniforms = device.createBuffer({
      label: "Codebase: Repulsion Uniforms",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.clearUniforms = device.createBuffer({
      label: "Codebase: Clear Uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.accumUniforms = device.createBuffer({
      label: "Codebase: Accum Uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.attractUniforms = device.createBuffer({
      label: "Codebase: Attract Uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.communityIdsBuffer = device.createBuffer({
      label: "Codebase: Community IDs",
      size: maxNodes * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.centroidSumXBuffer = device.createBuffer({
      label: "Codebase: Centroid Sum X",
      size: maxNodes * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.centroidSumYBuffer = device.createBuffer({
      label: "Codebase: Centroid Sum Y",
      size: maxNodes * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.centroidCountBuffer = device.createBuffer({
      label: "Codebase: Centroid Count",
      size: maxNodes * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    // Node degree buffer for degree-weighted repulsion (FA2 core innovation)
    this.degreesBuffer = device.createBuffer({
      label: "Codebase: Node Degrees",
      size: maxNodes * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new CodebaseLayoutBuffers(
      this.communityIdsBuffer,
      this.centroidSumXBuffer,
      this.centroidSumYBuffer,
      this.centroidCountBuffer,
      this.degreesBuffer,
      this.repulsionUniforms,
      this.clearUniforms,
      this.accumUniforms,
      this.attractUniforms,
    );
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    _algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    if (
      !this.repulsionUniforms ||
      !this.clearUniforms ||
      !this.accumUniforms ||
      !this.attractUniforms ||
      !this.communityIdsBuffer ||
      !this.centroidSumXBuffer ||
      !this.centroidSumYBuffer ||
      !this.centroidCountBuffer ||
      !this.degreesBuffer
    ) {
      throw new Error("Codebase layout buffers not initialized");
    }

    // Degree-weighted modulated repulsion: uniforms, positions, forces, communityIds, degrees
    const repulsion = device.createBindGroup({
      label: "Codebase: Repulsion Bind Group",
      layout: pipelines.repulsion.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.repulsionUniforms } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: this.communityIdsBuffer } },
        { binding: 4, resource: { buffer: this.degreesBuffer } },
      ],
    });

    const clearCentroids = device.createBindGroup({
      label: "Codebase: Clear Centroids Bind Group",
      layout: pipelines.clearCentroids.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.clearUniforms } },
        { binding: 1, resource: { buffer: this.centroidSumXBuffer } },
        { binding: 2, resource: { buffer: this.centroidSumYBuffer } },
        { binding: 3, resource: { buffer: this.centroidCountBuffer } },
      ],
    });

    const accumulate = device.createBindGroup({
      label: "Codebase: Accumulate Bind Group",
      layout: pipelines.accumulate.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.accumUniforms } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: this.communityIdsBuffer } },
        { binding: 3, resource: { buffer: this.centroidSumXBuffer } },
        { binding: 4, resource: { buffer: this.centroidSumYBuffer } },
        { binding: 5, resource: { buffer: this.centroidCountBuffer } },
      ],
    });

    const attract = device.createBindGroup({
      label: "Codebase: Attract Bind Group",
      layout: pipelines.attract.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.attractUniforms } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: this.communityIdsBuffer } },
        { binding: 4, resource: { buffer: this.centroidSumXBuffer } },
        { binding: 5, resource: { buffer: this.centroidSumYBuffer } },
        { binding: 6, resource: { buffer: this.centroidCountBuffer } },
        { binding: 7, resource: { buffer: this.degreesBuffer } },
      ],
    });

    return { repulsion, clearCentroids, accumulate, attract };
  }

  updateUniforms(
    device: GPUDevice,
    _algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void {
    // Modulated repulsion uniforms — ALL values derived from config sliders, nothing hardcoded.
    //   codebaseFilePadding       → intra_factor (within-community repulsion scale)
    //   codebaseDirectoryPadding  → inter_factor (between-community repulsion scale)
    //   codebaseSpreadFactor      → global repulsion multiplier
    //   centerStrength            → distance-independent gravity
    if (this.repulsionUniforms) {
      const cfg = context.forceConfig;
      const data = new ArrayBuffer(32);
      const view = new DataView(data);
      view.setUint32(0, context.nodeCount, true);
      view.setFloat32(4, Math.abs(cfg.repulsionStrength) * cfg.codebaseSpreadFactor, true);
      view.setFloat32(8, cfg.repulsionDistanceMin, true);
      view.setFloat32(12, cfg.codebaseFilePadding * 0.1, true);       // File Padding slider: 8 → 0.8
      view.setFloat32(16, cfg.codebaseDirectoryPadding * 0.1, true);  // Dir Padding slider: 15 → 1.5
      view.setFloat32(20, cfg.centerStrength * 100, true);            // Center Gravity slider: 0.01 → 1.0
      view.setFloat32(24, 0.0, true);
      view.setFloat32(28, 0.0, true);
      device.queue.writeBuffer(this.repulsionUniforms, 0, data);
    }

    // Compute and upload node degrees for degree-weighted repulsion (FA2 core innovation)
    if (this.degreesBuffer) {
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
          if (src < context.nodeCount) degrees[src]++;
          if (tgt < context.nodeCount) degrees[tgt]++;
        }
      }
      device.queue.writeBuffer(this.degreesBuffer, 0, degrees);
    }

    // Clear uniforms: { community_count, _pad x3 }
    if (this.clearUniforms) {
      const data = new ArrayBuffer(16);
      const view = new DataView(data);
      view.setUint32(0, this.communityCount, true);
      view.setUint32(4, 0, true);
      view.setUint32(8, 0, true);
      view.setUint32(12, 0, true);
      device.queue.writeBuffer(this.clearUniforms, 0, data);
    }

    // Accumulate uniforms: { node_count, _pad x3 }
    if (this.accumUniforms) {
      const data = new ArrayBuffer(16);
      const view = new DataView(data);
      view.setUint32(0, context.nodeCount, true);
      view.setUint32(4, 0, true);
      view.setUint32(8, 0, true);
      view.setUint32(12, 0, true);
      device.queue.writeBuffer(this.accumUniforms, 0, data);
    }

    // Attract uniforms: { node_count, attraction_strength, _pad x2 }
    if (this.attractUniforms) {
      const data = new ArrayBuffer(16);
      const view = new DataView(data);
      view.setUint32(0, context.nodeCount, true);
      // Stiffness slider controls centroid attraction directly — shader scales by 1/sqrt(count)
      view.setFloat32(4, context.forceConfig.codebaseStiffness, true);
      view.setUint32(8, 0, true);
      view.setUint32(12, 0, true);
      device.queue.writeBuffer(this.attractUniforms, 0, data);
    }
  }

  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: AlgorithmPipelines,
    bindGroups: AlgorithmBindGroups,
    nodeCount: number,
  ): void {
    const nodeWorkgroups = calculateWorkgroups(nodeCount, 256);
    const commWorkgroups = calculateWorkgroups(this.communityCount, 256);

    // Phase 1: Clear centroid accumulators
    {
      const pass = encoder.beginComputePass({ label: "Codebase: Clear Centroids" });
      pass.setPipeline(pipelines.clearCentroids);
      pass.setBindGroup(0, bindGroups.clearCentroids);
      pass.dispatchWorkgroups(commWorkgroups);
      pass.end();
    }

    // Phase 2: Community-modulated N² repulsion
    {
      const pass = encoder.beginComputePass({ label: "Codebase: Modulated Repulsion" });
      pass.setPipeline(pipelines.repulsion);
      pass.setBindGroup(0, bindGroups.repulsion);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }

    // Phase 3: Accumulate community centroids
    {
      const pass = encoder.beginComputePass({ label: "Codebase: Accumulate Centroids" });
      pass.setPipeline(pipelines.accumulate);
      pass.setBindGroup(0, bindGroups.accumulate);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }

    // Phase 4: Apply cluster attraction
    {
      const pass = encoder.beginComputePass({ label: "Codebase: Cluster Attraction" });
      pass.setPipeline(pipelines.attract);
      pass.setBindGroup(0, bindGroups.attract);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }
  }

  destroy(): void {
    this.repulsionUniforms?.destroy();
    this.clearUniforms?.destroy();
    this.accumUniforms?.destroy();
    this.attractUniforms?.destroy();
    this.repulsionUniforms = null;
    this.clearUniforms = null;
    this.accumUniforms = null;
    this.attractUniforms = null;
    this.communityIdsBuffer = null;
    this.centroidSumXBuffer = null;
    this.centroidSumYBuffer = null;
    this.centroidCountBuffer = null;
    this.degreesBuffer = null;
  }
}

/**
 * Create Codebase Layout algorithm instance
 */
export function createCodebaseLayoutAlgorithm(): CodebaseLayoutAlgorithm {
  return new CodebaseLayoutAlgorithm();
}
