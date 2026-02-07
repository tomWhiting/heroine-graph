/**
 * Community Layout Algorithm — Cluster-Aware Force-Directed
 *
 * Detects communities using Louvain modularity optimization (via WASM),
 * then runs a force-directed simulation with:
 *   - Community-modulated N² repulsion (reduced for same-community pairs)
 *   - Cluster centroid attraction (pulls nodes toward their community center)
 *
 * GPU pipeline per frame:
 *   1. Clear centroid accumulators
 *   2. Community-modulated N² repulsion (same community = reduced, different = full)
 *   3. Accumulate community centroids (atomicAdd positions)
 *   4. Apply cluster attraction (pull toward centroid)
 *
 * Default edge springs are handled by the simulation pipeline
 * (handlesSprings = false). Gravity is handled by cluster attraction
 * (handlesGravity = true).
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

// Shaders
import REPULSION_COMMUNITY_WGSL from "../shaders/repulsion_community.comp.wgsl";
import CLUSTER_CLEAR_WGSL from "../shaders/cluster_clear.comp.wgsl";
import CLUSTER_ACCUMULATE_WGSL from "../shaders/cluster_accumulate.comp.wgsl";
import CLUSTER_ATTRACT_WGSL from "../shaders/cluster_attract.comp.wgsl";

const COMMUNITY_LAYOUT_INFO: ForceAlgorithmInfo = {
  id: "community" as "community",
  name: "Community Layout",
  description:
    "Louvain community detection with cluster-aware force simulation. " +
    "Same-community nodes attract, producing natural visual clusters.",
  minNodes: 0,
  maxNodes: -1,
  complexity: "O(n²)",
};

/**
 * Community layout algorithm buffers
 */
class CommunityLayoutBuffers implements AlgorithmBuffers {
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
 * Community Layout: force-directed with Louvain cluster attraction
 * and community-modulated repulsion.
 *
 * Call `uploadCommunityIds()` after Louvain detection to set community
 * assignments. The GPU simulation then continuously applies modulated
 * repulsion and cluster attraction forces alongside edge springs.
 */
export class CommunityLayoutAlgorithm implements ForceAlgorithm {
  readonly info = COMMUNITY_LAYOUT_INFO;
  readonly handlesGravity = true; // Cluster attraction replaces center gravity
  readonly handlesSprings = false; // Use default edge springs

  // Uniform buffers (16 bytes each)
  private repulsionUniforms: GPUBuffer | null = null;
  private clearUniforms: GPUBuffer | null = null;
  private accumUniforms: GPUBuffer | null = null;
  private attractUniforms: GPUBuffer | null = null;

  // Algorithm state
  private communityIdsBuffer: GPUBuffer | null = null;
  private centroidSumXBuffer: GPUBuffer | null = null;
  private centroidSumYBuffer: GPUBuffer | null = null;
  private centroidCountBuffer: GPUBuffer | null = null;
  private degreesBuffer: GPUBuffer | null = null;
  private maxNodes = 0;
  private communityCount = 1;

  /**
   * Upload community assignments to the GPU.
   * Called after Louvain detection in WASM.
   */
  uploadCommunityIds(
    device: GPUDevice,
    assignments: Uint32Array,
    communityCount: number,
  ): void {
    if (!this.communityIdsBuffer) {
      throw new Error("Community layout buffers not initialized");
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
      label: "Community: Modulated Repulsion Shader",
      code: REPULSION_COMMUNITY_WGSL,
    });
    const clearModule = device.createShaderModule({
      label: "Community: Clear Centroids Shader",
      code: CLUSTER_CLEAR_WGSL,
    });
    const accumModule = device.createShaderModule({
      label: "Community: Accumulate Centroids Shader",
      code: CLUSTER_ACCUMULATE_WGSL,
    });
    const attractModule = device.createShaderModule({
      label: "Community: Cluster Attract Shader",
      code: CLUSTER_ATTRACT_WGSL,
    });

    return {
      repulsion: device.createComputePipeline({
        label: "Community: Modulated Repulsion Pipeline",
        layout: "auto",
        compute: { module: repulsionModule, entryPoint: "main" },
      }),
      clearCentroids: device.createComputePipeline({
        label: "Community: Clear Centroids Pipeline",
        layout: "auto",
        compute: { module: clearModule, entryPoint: "main" },
      }),
      accumulate: device.createComputePipeline({
        label: "Community: Accumulate Centroids Pipeline",
        layout: "auto",
        compute: { module: accumModule, entryPoint: "main" },
      }),
      attract: device.createComputePipeline({
        label: "Community: Cluster Attract Pipeline",
        layout: "auto",
        compute: { module: attractModule, entryPoint: "main" },
      }),
    };
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    this.maxNodes = maxNodes;

    // 4 uniform buffers (16 bytes each)
    this.repulsionUniforms = device.createBuffer({
      label: "Community: Repulsion Uniforms",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.clearUniforms = device.createBuffer({
      label: "Community: Clear Uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.accumUniforms = device.createBuffer({
      label: "Community: Accum Uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.attractUniforms = device.createBuffer({
      label: "Community: Attract Uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Community ID per node (u32)
    this.communityIdsBuffer = device.createBuffer({
      label: "Community: IDs",
      size: maxNodes * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Centroid accumulators — sized to maxNodes (worst case: every node is its own community)
    this.centroidSumXBuffer = device.createBuffer({
      label: "Community: Centroid Sum X",
      size: maxNodes * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.centroidSumYBuffer = device.createBuffer({
      label: "Community: Centroid Sum Y",
      size: maxNodes * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.centroidCountBuffer = device.createBuffer({
      label: "Community: Centroid Count",
      size: maxNodes * 4,
      usage: GPUBufferUsage.STORAGE,
    });

    // Node degree buffer for degree-weighted repulsion (FA2 core innovation)
    this.degreesBuffer = device.createBuffer({
      label: "Community: Node Degrees",
      size: maxNodes * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new CommunityLayoutBuffers(
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
      throw new Error("Community layout buffers not initialized");
    }

    // Degree-weighted modulated repulsion: uniforms, positions, forces, communityIds, degrees
    const repulsion = device.createBindGroup({
      label: "Community: Repulsion Bind Group",
      layout: pipelines.repulsion.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.repulsionUniforms } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: this.communityIdsBuffer } },
        { binding: 4, resource: { buffer: this.degreesBuffer } },
      ],
    });

    // Clear centroids: uniforms, centroidSumX, centroidSumY, centroidCount
    const clearCentroids = device.createBindGroup({
      label: "Community: Clear Centroids Bind Group",
      layout: pipelines.clearCentroids.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.clearUniforms } },
        { binding: 1, resource: { buffer: this.centroidSumXBuffer } },
        { binding: 2, resource: { buffer: this.centroidSumYBuffer } },
        { binding: 3, resource: { buffer: this.centroidCountBuffer } },
      ],
    });

    // Accumulate centroids: uniforms, positions, communityIds, centroidSumX/Y, centroidCount
    const accumulate = device.createBindGroup({
      label: "Community: Accumulate Bind Group",
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

    // Cluster attraction: uniforms, positions, forces, communityIds, centroidSumX/Y, centroidCount, degrees
    const attract = device.createBindGroup({
      label: "Community: Attract Bind Group",
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
    //   communityNodeSpacing  → intra_factor (within-community repulsion scale)
    //   communitySpacing      → inter_factor (between-community repulsion scale)
    //   communitySpreadFactor → global repulsion multiplier
    //   centerStrength        → distance-independent gravity
    if (this.repulsionUniforms) {
      const cfg = context.forceConfig;
      const data = new ArrayBuffer(32);
      const view = new DataView(data);
      view.setUint32(0, context.nodeCount, true);
      view.setFloat32(4, Math.abs(cfg.repulsionStrength) * cfg.communitySpreadFactor, true);
      view.setFloat32(8, cfg.repulsionDistanceMin, true);
      view.setFloat32(12, cfg.communityNodeSpacing * 0.1, true);  // Node Spacing slider: 10 → 1.0
      view.setFloat32(16, cfg.communitySpacing * 0.02, true);     // Cluster Spacing slider: 50 → 1.0
      view.setFloat32(20, cfg.centerStrength * 100, true);        // Center Gravity slider: 0.01 → 1.0
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
      view.setFloat32(4, context.forceConfig.communityStiffness, true);
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
      const pass = encoder.beginComputePass({ label: "Community: Clear Centroids" });
      pass.setPipeline(pipelines.clearCentroids);
      pass.setBindGroup(0, bindGroups.clearCentroids);
      pass.dispatchWorkgroups(commWorkgroups);
      pass.end();
    }

    // Phase 2: Community-modulated N² repulsion
    {
      const pass = encoder.beginComputePass({ label: "Community: Modulated Repulsion" });
      pass.setPipeline(pipelines.repulsion);
      pass.setBindGroup(0, bindGroups.repulsion);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }

    // Phase 3: Accumulate community centroids
    {
      const pass = encoder.beginComputePass({ label: "Community: Accumulate Centroids" });
      pass.setPipeline(pipelines.accumulate);
      pass.setBindGroup(0, bindGroups.accumulate);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }

    // Phase 4: Apply cluster attraction toward centroids
    {
      const pass = encoder.beginComputePass({ label: "Community: Cluster Attraction" });
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
    // Owned buffers destroyed via CommunityLayoutBuffers.destroy()
    this.communityIdsBuffer = null;
    this.centroidSumXBuffer = null;
    this.centroidSumYBuffer = null;
    this.centroidCountBuffer = null;
    this.degreesBuffer = null;
  }
}

/**
 * Create Community Layout algorithm instance
 */
export function createCommunityLayoutAlgorithm(): CommunityLayoutAlgorithm {
  return new CommunityLayoutAlgorithm();
}
