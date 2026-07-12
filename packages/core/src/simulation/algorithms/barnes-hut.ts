/**
 * Barnes-Hut Force Algorithm - Karras Binary Radix Tree Implementation
 *
 * O(N log N) approximation using parallel binary radix tree construction.
 * Implements Karras 2012 algorithm for GPU-parallel tree building.
 *
 * Pipeline:
 * 1. Compute bounding box (existing bounds or fixed)
 * 2. Generate Morton codes for spatial locality
 * 3. Radix sort particles by Morton code
 * 4. Build binary radix tree topology (Karras algorithm) - O(N)
 * 5. Initialize leaf nodes with particle data - O(N)
 * 6. Aggregate centers of mass bottom-up - O(N)
 * 7. Traverse tree for force computation - O(N log N)
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
  createRadixSortBindGroups,
  createRadixSortBuffers,
  createRadixSortPipeline,
  destroyRadixSortBuffers,
  recordRadixSort,
  updateRadixSortUniforms,
  wasSimpleSort,
} from "../utils/radix_sort.ts";
import type {
  RadixSortBindGroups,
  RadixSortBuffers,
  RadixSortPipeline,
} from "../utils/radix_sort.ts";

// Import shader sources
import MORTON_WGSL from "../shaders/morton.comp.wgsl";
import KARRAS_TREE_WGSL from "../shaders/karras_tree.comp.wgsl";
import TRAVERSE_WGSL from "../shaders/barnes_hut_binary.comp.wgsl";

/**
 * Barnes-Hut algorithm info
 */
const BARNES_HUT_ALGORITHM_INFO: ForceAlgorithmInfo = {
  id: "barnes-hut",
  name: "Barnes-Hut (Karras)",
  description: "GPU binary radix tree with O(N log N) construction. Optimal for 5K-100K+ nodes.",
  minNodes: 100,
  maxNodes: 131072, // Limited by radix sort prefix scan (512 workgroups * 256 threads)
  complexity: "O(N log N)",
};

// Configuration constants
const WORKGROUP_SIZE = 256;

/**
 * Maximum bottom-up aggregation dispatches per frame. Aggregation is
 * multi-pass because WGSL provides no cross-workgroup visibility for plain
 * stores within one dispatch (see karras_tree.comp.wgsl). With the in-pass
 * upward walk, ceil(log2(N)) passes provably complete the tree; for the
 * 131,072-node ceiling that is 17, so 24 leaves generous margin.
 */
const MAX_AGGREGATION_PASSES = 24;

/** Bytes per TreeUniforms record (must match karras_tree.comp.wgsl). */
const TREE_UNIFORMS_SIZE = 32;

/**
 * Records in the tree-uniforms staging buffer: record 0 (aggregation_pass = 0)
 * serves the setup phases (clear/topology/leaves); record i >= 1 carries
 * aggregation_pass = i + 1 for the i-th aggregation dispatch (leaves are
 * stamped 1, so aggregate passes count from 2).
 */
const TREE_UNIFORMS_RECORDS = MAX_AGGREGATION_PASSES + 1;

/** Number of aggregation dispatches needed for nodeCount particles. */
function aggregationPassCount(nodeCount: number): number {
  return Math.min(
    MAX_AGGREGATION_PASSES,
    Math.max(1, Math.ceil(Math.log2(Math.max(nodeCount, 2))) + 1),
  );
}

/**
 * Extended pipelines for Barnes-Hut Karras implementation
 */
interface BarnesHutPipelines extends AlgorithmPipelines {
  // Morton code generation
  morton: GPUComputePipeline;

  // Radix sort (shared utility)
  sort: RadixSortPipeline;

  // Karras tree construction
  clearTree: GPUComputePipeline;
  buildTopology: GPUComputePipeline;
  initLeaves: GPUComputePipeline;
  aggregatePass: GPUComputePipeline;

  // Bind group layouts
  mortonLayout: GPUBindGroupLayout;
  treeLayout: GPUBindGroupLayout;
  traverseLayout: GPUBindGroupLayout;
}

/**
 * Barnes-Hut algorithm-specific buffers
 */
class BarnesHutBuffers implements AlgorithmBuffers {
  constructor(
    // Uniform buffers
    public mortonUniforms: GPUBuffer,
    public treeUniforms: GPUBuffer,
    /** Per-pass TreeUniforms records, copied into treeUniforms mid-encoder */
    public treeUniformsStaging: GPUBuffer,
    public traverseUniforms: GPUBuffer,
    // Shared radix sort buffers (keys = morton codes, values = node indices)
    public sortBuffers: RadixSortBuffers,
    // Tree structure (N-1 internal nodes)
    public leftChild: GPUBuffer,
    public rightChild: GPUBuffer,
    public parent: GPUBuffer,
    // Node properties (2N-1 total: internal + leaves)
    public nodeCom: GPUBuffer, // vec2<f32> per node (center of mass)
    public nodeMass: GPUBuffer,
    public nodeSize: GPUBuffer,
    // Aggregation readiness stamps (atomic u32 per tree node)
    public visitCount: GPUBuffer,
    // Zeroed (= all alive) flags used when the context provides none
    public fallbackNodeFlags: GPUBuffer,
    // Maximum node count this buffer set supports
    public maxNodes: number,
  ) {}

  destroy(): void {
    this.mortonUniforms.destroy();
    this.treeUniforms.destroy();
    this.treeUniformsStaging.destroy();
    this.traverseUniforms.destroy();

    destroyRadixSortBuffers(this.sortBuffers);

    this.leftChild.destroy();
    this.rightChild.destroy();
    this.parent.destroy();

    this.nodeCom.destroy();
    this.nodeMass.destroy();
    this.nodeSize.destroy();

    this.visitCount.destroy();
    this.fallbackNodeFlags.destroy();
  }
}

/**
 * Extended bind groups for Barnes-Hut
 */
interface BarnesHutBindGroups extends AlgorithmBindGroups {
  morton: GPUBindGroup;
  sort: RadixSortBindGroups;
  tree: GPUBindGroup; // For full radix sort (reads from keysA/valuesA after 8 passes)
  treeSimpleSort: GPUBindGroup; // For simple sort (reads from keysB/valuesB after 1 pass)
  // repulsion from base interface is for traversal

  // Buffer reference for radix sort recording (copyBufferToBuffer)
  sortBuffers: RadixSortBuffers;

  // Buffer references for per-pass tree uniform copies during recording
  treeUniforms: GPUBuffer;
  treeUniformsStaging: GPUBuffer;
}

/**
 * Barnes-Hut repulsion algorithm using Karras binary radix tree
 */
export class BarnesHutForceAlgorithm implements ForceAlgorithm {
  readonly info = BARNES_HUT_ALGORITHM_INFO;
  readonly handlesGravity = false;

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

    // Create shader modules
    const mortonModule = device.createShaderModule({
      label: "Morton Code Shader",
      code: MORTON_WGSL,
    });

    const karrasTreeModule = device.createShaderModule({
      label: "Karras Tree Shader",
      code: KARRAS_TREE_WGSL,
    });

    const traverseModule = device.createShaderModule({
      label: "Barnes-Hut Binary Traversal Shader",
      code: TRAVERSE_WGSL,
    });

    // Shared radix sort pipelines
    const sort = createRadixSortPipeline(device, "BH");

    // === Morton Code Layout ===
    const mortonLayout = device.createBindGroupLayout({
      label: "Morton Code Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    // === Karras Tree Layout ===
    const treeLayout = device.createBindGroupLayout({
      label: "Karras Tree Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // === Traversal Layout ===
    const traverseLayout = device.createBindGroupLayout({
      label: "Barnes-Hut Traversal Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    // Create pipeline layouts
    const mortonPipelineLayout = device.createPipelineLayout({
      label: "Morton Pipeline Layout",
      bindGroupLayouts: [mortonLayout],
    });

    const treePipelineLayout = device.createPipelineLayout({
      label: "Karras Tree Pipeline Layout",
      bindGroupLayouts: [treeLayout],
    });

    const traversePipelineLayout = device.createPipelineLayout({
      label: "Barnes-Hut Traversal Pipeline Layout",
      bindGroupLayouts: [traverseLayout],
    });

    const pipelines: BarnesHutPipelines = {
      morton: device.createComputePipeline({
        label: "Morton Code Pipeline",
        layout: mortonPipelineLayout,
        compute: { module: mortonModule, entryPoint: "main" },
      }),

      sort,

      clearTree: device.createComputePipeline({
        label: "Clear Tree Pipeline",
        layout: treePipelineLayout,
        compute: { module: karrasTreeModule, entryPoint: "clear_tree" },
      }),
      buildTopology: device.createComputePipeline({
        label: "Build Topology Pipeline",
        layout: treePipelineLayout,
        compute: { module: karrasTreeModule, entryPoint: "build_topology" },
      }),
      initLeaves: device.createComputePipeline({
        label: "Init Leaves Pipeline",
        layout: treePipelineLayout,
        compute: { module: karrasTreeModule, entryPoint: "init_leaves" },
      }),
      aggregatePass: device.createComputePipeline({
        label: "Aggregate Pass Pipeline",
        layout: treePipelineLayout,
        compute: { module: karrasTreeModule, entryPoint: "aggregate_pass" },
      }),

      repulsion: device.createComputePipeline({
        label: "Barnes-Hut Traversal Pipeline",
        layout: traversePipelineLayout,
        compute: { module: traverseModule, entryPoint: "main" },
      }),

      mortonLayout,
      treeLayout,
      traverseLayout,
    };

    return pipelines;
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    const safeMaxNodes = Math.max(maxNodes, 4);

    const treeNodeBytes = (2 * safeMaxNodes - 1) * 4;
    const internalNodeBytes = Math.max((safeMaxNodes - 1) * 4, 4);

    // Uniform buffers
    const mortonUniforms = device.createBuffer({
      label: "BH Morton Uniforms",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const treeUniforms = device.createBuffer({
      label: "BH Tree Uniforms",
      size: TREE_UNIFORMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Per-pass TreeUniforms records; copyBufferToBuffer is the only way to
    // change a uniform between dispatches recorded in one command encoder.
    const treeUniformsStaging = device.createBuffer({
      label: "BH Tree Uniforms Staging",
      size: TREE_UNIFORMS_RECORDS * TREE_UNIFORMS_SIZE,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const traverseUniforms = device.createBuffer({
      label: "BH Traverse Uniforms",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Shared radix sort buffers (keys = morton codes, values = node indices)
    const sortBuffers = createRadixSortBuffers(device, safeMaxNodes, "BH");

    // Tree structure buffers (N-1 internal nodes)
    const leftChild = device.createBuffer({
      label: "BH Left Child",
      size: internalNodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const rightChild = device.createBuffer({
      label: "BH Right Child",
      size: internalNodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const parent = device.createBuffer({
      label: "BH Parent",
      size: treeNodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Node properties (2N-1 total)
    const nodeCom = device.createBuffer({
      label: "BH Node CoM",
      size: treeNodeBytes * 2, // vec2<f32> = 8 bytes per node
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const nodeMass = device.createBuffer({
      label: "BH Node Mass",
      size: treeNodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const nodeSize = device.createBuffer({
      label: "BH Node Size",
      size: treeNodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Aggregation readiness stamps: one atomic u32 per tree node (2N-1)
    const visitCount = device.createBuffer({
      label: "BH Aggregation Stamps",
      size: treeNodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Zero-initialized (all alive) node flags for contexts without a
    // nodeFlags buffer (GPU buffers are created zeroed per WebGPU spec)
    const fallbackNodeFlags = device.createBuffer({
      label: "BH Fallback Node Flags",
      size: safeMaxNodes * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new BarnesHutBuffers(
      mortonUniforms,
      treeUniforms,
      treeUniformsStaging,
      traverseUniforms,
      sortBuffers,
      leftChild,
      rightChild,
      parent,
      nodeCom,
      nodeMass,
      nodeSize,
      visitCount,
      fallbackNodeFlags,
      safeMaxNodes,
    );
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    const p = pipelines as BarnesHutPipelines;
    const b = algorithmBuffers as BarnesHutBuffers;

    // Dead-slot masking: dead slots must neither seed leaves (morton) nor
    // receive traversal forces. Fall back to the zeroed all-alive buffer.
    const nodeFlags = context.nodeFlags ?? b.fallbackNodeFlags;

    // Morton code bind group — writes keysA (morton codes) and valuesA (node indices)
    const morton = device.createBindGroup({
      label: "BH Morton Bind Group",
      layout: p.mortonLayout,
      entries: [
        { binding: 0, resource: { buffer: b.mortonUniforms } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: b.sortBuffers.keysA } },
        { binding: 3, resource: { buffer: b.sortBuffers.valuesA } },
        { binding: 4, resource: { buffer: nodeFlags } },
      ],
    });

    // Shared radix sort bind groups
    const sort = createRadixSortBindGroups(device, p.sort, b.sortBuffers);

    // After 8 passes (even number), results are in keysA/valuesA
    const tree = device.createBindGroup({
      label: "BH Tree Bind Group (Full Sort)",
      layout: p.treeLayout,
      entries: [
        { binding: 0, resource: { buffer: b.treeUniforms } },
        { binding: 1, resource: { buffer: b.sortBuffers.keysA } },
        { binding: 2, resource: { buffer: b.sortBuffers.valuesA } },
        { binding: 3, resource: { buffer: context.positions } },
        { binding: 4, resource: { buffer: b.leftChild } },
        { binding: 5, resource: { buffer: b.rightChild } },
        { binding: 6, resource: { buffer: b.parent } },
        { binding: 7, resource: { buffer: b.nodeCom } },
        { binding: 8, resource: { buffer: b.nodeMass } },
        { binding: 9, resource: { buffer: b.nodeSize } },
        { binding: 10, resource: { buffer: b.visitCount } },
      ],
    });

    // For simple sort (1 pass), results are in keysB/valuesB
    const treeSimpleSort = device.createBindGroup({
      label: "BH Tree Bind Group (Simple Sort)",
      layout: p.treeLayout,
      entries: [
        { binding: 0, resource: { buffer: b.treeUniforms } },
        { binding: 1, resource: { buffer: b.sortBuffers.keysB } },
        { binding: 2, resource: { buffer: b.sortBuffers.valuesB } },
        { binding: 3, resource: { buffer: context.positions } },
        { binding: 4, resource: { buffer: b.leftChild } },
        { binding: 5, resource: { buffer: b.rightChild } },
        { binding: 6, resource: { buffer: b.parent } },
        { binding: 7, resource: { buffer: b.nodeCom } },
        { binding: 8, resource: { buffer: b.nodeMass } },
        { binding: 9, resource: { buffer: b.nodeSize } },
        { binding: 10, resource: { buffer: b.visitCount } },
      ],
    });

    // Traversal bind group
    const repulsion = device.createBindGroup({
      label: "BH Traversal Bind Group",
      layout: p.traverseLayout,
      entries: [
        { binding: 0, resource: { buffer: b.traverseUniforms } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: b.leftChild } },
        { binding: 4, resource: { buffer: b.rightChild } },
        { binding: 5, resource: { buffer: b.nodeCom } },
        { binding: 6, resource: { buffer: b.nodeMass } },
        { binding: 7, resource: { buffer: b.nodeSize } },
        { binding: 8, resource: { buffer: nodeFlags } },
      ],
    });

    const bindGroups: BarnesHutBindGroups = {
      morton,
      sort,
      tree,
      treeSimpleSort,
      repulsion,
      sortBuffers: b.sortBuffers,
      treeUniforms: b.treeUniforms,
      treeUniformsStaging: b.treeUniformsStaging,
    };

    return bindGroups;
  }

  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void {
    const b = algorithmBuffers as BarnesHutBuffers;

    // CRITICAL: Validate node count doesn't exceed buffer capacity.
    // Buffer overflow from undersized buffers is a security issue that can corrupt
    // GPU memory, cause crashes, or produce undefined behavior.
    if (context.nodeCount > b.maxNodes) {
      throw new Error(
        `Barnes-Hut buffer overflow: nodeCount (${context.nodeCount}) exceeds buffer capacity (${b.maxNodes}). ` +
          `Buffers must be recreated with createBuffers() when node count increases.`,
      );
    }

    // Barnes-Hut REQUIRES bounds for correct Morton code computation.
    // Without proper bounds, Morton codes cannot map positions to the [0,1] normalized
    // space, causing spatial tree degeneration and incorrect force calculations.
    // The caller MUST provide bounds computed from actual node positions.
    if (!context.bounds) {
      throw new Error(
        "Barnes-Hut algorithm requires bounds to be provided in AlgorithmRenderContext. " +
          "Bounds must be computed from actual node positions. Without bounds, Morton codes " +
          "cannot correctly encode spatial locality, causing tree degeneration and incorrect forces.",
      );
    }

    const boundsMinX = context.bounds.minX;
    const boundsMinY = context.bounds.minY;
    const boundsMaxX = context.bounds.maxX;
    const boundsMaxY = context.bounds.maxY;
    const rootSize = Math.max(boundsMaxX - boundsMinX, boundsMaxY - boundsMinY);

    // Morton uniforms (48 bytes due to vec3 alignment)
    // struct SimulationUniforms { bounds_min: vec2<f32>, bounds_max: vec2<f32>, node_count: u32, _padding: vec3<u32> }
    // Layout: bounds_min at 0, bounds_max at 8, node_count at 16, _padding at 32 (16-byte aligned)
    const mortonData = new ArrayBuffer(48);
    const mortonView = new DataView(mortonData);
    mortonView.setFloat32(0, boundsMinX, true); // bounds_min.x
    mortonView.setFloat32(4, boundsMinY, true); // bounds_min.y
    mortonView.setFloat32(8, boundsMaxX, true); // bounds_max.x
    mortonView.setFloat32(12, boundsMaxY, true); // bounds_max.y
    mortonView.setUint32(16, context.nodeCount, true); // node_count
    // Implicit padding at offsets 20, 24, 28 to align vec3 at offset 32
    mortonView.setUint32(20, 0, true);
    mortonView.setUint32(24, 0, true);
    mortonView.setUint32(28, 0, true);
    // _padding vec3<u32> at offset 32 (16-byte aligned)
    mortonView.setUint32(32, 0, true); // _padding[0]
    mortonView.setUint32(36, 0, true); // _padding[1]
    mortonView.setUint32(40, 0, true); // _padding[2]
    // Final padding to 48 bytes
    mortonView.setUint32(44, 0, true);
    device.queue.writeBuffer(b.mortonUniforms, 0, mortonData);

    // Tree uniforms staging: one TreeUniforms record per dispatch that needs
    // a distinct aggregation_pass value. Record 0 (aggregation_pass = 0) is
    // for clear/topology/leaves; record i >= 1 is for the i-th aggregation
    // dispatch (aggregation_pass = i + 1; leaves are stamped 1).
    // struct TreeUniforms { node_count: u32, bounds_min_x: f32, bounds_min_y: f32, bounds_max_x: f32,
    //                       bounds_max_y: f32, root_size: f32, aggregation_pass: u32, _padding: u32 }
    const treeData = new ArrayBuffer(TREE_UNIFORMS_RECORDS * TREE_UNIFORMS_SIZE);
    const treeView = new DataView(treeData);
    for (let rec = 0; rec < TREE_UNIFORMS_RECORDS; rec++) {
      const base = rec * TREE_UNIFORMS_SIZE;
      treeView.setUint32(base + 0, context.nodeCount, true);
      treeView.setFloat32(base + 4, boundsMinX, true);
      treeView.setFloat32(base + 8, boundsMinY, true);
      treeView.setFloat32(base + 12, boundsMaxX, true);
      treeView.setFloat32(base + 16, boundsMaxY, true);
      treeView.setFloat32(base + 20, rootSize, true);
      treeView.setUint32(base + 24, rec === 0 ? 0 : rec + 1, true); // aggregation_pass
      treeView.setUint32(base + 28, 0, true);
    }
    device.queue.writeBuffer(b.treeUniformsStaging, 0, treeData);
    // Seed the live uniform with record 0 for the setup phases; the
    // aggregation loop overwrites it per dispatch via copyBufferToBuffer.
    device.queue.writeBuffer(b.treeUniforms, 0, treeData, 0, TREE_UNIFORMS_SIZE);

    // Traverse uniforms (32 bytes)
    // struct ForceUniforms { particle_count: u32, repulsion_strength: f32, theta: f32, min_distance: f32,
    //                        leaf_size: f32, max_distance: f32, _pad2: f32, _pad3: f32 }
    const leafSize = rootSize / 256.0; // Approximate leaf size
    const traverseData = new ArrayBuffer(32);
    const traverseView = new DataView(traverseData);
    traverseView.setUint32(0, context.nodeCount, true);
    traverseView.setFloat32(4, Math.abs(context.forceConfig.repulsionStrength), true);
    traverseView.setFloat32(8, context.forceConfig.theta, true);
    traverseView.setFloat32(12, context.forceConfig.repulsionDistanceMin, true);
    traverseView.setFloat32(16, leafSize, true);
    traverseView.setFloat32(20, context.forceConfig.repulsionDistanceMax, true);
    traverseView.setFloat32(24, 0.0, true);
    traverseView.setFloat32(28, 0.0, true);
    device.queue.writeBuffer(b.traverseUniforms, 0, traverseData);

    // Radix sort uniforms (scan + per-pass staging)
    updateRadixSortUniforms(device, b.sortBuffers, context.nodeCount);
  }

  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: AlgorithmPipelines,
    bindGroups: AlgorithmBindGroups,
    nodeCount: number,
  ): void {
    const p = pipelines as BarnesHutPipelines;
    const bg = bindGroups as BarnesHutBindGroups;

    // Handle degenerate cases
    if (nodeCount < 2) {
      return;
    }

    const nodeWorkgroups = calculateWorkgroups(nodeCount, WORKGROUP_SIZE);
    const internalNodes = nodeCount - 1;
    const internalWorkgroups = calculateWorkgroups(internalNodes, WORKGROUP_SIZE);
    const totalTreeNodes = 2 * nodeCount - 1;
    const treeWorkgroups = calculateWorkgroups(totalTreeNodes, WORKGROUP_SIZE);

    // === PHASE 1: Generate Morton codes ===
    {
      const pass = encoder.beginComputePass({ label: "BH Morton Codes" });
      pass.setPipeline(p.morton);
      pass.setBindGroup(0, bg.morton);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }

    // === PHASE 2: Sort by Morton code ===
    // Uses shared radix sort utility: simple sort for <1024 nodes, full 8-pass LSD for larger.
    // Returns false if workgroup count exceeds prefix scan capacity (~131K nodes).
    const sortSucceeded = recordRadixSort(
      encoder,
      p.sort,
      bg.sort,
      bg.sortBuffers,
      nodeCount,
      "BH",
    );
    if (!sortSucceeded) {
      return;
    }

    // Select the correct tree bind group based on sort method.
    // Simple sort (1 pass) outputs to keysB/valuesB; full radix (8 passes) to keysA/valuesA.
    const treeBindGroup = wasSimpleSort(nodeCount) ? bg.treeSimpleSort : bg.tree;

    // Reset the live tree uniform to record 0 (aggregation_pass = 0) — a
    // previous frame's aggregation loop left it at the last pass number.
    encoder.copyBufferToBuffer(
      bg.treeUniformsStaging,
      0,
      bg.treeUniforms,
      0,
      TREE_UNIFORMS_SIZE,
    );

    // === PHASE 3: Clear tree data ===
    {
      const pass = encoder.beginComputePass({ label: "BH Clear Tree" });
      pass.setPipeline(p.clearTree);
      pass.setBindGroup(0, treeBindGroup);
      pass.dispatchWorkgroups(treeWorkgroups);
      pass.end();
    }

    // === PHASE 4: Build tree topology (Karras algorithm) ===
    {
      const pass = encoder.beginComputePass({ label: "BH Build Topology" });
      pass.setPipeline(p.buildTopology);
      pass.setBindGroup(0, treeBindGroup);
      pass.dispatchWorkgroups(internalWorkgroups);
      pass.end();
    }

    // === PHASE 5: Initialize leaf nodes ===
    {
      const pass = encoder.beginComputePass({ label: "BH Init Leaves" });
      pass.setPipeline(p.initLeaves);
      pass.setBindGroup(0, treeBindGroup);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }

    // === PHASE 6: Aggregate centers of mass bottom-up (multi-pass) ===
    // WGSL provides no cross-workgroup visibility for plain stores within a
    // dispatch, so each dispatch only consumes child data stamped by EARLIER
    // dispatches (inter-dispatch ordering guarantees visibility) or written
    // by the walking thread itself. ceil(log2(N)) + 1 dispatches provably
    // complete the tree (see karras_tree.comp.wgsl).
    const aggregationPasses = aggregationPassCount(nodeCount);
    for (let i = 1; i <= aggregationPasses; i++) {
      // Load this dispatch's aggregation_pass value (record i => pass i + 1)
      encoder.copyBufferToBuffer(
        bg.treeUniformsStaging,
        i * TREE_UNIFORMS_SIZE,
        bg.treeUniforms,
        0,
        TREE_UNIFORMS_SIZE,
      );
      const pass = encoder.beginComputePass({ label: `BH Aggregate Pass ${i}` });
      pass.setPipeline(p.aggregatePass);
      pass.setBindGroup(0, treeBindGroup);
      pass.dispatchWorkgroups(internalWorkgroups);
      pass.end();
    }

    // === PHASE 7: Tree traversal for force computation ===
    {
      const pass = encoder.beginComputePass({ label: "BH Traversal" });
      pass.setPipeline(p.repulsion);
      pass.setBindGroup(0, bg.repulsion);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }
  }

  destroy(): void {
    // Buffers are destroyed via AlgorithmBuffers.destroy()
  }
}

/**
 * Create Barnes-Hut force algorithm instance
 */
export function createBarnesHutAlgorithm(): ForceAlgorithm {
  return new BarnesHutForceAlgorithm();
}
