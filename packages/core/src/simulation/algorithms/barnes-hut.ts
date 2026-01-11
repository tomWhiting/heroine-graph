/**
 * Barnes-Hut Force Algorithm - Full GPU Implementation
 *
 * O(n log n) approximation using quadtree spatial partitioning.
 * The entire pipeline runs on GPU:
 * 1. Clear tree and set cell sizes
 * 2. Compute leaf cell centers of mass
 * 3. Build internal nodes bottom-up
 * 4. Traverse tree for force computation
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

// Import shader sources
import BUILD_WGSL from "../shaders/quadtree_build.comp.wgsl";
import TRAVERSE_WGSL from "../shaders/barnes_hut.comp.wgsl";

/**
 * Barnes-Hut algorithm info
 */
const BARNES_HUT_ALGORITHM_INFO: ForceAlgorithmInfo = {
  id: "barnes-hut",
  name: "Barnes-Hut",
  description:
    "GPU quadtree-based approximation. Best for medium graphs (5K-50K nodes).",
  minNodes: 100,
  maxNodes: 50000,
  complexity: "O(n log n)",
};

// Tree configuration
const MAX_TREE_NODES = 262144; // Maximum tree nodes
const WORKGROUP_SIZE = 256;
// Tree depth controls resolution vs performance
// Note: Current leaf CoM computation is O(n * leaf_cells), so deeper trees are slower
// Depth 5 = 256 leaves, good balance for up to ~50K nodes
// TODO: Implement proper Morton-based tree building for O(n log n) at any scale
const TREE_DEPTH = 5;

/**
 * Extended pipelines for Barnes-Hut
 */
interface BarnesHutPipelines extends AlgorithmPipelines {
  clearTree: GPUComputePipeline;
  computeLeafCom: GPUComputePipeline;
  buildLevel: GPUComputePipeline;
  // 'repulsion' from base interface is the traversal

  // Shared bind group layouts
  buildBindGroupLayout: GPUBindGroupLayout;
  traverseBindGroupLayout: GPUBindGroupLayout;
}

/**
 * Barnes-Hut algorithm-specific buffers
 */
class BarnesHutBuffers implements AlgorithmBuffers {
  constructor(
    // Uniform buffers
    public buildUniforms: GPUBuffer,
    public traverseUniforms: GPUBuffer,
    // Tree structure
    public treeComX: GPUBuffer,
    public treeComY: GPUBuffer,
    public treeMass: GPUBuffer,
    public treeSizes: GPUBuffer,
    public treeCount: GPUBuffer,
  ) {}

  destroy(): void {
    this.buildUniforms.destroy();
    this.traverseUniforms.destroy();
    this.treeComX.destroy();
    this.treeComY.destroy();
    this.treeMass.destroy();
    this.treeSizes.destroy();
    this.treeCount.destroy();
  }
}

/**
 * Extended bind groups for Barnes-Hut
 */
interface BarnesHutBindGroups extends AlgorithmBindGroups {
  build: GPUBindGroup;
  // 'repulsion' from base interface is used for traversal
}

/**
 * Barnes-Hut repulsion algorithm - Full GPU implementation
 */
export class BarnesHutForceAlgorithm implements ForceAlgorithm {
  readonly info = BARNES_HUT_ALGORITHM_INFO;

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

    // Create shader modules
    const buildModule = device.createShaderModule({
      label: "Barnes-Hut Build Shader",
      code: BUILD_WGSL,
    });

    const traverseModule = device.createShaderModule({
      label: "Barnes-Hut Traverse Shader",
      code: TRAVERSE_WGSL,
    });

    // Create explicit bind group layout for build pipelines
    // Bindings: uniforms, positions_x, positions_y, tree_com_x, tree_com_y, tree_mass, tree_sizes, tree_count
    const buildBindGroupLayout = device.createBindGroupLayout({
      label: "Barnes-Hut Build Bind Group Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // Create explicit bind group layout for traversal
    // Bindings: uniforms, positions_x, positions_y, forces_x, forces_y, tree_com_x, tree_com_y, tree_mass, tree_sizes
    const traverseBindGroupLayout = device.createBindGroupLayout({
      label: "Barnes-Hut Traverse Bind Group Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const buildPipelineLayout = device.createPipelineLayout({
      label: "Barnes-Hut Build Pipeline Layout",
      bindGroupLayouts: [buildBindGroupLayout],
    });

    const traversePipelineLayout = device.createPipelineLayout({
      label: "Barnes-Hut Traverse Pipeline Layout",
      bindGroupLayouts: [traverseBindGroupLayout],
    });

    const pipelines: BarnesHutPipelines = {
      clearTree: device.createComputePipeline({
        label: "Barnes-Hut Clear Tree",
        layout: buildPipelineLayout,
        compute: { module: buildModule, entryPoint: "clear_tree" },
      }),
      computeLeafCom: device.createComputePipeline({
        label: "Barnes-Hut Compute Leaf CoM",
        layout: buildPipelineLayout,
        compute: { module: buildModule, entryPoint: "compute_leaf_com" },
      }),
      buildLevel: device.createComputePipeline({
        label: "Barnes-Hut Build Level",
        layout: buildPipelineLayout,
        compute: { module: buildModule, entryPoint: "build_level" },
      }),
      repulsion: device.createComputePipeline({
        label: "Barnes-Hut Traversal",
        layout: traversePipelineLayout,
        compute: { module: traverseModule, entryPoint: "main" },
      }),
      buildBindGroupLayout,
      traverseBindGroupLayout,
    };

    return pipelines;
  }

  createBuffers(device: GPUDevice, _maxNodes: number): AlgorithmBuffers {
    // Build uniforms: QuadtreeUniforms struct (32 bytes)
    // node_count: u32, max_depth: u32, bounds_min_x: f32, bounds_min_y: f32,
    // bounds_max_x: f32, bounds_max_y: f32, root_size: f32, _padding: u32
    const buildUniforms = device.createBuffer({
      label: "BH Build Uniforms",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Traverse uniforms: ForceUniforms struct (32 bytes = 8 x f32)
    // node_count: u32, repulsion_strength: f32, theta: f32, min_distance: f32,
    // min_cell_size: f32, _pad1: f32, _pad2: f32, _pad3: f32
    const traverseUniforms = device.createBuffer({
      label: "BH Traverse Uniforms",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Tree structure buffers
    const treeComX = device.createBuffer({
      label: "BH Tree CoM X",
      size: MAX_TREE_NODES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const treeComY = device.createBuffer({
      label: "BH Tree CoM Y",
      size: MAX_TREE_NODES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const treeMass = device.createBuffer({
      label: "BH Tree Mass",
      size: MAX_TREE_NODES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const treeSizes = device.createBuffer({
      label: "BH Tree Sizes",
      size: MAX_TREE_NODES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const treeCount = device.createBuffer({
      label: "BH Tree Count",
      size: MAX_TREE_NODES * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const buffers = new BarnesHutBuffers(
      buildUniforms,
      traverseUniforms,
      treeComX,
      treeComY,
      treeMass,
      treeSizes,
      treeCount,
    );
    return buffers;
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    const p = pipelines as BarnesHutPipelines;
    const b = algorithmBuffers as BarnesHutBuffers;

    // Build bind group
    const build = device.createBindGroup({
      label: "BH Build Bind Group",
      layout: p.buildBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: b.buildUniforms } },
        { binding: 1, resource: { buffer: context.positionsX } },
        { binding: 2, resource: { buffer: context.positionsY } },
        { binding: 3, resource: { buffer: b.treeComX } },
        { binding: 4, resource: { buffer: b.treeComY } },
        { binding: 5, resource: { buffer: b.treeMass } },
        { binding: 6, resource: { buffer: b.treeSizes } },
        { binding: 7, resource: { buffer: b.treeCount } },
      ],
    });

    // Traversal bind group
    const repulsion = device.createBindGroup({
      label: "BH Traversal Bind Group",
      layout: p.traverseBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: b.traverseUniforms } },
        { binding: 1, resource: { buffer: context.positionsX } },
        { binding: 2, resource: { buffer: context.positionsY } },
        { binding: 3, resource: { buffer: context.forcesX } },
        { binding: 4, resource: { buffer: context.forcesY } },
        { binding: 5, resource: { buffer: b.treeComX } },
        { binding: 6, resource: { buffer: b.treeComY } },
        { binding: 7, resource: { buffer: b.treeMass } },
        { binding: 8, resource: { buffer: b.treeSizes } },
      ],
    });

    const bindGroups: BarnesHutBindGroups = {
      build,
      repulsion,
    };

    return bindGroups;
  }

  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void {
    const b = algorithmBuffers as BarnesHutBuffers;

    // Use large bounds to accommodate spread-out graphs
    // With 100K nodes at high repulsion, graphs can spread very wide
    // These bounds should be large enough to contain any reasonable layout
    const boundsMin = -5000.0;
    const boundsMax = 5000.0;
    const rootSize = boundsMax - boundsMin;

    // Build uniforms (32 bytes)
    const buildData = new ArrayBuffer(32);
    const buildView = new DataView(buildData);
    buildView.setUint32(0, context.nodeCount, true);     // node_count
    buildView.setUint32(4, TREE_DEPTH, true);            // max_depth
    buildView.setFloat32(8, boundsMin, true);            // bounds_min_x
    buildView.setFloat32(12, boundsMin, true);           // bounds_min_y
    buildView.setFloat32(16, boundsMax, true);           // bounds_max_x
    buildView.setFloat32(20, boundsMax, true);           // bounds_max_y
    buildView.setFloat32(24, rootSize, true);            // root_size
    buildView.setUint32(28, 0, true);                    // _padding
    device.queue.writeBuffer(b.buildUniforms, 0, buildData);

    // Traverse uniforms (32 bytes)
    // Calculate minimum cell size (leaf level)
    const minCellSize = rootSize / Math.pow(2, TREE_DEPTH - 1);

    const traverseData = new ArrayBuffer(32);
    const traverseView = new DataView(traverseData);
    traverseView.setUint32(0, context.nodeCount, true);
    traverseView.setFloat32(4, Math.abs(context.forceConfig.repulsionStrength), true);
    traverseView.setFloat32(8, context.forceConfig.theta, true);
    traverseView.setFloat32(12, context.forceConfig.repulsionDistanceMin, true);
    traverseView.setFloat32(16, minCellSize, true);  // min_cell_size
    // _padding: vec3<f32> at bytes 20-31 (initialized to 0 by ArrayBuffer)
    device.queue.writeBuffer(b.traverseUniforms, 0, traverseData);
  }

  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: AlgorithmPipelines,
    bindGroups: AlgorithmBindGroups,
    nodeCount: number,
  ): void {
    const p = pipelines as BarnesHutPipelines;
    const bg = bindGroups as BarnesHutBindGroups;

    const nodeWorkgroups = calculateWorkgroups(nodeCount, WORKGROUP_SIZE);
    const treeWorkgroups = calculateWorkgroups(MAX_TREE_NODES, WORKGROUP_SIZE);

    // Number of leaf cells at depth TREE_DEPTH-1: 4^(TREE_DEPTH-1)
    const numLeaves = Math.pow(4, TREE_DEPTH - 1);
    const leafWorkgroups = calculateWorkgroups(numLeaves, WORKGROUP_SIZE);

    // === PHASE 1: Clear tree and initialize sizes ===
    {
      const pass = encoder.beginComputePass({ label: "BH Clear Tree" });
      pass.setPipeline(p.clearTree);
      pass.setBindGroup(0, bg.build);
      pass.dispatchWorkgroups(treeWorkgroups);
      pass.end();
    }

    // === PHASE 2: Compute leaf cell centers of mass ===
    {
      const pass = encoder.beginComputePass({ label: "BH Compute Leaf CoM" });
      pass.setPipeline(p.computeLeafCom);
      pass.setBindGroup(0, bg.build);
      pass.dispatchWorkgroups(leafWorkgroups);
      pass.end();
    }

    // === PHASE 3: Build internal nodes bottom-up (level by level) ===
    // Must go from level (TREE_DEPTH-2) down to level 0
    // Each level depends on the level below it being complete
    // Level is passed via dispatch z-count (can't update uniforms between GPU passes)
    for (let level = TREE_DEPTH - 2; level >= 0; level--) {
      const levelSize = Math.pow(4, level);
      const levelWorkgroups = calculateWorkgroups(levelSize, WORKGROUP_SIZE);

      const pass = encoder.beginComputePass({ label: `BH Build Level ${level}` });
      pass.setPipeline(p.buildLevel);
      pass.setBindGroup(0, bg.build);
      // Pass level via z-coordinate of dispatch (shader reads num_workgroups.z - 1)
      pass.dispatchWorkgroups(levelWorkgroups, 1, level + 1);
      pass.end();
    }

    // === PHASE 4: Tree traversal for force computation ===
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
