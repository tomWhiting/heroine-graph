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

// Import shader sources
import MORTON_WGSL from "../shaders/morton.comp.wgsl";
import RADIX_SORT_WGSL from "../shaders/radix_sort.comp.wgsl";
import PREFIX_SCAN_WGSL from "../shaders/prefix_scan.comp.wgsl";
import KARRAS_TREE_WGSL from "../shaders/karras_tree.comp.wgsl";
import TRAVERSE_WGSL from "../shaders/barnes_hut_binary.comp.wgsl";

/**
 * Barnes-Hut algorithm info
 */
const BARNES_HUT_ALGORITHM_INFO: ForceAlgorithmInfo = {
  id: "barnes-hut",
  name: "Barnes-Hut (Karras)",
  description:
    "GPU binary radix tree with O(N log N) construction. Optimal for 5K-100K+ nodes.",
  minNodes: 100,
  maxNodes: 500000,
  complexity: "O(N log N)",
};

// Configuration constants
const WORKGROUP_SIZE = 256;
const RADIX_PASSES = 8; // 32-bit keys / 4 bits per pass = 8 passes
const RADIX_SIZE = 16; // 2^4 = 16 buckets per radix digit

/**
 * Extended pipelines for Barnes-Hut Karras implementation
 */
interface BarnesHutPipelines extends AlgorithmPipelines {
  // Morton code generation
  morton: GPUComputePipeline;

  // Radix sort pipelines
  radixClearHistogram: GPUComputePipeline; // Clear histogram before each pass
  radixHistogram: GPUComputePipeline;
  radixScan: GPUComputePipeline;
  radixScatter: GPUComputePipeline;
  radixSimple: GPUComputePipeline; // Fallback for small arrays

  // Karras tree construction
  clearTree: GPUComputePipeline;
  buildTopology: GPUComputePipeline;
  initLeaves: GPUComputePipeline;
  aggregateBottomUp: GPUComputePipeline;

  // Force computation (repulsion from base interface)
  // repulsion: GPUComputePipeline;

  // Bind group layouts for different stages
  mortonLayout: GPUBindGroupLayout;
  sortLayout: GPUBindGroupLayout;
  scanLayout: GPUBindGroupLayout;
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
    public sortUniforms: GPUBuffer,
    public scanUniforms: GPUBuffer,
    public treeUniforms: GPUBuffer,
    public traverseUniforms: GPUBuffer,

    // Staging buffer for per-pass sort uniforms (8 passes * 16 bytes = 128 bytes)
    // This allows us to copy the correct pass_number during command encoding
    public sortUniformsStaging: GPUBuffer,

    // Morton code buffers (ping-pong for sorting)
    public mortonCodes: GPUBuffer,
    public mortonCodesOut: GPUBuffer,
    public nodeIndices: GPUBuffer,
    public nodeIndicesOut: GPUBuffer,

    // Radix sort working buffers
    public histogram: GPUBuffer,
    public prefixSums: GPUBuffer,

    // Tree structure (N-1 internal nodes)
    public leftChild: GPUBuffer,
    public rightChild: GPUBuffer,
    public parent: GPUBuffer,

    // Node properties (2N-1 total: internal + leaves)
    public nodeComX: GPUBuffer,
    public nodeComY: GPUBuffer,
    public nodeMass: GPUBuffer,
    public nodeSize: GPUBuffer,

    // Atomic visit counter for bottom-up aggregation
    public visitCount: GPUBuffer,

    // Maximum node count this buffer set supports
    public maxNodes: number,
  ) {}

  destroy(): void {
    this.mortonUniforms.destroy();
    this.sortUniforms.destroy();
    this.scanUniforms.destroy();
    this.treeUniforms.destroy();
    this.traverseUniforms.destroy();
    this.sortUniformsStaging.destroy();

    this.mortonCodes.destroy();
    this.mortonCodesOut.destroy();
    this.nodeIndices.destroy();
    this.nodeIndicesOut.destroy();

    this.histogram.destroy();
    this.prefixSums.destroy();

    this.leftChild.destroy();
    this.rightChild.destroy();
    this.parent.destroy();

    this.nodeComX.destroy();
    this.nodeComY.destroy();
    this.nodeMass.destroy();
    this.nodeSize.destroy();

    this.visitCount.destroy();
  }
}

/**
 * Extended bind groups for Barnes-Hut
 */
interface BarnesHutBindGroups extends AlgorithmBindGroups {
  morton: GPUBindGroup;
  sortPass: GPUBindGroup[];  // One per radix pass (ping-pong)
  scan: GPUBindGroup;
  tree: GPUBindGroup;           // For full radix sort (reads from non-Out buffers after 8 passes)
  treeSimpleSort: GPUBindGroup; // For simple sort (reads from Out buffers after 1 pass)
  // repulsion from base interface is for traversal

  // Buffer references needed for command encoding (copyBufferToBuffer)
  // These enable per-pass uniform updates during radix sort
  sortUniforms: GPUBuffer;
  sortUniformsStaging: GPUBuffer;
}

/**
 * Barnes-Hut repulsion algorithm using Karras binary radix tree
 */
export class BarnesHutForceAlgorithm implements ForceAlgorithm {
  readonly info = BARNES_HUT_ALGORITHM_INFO;

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

    // Create shader modules
    const mortonModule = device.createShaderModule({
      label: "Morton Code Shader",
      code: MORTON_WGSL,
    });

    const radixSortModule = device.createShaderModule({
      label: "Radix Sort Shader",
      code: RADIX_SORT_WGSL,
    });

    const prefixScanModule = device.createShaderModule({
      label: "Prefix Scan Shader",
      code: PREFIX_SCAN_WGSL,
    });

    const karrasTreeModule = device.createShaderModule({
      label: "Karras Tree Shader",
      code: KARRAS_TREE_WGSL,
    });

    const traverseModule = device.createShaderModule({
      label: "Barnes-Hut Binary Traversal Shader",
      code: TRAVERSE_WGSL,
    });

    // === Morton Code Layout ===
    // Bindings: uniforms, positions (vec2), morton_codes, node_indices
    const mortonLayout = device.createBindGroupLayout({
      label: "Morton Code Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // === Radix Sort Layout ===
    // Bindings: uniforms, keys_in, values_in, keys_out, values_out, histogram, prefix_sums
    const sortLayout = device.createBindGroupLayout({
      label: "Radix Sort Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    // === Prefix Scan Layout ===
    // Bindings: uniforms, histogram, prefix_sums
    const scanLayout = device.createBindGroupLayout({
      label: "Prefix Scan Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // === Karras Tree Layout ===
    // Bindings: uniforms, morton_codes, sorted_indices, positions (vec2),
    //           left_child, right_child, parent, node_com_x, node_com_y,
    //           node_mass, node_size, visit_count
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
        { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // === Traversal Layout ===
    // Bindings: uniforms, positions (vec2), forces (vec2),
    //           left_child, right_child, node_com_x, node_com_y, node_mass, node_size
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

    const sortPipelineLayout = device.createPipelineLayout({
      label: "Radix Sort Pipeline Layout",
      bindGroupLayouts: [sortLayout],
    });

    const scanPipelineLayout = device.createPipelineLayout({
      label: "Prefix Scan Pipeline Layout",
      bindGroupLayouts: [scanLayout],
    });

    const treePipelineLayout = device.createPipelineLayout({
      label: "Karras Tree Pipeline Layout",
      bindGroupLayouts: [treeLayout],
    });

    const traversePipelineLayout = device.createPipelineLayout({
      label: "Barnes-Hut Traversal Pipeline Layout",
      bindGroupLayouts: [traverseLayout],
    });

    // Create compute pipelines
    const pipelines: BarnesHutPipelines = {
      // Morton code generation
      morton: device.createComputePipeline({
        label: "Morton Code Pipeline",
        layout: mortonPipelineLayout,
        compute: { module: mortonModule, entryPoint: "main" },
      }),

      // Radix sort
      radixClearHistogram: device.createComputePipeline({
        label: "Radix Clear Histogram Pipeline",
        layout: sortPipelineLayout,
        compute: { module: radixSortModule, entryPoint: "clear_histogram" },
      }),
      radixHistogram: device.createComputePipeline({
        label: "Radix Histogram Pipeline",
        layout: sortPipelineLayout,
        compute: { module: radixSortModule, entryPoint: "histogram_count" },
      }),
      radixScan: device.createComputePipeline({
        label: "Radix Scan Pipeline",
        layout: scanPipelineLayout,
        compute: { module: prefixScanModule, entryPoint: "scan_sequential" },
      }),
      radixScatter: device.createComputePipeline({
        label: "Radix Scatter Pipeline",
        layout: sortPipelineLayout,
        compute: { module: radixSortModule, entryPoint: "scatter" },
      }),
      radixSimple: device.createComputePipeline({
        label: "Radix Simple Sort Pipeline",
        layout: sortPipelineLayout,
        compute: { module: radixSortModule, entryPoint: "simple_sort" },
      }),

      // Karras tree construction
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
      aggregateBottomUp: device.createComputePipeline({
        label: "Aggregate Bottom-Up Pipeline",
        layout: treePipelineLayout,
        compute: { module: karrasTreeModule, entryPoint: "aggregate_bottom_up" },
      }),

      // Force computation
      repulsion: device.createComputePipeline({
        label: "Barnes-Hut Traversal Pipeline",
        layout: traversePipelineLayout,
        compute: { module: traverseModule, entryPoint: "main" },
      }),

      // Store layouts for bind group creation
      mortonLayout,
      sortLayout,
      scanLayout,
      treeLayout,
      traverseLayout,
    };

    return pipelines;
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    // Ensure minimum size for all buffers
    const safeMaxNodes = Math.max(maxNodes, 4);

    // Calculate buffer sizes
    const nodeBytes = safeMaxNodes * 4;
    const treeNodeBytes = (2 * safeMaxNodes - 1) * 4; // Internal + leaves
    const internalNodeBytes = Math.max((safeMaxNodes - 1) * 4, 4);

    // Calculate histogram size: workgroups * RADIX_SIZE
    const workgroupCount = calculateWorkgroups(safeMaxNodes, WORKGROUP_SIZE);
    const histogramBytes = workgroupCount * RADIX_SIZE * 4;

    // Uniform buffers
    const mortonUniforms = device.createBuffer({
      label: "BH Morton Uniforms",
      size: 32, // SimulationUniforms: bounds (vec2 x2), node_count, padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const sortUniforms = device.createBuffer({
      label: "BH Sort Uniforms",
      size: 16, // SortUniforms: node_count, pass_number, padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Staging buffer with pre-computed uniforms for all 8 radix passes
    // Each pass needs 16 bytes (node_count: u32, pass_number: u32, _padding: vec2<u32>)
    // Total: 8 passes * 16 bytes = 128 bytes
    // This enables per-pass uniform updates during command encoding via copyBufferToBuffer
    const sortUniformsStaging = device.createBuffer({
      label: "BH Sort Uniforms Staging",
      size: RADIX_PASSES * 16,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const scanUniforms = device.createBuffer({
      label: "BH Scan Uniforms",
      size: 16, // ScanUniforms: element_count, pass_number, workgroup_count, padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const treeUniforms = device.createBuffer({
      label: "BH Tree Uniforms",
      size: 32, // TreeUniforms: node_count, bounds (4 floats), root_size, padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const traverseUniforms = device.createBuffer({
      label: "BH Traverse Uniforms",
      size: 32, // ForceUniforms: particle_count, repulsion_strength, theta, min_distance, leaf_size, padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Morton code buffers (ping-pong for sorting)
    const mortonCodes = device.createBuffer({
      label: "BH Morton Codes",
      size: nodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const mortonCodesOut = device.createBuffer({
      label: "BH Morton Codes Out",
      size: nodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const nodeIndices = device.createBuffer({
      label: "BH Node Indices",
      size: nodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const nodeIndicesOut = device.createBuffer({
      label: "BH Node Indices Out",
      size: nodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Radix sort working buffers
    const histogram = device.createBuffer({
      label: "BH Histogram",
      size: Math.max(histogramBytes, 64), // Minimum size for small arrays
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const prefixSums = device.createBuffer({
      label: "BH Prefix Sums",
      size: Math.max(histogramBytes, 64),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

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
      size: treeNodeBytes, // All nodes need parent reference
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Node properties (2N-1 total)
    const nodeComX = device.createBuffer({
      label: "BH Node CoM X",
      size: treeNodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const nodeComY = device.createBuffer({
      label: "BH Node CoM Y",
      size: treeNodeBytes,
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

    // Atomic visit counter for bottom-up aggregation
    const visitCount = device.createBuffer({
      label: "BH Visit Count",
      size: internalNodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new BarnesHutBuffers(
      mortonUniforms,
      sortUniforms,
      scanUniforms,
      treeUniforms,
      traverseUniforms,
      sortUniformsStaging,
      mortonCodes,
      mortonCodesOut,
      nodeIndices,
      nodeIndicesOut,
      histogram,
      prefixSums,
      leftChild,
      rightChild,
      parent,
      nodeComX,
      nodeComY,
      nodeMass,
      nodeSize,
      visitCount,
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

    // Morton code bind group
    const morton = device.createBindGroup({
      label: "BH Morton Bind Group",
      layout: p.mortonLayout,
      entries: [
        { binding: 0, resource: { buffer: b.mortonUniforms } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: b.mortonCodes } },
        { binding: 3, resource: { buffer: b.nodeIndices } },
      ],
    });

    // Radix sort bind groups (ping-pong between passes)
    // Even passes: mortonCodes -> mortonCodesOut
    // Odd passes: mortonCodesOut -> mortonCodes
    const sortPass: GPUBindGroup[] = [];

    for (let pass = 0; pass < RADIX_PASSES; pass++) {
      const isEven = pass % 2 === 0;
      sortPass.push(
        device.createBindGroup({
          label: `BH Sort Pass ${pass} Bind Group`,
          layout: p.sortLayout,
          entries: [
            { binding: 0, resource: { buffer: b.sortUniforms } },
            { binding: 1, resource: { buffer: isEven ? b.mortonCodes : b.mortonCodesOut } },
            { binding: 2, resource: { buffer: isEven ? b.nodeIndices : b.nodeIndicesOut } },
            { binding: 3, resource: { buffer: isEven ? b.mortonCodesOut : b.mortonCodes } },
            { binding: 4, resource: { buffer: isEven ? b.nodeIndicesOut : b.nodeIndices } },
            { binding: 5, resource: { buffer: b.histogram } },
            { binding: 6, resource: { buffer: b.prefixSums } },
          ],
        }),
      );
    }

    // Prefix scan bind group
    const scan = device.createBindGroup({
      label: "BH Scan Bind Group",
      layout: p.scanLayout,
      entries: [
        { binding: 0, resource: { buffer: b.scanUniforms } },
        { binding: 1, resource: { buffer: b.histogram } },
        { binding: 2, resource: { buffer: b.prefixSums } },
      ],
    });

    // For tree construction, we use the final sorted buffers
    // After 8 passes (even number), results are in mortonCodes/nodeIndices
    const tree = device.createBindGroup({
      label: "BH Tree Bind Group (Full Sort)",
      layout: p.treeLayout,
      entries: [
        { binding: 0, resource: { buffer: b.treeUniforms } },
        { binding: 1, resource: { buffer: b.mortonCodes } },  // Sorted Morton codes
        { binding: 2, resource: { buffer: b.nodeIndices } },  // Sorted particle indices
        { binding: 3, resource: { buffer: context.positions } },
        { binding: 4, resource: { buffer: b.leftChild } },
        { binding: 5, resource: { buffer: b.rightChild } },
        { binding: 6, resource: { buffer: b.parent } },
        { binding: 7, resource: { buffer: b.nodeComX } },
        { binding: 8, resource: { buffer: b.nodeComY } },
        { binding: 9, resource: { buffer: b.nodeMass } },
        { binding: 10, resource: { buffer: b.nodeSize } },
        { binding: 11, resource: { buffer: b.visitCount } },
      ],
    });

    // For simple sort (1 pass), results are in mortonCodesOut/nodeIndicesOut
    const treeSimpleSort = device.createBindGroup({
      label: "BH Tree Bind Group (Simple Sort)",
      layout: p.treeLayout,
      entries: [
        { binding: 0, resource: { buffer: b.treeUniforms } },
        { binding: 1, resource: { buffer: b.mortonCodesOut } },  // Sorted Morton codes (Out buffer)
        { binding: 2, resource: { buffer: b.nodeIndicesOut } },  // Sorted particle indices (Out buffer)
        { binding: 3, resource: { buffer: context.positions } },
        { binding: 4, resource: { buffer: b.leftChild } },
        { binding: 5, resource: { buffer: b.rightChild } },
        { binding: 6, resource: { buffer: b.parent } },
        { binding: 7, resource: { buffer: b.nodeComX } },
        { binding: 8, resource: { buffer: b.nodeComY } },
        { binding: 9, resource: { buffer: b.nodeMass } },
        { binding: 10, resource: { buffer: b.nodeSize } },
        { binding: 11, resource: { buffer: b.visitCount } },
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
        { binding: 5, resource: { buffer: b.nodeComX } },
        { binding: 6, resource: { buffer: b.nodeComY } },
        { binding: 7, resource: { buffer: b.nodeMass } },
        { binding: 8, resource: { buffer: b.nodeSize } },
      ],
    });

    const bindGroups: BarnesHutBindGroups = {
      morton,
      sortPass,
      scan,
      tree,
      treeSimpleSort,
      repulsion,
      // Buffer references for per-pass uniform updates during command encoding
      sortUniforms: b.sortUniforms,
      sortUniformsStaging: b.sortUniformsStaging,
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
    const boundsMinX = context.bounds?.minX ?? -5000.0;
    const boundsMinY = context.bounds?.minY ?? -5000.0;
    const boundsMaxX = context.bounds?.maxX ?? 5000.0;
    const boundsMaxY = context.bounds?.maxY ?? 5000.0;
    const rootSize = Math.max(boundsMaxX - boundsMinX, boundsMaxY - boundsMinY);

    // Morton uniforms (32 bytes)
    // struct SimulationUniforms { bounds_min: vec2<f32>, bounds_max: vec2<f32>, node_count: u32, _padding: vec3<u32> }
    const mortonData = new ArrayBuffer(32);
    const mortonView = new DataView(mortonData);
    mortonView.setFloat32(0, boundsMinX, true);   // bounds_min.x
    mortonView.setFloat32(4, boundsMinY, true);   // bounds_min.y
    mortonView.setFloat32(8, boundsMaxX, true);   // bounds_max.x
    mortonView.setFloat32(12, boundsMaxY, true);  // bounds_max.y
    mortonView.setUint32(16, context.nodeCount, true);  // node_count
    mortonView.setUint32(20, 0, true);  // _padding[0]
    mortonView.setUint32(24, 0, true);  // _padding[1]
    mortonView.setUint32(28, 0, true);  // _padding[2]
    device.queue.writeBuffer(b.mortonUniforms, 0, mortonData);

    // Tree uniforms (32 bytes)
    // struct TreeUniforms { node_count: u32, bounds_min_x: f32, bounds_min_y: f32, bounds_max_x: f32,
    //                       bounds_max_y: f32, root_size: f32, _padding: vec2<u32> }
    const treeData = new ArrayBuffer(32);
    const treeView = new DataView(treeData);
    treeView.setUint32(0, context.nodeCount, true);
    treeView.setFloat32(4, boundsMinX, true);
    treeView.setFloat32(8, boundsMinY, true);
    treeView.setFloat32(12, boundsMaxX, true);
    treeView.setFloat32(16, boundsMaxY, true);
    treeView.setFloat32(20, rootSize, true);
    treeView.setUint32(24, 0, true);
    treeView.setUint32(28, 0, true);
    device.queue.writeBuffer(b.treeUniforms, 0, treeData);

    // Traverse uniforms (32 bytes)
    // struct ForceUniforms { particle_count: u32, repulsion_strength: f32, theta: f32, min_distance: f32,
    //                        leaf_size: f32, _pad1: f32, _pad2: f32, _pad3: f32 }
    const leafSize = rootSize / 256.0; // Approximate leaf size
    const traverseData = new ArrayBuffer(32);
    const traverseView = new DataView(traverseData);
    traverseView.setUint32(0, context.nodeCount, true);
    traverseView.setFloat32(4, Math.abs(context.forceConfig.repulsionStrength), true);
    traverseView.setFloat32(8, context.forceConfig.theta, true);
    traverseView.setFloat32(12, context.forceConfig.repulsionDistanceMin, true);
    traverseView.setFloat32(16, leafSize, true);
    traverseView.setFloat32(20, 0.0, true);
    traverseView.setFloat32(24, 0.0, true);
    traverseView.setFloat32(28, 0.0, true);
    device.queue.writeBuffer(b.traverseUniforms, 0, traverseData);

    // Scan uniforms (16 bytes)
    const workgroupCount = calculateWorkgroups(context.nodeCount, WORKGROUP_SIZE);
    const scanData = new ArrayBuffer(16);
    const scanView = new DataView(scanData);
    scanView.setUint32(0, workgroupCount * RADIX_SIZE, true);  // element_count
    scanView.setUint32(4, 0, true);  // pass_number (unused in sequential scan)
    scanView.setUint32(8, workgroupCount, true);  // workgroup_count
    scanView.setUint32(12, 0, true);  // _padding
    device.queue.writeBuffer(b.scanUniforms, 0, scanData);

    // Sort uniforms staging buffer (128 bytes = 8 passes * 16 bytes each)
    // Pre-compute uniforms for all 8 radix passes so we can copy the correct
    // pass_number during command encoding without needing device.queue.writeBuffer
    // struct SortUniforms { node_count: u32, pass_number: u32, _padding: vec2<u32> }
    const sortStagingData = new ArrayBuffer(RADIX_PASSES * 16);
    const sortStagingView = new DataView(sortStagingData);
    for (let pass = 0; pass < RADIX_PASSES; pass++) {
      const offset = pass * 16;
      sortStagingView.setUint32(offset + 0, context.nodeCount, true);  // node_count
      sortStagingView.setUint32(offset + 4, pass, true);               // pass_number (0-7)
      sortStagingView.setUint32(offset + 8, 0, true);                  // _padding[0]
      sortStagingView.setUint32(offset + 12, 0, true);                 // _padding[1]
    }
    device.queue.writeBuffer(b.sortUniformsStaging, 0, sortStagingData);
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

    // Use simple O(n²) counting sort for small arrays (more efficient for < 1024 nodes)
    const useSimpleSort = nodeCount < 1024;

    // === PHASE 1: Generate Morton codes ===
    {
      const pass = encoder.beginComputePass({ label: "BH Morton Codes" });
      pass.setPipeline(p.morton);
      pass.setBindGroup(0, bg.morton);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }

    // === PHASE 2: Sort by Morton code ===
    if (useSimpleSort) {
      // Simple counting sort for small arrays
      // This writes to keys_out/values_out (the Out buffers)
      const pass = encoder.beginComputePass({ label: "BH Simple Sort" });
      pass.setPipeline(p.radixSimple);
      pass.setBindGroup(0, bg.sortPass[0]);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    } else {
      // Full radix sort for larger arrays (8 passes for 32-bit Morton codes)
      // Each pass sorts by a different 4-bit digit, from least significant to most significant
      //
      // CRITICAL: The histogram buffer must be cleared before each pass because
      // histogram_count uses atomicAdd to accumulate counts. Without clearing,
      // counts from previous passes would corrupt the current pass's histogram,
      // leading to incorrect prefix sums and wrong scatter positions.
      const histogramWorkgroups = calculateWorkgroups(nodeWorkgroups * RADIX_SIZE, WORKGROUP_SIZE);

      for (let pass = 0; pass < RADIX_PASSES; pass++) {
        // Copy the pre-computed uniforms for this pass from staging buffer to active uniform buffer
        // The staging buffer contains 8 sets of uniforms (one per pass), each 16 bytes
        // This updates pass_number so the shader extracts the correct 4-bit digit
        encoder.copyBufferToBuffer(
          bg.sortUniformsStaging,  // source: staging buffer with all 8 pass uniforms
          pass * 16,               // source offset: pass * sizeof(SortUniforms)
          bg.sortUniforms,         // destination: active uniform buffer read by shader
          0,                       // destination offset: start of buffer
          16,                      // size: sizeof(SortUniforms)
        );

        // Step 0: Clear histogram buffer - MUST happen before histogram_count
        // The histogram uses atomicAdd, so stale values from the previous pass
        // would corrupt this pass's counts
        {
          const computePass = encoder.beginComputePass({ label: `BH Clear Histogram ${pass}` });
          computePass.setPipeline(p.radixClearHistogram);
          computePass.setBindGroup(0, bg.sortPass[pass]);
          computePass.dispatchWorkgroups(histogramWorkgroups);
          computePass.end();
        }

        // Step 1: Count histogram - count occurrences of each 4-bit digit value
        {
          const computePass = encoder.beginComputePass({ label: `BH Radix Histogram ${pass}` });
          computePass.setPipeline(p.radixHistogram);
          computePass.setBindGroup(0, bg.sortPass[pass]);
          computePass.dispatchWorkgroups(nodeWorkgroups);
          computePass.end();
        }

        // Step 2: Prefix scan on histogram - compute output positions for each bucket
        {
          const computePass = encoder.beginComputePass({ label: `BH Prefix Scan ${pass}` });
          computePass.setPipeline(p.radixScan);
          computePass.setBindGroup(0, bg.scan);
          computePass.dispatchWorkgroups(1);  // Sequential scan uses single workgroup
          computePass.end();
        }

        // Step 3: Scatter to sorted positions - move elements to their sorted locations
        {
          const computePass = encoder.beginComputePass({ label: `BH Radix Scatter ${pass}` });
          computePass.setPipeline(p.radixScatter);
          computePass.setBindGroup(0, bg.sortPass[pass]);
          computePass.dispatchWorkgroups(nodeWorkgroups);
          computePass.end();
        }
      }
    }

    // Select the correct tree bind group based on sort method
    // Simple sort outputs to Out buffers, full radix outputs to non-Out buffers
    const treeBindGroup = useSimpleSort ? bg.treeSimpleSort : bg.tree;

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

    // === PHASE 6: Aggregate centers of mass bottom-up ===
    {
      const pass = encoder.beginComputePass({ label: "BH Aggregate Bottom-Up" });
      pass.setPipeline(p.aggregateBottomUp);
      pass.setBindGroup(0, treeBindGroup);
      pass.dispatchWorkgroups(nodeWorkgroups);
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
