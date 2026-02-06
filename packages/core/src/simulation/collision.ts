/**
 * Collision Detection and Resolution Module
 *
 * Provides GPU-accelerated collision detection and resolution for node overlap
 * prevention. Runs as an optional post-integration pass.
 *
 * @module
 */

import type { GPUContext } from "../webgpu/context.ts";
import { calculateWorkgroups } from "../renderer/commands.ts";
import type { FullForceConfig } from "./config.ts";
import type { BoundingBox } from "../types.ts";

// Import shader sources
import COLLISION_WGSL from "./shaders/collision.comp.wgsl";
import COLLISION_GRID_WGSL from "./shaders/collision_grid.comp.wgsl";

const WORKGROUP_SIZE = 256;
const DEFAULT_RADIUS = 5.0;

/** Maximum grid dimension (each axis). Total cells <= MAX_GRID_DIM^2 = 65536. */
const MAX_GRID_DIM = 256;

/**
 * Collision pipeline resources
 */
export interface CollisionPipeline {
  /** Main collision resolution pipeline */
  resolve: GPUComputePipeline;
  /** Tiled version for larger graphs */
  resolveTiled: GPUComputePipeline;
  /** Bind group layout */
  bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Collision-specific buffers
 */
export interface CollisionBuffers {
  /** Uniform buffer for collision parameters */
  uniforms: GPUBuffer;
  /** Node sizes/radii buffer */
  nodeSizes: GPUBuffer;
  /** Maximum nodes this buffer set supports */
  maxNodes: number;
}

/**
 * Collision bind group
 */
export interface CollisionBindGroup {
  /** Bind group for collision pass */
  bindGroup: GPUBindGroup;
}

/**
 * Creates the collision compute pipeline
 *
 * @param context - GPU context
 * @returns Collision pipeline resources
 */
export function createCollisionPipeline(context: GPUContext): CollisionPipeline {
  const { device } = context;

  const shaderModule = device.createShaderModule({
    label: "Collision Shader",
    code: COLLISION_WGSL,
  });

  // Bind group layout for collision pass
  // Bindings: uniforms, positions (vec2, read-write), node_sizes
  const bindGroupLayout = device.createBindGroupLayout({
    label: "Collision Bind Group Layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "Collision Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });

  const resolve = device.createComputePipeline({
    label: "Collision Resolve Pipeline",
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "main" },
  });

  const resolveTiled = device.createComputePipeline({
    label: "Collision Resolve Tiled Pipeline",
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "resolve_tiled" },
  });

  return {
    resolve,
    resolveTiled,
    bindGroupLayout,
  };
}

/**
 * Creates collision-specific buffers
 *
 * @param device - GPU device
 * @param maxNodes - Maximum number of nodes
 * @returns Collision buffers
 */
export function createCollisionBuffers(
  device: GPUDevice,
  maxNodes: number,
): CollisionBuffers {
  const safeMaxNodes = Math.max(maxNodes, 4);
  const nodeBytes = safeMaxNodes * 4;

  // Uniform buffer (32 bytes for CollisionUniforms struct)
  const uniforms = device.createBuffer({
    label: "Collision Uniforms",
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Node sizes buffer
  const nodeSizes = device.createBuffer({
    label: "Node Sizes",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  return {
    uniforms,
    nodeSizes,
    maxNodes: safeMaxNodes,
  };
}

/**
 * Creates collision bind group
 *
 * @param device - GPU device
 * @param pipeline - Collision pipeline
 * @param collisionBuffers - Collision-specific buffers
 * @param positions - Position buffer (vec2, read-write)
 * @returns Collision bind group
 */
export function createCollisionBindGroup(
  device: GPUDevice,
  pipeline: CollisionPipeline,
  collisionBuffers: CollisionBuffers,
  positions: GPUBuffer,
): CollisionBindGroup {
  const bindGroup = device.createBindGroup({
    label: "Collision Bind Group",
    layout: pipeline.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: collisionBuffers.uniforms } },
      { binding: 1, resource: { buffer: positions } },
      { binding: 2, resource: { buffer: collisionBuffers.nodeSizes } },
    ],
  });

  return { bindGroup };
}

/**
 * Updates collision uniform buffer
 *
 * @param device - GPU device
 * @param collisionBuffers - Collision buffers
 * @param nodeCount - Number of nodes
 * @param forceConfig - Force configuration
 */
export function updateCollisionUniforms(
  device: GPUDevice,
  collisionBuffers: CollisionBuffers,
  nodeCount: number,
  forceConfig: FullForceConfig,
): void {
  // CollisionUniforms struct (32 bytes):
  // node_count: u32, collision_strength: f32, radius_multiplier: f32, iterations: u32,
  // default_radius: f32, _pad0: f32, _pad1: f32, _pad2: f32
  const data = new ArrayBuffer(32);
  const view = new DataView(data);
  view.setUint32(0, nodeCount, true);
  view.setFloat32(4, forceConfig.collisionStrength, true);
  view.setFloat32(8, forceConfig.collisionRadiusMultiplier, true);
  view.setUint32(12, forceConfig.collisionIterations, true);
  view.setFloat32(16, DEFAULT_RADIUS, true);
  view.setFloat32(20, 0.0, true);  // _pad0
  view.setFloat32(24, 0.0, true);  // _pad1
  view.setFloat32(28, 0.0, true);  // _pad2
  device.queue.writeBuffer(collisionBuffers.uniforms, 0, data);
}

/**
 * Uploads node sizes to GPU
 *
 * @param device - GPU device
 * @param collisionBuffers - Collision buffers
 * @param nodeSizes - Array of node sizes/radii
 */
export function uploadNodeSizes(
  device: GPUDevice,
  collisionBuffers: CollisionBuffers,
  nodeSizes: Float32Array,
): void {
  // Create a proper ArrayBuffer copy to satisfy BufferSource type
  const buffer = new ArrayBuffer(nodeSizes.byteLength);
  new Float32Array(buffer).set(nodeSizes);
  device.queue.writeBuffer(collisionBuffers.nodeSizes, 0, buffer);
}

/**
 * Records collision detection pass(es) to command encoder
 *
 * @param encoder - Command encoder
 * @param pipeline - Collision pipeline
 * @param bindGroup - Collision bind group
 * @param nodeCount - Number of nodes
 * @param iterations - Number of collision resolution iterations
 * @param useTiled - Use tiled version for large graphs (>5000 nodes)
 */
export function recordCollisionPass(
  encoder: GPUCommandEncoder,
  pipeline: CollisionPipeline,
  bindGroup: CollisionBindGroup,
  nodeCount: number,
  iterations: number = 1,
  useTiled: boolean = false,
): void {
  const workgroups = calculateWorkgroups(nodeCount, WORKGROUP_SIZE);
  const selectedPipeline = useTiled ? pipeline.resolveTiled : pipeline.resolve;

  for (let i = 0; i < iterations; i++) {
    const pass = encoder.beginComputePass({
      label: `Collision Resolution ${i + 1}/${iterations}`,
    });
    pass.setPipeline(selectedPipeline);
    pass.setBindGroup(0, bindGroup.bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
  }
}

/**
 * Destroys collision buffers
 *
 * @param buffers - Collision buffers to destroy
 */
export function destroyCollisionBuffers(buffers: CollisionBuffers): void {
  buffers.uniforms.destroy();
  buffers.nodeSizes.destroy();
}

// ============================================================================
// Grid-Based Collision (O(n·k) atomic linked list spatial hash)
// ============================================================================

/**
 * Grid collision pipeline resources.
 *
 * Uses a spatial hash grid with atomic linked lists to reduce collision
 * detection from O(n^2) to O(n·k) where k is the average number of nodes
 * per cell neighborhood. Only 3 GPU dispatches per iteration.
 */
export interface GridCollisionPipeline {
  /** Clear all cell head pointers to EMPTY sentinel */
  clearCells: GPUComputePipeline;
  /** Build per-cell linked lists via atomic prepend */
  buildLists: GPUComputePipeline;
  /** Resolve collisions by walking linked lists in 3x3 neighborhood */
  resolveGrid: GPUComputePipeline;
  /** Bind group layout (6 bindings) */
  gridLayout: GPUBindGroupLayout;
}

/**
 * Grid collision buffers.
 */
export interface GridCollisionBuffers {
  /** Grid collision uniform buffer (48 bytes) */
  gridUniforms: GPUBuffer;
  /** Per-cell linked list head pointers (MAX_GRID_DIM^2 atomic u32 entries) */
  cellHead: GPUBuffer;
  /** Per-node next pointer for linked list traversal (maxNodes u32 entries) */
  nodeNext: GPUBuffer;
  /** Per-node cell hash (maxNodes u32 entries, avoids recomputing in resolve) */
  nodeCell: GPUBuffer;
  /** Maximum node count this buffer set supports */
  maxNodes: number;
  /** Maximum cell count (MAX_GRID_DIM^2) */
  maxCells: number;
}

/**
 * Grid collision bind groups.
 */
export interface GridCollisionBindGroups {
  /** Single bind group for all 3 entry points (6 bindings) */
  grid: GPUBindGroup;
}

/**
 * Creates grid collision compute pipelines.
 *
 * @param context - GPU context
 * @returns Grid collision pipeline resources
 */
export function createGridCollisionPipeline(
  context: GPUContext,
): GridCollisionPipeline {
  const { device } = context;

  const shaderModule = device.createShaderModule({
    label: "Grid Collision Shader",
    code: COLLISION_GRID_WGSL,
  });

  // All 3 entry points share this layout (6 bindings).
  const gridLayout = device.createBindGroupLayout({
    label: "Grid Collision Layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // positions (rw)
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // node_sizes
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // cell_head (atomic rw)
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // node_next (rw)
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // node_cell (rw)
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "Grid Collision Pipeline Layout",
    bindGroupLayouts: [gridLayout],
  });

  const clearCells = device.createComputePipeline({
    label: "Grid Collision Clear Cells",
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "clear_cells" },
  });

  const buildLists = device.createComputePipeline({
    label: "Grid Collision Build Lists",
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "build_lists" },
  });

  const resolveGrid = device.createComputePipeline({
    label: "Grid Collision Resolve",
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "resolve_grid" },
  });

  return {
    clearCells,
    buildLists,
    resolveGrid,
    gridLayout,
  };
}

/**
 * Creates grid collision buffers.
 *
 * @param device - GPU device
 * @param maxNodes - Maximum number of nodes
 * @returns Grid collision buffers
 */
export function createGridCollisionBuffers(
  device: GPUDevice,
  maxNodes: number,
): GridCollisionBuffers {
  const safeMaxNodes = Math.max(maxNodes, 4);
  const maxCells = MAX_GRID_DIM * MAX_GRID_DIM;
  const cellBytes = maxCells * 4;
  const nodeBytes = safeMaxNodes * 4;

  const gridUniforms = device.createBuffer({
    label: "Grid Collision Uniforms",
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const cellHead = device.createBuffer({
    label: "Grid Collision Cell Head",
    size: cellBytes,
    usage: GPUBufferUsage.STORAGE,
  });

  const nodeNext = device.createBuffer({
    label: "Grid Collision Node Next",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE,
  });

  const nodeCell = device.createBuffer({
    label: "Grid Collision Node Cell",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE,
  });

  return {
    gridUniforms,
    cellHead,
    nodeNext,
    nodeCell,
    maxNodes: safeMaxNodes,
    maxCells,
  };
}

/**
 * Creates grid collision bind groups.
 *
 * @param device - GPU device
 * @param pipeline - Grid collision pipeline
 * @param gridBuffers - Grid collision buffers
 * @param nodeSizes - Node sizes buffer (from CollisionBuffers)
 * @param positions - Position buffer (positionsOut for ping-pong consistency)
 * @returns Grid collision bind groups
 */
export function createGridCollisionBindGroups(
  device: GPUDevice,
  pipeline: GridCollisionPipeline,
  gridBuffers: GridCollisionBuffers,
  nodeSizes: GPUBuffer,
  positions: GPUBuffer,
): GridCollisionBindGroups {
  const grid = device.createBindGroup({
    label: "Grid Collision Bind Group",
    layout: pipeline.gridLayout,
    entries: [
      { binding: 0, resource: { buffer: gridBuffers.gridUniforms } },
      { binding: 1, resource: { buffer: positions } },
      { binding: 2, resource: { buffer: nodeSizes } },
      { binding: 3, resource: { buffer: gridBuffers.cellHead } },
      { binding: 4, resource: { buffer: gridBuffers.nodeNext } },
      { binding: 5, resource: { buffer: gridBuffers.nodeCell } },
    ],
  });

  return { grid };
}

/**
 * Computes grid dimensions from bounds and maximum node radius.
 *
 * Cell size starts at 2 * maxRadius * radiusMultiplier (guaranteeing that
 * overlapping nodes are always in the same or adjacent cells). If the
 * resulting grid exceeds MAX_GRID_DIM on either axis, cell size is increased
 * until the grid fits.
 */
function computeGridDimensions(
  bounds: BoundingBox,
  maxRadius: number,
  radiusMultiplier: number,
): { gridWidth: number; gridHeight: number; cellSize: number } {
  let cellSize = Math.max(2 * maxRadius * radiusMultiplier, 1.0);
  const spanX = Math.max(bounds.maxX - bounds.minX, cellSize);
  const spanY = Math.max(bounds.maxY - bounds.minY, cellSize);

  // Ensure grid fits within MAX_GRID_DIM on each axis
  const minCellSizeX = spanX / (MAX_GRID_DIM - 1);
  const minCellSizeY = spanY / (MAX_GRID_DIM - 1);
  cellSize = Math.max(cellSize, minCellSizeX, minCellSizeY);

  const gridWidth = Math.min(Math.ceil(spanX / cellSize) + 1, MAX_GRID_DIM);
  const gridHeight = Math.min(Math.ceil(spanY / cellSize) + 1, MAX_GRID_DIM);

  return { gridWidth, gridHeight, cellSize };
}

/**
 * Updates grid collision uniform buffer.
 *
 * Must be called each frame before recordGridCollisionPass because bounds
 * change as nodes move, which changes grid dimensions.
 *
 * @param device - GPU device
 * @param gridBuffers - Grid collision buffers
 * @param nodeCount - Current node count
 * @param forceConfig - Force configuration
 * @param bounds - Current graph bounding box
 * @param maxRadius - Maximum node radius (for cell size computation)
 */
export function updateGridCollisionUniforms(
  device: GPUDevice,
  gridBuffers: GridCollisionBuffers,
  nodeCount: number,
  forceConfig: FullForceConfig,
  bounds: BoundingBox,
  maxRadius: number,
): void {
  if (nodeCount > gridBuffers.maxNodes) {
    throw new Error(
      `Grid collision buffer overflow: nodeCount (${nodeCount}) exceeds capacity (${gridBuffers.maxNodes}).`
    );
  }

  const { gridWidth, gridHeight, cellSize } = computeGridDimensions(
    bounds, maxRadius, forceConfig.collisionRadiusMultiplier,
  );
  const totalCells = gridWidth * gridHeight;

  // GridCollisionUniforms (48 bytes, 16-byte aligned)
  const data = new ArrayBuffer(48);
  const view = new DataView(data);
  view.setUint32(0, nodeCount, true);                               // node_count
  view.setUint32(4, gridWidth, true);                                // grid_width
  view.setUint32(8, gridHeight, true);                               // grid_height
  view.setFloat32(12, cellSize, true);                               // cell_size
  view.setFloat32(16, bounds.minX, true);                            // bounds_min_x
  view.setFloat32(20, bounds.minY, true);                            // bounds_min_y
  view.setFloat32(24, forceConfig.collisionStrength, true);          // collision_strength
  view.setFloat32(28, forceConfig.collisionRadiusMultiplier, true);  // radius_multiplier
  view.setFloat32(32, DEFAULT_RADIUS, true);                         // default_radius
  view.setUint32(36, totalCells, true);                              // total_cells
  view.setUint32(40, 0, true);                                      // _pad0
  view.setUint32(44, 0, true);                                      // _pad1
  device.queue.writeBuffer(gridBuffers.gridUniforms, 0, data);
}

/**
 * Records grid-based collision detection pass(es).
 *
 * Per iteration (3 GPU dispatches):
 * 1. clear_cells  — reset all cell head pointers to EMPTY
 * 2. build_lists  — each node atomically prepends itself to its cell's list
 * 3. resolve_grid — walk linked lists in 3x3 neighborhood for overlaps
 *
 * @param encoder - GPU command encoder
 * @param pipeline - Grid collision pipeline
 * @param bindGroups - Grid collision bind groups
 * @param gridBuffers - Grid collision buffers (for maxCells dispatch sizing)
 * @param nodeCount - Number of nodes
 * @param iterations - Number of collision resolution iterations
 */
export function recordGridCollisionPass(
  encoder: GPUCommandEncoder,
  pipeline: GridCollisionPipeline,
  bindGroups: GridCollisionBindGroups,
  gridBuffers: GridCollisionBuffers,
  nodeCount: number,
  iterations: number = 1,
): void {
  if (nodeCount < 2) {
    return;
  }

  const nodeWorkgroups = calculateWorkgroups(nodeCount, WORKGROUP_SIZE);
  // Clear all cells each iteration even if some are unused — over-clearing
  // with EMPTY sentinels is harmless and avoids tracking exact grid dimensions.
  const maxCells = gridBuffers.maxCells;
  const cellWorkgroups = calculateWorkgroups(maxCells, WORKGROUP_SIZE);

  for (let iter = 0; iter < iterations; iter++) {
    // Phase 1: Clear cell head pointers
    {
      const pass = encoder.beginComputePass({
        label: `GridCollision Clear Cells ${iter + 1}/${iterations}`,
      });
      pass.setPipeline(pipeline.clearCells);
      pass.setBindGroup(0, bindGroups.grid);
      pass.dispatchWorkgroups(cellWorkgroups);
      pass.end();
    }

    // Phase 2: Build linked lists (each node atomically prepends to its cell)
    {
      const pass = encoder.beginComputePass({
        label: `GridCollision Build Lists ${iter + 1}/${iterations}`,
      });
      pass.setPipeline(pipeline.buildLists);
      pass.setBindGroup(0, bindGroups.grid);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }

    // Phase 3: Resolve collisions by walking 3x3 neighborhood lists
    {
      const pass = encoder.beginComputePass({
        label: `GridCollision Resolve ${iter + 1}/${iterations}`,
      });
      pass.setPipeline(pipeline.resolveGrid);
      pass.setBindGroup(0, bindGroups.grid);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }
  }
}

/**
 * Destroys grid collision buffers.
 *
 * @param buffers - Grid collision buffers to destroy
 */
export function destroyGridCollisionBuffers(
  buffers: GridCollisionBuffers,
): void {
  buffers.gridUniforms.destroy();
  buffers.cellHead.destroy();
  buffers.nodeNext.destroy();
  buffers.nodeCell.destroy();
}
