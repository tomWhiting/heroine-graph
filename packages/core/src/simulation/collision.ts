/**
 * Collision Detection and Resolution Module
 *
 * Provides GPU-accelerated collision detection and resolution for node overlap
 * prevention. Runs as an optional post-integration pass.
 *
 * Every collision iteration is two dispatches: a resolve pass that reads the
 * position snapshot and writes a per-node displacement, then an apply pass
 * that adds those displacements to the positions. Resolving in place would
 * mean threads read neighbour positions other threads are concurrently
 * writing — a cross-workgroup race with no ordering guarantee, which made the
 * resolved layout depend on how the driver scheduled workgroups.
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
import COLLISION_APPLY_WGSL from "./shaders/collision_apply.comp.wgsl";

const WORKGROUP_SIZE = 256;
const DEFAULT_RADIUS = 5.0;

/** Maximum grid dimension (each axis). Total cells <= MAX_GRID_DIM^2 = 65536. */
const MAX_GRID_DIM = 256;

/**
 * Collision pipeline resources
 */
export interface CollisionPipeline {
  /** Main collision resolution pipeline (pass 1: positions -> displacements) */
  resolve: GPUComputePipeline;
  /** Tiled version for larger graphs (pass 1) */
  resolveTiled: GPUComputePipeline;
  /** Pass 2: adds the accumulated displacements to the positions */
  apply: GPUComputePipeline;
  /** Bind group layout for the resolve passes */
  bindGroupLayout: GPUBindGroupLayout;
  /** Bind group layout for the apply pass */
  applyLayout: GPUBindGroupLayout;
}

/**
 * Collision-specific buffers
 */
export interface CollisionBuffers {
  /** Uniform buffer for collision parameters */
  uniforms: GPUBuffer;
  /** Node sizes/radii buffer */
  nodeSizes: GPUBuffer;
  /**
   * Per-node displacement handed from the resolve pass to the apply pass
   * (vec2<f32> per node). Scratch space only: the resolve pass overwrites
   * every slot it dispatches over, so nothing carries across iterations.
   * Shared with the grid collision path, which accumulates into the same
   * buffer.
   */
  displacements: GPUBuffer;
  /**
   * Zeroed fallback for the node_flags binding, used when the caller does not
   * pass the simulation's nodeFlags buffer (every slot treated as live,
   * unpinned and visible). Pass SimulationBuffers.nodeFlags to
   * createCollisionBindGroup so collision respects NODE_FLAG_DEAD /
   * NODE_FLAG_PINNED / NODE_FLAG_HIDDEN_LOD — and so the apply pass can tell a
   * fresh displacement from one a shortened resolve dispatch left behind.
   */
  fallbackNodeFlags: GPUBuffer;
  /**
   * Identity list `[0, maxNodes)` for the `live_idx` binding, used when the
   * caller passes no active-index list. It describes the whole graph in slot
   * order, which is exactly what the resolve passes swept before the list
   * existed. Never zero-filled: a zeroed list is every thread colliding slot 0
   * with itself.
   */
  fallbackLiveIndices: GPUBuffer;
  /** Maximum nodes this buffer set supports */
  maxNodes: number;
}

/**
 * Collision bind group
 */
export interface CollisionBindGroup {
  /** Bind group for the resolve pass */
  bindGroup: GPUBindGroup;
  /** Bind group for the apply pass (positions + displacements) */
  applyBindGroup: GPUBindGroup;
}

/**
 * Creates the pass-2 pipeline that folds accumulated displacements into
 * positions. Both the O(n^2) and the grid collision path build their own
 * instance: the resolve shaders declare positions read-only, so the write
 * has to live in a separate WGSL module with its own bind group layout.
 */
function createApplyPipeline(
  device: GPUDevice,
): { apply: GPUComputePipeline; applyLayout: GPUBindGroupLayout } {
  const shaderModule = device.createShaderModule({
    label: "Collision Apply Shader",
    code: COLLISION_APPLY_WGSL,
  });

  // Bindings: positions (vec2, read-write), displacements (vec2, read-only),
  // node_flags
  const applyLayout = device.createBindGroupLayout({
    label: "Collision Apply Bind Group Layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });

  const apply = device.createComputePipeline({
    label: "Collision Apply Pipeline",
    layout: device.createPipelineLayout({
      label: "Collision Apply Pipeline Layout",
      bindGroupLayouts: [applyLayout],
    }),
    compute: { module: shaderModule, entryPoint: "apply_displacements" },
  });

  return { apply, applyLayout };
}

/**
 * One apply pipeline per device, shared by the n2/tiled and grid paths.
 *
 * Both resolve paths write the same displacements buffer and both fold it in
 * with the same shader, layout and entry point — and they are mutually
 * exclusive per frame — so building it twice created two identical GPU objects
 * and a second place for them to drift apart. Keyed weakly on the device so a
 * destroyed device's pipeline is collectable and a fresh device never reuses
 * one.
 */
const applyPipelineByDevice = new WeakMap<
  GPUDevice,
  { apply: GPUComputePipeline; applyLayout: GPUBindGroupLayout }
>();

function getApplyPipeline(
  device: GPUDevice,
): { apply: GPUComputePipeline; applyLayout: GPUBindGroupLayout } {
  let cached = applyPipelineByDevice.get(device);
  if (!cached) {
    cached = createApplyPipeline(device);
    applyPipelineByDevice.set(device, cached);
  }
  return cached;
}

/**
 * Builds the apply-pass bind group for a (positions, displacements) pair.
 */
function createApplyBindGroup(
  device: GPUDevice,
  applyLayout: GPUBindGroupLayout,
  positions: GPUBuffer,
  displacements: GPUBuffer,
  nodeFlags: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: "Collision Apply Bind Group",
    layout: applyLayout,
    entries: [
      { binding: 0, resource: { buffer: positions } },
      { binding: 1, resource: { buffer: displacements } },
      { binding: 2, resource: { buffer: nodeFlags } },
    ],
  });
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

  // Bind group layout for the resolve pass
  // Bindings: uniforms, positions (vec2, read-only), node_sizes, node_flags,
  // displacements (vec2, read-write), live_idx (active-index list)
  const bindGroupLayout = device.createBindGroupLayout({
    label: "Collision Bind Group Layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
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

  const { apply, applyLayout } = getApplyPipeline(device);

  return {
    resolve,
    resolveTiled,
    apply,
    bindGroupLayout,
    applyLayout,
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

  // Resolve -> apply hand-off, one vec2<f32> per node. Never uploaded from
  // the CPU: the resolve pass writes every slot before the apply pass reads it.
  const displacements = device.createBuffer({
    label: "Collision Displacements",
    size: safeMaxNodes * 8,
    usage: GPUBufferUsage.STORAGE,
  });

  // Zeroed fallback flags (all live/unpinned) for callers without a
  // simulation nodeFlags buffer
  const fallbackNodeFlags = device.createBuffer({
    label: "Collision Fallback Node Flags",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const fallbackLiveIndices = createIdentityLiveIndices(
    device,
    safeMaxNodes,
    "Collision Fallback Live Indices",
  );

  return {
    uniforms,
    nodeSizes,
    displacements,
    fallbackNodeFlags,
    fallbackLiveIndices,
    maxNodes: safeMaxNodes,
  };
}

/** Allocate and fill an ascending identity list `[0, slotCount)`. */
function createIdentityLiveIndices(
  device: GPUDevice,
  slotCount: number,
  label: string,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: Math.max(slotCount, 1) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const identity = new Uint32Array(Math.max(slotCount, 1));
  for (let i = 0; i < identity.length; i++) identity[i] = i;
  device.queue.writeBuffer(buffer, 0, identity);
  return buffer;
}

/**
 * Creates collision bind group
 *
 * Builds both bind groups an iteration needs: the resolve pass (positions
 * read-only, displacements written) and the apply pass (positions written).
 *
 * @param device - GPU device
 * @param pipeline - Collision pipeline
 * @param collisionBuffers - Collision-specific buffers
 * @param positions - Position buffer (vec2; read in resolve, written in apply)
 * @param nodeFlags - Simulation nodeFlags buffer (u32 per node; bit 0 = dead,
 *   bit 1 = pinned, bit 2 = hidden by LOD). Both passes read it: resolve skips
 *   dead and LOD-hidden slots as neighbours and computes no displacement for a
 *   slot that is dead, LOD-hidden or pinned; apply masks the same three bits
 *   before folding a displacement into a position.
 *
 *   Omitting it binds an all-zero buffer, which declares every slot live,
 *   unpinned and visible — and that forfeits the guarantee that makes a
 *   shortened resolve dispatch safe. `liveIndices` lets resolve sweep fewer
 *   slots than apply does, so an omitted slot keeps the displacement it was
 *   left with when it last took part; the IMMOVABLE mask in apply is the only
 *   thing that stops that stale value being re-applied every iteration. Pass
 *   SimulationBuffers.nodeFlags whenever `liveIndices` is anything but the
 *   whole graph.
 * @param liveIndices - Active-index list bound to the resolve pass: its first
 *   `activeCount` entries (see {@link updateCollisionUniforms}) are the slots
 *   that take part. Falls back to
 *   {@link CollisionBuffers.fallbackLiveIndices}, the identity list over every
 *   slot, which is the whole graph in slot order.
 * @returns Collision bind groups
 */
export function createCollisionBindGroup(
  device: GPUDevice,
  pipeline: CollisionPipeline,
  collisionBuffers: CollisionBuffers,
  positions: GPUBuffer,
  nodeFlags?: GPUBuffer,
  liveIndices?: GPUBuffer,
): CollisionBindGroup {
  const flags = nodeFlags ?? collisionBuffers.fallbackNodeFlags;
  const bindGroup = device.createBindGroup({
    label: "Collision Bind Group",
    layout: pipeline.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: collisionBuffers.uniforms } },
      { binding: 1, resource: { buffer: positions } },
      { binding: 2, resource: { buffer: collisionBuffers.nodeSizes } },
      { binding: 3, resource: { buffer: flags } },
      { binding: 4, resource: { buffer: collisionBuffers.displacements } },
      { binding: 5, resource: { buffer: liveIndices ?? collisionBuffers.fallbackLiveIndices } },
    ],
  });

  const applyBindGroup = createApplyBindGroup(
    device,
    pipeline.applyLayout,
    positions,
    collisionBuffers.displacements,
    flags,
  );

  return { bindGroup, applyBindGroup };
}

/**
 * Updates collision uniform buffer
 *
 * @param device - GPU device
 * @param collisionBuffers - Collision buffers
 * @param nodeCount - Number of nodes
 * @param forceConfig - Force configuration
 * @param activeCount - Entries of the active-index list bound to the resolve
 *   pass. Defaults to `nodeCount`, which is what the identity fallback list
 *   describes. It must come from the same field as the dispatch size passed to
 *   {@link recordCollisionPass}, or threads sweep past the end of the list.
 */
export function updateCollisionUniforms(
  device: GPUDevice,
  collisionBuffers: CollisionBuffers,
  nodeCount: number,
  forceConfig: FullForceConfig,
  activeCount?: number,
): void {
  // CollisionUniforms struct (32 bytes):
  // node_count: u32, collision_strength: f32, radius_multiplier: f32,
  // _pad_iterations: u32, default_radius: f32, _pad0: f32, _pad1: f32, _pad2: f32
  //
  // Offset 12 used to carry collisionIterations. No shader ever read it — the
  // iteration loop is on the CPU (recordCollisionPass), because each iteration
  // is a separate resolve+apply dispatch pair. The slot stays as explicit
  // padding rather than being removed, so the struct layout and the 32-byte
  // buffer size are unchanged.
  const data = new ArrayBuffer(32);
  const view = new DataView(data);
  view.setUint32(0, nodeCount, true);
  view.setFloat32(4, forceConfig.collisionStrength, true);
  view.setFloat32(8, forceConfig.collisionRadiusMultiplier, true);
  view.setUint32(12, 0, true); // _pad_iterations
  view.setFloat32(16, DEFAULT_RADIUS, true);
  // The dispatch and both sweep bounds come from this one number. Defaulting
  // to nodeCount is what the identity fallback list describes.
  view.setUint32(20, activeCount ?? nodeCount, true); // active_count
  view.setFloat32(24, 0.0, true); // _pad1
  view.setFloat32(28, 0.0, true); // _pad2
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
 * Records collision detection pass(es) to command encoder.
 *
 * Each iteration is a resolve dispatch followed by an apply dispatch. Both
 * cover the same threads, so every displacement the resolve pass writes is
 * consumed by the apply pass that follows it, and each iteration starts from
 * the previous iteration's fully applied positions.
 *
 * @param encoder - Command encoder
 * @param pipeline - Collision pipeline
 * @param bindGroup - Collision bind groups
 * @param nodeCount - Number of nodes
 * @param iterations - Number of collision resolution iterations
 * @param useTiled - Use tiled version for large graphs (>5000 nodes)
 * @param activeCount - Entries of the active-index list, sizing the resolve
 *   dispatch. Must be the same number given to
 *   {@link updateCollisionUniforms}. Defaults to `nodeCount`.
 */
export function recordCollisionPass(
  encoder: GPUCommandEncoder,
  pipeline: CollisionPipeline,
  bindGroup: CollisionBindGroup,
  nodeCount: number,
  iterations: number = 1,
  useTiled: boolean = false,
  activeCount?: number,
): void {
  // The resolve pass is indexed by ACTIVE-LIST ENTRY and shrinks with the cut;
  // the apply pass stays indexed by SLOT, because its body is a load, a
  // compare and an add and it is what makes the shortened resolve dispatch
  // safe (it drops every IMMOVABLE slot on the flags rather than trusting the
  // list).
  const resolveWorkgroups = calculateWorkgroups(activeCount ?? nodeCount, WORKGROUP_SIZE);
  const applyWorkgroups = calculateWorkgroups(nodeCount, WORKGROUP_SIZE);
  const selectedPipeline = useTiled ? pipeline.resolveTiled : pipeline.resolve;

  for (let i = 0; i < iterations; i++) {
    {
      const pass = encoder.beginComputePass({
        label: `Collision Resolution ${i + 1}/${iterations}`,
      });
      pass.setPipeline(selectedPipeline);
      pass.setBindGroup(0, bindGroup.bindGroup);
      pass.dispatchWorkgroups(resolveWorkgroups);
      pass.end();
    }
    {
      const pass = encoder.beginComputePass({
        label: `Collision Apply ${i + 1}/${iterations}`,
      });
      pass.setPipeline(pipeline.apply);
      pass.setBindGroup(0, bindGroup.applyBindGroup);
      pass.dispatchWorkgroups(applyWorkgroups);
      pass.end();
    }
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
  buffers.displacements.destroy();
  buffers.fallbackNodeFlags.destroy();
  buffers.fallbackLiveIndices.destroy();
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
  /** Adds the accumulated displacements to the positions */
  apply: GPUComputePipeline;
  /** Bind group layout for the three grid phases (8 bindings) */
  gridLayout: GPUBindGroupLayout;
  /** Bind group layout for the apply pass */
  applyLayout: GPUBindGroupLayout;
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
  /**
   * Zeroed fallback for the node_flags binding (every slot live, unpinned and
   * visible). Pass SimulationBuffers.nodeFlags to
   * createGridCollisionBindGroups so grid collision respects NODE_FLAG_DEAD /
   * NODE_FLAG_PINNED / NODE_FLAG_HIDDEN_LOD.
   */
  fallbackNodeFlags: GPUBuffer;
  /**
   * Identity list `[0, maxNodes)` for the `live_idx` binding, used when the
   * caller passes no active-index list. See CollisionBuffers.fallbackLiveIndices.
   */
  fallbackLiveIndices: GPUBuffer;
  /** Maximum node count this buffer set supports */
  maxNodes: number;
  /** Maximum cell count (MAX_GRID_DIM^2) */
  maxCells: number;
}

/**
 * Grid collision bind groups.
 */
export interface GridCollisionBindGroups {
  /** Single bind group for the 3 grid entry points (9 bindings) */
  grid: GPUBindGroup;
  /** Bind group for the apply pass (positions + displacements) */
  apply: GPUBindGroup;
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

  // All 3 grid entry points share this layout (9 bindings).
  const gridLayout = device.createBindGroupLayout({
    label: "Grid Collision Layout",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // positions (ro)
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // node_sizes
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // cell_head (atomic rw)
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // node_next (rw)
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // node_cell (rw)
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // node_flags
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // displacements (rw)
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // live_idx
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

  const { apply, applyLayout } = getApplyPipeline(device);

  return {
    clearCells,
    buildLists,
    resolveGrid,
    apply,
    gridLayout,
    applyLayout,
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

  // Zeroed fallback flags (all live/unpinned) for callers without a
  // simulation nodeFlags buffer
  const fallbackNodeFlags = device.createBuffer({
    label: "Grid Collision Fallback Node Flags",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  return {
    gridUniforms,
    cellHead,
    nodeNext,
    nodeCell,
    fallbackNodeFlags,
    fallbackLiveIndices: createIdentityLiveIndices(
      device,
      safeMaxNodes,
      "Grid Collision Fallback Live Indices",
    ),
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
 * @param displacements - Resolve -> apply scratch buffer (from CollisionBuffers;
 *   the two collision paths never run in the same frame, so they share it)
 * @param positions - Position buffer (positionsOut for ping-pong consistency)
 * @param nodeFlags - Simulation nodeFlags buffer (u32 per node; bit 0 = dead,
 *   bit 1 = pinned, bit 2 = hidden by LOD). Dead and LOD-hidden slots are kept
 *   out of the cell lists entirely and pinned slots take no displacement; the
 *   apply pass masks the same three bits. Omitting it binds an all-zero buffer
 *   — every slot live, unpinned and visible — which also forfeits the
 *   stale-displacement guarantee described on
 *   {@link createCollisionBindGroup}, since the two paths share both the
 *   displacement buffer and the apply shader.
 * @param liveIndices - Active-index list bound to the build and resolve
 *   passes; see {@link createCollisionBindGroup}. Falls back to
 *   {@link GridCollisionBuffers.fallbackLiveIndices}.
 * @returns Grid collision bind groups
 */
export function createGridCollisionBindGroups(
  device: GPUDevice,
  pipeline: GridCollisionPipeline,
  gridBuffers: GridCollisionBuffers,
  nodeSizes: GPUBuffer,
  displacements: GPUBuffer,
  positions: GPUBuffer,
  nodeFlags?: GPUBuffer,
  liveIndices?: GPUBuffer,
): GridCollisionBindGroups {
  const flags = nodeFlags ?? gridBuffers.fallbackNodeFlags;
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
      { binding: 6, resource: { buffer: flags } },
      { binding: 7, resource: { buffer: displacements } },
      { binding: 8, resource: { buffer: liveIndices ?? gridBuffers.fallbackLiveIndices } },
    ],
  });

  const apply = createApplyBindGroup(
    device,
    pipeline.applyLayout,
    positions,
    displacements,
    flags,
  );

  return { grid, apply };
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
 * @param activeCount - Entries of the active-index list bound to the grid
 *   passes. Defaults to `nodeCount`, which is what the identity fallback list
 *   describes. It must come from the same field as the dispatch size passed to
 *   {@link recordGridCollisionPass}, or threads read past the end of the list.
 */
export function updateGridCollisionUniforms(
  device: GPUDevice,
  gridBuffers: GridCollisionBuffers,
  nodeCount: number,
  forceConfig: FullForceConfig,
  bounds: BoundingBox,
  maxRadius: number,
  activeCount?: number,
): void {
  if (nodeCount > gridBuffers.maxNodes) {
    throw new Error(
      `Grid collision buffer overflow: nodeCount (${nodeCount}) exceeds capacity (${gridBuffers.maxNodes}).`,
    );
  }

  const { gridWidth, gridHeight, cellSize } = computeGridDimensions(
    bounds,
    maxRadius,
    forceConfig.collisionRadiusMultiplier,
  );
  const totalCells = gridWidth * gridHeight;

  // GridCollisionUniforms (48 bytes, 16-byte aligned)
  const data = new ArrayBuffer(48);
  const view = new DataView(data);
  view.setUint32(0, nodeCount, true); // node_count
  view.setUint32(4, gridWidth, true); // grid_width
  view.setUint32(8, gridHeight, true); // grid_height
  view.setFloat32(12, cellSize, true); // cell_size
  view.setFloat32(16, bounds.minX, true); // bounds_min_x
  view.setFloat32(20, bounds.minY, true); // bounds_min_y
  view.setFloat32(24, forceConfig.collisionStrength, true); // collision_strength
  view.setFloat32(28, forceConfig.collisionRadiusMultiplier, true); // radius_multiplier
  view.setFloat32(32, DEFAULT_RADIUS, true); // default_radius
  view.setUint32(36, totalCells, true); // total_cells
  view.setUint32(40, activeCount ?? nodeCount, true); // active_count
  view.setUint32(44, 0, true); // _pad1
  device.queue.writeBuffer(gridBuffers.gridUniforms, 0, data);
}

/**
 * Records grid-based collision detection pass(es).
 *
 * Per iteration (4 GPU dispatches):
 * 1. clear_cells  — reset all cell head pointers to EMPTY
 * 2. build_lists  — each node atomically prepends itself to its cell's list
 * 3. resolve_grid — walk linked lists in 3x3 neighborhood for overlaps,
 *                   accumulating a per-node displacement
 * 4. apply        — add the displacements to the positions
 *
 * @param encoder - GPU command encoder
 * @param pipeline - Grid collision pipeline
 * @param bindGroups - Grid collision bind groups
 * @param gridBuffers - Grid collision buffers (for maxCells dispatch sizing)
 * @param nodeCount - Number of nodes
 * @param iterations - Number of collision resolution iterations
 * @param activeCount - Entries of the active-index list, sizing the build and
 *   resolve dispatches. Must be the same number given to
 *   {@link updateGridCollisionUniforms}. Defaults to `nodeCount`.
 */
export function recordGridCollisionPass(
  encoder: GPUCommandEncoder,
  pipeline: GridCollisionPipeline,
  bindGroups: GridCollisionBindGroups,
  gridBuffers: GridCollisionBuffers,
  nodeCount: number,
  iterations: number = 1,
  activeCount?: number,
): void {
  if (nodeCount < 2) {
    return;
  }

  // Build and resolve are indexed by ACTIVE-LIST ENTRY and shrink with the
  // cut; apply stays indexed by SLOT (see recordCollisionPass).
  const activeWorkgroups = calculateWorkgroups(activeCount ?? nodeCount, WORKGROUP_SIZE);
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
      pass.dispatchWorkgroups(activeWorkgroups);
      pass.end();
    }

    // Phase 3: Resolve collisions by walking 3x3 neighborhood lists
    {
      const pass = encoder.beginComputePass({
        label: `GridCollision Resolve ${iter + 1}/${iterations}`,
      });
      pass.setPipeline(pipeline.resolveGrid);
      pass.setBindGroup(0, bindGroups.grid);
      pass.dispatchWorkgroups(activeWorkgroups);
      pass.end();
    }

    // Phase 4: Fold the displacements into the positions. Slot-indexed and
    // wider than phase 3 on purpose: it drops every IMMOVABLE slot on the
    // flags, which is what makes the shortened resolve dispatch safe.
    {
      const pass = encoder.beginComputePass({
        label: `GridCollision Apply ${iter + 1}/${iterations}`,
      });
      pass.setPipeline(pipeline.apply);
      pass.setBindGroup(0, bindGroups.apply);
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
  buffers.fallbackNodeFlags.destroy();
  buffers.fallbackLiveIndices.destroy();
}
