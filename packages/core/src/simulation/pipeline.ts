/**
 * Simulation Compute Pipeline Orchestration
 *
 * Coordinates the GPU compute pipeline for force simulation:
 * 1. Clear forces
 * 2. N^2 repulsion forces
 * 3. Spring attractive forces
 * 4. Integration (position/velocity update)
 *
 * @module
 */

import type { GPUContext } from "../webgpu/context.ts";
import { calculateWorkgroups } from "../renderer/commands.ts";
import { toArrayBuffer } from "../webgpu/buffer_utils.ts";
import { DEFAULT_FORCE_CONFIG, type FullForceConfig } from "./config.ts";

// Import shader sources (bundled as text by esbuild)
import CLEAR_FORCES_WGSL from "./shaders/clear_forces.comp.wgsl";
import REPULSION_N2_WGSL from "./shaders/repulsion_n2.comp.wgsl";
import SPRINGS_SIMPLE_WGSL from "./shaders/springs_simple.comp.wgsl";
import INTEGRATE_WGSL from "./shaders/integrate.comp.wgsl";

/**
 * Simulation pipeline configuration
 */
export interface SimulationPipelineConfig {
  /** Maximum number of nodes */
  maxNodes?: number | undefined;
  /** Maximum number of edges */
  maxEdges?: number | undefined;
  /** Workgroup size for compute shaders */
  workgroupSize?: number | undefined;
}

/**
 * Default simulation pipeline configuration
 */
export const DEFAULT_SIMULATION_PIPELINE_CONFIG: Required<SimulationPipelineConfig> = {
  maxNodes: 1_000_000,
  maxEdges: 2_000_000,
  workgroupSize: 256,
};

/**
 * Simulation pipeline resources
 */
export interface SimulationPipeline {
  /** Compute pipelines for each stage */
  pipelines: {
    clearForces: GPUComputePipeline;
    repulsion: GPUComputePipeline;
    springs: GPUComputePipeline;
    integrate: GPUComputePipeline;
  };
  /** Pipeline configuration */
  config: Required<SimulationPipelineConfig>;
}

/**
 * Simulation buffers required for compute passes
 *
 * All position, velocity, force, and readback buffers use vec2<f32> layout
 * (8 bytes per node) for better memory access patterns and reduced binding count.
 */
export interface SimulationBuffers {
  // Position buffers (ping-pong for integration) - vec2<f32> per node
  positions: GPUBuffer;
  positionsOut: GPUBuffer;
  // Velocity buffers (ping-pong for integration) - vec2<f32> per node
  velocities: GPUBuffer;
  velocitiesOut: GPUBuffer;
  // Force accumulators - vec2<f32> per node
  forces: GPUBuffer;
  // Edge data
  edgeSources: GPUBuffer;
  edgeTargets: GPUBuffer;
  // Uniform buffers for each stage
  clearUniforms: GPUBuffer;
  repulsionUniforms: GPUBuffer;
  springUniforms: GPUBuffer;
  integrationUniforms: GPUBuffer;
  // Node state flags (for pinned nodes)
  nodeFlags: GPUBuffer;
  // Readback buffer for syncing positions to CPU - vec2<f32> per node
  readback: GPUBuffer;
  // Node count for readback sizing
  nodeCount: number;
  // Allocated capacity (may be larger than count for incremental mutations)
  nodeCapacity: number;
  edgeCapacity: number;
}

/**
 * Creates the simulation compute pipelines
 *
 * @param context - GPU context
 * @param config - Pipeline configuration
 * @returns Simulation pipeline resources
 */
export function createSimulationPipeline(
  context: GPUContext,
  config: SimulationPipelineConfig = {},
): SimulationPipeline {
  const { device } = context;
  const finalConfig = { ...DEFAULT_SIMULATION_PIPELINE_CONFIG, ...config };

  // Helper to create shader module
  const createModule = (label: string, code: string): GPUShaderModule => {
    return device.createShaderModule({ label, code });
  };

  // Create shader modules - each module now has only bindings it needs
  const clearForcesModule = createModule("Clear Forces Shader", CLEAR_FORCES_WGSL);
  const repulsionModule = createModule("N^2 Repulsion Shader", REPULSION_N2_WGSL);
  const springsModule = createModule("Springs Shader", SPRINGS_SIMPLE_WGSL);
  const integrateModule = createModule("Integration Shader", INTEGRATE_WGSL);

  // Create compute pipelines with auto layout
  // Each shader module now only declares bindings it uses, so auto layout works correctly
  const clearForcesPipeline = device.createComputePipeline({
    label: "Clear Forces Pipeline",
    layout: "auto",
    compute: {
      module: clearForcesModule,
      entryPoint: "main",
    },
  });

  const repulsionPipeline = device.createComputePipeline({
    label: "Repulsion Pipeline (N^2)",
    layout: "auto",
    compute: {
      module: repulsionModule,
      entryPoint: "main",
    },
  });

  const springsPipeline = device.createComputePipeline({
    label: "Springs Pipeline",
    layout: "auto",
    compute: {
      module: springsModule,
      entryPoint: "main",
    },
  });

  const integratePipeline = device.createComputePipeline({
    label: "Integration Pipeline",
    layout: "auto",
    compute: {
      module: integrateModule,
      entryPoint: "main",
    },
  });

  return {
    pipelines: {
      clearForces: clearForcesPipeline,
      repulsion: repulsionPipeline,
      springs: springsPipeline,
      integrate: integratePipeline,
    },
    config: finalConfig,
  };
}

/**
 * Records a simulation step to a command encoder
 *
 * @param encoder - Command encoder
 * @param pipeline - Simulation pipeline
 * @param bindGroups - Pre-created bind groups for each stage
 * @param nodeCount - Number of nodes
 * @param edgeCount - Number of edges
 */
export function recordSimulationStep(
  encoder: GPUCommandEncoder,
  pipeline: SimulationPipeline,
  bindGroups: SimulationBindGroups,
  nodeCount: number,
  edgeCount: number,
): void {
  const workgroupSize = pipeline.config.workgroupSize;
  const nodeWorkgroups = calculateWorkgroups(nodeCount, workgroupSize);
  const edgeWorkgroups = calculateWorkgroups(edgeCount, workgroupSize);

  // Stage 1: Clear forces
  const clearPass = encoder.beginComputePass({ label: "Clear Forces" });
  clearPass.setPipeline(pipeline.pipelines.clearForces);
  clearPass.setBindGroup(0, bindGroups.clearForces);
  clearPass.dispatchWorkgroups(nodeWorkgroups);
  clearPass.end();

  // Stage 2: Compute repulsion forces (N^2)
  const repulsionPass = encoder.beginComputePass({ label: "N^2 Repulsion" });
  repulsionPass.setPipeline(pipeline.pipelines.repulsion);
  repulsionPass.setBindGroup(0, bindGroups.repulsion);
  repulsionPass.dispatchWorkgroups(nodeWorkgroups);
  repulsionPass.end();

  // Stage 3: Compute spring forces
  if (edgeCount > 0) {
    const springsPass = encoder.beginComputePass({ label: "Springs" });
    springsPass.setPipeline(pipeline.pipelines.springs);
    springsPass.setBindGroup(0, bindGroups.springs);
    springsPass.dispatchWorkgroups(edgeWorkgroups);
    springsPass.end();
  }

  // Stage 4: Integration
  const integratePass = encoder.beginComputePass({ label: "Integration" });
  integratePass.setPipeline(pipeline.pipelines.integrate);
  integratePass.setBindGroup(0, bindGroups.integration);
  integratePass.dispatchWorkgroups(nodeWorkgroups);
  integratePass.end();
}

/**
 * Options for recording simulation step with custom algorithm
 */
export interface RecordSimulationOptions {
  /** Custom repulsion pass recorder (replaces default N² repulsion) */
  recordRepulsionPass?: ((encoder: GPUCommandEncoder) => void) | undefined;
}

/**
 * Records a simulation step with optional custom algorithm for repulsion
 *
 * @param encoder - Command encoder
 * @param pipeline - Simulation pipeline
 * @param bindGroups - Pre-created bind groups for each stage
 * @param nodeCount - Number of nodes
 * @param edgeCount - Number of edges
 * @param options - Optional configuration including custom repulsion
 */
export function recordSimulationStepWithOptions(
  encoder: GPUCommandEncoder,
  pipeline: SimulationPipeline,
  bindGroups: SimulationBindGroups,
  nodeCount: number,
  edgeCount: number,
  options: RecordSimulationOptions = {},
): void {
  const workgroupSize = pipeline.config.workgroupSize;
  const nodeWorkgroups = calculateWorkgroups(nodeCount, workgroupSize);
  const edgeWorkgroups = calculateWorkgroups(edgeCount, workgroupSize);

  // Stage 1: Clear forces
  const clearPass = encoder.beginComputePass({ label: "Clear Forces" });
  clearPass.setPipeline(pipeline.pipelines.clearForces);
  clearPass.setBindGroup(0, bindGroups.clearForces);
  clearPass.dispatchWorkgroups(nodeWorkgroups);
  clearPass.end();

  // Stage 2: Compute repulsion forces (custom algorithm or default N²)
  if (options.recordRepulsionPass) {
    options.recordRepulsionPass(encoder);
  } else {
    const repulsionPass = encoder.beginComputePass({ label: "N^2 Repulsion" });
    repulsionPass.setPipeline(pipeline.pipelines.repulsion);
    repulsionPass.setBindGroup(0, bindGroups.repulsion);
    repulsionPass.dispatchWorkgroups(nodeWorkgroups);
    repulsionPass.end();
  }

  // Stage 3: Compute spring forces
  if (edgeCount > 0) {
    const springsPass = encoder.beginComputePass({ label: "Springs" });
    springsPass.setPipeline(pipeline.pipelines.springs);
    springsPass.setBindGroup(0, bindGroups.springs);
    springsPass.dispatchWorkgroups(edgeWorkgroups);
    springsPass.end();
  }

  // Stage 4: Integration
  const integratePass = encoder.beginComputePass({ label: "Integration" });
  integratePass.setPipeline(pipeline.pipelines.integrate);
  integratePass.setBindGroup(0, bindGroups.integration);
  integratePass.dispatchWorkgroups(nodeWorkgroups);
  integratePass.end();
}

/**
 * Bind groups for simulation stages
 */
export interface SimulationBindGroups {
  clearForces: GPUBindGroup;
  repulsion: GPUBindGroup;
  springs: GPUBindGroup;
  integration: GPUBindGroup;
}

/**
 * Creates simulation bind groups from buffers
 * Uses getBindGroupLayout(0) to get the auto-inferred layout from each pipeline
 *
 * All buffers use vec2<f32> layout for consolidated X/Y data.
 */
export function createSimulationBindGroups(
  device: GPUDevice,
  pipeline: SimulationPipeline,
  buffers: SimulationBuffers,
): SimulationBindGroups {
  // Clear forces bind group (bindings 0-1)
  const clearForces = device.createBindGroup({
    label: "Clear Forces Bind Group",
    layout: pipeline.pipelines.clearForces.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.clearUniforms } },
      { binding: 1, resource: { buffer: buffers.forces } },
    ],
  });

  // Repulsion bind group (bindings 0-2)
  const repulsion = device.createBindGroup({
    label: "Repulsion Bind Group",
    layout: pipeline.pipelines.repulsion.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.repulsionUniforms } },
      { binding: 1, resource: { buffer: buffers.positions } },
      { binding: 2, resource: { buffer: buffers.forces } },
    ],
  });

  // Springs bind group (bindings 0-4)
  const springs = device.createBindGroup({
    label: "Springs Bind Group",
    layout: pipeline.pipelines.springs.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.springUniforms } },
      { binding: 1, resource: { buffer: buffers.positions } },
      { binding: 2, resource: { buffer: buffers.forces } },
      { binding: 3, resource: { buffer: buffers.edgeSources } },
      { binding: 4, resource: { buffer: buffers.edgeTargets } },
    ],
  });

  // Integration bind group (bindings 0-5)
  // With vec2 consolidation, we now have room for node_flags if needed
  const integration = device.createBindGroup({
    label: "Integration Bind Group",
    layout: pipeline.pipelines.integrate.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.integrationUniforms } },
      { binding: 1, resource: { buffer: buffers.positions } },
      { binding: 2, resource: { buffer: buffers.positionsOut } },
      { binding: 3, resource: { buffer: buffers.velocities } },
      { binding: 4, resource: { buffer: buffers.velocitiesOut } },
      { binding: 5, resource: { buffer: buffers.forces } },
    ],
  });

  return { clearForces, repulsion, springs, integration };
}

/**
 * Create simulation buffers
 *
 * All position, velocity, force, and readback buffers use vec2<f32> layout
 * (8 bytes per node) for consolidated X/Y data.
 *
 * @param nodeCapacity - GPU buffer capacity for nodes (defaults to nodeCount).
 *                       Set larger than nodeCount to enable incremental additions.
 * @param edgeCapacity - GPU buffer capacity for edges (defaults to edgeCount).
 */
export function createSimulationBuffers(
  device: GPUDevice,
  nodeCount: number,
  edgeCount: number,
  nodeCapacity?: number,
  edgeCapacity?: number,
): SimulationBuffers {
  const effectiveNodeCap = Math.max(nodeCapacity ?? nodeCount, nodeCount);
  const effectiveEdgeCap = Math.max(edgeCapacity ?? edgeCount, edgeCount);

  const nodeVec2Bytes = effectiveNodeCap * 8; // vec2<f32> = 8 bytes
  const nodeFlagBytes = effectiveNodeCap * 4; // u32 = 4 bytes
  const edgeBytes = Math.max(effectiveEdgeCap * 4, 4); // Minimum 4 bytes

  // Position buffers (ping-pong) - vec2<f32> per node
  const positions = device.createBuffer({
    label: "Sim Positions",
    size: nodeVec2Bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const positionsOut = device.createBuffer({
    label: "Sim Positions Out",
    size: nodeVec2Bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  // Velocity buffers (ping-pong) - vec2<f32> per node
  const velocities = device.createBuffer({
    label: "Sim Velocities",
    size: nodeVec2Bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const velocitiesOut = device.createBuffer({
    label: "Sim Velocities Out",
    size: nodeVec2Bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Force accumulators - vec2<f32> per node
  const forces = device.createBuffer({
    label: "Sim Forces",
    size: nodeVec2Bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Edge data
  const edgeSources = device.createBuffer({
    label: "Sim Edge Sources",
    size: edgeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const edgeTargets = device.createBuffer({
    label: "Sim Edge Targets",
    size: edgeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Uniform buffers (aligned to 16 bytes)
  // ClearUniforms: 16 bytes (node_count u32 + padding)
  const clearUniforms = device.createBuffer({
    label: "Clear Uniforms",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // RepulsionUniforms: 16 bytes (node_count + 2 f32 + padding)
  const repulsionUniforms = device.createBuffer({
    label: "Repulsion Uniforms",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // SpringUniforms: 16 bytes (edge_count + 2 f32 + padding)
  const springUniforms = device.createBuffer({
    label: "Spring Uniforms",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // IntegrationUniforms: 48 bytes for full shader
  // (node_count, dt, damping, max_velocity, alpha, alpha_decay, alpha_min,
  //  gravity_strength, center_x, center_y, padding)
  const integrationUniforms = device.createBuffer({
    label: "Integration Uniforms",
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Node state flags (for pinned nodes, etc.)
  const nodeFlags = device.createBuffer({
    label: "Node Flags",
    size: nodeFlagBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Readback buffer for syncing positions to CPU - vec2<f32> per node
  const readback = device.createBuffer({
    label: "Position Readback",
    size: nodeVec2Bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  return {
    positions,
    positionsOut,
    velocities,
    velocitiesOut,
    forces,
    edgeSources,
    edgeTargets,
    clearUniforms,
    repulsionUniforms,
    springUniforms,
    integrationUniforms,
    nodeFlags,
    readback,
    nodeCount,
    nodeCapacity: effectiveNodeCap,
    edgeCapacity: effectiveEdgeCap,
  };
}

/**
 * Update simulation uniform buffers
 *
 * @param device - GPU device
 * @param buffers - Simulation buffers
 * @param nodeCount - Number of nodes
 * @param edgeCount - Number of edges
 * @param alpha - Simulation temperature (0-1)
 * @param forceConfig - Force configuration (optional, uses defaults if not provided)
 */
export function updateSimulationUniforms(
  device: GPUDevice,
  buffers: SimulationBuffers,
  nodeCount: number,
  edgeCount: number,
  alpha: number,
  forceConfig: FullForceConfig = DEFAULT_FORCE_CONFIG,
): void {
  // ClearUniforms: { node_count: u32 }
  const clearData = new ArrayBuffer(16);
  const clearView = new DataView(clearData);
  clearView.setUint32(0, nodeCount, true);
  device.queue.writeBuffer(buffers.clearUniforms, 0, clearData);

  // RepulsionUniforms: { node_count, repulsion_strength, min_distance, _padding }
  // Note: repulsionStrength is negative in config (d3 convention), shader uses positive
  const repulsionData = new ArrayBuffer(16);
  const repulsionView = new DataView(repulsionData);
  repulsionView.setUint32(0, nodeCount, true);
  repulsionView.setFloat32(4, Math.abs(forceConfig.repulsionStrength), true);
  repulsionView.setFloat32(8, forceConfig.repulsionDistanceMin, true);
  repulsionView.setUint32(12, 0, true); // padding
  device.queue.writeBuffer(buffers.repulsionUniforms, 0, repulsionData);

  // SpringUniforms: { edge_count, spring_strength, rest_length, _padding }
  const springData = new ArrayBuffer(16);
  const springView = new DataView(springData);
  springView.setUint32(0, edgeCount, true);
  springView.setFloat32(4, forceConfig.springStrength, true);
  springView.setFloat32(8, forceConfig.springLength, true);
  springView.setUint32(12, 0, true); // padding
  device.queue.writeBuffer(buffers.springUniforms, 0, springData);

  // IntegrationUniforms: full 48-byte struct
  // {
  //   node_count: u32,         // offset 0
  //   dt: f32,                 // offset 4
  //   damping: f32,            // offset 8
  //   max_velocity: f32,       // offset 12
  //   alpha: f32,              // offset 16
  //   alpha_decay: f32,        // offset 20
  //   alpha_min: f32,          // offset 24
  //   gravity_strength: f32,   // offset 28
  //   center_x: f32,           // offset 32
  //   center_y: f32,           // offset 36
  //   _padding: vec2<u32>,     // offset 40 (8 bytes)
  // }
  // Note: velocityDecay is the fraction lost per frame, damping is fraction retained
  // damping = 1 - velocityDecay
  const intData = new ArrayBuffer(48);
  const intView = new DataView(intData);
  intView.setUint32(0, nodeCount, true);                          // node_count
  intView.setFloat32(4, forceConfig.timeStep, true);              // dt
  intView.setFloat32(8, 1 - forceConfig.velocityDecay, true);     // damping
  intView.setFloat32(12, forceConfig.maxVelocity, true);          // max_velocity
  intView.setFloat32(16, alpha, true);                            // alpha
  intView.setFloat32(20, 0.0228, true);                           // alpha_decay (default)
  intView.setFloat32(24, 0.001, true);                            // alpha_min (default)
  intView.setFloat32(28, forceConfig.centerStrength, true);       // gravity_strength
  intView.setFloat32(32, forceConfig.centerX, true);              // center_x
  intView.setFloat32(36, forceConfig.centerY, true);              // center_y
  intView.setUint32(40, 0, true);                                 // padding[0]
  intView.setUint32(44, 0, true);                                 // padding[1]
  device.queue.writeBuffer(buffers.integrationUniforms, 0, intData);
}

/**
 * Copy initial positions to simulation buffers
 *
 * Accepts separate X/Y arrays for API compatibility and interleaves them
 * into vec2<f32> format for GPU buffers.
 */
export function copyPositionsToSimulation(
  device: GPUDevice,
  buffers: SimulationBuffers,
  positionsX: Float32Array,
  positionsY: Float32Array,
): void {
  const nodeCount = positionsX.length;

  // Interleave X/Y into vec2<f32> format
  const positionsVec2 = new Float32Array(nodeCount * 2);
  for (let i = 0; i < nodeCount; i++) {
    positionsVec2[i * 2] = positionsX[i];
    positionsVec2[i * 2 + 1] = positionsY[i];
  }

  device.queue.writeBuffer(buffers.positions, 0, toArrayBuffer(positionsVec2));
  device.queue.writeBuffer(buffers.positionsOut, 0, toArrayBuffer(positionsVec2));

  // Zero out velocities and forces (vec2<f32> per node)
  const zerosVec2 = new Float32Array(nodeCount * 2);
  device.queue.writeBuffer(buffers.velocities, 0, toArrayBuffer(zerosVec2));
  device.queue.writeBuffer(buffers.velocitiesOut, 0, toArrayBuffer(zerosVec2));
  device.queue.writeBuffer(buffers.forces, 0, toArrayBuffer(zerosVec2));

  // Initialize node flags to 0 (all unpinned, visible)
  const zeroFlags = new Uint32Array(nodeCount);
  device.queue.writeBuffer(buffers.nodeFlags, 0, toArrayBuffer(zeroFlags));
}

/**
 * Copy edge data to simulation buffers
 */
export function copyEdgesToSimulation(
  device: GPUDevice,
  buffers: SimulationBuffers,
  edgeSources: Uint32Array,
  edgeTargets: Uint32Array,
): void {
  device.queue.writeBuffer(buffers.edgeSources, 0, toArrayBuffer(edgeSources));
  device.queue.writeBuffer(buffers.edgeTargets, 0, toArrayBuffer(edgeTargets));
}

/**
 * Swap ping-pong buffers after integration
 */
export function swapSimulationBuffers(buffers: SimulationBuffers): void {
  // Swap position buffers
  const tempPos = buffers.positions;
  buffers.positions = buffers.positionsOut;
  buffers.positionsOut = tempPos;

  // Swap velocity buffers
  const tempVel = buffers.velocities;
  buffers.velocities = buffers.velocitiesOut;
  buffers.velocitiesOut = tempVel;
}

/**
 * Schedule a copy of positions to readback buffer.
 * Call this during command encoding, then call readbackPositions later.
 */
export function copyPositionsToReadback(
  encoder: GPUCommandEncoder,
  buffers: SimulationBuffers,
): void {
  const byteSize = buffers.nodeCount * 8; // vec2<f32> = 8 bytes per node
  encoder.copyBufferToBuffer(buffers.positions, 0, buffers.readback, 0, byteSize);
}

/**
 * Read positions from GPU to CPU arrays.
 * This is async and causes a GPU pipeline stall - use sparingly.
 *
 * Accepts separate X/Y arrays for API compatibility and de-interleaves
 * the vec2<f32> data from the GPU buffer.
 */
export async function readbackPositions(
  buffers: SimulationBuffers,
  targetX: Float32Array,
  targetY: Float32Array,
): Promise<void> {
  const nodeCount = buffers.nodeCount;
  const byteSize = nodeCount * 8; // vec2<f32> = 8 bytes per node

  // Map and read interleaved positions
  await buffers.readback.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(buffers.readback.getMappedRange(0, byteSize));

  // De-interleave into separate X/Y arrays
  for (let i = 0; i < nodeCount; i++) {
    targetX[i] = data[i * 2];
    targetY[i] = data[i * 2 + 1];
  }

  buffers.readback.unmap();
}
