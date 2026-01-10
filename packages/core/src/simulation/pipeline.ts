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
import INTEGRATE_SIMPLE_WGSL from "./shaders/integrate_simple.comp.wgsl";

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
 */
export interface SimulationBuffers {
  // Position buffers (ping-pong for integration)
  positionsX: GPUBuffer;
  positionsY: GPUBuffer;
  positionsXOut: GPUBuffer;
  positionsYOut: GPUBuffer;
  // Velocity buffers (ping-pong for integration)
  velocitiesX: GPUBuffer;
  velocitiesY: GPUBuffer;
  velocitiesXOut: GPUBuffer;
  velocitiesYOut: GPUBuffer;
  // Force accumulators
  forcesX: GPUBuffer;
  forcesY: GPUBuffer;
  // Edge data
  edgeSources: GPUBuffer;
  edgeTargets: GPUBuffer;
  // Uniform buffers for each stage
  clearUniforms: GPUBuffer;
  repulsionUniforms: GPUBuffer;
  springUniforms: GPUBuffer;
  integrationUniforms: GPUBuffer;
  // Readback buffers for syncing positions to CPU
  readbackX: GPUBuffer;
  readbackY: GPUBuffer;
  // Node count for readback sizing
  nodeCount: number;
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
  const integrateModule = createModule("Integration Shader", INTEGRATE_SIMPLE_WGSL);

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
 */
export function createSimulationBindGroups(
  device: GPUDevice,
  pipeline: SimulationPipeline,
  buffers: SimulationBuffers,
): SimulationBindGroups {
  // Clear forces bind group (bindings 0-2)
  const clearForces = device.createBindGroup({
    label: "Clear Forces Bind Group",
    layout: pipeline.pipelines.clearForces.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.clearUniforms } },
      { binding: 1, resource: { buffer: buffers.forcesX } },
      { binding: 2, resource: { buffer: buffers.forcesY } },
    ],
  });

  // Repulsion bind group (bindings 0-4)
  const repulsion = device.createBindGroup({
    label: "Repulsion Bind Group",
    layout: pipeline.pipelines.repulsion.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.repulsionUniforms } },
      { binding: 1, resource: { buffer: buffers.positionsX } },
      { binding: 2, resource: { buffer: buffers.positionsY } },
      { binding: 3, resource: { buffer: buffers.forcesX } },
      { binding: 4, resource: { buffer: buffers.forcesY } },
    ],
  });

  // Springs bind group (bindings 0-6)
  const springs = device.createBindGroup({
    label: "Springs Bind Group",
    layout: pipeline.pipelines.springs.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.springUniforms } },
      { binding: 1, resource: { buffer: buffers.positionsX } },
      { binding: 2, resource: { buffer: buffers.positionsY } },
      { binding: 3, resource: { buffer: buffers.forcesX } },
      { binding: 4, resource: { buffer: buffers.forcesY } },
      { binding: 5, resource: { buffer: buffers.edgeSources } },
      { binding: 6, resource: { buffer: buffers.edgeTargets } },
    ],
  });

  // Integration bind group (bindings 0-10)
  const integration = device.createBindGroup({
    label: "Integration Bind Group",
    layout: pipeline.pipelines.integrate.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.integrationUniforms } },
      { binding: 1, resource: { buffer: buffers.positionsX } },
      { binding: 2, resource: { buffer: buffers.positionsY } },
      { binding: 3, resource: { buffer: buffers.positionsXOut } },
      { binding: 4, resource: { buffer: buffers.positionsYOut } },
      { binding: 5, resource: { buffer: buffers.velocitiesX } },
      { binding: 6, resource: { buffer: buffers.velocitiesY } },
      { binding: 7, resource: { buffer: buffers.velocitiesXOut } },
      { binding: 8, resource: { buffer: buffers.velocitiesYOut } },
      { binding: 9, resource: { buffer: buffers.forcesX } },
      { binding: 10, resource: { buffer: buffers.forcesY } },
    ],
  });

  return { clearForces, repulsion, springs, integration };
}

/**
 * Create simulation buffers
 */
export function createSimulationBuffers(
  device: GPUDevice,
  nodeCount: number,
  edgeCount: number,
): SimulationBuffers {
  const nodeBytes = nodeCount * 4; // f32 = 4 bytes
  const edgeBytes = Math.max(edgeCount * 4, 4); // Minimum 4 bytes

  // Position buffers (ping-pong)
  const positionsX = device.createBuffer({
    label: "Sim Positions X",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const positionsY = device.createBuffer({
    label: "Sim Positions Y",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const positionsXOut = device.createBuffer({
    label: "Sim Positions X Out",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const positionsYOut = device.createBuffer({
    label: "Sim Positions Y Out",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  // Velocity buffers (ping-pong)
  const velocitiesX = device.createBuffer({
    label: "Sim Velocities X",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const velocitiesY = device.createBuffer({
    label: "Sim Velocities Y",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const velocitiesXOut = device.createBuffer({
    label: "Sim Velocities X Out",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const velocitiesYOut = device.createBuffer({
    label: "Sim Velocities Y Out",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Force accumulators
  const forcesX = device.createBuffer({
    label: "Sim Forces X",
    size: nodeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const forcesY = device.createBuffer({
    label: "Sim Forces Y",
    size: nodeBytes,
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

  // IntegrationUniforms: 16 bytes (4 f32/u32)
  const integrationUniforms = device.createBuffer({
    label: "Integration Uniforms",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Readback buffers for syncing positions to CPU
  const readbackX = device.createBuffer({
    label: "Position Readback X",
    size: nodeBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const readbackY = device.createBuffer({
    label: "Position Readback Y",
    size: nodeBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  return {
    positionsX,
    positionsY,
    positionsXOut,
    positionsYOut,
    velocitiesX,
    velocitiesY,
    velocitiesXOut,
    velocitiesYOut,
    forcesX,
    forcesY,
    edgeSources,
    edgeTargets,
    clearUniforms,
    repulsionUniforms,
    springUniforms,
    integrationUniforms,
    readbackX,
    readbackY,
    nodeCount,
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

  // IntegrationUniforms: { node_count, dt, damping, alpha }
  // Note: velocityDecay is the fraction lost per frame, damping is fraction retained
  // damping = 1 - velocityDecay
  const intData = new ArrayBuffer(16);
  const intView = new DataView(intData);
  intView.setUint32(0, nodeCount, true);
  intView.setFloat32(4, forceConfig.timeStep, true);
  intView.setFloat32(8, 1 - forceConfig.velocityDecay, true);
  intView.setFloat32(12, alpha, true);
  device.queue.writeBuffer(buffers.integrationUniforms, 0, intData);
}

/**
 * Copy initial positions to simulation buffers
 */
export function copyPositionsToSimulation(
  device: GPUDevice,
  buffers: SimulationBuffers,
  positionsX: Float32Array,
  positionsY: Float32Array,
): void {
  device.queue.writeBuffer(buffers.positionsX, 0, toArrayBuffer(positionsX));
  device.queue.writeBuffer(buffers.positionsY, 0, toArrayBuffer(positionsY));
  device.queue.writeBuffer(buffers.positionsXOut, 0, toArrayBuffer(positionsX));
  device.queue.writeBuffer(buffers.positionsYOut, 0, toArrayBuffer(positionsY));

  // Zero out velocities
  const zeros = new Float32Array(positionsX.length);
  device.queue.writeBuffer(buffers.velocitiesX, 0, toArrayBuffer(zeros));
  device.queue.writeBuffer(buffers.velocitiesY, 0, toArrayBuffer(zeros));
  device.queue.writeBuffer(buffers.velocitiesXOut, 0, toArrayBuffer(zeros));
  device.queue.writeBuffer(buffers.velocitiesYOut, 0, toArrayBuffer(zeros));
  device.queue.writeBuffer(buffers.forcesX, 0, toArrayBuffer(zeros));
  device.queue.writeBuffer(buffers.forcesY, 0, toArrayBuffer(zeros));
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
  const tempPosX = buffers.positionsX;
  const tempPosY = buffers.positionsY;
  buffers.positionsX = buffers.positionsXOut;
  buffers.positionsY = buffers.positionsYOut;
  buffers.positionsXOut = tempPosX;
  buffers.positionsYOut = tempPosY;

  // Swap velocity buffers
  const tempVelX = buffers.velocitiesX;
  const tempVelY = buffers.velocitiesY;
  buffers.velocitiesX = buffers.velocitiesXOut;
  buffers.velocitiesY = buffers.velocitiesYOut;
  buffers.velocitiesXOut = tempVelX;
  buffers.velocitiesYOut = tempVelY;
}

/**
 * Schedule a copy of positions to readback buffers.
 * Call this during command encoding, then call readbackPositions later.
 */
export function copyPositionsToReadback(
  encoder: GPUCommandEncoder,
  buffers: SimulationBuffers,
): void {
  const byteSize = buffers.nodeCount * 4;
  encoder.copyBufferToBuffer(buffers.positionsX, 0, buffers.readbackX, 0, byteSize);
  encoder.copyBufferToBuffer(buffers.positionsY, 0, buffers.readbackY, 0, byteSize);
}

/**
 * Read positions from GPU to CPU arrays.
 * This is async and causes a GPU pipeline stall - use sparingly.
 */
export async function readbackPositions(
  buffers: SimulationBuffers,
  targetX: Float32Array,
  targetY: Float32Array,
): Promise<void> {
  const nodeCount = buffers.nodeCount;
  const byteSize = nodeCount * 4;

  // Map and read X positions
  await buffers.readbackX.mapAsync(GPUMapMode.READ);
  const dataX = new Float32Array(buffers.readbackX.getMappedRange(0, byteSize));
  targetX.set(dataX);
  buffers.readbackX.unmap();

  // Map and read Y positions
  await buffers.readbackY.mapAsync(GPUMapMode.READ);
  const dataY = new Float32Array(buffers.readbackY.getMappedRange(0, byteSize));
  targetY.set(dataY);
  buffers.readbackY.unmap();
}
