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
import { ErrorCode, GraphMotherError } from "../errors.ts";
import { calculateWorkgroups } from "../renderer/commands.ts";
import { toArrayBuffer } from "../webgpu/buffer_utils.ts";
import { NODE_ALPHA_OPAQUE } from "../lod/crossfade.ts";
import { NODE_MASS_UNIT } from "../lod/mass.ts";
import { DEFAULT_FORCE_CONFIG, type FullForceConfig } from "./config.ts";

// --- Settling telemetry (temporary, for tuning decay curves) ---
interface SettlingFrame {
  frame: number;
  alpha: number;
  damping: number;
  effectiveAlpha: Record<string, number>; // by depth level
}

let _settlingLog: SettlingFrame[] = [];
let _settlingFrame = 0;
let _settlingActive = false;
const SETTLING_SAMPLE_DEPTHS = [0, 1, 2, 3, 5, 8];

export function startSettlingTelemetry(): void {
  _settlingLog = [];
  _settlingFrame = 0;
  _settlingActive = true;
  console.log("[settling] Telemetry started. Will dump on simulation stop.");
}

export function dumpSettlingTelemetry(): void {
  if (_settlingLog.length === 0) {
    console.log("[settling] No data captured.");
    return;
  }
  // Print a compact table to console
  const header = ["frame", "alpha", "damping", ...SETTLING_SAMPLE_DEPTHS.map((d) => `d${d}`)];
  console.log("[settling] " + header.join("\t"));
  for (const f of _settlingLog) {
    const row = [
      f.frame,
      f.alpha.toFixed(4),
      f.damping.toFixed(4),
      ...SETTLING_SAMPLE_DEPTHS.map((d) => f.effectiveAlpha[`d${d}`].toFixed(4)),
    ];
    console.log("[settling] " + row.join("\t"));
  }
  console.log(`[settling] ${_settlingLog.length} samples captured over ${_settlingFrame} frames.`);
  _settlingActive = false;
}

function recordSettlingFrame(alpha: number, damping: number, spread: number): void {
  if (!_settlingActive) return;
  // Sample every 5 frames to keep output manageable
  if (_settlingFrame % 5 === 0) {
    const effectiveAlpha: Record<string, number> = {};
    for (const d of SETTLING_SAMPLE_DEPTHS) {
      effectiveAlpha[`d${d}`] = Math.min(alpha * (1 + d * spread), 1.0);
    }
    _settlingLog.push({
      frame: _settlingFrame,
      alpha,
      damping,
      effectiveAlpha,
    });
  }
  _settlingFrame++;
  // Auto-dump when alpha gets very low (simulation essentially done)
  if (_settlingActive && alpha < 0.0005 && _settlingFrame > 10) {
    dumpSettlingTelemetry();
  }
}
// --- End settling telemetry ---

// Import shader sources (bundled as text by esbuild)
import CLEAR_FORCES_WGSL from "./shaders/clear_forces.comp.wgsl";
import REPULSION_N2_WGSL from "./shaders/repulsion_n2.comp.wgsl";
import SPRINGS_SIMPLE_WGSL from "./shaders/springs_simple.comp.wgsl";
import INTEGRATE_WGSL from "./shaders/integrate.comp.wgsl";

/**
 * nodeFlags bit 0: the slot is a hole left by a removal. Simulation shaders
 * (repulsion, springs, integration) skip dead slots entirely — they neither
 * exert nor receive forces and are never integrated.
 */
export const NODE_FLAG_DEAD = 1;

/**
 * nodeFlags bit 1: the node is pinned (pinNode / active drag). The integrate
 * shader carries its position through the ping-pong unchanged and zeroes its
 * velocity, so external writes to the position buffers hold. Pinned nodes
 * still exert repulsion and spring forces on their neighbors.
 */
export const NODE_FLAG_PINNED = 2;

/**
 * nodeFlags bit 2: the node is hidden (setNodeVisibility, and later the
 * semantic-LOD cut). The node render pipeline drops the instance; the
 * simulation is deliberately unaffected, so a hidden node keeps its place in
 * the layout and reappears where it belongs.
 *
 * Distinct from {@link NODE_FLAG_DEAD} because a hidden node is still a live
 * slot: reusing the dead bit would corrupt the free-list bookkeeping that
 * nodeFreeList/nodeFreeSet and the WASM engine keep in lockstep.
 */
export const NODE_FLAG_HIDDEN_LOD = 4;

/**
 * Sentinel written into the collision node_sizes buffer for dead slots.
 * Collision shaders treat any negative radius as "slot does not exist"
 * (0 means "use default radius", so it cannot mark dead slots).
 */
export const DEAD_SLOT_RADIUS = -1;

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
  // Previous tick's total force - vec2<f32> per node (adaptive speed swing/traction)
  prevForces: GPUBuffer;
  // Edge data
  edgeSources: GPUBuffer;
  edgeTargets: GPUBuffer;
  // Uniform buffers for each stage
  clearUniforms: GPUBuffer;
  repulsionUniforms: GPUBuffer;
  springUniforms: GPUBuffer;
  integrationUniforms: GPUBuffer;
  // Node state flags (u32 per node, bit 0 = dead slot, bit 1 = pinned,
  // bit 2 = hidden; see NODE_FLAG_DEAD / NODE_FLAG_PINNED / NODE_FLAG_HIDDEN_LOD)
  nodeFlags: GPUBuffer;
  // Per-node render alpha (f32 per node, 1.0 = opaque). Written only by the
  // crossfade scheduler (lod/crossfade.ts) and read only by the node render
  // pipeline; the simulation never touches it. See NODE_ALPHA_OPAQUE.
  nodeAlpha: GPUBuffer;
  // Per-node simulation mass (f32 per node, 1.0 = one body). Shared by every
  // repulsion path so a collapsed LOD subtree's proxy repels like the subtree
  // it replaces. Not ping-ponged: a collapse rewrites contents only, so bind
  // groups referencing it never go stale. See NODE_MASS_UNIT / rollUpMass.
  nodeMass: GPUBuffer;
  // Node depth from root (f32 per node) for hierarchical settling
  nodeDepth: GPUBuffer;
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
      // Dead-slot-masked variant; plain "main" is used by the N² algorithm
      // plugin whose bind group has no node_flags buffer
      entryPoint: "main_masked",
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
  /** Skip the edge spring pass (for algorithms that provide their own positioning) */
  skipSprings?: boolean;
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

  // Stage 3: Compute spring forces (skip for algorithms that handle their own positioning)
  if (edgeCount > 0 && !options.skipSprings) {
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

  // Repulsion bind group (bindings 0-4)
  const repulsion = device.createBindGroup({
    label: "Repulsion Bind Group",
    layout: pipeline.pipelines.repulsion.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.repulsionUniforms } },
      { binding: 1, resource: { buffer: buffers.positions } },
      { binding: 2, resource: { buffer: buffers.forces } },
      { binding: 3, resource: { buffer: buffers.nodeFlags } },
      { binding: 4, resource: { buffer: buffers.nodeMass } },
    ],
  });

  // Springs bind group (bindings 0-5)
  const springs = device.createBindGroup({
    label: "Springs Bind Group",
    layout: pipeline.pipelines.springs.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.springUniforms } },
      { binding: 1, resource: { buffer: buffers.positions } },
      { binding: 2, resource: { buffer: buffers.forces } },
      { binding: 3, resource: { buffer: buffers.edgeSources } },
      { binding: 4, resource: { buffer: buffers.edgeTargets } },
      { binding: 5, resource: { buffer: buffers.nodeFlags } },
    ],
  });

  // Integration bind group (bindings 0-7)
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
      { binding: 6, resource: { buffer: buffers.nodeDepth } },
      { binding: 7, resource: { buffer: buffers.nodeFlags } },
      { binding: 8, resource: { buffer: buffers.prevForces } },
    ],
  });

  return { clearForces, repulsion, springs, integration };
}

/**
 * Fill a node-alpha buffer with {@link NODE_ALPHA_OPAQUE} over `[0, nodeCount)`.
 *
 * Every path that creates or grows the alpha buffer must run this: a fresh GPU
 * buffer is zero-filled, and zero alpha renders nothing.
 */
export function writeOpaqueNodeAlpha(
  device: GPUDevice,
  nodeAlpha: GPUBuffer,
  nodeCount: number,
): void {
  // Deno's WebGPU backend panics on a zero-length write.
  if (nodeCount <= 0) return;
  const opaque = new Float32Array(nodeCount).fill(NODE_ALPHA_OPAQUE);
  device.queue.writeBuffer(nodeAlpha, 0, toArrayBuffer(opaque));
}

/**
 * Fill a node-mass buffer with {@link NODE_MASS_UNIT} over `[0, nodeCount)`.
 *
 * Every path that creates or grows the mass buffer must run this: a fresh GPU
 * buffer is zero-filled, and zero mass means no repulsion at all.
 */
export function writeUnitNodeMass(
  device: GPUDevice,
  nodeMass: GPUBuffer,
  nodeCount: number,
): void {
  // Deno's WebGPU backend panics on a zero-length write.
  if (nodeCount <= 0) return;
  const unit = new Float32Array(nodeCount).fill(NODE_MASS_UNIT);
  device.queue.writeBuffer(nodeMass, 0, toArrayBuffer(unit));
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

  // Previous tick's total force - vec2<f32> per node (adaptive speed)
  const prevForces = device.createBuffer({
    label: "Sim Prev Forces",
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

  // Node state flags (dead-slot masking; zero-initialized = all live)
  const nodeFlags = device.createBuffer({
    label: "Node Flags",
    size: nodeFlagBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Per-node render alpha, read only by the node render pipeline. A fresh GPU
  // buffer is zero-filled, which would render nothing, so it is explicitly
  // initialized to fully opaque: with no crossfade in flight the multiply in
  // node.frag.wgsl is bit-exactly the identity. COPY_SRC is not needed by the
  // renderer — it exists so the contents can be read back and asserted on
  // (tests/gpu/node_alpha_test.ts), the same allowance
  // layers/stream_intensity.ts makes.
  const nodeAlpha = device.createBuffer({
    label: "Node Alpha",
    size: nodeFlagBytes, // f32 per node = same size as flags
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  writeOpaqueNodeAlpha(device, nodeAlpha, effectiveNodeCap);

  // Per-node simulation mass, read by every repulsion path. Allocated at
  // capacity here so the existing reallocation path already covers it and no
  // LOD transition ever changes a buffer identity — a collapse is a contents
  // write, which never invalidates a bind group. COPY_SRC is for test readback
  // only, the same allowance nodeAlpha makes.
  const nodeMass = device.createBuffer({
    label: "Node Mass",
    size: nodeFlagBytes, // f32 per node = same size as flags
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  writeUnitNodeMass(device, nodeMass, effectiveNodeCap);

  // Node depth from root (f32 per node) for hierarchical settling.
  // Zero-initialized: non-hierarchical algorithms get depth=0 (no effect).
  const nodeDepth = device.createBuffer({
    label: "Node Depth",
    size: nodeFlagBytes, // f32 per node = same size as flags
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
    prevForces,
    edgeSources,
    edgeTargets,
    clearUniforms,
    repulsionUniforms,
    springUniforms,
    integrationUniforms,
    nodeFlags,
    nodeAlpha,
    nodeMass,
    nodeDepth,
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
 * @param adaptiveSpeedOverride - Per-algorithm override for adaptive speed
 *   (e.g. from ForceAlgorithm.prefersAdaptiveSpeed). When set, wins over
 *   forceConfig.adaptiveSpeed.
 */
export function updateSimulationUniforms(
  device: GPUDevice,
  buffers: SimulationBuffers,
  nodeCount: number,
  edgeCount: number,
  alpha: number,
  forceConfig: FullForceConfig = DEFAULT_FORCE_CONFIG,
  adaptiveSpeedOverride?: boolean,
): void {
  // ClearUniforms: { node_count: u32 }
  const clearData = new ArrayBuffer(16);
  const clearView = new DataView(clearData);
  clearView.setUint32(0, nodeCount, true);
  device.queue.writeBuffer(buffers.clearUniforms, 0, clearData);

  // RepulsionUniforms: { node_count, repulsion_strength, min_distance, max_distance }
  // Note: repulsionStrength is negative in config (d3 convention), shader uses positive
  // max_distance = 0 means no limit (shader checks for > 0 before applying cutoff)
  const repulsionData = new ArrayBuffer(16);
  const repulsionView = new DataView(repulsionData);
  repulsionView.setUint32(0, nodeCount, true);
  repulsionView.setFloat32(4, Math.abs(forceConfig.repulsionStrength), true);
  repulsionView.setFloat32(8, forceConfig.repulsionDistanceMin, true);
  repulsionView.setFloat32(12, forceConfig.repulsionDistanceMax, true);
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
  //   depth_settling_spread: f32, // offset 20 (parents settle before children)
  //   alpha_min: f32,          // offset 24
  //   gravity_strength: f32,   // offset 28
  //   center_x: f32,           // offset 32
  //   center_y: f32,           // offset 36
  //   pinned_node: u32,        // offset 40
  //   adaptive_speed: f32,     // offset 44 (0 = disabled)
  // }
  // Note: velocityDecay is the fraction lost per frame, damping is fraction retained
  // damping = 1 - velocityDecay
  //
  // Progressive damping: as simulation cools (alpha→0), boost velocity drain
  // so nodes decelerate smoothly instead of coasting on residual momentum.
  // At alpha=1 (hot): baseDamping unchanged. At alpha=0 (cold): extra 12% drain.
  const baseDamping = 1 - forceConfig.velocityDecay;
  const progressiveBoost = (1 - Math.min(1, Math.max(0, alpha))) * 0.12;
  const effectiveDamping = Math.max(0.05, baseDamping - progressiveBoost);

  const intData = new ArrayBuffer(48);
  const intView = new DataView(intData);
  intView.setUint32(0, nodeCount, true); // node_count
  intView.setFloat32(4, forceConfig.timeStep, true); // dt
  intView.setFloat32(8, effectiveDamping, true); // damping (progressive)
  intView.setFloat32(12, forceConfig.maxVelocity, true); // max_velocity
  intView.setFloat32(16, alpha, true); // alpha
  intView.setFloat32(20, forceConfig.depthSettlingSpread, true); // depth_settling_spread (parents settle before children)
  intView.setFloat32(24, 0.0, true); // alpha_min (unused by shader — convergence managed on CPU)
  intView.setFloat32(28, forceConfig.centerStrength, true); // gravity_strength
  intView.setFloat32(32, forceConfig.centerX, true); // center_x
  intView.setFloat32(36, forceConfig.centerY, true); // center_y
  intView.setUint32(40, forceConfig.pinnedNode >>> 0, true); // pinned_node (0xFFFFFFFF = none)
  const adaptiveOn = adaptiveSpeedOverride ?? forceConfig.adaptiveSpeed;
  intView.setFloat32(44, adaptiveOn ? forceConfig.adaptiveSpeedStrength : 0, true); // adaptive_speed
  device.queue.writeBuffer(buffers.integrationUniforms, 0, intData);

  // Record settling telemetry
  recordSettlingFrame(alpha, effectiveDamping, forceConfig.depthSettlingSpread);
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
  device.queue.writeBuffer(buffers.prevForces, 0, toArrayBuffer(zerosVec2));

  // Initialize node flags to 0 (all slots live)
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
 * Which of the two ping-pong orientations the position/velocity buffers are
 * currently in. Only the parity of the swap count matters — two swaps restore
 * the original assignment — so every bind group that references a ping-pong
 * buffer has exactly two possible forms.
 */
export type BufferParity = 0 | 1;

/**
 * A value built once for each {@link BufferParity}. Index it with the current
 * parity; never rebuild its members per frame.
 */
export type ParityPair<T> = readonly [T, T];

/**
 * Shallow copy of `buffers` with the position and velocity ping-pong pairs
 * exchanged, for building the opposite-parity bind groups without mutating
 * (or transiently corrupting) the live buffer struct.
 *
 * Only the four ping-pong fields differ; every other buffer is shared by
 * reference, which is what makes both parity variants valid simultaneously.
 */
export function swappedPingPongView(buffers: SimulationBuffers): SimulationBuffers {
  return {
    ...buffers,
    positions: buffers.positionsOut,
    positionsOut: buffers.positions,
    velocities: buffers.velocitiesOut,
    velocitiesOut: buffers.velocities,
  };
}

/**
 * Builds `build` once per ping-pong parity so the per-frame path becomes an
 * index flip instead of a bind-group rebuild.
 *
 * `buffers` is assumed to be in orientation `currentParity` (i.e. its
 * `positions` field is the buffer the simulation reads while the parity
 * counter holds `currentParity`), so the as-is result lands at index
 * `currentParity` and the swapped-view result at the other index.
 *
 * Call this again — for both parities — whenever any buffer reachable from
 * `buffers` is reallocated; a bind group referencing a destroyed buffer is a
 * device-loss bug.
 *
 * @param buffers - Live simulation buffers, in orientation `currentParity`
 * @param currentParity - Parity the caller's frame counter is currently at
 * @param build - Builds the bind group(s) for one buffer orientation. Must be
 *   a pure function of the buffers in its argument.
 */
export function buildForBothParities<T>(
  buffers: SimulationBuffers,
  currentParity: BufferParity,
  build: (view: SimulationBuffers) => T,
): ParityPair<T> {
  const asIs = build(buffers);
  const flipped = build(swappedPingPongView(buffers));
  return currentParity === 0 ? [asIs, flipped] : [flipped, asIs];
}

/**
 * One parity-indexed group of bind groups (e.g. "the simulation compute bind
 * groups", "the collision bind group").
 *
 * A slot holds the {@link ParityPair} produced by {@link buildForBothParities}
 * plus the identity of the positions buffer each half was built from. Reading
 * {@link current} checks that identity against the buffer the simulation is
 * reading this frame, which is what turns the otherwise-silent failure mode
 * (an off-by-one in the parity index, a reallocation that forgot to rebuild,
 * a swap without a flip) into a thrown error instead of a simulation that
 * reads the buffer it is writing.
 *
 * Slots are created by {@link BindGroupParitySets.slot} and never constructed
 * directly; the owner supplies the parity counter and the live buffers.
 */
export class ParitySlot<T> {
  readonly #label: string;
  readonly #owner: BindGroupParitySets;
  /**
   * Both variants, each carrying the positions buffer it was built from.
   *
   * The buffer identity travels *with* the value rather than in a parallel
   * array, so any mis-ordering of the pair — including one introduced inside
   * buildForBothParities — moves the identity with it and trips the check in
   * {@link current} instead of silently swapping the two bind groups.
   */
  #variants: ParityPair<{ value: T; positions: GPUBuffer }> | null = null;

  /** @internal Use {@link BindGroupParitySets.slot}. */
  constructor(label: string, owner: BindGroupParitySets) {
    this.#label = label;
    this.#owner = owner;
  }

  /** Human-readable name, used in invariant failures. */
  get label(): string {
    return this.#label;
  }

  /** True once {@link rebuild} has run and {@link clear} has not. */
  get built(): boolean {
    return this.#variants !== null;
  }

  /**
   * Build both parity variants from the owner's current buffers.
   *
   * `build` must be a pure function of the buffers in its argument (see
   * {@link buildForBothParities}); it is called twice, once per orientation.
   * No-op when the owner holds no buffers.
   */
  rebuild(build: (view: SimulationBuffers) => T): void {
    const buffers = this.#owner.buffers;
    if (!buffers) {
      this.clear();
      return;
    }
    this.#variants = buildForBothParities(
      buffers,
      this.#owner.parity,
      (view) => ({ value: build(view), positions: view.positions }),
    );
  }

  /** Drop both variants (the buffers they reference are gone or stale). */
  clear(): void {
    this.#variants = null;
  }

  /**
   * The variant matching the current parity, or null when nothing is built.
   *
   * @throws GraphMotherError if the selected variant was not built from the
   *   buffer currently being read — i.e. the parity bookkeeping and the buffer
   *   orientation have diverged.
   */
  get current(): T | null {
    const variants = this.#variants;
    if (!variants) return null;
    const buffers = this.#owner.buffers;
    if (!buffers) {
      throw new GraphMotherError(
        ErrorCode.INVALID_POSITIONS,
        `Bind group set "${this.#label}" outlived its simulation buffers`,
        { slot: this.#label },
        "Clear the parity sets whenever the simulation buffers are destroyed.",
      );
    }
    const parity = this.#owner.parity;
    const selected = variants[parity];
    if (selected.positions !== buffers.positions) {
      throw new GraphMotherError(
        ErrorCode.INVALID_POSITIONS,
        `Bind group set "${this.#label}" is stale: the variant selected for parity ` +
          `${parity} was built from a different positions buffer than the one the ` +
          `simulation reads this frame`,
        { slot: this.#label, parity },
        "Rebuild this set (ParitySlot.rebuild) after every reallocation of a " +
          "ping-pong buffer, and advance parity only through BindGroupParitySets.advance().",
      );
    }
    return selected.value;
  }
}

/**
 * The ping-pong parity state machine: the parity counter, the buffers it
 * describes, and every parity-indexed bind-group set built against them.
 *
 * Position/velocity buffers ping-pong every frame, so a bind group that
 * references them has exactly two forms. Both are built once at allocation
 * time ({@link ParitySlot.rebuild}) and the per-frame path is an index flip
 * ({@link advance}) rather than ~14 createBindGroup calls.
 *
 * The counter and the buffer orientation MUST move together, which is why
 * {@link advance} owns both: it swaps the buffers and flips the counter in one
 * step. Every read goes through {@link ParitySlot.current}, which verifies the
 * two are still in agreement.
 *
 * The owner keeps the buffers; this object reads them through the accessor
 * passed to the constructor, so it never holds a stale reference of its own.
 */
export class BindGroupParitySets {
  readonly #buffersRef: () => SimulationBuffers | null;
  readonly #slots: ParitySlot<unknown>[] = [];
  #parity: BufferParity = 0;

  /**
   * @param buffersRef - Reads the owner's live simulation buffers. Must return
   *   the same object the owner passes to the GPU, or null when none exist.
   */
  constructor(buffersRef: () => SimulationBuffers | null) {
    this.#buffersRef = buffersRef;
  }

  /** The live simulation buffers, in orientation {@link parity}. */
  get buffers(): SimulationBuffers | null {
    return this.#buffersRef();
  }

  /** Which ping-pong orientation the buffers are in this frame. */
  get parity(): BufferParity {
    return this.#parity;
  }

  /** Create a parity-indexed slot owned by this set. */
  slot<T>(label: string): ParitySlot<T> {
    const slot = new ParitySlot<T>(label, this);
    this.#slots.push(slot as ParitySlot<unknown>);
    return slot;
  }

  /**
   * Rotate the ping-pong buffers and the parity counter together, after a
   * simulation step has been submitted.
   *
   * No bind group is created: both variants of every slot were built when the
   * buffers were allocated, so advancing a frame is an index flip. The buffers
   * are still swapped in place because callers read `buffers.positions` as
   * "the buffer holding current positions" (readback, drag writes, layers).
   */
  advance(): void {
    const buffers = this.#buffersRef();
    if (!buffers) return;
    swapSimulationBuffers(buffers);
    this.#parity = (this.#parity ^ 1) as BufferParity;
  }

  /** Drop every slot's bind groups (their buffers are gone or stale). */
  clearAll(): void {
    for (const slot of this.#slots) slot.clear();
  }
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
