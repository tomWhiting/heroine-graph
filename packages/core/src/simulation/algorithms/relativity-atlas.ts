/**
 * Relativity Atlas Force Algorithm
 *
 * O(N + E) hierarchical force model for directed/hierarchical graphs.
 *
 * Key innovations:
 * - Mass inheritance: mass = 1 + 0.5 × Σ child_mass
 * - Sibling repulsion: Only nodes sharing a parent repel each other
 * - Stability zones: High-mass nodes have gentler force gradients
 *
 * This algorithm is optimal for:
 * - Directed acyclic graphs (DAGs)
 * - Tree structures
 * - Hierarchical data with clear parent-child relationships
 * - Large graphs where O(N²) is too slow
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";
import { calculateWorkgroups } from "../../renderer/commands.ts";
import { assertAlgorithmSupportedOnDevice } from "./types.ts";
import type {
  AlgorithmBindGroups,
  AlgorithmBuffers,
  AlgorithmPipelines,
  AlgorithmRenderContext,
  ForceAlgorithm,
  ForceAlgorithmInfo,
  ForceAlgorithmType,
} from "./types.ts";
import {
  type AlgorithmLodFallbacks,
  createAlgorithmLodFallbacks,
  type LodEdgeDispatch,
  lodEdgeDispatch,
  lodEdgeSet,
  lodNodeMass,
} from "./lod_bindings.ts";

// Import shader sources
import DEGREES_WGSL from "../shaders/relativity_degrees.comp.wgsl";
import MASS_WGSL from "../shaders/relativity_mass.comp.wgsl";
import SIBLING_WGSL from "../shaders/relativity_sibling.comp.wgsl";
import GRAVITY_WGSL from "../shaders/relativity_gravity.comp.wgsl";
import DENSITY_FIELD_WGSL from "../shaders/density_field.comp.wgsl";
import FA2_ATTRACTION_WGSL from "../shaders/fa2_attraction.comp.wgsl";

/**
 * Relativity Atlas algorithm info
 */
const RELATIVITY_ATLAS_INFO: ForceAlgorithmInfo = {
  id: "relativity-atlas" as ForceAlgorithmType,
  name: "Relativity Atlas",
  description: "Hierarchical O(N+E) for directed/hierarchical graphs. Optimal for DAGs and trees.",
  minNodes: 100,
  maxNodes: -1, // Unlimited
  complexity: "O(N + E)",
  // Two passes tie for the widest layout and both sit exactly on the WebGPU
  // default. Sibling repulsion binds positions, forces, the merged inverse
  // CSR, the forward CSR's two rows, node mass, well radius and node flags;
  // bundled attraction binds its six un-cut buffers plus the merged LOD edge
  // set and node mass. Each was 9 before its merge, and 9 is not slower but
  // unselectable — the layout is invalid on a default-limit adapter. Declared
  // rather than omitted so a ninth binding fails a test instead of silently
  // costing the algorithm those adapters.
  minStorageBuffersPerShaderStage: 8,
};

// Configuration constants
const WORKGROUP_SIZE = 256;
const DENSITY_GRID_SIZE = 128; // 128x128 density grid cells
const DEFAULT_SPLAT_RADIUS = 3.0; // In grid cells

/**
 * Number of mass aggregation iterations to run.
 *
 * DESIGN NOTE: We use a fixed iteration count rather than GPU-to-CPU convergence
 * readback for the following reasons:
 *
 * 1. GPU-to-CPU readback is expensive: Reading the convergence flag from GPU
 *    requires a buffer map operation with synchronization, adding significant
 *    latency (often more than the cost of extra iterations).
 *
 * 2. Fixed iteration approach: We use 20 iterations as the default.
 *    For hierarchies deeper than 20 levels, increase MAX_MASS_ITERATIONS
 *    to match the maximum tree depth.
 *
 * 3. Per-frame overhead is negligible: Mass initialization only runs when the
 *    graph changes, not every frame. The force passes (sibling repulsion, gravity)
 *    run every frame and dominate the cost.
 *
 * 4. Complexity cost: Async readback would require restructuring the rendering
 *    pipeline, adding state management for in-flight reads, and complicating
 *    the frame timing. The benefit doesn't justify this complexity.
 *
 * The shader still writes to a convergence flag buffer (for potential future use
 * or debugging), but it is not read back on the CPU side.
 */
const MAX_MASS_ITERATIONS = 20;

/**
 * Threshold below which mass changes are considered converged.
 * Used by the shader to set the convergence flag (for debugging/future use).
 */
const MASS_CONVERGENCE_THRESHOLD = 0.01;

/**
 * Extended pipelines for Relativity Atlas
 */
interface RelativityAtlasPipelines extends AlgorithmPipelines {
  // Degree computation
  computeDegrees: GPUComputePipeline;

  // Mass aggregation
  initMass: GPUComputePipeline;
  aggregateMass: GPUComputePipeline;

  // Sibling repulsion
  siblingRepulsion: GPUComputePipeline;

  // Gravity
  gravity: GPUComputePipeline;

  // Linear attraction (F = d, no rest length — replaces Hooke's law springs)
  attraction: GPUComputePipeline;
  /**
   * The same attraction over the LOD active-edge list plus the aggregated
   * bundles. A second entry point rather than a branch in the first, so that
   * with no cut the pipeline, its bind group and its arithmetic are the ones
   * that shipped before edge aggregation existed.
   */
  attractionBundled: GPUComputePipeline;

  // Density field (global repulsion)
  densityClearGrid: GPUComputePipeline;
  densityAccumulate: GPUComputePipeline;
  densityApplyForces: GPUComputePipeline;

  // Bind group layouts
  degreesLayout: GPUBindGroupLayout;
  massLayout: GPUBindGroupLayout;
  siblingLayout: GPUBindGroupLayout;
  gravityLayout: GPUBindGroupLayout;
  attractionLayout: GPUBindGroupLayout;
  attractionBundledLayout: GPUBindGroupLayout;
  densityLayout: GPUBindGroupLayout;
}

/** Byte size of FA2AttractionUniforms in fa2_attraction.comp.wgsl. */
const RA_ATTRACTION_UNIFORM_BYTES = 32;

/**
 * First element of the source region of {@link RelativityAtlasBuffers.csrInverse}.
 *
 * ## INVERSE CSR REGION MAP
 *
 * One buffer, two regions, in `u32` elements (N = `maxNodes`, E = `maxEdges`):
 *
 * ```text
 *   [0, N+1)              offset row: incoming edges of node i are the source
 *                         entries [offsets[i], offsets[i+1])
 *   [N+1, N+1+E)          source row: the slot each incoming edge comes from
 * ```
 *
 * The base is the allocated offset-row length, not the live node count, so a
 * graph change rewrites the regions in place and never moves them. It is
 * uploaded to the shaders as `csr_inverse_sources_base`, which
 * relativity_degrees.comp.wgsl and relativity_sibling.comp.wgsl state the same
 * map against.
 */
function csrInverseSourcesBase(maxNodes: number): number {
  return maxNodes + 1;
}

/**
 * Relativity Atlas algorithm-specific buffers
 */

export class RelativityAtlasBuffers implements AlgorithmBuffers {
  constructor(
    // Uniform buffers
    public degreesUniforms: GPUBuffer,
    public massUniforms: GPUBuffer,
    public siblingUniforms: GPUBuffer,
    public gravityUniforms: GPUBuffer,
    // CSR edge data (outgoing edges)
    public csrOffsets: GPUBuffer,
    public csrTargets: GPUBuffer,
    /**
     * Inverse CSR (incoming edges / parents): the offset row then the source
     * row, one buffer, two regions — see {@link csrInverseSourcesBase}.
     *
     * Merged because the sibling pass reads both and, with them apart, binds
     * nine storage buffers in one stage against a WebGPU default of eight,
     * which makes Relativity Atlas unselectable on a default-limit adapter.
     * The forward pair is deliberately NOT merged: no pass is over the limit
     * because of it, and merging it would put a second region base in three
     * more uniform structs to buy headroom nothing needs.
     */
    public csrInverse: GPUBuffer,
    // Computed data
    public degrees: GPUBuffer, // [out_deg, in_deg] pairs
    public mass: GPUBuffer, // Ping buffer
    public massOut: GPUBuffer, // Pong buffer
    /**
     * Atomic convergence flag buffer (written by shader, not read by CPU).
     *
     * NOTE: This buffer is NOT initialized to 1 before iterations. The shader
     * atomically stores 0 when mass values haven't converged, but we don't
     * read it back due to the high cost of GPU-to-CPU sync. Instead, we run
     * a fixed number of iterations (MAX_MASS_ITERATIONS). This buffer is
     * retained for shader binding compatibility and potential future debugging use.
     */
    public converged: GPUBuffer,
    // Density field buffers (global repulsion)
    public densityUniforms: GPUBuffer,
    public densityGrid: GPUBuffer,
    // Linear attraction buffers (replaces Hooke's law springs)
    public attractionUniforms: GPUBuffer,
    public edgeWeightsBuffer: GPUBuffer,
    // Bubble mode buffers (per-node data from WASM)
    public wellRadius: GPUBuffer, // f32 per node: subtree-based collision radius
    public nodeDepth: GPUBuffer, // f32 per node: BFS depth from root
    /**
     * Zero-filled node flags used when AlgorithmRenderContext.nodeFlags is
     * not provided (all slots treated as live). Production passes the shared
     * simulation nodeFlags buffer for dead-slot masking.
     */
    public fallbackNodeFlags: GPUBuffer,
    /** Identity list / unit mass / empty LOD edge lists for a context supplying none */
    public lodFallbacks: AlgorithmLodFallbacks,
    // Capacities
    public maxNodes: number,
    public maxEdges: number,
  ) {}

  destroy(): void {
    this.degreesUniforms.destroy();
    this.massUniforms.destroy();
    this.siblingUniforms.destroy();
    this.gravityUniforms.destroy();

    this.csrOffsets.destroy();
    this.csrTargets.destroy();
    this.csrInverse.destroy();

    this.degrees.destroy();
    this.mass.destroy();
    this.massOut.destroy();
    this.converged.destroy();

    this.densityUniforms.destroy();
    this.densityGrid.destroy();

    this.attractionUniforms.destroy();
    this.edgeWeightsBuffer.destroy();

    this.wellRadius.destroy();
    this.nodeDepth.destroy();
    this.fallbackNodeFlags.destroy();
    this.lodFallbacks.destroy();
  }
}

/**
 * Extended bind groups for Relativity Atlas
 */
interface RelativityAtlasBindGroups extends AlgorithmBindGroups {
  degrees: GPUBindGroup;
  massInit: GPUBindGroup;
  massAggregate: GPUBindGroup[]; // Ping-pong
  sibling: GPUBindGroup;
  gravity: GPUBindGroup;
  attraction: GPUBindGroup; // Linear edge attraction (F=d, no rest length)
  attractionBundled: GPUBindGroup; // Same, over the LOD active-edge list + bundles
  density: GPUBindGroup | null; // null when bounds unavailable
  // repulsion from base is used for main force pass
}

/**
 * Relativity Atlas force algorithm implementation
 */
export class RelativityAtlasAlgorithm implements ForceAlgorithm {
  readonly info = RELATIVITY_ATLAS_INFO;
  readonly handlesGravity = true;
  readonly handlesSprings = true; // LinLog attraction (F=log(1+d), no rest length) instead of Hooke's law
  // Per-edge attraction has no equilibrium distance, so high-degree parents
  // accumulate stiff net pulls that overshoot under plain integration —
  // FA2-style swing/traction speed control is required for convergence.
  readonly prefersAdaptiveSpeed = true;

  // Track whether mass has been initialized for current graph
  private massInitialized = false;
  private lastNodeCount = 0;
  private currentEdgeCount = 0;
  /** The aggregated edge set the last updateUniforms saw, or null for no cut. */
  private lastLodEdges: LodEdgeDispatch | null = null;

  /**
   * Reset mass state when graph changes
   */
  resetMassState(): void {
    this.massInitialized = false;
  }

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

    // Same hazard as Barnes-Hut, one buffer lower: an over-limit layout is
    // invalid and poisons every encoder its bind group is recorded into.
    assertAlgorithmSupportedOnDevice(this.info, device);

    // Create shader modules
    const degreesModule = device.createShaderModule({
      label: "Relativity Atlas Degrees",
      code: DEGREES_WGSL,
    });

    const massModule = device.createShaderModule({
      label: "Relativity Atlas Mass",
      code: MASS_WGSL,
    });

    const siblingModule = device.createShaderModule({
      label: "Relativity Atlas Sibling",
      code: SIBLING_WGSL,
    });

    const gravityModule = device.createShaderModule({
      label: "Relativity Atlas Gravity",
      code: GRAVITY_WGSL,
    });

    // === Degrees Layout ===
    const degreesLayout = device.createBindGroupLayout({
      label: "RA Degrees Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // csr_inverse
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // === Mass Layout ===
    const massLayout = device.createBindGroupLayout({
      label: "RA Mass Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // === Sibling Layout ===
    // Bindings: uniforms, positions (vec2), forces (vec2), csr_inverse_offsets, csr_inverse_sources,
    //           csr_offsets, csr_targets, node_mass, well_radius, node_flags
    const siblingLayout = device.createBindGroupLayout({
      label: "RA Sibling Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // csr_inverse
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // well_radius
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // node_flags
      ],
    });

    // === Gravity Layout ===
    // Bindings: uniforms, positions (vec2), forces (vec2), node_mass, node_depth, node_flags
    const gravityLayout = device.createBindGroupLayout({
      label: "RA Gravity Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // node_depth
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // node_flags
      ],
    });

    // Create pipeline layouts
    const degreesPipelineLayout = device.createPipelineLayout({
      label: "RA Degrees Pipeline Layout",
      bindGroupLayouts: [degreesLayout],
    });

    const massPipelineLayout = device.createPipelineLayout({
      label: "RA Mass Pipeline Layout",
      bindGroupLayouts: [massLayout],
    });

    const siblingPipelineLayout = device.createPipelineLayout({
      label: "RA Sibling Pipeline Layout",
      bindGroupLayouts: [siblingLayout],
    });

    const gravityPipelineLayout = device.createPipelineLayout({
      label: "RA Gravity Pipeline Layout",
      bindGroupLayouts: [gravityLayout],
    });

    // === Attraction Layout (linear F=d, replaces Hooke's law springs) ===
    // Bindings: uniforms, positions, forces, edge_sources, edge_targets,
    // edge_weights, node_flags
    const attractionModule = device.createShaderModule({
      label: "RA Attraction Shader",
      code: FA2_ATTRACTION_WGSL,
    });

    const attractionEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // node_flags
    ];

    const attractionLayout = device.createBindGroupLayout({
      label: "RA Attraction Layout",
      entries: attractionEntries,
    });

    const attractionPipelineLayout = device.createPipelineLayout({
      label: "RA Attraction Pipeline Layout",
      bindGroupLayouts: [attractionLayout],
    });

    // Bundled attraction: the un-cut bindings plus the combined LOD edge set
    // and the per-node mass each arriving pull is divided by. Eight storage
    // buffers, the WebGPU default, exactly — the same ceiling the sibling pass
    // sits on.
    const attractionBundledLayout = device.createBindGroupLayout({
      label: "RA Attraction Layout (LOD bundles)",
      entries: [
        ...attractionEntries,
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const attractionBundledPipelineLayout = device.createPipelineLayout({
      label: "RA Attraction Pipeline Layout (LOD bundles)",
      bindGroupLayouts: [attractionBundledLayout],
    });

    // === Density Field Layout ===
    // Reuses the density_field.comp.wgsl shader for global O(n) repulsion.
    // Bindings: uniforms, positions (vec2), forces (vec2), density_grid, well_radius, node_flags
    const densityModule = device.createShaderModule({
      label: "RA Density Field Shader",
      code: DENSITY_FIELD_WGSL,
    });

    const densityLayout = device.createBindGroupLayout({
      label: "RA Density Field Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // well_radius
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // node_flags
      ],
    });

    const densityPipelineLayout = device.createPipelineLayout({
      label: "RA Density Pipeline Layout",
      bindGroupLayouts: [densityLayout],
    });

    // Create pipelines
    const pipelines: RelativityAtlasPipelines = {
      computeDegrees: device.createComputePipeline({
        label: "RA Compute Degrees",
        layout: degreesPipelineLayout,
        compute: { module: degreesModule, entryPoint: "main" },
      }),

      initMass: device.createComputePipeline({
        label: "RA Init Mass",
        layout: massPipelineLayout,
        compute: { module: massModule, entryPoint: "init_mass" },
      }),

      aggregateMass: device.createComputePipeline({
        label: "RA Aggregate Mass",
        layout: massPipelineLayout,
        compute: { module: massModule, entryPoint: "aggregate_mass" },
      }),

      siblingRepulsion: device.createComputePipeline({
        label: "RA Sibling Repulsion",
        layout: siblingPipelineLayout,
        compute: { module: siblingModule, entryPoint: "main" },
      }),

      gravity: device.createComputePipeline({
        label: "RA Gravity",
        layout: gravityPipelineLayout,
        compute: { module: gravityModule, entryPoint: "main" },
      }),

      // Alias: reuse siblingRepulsion as the main "repulsion" pass (avoids duplicate pipeline)
      repulsion: null as unknown as GPUComputePipeline, // Assigned below after object creation

      // Linear attraction (F = d, no rest length)
      attraction: device.createComputePipeline({
        label: "RA Attraction",
        layout: attractionPipelineLayout,
        compute: { module: attractionModule, entryPoint: "main" },
      }),

      // Same law over the LOD active-edge list plus the aggregated bundles
      attractionBundled: device.createComputePipeline({
        label: "RA Attraction (LOD bundles)",
        layout: attractionBundledPipelineLayout,
        compute: { module: attractionModule, entryPoint: "main_bundled" },
      }),

      // Density field pipelines (global repulsion)
      densityClearGrid: device.createComputePipeline({
        label: "RA Density Clear Grid",
        layout: densityPipelineLayout,
        compute: { module: densityModule, entryPoint: "clear_grid" },
      }),
      densityAccumulate: device.createComputePipeline({
        label: "RA Density Accumulate",
        layout: densityPipelineLayout,
        compute: { module: densityModule, entryPoint: "accumulate_density" },
      }),
      densityApplyForces: device.createComputePipeline({
        label: "RA Density Apply Forces",
        layout: densityPipelineLayout,
        compute: { module: densityModule, entryPoint: "apply_forces" },
      }),

      degreesLayout,
      massLayout,
      siblingLayout,
      gravityLayout,
      attractionLayout,
      attractionBundledLayout,
      densityLayout,
    };

    // Assign repulsion alias to reuse siblingRepulsion (avoids duplicate pipeline)
    pipelines.repulsion = pipelines.siblingRepulsion;

    return pipelines;
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    const safeMaxNodes = Math.max(maxNodes, 4);
    const maxEdges = safeMaxNodes * 4; // Estimate 4 edges per node on average

    const nodeBytes = safeMaxNodes * 4;
    const edgeBytes = maxEdges * 4;
    const offsetBytes = (safeMaxNodes + 1) * 4;
    const degreeBytes = safeMaxNodes * 2 * 4; // 2 degrees per node

    // Uniform buffers
    const degreesUniforms = device.createBuffer({
      label: "RA Degrees Uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // MassUniforms: { node_count: u32, edge_count: u32, iteration: u32, convergence_threshold: f32,
    //                 base_mass: f32, child_mass_factor: f32, _padding: vec2<u32> }
    // Total: 32 bytes
    const massUniforms = device.createBuffer({
      label: "RA Mass Uniforms",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // SiblingUniforms: { node_count: u32, edge_count: u32, repulsion_strength: f32, min_distance: f32,
    //                   max_siblings: u32, parent_child_multiplier: f32,
    //                   cousin_enabled: u32, cousin_strength: f32,
    //                   phantom_enabled: u32, phantom_multiplier: f32,
    //                   orbit_strength: f32, tangential_multiplier: f32,
    //                   orbit_radius_base: f32, bubble_mode: u32, orbit_scale: f32 }
    // Total: 64 bytes (60 used + 4 implicit padding)
    const siblingUniforms = device.createBuffer({
      label: "RA Sibling Uniforms",
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // GravityUniforms: { node_count: u32, gravity_strength: f32, center_x: f32, center_y: f32,
    //                   mass_exponent: f32, gravity_curve: u32, gravity_exponent: f32, depth_decay_rate: f32 }
    // Total: 32 bytes
    const gravityUniforms = device.createBuffer({
      label: "RA Gravity Uniforms",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // CSR edge data
    const csrOffsets = device.createBuffer({
      label: "RA CSR Offsets",
      size: offsetBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const csrTargets = device.createBuffer({
      label: "RA CSR Targets",
      size: edgeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Offset row then source row; see csrInverseSourcesBase for the map.
    const csrInverse = device.createBuffer({
      label: "RA CSR Inverse",
      size: offsetBytes + edgeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Computed data
    const degrees = device.createBuffer({
      label: "RA Degrees",
      size: degreeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const mass = device.createBuffer({
      label: "RA Mass",
      size: nodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const massOut = device.createBuffer({
      label: "RA Mass Out",
      size: nodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const converged = device.createBuffer({
      label: "RA Converged",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Density field buffers for global repulsion
    // DensityUniforms struct: 48 bytes (12 × f32)
    const densityUniforms = device.createBuffer({
      label: "RA Density Uniforms",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Density grid: DENSITY_GRID_SIZE × DENSITY_GRID_SIZE cells, each atomic u32
    const gridCells = DENSITY_GRID_SIZE * DENSITY_GRID_SIZE;
    const densityGrid = device.createBuffer({
      label: "RA Density Grid",
      size: gridCells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Linear attraction uniforms (RA_ATTRACTION_UNIFORM_BYTES):
    // { edge_count, edge_weight_influence, flags, active_edge_count,
    //   bundle_count, 3 x padding }
    const attractionUniforms = device.createBuffer({
      label: "RA Attraction Uniforms",
      size: RA_ATTRACTION_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Edge weights: f32 per edge (1.0 for unweighted)
    const edgeWeightsBuffer = device.createBuffer({
      label: "RA Edge Weights",
      size: Math.max(maxEdges * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Bubble mode: per-node well radius (subtree-based collision boundary)
    // Always allocated with safe defaults (base_radius when bubble mode off)
    const wellRadius = device.createBuffer({
      label: "RA Well Radius",
      size: nodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Bubble mode: per-node depth (BFS distance from root)
    // Always allocated with safe defaults (0.0 when bubble mode off)
    const nodeDepth = device.createBuffer({
      label: "RA Node Depth",
      size: nodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Zero-filled fallback for contexts without a shared nodeFlags buffer
    // (GPU buffers are created zero-initialized: all slots live)
    const fallbackNodeFlags = device.createBuffer({
      label: "RA Fallback Node Flags",
      size: nodeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new RelativityAtlasBuffers(
      degreesUniforms,
      massUniforms,
      siblingUniforms,
      gravityUniforms,
      csrOffsets,
      csrTargets,
      csrInverse,
      degrees,
      mass,
      massOut,
      converged,
      densityUniforms,
      densityGrid,
      attractionUniforms,
      edgeWeightsBuffer,
      wellRadius,
      nodeDepth,
      fallbackNodeFlags,
      createAlgorithmLodFallbacks(device, safeMaxNodes, "RA"),
      safeMaxNodes,
      maxEdges,
    );
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    const p = pipelines as RelativityAtlasPipelines;
    const b = algorithmBuffers as RelativityAtlasBuffers;

    // Dead-slot masking: use the shared simulation nodeFlags buffer when the
    // host provides it, otherwise a zero-filled fallback (all slots live)
    const nodeFlags = context.nodeFlags ?? b.fallbackNodeFlags;

    // Degrees bind group
    const degrees = device.createBindGroup({
      label: "RA Degrees Bind Group",
      layout: p.degreesLayout,
      entries: [
        { binding: 0, resource: { buffer: b.degreesUniforms } },
        { binding: 1, resource: { buffer: b.csrOffsets } },
        { binding: 2, resource: { buffer: b.csrTargets } },
        { binding: 3, resource: { buffer: b.csrInverse } },
        { binding: 4, resource: { buffer: b.degrees } },
      ],
    });

    // Mass init bind group (writes to massOut)
    const massInit = device.createBindGroup({
      label: "RA Mass Init Bind Group",
      layout: p.massLayout,
      entries: [
        { binding: 0, resource: { buffer: b.massUniforms } },
        { binding: 1, resource: { buffer: b.csrOffsets } },
        { binding: 2, resource: { buffer: b.csrTargets } },
        { binding: 3, resource: { buffer: b.degrees } },
        { binding: 4, resource: { buffer: b.mass } }, // unused for init
        { binding: 5, resource: { buffer: b.massOut } }, // output
        { binding: 6, resource: { buffer: b.converged } },
      ],
    });

    // Mass aggregate bind groups (ping-pong)
    const massAggregate: GPUBindGroup[] = [
      // Even iterations: read from massOut, write to mass
      device.createBindGroup({
        label: "RA Mass Aggregate Even",
        layout: p.massLayout,
        entries: [
          { binding: 0, resource: { buffer: b.massUniforms } },
          { binding: 1, resource: { buffer: b.csrOffsets } },
          { binding: 2, resource: { buffer: b.csrTargets } },
          { binding: 3, resource: { buffer: b.degrees } },
          { binding: 4, resource: { buffer: b.massOut } }, // read
          { binding: 5, resource: { buffer: b.mass } }, // write
          { binding: 6, resource: { buffer: b.converged } },
        ],
      }),
      // Odd iterations: read from mass, write to massOut
      device.createBindGroup({
        label: "RA Mass Aggregate Odd",
        layout: p.massLayout,
        entries: [
          { binding: 0, resource: { buffer: b.massUniforms } },
          { binding: 1, resource: { buffer: b.csrOffsets } },
          { binding: 2, resource: { buffer: b.csrTargets } },
          { binding: 3, resource: { buffer: b.degrees } },
          { binding: 4, resource: { buffer: b.mass } }, // read
          { binding: 5, resource: { buffer: b.massOut } }, // write
          { binding: 6, resource: { buffer: b.converged } },
        ],
      }),
    ];

    // Sibling repulsion bind group
    // Note: After mass iterations (even count), final mass values are in massOut
    const sibling = device.createBindGroup({
      label: "RA Sibling Bind Group",
      layout: p.siblingLayout,
      entries: [
        { binding: 0, resource: { buffer: b.siblingUniforms } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: b.csrInverse } },
        { binding: 4, resource: { buffer: b.csrOffsets } },
        { binding: 5, resource: { buffer: b.csrTargets } },
        { binding: 6, resource: { buffer: b.massOut } }, // Final mass is in massOut
        { binding: 7, resource: { buffer: b.wellRadius } },
        { binding: 8, resource: { buffer: nodeFlags } },
      ],
    });

    // Gravity bind group
    // Note: After mass iterations (even count), final mass values are in massOut
    const gravity = device.createBindGroup({
      label: "RA Gravity Bind Group",
      layout: p.gravityLayout,
      entries: [
        { binding: 0, resource: { buffer: b.gravityUniforms } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: b.massOut } }, // Final mass is in massOut
        { binding: 4, resource: { buffer: b.nodeDepth } },
        { binding: 5, resource: { buffer: nodeFlags } },
      ],
    });

    // Linear attraction bind group (replaces Hooke's law springs)
    if (!context.edgeSources || !context.edgeTargets) {
      throw new Error(
        "RelativityAtlas requires edge source/target buffers in AlgorithmRenderContext. " +
          "Ensure graph.ts populates edgeSources and edgeTargets.",
      );
    }

    const attractionEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: b.attractionUniforms } },
      { binding: 1, resource: { buffer: context.positions } },
      { binding: 2, resource: { buffer: context.forces } },
      { binding: 3, resource: { buffer: context.edgeSources } },
      { binding: 4, resource: { buffer: context.edgeTargets } },
      { binding: 5, resource: { buffer: b.edgeWeightsBuffer } },
      { binding: 6, resource: { buffer: nodeFlags } },
    ];

    const attraction = device.createBindGroup({
      label: "RA Attraction Bind Group",
      layout: p.attractionLayout,
      entries: attractionEntries,
    });

    // Built unconditionally, dispatched only under a cut: whether an
    // aggregation is live is per-frame state, and createBindGroups is
    // contractually forbidden from capturing any of that.
    const attractionBundled = device.createBindGroup({
      label: "RA Attraction Bind Group (LOD bundles)",
      layout: p.attractionBundledLayout,
      entries: [
        ...attractionEntries,
        { binding: 7, resource: { buffer: lodEdgeSet(context, b.lodFallbacks) } },
        { binding: 8, resource: { buffer: lodNodeMass(context, b.lodFallbacks) } },
      ],
    });

    // Density field bind group for global repulsion.
    // Uses the same position/force buffers as sibling and gravity.
    const density = device.createBindGroup({
      label: "RA Density Bind Group",
      layout: p.densityLayout,
      entries: [
        { binding: 0, resource: { buffer: b.densityUniforms } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: b.densityGrid } },
        { binding: 4, resource: { buffer: b.wellRadius } },
        { binding: 5, resource: { buffer: nodeFlags } },
      ],
    });

    const bindGroups: RelativityAtlasBindGroups = {
      degrees,
      massInit,
      massAggregate,
      sibling,
      gravity,
      attraction,
      attractionBundled,
      density,
      repulsion: sibling, // Use sibling as main repulsion
    };

    return bindGroups;
  }

  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void {
    const b = algorithmBuffers as RelativityAtlasBuffers;

    // CRITICAL: Validate node count doesn't exceed buffer capacity.
    // Buffer overflow from undersized buffers is a security issue that can corrupt
    // GPU memory, cause crashes, or produce undefined behavior.
    if (context.nodeCount > b.maxNodes) {
      throw new Error(
        `RelativityAtlas buffer overflow: nodeCount (${context.nodeCount}) exceeds buffer capacity (${b.maxNodes}). ` +
          `Buffers must be recreated with createBuffers() when node count increases.`,
      );
    }

    // Validate edge count doesn't exceed buffer capacity
    if (context.edgeCount > b.maxEdges) {
      throw new Error(
        `RelativityAtlas buffer overflow: edgeCount (${context.edgeCount}) exceeds buffer capacity (${b.maxEdges}). ` +
          `Buffers must be recreated with createBuffers() when edge count increases.`,
      );
    }

    const edgeCount = context.edgeCount;

    // Degrees uniforms (16 bytes)
    // Struct layout: { node_count: u32, edge_count: u32,
    //                  csr_inverse_sources_base: u32, _padding: u32 }
    const degreesData = new ArrayBuffer(16);
    const degreesView = new DataView(degreesData);
    degreesView.setUint32(0, context.nodeCount, true);
    degreesView.setUint32(4, edgeCount, true);
    degreesView.setUint32(8, csrInverseSourcesBase(b.maxNodes), true);
    degreesView.setUint32(12, 0, true); // padding
    device.queue.writeBuffer(b.degreesUniforms, 0, degreesData);

    // Mass uniforms (32 bytes)
    // Struct layout: { node_count: u32, edge_count: u32, iteration: u32, convergence_threshold: f32,
    //                  base_mass: f32, child_mass_factor: f32, _padding: vec2<u32> }
    const massData = new ArrayBuffer(32);
    const massView = new DataView(massData);
    massView.setUint32(0, context.nodeCount, true);
    massView.setUint32(4, edgeCount, true);
    massView.setUint32(8, 0, true); // iteration (updated per-iteration if needed)
    massView.setFloat32(12, MASS_CONVERGENCE_THRESHOLD, true);
    massView.setFloat32(16, context.forceConfig.relativityBaseMass, true);
    massView.setFloat32(20, context.forceConfig.relativityChildMassFactor, true);
    massView.setUint32(24, 0, true); // _padding
    massView.setUint32(28, 0, true); // _padding
    device.queue.writeBuffer(b.massUniforms, 0, massData);

    // Sibling uniforms (64 bytes)
    // Struct layout: { node_count: u32, edge_count: u32, repulsion_strength: f32, min_distance: f32,
    //                  max_siblings: u32, parent_child_multiplier: f32,
    //                  cousin_enabled: u32, cousin_strength: f32,
    //                  phantom_enabled: u32, phantom_multiplier: f32,
    //                  orbit_strength: f32, tangential_multiplier: f32,
    //                  orbit_radius_base: f32, bubble_mode: u32, orbit_scale: f32,
    //                  csr_inverse_sources_base: u32 }
    // The last field lands in what used to be the struct's implicit tail
    // padding, so the buffer stays 64 bytes.
    const siblingData = new ArrayBuffer(64);
    const siblingView = new DataView(siblingData);
    siblingView.setUint32(0, context.nodeCount, true);
    siblingView.setUint32(4, edgeCount, true);
    siblingView.setFloat32(8, Math.abs(context.forceConfig.repulsionStrength), true);
    siblingView.setFloat32(12, context.forceConfig.repulsionDistanceMin, true);
    siblingView.setUint32(16, context.forceConfig.relativityMaxSiblings, true);
    siblingView.setFloat32(20, context.forceConfig.relativityParentChildMultiplier, true);
    siblingView.setUint32(24, context.forceConfig.relativityCousinRepulsion ? 1 : 0, true);
    siblingView.setFloat32(28, context.forceConfig.relativityCousinStrength, true);
    siblingView.setUint32(32, context.forceConfig.relativityPhantomZone ? 1 : 0, true);
    siblingView.setFloat32(36, context.forceConfig.relativityPhantomMultiplier, true);
    siblingView.setFloat32(40, context.forceConfig.relativityOrbitStrength, true);
    siblingView.setFloat32(44, context.forceConfig.relativityTangentialMultiplier, true);
    siblingView.setFloat32(48, context.forceConfig.relativityOrbitRadius, true);
    siblingView.setUint32(52, context.forceConfig.relativityBubbleMode ? 1 : 0, true); // bubble_mode
    siblingView.setFloat32(56, context.forceConfig.relativityBubbleOrbitScale, true); // orbit_scale
    siblingView.setUint32(60, csrInverseSourcesBase(b.maxNodes), true);
    device.queue.writeBuffer(b.siblingUniforms, 0, siblingData);

    // Gravity uniforms (32 bytes)
    // Struct layout: { node_count: u32, gravity_strength: f32, center_x: f32, center_y: f32,
    //                  mass_exponent: f32, gravity_curve: u32, gravity_exponent: f32, depth_decay_rate: f32 }
    const gravityData = new ArrayBuffer(32);
    const gravityView = new DataView(gravityData);
    gravityView.setUint32(0, context.nodeCount, true);
    gravityView.setFloat32(4, context.forceConfig.centerStrength, true);
    gravityView.setFloat32(8, context.forceConfig.centerX, true);
    gravityView.setFloat32(12, context.forceConfig.centerY, true);
    gravityView.setFloat32(16, context.forceConfig.relativityMassExponent, true);
    // Map gravity curve string to u32: linear=0, inverse=1, soft=2, custom=3
    const gravityCurveMap: Record<string, number> = { linear: 0, inverse: 1, soft: 2, custom: 3 };
    const gravityCurveValue = gravityCurveMap[context.forceConfig.relativityGravityCurve] ?? 0;
    gravityView.setUint32(20, gravityCurveValue, true);
    gravityView.setFloat32(24, context.forceConfig.relativityGravityExponent, true);
    gravityView.setFloat32(28, context.forceConfig.relativityDepthDecay, true); // depth_decay_rate
    device.queue.writeBuffer(b.gravityUniforms, 0, gravityData);

    // Cache edge count for attraction pass workgroup dispatch
    this.currentEdgeCount = edgeCount;

    // Attraction uniforms (RA_ATTRACTION_UNIFORM_BYTES):
    // { edge_count, edge_weight_influence, flags, active_edge_count,
    //   bundle_count, 3 x padding }
    //
    // flags bit 0 = LinLog mode: F = log(1 + d) instead of F = d. The raw
    // linear pull is unbounded (a child 500 units out received force 500,
    // pinning velocity at the cap and overshooting every frame — the main
    // jitter driver), and it overpowered the orbit spring so children
    // equilibrated at roughly half the configured orbit radius. log(1 + d)
    // keeps the always-pulling/no-rest-length design while capping
    // long-range magnitude (~6 at d = 500, ~3 at the default orbit radius).
    this.lastLodEdges = lodEdgeDispatch(context);

    const attractData = new ArrayBuffer(RA_ATTRACTION_UNIFORM_BYTES);
    const attractView = new DataView(attractData);
    attractView.setUint32(0, edgeCount, true);
    attractView.setFloat32(4, 1.0, true); // edge_weight_influence
    attractView.setUint32(8, 1, true); // flags (bit 0: linlog attraction)
    // Zero with no cut, which is what keeps `main` reading nothing new.
    attractView.setUint32(12, this.lastLodEdges?.activeEdgeCount ?? 0, true);
    attractView.setUint32(16, this.lastLodEdges?.bundleCount ?? 0, true);
    attractView.setUint32(20, 0, true);
    attractView.setUint32(24, 0, true);
    attractView.setUint32(28, 0, true);
    device.queue.writeBuffer(b.attractionUniforms, 0, attractData);

    // Upload edge weights (all 1.0 for unweighted graphs)
    if (edgeCount > 0 && edgeCount <= b.maxEdges) {
      const weights = new Float32Array(edgeCount);
      weights.fill(1.0);
      device.queue.writeBuffer(b.edgeWeightsBuffer, 0, weights);
    }

    // Density field uniforms (48 bytes = 12 × f32)
    // Only meaningful when bounds are available; the shader is skipped otherwise.
    // DensityUniforms struct: { node_count, grid_width, grid_height, repulsion_strength,
    //   bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y, splat_radius, _pad × 3 }
    if (context.bounds) {
      const densityRepulsionStrength = Math.abs(context.forceConfig.repulsionStrength) *
        context.forceConfig.relativityDensityRepulsion;
      const densityData = new ArrayBuffer(48);
      const densityView = new DataView(densityData);
      densityView.setUint32(0, context.nodeCount, true); // node_count
      densityView.setUint32(4, DENSITY_GRID_SIZE, true); // grid_width
      densityView.setUint32(8, DENSITY_GRID_SIZE, true); // grid_height
      densityView.setFloat32(12, densityRepulsionStrength, true); // repulsion_strength
      densityView.setFloat32(16, context.bounds.minX, true); // bounds_min_x
      densityView.setFloat32(20, context.bounds.minY, true); // bounds_min_y
      densityView.setFloat32(24, context.bounds.maxX, true); // bounds_max_x
      densityView.setFloat32(28, context.bounds.maxY, true); // bounds_max_y
      densityView.setFloat32(32, DEFAULT_SPLAT_RADIUS, true); // splat_radius
      densityView.setFloat32(36, 0, true); // _pad1
      densityView.setFloat32(40, 0, true); // _pad2
      densityView.setFloat32(44, 0, true); // _pad3
      device.queue.writeBuffer(b.densityUniforms, 0, densityData);
    }
  }

  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: AlgorithmPipelines,
    bindGroups: AlgorithmBindGroups,
    nodeCount: number,
  ): void {
    const p = pipelines as RelativityAtlasPipelines;
    const bg = bindGroups as RelativityAtlasBindGroups;

    if (nodeCount < 2) {
      return;
    }

    const nodeWorkgroups = calculateWorkgroups(nodeCount, WORKGROUP_SIZE);

    // Check if we need to reinitialize mass (graph changed)
    if (!this.massInitialized || nodeCount !== this.lastNodeCount) {
      this.lastNodeCount = nodeCount;

      // === PHASE 1: Compute degrees from CSR data ===
      {
        const pass = encoder.beginComputePass({ label: "RA Compute Degrees" });
        pass.setPipeline(p.computeDegrees);
        pass.setBindGroup(0, bg.degrees);
        pass.dispatchWorkgroups(nodeWorkgroups);
        pass.end();
      }

      // === PHASE 2: Initialize mass values ===
      {
        const pass = encoder.beginComputePass({ label: "RA Init Mass" });
        pass.setPipeline(p.initMass);
        pass.setBindGroup(0, bg.massInit);
        pass.dispatchWorkgroups(nodeWorkgroups);
        pass.end();
      }

      // === PHASE 3: Iterative mass aggregation ===
      // Run a fixed number of iterations to propagate mass through the hierarchy.
      // Each iteration aggregates child masses to parents (mass = 1 + 0.5 * sum(child_mass)).
      //
      // We use fixed iterations rather than convergence checking because:
      // - GPU-to-CPU readback for the convergence flag would add more latency than
      //   running extra iterations
      // - This phase only runs when the graph changes, not every frame
      //
      // See MAX_MASS_ITERATIONS constant for detailed rationale.
      for (let iter = 0; iter < MAX_MASS_ITERATIONS; iter++) {
        const pass = encoder.beginComputePass({ label: `RA Aggregate Mass ${iter}` });
        pass.setPipeline(p.aggregateMass);
        // Alternate between ping-pong buffers
        pass.setBindGroup(0, bg.massAggregate[iter % 2]);
        pass.dispatchWorkgroups(nodeWorkgroups);
        pass.end();
      }

      this.massInitialized = true;
    }

    // === PHASE 4: Density field global repulsion (every frame, when bounds available) ===
    // Provides O(n) mass-independent repulsion between all nodes regardless of
    // hierarchy. Without this, nodes in different subtrees have zero mutual
    // repulsion and can overlap. Runs before sibling repulsion so both forces
    // contribute to the accumulated force buffer.
    if (bg.density) {
      const gridCells = DENSITY_GRID_SIZE * DENSITY_GRID_SIZE;
      const gridWorkgroups = calculateWorkgroups(gridCells, WORKGROUP_SIZE);

      // Clear density grid
      {
        const pass = encoder.beginComputePass({ label: "RA Density Clear Grid" });
        pass.setPipeline(p.densityClearGrid);
        pass.setBindGroup(0, bg.density);
        pass.dispatchWorkgroups(gridWorkgroups);
        pass.end();
      }

      // Accumulate density from node positions
      {
        const pass = encoder.beginComputePass({ label: "RA Density Accumulate" });
        pass.setPipeline(p.densityAccumulate);
        pass.setBindGroup(0, bg.density);
        pass.dispatchWorkgroups(nodeWorkgroups);
        pass.end();
      }

      // Apply density gradient forces (away from high density)
      {
        const pass = encoder.beginComputePass({ label: "RA Density Apply Forces" });
        pass.setPipeline(p.densityApplyForces);
        pass.setBindGroup(0, bg.density);
        pass.dispatchWorkgroups(nodeWorkgroups);
        pass.end();
      }
    }

    // === PHASE 5: Sibling repulsion (every frame) ===
    {
      const pass = encoder.beginComputePass({ label: "RA Sibling Repulsion" });
      pass.setPipeline(p.siblingRepulsion);
      pass.setBindGroup(0, bg.sibling);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }

    // === PHASE 6: Mass-weighted gravity (every frame) ===
    {
      const pass = encoder.beginComputePass({ label: "RA Gravity" });
      pass.setPipeline(p.gravity);
      pass.setBindGroup(0, bg.gravity);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }

    // === PHASE 7: LinLog edge attraction (every frame) ===
    // F = log(1 + distance) * direction — always pulling, no rest length, no
    // equilibrium. Replaces Hooke's law springs which created grid/lattice
    // patterns; bounded magnitude (unlike F = d) so distant children cannot
    // saturate the velocity cap and oscillate. See updateUniforms flags.
    //
    // Under a live LOD cut it runs over the aggregated edge set instead: the
    // visible source edges plus the bundles standing in for everything that
    // crosses a collapse boundary. Every source edge is covered by exactly one
    // of the two, so an edge is never pulled twice and a collapsed subtree's
    // cross-cutting orbit spring is transferred rather than dropped — this is
    // the pass that holds a child to its parent, so dropping it under a
    // collapse let the visible frontier drift free. Zero threads is a legal
    // aggregation and must dispatch nothing rather than fall back to the whole
    // edge array.
    const lodEdges = this.lastLodEdges;
    if (lodEdges !== null) {
      if (lodEdges.total > 0) {
        const pass = encoder.beginComputePass({ label: "RA Linear Attraction (LOD bundles)" });
        pass.setPipeline(p.attractionBundled);
        pass.setBindGroup(0, bg.attractionBundled);
        pass.dispatchWorkgroups(calculateWorkgroups(lodEdges.total, WORKGROUP_SIZE));
        pass.end();
      }
    } else if (this.currentEdgeCount > 0) {
      const edgeWorkgroups = calculateWorkgroups(this.currentEdgeCount, WORKGROUP_SIZE);
      const pass = encoder.beginComputePass({ label: "RA Linear Attraction" });
      pass.setPipeline(p.attraction);
      pass.setBindGroup(0, bg.attraction);
      pass.dispatchWorkgroups(edgeWorkgroups);
      pass.end();
    }
  }

  destroy(): void {
    // Buffers are destroyed via AlgorithmBuffers.destroy()
    this.lastLodEdges = null;
  }
}

/**
 * Create Relativity Atlas force algorithm instance
 */
export function createRelativityAtlasAlgorithm(): ForceAlgorithm {
  return new RelativityAtlasAlgorithm();
}

/**
 * CSR validation error with detailed diagnostic information
 */
export class CSRValidationError extends Error {
  constructor(
    message: string,
    public readonly details: {
      field: string;
      expected?: number | string;
      actual?: number | string;
      nodeCount?: number;
      edgeCount?: number;
    },
  ) {
    super(message);
    this.name = "CSRValidationError";
  }
}

/**
 * Validates CSR (Compressed Sparse Row) data structure integrity.
 *
 * CSR format stores a sparse graph as:
 * - offsets: Array of length (nodeCount + 1) where offsets[i] is the start index
 *   of node i's adjacency list in the indices array, and offsets[nodeCount] equals
 *   the total edge count.
 * - indices: Array of length edgeCount containing target/source node indices.
 *
 * This validation prevents GPU memory corruption by ensuring:
 * 1. Array sizes match expected dimensions
 * 2. Offsets array has exactly nodeCount + 1 elements
 * 3. Indices don't exceed valid node ranges
 * 4. Edge count matches actual data size
 * 5. Offsets are monotonically non-decreasing
 * 6. Indices array doesn't exceed buffer capacity (DoS protection)
 * 7. All indices are non-negative (defense-in-depth)
 *
 * @param offsets - CSR offsets array (nodeCount + 1 elements)
 * @param indices - CSR indices array (edge targets or sources)
 * @param nodeCount - Expected number of nodes
 * @param maxEdges - Maximum allowed edges (buffer capacity)
 * @param label - Label for error messages (e.g., "Forward CSR" or "Inverse CSR")
 * @throws CSRValidationError if validation fails
 */
export function validateCSRData(
  offsets: Uint32Array,
  indices: Uint32Array,
  nodeCount: number,
  maxEdges: number,
  label: string,
): { edgeCount: number } {
  const expectedOffsetsLength = nodeCount + 1;

  // Validation 1: Offsets array must be correctly sized
  if (offsets.length < expectedOffsetsLength) {
    throw new CSRValidationError(
      `${label}: Offsets array too small. Expected ${expectedOffsetsLength} elements ` +
        `(nodeCount + 1), but got ${offsets.length} elements.`,
      {
        field: "offsets.length",
        expected: expectedOffsetsLength,
        actual: offsets.length,
        nodeCount,
      },
    );
  }

  // Validation 2: First offset must be 0
  if (offsets[0] !== 0) {
    throw new CSRValidationError(
      `${label}: First offset must be 0, but got ${offsets[0]}. ` +
        `CSR offsets must start at 0 as the base index into the indices array.`,
      {
        field: "offsets[0]",
        expected: 0,
        actual: offsets[0],
        nodeCount,
      },
    );
  }

  // Validation 3: Last offset equals edge count and matches indices array length
  const declaredEdgeCount = offsets[nodeCount];
  if (declaredEdgeCount !== indices.length) {
    throw new CSRValidationError(
      `${label}: Edge count mismatch. Offsets array declares ${declaredEdgeCount} edges ` +
        `(offsets[${nodeCount}] = ${declaredEdgeCount}), but indices array has ${indices.length} elements. ` +
        `These must match exactly.`,
      {
        field: "edgeCount",
        expected: declaredEdgeCount,
        actual: indices.length,
        nodeCount,
        edgeCount: declaredEdgeCount,
      },
    );
  }

  // Validation 4: Edge count doesn't exceed buffer capacity
  if (declaredEdgeCount > maxEdges) {
    throw new CSRValidationError(
      `${label}: Edge count ${declaredEdgeCount} exceeds buffer capacity ${maxEdges}. ` +
        `Buffers must be recreated with larger capacity before uploading this data.`,
      {
        field: "edgeCount",
        expected: `<= ${maxEdges}`,
        actual: declaredEdgeCount,
        nodeCount,
        edgeCount: declaredEdgeCount,
      },
    );
  }

  // Validation 5: Offsets are monotonically non-decreasing
  for (let i = 1; i <= nodeCount; i++) {
    if (offsets[i] < offsets[i - 1]) {
      throw new CSRValidationError(
        `${label}: Offsets must be monotonically non-decreasing. ` +
          `Found offsets[${i}] = ${offsets[i]} < offsets[${i - 1}] = ${offsets[i - 1]}. ` +
          `This indicates corrupted CSR data.`,
        {
          field: `offsets[${i}]`,
          expected: `>= ${offsets[i - 1]}`,
          actual: offsets[i],
          nodeCount,
        },
      );
    }
  }

  // Validation 6: Indices array doesn't exceed buffer capacity
  if (indices.length > maxEdges) {
    throw new CSRValidationError(
      `${label}: Indices array exceeds maximum edge capacity (${indices.length} > ${maxEdges}).`,
      {
        field: "indices.length",
        expected: `<= ${maxEdges}`,
        actual: indices.length,
        nodeCount,
        edgeCount: declaredEdgeCount,
      },
    );
  }

  // Validation 7: All indices are valid node references (in range [0, nodeCount))
  // Critical for GPU safety — out-of-range indices cause memory corruption
  for (let i = 0; i < indices.length; i++) {
    const targetIndex = indices[i];
    if (targetIndex >= nodeCount) {
      // Find which node this edge belongs to for better error reporting
      let sourceNode = 0;
      for (let n = 0; n < nodeCount; n++) {
        if (offsets[n + 1] > i) {
          sourceNode = n;
          break;
        }
      }
      throw new CSRValidationError(
        `${label}: Invalid node index at indices[${i}] = ${targetIndex}. ` +
          `Index must be < ${nodeCount} (nodeCount). This edge originates from node ${sourceNode}. ` +
          `Out-of-range indices cause GPU memory corruption.`,
        {
          field: `indices[${i}]`,
          expected: `< ${nodeCount}`,
          actual: targetIndex,
          nodeCount,
          edgeCount: declaredEdgeCount,
        },
      );
    }
  }

  return { edgeCount: declaredEdgeCount };
}

/**
 * CSR data with separate offsets and indices arrays.
 */
export interface CSRData {
  /** Offsets array (nodeCount + 1 elements) */
  offsets: Uint32Array;
  /** Indices array (edge targets or sources) */
  indices: Uint32Array;
}

/**
 * Upload CSR edge data to algorithm buffers with comprehensive validation.
 *
 * Validates CSR data integrity before uploading to GPU buffers to prevent
 * memory corruption from malformed data.
 *
 * @param device - GPU device
 * @param buffers - Relativity Atlas buffers
 * @param forwardCSR - Forward CSR (outgoing edges: offsets + targets)
 * @param inverseCSR - Inverse CSR (incoming edges: offsets + sources)
 * @param nodeCount - Actual node count (must match CSR offsets sizing)
 * @throws CSRValidationError if CSR data is malformed
 */
export function uploadRelativityAtlasEdges(
  device: GPUDevice,
  buffers: AlgorithmBuffers,
  forwardCSR: CSRData,
  inverseCSR: CSRData,
  nodeCount: number,
): void {
  const b = buffers as RelativityAtlasBuffers;

  // Validate node count doesn't exceed buffer capacity
  if (nodeCount > b.maxNodes) {
    throw new CSRValidationError(
      `Node count ${nodeCount} exceeds buffer capacity ${b.maxNodes}. ` +
        `Recreate buffers with createBuffers() using a larger maxNodes value.`,
      {
        field: "nodeCount",
        expected: `<= ${b.maxNodes}`,
        actual: nodeCount,
        nodeCount,
      },
    );
  }

  // Validate forward and inverse CSR data
  const forwardResult = validateCSRData(
    forwardCSR.offsets,
    forwardCSR.indices,
    nodeCount,
    b.maxEdges,
    "Forward CSR",
  );

  const inverseResult = validateCSRData(
    inverseCSR.offsets,
    inverseCSR.indices,
    nodeCount,
    b.maxEdges,
    "Inverse CSR",
  );

  // Cross-validation: forward and inverse CSR should have same edge count
  if (forwardResult.edgeCount !== inverseResult.edgeCount) {
    throw new CSRValidationError(
      `Edge count mismatch between forward and inverse CSR. ` +
        `Forward CSR has ${forwardResult.edgeCount} edges, inverse CSR has ${inverseResult.edgeCount} edges. ` +
        `Both must represent the same graph and have identical edge counts.`,
      {
        field: "edgeCount",
        expected: forwardResult.edgeCount,
        actual: inverseResult.edgeCount,
        nodeCount,
        edgeCount: forwardResult.edgeCount,
      },
    );
  }

  // Upload forward CSR offsets
  const forwardOffsetBuffer = new ArrayBuffer(forwardCSR.offsets.byteLength);
  new Uint32Array(forwardOffsetBuffer).set(forwardCSR.offsets);
  device.queue.writeBuffer(b.csrOffsets, 0, forwardOffsetBuffer);

  // Upload forward CSR targets
  if (forwardCSR.indices.length > 0) {
    const forwardTargetBuffer = new ArrayBuffer(forwardCSR.indices.byteLength);
    new Uint32Array(forwardTargetBuffer).set(forwardCSR.indices);
    device.queue.writeBuffer(b.csrTargets, 0, forwardTargetBuffer);
  }

  // Upload the inverse CSR's two regions into the one buffer they share; see
  // csrInverseSourcesBase for the map. Two writes rather than a concatenated
  // one: the regions are allocated at capacity, so the source row starts at a
  // fixed offset whatever the live offset row's length.
  const inverseOffsetBuffer = new ArrayBuffer(inverseCSR.offsets.byteLength);
  new Uint32Array(inverseOffsetBuffer).set(inverseCSR.offsets);
  device.queue.writeBuffer(b.csrInverse, 0, inverseOffsetBuffer);

  if (inverseCSR.indices.length > 0) {
    const inverseSourceBuffer = new ArrayBuffer(inverseCSR.indices.byteLength);
    new Uint32Array(inverseSourceBuffer).set(inverseCSR.indices);
    device.queue.writeBuffer(
      b.csrInverse,
      csrInverseSourcesBase(b.maxNodes) * Uint32Array.BYTES_PER_ELEMENT,
      inverseSourceBuffer,
    );
  }
}
