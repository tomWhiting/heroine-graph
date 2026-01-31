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
import type {
  AlgorithmBindGroups,
  AlgorithmBuffers,
  AlgorithmPipelines,
  AlgorithmRenderContext,
  ForceAlgorithm,
  ForceAlgorithmInfo,
  ForceAlgorithmType,
} from "./types.ts";

// Import shader sources
import DEGREES_WGSL from "../shaders/relativity_degrees.comp.wgsl";
import MASS_WGSL from "../shaders/relativity_mass.comp.wgsl";
import SIBLING_WGSL from "../shaders/relativity_sibling.comp.wgsl";
import GRAVITY_WGSL from "../shaders/relativity_gravity.comp.wgsl";

/**
 * Relativity Atlas algorithm info
 */
const RELATIVITY_ATLAS_INFO: ForceAlgorithmInfo = {
  id: "relativity-atlas" as ForceAlgorithmType,
  name: "Relativity Atlas",
  description:
    "Hierarchical O(N+E) for directed/hierarchical graphs. Optimal for DAGs and trees.",
  minNodes: 100,
  maxNodes: -1, // Unlimited
  complexity: "O(N + E)",
};

// Configuration constants
const WORKGROUP_SIZE = 256;
const MASS_CONVERGENCE_THRESHOLD = 0.01;
const DEFAULT_MAX_SIBLINGS = 100;
const MAX_MASS_ITERATIONS = 10;

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

  // Bind group layouts
  degreesLayout: GPUBindGroupLayout;
  massLayout: GPUBindGroupLayout;
  siblingLayout: GPUBindGroupLayout;
  gravityLayout: GPUBindGroupLayout;
}

/**
 * Relativity Atlas algorithm-specific buffers
 */
class RelativityAtlasBuffers implements AlgorithmBuffers {
  constructor(
    // Uniform buffers
    public degreesUniforms: GPUBuffer,
    public massUniforms: GPUBuffer,
    public siblingUniforms: GPUBuffer,
    public gravityUniforms: GPUBuffer,

    // CSR edge data (outgoing edges)
    public csrOffsets: GPUBuffer,
    public csrTargets: GPUBuffer,

    // Inverse CSR (incoming edges / parents)
    public csrInverseOffsets: GPUBuffer,
    public csrInverseSources: GPUBuffer,

    // Computed data
    public degrees: GPUBuffer,        // [out_deg, in_deg] pairs
    public mass: GPUBuffer,           // Ping buffer
    public massOut: GPUBuffer,        // Pong buffer
    public converged: GPUBuffer,      // Atomic convergence flag

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
    this.csrInverseOffsets.destroy();
    this.csrInverseSources.destroy();

    this.degrees.destroy();
    this.mass.destroy();
    this.massOut.destroy();
    this.converged.destroy();
  }
}

/**
 * Extended bind groups for Relativity Atlas
 */
interface RelativityAtlasBindGroups extends AlgorithmBindGroups {
  degrees: GPUBindGroup;
  massInit: GPUBindGroup;
  massAggregate: GPUBindGroup[];  // Ping-pong
  sibling: GPUBindGroup;
  gravity: GPUBindGroup;
  // repulsion from base is used for main force pass
}

/**
 * Relativity Atlas force algorithm implementation
 */
export class RelativityAtlasAlgorithm implements ForceAlgorithm {
  readonly info = RELATIVITY_ATLAS_INFO;

  // Track whether mass has been initialized for current graph
  private massInitialized = false;
  private lastNodeCount = 0;

  /**
   * Reset mass state when graph changes
   */
  resetMassState(): void {
    this.massInitialized = false;
  }

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

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
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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
    //           csr_offsets, csr_targets, node_mass
    const siblingLayout = device.createBindGroupLayout({
      label: "RA Sibling Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    // === Gravity Layout ===
    // Bindings: uniforms, positions (vec2), forces (vec2), node_mass
    const gravityLayout = device.createBindGroupLayout({
      label: "RA Gravity Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
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

      // Use sibling repulsion as the main "repulsion" pass
      repulsion: device.createComputePipeline({
        label: "RA Main Repulsion",
        layout: siblingPipelineLayout,
        compute: { module: siblingModule, entryPoint: "main" },
      }),

      degreesLayout,
      massLayout,
      siblingLayout,
      gravityLayout,
    };

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

    const massUniforms = device.createBuffer({
      label: "RA Mass Uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const siblingUniforms = device.createBuffer({
      label: "RA Sibling Uniforms",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

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

    const csrInverseOffsets = device.createBuffer({
      label: "RA CSR Inverse Offsets",
      size: offsetBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const csrInverseSources = device.createBuffer({
      label: "RA CSR Inverse Sources",
      size: edgeBytes,
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

    return new RelativityAtlasBuffers(
      degreesUniforms,
      massUniforms,
      siblingUniforms,
      gravityUniforms,
      csrOffsets,
      csrTargets,
      csrInverseOffsets,
      csrInverseSources,
      degrees,
      mass,
      massOut,
      converged,
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

    // Degrees bind group
    const degrees = device.createBindGroup({
      label: "RA Degrees Bind Group",
      layout: p.degreesLayout,
      entries: [
        { binding: 0, resource: { buffer: b.degreesUniforms } },
        { binding: 1, resource: { buffer: b.csrOffsets } },
        { binding: 2, resource: { buffer: b.csrTargets } },
        { binding: 3, resource: { buffer: b.csrInverseOffsets } },
        { binding: 4, resource: { buffer: b.csrInverseSources } },
        { binding: 5, resource: { buffer: b.degrees } },
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
        { binding: 4, resource: { buffer: b.mass } },      // unused for init
        { binding: 5, resource: { buffer: b.massOut } },   // output
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
          { binding: 4, resource: { buffer: b.massOut } },  // read
          { binding: 5, resource: { buffer: b.mass } },     // write
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
          { binding: 4, resource: { buffer: b.mass } },     // read
          { binding: 5, resource: { buffer: b.massOut } },  // write
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
        { binding: 3, resource: { buffer: b.csrInverseOffsets } },
        { binding: 4, resource: { buffer: b.csrInverseSources } },
        { binding: 5, resource: { buffer: b.csrOffsets } },
        { binding: 6, resource: { buffer: b.csrTargets } },
        { binding: 7, resource: { buffer: b.massOut } },  // Final mass is in massOut
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
        { binding: 3, resource: { buffer: b.massOut } },  // Final mass is in massOut
      ],
    });

    const bindGroups: RelativityAtlasBindGroups = {
      degrees,
      massInit,
      massAggregate,
      sibling,
      gravity,
      repulsion: sibling,  // Use sibling as main repulsion
    };

    return bindGroups;
  }

  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void {
    const b = algorithmBuffers as RelativityAtlasBuffers;

    // Degrees uniforms (16 bytes)
    const degreesData = new ArrayBuffer(16);
    const degreesView = new DataView(degreesData);
    degreesView.setUint32(0, context.nodeCount, true);
    degreesView.setUint32(4, 0, true);  // edge_count (set during edge upload)
    degreesView.setUint32(8, 0, true);
    degreesView.setUint32(12, 0, true);
    device.queue.writeBuffer(b.degreesUniforms, 0, degreesData);

    // Mass uniforms (16 bytes)
    const massData = new ArrayBuffer(16);
    const massView = new DataView(massData);
    massView.setUint32(0, context.nodeCount, true);
    massView.setUint32(4, 0, true);  // edge_count
    massView.setUint32(8, 0, true);  // iteration
    massView.setFloat32(12, MASS_CONVERGENCE_THRESHOLD, true);
    device.queue.writeBuffer(b.massUniforms, 0, massData);

    // Sibling uniforms (32 bytes)
    const siblingData = new ArrayBuffer(32);
    const siblingView = new DataView(siblingData);
    siblingView.setUint32(0, context.nodeCount, true);
    siblingView.setUint32(4, 0, true);  // edge_count
    siblingView.setFloat32(8, Math.abs(context.forceConfig.repulsionStrength), true);
    siblingView.setFloat32(12, context.forceConfig.repulsionDistanceMin, true);
    siblingView.setUint32(16, DEFAULT_MAX_SIBLINGS, true);
    siblingView.setUint32(20, 0, true);
    siblingView.setUint32(24, 0, true);
    siblingView.setUint32(28, 0, true);
    device.queue.writeBuffer(b.siblingUniforms, 0, siblingData);

    // Gravity uniforms (32 bytes)
    const gravityData = new ArrayBuffer(32);
    const gravityView = new DataView(gravityData);
    gravityView.setUint32(0, context.nodeCount, true);
    gravityView.setFloat32(4, context.forceConfig.centerStrength, true);
    gravityView.setFloat32(8, context.forceConfig.centerX, true);
    gravityView.setFloat32(12, context.forceConfig.centerY, true);
    gravityView.setFloat32(16, 0.5, true);  // mass_exponent
    gravityView.setUint32(20, 0, true);
    gravityView.setUint32(24, 0, true);
    gravityView.setUint32(28, 0, true);
    device.queue.writeBuffer(b.gravityUniforms, 0, gravityData);
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
      // Run multiple iterations to propagate mass through the hierarchy
      // Each iteration aggregates child masses to parents
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

    // === PHASE 4: Sibling repulsion (every frame) ===
    {
      const pass = encoder.beginComputePass({ label: "RA Sibling Repulsion" });
      pass.setPipeline(p.siblingRepulsion);
      pass.setBindGroup(0, bg.sibling);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }

    // === PHASE 5: Mass-weighted gravity (every frame) ===
    {
      const pass = encoder.beginComputePass({ label: "RA Gravity" });
      pass.setPipeline(p.gravity);
      pass.setBindGroup(0, bg.gravity);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }
  }

  destroy(): void {
    // Buffers are destroyed via AlgorithmBuffers.destroy()
  }
}

/**
 * Create Relativity Atlas force algorithm instance
 */
export function createRelativityAtlasAlgorithm(): ForceAlgorithm {
  return new RelativityAtlasAlgorithm();
}

/**
 * Upload CSR edge data to algorithm buffers
 *
 * @param device - GPU device
 * @param buffers - Relativity Atlas buffers
 * @param csrData - CSR data [offsets..., targets...]
 * @param inverseCsrData - Inverse CSR data [offsets..., sources...]
 */
export function uploadRelativityAtlasEdges(
  device: GPUDevice,
  buffers: AlgorithmBuffers,
  csrData: Uint32Array,
  inverseCsrData: Uint32Array,
): void {
  const b = buffers as RelativityAtlasBuffers;

  // Parse CSR data (offsets are first nodeCount+1 elements)
  const nodeCount = b.maxNodes;
  const offsetsEnd = nodeCount + 1;

  // Upload forward CSR
  if (csrData.length >= offsetsEnd) {
    const offsets = csrData.subarray(0, offsetsEnd);
    const targets = csrData.subarray(offsetsEnd);

    const offsetBuffer = new ArrayBuffer(offsets.byteLength);
    new Uint32Array(offsetBuffer).set(offsets);
    device.queue.writeBuffer(b.csrOffsets, 0, offsetBuffer);

    if (targets.length > 0) {
      const targetBuffer = new ArrayBuffer(targets.byteLength);
      new Uint32Array(targetBuffer).set(targets);
      device.queue.writeBuffer(b.csrTargets, 0, targetBuffer);
    }
  }

  // Upload inverse CSR
  if (inverseCsrData.length >= offsetsEnd) {
    const offsets = inverseCsrData.subarray(0, offsetsEnd);
    const sources = inverseCsrData.subarray(offsetsEnd);

    const offsetBuffer = new ArrayBuffer(offsets.byteLength);
    new Uint32Array(offsetBuffer).set(offsets);
    device.queue.writeBuffer(b.csrInverseOffsets, 0, offsetBuffer);

    if (sources.length > 0) {
      const sourceBuffer = new ArrayBuffer(sources.byteLength);
      new Uint32Array(sourceBuffer).set(sources);
      device.queue.writeBuffer(b.csrInverseSources, 0, sourceBuffer);
    }
  }
}
