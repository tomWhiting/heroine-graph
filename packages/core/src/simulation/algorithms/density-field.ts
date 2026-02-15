/**
 * Density Field Force Algorithm - O(n) GPU Implementation
 *
 * Uses a density grid to approximate repulsion forces:
 * 1. Clear grid
 * 2. Each node splatters density to nearby cells
 * 3. Each node samples gradient and moves away from high density
 *
 * This is the fastest algorithm for very large graphs (100K+ nodes)
 * but produces less precise results than pairwise methods.
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

// Import shader source
import DENSITY_FIELD_WGSL from "../shaders/density_field.comp.wgsl?raw";

/**
 * Density Field algorithm info
 */
const DENSITY_FIELD_ALGORITHM_INFO: ForceAlgorithmInfo = {
  id: "density",
  name: "Density Field",
  description:
    "O(n) grid-based approximation. Best for very large graphs (100K+ nodes).",
  minNodes: 1000,
  maxNodes: -1, // Unlimited
  complexity: "O(n)",
};

// Grid configuration — densityGridSize from ForceConfig overrides at runtime
const MAX_GRID_SIZE = 512; // Maximum supported grid size (buffer allocated at this size)
const DEFAULT_GRID_SIZE = 128; // Default used when ForceConfig not available
const DEFAULT_SPLAT_RADIUS = 3.0; // In grid cells
const WORKGROUP_SIZE = 256;

/**
 * Extended pipelines for Density Field
 */
interface DensityFieldPipelines extends AlgorithmPipelines {
  clearGrid: GPUComputePipeline;
  accumulateDensity: GPUComputePipeline;
  // 'repulsion' from base interface is the apply_forces pass

  // Shared bind group layout
  bindGroupLayout: GPUBindGroupLayout;
}

/**
 * Density Field algorithm-specific buffers
 */
class DensityFieldBuffers implements AlgorithmBuffers {
  constructor(
    public uniforms: GPUBuffer,
    public densityGrid: GPUBuffer,
    public wellRadius: GPUBuffer,
  ) {}

  destroy(): void {
    this.uniforms.destroy();
    this.densityGrid.destroy();
    this.wellRadius.destroy();
  }
}

/**
 * Density Field repulsion algorithm
 */
export class DensityFieldAlgorithm implements ForceAlgorithm {
  readonly info = DENSITY_FIELD_ALGORITHM_INFO;
  readonly handlesGravity = false;

  /** Current grid size from ForceConfig, updated each frame in updateUniforms */
  private currentGridSize = DEFAULT_GRID_SIZE;

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

    // Create shader module
    const shaderModule = device.createShaderModule({
      label: "Density Field Shader",
      code: DENSITY_FIELD_WGSL,
    });

    // Create explicit bind group layout
    // Bindings: uniforms, positions (vec2), forces (vec2), density_grid
    const bindGroupLayout = device.createBindGroupLayout({
      label: "Density Field Bind Group Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      label: "Density Field Pipeline Layout",
      bindGroupLayouts: [bindGroupLayout],
    });

    const pipelines: DensityFieldPipelines = {
      clearGrid: device.createComputePipeline({
        label: "Density Field Clear Grid",
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: "clear_grid" },
      }),
      accumulateDensity: device.createComputePipeline({
        label: "Density Field Accumulate",
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: "accumulate_density" },
      }),
      repulsion: device.createComputePipeline({
        label: "Density Field Apply Forces",
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: "apply_forces" },
      }),
      bindGroupLayout,
    };

    return pipelines;
  }

  createBuffers(device: GPUDevice, maxNodes: number): AlgorithmBuffers {
    const safeMaxNodes = Math.max(maxNodes, 4);

    // Uniforms: DensityUniforms struct (48 bytes = 12 x f32)
    const uniforms = device.createBuffer({
      label: "Density Field Uniforms",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Density grid: allocate for max grid size (512x512 = 1MB)
    // Runtime grid size from ForceConfig uses a subset of this buffer.
    const gridCells = MAX_GRID_SIZE * MAX_GRID_SIZE;
    const densityGrid = device.createBuffer({
      label: "Density Grid",
      size: gridCells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Well radius buffer (zero-filled = uses default splat_radius in shader)
    const wellRadius = device.createBuffer({
      label: "Density Field Well Radius",
      size: safeMaxNodes * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new DensityFieldBuffers(uniforms, densityGrid, wellRadius);
  }

  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelines,
    context: AlgorithmRenderContext,
    algorithmBuffers: AlgorithmBuffers,
  ): AlgorithmBindGroups {
    const p = pipelines as DensityFieldPipelines;
    const b = algorithmBuffers as DensityFieldBuffers;

    const repulsion = device.createBindGroup({
      label: "Density Field Bind Group",
      layout: p.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: b.uniforms } },
        { binding: 1, resource: { buffer: context.positions } },
        { binding: 2, resource: { buffer: context.forces } },
        { binding: 3, resource: { buffer: b.densityGrid } },
        { binding: 4, resource: { buffer: b.wellRadius } },
      ],
    });

    return { repulsion };
  }

  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: AlgorithmBuffers,
    context: AlgorithmRenderContext,
  ): void {
    const b = algorithmBuffers as DensityFieldBuffers;

    // Density field REQUIRES bounds for correct grid mapping.
    // Without bounds, positions cannot be mapped to grid cells correctly,
    // causing density accumulation to be concentrated in wrong areas.
    if (!context.bounds) {
      throw new Error(
        "Density field algorithm requires bounds to be provided in AlgorithmRenderContext. " +
        "Bounds must be computed from actual node positions. Without bounds, the density " +
        "grid cannot correctly map positions to cells."
      );
    }

    const boundsMinX = context.bounds.minX;
    const boundsMinY = context.bounds.minY;
    const boundsMaxX = context.bounds.maxX;
    const boundsMaxY = context.bounds.maxY;

    // Uniforms (48 bytes = 12 x f32)
    const data = new ArrayBuffer(48);
    const view = new DataView(data);
    const gridSize = context.forceConfig.densityGridSize || DEFAULT_GRID_SIZE;
    this.currentGridSize = gridSize;
    view.setUint32(0, context.nodeCount, true);           // node_count
    view.setUint32(4, gridSize, true);                    // grid_width
    view.setUint32(8, gridSize, true);                    // grid_height
    view.setFloat32(12, Math.abs(context.forceConfig.repulsionStrength), true); // repulsion_strength
    view.setFloat32(16, boundsMinX, true);                // bounds_min_x
    view.setFloat32(20, boundsMinY, true);                // bounds_min_y
    view.setFloat32(24, boundsMaxX, true);                // bounds_max_x
    view.setFloat32(28, boundsMaxY, true);                // bounds_max_y
    view.setFloat32(32, DEFAULT_SPLAT_RADIUS, true);      // splat_radius
    view.setFloat32(36, 0, true);                         // _pad1
    view.setFloat32(40, 0, true);                         // _pad2
    view.setFloat32(44, 0, true);                         // _pad3
    device.queue.writeBuffer(b.uniforms, 0, data);
  }

  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: AlgorithmPipelines,
    bindGroups: AlgorithmBindGroups,
    nodeCount: number,
  ): void {
    const p = pipelines as DensityFieldPipelines;

    const nodeWorkgroups = calculateWorkgroups(nodeCount, WORKGROUP_SIZE);
    const gridCells = this.currentGridSize * this.currentGridSize;
    const gridWorkgroups = calculateWorkgroups(gridCells, WORKGROUP_SIZE);

    // Phase 1: Clear density grid
    {
      const pass = encoder.beginComputePass({ label: "Density Clear Grid" });
      pass.setPipeline(p.clearGrid);
      pass.setBindGroup(0, bindGroups.repulsion);
      pass.dispatchWorkgroups(gridWorkgroups);
      pass.end();
    }

    // Phase 2: Accumulate density from nodes
    {
      const pass = encoder.beginComputePass({ label: "Density Accumulate" });
      pass.setPipeline(p.accumulateDensity);
      pass.setBindGroup(0, bindGroups.repulsion);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }

    // Phase 3: Apply forces based on gradient
    {
      const pass = encoder.beginComputePass({ label: "Density Apply Forces" });
      pass.setPipeline(p.repulsion);
      pass.setBindGroup(0, bindGroups.repulsion);
      pass.dispatchWorkgroups(nodeWorkgroups);
      pass.end();
    }
  }
}

/**
 * Create Density Field force algorithm instance
 */
export function createDensityFieldAlgorithm(): ForceAlgorithm {
  return new DensityFieldAlgorithm();
}
