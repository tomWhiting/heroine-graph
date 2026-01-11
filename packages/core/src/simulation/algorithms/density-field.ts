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

// Grid configuration
const GRID_SIZE = 128; // 128x128 grid
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
  ) {}

  destroy(): void {
    this.uniforms.destroy();
    this.densityGrid.destroy();
  }
}

/**
 * Density Field repulsion algorithm
 */
export class DensityFieldAlgorithm implements ForceAlgorithm {
  readonly info = DENSITY_FIELD_ALGORITHM_INFO;

  createPipelines(context: GPUContext): AlgorithmPipelines {
    const { device } = context;

    // Create shader module
    const shaderModule = device.createShaderModule({
      label: "Density Field Shader",
      code: DENSITY_FIELD_WGSL,
    });

    // Create explicit bind group layout
    // Bindings: uniforms, positions_x, positions_y, forces_x, forces_y, density_grid
    const bindGroupLayout = device.createBindGroupLayout({
      label: "Density Field Bind Group Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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

  createBuffers(device: GPUDevice, _maxNodes: number): AlgorithmBuffers {
    // Uniforms: DensityUniforms struct (48 bytes = 12 x f32)
    const uniforms = device.createBuffer({
      label: "Density Field Uniforms",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Density grid: GRID_SIZE x GRID_SIZE cells, each u32
    const gridCells = GRID_SIZE * GRID_SIZE;
    const densityGrid = device.createBuffer({
      label: "Density Grid",
      size: gridCells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    return new DensityFieldBuffers(uniforms, densityGrid);
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
        { binding: 1, resource: { buffer: context.positionsX } },
        { binding: 2, resource: { buffer: context.positionsY } },
        { binding: 3, resource: { buffer: context.forcesX } },
        { binding: 4, resource: { buffer: context.forcesY } },
        { binding: 5, resource: { buffer: b.densityGrid } },
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

    // Use large bounds for the density grid
    const boundsMin = -5000.0;
    const boundsMax = 5000.0;

    // Uniforms (48 bytes = 12 x f32)
    const data = new ArrayBuffer(48);
    const view = new DataView(data);
    view.setUint32(0, context.nodeCount, true);           // node_count
    view.setUint32(4, GRID_SIZE, true);                   // grid_width
    view.setUint32(8, GRID_SIZE, true);                   // grid_height
    view.setFloat32(12, Math.abs(context.forceConfig.repulsionStrength), true); // repulsion_strength
    view.setFloat32(16, boundsMin, true);                 // bounds_min_x
    view.setFloat32(20, boundsMin, true);                 // bounds_min_y
    view.setFloat32(24, boundsMax, true);                 // bounds_max_x
    view.setFloat32(28, boundsMax, true);                 // bounds_max_y
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
    const gridCells = GRID_SIZE * GRID_SIZE;
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
