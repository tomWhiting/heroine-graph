/**
 * Heatmap Render Pipeline
 *
 * Creates and manages the GPU render pipelines for heatmap visualization:
 * 1. Splat pass: Renders Gaussian splats additively to density texture
 * 2. Colormap pass: Maps density values to colors and composites to screen
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";
import type { DensityTexture } from "./texture.ts";
import type { ColorScaleTexture } from "./colorscale.ts";

// Import shader source code
import splatVertSource from "./shaders/splat.vert.wgsl?raw";
import splatFragSource from "./shaders/splat.frag.wgsl?raw";
import colormapVertSource from "./shaders/colormap.vert.wgsl?raw";
import colormapFragSource from "./shaders/colormap.frag.wgsl?raw";

/**
 * Heatmap uniform buffer data
 */
export interface HeatmapUniforms {
  /** Splat radius in graph units */
  radius: number;
  /** Intensity multiplier */
  intensity: number;
}

/**
 * Colormap uniform buffer data
 */
export interface ColormapUniforms {
  /** Minimum density for normalization */
  minDensity: number;
  /** Maximum density for normalization */
  maxDensity: number;
  /** Opacity of heatmap overlay */
  opacity: number;
}

/**
 * Heatmap pipeline resources
 */
export interface HeatmapPipeline {
  /** Splat render pipeline */
  splatPipeline: GPURenderPipeline;
  /** Colormap render pipeline */
  colormapPipeline: GPURenderPipeline;
  /** Heatmap uniforms bind group layout (group 0) */
  uniformsLayout: GPUBindGroupLayout;
  /** Position data bind group layout (group 1) */
  positionLayout: GPUBindGroupLayout;
  /** Colormap bind group layout */
  colormapLayout: GPUBindGroupLayout;
  /** Heatmap uniform buffer */
  heatmapUniformBuffer: GPUBuffer;
  /** Colormap uniform buffer */
  colormapUniformBuffer: GPUBuffer;
  /** Update heatmap uniforms */
  updateHeatmapUniforms: (uniforms: HeatmapUniforms) => void;
  /** Update colormap uniforms */
  updateColormapUniforms: (uniforms: ColormapUniforms) => void;
  /** Create splat bind group for positions */
  createSplatBindGroup: (
    viewportUniformBuffer: GPUBuffer,
    positionsX: GPUBuffer,
    positionsY: GPUBuffer
  ) => { uniformsBindGroup: GPUBindGroup; positionBindGroup: GPUBindGroup };
  /** Create colormap bind group */
  createColormapBindGroup: (
    densityTexture: DensityTexture,
    colorScale: ColorScaleTexture
  ) => GPUBindGroup;
  /** Destroy resources */
  destroy: () => void;
}

/**
 * Default heatmap configuration
 */
export const DEFAULT_HEATMAP_UNIFORMS: HeatmapUniforms = {
  radius: 50.0,
  intensity: 0.1,
};

/**
 * Default colormap configuration
 */
export const DEFAULT_COLORMAP_UNIFORMS: ColormapUniforms = {
  minDensity: 0.0,
  maxDensity: 1.0,
  opacity: 0.7,
};

/**
 * Creates the heatmap render pipelines
 */
export function createHeatmapPipeline(context: GPUContext): HeatmapPipeline {
  const { device } = context;

  // Create shader modules
  const splatVertModule = device.createShaderModule({
    label: "Heatmap Splat Vertex Shader",
    code: splatVertSource,
  });

  const splatFragModule = device.createShaderModule({
    label: "Heatmap Splat Fragment Shader",
    code: splatFragSource,
  });

  const colormapVertModule = device.createShaderModule({
    label: "Heatmap Colormap Vertex Shader",
    code: colormapVertSource,
  });

  const colormapFragModule = device.createShaderModule({
    label: "Heatmap Colormap Fragment Shader",
    code: colormapFragSource,
  });

  // Create bind group layouts for splat pass

  // Group 0: Viewport and heatmap uniforms
  const uniformsLayout = device.createBindGroupLayout({
    label: "Heatmap Uniforms Layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
    ],
  });

  // Group 1: Position buffers
  const positionLayout = device.createBindGroupLayout({
    label: "Heatmap Position Layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
      },
    ],
  });

  // Create splat pipeline layout
  const splatPipelineLayout = device.createPipelineLayout({
    label: "Heatmap Splat Pipeline Layout",
    bindGroupLayouts: [uniformsLayout, positionLayout],
  });

  // Create splat render pipeline with additive blending
  const splatPipeline = device.createRenderPipeline({
    label: "Heatmap Splat Pipeline",
    layout: splatPipelineLayout,
    vertex: {
      module: splatVertModule,
      entryPoint: "vs_main",
    },
    fragment: {
      module: splatFragModule,
      entryPoint: "fs_main",
      targets: [
        {
          format: "rgba16float",
          blend: {
            color: {
              srcFactor: "one",
              dstFactor: "one",
              operation: "add",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one",
              operation: "add",
            },
          },
        },
      ],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  // Create colormap bind group layout
  const colormapLayout = device.createBindGroupLayout({
    label: "Heatmap Colormap Layout",
    entries: [
      // Density texture
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float" },
      },
      // Density sampler
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "filtering" },
      },
      // Color scale texture (1D)
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float", viewDimension: "1d" },
      },
      // Color scale sampler
      {
        binding: 3,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "filtering" },
      },
      // Colormap uniforms
      {
        binding: 4,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
    ],
  });

  // Create colormap pipeline layout
  const colormapPipelineLayout = device.createPipelineLayout({
    label: "Heatmap Colormap Pipeline Layout",
    bindGroupLayouts: [colormapLayout],
  });

  // Create colormap render pipeline with alpha blending
  const colormapPipeline = device.createRenderPipeline({
    label: "Heatmap Colormap Pipeline",
    layout: colormapPipelineLayout,
    vertex: {
      module: colormapVertModule,
      entryPoint: "vs_main",
    },
    fragment: {
      module: colormapFragModule,
      entryPoint: "fs_main",
      targets: [
        {
          format: "bgra8unorm",
          blend: {
            color: {
              srcFactor: "src-alpha",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
          },
        },
      ],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  // Create uniform buffers
  const heatmapUniformBuffer = device.createBuffer({
    label: "Heatmap Uniform Buffer",
    size: 16, // 2 floats + 2 padding
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const colormapUniformBuffer = device.createBuffer({
    label: "Colormap Uniform Buffer",
    size: 16, // 3 floats + 1 padding
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Initialize with defaults
  const uniformData = new Float32Array([
    DEFAULT_HEATMAP_UNIFORMS.radius,
    DEFAULT_HEATMAP_UNIFORMS.intensity,
    0,
    0, // padding
  ]);
  device.queue.writeBuffer(heatmapUniformBuffer, 0, uniformData);

  const colormapData = new Float32Array([
    DEFAULT_COLORMAP_UNIFORMS.minDensity,
    DEFAULT_COLORMAP_UNIFORMS.maxDensity,
    DEFAULT_COLORMAP_UNIFORMS.opacity,
    0, // padding
  ]);
  device.queue.writeBuffer(colormapUniformBuffer, 0, colormapData);

  function updateHeatmapUniforms(uniforms: HeatmapUniforms): void {
    const data = new Float32Array([uniforms.radius, uniforms.intensity, 0, 0]);
    device.queue.writeBuffer(heatmapUniformBuffer, 0, data);
  }

  function updateColormapUniforms(uniforms: ColormapUniforms): void {
    const data = new Float32Array([
      uniforms.minDensity,
      uniforms.maxDensity,
      uniforms.opacity,
      0,
    ]);
    device.queue.writeBuffer(colormapUniformBuffer, 0, data);
  }

  function createSplatBindGroup(
    viewportUniformBuffer: GPUBuffer,
    positionsX: GPUBuffer,
    positionsY: GPUBuffer
  ): { uniformsBindGroup: GPUBindGroup; positionBindGroup: GPUBindGroup } {
    const uniformsBindGroup = device.createBindGroup({
      label: "Heatmap Uniforms Bind Group",
      layout: uniformsLayout,
      entries: [
        { binding: 0, resource: { buffer: viewportUniformBuffer } },
        { binding: 1, resource: { buffer: heatmapUniformBuffer } },
      ],
    });

    const positionBindGroup = device.createBindGroup({
      label: "Heatmap Position Bind Group",
      layout: positionLayout,
      entries: [
        { binding: 0, resource: { buffer: positionsX } },
        { binding: 1, resource: { buffer: positionsY } },
      ],
    });

    return { uniformsBindGroup, positionBindGroup };
  }

  function createColormapBindGroup(
    densityTexture: DensityTexture,
    colorScale: ColorScaleTexture
  ): GPUBindGroup {
    return device.createBindGroup({
      label: "Heatmap Colormap Bind Group",
      layout: colormapLayout,
      entries: [
        { binding: 0, resource: densityTexture.sampleView },
        { binding: 1, resource: densityTexture.sampler },
        { binding: 2, resource: colorScale.view },
        { binding: 3, resource: colorScale.sampler },
        { binding: 4, resource: { buffer: colormapUniformBuffer } },
      ],
    });
  }

  function destroy(): void {
    heatmapUniformBuffer.destroy();
    colormapUniformBuffer.destroy();
  }

  return {
    splatPipeline,
    colormapPipeline,
    uniformsLayout,
    positionLayout,
    colormapLayout,
    heatmapUniformBuffer,
    colormapUniformBuffer,
    updateHeatmapUniforms,
    updateColormapUniforms,
    createSplatBindGroup,
    createColormapBindGroup,
    destroy,
  };
}
