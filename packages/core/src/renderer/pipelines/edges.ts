/**
 * Edge Render Pipeline
 *
 * Sets up the WebGPU render pipeline for edge rendering using instanced
 * quad rendering with anti-aliased line rasterization.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";

// Import shader source (bundled as text by esbuild)
import EDGE_VERT_WGSL from "../shaders/edge.vert.wgsl";
import EDGE_FRAG_WGSL from "../shaders/edge.frag.wgsl";

/**
 * Configuration for edge render pipeline
 */
export interface EdgePipelineConfig {
  /** Maximum number of edges to render */
  maxEdges: number;
  /** Sample count for MSAA (1, 4, or 8) */
  sampleCount?: number;
  /** Texture format for color attachment */
  format?: GPUTextureFormat;
}

/**
 * Default edge pipeline configuration
 */
export const DEFAULT_EDGE_PIPELINE_CONFIG: Required<EdgePipelineConfig> = {
  maxEdges: 2_000_000,
  sampleCount: 1,
  format: "bgra8unorm",
};

/**
 * Edge render pipeline resources
 */
export interface EdgeRenderPipeline {
  /** The WebGPU render pipeline */
  pipeline: GPURenderPipeline;
  /** Bind group layout for viewport uniforms */
  viewportBindGroupLayout: GPUBindGroupLayout;
  /** Bind group layout for edge data */
  edgeBindGroupLayout: GPUBindGroupLayout;
  /** Shader module */
  shaderModule: GPUShaderModule;
  /** Pipeline configuration */
  config: Required<EdgePipelineConfig>;
}

/**
 * Creates the edge render pipeline
 *
 * @param context - GPU context
 * @param config - Pipeline configuration
 * @returns Edge render pipeline resources
 */
export function createEdgeRenderPipeline(
  context: GPUContext,
  config: EdgePipelineConfig = {},
): EdgeRenderPipeline {
  const { device } = context;
  const finalConfig = { ...DEFAULT_EDGE_PIPELINE_CONFIG, ...config };

  // Create shader module
  const shaderModule = device.createShaderModule({
    label: "Edge Shader",
    code: `${EDGE_VERT_WGSL}\n${EDGE_FRAG_WGSL}`,
  });

  // Viewport uniform bind group layout (group 0)
  const viewportBindGroupLayout = device.createBindGroupLayout({
    label: "Edge Pipeline - Viewport Uniforms",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
    ],
  });

  // Edge data bind group layout (group 1)
  // - binding 0: positions_x (storage buffer) - node positions for endpoints
  // - binding 1: positions_y (storage buffer)
  // - binding 2: edge_indices (storage buffer) - source/target pairs
  // - binding 3: edge_attrs (storage buffer) - width, color, state
  const edgeBindGroupLayout = device.createBindGroupLayout({
    label: "Edge Pipeline - Edge Data",
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
      {
        binding: 2,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
      },
    ],
  });

  // Create pipeline layout
  const pipelineLayout = device.createPipelineLayout({
    label: "Edge Pipeline Layout",
    bindGroupLayouts: [viewportBindGroupLayout, edgeBindGroupLayout],
  });

  // Create render pipeline
  const pipeline = device.createRenderPipeline({
    label: "Edge Render Pipeline",
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      // No vertex buffers - we use instancing with storage buffers
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [
        {
          format: finalConfig.format,
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
          writeMask: GPUColorWrite.ALL,
        },
      ],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "none", // Edges can be viewed from either side
    },
    multisample: {
      count: finalConfig.sampleCount,
    },
  });

  return {
    pipeline,
    viewportBindGroupLayout,
    edgeBindGroupLayout,
    shaderModule,
    config: finalConfig,
  };
}

/**
 * Creates a bind group for edge rendering
 *
 * @param device - GPU device
 * @param pipeline - Edge render pipeline
 * @param positionsX - Node X position buffer
 * @param positionsY - Node Y position buffer
 * @param edgeIndices - Edge source/target indices buffer
 * @param edgeAttrs - Edge attributes buffer
 * @returns Bind group for edge data
 */
export function createEdgeBindGroup(
  device: GPUDevice,
  pipeline: EdgeRenderPipeline,
  positionsX: GPUBuffer,
  positionsY: GPUBuffer,
  edgeIndices: GPUBuffer,
  edgeAttrs: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: "Edge Data Bind Group",
    layout: pipeline.edgeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: positionsX } },
      { binding: 1, resource: { buffer: positionsY } },
      { binding: 2, resource: { buffer: edgeIndices } },
      { binding: 3, resource: { buffer: edgeAttrs } },
    ],
  });
}

/**
 * Creates a bind group for viewport uniforms (edge pipeline)
 *
 * @param device - GPU device
 * @param pipeline - Edge render pipeline
 * @param viewportUniformBuffer - Viewport uniform buffer
 * @returns Bind group for viewport uniforms
 */
export function createEdgeViewportBindGroup(
  device: GPUDevice,
  pipeline: EdgeRenderPipeline,
  viewportUniformBuffer: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: "Edge Viewport Uniform Bind Group",
    layout: pipeline.viewportBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: viewportUniformBuffer } }],
  });
}

/**
 * Records edge rendering commands
 *
 * @param pass - Render pass encoder
 * @param pipeline - Edge render pipeline
 * @param viewportBindGroup - Viewport uniforms bind group
 * @param edgeBindGroup - Edge data bind group
 * @param edgeCount - Number of edges to render
 */
export function renderEdges(
  pass: GPURenderPassEncoder,
  pipeline: EdgeRenderPipeline,
  viewportBindGroup: GPUBindGroup,
  edgeBindGroup: GPUBindGroup,
  edgeCount: number,
): void {
  if (edgeCount <= 0) return;

  pass.setPipeline(pipeline.pipeline);
  pass.setBindGroup(0, viewportBindGroup);
  pass.setBindGroup(1, edgeBindGroup);

  // 6 vertices per quad (2 triangles), 1 instance per edge
  pass.draw(6, edgeCount);
}

/**
 * Destroys edge render pipeline resources
 *
 * @param pipeline - Edge render pipeline to destroy
 */
export function destroyEdgeRenderPipeline(_pipeline: EdgeRenderPipeline): void {
  // WebGPU resources are garbage collected
}
