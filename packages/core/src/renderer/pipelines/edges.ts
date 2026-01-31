/**
 * Edge Render Pipeline
 *
 * Sets up the WebGPU render pipeline for edge rendering using instanced
 * quad rendering with anti-aliased line rasterization.
 * Supports both straight and curved (conic Bezier) edges.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";
import type { EdgeFlowConfig } from "../../types.ts";
import {
  DEFAULT_EDGE_FLOW_CONFIG,
  EDGE_FLOW_UNIFORM_SIZE,
  writeEdgeFlowUniforms,
} from "../edge_flow.ts";

// Import shader source (bundled as text by esbuild)
import EDGE_VERT_WGSL from "../shaders/edge.vert.wgsl";
import EDGE_FRAG_WGSL from "../shaders/edge.frag.wgsl";

/**
 * Configuration for curved edges
 */
export interface CurvedEdgeConfig {
  /** Enable curved edge rendering */
  enabled: boolean;
  /** Number of tessellation segments (default 19) */
  segments: number;
  /** Rational curve weight (default 0.8) */
  weight: number;
}

/**
 * Default curved edge configuration
 */
export const DEFAULT_CURVED_EDGE_CONFIG: CurvedEdgeConfig = {
  enabled: false,
  segments: 19,
  weight: 0.8,
};

/**
 * Size of curve config uniform buffer in bytes
 * Layout: enabled (u32), segments (u32), weight (f32), padding (f32)
 */
export const CURVE_CONFIG_UNIFORM_SIZE = 16;

/**
 * Configuration for edge render pipeline
 */
export interface EdgePipelineConfig {
  /** Maximum number of edges to render */
  maxEdges?: number | undefined;
  /** Sample count for MSAA (1, 4, or 8) */
  sampleCount?: number | undefined;
  /** Texture format for color attachment */
  format?: GPUTextureFormat | undefined;
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
  /** Bind group layout for flow uniforms */
  flowBindGroupLayout: GPUBindGroupLayout;
  /** Flow uniform buffer */
  flowUniformBuffer: GPUBuffer;
  /** Flow uniform bind group */
  flowBindGroup: GPUBindGroup;
  /** Curve config buffer */
  curveConfigBuffer: GPUBuffer;
  /** Current curved edge configuration */
  curveConfig: CurvedEdgeConfig;
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
  // - binding 0: positions (vec2 storage buffer) - node positions for endpoints
  // - binding 1: edge_indices (storage buffer) - source/target pairs
  // - binding 2: edge_attrs (storage buffer) - width, color, state, curvature
  // - binding 3: curve_config (storage buffer) - curve configuration
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

  // Create curve config buffer
  const curveConfigBuffer = device.createBuffer({
    label: "Edge Curve Config",
    size: CURVE_CONFIG_UNIFORM_SIZE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Initialize curve config with defaults
  const curveConfigData = new ArrayBuffer(CURVE_CONFIG_UNIFORM_SIZE);
  const curveConfigView = new DataView(curveConfigData);
  curveConfigView.setUint32(0, DEFAULT_CURVED_EDGE_CONFIG.enabled ? 1 : 0, true);
  curveConfigView.setUint32(4, DEFAULT_CURVED_EDGE_CONFIG.segments, true);
  curveConfigView.setFloat32(8, DEFAULT_CURVED_EDGE_CONFIG.weight, true);
  curveConfigView.setFloat32(12, 0.0, true); // padding
  device.queue.writeBuffer(curveConfigBuffer, 0, curveConfigData);

  // Flow uniform bind group layout (group 2)
  const flowBindGroupLayout = device.createBindGroupLayout({
    label: "Edge Pipeline - Flow Uniforms",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
    ],
  });

  // Create flow uniform buffer
  const flowUniformBuffer = device.createBuffer({
    label: "Edge Flow Uniforms",
    size: EDGE_FLOW_UNIFORM_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create flow bind group
  const flowBindGroup = device.createBindGroup({
    label: "Edge Flow Bind Group",
    layout: flowBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: flowUniformBuffer } },
    ],
  });

  // Initialize flow uniforms with defaults
  const flowData = new ArrayBuffer(EDGE_FLOW_UNIFORM_SIZE);
  writeEdgeFlowUniforms(flowData, DEFAULT_EDGE_FLOW_CONFIG, 0.0);
  device.queue.writeBuffer(flowUniformBuffer, 0, flowData);

  // Create pipeline layout
  const pipelineLayout = device.createPipelineLayout({
    label: "Edge Pipeline Layout",
    bindGroupLayouts: [viewportBindGroupLayout, edgeBindGroupLayout, flowBindGroupLayout],
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
          format: finalConfig.format ?? "bgra8unorm",
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
      count: finalConfig.sampleCount ?? 1,
    },
  });

  return {
    pipeline,
    viewportBindGroupLayout,
    edgeBindGroupLayout,
    flowBindGroupLayout,
    flowUniformBuffer,
    flowBindGroup,
    curveConfigBuffer,
    curveConfig: { ...DEFAULT_CURVED_EDGE_CONFIG },
    shaderModule,
    config: finalConfig,
  };
}

/**
 * Creates a bind group for edge rendering
 *
 * @param device - GPU device
 * @param pipeline - Edge render pipeline
 * @param positions - Node position buffer (vec2 per node)
 * @param edgeIndices - Edge source/target indices buffer
 * @param edgeAttrs - Edge attributes buffer
 * @returns Bind group for edge data
 */
export function createEdgeBindGroup(
  device: GPUDevice,
  pipeline: EdgeRenderPipeline,
  positions: GPUBuffer,
  edgeIndices: GPUBuffer,
  edgeAttrs: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: "Edge Data Bind Group",
    layout: pipeline.edgeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: positions } },
      { binding: 1, resource: { buffer: edgeIndices } },
      { binding: 2, resource: { buffer: edgeAttrs } },
      { binding: 3, resource: { buffer: pipeline.curveConfigBuffer } },
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
 * Update edge flow uniforms
 *
 * @param device - GPU device
 * @param pipeline - Edge render pipeline
 * @param config - Flow configuration
 * @param time - Current animation time in seconds
 */
export function updateEdgeFlowUniforms(
  device: GPUDevice,
  pipeline: EdgeRenderPipeline,
  config: EdgeFlowConfig,
  time: number,
): void {
  const data = new ArrayBuffer(EDGE_FLOW_UNIFORM_SIZE);
  writeEdgeFlowUniforms(data, config, time);
  device.queue.writeBuffer(pipeline.flowUniformBuffer, 0, data);
}

/**
 * Update curved edge configuration
 *
 * @param device - GPU device
 * @param pipeline - Edge render pipeline
 * @param config - Curved edge configuration
 */
export function updateCurveConfig(
  device: GPUDevice,
  pipeline: EdgeRenderPipeline,
  config: Partial<CurvedEdgeConfig>,
): void {
  // Merge with current config
  pipeline.curveConfig = {
    ...pipeline.curveConfig,
    ...config,
  };

  // Write to GPU buffer
  const data = new ArrayBuffer(CURVE_CONFIG_UNIFORM_SIZE);
  const view = new DataView(data);
  view.setUint32(0, pipeline.curveConfig.enabled ? 1 : 0, true);
  view.setUint32(4, pipeline.curveConfig.segments, true);
  view.setFloat32(8, pipeline.curveConfig.weight, true);
  view.setFloat32(12, 0.0, true); // padding
  device.queue.writeBuffer(pipeline.curveConfigBuffer, 0, data);
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
  pass.setBindGroup(2, pipeline.flowBindGroup);

  // For curved edges, we need more vertices per edge to tessellate the curve
  // 6 vertices per segment, segments per edge
  const verticesPerEdge = pipeline.curveConfig.enabled
    ? 6 * pipeline.curveConfig.segments
    : 6;

  pass.draw(verticesPerEdge, edgeCount);
}

/**
 * Destroys edge render pipeline resources
 *
 * @param pipeline - Edge render pipeline to destroy
 */
export function destroyEdgeRenderPipeline(pipeline: EdgeRenderPipeline): void {
  // Clean up buffers
  pipeline.flowUniformBuffer.destroy();
  pipeline.curveConfigBuffer.destroy();
}
