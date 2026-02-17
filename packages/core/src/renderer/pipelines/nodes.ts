/**
 * Node Render Pipeline
 *
 * Sets up the WebGPU render pipeline for node rendering using instanced
 * quad rendering with SDF-based circle rasterization.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";

// Import shader source (bundled as text by esbuild)
import NODE_VERT_WGSL from "../shaders/node.vert.wgsl";
import NODE_FRAG_WGSL from "../shaders/node.frag.wgsl";

/**
 * Configuration for node render pipeline
 */
export interface NodePipelineConfig {
  /** Maximum number of nodes to render */
  maxNodes?: number | undefined;
  /** Sample count for MSAA (1, 4, or 8) */
  sampleCount?: number | undefined;
  /** Texture format for color attachment */
  format?: GPUTextureFormat | undefined;
}

/**
 * Default node pipeline configuration
 */
export const DEFAULT_NODE_PIPELINE_CONFIG: Required<NodePipelineConfig> = {
  maxNodes: 1_000_000,
  sampleCount: 1,
  format: "bgra8unorm",
};

/**
 * Node render pipeline resources
 */
export interface NodeRenderPipeline {
  /** The WebGPU render pipeline */
  pipeline: GPURenderPipeline;
  /** Bind group layout for viewport uniforms */
  viewportBindGroupLayout: GPUBindGroupLayout;
  /** Bind group layout for node data */
  nodeBindGroupLayout: GPUBindGroupLayout;
  /** Bind group layout for render config */
  renderConfigBindGroupLayout: GPUBindGroupLayout;
  /** Shader module */
  shaderModule: GPUShaderModule;
  /** Pipeline configuration */
  config: Required<NodePipelineConfig>;
}

/**
 * Creates the node render pipeline
 *
 * @param context - GPU context
 * @param config - Pipeline configuration
 * @returns Node render pipeline resources
 */
export function createNodeRenderPipeline(
  context: GPUContext,
  config: NodePipelineConfig = {},
): NodeRenderPipeline {
  const { device } = context;
  const finalConfig = { ...DEFAULT_NODE_PIPELINE_CONFIG, ...config };

  // Create shader module
  const shaderModule = device.createShaderModule({
    label: "Node Shader",
    code: `${NODE_VERT_WGSL}\n${NODE_FRAG_WGSL}`,
  });

  // Viewport uniform bind group layout (group 0)
  const viewportBindGroupLayout = device.createBindGroupLayout({
    label: "Node Pipeline - Viewport Uniforms",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
    ],
  });

  // Node data bind group layout (group 1)
  // - binding 0: positions (vec2 storage buffer)
  // - binding 1: node_attrs (storage buffer)
  const nodeBindGroupLayout = device.createBindGroupLayout({
    label: "Node Pipeline - Node Data",
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

  // Render config bind group layout (group 2)
  // - binding 0: render config uniform
  const renderConfigBindGroupLayout = device.createBindGroupLayout({
    label: "Node Pipeline - Render Config",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
    ],
  });

  // Create pipeline layout
  const pipelineLayout = device.createPipelineLayout({
    label: "Node Pipeline Layout",
    bindGroupLayouts: [viewportBindGroupLayout, nodeBindGroupLayout, renderConfigBindGroupLayout],
  });

  // Create render pipeline
  const pipeline = device.createRenderPipeline({
    label: "Node Render Pipeline",
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
      cullMode: "none", // Nodes are always visible
    },
    multisample: {
      count: finalConfig.sampleCount ?? 1,
    },
  });

  return {
    pipeline,
    viewportBindGroupLayout,
    nodeBindGroupLayout,
    renderConfigBindGroupLayout,
    shaderModule,
    config: finalConfig,
  };
}

/**
 * Creates a bind group for node rendering
 *
 * @param device - GPU device
 * @param pipeline - Node render pipeline
 * @param positions - Position buffer (vec2 per node)
 * @param nodeAttrs - Node attributes buffer
 * @returns Bind group for node data
 */
export function createNodeBindGroup(
  device: GPUDevice,
  pipeline: NodeRenderPipeline,
  positions: GPUBuffer,
  nodeAttrs: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: "Node Data Bind Group",
    layout: pipeline.nodeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: positions } },
      { binding: 1, resource: { buffer: nodeAttrs } },
    ],
  });
}

/**
 * Creates a bind group for viewport uniforms
 *
 * @param device - GPU device
 * @param pipeline - Node render pipeline
 * @param viewportUniformBuffer - Viewport uniform buffer
 * @returns Bind group for viewport uniforms
 */
export function createViewportBindGroup(
  device: GPUDevice,
  pipeline: NodeRenderPipeline,
  viewportUniformBuffer: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: "Viewport Uniform Bind Group",
    layout: pipeline.viewportBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: viewportUniformBuffer } }],
  });
}

/**
 * Creates a bind group for render config uniforms
 *
 * @param device - GPU device
 * @param pipeline - Node render pipeline
 * @param renderConfigBuffer - Render config uniform buffer
 * @returns Bind group for render config
 */
export function createRenderConfigBindGroup(
  device: GPUDevice,
  pipeline: NodeRenderPipeline,
  renderConfigBuffer: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: "Render Config Bind Group",
    layout: pipeline.renderConfigBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: renderConfigBuffer } }],
  });
}

/**
 * Records node rendering commands
 *
 * @param pass - Render pass encoder
 * @param pipeline - Node render pipeline
 * @param viewportBindGroup - Viewport uniforms bind group
 * @param nodeBindGroup - Node data bind group
 * @param renderConfigBindGroup - Render config bind group
 * @param nodeCount - Number of nodes to render
 */
export function renderNodes(
  pass: GPURenderPassEncoder,
  pipeline: NodeRenderPipeline,
  viewportBindGroup: GPUBindGroup,
  nodeBindGroup: GPUBindGroup,
  renderConfigBindGroup: GPUBindGroup,
  nodeCount: number,
): void {
  if (nodeCount <= 0) return;

  pass.setPipeline(pipeline.pipeline);
  pass.setBindGroup(0, viewportBindGroup);
  pass.setBindGroup(1, nodeBindGroup);
  pass.setBindGroup(2, renderConfigBindGroup);

  // 6 vertices per quad (2 triangles), 1 instance per node
  pass.draw(6, nodeCount);
}

/**
 * Destroys node render pipeline resources
 *
 * @param pipeline - Node render pipeline to destroy
 */
export function destroyNodeRenderPipeline(_pipeline: NodeRenderPipeline): void {
  // WebGPU resources are garbage collected, but we can help by
  // explicitly releasing references
  // Note: GPURenderPipeline doesn't have a destroy() method
}
