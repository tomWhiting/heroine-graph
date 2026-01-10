/**
 * Metaball Render Pipeline
 *
 * Creates and manages the GPU render pipeline for metaball visualization.
 * Uses screen-space SDF evaluation with smooth minimum blending.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";

// Import shader source code
import sdfVertSource from "./shaders/sdf.vert.wgsl?raw";
import sdfFragSource from "./shaders/sdf.frag.wgsl?raw";

/**
 * Metaball uniform data
 */
export interface MetaballUniforms {
  /** Viewport pan offset */
  viewportOffset: [number, number];
  /** Viewport zoom scale */
  viewportScale: number;
  /** SDF threshold */
  threshold: number;
  /** Smooth minimum blend radius */
  blendRadius: number;
  /** Node influence radius */
  nodeRadius: number;
  /** Screen dimensions */
  screenWidth: number;
  screenHeight: number;
  /** Fill color (RGBA 0-1) */
  fillColor: [number, number, number, number];
  /** Whether to render outline only */
  outlineOnly: boolean;
  /** Outline width in pixels */
  outlineWidth: number;
  /** Number of nodes */
  nodeCount: number;
}

/**
 * Metaball pipeline resources
 */
export interface MetaballPipeline {
  /** Render pipeline */
  pipeline: GPURenderPipeline;
  /** Bind group layout */
  bindGroupLayout: GPUBindGroupLayout;
  /** Uniform buffer */
  uniformBuffer: GPUBuffer;
  /** Update uniforms */
  updateUniforms: (uniforms: MetaballUniforms) => void;
  /** Create bind group */
  createBindGroup: (positionsX: GPUBuffer, positionsY: GPUBuffer) => GPUBindGroup;
  /** Destroy resources */
  destroy: () => void;
}

/**
 * Default metaball uniforms
 */
export const DEFAULT_METABALL_UNIFORMS: MetaballUniforms = {
  viewportOffset: [0, 0],
  viewportScale: 1.0,
  threshold: 0.5,
  blendRadius: 30.0,
  nodeRadius: 50.0,
  screenWidth: 800,
  screenHeight: 600,
  fillColor: [0.388, 0.4, 0.945, 0.3],
  outlineOnly: false,
  outlineWidth: 2.0,
  nodeCount: 0,
};

/**
 * Creates the metaball render pipeline
 */
export function createMetaballPipeline(context: GPUContext): MetaballPipeline {
  const { device } = context;

  // Create shader modules
  const vertModule = device.createShaderModule({
    label: "Metaball Vertex Shader",
    code: sdfVertSource,
  });

  const fragModule = device.createShaderModule({
    label: "Metaball Fragment Shader",
    code: sdfFragSource,
  });

  // Create bind group layout
  const bindGroupLayout = device.createBindGroupLayout({
    label: "Metaball Bind Group Layout",
    entries: [
      // Uniforms
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
      // Positions X
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "read-only-storage" },
      },
      // Positions Y
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "read-only-storage" },
      },
    ],
  });

  // Create pipeline layout
  const pipelineLayout = device.createPipelineLayout({
    label: "Metaball Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });

  // Create render pipeline
  const pipeline = device.createRenderPipeline({
    label: "Metaball Pipeline",
    layout: pipelineLayout,
    vertex: {
      module: vertModule,
      entryPoint: "vs_main",
    },
    fragment: {
      module: fragModule,
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

  // Create uniform buffer
  // Layout matches MetaballUniforms struct in shader:
  // vec2 viewport_offset: 8 bytes
  // f32 viewport_scale: 4 bytes
  // f32 threshold: 4 bytes
  // f32 blend_radius: 4 bytes
  // f32 node_radius: 4 bytes
  // f32 screen_width: 4 bytes
  // f32 screen_height: 4 bytes
  // vec4 fill_color: 16 bytes
  // u32 outline_only: 4 bytes
  // f32 outline_width: 4 bytes
  // u32 node_count: 4 bytes
  // u32 padding: 4 bytes
  // Total: 64 bytes
  const uniformBuffer = device.createBuffer({
    label: "Metaball Uniform Buffer",
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Initialize with defaults
  updateUniforms(DEFAULT_METABALL_UNIFORMS);

  function updateUniforms(uniforms: MetaballUniforms): void {
    const data = new ArrayBuffer(64);
    const floatView = new Float32Array(data);
    const uintView = new Uint32Array(data);

    // viewport_offset (vec2)
    floatView[0] = uniforms.viewportOffset[0];
    floatView[1] = uniforms.viewportOffset[1];
    // viewport_scale (f32)
    floatView[2] = uniforms.viewportScale;
    // threshold (f32)
    floatView[3] = uniforms.threshold;
    // blend_radius (f32)
    floatView[4] = uniforms.blendRadius;
    // node_radius (f32)
    floatView[5] = uniforms.nodeRadius;
    // screen_width (f32)
    floatView[6] = uniforms.screenWidth;
    // screen_height (f32)
    floatView[7] = uniforms.screenHeight;
    // fill_color (vec4)
    floatView[8] = uniforms.fillColor[0];
    floatView[9] = uniforms.fillColor[1];
    floatView[10] = uniforms.fillColor[2];
    floatView[11] = uniforms.fillColor[3];
    // outline_only (u32)
    uintView[12] = uniforms.outlineOnly ? 1 : 0;
    // outline_width (f32)
    floatView[13] = uniforms.outlineWidth;
    // node_count (u32)
    uintView[14] = uniforms.nodeCount;
    // padding (u32)
    uintView[15] = 0;

    device.queue.writeBuffer(uniformBuffer, 0, data);
  }

  function createBindGroup(
    positionsX: GPUBuffer,
    positionsY: GPUBuffer,
  ): GPUBindGroup {
    return device.createBindGroup({
      label: "Metaball Bind Group",
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: positionsX } },
        { binding: 2, resource: { buffer: positionsY } },
      ],
    });
  }

  function destroy(): void {
    uniformBuffer.destroy();
  }

  return {
    pipeline,
    bindGroupLayout,
    uniformBuffer,
    updateUniforms,
    createBindGroup,
    destroy,
  };
}
