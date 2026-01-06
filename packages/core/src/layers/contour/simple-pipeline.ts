/**
 * Simple Contour Pipeline
 *
 * Renders contour lines using a fullscreen fragment shader approach.
 * Based on Sigma.js implementation using screen-space derivatives
 * for anti-aliased contour lines.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";

// Import shader source code
import contourVertSource from "./shaders/contour.vert.wgsl?raw";
import contourFragSource from "./shaders/contour.frag.wgsl?raw";

/**
 * Contour uniform data
 */
export interface SimpleContourUniforms {
  /** Line color (RGBA, 0-1) */
  lineColor: [number, number, number, number];
  /** Line thickness in pixels */
  lineThickness: number;
  /** Anti-aliasing feather amount */
  feather: number;
  /** Threshold value (0-1) */
  threshold: number;
  /** Maximum density for normalization */
  maxDensity: number;
}

/**
 * Simple contour pipeline resources
 */
export interface SimpleContourPipeline {
  /** Render pipeline */
  pipeline: GPURenderPipeline;
  /** Bind group layout */
  bindGroupLayout: GPUBindGroupLayout;
  /** Uniform buffer */
  uniformBuffer: GPUBuffer;
  /** Sampler for density texture */
  sampler: GPUSampler;
  /** Update uniforms */
  updateUniforms: (uniforms: SimpleContourUniforms) => void;
  /** Create bind group */
  createBindGroup: (densityTextureView: GPUTextureView) => GPUBindGroup;
  /** Destroy resources */
  destroy: () => void;
}

/**
 * Creates the simple contour pipeline
 */
export function createSimpleContourPipeline(context: GPUContext): SimpleContourPipeline {
  const { device, format: canvasFormat } = context;

  // Create shader modules
  const vertModule = device.createShaderModule({
    label: "Simple Contour Vertex Shader",
    code: contourVertSource,
  });

  const fragModule = device.createShaderModule({
    label: "Simple Contour Fragment Shader",
    code: contourFragSource,
  });

  // Create sampler for density texture
  const sampler = device.createSampler({
    label: "Contour Density Sampler",
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
  });

  // Create uniform buffer
  // Layout: vec4 lineColor, f32 lineThickness, f32 feather, f32 threshold, f32 maxDensity
  const uniformBuffer = device.createBuffer({
    label: "Simple Contour Uniform Buffer",
    size: 32, // 4 floats for color + 4 floats for params = 8 * 4 = 32 bytes
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create bind group layout
  const bindGroupLayout = device.createBindGroupLayout({
    label: "Simple Contour Bind Group Layout",
    entries: [
      // Density texture
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float" },
      },
      // Sampler
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "filtering" },
      },
      // Uniforms
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
    ],
  });

  // Create pipeline layout
  const pipelineLayout = device.createPipelineLayout({
    label: "Simple Contour Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });

  // Create render pipeline
  const pipeline = device.createRenderPipeline({
    label: "Simple Contour Pipeline",
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
          format: canvasFormat,
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

  // Update uniforms function
  function updateUniforms(uniforms: SimpleContourUniforms): void {
    const data = new Float32Array([
      uniforms.lineColor[0],
      uniforms.lineColor[1],
      uniforms.lineColor[2],
      uniforms.lineColor[3],
      uniforms.lineThickness,
      uniforms.feather,
      uniforms.threshold,
      uniforms.maxDensity,
    ]);
    device.queue.writeBuffer(uniformBuffer, 0, data);
  }

  // Create bind group function
  function createBindGroup(densityTextureView: GPUTextureView): GPUBindGroup {
    return device.createBindGroup({
      label: "Simple Contour Bind Group",
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: densityTextureView },
        { binding: 1, resource: sampler },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });
  }

  // Destroy function
  function destroy(): void {
    uniformBuffer.destroy();
  }

  return {
    pipeline,
    bindGroupLayout,
    uniformBuffer,
    sampler,
    updateUniforms,
    createBindGroup,
    destroy,
  };
}
