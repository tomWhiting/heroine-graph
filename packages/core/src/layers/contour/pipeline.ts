/**
 * Contour Render Pipeline
 *
 * Creates and manages GPU pipelines for contour visualization:
 * 1. Identify pass: Mark cells that cross the threshold
 * 2. Prefix sum: Compute output offsets for each cell
 * 3. Generate: Create line segment vertices
 * 4. Render: Draw line segments to screen
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";

// Import shader source code
import identifySource from "./shaders/identify.comp.wgsl?raw";
import prefixSumSource from "./shaders/prefix_sum.comp.wgsl?raw";
import generateSource from "./shaders/generate.comp.wgsl?raw";
import lineVertSource from "./shaders/line.vert.wgsl?raw";
import lineFragSource from "./shaders/line.frag.wgsl?raw";

/**
 * Contour uniform data
 */
export interface ContourUniforms {
  /** Texture width */
  width: number;
  /** Texture height */
  height: number;
  /** Density threshold (normalized 0-1) */
  threshold: number;
  /** Maximum density for normalization */
  maxDensity: number;
}

/**
 * Line render uniform data
 */
export interface LineUniforms {
  /** Line width in pixels */
  lineWidth: number;
  /** Screen width */
  screenWidth: number;
  /** Screen height */
  screenHeight: number;
}

/**
 * Line color uniform data
 */
export interface LineColorUniforms {
  /** RGBA color (0-1) */
  color: [number, number, number, number];
}

/**
 * Contour pipeline resources
 */
export interface ContourPipeline {
  /** Identify compute pipeline */
  identifyPipeline: GPUComputePipeline;
  /** Prefix sum compute pipeline */
  prefixSumPipeline: GPUComputePipeline;
  /** Generate compute pipeline */
  generatePipeline: GPUComputePipeline;
  /** Line render pipeline */
  linePipeline: GPURenderPipeline;
  /** Uniform buffer for contour params */
  contourUniformBuffer: GPUBuffer;
  /** Uniform buffer for prefix sum */
  prefixSumUniformBuffer: GPUBuffer;
  /** Uniform buffer for line rendering */
  lineUniformBuffer: GPUBuffer;
  /** Uniform buffer for line color */
  lineColorUniformBuffer: GPUBuffer;
  /** Update contour uniforms */
  updateContourUniforms: (uniforms: ContourUniforms) => void;
  /** Update line uniforms */
  updateLineUniforms: (uniforms: LineUniforms) => void;
  /** Update line color */
  updateLineColor: (uniforms: LineColorUniforms) => void;
  /** Create identify bind group */
  createIdentifyBindGroup: (
    densityTexture: GPUTextureView,
    cellCases: GPUBuffer,
    activeCount: GPUBuffer,
  ) => GPUBindGroup;
  /** Create prefix sum bind group */
  createPrefixSumBindGroup: (data: GPUBuffer, elementCount: number) => GPUBindGroup;
  /** Create generate bind group */
  createGenerateBindGroup: (
    densityTexture: GPUTextureView,
    cellCases: GPUBuffer,
    prefixSums: GPUBuffer,
    vertices: GPUBuffer,
  ) => GPUBindGroup;
  /** Create line render bind group */
  createLineBindGroup: (segments: GPUBuffer) => GPUBindGroup;
  /** Destroy resources */
  destroy: () => void;
}

/**
 * Creates the contour pipelines
 */
export function createContourPipeline(context: GPUContext): ContourPipeline {
  const { device, format: canvasFormat } = context;

  // Create shader modules
  const identifyModule = device.createShaderModule({
    label: "Contour Identify Shader",
    code: identifySource,
  });

  const prefixSumModule = device.createShaderModule({
    label: "Contour Prefix Sum Shader",
    code: prefixSumSource,
  });

  const generateModule = device.createShaderModule({
    label: "Contour Generate Shader",
    code: generateSource,
  });

  const lineVertModule = device.createShaderModule({
    label: "Contour Line Vertex Shader",
    code: lineVertSource,
  });

  const lineFragModule = device.createShaderModule({
    label: "Contour Line Fragment Shader",
    code: lineFragSource,
  });

  // Create bind group layouts

  // Identify layout
  const identifyLayout = device.createBindGroupLayout({
    label: "Contour Identify Layout",
    entries: [
      // Density texture
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float" },
      },
      // Uniforms
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      // Cell cases output
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      // Active count (atomic)
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });

  // Prefix sum layout
  const prefixSumLayout = device.createBindGroupLayout({
    label: "Contour Prefix Sum Layout",
    entries: [
      // Uniforms
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      // Data buffer (in-place)
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });

  // Generate layout
  const generateLayout = device.createBindGroupLayout({
    label: "Contour Generate Layout",
    entries: [
      // Density texture
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float" },
      },
      // Uniforms
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      // Cell cases
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      // Segment counter (atomic)
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      // Vertices output
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });

  // Line render layout
  const lineLayout = device.createBindGroupLayout({
    label: "Contour Line Layout",
    entries: [
      // Line uniforms
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
      // Segments buffer
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" },
      },
      // Color uniforms
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
    ],
  });

  // Create pipeline layouts
  const identifyPipelineLayout = device.createPipelineLayout({
    label: "Contour Identify Pipeline Layout",
    bindGroupLayouts: [identifyLayout],
  });

  const prefixSumPipelineLayout = device.createPipelineLayout({
    label: "Contour Prefix Sum Pipeline Layout",
    bindGroupLayouts: [prefixSumLayout],
  });

  const generatePipelineLayout = device.createPipelineLayout({
    label: "Contour Generate Pipeline Layout",
    bindGroupLayouts: [generateLayout],
  });

  const linePipelineLayout = device.createPipelineLayout({
    label: "Contour Line Pipeline Layout",
    bindGroupLayouts: [lineLayout],
  });

  // Create compute pipelines
  const identifyPipeline = device.createComputePipeline({
    label: "Contour Identify Pipeline",
    layout: identifyPipelineLayout,
    compute: {
      module: identifyModule,
      entryPoint: "main",
    },
  });

  const prefixSumPipeline = device.createComputePipeline({
    label: "Contour Prefix Sum Pipeline",
    layout: prefixSumPipelineLayout,
    compute: {
      module: prefixSumModule,
      entryPoint: "main",
    },
  });

  const generatePipeline = device.createComputePipeline({
    label: "Contour Generate Pipeline",
    layout: generatePipelineLayout,
    compute: {
      module: generateModule,
      entryPoint: "main",
    },
  });

  // Create render pipeline
  const linePipeline = device.createRenderPipeline({
    label: "Contour Line Pipeline",
    layout: linePipelineLayout,
    vertex: {
      module: lineVertModule,
      entryPoint: "vs_main",
    },
    fragment: {
      module: lineFragModule,
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

  // Create uniform buffers
  const contourUniformBuffer = device.createBuffer({
    label: "Contour Uniform Buffer",
    size: 16, // 3 u32/f32 + 1 padding
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const prefixSumUniformBuffer = device.createBuffer({
    label: "Prefix Sum Uniform Buffer",
    size: 16, // 2 u32 + 2 padding
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const lineUniformBuffer = device.createBuffer({
    label: "Line Uniform Buffer",
    size: 16, // 3 f32 + 1 padding
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const lineColorUniformBuffer = device.createBuffer({
    label: "Line Color Uniform Buffer",
    size: 16, // vec4<f32>
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Uniform update functions
  function updateContourUniforms(uniforms: ContourUniforms): void {
    const data = new ArrayBuffer(16);
    const view = new DataView(data);
    view.setUint32(0, uniforms.width, true);
    view.setUint32(4, uniforms.height, true);
    view.setFloat32(8, uniforms.threshold, true);
    view.setFloat32(12, uniforms.maxDensity, true);
    device.queue.writeBuffer(contourUniformBuffer, 0, data);
  }

  function updateLineUniforms(uniforms: LineUniforms): void {
    const data = new Float32Array([
      uniforms.lineWidth,
      uniforms.screenWidth,
      uniforms.screenHeight,
      0, // padding
    ]);
    device.queue.writeBuffer(lineUniformBuffer, 0, data);
  }

  function updateLineColor(uniforms: LineColorUniforms): void {
    const data = new Float32Array(uniforms.color);
    device.queue.writeBuffer(lineColorUniformBuffer, 0, data);
  }

  // Bind group creation functions
  function createIdentifyBindGroup(
    densityTexture: GPUTextureView,
    cellCases: GPUBuffer,
    activeCount: GPUBuffer,
  ): GPUBindGroup {
    return device.createBindGroup({
      label: "Contour Identify Bind Group",
      layout: identifyLayout,
      entries: [
        { binding: 0, resource: densityTexture },
        { binding: 1, resource: { buffer: contourUniformBuffer } },
        { binding: 2, resource: { buffer: cellCases } },
        { binding: 3, resource: { buffer: activeCount } },
      ],
    });
  }

  function createPrefixSumBindGroup(
    data: GPUBuffer,
    elementCount: number,
  ): GPUBindGroup {
    // Update uniforms
    const uniformData = new Uint32Array([elementCount, 0, 0, 0]);
    device.queue.writeBuffer(prefixSumUniformBuffer, 0, uniformData);

    return device.createBindGroup({
      label: "Contour Prefix Sum Bind Group",
      layout: prefixSumLayout,
      entries: [
        { binding: 0, resource: { buffer: prefixSumUniformBuffer } },
        { binding: 1, resource: { buffer: data } },
      ],
    });
  }

  function createGenerateBindGroup(
    densityTexture: GPUTextureView,
    cellCases: GPUBuffer,
    prefixSums: GPUBuffer,
    vertices: GPUBuffer,
  ): GPUBindGroup {
    return device.createBindGroup({
      label: "Contour Generate Bind Group",
      layout: generateLayout,
      entries: [
        { binding: 0, resource: densityTexture },
        { binding: 1, resource: { buffer: contourUniformBuffer } },
        { binding: 2, resource: { buffer: cellCases } },
        { binding: 3, resource: { buffer: prefixSums } },
        { binding: 4, resource: { buffer: vertices } },
      ],
    });
  }

  function createLineBindGroup(segments: GPUBuffer): GPUBindGroup {
    return device.createBindGroup({
      label: "Contour Line Bind Group",
      layout: lineLayout,
      entries: [
        { binding: 0, resource: { buffer: lineUniformBuffer } },
        { binding: 1, resource: { buffer: segments } },
        { binding: 2, resource: { buffer: lineColorUniformBuffer } },
      ],
    });
  }

  function destroy(): void {
    contourUniformBuffer.destroy();
    prefixSumUniformBuffer.destroy();
    lineUniformBuffer.destroy();
    lineColorUniformBuffer.destroy();
  }

  return {
    identifyPipeline,
    prefixSumPipeline,
    generatePipeline,
    linePipeline,
    contourUniformBuffer,
    prefixSumUniformBuffer,
    lineUniformBuffer,
    lineColorUniformBuffer,
    updateContourUniforms,
    updateLineUniforms,
    updateLineColor,
    createIdentifyBindGroup,
    createPrefixSumBindGroup,
    createGenerateBindGroup,
    createLineBindGroup,
    destroy,
  };
}
