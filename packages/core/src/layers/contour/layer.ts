/**
 * Contour Layer
 *
 * A visualization layer that renders iso-lines at specified density thresholds.
 * Uses marching squares algorithm executed on the GPU via compute shaders.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";
import type { Layer } from "../heatmap/layer.ts";
import type { ContourConfig } from "./config.ts";
import type { ContourPipeline } from "./pipeline.ts";

import { mergeContourConfig, parseColor } from "./config.ts";
import { createContourPipeline } from "./pipeline.ts";

/**
 * Contour layer render context
 */
export interface ContourRenderContext {
  /** Density texture to sample from (from heatmap) */
  densityTexture: GPUTexture;
  /** Density texture view */
  densityTextureView: GPUTextureView;
  /** Width of density texture */
  width: number;
  /** Height of density texture */
  height: number;
}

/**
 * Maximum segments per threshold (for buffer allocation)
 */
const MAX_SEGMENTS_PER_THRESHOLD = 100000;

/**
 * Contour layer implementation
 */
export class ContourLayer implements Layer {
  readonly id: string;
  readonly type = "contour";

  private context: GPUContext;
  private config: Required<ContourConfig>;
  private pipeline: ContourPipeline;
  private renderContext: ContourRenderContext | null = null;

  // GPU buffers for marching squares
  private cellCasesBuffer: GPUBuffer | null = null;
  private activeCountBuffer: GPUBuffer | null = null;
  private prefixSumBuffer: GPUBuffer | null = null;
  private vertexBuffer: GPUBuffer | null = null;
  private stagingBuffer: GPUBuffer | null = null;

  // Cached bind groups
  private bindGroupsDirty = true;
  private identifyBindGroup: GPUBindGroup | null = null;

  // Screen dimensions for line rendering
  private screenWidth = 800;
  private screenHeight = 600;

  constructor(
    id: string,
    context: GPUContext,
    config: ContourConfig = {}
  ) {
    this.id = id;
    this.context = context;
    this.config = mergeContourConfig(config);
    this.pipeline = createContourPipeline(context);

    // Initialize line color
    this.updateLineColor();
  }

  get enabled(): boolean {
    return this.config.enabled;
  }

  set enabled(value: boolean) {
    this.config.enabled = value;
  }

  get order(): number {
    return 1; // Contours render above heatmap but below nodes
  }

  set order(_value: number) {
    // Contour order is fixed
  }

  /**
   * Set the render context (density texture from heatmap)
   */
  setRenderContext(context: ContourRenderContext): void {
    this.renderContext = context;
    this.bindGroupsDirty = true;

    // Allocate GPU buffers based on texture size
    this.allocateBuffers(context.width, context.height);
  }

  /**
   * Update contour configuration
   */
  setConfig(config: Partial<ContourConfig>): void {
    Object.assign(this.config, config);

    if (config.thresholds) {
      this.config.thresholds = [...config.thresholds];
    }

    if (config.strokeColor) {
      this.updateLineColor();
    }

    this.bindGroupsDirty = true;
  }

  /**
   * Get current configuration
   */
  getConfig(): Required<ContourConfig> {
    return {
      ...this.config,
      thresholds: [...this.config.thresholds],
    };
  }

  /**
   * Allocate GPU buffers for marching squares
   */
  private allocateBuffers(width: number, height: number): void {
    const { device } = this.context;

    // Clean up existing buffers
    this.destroyBuffers();

    const cellCount = (width - 1) * (height - 1);

    // Cell cases buffer (one u32 per cell)
    this.cellCasesBuffer = device.createBuffer({
      label: "Contour Cell Cases Buffer",
      size: cellCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Active count buffer (single atomic u32)
    this.activeCountBuffer = device.createBuffer({
      label: "Contour Active Count Buffer",
      size: 4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    // Prefix sum buffer (counts per cell, then transformed to offsets)
    this.prefixSumBuffer = device.createBuffer({
      label: "Contour Prefix Sum Buffer",
      size: cellCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Vertex buffer (4 floats per segment: x1, y1, x2, y2)
    this.vertexBuffer = device.createBuffer({
      label: "Contour Vertex Buffer",
      size: MAX_SEGMENTS_PER_THRESHOLD * 4 * 4, // 4 floats * 4 bytes
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Staging buffer for reading active count
    this.stagingBuffer = device.createBuffer({
      label: "Contour Staging Buffer",
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * Destroy GPU buffers
   */
  private destroyBuffers(): void {
    this.cellCasesBuffer?.destroy();
    this.activeCountBuffer?.destroy();
    this.prefixSumBuffer?.destroy();
    this.vertexBuffer?.destroy();
    this.stagingBuffer?.destroy();

    this.cellCasesBuffer = null;
    this.activeCountBuffer = null;
    this.prefixSumBuffer = null;
    this.vertexBuffer = null;
    this.stagingBuffer = null;
  }

  /**
   * Update line color from config
   */
  private updateLineColor(): void {
    const [r, g, b, a] = parseColor(this.config.strokeColor);
    this.pipeline.updateLineColor({
      color: [r, g, b, a * this.config.opacity],
    });
  }

  /**
   * Render the contour layer
   */
  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void {
    if (!this.config.enabled || !this.renderContext) {
      return;
    }

    if (
      !this.cellCasesBuffer ||
      !this.activeCountBuffer ||
      !this.prefixSumBuffer ||
      !this.vertexBuffer
    ) {
      return;
    }

    const { width, height } = this.renderContext;

    // Update line uniforms with screen dimensions
    this.pipeline.updateLineUniforms({
      lineWidth: this.config.strokeWidth,
      screenWidth: this.screenWidth,
      screenHeight: this.screenHeight,
    });

    // Render contours for each threshold
    for (const threshold of this.config.thresholds) {
      this.renderThreshold(encoder, targetView, threshold, width, height);
    }
  }

  /**
   * Render contours for a single threshold
   */
  private renderThreshold(
    encoder: GPUCommandEncoder,
    targetView: GPUTextureView,
    threshold: number,
    width: number,
    height: number
  ): void {
    if (!this.renderContext) return;

    const cellCount = (width - 1) * (height - 1);

    // Update contour uniforms
    this.pipeline.updateContourUniforms({
      width,
      height,
      threshold,
    });

    // Clear active count
    const zeroData = new Uint32Array([0]);
    this.context.device.queue.writeBuffer(this.activeCountBuffer!, 0, zeroData);

    // Ensure bind groups exist
    this.ensureBindGroups();

    // Pass 1: Identify active cells
    const identifyPass = encoder.beginComputePass({
      label: `Contour Identify (threshold=${threshold})`,
    });
    identifyPass.setPipeline(this.pipeline.identifyPipeline);
    identifyPass.setBindGroup(0, this.identifyBindGroup!);

    const workgroupsX = Math.ceil((width - 1) / 16);
    const workgroupsY = Math.ceil((height - 1) / 16);
    identifyPass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
    identifyPass.end();

    // Note: For simplicity, we skip the full prefix sum and generate
    // a fixed number of segments based on estimated density.
    // A production implementation would read back active_count
    // and perform a proper prefix sum.

    // Pass 2: Generate vertices (simplified - process all cells)
    const generateBindGroup = this.pipeline.createGenerateBindGroup(
      this.renderContext.densityTextureView,
      this.cellCasesBuffer!,
      this.prefixSumBuffer!, // In simplified mode, we don't use this properly
      this.vertexBuffer!
    );

    const generatePass = encoder.beginComputePass({
      label: `Contour Generate (threshold=${threshold})`,
    });
    generatePass.setPipeline(this.pipeline.generatePipeline);
    generatePass.setBindGroup(0, generateBindGroup);

    const generateWorkgroups = Math.ceil(cellCount / 256);
    generatePass.dispatchWorkgroups(generateWorkgroups, 1, 1);
    generatePass.end();

    // Pass 3: Render lines
    // Estimate segment count (conservative estimate)
    const estimatedSegments = Math.min(cellCount / 4, MAX_SEGMENTS_PER_THRESHOLD);

    if (estimatedSegments > 0) {
      const lineBindGroup = this.pipeline.createLineBindGroup(this.vertexBuffer!);

      const renderPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: targetView,
            loadOp: "load",
            storeOp: "store",
          },
        ],
      });

      renderPass.setPipeline(this.pipeline.linePipeline);
      renderPass.setBindGroup(0, lineBindGroup);
      // 6 vertices per line segment (2 triangles = quad)
      renderPass.draw(6, Math.floor(estimatedSegments), 0, 0);
      renderPass.end();
    }
  }

  /**
   * Ensure bind groups are created
   */
  private ensureBindGroups(): void {
    if (!this.bindGroupsDirty || !this.renderContext) return;

    this.identifyBindGroup = this.pipeline.createIdentifyBindGroup(
      this.renderContext.densityTextureView,
      this.cellCasesBuffer!,
      this.activeCountBuffer!
    );

    this.bindGroupsDirty = false;
  }

  /**
   * Resize layer resources
   */
  resize(width: number, height: number): void {
    this.screenWidth = width;
    this.screenHeight = height;

    if (this.renderContext) {
      this.allocateBuffers(this.renderContext.width, this.renderContext.height);
      this.bindGroupsDirty = true;
    }
  }

  /**
   * Destroy layer resources
   */
  destroy(): void {
    this.destroyBuffers();
    this.pipeline.destroy();
  }
}

/**
 * Create a contour layer
 */
export function createContourLayer(
  id: string,
  context: GPUContext,
  config?: ContourConfig
): ContourLayer {
  return new ContourLayer(id, context, config);
}
