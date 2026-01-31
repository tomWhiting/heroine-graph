/**
 * Simple Contour Layer
 *
 * A visualization layer that renders contour lines around density thresholds.
 *
 * TODO: CONTOUR LAYER NOT WORKING
 * ================================
 * This layer is currently broken and produces no visible output.
 * Multiple approaches have been tried:
 * 1. Screen-space derivatives (dpdx/dpdy) - gradients too small
 * 2. Simple band test (like metaball outline) - still not rendering
 *
 * The heatmap and metaball layers work correctly. The issue may be:
 * - Density values not in expected range for thresholds
 * - Texture sampling issues
 * - Bind group/pipeline configuration
 *
 * Consider using D3 contour for CPU-based isoline computation as fallback.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";
import type { Layer } from "../heatmap/layer.ts";
import type { ContourConfig } from "./config.ts";
import type { SimpleContourPipeline } from "./simple-pipeline.ts";

import { mergeContourConfig, parseColor } from "./config.ts";
import { createSimpleContourPipeline } from "./simple-pipeline.ts";

/**
 * Contour layer render context
 */
export interface SimpleContourRenderContext {
  /** Density texture view to sample from (from heatmap) */
  densityTextureView: GPUTextureView;
  /** Maximum density value for normalization (from heatmap config) */
  maxDensity: number;
}

/**
 * Simple contour layer implementation
 */
export class SimpleContourLayer implements Layer {
  readonly id: string;
  readonly type = "contour";

  private config: Required<ContourConfig>;
  private pipeline: SimpleContourPipeline;
  private renderContext: SimpleContourRenderContext | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private bindGroupDirty = true;

  constructor(
    id: string,
    context: GPUContext,
    config: ContourConfig = {},
  ) {
    this.id = id;
    this.config = mergeContourConfig(config);
    this.pipeline = createSimpleContourPipeline(context);

    // Initialize uniforms
    this.updateUniforms();
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
  setRenderContext(context: SimpleContourRenderContext): void {
    this.renderContext = context;
    this.bindGroupDirty = true;
    this.updateUniforms();
  }

  /**
   * Update contour configuration
   */
  setConfig(config: Partial<ContourConfig>): void {
    Object.assign(this.config, config);

    if (config.thresholds) {
      this.config.thresholds = [...config.thresholds];
    }

    this.updateUniforms();
    this.bindGroupDirty = true;
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
   * Set the data source for contour values.
   * @param source - 'density' for uniform node density, or a stream ID
   */
  setDataSource(source: string): void {
    if (source === this.config.dataSource) return;
    this.config.dataSource = source;
    this.bindGroupDirty = true;
  }

  /**
   * Get the current data source
   */
  getDataSource(): string {
    return this.config.dataSource;
  }

  /**
   * Update GPU uniforms
   */
  private updateUniforms(): void {
    const [r, g, b, a] = parseColor(this.config.strokeColor);
    const threshold = this.config.thresholds[0] ?? 0.5;
    const maxDensity = this.renderContext?.maxDensity ?? 10;

    this.pipeline.updateUniforms({
      lineColor: [r, g, b, a * this.config.opacity],
      lineThickness: this.config.strokeWidth,
      feather: 1.5, // Anti-aliasing amount
      threshold,
      maxDensity,
    });
  }

  /**
   * Ensure bind group is created
   */
  private ensureBindGroup(): void {
    if (!this.bindGroupDirty || !this.renderContext) return;

    this.bindGroup = this.pipeline.createBindGroup(
      this.renderContext.densityTextureView,
    );
    this.bindGroupDirty = false;
  }

  /**
   * Render the contour layer
   */
  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void {
    if (!this.config.enabled) return;
    if (!this.renderContext) return;

    // Ensure bind group exists
    this.ensureBindGroup();
    if (!this.bindGroup) return;

    // Update uniforms for each threshold
    for (const threshold of this.config.thresholds) {
      this.renderThreshold(encoder, targetView, threshold);
    }
  }

  /**
   * Render a single threshold level
   */
  private renderThreshold(
    encoder: GPUCommandEncoder,
    targetView: GPUTextureView,
    threshold: number,
  ): void {
    // Update threshold in uniforms
    const [r, g, b, a] = parseColor(this.config.strokeColor);
    const maxDensity = this.renderContext?.maxDensity ?? 10;

    this.pipeline.updateUniforms({
      lineColor: [r, g, b, a * this.config.opacity],
      lineThickness: this.config.strokeWidth,
      feather: 1.5,
      threshold,
      maxDensity,
    });

    // Render pass
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: targetView,
          loadOp: "load",
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(this.pipeline.pipeline);
    pass.setBindGroup(0, this.bindGroup!);
    pass.draw(3, 1, 0, 0); // Fullscreen triangle
    pass.end();
  }

  /**
   * Resize layer resources
   */
  resize(_width: number, _height: number): void {
    // No resize needed - we render fullscreen
    this.bindGroupDirty = true;
  }

  /**
   * Destroy layer resources
   */
  destroy(): void {
    this.pipeline.destroy();
  }
}

/**
 * Create a simple contour layer
 */
export function createSimpleContourLayer(
  id: string,
  context: GPUContext,
  config?: ContourConfig,
): SimpleContourLayer {
  return new SimpleContourLayer(id, context, config);
}
