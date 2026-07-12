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
 * FIXED since this TODO was written: all thresholds used to share one
 * uniform buffer, so queue.writeBuffer raced ahead of the encoded passes
 * and every pass sampled only the LAST threshold (typically 0.8, almost
 * never reached with default maxDensity). Each threshold now has its own
 * uniform buffer + bind group.
 *
 * The heatmap and metaball layers work correctly. Remaining suspects:
 * - Density values not in expected range for thresholds
 * - Texture sampling issues
 *
 * Consider using D3 contour for CPU-based isoline computation as fallback.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";
import type { Layer } from "../types.ts";
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
  // One uniform buffer + bind group per threshold: queue.writeBuffer
  // executes immediately while encoded passes run at submit, so N passes
  // sharing one buffer would all sample the LAST threshold written.
  private thresholdBuffers: GPUBuffer[] = [];
  private thresholdBindGroups: GPUBindGroup[] = [];
  private bindGroupDirty = true;
  private uniformsDirty = true;

  constructor(
    id: string,
    context: GPUContext,
    config: ContourConfig = {},
  ) {
    this.id = id;
    this.config = mergeContourConfig(config);
    this.pipeline = createSimpleContourPipeline(context);
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
    this.uniformsDirty = true;
  }

  /**
   * Update contour configuration
   */
  setConfig(config: Partial<ContourConfig>): void {
    Object.assign(this.config, config);

    if (config.thresholds) {
      this.config.thresholds = [...config.thresholds];
    }

    this.uniformsDirty = true;
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
   * Ensure one uniform buffer + bind group exists per threshold
   */
  private ensureThresholdResources(): void {
    if (!this.renderContext) return;

    const count = this.config.thresholds.length;

    if (this.bindGroupDirty || this.thresholdBindGroups.length !== count) {
      // Grow/shrink the buffer pool to match the threshold count
      while (this.thresholdBuffers.length < count) {
        this.thresholdBuffers.push(this.pipeline.createUniformBuffer());
      }
      while (this.thresholdBuffers.length > count) {
        this.thresholdBuffers.pop()!.destroy();
      }

      this.thresholdBindGroups = this.thresholdBuffers.map((buffer) =>
        this.pipeline.createBindGroup(this.renderContext!.densityTextureView, buffer)
      );

      this.bindGroupDirty = false;
      this.uniformsDirty = true;
    }

    if (this.uniformsDirty) {
      const [r, g, b, a] = parseColor(this.config.strokeColor);
      const maxDensity = this.renderContext.maxDensity;

      // One write per buffer: every encoded pass keeps its own threshold
      // even though all writes land before the frame's single submit.
      this.config.thresholds.forEach((threshold, i) => {
        this.pipeline.updateUniforms({
          lineColor: [r, g, b, a * this.config.opacity],
          lineThickness: this.config.strokeWidth,
          feather: 1.5, // Anti-aliasing amount
          threshold,
          maxDensity,
        }, this.thresholdBuffers[i]);
      });

      this.uniformsDirty = false;
    }
  }

  /**
   * Per-threshold uniform buffers (one per configured threshold).
   * Exposed for tests/diagnostics: each encoded pass must bind its own
   * buffer so every threshold survives to the frame's single queue.submit.
   */
  getThresholdUniformBuffers(): readonly GPUBuffer[] {
    return this.thresholdBuffers;
  }

  /**
   * Render the contour layer (one pass per threshold)
   */
  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void {
    if (!this.config.enabled) return;
    if (!this.renderContext) return;

    this.ensureThresholdResources();

    for (const bindGroup of this.thresholdBindGroups) {
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
      pass.setBindGroup(0, bindGroup);
      pass.draw(3, 1, 0, 0); // Fullscreen triangle
      pass.end();
    }
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
    for (const buffer of this.thresholdBuffers) {
      buffer.destroy();
    }
    this.thresholdBuffers = [];
    this.thresholdBindGroups = [];
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
