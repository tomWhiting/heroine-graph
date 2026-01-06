/**
 * Metaball Layer
 *
 * A visualization layer that renders smooth blob-like shapes around node clusters.
 * Uses screen-space SDF evaluation with quadratic smooth minimum for organic blending.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";
import type { Layer } from "../heatmap/layer.ts";
import type { MetaballConfig } from "./config.ts";
import type { MetaballPipeline } from "./pipeline.ts";

import { mergeMetaballConfig, parseMetaballColor } from "./config.ts";
import { createMetaballPipeline } from "./pipeline.ts";

/**
 * Metaball layer render context
 */
export interface MetaballRenderContext {
  /** Viewport uniform buffer */
  viewportUniformBuffer: GPUBuffer;
  /** Position X buffer */
  positionsX: GPUBuffer;
  /** Position Y buffer */
  positionsY: GPUBuffer;
  /** Number of nodes */
  nodeCount: number;
  /** Viewport offset (pan) */
  viewportOffset: [number, number];
  /** Viewport scale (zoom) */
  viewportScale: number;
}

/**
 * Metaball layer implementation
 */
export class MetaballLayer implements Layer {
  readonly id: string;
  readonly type = "metaball";

  private config: Required<MetaballConfig>;
  private pipeline: MetaballPipeline;
  private renderContext: MetaballRenderContext | null = null;

  // Cached bind group
  private bindGroup: GPUBindGroup | null = null;
  private bindGroupDirty = true;

  // Screen dimensions
  private screenWidth = 800;
  private screenHeight = 600;

  constructor(id: string, context: GPUContext, config: MetaballConfig = {}) {
    this.id = id;
    this.config = mergeMetaballConfig(config);
    this.pipeline = createMetaballPipeline(context);
  }

  get enabled(): boolean {
    return this.config.enabled;
  }

  set enabled(value: boolean) {
    this.config.enabled = value;
  }

  get order(): number {
    return 2; // Metaballs render above heatmap and contours, below nodes
  }

  set order(_value: number) {
    // Metaball order is fixed
  }

  /**
   * Set the render context (position buffers, etc.)
   */
  setRenderContext(context: MetaballRenderContext): void {
    this.renderContext = context;
    this.bindGroupDirty = true;
  }

  /**
   * Update metaball configuration
   */
  setConfig(config: Partial<MetaballConfig>): void {
    Object.assign(this.config, config);
  }

  /**
   * Get current configuration
   */
  getConfig(): Required<MetaballConfig> {
    return { ...this.config };
  }

  /**
   * Render the metaball layer
   */
  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void {
    if (!this.config.enabled || !this.renderContext) {
      return;
    }

    const { nodeCount, viewportOffset, viewportScale } = this.renderContext;
    if (nodeCount === 0) return;

    // Ensure bind group exists
    this.ensureBindGroup();

    // Update uniforms
    const [r, g, b, a] = parseMetaballColor(this.config.fillColor);
    this.pipeline.updateUniforms({
      viewportOffset,
      viewportScale,
      threshold: this.config.threshold,
      blendRadius: this.config.blendRadius,
      nodeRadius: this.config.nodeRadius,
      screenWidth: this.screenWidth,
      screenHeight: this.screenHeight,
      fillColor: [r, g, b, a * this.config.opacity],
      outlineOnly: this.config.outlineOnly,
      outlineWidth: this.config.outlineWidth,
      nodeCount,
    });

    // Render fullscreen triangle
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
   * Ensure bind group is created
   */
  private ensureBindGroup(): void {
    if (!this.bindGroupDirty || !this.renderContext) return;

    this.bindGroup = this.pipeline.createBindGroup(
      this.renderContext.positionsX,
      this.renderContext.positionsY
    );

    this.bindGroupDirty = false;
  }

  /**
   * Resize layer resources
   */
  resize(width: number, height: number): void {
    this.screenWidth = width;
    this.screenHeight = height;
  }

  /**
   * Destroy layer resources
   */
  destroy(): void {
    this.pipeline.destroy();
  }
}

/**
 * Create a metaball layer
 */
export function createMetaballLayer(
  id: string,
  context: GPUContext,
  config?: MetaballConfig
): MetaballLayer {
  return new MetaballLayer(id, context, config);
}
