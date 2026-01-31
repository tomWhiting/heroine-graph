/**
 * Heatmap Layer
 *
 * A visualization layer that renders node density as a heatmap overlay.
 * Uses Gaussian splatting for smooth density accumulation and a color
 * scale for mapping density values to colors.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";
import type { HeatmapConfig } from "./config.ts";
import type { DensityTexture } from "./texture.ts";
import type { ColorScaleName, ColorScaleTexture, ColorStop } from "./colorscale.ts";
import type { HeatmapPipeline } from "./pipeline.ts";

import { mergeHeatmapConfig } from "./config.ts";
import { clearDensityTexture, createDensityTexture } from "./texture.ts";
import { createColorScaleTexture, createCustomColorScaleTexture } from "./colorscale.ts";
import { createHeatmapPipeline } from "./pipeline.ts";

/**
 * Layer interface for the layer system
 */
export interface Layer {
  /** Unique layer ID */
  readonly id: string;
  /** Layer type */
  readonly type: string;
  /** Whether layer is enabled */
  enabled: boolean;
  /** Render order (higher = on top) */
  order: number;
  /** Render the layer */
  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void;
  /** Resize layer resources */
  resize(width: number, height: number): void;
  /** Destroy layer resources */
  destroy(): void;
}

/**
 * Heatmap layer rendering context
 */
export interface HeatmapRenderContext {
  /** Viewport uniform buffer */
  viewportUniformBuffer: GPUBuffer;
  /** Position buffer (vec2 per node) */
  positions: GPUBuffer;
  /** Number of nodes */
  nodeCount: number;
  /**
   * Optional per-node intensity buffer (Float32Array values).
   * If provided, each node's splat uses this intensity value.
   * If null, all nodes use the uniform intensity from config.
   */
  nodeIntensities?: GPUBuffer | null;
}

/**
 * Heatmap layer implementation
 */
export class HeatmapLayer implements Layer {
  readonly id: string;
  readonly type = "heatmap";

  private context: GPUContext;
  private config: Required<HeatmapConfig>;
  private pipeline: HeatmapPipeline;
  private densityTexture: DensityTexture;
  private colorScale: ColorScaleTexture;
  private renderContext: HeatmapRenderContext | null = null;

  // Cached bind groups (recreated when resources change)
  private uniformsBindGroup: GPUBindGroup | null = null;
  private positionBindGroup: GPUBindGroup | null = null;
  private colormapBindGroup: GPUBindGroup | null = null;
  private bindGroupsDirty = true;

  // Deferred destruction queue (to avoid destroying textures still in use by GPU)
  private pendingDestruction: { resource: ColorScaleTexture; framesRemaining: number }[] = [];

  constructor(
    id: string,
    context: GPUContext,
    width: number,
    height: number,
    config: HeatmapConfig = {},
  ) {
    this.id = id;
    this.context = context;
    this.config = mergeHeatmapConfig(config);

    // Create pipeline
    this.pipeline = createHeatmapPipeline(context);

    // Create density texture
    this.densityTexture = createDensityTexture(context, {
      width,
      height,
      scale: this.config.resolutionScale,
    });

    // Create color scale texture
    this.colorScale = createColorScaleTexture(context, this.config.colorScale);

    // Apply initial uniforms
    this.updateUniforms();
  }

  get enabled(): boolean {
    return this.config.enabled;
  }

  set enabled(value: boolean) {
    this.config.enabled = value;
  }

  get order(): number {
    return 0; // Heatmap renders below nodes
  }

  set order(_value: number) {
    // Heatmap always at order 0
  }

  /**
   * Set the render context (position buffers, etc.)
   */
  setRenderContext(context: HeatmapRenderContext): void {
    this.renderContext = context;
    this.bindGroupsDirty = true;
  }

  /**
   * Update heatmap configuration
   */
  setConfig(config: Partial<HeatmapConfig>): void {
    const prevColorScale = this.config.colorScale;

    Object.assign(this.config, config);

    // Recreate color scale if changed
    if (config.colorScale && config.colorScale !== prevColorScale) {
      // Defer destruction of old texture to avoid GPU race condition
      this.scheduleDestruction(this.colorScale);
      this.colorScale = createColorScaleTexture(this.context, config.colorScale);
      this.bindGroupsDirty = true;
    }

    this.updateUniforms();
  }

  /**
   * Get current configuration
   */
  getConfig(): Required<HeatmapConfig> {
    return { ...this.config };
  }

  /**
   * Change the color scale
   */
  setColorScale(name: ColorScaleName): void {
    if (name === this.config.colorScale) return;

    // Defer destruction of old texture to avoid GPU race condition
    this.scheduleDestruction(this.colorScale);
    this.colorScale = createColorScaleTexture(this.context, name);
    this.config.colorScale = name;
    this.bindGroupsDirty = true;
  }

  /**
   * Set a custom color scale from color stops
   * @param stops - Array of color stops with position (0-1) and color (RGBA 0-1)
   */
  setCustomColorScale(stops: ColorStop[]): void {
    // Defer destruction of old texture to avoid GPU race condition
    this.scheduleDestruction(this.colorScale);
    this.colorScale = createCustomColorScaleTexture(this.context, stops, 256, "custom");
    this.config.colorScale = "custom" as ColorScaleName;
    this.bindGroupsDirty = true;
  }

  /**
   * Schedule a color scale texture for deferred destruction.
   * This prevents "destroyed texture used in submit" errors when
   * the GPU is still executing commands that reference the texture.
   */
  private scheduleDestruction(resource: ColorScaleTexture): void {
    // Wait 3 frames to ensure GPU is done with the resource
    this.pendingDestruction.push({ resource, framesRemaining: 3 });
  }

  /**
   * Process deferred destruction queue.
   * Called at the start of each render to clean up old resources.
   */
  private processPendingDestruction(): void {
    for (let i = this.pendingDestruction.length - 1; i >= 0; i--) {
      const item = this.pendingDestruction[i];
      item.framesRemaining--;
      if (item.framesRemaining <= 0) {
        item.resource.destroy();
        this.pendingDestruction.splice(i, 1);
      }
    }
  }

  /**
   * Set the data source for per-node intensity values.
   * @param source - 'density' for uniform intensity, or a stream ID
   */
  setDataSource(source: string): void {
    if (source === this.config.dataSource) return;
    this.config.dataSource = source;
    this.bindGroupsDirty = true;
  }

  /**
   * Get the current data source
   */
  getDataSource(): string {
    return this.config.dataSource;
  }

  /**
   * Render the heatmap layer
   */
  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void {
    // Always process deferred destruction, even when disabled
    this.processPendingDestruction();

    if (!this.config.enabled || !this.renderContext) {
      return;
    }

    const { nodeCount } = this.renderContext;
    if (nodeCount === 0) return;

    // Rebuild bind groups if needed
    this.ensureBindGroups();

    // Pass 1: Render Gaussian splats to density texture
    this.renderSplatPass(encoder, nodeCount);

    // Pass 2: Map density to colors and composite to screen
    this.renderColormapPass(encoder, targetView);
  }

  /**
   * Render splats to density texture
   */
  private renderSplatPass(encoder: GPUCommandEncoder, nodeCount: number): void {
    // Clear density texture
    clearDensityTexture(encoder, this.densityTexture);

    // Render splats
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.densityTexture.renderView,
          loadOp: "load", // Already cleared
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(this.pipeline.splatPipeline);
    pass.setBindGroup(0, this.uniformsBindGroup!);
    pass.setBindGroup(1, this.positionBindGroup!);

    // Draw 6 vertices per node (2 triangles = 1 quad)
    pass.draw(6, nodeCount, 0, 0);

    pass.end();
  }

  /**
   * Map density to colors and composite
   */
  private renderColormapPass(
    encoder: GPUCommandEncoder,
    targetView: GPUTextureView,
  ): void {
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: targetView,
          loadOp: "load", // Preserve existing content
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(this.pipeline.colormapPipeline);
    pass.setBindGroup(0, this.colormapBindGroup!);

    // Draw fullscreen triangle
    pass.draw(3, 1, 0, 0);

    pass.end();
  }

  /**
   * Ensure bind groups are created and up to date
   */
  private ensureBindGroups(): void {
    if (!this.bindGroupsDirty || !this.renderContext) return;

    const { uniformsBindGroup, positionBindGroup } = this.pipeline.createSplatBindGroup(
      this.renderContext.viewportUniformBuffer,
      this.renderContext.positions,
      this.renderContext.nodeIntensities,
    );

    this.uniformsBindGroup = uniformsBindGroup;
    this.positionBindGroup = positionBindGroup;

    this.colormapBindGroup = this.pipeline.createColormapBindGroup(
      this.densityTexture,
      this.colorScale,
    );

    this.bindGroupsDirty = false;
  }

  /**
   * Update GPU uniforms from config
   */
  private updateUniforms(): void {
    this.pipeline.updateHeatmapUniforms({
      radius: this.config.radius,
      intensity: this.config.intensity,
    });

    this.pipeline.updateColormapUniforms({
      minDensity: this.config.minDensity,
      maxDensity: this.config.maxDensity,
      opacity: this.config.opacity,
    });
  }

  /**
   * Get the density texture for use by other layers (e.g., contour layer)
   */
  getDensityTexture(): DensityTexture {
    return this.densityTexture;
  }

  /**
   * Resize layer resources
   */
  resize(width: number, height: number): void {
    this.densityTexture.resize(width, height);
    this.bindGroupsDirty = true;
  }

  /**
   * Destroy layer resources
   */
  destroy(): void {
    this.densityTexture.destroy();
    this.colorScale.destroy();
    this.pipeline.destroy();

    // Clean up any pending destruction items
    for (const item of this.pendingDestruction) {
      item.resource.destroy();
    }
    this.pendingDestruction = [];
  }
}

/**
 * Create a heatmap layer
 */
export function createHeatmapLayer(
  id: string,
  context: GPUContext,
  width: number,
  height: number,
  config?: HeatmapConfig,
): HeatmapLayer {
  return new HeatmapLayer(id, context, width, height, config);
}
