/**
 * Labels Layer
 *
 * Renders text labels for nodes using MSDF (Multi-channel Signed Distance Field)
 * text rendering. Labels remain sharp at any zoom level.
 *
 * Features:
 * - Priority-based label selection
 * - Collision detection to prevent overlap
 * - Level-of-detail culling based on zoom
 * - GPU-accelerated instanced rendering
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";
import type { Layer } from "../heatmap/layer.ts";
import type { FontAtlas } from "./atlas.ts";
import { loadDefaultFontAtlas } from "./atlas.ts";
import { type LabelData, LabelManager } from "./manager.ts";
import type { PositionProvider } from "./manager.ts";
import { DEFAULT_LABEL_CONFIG, type LabelConfig, parseColor } from "./config.ts";

// Import shaders as strings
import labelVertexShader from "./shaders/label.vert.wgsl?raw";
import labelFragmentShader from "./shaders/label.frag.wgsl?raw";

// Re-export PositionProvider for convenience
export type { PositionProvider } from "./manager.ts";

/**
 * Render context for the labels layer
 */
export interface LabelsRenderContext {
  /** Viewport offset X */
  viewportX: number;
  /** Viewport offset Y */
  viewportY: number;
  /** Viewport scale */
  scale: number;
  /** Canvas width */
  canvasWidth: number;
  /** Canvas height */
  canvasHeight: number;
  /** Optional position provider for dynamic node positions */
  positionProvider?: PositionProvider;
}

/**
 * Labels visualization layer
 */
export class LabelsLayer implements Layer {
  readonly id: string;
  readonly type = "labels";

  private context: GPUContext;
  private config: LabelConfig;
  private manager: LabelManager;
  private fontAtlas: FontAtlas | null = null;
  private atlasLoading: Promise<void> | null = null;
  private renderContext: LabelsRenderContext | null = null;

  // GPU resources
  private pipeline: GPURenderPipeline | null = null;
  private viewportUniformBuffer: GPUBuffer | null = null;
  private labelUniformBuffer: GPUBuffer | null = null;
  private glyphStorageBuffer: GPUBuffer | null = null;
  private bindGroup0: GPUBindGroup | null = null;
  private bindGroup1: GPUBindGroup | null = null;
  private bindGroup2: GPUBindGroup | null = null;

  // Current state
  private _currentGlyphCount: number = 0;
  private maxGlyphCapacity: number = 10000;
  private needsRebuild: boolean = true;
  private _order: number = 100; // Labels render on top
  private isInitialized: boolean = false;

  constructor(
    id: string,
    context: GPUContext,
    config: Partial<LabelConfig> = {},
  ) {
    this.id = id;
    this.context = context;
    this.config = { ...DEFAULT_LABEL_CONFIG, ...config };
    this.manager = new LabelManager({
      maxLabels: this.config.maxLabels,
      fontSize: this.config.fontSize,
      minZoom: this.config.minZoom,
      labelPadding: this.config.labelPadding,
    });

    // Start loading font atlas and auto-initialize
    this.atlasLoading = this.loadAtlas();
    this.initialize().catch((err) => {
      console.error("[HeroineGraph] LabelsLayer initialization failed:", err);
    });
  }

  get enabled(): boolean {
    return this.config.visible;
  }

  set enabled(value: boolean) {
    this.config.visible = value;
  }

  get order(): number {
    return this._order;
  }

  set order(value: number) {
    this._order = value;
  }

  /**
   * Get the current number of glyphs being rendered
   */
  get glyphCount(): number {
    return this._currentGlyphCount;
  }

  /**
   * Set the render context (viewport info)
   */
  setRenderContext(context: LabelsRenderContext): void {
    this.renderContext = context;
  }

  /**
   * Load the font atlas asynchronously
   */
  private async loadAtlas(): Promise<void> {
    try {
      this.fontAtlas = await loadDefaultFontAtlas(this.context);
      this.manager.setFontAtlas(this.fontAtlas);
      this.needsRebuild = true;
    } catch (error) {
      console.error("[LabelsLayer] Failed to load font atlas:", error);
    }
  }

  /**
   * Initialize GPU resources
   */
  async initialize(): Promise<void> {
    // Wait for atlas to load
    if (this.atlasLoading) {
      await this.atlasLoading;
      this.atlasLoading = null;
    }

    if (!this.fontAtlas) {
      console.warn("[LabelsLayer] Font atlas not loaded, skipping initialization");
      return;
    }

    const { device } = this.context;

    // Create uniform buffers
    // ViewportUniforms struct alignment:
    // - offset: vec2<f32> (8 bytes) at 0
    // - scale: f32 (4 bytes) at 8
    // - canvas_width: f32 (4 bytes) at 12
    // - canvas_height: f32 (4 bytes) at 16
    // - padding (12 bytes) at 20 to align vec3 to 16-byte boundary
    // - _padding: vec3<f32> (12 bytes) at 32
    // - final padding (4 bytes) at 44 to round struct to 16-byte multiple
    // Total: 48 bytes
    this.viewportUniformBuffer = device.createBuffer({
      label: "Label Viewport Uniforms",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.labelUniformBuffer = device.createBuffer({
      label: "Label Uniforms",
      size: 48, // vec4 + 4 floats + atlas_font_size + 3 padding = 48 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create glyph storage buffer with initial capacity
    // 12 floats (48 bytes) per glyph to match WGSL struct alignment
    this.glyphStorageBuffer = device.createBuffer({
      label: "Glyph Instance Storage",
      size: this.maxGlyphCapacity * 12 * 4, // 12 floats per glyph (48 bytes)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Create shader modules
    const vertexModule = device.createShaderModule({
      label: "Label Vertex Shader",
      code: labelVertexShader,
    });

    const fragmentModule = device.createShaderModule({
      label: "Label Fragment Shader",
      code: labelFragmentShader,
    });

    // Create bind group layouts
    const bindGroupLayout0 = device.createBindGroupLayout({
      label: "Label Uniforms Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    const bindGroupLayout1 = device.createBindGroupLayout({
      label: "Glyph Storage Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" },
        },
      ],
    });

    const bindGroupLayout2 = device.createBindGroupLayout({
      label: "Font Atlas Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" },
        },
      ],
    });

    // Create pipeline layout
    const pipelineLayout = device.createPipelineLayout({
      label: "Labels Pipeline Layout",
      bindGroupLayouts: [bindGroupLayout0, bindGroupLayout1, bindGroupLayout2],
    });

    // Create render pipeline
    this.pipeline = device.createRenderPipeline({
      label: "Labels Render Pipeline",
      layout: pipelineLayout,
      vertex: {
        module: vertexModule,
        entryPoint: "vs_main",
      },
      fragment: {
        module: fragmentModule,
        entryPoint: "fs_main",
        targets: [
          {
            format: this.context.format,
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

    // Create bind groups
    this.bindGroup0 = device.createBindGroup({
      label: "Label Uniforms Bind Group",
      layout: bindGroupLayout0,
      entries: [
        { binding: 0, resource: { buffer: this.viewportUniformBuffer } },
        { binding: 1, resource: { buffer: this.labelUniformBuffer } },
      ],
    });

    this.bindGroup1 = device.createBindGroup({
      label: "Glyph Storage Bind Group",
      layout: bindGroupLayout1,
      entries: [
        { binding: 0, resource: { buffer: this.glyphStorageBuffer } },
      ],
    });

    this.bindGroup2 = device.createBindGroup({
      label: "Font Atlas Bind Group",
      layout: bindGroupLayout2,
      entries: [
        { binding: 0, resource: this.fontAtlas.view },
        { binding: 1, resource: this.fontAtlas.sampler },
      ],
    });

    this.needsRebuild = false;
    this.isInitialized = true;
  }

  /**
   * Set labels from node data
   */
  setLabels(labels: LabelData[]): void {
    this.manager.setLabels(labels);
  }

  /**
   * Update label positions (call when nodes move)
   */
  updateLabelPositions(
    _positions: { nodeId: number; x: number; y: number }[],
  ): void {
    // This would update the label positions based on node movement
    // For now, we rebuild labels each frame from the render context
  }

  /**
   * Get current configuration
   */
  getConfig(): LabelConfig {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  setConfig(config: Partial<LabelConfig>): void {
    this.config = { ...this.config, ...config };
    this.manager.setConfig({
      maxLabels: this.config.maxLabels,
      fontSize: this.config.fontSize,
      minZoom: this.config.minZoom,
      labelPadding: this.config.labelPadding,
    });
  }

  /**
   * Check if the layer is visible
   */
  isVisible(): boolean {
    return this.config.visible;
  }

  /**
   * Set layer visibility
   */
  setVisible(visible: boolean): void {
    this.config.visible = visible;
  }

  /**
   * Render the labels layer
   */
  render(
    encoder: GPUCommandEncoder,
    textureView: GPUTextureView,
  ): void {
    if (!this.config.visible) {
      return;
    }

    if (!this.isInitialized || !this.fontAtlas || !this.pipeline || this.needsRebuild) {
      // Not ready yet - initialization is in progress
      return;
    }

    if (!this.renderContext) {
      // No render context set
      return;
    }

    const { device } = this.context;
    const { viewportX, viewportY, scale, canvasWidth, canvasHeight, positionProvider } =
      this.renderContext;

    // Get visible labels with culling (use position provider for dynamic positions)
    const visibleLabels = this.manager.getVisibleLabels(
      viewportX,
      viewportY,
      scale,
      canvasWidth,
      canvasHeight,
      positionProvider,
    );

    if (visibleLabels.length === 0) {
      return;
    }

    // Generate glyph instances
    const { instances, count } = this.manager.generateGlyphInstances(
      visibleLabels,
      viewportX,
      viewportY,
      scale,
      canvasWidth,
      canvasHeight,
    );

    if (count === 0) {
      return;
    }

    // Resize storage buffer if needed
    if (count > this.maxGlyphCapacity) {
      this.maxGlyphCapacity = Math.ceil(count * 1.5);
      this.glyphStorageBuffer?.destroy();
      this.glyphStorageBuffer = device.createBuffer({
        label: "Glyph Instance Storage",
        size: this.maxGlyphCapacity * 12 * 4, // 12 floats per glyph (48 bytes)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      // Recreate bind group with new buffer
      const bindGroupLayout1 = this.pipeline.getBindGroupLayout(1);
      this.bindGroup1 = device.createBindGroup({
        label: "Glyph Storage Bind Group",
        layout: bindGroupLayout1,
        entries: [
          { binding: 0, resource: { buffer: this.glyphStorageBuffer } },
        ],
      });
    }

    // Upload glyph data - create new Float32Array to ensure proper ArrayBuffer type
    const glyphData = new Float32Array(instances);
    device.queue.writeBuffer(this.glyphStorageBuffer!, 0, glyphData);

    // Update viewport uniforms (48 bytes = 12 floats)
    // Layout matches ViewportUniforms struct in shader
    const viewportData = new Float32Array([
      viewportX, // offset.x at byte 0
      viewportY, // offset.y at byte 4
      scale, // scale at byte 8
      canvasWidth, // canvas_width at byte 12
      canvasHeight, // canvas_height at byte 16
      0,
      0,
      0, // padding to align vec3 at byte 20-31
      0,
      0,
      0, // _padding vec3 at byte 32-43
      0, // final padding at byte 44-47
    ]);
    device.queue.writeBuffer(this.viewportUniformBuffer!, 0, viewportData);

    // Update label uniforms
    const [r, g, b, a] = parseColor(this.config.fontColor);
    const atlasFontSize = this.fontAtlas.info?.size ?? 42;
    const labelData = new Float32Array([
      r,
      g,
      b,
      a, // color
      this.config.fontSize,
      this.fontAtlas.distanceRange,
      this.fontAtlas.common.scaleW,
      this.fontAtlas.common.scaleH,
      atlasFontSize, // atlas_font_size
      0, // _pad0
      0, // _pad1
      0, // _pad2
    ]);
    device.queue.writeBuffer(this.labelUniformBuffer!, 0, labelData);

    // Create render pass
    const renderPass = encoder.beginRenderPass({
      label: "Labels Render Pass",
      colorAttachments: [
        {
          view: textureView,
          loadOp: "load",
          storeOp: "store",
        },
      ],
    });

    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, this.bindGroup0!);
    renderPass.setBindGroup(1, this.bindGroup1!);
    renderPass.setBindGroup(2, this.bindGroup2!);

    // Draw instanced quads (6 vertices per quad, one instance per glyph)
    renderPass.draw(6, count);

    renderPass.end();

    this._currentGlyphCount = count;
  }

  /**
   * Resize handler
   */
  resize(_width: number, _height: number): void {
    // No resize-dependent resources to update
  }

  /**
   * Destroy all GPU resources
   */
  destroy(): void {
    this.viewportUniformBuffer?.destroy();
    this.labelUniformBuffer?.destroy();
    this.glyphStorageBuffer?.destroy();
    this.fontAtlas?.destroy();
  }
}
