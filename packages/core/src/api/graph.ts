/**
 * HeroineGraph Main Class
 *
 * The main class for graph visualization. Handles data loading, rendering,
 * simulation control, and user interaction.
 *
 * @module
 */

import type {
  GraphInput,
  GraphTypedInput,
  GraphConfig,
  Vec2,
  BoundingBox,
  SimulationStatus,
  ViewportState,
  EventHandler,
  HeroineGraphEvent,
} from "../types.ts";
import type { GPUContext } from "../webgpu/context.ts";
import { HeroineGraphError, ErrorCode } from "../errors.ts";
import { EventEmitter, createEventEmitter } from "../events/emitter.ts";
import { Viewport, createViewport } from "../viewport/viewport.ts";
import { createViewportUniformBuffer, type ViewportUniformBuffer } from "../viewport/uniforms.ts";
import { parseGraphInput, type ParsedGraph } from "../graph/parser.ts";
import { parseGraphTypedInput } from "../graph/typed_parser.ts";
import { initializePositions, needsInitialization } from "../graph/initialize.ts";
import { createNodeRenderPipeline, createNodeBindGroup, createViewportBindGroup, renderNodes, type NodeRenderPipeline } from "../renderer/pipelines/nodes.ts";
import { createEdgeRenderPipeline, createEdgeBindGroup, renderEdges, type EdgeRenderPipeline } from "../renderer/pipelines/edges.ts";
import { createRenderLoop, type RenderLoop, type FrameStats } from "../renderer/render_loop.ts";
import { createSimulationController, type SimulationController } from "../simulation/controller.ts";
import { createEdgeIndicesBuffer } from "../graph/parser.ts";
import { fitBoundsScale, boundsCenter } from "../viewport/transforms.ts";
import {
  createSimulationPipeline,
  createSimulationBuffers,
  createSimulationBindGroups,
  copyPositionsToSimulation,
  copyEdgesToSimulation,
  updateSimulationUniforms,
  recordSimulationStep,
  swapSimulationBuffers,
  type SimulationPipeline,
  type SimulationBuffers,
  type SimulationBindGroups,
} from "../simulation/pipeline.ts";

/**
 * HeroineGraph configuration
 */
export interface HeroineGraphConfig {
  /** GPU context */
  gpuContext: GPUContext;
  /** WASM engine instance */
  wasmEngine: unknown;
  /** Canvas element */
  canvas: HTMLCanvasElement;
  /** Graph configuration */
  config?: Partial<GraphConfig>;
  /** Debug mode */
  debug?: boolean;
}

/**
 * Internal graph state
 */
interface GraphState {
  loaded: boolean;
  nodeCount: number;
  edgeCount: number;
  parsedGraph: ParsedGraph | null;
}

/**
 * GPU buffers for rendering
 */
interface GPUBuffers {
  positionsX: GPUBuffer;
  positionsY: GPUBuffer;
  nodeAttributes: GPUBuffer;
  edgeIndices: GPUBuffer;
  edgeAttributes: GPUBuffer;
  viewportUniforms: GPUBuffer;
}

/**
 * Main HeroineGraph class
 */
export class HeroineGraph {
  // Configuration
  private readonly gpuContext: GPUContext;
  private readonly canvas: HTMLCanvasElement;
  private readonly debug: boolean;

  // State
  private state: GraphState;
  private disposed: boolean = false;

  // Components
  private viewport: Viewport;
  private viewportUniformBuffer: ViewportUniformBuffer;
  private events: EventEmitter<Record<string, HeroineGraphEvent>>;
  private renderLoop: RenderLoop;
  private simulationController: SimulationController;

  // Pipelines
  private nodePipeline: NodeRenderPipeline | null = null;
  private edgePipeline: EdgeRenderPipeline | null = null;
  private simulationPipeline: SimulationPipeline | null = null;

  // GPU resources
  private buffers: GPUBuffers | null = null;
  private nodeBindGroup: GPUBindGroup | null = null;
  private edgeBindGroup: GPUBindGroup | null = null;
  private viewportBindGroup: GPUBindGroup | null = null;

  // GPU Simulation resources
  private simBuffers: SimulationBuffers | null = null;
  private simBindGroups: SimulationBindGroups | null = null;

  constructor(config: HeroineGraphConfig) {
    this.gpuContext = config.gpuContext;
    this.canvas = config.canvas;
    this.debug = config.debug ?? false;

    // Initialize state
    this.state = {
      loaded: false,
      nodeCount: 0,
      edgeCount: 0,
      parsedGraph: null,
    };

    // Create viewport
    this.viewport = createViewport(this.canvas, {
      onViewportChange: (state) => {
        this.updateViewportUniforms();
        this.events.emit("viewport:change", {
          type: "viewport:change",
          timestamp: Date.now(),
          ...state,
        });
      },
    });

    // Create viewport uniform buffer
    this.viewportUniformBuffer = createViewportUniformBuffer(
      this.gpuContext.device,
    );

    // Create event emitter
    this.events = createEventEmitter();

    // Create render loop
    this.renderLoop = createRenderLoop((deltaTime, stats) => {
      this.renderFrame(deltaTime, stats);
    });

    // Create simulation controller
    this.simulationController = createSimulationController();

    // Initialize pipelines
    this.initializePipelines();

    // Start render loop
    this.renderLoop.start();

    if (this.debug) {
      console.log("HeroineGraph instance created");
    }
  }

  /**
   * Initialize render pipelines
   */
  private initializePipelines(): void {
    const format = this.gpuContext.format;

    this.nodePipeline = createNodeRenderPipeline(this.gpuContext, { format });
    this.edgePipeline = createEdgeRenderPipeline(this.gpuContext, { format });
    this.simulationPipeline = createSimulationPipeline(this.gpuContext);
  }

  /**
   * Update viewport uniform buffer
   */
  private updateViewportUniforms(): void {
    if (!this.viewportUniformBuffer) return;

    const state = this.viewport.state;
    const { width, height } = this.canvas;

    this.viewportUniformBuffer.update(
      this.gpuContext.device,
      state,
      width,
      height,
    );
  }

  /**
   * Record GPU simulation commands
   * Returns the command encoder with simulation passes recorded
   */
  private recordSimulationCommands(encoder: GPUCommandEncoder): void {
    if (!this.simulationPipeline || !this.simBuffers || !this.simBindGroups) return;
    if (!this.simulationController.isRunning) return;

    const { device } = this.gpuContext;
    const alpha = this.simulationController.state.alpha;

    // Update uniforms with current alpha
    updateSimulationUniforms(
      device,
      this.simBuffers,
      this.state.nodeCount,
      this.state.edgeCount,
      alpha
    );

    // Record simulation compute passes
    recordSimulationStep(
      encoder,
      this.simulationPipeline,
      this.simBindGroups,
      this.state.nodeCount,
      this.state.edgeCount
    );

    // Tick the simulation controller
    this.simulationController.tick();
  }

  /**
   * Swap buffers after simulation and rebuild bind groups
   */
  private swapAndRebuildBindGroups(): void {
    if (!this.simBuffers || !this.simulationPipeline) return;

    // Swap the ping-pong buffers
    swapSimulationBuffers(this.simBuffers);

    // Rebuild bind groups with swapped buffers
    this.simBindGroups = createSimulationBindGroups(
      this.gpuContext.device,
      this.simulationPipeline,
      this.simBuffers
    );

    // Also update render bind groups to use new position buffers
    if (this.nodePipeline) {
      this.nodeBindGroup = createNodeBindGroup(
        this.gpuContext.device,
        this.nodePipeline,
        this.simBuffers.positionsX,
        this.simBuffers.positionsY,
        this.buffers!.nodeAttributes
      );
    }

    if (this.edgePipeline && this.buffers) {
      this.edgeBindGroup = createEdgeBindGroup(
        this.gpuContext.device,
        this.edgePipeline,
        this.simBuffers.positionsX,
        this.simBuffers.positionsY,
        this.buffers.edgeIndices,
        this.buffers.edgeAttributes
      );
    }
  }

  /**
   * Render a frame
   */
  private renderFrame(_deltaTime: number, _stats: FrameStats): void {
    if (this.disposed || !this.state.loaded) return;

    const { device, context } = this.gpuContext;

    // Create command encoder
    const encoder = device.createCommandEncoder();

    // Run GPU simulation compute passes
    if (this.simulationController.isRunning) {
      this.recordSimulationCommands(encoder);
    }

    // Get current texture
    const texture = context.getCurrentTexture();
    const textureView = texture.createView();

    // Begin render pass
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.95, g: 0.95, b: 0.95, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    // Render edges first (below nodes)
    if (
      this.edgePipeline &&
      this.viewportBindGroup &&
      this.edgeBindGroup &&
      this.state.edgeCount > 0
    ) {
      renderEdges(
        renderPass,
        this.edgePipeline,
        this.viewportBindGroup,
        this.edgeBindGroup,
        this.state.edgeCount,
      );
    }

    // Render nodes
    if (
      this.nodePipeline &&
      this.viewportBindGroup &&
      this.nodeBindGroup &&
      this.state.nodeCount > 0
    ) {
      renderNodes(
        renderPass,
        this.nodePipeline,
        this.viewportBindGroup,
        this.nodeBindGroup,
        this.state.nodeCount,
      );
    }

    renderPass.end();

    // Submit commands
    device.queue.submit([encoder.finish()]);

    // Swap buffers after GPU execution for next frame
    if (this.simulationController.isRunning) {
      this.swapAndRebuildBindGroups();
    }
  }

  // ==========================================================================
  // Public API - Data Loading
  // ==========================================================================

  /**
   * Load graph data
   *
   * @param data - Graph data (GraphInput or GraphTypedInput)
   * @returns Promise that resolves when data is loaded
   */
  async load(data: GraphInput | GraphTypedInput): Promise<void> {
    if (this.disposed) {
      throw new HeroineGraphError(
        ErrorCode.DISPOSED_ACCESS,
        "Cannot load data on disposed graph",
      );
    }

    // Parse input
    const isTyped = "nodeCount" in data;
    const parsed = isTyped
      ? parseGraphTypedInput(data as GraphTypedInput)
      : parseGraphInput(data as GraphInput);

    // Initialize positions if needed
    if (needsInitialization(parsed.positionsX, parsed.positionsY)) {
      initializePositions(parsed.positionsX, parsed.positionsY, {
        strategy: "phyllotaxis",
        radius: Math.sqrt(parsed.nodeCount) * 10,
      });
    }

    // Create GPU buffers for rendering
    await this.createBuffers(parsed);

    // Update state
    this.state.loaded = true;
    this.state.nodeCount = parsed.nodeCount;
    this.state.edgeCount = parsed.edgeCount;
    this.state.parsedGraph = parsed;

    // Create GPU simulation buffers and bind groups
    await this.createSimulationResources(parsed);

    // Fit view to content
    this.fitToView();

    // Start simulation automatically
    this.simulationController.restart();

    // Emit load event
    this.events.emit("graph:load", {
      type: "graph:load",
      timestamp: Date.now(),
      nodeCount: parsed.nodeCount,
      edgeCount: parsed.edgeCount,
    });

    if (this.debug) {
      console.log(
        `Loaded graph: ${parsed.nodeCount} nodes, ${parsed.edgeCount} edges (GPU simulation enabled)`,
      );
    }
  }

  /**
   * Create GPU simulation resources
   */
  private async createSimulationResources(parsed: ParsedGraph): Promise<void> {
    if (!this.simulationPipeline) return;

    const { device } = this.gpuContext;

    // Create simulation buffers
    this.simBuffers = createSimulationBuffers(
      device,
      parsed.nodeCount,
      parsed.edgeCount
    );

    // Copy initial positions to simulation buffers
    copyPositionsToSimulation(
      device,
      this.simBuffers,
      parsed.positionsX,
      parsed.positionsY
    );

    // Copy edge data to simulation buffers
    copyEdgesToSimulation(
      device,
      this.simBuffers,
      parsed.edgeSources,
      parsed.edgeTargets
    );

    // Initialize uniforms
    updateSimulationUniforms(
      device,
      this.simBuffers,
      parsed.nodeCount,
      parsed.edgeCount,
      1.0 // Initial alpha
    );

    // Create simulation bind groups
    this.simBindGroups = createSimulationBindGroups(
      device,
      this.simulationPipeline,
      this.simBuffers
    );

    // Update render bind groups to use simulation position buffers
    if (this.nodePipeline) {
      this.nodeBindGroup = createNodeBindGroup(
        device,
        this.nodePipeline,
        this.simBuffers.positionsX,
        this.simBuffers.positionsY,
        this.buffers!.nodeAttributes
      );
    }

    if (this.edgePipeline && this.buffers) {
      this.edgeBindGroup = createEdgeBindGroup(
        device,
        this.edgePipeline,
        this.simBuffers.positionsX,
        this.simBuffers.positionsY,
        this.buffers.edgeIndices,
        this.buffers.edgeAttributes
      );
    }
  }

  /**
   * Create GPU buffers from parsed graph
   */
  private async createBuffers(parsed: ParsedGraph): Promise<void> {
    const { device } = this.gpuContext;

    // Destroy old buffers
    this.destroyBuffers();

    // Create position buffers
    const positionsX = device.createBuffer({
      label: "Positions X",
      size: parsed.positionsX.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(positionsX, 0, parsed.positionsX);

    const positionsY = device.createBuffer({
      label: "Positions Y",
      size: parsed.positionsY.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(positionsY, 0, parsed.positionsY);

    // Create node attributes buffer
    const nodeAttributes = device.createBuffer({
      label: "Node Attributes",
      size: parsed.nodeAttributes.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(nodeAttributes, 0, parsed.nodeAttributes);

    // Create edge buffers
    const edgeIndicesData = createEdgeIndicesBuffer(
      parsed.edgeSources,
      parsed.edgeTargets,
    );
    const edgeIndices = device.createBuffer({
      label: "Edge Indices",
      size: Math.max(edgeIndicesData.byteLength, 4), // Minimum size
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (edgeIndicesData.byteLength > 0) {
      device.queue.writeBuffer(edgeIndices, 0, edgeIndicesData);
    }

    const edgeAttributes = device.createBuffer({
      label: "Edge Attributes",
      size: Math.max(parsed.edgeAttributes.byteLength, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (parsed.edgeAttributes.byteLength > 0) {
      device.queue.writeBuffer(edgeAttributes, 0, parsed.edgeAttributes);
    }

    // Store buffers
    this.buffers = {
      positionsX,
      positionsY,
      nodeAttributes,
      edgeIndices,
      edgeAttributes,
      viewportUniforms: this.viewportUniformBuffer.buffer,
    };

    // Create bind groups
    if (this.nodePipeline) {
      this.viewportBindGroup = createViewportBindGroup(
        device,
        this.nodePipeline,
        this.viewportUniformBuffer.buffer,
      );

      this.nodeBindGroup = createNodeBindGroup(
        device,
        this.nodePipeline,
        positionsX,
        positionsY,
        nodeAttributes,
      );
    }

    if (this.edgePipeline) {
      this.edgeBindGroup = createEdgeBindGroup(
        device,
        this.edgePipeline,
        positionsX,
        positionsY,
        edgeIndices,
        edgeAttributes,
      );
    }

    // Update viewport uniforms
    this.updateViewportUniforms();
  }

  /**
   * Destroy GPU buffers
   */
  private destroyBuffers(): void {
    if (this.buffers) {
      this.buffers.positionsX.destroy();
      this.buffers.positionsY.destroy();
      this.buffers.nodeAttributes.destroy();
      this.buffers.edgeIndices.destroy();
      this.buffers.edgeAttributes.destroy();
      this.buffers = null;
    }

    this.nodeBindGroup = null;
    this.edgeBindGroup = null;
  }

  // ==========================================================================
  // Public API - Viewport Control
  // ==========================================================================

  /**
   * Pan the viewport by a delta
   */
  pan(dx: number, dy: number): void {
    this.viewport.pan(dx, dy);
  }

  /**
   * Zoom the viewport
   */
  zoom(factor: number, center?: Vec2): void {
    this.viewport.zoom(factor, center);
  }

  /**
   * Set the viewport center
   */
  setCenter(x: number, y: number): void {
    this.viewport.setCenter(x, y);
  }

  /**
   * Set the viewport scale
   */
  setScale(scale: number): void {
    this.viewport.setScale(scale);
  }

  /**
   * Fit the viewport to show all content
   */
  fitToView(padding: number = 50): void {
    if (!this.state.parsedGraph) return;

    const { positionsX, positionsY, nodeCount } = this.state.parsedGraph;

    // Calculate bounds
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;

    for (let i = 0; i < nodeCount; i++) {
      const x = positionsX[i];
      const y = positionsY[i];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }

    if (minX === Infinity) return; // No nodes

    const bounds: BoundingBox = {
      minX: minX - padding,
      minY: minY - padding,
      maxX: maxX + padding,
      maxY: maxY + padding,
    };

    const { width, height } = this.canvas;
    const scale = fitBoundsScale(bounds, width, height);
    const center = boundsCenter(bounds);

    this.viewport.setScale(scale);
    this.viewport.setCenter(center.x, center.y);
  }

  /**
   * Get current viewport state
   */
  getViewportState(): ViewportState {
    return { ...this.viewport.state };
  }

  // ==========================================================================
  // Public API - Simulation Control
  // ==========================================================================

  /**
   * Start the force simulation
   */
  startSimulation(): void {
    this.simulationController.start();
  }

  /**
   * Pause the force simulation
   */
  pauseSimulation(): void {
    this.simulationController.pause();
  }

  /**
   * Stop the force simulation
   */
  stopSimulation(): void {
    this.simulationController.stop();
  }

  /**
   * Restart the force simulation
   */
  restartSimulation(): void {
    this.simulationController.restart();
  }

  /**
   * Get current simulation status
   */
  getSimulationStatus(): SimulationStatus {
    return this.simulationController.state.status;
  }

  /**
   * Set simulation alpha
   */
  setSimulationAlpha(alpha: number): void {
    this.simulationController.setAlpha(alpha);
  }

  // ==========================================================================
  // Public API - Events
  // ==========================================================================

  /**
   * Subscribe to an event
   */
  on<K extends string>(event: K, handler: EventHandler<HeroineGraphEvent>): void {
    this.events.on(event, handler);
  }

  /**
   * Unsubscribe from an event
   */
  off<K extends string>(event: K, handler: EventHandler<HeroineGraphEvent>): void {
    this.events.off(event, handler);
  }

  /**
   * Subscribe to an event once
   */
  once<K extends string>(event: K, handler: EventHandler<HeroineGraphEvent>): void {
    this.events.once(event, handler);
  }

  // ==========================================================================
  // Public API - Lifecycle
  // ==========================================================================

  /**
   * Resize the graph canvas
   */
  resize(width?: number, height?: number): void {
    if (width !== undefined && height !== undefined) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    this.updateViewportUniforms();
  }

  /**
   * Dispose of all resources
   */
  dispose(): void {
    if (this.disposed) return;

    this.disposed = true;

    // Stop render loop
    this.renderLoop.stop();

    // Stop simulation
    this.simulationController.stop();

    // Dispose viewport
    this.viewport.dispose();

    // Destroy render buffers
    this.destroyBuffers();

    // Destroy simulation buffers
    this.destroySimulationBuffers();

    // Destroy viewport uniform buffer
    this.viewportUniformBuffer.destroy();

    // Clear event listeners
    this.events.clear();

    if (this.debug) {
      console.log("HeroineGraph disposed");
    }
  }

  /**
   * Destroy simulation GPU buffers
   */
  private destroySimulationBuffers(): void {
    if (this.simBuffers) {
      this.simBuffers.positionsX.destroy();
      this.simBuffers.positionsY.destroy();
      this.simBuffers.positionsXOut.destroy();
      this.simBuffers.positionsYOut.destroy();
      this.simBuffers.velocitiesX.destroy();
      this.simBuffers.velocitiesY.destroy();
      this.simBuffers.velocitiesXOut.destroy();
      this.simBuffers.velocitiesYOut.destroy();
      this.simBuffers.forcesX.destroy();
      this.simBuffers.forcesY.destroy();
      this.simBuffers.edgeSources.destroy();
      this.simBuffers.edgeTargets.destroy();
      this.simBuffers.clearUniforms.destroy();
      this.simBuffers.repulsionUniforms.destroy();
      this.simBuffers.springUniforms.destroy();
      this.simBuffers.integrationUniforms.destroy();
      this.simBuffers = null;
    }
    this.simBindGroups = null;
  }

  // ==========================================================================
  // Public API - Info
  // ==========================================================================

  /**
   * Get node count
   */
  get nodeCount(): number {
    return this.state.nodeCount;
  }

  /**
   * Get edge count
   */
  get edgeCount(): number {
    return this.state.edgeCount;
  }

  /**
   * Check if graph is loaded
   */
  get isLoaded(): boolean {
    return this.state.loaded;
  }

  /**
   * Get frame stats
   */
  get frameStats(): FrameStats {
    return this.renderLoop.stats;
  }
}
