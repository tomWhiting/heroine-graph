/**
 * HeroineGraph Main Class
 *
 * The main class for graph visualization. Handles data loading, rendering,
 * simulation control, and user interaction.
 *
 * @module
 */

import type {
  BoundingBox,
  EdgeFlowConfig,
  EdgeId,
  EventHandler,
  EventMap,
  GraphConfig,
  GraphInput,
  GraphTypedInput,
  NodeId,
  SimulationStatus,
  Vec2,
  ViewportState,
} from "../types.ts";
import type { GPUContext } from "../webgpu/context.ts";
import { toArrayBuffer } from "../webgpu/buffer_utils.ts";
import { ErrorCode, HeroineGraphError } from "../errors.ts";
import { createEventEmitter, type EventEmitter } from "../events/emitter.ts";
import { createViewport, type Viewport } from "../viewport/viewport.ts";
import { createViewportUniformBuffer, type ViewportUniformBuffer } from "../viewport/uniforms.ts";
import { type ParsedGraph, parseGraphInput } from "../graph/parser.ts";
import { parseGraphTypedInput } from "../graph/typed_parser.ts";
import { initializePositions, needsInitialization } from "../graph/initialize.ts";
import {
  createNodeBindGroup,
  createNodeRenderPipeline,
  createViewportBindGroup,
  type NodeRenderPipeline,
  renderNodes,
} from "../renderer/pipelines/nodes.ts";
import {
  createEdgeBindGroup,
  createEdgeRenderPipeline,
  type EdgeRenderPipeline,
  renderEdges,
  updateEdgeFlowUniforms,
} from "../renderer/pipelines/edges.ts";
import {
  DEFAULT_EDGE_FLOW_CONFIG,
  type EdgeFlowPreset,
  EDGE_FLOW_PRESETS,
} from "../renderer/edge_flow.ts";
import { createRenderLoop, type FrameStats, type RenderLoop } from "../renderer/render_loop.ts";
import { createSimulationController, type SimulationController } from "../simulation/controller.ts";
import {
  DEFAULT_FORCE_CONFIG,
  type FullForceConfig,
  validateForceConfig,
} from "../simulation/config.ts";
import { createEdgeIndicesBuffer } from "../graph/parser.ts";
import { boundsCenter, fitBoundsScale } from "../viewport/transforms.ts";
import {
  copyEdgesToSimulation,
  copyPositionsToReadback,
  copyPositionsToSimulation,
  createSimulationBindGroups,
  createSimulationBuffers,
  createSimulationPipeline,
  readbackPositions,
  recordSimulationStepWithOptions,
  type SimulationBindGroups,
  type SimulationBuffers,
  type SimulationPipeline,
  swapSimulationBuffers,
  updateSimulationUniforms,
} from "../simulation/pipeline.ts";
import {
  createHitTester,
  type HitTester,
  type SpatialQueryEngine,
} from "../interaction/hit_test.ts";
import { createPointerManager, type PointerManager } from "../interaction/pointer.ts";
import {
  type ColorScaleName,
  type ContourConfig,
  // Contour layer
  type ContourLayer,
  type ContourRenderContext,
  createContourLayer,
  createHeatmapLayer,
  createLayerManager,
  createMetaballLayer,
  type HeatmapConfig,
  type HeatmapLayer,
  type HeatmapRenderContext,
  type LabelConfig,
  type LabelData,
  // Labels layer
  LabelsLayer,
  type LabelsRenderContext,
  type LayerInfo,
  type LayerManager,
  type MetaballConfig,
  // Metaball layer
  type MetaballLayer,
  type MetaballRenderContext,
} from "../layers/mod.ts";
import {
  type AlgorithmBindGroups,
  type AlgorithmBuffers,
  type AlgorithmPipelines,
  type AlgorithmRenderContext,
  type ForceAlgorithm,
  type ForceAlgorithmType,
  getAlgorithmRegistry,
  initializeBuiltinAlgorithms,
} from "../simulation/algorithms/mod.ts";
import {
  createStreamManager,
  type StreamBulkData,
  type StreamDataPoint,
  type StreamInfo,
  type StreamManager,
  type ValueStreamConfig,
} from "../streams/mod.ts";

/**
 * WASM engine interface for spatial queries.
 * This matches the HeroineGraphWasm API exposed by the WASM module.
 */
interface WasmEngine extends SpatialQueryEngine {
  findNearestNode(x: number, y: number): number | undefined;
}

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
  private readonly wasmEngine: WasmEngine | null;
  private readonly canvas: HTMLCanvasElement;
  private readonly debug: boolean;

  // State
  private state: GraphState;
  private disposed: boolean = false;

  // Components
  private viewport: Viewport;
  private viewportUniformBuffer: ViewportUniformBuffer;
  private events: EventEmitter;
  private renderLoop: RenderLoop;
  private simulationController: SimulationController;
  private forceConfig: FullForceConfig;

  // Edge flow animation
  private flowConfig: EdgeFlowConfig;
  private flowStartTime: number = 0;

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

  // Force algorithm resources
  private currentAlgorithm: ForceAlgorithm | null = null;
  private algorithmPipelines: AlgorithmPipelines | null = null;
  private algorithmBuffers: AlgorithmBuffers | null = null;
  private algorithmBindGroups: AlgorithmBindGroups | null = null;

  // Interaction
  private hitTester: HitTester;
  private pointerManager: PointerManager | null = null;
  private selectedNodes: Set<NodeId> = new Set();
  private selectedEdges: Set<EdgeId> = new Set();
  private hoveredNode: NodeId | null = null;
  private hoveredEdge: EdgeId | null = null;
  private draggedNode: NodeId | null = null;
  private lastDragPosition: Vec2 | null = null;
  private pinnedNodes: Set<NodeId> = new Set();

  // Viewport panning state
  private isPanning: boolean = false;
  private lastPanPosition: Vec2 | null = null;

  // Position sync (GPU -> JS for hit testing)
  private syncFrameCounter: number = 0;
  private syncInProgress: boolean = false;
  private readonly SYNC_INTERVAL: number = 5; // Sync every N frames

  // Layer system
  private layerManager: LayerManager;

  // Value stream system
  private streamManager: StreamManager;

  // Heatmap stream intensity buffer (per-node values from stream)
  private heatmapIntensityBuffer: GPUBuffer | null = null;

  constructor(config: HeroineGraphConfig) {
    this.gpuContext = config.gpuContext;
    this.wasmEngine = config.wasmEngine as WasmEngine | null;
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
        this.events.emit({
          type: "viewport:change",
          timestamp: Date.now(),
          viewport: state,
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

    // Initialize force configuration
    this.forceConfig = { ...DEFAULT_FORCE_CONFIG };

    // Initialize edge flow configuration (disabled by default)
    this.flowConfig = { ...DEFAULT_EDGE_FLOW_CONFIG };
    this.flowStartTime = performance.now();

    // Initialize force algorithm registry and default algorithm
    initializeBuiltinAlgorithms();
    const registry = getAlgorithmRegistry();
    this.currentAlgorithm = registry.get("n2") ?? null;
    if (this.currentAlgorithm) {
      this.algorithmPipelines = this.currentAlgorithm.createPipelines(this.gpuContext);
    }

    // Initialize pipelines
    this.initializePipelines();

    // Initialize hit tester
    // Hit testing uses per-node radius from nodeAttributes (+ 2 unit tolerance for easier clicking).
    // nodeHitRadius is a fallback maximum if per-node radius isn't available.
    this.hitTester = createHitTester({
      nodeHitRadius: 20,
      edgeHitRadius: 5,
      prioritizeNodes: true,
    });

    // Note: WASM spatial engine is not wired up yet - it needs to be populated
    // with node positions first. Using brute-force fallback for now.
    // TODO: Populate WASM engine with nodes and wire up for O(log n) hit testing
    if (this.debug) {
      console.log("Hit tester using brute-force fallback");
    }

    // Initialize pointer manager for interaction
    this.pointerManager = createPointerManager({
      canvas: this.canvas,
      viewport: this.viewport,
      preventDefault: true,
    });
    this.setupInteractionHandlers();

    // Initialize layer manager
    this.layerManager = createLayerManager();

    // Initialize stream manager for value streams
    this.streamManager = createStreamManager();

    // Note: render loop starts on first load() call, not here
    // This prevents rendering before canvas has valid dimensions

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
    // Use CSS dimensions for uniforms to match hit testing coordinate system.
    // WebGPU's canvas context handles devicePixelRatio internally.
    const cssWidth = this.canvas.clientWidth || this.canvas.width;
    const cssHeight = this.canvas.clientHeight || this.canvas.height;

    this.viewportUniformBuffer.update(
      this.gpuContext.device,
      state,
      cssWidth,
      cssHeight,
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

    // Update uniforms with current alpha and force config
    updateSimulationUniforms(
      device,
      this.simBuffers,
      this.state.nodeCount,
      this.state.edgeCount,
      alpha,
      this.forceConfig,
    );

    // Update algorithm uniforms if using custom algorithm
    if (this.currentAlgorithm && this.algorithmBuffers && this.algorithmBindGroups) {
      const context: AlgorithmRenderContext = {
        device,
        positionsX: this.simBuffers.positionsX,
        positionsY: this.simBuffers.positionsY,
        forcesX: this.simBuffers.forcesX,
        forcesY: this.simBuffers.forcesY,
        nodeCount: this.state.nodeCount,
        forceConfig: this.forceConfig,
      };
      this.currentAlgorithm.updateUniforms(device, this.algorithmBuffers, context);
    }

    // Record simulation compute passes with custom algorithm for repulsion
    recordSimulationStepWithOptions(
      encoder,
      this.simulationPipeline,
      this.simBindGroups,
      this.state.nodeCount,
      this.state.edgeCount,
      {
        recordRepulsionPass:
          this.currentAlgorithm && this.algorithmPipelines && this.algorithmBindGroups
            ? (enc) => {
              this.currentAlgorithm!.recordRepulsionPass(
                enc,
                this.algorithmPipelines!,
                this.algorithmBindGroups!,
                this.state.nodeCount,
              );
            }
            : undefined,
      },
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
      this.simBuffers,
    );

    // Also update render bind groups to use new position buffers
    if (this.nodePipeline) {
      this.nodeBindGroup = createNodeBindGroup(
        this.gpuContext.device,
        this.nodePipeline,
        this.simBuffers.positionsX,
        this.simBuffers.positionsY,
        this.buffers!.nodeAttributes,
      );
    }

    if (this.edgePipeline && this.buffers) {
      this.edgeBindGroup = createEdgeBindGroup(
        this.gpuContext.device,
        this.edgePipeline,
        this.simBuffers.positionsX,
        this.simBuffers.positionsY,
        this.buffers.edgeIndices,
        this.buffers.edgeAttributes,
      );
    }

    // Rebuild algorithm bind groups with swapped position/force buffers
    if (this.currentAlgorithm && this.algorithmPipelines && this.algorithmBuffers) {
      const context: AlgorithmRenderContext = {
        device: this.gpuContext.device,
        positionsX: this.simBuffers.positionsX,
        positionsY: this.simBuffers.positionsY,
        forcesX: this.simBuffers.forcesX,
        forcesY: this.simBuffers.forcesY,
        nodeCount: this.state.nodeCount,
        forceConfig: this.forceConfig,
      };

      this.algorithmBindGroups = this.currentAlgorithm.createBindGroups(
        this.gpuContext.device,
        this.algorithmPipelines,
        context,
        this.algorithmBuffers,
      );
    }

    // Update layer render contexts to use the new position buffers
    // This is critical: layers must point to the current read buffer,
    // not the output buffer being written to by the simulation
    this.updateLayerRenderContext();
  }

  /**
   * Render a frame
   */
  private renderFrame(_deltaTime: number, _stats: FrameStats): void {
    if (this.disposed || !this.state.loaded) return;

    // Skip rendering if canvas has no valid dimensions
    if (this.canvas.width === 0 || this.canvas.height === 0) return;

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

    // Clear the canvas first
    const clearPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.95, g: 0.95, b: 0.95, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    clearPass.end();

    // Update layer render contexts before rendering (ensures fresh texture references)
    this.updateLayerRenderContext();

    // Render background visualization layers FIRST (heatmap, contour, metaball render behind nodes)
    // Skip labels layer - it renders after nodes
    this.layerManager.render(encoder, textureView, ["labels"]);

    // Begin main render pass (loads existing content from layers)
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          loadOp: "load", // Preserve heatmap content
          storeOp: "store",
        },
      ],
    });

    // Render edges (below nodes, above heatmap)
    if (
      this.edgePipeline &&
      this.viewportBindGroup &&
      this.edgeBindGroup &&
      this.state.edgeCount > 0
    ) {
      // Update flow animation time
      const flowTime = (performance.now() - this.flowStartTime) / 1000.0;
      updateEdgeFlowUniforms(device, this.edgePipeline, this.flowConfig, flowTime);

      renderEdges(
        renderPass,
        this.edgePipeline,
        this.viewportBindGroup,
        this.edgeBindGroup,
        this.state.edgeCount,
      );
    }

    // Render nodes (on top)
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

    // Render overlay layers AFTER nodes (labels render on top)
    const labelsLayer = this.layerManager.getLayer<LabelsLayer>("labels");
    if (labelsLayer && labelsLayer.enabled) {
      labelsLayer.render(encoder, textureView);
    }

    // Schedule position sync if simulation is running
    let syncEncoder: GPUCommandEncoder | null = null;
    if (this.simulationController.isRunning && this.simBuffers && !this.syncInProgress) {
      this.syncFrameCounter++;
      if (this.syncFrameCounter >= this.SYNC_INTERVAL) {
        this.syncFrameCounter = 0;
        syncEncoder = device.createCommandEncoder();
        copyPositionsToReadback(syncEncoder, this.simBuffers);
      }
    }

    // Submit render commands
    device.queue.submit([encoder.finish()]);

    // Submit sync commands separately (if scheduled)
    if (syncEncoder) {
      device.queue.submit([syncEncoder.finish()]);
      this.performPositionReadback();
    }

    // Swap buffers after GPU execution for next frame
    if (this.simulationController.isRunning) {
      this.swapAndRebuildBindGroups();
    }
  }

  /**
   * Async readback of positions from GPU to JS arrays for hit testing
   */
  private async performPositionReadback(): Promise<void> {
    if (!this.simBuffers || !this.state.parsedGraph || this.syncInProgress) return;

    this.syncInProgress = true;
    try {
      await readbackPositions(
        this.simBuffers,
        this.state.parsedGraph.positionsX,
        this.state.parsedGraph.positionsY,
      );
    } catch (e) {
      // Readback failed - might happen if buffers are destroyed
      if (this.debug) {
        console.warn("Position readback failed:", e);
      }
    } finally {
      this.syncInProgress = false;
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
    this.createBuffers(parsed);

    // Update state
    this.state.loaded = true;
    this.state.nodeCount = parsed.nodeCount;
    this.state.edgeCount = parsed.edgeCount;
    this.state.parsedGraph = parsed;

    // Create GPU simulation buffers and bind groups
    this.createSimulationResources(parsed);

    // Update layer render contexts with new position buffers
    this.updateLayerRenderContext();

    // Start render loop on first load (delayed from constructor to ensure canvas is sized)
    if (!this.renderLoop.isRunning) {
      this.renderLoop.start();
    }

    // Fit view to content
    this.fitToView();

    // Start simulation automatically
    this.simulationController.restart();

    // Emit load event
    // Update hit tester with new graph data
    this.updateHitTester();

    this.events.emit({
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
  private createSimulationResources(parsed: ParsedGraph): void {
    if (!this.simulationPipeline) return;

    const { device } = this.gpuContext;

    // Create simulation buffers
    this.simBuffers = createSimulationBuffers(
      device,
      parsed.nodeCount,
      parsed.edgeCount,
    );

    // Copy initial positions to simulation buffers
    copyPositionsToSimulation(
      device,
      this.simBuffers,
      parsed.positionsX,
      parsed.positionsY,
    );

    // Copy edge data to simulation buffers
    copyEdgesToSimulation(
      device,
      this.simBuffers,
      parsed.edgeSources,
      parsed.edgeTargets,
    );

    // Initialize uniforms with force config
    updateSimulationUniforms(
      device,
      this.simBuffers,
      parsed.nodeCount,
      parsed.edgeCount,
      1.0, // Initial alpha
      this.forceConfig,
    );

    // Create simulation bind groups
    this.simBindGroups = createSimulationBindGroups(
      device,
      this.simulationPipeline,
      this.simBuffers,
    );

    // Update render bind groups to use simulation position buffers
    if (this.nodePipeline) {
      this.nodeBindGroup = createNodeBindGroup(
        device,
        this.nodePipeline,
        this.simBuffers.positionsX,
        this.simBuffers.positionsY,
        this.buffers!.nodeAttributes,
      );
    }

    if (this.edgePipeline && this.buffers) {
      this.edgeBindGroup = createEdgeBindGroup(
        device,
        this.edgePipeline,
        this.simBuffers.positionsX,
        this.simBuffers.positionsY,
        this.buffers.edgeIndices,
        this.buffers.edgeAttributes,
      );
    }

    // Create algorithm-specific buffers and bind groups
    if (this.currentAlgorithm && this.algorithmPipelines) {
      this.algorithmBuffers = this.currentAlgorithm.createBuffers(
        device,
        parsed.nodeCount,
      );

      const context: AlgorithmRenderContext = {
        device,
        positionsX: this.simBuffers.positionsX,
        positionsY: this.simBuffers.positionsY,
        forcesX: this.simBuffers.forcesX,
        forcesY: this.simBuffers.forcesY,
        nodeCount: parsed.nodeCount,
        forceConfig: this.forceConfig,
      };

      this.algorithmBindGroups = this.currentAlgorithm.createBindGroups(
        device,
        this.algorithmPipelines,
        context,
        this.algorithmBuffers,
      );
    }
  }

  /**
   * Create GPU buffers from parsed graph
   */
  private createBuffers(parsed: ParsedGraph): void {
    const { device } = this.gpuContext;

    // Destroy old buffers
    this.destroyBuffers();

    // Create position buffers
    const positionsX = device.createBuffer({
      label: "Positions X",
      size: parsed.positionsX.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(positionsX, 0, toArrayBuffer(parsed.positionsX));

    const positionsY = device.createBuffer({
      label: "Positions Y",
      size: parsed.positionsY.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(positionsY, 0, toArrayBuffer(parsed.positionsY));

    // Create node attributes buffer
    const nodeAttributes = device.createBuffer({
      label: "Node Attributes",
      size: parsed.nodeAttributes.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(nodeAttributes, 0, toArrayBuffer(parsed.nodeAttributes));

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
      device.queue.writeBuffer(edgeIndices, 0, toArrayBuffer(edgeIndicesData));
    }

    const edgeAttributes = device.createBuffer({
      label: "Edge Attributes",
      size: Math.max(parsed.edgeAttributes.byteLength, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (parsed.edgeAttributes.byteLength > 0) {
      device.queue.writeBuffer(edgeAttributes, 0, toArrayBuffer(parsed.edgeAttributes));
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
    this.viewport.zoom(factor, center?.x, center?.y);
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

    // Use CSS dimensions for scale calculation to match viewport coordinate system
    // (hit testing uses CSS coordinates from getBoundingClientRect)
    const cssWidth = this.canvas.clientWidth || this.canvas.width;
    const cssHeight = this.canvas.clientHeight || this.canvas.height;
    const scale = fitBoundsScale(bounds, cssWidth, cssHeight);
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

  /**
   * Set force configuration parameters.
   * Updates take effect immediately on the running simulation.
   *
   * @param config - Partial force configuration to merge with current config
   */
  setForceConfig(config: Partial<FullForceConfig>): void {
    this.forceConfig = validateForceConfig({
      ...this.forceConfig,
      ...config,
    });

    // Reheat simulation so changes take effect
    const currentAlpha = this.simulationController.state.alpha;
    if (currentAlpha < 0.3) {
      this.simulationController.setAlpha(0.3);
    }
  }

  /**
   * Get current force configuration.
   *
   * @returns A copy of the current force configuration
   */
  getForceConfig(): FullForceConfig {
    return { ...this.forceConfig };
  }

  /**
   * Get information about available force algorithms.
   *
   * @returns Array of available algorithm info
   */
  getAvailableAlgorithms(): Array<{ id: string; name: string; description: string; complexity: string }> {
    const registry = getAlgorithmRegistry();
    return registry.listInfo().map((info) => ({
      id: info.id,
      name: info.name,
      description: info.description,
      complexity: info.complexity,
    }));
  }

  /**
   * Get the current force algorithm type.
   *
   * @returns Current algorithm ID or null if no algorithm set
   */
  getForceAlgorithm(): ForceAlgorithmType | null {
    return this.currentAlgorithm?.info.id ?? null;
  }

  /**
   * Set the force algorithm for repulsion calculations.
   *
   * Available algorithms:
   * - "n2": Simple O(n²) all-pairs repulsion (< 10K nodes)
   * - "barnes-hut": O(n log n) quadtree approximation (5K-100K nodes)
   *
   * @param type - Algorithm type to use
   */
  setForceAlgorithm(type: ForceAlgorithmType): void {
    const registry = getAlgorithmRegistry();
    const algorithm = registry.get(type);

    if (!algorithm) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        `Unknown force algorithm: ${type}. Available: ${registry.listInfo().map((i) => i.id).join(", ")}`,
      );
    }

    // Skip if already using this algorithm
    if (this.currentAlgorithm?.info.id === type) {
      return;
    }

    // Destroy old algorithm resources
    this.algorithmBuffers?.destroy();
    this.algorithmBuffers = null;
    this.algorithmBindGroups = null;

    // Set new algorithm and create pipelines
    this.currentAlgorithm = algorithm;
    this.algorithmPipelines = algorithm.createPipelines(this.gpuContext);

    // Create buffers if graph is loaded
    if (this.state.loaded && this.simBuffers) {
      this.algorithmBuffers = algorithm.createBuffers(
        this.gpuContext.device,
        this.state.nodeCount,
      );

      // Create bind groups
      const context: AlgorithmRenderContext = {
        device: this.gpuContext.device,
        positionsX: this.simBuffers.positionsX,
        positionsY: this.simBuffers.positionsY,
        forcesX: this.simBuffers.forcesX,
        forcesY: this.simBuffers.forcesY,
        nodeCount: this.state.nodeCount,
        forceConfig: this.forceConfig,
      };

      this.algorithmBindGroups = algorithm.createBindGroups(
        this.gpuContext.device,
        this.algorithmPipelines,
        context,
        this.algorithmBuffers,
      );
    }

    // Reheat simulation
    const currentAlpha = this.simulationController.state.alpha;
    if (currentAlpha < 0.5) {
      this.simulationController.setAlpha(0.5);
    }

    if (this.debug) {
      console.log(`Force algorithm switched to: ${algorithm.info.name}`);
    }
  }

  // ==========================================================================
  // Public API - Layers
  // ==========================================================================

  /**
   * Enable the heatmap layer.
   * Creates the layer if it doesn't exist.
   */
  enableHeatmap(config?: HeatmapConfig): void {
    const layerId = "heatmap";

    if (!this.layerManager.hasLayer(layerId)) {
      // Create heatmap layer
      const cssWidth = this.canvas.clientWidth || this.canvas.width;
      const cssHeight = this.canvas.clientHeight || this.canvas.height;

      const heatmapLayer = createHeatmapLayer(
        layerId,
        this.gpuContext,
        cssWidth,
        cssHeight,
        { ...config, enabled: true },
      );

      this.layerManager.addLayer(heatmapLayer);

      // Set render context if graph is loaded
      this.updateLayerRenderContext();
    } else {
      // Enable existing layer
      const layer = this.layerManager.getLayer<HeatmapLayer>(layerId);
      if (layer) {
        layer.enabled = true;
        if (config) {
          layer.setConfig(config);
        }
      }
    }
  }

  /**
   * Disable the heatmap layer.
   */
  disableHeatmap(): void {
    this.layerManager.disableLayer("heatmap");
  }

  /**
   * Check if heatmap is enabled.
   */
  isHeatmapEnabled(): boolean {
    return this.layerManager.isLayerVisible("heatmap");
  }

  /**
   * Configure the heatmap layer.
   */
  setHeatmapConfig(config: Partial<HeatmapConfig>): void {
    const layer = this.layerManager.getLayer<HeatmapLayer>("heatmap");
    if (layer) {
      layer.setConfig(config);
    }
  }

  /**
   * Get heatmap configuration.
   */
  getHeatmapConfig(): HeatmapConfig | null {
    const layer = this.layerManager.getLayer<HeatmapLayer>("heatmap");
    return layer?.getConfig() ?? null;
  }

  /**
   * Set heatmap color scale.
   */
  setHeatmapColorScale(name: ColorScaleName): void {
    const layer = this.layerManager.getLayer<HeatmapLayer>("heatmap");
    if (layer) {
      layer.setColorScale(name);
    }
  }

  /**
   * Set a custom heatmap color scale from color stops.
   *
   * @param stops - Array of color stops, each with position (0-1) and color (RGBA 0-1)
   *
   * @example
   * ```typescript
   * graph.setCustomHeatmapColorScale([
   *   { position: 0, color: [0, 0, 0, 1] },      // Black at low density
   *   { position: 0.5, color: [1, 0.4, 0, 1] },  // Orange at mid density
   *   { position: 1, color: [1, 1, 0, 1] },      // Yellow at high density
   * ]);
   * ```
   */
  setCustomHeatmapColorScale(stops: Array<{ position: number; color: [number, number, number, number] }>): void {
    const layer = this.layerManager.getLayer<HeatmapLayer>("heatmap");
    if (layer) {
      layer.setCustomColorScale(stops);
    }
  }

  /**
   * Set heatmap data source.
   *
   * @param source - 'density' for uniform intensity (all nodes contribute equally),
   *                 or a stream ID to use stream values as per-node intensity
   *
   * @example
   * ```typescript
   * // Use uniform density (default)
   * graph.setHeatmapDataSource('density');
   *
   * // Use error stream values - nodes with more errors contribute more to heatmap
   * graph.setHeatmapDataSource('errors');
   * ```
   */
  setHeatmapDataSource(source: string): void {
    const layer = this.layerManager.getLayer<HeatmapLayer>("heatmap");
    if (layer) {
      layer.setDataSource(source);
      // Rebuild intensity buffer on next render
      this.updateLayerRenderContext();
    }
  }

  /**
   * Get heatmap data source.
   *
   * @returns Current data source ('density' or stream ID)
   */
  getHeatmapDataSource(): string | null {
    const layer = this.layerManager.getLayer<HeatmapLayer>("heatmap");
    return layer?.getDataSource() ?? null;
  }

  /**
   * Get info about all layers.
   */
  getLayers(): LayerInfo[] {
    return this.layerManager.getLayerInfo();
  }

  /**
   * Toggle a layer's visibility.
   */
  toggleLayer(layerId: string): boolean {
    return this.layerManager.toggleLayer(layerId);
  }

  /**
   * Update render context for all layers.
   * Called when graph data changes.
   */
  private updateLayerRenderContext(): void {
    if (!this.simBuffers || !this.state.loaded) return;

    // Update heatmap layer if it exists
    const heatmapLayer = this.layerManager.getLayer<HeatmapLayer>("heatmap");
    if (heatmapLayer) {
      // Check if heatmap is using a stream for per-node intensity
      const dataSource = heatmapLayer.getDataSource();
      let nodeIntensities: GPUBuffer | null = null;

      if (dataSource !== "density" && this.streamManager.hasStream(dataSource)) {
        // Get stream values and create intensity buffer
        const stream = this.streamManager.getStream(dataSource);
        if (stream) {
          // Get normalized values from stream (domain mapped to 0-1)
          const intensities = new Float32Array(this.state.nodeCount);

          // Get domain for normalization
          const colorScale = stream.getColorScale();
          const domain = colorScale.domain;
          const domainMin = domain[0];
          const domainRange = domain[1] - domainMin;

          // Fill intensity array from stream data
          for (let i = 0; i < this.state.nodeCount; i++) {
            const value = stream.getValue(i);
            if (value !== undefined && domainRange > 0) {
              // Normalize to 0-1 range based on domain
              intensities[i] = Math.max(0, Math.min(1, (value - domainMin) / domainRange));
            } else {
              // Nodes without values don't contribute (intensity = 0)
              intensities[i] = 0;
            }
          }

          // Create or update GPU buffer
          const { device } = this.gpuContext;
          const requiredSize = this.state.nodeCount * 4; // 1 f32 per node

          // Recreate buffer if size changed
          if (!this.heatmapIntensityBuffer || this.heatmapIntensityBuffer.size < requiredSize) {
            this.heatmapIntensityBuffer?.destroy();
            this.heatmapIntensityBuffer = device.createBuffer({
              label: "Heatmap Stream Intensity Buffer",
              size: requiredSize,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
          }

          // Write intensity data
          device.queue.writeBuffer(this.heatmapIntensityBuffer, 0, intensities);
          nodeIntensities = this.heatmapIntensityBuffer;
        }
      }

      const heatmapContext: HeatmapRenderContext = {
        viewportUniformBuffer: this.viewportUniformBuffer.buffer,
        positionsX: this.simBuffers.positionsX,
        positionsY: this.simBuffers.positionsY,
        nodeCount: this.state.nodeCount,
        nodeIntensities,
      };

      heatmapLayer.setRenderContext(heatmapContext);
    }

    // Update contour layer if it exists
    // Contour layer uses the density texture from heatmap
    const contourLayer = this.layerManager.getLayer<ContourLayer>("contour");
    if (contourLayer && heatmapLayer) {
      const densityTexture = heatmapLayer.getDensityTexture();
      const heatmapConfig = heatmapLayer.getConfig();
      const contourContext: ContourRenderContext = {
        densityTextureView: densityTexture.sampleView,
        maxDensity: heatmapConfig.maxDensity,
      };
      contourLayer.setRenderContext(contourContext);
    }

    // Update metaball layer if it exists
    const metaballLayer = this.layerManager.getLayer<MetaballLayer>("metaball");
    if (metaballLayer) {
      const viewportState = this.viewport.state;
      const metaballContext: MetaballRenderContext = {
        viewportUniformBuffer: this.viewportUniformBuffer.buffer,
        positionsX: this.simBuffers.positionsX,
        positionsY: this.simBuffers.positionsY,
        nodeCount: this.state.nodeCount,
        viewportOffset: [viewportState.x, viewportState.y],
        viewportScale: viewportState.scale,
      };
      metaballLayer.setRenderContext(metaballContext);
    }

    // Update labels layer if it exists
    const labelsLayer = this.layerManager.getLayer<LabelsLayer>("labels");
    if (labelsLayer && this.state.parsedGraph) {
      const viewportState = this.viewport.state;
      const cssWidth = this.canvas.clientWidth || this.canvas.width;
      const cssHeight = this.canvas.clientHeight || this.canvas.height;

      // Create position provider that reads from parsedGraph arrays
      const positionsX = this.state.parsedGraph.positionsX;
      const positionsY = this.state.parsedGraph.positionsY;
      const positionProvider = {
        getX: (nodeId: number) => positionsX[nodeId] ?? 0,
        getY: (nodeId: number) => positionsY[nodeId] ?? 0,
      };

      const labelsContext: LabelsRenderContext = {
        viewportX: viewportState.x,
        viewportY: viewportState.y,
        scale: viewportState.scale,
        canvasWidth: cssWidth,
        canvasHeight: cssHeight,
        positionProvider,
      };
      labelsLayer.setRenderContext(labelsContext);
    }
  }

  // ==========================================================================
  // Public API - Contour Layer
  // ==========================================================================

  /**
   * Enable the contour layer.
   * Creates the layer if it doesn't exist.
   */
  enableContour(config?: ContourConfig): void {
    const layerId = "contour";

    if (!this.layerManager.hasLayer(layerId)) {
      const contourLayer = createContourLayer(
        layerId,
        this.gpuContext,
        { ...config, enabled: true },
      );

      this.layerManager.addLayer(contourLayer);
      this.updateLayerRenderContext();
    } else {
      const layer = this.layerManager.getLayer<ContourLayer>(layerId);
      if (layer) {
        layer.enabled = true;
        if (config) {
          layer.setConfig(config);
        }
      }
    }
  }

  /**
   * Disable the contour layer.
   */
  disableContour(): void {
    this.layerManager.disableLayer("contour");
  }

  /**
   * Check if contour is enabled.
   */
  isContourEnabled(): boolean {
    return this.layerManager.isLayerVisible("contour");
  }

  /**
   * Configure the contour layer.
   */
  setContourConfig(config: Partial<ContourConfig>): void {
    const layer = this.layerManager.getLayer<ContourLayer>("contour");
    if (layer) {
      layer.setConfig(config);
    }
  }

  /**
   * Get contour configuration.
   */
  getContourConfig(): ContourConfig | null {
    const layer = this.layerManager.getLayer<ContourLayer>("contour");
    return layer?.getConfig() ?? null;
  }

  // ==========================================================================
  // Public API - Metaball Layer
  // ==========================================================================

  /**
   * Enable the metaball layer.
   * Creates the layer if it doesn't exist.
   */
  enableMetaball(config?: MetaballConfig): void {
    const layerId = "metaball";

    if (!this.layerManager.hasLayer(layerId)) {
      const metaballLayer = createMetaballLayer(
        layerId,
        this.gpuContext,
        { ...config, enabled: true },
      );

      this.layerManager.addLayer(metaballLayer);
      this.updateLayerRenderContext();
    } else {
      const layer = this.layerManager.getLayer<MetaballLayer>(layerId);
      if (layer) {
        layer.enabled = true;
        if (config) {
          layer.setConfig(config);
        }
      }
    }
  }

  /**
   * Disable the metaball layer.
   */
  disableMetaball(): void {
    this.layerManager.disableLayer("metaball");
  }

  /**
   * Check if metaball is enabled.
   */
  isMetaballEnabled(): boolean {
    return this.layerManager.isLayerVisible("metaball");
  }

  /**
   * Configure the metaball layer.
   */
  setMetaballConfig(config: Partial<MetaballConfig>): void {
    const layer = this.layerManager.getLayer<MetaballLayer>("metaball");
    if (layer) {
      layer.setConfig(config);
    }
  }

  /**
   * Get metaball configuration.
   */
  getMetaballConfig(): MetaballConfig | null {
    const layer = this.layerManager.getLayer<MetaballLayer>("metaball");
    return layer?.getConfig() ?? null;
  }

  // ==========================================================================
  // Public API - Labels Layer
  // ==========================================================================

  /**
   * Enable the labels layer.
   * Creates the layer if it doesn't exist.
   */
  async enableLabels(config?: Partial<LabelConfig>): Promise<void> {
    const layerId = "labels";

    if (!this.layerManager.hasLayer(layerId)) {
      const labelsLayer = new LabelsLayer(
        layerId,
        this.gpuContext,
        { ...config, visible: true },
      );

      // Initialize the layer (loads font atlas)
      await labelsLayer.initialize();

      this.layerManager.addLayer(labelsLayer);
      this.updateLayerRenderContext();
    } else {
      const layer = this.layerManager.getLayer<LabelsLayer>(layerId);
      if (layer) {
        layer.enabled = true;
        if (config) {
          layer.setConfig(config);
        }
      }
    }
  }

  /**
   * Disable the labels layer.
   */
  disableLabels(): void {
    this.layerManager.disableLayer("labels");
  }

  /**
   * Check if labels are enabled.
   */
  isLabelsEnabled(): boolean {
    return this.layerManager.isLayerVisible("labels");
  }

  /**
   * Configure the labels layer.
   */
  setLabelsConfig(config: Partial<LabelConfig>): void {
    const layer = this.layerManager.getLayer<LabelsLayer>("labels");
    if (layer) {
      layer.setConfig(config);
    }
  }

  /**
   * Get labels configuration.
   */
  getLabelsConfig(): LabelConfig | null {
    const layer = this.layerManager.getLayer<LabelsLayer>("labels");
    return layer?.getConfig() ?? null;
  }

  /**
   * Set labels data for the labels layer.
   * @param labels Array of label data to display
   */
  setLabels(labels: LabelData[]): void {
    const layer = this.layerManager.getLayer<LabelsLayer>("labels");
    if (layer) {
      layer.setLabels(labels);
    }
  }

  // ==========================================================================
  // Public API - Edge Flow Animation
  // ==========================================================================

  /**
   * Enable edge flow animation with a preset.
   * @param preset Preset name: "particles", "waves", "dataStream", "sparks", "warning", "dualLayer", "energy"
   */
  setEdgeFlowPreset(preset: EdgeFlowPreset): void {
    const config = EDGE_FLOW_PRESETS[preset];
    if (!config) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        `Unknown flow preset: ${preset}. Available: ${Object.keys(EDGE_FLOW_PRESETS).join(", ")}`,
      );
    }
    this.flowConfig = config;
  }

  /**
   * Set custom edge flow configuration.
   * @param config Partial configuration to merge with current config
   */
  setEdgeFlowConfig(config: Partial<EdgeFlowConfig>): void {
    this.flowConfig = {
      layer1: config.layer1 ?? this.flowConfig.layer1,
      layer2: config.layer2 ?? this.flowConfig.layer2,
    };
  }

  /**
   * Get current edge flow configuration.
   * @returns A copy of the current flow configuration
   */
  getEdgeFlowConfig(): EdgeFlowConfig {
    return {
      layer1: { ...this.flowConfig.layer1 },
      layer2: { ...this.flowConfig.layer2 },
    };
  }

  /**
   * Disable edge flow animation.
   */
  disableEdgeFlow(): void {
    this.flowConfig = { ...DEFAULT_EDGE_FLOW_CONFIG };
  }

  /**
   * Check if edge flow is enabled.
   */
  isEdgeFlowEnabled(): boolean {
    return this.flowConfig.layer1.enabled || this.flowConfig.layer2.enabled;
  }

  /**
   * Get available flow preset names.
   */
  getFlowPresets(): EdgeFlowPreset[] {
    return Object.keys(EDGE_FLOW_PRESETS) as EdgeFlowPreset[];
  }

  // ==========================================================================
  // Public API - Per-Item Styling
  // ==========================================================================

  /**
   * Set colors for individual nodes.
   *
   * @param colors Float32Array with 4 values (RGBA) per node.
   *               Length must equal nodeCount × 4.
   *               Values should be in range 0-1.
   * @throws HeroineGraphError if array length doesn't match nodeCount × 4
   *
   * @example
   * ```typescript
   * const colors = new Float32Array(nodeCount * 4);
   * for (let i = 0; i < nodeCount; i++) {
   *   colors[i * 4 + 0] = Math.random(); // R
   *   colors[i * 4 + 1] = Math.random(); // G
   *   colors[i * 4 + 2] = Math.random(); // B
   *   colors[i * 4 + 3] = 1.0;           // A
   * }
   * graph.setNodeColors(colors);
   * ```
   */
  setNodeColors(colors: Float32Array): void {
    if (!this.state.loaded || !this.buffers || !this.state.parsedGraph) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Cannot set node colors: graph not loaded",
      );
    }

    const expected = this.state.nodeCount * 4;
    if (colors.length !== expected) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        `Expected ${expected} values for ${this.state.nodeCount} nodes (4 per node), got ${colors.length}`,
      );
    }

    const { device } = this.gpuContext;
    const nodeAttrs = this.state.parsedGraph.nodeAttributes;

    // Update CPU-side array and GPU buffer
    // Node attrs layout: [radius, r, g, b, selected, hovered] per node
    for (let i = 0; i < this.state.nodeCount; i++) {
      const colorBase = i * 4;
      const attrBase = i * 6;

      // Skip NaN values (keep existing color)
      const r = colors[colorBase];
      const g = colors[colorBase + 1];
      const b = colors[colorBase + 2];
      // Alpha (colors[colorBase + 3]) is currently ignored - shader uses RGB only

      if (!Number.isNaN(r)) nodeAttrs[attrBase + 1] = r;
      if (!Number.isNaN(g)) nodeAttrs[attrBase + 2] = g;
      if (!Number.isNaN(b)) nodeAttrs[attrBase + 3] = b;
    }

    // Upload entire buffer to GPU
    device.queue.writeBuffer(
      this.buffers.nodeAttributes,
      0,
      toArrayBuffer(nodeAttrs),
    );
  }

  /**
   * Set sizes (radii) for individual nodes.
   *
   * @param sizes Float32Array with 1 value per node.
   *              Length must equal nodeCount.
   *              Values are in graph units.
   * @throws HeroineGraphError if array length doesn't match nodeCount
   *
   * @example
   * ```typescript
   * const sizes = new Float32Array(nodeCount);
   * for (let i = 0; i < nodeCount; i++) {
   *   sizes[i] = 5 + Math.random() * 10; // Random sizes 5-15
   * }
   * graph.setNodeSizes(sizes);
   * ```
   */
  setNodeSizes(sizes: Float32Array): void {
    if (!this.state.loaded || !this.buffers || !this.state.parsedGraph) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Cannot set node sizes: graph not loaded",
      );
    }

    const expected = this.state.nodeCount;
    if (sizes.length !== expected) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        `Expected ${expected} values for ${this.state.nodeCount} nodes (1 per node), got ${sizes.length}`,
      );
    }

    const { device } = this.gpuContext;
    const nodeAttrs = this.state.parsedGraph.nodeAttributes;

    // Update CPU-side array
    // Node attrs layout: [radius, r, g, b, selected, hovered] per node
    for (let i = 0; i < this.state.nodeCount; i++) {
      const size = sizes[i];
      if (!Number.isNaN(size) && size > 0) {
        nodeAttrs[i * 6] = size; // radius is at offset 0
      }
    }

    // Upload entire buffer to GPU
    device.queue.writeBuffer(
      this.buffers.nodeAttributes,
      0,
      toArrayBuffer(nodeAttrs),
    );
  }

  /**
   * Set colors for individual edges.
   *
   * @param colors Float32Array with 4 values (RGBA) per edge.
   *               Length must equal edgeCount × 4.
   *               Values should be in range 0-1.
   * @throws HeroineGraphError if array length doesn't match edgeCount × 4
   *
   * @example
   * ```typescript
   * const colors = new Float32Array(edgeCount * 4);
   * for (let i = 0; i < edgeCount; i++) {
   *   colors[i * 4 + 0] = 0.5; // R
   *   colors[i * 4 + 1] = 0.5; // G
   *   colors[i * 4 + 2] = 0.5; // B
   *   colors[i * 4 + 3] = 0.6; // A (opacity)
   * }
   * graph.setEdgeColors(colors);
   * ```
   */
  setEdgeColors(colors: Float32Array): void {
    if (!this.state.loaded || !this.buffers || !this.state.parsedGraph) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Cannot set edge colors: graph not loaded",
      );
    }

    const expected = this.state.edgeCount * 4;
    if (colors.length !== expected) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        `Expected ${expected} values for ${this.state.edgeCount} edges (4 per edge), got ${colors.length}`,
      );
    }

    const { device } = this.gpuContext;
    const edgeAttrs = this.state.parsedGraph.edgeAttributes;

    // Update CPU-side array and GPU buffer
    // Edge attrs layout: [width, r, g, b, selected, hovered] per edge
    for (let i = 0; i < this.state.edgeCount; i++) {
      const colorBase = i * 4;
      const attrBase = i * 6;

      const r = colors[colorBase];
      const g = colors[colorBase + 1];
      const b = colors[colorBase + 2];
      // Alpha (colors[colorBase + 3]) is currently ignored - shader uses RGB only

      if (!Number.isNaN(r)) edgeAttrs[attrBase + 1] = r;
      if (!Number.isNaN(g)) edgeAttrs[attrBase + 2] = g;
      if (!Number.isNaN(b)) edgeAttrs[attrBase + 3] = b;
    }

    // Upload entire buffer to GPU
    device.queue.writeBuffer(
      this.buffers.edgeAttributes,
      0,
      toArrayBuffer(edgeAttrs),
    );
  }

  /**
   * Set widths for individual edges.
   *
   * @param widths Float32Array with 1 value per edge.
   *               Length must equal edgeCount.
   *               Values are in pixels.
   * @throws HeroineGraphError if array length doesn't match edgeCount
   *
   * @example
   * ```typescript
   * const widths = new Float32Array(edgeCount);
   * for (let i = 0; i < edgeCount; i++) {
   *   widths[i] = 1 + Math.random() * 3; // Random widths 1-4px
   * }
   * graph.setEdgeWidths(widths);
   * ```
   */
  setEdgeWidths(widths: Float32Array): void {
    if (!this.state.loaded || !this.buffers || !this.state.parsedGraph) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Cannot set edge widths: graph not loaded",
      );
    }

    const expected = this.state.edgeCount;
    if (widths.length !== expected) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        `Expected ${expected} values for ${this.state.edgeCount} edges (1 per edge), got ${widths.length}`,
      );
    }

    const { device } = this.gpuContext;
    const edgeAttrs = this.state.parsedGraph.edgeAttributes;

    // Update CPU-side array
    // Edge attrs layout: [width, r, g, b, selected, hovered] per edge
    for (let i = 0; i < this.state.edgeCount; i++) {
      const width = widths[i];
      if (!Number.isNaN(width) && width > 0) {
        edgeAttrs[i * 6] = width; // width is at offset 0
      }
    }

    // Upload entire buffer to GPU
    device.queue.writeBuffer(
      this.buffers.edgeAttributes,
      0,
      toArrayBuffer(edgeAttrs),
    );
  }

  // ==========================================================================
  // Public API - Value Streams
  // ==========================================================================

  /**
   * Define a new value stream for visualizing numeric data as heat colors.
   *
   * Value streams map numeric values to colors using a configurable color scale.
   * Multiple streams can be active and their colors are blended together.
   *
   * @param config - Stream configuration
   * @returns The created stream's ID
   * @throws Error if stream ID already exists or max streams exceeded
   *
   * @example
   * ```typescript
   * // Define an error stream with red gradient
   * graph.defineValueStream({
   *   id: 'errors',
   *   colorScale: {
   *     domain: [0, 10],
   *     stops: [
   *       { position: 0, color: [0, 0, 0, 0] },        // transparent at 0
   *       { position: 0.5, color: [0.8, 0.2, 0.1, 0.5] }, // semi-transparent red
   *       { position: 1, color: [1, 0, 0, 1] }         // solid red at max
   *     ]
   *   },
   *   blendMode: 'additive'
   * });
   * ```
   */
  defineValueStream(config: ValueStreamConfig): string {
    this.streamManager.defineStream(config);
    return config.id;
  }

  /**
   * Set values for nodes in a stream.
   *
   * @param streamId - Stream ID
   * @param data - Array of node index/value pairs
   *
   * @example
   * ```typescript
   * graph.setStreamValues('errors', [
   *   { nodeIndex: 0, value: 5 },
   *   { nodeIndex: 1, value: 10 },
   *   { nodeIndex: 5, value: 3 }
   * ]);
   * ```
   */
  setStreamValues(streamId: string, data: StreamDataPoint[]): void {
    this.streamManager.setStreamData(streamId, data);
    this.applyStreamColors();
    this.updateHeatmapIfUsingStream(streamId);
  }

  /**
   * Set bulk values for nodes in a stream (more efficient for large updates).
   *
   * @param streamId - Stream ID
   * @param data - Bulk data with indices and values arrays
   *
   * @example
   * ```typescript
   * graph.setStreamBulkValues('activity', {
   *   indices: new Int32Array([0, 1, 2, 3, 4]),
   *   values: new Float32Array([0.5, 0.8, 0.3, 1.0, 0.2])
   * });
   * ```
   */
  setStreamBulkValues(streamId: string, data: StreamBulkData): void {
    this.streamManager.setStreamBulkData(streamId, data);
    this.applyStreamColors();
    this.updateHeatmapIfUsingStream(streamId);
  }

  /**
   * Clear all values from a stream.
   *
   * @param streamId - Stream ID
   */
  clearStreamValues(streamId: string): void {
    this.streamManager.clearStreamData(streamId);
    this.applyStreamColors();
    this.updateHeatmapIfUsingStream(streamId);
  }

  /**
   * Remove a value stream entirely.
   *
   * @param streamId - Stream ID to remove
   * @returns true if stream was found and removed
   */
  removeValueStream(streamId: string): boolean {
    const removed = this.streamManager.removeStream(streamId);
    if (removed) {
      this.applyStreamColors();
    }
    return removed;
  }

  /**
   * Enable a value stream.
   *
   * @param streamId - Stream ID
   */
  enableValueStream(streamId: string): void {
    this.streamManager.enableStream(streamId);
    this.applyStreamColors();
  }

  /**
   * Disable a value stream (keeps data, just hides visualization).
   *
   * @param streamId - Stream ID
   */
  disableValueStream(streamId: string): void {
    this.streamManager.disableStream(streamId);
    this.applyStreamColors();
  }

  /**
   * Toggle a value stream's enabled state.
   *
   * @param streamId - Stream ID
   * @returns New enabled state
   */
  toggleValueStream(streamId: string): boolean {
    const result = this.streamManager.toggleStream(streamId);
    this.applyStreamColors();
    return result;
  }

  /**
   * Get info about all defined value streams.
   *
   * @returns Array of stream information
   */
  getValueStreams(): StreamInfo[] {
    return this.streamManager.getStreamInfo();
  }

  /**
   * Check if a value stream exists.
   *
   * @param streamId - Stream ID
   */
  hasValueStream(streamId: string): boolean {
    return this.streamManager.hasStream(streamId);
  }

  /**
   * Set opacity for a value stream.
   *
   * @param streamId - Stream ID
   * @param opacity - Opacity value (0-1)
   */
  setStreamOpacity(streamId: string, opacity: number): void {
    const stream = this.streamManager.getStream(streamId);
    if (stream) {
      stream.setOpacity(opacity);
      this.streamManager.invalidateCache();
      this.applyStreamColors();
    }
  }

  /**
   * Set blend mode for a value stream.
   *
   * @param streamId - Stream ID
   * @param blendMode - Blend mode ('additive', 'multiply', 'max', 'replace')
   */
  setStreamBlendMode(streamId: string, blendMode: "additive" | "multiply" | "max" | "replace"): void {
    const stream = this.streamManager.getStream(streamId);
    if (stream) {
      stream.setBlendMode(blendMode);
      this.streamManager.invalidateCache();
      this.applyStreamColors();
    }
  }

  /**
   * Clear all value streams.
   */
  clearAllValueStreams(): void {
    this.streamManager.clear();
    this.applyStreamColors();
  }

  /**
   * Apply computed stream colors to nodes.
   * Called internally after stream data changes.
   */
  private applyStreamColors(): void {
    if (!this.state.loaded || this.state.nodeCount === 0) return;

    // Get blended colors from all active streams
    const colors = this.streamManager.computeBlendedColors(this.state.nodeCount);

    // Only apply if there are actual colors (any non-zero alpha)
    let hasColors = false;
    for (let i = 3; i < colors.length; i += 4) {
      if (colors[i] > 0) {
        hasColors = true;
        break;
      }
    }

    if (hasColors) {
      this.setNodeColors(colors);
    }
  }

  /**
   * Update heatmap render context if it's using the specified stream.
   * Called internally when stream data changes to ensure heatmap reflects updates.
   */
  private updateHeatmapIfUsingStream(streamId: string): void {
    const heatmapLayer = this.layerManager.getLayer<HeatmapLayer>("heatmap");
    if (heatmapLayer && heatmapLayer.getDataSource() === streamId) {
      this.updateLayerRenderContext();
    }
  }

  // ==========================================================================
  // Public API - Events
  // ==========================================================================

  /**
   * Subscribe to an event
   */
  on<K extends keyof EventMap>(event: K, handler: EventHandler<EventMap[K]>): void {
    this.events.on(event, handler);
  }

  /**
   * Unsubscribe from an event
   */
  off<K extends keyof EventMap>(event: K, handler: EventHandler<EventMap[K]>): void {
    this.events.off(event, handler);
  }

  /**
   * Subscribe to an event once
   */
  once<K extends keyof EventMap>(event: K, handler: EventHandler<EventMap[K]>): void {
    this.events.once(event, handler);
  }

  // ==========================================================================
  // Public API - Interaction
  // ==========================================================================

  /**
   * Get node at screen position.
   * @param screenX X position in screen/canvas coordinates
   * @param screenY Y position in screen/canvas coordinates
   * @returns Node ID or null if no node at position
   */
  getNodeAtPosition(screenX: number, screenY: number): NodeId | null {
    const graphPos = this.viewport.screenToGraph(screenX, screenY);
    const hitRadius = 20 / this.viewport.state.scale; // Adjust for zoom

    const result = this.hitTester.hitTestNode(graphPos.x, graphPos.y, hitRadius);

    return result?.nodeId ?? null;
  }

  /**
   * Get edge at screen position.
   * @param screenX X position in screen/canvas coordinates
   * @param screenY Y position in screen/canvas coordinates
   * @returns Edge ID or null if no edge at position
   */
  getEdgeAtPosition(screenX: number, screenY: number): EdgeId | null {
    const graphPos = this.viewport.screenToGraph(screenX, screenY);
    const hitRadius = 5 / this.viewport.state.scale; // Adjust for zoom
    const result = this.hitTester.hitTestEdge(graphPos.x, graphPos.y, hitRadius);
    return result?.edgeId ?? null;
  }

  /**
   * Select nodes by ID.
   * @param nodeIds Node IDs to select (replaces current selection)
   */
  selectNodes(nodeIds: NodeId[]): void {
    const previousSelection = new Set(this.selectedNodes);
    this.selectedNodes.clear();
    for (const id of nodeIds) {
      this.selectedNodes.add(id);
    }
    this.emitSelectionChange("node", previousSelection, this.selectedNodes);
  }

  /**
   * Select edges by ID.
   * @param edgeIds Edge IDs to select (replaces current selection)
   */
  selectEdges(edgeIds: EdgeId[]): void {
    const previousSelection = new Set(this.selectedEdges);
    this.selectedEdges.clear();
    for (const id of edgeIds) {
      this.selectedEdges.add(id);
    }
    this.emitSelectionChange("edge", previousSelection, this.selectedEdges);
  }

  /**
   * Add nodes to selection.
   * @param nodeIds Node IDs to add
   */
  addToSelection(nodeIds: NodeId[]): void {
    const previousSelection = new Set(this.selectedNodes);
    for (const id of nodeIds) {
      this.selectedNodes.add(id);
    }
    this.emitSelectionChange("node", previousSelection, this.selectedNodes);
  }

  /**
   * Remove nodes from selection.
   * @param nodeIds Node IDs to remove
   */
  removeFromSelection(nodeIds: NodeId[]): void {
    const previousSelection = new Set(this.selectedNodes);
    for (const id of nodeIds) {
      this.selectedNodes.delete(id);
    }
    this.emitSelectionChange("node", previousSelection, this.selectedNodes);
  }

  /**
   * Clear all selection.
   */
  clearSelection(): void {
    const previousNodeSelection = new Set(this.selectedNodes);
    const previousEdgeSelection = new Set(this.selectedEdges);
    this.selectedNodes.clear();
    this.selectedEdges.clear();
    if (previousNodeSelection.size > 0) {
      this.emitSelectionChange("node", previousNodeSelection, this.selectedNodes);
    }
    if (previousEdgeSelection.size > 0) {
      this.emitSelectionChange("edge", previousEdgeSelection, this.selectedEdges);
    }
  }

  /**
   * Get selected node IDs.
   */
  getSelectedNodes(): NodeId[] {
    return Array.from(this.selectedNodes);
  }

  /**
   * Get selected edge IDs.
   */
  getSelectedEdges(): EdgeId[] {
    return Array.from(this.selectedEdges);
  }

  /**
   * Check if a node is selected.
   */
  isNodeSelected(nodeId: NodeId): boolean {
    return this.selectedNodes.has(nodeId);
  }

  /**
   * Check if an edge is selected.
   */
  isEdgeSelected(edgeId: EdgeId): boolean {
    return this.selectedEdges.has(edgeId);
  }

  /**
   * Pin a node (exclude from simulation, fixed position).
   * @param nodeId Node ID to pin
   */
  pinNode(nodeId: NodeId): void {
    this.pinnedNodes.add(nodeId);
    this.events.emit({
      type: "node:pin",
      timestamp: Date.now(),
      nodeId,
    });
  }

  /**
   * Unpin a node (include in simulation).
   * @param nodeId Node ID to unpin
   */
  unpinNode(nodeId: NodeId): void {
    this.pinnedNodes.delete(nodeId);
    this.events.emit({
      type: "node:unpin",
      timestamp: Date.now(),
      nodeId,
    });
  }

  /**
   * Check if a node is pinned.
   */
  isNodePinned(nodeId: NodeId): boolean {
    return this.pinnedNodes.has(nodeId);
  }

  /**
   * Get all pinned node IDs.
   */
  getPinnedNodes(): NodeId[] {
    return Array.from(this.pinnedNodes);
  }

  /**
   * Set node position (also pins the node).
   * @param nodeId Node ID
   * @param x X position in graph coordinates
   * @param y Y position in graph coordinates
   */
  setNodePosition(nodeId: NodeId, x: number, y: number): void {
    if (!this.state.parsedGraph) return;

    // nodeId is the array index in our system
    const idx = nodeId;
    if (idx < 0 || idx >= this.state.parsedGraph.nodeCount) return;

    // Update local position data
    this.state.parsedGraph.positionsX[idx] = x;
    this.state.parsedGraph.positionsY[idx] = y;

    // Pin the node
    this.pinnedNodes.add(nodeId);

    // Update GPU buffer
    this.syncPositionToGPU(nodeId, x, y);

    // Disturb the simulation - boost alpha so neighbors react to the moved node.
    // This ensures the simulation is responsive when nodes are dragged.
    const currentAlpha = this.simulationController.state.alpha;
    if (currentAlpha < 0.3) {
      this.simulationController.setAlpha(0.3);
    }
  }

  /**
   * Get node position.
   * @param nodeId Node ID
   * @returns Position or undefined if node not found
   */
  getNodePosition(nodeId: NodeId): Vec2 | undefined {
    if (!this.state.parsedGraph) return undefined;

    // nodeId is the array index in our system
    const idx = nodeId;
    if (idx < 0 || idx >= this.state.parsedGraph.nodeCount) return undefined;

    return {
      x: this.state.parsedGraph.positionsX[idx],
      y: this.state.parsedGraph.positionsY[idx],
    };
  }

  /**
   * Get currently hovered node.
   */
  getHoveredNode(): NodeId | null {
    return this.hoveredNode;
  }

  /**
   * Get currently hovered edge.
   */
  getHoveredEdge(): EdgeId | null {
    return this.hoveredEdge;
  }

  // ==========================================================================
  // Private - Interaction Helpers
  // ==========================================================================

  /**
   * Setup interaction event handlers
   */
  private setupInteractionHandlers(): void {
    if (!this.pointerManager) return;

    // Handle pointer down (start drag or select)
    this.pointerManager.on("pointerdown", (e) => {
      if (e.button !== 0) return; // Only left click

      const nodeId = this.getNodeAtPosition(e.screenPosition.x, e.screenPosition.y);

      if (nodeId !== null) {
        // Start drag on node
        this.draggedNode = nodeId;
        this.lastDragPosition = { ...e.graphPosition };
        this.pinnedNodes.add(nodeId);

        // Select if not already selected (or add to selection with shift)
        if (!e.modifiers.shift && !this.selectedNodes.has(nodeId)) {
          this.selectNodes([nodeId]);
        } else if (e.modifiers.shift) {
          this.addToSelection([nodeId]);
        }

        this.events.emit({
          type: "node:dragstart",
          timestamp: Date.now(),
          nodeId,
          position: e.graphPosition,
        });
      } else {
        // No node hit - start panning (allow panning even when clicking on/near edges)
        // Check for edge click to select it, but still allow panning
        const edgeId = this.getEdgeAtPosition(e.screenPosition.x, e.screenPosition.y);
        if (edgeId !== null) {
          if (!e.modifiers.shift) {
            this.clearSelection();
          }
          this.selectEdges([edgeId]);
        } else if (!e.modifiers.shift) {
          // Empty space - clear selection
          this.clearSelection();
        }

        // Start panning regardless of edge hit
        this.isPanning = true;
        this.lastPanPosition = { ...e.screenPosition };
      }
    });

    // Handle pointer move (drag, pan, or hover)
    this.pointerManager.on("pointermove", (e) => {
      if (this.draggedNode !== null) {
        // Calculate delta from last position
        const delta: Vec2 = this.lastDragPosition
          ? {
            x: e.graphPosition.x - this.lastDragPosition.x,
            y: e.graphPosition.y - this.lastDragPosition.y,
          }
          : { x: 0, y: 0 };

        // Update last position
        this.lastDragPosition = { ...e.graphPosition };

        // Update dragged node position
        this.setNodePosition(this.draggedNode, e.graphPosition.x, e.graphPosition.y);

        this.events.emit({
          type: "node:dragmove",
          timestamp: Date.now(),
          nodeId: this.draggedNode,
          position: e.graphPosition,
          delta,
        });
      } else if (this.isPanning && this.lastPanPosition) {
        // Pan the viewport (inverted - like pushing a piece of paper)
        const dx = this.lastPanPosition.x - e.screenPosition.x;
        const dy = this.lastPanPosition.y - e.screenPosition.y;
        this.viewport.pan(dx, dy);
        this.lastPanPosition = { ...e.screenPosition };
      } else {
        // Hover detection
        this.updateHover(e.screenPosition.x, e.screenPosition.y);
      }
    });

    // Handle pointer up (end drag or pan)
    this.pointerManager.on("pointerup", (e) => {
      if (this.draggedNode !== null) {
        const nodeId = this.draggedNode;
        this.draggedNode = null;
        this.lastDragPosition = null;

        // Optionally unpin after drag (could be configurable)
        // this.pinnedNodes.delete(nodeId);

        this.events.emit({
          type: "node:dragend",
          timestamp: Date.now(),
          nodeId,
          position: e.graphPosition,
        });
      }

      // End panning
      if (this.isPanning) {
        this.isPanning = false;
        this.lastPanPosition = null;
      }
    });

    // Handle wheel (zoom) - use gradual zoom based on delta magnitude
    this.pointerManager.on("wheel", (e) => {
      if (e.wheelDelta) {
        // Normalize wheel delta and apply gradual zoom
        // deltaY is typically ~100 for one scroll tick
        const normalizedDelta = Math.sign(e.wheelDelta.y) *
          Math.min(Math.abs(e.wheelDelta.y), 100) / 100;
        const zoomFactor = 1 - normalizedDelta * 0.05; // 5% per scroll tick
        this.viewport.zoom(zoomFactor, e.screenPosition.x, e.screenPosition.y);
      }
    });
  }

  /**
   * Update hover state and sync to GPU
   */
  private updateHover(screenX: number, screenY: number): void {
    const nodeId = this.getNodeAtPosition(screenX, screenY);
    const position = this.viewport.screenToGraph(screenX, screenY);

    if (nodeId !== this.hoveredNode) {
      // Update previous hovered node
      if (this.hoveredNode !== null) {
        this.syncNodeHoverToGPU(this.hoveredNode, false);
        this.events.emit({
          type: "node:hoverleave",
          timestamp: Date.now(),
          nodeId: this.hoveredNode,
        });
      }

      this.hoveredNode = nodeId;

      // Update new hovered node
      if (nodeId !== null) {
        this.syncNodeHoverToGPU(nodeId, true);
        this.events.emit({
          type: "node:hoverenter",
          timestamp: Date.now(),
          nodeId,
          position,
        });
      }
    }

    // Only check edge hover if not hovering a node
    if (nodeId === null) {
      const edgeId = this.getEdgeAtPosition(screenX, screenY);

      if (edgeId !== this.hoveredEdge) {
        if (this.hoveredEdge !== null) {
          this.events.emit({
            type: "edge:hoverleave",
            timestamp: Date.now(),
            edgeId: this.hoveredEdge,
          });
        }

        this.hoveredEdge = edgeId;

        if (edgeId !== null) {
          this.events.emit({
            type: "edge:hoverenter",
            timestamp: Date.now(),
            edgeId,
            position,
          });
        }
      }
    } else {
      // Clear edge hover when hovering a node
      if (this.hoveredEdge !== null) {
        this.events.emit({
          type: "edge:hoverleave",
          timestamp: Date.now(),
          edgeId: this.hoveredEdge,
        });
        this.hoveredEdge = null;
      }
    }
  }

  /**
   * Emit selection change event and update GPU buffer
   */
  private emitSelectionChange(
    type: "node" | "edge",
    previous: Set<number>,
    current: Set<number>,
  ): void {
    const added = [...current].filter((id) => !previous.has(id));
    const removed = [...previous].filter((id) => !current.has(id));

    if (added.length > 0 || removed.length > 0) {
      // Update GPU selection state for nodes
      if (type === "node") {
        for (const nodeId of added) {
          this.syncNodeSelectionToGPU(nodeId, true);
        }
        for (const nodeId of removed) {
          this.syncNodeSelectionToGPU(nodeId, false);
        }
      }

      this.events.emit({
        type: "selection:change",
        timestamp: Date.now(),
        selectedNodes: [...this.selectedNodes],
        selectedEdges: [...this.selectedEdges],
      });
    }
  }

  /**
   * Sync a node's position to GPU buffer
   */
  private syncPositionToGPU(nodeId: NodeId, x: number, y: number): void {
    if (!this.buffers || !this.state.parsedGraph) return;

    // nodeId is the array index in our system
    const idx = nodeId;
    if (idx < 0 || idx >= this.state.parsedGraph.nodeCount) return;

    const { device } = this.gpuContext;
    const posX = new Float32Array([x]);
    const posY = new Float32Array([y]);

    device.queue.writeBuffer(this.buffers.positionsX, idx * 4, posX);
    device.queue.writeBuffer(this.buffers.positionsY, idx * 4, posY);

    // Also update simulation buffers if they exist
    if (this.simBuffers) {
      device.queue.writeBuffer(this.simBuffers.positionsX, idx * 4, posX);
      device.queue.writeBuffer(this.simBuffers.positionsY, idx * 4, posY);
      device.queue.writeBuffer(this.simBuffers.positionsXOut, idx * 4, posX);
      device.queue.writeBuffer(this.simBuffers.positionsYOut, idx * 4, posY);
    }
  }

  /**
   * Update node selection state in GPU buffer
   * Node attributes: [radius, r, g, b, selected, hovered] (6 floats per node)
   */
  private syncNodeSelectionToGPU(nodeId: NodeId, selected: boolean): void {
    if (!this.buffers || !this.state.parsedGraph) return;

    // nodeId is the array index in our system
    const idx = nodeId;
    if (idx < 0 || idx >= this.state.parsedGraph.nodeCount) return;

    const { device } = this.gpuContext;
    // Node attributes are 6 floats per node, selection is at offset 4
    const attrOffset = idx * 6 * 4 + 4 * 4; // 6 floats * 4 bytes, offset 4
    const selectionValue = new Float32Array([selected ? 1.0 : 0.0]);
    device.queue.writeBuffer(this.buffers.nodeAttributes, attrOffset, selectionValue);

    // Also update local parsed graph data
    this.state.parsedGraph.nodeAttributes[idx * 6 + 4] = selected ? 1.0 : 0.0;
  }

  /**
   * Update node hover state in GPU buffer
   * Node attributes: [radius, r, g, b, selected, hovered] (6 floats per node)
   */
  private syncNodeHoverToGPU(nodeId: NodeId, hovered: boolean): void {
    if (!this.buffers || !this.state.parsedGraph) return;

    // nodeId is the array index in our system
    const idx = nodeId;
    if (idx < 0 || idx >= this.state.parsedGraph.nodeCount) return;

    const { device } = this.gpuContext;
    // Node attributes are 6 floats per node, hover is at offset 5
    const attrOffset = idx * 6 * 4 + 5 * 4; // 6 floats * 4 bytes, offset 5
    const hoverValue = new Float32Array([hovered ? 1.0 : 0.0]);
    device.queue.writeBuffer(this.buffers.nodeAttributes, attrOffset, hoverValue);

    // Also update local parsed graph data
    this.state.parsedGraph.nodeAttributes[idx * 6 + 5] = hovered ? 1.0 : 0.0;
  }

  /**
   * Update hit tester with current position data
   */
  private updateHitTester(): void {
    if (!this.state.parsedGraph) return;

    const parsedGraph = this.state.parsedGraph;

    // Set position provider (uses node indices directly - WASM and edges use indices)
    this.hitTester.setPositionProvider({
      getNodePosition: (nodeId: NodeId): Vec2 | undefined => {
        // nodeId is the array index in our system
        if (nodeId < 0 || nodeId >= parsedGraph.nodeCount) return undefined;
        return {
          x: parsedGraph.positionsX[nodeId],
          y: parsedGraph.positionsY[nodeId],
        };
      },
      getNodeRadius: (nodeId: NodeId): number | undefined => {
        // nodeId is the array index in our system
        // nodeAttributes layout: [radius, r, g, b, selected, hovered] per node
        if (nodeId < 0 || nodeId >= parsedGraph.nodeCount) return undefined;
        return parsedGraph.nodeAttributes[nodeId * 6]; // radius is at offset 0
      },
      getNodeIds: function* () {
        for (let i = 0; i < parsedGraph.nodeCount; i++) {
          yield i;
        }
      },
      getNodeCount: () => parsedGraph.nodeCount,
    });

    // Set edge provider
    this.hitTester.setEdgeProvider({
      getEdges: function* () {
        let edgeId = 0;
        for (let i = 0; i < parsedGraph.edgeSources.length; i++) {
          yield [edgeId++, parsedGraph.edgeSources[i], parsedGraph.edgeTargets[i]];
        }
      },
      getEdgeCount: () => parsedGraph.edgeSources.length,
    });

    // Rebuild WASM spatial index with new graph data
    if (this.wasmEngine) {
      this.wasmEngine.rebuildSpatialIndex();
    }
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
    // Update viewport with CSS dimensions for coordinate transforms
    const cssWidth = this.canvas.clientWidth || this.canvas.width;
    const cssHeight = this.canvas.clientHeight || this.canvas.height;
    this.viewport.resize(cssWidth, cssHeight);
    this.updateViewportUniforms();

    // Resize layers
    this.layerManager.resize(cssWidth, cssHeight);

    // Update layer render contexts after resize (texture views may have changed)
    this.updateLayerRenderContext();
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

    // Dispose pointer manager
    this.pointerManager?.dispose();

    // Dispose viewport
    this.viewport.dispose();

    // Destroy render buffers
    this.destroyBuffers();

    // Destroy simulation buffers
    this.destroySimulationBuffers();

    // Destroy layer manager and all layers
    this.layerManager.destroy();

    // Destroy stream manager
    this.streamManager.destroy();

    // Destroy heatmap intensity buffer
    this.heatmapIntensityBuffer?.destroy();
    this.heatmapIntensityBuffer = null;

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

    // Destroy algorithm-specific buffers
    this.algorithmBuffers?.destroy();
    this.algorithmBuffers = null;
    this.algorithmBindGroups = null;
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
