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
  EdgeInput,
  EventHandler,
  EventMap,
  GraphConfig,
  GraphInput,
  GraphTypedInput,
  NodeId,
  NodeInput,
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
  createRenderConfigBindGroup,
  createViewportBindGroup,
  type NodeRenderPipeline,
  renderNodes,
} from "../renderer/pipelines/nodes.ts";
import {
  createEdgeBindGroup,
  createEdgeRenderPipeline,
  type CurvedEdgeConfig,
  DEFAULT_CURVED_EDGE_CONFIG,
  type EdgeRenderPipeline,
  renderEdges,
  updateCurveConfig,
  updateEdgeFlowUniforms,
} from "../renderer/pipelines/edges.ts";
import {
  DEFAULT_EDGE_FLOW_CONFIG,
  type EdgeFlowPreset,
  EDGE_FLOW_PRESETS,
} from "../renderer/edge_flow.ts";
import { parseColorToRGB } from "../utils/color.ts";
import {
  DEFAULT_NODE_BORDER_CONFIG as _DEFAULT_NODE_BORDER_CONFIG,
  type NodeBorderConfig as _NodeBorderConfig,
} from "../config/node_border.ts";
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
  type CollisionBindGroup,
  type CollisionBuffers,
  type CollisionPipeline,
  type GridCollisionPipeline,
  type GridCollisionBuffers,
  type GridCollisionBindGroups,
  createCollisionBindGroup,
  createCollisionBuffers,
  createCollisionPipeline,
  createGridCollisionPipeline,
  createGridCollisionBuffers,
  createGridCollisionBindGroups,
  destroyCollisionBuffers,
  destroyGridCollisionBuffers,
  recordCollisionPass,
  recordGridCollisionPass,
  updateCollisionUniforms,
  updateGridCollisionUniforms,
  uploadNodeSizes,
} from "../simulation/collision.ts";
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
  RelativityAtlasAlgorithm,
  uploadRelativityAtlasEdges,
} from "../simulation/algorithms/mod.ts";
import {
  createStreamManager,
  type StreamBulkData,
  type StreamDataPoint,
  type StreamInfo,
  type StreamManager,
  type ValueStreamConfig,
} from "../streams/mod.ts";
import {
  createTypeStyleManager,
  type EdgeTypeStyleMap,
  type NodeTypeStyleMap,
  type TypeStyleManager,
} from "../styling/mod.ts";
import { initialCapacity, growCapacity } from "./buffer_capacity.ts";
import { MutableGraphState } from "./graph_state.ts";

/**
 * WASM engine interface for spatial queries and graph structure.
 * This matches the HeroineGraphWasm API exposed by the WASM module.
 */
interface WasmEngine extends SpatialQueryEngine {
  findNearestNode(x: number, y: number): number | undefined;
  /** Clear all graph data */
  clear(): void;
  /** Add a single node at position */
  addNode(x: number, y: number): number;
  /** Add multiple nodes from interleaved positions [x0, y0, x1, y1, ...] */
  addNodesFromPositions(positions: Float32Array): number;
  /** Add an edge between two nodes */
  addEdge(source: number, target: number, weight: number): number | undefined;
  /** Add edges from interleaved pairs [src0, tgt0, src1, tgt1, ...] */
  addEdgesFromPairs(edges: Uint32Array): number;
  /** Remove a node by slot index */
  removeNode(id: number): boolean;
  /** Remove an edge by ID */
  removeEdge(id: number): boolean;
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

// Re-export NodeBorderConfig types for backwards compatibility
export { DEFAULT_NODE_BORDER_CONFIG, type NodeBorderConfig } from "../config/node_border.ts";

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
  positions: GPUBuffer;
  nodeAttributes: GPUBuffer;
  edgeIndices: GPUBuffer;
  edgeAttributes: GPUBuffer;
  viewportUniforms: GPUBuffer;
  /** Allocated node capacity (may be > nodeCount for incremental mutations) */
  nodeCapacity: number;
  /** Allocated edge capacity (may be > edgeCount for incremental mutations) */
  edgeCapacity: number;
}

/**
 * Compute bounding box from position arrays.
 *
 * Returns bounds with a margin to account for position changes between frames.
 * The margin is proportional to the graph extent to handle graphs of any scale.
 *
 * @param positionsX - X coordinates array
 * @param positionsY - Y coordinates array
 * @param nodeCount - Number of valid positions to consider
 * @returns Bounding box with margin, or undefined if no valid positions
 */
function computeBoundsFromPositions(
  positionsX: Float32Array,
  positionsY: Float32Array,
  nodeCount: number,
): BoundingBox | undefined {
  if (nodeCount === 0) {
    return undefined;
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (let i = 0; i < nodeCount; i++) {
    const x = positionsX[i];
    const y = positionsY[i];

    // Skip invalid positions (NaN or Infinity)
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      continue;
    }

    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }

  // No valid positions found
  if (minX === Infinity) {
    return undefined;
  }

  // Add margin proportional to graph extent to handle position drift between frames.
  // Using 10% margin ensures nodes moving during simulation don't fall outside bounds.
  // Minimum margin of 100 units handles small/clustered graphs.
  const extentX = maxX - minX;
  const extentY = maxY - minY;
  const margin = Math.max(100, Math.max(extentX, extentY) * 0.1);

  return {
    minX: minX - margin,
    minY: minY - margin,
    maxX: maxX + margin,
    maxY: maxY + margin,
  };
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
  private graphState: MutableGraphState | null = null;
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
  private renderConfigBindGroup: GPUBindGroup | null = null;
  private renderConfigBuffer: GPUBuffer | null = null;

  // Node border configuration
  private nodeBorderConfig: _NodeBorderConfig = { ..._DEFAULT_NODE_BORDER_CONFIG };

  // Background color (RGBA 0-1)
  private backgroundColor: { r: number; g: number; b: number; a: number } = { r: 0.04, g: 0.04, b: 0.06, a: 1.0 };

  // GPU Simulation resources
  private simBuffers: SimulationBuffers | null = null;
  private simBindGroups: SimulationBindGroups | null = null;

  // Force algorithm resources
  private currentAlgorithm: ForceAlgorithm | null = null;
  private algorithmPipelines: AlgorithmPipelines | null = null;
  private algorithmBuffers: AlgorithmBuffers | null = null;
  private algorithmBindGroups: AlgorithmBindGroups | null = null;

  // Collision detection resources
  private collisionPipeline: CollisionPipeline | null = null;
  private collisionBuffers: CollisionBuffers | null = null;
  private collisionBindGroup: CollisionBindGroup | null = null;

  // Grid collision resources (O(n·k) spatial hash for large graphs)
  private gridCollisionPipeline: GridCollisionPipeline | null = null;
  private gridCollisionBuffers: GridCollisionBuffers | null = null;
  private gridCollisionBindGroups: GridCollisionBindGroups | null = null;
  private maxNodeRadius: number = 5.0;
  private frameBounds: BoundingBox | undefined;

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

  // Type-based styling system
  private typeStyleManager: TypeStyleManager;

  // Heatmap stream intensity buffer (per-node values from stream)
  private heatmapIntensityBuffer: GPUBuffer | null = null;

  // Metaball stream intensity buffer (per-node values from stream)
  private metaballIntensityBuffer: GPUBuffer | null = null;

  // Default intensity buffer (all 1.0 values for density mode)
  private defaultIntensityBuffer: GPUBuffer | null = null;

  // Visibility change handling - pause simulation when tab hidden
  private visibilityChangeHandler: (() => void) | null = null;
  private wasRunningBeforeHidden: boolean = false;

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

    // WASM spatial engine is populated during load() with graph data.
    // Enables O(log n) spatial queries for hit testing.

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

    // Initialize type-based styling manager
    this.typeStyleManager = createTypeStyleManager();

    // Note: render loop starts on first load() call, not here
    // This prevents rendering before canvas has valid dimensions

    // Set up visibility change handling - pause simulation when tab is hidden
    this.setupVisibilityChangeHandler();

    if (this.debug) {
      console.log("HeroineGraph instance created");
    }
  }

  /**
   * Set up visibility change handling to pause simulation when tab is hidden.
   * This saves resources when the user switches tabs.
   */
  private setupVisibilityChangeHandler(): void {
    if (typeof document === "undefined") return;

    this.visibilityChangeHandler = () => {
      if (document.hidden) {
        // Tab became hidden - pause simulation if running
        if (this.simulationController.state.status === "running") {
          this.wasRunningBeforeHidden = true;
          this.simulationController.pause();
          if (this.debug) {
            console.log("Tab hidden - pausing simulation");
          }
        }
      } else {
        // Tab became visible - resume simulation if it was running before
        if (this.wasRunningBeforeHidden) {
          this.wasRunningBeforeHidden = false;
          this.simulationController.start();
          if (this.debug) {
            console.log("Tab visible - resuming simulation");
          }
        }
      }
    };

    document.addEventListener("visibilitychange", this.visibilityChangeHandler);
  }

  /**
   * Initialize render pipelines
   */
  private initializePipelines(): void {
    const format = this.gpuContext.format;
    const { device } = this.gpuContext;

    this.nodePipeline = createNodeRenderPipeline(this.gpuContext, { format });
    this.edgePipeline = createEdgeRenderPipeline(this.gpuContext, { format });
    this.simulationPipeline = createSimulationPipeline(this.gpuContext);
    this.collisionPipeline = createCollisionPipeline(this.gpuContext);
    this.gridCollisionPipeline = createGridCollisionPipeline(this.gpuContext);

    // Create render config buffer
    // Struct layout (must match node.frag.wgsl RenderConfig):
    // - selection_color: vec3<f32> (12 bytes) + selection_ring_width: f32 (4 bytes) = 16 bytes
    // - hover_brightness: f32 (4 bytes) + border_enabled: u32 (4 bytes) + border_width: f32 (4 bytes) + pad: f32 (4 bytes) = 16 bytes
    // - border_color: vec3<f32> (12 bytes) + pad: f32 (4 bytes) = 16 bytes
    // Total: 48 bytes
    this.renderConfigBuffer = device.createBuffer({
      label: "Render Config Uniform Buffer",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Initialize with default values
    this.updateRenderConfigBuffer();

    // Create render config bind group
    this.renderConfigBindGroup = createRenderConfigBindGroup(
      device,
      this.nodePipeline,
      this.renderConfigBuffer,
    );
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

    // Compute DPR from canvas buffer vs CSS dimensions.
    // This is more reliable than globalThis.devicePixelRatio because it reflects
    // the actual ratio between the GPU texture and the CSS layout, which may differ
    // if the caller set non-standard canvas dimensions.
    const dpr = cssWidth > 0 ? this.canvas.width / cssWidth : (globalThis.devicePixelRatio || 1);

    this.viewportUniformBuffer.update(
      this.gpuContext.device,
      state,
      cssWidth,
      cssHeight,
      dpr,
    );
  }

  /**
   * Update render config uniform buffer with current node border settings.
   * Called when border configuration changes.
   */
  private updateRenderConfigBuffer(): void {
    if (!this.renderConfigBuffer) return;

    const { device } = this.gpuContext;
    const data = new ArrayBuffer(48);
    const floatView = new Float32Array(data);
    const uintView = new Uint32Array(data);

    // Parse border color from hex/CSS string
    const borderColor = this.parseColorString(this.nodeBorderConfig.color);

    // Layout matches RenderConfig struct in node.frag.wgsl:
    // vec3 selection_color (0-2), f32 selection_ring_width (3)
    floatView[0] = 0.259; // selection_color.r (#4285f4)
    floatView[1] = 0.522; // selection_color.g
    floatView[2] = 0.957; // selection_color.b
    floatView[3] = 2.0; // selection_ring_width

    // f32 hover_brightness (4), u32 border_enabled (5), f32 border_width (6), f32 _pad1 (7)
    floatView[4] = 1.2; // hover_brightness
    uintView[5] = this.nodeBorderConfig.enabled ? 1 : 0; // border_enabled
    floatView[6] = this.nodeBorderConfig.width; // border_width
    floatView[7] = 0.0; // _pad1

    // vec3 border_color (8-10), f32 _pad2 (11)
    floatView[8] = borderColor[0]; // border_color.r
    floatView[9] = borderColor[1]; // border_color.g
    floatView[10] = borderColor[2]; // border_color.b
    floatView[11] = 0.0; // _pad2

    device.queue.writeBuffer(this.renderConfigBuffer, 0, data);
  }

  /**
   * Parse a CSS color string or hex to RGB values (0-1 range).
   */
  private parseColorString(color: string): [number, number, number] {
    // Use shared color parsing utility
    return parseColorToRGB(color);
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

    // When the current algorithm handles gravity itself, suppress integration
    // gravity to avoid double-applying center pull. The algorithm's gravity
    // pass uses mass-weighted gravity; the integration shader's is uniform.
    const algorithmHandlesGravity = this.currentAlgorithm?.handlesGravity ?? false;
    const effectiveForceConfig = algorithmHandlesGravity
      ? { ...this.forceConfig, centerStrength: 0 }
      : this.forceConfig;

    // Update uniforms with current alpha and force config
    updateSimulationUniforms(
      device,
      this.simBuffers,
      this.state.nodeCount,
      this.state.edgeCount,
      alpha,
      effectiveForceConfig,
    );

    // Compute bounds once per frame for all consumers (algorithm context, collision grid).
    // CPU-side position arrays are synced from GPU every SYNC_INTERVAL frames, so bounds
    // may be slightly stale. The computeBoundsFromPositions function adds a margin for drift.
    this.frameBounds = this.state.parsedGraph
      ? computeBoundsFromPositions(
          this.state.parsedGraph.positionsX,
          this.state.parsedGraph.positionsY,
          this.state.nodeCount,
        )
      : undefined;

    // Update algorithm uniforms if using custom algorithm
    if (this.currentAlgorithm && this.algorithmBuffers && this.algorithmBindGroups) {
      const bounds = this.frameBounds;

      // Spatial algorithms (Barnes-Hut, Density Field) require valid bounds.
      // If bounds are undefined, position data is corrupted (all NaN/Infinity).
      // Stop simulation gracefully rather than throwing errors every frame.
      const algorithmId = this.currentAlgorithm.info.id;
      const requiresBounds = algorithmId === "barnes-hut" || algorithmId === "density";
      if (requiresBounds && !bounds) {
        console.error(
          "CRITICAL: Position data corrupted (all NaN/Infinity). Stopping simulation."
        );
        this.simulationController.stop();
        return;
      }

      const context: AlgorithmRenderContext = {
        device,
        positions: this.simBuffers.positions,
        forces: this.simBuffers.forces,
        nodeCount: this.state.nodeCount,
        edgeCount: this.state.edgeCount,
        forceConfig: this.forceConfig,
        bounds,
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

    // Record collision detection pass (after integration, if enabled)
    if (
      this.forceConfig.collisionEnabled &&
      this.collisionPipeline &&
      this.collisionBuffers &&
      this.collisionBindGroup
    ) {
      const nodeCount = this.state.nodeCount;
      const useGridCollision = nodeCount > 5000 &&
        this.gridCollisionPipeline &&
        this.gridCollisionBuffers &&
        this.gridCollisionBindGroups;

      if (useGridCollision) {
        // Grid collision: O(n·k) spatial hash for large graphs.
        // Reuse frame bounds computed at top of recordSimulationCommands.
        if (this.frameBounds) {
          updateGridCollisionUniforms(
            device,
            this.gridCollisionBuffers!,
            nodeCount,
            this.forceConfig,
            this.frameBounds,
            this.maxNodeRadius,
          );
          recordGridCollisionPass(
            encoder,
            this.gridCollisionPipeline!,
            this.gridCollisionBindGroups!,
            this.gridCollisionBuffers!,
            nodeCount,
            this.forceConfig.collisionIterations,
          );
        } else {
          // Bounds unavailable — fall back to tiled collision
          updateCollisionUniforms(device, this.collisionBuffers, nodeCount, this.forceConfig);
          recordCollisionPass(
            encoder, this.collisionPipeline, this.collisionBindGroup,
            nodeCount, this.forceConfig.collisionIterations, true,
          );
        }
      } else {
        // Tiled/simple collision: O(n^2) for small graphs (<=5000 nodes)
        updateCollisionUniforms(device, this.collisionBuffers, nodeCount, this.forceConfig);
        recordCollisionPass(
          encoder, this.collisionPipeline, this.collisionBindGroup,
          nodeCount, this.forceConfig.collisionIterations,
          nodeCount > 1000,
        );
      }
    }

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
        this.simBuffers.positions,
        this.buffers!.nodeAttributes,
      );
    }

    if (this.edgePipeline && this.buffers) {
      this.edgeBindGroup = createEdgeBindGroup(
        this.gpuContext.device,
        this.edgePipeline,
        this.simBuffers.positions,
        this.buffers.edgeIndices,
        this.buffers.edgeAttributes,
      );
    }

    // Rebuild algorithm bind groups with swapped position/force buffers
    if (this.currentAlgorithm && this.algorithmPipelines && this.algorithmBuffers) {
      // Reuse frame bounds computed in recordSimulationCommands (same frame).
      // Bind group creation doesn't use bounds, but the AlgorithmRenderContext
      // interface includes them for consistency.
      const context: AlgorithmRenderContext = {
        device: this.gpuContext.device,
        positions: this.simBuffers.positions,
        forces: this.simBuffers.forces,
        nodeCount: this.state.nodeCount,
        edgeCount: this.state.edgeCount,
        forceConfig: this.forceConfig,
        bounds: this.frameBounds,
      };

      this.algorithmBindGroups = this.currentAlgorithm.createBindGroups(
        this.gpuContext.device,
        this.algorithmPipelines,
        context,
        this.algorithmBuffers,
      );
    }

    // Rebuild collision bind group with swapped position buffers.
    // Collision binds to positionsOut (integration's write target) so corrections
    // persist through the ping-pong swap.
    if (this.collisionPipeline && this.collisionBuffers) {
      this.collisionBindGroup = createCollisionBindGroup(
        this.gpuContext.device,
        this.collisionPipeline,
        this.collisionBuffers,
        this.simBuffers.positionsOut,
      );
    }

    // Rebuild grid collision bind groups with swapped position buffers
    if (this.gridCollisionPipeline && this.gridCollisionBuffers && this.collisionBuffers) {
      this.gridCollisionBindGroups = createGridCollisionBindGroups(
        this.gpuContext.device,
        this.gridCollisionPipeline,
        this.gridCollisionBuffers,
        this.collisionBuffers.nodeSizes,
        this.simBuffers.positionsOut,
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
          clearValue: this.backgroundColor,
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
      this.renderConfigBindGroup &&
      this.state.nodeCount > 0
    ) {
      renderNodes(
        renderPass,
        this.nodePipeline,
        this.viewportBindGroup,
        this.nodeBindGroup,
        this.renderConfigBindGroup,
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

    // Handle empty graph gracefully
    if (parsed.nodeCount === 0) {
      // Clear existing state
      this.destroyBuffers();
      this.destroySimulationBuffers();
      this.state.loaded = true;
      this.state.nodeCount = 0;
      this.state.edgeCount = 0;
      this.state.parsedGraph = parsed;

      // Start render loop (will just clear the screen)
      if (!this.renderLoop.isRunning) {
        this.renderLoop.start();
      }

      this.events.emit({
        type: "graph:load",
        timestamp: Date.now(),
        nodeCount: 0,
        edgeCount: 0,
      });

      if (this.debug) {
        console.log("Loaded empty graph (0 nodes)");
      }
      return;
    }

    // Initialize positions if needed
    if (needsInitialization(parsed.positionsX, parsed.positionsY)) {
      initializePositions(parsed.positionsX, parsed.positionsY, {
        strategy: "phyllotaxis",
        radius: Math.sqrt(parsed.nodeCount) * 10,
      });
    }

    // Create mutable graph state from parsed data
    this.graphState = MutableGraphState.fromParsedGraph(parsed);

    // Populate WASM engine with graph data for spatial indexing
    this.populateWasmEngine(parsed);

    // Create GPU buffers for rendering (with capacity from graph state)
    this.createBuffers(parsed, this.graphState.nodeCapacity, this.graphState.edgeCapacity);

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

    // CRITICAL: Destroy old simulation buffers before creating new ones.
    // Without this, algorithm buffers (sized for old node count) remain active,
    // causing out-of-bounds reads and NaN propagation that crashes the GPU.
    this.destroySimulationBuffers();

    // Create simulation buffers with capacity headroom for mutations
    const nodeCap = this.buffers?.nodeCapacity ?? parsed.nodeCount;
    const edgeCap = this.buffers?.edgeCapacity ?? parsed.edgeCount;
    this.simBuffers = createSimulationBuffers(
      device,
      parsed.nodeCount,
      parsed.edgeCount,
      nodeCap,
      edgeCap,
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
        this.simBuffers.positions,
        this.buffers!.nodeAttributes,
      );
    }

    if (this.edgePipeline && this.buffers) {
      this.edgeBindGroup = createEdgeBindGroup(
        device,
        this.edgePipeline,
        this.simBuffers.positions,
        this.buffers.edgeIndices,
        this.buffers.edgeAttributes,
      );
    }

    // Create algorithm-specific buffers and bind groups (use capacity)
    if (this.currentAlgorithm && this.algorithmPipelines) {
      this.algorithmBuffers = this.currentAlgorithm.createBuffers(
        device,
        nodeCap,
      );

      // Compute initial bounds from the parsed graph positions.
      // These are the initial positions before simulation starts.
      const bounds = computeBoundsFromPositions(
        parsed.positionsX,
        parsed.positionsY,
        parsed.nodeCount,
      );

      const context: AlgorithmRenderContext = {
        device,
        positions: this.simBuffers.positions,
        forces: this.simBuffers.forces,
        nodeCount: parsed.nodeCount,
        edgeCount: parsed.edgeCount,
        forceConfig: this.forceConfig,
        bounds,
      };

      this.algorithmBindGroups = this.currentAlgorithm.createBindGroups(
        device,
        this.algorithmPipelines,
        context,
        this.algorithmBuffers,
      );

      // Upload algorithm-specific edge data
      this.uploadAlgorithmEdgeData(device);
    }

    // Create collision detection resources (use capacity, not count)
    this.initializeCollisionResources(device, nodeCap, parsed.nodeAttributes);
  }

  /**
   * Initialize collision detection resources
   */
  private initializeCollisionResources(
    device: GPUDevice,
    nodeCount: number,
    nodeAttributes: Float32Array,
  ): void {
    if (!this.collisionPipeline || !this.simBuffers) {
      return;
    }

    // Destroy existing collision resources
    if (this.collisionBuffers) {
      destroyCollisionBuffers(this.collisionBuffers);
    }

    // Create new collision buffers
    this.collisionBuffers = createCollisionBuffers(device, nodeCount);

    // Extract node sizes from attributes and compute max radius
    // Node attributes layout: [radius, r, g, b, selected, hovered] per node (6 floats)
    const ATTRIBUTES_PER_NODE = 6;
    const nodeSizes = new Float32Array(nodeCount);
    let maxRadius = 0;
    for (let i = 0; i < nodeCount; i++) {
      const radius = nodeAttributes[i * ATTRIBUTES_PER_NODE];
      const r = radius > 0 ? radius : 5.0; // Default radius if not set
      nodeSizes[i] = r;
      if (r > maxRadius) maxRadius = r;
    }
    this.maxNodeRadius = maxRadius > 0 ? maxRadius : 5.0;
    uploadNodeSizes(device, this.collisionBuffers, nodeSizes);

    // Create collision bind group — bind to positionsOut so collision corrections
    // persist through the ping-pong swap (integration writes positionsOut, collision
    // modifies positionsOut, swap rotates positionsOut into next frame's read buffer)
    this.collisionBindGroup = createCollisionBindGroup(
      device,
      this.collisionPipeline,
      this.collisionBuffers,
      this.simBuffers.positionsOut,
    );

    // Update collision uniforms
    updateCollisionUniforms(device, this.collisionBuffers, nodeCount, this.forceConfig);

    // Create grid collision resources (spatial hash for O(n·k) at >5000 nodes)
    if (this.gridCollisionPipeline) {
      if (this.gridCollisionBuffers) {
        destroyGridCollisionBuffers(this.gridCollisionBuffers);
      }
      this.gridCollisionBuffers = createGridCollisionBuffers(device, nodeCount);
      this.gridCollisionBindGroups = createGridCollisionBindGroups(
        device,
        this.gridCollisionPipeline,
        this.gridCollisionBuffers,
        this.collisionBuffers.nodeSizes,
        this.simBuffers.positionsOut,
      );
    }

    if (this.debug) {
      console.log(`Collision detection initialized for ${nodeCount} nodes`);
    }
  }

  /**
   * Populate the WASM engine with graph data from a ParsedGraph.
   * Clears any existing data and bulk-loads nodes and edges.
   * This enables the rstar spatial index for O(log n) hit testing.
   */
  private populateWasmEngine(parsed: ParsedGraph): void {
    if (!this.wasmEngine) return;

    this.wasmEngine.clear();

    // Bulk-add nodes from interleaved positions
    const nodeCount = parsed.positionsX.length;
    const positions = new Float32Array(nodeCount * 2);
    for (let i = 0; i < nodeCount; i++) {
      positions[i * 2] = parsed.positionsX[i];
      positions[i * 2 + 1] = parsed.positionsY[i];
    }
    this.wasmEngine.addNodesFromPositions(positions);

    // Bulk-add edges from interleaved pairs
    const edgeCount = parsed.edgeSources.length;
    if (edgeCount > 0) {
      const edgePairs = new Uint32Array(edgeCount * 2);
      for (let i = 0; i < edgeCount; i++) {
        edgePairs[i * 2] = parsed.edgeSources[i];
        edgePairs[i * 2 + 1] = parsed.edgeTargets[i];
      }
      this.wasmEngine.addEdgesFromPairs(edgePairs);
    }

    this.wasmEngine.rebuildSpatialIndex();

    if (this.debug) {
      console.log(`WASM engine populated: ${nodeCount} nodes, ${edgeCount} edges`);
    }
  }

  /**
   * Upload edge data for algorithm-specific formats (CSR for Relativity Atlas).
   *
   * CSR data is generated from MutableGraphState's edge arrays, which are the
   * source of truth for GPU buffer slot indices. This ensures the CSR indices
   * match the actual position buffer layout.
   */
  private uploadAlgorithmEdgeData(device: GPUDevice): void {
    if (!this.currentAlgorithm || !this.algorithmBuffers || !this.graphState) {
      return;
    }

    if (this.currentAlgorithm instanceof RelativityAtlasAlgorithm) {
      const gs = this.graphState;
      const forward = gs.generateForwardCSR();
      const inverse = gs.generateInverseCSR();

      uploadRelativityAtlasEdges(
        device,
        this.algorithmBuffers,
        { offsets: forward.offsets, indices: forward.targets },
        { offsets: inverse.offsets, indices: inverse.sources },
        gs.nodeHighWater,
      );

      // Reset mass state so it gets recomputed on next frame
      (this.currentAlgorithm as RelativityAtlasAlgorithm).resetMassState();

      if (this.debug) {
        console.log(
          `Relativity Atlas: uploaded CSR (${gs.nodeHighWater} nodes, ${gs.edgeCount} edges)`,
        );
      }
    }
  }

  /**
   * Create GPU buffers from parsed graph.
   * Allocates with capacity headroom for incremental mutations.
   */
  private createBuffers(parsed: ParsedGraph, nodeCap?: number, edgeCap?: number): void {
    const { device } = this.gpuContext;

    // Destroy old buffers
    this.destroyBuffers();

    const nodeCount = parsed.positionsX.length;
    const edgeCount = parsed.edgeSources.length;
    const nodeCapacity = Math.max(nodeCap ?? initialCapacity(nodeCount), nodeCount);
    const edgeCapacity = Math.max(edgeCap ?? initialCapacity(edgeCount), edgeCount);

    // Create position buffer (vec2 per node) — sized to capacity
    const positionsVec2 = new Float32Array(nodeCount * 2);
    for (let i = 0; i < nodeCount; i++) {
      positionsVec2[i * 2] = parsed.positionsX[i];
      positionsVec2[i * 2 + 1] = parsed.positionsY[i];
    }

    const positions = device.createBuffer({
      label: "Positions",
      size: nodeCapacity * 8, // vec2<f32> = 8 bytes
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(positions, 0, toArrayBuffer(positionsVec2));

    // Create node attributes buffer (6 floats per node) — sized to capacity
    const nodeAttributes = device.createBuffer({
      label: "Node Attributes",
      size: nodeCapacity * 6 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(nodeAttributes, 0, toArrayBuffer(parsed.nodeAttributes));

    // Create edge buffers — sized to capacity
    const edgeIndicesData = createEdgeIndicesBuffer(
      parsed.edgeSources,
      parsed.edgeTargets,
    );
    const edgeIndices = device.createBuffer({
      label: "Edge Indices",
      size: Math.max(edgeCapacity * 2 * 4, 4), // 2 u32 per edge, minimum 4 bytes
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (edgeIndicesData.byteLength > 0) {
      device.queue.writeBuffer(edgeIndices, 0, toArrayBuffer(edgeIndicesData));
    }

    const edgeAttributes = device.createBuffer({
      label: "Edge Attributes",
      size: Math.max(edgeCapacity * 8 * 4, 4), // 8 floats per edge
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (parsed.edgeAttributes.byteLength > 0) {
      device.queue.writeBuffer(edgeAttributes, 0, toArrayBuffer(parsed.edgeAttributes));
    }

    // Store buffers
    this.buffers = {
      positions,
      nodeAttributes,
      edgeIndices,
      edgeAttributes,
      viewportUniforms: this.viewportUniformBuffer.buffer,
      nodeCapacity,
      edgeCapacity,
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
        positions,
        nodeAttributes,
      );
    }

    if (this.edgePipeline) {
      this.edgeBindGroup = createEdgeBindGroup(
        device,
        this.edgePipeline,
        positions,
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
      this.buffers.positions.destroy();
      this.buffers.nodeAttributes.destroy();
      this.buffers.edgeIndices.destroy();
      this.buffers.edgeAttributes.destroy();
      this.buffers = null;
    }

    this.nodeBindGroup = null;
    this.edgeBindGroup = null;
  }

  // ==========================================================================
  // Public API - Incremental Mutations
  // ==========================================================================

  /**
   * Add a single node to the graph.
   *
   * @param node - Node input data
   * @returns The assigned node ID (the user-provided id)
   */
  async addNode(node: NodeInput): Promise<NodeId> {
    if (!this.graphState || !this.buffers || !this.simBuffers) {
      throw new HeroineGraphError(ErrorCode.INVALID_GRAPH_DATA, "Cannot add node: graph not loaded");
    }

    const gs = this.graphState;

    // Check capacity
    if (gs.needsNodeReallocation(1)) {
      await this.reallocateNodeBuffers(growCapacity(gs.nodeHighWater + 1, gs.nodeCapacity));
    }

    // Allocate slot
    const slot = gs.allocateNodeSlot();
    const nodeId = node.id;
    gs.nodeIdMap.add(nodeId);

    // Parse position
    const x = node.x ?? (Math.random() - 0.5) * Math.sqrt(gs.nodeCount) * 20;
    const y = node.y ?? (Math.random() - 0.5) * Math.sqrt(gs.nodeCount) * 20;

    // Parse attributes
    const radius = node.radius ?? 5;
    const [r, g, b] = node.color ? parseColorToRGB(node.color) : [0.4, 0.6, 0.9];

    // Write to CPU shadow
    gs.positionsX[slot] = x;
    gs.positionsY[slot] = y;
    const attrBase = slot * 6;
    gs.nodeAttributes[attrBase] = radius;
    gs.nodeAttributes[attrBase + 1] = r;
    gs.nodeAttributes[attrBase + 2] = g;
    gs.nodeAttributes[attrBase + 3] = b;
    gs.nodeAttributes[attrBase + 4] = 0; // selected
    gs.nodeAttributes[attrBase + 5] = 0; // hovered

    // Write to GPU buffers (targeted writes)
    const { device } = this.gpuContext;
    const posVec2 = new Float32Array([x, y]);
    const attrData = new Float32Array([radius, r, g, b, 0, 0]);
    const zeroVec2 = new Float32Array([0, 0]);

    device.queue.writeBuffer(this.simBuffers.positions, slot * 8, posVec2);
    device.queue.writeBuffer(this.simBuffers.positionsOut, slot * 8, posVec2);
    device.queue.writeBuffer(this.buffers.nodeAttributes, slot * 24, attrData);
    device.queue.writeBuffer(this.simBuffers.velocities, slot * 8, zeroVec2);
    device.queue.writeBuffer(this.simBuffers.velocitiesOut, slot * 8, zeroVec2);
    device.queue.writeBuffer(this.simBuffers.forces, slot * 8, zeroVec2);

    // Update WASM engine
    if (this.wasmEngine) {
      this.wasmEngine.addNode(x, y);
      this.wasmEngine.rebuildSpatialIndex();
    }

    // Update counts
    this.state.nodeCount = gs.nodeCount;
    this.simBuffers.nodeCount = gs.nodeHighWater;

    // Update parsedGraph reference to point to graphState arrays
    this.syncParsedGraphFromState();

    // Ensure algorithm buffers can handle the new node count
    this.ensureAlgorithmCapacity();

    // Reheat simulation
    this.bumpSimulationAlpha(0.1);

    // Emit event
    this.events.emit({
      type: "node:add",
      timestamp: Date.now(),
      nodeId: nodeId,
      index: slot,
    });

    return slot;
  }

  /**
   * Remove a single node and all its connected edges.
   *
   * @param id - Node ID to remove
   * @returns true if the node was found and removed
   */
  async removeNode(id: NodeId | string): Promise<boolean> {
    if (!this.graphState || !this.buffers || !this.simBuffers) return false;

    const gs = this.graphState;
    const slot = typeof id === "number" && id < gs.nodeHighWater ? id : gs.nodeIdMap.get(id);
    if (slot === undefined) return false;

    // Remove all connected edges first
    const connectedEdges = new Set(gs.getConnectedEdges(slot));
    for (const edgeIndex of connectedEdges) {
      await this.removeEdgeByIndex(edgeIndex);
    }

    // Free the node slot (zeros CPU shadow)
    gs.nodeIdMap.remove(gs.nodeIdMap.getId(slot)!);
    gs.freeNodeSlot(slot);

    // Write zeros to GPU buffers
    const { device } = this.gpuContext;
    const zeroVec2 = new Float32Array([0, 0]);
    const zeroAttrs = new Float32Array(6);

    device.queue.writeBuffer(this.simBuffers.positions, slot * 8, zeroVec2);
    device.queue.writeBuffer(this.simBuffers.positionsOut, slot * 8, zeroVec2);
    device.queue.writeBuffer(this.buffers.nodeAttributes, slot * 24, zeroAttrs);
    device.queue.writeBuffer(this.simBuffers.velocities, slot * 8, zeroVec2);
    device.queue.writeBuffer(this.simBuffers.velocitiesOut, slot * 8, zeroVec2);
    device.queue.writeBuffer(this.simBuffers.forces, slot * 8, zeroVec2);

    // Update WASM engine
    if (this.wasmEngine) {
      this.wasmEngine.removeNode(slot);
      this.wasmEngine.rebuildSpatialIndex();
    }

    // Update counts
    this.state.nodeCount = gs.nodeCount;
    this.simBuffers.nodeCount = gs.nodeHighWater;

    this.syncParsedGraphFromState();
    this.bumpSimulationAlpha(0.05);

    // Emit event
    this.events.emit({
      type: "node:remove",
      timestamp: Date.now(),
      nodeId: id,
      index: slot,
    });

    return true;
  }

  /**
   * Add a single edge between two existing nodes.
   *
   * @param edge - Edge input data
   * @returns The edge index, or undefined if source/target not found
   */
  async addEdge(edge: EdgeInput): Promise<EdgeId | undefined> {
    if (!this.graphState || !this.buffers || !this.simBuffers) return undefined;

    const gs = this.graphState;

    // Resolve source/target to slot indices
    const srcSlot = gs.nodeIdMap.get(edge.source);
    const tgtSlot = gs.nodeIdMap.get(edge.target);
    if (srcSlot === undefined || tgtSlot === undefined) return undefined;

    // Check capacity
    if (gs.needsEdgeReallocation(1)) {
      await this.reallocateEdgeBuffers(growCapacity(gs.edgeCount + 1, gs.edgeCapacity));
    }

    // Allocate slot
    const slot = gs.allocateEdgeSlot();
    const edgeId = (edge as Record<string, unknown>)["id"] as string | number | undefined ?? `edge_${slot}`;
    gs.edgeIdMap.add(edgeId);

    // Parse attributes
    const width = edge.width ?? 1;
    const [r, g, b] = edge.color ? parseColorToRGB(edge.color) : [0.5, 0.5, 0.5];

    // Write to CPU shadow
    gs.edgeSources[slot] = srcSlot;
    gs.edgeTargets[slot] = tgtSlot;
    const eAttrBase = slot * 8;
    gs.edgeAttributes[eAttrBase] = width;
    gs.edgeAttributes[eAttrBase + 1] = r;
    gs.edgeAttributes[eAttrBase + 2] = g;
    gs.edgeAttributes[eAttrBase + 3] = b;
    gs.edgeAttributes[eAttrBase + 4] = 0; // selected
    gs.edgeAttributes[eAttrBase + 5] = 0; // hovered
    gs.edgeAttributes[eAttrBase + 6] = 0; // curvature
    gs.edgeAttributes[eAttrBase + 7] = 0; // reserved

    // Update adjacency
    gs.addEdgeAdjacency(slot, srcSlot, tgtSlot);

    // Write to GPU buffers
    const { device } = this.gpuContext;
    const edgeIndicesData = new Uint32Array([srcSlot, tgtSlot]);
    const edgeAttrData = new Float32Array([width, r, g, b, 0, 0, 0, 0]);
    const srcData = new Uint32Array([srcSlot]);
    const tgtData = new Uint32Array([tgtSlot]);

    device.queue.writeBuffer(this.buffers.edgeIndices, slot * 8, edgeIndicesData);
    device.queue.writeBuffer(this.buffers.edgeAttributes, slot * 32, edgeAttrData);
    device.queue.writeBuffer(this.simBuffers.edgeSources, slot * 4, srcData);
    device.queue.writeBuffer(this.simBuffers.edgeTargets, slot * 4, tgtData);

    // Update WASM engine
    if (this.wasmEngine) {
      const weight = edge.weight ?? 1.0;
      this.wasmEngine.addEdge(srcSlot, tgtSlot, weight);
    }

    // Update counts
    this.state.edgeCount = gs.edgeCount;

    this.syncParsedGraphFromState();
    this.bumpSimulationAlpha(0.05);

    // Emit event
    this.events.emit({
      type: "edge:add",
      timestamp: Date.now(),
      edgeId: edgeId,
      sourceId: edge.source,
      targetId: edge.target,
    });

    return slot;
  }

  /**
   * Remove a single edge.
   *
   * @param id - Edge ID to remove
   * @returns true if the edge was found and removed
   */
  async removeEdge(id: EdgeId | string): Promise<boolean> {
    if (!this.graphState) return false;

    const gs = this.graphState;
    const slot = typeof id === "number" && id < gs.edgeCount ? id : gs.edgeIdMap.get(id);
    if (slot === undefined) return false;

    return this.removeEdgeByIndex(slot);
  }

  /**
   * Internal: Remove an edge by its slot index using swap-remove.
   */
  private async removeEdgeByIndex(index: number): Promise<boolean> {
    if (!this.graphState || !this.buffers || !this.simBuffers) return false;

    const gs = this.graphState;
    if (index >= gs.edgeCount) return false;

    // Get the ID of the edge being removed
    const removedId = gs.edgeIdMap.getId(index);
    if (removedId !== undefined) {
      gs.edgeIdMap.remove(removedId);
    }

    // If not the last edge, we need to fix up the swapped edge's ID mapping
    const lastIndex = gs.edgeCount - 1;
    const swappedId = index < lastIndex ? gs.edgeIdMap.getId(lastIndex) : undefined;

    // Perform swap-remove on CPU shadow
    const swappedFromIndex = gs.freeEdgeSlot(index);

    // Update the swapped edge's ID map entry
    if (swappedId !== undefined && swappedFromIndex >= 0) {
      gs.edgeIdMap.remove(swappedId);
      gs.edgeIdMap.add(swappedId); // Re-add at new position
    }

    // Write the swapped edge data to GPU at the vacated slot
    if (swappedFromIndex >= 0 && index < gs.edgeCount) {
      const { device } = this.gpuContext;
      const edgeIndicesData = new Uint32Array([gs.edgeSources[index], gs.edgeTargets[index]]);
      const edgeAttrData = gs.edgeAttributes.subarray(index * 8, index * 8 + 8);
      const srcData = new Uint32Array([gs.edgeSources[index]]);
      const tgtData = new Uint32Array([gs.edgeTargets[index]]);

      device.queue.writeBuffer(this.buffers.edgeIndices, index * 8, edgeIndicesData);
      device.queue.writeBuffer(this.buffers.edgeAttributes, index * 32, new Float32Array(edgeAttrData));
      device.queue.writeBuffer(this.simBuffers.edgeSources, index * 4, srcData);
      device.queue.writeBuffer(this.simBuffers.edgeTargets, index * 4, tgtData);
    }

    // Update WASM engine
    if (this.wasmEngine && removedId !== undefined) {
      this.wasmEngine.removeEdge(
        typeof removedId === "number" ? removedId : index,
      );
    }

    // Update counts
    this.state.edgeCount = gs.edgeCount;

    this.syncParsedGraphFromState();

    // Emit event
    if (removedId !== undefined) {
      this.events.emit({
        type: "edge:remove",
        timestamp: Date.now(),
        edgeId: typeof removedId === "number" ? removedId : index,
      });
    }

    return true;
  }

  // ---------- Batch Mutations ----------

  /**
   * Add multiple nodes at once.
   *
   * @param nodes - Array of node inputs
   * @returns Array of assigned node IDs
   */
  async addNodes(nodes: NodeInput[]): Promise<NodeId[]> {
    if (!this.graphState || !this.buffers || !this.simBuffers) {
      throw new HeroineGraphError(ErrorCode.INVALID_GRAPH_DATA, "Cannot add nodes: graph not loaded");
    }

    const gs = this.graphState;

    // Check capacity
    if (gs.needsNodeReallocation(nodes.length)) {
      await this.reallocateNodeBuffers(
        growCapacity(gs.nodeHighWater + nodes.length, gs.nodeCapacity),
      );
    }

    const ids: NodeId[] = [];
    for (const node of nodes) {
      const slot = gs.allocateNodeSlot();
      gs.nodeIdMap.add(node.id);

      const x = node.x ?? (Math.random() - 0.5) * Math.sqrt(gs.nodeCount) * 20;
      const y = node.y ?? (Math.random() - 0.5) * Math.sqrt(gs.nodeCount) * 20;
      const radius = node.radius ?? 5;
      const [r, g, b] = node.color ? parseColorToRGB(node.color) : [0.4, 0.6, 0.9];

      gs.positionsX[slot] = x;
      gs.positionsY[slot] = y;
      const attrBase = slot * 6;
      gs.nodeAttributes[attrBase] = radius;
      gs.nodeAttributes[attrBase + 1] = r;
      gs.nodeAttributes[attrBase + 2] = g;
      gs.nodeAttributes[attrBase + 3] = b;
      gs.nodeAttributes[attrBase + 4] = 0;
      gs.nodeAttributes[attrBase + 5] = 0;

      ids.push(slot);
    }

    // Flush all node data to GPU in bulk
    this.flushNodeBuffersToGPU();

    // Update WASM engine in batch
    if (this.wasmEngine) {
      const positions = new Float32Array(nodes.length * 2);
      for (let i = 0; i < ids.length; i++) {
        const slot = ids[i];
        positions[i * 2] = gs.positionsX[slot];
        positions[i * 2 + 1] = gs.positionsY[slot];
      }
      this.wasmEngine.addNodesFromPositions(positions);
      this.wasmEngine.rebuildSpatialIndex();
    }

    this.state.nodeCount = gs.nodeCount;
    this.simBuffers.nodeCount = gs.nodeHighWater;
    this.syncParsedGraphFromState();
    this.ensureAlgorithmCapacity();
    this.bumpSimulationAlpha(0.2);

    // Emit batch summary
    this.events.emit({
      type: "graph:mutate",
      timestamp: Date.now(),
      nodesAdded: ids.length,
      nodesRemoved: 0,
      edgesAdded: 0,
      edgesRemoved: 0,
    });

    return ids;
  }

  /**
   * Remove multiple nodes.
   *
   * @param ids - Node IDs to remove
   * @returns Number of nodes actually removed
   */
  async removeNodes(ids: (NodeId | string)[]): Promise<number> {
    let removed = 0;
    for (const id of ids) {
      if (await this.removeNode(id)) removed++;
    }

    if (removed > 0) {
      this.events.emit({
        type: "graph:mutate",
        timestamp: Date.now(),
        nodesAdded: 0,
        nodesRemoved: removed,
        edgesAdded: 0,
        edgesRemoved: 0,
      });
    }

    return removed;
  }

  /**
   * Add multiple edges at once.
   *
   * @param edges - Array of edge inputs
   * @returns Array of edge IDs (undefined for edges with invalid source/target)
   */
  async addEdges(edges: EdgeInput[]): Promise<(EdgeId | undefined)[]> {
    const results: (EdgeId | undefined)[] = [];
    for (const edge of edges) {
      results.push(await this.addEdge(edge));
    }

    const added = results.filter((r) => r !== undefined).length;
    if (added > 0) {
      this.events.emit({
        type: "graph:mutate",
        timestamp: Date.now(),
        nodesAdded: 0,
        nodesRemoved: 0,
        edgesAdded: added,
        edgesRemoved: 0,
      });
    }

    return results;
  }

  /**
   * Remove multiple edges.
   *
   * @param ids - Edge IDs to remove
   * @returns Number of edges actually removed
   */
  async removeEdges(ids: (EdgeId | string)[]): Promise<number> {
    let removed = 0;
    for (const id of ids) {
      if (await this.removeEdge(id)) removed++;
    }

    if (removed > 0) {
      this.events.emit({
        type: "graph:mutate",
        timestamp: Date.now(),
        nodesAdded: 0,
        nodesRemoved: 0,
        edgesAdded: 0,
        edgesRemoved: removed,
      });
    }

    return removed;
  }

  // ---------- Mutation Helpers ----------

  /**
   * Flush all node position/attribute data to GPU.
   * Used for batch operations.
   */
  private flushNodeBuffersToGPU(): void {
    if (!this.graphState || !this.buffers || !this.simBuffers) return;

    const gs = this.graphState;
    const { device } = this.gpuContext;
    const hw = gs.nodeHighWater;

    // Interleave positions into vec2 format
    const posVec2 = new Float32Array(hw * 2);
    for (let i = 0; i < hw; i++) {
      posVec2[i * 2] = gs.positionsX[i];
      posVec2[i * 2 + 1] = gs.positionsY[i];
    }

    device.queue.writeBuffer(this.simBuffers.positions, 0, posVec2);
    device.queue.writeBuffer(this.simBuffers.positionsOut, 0, posVec2);
    device.queue.writeBuffer(
      this.buffers.nodeAttributes,
      0,
      toArrayBuffer(gs.nodeAttributes.subarray(0, hw * 6)),
    );

    // Zero velocities/forces for all slots
    const zeros = new Float32Array(hw * 2);
    device.queue.writeBuffer(this.simBuffers.velocities, 0, zeros);
    device.queue.writeBuffer(this.simBuffers.velocitiesOut, 0, zeros);
    device.queue.writeBuffer(this.simBuffers.forces, 0, zeros);
  }

  /**
   * Sync parsedGraph reference to use graphState's arrays.
   * This keeps the existing code that reads from parsedGraph working.
   */
  private syncParsedGraphFromState(): void {
    if (!this.graphState || !this.state.parsedGraph) return;

    const gs = this.graphState;
    this.state.parsedGraph.positionsX = gs.positionsX;
    this.state.parsedGraph.positionsY = gs.positionsY;
    this.state.parsedGraph.nodeAttributes = gs.nodeAttributes;
    this.state.parsedGraph.edgeSources = gs.edgeSources.subarray(0, gs.edgeCount);
    this.state.parsedGraph.edgeTargets = gs.edgeTargets.subarray(0, gs.edgeCount);
    this.state.parsedGraph.edgeAttributes = gs.edgeAttributes.subarray(0, gs.edgeCount * 8);
    this.state.parsedGraph.nodeCount = gs.nodeHighWater; // Use highWater for draw calls
    this.state.parsedGraph.edgeCount = gs.edgeCount;
    this.state.parsedGraph.nodeIdMap = gs.nodeIdMap;
    this.state.parsedGraph.edgeIdMap = gs.edgeIdMap;
  }

  /**
   * Ensure algorithm buffers can handle the current nodeHighWater.
   * Recreates algorithm buffers/bind groups if nodeHighWater exceeds their maxNodes.
   */
  private ensureAlgorithmCapacity(): void {
    if (!this.currentAlgorithm || !this.algorithmPipelines || !this.algorithmBuffers || !this.simBuffers || !this.graphState) return;

    const gs = this.graphState;
    const algMaxNodes = (this.algorithmBuffers as unknown as { maxNodes?: number }).maxNodes;
    if (algMaxNodes === undefined || gs.nodeHighWater <= algMaxNodes) return;

    // Algorithm buffers are too small — recreate with current nodeCapacity
    const { device } = this.gpuContext;
    const newCap = gs.nodeCapacity;

    this.algorithmBuffers.destroy();
    this.algorithmBuffers = this.currentAlgorithm.createBuffers(device, newCap);

    const bounds = computeBoundsFromPositions(gs.positionsX, gs.positionsY, gs.nodeHighWater);
    const context: AlgorithmRenderContext = {
      device,
      positions: this.simBuffers.positions,
      forces: this.simBuffers.forces,
      nodeCount: gs.nodeHighWater,
      edgeCount: gs.edgeCount,
      forceConfig: this.forceConfig,
      bounds,
    };

    this.algorithmBindGroups = this.currentAlgorithm.createBindGroups(
      device,
      this.algorithmPipelines,
      context,
      this.algorithmBuffers,
    );

    this.uploadAlgorithmEdgeData(device);

    if (this.debug) {
      console.log(`Algorithm buffers recreated: capacity ${newCap} (was ${algMaxNodes})`);
    }
  }

  /**
   * Bump simulation alpha for mutations
   */
  private bumpSimulationAlpha(minAlpha: number): void {
    const currentAlpha = this.simulationController.state.alpha;
    if (currentAlpha < minAlpha) {
      this.simulationController.setAlpha(minAlpha);
    }
    if (this.simulationController.state.status !== "running") {
      this.simulationController.start();
    }
  }

  // ---------- Buffer Reallocation ----------

  /**
   * Reallocate all node-related GPU buffers to a new capacity.
   * Re-uploads all data from CPU shadow arrays. Rebuilds all affected bind groups.
   */
  private async reallocateNodeBuffers(newCapacity: number): Promise<void> {
    if (!this.graphState || !this.buffers || !this.simBuffers) return;

    const gs = this.graphState;
    const { device } = this.gpuContext;

    // Grow CPU shadow arrays
    gs.growNodeCapacity(newCapacity);

    // === Render buffers ===
    // Destroy old render node buffers
    this.buffers.positions.destroy();
    this.buffers.nodeAttributes.destroy();

    // Create new render buffers at new capacity
    this.buffers.positions = device.createBuffer({
      label: "Positions",
      size: newCapacity * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.buffers.nodeAttributes = device.createBuffer({
      label: "Node Attributes",
      size: newCapacity * 6 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.buffers.nodeCapacity = newCapacity;

    // Upload data from CPU shadow
    const hw = gs.nodeHighWater;
    const posVec2 = new Float32Array(hw * 2);
    for (let i = 0; i < hw; i++) {
      posVec2[i * 2] = gs.positionsX[i];
      posVec2[i * 2 + 1] = gs.positionsY[i];
    }
    device.queue.writeBuffer(this.buffers.positions, 0, posVec2);
    device.queue.writeBuffer(
      this.buffers.nodeAttributes,
      0,
      toArrayBuffer(gs.nodeAttributes.subarray(0, hw * 6)),
    );

    // === Simulation buffers ===
    // Destroy old sim node buffers
    this.simBuffers.positions.destroy();
    this.simBuffers.positionsOut.destroy();
    this.simBuffers.velocities.destroy();
    this.simBuffers.velocitiesOut.destroy();
    this.simBuffers.forces.destroy();
    this.simBuffers.nodeFlags.destroy();
    this.simBuffers.readback.destroy();

    const nodeVec2Bytes = newCapacity * 8;
    const nodeFlagBytes = newCapacity * 4;

    // Create new sim buffers at new capacity
    this.simBuffers.positions = device.createBuffer({
      label: "Sim Positions",
      size: nodeVec2Bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.simBuffers.positionsOut = device.createBuffer({
      label: "Sim Positions Out",
      size: nodeVec2Bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.simBuffers.velocities = device.createBuffer({
      label: "Sim Velocities",
      size: nodeVec2Bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.simBuffers.velocitiesOut = device.createBuffer({
      label: "Sim Velocities Out",
      size: nodeVec2Bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.simBuffers.forces = device.createBuffer({
      label: "Sim Forces",
      size: nodeVec2Bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.simBuffers.nodeFlags = device.createBuffer({
      label: "Sim Node Flags",
      size: nodeFlagBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.simBuffers.readback = device.createBuffer({
      label: "Sim Readback",
      size: nodeVec2Bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.simBuffers.nodeCapacity = newCapacity;

    // Upload position data to both sim ping-pong buffers
    device.queue.writeBuffer(this.simBuffers.positions, 0, posVec2);
    device.queue.writeBuffer(this.simBuffers.positionsOut, 0, posVec2);

    // Zero velocities and forces (new capacity may have uninitialized data)
    const zeros = new Float32Array(hw * 2);
    device.queue.writeBuffer(this.simBuffers.velocities, 0, zeros);
    device.queue.writeBuffer(this.simBuffers.velocitiesOut, 0, zeros);
    device.queue.writeBuffer(this.simBuffers.forces, 0, zeros);

    // === Rebuild all affected bind groups ===
    this.rebuildAllBindGroups();

    // === Rebuild algorithm buffers if they're smaller than new capacity ===
    if (this.currentAlgorithm && this.algorithmPipelines) {
      this.algorithmBuffers?.destroy();
      this.algorithmBuffers = this.currentAlgorithm.createBuffers(device, newCapacity);

      const bounds = computeBoundsFromPositions(gs.positionsX, gs.positionsY, hw);
      const context: AlgorithmRenderContext = {
        device,
        positions: this.simBuffers.positions,
        forces: this.simBuffers.forces,
        nodeCount: hw,
        edgeCount: gs.edgeCount,
        forceConfig: this.forceConfig,
        bounds,
      };

      this.algorithmBindGroups = this.currentAlgorithm.createBindGroups(
        device,
        this.algorithmPipelines,
        context,
        this.algorithmBuffers,
      );

      this.uploadAlgorithmEdgeData(device);
    }

    // === Rebuild collision buffers ===
    this.initializeCollisionResources(device, newCapacity, gs.nodeAttributes);

    // === Update layer render contexts ===
    this.updateLayerRenderContext();

    if (this.debug) {
      console.log(`Node buffers reallocated: capacity ${newCapacity}`);
    }
  }

  /**
   * Reallocate all edge-related GPU buffers to a new capacity.
   * Re-uploads all data from CPU shadow arrays. Rebuilds affected bind groups.
   */
  private async reallocateEdgeBuffers(newCapacity: number): Promise<void> {
    if (!this.graphState || !this.buffers || !this.simBuffers) return;

    const gs = this.graphState;
    const { device } = this.gpuContext;

    // Grow CPU shadow arrays
    gs.growEdgeCapacity(newCapacity);

    // === Render edge buffers ===
    this.buffers.edgeIndices.destroy();
    this.buffers.edgeAttributes.destroy();

    this.buffers.edgeIndices = device.createBuffer({
      label: "Edge Indices",
      size: Math.max(newCapacity * 2 * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.buffers.edgeAttributes = device.createBuffer({
      label: "Edge Attributes",
      size: Math.max(newCapacity * 8 * 4, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.buffers.edgeCapacity = newCapacity;

    // Upload edge data from CPU shadow
    const ec = gs.edgeCount;
    if (ec > 0) {
      const edgeIndicesData = createEdgeIndicesBuffer(
        gs.edgeSources.subarray(0, ec),
        gs.edgeTargets.subarray(0, ec),
      );
      device.queue.writeBuffer(this.buffers.edgeIndices, 0, toArrayBuffer(edgeIndicesData));
      device.queue.writeBuffer(
        this.buffers.edgeAttributes,
        0,
        toArrayBuffer(gs.edgeAttributes.subarray(0, ec * 8)),
      );
    }

    // === Simulation edge buffers ===
    this.simBuffers.edgeSources.destroy();
    this.simBuffers.edgeTargets.destroy();

    const edgeBytes = Math.max(newCapacity * 4, 4);
    this.simBuffers.edgeSources = device.createBuffer({
      label: "Sim Edge Sources",
      size: edgeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.simBuffers.edgeTargets = device.createBuffer({
      label: "Sim Edge Targets",
      size: edgeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.simBuffers.edgeCapacity = newCapacity;

    // Upload edge source/target data
    if (ec > 0) {
      device.queue.writeBuffer(this.simBuffers.edgeSources, 0, toArrayBuffer(gs.edgeSources.subarray(0, ec)));
      device.queue.writeBuffer(this.simBuffers.edgeTargets, 0, toArrayBuffer(gs.edgeTargets.subarray(0, ec)));
    }

    // Rebuild bind groups that reference edge buffers
    this.rebuildAllBindGroups();

    // Rebuild algorithm edge data if Relativity Atlas is active
    if (this.currentAlgorithm && this.algorithmPipelines && this.algorithmBuffers) {
      this.uploadAlgorithmEdgeData(device);
    }

    if (this.debug) {
      console.log(`Edge buffers reallocated: capacity ${newCapacity}`);
    }
  }

  /**
   * Rebuild all bind groups after buffer reallocation.
   * This is the same pattern used in swapAndRebuildBindGroups() but
   * also updates edge bind groups for edge buffer changes.
   */
  private rebuildAllBindGroups(): void {
    if (!this.simBuffers || !this.simulationPipeline) return;

    const { device } = this.gpuContext;

    // Rebuild simulation bind groups
    this.simBindGroups = createSimulationBindGroups(
      device,
      this.simulationPipeline,
      this.simBuffers,
    );

    // Rebuild node render bind group
    if (this.nodePipeline) {
      this.nodeBindGroup = createNodeBindGroup(
        device,
        this.nodePipeline,
        this.simBuffers.positions,
        this.buffers!.nodeAttributes,
      );
    }

    // Rebuild edge render bind group
    if (this.edgePipeline && this.buffers) {
      this.edgeBindGroup = createEdgeBindGroup(
        device,
        this.edgePipeline,
        this.simBuffers.positions,
        this.buffers.edgeIndices,
        this.buffers.edgeAttributes,
      );
    }

    // Rebuild algorithm bind groups
    if (this.currentAlgorithm && this.algorithmPipelines && this.algorithmBuffers) {
      const gs = this.graphState!;
      const bounds = computeBoundsFromPositions(gs.positionsX, gs.positionsY, gs.nodeHighWater);

      const context: AlgorithmRenderContext = {
        device,
        positions: this.simBuffers.positions,
        forces: this.simBuffers.forces,
        nodeCount: gs.nodeHighWater,
        edgeCount: gs.edgeCount,
        forceConfig: this.forceConfig,
        bounds,
      };

      this.algorithmBindGroups = this.currentAlgorithm.createBindGroups(
        device,
        this.algorithmPipelines,
        context,
        this.algorithmBuffers,
      );
    }

    // Rebuild collision bind group — bind to positionsOut for ping-pong consistency
    if (this.collisionPipeline && this.collisionBuffers) {
      this.collisionBindGroup = createCollisionBindGroup(
        device,
        this.collisionPipeline,
        this.collisionBuffers,
        this.simBuffers.positionsOut,
      );
    }

    // Rebuild grid collision bind groups
    if (this.gridCollisionPipeline && this.gridCollisionBuffers && this.collisionBuffers) {
      this.gridCollisionBindGroups = createGridCollisionBindGroups(
        device,
        this.gridCollisionPipeline,
        this.gridCollisionBuffers,
        this.collisionBuffers.nodeSizes,
        this.simBuffers.positionsOut,
      );
    }
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
   * Enable or disable collision detection.
   *
   * @param enabled - Whether collision detection should be enabled
   * @param strength - Optional collision strength (0-1)
   */
  setCollisionEnabled(enabled: boolean, strength?: number): void {
    this.forceConfig.collisionEnabled = enabled;
    if (strength !== undefined) {
      this.forceConfig.collisionStrength = Math.max(0, Math.min(1, strength));
    }
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
      // Use nodeCapacity (not nodeCount) so algorithm buffers have headroom for mutations
      const algCapacity = this.buffers?.nodeCapacity ?? this.state.nodeCount;
      this.algorithmBuffers = algorithm.createBuffers(
        this.gpuContext.device,
        algCapacity,
      );

      // Compute bounds from current positions for spatial algorithms.
      const bounds = this.state.parsedGraph
        ? computeBoundsFromPositions(
            this.state.parsedGraph.positionsX,
            this.state.parsedGraph.positionsY,
            this.state.nodeCount,
          )
        : undefined;

      // Create bind groups
      const context: AlgorithmRenderContext = {
        device: this.gpuContext.device,
        positions: this.simBuffers.positions,
        forces: this.simBuffers.forces,
        nodeCount: this.state.nodeCount,
        edgeCount: this.state.edgeCount,
        forceConfig: this.forceConfig,
        bounds,
      };

      this.algorithmBindGroups = algorithm.createBindGroups(
        this.gpuContext.device,
        this.algorithmPipelines,
        context,
        this.algorithmBuffers,
      );

      // Upload algorithm-specific edge data
      this.uploadAlgorithmEdgeData(this.gpuContext.device);
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
   * Get or create the default intensity buffer (all 1.0 values).
   * Used for density mode where all nodes contribute equally.
   */
  private getOrCreateDefaultIntensityBuffer(): GPUBuffer {
    const { device } = this.gpuContext;
    const requiredSize = Math.max(4, this.state.nodeCount * 4); // 1 f32 per node, min 4 bytes

    // Recreate buffer if size changed
    if (!this.defaultIntensityBuffer || this.defaultIntensityBuffer.size < requiredSize) {
      this.defaultIntensityBuffer?.destroy();

      // Create buffer with all 1.0 values
      const intensities = new Float32Array(this.state.nodeCount || 1);
      intensities.fill(1.0);

      this.defaultIntensityBuffer = device.createBuffer({
        label: "Default Intensity Buffer",
        size: requiredSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });

      // Write initial data
      new Float32Array(this.defaultIntensityBuffer.getMappedRange()).set(intensities);
      this.defaultIntensityBuffer.unmap();
    }

    return this.defaultIntensityBuffer;
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
        positions: this.simBuffers.positions,
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

      // Check if metaball is using a stream for per-node intensity
      const metaballDataSource = metaballLayer.getDataSource();
      let metaballIntensities: GPUBuffer;

      if (metaballDataSource !== "density" && this.streamManager.hasStream(metaballDataSource)) {
        // Get stream values and create intensity buffer
        const stream = this.streamManager.getStream(metaballDataSource);
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
          if (!this.metaballIntensityBuffer || this.metaballIntensityBuffer.size < requiredSize) {
            this.metaballIntensityBuffer?.destroy();
            this.metaballIntensityBuffer = device.createBuffer({
              label: "Metaball Stream Intensity Buffer",
              size: requiredSize,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
          }

          // Write intensity data
          device.queue.writeBuffer(this.metaballIntensityBuffer, 0, intensities);
          metaballIntensities = this.metaballIntensityBuffer;
        } else {
          // Stream not found, use default density mode
          metaballIntensities = this.getOrCreateDefaultIntensityBuffer();
        }
      } else {
        // Density mode: all nodes contribute equally (intensity = 1.0)
        metaballIntensities = this.getOrCreateDefaultIntensityBuffer();
      }

      const metaballContext: MetaballRenderContext = {
        viewportUniformBuffer: this.viewportUniformBuffer.buffer,
        positions: this.simBuffers.positions,
        nodeIntensities: metaballIntensities,
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

  /**
   * Set contour data source.
   *
   * @param source - 'density' for uniform intensity (all nodes contribute equally),
   *                 or a stream ID to use stream values for contour thresholds
   *
   * @example
   * ```typescript
   * // Use uniform density (default)
   * graph.setContourDataSource('density');
   *
   * // Use activity stream values - contours follow activity level thresholds
   * graph.setContourDataSource('activity');
   * ```
   */
  setContourDataSource(source: string): void {
    const contourLayer = this.layerManager.getLayer<ContourLayer>("contour");
    if (contourLayer) {
      contourLayer.setDataSource(source);
    }

    // The contour layer uses the heatmap's density texture, so we need to
    // ensure the heatmap is configured with the same data source
    const heatmapLayer = this.layerManager.getLayer<HeatmapLayer>("heatmap");
    if (heatmapLayer) {
      heatmapLayer.setDataSource(source);
    }

    // Rebuild render context to update stream intensity buffer
    this.updateLayerRenderContext();
  }

  /**
   * Get contour data source.
   *
   * @returns Current data source ('density' or stream ID)
   */
  getContourDataSource(): string | null {
    const layer = this.layerManager.getLayer<ContourLayer>("contour");
    return layer?.getDataSource() ?? null;
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

  /**
   * Set metaball data source.
   *
   * @param source - 'density' for uniform intensity (all nodes contribute equally),
   *                 or a stream ID to use stream values for per-node blob size
   *
   * @example
   * ```typescript
   * // Use uniform density (default)
   * graph.setMetaballDataSource('density');
   *
   * // Use importance stream values - nodes with higher importance = larger blobs
   * graph.setMetaballDataSource('importance');
   * ```
   */
  setMetaballDataSource(source: string): void {
    const layer = this.layerManager.getLayer<MetaballLayer>("metaball");
    if (layer) {
      layer.setDataSource(source);
      // Rebuild render context to update stream data
      this.updateLayerRenderContext();
    }
  }

  /**
   * Get metaball data source.
   *
   * @returns Current data source ('density' or stream ID)
   */
  getMetaballDataSource(): string | null {
    const layer = this.layerManager.getLayer<MetaballLayer>("metaball");
    return layer?.getDataSource() ?? null;
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

    // Also update collision buffers if they're initialized
    if (this.collisionBuffers) {
      uploadNodeSizes(device, this.collisionBuffers, sizes);

      // Recompute max radius for grid collision cell sizing
      let maxRadius = 0;
      for (let i = 0; i < sizes.length; i++) {
        if (sizes[i] > maxRadius) maxRadius = sizes[i];
      }
      this.maxNodeRadius = maxRadius > 0 ? maxRadius : 5.0;

      if (this.debug) {
        console.log(`Updated collision radii for ${sizes.length} nodes`);
      }
    }
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
    // Edge attrs layout: [width, r, g, b, selected, hovered, curvature, reserved] per edge (8 floats)
    for (let i = 0; i < this.state.edgeCount; i++) {
      const colorBase = i * 4;
      const attrBase = i * 8;

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
    // Edge attrs layout: [width, r, g, b, selected, hovered, curvature, reserved] per edge (8 floats)
    for (let i = 0; i < this.state.edgeCount; i++) {
      const width = widths[i];
      if (!Number.isNaN(width) && width > 0) {
        edgeAttrs[i * 8] = width; // width is at offset 0
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
  // Public API - Curved Edges
  // ==========================================================================

  /**
   * Configure curved edge rendering.
   *
   * @param config Partial curved edge configuration to merge with current settings.
   *
   * @example
   * ```typescript
   * // Enable curved edges
   * graph.setCurvedEdges({ enabled: true });
   *
   * // Enable with custom segments and weight
   * graph.setCurvedEdges({ enabled: true, segments: 25, weight: 0.6 });
   *
   * // Disable curved edges
   * graph.setCurvedEdges({ enabled: false });
   * ```
   */
  setCurvedEdges(config: Partial<CurvedEdgeConfig>): void {
    if (!this.edgePipeline) return;

    updateCurveConfig(this.gpuContext.device, this.edgePipeline, config);
  }

  /**
   * Get current curved edge configuration.
   *
   * @returns Current curved edge configuration.
   */
  getCurvedEdgeConfig(): CurvedEdgeConfig {
    if (!this.edgePipeline) {
      return { ...DEFAULT_CURVED_EDGE_CONFIG };
    }
    return { ...this.edgePipeline.curveConfig };
  }

  /**
   * Enable curved edge rendering.
   *
   * @param segments Optional number of tessellation segments (default: 19).
   * @param weight Optional rational curve weight (default: 0.8).
   */
  enableCurvedEdges(segments?: number, weight?: number): void {
    this.setCurvedEdges({
      enabled: true,
      ...(segments !== undefined && { segments }),
      ...(weight !== undefined && { weight }),
    });
  }

  /**
   * Disable curved edge rendering (back to straight edges).
   */
  disableCurvedEdges(): void {
    this.setCurvedEdges({ enabled: false });
  }

  /**
   * Set curvature for individual edges.
   *
   * Curvature values control how much each edge bends:
   * - Positive values bend the edge to the right
   * - Negative values bend the edge to the left
   * - Zero means a straight edge
   * - Typical values are in the range -0.5 to 0.5
   *
   * Note: Curved edges must be enabled via setCurvedEdges({ enabled: true })
   * for curvature values to take effect.
   *
   * @param curvatures Float32Array with 1 value per edge.
   *                   Length must equal edgeCount.
   * @throws HeroineGraphError if array length doesn't match edgeCount
   *
   * @example
   * ```typescript
   * // Give all edges random curvature
   * const curvatures = new Float32Array(edgeCount);
   * for (let i = 0; i < edgeCount; i++) {
   *   curvatures[i] = (Math.random() - 0.5) * 0.6; // Range -0.3 to 0.3
   * }
   * graph.setEdgeCurvatures(curvatures);
   * graph.enableCurvedEdges(); // Don't forget to enable!
   * ```
   */
  setEdgeCurvatures(curvatures: Float32Array): void {
    if (!this.state.loaded || !this.buffers || !this.state.parsedGraph) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Cannot set edge curvatures: graph not loaded",
      );
    }

    const expected = this.state.edgeCount;
    if (curvatures.length !== expected) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        `Expected ${expected} values for ${this.state.edgeCount} edges (1 per edge), got ${curvatures.length}`,
      );
    }

    const { device } = this.gpuContext;
    const edgeAttrs = this.state.parsedGraph.edgeAttributes;

    // Update CPU-side array
    // Edge attrs layout: [width, r, g, b, selected, hovered, curvature, reserved] per edge (8 floats)
    for (let i = 0; i < this.state.edgeCount; i++) {
      const curvature = curvatures[i];
      if (!Number.isNaN(curvature)) {
        edgeAttrs[i * 8 + 6] = curvature; // curvature is at offset 6
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
  // Public API - Node Borders
  // ==========================================================================

  /**
   * Configure node border rendering.
   *
   * @param config Partial border configuration to merge with current settings.
   *
   * @example
   * ```typescript
   * // Enable thick dark borders
   * graph.setNodeBorder({ enabled: true, width: 2.0, color: "#000000" });
   *
   * // Disable borders
   * graph.setNodeBorder({ enabled: false });
   *
   * // Just change color
   * graph.setNodeBorder({ color: "#ff0000" });
   * ```
   */
  setNodeBorder(config: Partial<_NodeBorderConfig>): void {
    // Merge with current config
    this.nodeBorderConfig = {
      ...this.nodeBorderConfig,
      ...config,
    };

    // Update GPU buffer
    this.updateRenderConfigBuffer();
  }

  /**
   * Get current node border configuration.
   *
   * @returns Current border configuration.
   */
  getNodeBorderConfig(): _NodeBorderConfig {
    return { ...this.nodeBorderConfig };
  }

  /**
   * Enable node borders.
   *
   * @param width Optional border width in pixels (default: current width).
   * @param color Optional border color as CSS/hex string (default: current color).
   */
  enableNodeBorder(width?: number, color?: string): void {
    this.setNodeBorder({
      enabled: true,
      ...(width !== undefined && { width }),
      ...(color !== undefined && { color }),
    });
  }

  /**
   * Disable node borders.
   */
  disableNodeBorder(): void {
    this.setNodeBorder({ enabled: false });
  }

  // ==========================================================================
  // Public API - Display Settings
  // ==========================================================================

  /**
   * Set the background color of the graph canvas.
   *
   * @param color - Color as hex string (e.g., "#0a0a0f") or RGBA object
   *
   * @example
   * ```typescript
   * // Set dark background
   * graph.setBackgroundColor("#0a0a0f");
   *
   * // Set light background
   * graph.setBackgroundColor("#ffffff");
   *
   * // Set with RGBA object
   * graph.setBackgroundColor({ r: 0.04, g: 0.04, b: 0.06, a: 1.0 });
   * ```
   */
  setBackgroundColor(color: string | { r: number; g: number; b: number; a?: number }): void {
    if (typeof color === "string") {
      // Parse hex color
      const hex = color.startsWith("#") ? color.slice(1) : color;
      if (hex.length >= 6) {
        this.backgroundColor = {
          r: parseInt(hex.slice(0, 2), 16) / 255,
          g: parseInt(hex.slice(2, 4), 16) / 255,
          b: parseInt(hex.slice(4, 6), 16) / 255,
          a: hex.length >= 8 ? parseInt(hex.slice(6, 8), 16) / 255 : 1.0,
        };
      }
    } else {
      this.backgroundColor = {
        r: color.r,
        g: color.g,
        b: color.b,
        a: color.a ?? 1.0,
      };
    }
  }

  /**
   * Get the current background color.
   *
   * @returns Current background color as RGBA object (0-1 range)
   */
  getBackgroundColor(): { r: number; g: number; b: number; a: number } {
    return { ...this.backgroundColor };
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

  // ============================================================================
  // Type-Based Styling API
  // ============================================================================

  /**
   * Set visual styles for node types.
   *
   * @param styles - Map of type names to node styles
   *
   * @example
   * ```typescript
   * graph.setNodeTypeStyles({
   *   person: { color: '#4CAF50', size: 1.2 },
   *   company: { color: '#2196F3', size: 1.5 },
   *   document: { color: '#FF9800', size: 0.8 }
   * });
   * ```
   */
  setNodeTypeStyles(styles: NodeTypeStyleMap): void {
    this.typeStyleManager.setNodeTypeStyles(styles);
    this.applyTypeStyles();
  }

  /**
   * Set visual styles for edge types.
   *
   * @param styles - Map of type names to edge styles
   *
   * @example
   * ```typescript
   * graph.setEdgeTypeStyles({
   *   friendship: { color: '#4CAF50', width: 2.0 },
   *   collaboration: { color: '#2196F3', width: 1.5, opacity: 0.8 },
   *   dependency: { color: '#FF5722', width: 1.0 }
   * });
   * ```
   */
  setEdgeTypeStyles(styles: EdgeTypeStyleMap): void {
    this.typeStyleManager.setEdgeTypeStyles(styles);
    this.applyTypeStyles();
  }

  /**
   * Get all defined node type names.
   *
   * @returns Array of node type names
   */
  getNodeTypes(): string[] {
    return this.typeStyleManager.getNodeTypes();
  }

  /**
   * Get all defined edge type names.
   *
   * @returns Array of edge type names
   */
  getEdgeTypes(): string[] {
    return this.typeStyleManager.getEdgeTypes();
  }

  /**
   * Clear all type-based styles.
   */
  clearTypeStyles(): void {
    this.typeStyleManager.clear();
    this.applyTypeStyles();
  }

  /**
   * Apply type-based styles to nodes and edges.
   * Called internally after type style changes.
   */
  private applyTypeStyles(): void {
    if (!this.state.loaded || !this.state.parsedGraph) return;

    const parsed = this.state.parsedGraph;
    const { device } = this.gpuContext;

    // Update node attributes buffer with type-based colors and sizes
    if (this.buffers && this.typeStyleManager.hasNodeStyles()) {
      const nodeCount = this.state.nodeCount;
      const nodeAttributes = new Float32Array(nodeCount * 8); // 8 floats per node

      for (let i = 0; i < nodeCount; i++) {
        const nodeType = parsed.nodeTypes?.[i];
        const style = this.typeStyleManager.resolveNodeStyle(nodeType);

        const baseOffset = i * 8;
        // Copy original radius (from parsed graph or default)
        const originalRadius = parsed.nodeAttributes[i * 8 + 0] || 8.0;
        nodeAttributes[baseOffset + 0] = originalRadius * style.size;
        // Color (RGBA)
        nodeAttributes[baseOffset + 1] = style.color[0];
        nodeAttributes[baseOffset + 2] = style.color[1];
        nodeAttributes[baseOffset + 3] = style.color[2];
        nodeAttributes[baseOffset + 4] = style.color[3];
        // Copy remaining attributes (border color, border width, flags)
        nodeAttributes[baseOffset + 5] = parsed.nodeAttributes[i * 8 + 5] || 0;
        nodeAttributes[baseOffset + 6] = parsed.nodeAttributes[i * 8 + 6] || 0;
        nodeAttributes[baseOffset + 7] = parsed.nodeAttributes[i * 8 + 7] || 0;
      }

      device.queue.writeBuffer(
        this.buffers.nodeAttributes,
        0,
        nodeAttributes.buffer,
      );
    }

    // Update edge attributes buffer with type-based colors and widths
    if (this.buffers && this.typeStyleManager.hasEdgeStyles()) {
      const edgeCount = this.state.edgeCount;
      const edgeAttributes = new Float32Array(edgeCount * 8); // 8 floats per edge

      for (let i = 0; i < edgeCount; i++) {
        const edgeType = parsed.edgeTypes?.[i];
        const style = this.typeStyleManager.resolveEdgeStyle(edgeType);

        const baseOffset = i * 8;
        // Layout: [width, r, g, b, selected, hovered, curvature, reserved]
        edgeAttributes[baseOffset + 0] = style.width;
        edgeAttributes[baseOffset + 1] = style.color[0]; // r
        edgeAttributes[baseOffset + 2] = style.color[1]; // g
        edgeAttributes[baseOffset + 3] = style.color[2]; // b
        edgeAttributes[baseOffset + 4] = 0; // selected
        edgeAttributes[baseOffset + 5] = 0; // hovered
        edgeAttributes[baseOffset + 6] = 0; // curvature (default: straight)
        edgeAttributes[baseOffset + 7] = 0; // reserved
      }

      device.queue.writeBuffer(
        this.buffers.edgeAttributes,
        0,
        edgeAttributes.buffer,
      );
    }
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
        // Use panScreen since delta is in screen pixels, not graph units
        const dx = this.lastPanPosition.x - e.screenPosition.x;
        const dy = this.lastPanPosition.y - e.screenPosition.y;
        this.viewport.panScreen(dx, dy);
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
    const posVec2 = new Float32Array([x, y]);

    // Write vec2 at the node's offset (8 bytes per vec2)
    device.queue.writeBuffer(this.buffers.positions, idx * 8, posVec2);

    // Also update simulation buffers if they exist
    if (this.simBuffers) {
      device.queue.writeBuffer(this.simBuffers.positions, idx * 8, posVec2);
      device.queue.writeBuffer(this.simBuffers.positionsOut, idx * 8, posVec2);
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

    // Remove visibility change listener
    if (this.visibilityChangeHandler && typeof document !== "undefined") {
      document.removeEventListener("visibilitychange", this.visibilityChangeHandler);
      this.visibilityChangeHandler = null;
    }

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

    // Destroy metaball intensity buffer
    this.metaballIntensityBuffer?.destroy();
    this.metaballIntensityBuffer = null;

    // Destroy default intensity buffer
    this.defaultIntensityBuffer?.destroy();
    this.defaultIntensityBuffer = null;

    // Destroy render config buffer
    this.renderConfigBuffer?.destroy();
    this.renderConfigBuffer = null;

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
      this.simBuffers.positions.destroy();
      this.simBuffers.positionsOut.destroy();
      this.simBuffers.velocities.destroy();
      this.simBuffers.velocitiesOut.destroy();
      this.simBuffers.forces.destroy();
      this.simBuffers.edgeSources.destroy();
      this.simBuffers.edgeTargets.destroy();
      this.simBuffers.clearUniforms.destroy();
      this.simBuffers.repulsionUniforms.destroy();
      this.simBuffers.springUniforms.destroy();
      this.simBuffers.integrationUniforms.destroy();
      this.simBuffers.readback.destroy();
      this.simBuffers = null;
    }
    this.simBindGroups = null;

    // Destroy algorithm-specific buffers
    this.algorithmBuffers?.destroy();
    this.algorithmBuffers = null;
    this.algorithmBindGroups = null;

    // Destroy collision buffers
    if (this.collisionBuffers) {
      destroyCollisionBuffers(this.collisionBuffers);
      this.collisionBuffers = null;
    }
    this.collisionBindGroup = null;

    // Destroy grid collision buffers
    if (this.gridCollisionBuffers) {
      destroyGridCollisionBuffers(this.gridCollisionBuffers);
      this.gridCollisionBuffers = null;
    }
    this.gridCollisionBindGroups = null;
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
