/**
 * GraphMother Main Class
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
import { destroyGPUContext, type GPUContext, resizeGPUContext } from "../webgpu/context.ts";
import { toArrayBuffer } from "../webgpu/buffer_utils.ts";
import { ErrorCode, GraphMotherError } from "../errors.ts";
import { createEventEmitter, type EventEmitter, Events } from "../events/emitter.ts";
import { createViewport, type Viewport } from "../viewport/viewport.ts";
import { createViewportUniformBuffer, type ViewportUniformBuffer } from "../viewport/uniforms.ts";
import {
  clampTag,
  clampWeight,
  type ParsedGraph,
  parseGraphInput,
  type ParserConfig,
} from "../graph/parser.ts";
import { parseGraphTypedInput, type TypedParserConfig } from "../graph/typed_parser.ts";
import { initializePositions, needsInitialization } from "../graph/initialize.ts";
import {
  bubbleUploadChanged,
  deriveHierarchy,
  HIERARCHY_ROOT,
  type HierarchyColumns,
  type RetainedHierarchy,
  retainSuppliedHierarchy,
  selectContainmentEdges,
} from "../graph/hierarchy.ts";
import { commitNodeMass, NODE_MASS_UNIT } from "../lod/mass.ts";
import {
  aggregateEdges,
  buildBundleInstances,
  type BundleInstanceScratch,
  type BundleStyle,
  type EdgeAggregation,
  EdgeOpacityMask,
} from "../lod/edge_aggregation.ts";
import { forEachSlotRun } from "../lod/runs.ts";
import { type ProxyRadiusHost, ProxyRadiusTable } from "../lod/proxy_radius.ts";
import type { CrossfadeScheduler } from "../lod/crossfade.ts";
import {
  DEFAULT_LOD_CONFIG,
  type LodConfig,
  LODController,
  type LodHost,
  type LodPolicy,
} from "../lod/mod.ts";
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
  EDGE_FLOW_PRESETS,
  type EdgeFlowPreset,
} from "../renderer/edge_flow.ts";
import { parseColorToRGB, parseColorToRGBA } from "../utils/color.ts";
import {
  DEFAULT_NODE_BORDER_CONFIG as _DEFAULT_NODE_BORDER_CONFIG,
  type NodeBorderConfig as _NodeBorderConfig,
} from "../config/node_border.ts";
import { createRenderLoop, type FrameStats, type RenderLoop } from "../renderer/render_loop.ts";
import { RenderPauseGate } from "../renderer/render_pause.ts";
import { createSimulationController, type SimulationController } from "../simulation/controller.ts";
import {
  DEFAULT_FORCE_CONFIG,
  type FullForceConfig,
  validateForceConfig,
} from "../simulation/config.ts";
import { createEdgeIndicesBuffer } from "../graph/parser.ts";
import { boundsCenter, fitBoundsScale } from "../viewport/transforms.ts";
import {
  BindGroupParitySets,
  copyEdgesToSimulation,
  copyPositionsToReadback,
  copyPositionsToSimulation,
  createSimulationBindGroups,
  createSimulationBuffers,
  createSimulationPipeline,
  DEAD_SLOT_RADIUS,
  LOD_EDGE_SET_WORDS_PER_EDGE,
  lodEdgeDispatchCount,
  NODE_FLAG_DEAD,
  NODE_FLAG_HIDDEN_LOD,
  NODE_FLAG_PINNED,
  readbackPositions,
  recordSimulationStepWithOptions,
  releaseEdgeBundles,
  type SimulationBindGroups,
  type SimulationBuffers,
  type SimulationPipeline,
  startSettlingTelemetry,
  updateSimulationUniforms,
  uploadEdgeBundles,
  uploadLiveIndices,
  writeIdentityLiveIndices,
  writeOpaqueNodeAlpha,
  writeUnitNodeMass,
} from "../simulation/pipeline.ts";
import { activeIndicesUnchanged, deriveActiveIndices } from "../simulation/active_set.ts";
import {
  type CollisionBindGroup,
  type CollisionBuffers,
  type CollisionPipeline,
  createCollisionBindGroup,
  createCollisionBuffers,
  createCollisionPipeline,
  createGridCollisionBindGroups,
  createGridCollisionBuffers,
  createGridCollisionPipeline,
  destroyCollisionBuffers,
  destroyGridCollisionBuffers,
  type GridCollisionBindGroups,
  type GridCollisionBuffers,
  type GridCollisionPipeline,
  recordCollisionPass,
  recordGridCollisionPass,
  updateCollisionUniforms,
  updateGridCollisionUniforms,
  uploadNodeSizes,
} from "../simulation/collision.ts";
import { createHitTester, type HitTester } from "../interaction/hit_test.ts";
import { HoverTracker } from "../interaction/hover.ts";
import { NodeDragController } from "../interaction/node_drag.ts";
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
  type LabelConfig,
  type LabelData,
  type LabelNodeSource,
  // Labels layer
  LabelsLayer,
  type LabelsRenderContext,
  type Layer as VisualizationLayer,
  type LayerInfo,
  type LayerManager,
  type MetaballConfig,
  // Metaball layer
  type MetaballLayer,
  type MetaballRenderContext,
} from "../layers/mod.ts";
import { StreamIntensityCache } from "../layers/stream_intensity.ts";
import {
  type AlgorithmBindGroups,
  type AlgorithmBuffers,
  type AlgorithmPipelines,
  type AlgorithmRenderContext,
  CodebaseLayoutAlgorithm,
  CommunityLayoutAlgorithm,
  type ForceAlgorithm,
  type ForceAlgorithmType,
  getAlgorithmRegistry,
  initializeBuiltinAlgorithms,
  RelativityAtlasAlgorithm,
  RelativityAtlasBuffers,
  supportsAlgorithmOnDevice,
  TidyTreeAlgorithm,
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
import {
  type CardNodeSource,
  type CardProvider,
  type CardSyncEntry,
  DomCardOverlay,
  type DomOverlayConfig,
  externalIdForSlot,
  slotForExternalId,
} from "../overlay/mod.ts";
import type { IdLike } from "../graph/id_map.ts";
import { growCapacity, initialCapacity } from "./buffer_capacity.ts";
import {
  aliasParsedGraphToState,
  collisionRadiusColumn,
  compactNodeColumn,
  DEFAULT_COLLISION_RADIUS,
  growSlotColumn,
  MutableGraphState,
  NODE_ATTR_BYTES,
  NODE_ATTR_FLOATS,
} from "./graph_state.ts";

// Default well radius for non-bubble mode (matches density field default splat in grid cells)
const DEFAULT_WELL_RADIUS = 0.0;

// Where a DOM card sits when its node has no position left to read — the slot
// was freed under it, and the overlay releases the card on the same frame.
const CARD_ORIGIN: Vec2 = { x: 0, y: 0 };

// The flag bits that take a slot out of hit testing. Both mean "not on the
// screen": a freed slot holds no node, and a node the LOD cut folded away is
// represented by a proxy the pointer must reach instead. It coincides with
// NODE_FLAGS_INERT today but answers a different question — one is about what
// the simulation integrates, this is about what the pointer can name.
const NODE_FLAGS_UNHITTABLE = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD;

/** Screen-space travel a press may make and still read as a click, in CSS pixels. */
const CLICK_THRESHOLD_PX = 5;

/** Whether a release is close enough to its press for the pair to be a click. */
function isClickDistance(press: Vec2, release: Vec2): boolean {
  const dx = release.x - press.x;
  const dy = release.y - press.y;
  return Math.sqrt(dx * dx + dy * dy) < CLICK_THRESHOLD_PX;
}

/**
 * WASM engine interface for graph structure and layout.
 * This matches the GraphMotherWasm API exposed by the WASM module.
 */
interface WasmEngine {
  /** Clear all graph data */
  clear(): void;
  /**
   * Add a single node at position. Returns the node's id, which by contract
   * equals its slot index: the engine reuses freed slots in the same LIFO
   * order as MutableGraphState.allocateNodeSlot, so ids and GPU slots match.
   */
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
  /** Compute tidy tree layout from graph's own edges */
  computeTreeLayoutFromGraph(
    rootId: number,
    levelSeparation: number,
    siblingSeparation: number,
    subtreeSeparation: number,
    radial: boolean,
  ): Float32Array;
  /** Get upper bound on node indices (max index + 1) */
  nodeBound(): number;
  /** Detect communities using Louvain algorithm. Returns assignments array with community count as last element. */
  detectCommunities(
    resolution: number,
    maxIterations: number,
    minModularityGain: number,
  ): Uint32Array;
  /** Detect communities and compute layout in a single call (legacy — prefer detectCommunities) */
  computeCommunityLayoutFromGraph(
    resolution: number,
    maxIterations: number,
    communitySpacing: number,
    nodeSpacing: number,
    spreadFactor: number,
  ): Float32Array;
  /** Compute codebase layout using the graph's own edges */
  computeCodebaseLayoutFromGraph(
    nodeCategories: Uint8Array,
    rootId: number,
    directoryPadding: number,
    filePadding: number,
    spreadFactor: number,
  ): Float32Array;
  /**
   * Derive the containment hierarchy from the engine's own edges.
   *
   * Corrupted by any non-containment edge — prefer
   * {@link WasmEngine.computeBubbleDataFromEdges}. Same 4-column return.
   */
  computeBubbleData(baseRadius: number, padding: number): Float32Array;
  /**
   * Derive the containment hierarchy from containment-only edges.
   *
   * Returns `4 * nodeBound` floats in four concatenated per-slot columns:
   * `[wellRadius…, depth…, parent…, subtreeSize…]`, where the parent column
   * holds -1 for a forest root. Pass 0xFFFFFFFF as rootId for auto-detection.
   */
  computeBubbleDataFromEdges(
    containmentEdges: Uint32Array,
    rootId: number,
    baseRadius: number,
    padding: number,
  ): Float32Array;
  /**
   * Aggregate the edge set against a semantic-LOD visible cut.
   *
   * Returns `[liveCount, bundleCount, liveEdges…, (source, target, weight)…]`;
   * see `lod/edge_aggregation.ts` for the decode.
   */
  aggregateLodEdges(
    edgeSources: Uint32Array,
    edgeTargets: Uint32Array,
    parent: Uint32Array,
    visible: Uint8Array,
  ): Uint32Array;
  /** Release the WASM-side engine and its linear-memory allocations */
  free(): void;
}

/**
 * GraphMother configuration
 */
export interface GraphMotherConfig {
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
 * Floats per edge in the render attribute buffer:
 * width, r, g, b, selected, hovered, curvature, opacity.
 */
const EDGE_ATTR_FLOATS = 8;

/** Index of the opacity channel within an edge's attribute row. */
const EDGE_ATTR_OPACITY = 7;

/**
 * How a collapsed-edge bundle is drawn.
 *
 * One neutral style for every bundle: a bundle stands for a set of edges of
 * possibly different types, so borrowing one member's colour would claim more
 * than the aggregation knows. Width scales with the bundle count inside
 * `buildBundleInstances`.
 */
const LOD_BUNDLE_STYLE: BundleStyle = {
  width: 1.5,
  color: [0.55, 0.6, 0.7],
  opacity: 0.85,
};

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
 * Main GraphMother class
 */
export class GraphMother {
  // Configuration
  private readonly gpuContext: GPUContext;
  private readonly wasmEngine: WasmEngine | null;
  private readonly canvas: HTMLCanvasElement;
  private readonly debug: boolean;

  // State
  private state: GraphState;
  private graphState: MutableGraphState | null = null;
  private disposed: boolean = false;

  /**
   * Retained containment hierarchy, or null when it has not been built since
   * the last topology change. Rebuilt lazily by {@link getHierarchy}.
   */
  private hierarchy: RetainedHierarchy | null = null;
  /** Producer-supplied columns from the last load, if any. */
  private suppliedHierarchy: HierarchyColumns | null = null;
  /**
   * Node and edge counts the supplied columns arrived with.
   *
   * Supplied columns describe the snapshot as loaded. Any add or remove — of a
   * node or an edge — moves one of these counts, at which point the producer's
   * tree no longer describes the graph and core derives its own instead.
   */
  private suppliedHierarchyStamp: { nodes: number; edges: number } | null = null;
  /** Whether the "no WASM engine, no hierarchy" warning has already been issued. */
  private hierarchyUnavailableWarned = false;
  /**
   * CPU mirror of `simBuffers.nodeMass`, or null while the GPU buffer is known
   * to hold unit mass everywhere (fresh allocation). Exists so
   * {@link uploadNodeMass} can write only the range a transition actually
   * changed instead of the whole capacity.
   */
  private nodeMassShadow: Float32Array | null = null;
  /**
   * CPU mirror of `simBuffers.liveIndices`, sized to the buffer's capacity, or
   * null while the GPU buffer still holds the identity list a fresh allocation
   * writes. Retained so {@link refreshActiveIndices} can tell a flag change
   * that moved the active set from one that did not, and skip the upload for
   * the latter — pinning a node runs the same choke point as hiding one.
   */
  private activeIndexShadow: Uint32Array | null = null;
  /** Render radii borrowed by the current collapsed proxies. */
  private readonly lodProxies = new ProxyRadiusTable();
  /** Opacities borrowed from the source edges the current cut hides. */
  private readonly lodEdgeOpacity = new EdgeOpacityMask();
  /** Reused instance rows for the bundle draw, so a transition allocates nothing. */
  private readonly lodBundleScratch: BundleInstanceScratch = {
    indices: new Uint32Array(0),
    attributes: new Float32Array(0),
  };
  /**
   * Buffers the collapsed-edge bundles are drawn from, or null before the
   * first aggregation.
   *
   * Allocated at the render edge capacity and then only ever written: a bundle
   * count can never exceed the source edge count — every bundle stands for at
   * least one edge — so no band transition can outgrow them, and the bind group
   * built against them survives every collapse and expand. They are created
   * lazily because a graph that never collapses anything would otherwise carry
   * 40 bytes per edge it never reads.
   */
  private lodEdgeRenderBuffers:
    | { indices: GPUBuffer; attributes: GPUBuffer; capacity: number }
    | null = null;
  /** Bundle instances to draw this frame; 0 when no aggregation is live. */
  private lodBundleDrawCount = 0;
  /** Whether the "no WASM engine, no edge aggregation" warning has been issued. */
  private edgeAggregationUnavailableWarned = false;

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
  private viewportBindGroup: GPUBindGroup | null = null;
  private renderConfigBindGroup: GPUBindGroup | null = null;
  private renderConfigBuffer: GPUBuffer | null = null;

  // Node border configuration
  private nodeBorderConfig: _NodeBorderConfig = { ..._DEFAULT_NODE_BORDER_CONFIG };

  // Birth pulse animation configuration
  private birthPulseConfig = {
    enabled: true,
    duration: 0.5,
    intensity: 0.5,
    pulseColor: [1, 1, 1] as [number, number, number],
  };
  /** Animation time of the most recent birth pulse (for render loop dirty tracking) */
  private lastBirthTime = 0;

  // Background color (RGBA 0-1)
  private backgroundColor: { r: number; g: number; b: number; a: number } = {
    r: 0.04,
    g: 0.04,
    b: 0.06,
    a: 1.0,
  };

  // Visual defaults seeded from the constructor `config` (GraphConfig) option.
  // Held rather than applied once because the parsers consume them on every
  // load(), so a reload keeps the caller's defaults.
  private parserConfig: ParserConfig = {};
  private typedParserConfig: TypedParserConfig = {};

  // GPU Simulation resources
  private simBuffers: SimulationBuffers | null = null;

  /**
   * The ping-pong parity state machine (packages/core/src/simulation/pipeline.ts).
   *
   * Owns the parity counter, every parity-indexed bind-group set, and the
   * swap-and-flip step; this class only delegates. Nothing about the mechanism
   * is reimplemented here, so it is covered by the tests for that unit
   * (tests/unit/parity_sets_test.ts, tests/gpu/bind_group_parity_test.ts)
   * rather than by hand-reading graph.ts.
   *
   * simBuffers is swapped in place by `paritySets.advance()`, so
   * `simBuffers.positions` is always the buffer the simulation reads this frame.
   */
  private readonly paritySets = new BindGroupParitySets(() => this.simBuffers);

  private readonly simBindGroupSlot = this.paritySets.slot<SimulationBindGroups>("simulation");
  private readonly nodeBindGroupSlot = this.paritySets.slot<GPUBindGroup>("node render");
  private readonly edgeBindGroupSlot = this.paritySets.slot<GPUBindGroup>("edge render");
  private readonly lodEdgeBindGroupSlot = this.paritySets.slot<GPUBindGroup>("lod edge render");
  private readonly algorithmBindGroupSlot = this.paritySets.slot<AlgorithmBindGroups>("algorithm");
  private readonly collisionBindGroupSlot = this.paritySets.slot<CollisionBindGroup>("collision");
  private readonly gridCollisionBindGroupSlot = this.paritySets.slot<GridCollisionBindGroups>(
    "grid collision",
  );

  // Force algorithm resources
  private currentAlgorithm: ForceAlgorithm | null = null;
  private algorithmPipelines: AlgorithmPipelines | null = null;
  private algorithmBuffers: AlgorithmBuffers | null = null;

  // Collision detection resources
  private collisionPipeline: CollisionPipeline | null = null;
  private collisionBuffers: CollisionBuffers | null = null;

  // Grid collision resources (O(n·k) spatial hash for large graphs)
  private gridCollisionPipeline: GridCollisionPipeline | null = null;
  private gridCollisionBuffers: GridCollisionBuffers | null = null;
  private maxNodeRadius: number = 5.0;
  private frameBounds: BoundingBox | undefined;

  // ------------------------------------------------------------------------
  // Bind groups for the current ping-pong parity.
  // Each delegates to its ParitySlot, which selects the variant matching the
  // current parity and throws if that variant was not built from the buffer
  // the simulation is reading this frame. Nothing here allocates, so these are
  // safe to touch on the per-frame path.
  // ------------------------------------------------------------------------

  private get simBindGroups(): SimulationBindGroups | null {
    return this.simBindGroupSlot.current;
  }

  private get nodeBindGroup(): GPUBindGroup | null {
    return this.nodeBindGroupSlot.current;
  }

  private get edgeBindGroup(): GPUBindGroup | null {
    return this.edgeBindGroupSlot.current;
  }

  private get algorithmBindGroups(): AlgorithmBindGroups | null {
    return this.algorithmBindGroupSlot.current;
  }

  private get collisionBindGroup(): CollisionBindGroup | null {
    return this.collisionBindGroupSlot.current;
  }

  private get gridCollisionBindGroups(): GridCollisionBindGroups | null {
    return this.gridCollisionBindGroupSlot.current;
  }

  // Interaction
  private hitTester: HitTester;
  private pointerManager: PointerManager | null = null;
  private selectedNodes: Set<NodeId> = new Set();
  private selectedEdges: Set<EdgeId> = new Set();
  private hover: HoverTracker;
  /**
   * The one node drag, whichever gesture produced it: a pointer on the sprite,
   * or a pointer on the drag handle of the card standing in for it. Pointer
   * capture keeps the two from overlapping, and both leave the node pinned.
   */
  private readonly nodeDrag = new NodeDragController({
    setPinned: (node, pinned) => this.setNodePinnedState(node, pinned),
    setPosition: (node, x, y) => this.setNodePosition(node, x, y),
    emit: (event) => this.events.emit(event),
    now: () => Date.now(),
  });
  private dragStartScreenPosition: Vec2 | null = null;
  /** Where a press on empty canvas landed, or null when the press hit a node. */
  private backgroundPressPosition: Vec2 | null = null;
  private lastClick: { nodeId: NodeId; timestamp: number } | null = null;
  private pinnedNodes: Set<NodeId> = new Set();

  // Viewport panning state
  private isPanning: boolean = false;
  private lastPanPosition: Vec2 | null = null;

  // Position sync (GPU -> JS for hit testing)
  private syncFrameCounter: number = 0;
  private syncInProgress: boolean = false;
  private readonly SYNC_INTERVAL: number = 5; // Sync every N frames

  // Convergence detection — monitors actual node movement instead of using
  // a fixed alpha decay timer. When max displacement per frame drops below
  // threshold, the graph has genuinely settled and alpha is clamped to 0.
  private prevSyncPositionsX: Float32Array | null = null;
  private prevSyncPositionsY: Float32Array | null = null;
  private convergenceCheckCount: number = 0;
  private readonly CONVERGENCE_THRESHOLD: number = 0.15; // max px/frame displacement
  private readonly CONVERGENCE_CHECKS_REQUIRED: number = 3; // consecutive checks needed

  // Render activity tracking — when false, skip GPU work entirely.
  // Set true by: simulation compute, viewport changes, selection, hover, resize.
  // Cleared after each frame. Edge flow animation keeps this permanently true.
  private renderDirty: boolean = true;

  // Layer system
  private layerManager: LayerManager;

  /**
   * DOM card overlay — a peer of the layer manager rather than a Layer, since
   * a Layer renders into a GPUCommandEncoder and this renders into the DOM.
   * Created by the first {@link GraphMother.setDomOverlay} call and driven
   * from the same two signals the labels layer is: viewport changes and the
   * per-frame tick.
   */
  private domOverlay: DomCardOverlay | null = null;

  /**
   * Semantic-LOD controller. Created by the first LOD API call, and inert
   * until {@link GraphMother.setLodConfig} enables it, so a graph that never
   * asks for LOD pays nothing and behaves exactly as it did before it existed.
   */
  private lodController: LODController | null = null;

  /**
   * Bumped whenever the per-node inputs to the GPU label set change other than
   * by movement: the LOD cut and the collapsed-proxy radii. The card set is a
   * second counter, on the overlay, and the two are summed for the layer.
   */
  private labelNodeStateVersion = 0;

  // Value stream system
  private streamManager: StreamManager;

  // Original colors backed up before stream overrides, keyed by node index.
  // Used to restore base colors when streams are cleared, disabled, or removed.
  private streamColorBackups = new Map<number, [number, number, number]>();

  // Type-based styling system
  private typeStyleManager: TypeStyleManager;

  /**
   * Per-layer caches of the stream-derived per-node intensity upload, keyed by
   * layer ID. Each recomputes only when its stream, the stream's mutation
   * version, the node count, or the colour-scale domain changed — without this
   * a heatmap or metaball bound to a stream costs an O(nodeCount) loop plus a
   * full buffer upload on every frame.
   */
  private streamIntensityCaches = new Map<string, StreamIntensityCache>();

  // Default intensity buffer (all 1.0 values for density mode)
  private defaultIntensityBuffer: GPUBuffer | null = null;

  // Visibility change handling - pause simulation and rendering when tab hidden
  private visibilityChangeHandler: (() => void) | null = null;
  private wasRunningBeforeHidden: boolean = false;
  /** Arbitrates the host's render pause against the visibility-driven one. */
  private renderPause: RenderPauseGate;

  constructor(config: GraphMotherConfig) {
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
        this.markRenderDirty();
        // Viewport has no gesture-end signal; the overlay debounces here.
        this.domOverlay?.viewportChanged();
        // The LOD cut is a function of zoom, so this is its only input. It
        // marks itself dirty and decides on the next frame whether the change
        // was large enough to be worth an evaluation.
        this.lodController?.viewportChanged();
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

    // Nothing marks the scene dirty while frames are not being presented, so
    // the first frame back must be a full draw rather than a dirty-gate skip.
    this.renderPause = new RenderPauseGate({
      pause: () => this.renderLoop.pause(),
      resume: () => {
        this.renderLoop.resume();
        this.markRenderDirty();
      },
    });

    // Create simulation controller
    this.simulationController = createSimulationController();

    // Bridge the internal simulation events onto the public emitter. The
    // internal controller is clock-free and counts ticks; the public contract
    // speaks timestamps and "iteration(s)".
    this.simulationController.events.on("tick", ({ alpha, tickCount }) => {
      this.events.emit({
        type: "simulation:tick",
        timestamp: Date.now(),
        alpha,
        iteration: tickCount,
      });
    });
    this.simulationController.events.on("end", ({ tickCount }) => {
      this.events.emit({
        type: "simulation:end",
        timestamp: Date.now(),
        iterations: tickCount,
      });
    });

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

    // Hit testing brute-forces the CPU position shadow (refreshed from the
    // GPU every SYNC_INTERVAL frames). The WASM R-tree is not wired up
    // because its position copy is never synced from the simulation — see
    // populateWasmEngine for the full rationale.

    this.hover = new HoverTracker({
      nodeAt: (screenX, screenY) => this.getNodeAtPosition(screenX, screenY),
      edgeAt: (screenX, screenY) => this.getEdgeAtPosition(screenX, screenY),
      toGraph: (screenX, screenY) => this.viewport.screenToGraph(screenX, screenY),
      // The edge scan costs a pass over every edge and produces nothing but
      // these two events, so nobody listening means nothing to compute.
      edgeHoverWanted: () =>
        this.events.hasListeners("edge:hoverenter") ||
        this.events.hasListeners("edge:hoverleave"),
      onNodeEnter: (nodeId, position) => {
        this.syncNodeHoverToGPU(nodeId, true);
        // A carded node is a DOM element the pointer may never touch — the
        // hit test runs against the sprite underneath — so `:hover` cannot
        // stand in for this and the card has to be told.
        this.domOverlay?.notify(nodeId, { kind: "hover", hovered: true });
        this.markRenderDirty();
        this.events.emit({
          type: "node:hoverenter",
          timestamp: Date.now(),
          nodeId,
          position,
        });
      },
      onNodeLeave: (nodeId) => {
        this.syncNodeHoverToGPU(nodeId, false);
        this.domOverlay?.notify(nodeId, { kind: "hover", hovered: false });
        this.markRenderDirty();
        this.events.emit({
          type: "node:hoverleave",
          timestamp: Date.now(),
          nodeId,
        });
      },
      onEdgeEnter: (edgeId, position) => {
        this.markRenderDirty();
        this.events.emit({
          type: "edge:hoverenter",
          timestamp: Date.now(),
          edgeId,
          position,
        });
      },
      onEdgeLeave: (edgeId) => {
        this.markRenderDirty();
        this.events.emit({
          type: "edge:hoverleave",
          timestamp: Date.now(),
          edgeId,
        });
      },
    });

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

    // Apply caller-supplied graph-wide visual defaults
    this.applyGraphConfig(config.config);

    // Set up visibility change handling - pause simulation when tab is hidden
    this.setupVisibilityChangeHandler();

    if (this.debug) {
      console.log("GraphMother instance created");
    }
  }

  /**
   * Seed graph-wide visual defaults from the constructor `config` option.
   *
   * `backgroundColor` is applied immediately; the node/edge defaults are
   * stashed for the parsers, which consume them on every load().
   */
  private applyGraphConfig(config: Partial<GraphConfig> | undefined): void {
    if (!config) return;

    if (config.backgroundColor !== undefined) {
      this.setBackgroundColor(config.backgroundColor);
    }

    if (config.nodeDefaultRadius !== undefined) {
      this.parserConfig.defaultNodeRadius = config.nodeDefaultRadius;
      this.typedParserConfig.defaultNodeRadius = config.nodeDefaultRadius;
    }

    if (config.nodeDefaultColor !== undefined) {
      const [r, g, b, a] = parseColorToRGBA(config.nodeDefaultColor);
      this.parserConfig.defaultNodeColor = { r, g, b, a };
      this.typedParserConfig.defaultNodeColor = [r, g, b];
    }

    if (config.edgeDefaultWidth !== undefined) {
      this.parserConfig.defaultEdgeWidth = config.edgeDefaultWidth;
      this.typedParserConfig.defaultEdgeWidth = config.edgeDefaultWidth;
    }

    if (config.edgeDefaultColor !== undefined) {
      const [r, g, b, a] = parseColorToRGBA(config.edgeDefaultColor);
      this.parserConfig.defaultEdgeColor = { r, g, b, a };
      this.typedParserConfig.defaultEdgeColor = [r, g, b];
    }
  }

  /**
   * Set up visibility change handling to pause simulation and rendering when
   * the tab is hidden. This saves resources when the user switches tabs.
   *
   * A hidden tab gets throttled or stopped rAF callbacks anyway; the point of
   * pausing rendering here is that the two suspensions stay consistent, and
   * that the gate holds the pause across a `load` that restarts the loop.
   */
  private setupVisibilityChangeHandler(): void {
    if (typeof document === "undefined") return;

    this.visibilityChangeHandler = () => {
      // The gate keeps a host pause outstanding across both edges.
      this.renderPause.setHidden(document.hidden);

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
    // - time: f32 (4 bytes) + birth_pulse_duration: f32 (4 bytes) + birth_pulse_intensity: f32 (4 bytes) + pad: f32 (4 bytes) = 16 bytes
    // - pulse_color: vec3<f32> (12 bytes) + pad: f32 (4 bytes) = 16 bytes
    // Total: 80 bytes
    this.renderConfigBuffer = device.createBuffer({
      label: "Render Config Uniform Buffer",
      size: 80,
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
   * Mark that a visual redraw is needed this frame.
   * Called by viewport changes, selection, hover, resize, and any visual mutation.
   */
  private markRenderDirty(): void {
    this.renderDirty = true;
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
    const data = new ArrayBuffer(80);
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

    // Birth pulse animation (12-15)
    floatView[12] = 0.0; // time (updated per-frame in renderFrame)
    floatView[13] = this.birthPulseConfig.duration; // birth_pulse_duration
    floatView[14] = this.birthPulseConfig.enabled ? this.birthPulseConfig.intensity : 0.0; // birth_pulse_intensity (0 when disabled)
    floatView[15] = 0.0; // _pad3

    // Pulse color for looping pulses (16-18), _pad4 (19)
    const pc = this.birthPulseConfig.pulseColor;
    floatView[16] = pc[0];
    floatView[17] = pc[1];
    floatView[18] = pc[2];
    floatView[19] = 0.0; // _pad4

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

    // Simulation must cover ALL occupied slots (live + holes), like rendering
    // does — using the live nodeCount would skip live nodes in high slots
    // after removals, freezing them. Hole slots are inert: they carry the
    // dead flag in nodeFlags and a negative collision radius.
    const simNodeCount = this.graphState?.nodeHighWater ?? this.state.nodeCount;

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
      simNodeCount,
      this.state.edgeCount,
      alpha,
      effectiveForceConfig,
      this.currentAlgorithm?.prefersAdaptiveSpeed,
    );

    // Compute bounds once per frame for all consumers (algorithm context, collision grid).
    // CPU-side position arrays are synced from GPU every SYNC_INTERVAL frames, so bounds
    // may be slightly stale. The computeBoundsFromPositions function adds a margin for drift.
    this.frameBounds = this.state.parsedGraph
      ? computeBoundsFromPositions(
        this.state.parsedGraph.positionsX,
        this.state.parsedGraph.positionsY,
        simNodeCount,
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
          "CRITICAL: Position data corrupted (all NaN/Infinity). Stopping simulation.",
        );
        this.simulationController.stop();
        return;
      }

      // simBuffers is always in the current parity's orientation, so this
      // context matches the bind groups the parity slots select.
      this.currentAlgorithm.updateUniforms(
        device,
        this.algorithmBuffers,
        this.buildAlgorithmContext(this.simBuffers, bounds),
      );
    }

    // Record simulation compute passes with custom algorithm for repulsion
    const algorithmHandlesSprings = this.currentAlgorithm?.handlesSprings ?? false;
    recordSimulationStepWithOptions(
      encoder,
      this.simulationPipeline,
      this.simBindGroups,
      simNodeCount,
      this.state.edgeCount,
      {
        recordRepulsionPass:
          this.currentAlgorithm && this.algorithmPipelines && this.algorithmBindGroups
            ? (enc) => {
              this.currentAlgorithm!.recordRepulsionPass(
                enc,
                this.algorithmPipelines!,
                this.algorithmBindGroups!,
                simNodeCount,
              );
            }
            : undefined,
        skipSprings: algorithmHandlesSprings,
        // Kept in step with the uniform written above by both reading the
        // same field; the list itself is maintained by refreshActiveIndices.
        activeCount: this.simBuffers.activeCount,
        // Undefined unless an LOD aggregation is uploaded, in which case the
        // spring pass runs over the visible edges plus the bundles instead of
        // over every edge. Same source as the two counts the uniform carries.
        lodEdgeCount: lodEdgeDispatchCount(this.simBuffers),
      },
    );

    // Record collision detection pass (after integration, if enabled)
    if (
      this.forceConfig.collisionEnabled &&
      this.collisionPipeline &&
      this.collisionBuffers &&
      this.collisionBindGroup
    ) {
      const nodeCount = simNodeCount;
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
            this.simBuffers.activeCount,
          );
          recordGridCollisionPass(
            encoder,
            this.gridCollisionPipeline!,
            this.gridCollisionBindGroups!,
            this.gridCollisionBuffers!,
            nodeCount,
            this.forceConfig.collisionIterations,
            this.simBuffers.activeCount,
          );
        } else {
          // Bounds unavailable — fall back to tiled collision
          updateCollisionUniforms(
            device,
            this.collisionBuffers,
            nodeCount,
            this.forceConfig,
            this.simBuffers.activeCount,
          );
          recordCollisionPass(
            encoder,
            this.collisionPipeline,
            this.collisionBindGroup,
            nodeCount,
            this.forceConfig.collisionIterations,
            true,
            this.simBuffers.activeCount,
          );
        }
      } else {
        // Tiled/simple collision: O(n^2) for small graphs (<=5000 nodes).
        // The resolve dispatch tracks the LOD cut; the same field feeds the
        // uniform, so the two can never disagree.
        updateCollisionUniforms(
          device,
          this.collisionBuffers,
          nodeCount,
          this.forceConfig,
          this.simBuffers.activeCount,
        );
        recordCollisionPass(
          encoder,
          this.collisionPipeline,
          this.collisionBindGroup,
          nodeCount,
          this.forceConfig.collisionIterations,
          nodeCount > 1000,
          this.simBuffers.activeCount,
        );
      }
    }

    // Tick the simulation controller
    this.simulationController.tick();
  }

  /**
   * Rotate the ping-pong buffers after a simulation step.
   *
   * The swap and the parity flip happen together inside
   * BindGroupParitySets.advance() — they are one state transition, and every
   * bind-group read asserts the two are still in agreement. No bind group is
   * created here: both parity variants were built when the buffers were
   * allocated (rebuildAllBindGroups), so advancing a frame is an index flip.
   */
  private advanceFrameParity(): void {
    this.paritySets.advance();

    // Layers must point at the new read buffer, not the buffer the simulation
    // is about to write. Stream-derived uploads inside are dirty-gated.
    this.refreshLayerRenderContexts();
  }

  /**
   * Render a frame
   *
   * Two-tier activity gating minimizes GPU work when idle:
   * - Tier 1 (compute): Skip force simulation when alpha = 0 (no movement needed)
   * - Tier 2 (render): Skip visual passes when nothing changed on screen
   *
   * Edge flow animation forces continuous rendering when enabled.
   */
  private renderFrame(_deltaTime: number, _stats: FrameStats): void {
    if (this.disposed || !this.state.loaded) return;

    // A lost device cannot record or submit work. Stop the loops and tell the
    // host once instead of throwing (and logging) at 60fps forever.
    if (this.gpuContext.isDeviceLost) {
      this.renderLoop.stop();
      this.simulationController.stop();
      this.events.emit({
        type: "device:lost",
        timestamp: performance.now(),
        reason: "unknown",
        message: "GPU device was lost; recreate the graph to continue rendering",
      });
      return;
    }

    // Skip rendering if canvas has no valid dimensions
    if (this.canvas.width === 0 || this.canvas.height === 0) return;

    // Semantic LOD, before the dirty gate: a crossfade in flight is itself a
    // reason to draw, and the controller is the only thing that knows one is.
    if (this.lodController?.tick(performance.now())) {
      this.renderDirty = true;
    }

    // Tier 1: Gate GPU simulation compute on needsCompute (alpha > 0).
    // All wake triggers (drag, mutations, config changes) set alpha > 0,
    // so compute automatically resumes. When alpha decays to 0, compute stops.
    const needsCompute = this.simulationController.needsCompute;

    // Edge flow animation requires continuous rendering (time-based uniforms)
    const hasActiveFlow = this.isEdgeFlowEnabled();
    if (hasActiveFlow) {
      this.renderDirty = true;
    }

    // Birth pulse animation requires rendering while decaying (or looping)
    if (this.lastBirthTime !== 0 && this.birthPulseConfig.enabled) {
      const elapsed = this.getAnimationTime() - Math.abs(this.lastBirthTime);
      const isLooping = this.lastBirthTime < 0;
      if (isLooping || elapsed < this.birthPulseConfig.duration * 3.0) {
        this.renderDirty = true;
      } else {
        this.lastBirthTime = 0;
      }
    }

    // Simulation compute marks render dirty (positions change each frame)
    if (needsCompute) {
      this.renderDirty = true;
    }

    // Tier 2: Skip all GPU work when nothing visual has changed.
    // The rAF loop still runs (cost: ~0.01ms per empty callback) so we
    // respond to wake events within one frame (~16ms).
    if (!this.renderDirty) {
      // Still tick the controller so alpha decay continues if somehow > 0
      // (shouldn't happen since needsCompute was false, but defensive)
      if (this.simulationController.isRunning) {
        this.simulationController.tick();
      }
      return;
    }

    // Clear the dirty flag for this frame
    this.renderDirty = false;

    const { device, context } = this.gpuContext;

    // Create command encoder
    const encoder = device.createCommandEncoder();

    // Run GPU simulation compute passes (only when forces need computing)
    if (needsCompute) {
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
    this.refreshLayerRenderContexts();

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

    // Collapsed-edge bundles: the same pipeline over their own instance
    // buffers, drawn after the source edges so a thick bundle sits above the
    // edges it stands for rather than under them. The originals leading into
    // hidden nodes are already at zero opacity, so nothing is drawn twice.
    const lodEdgeBindGroup = this.lodEdgeBindGroupSlot.current;
    if (
      this.edgePipeline && this.viewportBindGroup && lodEdgeBindGroup &&
      this.lodBundleDrawCount > 0
    ) {
      renderEdges(
        renderPass,
        this.edgePipeline,
        this.viewportBindGroup,
        lodEdgeBindGroup,
        this.lodBundleDrawCount,
      );
    }

    // Render nodes (on top) — use nodeHighWater as instance count so all
    // occupied slots are drawn. Dead slots (from removals) have radius=0
    // and are invisible; using nodeCount would skip live nodes in high slots.
    const nodeInstanceCount = this.graphState?.nodeHighWater ?? this.state.nodeCount;

    // Update animation time in RenderConfig for birth pulse (targeted 4-byte write at byte 48)
    if (this.renderConfigBuffer && this.lastBirthTime !== 0) {
      const animTime = this.getAnimationTime();
      device.queue.writeBuffer(
        this.renderConfigBuffer,
        48,
        new Float32Array([animTime]),
      );
    }

    if (
      this.nodePipeline &&
      this.viewportBindGroup &&
      this.nodeBindGroup &&
      this.renderConfigBindGroup &&
      nodeInstanceCount > 0
    ) {
      renderNodes(
        renderPass,
        this.nodePipeline,
        this.viewportBindGroup,
        this.nodeBindGroup,
        this.renderConfigBindGroup,
        nodeInstanceCount,
      );
    }

    renderPass.end();

    // Render overlay layers AFTER nodes (labels render on top)
    const labelsLayer = this.layerManager.getLayer<LabelsLayer>("labels");
    if (labelsLayer && labelsLayer.enabled) {
      labelsLayer.render(encoder, textureView);
    }

    // Schedule position sync only when simulation is actively computing
    let syncEncoder: GPUCommandEncoder | null = null;
    if (needsCompute && this.simBuffers && !this.syncInProgress) {
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

    // Rotate ping-pong buffers after GPU execution for next frame
    if (needsCompute) {
      this.advanceFrameParity();
    }

    // DOM cards are not GPU work, so they run once the frame is submitted:
    // one camera transform write plus one placement per card, both no-ops
    // when neither the camera nor the positions moved.
    this.domOverlay?.syncFrame();
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

      // Convergence detection: compare with previous sync positions
      this.checkConvergence();
    } catch (e) {
      // Readback failed - might happen if buffers are destroyed
      if (this.debug) {
        console.warn("Position readback failed:", e);
      }
    } finally {
      this.syncInProgress = false;
    }
  }

  /**
   * Check whether nodes have stopped moving by comparing positions
   * between consecutive GPU→CPU syncs. When max displacement per frame
   * drops below threshold for several consecutive checks, clamp alpha to 0.
   */
  private checkConvergence(): void {
    const pg = this.state.parsedGraph;
    if (!pg) return;

    // Cover all occupied slots so live nodes in high slots (after removals)
    // count toward convergence; static hole slots contribute zero displacement.
    const nodeCount = this.graphState?.nodeHighWater ?? this.state.nodeCount;
    if (nodeCount === 0) return;

    // Already converged (alpha = 0) — nothing to do
    const alpha = this.simulationController.state.alpha;
    if (alpha === 0) return;

    const posX = pg.positionsX;
    const posY = pg.positionsY;

    // First sync — just store positions, can't compare yet
    if (
      !this.prevSyncPositionsX || !this.prevSyncPositionsY ||
      this.prevSyncPositionsX.length < nodeCount
    ) {
      this.prevSyncPositionsX = new Float32Array(posX.subarray(0, nodeCount));
      this.prevSyncPositionsY = new Float32Array(posY.subarray(0, nodeCount));
      this.convergenceCheckCount = 0;
      return;
    }

    // Find max displacement since last sync
    let maxDisplSq = 0;
    for (let i = 0; i < nodeCount; i++) {
      const dx = posX[i] - this.prevSyncPositionsX[i];
      const dy = posY[i] - this.prevSyncPositionsY[i];
      const dSq = dx * dx + dy * dy;
      if (dSq > maxDisplSq) maxDisplSq = dSq;
    }

    // Convert to per-frame displacement (sync happens every SYNC_INTERVAL frames)
    const maxDisplPerFrame = Math.sqrt(maxDisplSq) / this.SYNC_INTERVAL;

    if (maxDisplPerFrame < this.CONVERGENCE_THRESHOLD) {
      this.convergenceCheckCount++;
      if (this.convergenceCheckCount >= this.CONVERGENCE_CHECKS_REQUIRED) {
        // Graph has genuinely converged — clamp alpha to 0
        this.simulationController.setAlpha(0);
      }
    } else {
      // Still moving — reset counter
      this.convergenceCheckCount = 0;
    }

    // Store current positions for next comparison
    this.prevSyncPositionsX.set(posX.subarray(0, nodeCount));
    this.prevSyncPositionsY.set(posY.subarray(0, nodeCount));
  }

  /**
   * Reset interaction, styling and convergence state that is scoped to a
   * single loaded graph.
   *
   * Every field cleared here is keyed by node/edge slot index or holds a
   * per-graph animation timestamp, so carrying it across a load() would
   * apply it to unrelated nodes of the new graph (stale selection/hover
   * highlights, colors restored onto the wrong nodes, a convergence
   * comparison against the previous graph's positions).
   */
  private resetPerGraphState(): void {
    this.selectedNodes.clear();
    this.selectedEdges.clear();
    this.hover.reset();
    // Cards are keyed by slot, so every one of them refers to a node of the
    // outgoing graph; the release runs while their DOM is still valid.
    this.domOverlay?.releaseAll();
    this.streamColorBackups.clear();
    this.prevSyncPositionsX = null;
    this.prevSyncPositionsY = null;
    this.convergenceCheckCount = 0;
    this.lastBirthTime = 0;

    // Stream-derived intensity buffers are slot-indexed and grow-only, so a
    // large graph followed by a small one would otherwise pin the old scratch
    // array and GPU buffer for the process lifetime. syncStreamIntensities
    // recreates the cache for any layer that still needs one.
    for (const cache of this.streamIntensityCaches.values()) {
      cache.destroy();
    }
    this.streamIntensityCaches.clear();
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
      throw new GraphMotherError(
        ErrorCode.DISPOSED_ACCESS,
        "Cannot load data on disposed graph",
      );
    }

    // Parse input
    const isTyped = "nodeCount" in data;
    const parsed = isTyped
      ? parseGraphTypedInput(data as GraphTypedInput, this.typedParserConfig)
      : parseGraphInput(data as GraphInput, this.parserConfig);

    // Pins reference slots of the previous graph — the fresh nodeFlags buffer
    // is zero-filled, so the CPU set must reset to match. The LOD focus set is
    // slot-keyed on the same graph and no slot of it survives, which is what an
    // empty remap says.
    this.pinnedNodes.clear();
    this.lodController?.remapSlots(new Map());

    // Every other slot-indexed interaction/animation cache is equally stale
    this.resetPerGraphState();

    // Everything derived from the topology belongs to the outgoing graph, the
    // collapse set included. Adopt the producer's hierarchy columns when this
    // input carries them (typed path only — the object path assigns slots
    // itself, so supplied columns would be indexed against slots the producer
    // never saw).
    this.invalidateTopologyDerived();
    this.suppliedHierarchy = parsed.hierarchy ?? null;
    this.suppliedHierarchyStamp = parsed.hierarchy
      ? { nodes: parsed.nodeCount, edges: parsed.edgeCount }
      : null;

    // Handle empty graph gracefully
    if (parsed.nodeCount === 0) {
      // Clear existing state
      this.destroyBuffers();
      this.destroySimulationBuffers();
      this.graphState = null;
      this.wasmEngine?.clear();
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

    // Alias parsedGraph views onto graphState's arrays so there is exactly ONE
    // CPU shadow from the start. Readback and styling write through parsedGraph;
    // mutation flushes upload graphState's arrays — without this aliasing the
    // first batch mutation would upload stale load-time positions and colors.
    // Must run after populateWasmEngine/createBuffers/createSimulationResources,
    // which rely on parsed arrays being exactly nodeCount long.
    this.syncParsedGraphFromState();

    // Update layer render contexts with new position buffers
    this.refreshLayerRenderContexts();

    // Start render loop on first load (delayed from constructor to ensure canvas is sized)
    if (!this.renderLoop.isRunning) {
      this.renderLoop.start();
    }

    // Fit view to content
    this.fitToView();

    // Start simulation automatically
    startSettlingTelemetry();
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

    // Simulation and render bind groups (both parities) for the fresh buffers
    this.rebuildSimulationBindGroups();

    // Create algorithm-specific buffers and bind groups (use capacity)
    if (this.currentAlgorithm && this.algorithmPipelines) {
      this.algorithmBuffers = this.currentAlgorithm.createBuffers(
        device,
        nodeCap,
      );
      this.rebuildAlgorithmBindGroups();

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

    const { sizes, maxRadius } = collisionRadiusColumn(nodeAttributes, nodeCount, {
      isDead: (slot) => this.graphState?.nodeFreeSet.has(slot) ?? false,
      proxyRadius: (slot) => this.lodProxies.savedRadiusOf(slot),
      deadRadius: DEAD_SLOT_RADIUS,
    });
    this.maxNodeRadius = maxRadius > 0 ? maxRadius : DEFAULT_COLLISION_RADIUS;
    uploadNodeSizes(device, this.collisionBuffers, sizes);

    // Update collision uniforms
    updateCollisionUniforms(device, this.collisionBuffers, nodeCount, this.forceConfig);

    // Create grid collision resources (spatial hash for O(n·k) at >5000 nodes)
    if (this.gridCollisionPipeline) {
      if (this.gridCollisionBuffers) {
        destroyGridCollisionBuffers(this.gridCollisionBuffers);
      }
      this.gridCollisionBuffers = createGridCollisionBuffers(device, nodeCount);
    }

    // Both collision paths bind the buffers created above, so their bind
    // groups have to be rebuilt for both parities here.
    this.rebuildCollisionBindGroups();

    if (this.debug) {
      console.log(`Collision detection initialized for ${nodeCount} nodes`);
    }
  }

  /**
   * Populate the WASM engine with graph data from a ParsedGraph.
   * Clears any existing data and bulk-loads nodes and edges.
   *
   * The engine serves TOPOLOGY consumers (CSR generation, tree/community/
   * codebase layouts). Its R-tree spatial index is deliberately NOT
   * maintained: node positions live on the GPU and only the CPU shadow
   * (parsedGraph.positionsX/Y) is refreshed via readback, so the engine's
   * position copy goes stale the moment simulation starts. Hit testing
   * therefore brute-forces over the CPU shadow (see updateHitTester).
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
  }

  /**
   * The retained containment hierarchy for the loaded graph, or null when none
   * can be built (no graph, or no producer-supplied columns and no WASM engine
   * to derive them from).
   *
   * Algorithm-independent and CPU-side: the hierarchy exists whether or not any
   * force algorithm consumes it. Built lazily on first access after a topology
   * change, then reused — the LOD cut, the hit-test broad phase and DOM culling
   * all read the same instance.
   *
   * Producers SHOULD emit slots in depth-first order so that a subtree is a
   * contiguous slot range; core neither verifies nor reorders.
   */
  getHierarchy(): RetainedHierarchy | null {
    if (!this.hierarchy) {
      this.hierarchy = this.buildHierarchy();
    }
    return this.hierarchy;
  }

  /**
   * Build the containment hierarchy: producer-supplied when available,
   * WASM-derived from the containment edges otherwise.
   *
   * There is deliberately no TypeScript derivation. A second implementation is
   * a second root-selection rule, and the two disagreed: the old CPU BFS rooted
   * every zero-indegree node while WASM picked a single component by descendant
   * count, so depths and radii described different trees.
   */
  private buildHierarchy(): RetainedHierarchy | null {
    const gs = this.graphState;
    if (!gs || gs.nodeHighWater === 0) return null;

    const nodeCount = gs.nodeHighWater;

    if (this.suppliedHierarchy && this.suppliedHierarchyStamp) {
      const stamp = this.suppliedHierarchyStamp;
      if (stamp.nodes === nodeCount && stamp.edges === gs.edgeCount) {
        return retainSuppliedHierarchy(this.suppliedHierarchy, nodeCount);
      }
      this.suppliedHierarchy = null;
      this.suppliedHierarchyStamp = null;
      console.warn(
        "[GraphMother] the graph has been mutated since load, so the producer-supplied " +
          "hierarchy columns no longer describe it; deriving the hierarchy instead.",
      );
    }

    if (!this.wasmEngine) {
      if (!this.hierarchyUnavailableWarned) {
        this.hierarchyUnavailableWarned = true;
        console.warn(
          "[GraphMother] no WASM engine and no supplied hierarchy columns; the containment " +
            "hierarchy is unavailable. Bubble mode, depth-decaying gravity and any hierarchy " +
            "consumer will fall back to flat defaults.",
        );
      }
      return null;
    }

    const containmentEdges = selectContainmentEdges(
      gs.edgeSources,
      gs.edgeTargets,
      gs.edgeCount,
      gs.edgeTypes,
    );

    return deriveHierarchy(this.wasmEngine, containmentEdges, nodeCount, {
      baseRadius: this.forceConfig.relativityBubbleBaseRadius,
      padding: this.forceConfig.relativityBubblePadding,
      rootId: HIERARCHY_ROOT,
    });
  }

  /**
   * Settle what the operation about to run would otherwise destroy.
   *
   * A collapsed proxy goes on simulating while the subtree folded into it is
   * frozen, and the controller pays the difference back to the descendants when
   * the fold opens. That payment is measured against the slot space and the
   * hierarchy the fold was taken in, and
   * {@link GraphMother.invalidateTopologyDerived} ends both: it drops the
   * hierarchy, and the anchors that measured the drift go with it at the next
   * adoption, as the only safe thing to do with numbers naming a tree that no
   * longer exists. So the drift is paid out here, before any of that;
   * afterwards there is nothing coherent left to pay it with, and every such
   * operation would quietly discard whatever was owed at the time.
   *
   * The rule is therefore about the invalidation and not about mutation as
   * such: every path that can reach `invalidateTopologyDerived` passes through
   * here first, including the ones that change no topology at all and only drop
   * what was derived from it — switching force algorithm, re-uploading the
   * bubble knobs. A mutation that also *moves* the slot space has the stricter
   * obligation of reaching here before the move rather than merely before the
   * invalidation, because a proxy's position read after a compaction is some
   * other node's.
   *
   * Deliberately not called from {@link GraphMother.load}, which replaces every
   * position a translate would move: there the drift is not lost but moot, and
   * issuing it would be three buffer writes per fold spent on positions that
   * are about to be overwritten.
   */
  private beginTopologyChange(): void {
    this.lodController?.flushFoldDrift();
  }

  /**
   * Drop everything derived from the topology that just changed.
   *
   * The hierarchy, the rolled-up masses, the inflated proxy radii and the LOD
   * edge aggregation all name slots and edge indices of the graph they were
   * computed against, and a mutation moves both: a removal compacts the slot
   * space and swap-removes edges, an addition appends slots the aggregation's
   * live-edge list has never heard of. Kept, they do not merely go stale — the
   * bundled spring pass keeps driving springs from an edge list that now
   * describes different edges, and a proxy weight stays on whatever node
   * inherited the slot.
   *
   * Every mutation entry point calls this, singular and batch alike, including
   * the singular paths the plural ones delegate to. Calling it twice on one
   * mutation is deliberately cheap — each of the four is a no-op when the state
   * it drops is already neutral — because the alternative is reasoning about
   * which of two overlapping paths a mutation took.
   */
  private invalidateTopologyDerived(): void {
    this.hierarchy = null;
    this.resetNodeMass();
    this.resetLodProxyRadii();
    this.releaseLodEdgeAggregation();
    // Everything above is host-side state the controller believes it still
    // owns. Telling it forces the re-evaluation that re-derives all of it;
    // without this the cut only catches up when the camera next moves.
    this.lodController?.handleTopologyChange();
  }

  /**
   * Upload edge data for algorithm-specific formats (CSR for Relativity Atlas).
   *
   * CSR data is generated from MutableGraphState's edge arrays, which are the
   * source of truth for GPU buffer slot indices. This ensures the CSR indices
   * match the actual position buffer layout.
   *
   * Every caller reaches here after mutating nodes or edges, so the derived
   * state is dropped up front — before the algorithm early-out, since the
   * hierarchy is graph-level and outlives any particular algorithm.
   */
  private uploadAlgorithmEdgeData(device: GPUDevice): void {
    this.invalidateTopologyDerived();

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

      const b = this.algorithmBuffers as RelativityAtlasBuffers;
      const nodeCount = gs.nodeHighWater;
      const hierarchy = this.getHierarchy();

      // Depths drive depth-decaying gravity; they come from the same forest the
      // LOD cut walks, so the physics and the semantics cannot diverge.
      const depths = new Float32Array(nodeCount);
      if (hierarchy) {
        const { depth } = hierarchy.columns;
        for (let i = 0; i < nodeCount; i++) depths[i] = depth[i];
      }
      device.queue.writeBuffer(b.nodeDepth, 0, depths);
      device.queue.writeBuffer(this.simBuffers!.nodeDepth, 0, depths);

      // Well radii are a bubble-mode force parameter: outside bubble mode the
      // phantom-zone and orbit-spring shaders must see the uniform default, so
      // the hierarchy is retained but its radii are not uploaded.
      const wellRadii = new Float32Array(nodeCount);
      if (this.forceConfig.relativityBubbleMode && hierarchy) {
        wellRadii.set(hierarchy.columns.wellRadius.subarray(0, nodeCount));
      } else {
        wellRadii.fill(DEFAULT_WELL_RADIUS);
        if (this.forceConfig.relativityBubbleMode) {
          console.warn(
            "[GraphMother] relativityBubbleMode is enabled but no containment hierarchy is " +
              "available; falling back to uniform well radii (bubble collision will be " +
              "ineffective)",
          );
        }
      }
      device.queue.writeBuffer(b.wellRadius, 0, toArrayBuffer(wellRadii));

      if (this.debug) {
        console.log(
          `Relativity Atlas: uploaded CSR (${gs.nodeHighWater} nodes, ${gs.edgeCount} edges)` +
            (this.forceConfig.relativityBubbleMode ? " [bubble mode]" : "") +
            (hierarchy ? ` [hierarchy: ${hierarchy.source}]` : " [no hierarchy]"),
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

    // Create node attributes buffer (8 floats per node) — sized to capacity
    const nodeAttributes = device.createBuffer({
      label: "Node Attributes",
      size: nodeCapacity * NODE_ATTR_BYTES,
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

    // The viewport bind group references no ping-pong buffer, so it is built
    // once here. The node and edge bind groups DO (they read the simulation's
    // current position buffer, not `positions` above, which is only the
    // pre-simulation upload target) and are built per parity by
    // rebuildSimulationBindGroups, which createSimulationResources runs
    // immediately after this method.
    if (this.nodePipeline) {
      this.viewportBindGroup = createViewportBindGroup(
        device,
        this.nodePipeline,
        this.viewportUniformBuffer.buffer,
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

    this.nodeBindGroupSlot.clear();
    this.edgeBindGroupSlot.clear();
    // The bundle instances describe the edge set going away, and their bind
    // group holds the pipeline these buffers belonged to.
    this.destroyLodEdgeRenderBuffers();
    this.lodEdgeOpacity.release(null);
  }

  // ==========================================================================
  // Public API - Incremental Mutations
  // ==========================================================================

  /**
   * Record the per-slot columns of an added node that live outside the
   * attribute row: its metadata, and the two semantic columns.
   *
   * A node added after load reads back through the same accessors as one that
   * came with the graph — `metadata` is what a DOM card renders, `tag` and
   * `weight` are what an LOD policy ranks on — so dropping them here is a card
   * with no label and a policy input stuck at zero, silently.
   *
   * Every column is written unconditionally, absent values included: an added
   * node may be reusing a freed slot, and the previous occupant's label must
   * not become its own. The semantic columns are absent altogether unless the
   * graph carried them (an all-zero column and no column read the same), so a
   * value materialises one only when it has something to say, and grows it when
   * the slot space has outrun the load-time length.
   */
  private recordAddedNodeColumns(slot: number, node: NodeInput): void {
    const gs = this.graphState;
    if (!gs) return;

    const metadata = node["metadata"] as Record<string, unknown> | undefined;
    if (metadata) {
      gs.nodeMetadata.set(slot, metadata);
    } else {
      gs.nodeMetadata.delete(slot);
    }

    const parsed = this.state.parsedGraph;
    if (!parsed) return;
    if (node.tag !== undefined || parsed.nodeTags) {
      const tags = growSlotColumn(parsed.nodeTags, slot, Uint16Array);
      tags[slot] = node.tag === undefined ? 0 : clampTag(node.tag);
      parsed.nodeTags = tags;
    }
    if (node.weight !== undefined || parsed.nodeWeights) {
      const weights = growSlotColumn(parsed.nodeWeights, slot, Float32Array);
      weights[slot] = node.weight === undefined ? 0 : clampWeight(node.weight);
      parsed.nodeWeights = weights;
    }
  }

  /**
   * Add a single node to the graph.
   *
   * @param node - Node input data
   * @returns The assigned node ID (the user-provided id)
   */
  async addNode(node: NodeInput): Promise<NodeId> {
    this.beginTopologyChange();
    if (!this.graphState || !this.buffers || !this.simBuffers) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Cannot add node: graph not loaded",
      );
    }

    const gs = this.graphState;

    // Check capacity
    if (gs.needsNodeReallocation(1)) {
      await this.reallocateNodeBuffers(growCapacity(gs.nodeHighWater + 1, gs.nodeCapacity));
    }

    // Allocate slot — allocateNodeSlot owns slot allocation; the ID map must
    // adopt its index (add() would pop the map's own diverging free list)
    const slot = gs.allocateNodeSlot();
    const nodeId = node.id;
    gs.nodeIdMap.set(nodeId, slot);

    // Parse position
    const x = node.x ?? (Math.random() - 0.5) * Math.sqrt(gs.nodeCount) * 20;
    const y = node.y ?? (Math.random() - 0.5) * Math.sqrt(gs.nodeCount) * 20;

    // Parse attributes
    const radius = node.radius ?? 5;
    const [r, g, b] = node.color ? parseColorToRGB(node.color) : [0.4, 0.6, 0.9];
    const birthTime = node.birthTime ?? 0;

    // Write to CPU shadow
    gs.positionsX[slot] = x;
    gs.positionsY[slot] = y;
    const attrBase = slot * NODE_ATTR_FLOATS;
    gs.nodeAttributes[attrBase] = radius;
    gs.nodeAttributes[attrBase + 1] = r;
    gs.nodeAttributes[attrBase + 2] = g;
    gs.nodeAttributes[attrBase + 3] = b;
    gs.nodeAttributes[attrBase + 4] = 0; // selected
    gs.nodeAttributes[attrBase + 5] = 0; // hovered
    gs.nodeAttributes[attrBase + 6] = birthTime; // birth_time
    gs.nodeAttributes[attrBase + 7] = 0; // tex_index (reserved for Stage 2)

    // Record node type for type-based styling
    const nodeType = node["type"] as string | undefined;
    if (nodeType) {
      gs.nodeTypes[slot] = nodeType;
    }

    this.recordAddedNodeColumns(slot, node);

    // Track birth pulse for render loop dirty detection (negative = looping)
    if (birthTime !== 0) this.lastBirthTime = birthTime;

    // Write to GPU buffers (targeted writes)
    const { device } = this.gpuContext;
    const posVec2 = new Float32Array([x, y]);
    const attrData = new Float32Array([radius, r, g, b, 0, 0, birthTime, 0]);
    const zeroVec2 = new Float32Array([0, 0]);

    device.queue.writeBuffer(this.simBuffers.positions, slot * 8, posVec2);
    device.queue.writeBuffer(this.simBuffers.positionsOut, slot * 8, posVec2);
    device.queue.writeBuffer(this.buffers.nodeAttributes, slot * NODE_ATTR_BYTES, attrData);
    device.queue.writeBuffer(this.simBuffers.velocities, slot * 8, zeroVec2);
    device.queue.writeBuffer(this.simBuffers.velocitiesOut, slot * 8, zeroVec2);
    device.queue.writeBuffer(this.simBuffers.forces, slot * 8, zeroVec2);

    // Mark the slot live for simulation and collision (clears any dead-slot
    // flag / negative-radius sentinel left by a previous removal)
    this.writeNodeSlotLiveness(slot, radius);

    // Update WASM engine — by contract the engine reuses freed slots in the
    // same LIFO order as allocateNodeSlot, so the returned id equals `slot`.
    // No spatial index rebuild: the index has no consumers (hit testing
    // brute-forces the CPU shadow; see populateWasmEngine).
    if (this.wasmEngine) {
      const wasmId = this.wasmEngine.addNode(x, y);
      if (this.debug && wasmId !== slot) {
        console.warn(`WASM slot contract violated: engine id ${wasmId}, expected slot ${slot}`);
      }
    }

    // Update counts
    this.state.nodeCount = gs.nodeCount;
    this.simBuffers.nodeCount = gs.nodeHighWater;

    // Update parsedGraph reference to point to graphState arrays
    this.syncParsedGraphFromState();
    this.invalidateTopologyDerived();

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
    this.beginTopologyChange();
    if (!this.graphState || !this.buffers || !this.simBuffers) return false;

    const gs = this.graphState;
    const slot = typeof id === "number" && id < gs.nodeHighWater ? id : gs.nodeIdMap.get(id);
    // Reject already-freed slots — a double free would corrupt the free list
    if (slot === undefined || gs.nodeFreeSet.has(slot)) return false;

    // Remove all connected edges first — sort descending so swap-remove only
    // moves edges from higher indices to lower, not affecting edges still to process
    const connectedEdges = [...gs.getConnectedEdges(slot)].sort((a, b) => b - a);
    for (const edgeIndex of connectedEdges) {
      await this.removeEdgeByIndex(edgeIndex);
    }

    // Free the node slot (zeros CPU shadow)
    gs.nodeIdMap.remove(gs.nodeIdMap.getId(slot)!);
    gs.freeNodeSlot(slot);

    // Write zeros to GPU buffers
    const { device } = this.gpuContext;
    const zeroVec2 = new Float32Array([0, 0]);
    const zeroAttrs = new Float32Array(NODE_ATTR_FLOATS);

    device.queue.writeBuffer(this.simBuffers.positions, slot * 8, zeroVec2);
    device.queue.writeBuffer(this.simBuffers.positionsOut, slot * 8, zeroVec2);
    device.queue.writeBuffer(this.buffers.nodeAttributes, slot * NODE_ATTR_BYTES, zeroAttrs);
    device.queue.writeBuffer(this.simBuffers.velocities, slot * 8, zeroVec2);
    device.queue.writeBuffer(this.simBuffers.velocitiesOut, slot * 8, zeroVec2);
    device.queue.writeBuffer(this.simBuffers.forces, slot * 8, zeroVec2);

    // Mark the slot dead so simulation shaders and collision skip it entirely
    // (a zeroed slot at the origin would otherwise still repel and collide)
    this.writeNodeSlotLiveness(slot, undefined);

    // Update WASM engine (NodeId == slot by contract)
    if (this.wasmEngine) {
      this.wasmEngine.removeNode(slot);
    }

    // Update counts
    this.state.nodeCount = gs.nodeCount;
    this.simBuffers.nodeCount = gs.nodeHighWater;

    this.syncParsedGraphFromState();
    this.invalidateTopologyDerived();
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
    this.beginTopologyChange();
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

    // Allocate slot — edgeIdMap adopts the dense slot from allocateEdgeSlot
    const slot = gs.allocateEdgeSlot();
    const edgeId = (edge as Record<string, unknown>)["id"] as string | number | undefined ??
      `edge_${slot}`;
    gs.edgeIdMap.set(edgeId, slot);

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
    gs.edgeAttributes[eAttrBase + 7] = 1.0; // opacity (default: fully visible)

    // Update adjacency
    gs.addEdgeAdjacency(slot, srcSlot, tgtSlot);

    // Write to GPU buffers
    const { device } = this.gpuContext;
    const edgeIndicesData = new Uint32Array([srcSlot, tgtSlot]);
    const edgeAttrData = new Float32Array([width, r, g, b, 0, 0, 0, 1.0]);
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
    this.invalidateTopologyDerived();
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
    // Always resolve through IdMap — numeric shortcut is invalid after swap-remove
    const slot = gs.edgeIdMap.get(id);
    if (slot === undefined) return false;

    return this.removeEdgeByIndex(slot);
  }

  /**
   * Internal: Remove an edge by its slot index using swap-remove.
   */
  private async removeEdgeByIndex(index: number): Promise<boolean> {
    this.beginTopologyChange();
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

    // Update the swapped edge's ID map entry — force-assign to the vacated slot
    if (swappedId !== undefined && swappedFromIndex >= 0) {
      gs.edgeIdMap.remove(swappedId);
      gs.edgeIdMap.set(swappedId, index);
    }

    // Write the swapped edge data to GPU at the vacated slot
    if (swappedFromIndex >= 0 && index < gs.edgeCount) {
      const { device } = this.gpuContext;
      const edgeIndicesData = new Uint32Array([gs.edgeSources[index], gs.edgeTargets[index]]);
      const edgeAttrData = gs.edgeAttributes.subarray(index * 8, index * 8 + 8);
      const srcData = new Uint32Array([gs.edgeSources[index]]);
      const tgtData = new Uint32Array([gs.edgeTargets[index]]);

      device.queue.writeBuffer(this.buffers.edgeIndices, index * 8, edgeIndicesData);
      device.queue.writeBuffer(
        this.buffers.edgeAttributes,
        index * 32,
        new Float32Array(edgeAttrData),
      );
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
    this.invalidateTopologyDerived();

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
    this.beginTopologyChange();
    if (!this.graphState || !this.buffers || !this.simBuffers) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Cannot add nodes: graph not loaded",
      );
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
      gs.nodeIdMap.set(node.id, slot);
      // A reused slot must not inherit the previous occupant's pin
      this.pinnedNodes.delete(slot);

      const x = node.x ?? (Math.random() - 0.5) * Math.sqrt(gs.nodeCount) * 20;
      const y = node.y ?? (Math.random() - 0.5) * Math.sqrt(gs.nodeCount) * 20;
      const radius = node.radius ?? 5;
      const [r, g, b] = node.color ? parseColorToRGB(node.color) : [0.4, 0.6, 0.9];
      const birthTime = node.birthTime ?? 0;

      gs.positionsX[slot] = x;
      gs.positionsY[slot] = y;
      const attrBase = slot * NODE_ATTR_FLOATS;
      gs.nodeAttributes[attrBase] = radius;
      gs.nodeAttributes[attrBase + 1] = r;
      gs.nodeAttributes[attrBase + 2] = g;
      gs.nodeAttributes[attrBase + 3] = b;
      gs.nodeAttributes[attrBase + 4] = 0;
      gs.nodeAttributes[attrBase + 5] = 0;
      gs.nodeAttributes[attrBase + 6] = birthTime;
      gs.nodeAttributes[attrBase + 7] = 0; // tex_index

      // Track birth pulse for render loop dirty detection (negative = looping)
      if (birthTime !== 0) this.lastBirthTime = birthTime;

      // Record node type for type-based styling
      const nodeType = node["type"] as string | undefined;
      if (nodeType) {
        gs.nodeTypes[slot] = nodeType;
      }

      this.recordAddedNodeColumns(slot, node);

      ids.push(slot);
    }

    // Flush all node data to GPU in bulk
    this.flushNodeBuffersToGPU();
    this.flushNodeSlotFlagsToGPU();

    // Update WASM engine in batch
    if (this.wasmEngine) {
      const positions = new Float32Array(nodes.length * 2);
      for (let i = 0; i < ids.length; i++) {
        const slot = ids[i];
        positions[i * 2] = gs.positionsX[slot];
        positions[i * 2 + 1] = gs.positionsY[slot];
      }
      this.wasmEngine.addNodesFromPositions(positions);
    }

    this.state.nodeCount = gs.nodeCount;
    this.simBuffers.nodeCount = gs.nodeHighWater;
    this.syncParsedGraphFromState();
    this.invalidateTopologyDerived();
    this.ensureAlgorithmCapacity();

    // Re-apply type styles after GPU flush to prevent CPU shadow from clobbering styled data
    this.applyTypeStyles();

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
  async removeNodes(
    ids: (NodeId | string)[],
  ): Promise<{ removedCount: number; nodeSlotRemap: Map<number, number> }> {
    const emptyResult = { removedCount: 0, nodeSlotRemap: new Map<number, number>() };
    if (ids.length === 0) return emptyResult;
    if (ids.length === 1) {
      const ok = await this.removeNode(ids[0]);
      // Single-node removal uses freeNodeSlot (hole-based, no compaction),
      // so no surviving node changes slot — remap is empty.
      return ok ? { removedCount: 1, nodeSlotRemap: new Map() } : emptyResult;
    }

    // Delegate to batch for O(N+E) instead of O(N^2)
    return this.removeNodesBatch(ids);
  }

  /**
   * Batch-optimized node removal with compact layout.
   *
   * Unlike removeNodes (which calls removeNode per-ID), this method:
   * - Resolves all IDs to slots upfront
   * - Collects ALL connected edges across all removed nodes
   * - Compacts edge arrays in a single pass (no swap-remove instability)
   * - Compacts node arrays — shifts surviving nodes down to eliminate gaps
   * - Remaps edge source/target indices to use new compact node slots
   * - Performs a single GPU flush for nodes and edges
   * - Rebuilds the WASM engine once
   * - Returns a slot remap so callers can update external slot references
   * - Emits a single batch event
   *
   * After this call, nodeCount == nodeHighWater (no holes) and nodeFreeList is empty.
   *
   * Performance: O(N + E) CPU work, ~12 GPU writes total (vs 6N + 4DN for removeNodes).
   *
   * @param ids - Node IDs to remove
   * @returns Object with removedCount and nodeSlotRemap (oldSlot → newSlot for surviving nodes)
   */
  async removeNodesBatch(
    ids: (NodeId | string)[],
  ): Promise<{ removedCount: number; nodeSlotRemap: Map<number, number> }> {
    this.beginTopologyChange();
    const emptyResult = { removedCount: 0, nodeSlotRemap: new Map<number, number>() };
    if (!this.graphState || !this.buffers || !this.simBuffers) return emptyResult;

    const gs = this.graphState;

    // 1. Resolve all IDs to valid slot indices
    const slotsToRemove = new Set<number>();
    for (const id of ids) {
      const slot = typeof id === "number" && id < gs.nodeHighWater ? id : gs.nodeIdMap.get(id);
      if (slot !== undefined && !gs.nodeFreeSet.has(slot)) {
        slotsToRemove.add(slot);
      }
    }
    if (slotsToRemove.size === 0) return emptyResult;

    // 2. Scan ALL edges — remove any where either endpoint is being removed.
    // This is O(E) but bypasses the nodeEdges adjacency map which can accumulate
    // stale entries from previous swap-remove operations in freeEdgeSlot.
    const edgesToRemove = new Set<number>();
    for (let i = 0; i < gs.edgeCount; i++) {
      if (slotsToRemove.has(gs.edgeSources[i]!) || slotsToRemove.has(gs.edgeTargets[i]!)) {
        edgesToRemove.add(i);
      }
    }

    // 3. Compact edge arrays — single-pass filter keeping only surviving edges
    const prevEdgeCount = gs.edgeCount;
    let removedEdgeCount = 0;

    if (edgesToRemove.size > 0) {
      removedEdgeCount = edgesToRemove.size;

      // Collect surviving edge IDs and types in compact order
      const survivingEdgeIds: (string | number | undefined)[] = [];
      const survivingEdgeTypes: (string | undefined)[] = [];

      let writeIdx = 0;
      for (let readIdx = 0; readIdx < prevEdgeCount; readIdx++) {
        if (edgesToRemove.has(readIdx)) continue;

        survivingEdgeIds.push(gs.edgeIdMap.getId(readIdx));
        survivingEdgeTypes.push(gs.edgeTypes[readIdx]);

        if (readIdx !== writeIdx) {
          gs.edgeSources[writeIdx] = gs.edgeSources[readIdx]!;
          gs.edgeTargets[writeIdx] = gs.edgeTargets[readIdx]!;
          const srcAttr = readIdx * 8;
          const dstAttr = writeIdx * 8;
          for (let k = 0; k < 8; k++) {
            gs.edgeAttributes[dstAttr + k] = gs.edgeAttributes[srcAttr + k]!;
          }
        }
        writeIdx++;
      }

      gs.edgeCount = writeIdx;

      // Rebuild edge ID map with compact indices
      gs.edgeIdMap.clear();
      for (const eid of survivingEdgeIds) {
        if (eid !== undefined) gs.edgeIdMap.add(eid);
      }

      // Restore edge types
      gs.edgeTypes = survivingEdgeTypes;

      // Edge metadata uses slot indices — clear stale entries
      gs.edgeMetadata.clear();
    }

    // 4. Compact node arrays — single-pass filter shifting survivors down
    // deadSlots includes both the nodes being removed AND any pre-existing
    // holes from prior freeNodeSlot() calls, so compaction eliminates all gaps.
    const deadSlots = new Set<number>([...slotsToRemove, ...gs.nodeFreeList]);
    const nodeSlotRemap = new Map<number, number>(); // oldSlot → newSlot
    const survivingNodeIds: (string | number | undefined)[] = [];
    const survivingNodeTypes: (string | undefined)[] = [];
    const survivingNodeMeta = new Map<number, Record<string, unknown>>();
    const prevNodeHighWater = gs.nodeHighWater;

    let nodeWriteIdx = 0;
    for (let readIdx = 0; readIdx < prevNodeHighWater; readIdx++) {
      if (deadSlots.has(readIdx)) continue;

      survivingNodeIds.push(gs.nodeIdMap.getId(readIdx));
      survivingNodeTypes.push(gs.nodeTypes[readIdx]);

      const meta = gs.nodeMetadata.get(readIdx);
      if (meta) survivingNodeMeta.set(nodeWriteIdx, meta);

      nodeSlotRemap.set(readIdx, nodeWriteIdx);

      if (readIdx !== nodeWriteIdx) {
        gs.positionsX[nodeWriteIdx] = gs.positionsX[readIdx]!;
        gs.positionsY[nodeWriteIdx] = gs.positionsY[readIdx]!;
        // Pin and visibility travel with the slot's contents; survivors are
        // live by construction, so no dead bit can be carried across.
        gs.nodeFlagsShadow[nodeWriteIdx] = gs.nodeFlagsShadow[readIdx]!;
        const srcAttr = readIdx * NODE_ATTR_FLOATS;
        const dstAttr = nodeWriteIdx * NODE_ATTR_FLOATS;
        for (let k = 0; k < NODE_ATTR_FLOATS; k++) {
          gs.nodeAttributes[dstAttr + k] = gs.nodeAttributes[srcAttr + k]!;
        }
      }
      nodeWriteIdx++;
    }

    // Zero trailing data (previously occupied slots now beyond nodeWriteIdx)
    for (let i = nodeWriteIdx; i < prevNodeHighWater; i++) {
      gs.positionsX[i] = 0;
      gs.positionsY[i] = 0;
      gs.nodeFlagsShadow[i] = 0;
      const attrBase = i * NODE_ATTR_FLOATS;
      for (let k = 0; k < NODE_ATTR_FLOATS; k++) gs.nodeAttributes[attrBase + k] = 0;
    }

    // Update node tracking — compact, no holes
    gs.nodeCount = nodeWriteIdx;
    gs.nodeHighWater = nodeWriteIdx;
    gs.nodeFreeList = [];
    gs.nodeFreeSet.clear();

    // Rebuild nodeIdMap with compact indices
    gs.nodeIdMap.clear();
    for (const nid of survivingNodeIds) {
      if (nid !== undefined) gs.nodeIdMap.add(nid);
    }

    // Restore metadata maps with new slot indices
    gs.nodeTypes = survivingNodeTypes;
    gs.nodeMetadata = survivingNodeMeta;

    // The producer's semantic columns are slot-indexed like everything else and
    // have no MutableGraphState counterpart, so the compaction has to reach
    // across and move them too or getNodeTag/getNodeWeight answer with the
    // values of whichever node used to occupy the slot.
    const parsedColumns = this.state.parsedGraph;
    if (parsedColumns) {
      compactNodeColumn(parsedColumns.nodeTags, nodeSlotRemap, prevNodeHighWater);
      compactNodeColumn(parsedColumns.nodeWeights, nodeSlotRemap, prevNodeHighWater);
    }

    // 5. Remap edge source/target to use new compact node slot indices
    // Safe: edge compaction already removed edges touching dead nodes,
    // so all surviving edges reference nodes present in nodeSlotRemap.
    for (let i = 0; i < gs.edgeCount; i++) {
      const newSrc = nodeSlotRemap.get(gs.edgeSources[i]!);
      const newTgt = nodeSlotRemap.get(gs.edgeTargets[i]!);
      if (newSrc !== undefined) gs.edgeSources[i] = newSrc;
      if (newTgt !== undefined) gs.edgeTargets[i] = newTgt;
    }

    // 6. Rebuild adjacency from compact edges (using remapped node slots)
    gs.nodeEdges.clear();
    for (let i = 0; i < gs.edgeCount; i++) {
      // Safe: i < edgeCount guarantees valid index
      gs.addEdgeAdjacency(i, gs.edgeSources[i]!, gs.edgeTargets[i]!);
    }

    // Remap the slot-keyed sets a caller declared rather than the graph derived
    // — removed nodes drop their membership, survivors may have moved. The pin
    // bits themselves moved with the flag shadow above; this keeps the CPU-side
    // set (isNodePinned / getPinnedNodes) in agreement, and the LOD focus set
    // pointing at the nodes the caller named rather than at whichever nodes
    // inherited their slots.
    const remappedPinned = new Set<NodeId>();
    for (const slot of this.pinnedNodes) {
      const newSlot = nodeSlotRemap.get(slot);
      if (newSlot !== undefined) remappedPinned.add(newSlot);
    }
    this.pinnedNodes = remappedPinned;
    this.lodController?.remapSlots(nodeSlotRemap);

    // Drop the derived state before the flush, not after: a collapsed proxy's
    // borrowed radius is restored through the rebuilt id map, so it has to be
    // given back while the compacted attribute rows are still only on the CPU.
    this.invalidateTopologyDerived();

    // 7. Single GPU flush for all data (compaction moved slots, so the
    // liveness flags and collision radii must be rewritten for all slots)
    this.flushNodeBuffersToGPU();
    this.flushNodeSlotFlagsToGPU();
    this.flushEdgeBuffersToGPU();

    // 8. Rebuild WASM engine from current state (single rebuild instead of N removes)
    if (this.wasmEngine) {
      this.wasmEngine.clear();

      const positions = new Float32Array(gs.nodeHighWater * 2);
      for (let i = 0; i < gs.nodeHighWater; i++) {
        positions[i * 2] = gs.positionsX[i]!;
        positions[i * 2 + 1] = gs.positionsY[i]!;
      }
      this.wasmEngine.addNodesFromPositions(positions);

      if (gs.edgeCount > 0) {
        const edgePairs = new Uint32Array(gs.edgeCount * 2);
        for (let i = 0; i < gs.edgeCount; i++) {
          edgePairs[i * 2] = gs.edgeSources[i]!;
          edgePairs[i * 2 + 1] = gs.edgeTargets[i]!;
        }
        this.wasmEngine.addEdgesFromPairs(edgePairs);
      }
    }

    // 9. Update counts and sync
    this.state.nodeCount = gs.nodeCount;
    this.state.edgeCount = gs.edgeCount;
    this.simBuffers.nodeCount = gs.nodeHighWater;

    this.syncParsedGraphFromState();
    this.uploadAlgorithmEdgeData(this.gpuContext.device);

    // Re-apply type styles after GPU flush to prevent CPU shadow from clobbering styled data
    this.applyTypeStyles();

    this.bumpSimulationAlpha(0.3);

    // 10. Emit batch summary
    this.events.emit({
      type: "graph:mutate",
      timestamp: Date.now(),
      nodesAdded: 0,
      nodesRemoved: slotsToRemove.size,
      edgesAdded: 0,
      edgesRemoved: removedEdgeCount,
    });

    return { removedCount: slotsToRemove.size, nodeSlotRemap };
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
   * Batch-optimized edge addition.
   *
   * Unlike addEdges (which calls addEdge per-edge), this method:
   * - Checks capacity once for all edges
   * - Populates CPU shadow arrays in a tight loop
   * - Performs a single GPU flush
   * - Batch-adds to WASM engine via addEdgesFromPairs
   * - Emits a single event
   *
   * Performance: O(E) CPU work, ~4 GPU writes total (vs 4E for addEdges).
   *
   * @param edges - Array of edge inputs
   * @returns Array of edge IDs (undefined for edges with invalid source/target)
   */
  async addEdgesBatch(edges: EdgeInput[]): Promise<(EdgeId | undefined)[]> {
    this.beginTopologyChange();
    if (!this.graphState || !this.buffers || !this.simBuffers) return [];
    if (edges.length === 0) return [];

    const gs = this.graphState;

    // Check capacity once for all edges
    if (gs.needsEdgeReallocation(edges.length)) {
      await this.reallocateEdgeBuffers(
        growCapacity(gs.edgeCount + edges.length, gs.edgeCapacity),
      );
    }

    const ids: (EdgeId | undefined)[] = [];
    const newEdgePairs: number[] = []; // For WASM batch

    for (const edge of edges) {
      const srcSlot = gs.nodeIdMap.get(edge.source);
      const tgtSlot = gs.nodeIdMap.get(edge.target);
      if (srcSlot === undefined || tgtSlot === undefined) {
        ids.push(undefined);
        continue;
      }

      const slot = gs.allocateEdgeSlot();
      const edgeId = ((edge as Record<string, unknown>)["id"] as string | number | undefined) ??
        `edge_${slot}`;
      gs.edgeIdMap.set(edgeId, slot);

      const width = edge.width ?? 1;
      const [r, g, b] = edge.color ? parseColorToRGB(edge.color) : [0.5, 0.5, 0.5];

      gs.edgeSources[slot] = srcSlot;
      gs.edgeTargets[slot] = tgtSlot;
      const eAttrBase = slot * 8;
      gs.edgeAttributes[eAttrBase] = width;
      gs.edgeAttributes[eAttrBase + 1] = r;
      gs.edgeAttributes[eAttrBase + 2] = g;
      gs.edgeAttributes[eAttrBase + 3] = b;
      gs.edgeAttributes[eAttrBase + 4] = 0;
      gs.edgeAttributes[eAttrBase + 5] = 0;
      gs.edgeAttributes[eAttrBase + 6] = 0;
      gs.edgeAttributes[eAttrBase + 7] = 1.0; // opacity (default: fully visible)

      gs.addEdgeAdjacency(slot, srcSlot, tgtSlot);

      // Record edge type
      const edgeType = (edge as Record<string, unknown>)["type"] as
        | string
        | undefined;
      if (edgeType) gs.edgeTypes[slot] = edgeType;

      newEdgePairs.push(srcSlot, tgtSlot);
      ids.push(slot);
    }

    // Single GPU flush for all edge data
    this.flushEdgeBuffersToGPU();

    // Batch WASM update
    if (this.wasmEngine && newEdgePairs.length > 0) {
      this.wasmEngine.addEdgesFromPairs(new Uint32Array(newEdgePairs));
    }

    // Update counts
    this.state.edgeCount = gs.edgeCount;

    this.syncParsedGraphFromState();
    this.uploadAlgorithmEdgeData(this.gpuContext.device);

    // Re-apply type styles after GPU flush to prevent CPU shadow from clobbering styled data
    this.applyTypeStyles();

    this.bumpSimulationAlpha(0.1);

    const added = ids.filter((r) => r !== undefined).length;
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

    return ids;
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

  /**
   * Batch-optimized edge removal.
   *
   * Unlike removeEdges (which calls removeEdge per-edge with swap-remove),
   * this method:
   * - Resolves all IDs to slot indices in one pass
   * - Single-pass compact: shifts surviving edges down
   * - Rebuilds edge ID map, types, adjacency from compact data
   * - Single GPU flush
   * - WASM clear + full rebuild (positions unchanged, surviving edges only)
   * - Re-applies type styles to preserve styled GPU data
   *
   * Performance: O(E) CPU scan + ~4 GPU writes + 1 WASM rebuild.
   *
   * @param ids - Edge IDs to remove (string or number)
   * @returns Number of edges actually removed
   */
  async removeEdgesBatch(ids: (EdgeId | string)[]): Promise<number> {
    this.beginTopologyChange();
    if (!this.graphState || !this.buffers || !this.simBuffers) return 0;
    if (ids.length === 0) return 0;

    const gs = this.graphState;

    // 1. Resolve all IDs to valid edge slot indices
    const slotsToRemove = new Set<number>();
    for (const id of ids) {
      const slot = gs.edgeIdMap.get(id);
      if (slot !== undefined) {
        slotsToRemove.add(slot);
      }
    }
    if (slotsToRemove.size === 0) return 0;

    const removedCount = slotsToRemove.size;

    // 2. Single-pass compact: iterate edges, skip removed, shift survivors down
    const survivingEdgeIds: (string | number | undefined)[] = [];
    const survivingEdgeTypes: (string | undefined)[] = [];

    let writeIdx = 0;
    for (let readIdx = 0; readIdx < gs.edgeCount; readIdx++) {
      if (slotsToRemove.has(readIdx)) continue;

      survivingEdgeIds.push(gs.edgeIdMap.getId(readIdx));
      survivingEdgeTypes.push(gs.edgeTypes[readIdx]);

      if (readIdx !== writeIdx) {
        gs.edgeSources[writeIdx] = gs.edgeSources[readIdx]!;
        gs.edgeTargets[writeIdx] = gs.edgeTargets[readIdx]!;
        const srcAttr = readIdx * 8;
        const dstAttr = writeIdx * 8;
        for (let k = 0; k < 8; k++) {
          gs.edgeAttributes[dstAttr + k] = gs.edgeAttributes[srcAttr + k]!;
        }
      }
      writeIdx++;
    }

    gs.edgeCount = writeIdx;

    // 3. Rebuild edge ID map with compact indices
    gs.edgeIdMap.clear();
    for (const eid of survivingEdgeIds) {
      if (eid !== undefined) gs.edgeIdMap.add(eid);
    }

    // 4. Restore edge types
    gs.edgeTypes = survivingEdgeTypes;

    // 5. Clear stale edge metadata
    gs.edgeMetadata.clear();

    // 6. Rebuild adjacency from compact edges
    gs.nodeEdges.clear();
    for (let i = 0; i < gs.edgeCount; i++) {
      gs.addEdgeAdjacency(i, gs.edgeSources[i]!, gs.edgeTargets[i]!);
    }

    // 7. Single GPU flush
    this.flushEdgeBuffersToGPU();

    // 8. Rebuild WASM engine (edges changed, nodes stay the same)
    if (this.wasmEngine) {
      this.wasmEngine.clear();

      // Re-add all node positions (unchanged)
      const positions = new Float32Array(gs.nodeHighWater * 2);
      for (let i = 0; i < gs.nodeHighWater; i++) {
        positions[i * 2] = gs.positionsX[i]!;
        positions[i * 2 + 1] = gs.positionsY[i]!;
      }
      this.wasmEngine.addNodesFromPositions(positions);

      // Recreate hole slots so the engine's vacancy list matches nodeFreeList
      // (same push order → same LIFO reuse order → NodeId == slot holds)
      for (const slot of gs.nodeFreeList) {
        this.wasmEngine.removeNode(slot);
      }

      // Re-add surviving edges
      if (gs.edgeCount > 0) {
        const edgePairs = new Uint32Array(gs.edgeCount * 2);
        for (let i = 0; i < gs.edgeCount; i++) {
          edgePairs[i * 2] = gs.edgeSources[i]!;
          edgePairs[i * 2 + 1] = gs.edgeTargets[i]!;
        }
        this.wasmEngine.addEdgesFromPairs(edgePairs);
      }
    }

    // 9. Update counts and sync
    this.state.edgeCount = gs.edgeCount;

    this.syncParsedGraphFromState();
    this.uploadAlgorithmEdgeData(this.gpuContext.device);

    // Re-apply type styles after GPU flush to prevent CPU shadow from clobbering styled data
    this.applyTypeStyles();

    this.bumpSimulationAlpha(0.3);

    // 10. Emit batch summary
    this.events.emit({
      type: "graph:mutate",
      timestamp: Date.now(),
      nodesAdded: 0,
      nodesRemoved: 0,
      edgesAdded: 0,
      edgesRemoved: removedCount,
    });

    return removedCount;
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
      toArrayBuffer(gs.nodeAttributes.subarray(0, hw * NODE_ATTR_FLOATS)),
    );

    // Zero velocities/forces for all slots
    const zeros = new Float32Array(hw * 2);
    device.queue.writeBuffer(this.simBuffers.velocities, 0, zeros);
    device.queue.writeBuffer(this.simBuffers.velocitiesOut, 0, zeros);
    device.queue.writeBuffer(this.simBuffers.forces, 0, zeros);
  }

  /**
   * Upload one slot's flag word from the CPU shadow.
   */
  private flushNodeFlagSlot(slot: number): void {
    if (!this.graphState || !this.simBuffers) return;
    this.gpuContext.device.queue.writeBuffer(
      this.simBuffers.nodeFlags,
      slot * 4,
      toArrayBuffer(this.graphState.nodeFlagsShadow.subarray(slot, slot + 1)),
    );
    this.refreshActiveIndices();
  }

  /**
   * Upload the flag words for slots [lo, hi) from the CPU shadow in a single
   * queue write. Callers that touch a scattered set of slots should still use
   * one range spanning them all: the buffer is u32-per-slot, so even the whole
   * high-water range is one small write, where per-slot writes are one queue
   * operation each.
   */
  private flushNodeFlagRange(lo: number, hi: number): void {
    if (!this.graphState || !this.simBuffers || hi <= lo) return;
    this.gpuContext.device.queue.writeBuffer(
      this.simBuffers.nodeFlags,
      lo * 4,
      toArrayBuffer(this.graphState.nodeFlagsShadow.subarray(lo, hi)),
    );
    this.refreshActiveIndices();
  }

  /**
   * Re-derive the simulation's active-index list from the node-flag shadow and
   * upload it if it moved.
   *
   * Called from the two flag-upload helpers rather than from their callers, so
   * every writer of visibility or liveness — `writeNodeSlotLiveness`,
   * `applyLodVisibility`, `setNodeVisibility`, `setNodePinnedState`,
   * `flushNodeSlotFlagsToGPU` — maintains the list by construction. The flags
   * stay the only state anyone sets; the list is derived, never set.
   *
   * The rebuild is O(nodeHighWater) with no allocation (~0.3 ms at 220 000
   * slots, inside the design's transition budget), and the comparison lets a
   * pin toggle — which changes flags without changing the active set — cost no
   * upload at all.
   *
   * Nothing here touches simulation alpha: an LOD transition must never reheat
   * the simulation.
   */
  private refreshActiveIndices(): void {
    const gs = this.graphState;
    const buffers = this.simBuffers;
    if (!gs || !buffers) return;

    const shadow = this.activeIndexShadow ??
      (this.activeIndexShadow = new Uint32Array(buffers.nodeCapacity));
    if (activeIndicesUnchanged(gs.nodeFlagsShadow, gs.nodeHighWater, shadow, buffers.activeCount)) {
      return;
    }
    const count = deriveActiveIndices(gs.nodeFlagsShadow, gs.nodeHighWater, shadow);
    uploadLiveIndices(this.gpuContext.device, buffers, shadow.subarray(0, count));
  }

  /**
   * Return every slot to unit mass.
   *
   * Runs with the hierarchy invalidation on any topology change: a rolled-up
   * mass names a subtree of the *old* slot mapping, and removals compact slots,
   * so keeping it would leave a proxy weight on whatever node inherited the
   * slot. The collapse state is the LOD controller's to rebuild from the new
   * hierarchy; the buffer's job is to be neutral until it does.
   */
  private resetNodeMass(): void {
    if (!this.simBuffers) return;
    if (this.nodeMassShadow === null) return; // already unit everywhere
    this.nodeMassShadow = null;
    writeUnitNodeMass(
      this.gpuContext.device,
      this.simBuffers.nodeMass,
      this.simBuffers.nodeCapacity,
    );
  }

  /**
   * Upload per-slot simulation masses, writing only the range that changed.
   *
   * This is the LOD collapse path's other half: `rollUpMass` (lod/mass.ts)
   * decides what each slot weighs, this puts it on the GPU. The mass buffer is
   * allocated at capacity and never reallocated by an LOD operation, so a
   * collapse or expand is a contents write — no bind group is rebuilt and the
   * ping-pong parity sets are untouched.
   *
   * `mass` is indexed by GPU slot and covers `[0, mass.length)`; slots at or
   * beyond that keep whatever they hold, which is {@link NODE_MASS_UNIT} on a
   * freshly allocated buffer. Callers that shrink their collapse set must pass
   * the full array with those slots back at unit mass rather than a shorter
   * one, or an old proxy stays heavy.
   *
   * Nothing here touches simulation alpha: an LOD transition must never reheat
   * the simulation.
   *
   * @throws GraphMotherError if `mass` exceeds the buffer capacity or carries a
   *   value that is not a finite non-negative number.
   */
  uploadNodeMass(mass: Float32Array): void {
    if (!this.simBuffers) return;
    const shadow = this.nodeMassShadow ??
      (this.nodeMassShadow = new Float32Array(this.simBuffers.nodeCapacity).fill(NODE_MASS_UNIT));

    const { lo, hi } = commitNodeMass(shadow, mass);
    if (lo === hi) return;
    this.gpuContext.device.queue.writeBuffer(
      this.simBuffers.nodeMass,
      lo * 4,
      toArrayBuffer(shadow.subarray(lo, hi)),
    );
  }

  /**
   * Render the declared slots as collapsed bubbles at the given radii, and give
   * every slot that has stopped being one its own radius back.
   *
   * A collapsed parent is drawn through the ordinary node pipeline as a single
   * instance at its well radius (Phase 3 §5.4 v1) — no metaballs and, expressly,
   * nothing built on `SimpleContourLayer`, which is documented broken. Only the
   * render radius moves: the collision radius stays the node's own, so a
   * transition changes nothing the physics reads. That holds because the two
   * readers of the attribute row for a physical radius go through
   * {@link ProxyRadiusTable.savedRadiusOf} first — the row itself is where the
   * inflated value lives, so it cannot be trusted on its own.
   *
   * `proxies` is complete, not a diff, so a caller cannot leak an inflated
   * radius by forgetting to name an expansion.
   */
  private setCollapsedProxies(proxies: Uint32Array, radii: Float32Array): void {
    if (!this.graphState) return;
    this.flushNodeAttributeRows(
      this.lodProxies.declare(proxies, proxies.length, radii, this.proxyRadiusHost()),
    );
    // Proxy radii are a label-ranking input, so the cached glyph layout is
    // stale even though nothing moved.
    this.labelNodeStateVersion++;
  }

  /**
   * Un-inflate every collapsed proxy after a topology change.
   *
   * Runs beside {@link resetNodeMass} and for the same reason: the collapse set
   * belongs to the hierarchy that just went stale.
   */
  private resetLodProxyRadii(): void {
    if (this.lodProxies.size === 0 || !this.graphState) return;
    this.flushNodeAttributeRows(this.lodProxies.release(this.proxyRadiusHost()));
  }

  /**
   * Aggregate the edge set against the LOD cut and hand it to both consumers.
   *
   * The walk is WASM's: mapping every endpoint to its lowest visible ancestor
   * is the one genuinely O(E) step of a transition, and the design's budget for
   * the whole transition is 4 ms. There is deliberately no TypeScript
   * derivation — the same no-fallback rule the containment hierarchy follows,
   * and for the same reason: a second implementation is a second answer.
   *
   * Nothing here touches simulation alpha: an LOD transition must never reheat
   * the simulation.
   */
  private aggregateLodEdges(visible: Uint8Array): void {
    const gs = this.graphState;
    if (!gs || !this.simBuffers) return;
    const hierarchy = this.getHierarchy();
    if (!hierarchy) return;

    if (!this.wasmEngine) {
      if (!this.edgeAggregationUnavailableWarned) {
        this.edgeAggregationUnavailableWarned = true;
        console.warn(
          "[GraphMother] no WASM engine; LOD edge aggregation is unavailable. Collapsing a " +
            "subtree will drop its cross-cutting dependency attractions rather than " +
            "transferring them to the proxy, so the collapsed layout will differ from the " +
            "expanded one.",
        );
      }
      return;
    }

    const edgeCount = gs.edgeCount;
    const aggregation = aggregateEdges(
      this.wasmEngine,
      gs.edgeSources.subarray(0, edgeCount),
      gs.edgeTargets.subarray(0, edgeCount),
      hierarchy.columns.parent,
      visible,
    );

    uploadEdgeBundles(
      this.gpuContext.device,
      this.simBuffers,
      aggregation.liveEdges,
      aggregation.bundles,
    );
    this.uploadBundleInstances(aggregation);
    this.flushEdgeAttributeRows(
      this.lodEdgeOpacity.apply(aggregation.liveEdges, edgeCount, this.edgeOpacityHost()),
    );
  }

  /**
   * Return the spring pass and the edge render to the source edge list.
   *
   * Called when the cut is released, and on any topology change: an
   * aggregation names slots and edge indices of the graph it was computed
   * against, and both move underneath it.
   */
  private releaseLodEdgeAggregation(): void {
    if (this.simBuffers) releaseEdgeBundles(this.simBuffers);
    this.lodBundleDrawCount = 0;
    if (this.lodEdgeOpacity.size > 0) {
      this.flushEdgeAttributeRows(this.lodEdgeOpacity.release(this.edgeOpacityHost()));
    }
    this.markRenderDirty();
  }

  /**
   * The two edge-attribute operations the opacity mask drives.
   *
   * Edges are identified by index and nothing else — unlike a node, an edge has
   * no producer id to follow — so an index past the live edge count names an
   * edge a mutation has already removed and is dropped. That is the one case
   * where a restore is lost: swap-remove moves edges, so a saved opacity can
   * only be given back to whatever now holds its index. Leaving an edge
   * permanently invisible would be worse, and the value in question is the
   * default for every edge the consumer has not restyled.
   */
  private edgeOpacityHost() {
    const gs = this.graphState!;
    return {
      opacityOf: (edge: number) => gs.edgeAttributes[edge * EDGE_ATTR_FLOATS + EDGE_ATTR_OPACITY],
      setOpacity: (edge: number, opacity: number) => {
        if (edge < 0 || edge >= gs.edgeCount) return false;
        const index = edge * EDGE_ATTR_FLOATS + EDGE_ATTR_OPACITY;
        if (gs.edgeAttributes[index] === opacity) return false;
        gs.edgeAttributes[index] = opacity;
        return true;
      },
    };
  }

  /**
   * Upload the attribute rows of `edges` as one write per contiguous run.
   *
   * The edge twin of {@link flushNodeAttributeRows}, and safe for the same
   * reason: `gs.edgeAttributes` is the CPU-side authority for the render
   * buffer, so a run may span edges the caller never touched.
   */
  private flushEdgeAttributeRows(edges: number[]): void {
    const gs = this.graphState;
    if (!gs || !this.buffers || edges.length === 0) return;

    const { device } = this.gpuContext;
    const buffer = this.buffers.edgeAttributes;
    forEachSlotRun(edges, edges.length, (lo, hi) => {
      device.queue.writeBuffer(
        buffer,
        lo * EDGE_ATTR_FLOATS * 4,
        toArrayBuffer(gs.edgeAttributes.subarray(lo * EDGE_ATTR_FLOATS, hi * EDGE_ATTR_FLOATS)),
      );
    });
    this.markRenderDirty();
  }

  /**
   * Write the bundle instances the edge pipeline draws this frame.
   *
   * Bundles go through the ordinary edge pipeline — same shader, same vertex
   * layout, a second bind group over their own index and attribute buffers —
   * so a collapsed dependency looks like the edges it replaces, only thicker.
   */
  private uploadBundleInstances(aggregation: EdgeAggregation): void {
    this.lodBundleDrawCount = 0;
    if (!this.buffers || aggregation.bundleCount === 0) {
      this.markRenderDirty();
      return;
    }

    const instances = buildBundleInstances(
      aggregation.bundles,
      aggregation.bundleCount,
      LOD_BUNDLE_STYLE,
      EDGE_ATTR_FLOATS,
      this.lodBundleScratch,
    );

    const buffers = this.ensureLodEdgeRenderBuffers();
    const { device } = this.gpuContext;
    device.queue.writeBuffer(buffers.indices, 0, toArrayBuffer(instances.indices));
    device.queue.writeBuffer(buffers.attributes, 0, toArrayBuffer(instances.attributes));
    this.lodBundleDrawCount = instances.count;
    this.markRenderDirty();
  }

  /** Allocate the bundle render buffers at edge capacity on first use. */
  private ensureLodEdgeRenderBuffers(): { indices: GPUBuffer; attributes: GPUBuffer } {
    const capacity = Math.max(this.buffers!.edgeCapacity, 1);
    const existing = this.lodEdgeRenderBuffers;
    if (existing && existing.capacity >= capacity) return existing;

    this.destroyLodEdgeRenderBuffers();
    const { device } = this.gpuContext;
    const created = {
      indices: device.createBuffer({
        label: "LOD Bundle Indices",
        size: capacity * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      attributes: device.createBuffer({
        label: "LOD Bundle Attributes",
        size: capacity * EDGE_ATTR_FLOATS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      capacity,
    };
    this.lodEdgeRenderBuffers = created;
    this.rebuildLodEdgeBindGroup();
    return created;
  }

  /** Drop the bundle render buffers and the bind group built against them. */
  private destroyLodEdgeRenderBuffers(): void {
    const existing = this.lodEdgeRenderBuffers;
    if (!existing) return;
    existing.indices.destroy();
    existing.attributes.destroy();
    this.lodEdgeRenderBuffers = null;
    this.lodBundleDrawCount = 0;
    this.lodEdgeBindGroupSlot.clear();
  }

  /** Rebuild the bundle draw's bind group for both ping-pong parities. */
  private rebuildLodEdgeBindGroup(): void {
    const buffers = this.lodEdgeRenderBuffers;
    if (!buffers || !this.edgePipeline) {
      this.lodEdgeBindGroupSlot.clear();
      return;
    }
    const { device } = this.gpuContext;
    const edgePipeline = this.edgePipeline;
    this.lodEdgeBindGroupSlot.rebuild(
      (view) =>
        createEdgeBindGroup(
          device,
          edgePipeline,
          view.positions,
          buffers.indices,
          buffers.attributes,
        ),
    );
  }

  /** The per-slot readings and radius write the proxy table drives. */
  private proxyRadiusHost(): ProxyRadiusHost {
    const gs = this.graphState!;
    return {
      radiusOf: (slot) => gs.nodeAttributes[slot * NODE_ATTR_FLOATS],
      externalIdOf: (slot) => externalIdForSlot(gs, slot),
      slotOf: (externalId) => slotForExternalId(gs, externalId),
      setRadius: (slot, radius) => {
        if (slot < 0 || slot >= gs.nodeHighWater) return false;
        const index = slot * NODE_ATTR_FLOATS;
        if (gs.nodeAttributes[index] === radius) return false;
        gs.nodeAttributes[index] = radius;
        return true;
      },
    };
  }

  /**
   * Upload the attribute rows of `slots` as one write per contiguous run.
   *
   * A run may span slots the caller never touched, which is safe here and only
   * here: the attribute shadow is the CPU-side authority for the render buffer,
   * where the position shadow merely lags it between readbacks. Sorting is what
   * makes a depth-first collapse set a single queue operation.
   */
  private flushNodeAttributeRows(slots: number[]): void {
    const gs = this.graphState;
    if (!gs || !this.buffers || slots.length === 0) return;
    slots.sort((a, b) => a - b);

    const { device } = this.gpuContext;
    const buffer = this.buffers.nodeAttributes;
    forEachSlotRun(slots, slots.length, (lo, hi) => {
      device.queue.writeBuffer(
        buffer,
        lo * NODE_ATTR_BYTES,
        toArrayBuffer(gs.nodeAttributes.subarray(lo * NODE_ATTR_FLOATS, hi * NODE_ATTR_FLOATS)),
      );
    });
    this.markRenderDirty();
  }

  /**
   * Translate the positions of `[lo, hi)` by `(dx, dy)`, CPU shadow and GPU.
   *
   * The LOD expand fix-up: a folded subtree is frozen while its proxy goes on
   * simulating, so it is moved by the proxy's drift before it is revealed.
   * Unlike {@link setNodePosition} this neither pins nor disturbs alpha — an
   * LOD transition must never reheat the simulation, and the nodes involved are
   * being restored to an arrangement, not placed by a user.
   *
   * Both ping-pong position buffers are written, because the caller does not
   * know which orientation the next tick reads.
   */
  private translateNodeRange(lo: number, hi: number, dx: number, dy: number): void {
    const gs = this.graphState;
    if (!gs || !this.buffers) return;
    const start = Math.max(0, lo);
    const end = Math.min(hi, gs.nodeHighWater);
    if (end <= start) return;

    const moved = new Float32Array((end - start) * 2);
    for (let slot = start; slot < end; slot++) {
      const x = gs.positionsX[slot] + dx;
      const y = gs.positionsY[slot] + dy;
      gs.positionsX[slot] = x;
      gs.positionsY[slot] = y;
      moved[(slot - start) * 2] = x;
      moved[(slot - start) * 2 + 1] = y;
    }

    const { device } = this.gpuContext;
    const offset = start * 8;
    device.queue.writeBuffer(this.buffers.positions, offset, moved);
    if (this.simBuffers) {
      device.queue.writeBuffer(this.simBuffers.positions, offset, moved);
      device.queue.writeBuffer(this.simBuffers.positionsOut, offset, moved);
    }
    this.markRenderDirty();
  }

  /**
   * Update one slot's liveness: the dead-slot flag in nodeFlags and the
   * collision radius. Pass the node's radius to mark the slot live, or
   * undefined to mark it dead (skipped by simulation and collision shaders).
   *
   * Both call sites (re)assign the slot to a fresh or removed node, so the
   * whole flag word is reset — any pin or hidden state held by the slot's
   * previous occupant belongs to a node that no longer exists.
   */
  private writeNodeSlotLiveness(slot: number, radius: number | undefined): void {
    const { device } = this.gpuContext;
    const dead = radius === undefined;

    this.pinnedNodes.delete(slot);
    this.graphState?.resetNodeFlags(slot, dead ? NODE_FLAG_DEAD : 0);
    this.flushNodeFlagSlot(slot);

    if (this.collisionBuffers) {
      device.queue.writeBuffer(
        this.collisionBuffers.nodeSizes,
        slot * 4,
        new Float32Array([dead ? DEAD_SLOT_RADIUS : radius]),
      );
      if (!dead && radius > this.maxNodeRadius) {
        this.maxNodeRadius = radius;
      }
    }
  }

  /**
   * Rewrite all per-slot liveness data ([0, nodeHighWater)) from graph state.
   * Used after batch mutations and buffer reallocation, where slots may have
   * moved (compaction) or the GPU buffers were recreated.
   *
   * The flag words come straight from the CPU shadow rather than being
   * recomposed from nodeFreeSet and pinnedNodes: recomposition can only carry
   * the bits it knows about, and would silently drop the rest.
   */
  private flushNodeSlotFlagsToGPU(): void {
    if (!this.graphState || !this.simBuffers) return;

    const gs = this.graphState;
    const { device } = this.gpuContext;
    const hw = gs.nodeHighWater;

    this.flushNodeFlagRange(0, hw);

    if (this.collisionBuffers) {
      const { sizes, maxRadius } = collisionRadiusColumn(gs.nodeAttributes, hw, {
        isDead: (slot) => gs.nodeFreeSet.has(slot),
        proxyRadius: (slot) => this.lodProxies.savedRadiusOf(slot),
        deadRadius: DEAD_SLOT_RADIUS,
      });
      // Recomputed, not ratcheted: this covers every live slot, so a radius
      // that has since shrunk must be allowed to bring the grid cell size back
      // down with it.
      this.maxNodeRadius = maxRadius > 0 ? maxRadius : DEFAULT_COLLISION_RADIUS;
      device.queue.writeBuffer(this.collisionBuffers.nodeSizes, 0, toArrayBuffer(sizes));
    }
  }

  /**
   * Flush all edge source/target/attribute data to GPU.
   * Used for batch operations. Writes the entire live edge range.
   */
  private flushEdgeBuffersToGPU(): void {
    if (!this.graphState || !this.buffers || !this.simBuffers) return;

    const gs = this.graphState;
    const { device } = this.gpuContext;
    const ec = gs.edgeCount;

    // Interleave sources/targets for the render edge-indices buffer
    const edgeIndices = new Uint32Array(ec * 2);
    for (let i = 0; i < ec; i++) {
      edgeIndices[i * 2] = gs.edgeSources[i]!;
      edgeIndices[i * 2 + 1] = gs.edgeTargets[i]!;
    }
    device.queue.writeBuffer(this.buffers.edgeIndices, 0, toArrayBuffer(edgeIndices));

    // Edge attributes
    device.queue.writeBuffer(
      this.buffers.edgeAttributes,
      0,
      toArrayBuffer(gs.edgeAttributes.subarray(0, ec * 8)),
    );

    // Simulation edge data
    device.queue.writeBuffer(
      this.simBuffers.edgeSources,
      0,
      toArrayBuffer(gs.edgeSources.subarray(0, ec)),
    );
    device.queue.writeBuffer(
      this.simBuffers.edgeTargets,
      0,
      toArrayBuffer(gs.edgeTargets.subarray(0, ec)),
    );
  }

  /**
   * Re-point `parsedGraph` at graph state after a mutation.
   *
   * `MutableGraphState` is the single authority; `parsedGraph` is a view of it
   * kept for the readers written against the load-time shape (see
   * {@link aliasParsedGraphToState}).
   */
  private syncParsedGraphFromState(): void {
    if (!this.graphState || !this.state.parsedGraph) return;
    aliasParsedGraphToState(this.state.parsedGraph, this.graphState);
  }

  /**
   * Ensure algorithm buffers can handle the current nodeHighWater.
   * Recreates algorithm buffers/bind groups if nodeHighWater exceeds their maxNodes.
   */
  private ensureAlgorithmCapacity(): void {
    if (
      !this.currentAlgorithm || !this.algorithmPipelines || !this.algorithmBuffers ||
      !this.simBuffers || !this.graphState
    ) return;

    const gs = this.graphState;
    const algMaxNodes = (this.algorithmBuffers as unknown as { maxNodes?: number }).maxNodes;
    if (algMaxNodes === undefined || gs.nodeHighWater <= algMaxNodes) return;

    // Algorithm buffers are too small — recreate with current nodeCapacity
    const { device } = this.gpuContext;
    const newCap = gs.nodeCapacity;

    this.algorithmBuffers.destroy();
    this.algorithmBuffers = this.currentAlgorithm.createBuffers(device, newCap);
    this.rebuildAlgorithmBindGroups();

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
    // Reset convergence detection — new data means the graph needs to re-settle
    this.convergenceCheckCount = 0;
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
      size: newCapacity * NODE_ATTR_BYTES,
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
      toArrayBuffer(gs.nodeAttributes.subarray(0, hw * NODE_ATTR_FLOATS)),
    );

    // === Simulation buffers ===
    // Destroy old sim node buffers
    this.simBuffers.positions.destroy();
    this.simBuffers.positionsOut.destroy();
    this.simBuffers.velocities.destroy();
    this.simBuffers.velocitiesOut.destroy();
    this.simBuffers.forces.destroy();
    this.simBuffers.prevForces.destroy();
    this.simBuffers.nodeFlags.destroy();
    this.simBuffers.nodeAlpha.destroy();
    this.simBuffers.nodeMass.destroy();
    this.simBuffers.nodeDepth.destroy();
    this.simBuffers.liveIndices.destroy();
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
    this.simBuffers.prevForces = device.createBuffer({
      label: "Sim Prev Forces",
      size: nodeVec2Bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.simBuffers.nodeFlags = device.createBuffer({
      label: "Sim Node Flags",
      size: nodeFlagBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // A capacity change is not an LOD operation, so nothing is mid-crossfade
    // that this must preserve: the buffer comes back fully opaque, and a
    // scheduler driving it re-flushes its shadow after resizing.
    this.simBuffers.nodeAlpha = device.createBuffer({
      label: "Sim Node Alpha",
      size: nodeFlagBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    writeOpaqueNodeAlpha(device, this.simBuffers.nodeAlpha, newCapacity);
    // Likewise not an LOD operation: masses come back at unit and the CPU
    // shadow is resized to match, so the next uploadNodeMass diffs against
    // what the GPU actually holds.
    this.simBuffers.nodeMass = device.createBuffer({
      label: "Sim Node Mass",
      size: nodeFlagBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    writeUnitNodeMass(device, this.simBuffers.nodeMass, newCapacity);
    this.nodeMassShadow = null;
    // Also not an LOD operation: the list comes back as the identity over the
    // new capacity, and the shadow is dropped so the next flag write derives
    // the real active set against what the GPU actually holds.
    this.simBuffers.liveIndices = device.createBuffer({
      label: "Sim Live Indices",
      size: nodeFlagBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    writeIdentityLiveIndices(device, this.simBuffers.liveIndices, newCapacity);
    this.simBuffers.activeCount = this.graphState?.nodeHighWater ?? this.simBuffers.nodeCount;
    this.activeIndexShadow = null;
    this.simBuffers.nodeDepth = device.createBuffer({
      label: "Sim Node Depth",
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
    device.queue.writeBuffer(this.simBuffers.prevForces, 0, zeros);

    // === Rebuild simulation/render bind groups for the new buffers ===
    this.rebuildSimulationBindGroups();

    // === Rebuild algorithm buffers if they're smaller than new capacity ===
    if (this.currentAlgorithm && this.algorithmPipelines) {
      this.algorithmBuffers?.destroy();
      this.algorithmBuffers = this.currentAlgorithm.createBuffers(device, newCapacity);
      this.uploadAlgorithmEdgeData(device);
    }
    // Runs even without an algorithm-buffer recreation: the algorithm binds
    // the simulation position/force/flag buffers replaced above.
    this.rebuildAlgorithmBindGroups();

    // === Rebuild collision buffers (rebuilds their bind groups too) ===
    this.initializeCollisionResources(device, newCapacity, gs.nodeAttributes);

    // Restore per-slot liveness — the recreated nodeFlags buffer is zeroed, so
    // every pin and every LOD-hidden bit has to be written back from the shadow
    this.flushNodeSlotFlagsToGPU();

    // The recreated alpha buffer is fully opaque; the crossfade scheduler
    // rebuilds its shadow against the live cut and re-uploads it whole.
    this.lodController?.handleNodeCapacityChange(newCapacity, performance.now());

    // === Update layer render contexts ===
    this.refreshLayerRenderContexts();

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

    // The bundle render buffers are sized to the edge capacity; the next
    // aggregation reallocates them against the new one.
    this.destroyLodEdgeRenderBuffers();

    // === Simulation edge buffers ===
    this.simBuffers.edgeSources.destroy();
    this.simBuffers.edgeTargets.destroy();
    this.simBuffers.lodEdgeSet.destroy();

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
    // Not an LOD operation: the aggregation named the edge set that just
    // changed, so it is released rather than carried, and the next band
    // transition recomputes it against the new one.
    this.simBuffers.lodEdgeSet = device.createBuffer({
      label: "Sim LOD Edge Set",
      size: edgeBytes * LOD_EDGE_SET_WORDS_PER_EDGE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    releaseEdgeBundles(this.simBuffers);
    this.simBuffers.edgeCapacity = newCapacity;

    // Upload edge source/target data
    if (ec > 0) {
      device.queue.writeBuffer(
        this.simBuffers.edgeSources,
        0,
        toArrayBuffer(gs.edgeSources.subarray(0, ec)),
      );
      device.queue.writeBuffer(
        this.simBuffers.edgeTargets,
        0,
        toArrayBuffer(gs.edgeTargets.subarray(0, ec)),
      );
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

  // ==========================================================================
  // Bind group construction
  //
  // Every bind group below references at least one ping-pong buffer, so each
  // is built twice — once per BufferParity, by its ParitySlot — and the
  // per-frame path only flips the index (see advanceFrameParity). These MUST
  // be re-run whenever any referenced buffer is reallocated (load, node/edge
  // capacity growth, algorithm switch, collision buffer recreation): a bind
  // group holding a destroyed buffer is a device-loss bug. A missed rebuild
  // that leaves a slot pointing at a replaced positions buffer is caught by
  // the slot's own staleness check on the next read.
  // ==========================================================================

  /**
   * Rebuild the simulation compute bind groups and the node/edge render bind
   * groups. Call after reallocating any simulation, node-attribute, or edge
   * buffer.
   */
  private rebuildSimulationBindGroups(): void {
    if (!this.simBuffers || !this.simulationPipeline) return;

    const { device } = this.gpuContext;
    const pipeline = this.simulationPipeline;

    this.simBindGroupSlot.rebuild((view) => createSimulationBindGroups(device, pipeline, view));

    // Render reads the simulation's current position buffer directly — the
    // separate render position buffer is only the pre-simulation upload target.
    if (this.nodePipeline && this.buffers) {
      const nodePipeline = this.nodePipeline;
      const nodeAttributes = this.buffers.nodeAttributes;
      this.nodeBindGroupSlot.rebuild(
        (view) =>
          createNodeBindGroup(
            device,
            nodePipeline,
            view.positions,
            nodeAttributes,
            view.nodeFlags,
            view.nodeAlpha,
          ),
      );
    }

    if (this.edgePipeline && this.buffers) {
      const edgePipeline = this.edgePipeline;
      const { edgeIndices, edgeAttributes } = this.buffers;
      this.edgeBindGroupSlot.rebuild(
        (view) =>
          createEdgeBindGroup(device, edgePipeline, view.positions, edgeIndices, edgeAttributes),
      );
    }

    // Bundles read the same ping-pong positions buffer as the source edges, so
    // their bind group has to be rebuilt on exactly the same occasions.
    this.rebuildLodEdgeBindGroup();
  }

  /**
   * Rebuild the current algorithm's bind groups for both parities. Call after
   * creating or recreating the algorithm's buffers, or after reallocating any
   * simulation buffer the algorithm binds.
   */
  private rebuildAlgorithmBindGroups(): void {
    if (
      !this.simBuffers || !this.currentAlgorithm || !this.algorithmPipelines ||
      !this.algorithmBuffers
    ) {
      this.algorithmBindGroupSlot.clear();
      return;
    }

    const { device } = this.gpuContext;
    const algorithm = this.currentAlgorithm;
    const pipelines = this.algorithmPipelines;
    const algorithmBuffers = this.algorithmBuffers;
    // O(n) over the CPU position shadow — computed once, not once per parity.
    const bounds = this.computeCurrentBounds();

    this.algorithmBindGroupSlot.rebuild(
      (view) =>
        algorithm.createBindGroups(
          device,
          pipelines,
          this.buildAlgorithmContext(view, bounds),
          algorithmBuffers,
        ),
    );
  }

  /**
   * Rebuild the collision bind groups for both parities. Call after creating
   * or recreating the collision/grid-collision buffers, or after reallocating
   * the simulation position or node-flag buffers.
   *
   * Collision binds positionsOut — the integration pass's write target — so
   * its corrections land in the buffer that becomes next frame's read buffer.
   */
  private rebuildCollisionBindGroups(): void {
    if (!this.simBuffers) return;

    const { device } = this.gpuContext;

    if (this.collisionPipeline && this.collisionBuffers) {
      const pipeline = this.collisionPipeline;
      const buffers = this.collisionBuffers;
      this.collisionBindGroupSlot.rebuild(
        (view) =>
          createCollisionBindGroup(
            device,
            pipeline,
            buffers,
            view.positionsOut,
            view.nodeFlags,
            view.liveIndices,
          ),
      );
    }

    if (this.gridCollisionPipeline && this.gridCollisionBuffers && this.collisionBuffers) {
      const pipeline = this.gridCollisionPipeline;
      const gridBuffers = this.gridCollisionBuffers;
      const { nodeSizes, displacements } = this.collisionBuffers;
      this.gridCollisionBindGroupSlot.rebuild(
        (view) =>
          createGridCollisionBindGroups(
            device,
            pipeline,
            gridBuffers,
            nodeSizes,
            displacements,
            view.positionsOut,
            view.nodeFlags,
            view.liveIndices,
          ),
      );
    }
  }

  /**
   * Rebuild every parity-dependent bind group. Used by the reallocation paths
   * that touch more than one resource group at once.
   */
  private rebuildAllBindGroups(): void {
    this.rebuildSimulationBindGroups();
    this.rebuildAlgorithmBindGroups();
    this.rebuildCollisionBindGroups();
  }

  /**
   * Bounding box of the CPU position shadow over all occupied slots, or
   * undefined when no graph is loaded or every position is non-finite.
   */
  private computeCurrentBounds(): BoundingBox | undefined {
    const parsed = this.state.parsedGraph;
    if (!parsed) return undefined;
    return computeBoundsFromPositions(
      parsed.positionsX,
      parsed.positionsY,
      this.graphState?.nodeHighWater ?? this.state.nodeCount,
    );
  }

  /**
   * Build the algorithm render context for one ping-pong orientation.
   *
   * `view` supplies the buffers; everything else is current frame state. Note
   * that createBindGroups is contractually forbidden from capturing the
   * non-buffer fields (see ForceAlgorithm.createBindGroups) — they exist for
   * updateUniforms, which runs per frame.
   */
  private buildAlgorithmContext(
    view: SimulationBuffers,
    bounds: BoundingBox | undefined,
  ): AlgorithmRenderContext {
    const edgeCount = this.state.edgeCount;
    return {
      device: this.gpuContext.device,
      positions: view.positions,
      forces: view.forces,
      nodeCount: this.graphState?.nodeHighWater ?? this.state.nodeCount,
      edgeCount,
      forceConfig: this.forceConfig,
      bounds,
      edgeSources: view.edgeSources,
      edgeTargets: view.edgeTargets,
      edgeSourcesData: this.graphState?.edgeSources.subarray(0, edgeCount),
      edgeTargetsData: this.graphState?.edgeTargets.subarray(0, edgeCount),
      nodeFlags: view.nodeFlags,
      nodeMass: view.nodeMass,
      // The LOD state, from the same fields the core pipeline reads, so a
      // plugin's passes are cut the same way the built-in ones are. The
      // buffers are stable for the buffer set's lifetime; the counts are
      // per-frame and are consumed by updateUniforms, never captured in a
      // bind group (see ForceAlgorithm.createBindGroups).
      liveIndices: view.liveIndices,
      activeCount: view.activeCount,
      lodEdgesActive: view.lodEdgesActive,
      lodEdgeSet: view.lodEdgeSet,
      activeEdgeCount: view.activeEdgeCount,
      bundleCount: view.bundleCount,
    };
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
    // Dead slots and LOD-hidden nodes are not content: both are frozen out of
    // the simulation, so their positions are stale (a hidden subtree keeps the
    // coordinates it had when it was folded, which after a settle can be
    // nowhere near its visible ancestor). Fitting to them frames empty space.
    const flags = this.graphState?.nodeFlagsShadow;
    const skipMask = NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD;
    // Nodes are discs, not points: a bounds over centers alone clips every
    // rim node by its own radius, which is very visible when LOD leaves a
    // handful of large folded bubbles. Same world-unit radius hit testing uses.
    const attributes = this.graphState?.nodeAttributes;

    // Calculate bounds
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;

    for (let i = 0; i < nodeCount; i++) {
      if (flags !== undefined && (flags[i] & skipMask) !== 0) continue;
      const x = positionsX[i];
      const y = positionsY[i];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const r = attributes === undefined ? 0 : attributes[i * NODE_ATTR_FLOATS];
      if (x - r < minX) minX = x - r;
      if (y - r < minY) minY = y - r;
      if (x + r > maxX) maxX = x + r;
      if (y + r > maxY) maxY = y + r;
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
   * Set simulation alpha target.
   *
   * Set to 0 for full convergence (simulation cools to rest, reheats on mutation).
   * Set to a positive value (e.g. 0.1) for continuous mode (never fully stops).
   */
  setAlphaTarget(target: number): void {
    this.simulationController.setAlphaTarget(target);
  }

  /**
   * Set simulation alpha decay rate.
   *
   * Controls how quickly the simulation cools down.
   * Lower values = slower cooling = more time to organize.
   * Default is 0.0228 (~300 iterations). Use calculateAlphaDecay() for target iteration counts.
   */
  setAlphaDecay(decay: number): void {
    this.simulationController.setConfig({ alphaDecay: decay });
  }

  /**
   * Set force configuration parameters.
   * Updates take effect immediately on the running simulation.
   *
   * @param config - Partial force configuration to merge with current config
   */
  setForceConfig(config: Partial<FullForceConfig>): void {
    const previous = this.forceConfig;
    this.forceConfig = validateForceConfig({
      ...this.forceConfig,
      ...config,
    });

    // The bubble knobs feed a load-time buffer write rather than the per-tick
    // uniforms, so they need an explicit re-upload (see bubbleUploadChanged).
    if (
      bubbleUploadChanged(previous, this.forceConfig) && this.state.loaded &&
      this.algorithmBuffers
    ) {
      // Inside the branch rather than at the top: the re-upload drops the
      // hierarchy and with it the fold anchors, but the calls that skip it
      // change nothing a fold was measured against, and this runs per knob
      // movement.
      this.beginTopologyChange();
      this.uploadAlgorithmEdgeData(this.gpuContext.device);
    }

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
  getAvailableAlgorithms(): Array<
    { id: string; name: string; description: string; complexity: string }
  > {
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
    let algorithm = registry.get(type);

    if (!algorithm) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        `Unknown force algorithm: ${type}. Available: ${
          registry.listInfo().map((i) => i.id).join(", ")
        }`,
      );
    }

    // Some algorithms bind more storage buffers per compute stage than the
    // WebGPU default of 8 (Barnes-Hut 10, Relativity Atlas 9). On a device
    // that cannot supply them, their bind group layouts are invalid — and an
    // invalid bind group poisons the compute pass, the pass poisons the
    // encoder, and submit() then discards the whole frame, springs and
    // integration included. That is a frozen canvas, not a slower layout, so
    // substitute a working algorithm and say so rather than honouring the
    // request. createPipelines throws on the same condition; this keeps the
    // public setter from being the thing that throws.
    const { device } = this.gpuContext;
    if (!supportsAlgorithmOnDevice(algorithm.info, device)) {
      const fallback = registry.get("n2") ?? registry.getRecommended(this.state.nodeCount, device);
      console.warn(
        `[GraphMother] ${algorithm.info.name} needs maxStorageBuffersPerShaderStage >= ` +
          `${algorithm.info.minStorageBuffersPerShaderStage}, but this device supports ` +
          `${device.limits.maxStorageBuffersPerShaderStage}. Falling back to ` +
          `${fallback?.info.name ?? "no algorithm"}.`,
      );
      if (!fallback || fallback.info.id === this.currentAlgorithm?.info.id) return;
      algorithm = fallback;
    }

    // Skip if already using this algorithm
    if (this.currentAlgorithm?.info.id === algorithm.info.id) {
      return;
    }

    // Destroy old algorithm resources. Clearing the bind group sets first is
    // what keeps the compute path from binding groups whose buffers are gone.
    this.algorithmBindGroupSlot.clear();
    this.algorithmBuffers?.destroy();
    this.algorithmBuffers = null;

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

      this.rebuildAlgorithmBindGroups();

      // The switch drops the hierarchy along with everything else derived from
      // the topology, so a live fold is owed its drift here exactly as it is
      // before a mutation — nothing above this point moved a node.
      this.beginTopologyChange();

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

  /**
   * Compute tidy tree layout and upload target positions to GPU.
   *
   * This uses the WASM Buchheim algorithm to compute analytical target
   * positions, then uploads them to the GPU for spring-force animation.
   * The current algorithm must be "tidy-tree".
   *
   * @param rootId - Root node ID, or undefined for auto-detection
   */
  computeTreeLayout(rootId?: number): void {
    if (!this.wasmEngine) {
      throw new GraphMotherError(
        ErrorCode.WASM_LOAD_FAILED,
        "WASM engine not available for tree layout computation",
      );
    }

    if (!(this.currentAlgorithm instanceof TidyTreeAlgorithm)) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Tree layout requires tidy-tree algorithm. Call setForceAlgorithm('tidy-tree') first.",
      );
    }

    const config = this.forceConfig;
    const root = rootId ?? 0xFFFFFFFF; // u32::MAX = auto-detect

    const positions = this.wasmEngine.computeTreeLayoutFromGraph(
      root,
      config.tidyTreeLevelSeparation,
      config.tidyTreeSiblingSeparation,
      config.tidyTreeSubtreeSeparation,
      config.tidyTreeRadial,
    );

    this.currentAlgorithm.uploadTargetPositions(
      this.gpuContext.device,
      positions,
    );

    // Reheat simulation so spring forces can animate toward targets
    this.simulationController.setAlpha(1.0);
  }

  /**
   * Compute community layout and upload target positions to GPU.
   *
   * Detects communities using Louvain modularity optimization, then computes
   * a circular cluster layout. The current algorithm must be "community".
   *
   * @param rootId - Unused (present for API symmetry with tree layout)
   */
  computeCommunityLayout(): void {
    if (!this.wasmEngine) {
      throw new GraphMotherError(
        ErrorCode.WASM_LOAD_FAILED,
        "WASM engine not available for community layout computation",
      );
    }

    if (!(this.currentAlgorithm instanceof CommunityLayoutAlgorithm)) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Community layout requires community algorithm. Call setForceAlgorithm('community') first.",
      );
    }

    const config = this.forceConfig;

    // Detect communities via Louvain (returns assignments + count as last element)
    const raw = this.wasmEngine.detectCommunities(
      config.communityResolution,
      config.communityMaxIterations,
      0.0001, // convergence threshold
    );

    // Last element is community count, rest are assignments
    const communityCount = raw[raw.length - 1];
    const assignments = new Uint32Array(raw.buffer as ArrayBuffer, raw.byteOffset, raw.length - 1);

    // Log community detection results for diagnostics
    const commDistribution = new Map<number, number>();
    for (let i = 0; i < assignments.length; i++) {
      commDistribution.set(assignments[i], (commDistribution.get(assignments[i]) ?? 0) + 1);
    }
    const sizes = [...commDistribution.values()].sort((a, b) => b - a);
    console.log(
      `[Community] Detected ${communityCount} communities from ${assignments.length} nodes. ` +
        `Largest: ${sizes.slice(0, 5).join(", ")}`,
    );

    this.currentAlgorithm.uploadCommunityIds(
      this.gpuContext.device,
      assignments,
      communityCount,
    );

    // Reheat simulation so cluster forces can take effect
    this.simulationController.setAlpha(1.0);
  }

  /**
   * Compute codebase layout and upload target positions to GPU.
   *
   * Uses the graph's edges as containment hierarchy and produces a
   * circle-packing layout. The current algorithm must be "codebase".
   *
   * @param nodeCategories - Uint8Array mapping node IDs to categories
   *   (0=repo, 1=dir, 2=file, 3=symbol, 4=other).
   *   If not provided, all nodes are treated as "other" (category 4).
   * @param rootId - Root node ID, or undefined for auto-detection
   */
  computeCodebaseLayout(
    _nodeCategories?: Uint8Array,
    _rootId?: number,
  ): void {
    if (!this.wasmEngine) {
      throw new GraphMotherError(
        ErrorCode.WASM_LOAD_FAILED,
        "WASM engine not available for codebase layout computation",
      );
    }

    if (!(this.currentAlgorithm instanceof CodebaseLayoutAlgorithm)) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Codebase layout requires codebase algorithm. Call setForceAlgorithm('codebase') first.",
      );
    }

    const config = this.forceConfig;

    // Detect communities via Louvain using the graph's natural structure.
    // For hierarchical codebases, Louvain detects subtree clusters automatically.
    const raw = this.wasmEngine.detectCommunities(
      config.communityResolution,
      config.communityMaxIterations,
      0.0001,
    );

    const communityCount = raw[raw.length - 1];
    const assignments = new Uint32Array(raw.buffer as ArrayBuffer, raw.byteOffset, raw.length - 1);

    // Log community detection results for diagnostics
    const commDist = new Map<number, number>();
    for (let i = 0; i < assignments.length; i++) {
      commDist.set(assignments[i], (commDist.get(assignments[i]) ?? 0) + 1);
    }
    const sizes2 = [...commDist.values()].sort((a, b) => b - a);
    console.log(
      `[Codebase] Detected ${communityCount} communities from ${assignments.length} nodes. ` +
        `Largest: ${sizes2.slice(0, 5).join(", ")}`,
    );

    this.currentAlgorithm.uploadCommunityIds(
      this.gpuContext.device,
      assignments,
      communityCount,
    );

    // Reheat simulation so cluster forces can take effect
    this.simulationController.setAlpha(1.0);
  }

  // ==========================================================================
  // Public API - Render Control
  // ==========================================================================

  /**
   * Stop presenting frames.
   *
   * For hosts that know the graph surface is not being seen while the page
   * itself is visible: a canvas covered by other UI, or in a collapsed panel,
   * still receives requestAnimationFrame callbacks and `document.hidden` stays
   * false, so nothing inside the engine can detect the occlusion. The host can.
   *
   * Independent of the simulation — physics keeps converging unless
   * {@link pauseSimulation} is called as well, and a paused renderer will show
   * the settled layout on resume. Idempotent, and it survives anything that
   * restarts the render loop internally: a {@link load} while paused stays
   * paused. Only {@link resumeRendering} lifts it, never a visibility change.
   */
  pauseRendering(): void {
    this.renderPause.pauseByHost();
  }

  /**
   * Resume presenting frames after {@link pauseRendering}.
   *
   * The next frame is a full redraw, since nothing marks the scene dirty while
   * presentation is suspended. A no-op when the host has not paused; if the
   * page is hidden, frames stay suspended until it is visible again.
   */
  resumeRendering(): void {
    this.renderPause.resumeByHost();
  }

  /**
   * Whether presentation is currently suspended.
   *
   * True for either cause — an explicit {@link pauseRendering} or a hidden
   * page — because what a host polling this wants to know is whether frames
   * are reaching the screen, not who stopped them.
   */
  isRenderingPaused(): boolean {
    return this.renderPause.isPaused;
  }

  // ==========================================================================
  // Public API - Layers
  // ==========================================================================

  /**
   * Enable the heatmap layer.
   * Creates the layer if it doesn't exist.
   */
  // ---- Layer-ID-aware heatmap methods ----

  /**
   * Enable a heatmap layer by ID. Creates if it doesn't exist.
   */
  enableHeatmapLayer(layerId: string, config?: HeatmapConfig): void {
    if (!this.layerManager.hasLayer(layerId)) {
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
      this.refreshLayerRenderContexts();
    } else {
      const layer = this.layerManager.getLayer<HeatmapLayer>(layerId);
      if (layer) {
        layer.enabled = true;
        if (config) {
          layer.setConfig(config);
        }
      }
    }
    this.markRenderDirty();
  }

  /**
   * Disable a heatmap layer by ID.
   */
  disableHeatmapLayer(layerId: string): void {
    this.layerManager.disableLayer(layerId);
    this.markRenderDirty();
  }

  /**
   * Configure a heatmap layer by ID.
   */
  configureHeatmapLayer(layerId: string, config: Partial<HeatmapConfig>): void {
    const layer = this.layerManager.getLayer<HeatmapLayer>(layerId);
    if (layer) {
      layer.setConfig(config);
      this.markRenderDirty();
    }
  }

  /**
   * Set color scale on a heatmap layer by ID.
   */
  setHeatmapLayerColorScale(layerId: string, name: ColorScaleName): void {
    const layer = this.layerManager.getLayer<HeatmapLayer>(layerId);
    if (layer) {
      layer.setColorScale(name);
      this.markRenderDirty();
    }
  }

  /**
   * Set custom color scale on a heatmap layer by ID.
   */
  setCustomHeatmapLayerColorScale(
    layerId: string,
    stops: Array<{ position: number; color: [number, number, number, number] }>,
  ): void {
    const layer = this.layerManager.getLayer<HeatmapLayer>(layerId);
    if (layer) {
      layer.setCustomColorScale(stops);
      this.markRenderDirty();
    }
  }

  /**
   * Set data source on a heatmap layer by ID.
   */
  setHeatmapLayerDataSource(layerId: string, source: string): void {
    const layer = this.layerManager.getLayer<HeatmapLayer>(layerId);
    if (layer) {
      layer.setDataSource(source);
      this.refreshLayerRenderContexts();
    }
  }

  // ---- Legacy single-layer heatmap aliases (delegate to "heatmap" layer) ----

  enableHeatmap(config?: HeatmapConfig): void {
    this.enableHeatmapLayer("heatmap", config);
  }

  disableHeatmap(): void {
    this.disableHeatmapLayer("heatmap");
  }

  isHeatmapEnabled(): boolean {
    return this.layerManager.isLayerVisible("heatmap");
  }

  setHeatmapConfig(config: Partial<HeatmapConfig>): void {
    this.configureHeatmapLayer("heatmap", config);
  }

  getHeatmapConfig(): HeatmapConfig | null {
    const layer = this.layerManager.getLayer<HeatmapLayer>("heatmap");
    return layer?.getConfig() ?? null;
  }

  setHeatmapColorScale(name: ColorScaleName): void {
    this.setHeatmapLayerColorScale("heatmap", name);
  }

  setCustomHeatmapColorScale(
    stops: Array<{ position: number; color: [number, number, number, number] }>,
  ): void {
    this.setCustomHeatmapLayerColorScale("heatmap", stops);
  }

  setHeatmapDataSource(source: string): void {
    this.setHeatmapLayerDataSource("heatmap", source);
  }

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
   * Register a custom visualization layer.
   *
   * This is the plugin seam for layers the library does not ship: implement
   * the `Layer` interface (exported as `VisualizationLayer`) and register it
   * here. Layers render in `order` sequence after the base graph.
   *
   * @param layer - Layer instance to register
   * @throws Error if the layer ID is already registered or the layer limit is reached
   */
  addLayer(layer: VisualizationLayer): void {
    this.layerManager.addLayer(layer);
    this.markRenderDirty();
  }

  /**
   * Remove a layer by ID, destroying its GPU resources.
   *
   * @param id - Layer ID
   * @returns True if a layer was removed, false if no layer had that ID
   */
  removeLayer(id: string): boolean {
    const removed = this.layerManager.removeLayer(id);
    if (removed) {
      // The layer's stream intensity buffer is owned here, not by the layer,
      // so removing the layer has to release it.
      this.streamIntensityCaches.get(id)?.destroy();
      this.streamIntensityCaches.delete(id);
      this.markRenderDirty();
    }
    return removed;
  }

  /**
   * Toggle a layer's visibility.
   */
  toggleLayer(layerId: string): boolean {
    const result = this.layerManager.toggleLayer(layerId);
    this.markRenderDirty();
    return result;
  }

  /**
   * Set a layer's render order. Higher values render on top.
   */
  setLayerOrder(layerId: string, order: number): void {
    this.layerManager.setLayerOrder(layerId, order);
    this.markRenderDirty();
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
   * Push fresh render contexts to every layer.
   *
   * Runs on every ping-pong flip (layers must bind the simulation's current
   * read buffer, not the buffer being written) and again just before the
   * render passes. Everything in here must therefore be cheap: the one
   * expensive part — deriving per-node intensities from a value stream — is
   * dirty-gated by StreamIntensityCache and only recomputes when its inputs
   * actually changed.
   */
  private refreshLayerRenderContexts(): void {
    if (!this.simBuffers || !this.state.loaded) return;

    // Update all heatmap layers (skip disabled to avoid wasted GPU allocation)
    const heatmapLayers = this.layerManager.getLayersByType<HeatmapLayer>("heatmap");
    for (const heatmapLayer of heatmapLayers) {
      if (!heatmapLayer.enabled) continue;

      heatmapLayer.setRenderContext({
        viewportUniformBuffer: this.viewportUniformBuffer.buffer,
        positions: this.simBuffers.positions,
        nodeCount: this.state.nodeCount,
        nodeIntensities: this.syncStreamIntensities(
          heatmapLayer.id,
          heatmapLayer.getDataSource(),
        ),
      });
    }

    // Update contour layer — uses density texture from first enabled heatmap
    const contourLayer = this.layerManager.getLayer<ContourLayer>("contour");
    const primaryHeatmap = heatmapLayers.find((l) => l.enabled);
    if (contourLayer && primaryHeatmap) {
      const densityTexture = primaryHeatmap.getDensityTexture();
      const heatmapConfig = primaryHeatmap.getConfig();
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

      // In density mode (or when the configured stream does not exist) every
      // node contributes equally, so the shared all-ones buffer is bound.
      const metaballIntensities =
        this.syncStreamIntensities(metaballLayer.id, metaballLayer.getDataSource()) ??
          this.getOrCreateDefaultIntensityBuffer();

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

      const labelsContext: LabelsRenderContext = {
        viewportX: viewportState.x,
        viewportY: viewportState.y,
        scale: viewportState.scale,
        canvasWidth: cssWidth,
        canvasHeight: cssHeight,
        nodeSource: this.labelNodeSource(),
        // Two counters summed: both only ever increase, so the sum changes
        // whenever either does, and the cache compares it for inequality.
        nodeStateVersion: this.labelNodeStateVersion + (this.domOverlay?.cardEpoch ?? 0),
      };
      labelsLayer.setRenderContext(labelsContext);
    }
  }

  /**
   * The live per-node readings the GPU label set is derived from.
   *
   * Radius is read from the same attribute row the sprite is drawn at, so a
   * collapsed LOD proxy — inflated to its well radius by
   * {@link GraphMother.setCollapsedProxies} — outranks the leaves it stands
   * for without the label path knowing that bubbles exist.
   */
  private labelNodeSource(): LabelNodeSource {
    const positionsX = this.state.parsedGraph!.positionsX;
    const positionsY = this.state.parsedGraph!.positionsY;
    return {
      getX: (nodeId) => positionsX[nodeId] ?? 0,
      getY: (nodeId) => positionsY[nodeId] ?? 0,
      getRadius: (nodeId) => this.getNodeRenderRadius(nodeId),
      isSuppressed: (nodeId) => this.isLabelSuppressed(nodeId),
    };
  }

  /**
   * Whether a node's GPU label is suppressed.
   *
   * Two reasons, both of which would otherwise draw text the user cannot use:
   * a node the LOD cut hides is not on screen at all, and a node holding a DOM
   * card already shows richer text on top of the sprite.
   */
  private isLabelSuppressed(nodeId: NodeId): boolean {
    if (this.lodController !== null && !this.lodController.isVisible(nodeId)) return true;
    return this.domOverlay?.hasCard(nodeId) === true;
  }

  /** Graph-space render radius of a slot, from the attribute shadow. */
  private getNodeRenderRadius(nodeId: NodeId): number {
    const gs = this.graphState;
    if (!gs || nodeId < 0 || nodeId >= gs.nodeHighWater) return 0;
    return gs.nodeAttributes[nodeId * NODE_ATTR_FLOATS];
  }

  /**
   * Returns the GPU buffer of per-node intensities for a layer bound to a
   * value stream, or null when the layer is in density mode or names a stream
   * that does not exist.
   *
   * Delegates the dirty tracking to the layer's StreamIntensityCache: the
   * O(nodeCount) fill and the upload only happen when the bound stream, its
   * mutation version, the node count, or the colour-scale domain changed.
   *
   * @param layerId - Layer the cache belongs to
   * @param dataSource - The layer's configured data source ("density" or a stream ID)
   */
  private syncStreamIntensities(layerId: string, dataSource: string): GPUBuffer | null {
    const stream = dataSource === "density" ? undefined : this.streamManager.getStream(dataSource);
    if (!stream) return null;

    let cache = this.streamIntensityCaches.get(layerId);
    if (!cache) {
      cache = new StreamIntensityCache(`Stream Intensity [${layerId}]`);
      this.streamIntensityCaches.set(layerId, cache);
    }

    return cache.sync(
      this.gpuContext.device,
      dataSource,
      stream,
      this.streamManager.version,
      this.state.nodeCount,
    );
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
      this.refreshLayerRenderContexts();
    } else {
      const layer = this.layerManager.getLayer<ContourLayer>(layerId);
      if (layer) {
        layer.enabled = true;
        if (config) {
          layer.setConfig(config);
        }
      }
    }
    this.markRenderDirty();
  }

  /**
   * Disable the contour layer.
   */
  disableContour(): void {
    this.layerManager.disableLayer("contour");
    this.markRenderDirty();
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
    this.refreshLayerRenderContexts();
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
      this.refreshLayerRenderContexts();
    } else {
      const layer = this.layerManager.getLayer<MetaballLayer>(layerId);
      if (layer) {
        layer.enabled = true;
        if (config) {
          layer.setConfig(config);
        }
      }
    }
    this.markRenderDirty();
  }

  /**
   * Disable the metaball layer.
   */
  disableMetaball(): void {
    this.layerManager.disableLayer("metaball");
    this.markRenderDirty();
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
      this.refreshLayerRenderContexts();
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
      this.refreshLayerRenderContexts();
    } else {
      const layer = this.layerManager.getLayer<LabelsLayer>(layerId);
      if (layer) {
        layer.enabled = true;
        if (config) {
          layer.setConfig(config);
        }
      }
    }
    this.markRenderDirty();
  }

  /**
   * Disable the labels layer.
   */
  disableLabels(): void {
    this.layerManager.disableLayer("labels");
    this.markRenderDirty();
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
  // Public API - DOM Card Overlay
  // ==========================================================================

  /**
   * Configure the DOM card overlay.
   *
   * The overlay promotes nodes to real DOM elements — selectable text, working
   * links, find-in-page — and keeps them on the pixel the GPU would have drawn
   * the sprite on, by carrying the same camera as a CSS transform.
   *
   * Cards are created inside a container appended to `config.host`, defaulting
   * to the canvas's parent element, so the factory signature stays a canvas
   * and core still creates no DOM until asked. The host must be a positioned
   * element that the canvas fills, which is what an embedding adapter already
   * owns.
   *
   * @param config - Settings to change; anything omitted keeps its value
   * @throws if enabling with no host and no `canvas.parentElement`
   */
  setDomOverlay(config: Partial<DomOverlayConfig>): void {
    this.ensureDomOverlay().setConfig(config);
  }

  /**
   * Register the renderer for card content, or `null` for the built-in one.
   *
   * Core owns where a card is and how big it is; everything inside it belongs
   * to the provider. Swapping providers releases every mounted card first, so
   * the outgoing provider tears down its own state.
   */
  setCardProvider(provider: CardProvider | null): void {
    this.ensureDomOverlay().setProvider(provider);
  }

  /**
   * Declare which nodes should currently be carded.
   *
   * Imperative for now: the semantic-LOD controller becomes the caller once it
   * lands, and this is the interface it will drive. The overlay applies its own
   * budget and anti-flicker floor to the set, so a caller may ask for more
   * cards than `maxCards` and get the highest-priority ones.
   *
   * No-op until {@link GraphMother.setDomOverlay} has enabled the overlay.
   */
  syncDomCards(entries: readonly CardSyncEntry[]): void {
    this.domOverlay?.syncCards(entries);
  }

  /**
   * Create the overlay on first use; it holds no GPU resources.
   *
   * Seeded with the LOD budget so the two agree from the first sync, whichever
   * order the caller enabled them in.
   */
  private ensureDomOverlay(): DomCardOverlay {
    if (this.domOverlay === null) {
      const lod = this.lodController?.getConfig() ?? DEFAULT_LOD_CONFIG;
      this.domOverlay = new DomCardOverlay({
        canvas: this.canvas,
        viewport: () => this.viewport.state,
        nodes: this.createCardNodeSource(),
        // A card dragged by its handle is a node drag: the same controller the
        // canvas gesture drives, so the pin, the position writes and the
        // `node:drag*` events are not merely similar but identical.
        drag: this.nodeDrag,
        maxCards: lod.maxCards,
        minCardLifetimeMs: lod.minCardLifetimeMs,
      });
    }
    return this.domOverlay;
  }

  /**
   * The per-node readings cards are a live view of.
   *
   * Closed over `this` rather than over the current graph, so cards follow a
   * reload and a mutation instead of snapshotting whatever was loaded when the
   * overlay was enabled. `tag` and `weight` are the producer's semantic columns
   * and `depth` comes from the retained containment hierarchy; each reports the
   * neutral value when the graph carries no such column, which is what a card
   * then displays.
   */
  private createCardNodeSource(): CardNodeSource {
    // Read from graph state, never from the load-time parse: a compaction
    // rebuilds this map against the new slots, and the card would otherwise
    // render — and fetch by `contentRef` — a neighbour's document.
    const metadataText = (nodeId: NodeId, key: string): string | undefined => {
      const value = this.graphState?.nodeMetadata.get(nodeId)?.[key];
      return typeof value === "string" ? value : undefined;
    };

    return {
      externalId: (nodeId) => this.getExternalId(nodeId),
      label: (nodeId) => metadataText(nodeId, "label"),
      tag: (nodeId) => this.getNodeTag(nodeId),
      weight: (nodeId) => this.getNodeWeight(nodeId),
      depth: (nodeId) => this.getNodeDepth(nodeId),
      contentRef: (nodeId) => metadataText(nodeId, "contentRef"),
      // The CPU position shadow, bounded by the slot space rather than by
      // nodeCount: after a removal a live node can sit in a slot above the
      // count, and a card on it must still track its node.
      position: (nodeId) => this.getNodePositionOrOrigin(nodeId),
      isPinned: (nodeId) => this.isNodePinned(nodeId),
      setPinned: (nodeId, pinned) => {
        if (pinned) {
          this.pinNode(nodeId);
        } else {
          this.unpinNode(nodeId);
        }
      },
    };
  }

  /** Graph-space position of a slot, or the origin when the slot has none. */
  private getNodePositionOrOrigin(nodeId: NodeId): Vec2 {
    const parsed = this.state.parsedGraph;
    if (parsed === null || nodeId < 0 || nodeId >= parsed.positionsX.length) {
      return CARD_ORIGIN;
    }
    return { x: parsed.positionsX[nodeId], y: parsed.positionsY[nodeId] };
  }

  /**
   * Producer-supplied semantic tag for a slot, or 0.
   *
   * Opaque to core: it exists so an LOD policy can compare an integer instead
   * of walking a metadata map on the evaluation path.
   */
  getNodeTag(nodeId: NodeId): number {
    const tags = this.state.parsedGraph?.nodeTags;
    return tags && nodeId >= 0 && nodeId < tags.length ? tags[nodeId] : 0;
  }

  /** Producer-supplied importance for a slot in 0..1, or 0. */
  getNodeWeight(nodeId: NodeId): number {
    const weights = this.state.parsedGraph?.nodeWeights;
    return weights && nodeId >= 0 && nodeId < weights.length ? weights[nodeId] : 0;
  }

  /** Containment depth of a slot; roots and graphs with no hierarchy report 0. */
  getNodeDepth(nodeId: NodeId): number {
    const hierarchy = this.getHierarchy();
    if (!hierarchy || nodeId < 0 || nodeId >= hierarchy.nodeCount) return 0;
    return hierarchy.columns.depth[nodeId];
  }

  // ==========================================================================
  // Public API - Semantic LOD
  // ==========================================================================

  /**
   * Configure semantic level of detail.
   *
   * LOD folds subtrees that are too small to read into a single visible proxy,
   * and unfolds them as the camera comes in. It is independent of the DOM card
   * overlay — collapsing works with no cards enabled, and cards work with LOD
   * off — and is disabled until this is called with `enabled: true`.
   *
   * A transition never reheats the simulation. Hidden nodes keep their
   * positions and are restored exactly where they were, so zooming in and out
   * is not a layout operation.
   *
   * @param config - Settings to change; anything omitted keeps its value
   */
  setLodConfig(config: Partial<LodConfig>): void {
    const controller = this.ensureLodController();
    controller.setConfig(config, performance.now());
    // The card budget and the anti-flicker floor exist on both sides — the
    // controller ranks and truncates, the overlay admits and holds — and two
    // separately settable copies of one knob disagree silently. The controller's
    // resolved value wins, so the clamps apply once.
    const resolved = controller.getConfig();
    this.domOverlay?.setConfig({
      maxCards: resolved.maxCards,
      minCardLifetimeMs: resolved.minCardLifetimeMs,
    });
  }

  /** Current LOD configuration, including defaults never explicitly set. */
  getLodConfig(): LodConfig {
    return this.ensureLodController().getConfig();
  }

  /**
   * Register a semantic policy, or `null` for the built-in geometric rule.
   *
   * The policy is consulted only for nodes that crossed a threshold since the
   * last evaluation, and must be pure, synchronous and allocation-free.
   */
  setLodPolicy(policy: LodPolicy | null): void {
    this.ensureLodController().setPolicy(policy);
  }

  /**
   * Declare the focus set: nodes carded regardless of their screen size.
   *
   * Focus does not expand ancestors — a focus node inside a collapsed subtree
   * stays folded, and {@link GraphMother.getVisibleAncestor} names the proxy
   * standing in for it.
   */
  setLodFocus(nodes: Iterable<NodeId>): void {
    this.ensureLodController().setFocus(nodes);
  }

  /** Expand a node now, whatever its screen size says. */
  expandNode(node: NodeId): void {
    this.ensureLodController().expandNode(node, performance.now());
  }

  /** Collapse a node now, whatever its screen size says. */
  collapseNode(node: NodeId): void {
    this.ensureLodController().collapseNode(node, performance.now());
  }

  /**
   * The nodes the current cut leaves on screen, ascending by slot.
   *
   * Every live slot when LOD is off or has not evaluated yet, so a caller can
   * treat this as "what is drawn" without asking whether LOD is on.
   */
  getVisibleNodes(): Uint32Array {
    const controller = this.lodController;
    if (controller !== null && controller.hasCut) return controller.getVisibleNodes();

    const gs = this.graphState;
    if (!gs) return new Uint32Array(0);
    const live: number[] = [];
    for (let slot = 0; slot < gs.nodeHighWater; slot++) {
      if ((gs.nodeFlagsShadow[slot] & NODE_FLAG_DEAD) === 0) live.push(slot);
    }
    return Uint32Array.from(live);
  }

  /**
   * Whether `node` is on screen standing in for a subtree that is folded away.
   *
   * This is the question a click handler has to ask. A proxy looks like an
   * ordinary node — it is the parent itself, drawn at its subtree's radius — so
   * a host that acts on the node's own type alone will treat a folded directory
   * as a directory the user asked to open, rather than as a cluster the user
   * asked to expand.
   *
   * False whenever LOD is off or has not evaluated yet, so the check is safe to
   * make unconditionally.
   */
  isCollapsed(node: NodeId): boolean {
    return this.lodController?.isCollapsed(node) ?? false;
  }

  /**
   * The lowest ancestor of `node` that is in the cut — `node` itself when it is
   * visible, or -1 when nothing on its root path is.
   */
  getVisibleAncestor(node: NodeId): NodeId {
    const controller = this.lodController;
    return controller === null ? node : controller.getVisibleAncestor(node);
  }

  /** Create the controller on first use; it holds no GPU resources. */
  private ensureLodController(): LODController {
    if (this.lodController === null) {
      this.lodController = new LODController(this.createLodHost());
    }
    return this.lodController;
  }

  /**
   * The graph seam the LOD controller drives.
   *
   * Deliberately narrow, and deliberately without any route to simulation
   * alpha: an LOD transition must never reheat the simulation, and a seam that
   * cannot express a reheat is a stronger guarantee than a comment saying not
   * to.
   */
  private createLodHost(): LodHost {
    return {
      getHierarchy: () => this.getHierarchy(),
      getViewport: () => this.viewport.state,
      getNodePosition: (node) => this.getNodePositionOrOrigin(node),
      getNodeRadius: (node) => this.getNodeRenderRadius(node),
      getNodeTag: (node) => this.getNodeTag(node),
      getNodeWeight: (node) => this.getNodeWeight(node),
      applyVisibility: (lo, hi, visible) => this.applyLodVisibility(lo, hi, visible),
      uploadNodeAlpha: (scheduler) => this.uploadNodeAlpha(scheduler),
      uploadNodeMass: (mass) => this.uploadNodeMass(mass),
      setCollapsedProxies: (proxies, radii) => this.setCollapsedProxies(proxies, radii),
      aggregateEdges: (visible) => this.aggregateLodEdges(visible),
      releaseEdgeAggregation: () => this.releaseLodEdgeAggregation(),
      translateNodeRange: (lo, hi, dx, dy) => this.translateNodeRange(lo, hi, dx, dy),
      syncCards: (entries) => this.syncDomCards(entries),
      emit: (event) => {
        // A changed cut is the LOD epoch boundary, and the only one core has:
        // the set of nodes on screen is different, so content warmed for the
        // previous cut and never carded is spent and has to become offerable
        // again. The controller emits this before it re-declares the card set,
        // which is the order that matters — the epoch turns first, then the
        // new ring's prefetches land inside it.
        if (event.type === "lod:change") this.domOverlay?.beginEpoch();
        this.events.emit(event);
      },
    };
  }

  /**
   * Apply the cut's visibility across a slot range as one flag upload.
   *
   * Dead slots are skipped: they are already invisible, and clearing
   * `HIDDEN_LOD` on one would say the opposite about a slot the free list owns.
   */
  private applyLodVisibility(lo: number, hi: number, visible: Uint8Array): void {
    const gs = this.graphState;
    if (!gs) return;

    const end = Math.min(hi, gs.nodeHighWater);
    let changedLo = end;
    let changedHi = lo;
    for (let slot = Math.max(0, lo); slot < end; slot++) {
      if ((gs.nodeFlagsShadow[slot] & NODE_FLAG_DEAD) !== 0) continue;
      if (!gs.setNodeFlagBits(slot, NODE_FLAG_HIDDEN_LOD, visible[slot] === 0)) continue;
      if (slot < changedLo) changedLo = slot;
      changedHi = slot + 1;
    }
    this.flushNodeFlagRange(changedLo, changedHi);
    // A node leaving or entering the cut also leaves or enters the label set.
    if (changedHi > changedLo) this.labelNodeStateVersion++;
  }

  /** Upload the crossfade scheduler's dirty alpha range, if buffers exist. */
  private uploadNodeAlpha(scheduler: CrossfadeScheduler): void {
    if (this.simBuffers) {
      scheduler.flush(this.gpuContext.device, this.simBuffers.nodeAlpha);
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
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        `Unknown flow preset: ${preset}. Available: ${Object.keys(EDGE_FLOW_PRESETS).join(", ")}`,
      );
    }
    this.flowConfig = config;
    this.markRenderDirty();
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
    this.markRenderDirty();
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
    this.markRenderDirty();
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
  // Public API - Birth Pulse Animation
  // ==========================================================================

  /**
   * Get the current animation time (seconds since graph start).
   * Use this to stamp `birthTime` on nodes for birth pulse animation.
   */
  getAnimationTime(): number {
    return (performance.now() - this.flowStartTime) / 1000.0;
  }

  /**
   * Configure the birth pulse animation parameters.
   * @param config - Partial config: `enabled`, `duration` (seconds), `intensity` (0-1),
   *   `pulseColor` (hex string for looping pulse color, e.g. "#a78bfa")
   */
  setBirthPulseConfig(config: {
    enabled?: boolean;
    duration?: number;
    intensity?: number;
    pulseColor?: string;
  }): void {
    if (config.enabled !== undefined) this.birthPulseConfig.enabled = config.enabled;
    if (config.duration !== undefined) this.birthPulseConfig.duration = config.duration;
    if (config.intensity !== undefined) this.birthPulseConfig.intensity = config.intensity;
    if (config.pulseColor !== undefined) {
      this.birthPulseConfig.pulseColor = parseColorToRGB(config.pulseColor);
    }
    this.updateRenderConfigBuffer();
    this.markRenderDirty();
  }

  /**
   * Set the birth time for a specific node (triggers birth pulse animation).
   * @param nodeId - The node slot index
   * @param time - Animation time from `getAnimationTime()`. 0 = no animation.
   *   Negative values enable continuous looping (abs value = cycle start time).
   */
  setNodeBirthTime(nodeId: NodeId, time: number): void {
    if (!this.buffers || !this.state.parsedGraph) return;
    const idx = nodeId;
    if (idx < 0 || idx >= (this.graphState?.nodeHighWater ?? 0)) return;

    this.state.parsedGraph.nodeAttributes[idx * NODE_ATTR_FLOATS + 6] = time;
    if (this.graphState) {
      this.graphState.nodeAttributes[idx * NODE_ATTR_FLOATS + 6] = time;
    }

    const byteOffset = idx * NODE_ATTR_BYTES + 6 * 4;
    this.gpuContext.device.queue.writeBuffer(
      this.buffers.nodeAttributes,
      byteOffset,
      new Float32Array([time]),
    );
    // Only update dirty tracking when starting/continuing a pulse, not when clearing one.
    // Clearing a single node (time=0) must not kill dirty tracking for other active pulses.
    if (time !== 0) {
      this.lastBirthTime = time;
    }
    this.markRenderDirty();
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
   * @throws GraphMotherError if array length doesn't match nodeCount × 4
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
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Cannot set node colors: graph not loaded",
      );
    }

    const expected = this.state.nodeCount * 4;
    if (colors.length !== expected) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        `Expected ${expected} values for ${this.state.nodeCount} nodes (4 per node), got ${colors.length}`,
      );
    }

    const { device } = this.gpuContext;
    const nodeAttrs = this.state.parsedGraph.nodeAttributes;

    // Update CPU-side array and GPU buffer
    // Node attrs layout: [radius, r, g, b, selected, hovered, birth_time, tex_index] per node
    for (let i = 0; i < this.state.nodeCount; i++) {
      const colorBase = i * 4;
      const attrBase = i * NODE_ATTR_FLOATS;

      // Skip NaN values (keep existing color)
      const r = colors[colorBase];
      const g = colors[colorBase + 1];
      const b = colors[colorBase + 2];
      // Alpha (colors[colorBase + 3]) is currently ignored - shader uses RGB only

      if (!Number.isNaN(r)) nodeAttrs[attrBase + 1] = r;
      if (!Number.isNaN(g)) nodeAttrs[attrBase + 2] = g;
      if (!Number.isNaN(b)) nodeAttrs[attrBase + 3] = b;

      // If this node has a stream color backup, update it so the new
      // base color is restored when the stream clears.
      const backup = this.streamColorBackups.get(i);
      if (backup) {
        if (typeof r === "number" && !Number.isNaN(r)) backup[0] = r;
        if (typeof g === "number" && !Number.isNaN(g)) backup[1] = g;
        if (typeof b === "number" && !Number.isNaN(b)) backup[2] = b;
      }
    }

    // Upload entire buffer to GPU
    device.queue.writeBuffer(
      this.buffers.nodeAttributes,
      0,
      toArrayBuffer(nodeAttrs),
    );
    this.markRenderDirty();
  }

  /**
   * Set sizes (radii) for individual nodes.
   *
   * @param sizes Float32Array with 1 value per node.
   *              Length must equal nodeCount.
   *              Values are in graph units.
   * @throws GraphMotherError if array length doesn't match nodeCount
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
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Cannot set node sizes: graph not loaded",
      );
    }

    const expected = this.state.nodeCount;
    if (sizes.length !== expected) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        `Expected ${expected} values for ${this.state.nodeCount} nodes (1 per node), got ${sizes.length}`,
      );
    }

    const { device } = this.gpuContext;
    const nodeAttrs = this.state.parsedGraph.nodeAttributes;

    // Update CPU-side array
    // Node attrs layout: [radius, r, g, b, selected, hovered, birth_time, tex_index] per node
    for (let i = 0; i < this.state.nodeCount; i++) {
      const size = sizes[i];
      if (!Number.isNaN(size) && size > 0) {
        nodeAttrs[i * NODE_ATTR_FLOATS] = size; // radius is at offset 0
      }
    }

    // Upload entire buffer to GPU
    device.queue.writeBuffer(
      this.buffers.nodeAttributes,
      0,
      toArrayBuffer(nodeAttrs),
    );
    this.markRenderDirty();

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
   * @throws GraphMotherError if array length doesn't match edgeCount × 4
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
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Cannot set edge colors: graph not loaded",
      );
    }

    const expected = this.state.edgeCount * 4;
    if (colors.length !== expected) {
      throw new GraphMotherError(
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
   * @throws GraphMotherError if array length doesn't match edgeCount
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
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Cannot set edge widths: graph not loaded",
      );
    }

    const expected = this.state.edgeCount;
    if (widths.length !== expected) {
      throw new GraphMotherError(
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
   * @throws GraphMotherError if array length doesn't match edgeCount
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
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        "Cannot set edge curvatures: graph not loaded",
      );
    }

    const expected = this.state.edgeCount;
    if (curvatures.length !== expected) {
      throw new GraphMotherError(
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
    this.markRenderDirty();
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
    this.markRenderDirty();
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
  setStreamBlendMode(
    streamId: string,
    blendMode: "additive" | "multiply" | "max" | "replace",
  ): void {
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

  // ============================================================================
  // Node Visibility API
  // ============================================================================

  /**
   * Show or hide nodes by ID.
   *
   * Visibility is a bit in the shared nodeFlags buffer (NODE_FLAG_HIDDEN_LOD),
   * and hiding is render *and* physics: the node pipeline drops the instance,
   * and the simulation treats the slot as inert — it exerts no force, receives
   * none, is excluded from the collision set, and is frozen where it stands.
   * Its radius, color and position are untouched, so it reappears exactly where
   * it was; but the nodes still simulating no longer feel it, so hiding a large
   * share of a graph lets the rest contract, and showing them again re-inflates
   * it. Hide for a filter, not to keep a layout still. Freed slots are left
   * alone; they are already invisible.
   *
   * Edges are unaffected: the edge pipeline reads positions only, so an edge
   * with a hidden endpoint still draws.
   *
   * @param ids - Node IDs to show or hide
   * @param visible - `true` to show, `false` to hide
   *
   * @example
   * ```typescript
   * // Hide all "json" nodes
   * graph.setNodeVisibility(jsonNodeIds, false);
   * // Show them again
   * graph.setNodeVisibility(jsonNodeIds, true);
   * ```
   */
  setNodeVisibility(ids: (string | number)[], visible: boolean): void {
    const gs = this.graphState;
    if (!gs) return;

    // One contiguous range write covering every slot that actually changed,
    // rather than one queue write per node.
    let lo = gs.nodeHighWater;
    let hi = 0;

    for (const id of ids) {
      const slot = typeof id === "number" && id < gs.nodeHighWater ? id : gs.nodeIdMap.get(id);
      if (slot === undefined || slot >= gs.nodeHighWater) continue;
      if ((gs.nodeFlagsShadow[slot] & NODE_FLAG_DEAD) !== 0) continue;
      if (!gs.setNodeFlagBits(slot, NODE_FLAG_HIDDEN_LOD, !visible)) continue;

      if (slot < lo) lo = slot;
      if (slot >= hi) hi = slot + 1;
    }

    this.flushNodeFlagRange(lo, hi);
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
      // Use parsed.nodeCount (= nodeHighWater) to cover all slots including gaps from removals
      const nodeCount = parsed.nodeCount;
      // Node attribute layout: [radius, r, g, b, selected, hovered, birth_time, tex_index] — 8 floats per node
      const nodeAttributes = new Float32Array(nodeCount * NODE_ATTR_FLOATS);

      for (let i = 0; i < nodeCount; i++) {
        const baseOffset = i * NODE_ATTR_FLOATS;
        const originalRadius = parsed.nodeAttributes[baseOffset];

        // Skip freed slots (radius 0). Hidden slots need no special case:
        // visibility lives in nodeFlags, not in the attributes rewritten here.
        if (originalRadius === 0) {
          continue;
        }

        const nodeType = parsed.nodeTypes?.[i];

        // Only restyle nodes whose type has a registered style. Untyped nodes
        // and nodes with unstyled types keep their explicitly-set or
        // parser-provided attributes from the CPU shadow — resolving them to
        // the default gray would permanently destroy per-node colors.
        if (!nodeType || !this.typeStyleManager.getNodeTypeStyle(nodeType)) {
          for (let k = 0; k < NODE_ATTR_FLOATS; k++) {
            nodeAttributes[baseOffset + k] = parsed.nodeAttributes[baseOffset + k] ?? 0;
          }
          continue;
        }

        const style = this.typeStyleManager.resolveNodeStyle(nodeType);

        nodeAttributes[baseOffset + 0] = originalRadius * style.size;
        // Color (RGBA — alpha baked into RGB for the shader)
        nodeAttributes[baseOffset + 1] = style.color[0];
        nodeAttributes[baseOffset + 2] = style.color[1];
        nodeAttributes[baseOffset + 3] = style.color[2];

        // Sync type-styled colors to CPU shadow so stream color backups
        // capture the correct base color (not stale parse-time values).
        parsed.nodeAttributes[baseOffset + 1] = style.color[0];
        parsed.nodeAttributes[baseOffset + 2] = style.color[1];
        parsed.nodeAttributes[baseOffset + 3] = style.color[2];

        // If this node has a stream color backup, update it to reflect
        // the new type-styled base color.
        if (this.streamColorBackups.has(i)) {
          this.streamColorBackups.set(i, [style.color[0], style.color[1], style.color[2]]);
        }

        // Preserve selected/hovered state
        nodeAttributes[baseOffset + 4] = parsed.nodeAttributes[baseOffset + 4];
        nodeAttributes[baseOffset + 5] = parsed.nodeAttributes[baseOffset + 5];
        // Preserve birth_time and tex_index
        nodeAttributes[baseOffset + 6] = parsed.nodeAttributes[baseOffset + 6];
        nodeAttributes[baseOffset + 7] = parsed.nodeAttributes[baseOffset + 7];
      }

      device.queue.writeBuffer(
        this.buffers.nodeAttributes,
        0,
        nodeAttributes.buffer,
        0,
        nodeCount * NODE_ATTR_BYTES,
      );
    }

    // Update edge attributes buffer with type-based colors and widths
    if (this.buffers && this.typeStyleManager.hasEdgeStyles()) {
      const edgeCount = this.state.edgeCount;
      const edgeAttributes = new Float32Array(edgeCount * 8); // 8 floats per edge

      for (let i = 0; i < edgeCount; i++) {
        const edgeType = parsed.edgeTypes?.[i];
        const baseOffset = i * 8;

        // Only restyle edges whose type has a registered style. Untyped edges
        // and edges with unstyled types keep their per-edge attributes from
        // the CPU shadow (e.g. followed_by::session_* sub-types discovered
        // after the last setEdgeTypeStyles, or parser-provided colors).
        if (!edgeType || !this.typeStyleManager.getEdgeTypeStyle(edgeType)) {
          for (let k = 0; k < 8; k++) {
            edgeAttributes[baseOffset + k] = parsed.edgeAttributes[baseOffset + k] ?? 0;
          }
          continue;
        }

        const style = this.typeStyleManager.resolveEdgeStyle(edgeType);

        // Layout: [width, r, g, b, selected, hovered, curvature, opacity]
        edgeAttributes[baseOffset + 0] = style.width;
        edgeAttributes[baseOffset + 1] = style.color[0]; // r
        edgeAttributes[baseOffset + 2] = style.color[1]; // g
        edgeAttributes[baseOffset + 3] = style.color[2]; // b
        // Preserve interaction state and per-edge curvature (setEdgeCurvatures
        // writes the shadow) — type styles only own width/color/opacity
        edgeAttributes[baseOffset + 4] = parsed.edgeAttributes[baseOffset + 4] ?? 0; // selected
        edgeAttributes[baseOffset + 5] = parsed.edgeAttributes[baseOffset + 5] ?? 0; // hovered
        edgeAttributes[baseOffset + 6] = parsed.edgeAttributes[baseOffset + 6] ?? 0; // curvature
        edgeAttributes[baseOffset + 7] = style.color[3]; // opacity from resolved alpha (0.0 = hidden)

        // Sync type-styled values to the CPU shadow so later shadow-based
        // uploads (mutation flushes, the preserve path above) keep them
        parsed.edgeAttributes[baseOffset + 0] = style.width;
        parsed.edgeAttributes[baseOffset + 1] = style.color[0];
        parsed.edgeAttributes[baseOffset + 2] = style.color[1];
        parsed.edgeAttributes[baseOffset + 3] = style.color[2];
        parsed.edgeAttributes[baseOffset + 7] = style.color[3];
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
   *
   * Only updates nodes that have actual stream color data (alpha > 0).
   * Nodes without stream values are left untouched — their existing
   * colors are preserved. This prevents streams with sparse coverage
   * from blanking unaffected nodes to black.
   */
  private applyStreamColors(): void {
    if (!this.state.loaded || this.state.nodeCount === 0) return;
    if (!this.state.parsedGraph || !this.buffers) return;

    const colors = this.streamManager.computeBlendedColors(this.state.nodeCount);
    const nodeAttrs = this.state.parsedGraph.nodeAttributes;
    let changed = false;
    let minChanged = this.state.nodeCount;
    let maxChanged = -1;
    const currentlyColored = new Set<number>();

    // Apply stream colors to nodes with data (alpha > 0).
    // Back up original colors before the first override so they can
    // be restored when the stream is cleared, disabled, or removed.
    for (let i = 0; i < this.state.nodeCount; i++) {
      const colorBase = i * 4;
      const alpha = colors[colorBase + 3] ?? 0;

      if (alpha > 0) {
        const attrBase = i * NODE_ATTR_FLOATS;

        // Save base color before first stream override
        if (!this.streamColorBackups.has(i)) {
          this.streamColorBackups.set(i, [
            nodeAttrs[attrBase + 1] ?? 0,
            nodeAttrs[attrBase + 2] ?? 0,
            nodeAttrs[attrBase + 3] ?? 0,
          ]);
        }

        // Node attrs layout: [radius, r, g, b, selected, hovered, birth_time, tex_index]
        nodeAttrs[attrBase + 1] = colors[colorBase] ?? 0; // R
        nodeAttrs[attrBase + 2] = colors[colorBase + 1] ?? 0; // G
        nodeAttrs[attrBase + 3] = colors[colorBase + 2] ?? 0; // B
        currentlyColored.add(i);
        changed = true;
        if (i < minChanged) minChanged = i;
        if (i > maxChanged) maxChanged = i;
      }
    }

    // Restore base colors for nodes that were stream-colored but no longer are.
    // This handles stream clear, disable, and removal without leaving stale colors.
    const toRemove: number[] = [];
    for (const [nodeIdx, backup] of this.streamColorBackups) {
      if (!currentlyColored.has(nodeIdx)) {
        if (nodeIdx < this.state.nodeCount) {
          const attrBase = nodeIdx * NODE_ATTR_FLOATS;
          nodeAttrs[attrBase + 1] = backup[0];
          nodeAttrs[attrBase + 2] = backup[1];
          nodeAttrs[attrBase + 3] = backup[2];
          changed = true;
          if (nodeIdx < minChanged) minChanged = nodeIdx;
          if (nodeIdx > maxChanged) maxChanged = nodeIdx;
        }
        toRemove.push(nodeIdx);
      }
    }
    for (const nodeIdx of toRemove) {
      this.streamColorBackups.delete(nodeIdx);
    }

    if (changed && maxChanged >= 0) {
      const { device } = this.gpuContext;
      // Upload only the affected subrange of the node attributes buffer
      const byteOffset = minChanged * NODE_ATTR_BYTES;
      const byteLength = (maxChanged - minChanged + 1) * NODE_ATTR_BYTES;
      device.queue.writeBuffer(
        this.buffers.nodeAttributes,
        byteOffset,
        toArrayBuffer(nodeAttrs),
        byteOffset,
        byteLength,
      );
      this.markRenderDirty();
    }
  }

  /**
   * Update heatmap render context if it's using the specified stream.
   * Called internally when stream data changes to ensure heatmap reflects updates.
   */
  private updateHeatmapIfUsingStream(streamId: string): void {
    for (const layer of this.layerManager.getLayersByType<HeatmapLayer>("heatmap")) {
      if (layer.getDataSource() === streamId) {
        this.refreshLayerRenderContexts();
        return;
      }
    }
  }

  // ==========================================================================
  // Public API - Events
  // ==========================================================================

  /**
   * Subscribe to an event
   *
   * @returns Unsubscribe function — call it to remove the handler
   */
  on<K extends keyof EventMap>(event: K, handler: EventHandler<EventMap[K]>): () => void {
    return this.events.on(event, handler);
  }

  /**
   * Unsubscribe from an event
   */
  off<K extends keyof EventMap>(event: K, handler: EventHandler<EventMap[K]>): void {
    this.events.off(event, handler);
  }

  /**
   * Subscribe to an event once
   *
   * @returns Unsubscribe function — call it to remove the handler before it fires
   */
  once<K extends keyof EventMap>(event: K, handler: EventHandler<EventMap[K]>): () => void {
    return this.events.once(event, handler);
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
   * Set a node's pinned state in both the CPU set and the GPU nodeFlags
   * buffer. The integrate shader holds pinned nodes in place: their position
   * is carried through the ping-pong unchanged and their velocity zeroed,
   * so forces stop moving them while they keep repelling/attracting others.
   */
  private setNodePinnedState(slot: number, pinned: boolean): void {
    if (pinned) {
      this.pinnedNodes.add(slot);
    } else {
      this.pinnedNodes.delete(slot);
    }

    const gs = this.graphState;
    if (!gs || !this.simBuffers) return;
    if (slot < 0 || slot >= gs.nodeHighWater) return;

    // A freed slot is not a node: pinning it must not resurrect it. Every
    // other bit in the word (dead, hidden) belongs to someone else and is
    // left exactly as it was.
    const dead = (gs.nodeFlagsShadow[slot] & NODE_FLAG_DEAD) !== 0;
    if (gs.setNodeFlagBits(slot, NODE_FLAG_PINNED, pinned && !dead)) {
      this.flushNodeFlagSlot(slot);
    }
  }

  /**
   * Pin a node (exclude from simulation, fixed position).
   * @param nodeId Node ID to pin
   */
  pinNode(nodeId: NodeId): void {
    this.setNodePinnedState(nodeId, true);
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
    this.setNodePinnedState(nodeId, false);
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

    // Pin the node (sets the GPU flag so integration holds the written position)
    this.setNodePinnedState(nodeId, true);

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
   * Get the identifier the producer supplied for a node slot.
   *
   * `NodeId` is a GPU slot index, recycled on removal and meaningless outside
   * this instance, so anything keeping its own per-node record — a DOM card,
   * a selection set, a search result — needs this to key on. Without it a
   * consumer has to maintain a duplicate slot table of its own.
   *
   * Returns `undefined` — never throws — for a slot that is out of range, was
   * freed by a removal, or was never mapped, so a stale id held across a
   * mutation is safe to look up.
   *
   * @param nodeId Node slot index
   * @returns Producer identifier, or undefined if the slot holds no live node
   */
  getExternalId(nodeId: NodeId): IdLike | undefined {
    const gs = this.graphState;
    if (!gs) return undefined;
    return externalIdForSlot(gs, nodeId);
  }

  /**
   * Get the node slot currently holding a producer identifier.
   *
   * The inverse of {@link GraphMother.getExternalId}. Returns `undefined` if
   * the identifier was never loaded or its node has been removed; because
   * slots are recycled, the returned value is only valid until the next
   * mutation.
   *
   * @param externalId Identifier the producer supplied for the node
   * @returns Node slot index, or undefined if no live node carries that id
   */
  getNodeId(externalId: IdLike): NodeId | undefined {
    const gs = this.graphState;
    if (!gs) return undefined;
    return slotForExternalId(gs, externalId);
  }

  /**
   * Get currently hovered node.
   */
  getHoveredNode(): NodeId | null {
    return this.hover.node;
  }

  /**
   * Get currently hovered edge.
   *
   * Edge hover is only tracked while something is subscribed to
   * `edge:hoverenter` or `edge:hoverleave` — the scan costs a pass over every
   * edge on every frame the pointer moves, and it is skipped when nothing
   * observes the result. Subscribe (even with a no-op handler) to poll this.
   */
  getHoveredEdge(): EdgeId | null {
    return this.hover.edge;
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
        this.nodeDrag.begin(nodeId, e.graphPosition);
        this.dragStartScreenPosition = { ...e.screenPosition };

        // Select if not already selected (or add to selection with shift)
        if (!e.modifiers.shift && !this.selectedNodes.has(nodeId)) {
          this.selectNodes([nodeId]);
        } else if (e.modifiers.shift) {
          this.addToSelection([nodeId]);
        }
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

        // Where the press landed, kept apart from lastPanPosition (which the
        // pan consumes as it goes): the click test at pointerup measures total
        // displacement from the press, not from the last move.
        this.backgroundPressPosition = { ...e.screenPosition };

        // Start panning regardless of edge hit
        this.isPanning = true;
        this.lastPanPosition = { ...e.screenPosition };
      }
    });

    // Handle pointer move (drag, pan, or hover)
    this.pointerManager.on("pointermove", (e) => {
      const dragging = this.nodeDrag.node;
      if (dragging !== null) {
        this.nodeDrag.move(dragging, e.graphPosition);
      } else if (this.isPanning && this.lastPanPosition) {
        // Pan the viewport (inverted - like pushing a piece of paper)
        // Use panScreen since delta is in screen pixels, not graph units
        const dx = this.lastPanPosition.x - e.screenPosition.x;
        const dy = this.lastPanPosition.y - e.screenPosition.y;
        this.viewport.panScreen(dx, dy);
        this.lastPanPosition = { ...e.screenPosition };
      } else {
        // Hover detection — evaluated once per frame, not once per event.
        this.hover.move(e.screenPosition.x, e.screenPosition.y);
      }
    });

    // Handle pointer up (end drag or pan)
    this.pointerManager.on("pointerup", (e) => {
      const nodeId = this.nodeDrag.node;
      if (nodeId !== null) {
        const dragStart = this.dragStartScreenPosition;
        this.dragStartScreenPosition = null;

        // Distinguish click from drag by how far the pointer travelled.
        const isClick = dragStart !== null && isClickDistance(dragStart, e.screenPosition);

        if (isClick) {
          // Not a drag after all: no `node:dragend`, and the pin the gesture
          // took at pointerdown goes back.
          this.nodeDrag.cancel();
          this.setNodePinnedState(nodeId, false);

          // pointerup always has a PointerEvent (not WheelEvent)
          this.events.emit(
            Events.nodeClick(nodeId, e.graphPosition, e.originalEvent as PointerEvent),
          );

          // Two clicks on the same node within the interval form a double-click
          const DBLCLICK_INTERVAL_MS = 350;
          const now = Date.now();
          if (
            this.lastClick !== null &&
            this.lastClick.nodeId === nodeId &&
            now - this.lastClick.timestamp <= DBLCLICK_INTERVAL_MS
          ) {
            this.events.emit(
              Events.nodeDoubleClick(
                nodeId,
                e.graphPosition,
                e.originalEvent as PointerEvent,
              ),
            );
            this.lastClick = null;
          } else {
            this.lastClick = { nodeId, timestamp: now };
          }
        } else {
          this.nodeDrag.end(nodeId, e.graphPosition);
        }
      }

      // A press on empty canvas that released without travelling is a click on
      // the background — the gesture a consumer dismisses a selection or an LOD
      // focus with. A pan is the same press with movement in the middle, so the
      // distance test is what separates them, and a press that hit a node
      // cannot reach here: it took the branch above.
      const pressed = this.backgroundPressPosition;
      this.backgroundPressPosition = null;
      if (nodeId === null && pressed !== null && isClickDistance(pressed, e.screenPosition)) {
        this.events.emit(
          Events.backgroundClick(e.graphPosition, e.originalEvent as PointerEvent),
        );
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
          // A carded node's sprite is behind its card, so the GPU selection
          // ring above is invisible for exactly the nodes the user is closest
          // to. The card renders the state instead.
          this.domOverlay?.notify(nodeId, { kind: "selection", selected: true });
        }
        for (const nodeId of removed) {
          this.syncNodeSelectionToGPU(nodeId, false);
          this.domOverlay?.notify(nodeId, { kind: "selection", selected: false });
        }
      }

      this.markRenderDirty();

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
   * Node attributes: [radius, r, g, b, selected, hovered, birth_time, tex_index] (8 floats per node)
   */
  private syncNodeSelectionToGPU(nodeId: NodeId, selected: boolean): void {
    if (!this.buffers || !this.state.parsedGraph) return;

    // nodeId is the array index in our system
    const idx = nodeId;
    if (idx < 0 || idx >= this.state.parsedGraph.nodeCount) return;

    const { device } = this.gpuContext;
    // Node attributes are 8 floats per node, selection is at offset 4
    const attrOffset = idx * NODE_ATTR_BYTES + 4 * 4; // 8 floats * 4 bytes, offset 4
    const selectionValue = new Float32Array([selected ? 1.0 : 0.0]);
    device.queue.writeBuffer(this.buffers.nodeAttributes, attrOffset, selectionValue);

    // Also update local parsed graph data
    this.state.parsedGraph.nodeAttributes[idx * NODE_ATTR_FLOATS + 4] = selected ? 1.0 : 0.0;
  }

  /**
   * Update node hover state in GPU buffer
   * Node attributes: [radius, r, g, b, selected, hovered, birth_time, tex_index] (8 floats per node)
   */
  private syncNodeHoverToGPU(nodeId: NodeId, hovered: boolean): void {
    if (!this.buffers || !this.state.parsedGraph) return;

    // nodeId is the array index in our system
    const idx = nodeId;
    if (idx < 0 || idx >= this.state.parsedGraph.nodeCount) return;

    const { device } = this.gpuContext;
    // Node attributes are 8 floats per node, hover is at offset 5
    const attrOffset = idx * NODE_ATTR_BYTES + 5 * 4; // 8 floats * 4 bytes, offset 5
    const hoverValue = new Float32Array([hovered ? 1.0 : 0.0]);
    device.queue.writeBuffer(this.buffers.nodeAttributes, attrOffset, hoverValue);

    // Also update local parsed graph data
    this.state.parsedGraph.nodeAttributes[idx * NODE_ATTR_FLOATS + 5] = hovered ? 1.0 : 0.0;
  }

  /**
   * Update hit tester with current position data.
   *
   * Deliberately does NOT wire the WASM spatial engine: the R-tree indexes
   * the engine's own position copy, which is only written at load/mutation
   * time — the GPU simulation moves nodes every frame, so R-tree queries
   * would hit-test against stale layout. Keeping it fresh would require
   * pushing all positions into WASM and an O(n log n) rebuild every
   * position sync (every SYNC_INTERVAL frames), which at target scale
   * (~35K nodes) costs far more than the on-demand O(N) scan of the CPU
   * shadow that only runs on pointer events.
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
        // nodeAttributes layout: [radius, r, g, b, selected, hovered, birth_time, tex_index] per node
        if (nodeId < 0 || nodeId >= parsedGraph.nodeCount) return undefined;
        return parsedGraph.nodeAttributes[nodeId * NODE_ATTR_FLOATS]; // radius is at offset 0
      },
      getNodeIds: function* () {
        for (let i = 0; i < parsedGraph.nodeCount; i++) {
          yield i;
        }
      },
      getNodeCount: () => parsedGraph.nodeCount,
      isNodeHittable: (nodeId: NodeId): boolean => {
        const flags = this.graphState?.nodeFlagsShadow;
        return flags === undefined || (flags[nodeId] & NODE_FLAGS_UNHITTABLE) === 0;
      },
      // Read per scan, never captured: growth replaces these arrays.
      getNodeColumns: () => ({
        count: parsedGraph.nodeCount,
        x: parsedGraph.positionsX,
        y: parsedGraph.positionsY,
        radii: parsedGraph.nodeAttributes,
        radiusStride: NODE_ATTR_FLOATS,
        radiusOffset: 0,
        // A freed slot and a node the LOD cut folded away are both absent from
        // the screen, and neither may answer for the proxy drawn over it.
        flags: this.graphState?.nodeFlagsShadow,
        skipMask: NODE_FLAGS_UNHITTABLE,
      }),
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
      getEdgeColumns: () => ({
        count: parsedGraph.edgeSources.length,
        sources: parsedGraph.edgeSources,
        targets: parsedGraph.edgeTargets,
      }),
    });
  }

  // ==========================================================================
  // Public API - Lifecycle
  // ==========================================================================

  /**
   * Resize the graph canvas.
   *
   * With explicit dimensions, sets the drawing-buffer size directly (the
   * caller owns devicePixelRatio scaling). With no arguments, measures the
   * canvas's CSS size and sizes the drawing buffer to
   * `clientSize × devicePixelRatio` — the right call from a ResizeObserver.
   */
  resize(width?: number, height?: number): void {
    if (width !== undefined && height !== undefined) {
      this.canvas.width = width;
      this.canvas.height = height;
    } else if (this.canvas.clientWidth > 0 && this.canvas.clientHeight > 0) {
      resizeGPUContext(this.gpuContext, this.canvas.clientWidth, this.canvas.clientHeight);
    }
    // Update viewport with CSS dimensions for coordinate transforms
    const cssWidth = this.canvas.clientWidth || this.canvas.width;
    const cssHeight = this.canvas.clientHeight || this.canvas.height;
    this.viewport.resize(cssWidth, cssHeight);
    this.updateViewportUniforms();
    this.markRenderDirty();

    // Resize layers
    this.layerManager.resize(cssWidth, cssHeight);

    // Update layer render contexts after resize (texture views may have changed)
    this.refreshLayerRenderContexts();
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

    // Release every card and detach the overlay container from the host
    this.domOverlay?.dispose();
    this.domOverlay = null;

    // Drop any hover evaluation still waiting on a frame callback
    this.hover.reset();

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

    // Destroy per-layer stream intensity buffers
    for (const cache of this.streamIntensityCaches.values()) {
      cache.destroy();
    }
    this.streamIntensityCaches.clear();

    // Destroy default intensity buffer
    this.defaultIntensityBuffer?.destroy();
    this.defaultIntensityBuffer = null;

    // Destroy render config buffer
    this.renderConfigBuffer?.destroy();
    this.renderConfigBuffer = null;

    // Destroy viewport uniform buffer
    this.viewportUniformBuffer.destroy();

    // Release the WASM engine's linear-memory allocations (the JS GC cannot
    // reclaim wasm-bindgen objects, and WASM memory never shrinks)
    this.wasmEngine?.free();

    // Unconfigure the canvas context and destroy the GPU device. Each
    // instance requests its own device, so skipping this leaks a GPUDevice
    // per mount/unmount cycle.
    destroyGPUContext(this.gpuContext);

    // Clear event listeners
    this.events.clear();

    if (this.debug) {
      console.log("GraphMother disposed");
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
      this.simBuffers.prevForces.destroy();
      this.simBuffers.nodeFlags.destroy();
      this.simBuffers.nodeAlpha.destroy();
      this.simBuffers.nodeMass.destroy();
      this.simBuffers.edgeSources.destroy();
      this.simBuffers.edgeTargets.destroy();
      this.simBuffers.lodEdgeSet.destroy();
      this.simBuffers.clearUniforms.destroy();
      this.simBuffers.repulsionUniforms.destroy();
      this.simBuffers.springUniforms.destroy();
      this.simBuffers.integrationUniforms.destroy();
      this.simBuffers.nodeDepth.destroy();
      this.simBuffers.liveIndices.destroy();
      this.simBuffers.readback.destroy();
      this.simBuffers = null;
    }
    this.nodeMassShadow = null;
    this.activeIndexShadow = null;
    // The bundle draw reads the destroyed position buffers through its own
    // bind group, and its instance buffers describe the edge set going away.
    this.destroyLodEdgeRenderBuffers();
    this.lodEdgeOpacity.release(null);
    // Every parity-indexed bind group references at least one buffer destroyed
    // above — including the node/edge render sets, which read the simulation
    // position buffers directly. rebuildAllBindGroups recreates them once
    // fresh buffers exist.
    this.paritySets.clearAll();

    // Destroy algorithm-specific buffers
    this.algorithmBuffers?.destroy();
    this.algorithmBuffers = null;

    // Destroy collision buffers
    if (this.collisionBuffers) {
      destroyCollisionBuffers(this.collisionBuffers);
      this.collisionBuffers = null;
    }

    // Destroy grid collision buffers
    if (this.gridCollisionBuffers) {
      destroyGridCollisionBuffers(this.gridCollisionBuffers);
      this.gridCollisionBuffers = null;
    }
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
