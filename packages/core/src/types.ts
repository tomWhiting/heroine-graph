/**
 * GraphMother - Core Type Definitions
 *
 * This file defines the public API types for the GraphMother library.
 * These are contracts - implementations must conform to these interfaces.
 */

// =============================================================================
// Identifiers
// =============================================================================

import type { HierarchyColumns } from "./graph/hierarchy.ts";

/** Stable node identifier (survives graph mutations) */
export type NodeId = number;

/** Stable edge identifier (survives graph mutations) */
export type EdgeId = number;

// =============================================================================
// Primitives
// =============================================================================

/** 2D vector */
export interface Vec2 {
  readonly x: number;
  readonly y: number;
}

/** Axis-aligned bounding box */
export interface BoundingBox {
  readonly minX: number;
  readonly minY: number;
  readonly maxX: number;
  readonly maxY: number;
}

/** CSS color string */
export type Color = string;

/** Color scale for visualizations */
export type ColorScale =
  | "viridis"
  | "plasma"
  | "inferno"
  | "magma"
  | "turbo"
  | readonly Color[];

// =============================================================================
// Node Types
// =============================================================================

/** User-provided node metadata */
export interface NodeMetadata {
  /** Text label for display */
  readonly label?: string;
  /** Override default node color */
  readonly color?: Color;
  /** Override default node radius */
  readonly radius?: number;
  /** Grouping identifier for clustering */
  readonly group?: string;
  /** Label priority (0-1, higher = more important) */
  readonly importance?: number;
  /** Arbitrary user data */
  readonly data?: Readonly<Record<string, unknown>>;
}

/** Node interaction state (read-only from user perspective) */
export interface NodeState {
  /** Currently selected */
  readonly selected: boolean;
  /** Mouse is hovering over node */
  readonly hovered: boolean;
  /** Fixed position (excluded from simulation) */
  readonly pinned: boolean;
  /** Not rendered */
  readonly hidden: boolean;
}

/** Complete node representation */
export interface Node {
  /** Stable unique identifier */
  readonly id: NodeId;
  /** X position in graph space */
  readonly x: number;
  /** Y position in graph space */
  readonly y: number;
  /** User-defined metadata */
  readonly metadata: NodeMetadata;
  /** Interaction state */
  readonly state: NodeState;
}

// =============================================================================
// Edge Types
// =============================================================================

/** User-provided edge metadata */
export interface EdgeMetadata {
  /** Force simulation weight */
  readonly weight?: number;
  /** Override default edge color */
  readonly color?: Color;
  /** Override default edge width */
  readonly width?: number;
  /** Text label for display */
  readonly label?: string;
  /** Show directional arrow */
  readonly directed?: boolean;
  /** Arbitrary user data */
  readonly data?: Readonly<Record<string, unknown>>;
}

/** Edge interaction state */
export interface EdgeState {
  /** Currently selected */
  readonly selected: boolean;
  /** Mouse is hovering over edge */
  readonly hovered: boolean;
  /** Not rendered */
  readonly hidden: boolean;
}

/** Complete edge representation */
export interface Edge {
  /** Stable unique identifier */
  readonly id: EdgeId;
  /** Source node identifier */
  readonly source: NodeId;
  /** Target node identifier */
  readonly target: NodeId;
  /** User-defined metadata */
  readonly metadata: EdgeMetadata;
  /** Interaction state */
  readonly state: EdgeState;
}

// =============================================================================
// Simulation Types
// =============================================================================

/** Simulation running state */
export type SimulationStatus =
  | "idle"
  | "stopped"
  | "running"
  | "paused"
  | "cooling";

/** Force configuration parameters */
export interface ForceConfig {
  /** Many-body repulsion strength (negative = repel) */
  readonly repulsion: number;
  /** Link spring strength */
  readonly attraction: number;
  /** Center gravity strength */
  readonly gravity: number;
  /** Gravity center X coordinate */
  readonly centerX: number;
  /** Gravity center Y coordinate */
  readonly centerY: number;
  /** Ideal link distance */
  readonly linkDistance: number;
  /** Barnes-Hut approximation parameter (0.5-1.5 typical) */
  readonly theta: number;
}

/** Simulation state */
export interface SimulationState {
  /** Current running state */
  readonly status: SimulationStatus;
  /** Current simulation energy (0-1) */
  readonly alpha: number;
  /** Target alpha for decay */
  readonly alphaTarget: number;
  /** Rate of energy loss per tick */
  readonly alphaDecay: number;
  /** Stop threshold */
  readonly alphaMin: number;
  /** Velocity damping factor */
  readonly velocityDecay: number;
  /** Force parameters */
  readonly forces: ForceConfig;
}

// =============================================================================
// Layer Types
// =============================================================================

/** Layer type discriminator */
export type LayerType = "heatmap" | "contour" | "metaball" | "labels";

/** Heatmap layer configuration */
export interface HeatmapLayerConfig {
  readonly type: "heatmap";
  /** Gaussian kernel radius */
  readonly radius: number;
  /** Color intensity multiplier */
  readonly intensity: number;
  /** Color gradient */
  readonly colorScale: ColorScale;
  /** Layer opacity (0-1) */
  readonly opacity: number;
}

/** Contour layer configuration */
export interface ContourLayerConfig {
  readonly type: "contour";
  /** Density values for iso-lines */
  readonly thresholds: readonly number[];
  /** Line stroke width */
  readonly strokeWidth: number;
  /** Line color */
  readonly strokeColor: Color;
  /** Layer opacity (0-1) */
  readonly opacity: number;
}

/** Metaball layer configuration */
export interface MetaballLayerConfig {
  readonly type: "metaball";
  /** SDF threshold for boundary */
  readonly threshold: number;
  /** Smooth union blend radius */
  readonly blendRadius: number;
  /** Fill color */
  readonly fillColor: Color;
  /** Layer opacity (0-1) */
  readonly opacity: number;
}

/** Label layer configuration */
export interface LabelLayerConfig {
  readonly type: "labels";
  /** Font family name */
  readonly fontFamily: string;
  /** Base font size in pixels */
  readonly fontSize: number;
  /** Text color */
  readonly fontColor: Color;
  /** Hide labels below this zoom level */
  readonly minZoom: number;
  /** Maximum number of visible labels */
  readonly maxLabels: number;
  /** Label ranking strategy */
  readonly priority: "importance" | "degree";
}

/** Union of all layer configurations */
export type LayerConfig =
  | HeatmapLayerConfig
  | ContourLayerConfig
  | MetaballLayerConfig
  | LabelLayerConfig;

/** Layer state */
export interface Layer {
  /** Layer type */
  readonly type: LayerType;
  /** Currently visible */
  readonly visible: boolean;
  /** Render order (lower = behind) */
  readonly order: number;
  /** Type-specific configuration */
  readonly config: LayerConfig;
}

// =============================================================================
// Viewport Types
// =============================================================================

/** Viewport (camera) state */
export interface ViewportState {
  /** Pan offset X in graph units */
  readonly x: number;
  /** Pan offset Y in graph units */
  readonly y: number;
  /** Zoom scale factor */
  readonly scale: number;
  /** Canvas width in pixels */
  readonly width: number;
  /** Canvas height in pixels */
  readonly height: number;
  /** Minimum allowed zoom */
  readonly minScale: number;
  /** Maximum allowed zoom */
  readonly maxScale: number;
}

// =============================================================================
// Graph Configuration
// =============================================================================

/** Graph-wide visual defaults */
export interface GraphConfig {
  /** Default node radius */
  readonly nodeDefaultRadius: number;
  /** Default node color */
  readonly nodeDefaultColor: Color;
  /** Default edge width */
  readonly edgeDefaultWidth: number;
  /** Default edge color */
  readonly edgeDefaultColor: Color;
  /** Canvas background color */
  readonly backgroundColor: Color;
}

// =============================================================================
// Edge Flow Types
// =============================================================================

/** Wave shape for flow animation */
export type EdgeFlowWaveShape = "square" | "triangle" | "sine";

/** Single layer of edge flow animation */
export interface EdgeFlowLayerConfig {
  /** Whether this layer is enabled */
  readonly enabled: boolean;
  /** Width of each pulse (0.005 - 0.8, normalized) */
  readonly pulseWidth: number;
  /** Number of pulses along each edge (1 - 8) */
  readonly pulseCount: number;
  /** Animation speed multiplier (0.01 - 2.0) */
  readonly speed: number;
  /** Wave shape: square (hard edges), triangle (linear), sine (smooth) */
  readonly waveShape: EdgeFlowWaveShape;
  /** Brightness multiplier (1.0 - 5.0) */
  readonly brightness: number;
  /** Trail fade amount (0.0 - 1.0, 0 = no trail) */
  readonly fade: number;
  /** Optional color override (null = use edge color) */
  readonly color: readonly [number, number, number, number] | null;
}

/** Complete edge flow configuration (dual-layer) */
export interface EdgeFlowConfig {
  /** Primary flow layer */
  readonly layer1: EdgeFlowLayerConfig;
  /** Secondary flow layer (sparks, highlights) */
  readonly layer2: EdgeFlowLayerConfig;
}

// =============================================================================
// Input Types (for creating/loading graphs)
// =============================================================================

/** Node input for graph creation */
export interface NodeInput {
  /** Unique identifier (string or number) */
  readonly id: string | number;
  /** Initial X position (random if omitted) */
  readonly x?: number;
  /** Initial Y position (random if omitted) */
  readonly y?: number;
  /** Text label */
  readonly label?: string;
  /** Node color */
  readonly color?: Color;
  /** Node radius */
  readonly radius?: number;
  /** Group identifier */
  readonly group?: string;
  /** Label importance (0-1) */
  readonly importance?: number;
  /**
   * Producer-defined semantic tag index, for the LOD policy hook.
   *
   * Opaque to core: it is compared, never interpreted, and the producer's tag
   * table never crosses the boundary. Defaults to 0.
   */
  readonly tag?: number;
  /**
   * Producer-defined importance in 0..1, for the LOD policy hook.
   *
   * Distinct from {@link NodeInput.importance}, which ranks *labels*. This one
   * ranks the node itself and is used as a tie-break by core even with no
   * policy registered. Values outside 0..1 are clamped. Defaults to 0.
   */
  readonly weight?: number;
  /** Birth time for animation (from getAnimationTime()). 0 = no animation. */
  readonly birthTime?: number;
  /** Additional metadata */
  readonly [key: string]: unknown;
}

/** Edge input for graph creation */
export interface EdgeInput {
  /** Source node ID (matches NodeInput.id) */
  readonly source: string | number;
  /** Target node ID (matches NodeInput.id) */
  readonly target: string | number;
  /** Edge weight for simulation */
  readonly weight?: number;
  /** Edge color */
  readonly color?: Color;
  /** Edge width */
  readonly width?: number;
  /** Show direction arrow */
  readonly directed?: boolean;
  /** Additional metadata */
  readonly [key: string]: unknown;
}

/** Standard graph input format */
export interface GraphInput {
  readonly nodes: readonly NodeInput[];
  readonly edges: readonly EdgeInput[];
}

/** High-performance typed array input format */
export interface GraphTypedInput {
  /** Number of nodes */
  readonly nodeCount: number;
  /** Number of edges */
  readonly edgeCount?: number | undefined;
  /**
   * Producer's revision identifier for this snapshot.
   *
   * Reserved: accepted and **ignored in v1**. The field exists now so that
   * content-addressed transport and snapshot reconciliation have somewhere to
   * land without a breaking input change.
   */
  readonly revision?: string | undefined;
  /** Positions as [x0, y0, x1, y1, ...] */
  readonly positions?: Float32Array | undefined;
  /** Edge pairs as [src0, tgt0, src1, tgt1, ...] */
  readonly edgePairs?: Uint32Array | undefined;
  /**
   * Edges as [src0, tgt0, src1, tgt1, ...].
   *
   * @deprecated Alias for {@link GraphTypedInput.edgePairs}, honoured for
   * producers already sending it. Supplying both is an error, since there is no
   * way to tell which one the producer meant. Prefer `edgePairs`.
   */
  readonly edges?: Uint32Array | undefined;
  /**
   * Optional per-edge kind index, consumer-defined.
   *
   * Core interprets exactly one value — {@link GraphTypedInput.containmentKind}
   * — and never reads the producer's kind table. Requires `containmentKind`;
   * without it the column carries no meaning core can act on.
   */
  readonly edgeKinds?: Uint16Array | undefined;
  /**
   * Which {@link GraphTypedInput.edgeKinds} value means containment
   * (parent → child).
   *
   * This is what makes a containment hierarchy expressible on the typed fast
   * path: without it, every edge is treated as containment and cross-cutting
   * dependency edges corrupt depths and bubble radii.
   */
  readonly containmentKind?: number | undefined;
  /**
   * Precomputed containment hierarchy, indexed by slot.
   *
   * Supplied-or-computed: when present it is validated and retained as-is;
   * when absent the hierarchy is derived in WASM from the containment edges.
   * Slot `i` here is node `i` of this input.
   *
   * Producers SHOULD emit slots in depth-first order so that each subtree is a
   * contiguous slot range; core neither verifies nor reorders.
   */
  readonly hierarchy?: HierarchyColumns | undefined;
  /** Optional node IDs (string or number) */
  readonly nodeIds?: readonly (string | number)[] | undefined;
  /** Optional edge IDs (string or number) */
  readonly edgeIds?: readonly (string | number)[] | undefined;
  /**
   * Producer-defined semantic tag per slot, for the LOD policy hook.
   *
   * Opaque to core: the values are compared against whatever the consumer's
   * policy expects and the tag table itself never crosses the boundary. An
   * integer column rather than a metadata lookup because it is read on the
   * evaluation path, once per node that crossed a band. Missing slots read 0.
   */
  readonly tag?: Uint16Array | undefined;
  /**
   * Producer-defined importance per slot, in 0..1, for the LOD policy hook.
   *
   * Used by core as a tie-break between nodes of equal screen size even with
   * no policy registered, so supplying it improves default behaviour. Values
   * outside 0..1 are clamped. Missing slots read 0.
   */
  readonly weight?: Float32Array | undefined;
  /** Optional node radii */
  readonly nodeRadii?: Float32Array | undefined;
  /** Optional node colors as [r0, g0, b0, r1, g1, b1, ...] */
  readonly nodeColors?: Float32Array | undefined;
  /** Optional edge widths */
  readonly edgeWidths?: Float32Array | undefined;
  /** Optional edge colors as [r0, g0, b0, r1, g1, b1, ...] */
  readonly edgeColors?: Float32Array | undefined;
  /** Optional node metadata */
  readonly nodeMetadata?: readonly NodeMetadata[] | undefined;
  /** Optional edge metadata */
  readonly edgeMetadata?: readonly EdgeMetadata[] | undefined;
}

// =============================================================================
// Event Types
// =============================================================================

/** Base event interface */
export interface GraphEvent {
  /** Event type discriminator */
  readonly type: string;
  /** Timestamp (performance.now()) */
  readonly timestamp: number;
}

/** Node click event */
export interface NodeClickEvent extends GraphEvent {
  readonly type: "node:click";
  readonly nodeId: NodeId;
  readonly position: Vec2;
  readonly originalEvent: PointerEvent;
}

/** Node double-click event */
export interface NodeDoubleClickEvent extends GraphEvent {
  readonly type: "node:dblclick";
  readonly nodeId: NodeId;
  readonly position: Vec2;
  readonly originalEvent: PointerEvent;
}

/** Node hover enter event */
export interface NodeHoverEnterEvent extends GraphEvent {
  readonly type: "node:hoverenter";
  readonly nodeId: NodeId;
  readonly position: Vec2;
}

/** Node hover leave event */
export interface NodeHoverLeaveEvent extends GraphEvent {
  readonly type: "node:hoverleave";
  readonly nodeId: NodeId;
}

/** Node drag start event */
export interface NodeDragStartEvent extends GraphEvent {
  readonly type: "node:dragstart";
  readonly nodeId: NodeId;
  readonly position: Vec2;
}

/** Node drag move event */
export interface NodeDragMoveEvent extends GraphEvent {
  readonly type: "node:dragmove";
  readonly nodeId: NodeId;
  readonly position: Vec2;
  readonly delta: Vec2;
}

/** Node drag end event */
export interface NodeDragEndEvent extends GraphEvent {
  readonly type: "node:dragend";
  readonly nodeId: NodeId;
  readonly position: Vec2;
}

/** Edge click event */
export interface EdgeClickEvent extends GraphEvent {
  readonly type: "edge:click";
  readonly edgeId: EdgeId;
  readonly position: Vec2;
  readonly originalEvent: PointerEvent;
}

/** Edge hover enter event */
export interface EdgeHoverEnterEvent extends GraphEvent {
  readonly type: "edge:hoverenter";
  readonly edgeId: EdgeId;
  readonly position: Vec2;
}

/** Edge hover leave event */
export interface EdgeHoverLeaveEvent extends GraphEvent {
  readonly type: "edge:hoverleave";
  readonly edgeId: EdgeId;
}

/** Viewport change event */
export interface ViewportChangeEvent extends GraphEvent {
  readonly type: "viewport:change";
  readonly viewport: ViewportState;
}

/** Simulation tick event */
export interface SimulationTickEvent extends GraphEvent {
  readonly type: "simulation:tick";
  readonly alpha: number;
  readonly iteration: number;
}

/** Simulation end event */
export interface SimulationEndEvent extends GraphEvent {
  readonly type: "simulation:end";
  readonly iterations: number;
}

/** GPU device lost event — the graph can no longer render and must be recreated */
export interface DeviceLostEvent extends GraphEvent {
  readonly type: "device:lost";
  /** Reason reported by WebGPU ("destroyed" is expected during dispose) */
  readonly reason: string;
  /** Human-readable message from the browser */
  readonly message: string;
}

/** Selection change event */
export interface SelectionChangeEvent extends GraphEvent {
  readonly type: "selection:change";
  readonly selectedNodes: readonly NodeId[];
  readonly selectedEdges: readonly EdgeId[];
}

/** Background click event */
export interface BackgroundClickEvent extends GraphEvent {
  readonly type: "background:click";
  readonly position: Vec2;
  readonly originalEvent: PointerEvent;
}

/** Graph load event */
export interface GraphLoadEvent extends GraphEvent {
  readonly type: "graph:load";
  readonly nodeCount: number;
  readonly edgeCount: number;
}

/** Node pin event */
export interface NodePinEvent extends GraphEvent {
  readonly type: "node:pin";
  readonly nodeId: NodeId;
}

/** Node unpin event */
export interface NodeUnpinEvent extends GraphEvent {
  readonly type: "node:unpin";
  readonly nodeId: NodeId;
}

/** Node added event */
export interface NodeAddEvent extends GraphEvent {
  readonly type: "node:add";
  readonly nodeId: string | number;
  readonly index: number;
}

/** Node removed event */
export interface NodeRemoveEvent extends GraphEvent {
  readonly type: "node:remove";
  readonly nodeId: string | number;
  readonly index: number;
}

/** Edge added event */
export interface EdgeAddEvent extends GraphEvent {
  readonly type: "edge:add";
  readonly edgeId: string | number;
  readonly sourceId: string | number;
  readonly targetId: string | number;
}

/** Edge removed event */
export interface EdgeRemoveEvent extends GraphEvent {
  readonly type: "edge:remove";
  readonly edgeId: string | number;
}

/** What drove a semantic-LOD state change. */
export type LodTransitionReason = "zoom" | "policy" | "imperative" | "budget";

/**
 * Summary of one LOD evaluation that changed the cut.
 *
 * `expanded` and `collapsed` list the nodes whose *proxy* status changed — a
 * node entering `collapsed` now stands for its whole subtree, a node leaving it
 * has handed that role back to its children. Nodes merely hidden underneath a
 * collapsed proxy are not listed; they are accounted for by `visibleCount`.
 */
export interface LodChangeEvent extends GraphEvent {
  readonly type: "lod:change";
  readonly expanded: readonly NodeId[];
  readonly collapsed: readonly NodeId[];
  readonly visibleCount: number;
  readonly zoom: number;
}

/** A node became the visible proxy for its subtree. */
export interface NodeCollapseEvent extends GraphEvent {
  readonly type: "node:collapse";
  readonly nodeId: NodeId;
  readonly subtreeSize: number;
  readonly reason: LodTransitionReason;
}

/** A node stopped standing in for its subtree; its children are in the cut. */
export interface NodeExpandEvent extends GraphEvent {
  readonly type: "node:expand";
  readonly nodeId: NodeId;
  readonly childCount: number;
  readonly reason: LodTransitionReason;
}

/** Batch mutation summary event */
export interface GraphMutateEvent extends GraphEvent {
  readonly type: "graph:mutate";
  readonly nodesAdded: number;
  readonly nodesRemoved: number;
  readonly edgesAdded: number;
  readonly edgesRemoved: number;
}

/** Union of all events */
export type GraphMotherEvent =
  | NodeClickEvent
  | NodeDoubleClickEvent
  | NodeHoverEnterEvent
  | NodeHoverLeaveEvent
  | NodeDragStartEvent
  | NodeDragMoveEvent
  | NodeDragEndEvent
  | EdgeClickEvent
  | EdgeHoverEnterEvent
  | EdgeHoverLeaveEvent
  | ViewportChangeEvent
  | SimulationTickEvent
  | SimulationEndEvent
  | SelectionChangeEvent
  | BackgroundClickEvent
  | GraphLoadEvent
  | NodePinEvent
  | NodeUnpinEvent
  | NodeAddEvent
  | NodeRemoveEvent
  | EdgeAddEvent
  | EdgeRemoveEvent
  | GraphMutateEvent
  | DeviceLostEvent
  | LodChangeEvent
  | NodeCollapseEvent
  | NodeExpandEvent;

/** Event handler function */
export type EventHandler<E extends GraphEvent> = (event: E) => void;

/** Event type to event mapping */
export interface EventMap {
  "node:click": NodeClickEvent;
  "node:dblclick": NodeDoubleClickEvent;
  "node:hoverenter": NodeHoverEnterEvent;
  "node:hoverleave": NodeHoverLeaveEvent;
  "node:dragstart": NodeDragStartEvent;
  "node:dragmove": NodeDragMoveEvent;
  "node:dragend": NodeDragEndEvent;
  "node:pin": NodePinEvent;
  "node:unpin": NodeUnpinEvent;
  "edge:click": EdgeClickEvent;
  "edge:hoverenter": EdgeHoverEnterEvent;
  "edge:hoverleave": EdgeHoverLeaveEvent;
  "viewport:change": ViewportChangeEvent;
  "simulation:tick": SimulationTickEvent;
  "simulation:end": SimulationEndEvent;
  "selection:change": SelectionChangeEvent;
  "background:click": BackgroundClickEvent;
  "graph:load": GraphLoadEvent;
  "node:add": NodeAddEvent;
  "node:remove": NodeRemoveEvent;
  "edge:add": EdgeAddEvent;
  "edge:remove": EdgeRemoveEvent;
  "graph:mutate": GraphMutateEvent;
  "device:lost": DeviceLostEvent;
  "lod:change": LodChangeEvent;
  "node:collapse": NodeCollapseEvent;
  "node:expand": NodeExpandEvent;
}
