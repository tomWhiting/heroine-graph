/**
 * Heroine Graph - Public API Contract
 *
 * This file defines the public API that implementations must provide.
 */

import type {
  BoundingBox,
  Color,
  Edge,
  EdgeId,
  EdgeInput,
  EdgeMetadata,
  EventHandler,
  EventMap,
  ForceConfig,
  GraphConfig,
  GraphInput,
  GraphTypedInput,
  Layer,
  LayerConfig,
  LayerType,
  Node,
  NodeId,
  NodeInput,
  NodeMetadata,
  SimulationState,
  Vec2,
  ViewportState,
} from "./types.ts";

// =============================================================================
// Initialization Options
// =============================================================================

/** Options for creating a HeroineGraph instance */
export interface HeroineGraphOptions {
  /** Canvas element or CSS selector */
  readonly canvas: HTMLCanvasElement | string;
  /** Initial graph data (optional, can load later) */
  readonly data?: GraphInput | GraphTypedInput;
  /** Graph visual configuration */
  readonly config?: Partial<GraphConfig>;
  /** Initial simulation parameters */
  readonly simulation?: Partial<SimulationState>;
  /** Initial viewport state */
  readonly viewport?: Partial<ViewportState>;
  /** Enable WebGPU debug mode */
  readonly debug?: boolean;
}

/** Result of WebGPU initialization check */
export interface WebGPUStatus {
  /** WebGPU is available and functional */
  readonly supported: boolean;
  /** Detailed error message if not supported */
  readonly error?: string;
  /** GPU adapter info if available */
  readonly adapterInfo?: {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
  };
}

// =============================================================================
// Main API Interface
// =============================================================================

/** Main HeroineGraph API */
export interface HeroineGraph {
  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  /**
   * Load graph data, replacing any existing graph.
   * @param data Graph data in standard or typed array format
   */
  load(data: GraphInput | GraphTypedInput): void;

  /**
   * Destroy the graph instance and release all resources.
   * Instance cannot be used after calling dispose.
   */
  dispose(): void;

  /**
   * Resize the canvas and viewport.
   * Call this when the container size changes.
   * @param width New width in pixels
   * @param height New height in pixels
   */
  resize(width: number, height: number): void;

  // ---------------------------------------------------------------------------
  // Node Operations
  // ---------------------------------------------------------------------------

  /**
   * Get a node by ID.
   * @param id Node identifier
   * @returns Node or undefined if not found
   */
  getNode(id: NodeId): Node | undefined;

  /**
   * Get all nodes.
   * @returns Array of all nodes
   */
  getNodes(): readonly Node[];

  /**
   * Get node count.
   * @returns Number of nodes in the graph
   */
  getNodeCount(): number;

  /**
   * Add a new node to the graph.
   * @param node Node data
   * @returns Assigned NodeId
   */
  addNode(node: NodeInput): NodeId;

  /**
   * Add multiple nodes to the graph.
   * @param nodes Array of node data
   * @returns Array of assigned NodeIds
   */
  addNodes(nodes: readonly NodeInput[]): readonly NodeId[];

  /**
   * Remove a node and all connected edges.
   * @param id Node identifier
   * @returns true if node existed and was removed
   */
  removeNode(id: NodeId): boolean;

  /**
   * Update node metadata.
   * @param id Node identifier
   * @param metadata Partial metadata to merge
   */
  updateNode(id: NodeId, metadata: Partial<NodeMetadata>): void;

  /**
   * Set node position.
   * @param id Node identifier
   * @param x X coordinate
   * @param y Y coordinate
   */
  setNodePosition(id: NodeId, x: number, y: number): void;

  /**
   * Pin a node (exclude from simulation).
   * @param id Node identifier
   */
  pinNode(id: NodeId): void;

  /**
   * Unpin a node (include in simulation).
   * @param id Node identifier
   */
  unpinNode(id: NodeId): void;

  /**
   * Hide a node (not rendered).
   * @param id Node identifier
   */
  hideNode(id: NodeId): void;

  /**
   * Show a hidden node.
   * @param id Node identifier
   */
  showNode(id: NodeId): void;

  // ---------------------------------------------------------------------------
  // Edge Operations
  // ---------------------------------------------------------------------------

  /**
   * Get an edge by ID.
   * @param id Edge identifier
   * @returns Edge or undefined if not found
   */
  getEdge(id: EdgeId): Edge | undefined;

  /**
   * Get all edges.
   * @returns Array of all edges
   */
  getEdges(): readonly Edge[];

  /**
   * Get edge count.
   * @returns Number of edges in the graph
   */
  getEdgeCount(): number;

  /**
   * Add a new edge to the graph.
   * @param edge Edge data
   * @returns Assigned EdgeId, or undefined if source/target not found
   */
  addEdge(edge: EdgeInput): EdgeId | undefined;

  /**
   * Add multiple edges to the graph.
   * @param edges Array of edge data
   * @returns Array of assigned EdgeIds (undefined for invalid edges)
   */
  addEdges(edges: readonly EdgeInput[]): readonly (EdgeId | undefined)[];

  /**
   * Remove an edge.
   * @param id Edge identifier
   * @returns true if edge existed and was removed
   */
  removeEdge(id: EdgeId): boolean;

  /**
   * Update edge metadata.
   * @param id Edge identifier
   * @param metadata Partial metadata to merge
   */
  updateEdge(id: EdgeId, metadata: Partial<EdgeMetadata>): void;

  /**
   * Get edges connected to a node.
   * @param nodeId Node identifier
   * @returns Array of connected edges
   */
  getNodeEdges(nodeId: NodeId): readonly Edge[];

  /**
   * Get neighbor nodes of a node.
   * @param nodeId Node identifier
   * @returns Array of neighbor node IDs
   */
  getNeighbors(nodeId: NodeId): readonly NodeId[];

  // ---------------------------------------------------------------------------
  // Selection
  // ---------------------------------------------------------------------------

  /**
   * Get currently selected nodes.
   * @returns Array of selected node IDs
   */
  getSelectedNodes(): readonly NodeId[];

  /**
   * Get currently selected edges.
   * @returns Array of selected edge IDs
   */
  getSelectedEdges(): readonly EdgeId[];

  /**
   * Select nodes.
   * @param ids Node identifiers to select
   * @param additive If true, add to existing selection; if false, replace
   */
  selectNodes(ids: readonly NodeId[], additive?: boolean): void;

  /**
   * Select edges.
   * @param ids Edge identifiers to select
   * @param additive If true, add to existing selection; if false, replace
   */
  selectEdges(ids: readonly EdgeId[], additive?: boolean): void;

  /**
   * Clear all selection.
   */
  clearSelection(): void;

  /**
   * Select all nodes and edges.
   */
  selectAll(): void;

  // ---------------------------------------------------------------------------
  // Simulation
  // ---------------------------------------------------------------------------

  /**
   * Get current simulation state.
   * @returns Simulation state
   */
  getSimulationState(): SimulationState;

  /**
   * Start or resume the simulation.
   */
  startSimulation(): void;

  /**
   * Pause the simulation (preserves alpha).
   */
  pauseSimulation(): void;

  /**
   * Stop the simulation (resets alpha to 0).
   */
  stopSimulation(): void;

  /**
   * Restart the simulation from high energy.
   * @param alpha Initial alpha (default: 1.0)
   */
  restartSimulation(alpha?: number): void;

  /**
   * Run a single simulation step.
   */
  tickSimulation(): void;

  /**
   * Update simulation parameters.
   * @param config Partial configuration to merge
   */
  setSimulationConfig(config: Partial<ForceConfig>): void;

  /**
   * Set simulation alpha (energy level).
   * @param alpha Alpha value (0-1)
   */
  setAlpha(alpha: number): void;

  /**
   * Set alpha target for gradual decay.
   * @param target Target alpha (0-1)
   */
  setAlphaTarget(target: number): void;

  // ---------------------------------------------------------------------------
  // Viewport
  // ---------------------------------------------------------------------------

  /**
   * Get current viewport state.
   * @returns Viewport state
   */
  getViewport(): ViewportState;

  /**
   * Pan the viewport.
   * @param dx Delta X in graph units
   * @param dy Delta Y in graph units
   */
  pan(dx: number, dy: number): void;

  /**
   * Pan to center on a position.
   * @param x X coordinate in graph units
   * @param y Y coordinate in graph units
   * @param animate Animate the transition (default: false)
   */
  panTo(x: number, y: number, animate?: boolean): void;

  /**
   * Zoom the viewport.
   * @param factor Zoom factor (>1 = zoom in, <1 = zoom out)
   * @param centerX Zoom center X in screen pixels (default: canvas center)
   * @param centerY Zoom center Y in screen pixels (default: canvas center)
   */
  zoom(factor: number, centerX?: number, centerY?: number): void;

  /**
   * Set absolute zoom level.
   * @param scale Scale value
   * @param animate Animate the transition (default: false)
   */
  zoomTo(scale: number, animate?: boolean): void;

  /**
   * Fit all nodes in the viewport.
   * @param padding Padding around nodes in pixels (default: 50)
   * @param animate Animate the transition (default: false)
   */
  fitToView(padding?: number, animate?: boolean): void;

  /**
   * Fit specific nodes in the viewport.
   * @param nodeIds Node identifiers to fit
   * @param padding Padding in pixels (default: 50)
   * @param animate Animate the transition (default: false)
   */
  fitNodes(nodeIds: readonly NodeId[], padding?: number, animate?: boolean): void;

  /**
   * Convert screen coordinates to graph coordinates.
   * @param screenX X in screen pixels
   * @param screenY Y in screen pixels
   * @returns Position in graph units
   */
  screenToGraph(screenX: number, screenY: number): Vec2;

  /**
   * Convert graph coordinates to screen coordinates.
   * @param graphX X in graph units
   * @param graphY Y in graph units
   * @returns Position in screen pixels
   */
  graphToScreen(graphX: number, graphY: number): Vec2;

  // ---------------------------------------------------------------------------
  // Layers
  // ---------------------------------------------------------------------------

  /**
   * Get all layers.
   * @returns Array of layers in render order
   */
  getLayers(): readonly Layer[];

  /**
   * Get a layer by type.
   * @param type Layer type
   * @returns Layer or undefined
   */
  getLayer(type: LayerType): Layer | undefined;

  /**
   * Show a layer.
   * @param type Layer type
   */
  showLayer(type: LayerType): void;

  /**
   * Hide a layer.
   * @param type Layer type
   */
  hideLayer(type: LayerType): void;

  /**
   * Toggle layer visibility.
   * @param type Layer type
   * @returns New visibility state
   */
  toggleLayer(type: LayerType): boolean;

  /**
   * Configure a layer.
   * @param type Layer type
   * @param config Partial configuration to merge
   */
  setLayerConfig(type: LayerType, config: Partial<LayerConfig>): void;

  /**
   * Set layer render order.
   * @param type Layer type
   * @param order New order value (lower = behind)
   */
  setLayerOrder(type: LayerType, order: number): void;

  // ---------------------------------------------------------------------------
  // Hit Testing
  // ---------------------------------------------------------------------------

  /**
   * Find the node at a screen position.
   * @param screenX X in screen pixels
   * @param screenY Y in screen pixels
   * @returns NodeId or undefined if no node at position
   */
  getNodeAtPosition(screenX: number, screenY: number): NodeId | undefined;

  /**
   * Find the edge at a screen position.
   * @param screenX X in screen pixels
   * @param screenY Y in screen pixels
   * @param tolerance Hit tolerance in pixels (default: 5)
   * @returns EdgeId or undefined if no edge at position
   */
  getEdgeAtPosition(screenX: number, screenY: number, tolerance?: number): EdgeId | undefined;

  /**
   * Find all nodes in a screen rectangle.
   * @param x1 Top-left X in screen pixels
   * @param y1 Top-left Y in screen pixels
   * @param x2 Bottom-right X in screen pixels
   * @param y2 Bottom-right Y in screen pixels
   * @returns Array of NodeIds in the rectangle
   */
  getNodesInRect(x1: number, y1: number, x2: number, y2: number): readonly NodeId[];

  /**
   * Find the nearest node to a position.
   * @param screenX X in screen pixels
   * @param screenY Y in screen pixels
   * @param maxDistance Maximum distance in pixels (default: Infinity)
   * @returns NodeId or undefined if no node within maxDistance
   */
  getNearestNode(screenX: number, screenY: number, maxDistance?: number): NodeId | undefined;

  // ---------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------

  /**
   * Get current graph configuration.
   * @returns Graph config
   */
  getConfig(): GraphConfig;

  /**
   * Update graph configuration.
   * @param config Partial configuration to merge
   */
  setConfig(config: Partial<GraphConfig>): void;

  // ---------------------------------------------------------------------------
  // Events
  // ---------------------------------------------------------------------------

  /**
   * Subscribe to an event.
   * @param type Event type
   * @param handler Event handler function
   * @returns Unsubscribe function
   */
  on<K extends keyof EventMap>(type: K, handler: EventHandler<EventMap[K]>): () => void;

  /**
   * Subscribe to an event (one-time).
   * @param type Event type
   * @param handler Event handler function
   * @returns Unsubscribe function
   */
  once<K extends keyof EventMap>(type: K, handler: EventHandler<EventMap[K]>): () => void;

  /**
   * Unsubscribe from an event.
   * @param type Event type
   * @param handler Event handler function
   */
  off<K extends keyof EventMap>(type: K, handler: EventHandler<EventMap[K]>): void;

  // ---------------------------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------------------------

  /**
   * Get the bounding box of all nodes.
   * @returns Bounding box or undefined if no nodes
   */
  getBounds(): BoundingBox | undefined;

  /**
   * Get the bounding box of specific nodes.
   * @param nodeIds Node identifiers
   * @returns Bounding box or undefined if no valid nodes
   */
  getNodesBounds(nodeIds: readonly NodeId[]): BoundingBox | undefined;

  /**
   * Export current node positions.
   * @returns Float32Array of positions [x0, y0, x1, y1, ...]
   */
  exportPositions(): Float32Array;

  /**
   * Import node positions.
   * @param positions Float32Array of positions [x0, y0, x1, y1, ...]
   */
  importPositions(positions: Float32Array): void;

  /**
   * Take a screenshot of the current view.
   * @returns Promise resolving to a Blob (PNG image)
   */
  screenshot(): Promise<Blob>;
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Check if WebGPU is available and functional.
 * @returns Promise resolving to WebGPU status
 */
export declare function checkWebGPU(): Promise<WebGPUStatus>;

/**
 * Create a new HeroineGraph instance.
 * @param options Initialization options
 * @returns Promise resolving to HeroineGraph instance
 * @throws Error if WebGPU is not available
 */
export declare function createHeroineGraph(options: HeroineGraphOptions): Promise<HeroineGraph>;
