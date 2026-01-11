/**
 * TypeScript Contracts: Advanced Visualization Layer System
 * Feature: 002-visualization-layers
 *
 * These interfaces define the public API for visualization features.
 */

// =============================================================================
// Color & Style Primitives
// =============================================================================

/** RGBA color as 4-element array [r, g, b, a] with values 0-1 */
export type RGBA = [number, number, number, number];

/** Color scale for mapping numeric values to colors */
export interface ColorScale {
  /** Input domain [min, max] */
  domain: [number, number];
  /** Output range as color strings or RGBA arrays */
  range: (string | RGBA)[];
}

// =============================================================================
// Diagnostic Channel System
// =============================================================================

/** Aggregation mode for hierarchical data */
export type AggregationType = 'sum' | 'max' | 'avg' | 'min';

/** Blend mode for combining multiple channels */
export type BlendMode = 'additive' | 'multiply' | 'overlay';

/** Configuration for defining a diagnostic channel */
export interface ChannelConfig {
  /** Unique identifier */
  id: string;
  /** Human-readable name */
  name?: string;
  /** Single color for visualization (use this or colorScale) */
  color?: RGBA;
  /** Color scale for value-based coloring (overrides color) */
  colorScale?: ColorScale;
  /** How to aggregate child values */
  aggregation: AggregationType;
  /** How to blend with other channels */
  blendMode?: BlendMode;
}

/** Data point for a channel */
export interface ChannelDataPoint {
  /** Node identifier */
  nodeId: string | number;
  /** Numeric value for this node */
  value: number;
}

// =============================================================================
// Multi-Layer System
// =============================================================================

/** Types of visualizations that can be enabled per layer */
export type VisualizationType = 'nodes' | 'edges' | 'heatmap' | 'contours' | 'metaballs';

/** Node data passed to filter functions */
export interface NodeData {
  id: string | number;
  type?: string;
  [key: string]: unknown;
}

/** Edge data passed to filter functions */
export interface EdgeData {
  id: string | number;
  source: string | number;
  target: string | number;
  type?: string;
  [key: string]: unknown;
}

/** Filter function for nodes */
export type NodeFilterFunction = (node: NodeData) => boolean;

/** Filter function for edges */
export type EdgeFilterFunction = (edge: EdgeData) => boolean;

/** Configuration for defining a visualization layer */
export interface LayerConfig {
  /** Unique identifier */
  id: string;
  /** Human-readable name */
  name?: string;
  /** Predicate to determine which nodes appear in this layer */
  nodeFilter: NodeFilterFunction;
  /** Optional predicate for edges (defaults to edges between visible nodes) */
  edgeFilter?: EdgeFilterFunction;
  /** What visualizations to render in this layer */
  visualizations: VisualizationType[];
  /** Render order (higher = on top) */
  zOrder: number;
  /** Initial visibility state */
  visible?: boolean;
}

/** Visibility state for all layers */
export type LayerVisibility = Record<string, boolean>;

// =============================================================================
// Type-Based Styling
// =============================================================================

/** Border configuration for nodes */
export interface BorderConfig {
  /** Whether border is rendered */
  enabled: boolean;
  /** Border thickness in pixels */
  thickness?: number;
  /** Border color */
  color?: RGBA;
}

/** Style configuration for a node type */
export interface NodeTypeStyle {
  /** Fill color */
  color?: RGBA;
  /** Node size in pixels */
  size?: number;
  /** Border configuration */
  border?: BorderConfig;
  /** Opacity multiplier */
  opacity?: number;
}

/** Style configuration for an edge type */
export interface EdgeTypeStyle {
  /** Edge color */
  color?: RGBA;
  /** Edge width in pixels */
  width?: number;
  /** Curvature amount (0 = straight) */
  curvature?: number;
  /** Opacity multiplier */
  opacity?: number;
}

/** Map of node type names to styles */
export type NodeTypeStyleMap = Record<string, NodeTypeStyle>;

/** Map of edge type names to styles */
export type EdgeTypeStyleMap = Record<string, EdgeTypeStyle>;

// =============================================================================
// Edge Flow Animation
// =============================================================================

/** Configuration for a single flow animation layer */
export interface FlowLayerConfig {
  /** Whether this flow layer is active */
  enabled: boolean;
  /** Animation speed (0.01-2.0, default 0.5) */
  speed?: number;
  /** Pulse width (0.005-0.99, default 0.15) */
  pulseWidth?: number;
  /** Number of pulses (1-8, default 3) */
  pulseCount?: number;
  /** Wave shape: 0=square, 0.5=triangle, 1.0=sine (default 1.0) */
  waveShape?: number;
  /** Brightness multiplier (1.0-5.0, default 1.5) */
  brightness?: number;
  /** Edge fade amount (0-1, default 0.5) */
  fade?: number;
  /** Tint color with alpha as blend amount */
  color?: RGBA;
}

/** Complete edge flow configuration */
export interface EdgeFlowConfig {
  /** Primary/base flow layer */
  layer1?: FlowLayerConfig;
  /** Secondary/spark flow layer (punches through layer1) */
  layer2?: FlowLayerConfig;
}

// =============================================================================
// Curved Edges
// =============================================================================

/** Configuration for curved edge rendering */
export interface CurvedEdgeConfig {
  /** Enable curved edges globally */
  curvedEdges?: boolean;
  /** Number of curve segments for tessellation (default 19) */
  curveSegments?: number;
  /** Rational curve weight (default 0.8) */
  curveWeight?: number;
  /** Control point distance 0-1 (default 0.5) */
  curveControlPointDistance?: number;
}

// =============================================================================
// Contour Configuration
// =============================================================================

/** Configuration for topographical contour rendering */
export interface ContourConfig {
  /** Whether contours are rendered */
  enabled: boolean;
  /** Density threshold levels for contour lines */
  thresholds?: number[];
  /** Color for contour lines */
  lineColor?: RGBA;
  /** Width of contour lines in pixels */
  lineThickness?: number;
}

// =============================================================================
// HeroineGraph API Extensions
// =============================================================================

/**
 * Extended HeroineGraph API with visualization features.
 * These methods extend the base HeroineGraph class.
 */
export interface VisualizationAPI {
  // Per-Item Styling
  setNodeColors(colors: Float32Array): void;
  setNodeSizes(sizes: Float32Array): void;
  setEdgeColors(colors: Float32Array): void;
  setEdgeWidths(widths: Float32Array): void;
  setEdgeCurvatures(curvatures: Float32Array): void;

  // Type-Based Styling
  setNodeTypeStyles(styles: NodeTypeStyleMap): void;
  setEdgeTypeStyles(styles: EdgeTypeStyleMap): void;

  // Node Border
  setNodeBorder(config: BorderConfig): void;

  // Diagnostic Channels
  defineChannel(config: ChannelConfig): void;
  setChannelData(channelId: string, data: ChannelDataPoint[]): void;
  removeChannel(channelId: string): void;

  // Multi-Layer System
  defineLayer(config: LayerConfig): void;
  setLayerVisible(layerId: string, visible: boolean): void;
  getLayerVisibility(): LayerVisibility;
  removeLayer(layerId: string): void;

  // Edge Flow Animation
  setEdgeFlow(config: EdgeFlowConfig): void;

  // Curved Edges
  setConfig(config: CurvedEdgeConfig): void;

  // Contours
  setContours(config: ContourConfig): void;
}
