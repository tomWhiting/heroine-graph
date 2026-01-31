/**
 * Configuration Nexus
 *
 * Central access point for all configuration in heroine-graph.
 * Import from here to see all available settings in one place.
 *
 * Individual modules define their own defaults, but this file provides
 * a unified view for users who want to configure the graph as a whole.
 *
 * @module
 */

// =============================================================================
// Shared Types
// =============================================================================

/**
 * Data source for visualization layers (heatmap, contour, metaball).
 * - 'density': All nodes contribute equally (default)
 * - string: ID of a value stream - nodes contribute based on stream values
 */
export type LayerDataSource = "density" | string;

// =============================================================================
// Layer Configuration
// =============================================================================

// Heatmap
export {
  DEFAULT_HEATMAP_CONFIG,
  mergeHeatmapConfig,
  validateHeatmapConfig,
} from "../layers/heatmap/config.ts";
export type { HeatmapConfig, HeatmapDataSource } from "../layers/heatmap/config.ts";

// Contour
export {
  DEFAULT_CONTOUR_CONFIG,
  mergeContourConfig,
  validateContourConfig,
} from "../layers/contour/config.ts";
export type { ContourConfig, ContourDataSource } from "../layers/contour/config.ts";

// Metaball
export {
  DEFAULT_METABALL_CONFIG,
  mergeMetaballConfig,
  validateMetaballConfig,
} from "../layers/metaball/config.ts";
export type { MetaballConfig, MetaballDataSource } from "../layers/metaball/config.ts";

// Labels
export { DEFAULT_LABEL_CONFIG } from "../layers/labels/config.ts";
export type { LabelConfig } from "../layers/labels/config.ts";

// Layer Manager
export { DEFAULT_LAYER_MANAGER_CONFIG } from "../layers/manager.ts";
export type { LayerInfo, LayerManagerConfig } from "../layers/manager.ts";

// =============================================================================
// Render Configuration
// =============================================================================

// Node Pipeline
export { DEFAULT_NODE_PIPELINE_CONFIG } from "../renderer/pipelines/nodes.ts";
export type { NodePipelineConfig } from "../renderer/pipelines/nodes.ts";

// Edge Pipeline
export {
  DEFAULT_CURVED_EDGE_CONFIG,
  DEFAULT_EDGE_PIPELINE_CONFIG,
} from "../renderer/pipelines/edges.ts";
export type { CurvedEdgeConfig, EdgePipelineConfig } from "../renderer/pipelines/edges.ts";

// Edge Flow Animation
export {
  createEdgeFlowConfig,
  DEFAULT_EDGE_FLOW_CONFIG,
  DISABLED_FLOW_LAYER,
  EDGE_FLOW_PRESETS,
  getFlowPreset,
} from "../renderer/edge_flow.ts";
export type { EdgeFlowPreset } from "../renderer/edge_flow.ts";

// Node Border
export { DEFAULT_NODE_BORDER_CONFIG } from "./node_border.ts";
export type { NodeBorderConfig } from "./node_border.ts";

// Render Loop
export { DEFAULT_RENDER_LOOP_CONFIG } from "../renderer/render_loop.ts";
export type { RenderLoopConfig } from "../renderer/render_loop.ts";

// Commands
export { DEFAULT_CLEAR_COLOR } from "../renderer/commands.ts";
export type { ClearColor } from "../renderer/commands.ts";

// =============================================================================
// Simulation Configuration
// =============================================================================

// Force Configuration
export {
  DEFAULT_FORCE_CONFIG,
  FORCE_PRESETS,
  forceConfigBuilder,
  mergeForceConfig,
  validateForceConfig,
} from "../simulation/config.ts";
export type { ForceConfigBuilder, FullForceConfig } from "../simulation/config.ts";

// Simulation Controller
export {
  calculateAlphaDecay,
  DEFAULT_SIMULATION_CONFIG,
} from "../simulation/controller.ts";
export type { SimulationControllerConfig } from "../simulation/controller.ts";

// Alpha/Convergence
export { DEFAULT_ALPHA_CONFIG } from "../simulation/alpha.ts";
export type { AlphaConfig } from "../simulation/alpha.ts";

// =============================================================================
// Viewport Configuration
// =============================================================================

export { DEFAULT_VIEWPORT_CONFIG } from "../viewport/viewport.ts";
export type { ViewportConfig } from "../viewport/viewport.ts";

// =============================================================================
// Data Processing Configuration
// =============================================================================

// Graph Parser
export { DEFAULT_PARSER_CONFIG } from "../graph/parser.ts";
export type { ParserConfig } from "../graph/parser.ts";

// Typed Parser
export { DEFAULT_TYPED_PARSER_CONFIG } from "../graph/typed_parser.ts";
export type { TypedParserConfig } from "../graph/typed_parser.ts";

// Position Initialization
export { DEFAULT_INITIALIZE_CONFIG } from "../graph/initialize.ts";
export type { InitializationStrategy, InitializeConfig } from "../graph/initialize.ts";

// =============================================================================
// Interaction Configuration
// =============================================================================

export { DEFAULT_HIT_TESTER_CONFIG } from "../interaction/hit_test.ts";
export type { HitTesterConfig } from "../interaction/hit_test.ts";

// =============================================================================
// Buffer Configuration
// =============================================================================

export { DEFAULT_POSITION_BUFFER_CONFIG } from "../renderer/buffers/positions.ts";
export type { PositionBufferConfig } from "../renderer/buffers/positions.ts";

export { DEFAULT_EDGE_BUFFER_CONFIG } from "../renderer/buffers/edges.ts";
export type { EdgeBufferConfig } from "../renderer/buffers/edges.ts";

// =============================================================================
// Color Scales
// =============================================================================

export { COLOR_SCALES, getColorScaleNames } from "../layers/heatmap/colorscale.ts";
export type { ColorScaleName, ColorStop } from "../layers/heatmap/colorscale.ts";

// Value Stream Color Presets
export {
  createColorScaleFromPreset,
  createGradientScale,
  VALUE_COLOR_PRESETS,
} from "../streams/mod.ts";
export type { ValueColorScale } from "../streams/mod.ts";

// =============================================================================
// Utilities (re-exported for convenience)
// =============================================================================

export {
  colorToHex,
  DEFAULT_COLORS,
  lerpColor,
  NAMED_COLORS,
  parseColor,
  parseColorToRGB,
  parseColorToRGBA,
  withAlpha,
} from "../utils/mod.ts";
export type { ColorInput, ColorRGB, ColorRGBA, RgbaColor } from "../utils/mod.ts";
