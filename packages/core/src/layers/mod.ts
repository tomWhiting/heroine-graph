/**
 * Layers Module
 *
 * Provides an extensible layer system for graph visualization overlays.
 * Layers can render additional visualizations on top of the base graph.
 *
 * @module
 */

// Layer Manager
export type { LayerManagerConfig, LayerInfo } from "./manager.ts";
export {
  LayerManager,
  createLayerManager,
  DEFAULT_LAYER_MANAGER_CONFIG,
} from "./manager.ts";

// Layer Interface
export type { Layer } from "./heatmap/layer.ts";

// Heatmap Layer
export * from "./heatmap/mod.ts";

// Contour Layer
export * from "./contour/mod.ts";

// Metaball Layer
export * from "./metaball/mod.ts";

// Labels Layer
export * from "./labels/mod.ts";
