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

// Heatmap Layer
export * from "./heatmap/mod.ts";
