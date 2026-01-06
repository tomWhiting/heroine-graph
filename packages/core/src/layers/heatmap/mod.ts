/**
 * Heatmap Layer Module
 *
 * Exports all components for the heatmap visualization layer.
 *
 * @module
 */

// Configuration
export type { HeatmapConfig } from "./config.ts";
export {
  DEFAULT_HEATMAP_CONFIG,
  mergeHeatmapConfig,
  validateHeatmapConfig,
} from "./config.ts";

// Color scales
export type {
  ColorRGBA,
  ColorStop,
  ColorScaleName,
  ColorScaleTexture,
} from "./colorscale.ts";
export {
  COLOR_SCALES,
  generateColorScaleData,
  createColorScaleTexture,
  createCustomColorScaleTexture,
  getColorScaleNames,
} from "./colorscale.ts";

// Density texture
export type { DensityTextureConfig, DensityTexture } from "./texture.ts";
export {
  DEFAULT_DENSITY_TEXTURE_CONFIG,
  createDensityTexture,
  clearDensityTexture,
} from "./texture.ts";

// Render pipeline
export type {
  HeatmapUniforms,
  ColormapUniforms,
  HeatmapPipeline,
} from "./pipeline.ts";
export {
  DEFAULT_HEATMAP_UNIFORMS,
  DEFAULT_COLORMAP_UNIFORMS,
  createHeatmapPipeline,
} from "./pipeline.ts";

// Layer
export type { Layer, HeatmapRenderContext } from "./layer.ts";
export { HeatmapLayer, createHeatmapLayer } from "./layer.ts";
