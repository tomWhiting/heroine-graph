/**
 * Heatmap Layer Module
 *
 * Exports all components for the heatmap visualization layer.
 *
 * @module
 */

// Configuration
export type { HeatmapConfig } from "./config.ts";
export { DEFAULT_HEATMAP_CONFIG, mergeHeatmapConfig, validateHeatmapConfig } from "./config.ts";

// Color scales
export type { ColorRGBA, ColorScaleName, ColorScaleTexture, ColorStop } from "./colorscale.ts";
export {
  COLOR_SCALES,
  createColorScaleTexture,
  createCustomColorScaleTexture,
  generateColorScaleData,
  getColorScaleNames,
} from "./colorscale.ts";

// Density texture
export type { DensityTexture, DensityTextureConfig } from "./texture.ts";
export {
  clearDensityTexture,
  createDensityTexture,
  DEFAULT_DENSITY_TEXTURE_CONFIG,
} from "./texture.ts";

// Render pipeline
export type { ColormapUniforms, HeatmapPipeline, HeatmapUniforms } from "./pipeline.ts";
export {
  createHeatmapPipeline,
  DEFAULT_COLORMAP_UNIFORMS,
  DEFAULT_HEATMAP_UNIFORMS,
} from "./pipeline.ts";

// Layer
export type { HeatmapRenderContext, Layer } from "./layer.ts";
export { createHeatmapLayer, HeatmapLayer } from "./layer.ts";
