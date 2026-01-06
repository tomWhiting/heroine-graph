/**
 * Contour Layer Module
 *
 * Provides contour line visualization for density fields.
 * Uses GPU-accelerated marching squares algorithm.
 *
 * @module
 */

// Configuration
export type { ContourConfig } from "./config.ts";
export {
  DEFAULT_CONTOUR_CONFIG,
  mergeContourConfig,
  validateContourConfig,
  parseColor,
} from "./config.ts";

// Pipeline
export type {
  ContourPipeline,
  ContourUniforms,
  LineUniforms,
  LineColorUniforms,
} from "./pipeline.ts";
export { createContourPipeline } from "./pipeline.ts";

// Layer
export type { ContourRenderContext } from "./layer.ts";
export { ContourLayer, createContourLayer } from "./layer.ts";
