/**
 * Contour Layer Module
 *
 * Provides contour line visualization for density fields.
 *
 * ⚠️  STATUS: NOT WORKING - needs investigation
 * The contour layer is currently broken. See simple-layer.ts for details.
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

// Simple Pipeline (new approach)
export type { SimpleContourPipeline, SimpleContourUniforms } from "./simple-pipeline.ts";
export { createSimpleContourPipeline } from "./simple-pipeline.ts";

// Simple Layer (new approach)
export type { SimpleContourRenderContext } from "./simple-layer.ts";
export {
  SimpleContourLayer,
  createSimpleContourLayer,
} from "./simple-layer.ts";

// Re-export with old names for backwards compatibility
export { SimpleContourLayer as ContourLayer } from "./simple-layer.ts";
export { createSimpleContourLayer as createContourLayer } from "./simple-layer.ts";
export type { SimpleContourRenderContext as ContourRenderContext } from "./simple-layer.ts";
