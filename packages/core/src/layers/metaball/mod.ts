/**
 * Metaball Layer Module
 *
 * Provides metaball visualization for organic cluster boundaries.
 * Uses screen-space SDF evaluation with smooth minimum blending.
 *
 * @module
 */

// Configuration
export type { MetaballConfig } from "./config.ts";
export {
  DEFAULT_METABALL_CONFIG,
  mergeMetaballConfig,
  validateMetaballConfig,
  parseMetaballColor,
} from "./config.ts";

// Pipeline
export type { MetaballPipeline, MetaballUniforms } from "./pipeline.ts";
export { createMetaballPipeline, DEFAULT_METABALL_UNIFORMS } from "./pipeline.ts";

// Layer
export type { MetaballRenderContext } from "./layer.ts";
export { MetaballLayer, createMetaballLayer } from "./layer.ts";
