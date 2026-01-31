/**
 * Metaball Layer Module
 *
 * Provides metaball visualization for organic cluster boundaries.
 * Uses screen-space SDF evaluation with smooth minimum blending.
 *
 * @module
 */

// Configuration
export type { MetaballConfig, MetaballDataSource } from "./config.ts";
export {
  DEFAULT_METABALL_CONFIG,
  mergeMetaballConfig,
  parseMetaballColor,
  validateMetaballConfig,
} from "./config.ts";

// Pipeline
export type { MetaballPipeline, MetaballUniforms } from "./pipeline.ts";
export { createMetaballPipeline, DEFAULT_METABALL_UNIFORMS } from "./pipeline.ts";

// Layer
export type { MetaballRenderContext } from "./layer.ts";
export { createMetaballLayer, MetaballLayer } from "./layer.ts";
