/**
 * Semantic LOD Module
 *
 * Machinery for the level-of-detail cut: what is visible at the current zoom,
 * and how a node gets there without popping.
 *
 * @module
 */

export { CrossfadeScheduler, NODE_ALPHA_OPAQUE, NODE_ALPHA_TRANSPARENT } from "./crossfade.ts";
export { commitNodeMass, NODE_MASS_HIDDEN, NODE_MASS_UNIT, rollUpMass } from "./mass.ts";
export type { MassDirtyRange } from "./mass.ts";
export {
  aggregateEdges,
  buildBundleInstances,
  BUNDLE_MAX_WIDTH_SCALE,
  decodeEdgeAggregation,
  EDGE_AGGREGATION_HEADER,
  EDGE_BUNDLE_STRIDE,
  EDGE_OPACITY_HIDDEN,
  EdgeOpacityMask,
} from "./edge_aggregation.ts";
export type {
  BundleInstances,
  BundleInstanceScratch,
  BundleStyle,
  EdgeAggregation,
  EdgeAggregator,
  EdgeOpacityHost,
} from "./edge_aggregation.ts";
export { DEFAULT_LOD_CONFIG, resolveLodConfig } from "./config.ts";
export type { LodConfig } from "./config.ts";
export { defaultLodPolicy } from "./policy.ts";
export type { LodCandidate, LodContext, LodDecision, LodPolicy } from "./policy.ts";
export {
  LOD_FORCE_CARD_PRIORITY,
  LOD_PAN_REEVALUATE_FRACTION,
  LOD_ZOOM_QUANTA_PER_OCTAVE,
  LODController,
  quantiseZoom,
} from "./controller.ts";
export type { LodHost } from "./controller.ts";
