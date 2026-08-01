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
