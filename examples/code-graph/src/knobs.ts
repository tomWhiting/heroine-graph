/**
 * The five knobs, and what each one means in library terms.
 *
 * The whole control surface of this demo is {@link KnobState}. Everything the
 * UI can change is one of these five fields, and every library call the UI
 * makes is a pure function of them — which is what makes the mapping testable
 * without a browser, and what keeps `main.ts` free of tuning decisions.
 *
 * @module
 */

// Typed against the modules that declare these, not the barrel — see the note
// at the top of `repo.ts`.
import type { LabelConfig } from "../../../packages/core/src/layers/labels/config.ts";
import type { LodConfig } from "../../../packages/core/src/lod/config.ts";
import type { RepoScaleId } from "./repo.ts";

/** The complete control surface. */
export interface KnobState {
  /** Which dataset size is loaded. */
  readonly scale: RepoScaleId;
  /** Whether semantic LOD evaluates at all. */
  readonly lodEnabled: boolean;
  /** Detail, 0 (coarse) to 1 (fine). Drives the expand/collapse band. */
  readonly detail: number;
  /** Ceiling on simultaneously mounted DOM cards. */
  readonly cardBudget: number;
  /** Ceiling on simultaneously drawn GPU labels. */
  readonly labelDensity: number;
  /** Whether the force simulation is advancing. */
  readonly running: boolean;
}

/** Where the demo starts. */
export const DEFAULT_KNOBS: KnobState = {
  scale: "small",
  lodEnabled: true,
  detail: 0.5,
  cardBudget: 60,
  labelDensity: 250,
  running: true,
};

/** Slider bounds, so the UI and the mapping cannot disagree about them. */
export const KNOB_RANGES = {
  detail: { min: 0, max: 1, step: 0.01 },
  cardBudget: { min: 0, max: 200, step: 5 },
  labelDensity: { min: 0, max: 1500, step: 25 },
} as const;

/**
 * Subtree screen extent, in px, at which a node expands — at detail 0 and at
 * detail 1 respectively.
 *
 * Detail runs backwards from the threshold it controls: a *high* expand
 * threshold means a subtree must be very large on screen before it unfolds,
 * which is the coarse end. Mapping the slider so that right means more detail
 * is the only reason this indirection exists.
 */
const EXPAND_AT_COARSE = 420;
const EXPAND_AT_FINE = 48;

/**
 * Collapse threshold as a fraction of the expand threshold.
 *
 * The gap between the two is the hysteresis band: a subtree that has just
 * expanded must shrink by a third before it folds again, so a camera hovering
 * at the threshold cannot flap it. One slider drives both because the *ratio*
 * is a physics decision and the *position* is a taste decision, and only the
 * second belongs to the user.
 */
const COLLAPSE_BAND = 0.66;

/** Linear interpolation, `t` clamped to 0..1. */
function lerp(from: number, to: number, t: number): number {
  const clamped = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
  return from + (to - from) * clamped;
}

/**
 * The LOD patch for a knob state.
 *
 * `maxCards` is here rather than on the overlay because `setLodConfig` forwards
 * it: the controller ranks and truncates the declared card set, the overlay
 * admits it, and two independently settable copies of one budget would disagree
 * silently.
 */
export function lodConfigFromKnobs(knobs: KnobState): Partial<LodConfig> {
  const expandThreshold = lerp(EXPAND_AT_COARSE, EXPAND_AT_FINE, knobs.detail);
  return {
    enabled: knobs.lodEnabled,
    expandThreshold,
    collapseThreshold: expandThreshold * COLLAPSE_BAND,
    maxCards: Math.max(0, Math.trunc(knobs.cardBudget)),
  };
}

/** The label-layer patch for a knob state. */
export function labelConfigFromKnobs(knobs: KnobState): Partial<LabelConfig> {
  return { maxLabels: Math.max(0, Math.trunc(knobs.labelDensity)) };
}

/**
 * How many slots to offer the label layer as candidates.
 *
 * The layer re-ranks its whole candidate set whenever the nodes move, so the
 * set is a per-frame cost and not merely a memory one. Holding a fixed multiple
 * of the visible budget keeps the ranking meaningful — there is always
 * something to promote when the camera moves — while bounding that cost by a
 * number the user can see.
 */
export const LABEL_CANDIDATE_RATIO = 20;

/** Hard ceiling on the candidate set, whatever the density knob asks for. */
export const LABEL_CANDIDATE_CEILING = 20_000;

/** Candidate-set size for a knob state. */
export function labelCandidateLimit(knobs: KnobState): number {
  const wanted = Math.max(0, Math.trunc(knobs.labelDensity)) * LABEL_CANDIDATE_RATIO;
  return Math.min(LABEL_CANDIDATE_CEILING, wanted);
}
