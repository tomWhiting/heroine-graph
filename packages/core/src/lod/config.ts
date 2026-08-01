/**
 * Semantic-LOD configuration.
 *
 * Four thresholds over two metrics. Collapse and expand read the **subtree**
 * screen extent (`wellRadius * zoom`); the DOM swap reads the **leaf** screen
 * radius against a fixed legible-card size. Each pair is an entry/exit band
 * rather than a single value, because a single value makes the state a
 * discontinuous function of the metric and any noise in the metric — zoom
 * jitter, a subtree still moving under a hot simulation — turns into flicker.
 *
 * The thresholds alone are not enough, which is why
 * {@link LodConfig.minBandCommitFrames} is here too: hysteresis answers zoom
 * oscillation, but a subtree that *resizes* while the camera is still can walk
 * across a band on its own.
 *
 * @module
 */

/**
 * Fraction of the entry threshold an inverted exit threshold is clamped to.
 *
 * A caller that sets an exit threshold at or above its entry threshold has
 * described a band with no interior, which leaves the state machine's
 * behaviour between the two undefined. Clamping keeps the ordering invariant
 * the controller relies on; the resulting band is degenerate but well-defined.
 */
const INVERTED_BAND_RATIO = 0.99;

/**
 * Semantic-LOD tuning. Every field is independent of the DOM overlay being
 * enabled: LOD collapses subtrees whether or not anything is carded.
 */
export interface LodConfig {
  /** Whether the controller evaluates at all. When false the cut is released. */
  readonly enabled: boolean;
  /** Subtree screen extent, in px, at or above which a node expands. */
  readonly expandThreshold: number;
  /** Subtree screen extent, in px, below which an expanded node collapses. */
  readonly collapseThreshold: number;
  /** Leaf screen radius, in px, at or above which a node is promoted to DOM. */
  readonly domThreshold: number;
  /** Leaf screen radius, in px, below which a carded node returns to a sprite. */
  readonly domExitThreshold: number;
  /**
   * How far outside the DOM band the prefetch ring reaches: a node is offered
   * to the provider as an abortable prefetch once its leaf screen radius
   * reaches `domThreshold / prefetchRatio`. Never below 1.
   */
  readonly prefetchRatio: number;
  /** Crossfade duration, in ms, for a node entering or leaving the cut. */
  readonly transitionMs: number;
  /** Hard ceiling on the size of the visible cut. */
  readonly maxVisibleNodes: number;
  /** Ceiling on simultaneously carded nodes. */
  readonly maxCards: number;
  /** Anti-flicker floor on how long a card stays once it exists, in ms. */
  readonly minCardLifetimeMs: number;
  /**
   * Frames a band state must hold before it may change again.
   *
   * The anti-oscillation primitive hysteresis cannot provide: it bounds how
   * fast a node's expansion state can change *regardless* of what drove the
   * change, so a subtree drifting or resizing under a hot simulation cannot
   * flap a band with the camera held still.
   */
  readonly minBandCommitFrames: number;
  /**
   * Whether edges crossing a collapsed boundary are bundled onto the visible
   * proxy. Carried here as the single LOD configuration surface; the
   * aggregation pass itself reads it. The controller does not act on it.
   */
  readonly edgeAggregation: boolean;
}

/** Defaults from the Phase 3 design. LOD is off until a caller enables it. */
export const DEFAULT_LOD_CONFIG: LodConfig = {
  enabled: false,
  expandThreshold: 96,
  collapseThreshold: 64,
  domThreshold: 48,
  domExitThreshold: 32,
  prefetchRatio: 2,
  transitionMs: 150,
  maxVisibleNodes: 4000,
  maxCards: 150,
  minCardLifetimeMs: 400,
  minBandCommitFrames: 6,
  edgeAggregation: true,
};

/** Coerce to a finite number at or above `min`, falling back on nonsense. */
function positive(value: number | undefined, fallback: number, min: number): number {
  if (value === undefined || !Number.isFinite(value)) return fallback;
  return value < min ? min : value;
}

/** Coerce to a non-negative integer, falling back on nonsense. */
function count(value: number | undefined, fallback: number, min: number): number {
  if (value === undefined || !Number.isFinite(value)) return fallback;
  return Math.max(min, Math.trunc(value));
}

/**
 * Merge a patch over a base configuration and enforce the invariants the
 * controller assumes.
 *
 * Validation clamps rather than throws. Every field here is a live tuning knob
 * — a slider in an inspector, a value interpolated by a consumer's own
 * animation — and a knob that throws mid-drag is worse than one that saturates.
 * The invariants enforced are the ones whose violation has no defined
 * behaviour: strictly ordered band edges, a prefetch ring outside the card
 * ring, and positive extents.
 */
export function resolveLodConfig(
  patch: Partial<LodConfig>,
  base: LodConfig = DEFAULT_LOD_CONFIG,
): LodConfig {
  const expandThreshold = positive(patch.expandThreshold, base.expandThreshold, Number.MIN_VALUE);
  const domThreshold = positive(patch.domThreshold, base.domThreshold, Number.MIN_VALUE);
  const collapseThreshold = Math.min(
    positive(patch.collapseThreshold, base.collapseThreshold, 0),
    expandThreshold * INVERTED_BAND_RATIO,
  );
  const domExitThreshold = Math.min(
    positive(patch.domExitThreshold, base.domExitThreshold, 0),
    domThreshold * INVERTED_BAND_RATIO,
  );

  return {
    enabled: patch.enabled ?? base.enabled,
    expandThreshold,
    collapseThreshold,
    domThreshold,
    domExitThreshold,
    prefetchRatio: positive(patch.prefetchRatio, base.prefetchRatio, 1),
    transitionMs: positive(patch.transitionMs, base.transitionMs, 0),
    maxVisibleNodes: count(patch.maxVisibleNodes, base.maxVisibleNodes, 1),
    maxCards: count(patch.maxCards, base.maxCards, 0),
    minCardLifetimeMs: positive(patch.minCardLifetimeMs, base.minCardLifetimeMs, 0),
    minBandCommitFrames: count(patch.minBandCommitFrames, base.minBandCommitFrames, 0),
    edgeAggregation: patch.edgeAggregation ?? base.edgeAggregation,
  };
}
