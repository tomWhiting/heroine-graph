/**
 * The HUD's view model.
 *
 * Everything the readout shows is derived here from events and frame
 * timestamps, with no DOM and no clock of its own — `main.ts` passes the time
 * in. That makes the derivation testable, and it makes the two rates honest
 * about what they measure:
 *
 * - **fps** is presentation rate, counted over a fixed window rather than from
 *   the last frame, because a per-frame reciprocal is unreadable.
 * - **tick** is the wall-clock interval between `simulation:tick` events,
 *   smoothed. It is not GPU time: the library exposes no timestamp queries, so
 *   nothing here can claim to be one.
 *
 * @module
 */

// Typed against the module that declares it, not the barrel — see the note at
// the top of `repo.ts`.
import type { LodChangeEvent, LodTransitionReason } from "../../../packages/core/src/types.ts";

/** Everything the readout draws, in one immutable snapshot. */
export interface HudSnapshot {
  readonly nodes: number;
  readonly edges: number;
  /** Nodes the current LOD cut leaves on screen. */
  readonly visible: number;
  /** Nodes the cut does not draw at all. */
  readonly undrawn: number;
  /**
   * What accounts for most of them, or null while everything is drawn.
   *
   * The number alone cannot be acted on: a graph folded away by the camera is
   * working as intended, while the same number under `budget` says the ceiling
   * is what the user is looking at and raising it would show them more.
   */
  readonly undrawnReason: LodTransitionReason | null;
  /** Nodes currently standing in for a folded subtree. */
  readonly folded: number;
  /** DOM cards currently mounted. */
  readonly cards: number;
  /** Smoothed ms between simulation ticks; 0 before the first pair. */
  readonly tickMs: number;
  /** Frames per second over the last completed window; 0 before the first. */
  readonly fps: number;
}

/** How long an fps window runs before it is published, in ms. */
const FPS_WINDOW_MS = 500;

/**
 * Smoothing applied to the tick interval.
 *
 * Low enough that a single long frame does not dominate the readout, high
 * enough that a real slowdown shows up within a second.
 */
const TICK_SMOOTHING = 0.15;

/**
 * Accumulates HUD state from graph events and frame timestamps.
 *
 * All time arguments are milliseconds on one monotonic clock; the caller
 * chooses which (`performance.now()` in the browser, a counter in tests).
 */
export class HudModel {
  #nodes = 0;
  #edges = 0;
  #visible = 0;
  #undrawn = 0;
  #undrawnReason: LodTransitionReason | null = null;
  #folded = 0;
  #cards = 0;
  #tickMs = 0;
  #fps = 0;

  // Both baselines carry an explicit "have we got one" flag rather than
  // treating 0 as absent: `performance.now()` starts at 0, so the first frame
  // and the first tick of a page load land exactly on the sentinel.
  #lastTickAt = 0;
  #hasTickBaseline = false;
  #windowStartedAt = 0;
  #hasWindow = false;
  #framesInWindow = 0;

  /**
   * Adopt a freshly loaded graph.
   *
   * Resets the derived counters rather than carrying them: the fold count and
   * the visible cut describe the previous dataset, and a stale card count would
   * outlive the cards themselves.
   */
  loaded(nodes: number, edges: number): void {
    this.#nodes = nodes;
    this.#edges = edges;
    this.#visible = nodes;
    this.#undrawn = 0;
    this.#undrawnReason = null;
    this.#folded = 0;
    this.#cards = 0;
  }

  /**
   * Record a `lod:change` event.
   *
   * The undrawn total is taken from the event's own accounting rather than from
   * the node count this model was loaded with: under a streaming producer the
   * two disagree between a mutation and the cut that follows it, and the event
   * is the one that describes the cut it came from.
   */
  lodChanged(event: LodChangeEvent): void {
    this.#visible = event.visibleCount;
    this.#undrawn = event.totalCount - event.visibleCount;
    this.#undrawnReason = dominantReason(event.hiddenByReason);
  }

  /**
   * Record a `node:collapse` or `node:expand` event.
   *
   * The net of the two is how many nodes are currently standing in for a folded
   * subtree. It is counted from the per-node events rather than from the
   * `lod:change` arrays so the same transition is not counted twice.
   */
  nodeFolded(): void {
    this.#folded++;
  }

  nodeUnfolded(): void {
    if (this.#folded > 0) this.#folded--;
  }

  /** Adopt the card provider's own count of mounted cards. */
  cardsMounted(count: number): void {
    this.#cards = count;
  }

  /** Record one presented frame; publishes a new fps figure once per window. */
  frame(nowMs: number): void {
    if (!this.#hasWindow) {
      this.#hasWindow = true;
      this.#windowStartedAt = nowMs;
      return;
    }
    this.#framesInWindow++;
    const elapsed = nowMs - this.#windowStartedAt;
    if (elapsed < FPS_WINDOW_MS) return;
    this.#fps = (this.#framesInWindow * 1000) / elapsed;
    this.#framesInWindow = 0;
    this.#windowStartedAt = nowMs;
  }

  /** Record one simulation tick; the first only establishes the baseline. */
  tick(nowMs: number): void {
    if (this.#hasTickBaseline) {
      const interval = nowMs - this.#lastTickAt;
      this.#tickMs = this.#tickMs === 0
        ? interval
        : this.#tickMs + (interval - this.#tickMs) * TICK_SMOOTHING;
    }
    this.#lastTickAt = nowMs;
    this.#hasTickBaseline = true;
  }

  /**
   * Forget the tick baseline, so a pause does not show up as one enormous
   * interval when the simulation resumes.
   */
  simulationStopped(): void {
    this.#hasTickBaseline = false;
  }

  snapshot(): HudSnapshot {
    return {
      nodes: this.#nodes,
      edges: this.#edges,
      visible: this.#visible,
      undrawn: this.#undrawn,
      undrawnReason: this.#undrawnReason,
      folded: this.#folded,
      cards: this.#cards,
      tickMs: this.#tickMs,
      fps: this.#fps,
    };
  }
}

/**
 * The reason with the largest share of the undrawn nodes, or null when nothing
 * is undrawn.
 *
 * Ties go to the first reason in iteration order, which is stable for a record
 * the library builds the same way every time; a HUD showing one of two equal
 * halves has nothing to be right or wrong about.
 */
function dominantReason(
  hidden: Readonly<Record<LodTransitionReason, number>>,
): LodTransitionReason | null {
  let best: LodTransitionReason | null = null;
  let most = 0;
  for (const reason of Object.keys(hidden) as LodTransitionReason[]) {
    if (hidden[reason] > most) {
      most = hidden[reason];
      best = reason;
    }
  }
  return best;
}

/**
 * Compact count for a readout: 12 400 nodes is `12.4K`.
 *
 * Thousands separators change width as the number grows, which makes a HUD
 * jitter; a fixed one-decimal suffix does not.
 */
export function formatCount(value: number): string {
  const n = Math.round(value);
  if (n < 1_000) return String(n);
  if (n < 1_000_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${(n / 1_000_000).toFixed(2)}M`;
}

/** Milliseconds for a readout: sub-10ms keeps a decimal, above it does not. */
export function formatMs(value: number): string {
  if (value <= 0) return "—";
  return value < 10 ? `${value.toFixed(1)}ms` : `${Math.round(value)}ms`;
}
