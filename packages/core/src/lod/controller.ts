/**
 * Semantic-LOD Controller
 *
 * The state machine behind US1: which nodes the current camera should show,
 * which subtrees should stand in for themselves as a single bubble, and which
 * of the survivors are large enough to become real DOM cards.
 *
 * Pure CPU, fully deterministic, and driven entirely through {@link LodHost} —
 * no GPU handle, no DOM, no clock of its own. Every timestamp and every frame
 * boundary arrives as an argument, so a viewport sequence replays to the same
 * decision stream every time.
 *
 * Four properties are load-bearing and none of them is expressible in a type:
 *
 * 1. **A transition must never reheat the simulation.** Nothing here touches
 *    simulation alpha, and nothing here may: at the codebase's standard reheat
 *    a node travels ~25 node-widths over a 150 ms crossfade, which tears the
 *    fade away from the geometry it belongs to. The host seam deliberately
 *    exposes no way to do it.
 * 2. **Evaluation is on viewport change, debounced — never per frame.** The
 *    candidates are the interior nodes whose subtree extent crossed a band
 *    since the last evaluation, plus the declared focus set. Steady zooming
 *    therefore costs tens of policy calls, not thousands.
 * 3. **Hysteresis is necessary but not sufficient.** It answers zoom
 *    oscillation. It does not answer a subtree that moves or resizes under a
 *    hot simulation with the camera held still, which is what the quantised
 *    zoom band and {@link LodConfig.minBandCommitFrames} are for.
 * 4. **Hiding is deferred behind the fade.** The `HIDDEN_LOD` flag culls a node
 *    in the vertex stage, so writing it at transition time would delete the
 *    node the crossfade is fading. The flag lands when the ramp does.
 * 5. **Expanding is a rigid-body move, not a re-layout.** A folded subtree is
 *    frozen while its proxy goes on simulating, so it is translated by the
 *    proxy's drift before it is revealed. That is what lets a collapse be
 *    undone exactly, and it is why the proxy renders at its own position
 *    rather than at a centroid.
 *
 * @module
 */

import type {
  LodChangeEvent,
  LodTransitionReason,
  NodeCollapseEvent,
  NodeExpandEvent,
  NodeId,
  Vec2,
  ViewportState,
} from "../types.ts";
import { HIERARCHY_ROOT, type RetainedHierarchy } from "../graph/hierarchy.ts";
import type { CardSyncEntry } from "../overlay/overlay.ts";
import { CrossfadeScheduler } from "./crossfade.ts";
import { DEFAULT_LOD_CONFIG, type LodConfig, resolveLodConfig } from "./config.ts";
import { NODE_MASS_UNIT, rollUpMass } from "./mass.ts";
import { forEachSlotRun } from "./runs.ts";
import type { LodCandidate, LodContext, LodDecision, LodPolicy } from "./policy.ts";

/**
 * Quantisation steps per octave of zoom.
 *
 * Band state is keyed off a quantised zoom rather than the continuous one, so
 * that a camera wobbling inside one step produces no evaluation at all. Eight
 * steps is a ~9 % ratio between neighbouring quanta — finer than the 1.5×
 * default hysteresis band, so quantisation never swallows a real crossing, and
 * coarse enough that trackpad noise never leaves a step. Powers of two are
 * exact quanta, which keeps the common zoom levels free of rounding.
 */
export const LOD_ZOOM_QUANTA_PER_OCTAVE = 8;

/**
 * Fraction of the viewport extent the camera must pan before a policy is
 * re-consulted.
 *
 * Panning cannot change the geometric cut — the metrics are zoom-only — but it
 * changes {@link LodCandidate.onScreen}, which a policy may read. So the pan
 * trigger exists, and is skipped entirely when no policy is registered.
 */
export const LOD_PAN_REEVALUATE_FRACTION = 0.25;

/**
 * Priority added to a force-carded node so it outranks every geometric
 * candidate under the card budget.
 *
 * Larger than any plausible screen radius in pixels, so ordering *within* the
 * forced set still follows screen size.
 */
export const LOD_FORCE_CARD_PRIORITY = 1e6;

const REASON_ZOOM = 0;
const REASON_POLICY = 1;
const REASON_IMPERATIVE = 2;
const REASON_BUDGET = 3;

const REASON_NAMES: readonly LodTransitionReason[] = [
  "zoom",
  "policy",
  "imperative",
  "budget",
];

/** Sentinel commit frame for a band state that has never changed. */
const NEVER_COMMITTED = -0x40000000;

/**
 * Quantise a zoom level onto the band-state grid.
 *
 * Monotone in `zoom`, idempotent, and exact on powers of two. A non-positive
 * or non-finite zoom quantises to 0, which suppresses evaluation rather than
 * producing an infinite screen extent.
 */
export function quantiseZoom(zoom: number): number {
  if (!Number.isFinite(zoom) || zoom <= 0) return 0;
  const step = Math.round(Math.log2(zoom) * LOD_ZOOM_QUANTA_PER_OCTAVE);
  return 2 ** (step / LOD_ZOOM_QUANTA_PER_OCTAVE);
}

/**
 * Everything the controller needs from the graph, and nothing more.
 *
 * A seam rather than a direct dependency on `GraphMother`: the controller is a
 * pure state machine over per-node readings, so this keeps it unit-testable
 * with no GPU and no DOM. It is also the enforcement point for the no-reheat
 * rule — there is deliberately no way to reach simulation alpha through it.
 *
 * Per-node getters are called with GPU slot indices below the hierarchy's node
 * count and must tolerate a slot holding no live node.
 */
export interface LodHost {
  /** Retained containment hierarchy, or `null` when the graph has none. */
  getHierarchy(): RetainedHierarchy | null;
  /** Current camera. */
  getViewport(): ViewportState;
  /** Graph-space position of a slot, from the CPU position shadow. */
  getNodePosition(node: NodeId): Vec2;
  /** Graph-space render radius of a slot: the leaf metric's input. */
  getNodeRadius(node: NodeId): number;
  /** Producer-supplied tag index; 0 when the graph carries no tag column. */
  getNodeTag(node: NodeId): number;
  /** Producer-supplied weight in 0..1; 0 when the graph carries no weight column. */
  getNodeWeight(node: NodeId): number;
  /**
   * Apply `HIDDEN_LOD` across the slot range `[lo, hi)` from `visible`, as one
   * contiguous flag upload. `visible[slot]` is 1 for a slot that must render.
   */
  applyVisibility(lo: number, hi: number, visible: Uint8Array): void;
  /** Upload the scheduler's dirty alpha range. */
  uploadNodeAlpha(scheduler: CrossfadeScheduler): void;
  /**
   * Upload per-slot simulation mass covering `[0, mass.length)`.
   *
   * The array is the controller's, reused across transitions and valid only
   * for the duration of the call: a host that keeps it must copy.
   */
  uploadNodeMass(mass: Float32Array): void;
  /**
   * Declare the collapsed proxies: `proxies[k]` renders at radius `radii[k]`.
   *
   * Complete, not incremental — a slot that was a proxy and is absent here
   * returns to the radius it had before it became one. Both arrays are
   * controller-owned scratch, valid only for the duration of the call.
   */
  setCollapsedProxies(proxies: Uint32Array, radii: Float32Array): void;
  /**
   * Translate the positions of slots `[lo, hi)` by `(dx, dy)`.
   *
   * The expand fix-up, issued once per contiguous run of a subtree that is
   * about to be revealed. Must not pin and must not reheat: the caller is
   * restoring an arrangement, not moving a node.
   */
  translateNodeRange(lo: number, hi: number, dx: number, dy: number): void;
  /** Declare the current card set to the DOM overlay. */
  syncCards(entries: readonly CardSyncEntry[]): void;
  /** Emit one LOD event. */
  emit(event: LodChangeEvent | NodeCollapseEvent | NodeExpandEvent): void;
}

/**
 * The single candidate and context objects handed to the policy, overwritten
 * per call. Derived from the public shapes so the two cannot drift apart.
 */
type MutableCandidate = { -readonly [K in keyof LodCandidate]: LodCandidate[K] };
type MutableContext = { -readonly [K in keyof LodContext]: LodContext[K] };

/** Empty stand-in so every accessor is total before a graph is attached. */
const EMPTY_U8 = new Uint8Array(0);
const EMPTY_U32 = new Uint32Array(0);
const EMPTY_F32 = new Float32Array(0);

/** Empty transition diff, for a proxy re-declaration that changes no state. */
const NO_NODES: readonly NodeId[] = [];

export class LODController {
  readonly #host: LodHost;
  readonly #crossfade = new CrossfadeScheduler();

  #config: LodConfig = DEFAULT_LOD_CONFIG;
  #policy: LodPolicy | null = null;
  #focus: ReadonlySet<NodeId> = new Set();

  /** The hierarchy the per-node state is indexed against; identity-compared. */
  #hierarchy: RetainedHierarchy | null = null;
  #nodeCount = 0;

  // ---- Band state, one entry per slot ----
  #expanded = EMPTY_U8;
  #passThrough = EMPTY_U8;
  #forceCard = EMPTY_U8;
  #reason = EMPTY_U8;
  #commitFrame = new Int32Array(0);
  #tag = new Uint16Array(0);
  #weight = EMPTY_F32;

  // ---- Derived cut, double-buffered so a transition is a diff ----
  #visibleMask = EMPTY_U8;
  /**
   * Visibility as the host currently has it.
   *
   * Distinct from {@link LODController.#visibleMask} because hiding lags the
   * cut by a fade: a node leaving the cut stays flagged visible until its ramp
   * reaches zero, or the vertex-stage cull would delete it mid-crossfade.
   */
  #appliedVisible = EMPTY_U8;
  #collapsedMask = EMPTY_U8;
  #nextVisibleMask = EMPTY_U8;
  #nextCollapsedMask = EMPTY_U8;
  #collapseReason = EMPTY_U8;
  #cardedMask = EMPTY_U8;
  #visible = EMPTY_U32;
  #hasCut = false;

  // ---- Scratch, all sized to the slot space ----
  #stack = EMPTY_U32;
  #scratch = EMPTY_U32;
  #heap = EMPTY_U32;
  #heapSize = 0;
  #candidates = EMPTY_U32;
  #candidateStamp = new Int32Array(0);
  #stamp = 0;

  /** Interior slots sorted ascending by well radius, with their radii. */
  #interior = EMPTY_U32;
  #interiorRadii = EMPTY_F32;

  // ---- Proxy state, all reused across transitions ----
  /** Per-slot mass, rebuilt in place on every collapsed-set change. */
  #mass = EMPTY_F32;
  /** The collapsed set as an ascending slot list, and its well radii. */
  #proxies = EMPTY_U32;
  #proxyRadii = EMPTY_F32;
  /**
   * Where each proxy stood when it collapsed; `NaN` for a slot that is not one.
   *
   * The whole point of the expand fix-up: the proxy keeps simulating while the
   * subtree under it is frozen, so on expand the subtree is rigid-body
   * translated by how far the proxy drifted. Without it a subtree reappears at
   * an absolute position its parent left behind long ago.
   */
  #anchorX = EMPTY_F32;
  #anchorY = EMPTY_F32;
  /**
   * The host's proxy state has not been written since the hierarchy changed.
   *
   * A topology change resets mass and proxy radii on the host's side (they
   * name subtrees of the old slot mapping), so the first transition against a
   * rebuilt hierarchy must re-declare them even when its diff is empty.
   */
  #proxiesStale = false;

  #frame = 0;
  #zoom = 0;
  /** Clock of the last card sync, so a focus change can re-derive at that time. */
  #lastCardClockMs = 0;
  #lastZoom = 0;
  #lastPanX = 0;
  #lastPanY = 0;
  #viewportDirty = true;
  #forceFull = true;
  /** A decision was blocked by the committed lifetime during this evaluation. */
  #refused = false;
  /** Whether the evaluation in progress ignores the committed lifetime. */
  #bypass = false;
  /** A blocked decision is still outstanding and must be retried. */
  #retryPending = false;
  /**
   * The next evaluation ignores the committed lifetime.
   *
   * Set by the explicit acts — a configuration change, a policy change, a focus
   * change. The window exists to absorb noise in the metric, not to make the
   * host wait six frames for an answer it asked for directly.
   */
  #bypassCommit = false;

  /** Slots faded out and waiting for the ramp to land before being flagged. */
  readonly #pendingHide = new Set<NodeId>();
  /** Clock value each currently-carded node was first carded at. */
  readonly #cardedSince = new Map<NodeId, number>();

  readonly #candidate: MutableCandidate = {
    node: 0,
    depth: 0,
    childCount: 0,
    subtreeSize: 0,
    wellRadius: 0,
    screenRadius: 0,
    onScreen: false,
    tag: 0,
    weight: 0,
  };
  readonly #context: MutableContext = {
    zoom: 0,
    expand: 0,
    collapse: 0,
    dom: 0,
    focus: this.#focus,
  };

  constructor(host: LodHost) {
    this.#host = host;
  }

  /** Current configuration. */
  getConfig(): LodConfig {
    return this.#config;
  }

  /**
   * Merge a patch into the configuration.
   *
   * Disabling releases the cut immediately — every node returns to visible and
   * fully opaque — so `enabled: false` is a true off switch and not a freeze.
   */
  setConfig(patch: Partial<LodConfig>): void {
    const previous = this.#config;
    this.#config = resolveLodConfig(patch, previous);
    if (!this.#config.enabled) {
      if (previous.enabled) this.#release(true);
      return;
    }
    this.#forceFull = true;
    this.#bypassCommit = true;
  }

  /** Register the semantic policy, or `null` for geometry only. */
  setPolicy(policy: LodPolicy | null): void {
    this.#policy = policy;
    if (policy === null && this.#nodeCount > 0) {
      // Pass-through and force-card are sticky between band crossings, so a
      // policy's decisions would outlive the policy itself.
      this.#passThrough.fill(0);
      this.#forceCard.fill(0);
    }
    this.#forceFull = true;
    this.#bypassCommit = true;
  }

  /**
   * Declare the focus set: nodes that are carded regardless of screen size.
   *
   * Focus does not expand ancestors in v1 — a focus node hidden under a
   * collapsed proxy stays hidden, and {@link LODController.getVisibleAncestor}
   * reports the proxy standing in for it.
   */
  setFocus(nodes: Iterable<NodeId>): void {
    this.#focus = new Set(nodes);
    this.#context.focus = this.#focus;
    // Focus members are candidates in their own right, so a policy can act on a
    // search hit that is far too small to have crossed anything.
    this.#forceFull = true;
    this.#bypassCommit = true;
    this.evaluateNow(this.#lastCardClockMs);
  }

  /** Nodes in the cut, ascending. Every slot when LOD is off. */
  getVisibleNodes(): Uint32Array {
    return this.#visible.slice();
  }

  /**
   * The lowest ancestor of `node` that is in the cut, `node` itself when it is
   * visible, or -1 when nothing on its root path is.
   */
  getVisibleAncestor(node: NodeId): NodeId {
    if (!this.#hasCut) return node;
    const h = this.#hierarchy;
    if (h === null) return node;
    const { parent } = h.columns;
    let slot = node;
    while (slot >= 0 && slot < this.#nodeCount) {
      if (this.#visibleMask[slot] === 1) return slot;
      const p = parent[slot];
      if (p === HIERARCHY_ROOT) return -1;
      slot = p;
    }
    return -1;
  }

  /**
   * Whether a cut is live.
   *
   * False before the first evaluation and whenever LOD is off, which is what
   * distinguishes "everything is visible because nothing has been decided" from
   * "everything is visible because that is the cut".
   */
  get hasCut(): boolean {
    return this.#hasCut;
  }

  /** Whether a node is in the cut. */
  isVisible(node: NodeId): boolean {
    if (!this.#hasCut) return true;
    return node >= 0 && node < this.#nodeCount && this.#visibleMask[node] === 1;
  }

  /** Whether a node is the visible proxy for its subtree. */
  isCollapsed(node: NodeId): boolean {
    return this.#hasCut && node >= 0 && node < this.#nodeCount &&
      this.#collapsedMask[node] === 1;
  }

  /** The alpha scheduler this controller drives. */
  get crossfade(): CrossfadeScheduler {
    return this.#crossfade;
  }

  /** Frames counted so far. The unit `minBandCommitFrames` is measured in. */
  get frame(): number {
    return this.#frame;
  }

  /** Note that the camera changed; the next tick decides whether to evaluate. */
  viewportChanged(): void {
    this.#viewportDirty = true;
  }

  /**
   * Track a change in node-buffer capacity.
   *
   * The reallocated alpha buffer comes back fully opaque, so the scheduler's
   * shadow is rebuilt to match the current cut and re-uploaded whole. In-flight
   * ramps are snapped to their endpoints: a capacity change is a topology
   * change, not an LOD transition, and there is nothing coherent to interpolate
   * across it.
   *
   * The mass buffer is reallocated at unit mass for the same reason, so the
   * live collapsed set is re-declared rather than left to the next transition,
   * which may be many frames away.
   */
  handleNodeCapacityChange(capacity: number, nowMs: number): void {
    this.#crossfade.resize(capacity);
    this.#crossfade.reset();
    if (this.#hasCut) {
      this.#crossfade.fadeOut(this.#pendingHideOrHidden(), nowMs, 0);
    }
    this.#host.uploadNodeAlpha(this.#crossfade);

    if (this.#hasCut) {
      this.#proxiesStale = true;
      this.#onCollapsedSetChanged(NO_NODES, NO_NODES);
    }
  }

  /**
   * Expand a node immediately, whatever the geometry says.
   *
   * Imperative changes bypass the committed-lifetime guard — the user asked —
   * but they take a commit slot, so zoom cannot undo one on the next frame.
   */
  expandNode(node: NodeId, nowMs: number): void {
    this.#setBandState(node, true, REASON_IMPERATIVE, true);
    this.#recut(nowMs);
  }

  /** Collapse a node immediately, whatever the geometry says. */
  collapseNode(node: NodeId, nowMs: number): void {
    this.#setBandState(node, false, REASON_IMPERATIVE, true);
    this.#recut(nowMs);
  }

  /**
   * Advance one frame: run a pending evaluation, then drive the crossfade.
   *
   * @param nowMs - Caller's clock; never read here (see the module doc)
   * @returns whether anything visible changed, i.e. whether to redraw
   */
  tick(nowMs: number): boolean {
    this.#frame++;

    let changed = false;
    if (this.#shouldEvaluate()) {
      changed = this.evaluateNow(nowMs);
    }

    if (this.#crossfade.advance(nowMs)) {
      this.#host.uploadNodeAlpha(this.#crossfade);
      this.#settleHidden();
      // A card's opacity tracks the node it replaced, so it is re-declared for
      // as long as a ramp is in flight and never again after it lands.
      this.#syncCards(nowMs);
      changed = true;
    }

    return changed;
  }

  /**
   * Evaluate now, skipping the debounce.
   *
   * @returns whether the cut or the card set changed
   */
  evaluateNow(nowMs: number): boolean {
    if (!this.#config.enabled) return false;

    const hierarchy = this.#host.getHierarchy();
    if (hierarchy === null) {
      if (this.#hasCut) this.#release(false);
      return false;
    }
    if (hierarchy !== this.#hierarchy) this.#adopt(hierarchy);

    const viewport = this.#host.getViewport();
    const zoom = quantiseZoom(viewport.scale);
    if (zoom === 0) return false;

    const full = this.#forceFull;
    this.#forceFull = false;
    this.#viewportDirty = false;
    this.#zoom = zoom;

    this.#bypass = this.#bypassCommit;
    this.#bypassCommit = false;
    this.#refused = false;
    const bandChanged = this.#evaluateCandidates(zoom, viewport, full);
    // A decision blocked by the committed lifetime has not been resolved, only
    // deferred. Leaving `lastZoom` where it was keeps the same crossing range
    // in the candidate set, and `retryPending` keeps evaluating until the
    // window expires — otherwise a refusal would stick until the next zoom.
    this.#retryPending = this.#refused;
    if (!this.#refused) this.#lastZoom = zoom;
    this.#lastPanX = viewport.x;
    this.#lastPanY = viewport.y;

    // A full evaluation follows an explicit act that may have changed the cut
    // without any band state moving — clearing a policy's sticky verdicts, for
    // one — so it always rebuilds rather than trusting the diff.
    if (!bandChanged && !full && this.#hasCut) {
      // The cut is unchanged, but the DOM band is measured in the same zoom
      // that just moved, so the card set still has to be re-derived.
      this.#syncCards(nowMs);
      return false;
    }

    return this.#recut(nowMs);
  }

  // ==========================================================================
  // Attachment
  // ==========================================================================

  /** Adopt a new hierarchy: every slot-indexed structure is re-sized and reset. */
  #adopt(hierarchy: RetainedHierarchy): void {
    const n = hierarchy.nodeCount;
    this.#hierarchy = hierarchy;
    this.#nodeCount = n;

    this.#expanded = new Uint8Array(n);
    this.#passThrough = new Uint8Array(n);
    this.#forceCard = new Uint8Array(n);
    this.#reason = new Uint8Array(n);
    this.#commitFrame = new Int32Array(n).fill(NEVER_COMMITTED);
    this.#visibleMask = new Uint8Array(n);
    // The recreated flag buffer holds no HIDDEN_LOD bits, so the host starts
    // out showing everything and the shadow has to say so.
    this.#appliedVisible = new Uint8Array(n).fill(1);
    this.#collapsedMask = new Uint8Array(n);
    this.#nextVisibleMask = new Uint8Array(n);
    this.#nextCollapsedMask = new Uint8Array(n);
    this.#collapseReason = new Uint8Array(n);
    this.#cardedMask = new Uint8Array(n);
    this.#stack = new Uint32Array(n);
    this.#scratch = new Uint32Array(n);
    this.#heap = new Uint32Array(n);
    this.#candidates = new Uint32Array(n);
    this.#candidateStamp = new Int32Array(n).fill(-1);
    this.#stamp = 0;
    this.#visible = EMPTY_U32;
    this.#hasCut = false;
    this.#pendingHide.clear();
    this.#cardedSince.clear();

    this.#mass = new Float32Array(n);
    this.#proxies = new Uint32Array(n);
    this.#proxyRadii = new Float32Array(n);
    // Anchors from the previous hierarchy name slots the mutation may have
    // reassigned, so they are dropped rather than carried: a fix-up against a
    // stale anchor would translate a subtree by an arbitrary distance.
    this.#anchorX = new Float32Array(n).fill(NaN);
    this.#anchorY = new Float32Array(n).fill(NaN);
    this.#proxiesStale = true;

    // The semantic columns are producer data, fixed for the graph's lifetime,
    // so they are read once here rather than on every evaluation.
    this.#tag = new Uint16Array(n);
    this.#weight = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      this.#tag[i] = this.#host.getNodeTag(i);
      const w = this.#host.getNodeWeight(i);
      this.#weight[i] = Number.isFinite(w) ? Math.min(1, Math.max(0, w)) : 0;
    }

    this.#buildInteriorIndex(hierarchy);

    this.#crossfade.resize(n);
    this.#crossfade.reset();
    this.#host.uploadNodeAlpha(this.#crossfade);
    this.#forceFull = true;
  }

  /**
   * Index the interior slots by well radius.
   *
   * Only interior nodes can collapse or expand, and their screen extent is
   * monotone in zoom, so "everything that crossed a threshold between two zoom
   * levels" is a contiguous run of this array — two binary searches instead of
   * a scan over every node.
   */
  #buildInteriorIndex(hierarchy: RetainedHierarchy): void {
    const { offsets } = hierarchy.children;
    const { wellRadius } = hierarchy.columns;
    const n = hierarchy.nodeCount;

    const interior: number[] = [];
    for (let i = 0; i < n; i++) {
      if (offsets[i + 1] > offsets[i]) interior.push(i);
    }
    // Ties broken by slot so the index — and therefore every crossing range —
    // is a pure function of the hierarchy.
    interior.sort((a, b) => wellRadius[a] - wellRadius[b] || a - b);

    this.#interior = Uint32Array.from(interior);
    this.#interiorRadii = new Float32Array(interior.length);
    for (let k = 0; k < interior.length; k++) {
      this.#interiorRadii[k] = wellRadius[interior[k]];
    }
  }

  // ==========================================================================
  // Evaluation
  // ==========================================================================

  #shouldEvaluate(): boolean {
    if (!this.#config.enabled) return false;
    if (this.#forceFull || this.#retryPending) return true;
    if (!this.#viewportDirty) return false;

    const viewport = this.#host.getViewport();
    const zoom = quantiseZoom(viewport.scale);
    if (zoom !== this.#lastZoom) return true;

    // Only a policy can see onScreen, so with none registered a pan can change
    // nothing and the evaluation is skipped outright.
    if (this.#policy === null || zoom === 0) return false;

    const extentX = viewport.width / zoom;
    const extentY = viewport.height / zoom;
    return Math.abs(viewport.x - this.#lastPanX) > extentX * LOD_PAN_REEVALUATE_FRACTION ||
      Math.abs(viewport.y - this.#lastPanY) > extentY * LOD_PAN_REEVALUATE_FRACTION;
  }

  /**
   * Decide every candidate for this evaluation.
   *
   * @returns whether any band state changed
   */
  #evaluateCandidates(zoom: number, viewport: ViewportState, full: boolean): boolean {
    const count = this.#buildCandidates(zoom, full);
    if (count === 0) return false;

    const h = this.#hierarchy!;
    const { depth, subtreeSize, wellRadius } = h.columns;
    const { offsets } = h.children;
    const config = this.#config;
    const policy = this.#policy;

    const context = this.#context;
    context.zoom = zoom;
    context.expand = config.expandThreshold;
    context.collapse = config.collapseThreshold;
    context.dom = config.domThreshold;

    const halfW = viewport.width / (2 * zoom);
    const halfH = viewport.height / (2 * zoom);
    const candidate = this.#candidate;

    let changed = false;
    for (let k = 0; k < count; k++) {
      const slot = this.#candidates[k];
      const extent = wellRadius[slot] * zoom;

      let decision: LodDecision = "default";
      if (policy !== null) {
        const position = this.#host.getNodePosition(slot);
        candidate.node = slot;
        candidate.depth = depth[slot];
        candidate.childCount = offsets[slot + 1] - offsets[slot];
        candidate.subtreeSize = subtreeSize[slot];
        candidate.wellRadius = wellRadius[slot];
        candidate.screenRadius = extent;
        candidate.onScreen = Math.abs(position.x - viewport.x) <= halfW + wellRadius[slot] &&
          Math.abs(position.y - viewport.y) <= halfH + wellRadius[slot];
        candidate.tag = this.#tag[slot];
        candidate.weight = this.#weight[slot];
        decision = policy(candidate, context);
      }

      changed = this.#applyDecision(slot, decision, extent, offsets[slot + 1] - offsets[slot]) ||
        changed;
    }
    return changed;
  }

  /** Apply one policy verdict to one slot. @returns whether state changed */
  #applyDecision(
    slot: number,
    decision: LodDecision,
    extent: number,
    childCount: number,
  ): boolean {
    const config = this.#config;

    if (decision === "pass-through") {
      return this.#setPassThrough(slot, true);
    }

    let changed = this.#setPassThrough(slot, false);
    if (decision === "force-card") {
      this.#forceCard[slot] = 1;
    } else if (this.#forceCard[slot] === 1) {
      this.#forceCard[slot] = 0;
      changed = true;
    }

    if (decision === "expand" || decision === "collapse") {
      return this.#setBandState(slot, decision === "expand", REASON_POLICY, false) || changed;
    }

    // A leaf has nothing to expand, so its band state is meaningless and
    // flipping it would only churn the cut. Focus members reach here as leaves.
    if (childCount === 0) return changed;

    // Hysteresis: the threshold a node must reach depends on where it is. An
    // expanded node holds until its extent falls below the exit threshold; a
    // collapsed one waits for the entry threshold.
    const want = this.#expanded[slot] === 1
      ? extent >= config.collapseThreshold
      : extent >= config.expandThreshold;
    return this.#setBandState(slot, want, REASON_ZOOM, false) || changed;
  }

  /**
   * Change a node's expansion state, subject to the committed lifetime.
   *
   * A state change holds for `minBandCommitFrames` frames whatever the metric
   * does in the meantime. This is the primitive hysteresis cannot supply: the
   * metric itself moves when the simulation is hot, so a band can be crossed
   * back with the camera perfectly still.
   */
  #setBandState(
    slot: number,
    expanded: boolean,
    reason: number,
    force: boolean,
  ): boolean {
    if (slot < 0 || slot >= this.#nodeCount) return false;
    const next = expanded ? 1 : 0;
    if (this.#expanded[slot] === next) return false;
    if (
      !force && !this.#bypass &&
      this.#frame - this.#commitFrame[slot] < this.#config.minBandCommitFrames
    ) {
      this.#refused = true;
      return false;
    }
    this.#expanded[slot] = next;
    this.#reason[slot] = reason;
    this.#commitFrame[slot] = this.#frame;
    return true;
  }

  /** Pass-through is a band state too, and holds for the same lifetime. */
  #setPassThrough(slot: number, value: boolean): boolean {
    const next = value ? 1 : 0;
    if (this.#passThrough[slot] === next) return false;
    if (
      !this.#bypass &&
      this.#frame - this.#commitFrame[slot] < this.#config.minBandCommitFrames
    ) {
      this.#refused = true;
      return false;
    }
    this.#passThrough[slot] = next;
    this.#reason[slot] = REASON_POLICY;
    this.#commitFrame[slot] = this.#frame;
    return true;
  }

  /**
   * The candidate set for one evaluation.
   *
   * Two sources. The interior nodes whose screen extent crossed a threshold
   * since the last evaluation: extent is `wellRadius * zoom`, so crossing
   * threshold `T` between `zLo` and `zHi` means `wellRadius` lies in
   * `[T / zHi, T / zLo]` — a contiguous run of the radius-sorted interior
   * index rather than a scan over every node. And the focus set, always, so a
   * policy can act on a search hit that is far too small to cross anything.
   *
   * Crossing bounds are inclusive: a candidate that did not need deciding is
   * wasted work, a missing one is a wrong cut.
   */
  #buildCandidates(zoom: number, full: boolean): number {
    const stamp = ++this.#stamp;
    let count = 0;

    if (full || this.#lastZoom === 0) {
      for (const slot of this.#interior) {
        this.#candidateStamp[slot] = stamp;
        this.#candidates[count++] = slot;
      }
    } else if (this.#lastZoom !== zoom) {
      const zLo = Math.min(this.#lastZoom, zoom);
      const zHi = Math.max(this.#lastZoom, zoom);
      count = this.#collectCrossings(this.#config.collapseThreshold, zLo, zHi, stamp, count);
      count = this.#collectCrossings(this.#config.expandThreshold, zLo, zHi, stamp, count);
    }

    for (const slot of this.#focus) {
      if (slot < 0 || slot >= this.#nodeCount) continue;
      if (this.#candidateStamp[slot] === stamp) continue;
      this.#candidateStamp[slot] = stamp;
      this.#candidates[count++] = slot;
    }
    return count;
  }

  /** Append the crossings of one threshold to the candidate buffer. */
  #collectCrossings(
    threshold: number,
    zLo: number,
    zHi: number,
    stamp: number,
    start: number,
  ): number {
    const radii = this.#interiorRadii;
    const hi = threshold / zLo;
    let count = start;
    for (let k = lowerBound(radii, threshold / zHi); k < radii.length && radii[k] <= hi; k++) {
      const slot = this.#interior[k];
      if (this.#candidateStamp[slot] === stamp) continue;
      this.#candidateStamp[slot] = stamp;
      this.#candidates[count++] = slot;
    }
    return count;
  }

  // ==========================================================================
  // The cut
  // ==========================================================================

  /** Rebuild the cut and apply everything that follows from it. */
  #recut(nowMs: number): boolean {
    if (this.#hierarchy === null || !this.#config.enabled) return false;
    if (this.#zoom === 0) return false;

    this.#buildCut();
    return this.#applyTransition(nowMs);
  }

  /**
   * Walk the containment forest into a visible cut.
   *
   * Best-first on subtree extent, so when {@link LodConfig.maxVisibleNodes}
   * binds it is the smallest subtrees that stay folded — the budget degrades
   * the cut the same way zooming out does, rather than truncating it wherever
   * the traversal happened to be.
   */
  #buildCut(): void {
    const h = this.#hierarchy!;
    const { parent } = h.columns;
    const { offsets, children } = h.children;
    const n = h.nodeCount;
    const budget = this.#config.maxVisibleNodes;

    const visible = this.#nextVisibleMask;
    const collapsed = this.#nextCollapsedMask;
    visible.fill(0);
    collapsed.fill(0);
    this.#heapSize = 0;

    // Roots are admitted unconditionally: a forest with more roots than the
    // budget still has to render, and there is nothing above them to fold into.
    let seeds = 0;
    for (let i = 0; i < n; i++) {
      if (parent[i] === HIERARCHY_ROOT) this.#stack[seeds++] = i;
    }
    let visibleCount = this.#commitAdmitted(this.#gather(seeds), visible, collapsed);

    while (this.#heapSize > 0) {
      const slot = this.#heapPop();
      let seedCount = 0;
      for (let c = offsets[slot]; c < offsets[slot + 1]; c++) {
        this.#stack[seedCount++] = children[c];
      }
      const admitted = this.#gather(seedCount);
      if (visibleCount + admitted > budget) {
        collapsed[slot] = 1;
        this.#collapseReason[slot] = REASON_BUDGET;
        continue;
      }
      visibleCount += this.#commitAdmitted(admitted, visible, collapsed);
    }

    this.#visible = materialise(visible, visibleCount);
  }

  /**
   * Resolve seeds sitting in `#stack[0, seedCount)` into the nodes that
   * actually enter the cut, written to `#scratch`.
   *
   * A pass-through node is not one of them: it contributes nothing and its
   * children are considered at its parent's level instead, so a chain of them
   * disappears entirely. Iterative because those chains are exactly the deep
   * single-child directory runs the verb exists for.
   */
  #gather(seedCount: number): number {
    const { offsets, children } = this.#hierarchy!.children;
    let top = seedCount;
    let out = 0;
    while (top > 0) {
      const slot = this.#stack[--top];
      if (this.#passThrough[slot] === 1) {
        for (let c = offsets[slot]; c < offsets[slot + 1]; c++) {
          this.#stack[top++] = children[c];
        }
        continue;
      }
      this.#scratch[out++] = slot;
    }
    return out;
  }

  /** Mark gathered nodes visible, queueing the expandable ones. */
  #commitAdmitted(count: number, visible: Uint8Array, collapsed: Uint8Array): number {
    const { offsets } = this.#hierarchy!.children;
    for (let k = 0; k < count; k++) {
      const slot = this.#scratch[k];
      visible[slot] = 1;
      if (offsets[slot + 1] === offsets[slot]) continue;
      if (this.#expanded[slot] === 1) {
        this.#heapPush(slot);
      } else {
        collapsed[slot] = 1;
        this.#collapseReason[slot] = this.#reason[slot];
      }
    }
    return count;
  }

  /**
   * Diff the new cut against the live one and drive everything downstream:
   * flags, crossfade, cards, events.
   */
  #applyTransition(nowMs: number): boolean {
    const h = this.#hierarchy!;
    const { subtreeSize } = h.columns;
    const { offsets } = h.children;
    const n = h.nodeCount;
    const previousVisible = this.#visibleMask;
    const previousCollapsed = this.#collapsedMask;
    const visible = this.#nextVisibleMask;
    const collapsed = this.#nextCollapsedMask;
    const first = !this.#hasCut;

    const shown: NodeId[] = [];
    const hidden: NodeId[] = [];
    const entered: NodeId[] = [];
    const left: NodeId[] = [];
    let lo = n;
    let hi = 0;

    for (let i = 0; i < n; i++) {
      const wasVisible = first ? 1 : previousVisible[i];
      if (visible[i] !== wasVisible) {
        if (visible[i] === 1) shown.push(i);
        else hidden.push(i);
        if (i < lo) lo = i;
        hi = i + 1;
      }
      const wasCollapsed = first ? 0 : previousCollapsed[i];
      if (collapsed[i] === 1 && wasCollapsed === 0) entered.push(i);
      else if (collapsed[i] === 0 && wasCollapsed === 1 && visible[i] === 1) left.push(i);
    }

    this.#visibleMask = visible;
    this.#collapsedMask = collapsed;
    this.#nextVisibleMask = previousVisible;
    this.#nextCollapsedMask = previousCollapsed;
    this.#hasCut = true;

    // Ahead of the reveal below: an expanding subtree has to be translated
    // under its proxy *before* it is unflagged, or it is drawn for one frame
    // wherever it was left when the proxy took over.
    this.#onCollapsedSetChanged(entered, left);

    if (shown.length === 0 && hidden.length === 0 && entered.length === 0 && left.length === 0) {
      this.#syncCards(nowMs);
      return false;
    }

    // A node that reappears mid-fade must lose its HIDDEN_LOD flag now, or it
    // would fade in behind a vertex-stage cull and never be drawn. Nodes on
    // their way out keep theirs until #settleHidden lands the ramp.
    for (const slot of shown) {
      this.#pendingHide.delete(slot);
      this.#appliedVisible[slot] = 1;
    }
    for (const slot of hidden) this.#pendingHide.add(slot);
    if (hi > lo) this.#host.applyVisibility(lo, hi, this.#appliedVisible);

    this.#crossfade.fadeIn(shown, nowMs, this.#config.transitionMs);
    this.#crossfade.fadeOut(hidden, nowMs, this.#config.transitionMs);
    this.#host.uploadNodeAlpha(this.#crossfade);
    this.#settleHidden();

    for (const slot of entered) {
      this.#host.emit({
        type: "node:collapse",
        timestamp: nowMs,
        nodeId: slot,
        subtreeSize: subtreeSize[slot],
        reason: REASON_NAMES[this.#collapseReason[slot]],
      });
    }
    for (const slot of left) {
      this.#host.emit({
        type: "node:expand",
        timestamp: nowMs,
        nodeId: slot,
        childCount: offsets[slot + 1] - offsets[slot],
        reason: REASON_NAMES[this.#reason[slot]],
      });
    }
    this.#host.emit({
      type: "lod:change",
      timestamp: nowMs,
      expanded: left,
      collapsed: entered,
      visibleCount: this.#visible.length,
      zoom: this.#zoom,
    });

    this.#syncCards(nowMs);
    return true;
  }

  /**
   * Everything a change in the collapsed set implies, in the only order that
   * is coherent.
   *
   * 1. **Expand fix-up first.** A proxy keeps simulating while its subtree is
   *    frozen, so a subtree revealed without being moved reappears wherever its
   *    parent used to be. It runs before the anchors are rewritten because an
   *    expand can immediately re-collapse a child, and that child's anchor has
   *    to be its translated position.
   * 2. **Aggregate mass.** A proxy standing for 5 000 nodes must repel like
   *    5 000 nodes, or the visible layout contracts on collapse and re-inflates
   *    on expand — the one thing semantic LOD must not do (SC-002).
   * 3. **Proxy radii.** The collapsed parent renders through the ordinary node
   *    pipeline as one instance at its well radius, so the fold reads as a
   *    bubble containing what it hides rather than as a lone leaf.
   */
  #onCollapsedSetChanged(entered: readonly NodeId[], left: readonly NodeId[]): void {
    const h = this.#hierarchy;
    if (h === null) return;
    const stale = this.#proxiesStale;
    if (!stale && entered.length === 0 && left.length === 0) return;
    this.#proxiesStale = false;

    for (const slot of left) this.#restoreSubtree(slot);

    const count = this.#gatherProxies();
    const proxies = this.#proxies.subarray(0, count);
    const { wellRadius } = h.columns;
    for (let k = 0; k < count; k++) this.#proxyRadii[k] = wellRadius[proxies[k]];

    rollUpMass(h.columns.parent, h.children, proxies, this.#mass);
    this.#host.uploadNodeMass(this.#mass);
    this.#host.setCollapsedProxies(proxies, this.#proxyRadii.subarray(0, count));

    for (const slot of left) {
      this.#anchorX[slot] = NaN;
      this.#anchorY[slot] = NaN;
    }
    for (const slot of entered) {
      const position = this.#host.getNodePosition(slot);
      this.#anchorX[slot] = position.x;
      this.#anchorY[slot] = position.y;
    }
  }

  /** The collapsed set as an ascending slot list in `#proxies`. */
  #gatherProxies(): number {
    const collapsed = this.#collapsedMask;
    let count = 0;
    for (let i = 0; i < this.#nodeCount; i++) {
      if (collapsed[i] === 1) this.#proxies[count++] = i;
    }
    return count;
  }

  /**
   * Rigid-body translate `root`'s descendants by how far `root` drifted while
   * they were folded into it.
   *
   * Issued as one call per contiguous slot run, which is a single buffer write
   * for the whole subtree when the producer emits slots depth-first — the
   * ordering the hierarchy module asks producers for, and the reason it does.
   */
  #restoreSubtree(root: NodeId): void {
    const position = this.#host.getNodePosition(root);
    const dx = position.x - this.#anchorX[root];
    const dy = position.y - this.#anchorY[root];
    // A proxy that never moved needs no fix-up, and an un-anchored root — one
    // whose anchor the hierarchy it was collapsed under took with it — has
    // nothing coherent to be moved to.
    if (!Number.isFinite(dx) || !Number.isFinite(dy) || (dx === 0 && dy === 0)) return;

    const { offsets, children } = this.#hierarchy!.children;
    let top = 0;
    let count = 0;
    for (let c = offsets[root]; c < offsets[root + 1]; c++) this.#stack[top++] = children[c];
    while (top > 0) {
      const slot = this.#stack[--top];
      this.#scratch[count++] = slot;
      for (let c = offsets[slot]; c < offsets[slot + 1]; c++) this.#stack[top++] = children[c];
    }

    const slots = this.#scratch.subarray(0, count);
    slots.sort();
    forEachSlotRun(slots, count, (lo, hi) => this.#host.translateNodeRange(lo, hi, dx, dy));
  }

  /**
   * Flag the nodes whose fade-out has landed.
   *
   * Deferring the flag is what makes the crossfade visible at all: the render
   * pipeline discards on `HIDDEN_LOD` in the vertex stage, so a node flagged at
   * transition time disappears instantly instead of fading.
   */
  #settleHidden(): void {
    if (this.#pendingHide.size === 0) return;

    let lo = this.#nodeCount;
    let hi = 0;
    for (const slot of this.#pendingHide) {
      if (this.#crossfade.alphaOf(slot) !== 0) continue;
      this.#pendingHide.delete(slot);
      this.#appliedVisible[slot] = 0;
      if (slot < lo) lo = slot;
      if (slot >= hi) hi = slot + 1;
    }
    if (hi > lo) this.#host.applyVisibility(lo, hi, this.#appliedVisible);
  }

  // ==========================================================================
  // Cards
  // ==========================================================================

  /**
   * Derive and declare the card set.
   *
   * Three inputs, in precedence order: the focus set and any `force-card`
   * policy verdict, which ignore screen size entirely; the DOM band with its
   * own hysteresis; and the minimum card lifetime, which holds a card that has
   * just appeared even once it has shrunk back out of the band. Nodes still
   * outside the band but inside the prefetch ring are declared as prefetch
   * only, so a provider can start work before the swap without any DOM existing.
   */
  #syncCards(nowMs: number): void {
    this.#lastCardClockMs = nowMs;
    if (!this.#hasCut) return;

    const config = this.#config;
    const prefetchThreshold = config.domThreshold / config.prefetchRatio;
    const carded: CardSyncEntry[] = [];
    const prefetch: CardSyncEntry[] = [];

    for (const slot of this.#visible) {
      const radius = this.#host.getNodeRadius(slot) * this.#zoom;
      const forced = this.#forceCard[slot] === 1 || this.#focus.has(slot);
      const wasCarded = this.#cardedMask[slot] === 1;
      const since = this.#cardedSince.get(slot);
      const held = since !== undefined && nowMs - since < config.minCardLifetimeMs;
      const inBand = wasCarded ? radius >= config.domExitThreshold : radius >= config.domThreshold;
      const wanted = forced || inBand || (wasCarded && held);

      if (!wanted) {
        this.#cardedMask[slot] = 0;
        this.#cardedSince.delete(slot);
        if (radius >= prefetchThreshold) {
          prefetch.push({ node: slot, priority: radius, prefetchOnly: true });
        }
        continue;
      }

      this.#cardedMask[slot] = 1;
      if (since === undefined) this.#cardedSince.set(slot, nowMs);
      carded.push({
        node: slot,
        // Weight is a sub-pixel tie-break: it can only reorder nodes whose
        // screen radii are within one pixel of each other.
        priority: radius + this.#weight[slot] + (forced ? LOD_FORCE_CARD_PRIORITY : 0),
        opacity: this.#crossfade.alphaOf(slot),
      });
    }

    // Nodes that left the cut cannot hold a card, whatever their lifetime says.
    for (const slot of this.#cardedSince.keys()) {
      if (this.#visibleMask[slot] === 0) {
        this.#cardedSince.delete(slot);
        this.#cardedMask[slot] = 0;
      }
    }

    byPriority(carded);
    byPriority(prefetch);
    const entries = carded.slice(0, config.maxCards);
    for (const entry of prefetch.slice(0, config.maxCards)) entries.push(entry);
    this.#host.syncCards(entries);
  }

  // ==========================================================================
  // Release
  // ==========================================================================

  /**
   * Return every node to visible and opaque, and drop the cut.
   *
   * @param restore - Whether to unfold the proxies as an expand would, rather
   *   than merely forgetting them. False only when the graph has dropped the
   *   hierarchy the collapsed set was named against, where the descendant
   *   walk would follow a tree that no longer describes the slot space.
   */
  #release(restore: boolean): void {
    if (restore) {
      const count = this.#gatherProxies();
      for (let k = 0; k < count; k++) this.#restoreSubtree(this.#proxies[k]);
    }
    this.#collapsedMask.fill(0);
    this.#anchorX.fill(NaN);
    this.#anchorY.fill(NaN);
    if (this.#nodeCount > 0) {
      this.#mass.fill(NODE_MASS_UNIT);
      this.#host.uploadNodeMass(this.#mass);
    }
    this.#host.setCollapsedProxies(EMPTY_U32, EMPTY_F32);
    this.#proxiesStale = true;

    this.#visibleMask.fill(1);
    this.#appliedVisible.fill(1);
    this.#cardedMask.fill(0);
    if (this.#nodeCount > 0) {
      this.#host.applyVisibility(0, this.#nodeCount, this.#appliedVisible);
    }
    this.#visible = materialise(this.#visibleMask, this.#nodeCount);
    this.#pendingHide.clear();
    this.#cardedSince.clear();
    this.#crossfade.reset();
    this.#host.uploadNodeAlpha(this.#crossfade);
    this.#host.syncCards([]);
    this.#hasCut = false;
  }

  /** Slots currently hidden or on their way there, for an alpha rebuild. */
  *#pendingHideOrHidden(): Generator<number> {
    for (let i = 0; i < this.#nodeCount; i++) {
      if (this.#visibleMask[i] === 0) yield i;
    }
  }

  // ==========================================================================
  // Max-heap over subtree extent
  // ==========================================================================

  #heapPush(slot: number): void {
    const heap = this.#heap;
    let child = this.#heapSize++;
    heap[child] = slot;
    while (child > 0) {
      const parent = (child - 1) >> 1;
      if (!this.#outranks(heap[child], heap[parent])) break;
      const swap = heap[parent];
      heap[parent] = heap[child];
      heap[child] = swap;
      child = parent;
    }
  }

  #heapPop(): number {
    const heap = this.#heap;
    const top = heap[0];
    const last = --this.#heapSize;
    heap[0] = heap[last];
    let parent = 0;
    for (;;) {
      const left = parent * 2 + 1;
      if (left >= last) break;
      const right = left + 1;
      let best = left;
      if (right < last && this.#outranks(heap[right], heap[left])) best = right;
      if (!this.#outranks(heap[best], heap[parent])) break;
      const swap = heap[parent];
      heap[parent] = heap[best];
      heap[best] = swap;
      parent = best;
    }
    return top;
  }

  /**
   * Expansion order: widest subtree first, ties to the heavier node, then to
   * the lower slot. Total and antisymmetric, so the budget cuts at the same
   * place every run.
   */
  #outranks(a: number, b: number): boolean {
    const { wellRadius } = this.#hierarchy!.columns;
    if (wellRadius[a] !== wellRadius[b]) return wellRadius[a] > wellRadius[b];
    if (this.#weight[a] !== this.#weight[b]) return this.#weight[a] > this.#weight[b];
    return a < b;
  }
}

/**
 * The slots a visibility mask marks, ascending.
 *
 * `count` is how many the caller already knows are set, so the result is
 * exactly sized in one pass.
 */
function materialise(mask: Uint8Array, count: number): Uint32Array<ArrayBuffer> {
  const slots = new Uint32Array(count);
  let out = 0;
  for (let i = 0; i < mask.length && out < count; i++) {
    if (mask[i] === 1) slots[out++] = i;
  }
  return slots;
}

/** Index of the first element at or above `value`; `array` is ascending. */
function lowerBound(array: Float32Array, value: number): number {
  let lo = 0;
  let hi = array.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (array[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Descending by priority, ties to the lower slot so the order is total. */
function byPriority(entries: CardSyncEntry[]): void {
  entries.sort((a, b) => b.priority - a.priority || a.node - b.node);
}
