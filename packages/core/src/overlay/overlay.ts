/**
 * DOM Card Overlay
 *
 * The engine behind US2: a handful of nodes stop being useful as GPU sprites
 * at high zoom, so they are promoted to real DOM elements with selectable
 * text, working links and find-in-page.
 *
 * **Not a `Layer`.** `Layer.render(encoder, targetView)` is a GPU contract and
 * `LayerManager.removeLayer` destroys with GPU lifecycle semantics; an overlay
 * implementing `render()` as a no-op would be a lie. This is a peer of the
 * layer manager, owned by `GraphMother` and driven from the same two signals
 * the labels layer is driven from — viewport changes and the per-frame tick.
 *
 * **What it owns and what it does not.** It owns the container element, the
 * one camera transform written onto it per frame, which cards exist, where
 * they sit and how long they live. It owns no geometry *policy*: which nodes
 * deserve a card arrives through {@link DomCardOverlay.syncCards}, which the
 * LOD controller (WP-C) becomes the caller of. Every DOM lifecycle call goes
 * through {@link CardDriver}, so the provider contract — mount once, release
 * before detach, never touch card children — is enforced in exactly one place
 * and this file cannot weaken it. That is also where a provider that breaks the
 * contract is contained; what is left for this layer is to forget the card the
 * containment took and to stop offering that node to a provider which has
 * already failed for it.
 *
 * @module
 */

import type { NodeId, Vec2, ViewportState } from "../types.ts";
import type { IdLike } from "../graph/id_map.ts";
import { ErrorCode, GraphMotherError } from "../errors.ts";
import { CardDriver, cardFailureForfeitsCard } from "./driver.ts";
import type { CardPlacement, CardProviderFailure } from "./driver.ts";
import { CardContainerPool } from "./pool.ts";
import { cardCounterScale, cardPlacementAt, formatCssMatrix, overlayMatrix } from "./projection.ts";
import { createDefaultCardProvider } from "./default_card.ts";
import { DEFAULT_DOM_OVERLAY_CONFIG } from "./types.ts";
import type {
  CardNode,
  CardProvider,
  CardSize,
  CardStateChange,
  DomOverlayConfig,
} from "./types.ts";

/** Card box used when a sync entry names no size: legible, and roughly a sprite. */
export const DEFAULT_CARD_SIZE: CardSize = { width: 200, height: 120 };

/** Simultaneously mounted cards, when the caller sets no budget (FR-009). */
export const DEFAULT_MAX_CARDS = 150;

/** Anti-flicker floor on a card's lifetime in ms, when unset (FR-009). */
export const DEFAULT_MIN_CARD_LIFETIME_MS = 400;

/**
 * Quiet period that ends a viewport gesture, in ms.
 *
 * `Viewport` emits per wheel tick and has no gesture-end signal of its own
 * (FR-012), and giving it one would mean teaching it about input devices. The
 * overlay debounces instead, which is also where the answer is needed.
 */
export const DEFAULT_GESTURE_IDLE_MS = 120;

/**
 * The graph data cards read through.
 *
 * A seam rather than a direct dependency on `GraphMother`: the overlay needs
 * nine per-node readings and no GPU, so this keeps it testable without a
 * device and keeps the growing set of columns (WP-B's hierarchy, WP-C's
 * semantics) on the graph's side of the boundary.
 *
 * Every method is called with a GPU slot index and must tolerate a stale one —
 * a card outlives the frame its node was removed in.
 */
export interface CardNodeSource {
  /** Producer identifier for a slot, or `undefined` if it holds no live node. */
  externalId(node: NodeId): IdLike | undefined;
  /** Display label, when the producer supplied one. */
  label(node: NodeId): string | undefined;
  /** Producer-defined tag index; 0 when the graph carries no tags. */
  tag(node: NodeId): number;
  /** Producer-defined importance in 0..1; 0 when the graph carries no weights. */
  weight(node: NodeId): number;
  /** Containment depth; 0 when the graph carries no hierarchy. */
  depth(node: NodeId): number;
  /** Opaque producer content reference. Core never dereferences it. */
  contentRef(node: NodeId): string | undefined;
  /** Current graph-space position, from the existing position readback. */
  position(node: NodeId): Vec2;
  isPinned(node: NodeId): boolean;
  setPinned(node: NodeId, pinned: boolean): void;
}

/**
 * One node the caller wants carded (or prefetched) this evaluation.
 *
 * Carries no position in the ordinary case: {@link CardNodeSource.position} is
 * the single authority and is re-read every frame, so a sync entry cannot go
 * stale against the simulation between evaluations. The one exception is
 * {@link CardSyncEntry.at}, for a node the simulation is not moving at all.
 */
export interface CardSyncEntry {
  /** Node slot to card. */
  readonly node: NodeId;
  /**
   * Rank under the budget: higher survives. The caller's own measure —
   * screen size, focus, search relevance — core only compares it.
   */
  readonly priority: number;
  /**
   * Card box in CSS pixels, read once when the card mounts and fixed from then
   * on — the card is laid out in it for its whole life, so a provider's content
   * never reflows under the user's pointer. What changes with the camera is the
   * size the box is *drawn* at, and only until it reaches this one; see
   * `cardCounterScale`.
   */
  readonly size?: CardSize;
  /** Crossfade opacity, 0..1. Defaults to fully opaque. */
  readonly opacity?: number;
  /**
   * Where to draw this card, overriding the node's own graph position.
   *
   * For a card the caller is holding open over a node the layout has folded
   * away. Such a node is frozen: its stored position is where it stood when the
   * fold began, and the fold it is inside has gone on moving without it, so the
   * card would drift away from the thing that swallowed it. The caller knows
   * that offset and supplies the corrected point here.
   *
   * Re-read on every sync, and the only thing that can move a card between
   * syncs, so a stale override cannot outlive the entry that set it. It does
   * not change {@link CardNode.position}, which goes on answering with the
   * node's true graph position — the card is displaced, the node is not.
   */
  readonly at?: Vec2;
  /**
   * Node is in the prefetch ring but not carded: the provider is offered an
   * advisory, abortable `prefetch` and no DOM is created.
   */
  readonly prefetchOnly?: boolean;
}

/**
 * Attribute a provider puts on one of its own elements to make that element
 * drag the card's node.
 *
 * Opt-in, and read rather than written: core never creates a handle, because
 * everything inside a card container belongs to the provider (SC-004). It also
 * has to be opt-in for a second reason — a card whose whole surface dragged
 * the node would have no selectable text, which is most of why cards exist.
 */
export const CARD_DRAG_HANDLE_ATTRIBUTE = "data-graphmother-drag";

/**
 * Attribute a provider puts on an element that owns its own wheel.
 *
 * A wheel over a card is forwarded to the canvas, because a user zooming back
 * out has the pointer over a card by construction and the canvas is the only
 * thing listening. That is right for a card the size of a caption and wrong
 * for one holding a scrollable control: an editor whose every wheel event is
 * `preventDefault`ed and handed to the camera cannot scroll at all.
 *
 * A wheel inside a subtree carrying this attribute is left entirely alone —
 * not forwarded, not prevented — so the browser scrolls the element natively.
 * Opt-in per element rather than per card, so a card can hold a scrolling
 * region and still zoom from its chrome, and read rather than written for the
 * same reason as {@link CARD_DRAG_HANDLE_ATTRIBUTE}: everything inside a card
 * container belongs to the provider.
 *
 * A region that has scrolled to its own end still keeps the event. Chaining
 * back to the camera at the boundary is what `overscroll-behavior` is for, and
 * the provider owns that CSS.
 */
export const CARD_SCROLL_ATTRIBUTE = "data-graphmother-scroll";

/**
 * Where a card drag is routed.
 *
 * The overlay resolves the gesture and converts it to graph units; what a
 * dragged node *does* — pin, position write, `node:drag*` events — belongs to
 * the graph, and is the same path a node dragged on the canvas takes.
 */
export interface CardDragSink {
  /** Gesture started on `node`, which is at `position`. */
  begin(node: NodeId, position: Vec2): void;
  /** Pointer moved: `node` should now be at `position`. */
  move(node: NodeId, position: Vec2): void;
  /** Gesture finished with `node` at `position`. The node stays pinned. */
  end(node: NodeId, position: Vec2): void;
}

/**
 * The two timer functions the gesture debounce needs.
 *
 * Injected so a test can end a gesture without waiting on wall-clock time;
 * defaults to the host's timers.
 */
export interface OverlayTimers {
  setTimeout(handler: () => void, ms: number): number;
  clearTimeout(handle: number): void;
}

/**
 * The host's timers.
 *
 * `setTimeout` returns a number in a browser and in Deno, but is *typed* as
 * returning an opaque handle wherever Node's declarations are also in scope,
 * so the handle is narrowed here rather than at every call site.
 */
const HOST_TIMERS: OverlayTimers = {
  setTimeout: (handler, ms) => Number(setTimeout(handler, ms)),
  clearTimeout: (handle) => clearTimeout(handle),
};

export interface DomCardOverlayOptions {
  /** Canvas the graph renders into: default host parent, and wheel target. */
  readonly canvas: HTMLCanvasElement;
  /** Current camera. Read on every frame and on every sync. */
  readonly viewport: () => ViewportState;
  /** Graph data the cards read through. */
  readonly nodes: CardNodeSource;
  /**
   * Where a drag on a card's handle is routed. Omit and cards do not drag
   * their nodes; the handle attribute is then inert.
   */
  readonly drag?: CardDragSink;
  /**
   * Initial card budget. Defaults to {@link DEFAULT_MAX_CARDS}, and is a
   * {@link DomOverlayConfig} field from then on — `setLodConfig` keeps it
   * equal to the controller's.
   */
  readonly maxCards?: number;
  /**
   * Initial anti-flicker lifetime floor. Defaults to
   * {@link DEFAULT_MIN_CARD_LIFETIME_MS}, and is a {@link DomOverlayConfig}
   * field from then on.
   */
  readonly minCardLifetimeMs?: number;
  /** Gesture debounce. Defaults to {@link DEFAULT_GESTURE_IDLE_MS}. */
  readonly gestureIdleMs?: number;
  /** Clock, injected so card lifetimes replay identically in a test. */
  readonly now?: () => number;
  /** Timers, injected so the gesture debounce is drivable without real time. */
  readonly timers?: OverlayTimers;
  /**
   * Called once per contained provider failure, after the overlay has dropped
   * whatever the failure cost. Omit it and {@link CardDriver} logs instead.
   */
  readonly onCardFailure?: (failure: CardProviderFailure) => void;
}

/** One mounted card's core-owned state. */
interface CardRecord {
  readonly card: CardNode;
  /** Clock value at mount; the minimum-lifetime floor is measured from it. */
  readonly mountedAtMs: number;
  /** Card layout box in CSS pixels, fixed for the card's lifetime. */
  readonly width: number;
  readonly height: number;
  /**
   * Camera scale at mount — the scale at which the card is rendered at its
   * natural size, and the ceiling on how large it is allowed to be drawn. See
   * {@link cardCounterScale}.
   */
  readonly mountScale: number;
  /**
   * Producer identifier the card mounted for.
   *
   * Cards are keyed by slot, and a batch removal compacts slots — so a
   * surviving node can move into a slot that is already carded. Every
   * {@link CardNode} accessor reads through to whatever the source says the
   * slot holds now, so without this the card would go on standing for a node
   * that is no longer there, showing one node's content against another's
   * position, and the provider would have no way to find out.
   */
  readonly externalId: IdLike | undefined;
  opacity: number;
  priority: number;
  /** Placement override from the last sync entry; see {@link CardSyncEntry.at}. */
  at: Vec2 | undefined;
}

/** A node competing for a place under the budget. */
interface Candidate {
  readonly node: NodeId;
  readonly priority: number;
  readonly mounted: boolean;
}

/** A card drag in flight. */
interface CardDrag {
  readonly node: NodeId;
  readonly pointerId: number;
  /** Where the pointer went down, in client pixels. */
  readonly clientX: number;
  readonly clientY: number;
  /** Where the node was then, in graph units. */
  readonly originX: number;
  readonly originY: number;
  /** Element holding the pointer capture, if the host granted one. */
  readonly captured: Element | null;
  /** Last position handed to the sink, so the end call is consistent with it. */
  position: Vec2;
}

export class DomCardOverlay {
  readonly #canvas: HTMLCanvasElement;
  readonly #viewport: () => ViewportState;
  readonly #nodes: CardNodeSource;
  readonly #dragSink: CardDragSink | null;
  readonly #gestureIdleMs: number;
  readonly #now: () => number;
  readonly #timers: OverlayTimers;
  readonly #onCardFailure: ((failure: CardProviderFailure) => void) | null;

  #config: DomOverlayConfig = { ...DEFAULT_DOM_OVERLAY_CONFIG };
  #provider: CardProvider<unknown> | null = null;

  #container: HTMLElement | null = null;
  #pool: CardContainerPool | null = null;
  #driver: CardDriver<unknown> | null = null;

  /** Mounted cards, in mount order. */
  readonly #cards = new Map<NodeId, CardRecord>();
  /**
   * `CardNode` views, kept alive from the first prefetch offer until the node
   * leaves the ring, so prefetch and mount see one stable object per node as
   * the contract promises.
   */
  readonly #views = new Map<NodeId, CardNode>();
  /** Nodes in the prefetch ring as of the last sync. */
  #ring: ReadonlySet<NodeId> = new Set();
  /**
   * Slots whose provider failed, against the producer id each held when it did.
   *
   * Without the quarantine the caller's next sync asks for the same card, the
   * provider throws again, and a broken provider costs one failure per node per
   * evaluation forever. A card is a piece of consumer code that has just
   * demonstrated it does not work for this node; retrying it is not resilience.
   *
   * The producer id is what makes it a quarantine on a *node* rather than on a
   * slot. Slots are compacted by a batch removal, so the slot one node's card
   * failed in comes to hold a different node an evaluation later — and holding
   * that node against its predecessor's defect is a hole in the graph with no
   * event, no diagnostic, and no way back short of the two global acts
   * (`setProvider`, a reload) a consumer has no reason to perform. Checked on
   * every read, so the lift needs no mutation event to hang off.
   */
  readonly #failed = new Map<NodeId, IdLike>();

  /** Last transform written, so an unchanged camera costs no DOM write. */
  #transform = "";
  #gestureHandle: number | null = null;
  #gestureActive = false;
  #drag: CardDrag | null = null;
  #cardEpoch = 0;

  constructor(options: DomCardOverlayOptions) {
    this.#canvas = options.canvas;
    this.#viewport = options.viewport;
    this.#nodes = options.nodes;
    this.#dragSink = options.drag ?? null;
    this.#config = clampBudget({
      ...DEFAULT_DOM_OVERLAY_CONFIG,
      maxCards: options.maxCards ?? DEFAULT_MAX_CARDS,
      minCardLifetimeMs: options.minCardLifetimeMs ?? DEFAULT_MIN_CARD_LIFETIME_MS,
    });
    this.#gestureIdleMs = Math.max(0, options.gestureIdleMs ?? DEFAULT_GESTURE_IDLE_MS);
    this.#now = options.now ?? (() => performance.now());
    this.#timers = options.timers ?? HOST_TIMERS;
    this.#onCardFailure = options.onCardFailure ?? null;
  }

  /** Current settings. */
  get config(): DomOverlayConfig {
    return this.#config;
  }

  /** Mounted cards. */
  get cardCount(): number {
    return this.#cards.size;
  }

  /**
   * Counter bumped whenever a card mounts or is released.
   *
   * Exists for the label path, which must suppress the GPU label under a card
   * and needs to know when to re-cull rather than asking every frame.
   */
  get cardEpoch(): number {
    return this.#cardEpoch;
  }

  /** Whether a node currently holds a mounted card. */
  hasCard(node: NodeId): boolean {
    return this.#cards.has(node);
  }

  /**
   * Nodes the current provider is no longer offered, because it threw for them.
   *
   * Read-only, and read by tests and inspectors rather than by the overlay's
   * own callers: a consumer acts on `card:error`, not on this set. Slots whose
   * occupant has changed since the failure are already gone from it — the
   * quarantine follows the node, not the slot index.
   */
  get failedCards(): ReadonlySet<NodeId> {
    this.#sweepQuarantine();
    return new Set(this.#failed.keys());
  }

  /** The node being dragged by its card handle, or `null`. */
  get draggingNode(): NodeId | null {
    return this.#drag?.node ?? null;
  }

  /**
   * The container element, or `null` while the overlay is disabled.
   *
   * Everything on it is core-owned: the camera transform, the click-through
   * discipline and the compositing hint.
   */
  get element(): HTMLElement | null {
    return this.#container;
  }

  /** Whether a viewport gesture is in flight (i.e. the debounce is armed). */
  get gestureActive(): boolean {
    return this.#gestureActive;
  }

  /**
   * Apply settings, creating or tearing down the container as `enabled`
   * changes.
   *
   * @throws if enabling with no host and no `canvas.parentElement` to fall
   * back to — an overlay with nowhere to put its cards is a caller error, not
   * a state worth carrying silently.
   */
  setConfig(config: Partial<DomOverlayConfig>): void {
    const previous = this.#config;
    const next = clampBudget({ ...previous, ...config });
    this.#config = next;

    const hostChanged = next.host !== previous.host;
    const classChanged = next.className !== previous.className;

    if (!next.enabled) {
      this.#teardown();
      return;
    }
    // A new host or class name has to reach the containers, and both are
    // applied at creation — so rebuild rather than patch.
    if (this.#container !== null && (hostChanged || classChanged)) {
      this.#teardown();
    }
    if (this.#container === null) this.#build(next);
  }

  /**
   * Register the renderer for card content. `null` restores the built-in one.
   *
   * Swapping providers releases every mounted card first: the outgoing
   * provider's `release` must run against its own state, and the incoming one
   * cannot adopt DOM it did not create.
   *
   * It also lifts the quarantine on nodes the outgoing provider failed for. A
   * new provider is a new answer to the question the old one got wrong, so it
   * is the documented way back from a failure — and holding nodes against it
   * for a predecessor's defect would be the library keeping a grudge.
   */
  setProvider(provider: CardProvider<unknown> | null): void {
    this.#provider = provider;
    // Also cleared by `releaseAll` below, but that path exists only once the
    // overlay has been enabled, and a provider may be registered before then.
    this.#failed.clear();
    const container = this.#container;
    const pool = this.#pool;
    if (container === null || pool === null) return;
    this.releaseAll();
    this.#driver = this.#createDriver(container, pool, this.#config);
  }

  /**
   * Declare the set of nodes that should be carded, and the ring around it.
   *
   * The caller decides *which* nodes; the overlay decides how many survive the
   * budget and how long they stay. Nodes absent from `entries` are released
   * once they are older than the minimum lifetime.
   *
   * Called on LOD evaluations, not per frame — this is the debounced,
   * band-crossing path, and {@link syncFrame} is the cheap one.
   */
  syncCards(entries: readonly CardSyncEntry[]): void {
    const driver = this.#driver;
    if (driver === null) return;

    const now = this.#now();
    const scale = this.#viewport().scale;

    // Ahead of every use of the quarantine below, so a slot that changed hands
    // since the last sync is a candidate again in the same evaluation its new
    // occupant is first asked for.
    this.#sweepQuarantine();

    const requested = new Map<NodeId, CardSyncEntry>();
    const ring = new Set<NodeId>();
    for (const entry of entries) {
      // Quarantined nodes are dropped here rather than at each use, so the
      // admission pass, the mount pass and the prefetch pass cannot disagree
      // about which of them a broken provider is still offered.
      if (this.#failed.has(entry.node)) continue;
      ring.add(entry.node);
      if (entry.prefetchOnly !== true) requested.set(entry.node, entry);
    }

    // Nodes that left the ring without ever being carded: abandon the fetch.
    for (const node of this.#ring) {
      if (ring.has(node) || this.#cards.has(node)) continue;
      driver.cancelPrefetch(node);
      this.#views.delete(node);
    }
    this.#ring = ring;

    // A card whose slot has changed hands is standing for a node that is no
    // longer there, so it goes before anything else looks at it — ahead of the
    // lifetime floor, which exists to stop a *correct* card flickering and has
    // no business holding a wrong one. Releasing rather than notifying keeps
    // the provider's contract intact: `mount` returns per-card state built for
    // one node, and there is no honest way to hand that state a different one.
    // The mount pass below re-cards the slot for its new occupant in the same
    // sync, so the swap costs one release and one mount, not a frame of wrong.
    for (const [node, record] of [...this.#cards]) {
      if (this.#nodes.externalId(node) !== record.externalId) this.#release(node);
    }

    // Refresh what a re-sync is allowed to change on a live card. Size is not
    // in that set: the layout box is fixed at mount so a provider's content
    // never reflows under the user (see CardSyncEntry.size).
    for (const [node, entry] of requested) {
      const record = this.#cards.get(node);
      if (record === undefined) continue;
      record.priority = entry.priority;
      record.opacity = entry.opacity ?? 1;
      record.at = entry.at;
    }

    // Unrequested cards go as soon as the anti-flicker floor allows.
    for (const [node, record] of [...this.#cards]) {
      if (requested.has(node)) continue;
      if (now - record.mountedAtMs < this.#config.minCardLifetimeMs) continue;
      this.#release(node);
    }

    const admitted = this.#admit(requested, now);

    for (const node of [...this.#cards.keys()]) {
      if (!admitted.has(node)) this.#release(node);
    }
    for (const node of admitted) {
      const entry = requested.get(node);
      if (entry === undefined || this.#cards.has(node)) continue;
      this.#mount(node, entry, now, scale);
    }

    // Prefetch last, so a node that just took a card is skipped by the driver
    // rather than offered a fetch it no longer needs.
    for (const node of ring) {
      if (this.#cards.has(node)) continue;
      driver.prefetch(this.#viewOf(node));
    }
  }

  /**
   * Per-frame tick: write the camera transform and re-place every card from
   * live positions.
   *
   * Both halves are no-ops when nothing moved — the transform is compared
   * against the last one written, and {@link CardDriver.place} drops an
   * unchanged placement — so a settled graph costs one string compare plus one
   * position read per card.
   */
  syncFrame(): void {
    const driver = this.#driver;
    if (driver === null) return;

    this.#applyTransform();

    const scale = this.#viewport().scale;
    for (const [node, record] of [...this.#cards]) {
      // The slot may have been freed by a mutation since the card mounted;
      // the node it stood for no longer exists, so neither may the card.
      if (this.#nodes.externalId(node) === undefined) {
        this.#release(node);
        continue;
      }
      driver.place(node, this.#placementOf(node, record, scale));
    }
  }

  /**
   * Report a data or interaction change to a node's card.
   *
   * The other half of the placement path: core derives where a card sits and
   * how big it is, and everything else it knows about the node — selected,
   * hovered, data changed — reaches the provider through here. Placement is
   * not expressible here, by type: it is core's to derive, per frame, in
   * {@link DomCardOverlay.syncFrame}.
   *
   * Cheap and safe to call for any node: one map lookup, and nodes without a
   * mounted card are ignored, so a caller fanning a change out over a
   * selection set does not have to ask which of them are carded.
   */
  notify(node: NodeId, change: CardStateChange): void {
    this.#driver?.notify(node, change);
  }

  /**
   * Note a viewport change and (re)arm the gesture debounce.
   *
   * While a gesture is in flight the container carries `will-change:
   * transform`, so the compositor moves the existing raster instead of
   * re-rasterizing card text on every wheel tick; dropping the hint on settle
   * is what makes the browser redraw the cards sharply at the final scale.
   */
  viewportChanged(): void {
    const container = this.#container;
    if (container === null) return;

    if (!this.#gestureActive) {
      this.#gestureActive = true;
      if (this.#config.rasterizeOnSettle) container.style.willChange = "transform";
    }
    if (this.#gestureHandle !== null) this.#timers.clearTimeout(this.#gestureHandle);
    this.#gestureHandle = this.#timers.setTimeout(() => {
      this.#gestureHandle = null;
      this.#gestureActive = false;
      container.style.willChange = "";
    }, this.#gestureIdleMs);
  }

  /**
   * Start a new LOD epoch: prefetches that never produced a card are aborted
   * and the once-per-node prefetch budget is re-armed. Mounted cards stay —
   * their content is still on screen.
   */
  beginEpoch(): void {
    this.#driver?.beginEpoch();
    for (const node of this.#ring) {
      if (!this.#cards.has(node)) this.#views.delete(node);
    }
    this.#ring = new Set();
  }

  /**
   * Release every card and abandon every prefetch, keeping the container.
   *
   * The quarantine goes with them. Everything dropped here is keyed by slot,
   * and so is the quarantine — this is the point a reload passes through, and
   * carrying a set of slot indices across one would hold nodes of the new graph
   * against a failure that belonged to the old.
   */
  releaseAll(): void {
    this.#endDrag();
    this.#driver?.releaseAll();
    if (this.#cards.size > 0) this.#cardEpoch++;
    this.#cards.clear();
    this.#views.clear();
    this.#ring = new Set();
    this.#failed.clear();
  }

  /** Release everything and detach the container. */
  dispose(): void {
    this.#teardown();
  }

  // --------------------------------------------------------------------------
  // Internals
  // --------------------------------------------------------------------------

  #build(config: DomOverlayConfig): void {
    const host = config.host ?? this.#canvas.parentElement;
    if (host === null) {
      throw new GraphMotherError(
        ErrorCode.INVALID_LAYER_CONFIG,
        "DOM overlay needs a host element: pass one, or mount the canvas inside a parent",
      );
    }

    const container = host.ownerDocument.createElement("div");
    const style = container.style;
    style.position = "absolute";
    style.left = "0";
    style.top = "0";
    style.width = "100%";
    style.height = "100%";
    // The camera is applied about the host's top-left, which is where the
    // canvas origin is; anything else would shear the overlay off the sprites.
    style.transformOrigin = "0 0";
    // Cards opt back in (see CardContainerPool.acquire); the container itself
    // must not swallow the pointer events the canvas is listening for.
    style.pointerEvents = "none";
    style.overflow = "visible";
    host.appendChild(container);

    const pool = new CardContainerPool(host.ownerDocument);
    this.#container = container;
    this.#pool = pool;
    this.#driver = this.#createDriver(container, pool, config);

    container.addEventListener("wheel", this.#onWheel, { passive: false });
    // Delegated: cards switch pointer events back on, so their events bubble
    // to the container even though the container itself is transparent to
    // hit-testing. Nothing here acts on an event that did not start inside a
    // card, so a gesture on empty overlay area reaches the canvas untouched.
    container.addEventListener("pointerdown", this.#onPointerDown);
    container.addEventListener("pointermove", this.#onPointerMove);
    container.addEventListener("pointerup", this.#onPointerUp);
    container.addEventListener("pointercancel", this.#onPointerUp);

    this.#transform = "";
    this.#applyTransform();
  }

  #createDriver(
    container: HTMLElement,
    pool: CardContainerPool,
    config: DomOverlayConfig,
  ): CardDriver<unknown> {
    return new CardDriver<unknown>({
      host: container,
      provider: this.#provider ?? createDefaultCardProvider(),
      createContainer: () => pool.acquire(),
      recycleContainer: (element) => pool.recycle(element),
      onProviderError: (failure) => this.#handleFailure(failure),
      ...(config.className === undefined ? {} : { cardClassName: config.className }),
    });
  }

  /**
   * Drop what a contained failure cost, then pass it on.
   *
   * The driver has already put the DOM back in order by the time this runs; what
   * is left is the overlay's own bookkeeping, which the driver cannot see. A
   * `released` failure took a card the overlay still has a record of, and the
   * label layer has to be told to re-cull because the node it was suppressing
   * is a sprite again.
   */
  #handleFailure(failure: CardProviderFailure): void {
    const node = failure.node.id;
    // Only the hooks a card cannot exist without cost the node the offer; see
    // `cardFailureForfeitsCard`. An advisory prefetch that threw is already
    // spent for the epoch inside the driver, and a card is exactly what a
    // failed `release` proves the provider *can* build.
    if (cardFailureForfeitsCard(failure.hook)) this.#failed.set(node, this.#identityOf(node));
    // The handle went with the card, so a gesture on it has no target left;
    // ending it leaves the node pinned where the drag put it, as a pointer-up
    // would — the same reasoning as an ordinary release.
    if (failure.released && this.#drag?.node === node) this.#endDrag();
    if (failure.released && this.#cards.delete(node)) {
      this.#cardEpoch++;
      if (!this.#ring.has(node)) this.#views.delete(node);
    }
    this.#onCardFailure?.(failure);
  }

  #teardown(): void {
    const container = this.#container;
    if (container === null) return;

    this.releaseAll();
    if (this.#gestureHandle !== null) {
      this.#timers.clearTimeout(this.#gestureHandle);
      this.#gestureHandle = null;
    }
    this.#gestureActive = false;
    container.removeEventListener("wheel", this.#onWheel);
    container.removeEventListener("pointerdown", this.#onPointerDown);
    container.removeEventListener("pointermove", this.#onPointerMove);
    container.removeEventListener("pointerup", this.#onPointerUp);
    container.removeEventListener("pointercancel", this.#onPointerUp);
    container.remove();

    this.#pool?.clear();
    this.#pool = null;
    this.#driver = null;
    this.#container = null;
    this.#transform = "";
  }

  /**
   * Decide which nodes hold a place under the budget.
   *
   * Two rules, in this order:
   *  1. A card younger than the minimum lifetime cannot be evicted, even by a
   *     higher-priority newcomer and even past `maxCards` — anti-flicker
   *     outranks the cap, and any overshoot drains within one lifetime. A
   *     budget honoured by making cards blink is not worth honouring.
   *  2. Everything else competes on priority, with mounted cards winning ties
   *     so a stable ranking does not churn the DOM, and the slot index
   *     breaking the remainder so the outcome is deterministic.
   */
  #admit(requested: ReadonlyMap<NodeId, CardSyncEntry>, now: number): Set<NodeId> {
    const admitted = new Set<NodeId>();
    const candidates: Candidate[] = [];

    for (const [node, record] of this.#cards) {
      if (now - record.mountedAtMs < this.#config.minCardLifetimeMs) {
        admitted.add(node);
        continue;
      }
      candidates.push({
        node,
        priority: requested.get(node)?.priority ?? record.priority,
        mounted: true,
      });
    }
    for (const [node, entry] of requested) {
      if (this.#cards.has(node)) continue;
      candidates.push({ node, priority: entry.priority, mounted: false });
    }

    candidates.sort((a, b) => {
      if (a.priority !== b.priority) return b.priority - a.priority;
      if (a.mounted !== b.mounted) return a.mounted ? -1 : 1;
      return a.node - b.node;
    });

    for (const candidate of candidates) {
      if (admitted.size >= this.#config.maxCards) break;
      admitted.add(candidate.node);
    }
    return admitted;
  }

  #mount(node: NodeId, entry: CardSyncEntry, now: number, scale: number): void {
    const driver = this.#driver;
    if (driver === null) return;

    // The provider is asked first and the entry is the fallback, because the
    // provider is the only party that knows what it is about to render — the
    // LOD pass that built the entry decided *whether* to card the node, not
    // what goes in it.
    const card = this.#viewOf(node);
    const size = driver.measure(card) ?? entry.size ?? DEFAULT_CARD_SIZE;
    const record: CardRecord = {
      card,
      mountedAtMs: now,
      width: size.width,
      height: size.height,
      mountScale: scale,
      externalId: this.#nodes.externalId(node),
      opacity: entry.opacity ?? 1,
      priority: entry.priority,
      at: entry.at,
    };
    // Registered before the provider runs: `CardNode.size()` reads this record
    // and mount() is exactly where a provider lays its content out.
    this.#cards.set(node, record);
    this.#cardEpoch++;

    if (driver.mount(record.card, this.#placementOf(node, record, scale))) return;
    // Which is also why it has to be taken back out again: a mount that never
    // completed leaves the overlay believing the node is carded, so the node
    // would never be mounted, never released, and left as neither sprite nor
    // card for as long as the graph lives.
    this.#cards.delete(node);
    this.#cardEpoch++;
    if (!this.#ring.has(node)) this.#views.delete(node);
  }

  /**
   * What a slot is standing for right now, for identity comparisons.
   *
   * A slot with no producer mapping answers with its own index, exactly as
   * `CardNode.externalId` does — so a slot that has gone dead compares unequal
   * to whatever producer id it last held, which is the answer identity
   * comparisons want from it.
   */
  #identityOf(node: NodeId): IdLike {
    return this.#nodes.externalId(node) ?? node;
  }

  /** Drop quarantines whose slot no longer holds the node that earned them. */
  #sweepQuarantine(): void {
    for (const [node, failedAs] of [...this.#failed]) {
      if (this.#identityOf(node) !== failedAs) this.#failed.delete(node);
    }
  }

  /** Where a card sits this frame, from its record and the live camera. */
  #placementOf(node: NodeId, record: CardRecord, scale: number): CardPlacement {
    return cardPlacementAt(
      record.at ?? this.#nodes.position(node),
      record.width,
      record.height,
      cardCounterScale(scale, record.mountScale),
      record.opacity,
    );
  }

  #release(node: NodeId): void {
    // The handle is about to be detached, so the gesture has no target left;
    // ending it leaves the node pinned where the drag put it, exactly as a
    // pointer-up would.
    if (this.#drag?.node === node) this.#endDrag();
    this.#cards.delete(node);
    this.#cardEpoch++;
    this.#driver?.release(node);
    if (!this.#ring.has(node)) this.#views.delete(node);
  }

  /**
   * The stable `CardNode` view of a slot.
   *
   * Accessors read through to the source on every call, so a card reflects the
   * live graph rather than a snapshot taken at mount, and the provider may
   * retain the object for the card's lifetime as the contract promises.
   */
  #viewOf(node: NodeId): CardNode {
    const existing = this.#views.get(node);
    if (existing !== undefined) return existing;

    const nodes = this.#nodes;
    const cards = this.#cards;
    const viewport = this.#viewport;
    const view: CardNode = {
      id: node,
      // A slot with no mapping (freed, or never mapped) still has to answer:
      // the slot index is itself an IdLike and is what the card was created
      // for. syncFrame() releases such a card on the next frame.
      get externalId(): IdLike {
        return nodes.externalId(node) ?? node;
      },
      get label(): string | undefined {
        return nodes.label(node);
      },
      get tag(): number {
        return nodes.tag(node);
      },
      get weight(): number {
        return nodes.weight(node);
      },
      get depth(): number {
        return nodes.depth(node);
      },
      get contentRef(): string | undefined {
        return nodes.contentRef(node);
      },
      position: () => nodes.position(node),
      size: (): CardSize => {
        const record = cards.get(node);
        if (record === undefined) return DEFAULT_CARD_SIZE;
        // The size the card is *drawn* at, which is its layout box only once
        // the camera has reached the scale it mounted at — from then on the two
        // agree and stay agreed, however far the camera goes in.
        const net = viewport().scale * cardCounterScale(viewport().scale, record.mountScale);
        return { width: record.width * net, height: record.height * net };
      },
      pin: () => nodes.setPinned(node, true),
      unpin: () => nodes.setPinned(node, false),
      isPinned: () => nodes.isPinned(node),
    };
    this.#views.set(node, view);
    return view;
  }

  #applyTransform(): void {
    const container = this.#container;
    if (container === null) return;

    const transform = formatCssMatrix(overlayMatrix(this.#viewport()));
    if (transform === this.#transform) return;
    this.#transform = transform;
    container.style.transform = transform;
  }

  /**
   * Forward a wheel that landed on a card to the canvas.
   *
   * Not optional: a user zooming back out from a card has the pointer over a
   * card by construction, and the canvas is the only element listening for the
   * gesture. The clone carries `clientX`/`clientY` because the canvas handler
   * derives the zoom focus from them.
   */
  readonly #onWheel = (event: Event): void => {
    if (!this.#config.forwardWheel) return;
    // Asked before anything is prevented: a claimed region has to see the
    // event untouched, or the browser will not scroll it.
    if (this.#claimsWheel(event.target)) return;
    const canvas = this.#canvas;
    event.preventDefault();
    canvas.dispatchEvent(cloneWheelEvent(event as WheelEvent, canvas.ownerDocument.defaultView));
  };

  /**
   * Begin a card drag, if the gesture started on a provider-declared handle.
   *
   * Deliberately silent otherwise: a pointer event that landed on card *text*
   * belongs to the browser (selection, links, focus), and one that landed on
   * empty overlay area never reaches here at all — the container is
   * transparent to hit-testing, so the canvas got it instead.
   */
  readonly #onPointerDown = (event: Event): void => {
    const sink = this.#dragSink;
    if (sink === null || this.#drag !== null) return;

    const pointer = event as PointerEvent;
    if (pointer.button > 0) return;
    const target = this.#resolveTarget(pointer.target);
    if (target === null || !target.onHandle) return;

    // Suppress the browser's own drag/selection gesture on the handle only.
    event.preventDefault();

    const origin = this.#nodes.position(target.node);
    this.#drag = {
      node: target.node,
      pointerId: pointer.pointerId,
      clientX: pointer.clientX,
      clientY: pointer.clientY,
      originX: origin.x,
      originY: origin.y,
      // Capture keeps the rest of the gesture on the card, which is what stops
      // the canvas from seeing it as a pan. Hosts without it (a DOM shim) still
      // work as long as the pointer stays over the card.
      captured: capturePointer(target.element, pointer.pointerId),
      position: { x: origin.x, y: origin.y },
    };
    sink.begin(target.node, origin);
  };

  readonly #onPointerMove = (event: Event): void => {
    const drag = this.#drag;
    if (drag === null) return;
    const pointer = event as PointerEvent;
    if (pointer.pointerId !== drag.pointerId) return;

    drag.position = this.#dragPosition(drag, pointer.clientX, pointer.clientY);
    // The node is the authority: the card is re-placed from its position on
    // the next frame, in graph units, by the same path that follows the
    // simulation. Nothing here writes CSS.
    this.#dragSink?.move(drag.node, drag.position);
  };

  readonly #onPointerUp = (event: Event): void => {
    const drag = this.#drag;
    if (drag === null) return;
    const pointer = event as PointerEvent;
    if (pointer.pointerId !== drag.pointerId) return;

    drag.position = this.#dragPosition(drag, pointer.clientX, pointer.clientY);
    this.#endDrag();
  };

  /** Finish the drag in flight, if any. The node keeps its pin. */
  #endDrag(): void {
    const drag = this.#drag;
    if (drag === null) return;
    this.#drag = null;
    releasePointer(drag.captured, drag.pointerId);
    this.#dragSink?.end(drag.node, drag.position);
  }

  /**
   * Where the dragged node goes for a pointer at `(clientX, clientY)`.
   *
   * Offset-preserving rather than centring the node under the pointer: the
   * grab point stays under the finger, so a card does not jump when it is
   * picked up by an edge. Client pixels divide by the live zoom, so a drag
   * survives a zoom mid-gesture.
   */
  #dragPosition(drag: CardDrag, clientX: number, clientY: number): Vec2 {
    const { scale } = this.#viewport();
    if (!Number.isFinite(scale) || scale <= 0) return drag.position;
    return {
      x: drag.originX + (clientX - drag.clientX) / scale,
      y: drag.originY + (clientY - drag.clientY) / scale,
    };
  }

  /**
   * Resolve an event target to the card container it sits in, and whether the
   * path from it crossed a drag handle.
   *
   * One walk answers both, and stopping at the container is what keeps a
   * handle inside one card from being seen by another.
   */
  /**
   * Whether this event landed inside a region that owns its own wheel.
   *
   * Walks to the card container and stops there, so an attribute outside the
   * overlay cannot claim the camera's gestures.
   */
  #claimsWheel(target: EventTarget | null): boolean {
    const container = this.#container;
    if (container === null) return false;
    let element = target as Element | null;
    while (element !== null && element !== container) {
      if (element.hasAttribute?.(CARD_SCROLL_ATTRIBUTE) === true) return true;
      element = element.parentElement;
    }
    return false;
  }

  #resolveTarget(
    target: EventTarget | null,
  ): { node: NodeId; element: Element; onHandle: boolean } | null {
    const container = this.#container;
    const driver = this.#driver;
    if (container === null || driver === null) return null;

    let element = target as Element | null;
    let onHandle = false;
    while (element !== null && element !== container) {
      if (element.hasAttribute?.(CARD_DRAG_HANDLE_ATTRIBUTE) === true) onHandle = true;
      const parent: Element | null = element.parentElement;
      if (parent === container) {
        const node = driver.nodeForContainer(element);
        return node === undefined ? null : { node, element, onHandle };
      }
      element = parent;
    }
    return null;
  }
}

/**
 * Route the rest of a gesture to `element`, and report whether it took.
 *
 * `setPointerCapture` is absent from every DOM shim the tests can reach, and
 * present but throwing in a browser when the pointer is already gone.
 */
function capturePointer(element: Element, pointerId: number): Element | null {
  if (typeof element.setPointerCapture !== "function") return null;
  try {
    element.setPointerCapture(pointerId);
    return element;
  } catch {
    return null;
  }
}

/** Undo {@link capturePointer}. A capture the host already dropped is fine. */
function releasePointer(element: Element | null, pointerId: number): void {
  if (element === null || typeof element.releasePointerCapture !== "function") return;
  try {
    element.releasePointerCapture(pointerId);
  } catch {
    // The host released it first: pointercancel, or the element left the DOM.
  }
}

/**
 * Coerce the two budget knobs to values the admission pass can compare against.
 *
 * They arrive from the same place `LodConfig` does — a live inspector slider,
 * a consumer's own interpolation — so nonsense saturates rather than throwing,
 * matching `resolveLodConfig`. A NaN budget would admit every card, and a NaN
 * lifetime would hold every card forever.
 */
function clampBudget(config: DomOverlayConfig): DomOverlayConfig {
  const maxCards = Number.isFinite(config.maxCards) ? Math.max(0, Math.trunc(config.maxCards)) : 0;
  const minCardLifetimeMs = Number.isFinite(config.minCardLifetimeMs)
    ? Math.max(0, config.minCardLifetimeMs)
    : 0;
  if (maxCards === config.maxCards && minCardLifetimeMs === config.minCardLifetimeMs) return config;
  return { ...config, maxCards, minCardLifetimeMs };
}

/** Enough of a constructor to rebuild an event inside its own DOM. */
type EventConstructor = new (type: string, init: EventInit) => Event;

/**
 * Rebuild a wheel event so it can be dispatched on the canvas. The original
 * cannot be reused: an event still in flight cannot be dispatched again.
 *
 * `WheelEvent` exists only in a real browser — the DOM shims Deno can reach
 * implement `Event`/`CustomEvent` and nothing else — and every implementation
 * rejects an event object that is not its own. So the clone is built from the
 * view's `WheelEvent` where there is one, and otherwise from the incoming
 * event's own constructor with the wheel fields copied on. The canvas handler
 * reads fields rather than the class, so both paths drive the same zoom.
 */
function cloneWheelEvent(
  source: WheelEvent,
  view: (Window & typeof globalThis) | null,
): Event {
  const fields = {
    deltaX: source.deltaX,
    deltaY: source.deltaY,
    deltaZ: source.deltaZ,
    deltaMode: source.deltaMode,
    clientX: source.clientX,
    clientY: source.clientY,
    ctrlKey: source.ctrlKey,
    shiftKey: source.shiftKey,
    altKey: source.altKey,
    metaKey: source.metaKey,
  };

  const Wheel = view?.WheelEvent;
  if (Wheel !== undefined) {
    return new Wheel("wheel", { ...fields, bubbles: false, cancelable: true });
  }
  const Shim = source.constructor as unknown as EventConstructor;
  return Object.assign(new Shim("wheel", { bubbles: false, cancelable: true }), fields);
}
