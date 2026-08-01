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
 * and this file cannot weaken it.
 *
 * @module
 */

import type { NodeId, Vec2, ViewportState } from "../types.ts";
import type { IdLike } from "../graph/id_map.ts";
import { ErrorCode, GraphMotherError } from "../errors.ts";
import { CardDriver } from "./driver.ts";
import { CardContainerPool } from "./pool.ts";
import { cardPlacementAt, formatCssMatrix, overlayMatrix } from "./projection.ts";
import { createDefaultCardProvider } from "./default_card.ts";
import { DEFAULT_DOM_OVERLAY_CONFIG } from "./types.ts";
import type { CardNode, CardProvider, CardSize, DomOverlayConfig } from "./types.ts";

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
 * Deliberately carries no position: {@link CardNodeSource.position} is the
 * single authority and is re-read every frame, so a sync entry cannot go stale
 * against the simulation between evaluations.
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
   * Card box in CSS pixels, read once when the card mounts. From then on the
   * box is held in graph units, so the card scales with the camera exactly
   * like the sprite it replaced instead of resizing under the user's pointer.
   */
  readonly size?: CardSize;
  /** Crossfade opacity, 0..1. Defaults to fully opaque. */
  readonly opacity?: number;
  /**
   * Node is in the prefetch ring but not carded: the provider is offered an
   * advisory, abortable `prefetch` and no DOM is created.
   */
  readonly prefetchOnly?: boolean;
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
  /** Simultaneously mounted cards. Defaults to {@link DEFAULT_MAX_CARDS}. */
  readonly maxCards?: number;
  /** Anti-flicker lifetime floor. Defaults to {@link DEFAULT_MIN_CARD_LIFETIME_MS}. */
  readonly minCardLifetimeMs?: number;
  /** Gesture debounce. Defaults to {@link DEFAULT_GESTURE_IDLE_MS}. */
  readonly gestureIdleMs?: number;
  /** Clock, injected so card lifetimes replay identically in a test. */
  readonly now?: () => number;
  /** Timers, injected so the gesture debounce is drivable without real time. */
  readonly timers?: OverlayTimers;
}

/** One mounted card's core-owned state. */
interface CardRecord {
  readonly card: CardNode;
  /** Clock value at mount; the minimum-lifetime floor is measured from it. */
  readonly mountedAtMs: number;
  /** Card box in graph units, fixed for the card's lifetime. */
  readonly width: number;
  readonly height: number;
  opacity: number;
  priority: number;
}

/** A node competing for a place under the budget. */
interface Candidate {
  readonly node: NodeId;
  readonly priority: number;
  readonly mounted: boolean;
}

export class DomCardOverlay {
  readonly #canvas: HTMLCanvasElement;
  readonly #viewport: () => ViewportState;
  readonly #nodes: CardNodeSource;
  readonly #maxCards: number;
  readonly #minCardLifetimeMs: number;
  readonly #gestureIdleMs: number;
  readonly #now: () => number;
  readonly #timers: OverlayTimers;

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

  /** Last transform written, so an unchanged camera costs no DOM write. */
  #transform = "";
  #gestureHandle: number | null = null;
  #gestureActive = false;

  constructor(options: DomCardOverlayOptions) {
    this.#canvas = options.canvas;
    this.#viewport = options.viewport;
    this.#nodes = options.nodes;
    this.#maxCards = Math.max(0, Math.trunc(options.maxCards ?? DEFAULT_MAX_CARDS));
    this.#minCardLifetimeMs = Math.max(
      0,
      options.minCardLifetimeMs ?? DEFAULT_MIN_CARD_LIFETIME_MS,
    );
    this.#gestureIdleMs = Math.max(0, options.gestureIdleMs ?? DEFAULT_GESTURE_IDLE_MS);
    this.#now = options.now ?? (() => performance.now());
    this.#timers = options.timers ?? HOST_TIMERS;
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
    const next: DomOverlayConfig = { ...previous, ...config };
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
   */
  setProvider(provider: CardProvider<unknown> | null): void {
    this.#provider = provider;
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

    const requested = new Map<NodeId, CardSyncEntry>();
    const ring = new Set<NodeId>();
    for (const entry of entries) {
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

    // Refresh what a re-sync is allowed to change on a live card. Size is not
    // in that set: the graph-unit box is fixed at mount so the card tracks the
    // camera rather than resizing under the user (see CardSyncEntry.size).
    for (const [node, entry] of requested) {
      const record = this.#cards.get(node);
      if (record === undefined) continue;
      record.priority = entry.priority;
      record.opacity = entry.opacity ?? 1;
    }

    // Unrequested cards go as soon as the anti-flicker floor allows.
    for (const [node, record] of [...this.#cards]) {
      if (requested.has(node)) continue;
      if (now - record.mountedAtMs < this.#minCardLifetimeMs) continue;
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

    for (const [node, record] of [...this.#cards]) {
      // The slot may have been freed by a mutation since the card mounted;
      // the node it stood for no longer exists, so neither may the card.
      if (this.#nodes.externalId(node) === undefined) {
        this.#release(node);
        continue;
      }
      const anchor = this.#nodes.position(node);
      driver.place(node, cardPlacementAt(anchor, record.width, record.height, record.opacity));
    }
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

  /** Release every card and abandon every prefetch, keeping the container. */
  releaseAll(): void {
    this.#driver?.releaseAll();
    this.#cards.clear();
    this.#views.clear();
    this.#ring = new Set();
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
      ...(config.className === undefined ? {} : { cardClassName: config.className }),
    });
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
      if (now - record.mountedAtMs < this.#minCardLifetimeMs) {
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
      if (admitted.size >= this.#maxCards) break;
      admitted.add(candidate.node);
    }
    return admitted;
  }

  #mount(node: NodeId, entry: CardSyncEntry, now: number, scale: number): void {
    const driver = this.#driver;
    if (driver === null) return;

    const size = entry.size ?? DEFAULT_CARD_SIZE;
    const record: CardRecord = {
      card: this.#viewOf(node),
      mountedAtMs: now,
      width: size.width / scale,
      height: size.height / scale,
      opacity: entry.opacity ?? 1,
      priority: entry.priority,
    };
    // Registered before the provider runs: `CardNode.size()` reads this record
    // and mount() is exactly where a provider lays its content out.
    this.#cards.set(node, record);

    const anchor = this.#nodes.position(node);
    driver.mount(record.card, cardPlacementAt(anchor, record.width, record.height, record.opacity));
  }

  #release(node: NodeId): void {
    this.#cards.delete(node);
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
        // Held in graph units, reported in CSS pixels: the card box is
        // whatever the current camera makes of it.
        const { scale } = viewport();
        return { width: record.width * scale, height: record.height * scale };
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
    const canvas = this.#canvas;
    event.preventDefault();
    canvas.dispatchEvent(cloneWheelEvent(event as WheelEvent, canvas.ownerDocument.defaultView));
  };
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
