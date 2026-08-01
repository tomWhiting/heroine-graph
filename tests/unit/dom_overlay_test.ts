/**
 * DOM card overlay engine.
 *
 * The overlay decides which cards exist, where they sit and how long they
 * live. Everything asserted here is a promise made to something core cannot
 * see — a consumer's card content, a user mid-gesture, a browser's rasterizer —
 * so each is pinned against a recording provider rather than left to a
 * screenshot:
 *
 *  - a released container is recycled, not recreated;
 *  - eviction respects priority, and never takes a card younger than
 *    `minCardLifetimeMs`;
 *  - `maxCards` is enforced;
 *  - a burst of viewport changes ends in exactly one settle (FR-012);
 *  - a wheel over a card reaches the canvas (zooming out from a card);
 *  - every teardown path goes through `CardDriver`, so SC-004 — core never
 *    touches card children — cannot be bypassed by this layer.
 *
 * linkedom supplies the DOM, as in `card_contract_test.ts`: the repo has no
 * browser runner and the overlay needs only element creation, attachment,
 * inline styles and event dispatch.
 */

import { assert, assertEquals, assertThrows } from "jsr:@std/assert@^1";
import { parseHTML } from "https://esm.sh/linkedom@0.18.11/worker";
import {
  type CardNodeSource,
  type CardSyncEntry,
  DEFAULT_CARD_SIZE,
  DomCardOverlay,
  type OverlayTimers,
} from "../../packages/core/src/overlay/overlay.ts";
import { CardContainerPool } from "../../packages/core/src/overlay/pool.ts";
import {
  cardPlacementAt,
  formatCssMatrix,
  overlayMatrix,
} from "../../packages/core/src/overlay/projection.ts";
import type { CardNode, CardProvider } from "../../packages/core/src/overlay/types.ts";
import type { NodeId, Vec2, ViewportState } from "../../packages/core/src/types.ts";

// -----------------------------------------------------------------------------
// Harness
// -----------------------------------------------------------------------------

interface Dom {
  readonly document: Document;
  readonly host: HTMLElement;
  readonly canvas: HTMLCanvasElement;
  /** linkedom's own `Event` subclass; its dispatch rejects foreign events. */
  readonly CustomEvent: new (type: string, init: EventInit) => Event;
}

function createDom(): Dom {
  const parsed = parseHTML(
    "<html><body><div id='host'><canvas id='canvas'></canvas></div></body></html>",
  );
  const document = parsed.document as unknown as Document;
  return {
    document,
    host: document.getElementById("host") as unknown as HTMLElement,
    canvas: document.getElementById("canvas") as unknown as HTMLCanvasElement,
    CustomEvent: (parsed.window as unknown as {
      CustomEvent: new (type: string, init: EventInit) => Event;
    }).CustomEvent,
  };
}

/** Deterministic stand-in for `setTimeout`, so a gesture ends on command. */
class FakeTimers implements OverlayTimers {
  #nextHandle = 1;
  readonly pending = new Map<number, () => void>();
  /** Timers armed and disarmed, which is what makes a debounce a debounce. */
  scheduled = 0;
  cancelled = 0;

  setTimeout(handler: () => void, _ms: number): number {
    const handle = this.#nextHandle++;
    this.pending.set(handle, handler);
    this.scheduled++;
    return handle;
  }

  clearTimeout(handle: number): void {
    if (this.pending.delete(handle)) this.cancelled++;
  }

  /** Fire everything currently armed; returns how many callbacks ran. */
  run(): number {
    const due = [...this.pending.values()];
    this.pending.clear();
    for (const handler of due) handler();
    return due.length;
  }
}

interface Call {
  readonly hook: "prefetch" | "mount" | "update" | "release";
  readonly nodeId: NodeId;
  /** Whether the container was still attached when the hook ran. */
  readonly connected?: boolean;
}

interface Recorder extends CardProvider<{ readonly nodeId: NodeId }> {
  readonly calls: Call[];
  readonly containers: Map<NodeId, HTMLElement>;
  count(hook: Call["hook"], nodeId?: NodeId): number;
}

function recorder(): Recorder {
  const calls: Call[] = [];
  const containers = new Map<NodeId, HTMLElement>();
  return {
    calls,
    containers,
    count(hook, nodeId) {
      return calls.filter((c) => c.hook === hook && (nodeId === undefined || c.nodeId === nodeId))
        .length;
    },
    prefetch(node) {
      calls.push({ hook: "prefetch", nodeId: node.id });
    },
    mount(container, node) {
      calls.push({ hook: "mount", nodeId: node.id, connected: container.isConnected });
      containers.set(node.id, container);
      // Real content, so the pool's "provider cleaned up" check has something
      // to check and SC-004 has something to protect.
      const body = container.ownerDocument.createElement("p");
      body.textContent = String(node.externalId);
      container.appendChild(body);
      return { nodeId: node.id };
    },
    update(_container, node) {
      calls.push({ hook: "update", nodeId: node.id });
    },
    release(container, node) {
      calls.push({ hook: "release", nodeId: node.id, connected: container.isConnected });
      while (container.firstChild !== null) container.firstChild.remove();
    },
  };
}

/** Mutable graph the overlay reads through. */
class Graph implements CardNodeSource {
  readonly positions = new Map<NodeId, Vec2>();
  readonly live = new Set<NodeId>();
  readonly pinned = new Set<NodeId>();

  constructor(nodeIds: readonly NodeId[]) {
    for (const id of nodeIds) {
      this.live.add(id);
      this.positions.set(id, { x: id * 10, y: id * -5 });
    }
  }

  externalId(node: NodeId): string | undefined {
    return this.live.has(node) ? `src/mod${node}.ts` : undefined;
  }
  label(node: NodeId): string | undefined {
    return `mod${node}.ts`;
  }
  tag(): number {
    return 0;
  }
  weight(): number {
    return 0;
  }
  depth(): number {
    return 0;
  }
  contentRef(): string | undefined {
    return undefined;
  }
  position(node: NodeId): Vec2 {
    return this.positions.get(node) ?? { x: 0, y: 0 };
  }
  isPinned(node: NodeId): boolean {
    return this.pinned.has(node);
  }
  setPinned(node: NodeId, pinned: boolean): void {
    if (pinned) this.pinned.add(node);
    else this.pinned.delete(node);
  }
}

interface Harness {
  readonly dom: Dom;
  readonly overlay: DomCardOverlay;
  readonly provider: Recorder;
  readonly graph: Graph;
  readonly timers: FakeTimers;
  /** Overlay container; the harness only enables overlays, so it exists. */
  readonly container: HTMLElement;
  /** Injected clock, in ms. */
  now: number;
  viewport: ViewportState;
}

function viewportState(overrides: Partial<ViewportState> = {}): ViewportState {
  return {
    x: 0,
    y: 0,
    scale: 1,
    width: 800,
    height: 600,
    minScale: 0.01,
    maxScale: 100,
    ...overrides,
  };
}

interface HarnessOptions {
  readonly nodeIds?: readonly NodeId[];
  readonly maxCards?: number;
  readonly minCardLifetimeMs?: number;
  readonly forwardWheel?: boolean;
  readonly rasterizeOnSettle?: boolean;
  readonly className?: string;
}

function harness(options: HarnessOptions = {}): Harness {
  const dom = createDom();
  const provider = recorder();
  const graph = new Graph(options.nodeIds ?? [1, 2, 3, 4, 5, 6, 7, 8]);
  const timers = new FakeTimers();

  const state: { now: number; viewport: ViewportState } = {
    now: 0,
    viewport: viewportState(),
  };

  const overlay = new DomCardOverlay({
    canvas: dom.canvas,
    viewport: () => state.viewport,
    nodes: graph,
    maxCards: options.maxCards ?? 16,
    minCardLifetimeMs: options.minCardLifetimeMs ?? 400,
    gestureIdleMs: 120,
    now: () => state.now,
    timers,
  });
  overlay.setProvider(provider);
  overlay.setConfig({
    enabled: true,
    rasterizeOnSettle: options.rasterizeOnSettle ?? true,
    forwardWheel: options.forwardWheel ?? true,
    ...(options.className === undefined ? {} : { className: options.className }),
  });

  const container = overlay.element;
  assert(container !== null, "overlay must have built a container when enabled");

  return {
    dom,
    overlay,
    provider,
    graph,
    timers,
    container,
    get now() {
      return state.now;
    },
    set now(value: number) {
      state.now = value;
    },
    get viewport() {
      return state.viewport;
    },
    set viewport(value: ViewportState) {
      state.viewport = value;
    },
  };
}

function entry(node: NodeId, priority: number, extra: Partial<CardSyncEntry> = {}): CardSyncEntry {
  return { node, priority, ...extra };
}

/** `will-change`, normalised: linkedom reports an unset property as undefined. */
function compositingHint(element: HTMLElement): string {
  return element.style.willChange ?? "";
}

function mountedNodes(provider: Recorder): NodeId[] {
  const open = new Set<NodeId>();
  for (const call of provider.calls) {
    if (call.hook === "mount") open.add(call.nodeId);
    if (call.hook === "release") open.delete(call.nodeId);
  }
  return [...open].sort((a, b) => a - b);
}

// -----------------------------------------------------------------------------
// Container lifecycle and pooling
// -----------------------------------------------------------------------------

Deno.test("overlay: the container is click-through and carries the camera", () => {
  const h = harness();

  // Element identity is compared with `assert`, never `assertEquals`: a
  // failing DOM-node comparison would try to diff two linkedom trees.
  assert(h.container.parentElement === h.dom.host, "defaults to the canvas's parent");
  assertEquals(h.container.style.position, "absolute");
  assertEquals(h.container.style.transformOrigin, "0 0");
  assertEquals(
    h.container.style.pointerEvents,
    "none",
    "the canvas below must keep receiving pointer events",
  );
  assertEquals(h.container.style.transform, formatCssMatrix(overlayMatrix(h.viewport)));

  // Disabling detaches it; nothing else on the host is disturbed.
  h.overlay.setConfig({ enabled: false });
  assert(h.overlay.element === null);
  assertEquals(h.container.isConnected, false);
  assertEquals(h.dom.host.querySelector("canvas") !== null, true);
});

Deno.test("overlay: enabling without a host is a caller error", () => {
  const dom = createDom();
  const orphan = dom.document.createElement("canvas") as unknown as HTMLCanvasElement;
  const overlay = new DomCardOverlay({
    canvas: orphan,
    viewport: () => viewportState(),
    nodes: new Graph([1]),
  });

  assertThrows(
    () => overlay.setConfig({ enabled: true }),
    Error,
    "host element",
  );
});

Deno.test("overlay: the configured class name reaches the cards", () => {
  const h = harness({ className: "gm-card", minCardLifetimeMs: 0 });

  h.overlay.syncCards([entry(1, 1)]);
  assertEquals(h.provider.containers.get(1)?.className, "gm-card");

  // The class is applied when a container is created, so changing it has to
  // rebuild rather than patch — and that releases the cards it invalidates.
  h.overlay.setConfig({ className: "gm-card-v2" });
  assertEquals(h.overlay.cardCount, 0);
  assertEquals(h.provider.count("release", 1), 1);

  h.overlay.syncCards([entry(1, 1)]);
  assertEquals(h.provider.containers.get(1)?.className, "gm-card-v2");
});

Deno.test("overlay: a released container is recycled, not recreated", () => {
  const h = harness({ minCardLifetimeMs: 0 });

  h.overlay.syncCards([entry(1, 1)]);
  const first = h.provider.containers.get(1);
  assert(first !== undefined);

  h.overlay.syncCards([]);
  assertEquals(h.provider.count("release", 1), 1);
  assertEquals(first.isConnected, false);

  h.overlay.syncCards([entry(2, 1)]);
  assert(
    h.provider.containers.get(2) === first,
    "the second card reuses the first card's element",
  );
  assertEquals(first.isConnected, true);
  // Recycled or fresh, the element arrives with nothing of the previous card
  // left in it and with the card-level pointer discipline reapplied.
  assertEquals(first.children.length, 1, "only the new provider's content");
  assertEquals(first.style.pointerEvents, "auto");
});

Deno.test("overlay: a container a provider left dirty is dropped instead of reused", () => {
  const { document } = createDom();
  const pool = new CardContainerPool(document, 4);

  const clean = pool.acquire();
  const dirty = pool.acquire();
  dirty.appendChild(document.createElement("span"));

  pool.recycle(clean);
  pool.recycle(dirty);
  assertEquals(pool.idleCount, 1, "leftover children would leak into the next card");
  assert(pool.acquire() === clean);

  // The idle limit is a limit, not a suggestion.
  const capped = new CardContainerPool(document, 1);
  capped.recycle(document.createElement("div"));
  capped.recycle(document.createElement("div"));
  assertEquals(capped.idleCount, 1);
});

// -----------------------------------------------------------------------------
// Budget, priority and lifetime
// -----------------------------------------------------------------------------

Deno.test("overlay: maxCards is enforced, highest priority first", () => {
  const h = harness({ maxCards: 3 });

  h.overlay.syncCards([
    entry(1, 0.1),
    entry(2, 0.9),
    entry(3, 0.5),
    entry(4, 0.7),
    entry(5, 0.3),
  ]);

  assertEquals(h.overlay.cardCount, 3);
  assertEquals(mountedNodes(h.provider), [2, 3, 4]);
});

Deno.test("overlay: eviction takes the lowest priority once the lifetime floor allows", () => {
  const h = harness({ maxCards: 2, minCardLifetimeMs: 400 });

  h.overlay.syncCards([entry(1, 0.2), entry(2, 0.8)]);
  assertEquals(mountedNodes(h.provider), [1, 2]);

  // Still inside the anti-flicker floor: a better card cannot displace either.
  h.now = 399;
  h.overlay.syncCards([entry(1, 0.2), entry(2, 0.8), entry(3, 1)]);
  assertEquals(mountedNodes(h.provider), [1, 2], "a card younger than the floor is not evictable");
  assertEquals(h.provider.count("mount", 3), 0);

  // Once both are old enough, the budget takes the weakest card, not the oldest.
  h.now = 400;
  h.overlay.syncCards([entry(1, 0.2), entry(2, 0.8), entry(3, 1)]);
  assertEquals(mountedNodes(h.provider), [2, 3]);
  assertEquals(h.provider.count("release", 1), 1);
});

Deno.test("overlay: the lifetime floor outranks the budget, and drains within one lifetime", () => {
  const h = harness({ maxCards: 1, minCardLifetimeMs: 400 });

  h.overlay.syncCards([entry(1, 0.1)]);
  h.now = 10;
  // Two protected-by-age cards would exceed maxCards; blinking a card out to
  // honour the cap is worse than briefly overshooting it.
  h.overlay.syncCards([entry(2, 9)]);
  assertEquals(mountedNodes(h.provider), [1], "the young card holds the only slot");

  h.now = 500;
  h.overlay.syncCards([entry(2, 9)]);
  assertEquals(mountedNodes(h.provider), [2]);
  assertEquals(h.provider.count("release", 1), 1);
});

Deno.test("overlay: an unrequested card survives until the floor, then goes", () => {
  const h = harness({ minCardLifetimeMs: 400 });

  h.overlay.syncCards([entry(1, 1)]);
  h.now = 399;
  h.overlay.syncCards([]);
  assertEquals(h.overlay.cardCount, 1, "released a card mid-flicker");

  h.now = 400;
  h.overlay.syncCards([]);
  assertEquals(h.overlay.cardCount, 0);
  assertEquals(h.provider.count("release", 1), 1);
});

Deno.test("overlay: ranking is stable and deterministic on ties", () => {
  const h = harness({ maxCards: 2, minCardLifetimeMs: 0 });

  // Equal priorities: the lower slot index wins, so the outcome does not
  // depend on iteration order.
  h.overlay.syncCards([entry(5, 1), entry(3, 1), entry(7, 1)]);
  assertEquals(mountedNodes(h.provider), [3, 5]);

  // A mounted card beats an equal-priority newcomer, so a steady ranking does
  // not churn the DOM.
  h.overlay.syncCards([entry(3, 1), entry(5, 1), entry(1, 1)]);
  assertEquals(mountedNodes(h.provider), [3, 5]);
  assertEquals(h.provider.count("mount"), 2, "no remount happened");
});

// -----------------------------------------------------------------------------
// Placement
// -----------------------------------------------------------------------------

Deno.test("overlay: cards are placed in graph units, centred on their node", () => {
  const h = harness({ minCardLifetimeMs: 0 });
  h.viewport = viewportState({ scale: 2 });

  h.overlay.syncCards([entry(1, 1)]);
  const container = h.provider.containers.get(1);
  assert(container !== undefined);

  // Size is given in CSS pixels and held in graph units, so the card scales
  // with the camera exactly like the sprite it replaced.
  const width = DEFAULT_CARD_SIZE.width / 2;
  const height = DEFAULT_CARD_SIZE.height / 2;
  const expected = cardPlacementAt(h.graph.position(1), width, height, 1);
  assertEquals(container.style.transform, `translate(${expected.x}px, ${expected.y}px)`);
  assertEquals(container.style.width, `${width}px`);
  assertEquals(container.style.height, `${height}px`);

  // A moving node follows on the frame tick, and the camera moves the whole
  // container rather than every card.
  h.graph.positions.set(1, { x: 111, y: -222 });
  h.viewport = viewportState({ scale: 2, x: 40 });
  h.overlay.syncFrame();
  const moved = cardPlacementAt({ x: 111, y: -222 }, width, height, 1);
  assertEquals(container.style.transform, `translate(${moved.x}px, ${moved.y}px)`);
  assertEquals(h.container.style.transform, formatCssMatrix(overlayMatrix(h.viewport)));

  // Nothing moved: no DOM write, no provider callback.
  const before = h.provider.calls.length;
  h.overlay.syncFrame();
  assertEquals(h.provider.calls.length, before);
});

Deno.test("overlay: a card whose node was removed is released on the next frame", () => {
  const h = harness({ minCardLifetimeMs: 10_000 });

  h.overlay.syncCards([entry(1, 1)]);
  h.graph.live.delete(1);

  h.overlay.syncFrame();
  assertEquals(h.overlay.cardCount, 0, "the lifetime floor cannot keep a card for a dead slot");
  assertEquals(h.provider.count("release", 1), 1);
});

Deno.test("overlay: CardNode is a live view of the graph", () => {
  const h = harness({ minCardLifetimeMs: 0 });
  let card: CardNode | undefined;
  h.overlay.setProvider({
    mount(_container, node) {
      card = node;
      return null;
    },
  });
  h.viewport = viewportState({ scale: 4 });
  h.overlay.syncCards([entry(2, 1, { size: { width: 400, height: 200 } })]);

  assert(card !== undefined);
  assertEquals(card.id, 2);
  assertEquals(card.externalId, "src/mod2.ts");
  assertEquals(card.position(), h.graph.position(2));
  assertEquals(card.size(), { width: 400, height: 200 }, "CSS pixels at the zoom it mounted at");

  // The box is fixed in graph units, so halving the zoom halves the card.
  h.viewport = viewportState({ scale: 2 });
  assertEquals(card.size(), { width: 200, height: 100 });

  h.graph.positions.set(2, { x: 5, y: 6 });
  assertEquals(card.position(), { x: 5, y: 6 });

  card.pin();
  assertEquals(card.isPinned(), true);
  assertEquals(h.graph.pinned.has(2), true);
  card.unpin();
  assertEquals(card.isPinned(), false);
});

// -----------------------------------------------------------------------------
// Prefetch ring
// -----------------------------------------------------------------------------

Deno.test("overlay: the prefetch ring offers nodes without creating DOM", () => {
  const h = harness({ minCardLifetimeMs: 0 });

  h.overlay.syncCards([entry(1, 1), entry(2, 0.5, { prefetchOnly: true })]);
  assertEquals(h.overlay.cardCount, 1);
  assertEquals(h.provider.count("prefetch", 2), 1);
  assertEquals(h.provider.count("mount", 2), 0);

  // Repeat offers inside an epoch are dropped by the driver.
  h.overlay.syncCards([entry(1, 1), entry(2, 0.5, { prefetchOnly: true })]);
  assertEquals(h.provider.count("prefetch", 2), 1);

  // A node promoted out of the ring mounts; a node that leaves it is dropped
  // and is not re-offered until a new epoch.
  h.overlay.syncCards([entry(2, 1)]);
  assertEquals(h.provider.count("mount", 2), 1);
  h.overlay.syncCards([]);
  h.overlay.beginEpoch();
  h.overlay.syncCards([entry(2, 0.5, { prefetchOnly: true })]);
  assertEquals(h.provider.count("prefetch", 2), 2);
});

// -----------------------------------------------------------------------------
// Gesture debounce (FR-012)
// -----------------------------------------------------------------------------

Deno.test("overlay: a burst of viewport changes settles exactly once", () => {
  const h = harness();

  for (let i = 0; i < 5; i++) h.overlay.viewportChanged();

  assertEquals(h.timers.scheduled, 5);
  assertEquals(h.timers.cancelled, 4, "each change disarms the previous timer");
  assertEquals(h.timers.pending.size, 1, "one settle is armed, not five");
  assertEquals(h.overlay.gestureActive, true);
  assertEquals(
    compositingHint(h.container),
    "transform",
    "the compositor moves the existing raster while the gesture runs",
  );

  assertEquals(h.timers.run(), 1);
  assertEquals(h.overlay.gestureActive, false);
  assertEquals(
    compositingHint(h.container),
    "",
    "dropping the hint is what re-rasterizes card text at the settled scale",
  );

  // Settled means settled: nothing is left armed to fire again.
  assertEquals(h.timers.run(), 0);

  // A later gesture arms a fresh one.
  h.overlay.viewportChanged();
  assertEquals(h.overlay.gestureActive, true);
  assertEquals(h.timers.pending.size, 1);
});

Deno.test("overlay: rasterizeOnSettle off leaves the compositing hint alone", () => {
  const h = harness({ rasterizeOnSettle: false });

  h.overlay.viewportChanged();
  assertEquals(h.overlay.gestureActive, true, "the gesture is still tracked");
  assertEquals(compositingHint(h.container), "");
  h.timers.run();
  assertEquals(h.overlay.gestureActive, false);
});

Deno.test("overlay: disabling disarms the gesture timer", () => {
  const h = harness();

  h.overlay.viewportChanged();
  h.overlay.setConfig({ enabled: false });

  assertEquals(h.timers.pending.size, 0);
  assertEquals(h.overlay.gestureActive, false);
  // A detached overlay does not arm timers it will never fire.
  h.overlay.viewportChanged();
  assertEquals(h.timers.pending.size, 0);
});

// -----------------------------------------------------------------------------
// Wheel forwarding
// -----------------------------------------------------------------------------

/** Dispatch a wheel over a card, the way a browser would deliver one. */
function wheelOverCard(h: Harness, card: HTMLElement, deltaY: number): Event {
  const event = new h.dom.CustomEvent("wheel", { bubbles: true, cancelable: true });
  Object.assign(event, {
    deltaX: 0,
    deltaY,
    deltaZ: 0,
    deltaMode: 0,
    clientX: 321,
    clientY: 123,
    ctrlKey: true,
    shiftKey: false,
    altKey: false,
    metaKey: false,
  });
  card.dispatchEvent(event);
  return event;
}

Deno.test("overlay: a wheel over a card reaches the canvas", () => {
  const h = harness({ minCardLifetimeMs: 0 });
  const received: WheelEvent[] = [];
  h.dom.canvas.addEventListener("wheel", (event) => received.push(event as WheelEvent));

  h.overlay.syncCards([entry(1, 1)]);
  const card = h.provider.containers.get(1);
  assert(card !== undefined);

  const source = wheelOverCard(h, card, -240);

  assertEquals(received.length, 1, "zooming back out from over a card must work");
  const forwarded = received[0];
  assertEquals(forwarded.type, "wheel");
  assertEquals(forwarded.deltaY, -240);
  // The canvas derives the zoom focus from the client coordinates, so a clone
  // that loses them zooms about the wrong point.
  assertEquals(forwarded.clientX, 321);
  assertEquals(forwarded.clientY, 123);
  assertEquals(forwarded.ctrlKey, true);
  assertEquals(source.defaultPrevented, true, "the page must not scroll behind the card");

  // The forwarded event must not bubble back into the overlay and loop.
  assertEquals(received.length, 1);
});

Deno.test("overlay: forwardWheel off leaves the wheel where it landed", () => {
  const h = harness({ minCardLifetimeMs: 0, forwardWheel: false });
  const received: Event[] = [];
  h.dom.canvas.addEventListener("wheel", (event) => received.push(event));

  h.overlay.syncCards([entry(1, 1)]);
  const card = h.provider.containers.get(1);
  assert(card !== undefined);
  const source = wheelOverCard(h, card, 100);

  assertEquals(received.length, 0);
  assertEquals(source.defaultPrevented, false);
});

// -----------------------------------------------------------------------------
// Teardown discipline (SC-004)
// -----------------------------------------------------------------------------

Deno.test("overlay: every teardown path releases through the driver", () => {
  // Eviction, an unrequested card, a disable, a dispose, a provider swap: each
  // removes a container, and each must run `provider.release` first, while the
  // container is still attached, so the consumer can tear its own state down.
  // `replaces` marks the one path that hands the element straight to another
  // card, which is the pool doing its job rather than a missing detach.
  const paths: readonly { name: string; replaces: boolean; run: (h: Harness) => void }[] = [
    { name: "unrequested", replaces: false, run: (h) => h.overlay.syncCards([]) },
    { name: "evicted", replaces: true, run: (h) => h.overlay.syncCards([entry(9, 100)]) },
    { name: "releaseAll", replaces: false, run: (h) => h.overlay.releaseAll() },
    { name: "disabled", replaces: false, run: (h) => h.overlay.setConfig({ enabled: false }) },
    { name: "disposed", replaces: false, run: (h) => h.overlay.dispose() },
    { name: "provider swap", replaces: false, run: (h) => h.overlay.setProvider(null) },
  ];

  for (const { name, replaces, run } of paths) {
    const h = harness({ maxCards: 1, minCardLifetimeMs: 0 });
    h.overlay.syncCards([entry(1, 1)]);
    const card = h.provider.containers.get(1);
    assert(card !== undefined, name);

    run(h);

    const releases = h.provider.calls.filter((c) => c.hook === "release");
    assertEquals(releases.length, 1, `${name}: released exactly once`);
    assertEquals(releases[0].nodeId, 1, name);
    assertEquals(releases[0].connected, true, `${name}: released before detach`);
    if (replaces) {
      // The release still ran before the element was handed on.
      const order = h.provider.calls.map((c) => `${c.hook}:${c.nodeId}`);
      assertEquals(order, ["mount:1", "release:1", "mount:9"], name);
      assertEquals(h.overlay.cardCount, 1, name);
    } else {
      assertEquals(card.isConnected, false, `${name}: detached afterwards`);
      assertEquals(h.overlay.cardCount, 0, name);
    }
  }
});

Deno.test("overlay: consumer content survives every core write (SC-004)", () => {
  const h = harness({ minCardLifetimeMs: 0 });
  h.overlay.syncCards([entry(1, 1)]);
  const card = h.provider.containers.get(1);
  assert(card !== undefined);

  const content = '<p class="body">export function main() {}</p>';
  card.innerHTML = content;

  for (let frame = 0; frame < 5; frame++) {
    h.graph.positions.set(1, { x: frame * 13, y: frame * -7 });
    h.viewport = viewportState({ scale: 1 + frame, x: frame * 100 });
    h.overlay.syncFrame();
  }
  h.overlay.syncCards([entry(1, 0.5, { opacity: 0.25 })]);
  h.overlay.syncFrame();

  assertEquals(card.innerHTML, content, "the overlay reaches card children through no path");
  assertEquals(card.style.opacity, "0.25", "...while core's own properties did move");
});

Deno.test("overlay: syncing before the overlay is enabled is inert", () => {
  const dom = createDom();
  const provider = recorder();
  const overlay = new DomCardOverlay({
    canvas: dom.canvas,
    viewport: () => viewportState(),
    nodes: new Graph([1]),
  });
  overlay.setProvider(provider);

  overlay.syncCards([entry(1, 1)]);
  overlay.syncFrame();
  overlay.viewportChanged();
  overlay.beginEpoch();
  overlay.releaseAll();

  assertEquals(provider.calls, []);
  assertEquals(overlay.cardCount, 0);
  assertEquals(dom.host.children.length, 1, "only the canvas");
});
