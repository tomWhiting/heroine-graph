/**
 * Card pointer routing and the card drag handle.
 *
 * The overlay sits on top of the canvas, so every pointer gesture has to end
 * up in exactly one of two places and never both. What is pinned here:
 *
 *  - the container is transparent to hit-testing and cards are not, so a pan
 *    started on empty overlay area reaches the canvas while a gesture on a
 *    card does not — and the overlay forwards *no* pointer event to the
 *    canvas, where it deliberately does forward the wheel;
 *  - a gesture on card text is left alone, so selection, links and find-in-page
 *    keep working; only a provider-declared handle drags;
 *  - a handle drag is a node drag: it drives the same {@link NodeDragController}
 *    the canvas gesture drives, so the pin, the position writes and the
 *    `node:drag*` stream are identical rather than merely similar, and the node
 *    is left pinned at drag end.
 *
 * linkedom supplies the DOM, as in `dom_overlay_test.ts`. Real hit-testing is
 * the browser's and is not simulated: what a shim can prove is the discipline
 * (which elements accept pointer events, and what the overlay does with the
 * ones it receives), and that is what is asserted.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import { parseHTML } from "https://esm.sh/linkedom@0.18.11/worker";
import {
  CARD_DRAG_HANDLE_ATTRIBUTE,
  type CardNodeSource,
  DomCardOverlay,
} from "../../packages/core/src/overlay/overlay.ts";
import type { CardNode, CardProvider } from "../../packages/core/src/overlay/types.ts";
import {
  NodeDragController,
  type NodeDragHost,
} from "../../packages/core/src/interaction/node_drag.ts";
import type { NodeId, Vec2, ViewportState } from "../../packages/core/src/types.ts";

// -----------------------------------------------------------------------------
// Harness
// -----------------------------------------------------------------------------

/** One `node:drag*` event, flattened to what the two entry points must agree on. */
interface DragEvent {
  readonly type: string;
  readonly nodeId: NodeId;
  readonly position: Vec2;
  readonly delta?: Vec2;
}

/** The graph a drag writes into, recording everything the gesture caused. */
class DragHost implements NodeDragHost {
  readonly events: DragEvent[] = [];
  readonly positions = new Map<NodeId, Vec2>();
  readonly pinned = new Set<NodeId>();
  clock = 1000;

  setPinned(node: NodeId, pinned: boolean): void {
    if (pinned) this.pinned.add(node);
    else this.pinned.delete(node);
  }

  setPosition(node: NodeId, x: number, y: number): void {
    this.positions.set(node, { x, y });
  }

  emit(event: DragEvent): void {
    this.events.push({
      type: event.type,
      nodeId: event.nodeId,
      position: { ...event.position },
      ...(event.delta === undefined ? {} : { delta: { ...event.delta } }),
    });
  }

  now(): number {
    return this.clock;
  }
}

interface Dom {
  readonly document: Document;
  readonly host: HTMLElement;
  readonly canvas: HTMLCanvasElement;
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

interface CardParts {
  readonly handle: HTMLElement;
  readonly body: HTMLElement;
}

/** A provider whose cards have a title-bar handle and selectable body text. */
function provider(parts: Map<NodeId, CardParts>): CardProvider<null> {
  return {
    mount(container: HTMLElement, node: CardNode): null {
      const document = container.ownerDocument;
      const handle = document.createElement("div");
      handle.setAttribute(CARD_DRAG_HANDLE_ATTRIBUTE, "");
      const title = document.createElement("span");
      title.textContent = String(node.externalId);
      handle.appendChild(title);
      const body = document.createElement("p");
      body.textContent = "selectable";
      container.appendChild(handle);
      container.appendChild(body);
      parts.set(node.id, { handle, body });
      return null;
    },
    release(container: HTMLElement): void {
      while (container.firstChild !== null) container.firstChild.remove();
    },
  };
}

class Graph implements CardNodeSource {
  readonly positions = new Map<NodeId, Vec2>();
  readonly pinned = new Set<NodeId>();

  constructor(nodeIds: readonly NodeId[]) {
    for (const id of nodeIds) this.positions.set(id, { x: id * 100, y: id * -50 });
  }

  externalId(node: NodeId): string {
    return `src/mod${node}.ts`;
  }
  label(node: NodeId): string {
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
  contentRef(): undefined {
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
  readonly graph: Graph;
  readonly drag: DragHost;
  readonly parts: Map<NodeId, CardParts>;
  readonly container: HTMLElement;
  /** Pointer events the canvas saw, by type. */
  readonly canvasEvents: string[];
  viewport: ViewportState;
}

function harness(scale = 1): Harness {
  const dom = createDom();
  const graph = new Graph([1, 2]);
  const parts = new Map<NodeId, CardParts>();
  const drag = new DragHost();
  const state = {
    viewport: {
      x: 0,
      y: 0,
      scale,
      width: 800,
      height: 600,
      minScale: 0.01,
      maxScale: 100,
    } as ViewportState,
  };

  // The graph wires its own controller in as the sink; the test wires the same
  // class, so what is asserted here is the path the product runs.
  const overlay = new DomCardOverlay({
    canvas: dom.canvas,
    viewport: () => state.viewport,
    nodes: graph,
    drag: new NodeDragController({
      setPinned: (node, pinned) => {
        drag.setPinned(node, pinned);
        graph.setPinned(node, pinned);
      },
      setPosition: (node, x, y) => {
        drag.setPosition(node, x, y);
        // The real `setNodePosition` writes the position shadow the overlay
        // re-places cards from, and pins as it goes.
        graph.positions.set(node, { x, y });
        graph.setPinned(node, true);
      },
      emit: (event) => drag.emit(event),
      now: () => drag.now(),
    }),
    minCardLifetimeMs: 0,
    now: () => 0,
  });
  overlay.setProvider(provider(parts));
  overlay.setConfig({ enabled: true });

  const container = overlay.element;
  assert(container !== null);

  const canvasEvents: string[] = [];
  for (const type of ["pointerdown", "pointermove", "pointerup", "wheel", "click"]) {
    dom.canvas.addEventListener(type, (event) => canvasEvents.push(event.type));
  }

  return {
    dom,
    overlay,
    graph,
    drag,
    parts,
    container,
    canvasEvents,
    get viewport() {
      return state.viewport;
    },
    set viewport(value: ViewportState) {
      state.viewport = value;
    },
  };
}

function pointer(
  h: Harness,
  target: HTMLElement,
  type: string,
  clientX: number,
  clientY: number,
  overrides: Record<string, unknown> = {},
): Event {
  const event = new h.dom.CustomEvent(type, { bubbles: true, cancelable: true });
  Object.assign(event, { pointerId: 7, button: 0, clientX, clientY, ...overrides });
  target.dispatchEvent(event);
  return event;
}

function card(h: Harness, node: NodeId): CardParts {
  const parts = h.parts.get(node);
  assert(parts !== undefined, `node ${node} must be carded`);
  return parts;
}

function mount(h: Harness, node: NodeId): CardParts {
  h.overlay.syncCards([{ node, priority: 1 }]);
  return card(h, node);
}

function types(events: readonly DragEvent[]): string[] {
  return events.map((event) => event.type);
}

// -----------------------------------------------------------------------------
// Routing discipline
// -----------------------------------------------------------------------------

Deno.test("card pointer: the container is transparent to events and cards are not", () => {
  // This is the whole pass-through mechanism: a pan or a click on empty
  // overlay area is hit-tested straight through to the canvas beneath, and a
  // card is the one thing that stops it.
  const h = harness();
  const parts = mount(h, 1);

  assertEquals(h.container.style.pointerEvents, "none");
  assertEquals((parts.handle.parentElement as HTMLElement).style.pointerEvents, "auto");
});

Deno.test("card pointer: no pointer event on a card is forwarded to the canvas", () => {
  // A click on a card must not also hit-test the canvas underneath it — that
  // would select whatever node happens to be behind the card, or start a pan.
  // The wheel is the deliberate exception, and the contrast is the discipline.
  const h = harness();
  const parts = mount(h, 1);

  pointer(h, parts.handle, "pointerdown", 100, 100);
  pointer(h, parts.handle, "pointermove", 140, 100);
  pointer(h, parts.handle, "pointerup", 140, 100);
  pointer(h, parts.body, "pointerdown", 100, 140);
  pointer(h, parts.body, "pointerup", 100, 140);

  assertEquals(h.canvasEvents, [], "pointer events on a card stay on the card");

  const wheel = new h.dom.CustomEvent("wheel", { bubbles: true, cancelable: true });
  Object.assign(wheel, {
    deltaX: 0,
    deltaY: -120,
    deltaZ: 0,
    deltaMode: 0,
    clientX: 1,
    clientY: 2,
  });
  parts.body.dispatchEvent(wheel);

  assertEquals(h.canvasEvents, ["wheel"], "zooming out from over a card must still work");
});

Deno.test("card pointer: a gesture on card text neither drags nor is swallowed", () => {
  // Card text has to stay selectable (SC-003), so a press on it must reach the
  // browser's own selection machinery with its default intact.
  const h = harness();
  const parts = mount(h, 1);

  const event = pointer(h, parts.body, "pointerdown", 100, 100);
  pointer(h, parts.body, "pointermove", 160, 100);

  assertEquals(h.overlay.draggingNode, null, "only a declared handle drags");
  assertEquals(event.defaultPrevented, false, "text selection must be left alone");
  assertEquals(h.drag.events, []);
  assertEquals(h.graph.positions.get(1), { x: 100, y: -50 }, "the node must not have moved");
});

Deno.test("card pointer: a gesture on empty overlay area is ignored by the overlay", () => {
  // It never reaches the overlay in a browser — the container is
  // pointer-events:none — but the delegated listener sits on the container, so
  // a synthesised one must still find no card and do nothing.
  const h = harness();
  mount(h, 1);

  const event = pointer(h, h.container, "pointerdown", 400, 300);

  assertEquals(h.overlay.draggingNode, null);
  assertEquals(event.defaultPrevented, false, "the canvas must keep the gesture's default");
  assertEquals(h.drag.events, []);
});

// -----------------------------------------------------------------------------
// The drag handle
// -----------------------------------------------------------------------------

Deno.test("card pointer: dragging a handle moves the node in graph units", () => {
  // Client pixels divide by the camera scale, and the grab point is preserved
  // rather than centring the node under the pointer — a card picked up by its
  // title bar must not jump.
  const h = harness(2);
  const parts = mount(h, 1);

  pointer(h, parts.handle, "pointerdown", 300, 200);
  pointer(h, parts.handle, "pointermove", 360, 180);

  assertEquals(h.graph.positions.get(1), { x: 130, y: -60 }, "60px right, 20px up at 2x zoom");

  pointer(h, parts.handle, "pointermove", 300, 200);
  assertEquals(h.graph.positions.get(1), { x: 100, y: -50 }, "back to where it was grabbed");
});

Deno.test("card pointer: a handle drag pins the node and leaves it pinned", () => {
  const h = harness();
  const parts = mount(h, 1);
  assert(!h.graph.isPinned(1), "the fixture must start unpinned");

  const down = pointer(h, parts.handle, "pointerdown", 300, 200);
  assertEquals(down.defaultPrevented, true, "the browser must not start its own drag");
  assert(h.graph.isPinned(1), "the pin is taken as the gesture starts, before any move");
  assertEquals(h.overlay.draggingNode, 1);

  pointer(h, parts.handle, "pointermove", 330, 200);
  pointer(h, parts.handle, "pointerup", 330, 200);

  assertEquals(h.overlay.draggingNode, null);
  assert(h.graph.isPinned(1), "a dropped node stays where it was dropped");
  assertEquals(types(h.drag.events), ["node:dragstart", "node:dragmove", "node:dragend"]);
});

Deno.test("card pointer: releasing a card mid-drag ends the drag", () => {
  // The handle is about to leave the DOM, so no pointerup can ever arrive; a
  // gesture left in flight would pin the graph's drag state open.
  const h = harness();
  const parts = mount(h, 1);

  pointer(h, parts.handle, "pointerdown", 300, 200);
  pointer(h, parts.handle, "pointermove", 340, 200);
  h.overlay.syncCards([]);

  assertEquals(h.overlay.draggingNode, null);
  assertEquals(types(h.drag.events), ["node:dragstart", "node:dragmove", "node:dragend"]);
  assert(h.graph.isPinned(1));
});

Deno.test("card pointer: a second card's handle cannot drive the first card's drag", () => {
  const h = harness();
  const first = mount(h, 1);
  h.overlay.syncCards([{ node: 1, priority: 2 }, { node: 2, priority: 1 }]);
  const second = card(h, 2);

  pointer(h, first.handle, "pointerdown", 300, 200);
  pointer(h, second.handle, "pointermove", 400, 200, { pointerId: 9 });

  assertEquals(h.overlay.draggingNode, 1);
  assertEquals(types(h.drag.events), ["node:dragstart"], "a foreign pointer moves nothing");
});

// -----------------------------------------------------------------------------
// Parity with the canvas node drag
// -----------------------------------------------------------------------------

Deno.test("node drag: a card drag and a canvas drag produce the same events and state", () => {
  // The requirement is not "similar to" but "the same as": both entry points
  // drive one controller, so the comparison is a real equality rather than two
  // implementations that happen to agree today.
  const canvas = new DragHost();
  const controller = new NodeDragController(canvas);
  controller.begin(1, { x: 10, y: 20 });
  controller.move(1, { x: 15, y: 20 });
  controller.end(1, { x: 15, y: 20 });

  const h = harness();
  const parts = mount(h, 1);
  h.graph.positions.set(1, { x: 10, y: 20 });
  pointer(h, parts.handle, "pointerdown", 0, 0);
  pointer(h, parts.handle, "pointermove", 5, 0);
  pointer(h, parts.handle, "pointerup", 5, 0);

  assertEquals(h.drag.events, canvas.events);
  assertEquals(h.drag.positions.get(1), canvas.positions.get(1));
  assertEquals([...h.drag.pinned], [...canvas.pinned]);
  assertEquals([...canvas.pinned], [1], "both leave the node pinned");
});

Deno.test("node drag: a press that turns out to be a click emits no dragend", () => {
  // The canvas path cancels rather than ending when the pointer barely moved,
  // and undoes the pin itself — a click is not a zero-length drag.
  const host = new DragHost();
  const controller = new NodeDragController(host);

  controller.begin(3, { x: 0, y: 0 });
  controller.cancel();
  host.setPinned(3, false);

  assertEquals(types(host.events), ["node:dragstart"]);
  assertEquals(controller.node, null);
  assertEquals([...host.pinned], []);
});

Deno.test("node drag: an abandoned gesture is ended before the next one begins", () => {
  const host = new DragHost();
  const controller = new NodeDragController(host);

  controller.begin(1, { x: 0, y: 0 });
  controller.move(1, { x: 4, y: 0 });
  controller.begin(2, { x: 9, y: 9 });

  assertEquals(types(host.events), [
    "node:dragstart",
    "node:dragmove",
    "node:dragend",
    "node:dragstart",
  ]);
  assertEquals(host.events[2].nodeId, 1);
  assertEquals(host.events[2].position, { x: 4, y: 0 }, "it ends where it was last seen");
  assertEquals(controller.node, 2);
});

Deno.test("node drag: events for a node that is not being dragged are dropped", () => {
  const host = new DragHost();
  const controller = new NodeDragController(host);

  controller.move(1, { x: 1, y: 1 });
  controller.end(1, { x: 1, y: 1 });
  controller.begin(1, { x: 0, y: 0 });
  controller.move(2, { x: 5, y: 5 });
  controller.end(2, { x: 5, y: 5 });

  assertEquals(types(host.events), ["node:dragstart"]);
  assertEquals(controller.node, 1, "the live drag survives a foreign end");
  assertEquals(host.positions.size, 0);
});
