/**
 * Hover tracking: coalescing, the edge-scan gate, and transition ordering.
 *
 * `HoverTracker` is the whole of `GraphMother`'s hover path — the graph wires
 * its hit tests, its viewport transform, its event emitter and its GPU hover
 * writes into `HoverTargets` and adds nothing else. Both scans are O(N) brute
 * force, so the two properties pinned here are the ones that keep pointer
 * motion affordable at scale: one evaluation per frame rather than per event,
 * and no edge scan at all when nothing observes edge hover.
 *
 * The emitter-driven test wires a real `EventEmitter` exactly as `graph.ts`
 * does, so the subscriber predicate itself is under test and not a stand-in.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  defaultHoverScheduler,
  type HoverTargets,
  HoverTracker,
} from "../../packages/core/src/interaction/hover.ts";
import { EventEmitter } from "../../packages/core/src/events/emitter.ts";
import type { EdgeId, NodeId, Vec2 } from "../../packages/core/src/types.ts";

interface Probe {
  targets: HoverTargets;
  /** Every position `nodeAt` was asked about. */
  nodeScans: [number, number][];
  /** Every position `edgeAt` was asked about — empty means the scan never ran. */
  edgeScans: [number, number][];
  transitions: string[];
  /** Screen position -> node, consulted by `nodeAt`. */
  nodeHits: Map<string, NodeId>;
  /** Screen position -> edge, consulted by `edgeAt`. */
  edgeHits: Map<string, EdgeId>;
  edgeHoverWanted: boolean;
}

function probe(): Probe {
  const p: Partial<Probe> = {
    nodeScans: [],
    edgeScans: [],
    transitions: [],
    nodeHits: new Map(),
    edgeHits: new Map(),
    edgeHoverWanted: true,
  };
  p.targets = {
    nodeAt: (x, y) => {
      p.nodeScans!.push([x, y]);
      return p.nodeHits!.get(`${x},${y}`) ?? null;
    },
    edgeAt: (x, y) => {
      p.edgeScans!.push([x, y]);
      return p.edgeHits!.get(`${x},${y}`) ?? null;
    },
    toGraph: (x, y): Vec2 => ({ x: x * 2, y: y * 2 }),
    edgeHoverWanted: () => p.edgeHoverWanted!,
    onNodeEnter: (nodeId, position) =>
      p.transitions!.push(`node:enter ${nodeId} @${position.x},${position.y}`),
    onNodeLeave: (nodeId) => p.transitions!.push(`node:leave ${nodeId}`),
    onEdgeEnter: (edgeId, position) =>
      p.transitions!.push(`edge:enter ${edgeId} @${position.x},${position.y}`),
    onEdgeLeave: (edgeId) => p.transitions!.push(`edge:leave ${edgeId}`),
  };
  return p as Probe;
}

/** Scheduler standing in for the frame clock; `flush` is the frame boundary. */
function manualScheduler() {
  let queued: (() => void)[] = [];
  return {
    schedule: (run: () => void) => {
      queued.push(run);
    },
    get depth(): number {
      return queued.length;
    },
    flush(): void {
      const pending = queued;
      queued = [];
      for (const run of pending) run();
    },
  };
}

Deno.test("hover: a burst of moves costs one scan, at the last position", () => {
  const p = probe();
  const clock = manualScheduler();
  const tracker = new HoverTracker(p.targets, clock.schedule);

  for (let i = 0; i < 20; i++) tracker.move(i, i * 2);

  assertEquals(p.nodeScans.length, 0, "nothing evaluates before the frame");
  assertEquals(clock.depth, 1, "one deferred evaluation, not twenty");

  clock.flush();
  assertEquals(p.nodeScans, [[19, 38]], "the last position is the one under the pointer");
  assertEquals(p.edgeScans, [[19, 38]]);

  // The next burst schedules again — coalescing is per frame, not once ever.
  tracker.move(100, 100);
  tracker.move(101, 101);
  clock.flush();
  assertEquals(p.nodeScans.length, 2);
  assertEquals(p.nodeScans[1], [101, 101]);
});

Deno.test("hover: with no frame clock the default scheduler evaluates inline", () => {
  assertEquals(
    typeof (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame,
    "undefined",
    "this test is only meaningful where rAF is absent",
  );

  const p = probe();
  const tracker = new HoverTracker(p.targets, defaultHoverScheduler);
  p.nodeHits.set("5,5", 7);

  tracker.move(5, 5);

  assertEquals(p.nodeScans, [[5, 5]], "headless embedders still get hover");
  assertEquals(tracker.node, 7);
  assertEquals(p.transitions, ["node:enter 7 @10,10"]);
});

Deno.test("hover: no edge:hover subscribers means no edge scan", () => {
  const p = probe();
  p.edgeHoverWanted = false;
  p.edgeHits.set("5,5", 3);
  const clock = manualScheduler();
  const tracker = new HoverTracker(p.targets, clock.schedule);

  for (const [x, y] of [[5, 5], [6, 6], [7, 7]]) {
    tracker.move(x, y);
    clock.flush();
  }

  assertEquals(p.edgeScans, [], "the edge scan never ran");
  assertEquals(p.nodeScans.length, 3, "node hover is unaffected");
  assertEquals(tracker.edge, null);
  assertEquals(p.transitions, []);
});

Deno.test("hover: a node under the pointer suppresses the edge scan", () => {
  const p = probe();
  p.nodeHits.set("5,5", 2);
  p.edgeHits.set("5,5", 9);
  const clock = manualScheduler();
  const tracker = new HoverTracker(p.targets, clock.schedule);

  tracker.move(5, 5);
  clock.flush();

  assertEquals(p.edgeScans, [], "edges are occluded by the node");
  assertEquals(tracker.node, 2);
  assertEquals(tracker.edge, null);
});

Deno.test("hover: transitions fire once per change, leave before enter", () => {
  const p = probe();
  p.nodeHits.set("1,1", 4);
  p.nodeHits.set("2,2", 5);
  p.edgeHits.set("3,3", 8);
  const clock = manualScheduler();
  const tracker = new HoverTracker(p.targets, clock.schedule);

  const step = (x: number, y: number) => {
    tracker.move(x, y);
    clock.flush();
  };

  step(1, 1); // enter node 4
  step(1, 1); // unchanged
  step(2, 2); // leave 4, enter 5
  step(3, 3); // leave 5, enter edge 8
  step(4, 4); // leave edge 8

  assertEquals(p.transitions, [
    "node:enter 4 @2,2",
    "node:leave 4",
    "node:enter 5 @4,4",
    "node:leave 5",
    "edge:enter 8 @6,6",
    "edge:leave 8",
  ]);
  assertEquals(tracker.node, null);
  assertEquals(tracker.edge, null);
});

Deno.test("hover: a leave handler still reads the target it is leaving", () => {
  // GraphMother emits node:hoverleave from onNodeLeave, and a handler calling
  // getHoveredNode() there must see the node being left, not the new one.
  const p = probe();
  const clock = manualScheduler();
  const observed: (NodeId | null | "before construction")[] = [];
  let constructed: HoverTracker | null = null;
  const readNode = () =>
    observed.push(constructed === null ? "before construction" : constructed.node);
  const tracker = new HoverTracker({
    ...p.targets,
    onNodeEnter: readNode,
    onNodeLeave: readNode,
  }, clock.schedule);
  constructed = tracker;

  p.nodeHits.set("1,1", 4);
  p.nodeHits.set("2,2", 5);

  tracker.move(1, 1);
  clock.flush();
  tracker.move(2, 2);
  clock.flush();

  assertEquals(observed, [4, 4, 5], "enter 4, leave 4 (still 4), enter 5");
});

Deno.test("hover: hovering a node clears an existing edge hover", () => {
  const p = probe();
  p.edgeHits.set("1,1", 8);
  p.nodeHits.set("2,2", 4);
  const clock = manualScheduler();
  const tracker = new HoverTracker(p.targets, clock.schedule);

  tracker.move(1, 1);
  clock.flush();
  assertEquals(tracker.edge, 8);

  tracker.move(2, 2);
  clock.flush();
  assertEquals(tracker.edge, null);
  assertEquals(tracker.node, 4);
  assertEquals(p.transitions.slice(1), ["node:enter 4 @4,4", "edge:leave 8"]);
});

Deno.test("hover: reset drops state and any pending evaluation", () => {
  const p = probe();
  p.nodeHits.set("1,1", 4);
  const clock = manualScheduler();
  const tracker = new HoverTracker(p.targets, clock.schedule);

  tracker.move(1, 1);
  clock.flush();
  assertEquals(tracker.node, 4);

  tracker.move(9, 9);
  tracker.reset();
  clock.flush();

  assertEquals(tracker.node, null, "state dropped without emitting a leave");
  assertEquals(p.nodeScans.length, 1, "the pending evaluation was abandoned");
  assertEquals(p.transitions, ["node:enter 4 @2,2"]);
});

Deno.test("hover: subscribing to edge:hover re-enables the scan and the events", () => {
  // Wired exactly as GraphMother wires it.
  const events = new EventEmitter();
  const p = probe();
  const clock = manualScheduler();
  const targets: HoverTargets = {
    ...p.targets,
    edgeHoverWanted: () =>
      events.hasListeners("edge:hoverenter") || events.hasListeners("edge:hoverleave"),
    onEdgeEnter: (edgeId, position) =>
      events.emit({ type: "edge:hoverenter", timestamp: 0, edgeId, position }),
    onEdgeLeave: (edgeId) => events.emit({ type: "edge:hoverleave", timestamp: 0, edgeId }),
  };
  const tracker = new HoverTracker(targets, clock.schedule);
  p.edgeHits.set("1,1", 8);

  const step = (x: number, y: number) => {
    tracker.move(x, y);
    clock.flush();
  };

  step(1, 1);
  assertEquals(p.edgeScans, [], "unsubscribed: no scan");

  const seen: string[] = [];
  const unsubscribe = events.on("edge:hoverenter", (e) => seen.push(`enter ${e.edgeId}`));
  events.on("edge:hoverleave", (e) => seen.push(`leave ${e.edgeId}`));

  step(1, 1);
  assertEquals(p.edgeScans.length, 1, "subscribed: the scan runs");
  assertEquals(seen, ["enter 8"]);

  step(2, 2);
  assertEquals(seen, ["enter 8", "leave 8"]);

  unsubscribe();
  events.removeAllListeners("edge:hoverleave");
  const scansAfterUnsubscribe = p.edgeScans.length;
  step(1, 1);
  assertEquals(p.edgeScans.length, scansAfterUnsubscribe, "unsubscribed again: no scan");

  // A one-shot listener is a subscriber until it fires.
  events.once("edge:hoverenter", (e) => seen.push(`once ${e.edgeId}`));
  step(1, 1);
  assert(p.edgeScans.length > scansAfterUnsubscribe, "a once-listener re-enables the scan");
  assertEquals(seen, ["enter 8", "leave 8", "once 8"]);
});
