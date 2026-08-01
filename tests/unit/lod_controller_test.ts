/**
 * LODController contract tests.
 *
 * The controller is a pure state machine, so everything here is exact: the
 * visible cut is compared against an independently computed one, oscillation
 * is asserted to produce literally zero state changes, and the same viewport
 * sequence is required to produce byte-identical decision streams twice.
 *
 * The fake host is wrapped in a recording `Proxy`, which is what makes the
 * no-reheat rule testable rather than merely documented: it also carries
 * simulation-alpha methods the seam does not declare, so any route the
 * controller takes to one is visible as a property access.
 */

import { assert, assertEquals, assertNotEquals, assertStrictEquals } from "jsr:@std/assert@^1";
import {
  LOD_FORCE_CARD_PRIORITY,
  LODController,
  type LodHost,
  quantiseZoom,
} from "../../packages/core/src/lod/controller.ts";
import type { LodPolicy } from "../../packages/core/src/lod/policy.ts";
import { DEFAULT_LOD_CONFIG, resolveLodConfig } from "../../packages/core/src/lod/config.ts";
import {
  HIERARCHY_ROOT,
  type HierarchyColumns,
  type RetainedHierarchy,
  retainSuppliedHierarchy,
} from "../../packages/core/src/graph/hierarchy.ts";
import type { CardSyncEntry } from "../../packages/core/src/overlay/overlay.ts";
import type { NodeId, ViewportState } from "../../packages/core/src/types.ts";
import { computeVisibleCut } from "../helpers/invariants.ts";
import { CODE_TREE_SCALES, generateCodeTree } from "../fixtures/code_tree.ts";

// =============================================================================
// Fixtures
// =============================================================================

const VIEWPORT: ViewportState = {
  x: 0,
  y: 0,
  scale: 1,
  width: 1200,
  height: 800,
  minScale: 0.01,
  maxScale: 100,
};

/**
 * Build a validated hierarchy from `parent[]`, giving each node a well radius
 * from `radiusOf`. Depths and subtree sizes are derived, so the fixture cannot
 * disagree with itself.
 */
function hierarchyOf(
  parentList: readonly number[],
  radiusOf: (slot: number, subtreeSize: number) => number,
): RetainedHierarchy {
  const n = parentList.length;
  const parent = new Uint32Array(n);
  const depth = new Uint16Array(n);
  const subtreeSize = new Uint32Array(n).fill(1);
  for (let i = 0; i < n; i++) {
    parent[i] = parentList[i] < 0 ? HIERARCHY_ROOT : parentList[i];
  }
  // Parents precede children in every fixture here, so one descending pass
  // accumulates sizes and one ascending pass fills depths.
  for (let i = 0; i < n; i++) {
    if (parent[i] !== HIERARCHY_ROOT) depth[i] = depth[parent[i]] + 1;
  }
  for (let i = n - 1; i >= 0; i--) {
    if (parent[i] !== HIERARCHY_ROOT) subtreeSize[parent[i]] += subtreeSize[i];
  }
  const wellRadius = new Float32Array(n);
  for (let i = 0; i < n; i++) wellRadius[i] = radiusOf(i, subtreeSize[i]);

  const columns: HierarchyColumns = { parent, wellRadius, depth, subtreeSize };
  return retainSuppliedHierarchy(columns, n);
}

/** The code-tree fixture as a hierarchy; radius grows with subtree size. */
function codeTreeHierarchy(): RetainedHierarchy {
  const tree = generateCodeTree(CODE_TREE_SCALES.small);
  return hierarchyOf(
    Array.from(tree.parent),
    (_slot, subtreeSize) => Math.sqrt(subtreeSize) * 4,
  );
}

// =============================================================================
// Recording host
// =============================================================================

interface VisibilityWrite {
  lo: number;
  hi: number;
  hidden: number[];
}

/** One `setCollapsedProxies` call, flattened to slot/radius pairs. */
interface ProxyWrite {
  slots: number[];
  radii: number[];
}

/** One `translateNodeRange` call. */
interface TranslateWrite {
  lo: number;
  hi: number;
  dx: number;
  dy: number;
}

/** Names that would reheat the simulation. None may ever be touched. */
const REHEAT_NAMES = [
  "setAlpha",
  "setAlphaTarget",
  "setAlphaDecay",
  "bumpSimulationAlpha",
  "simulationController",
  "restartSimulation",
] as const;

/**
 * The entire declared {@link LodHost} surface.
 *
 * The controller may read nothing else off its host. Kept as a literal rather
 * than derived from the type — the point is to notice when the seam grows, and
 * a list generated from the seam could not.
 */
const HOST_SURFACE: readonly string[] = [
  "getHierarchy",
  "getViewport",
  "getNodePosition",
  "getNodeRadius",
  "getNodeTag",
  "getNodeWeight",
  "applyVisibility",
  "uploadNodeAlpha",
  "uploadNodeMass",
  "setCollapsedProxies",
  "aggregateEdges",
  "releaseEdgeAggregation",
  "translateNodeRange",
  "syncCards",
  "emit",
];

interface Recorder {
  host: LodHost;
  /** Swappable, so a test can hand the controller a rebuilt hierarchy. */
  hierarchy: RetainedHierarchy | null;
  visibility: VisibilityWrite[];
  cards: CardSyncEntry[][];
  events: string[];
  /** Every property name read off the host, in order. */
  accessed: string[];
  reheatCalls: number;
  viewport: { x: number; y: number; scale: number };
  radiusOf: (node: NodeId) => number;
  weightOf: (node: NodeId) => number;
  positionOf: (node: NodeId) => { x: number; y: number };
  /** Every mass array handed over, copied — plus the identities, uncopied. */
  mass: Float32Array[];
  massIdentities: Float32Array[];
  proxies: ProxyWrite[];
  /** Every visible mask handed to `aggregateEdges`, copied. */
  aggregations: Uint8Array[];
  /** How many times the aggregation was released. */
  aggregationReleases: number;
  translations: TranslateWrite[];
  /** Host method names in call order, for claims about relative ordering. */
  calls: string[];
}

/**
 * A fake graph seam that records everything the controller does to it.
 *
 * The object deliberately carries the reheat methods the {@link LodHost}
 * interface does not declare: an implementation that reaches for one is only
 * observable if the target exists.
 */
function recorder(hierarchy: RetainedHierarchy | null): Recorder {
  const state: Recorder = {
    host: null as unknown as LodHost,
    hierarchy,
    visibility: [],
    cards: [],
    events: [],
    accessed: [],
    reheatCalls: 0,
    viewport: { x: VIEWPORT.x, y: VIEWPORT.y, scale: VIEWPORT.scale },
    radiusOf: () => 1,
    weightOf: () => 0,
    positionOf: () => ({ x: 0, y: 0 }),
    mass: [],
    massIdentities: [],
    proxies: [],
    aggregations: [],
    aggregationReleases: 0,
    translations: [],
    calls: [],
  };

  const reheat = () => {
    state.reheatCalls++;
  };

  const target = {
    getHierarchy: () => state.hierarchy,
    getViewport: (): ViewportState => ({ ...VIEWPORT, ...state.viewport }),
    getNodePosition: (node: NodeId) => state.positionOf(node),
    getNodeRadius: (node: NodeId) => state.radiusOf(node),
    getNodeTag: () => 0,
    getNodeWeight: (node: NodeId) => state.weightOf(node),
    applyVisibility: (lo: number, hi: number, visible: Uint8Array) => {
      const hidden: number[] = [];
      for (let slot = lo; slot < hi; slot++) {
        if (visible[slot] === 0) hidden.push(slot);
      }
      state.visibility.push({ lo, hi, hidden });
      state.calls.push("applyVisibility");
    },
    uploadNodeAlpha: () => {},
    uploadNodeMass: (mass: Float32Array) => {
      state.mass.push(mass.slice());
      state.massIdentities.push(mass);
      state.calls.push("uploadNodeMass");
    },
    setCollapsedProxies: (proxies: Uint32Array, radii: Float32Array) => {
      state.proxies.push({ slots: Array.from(proxies), radii: Array.from(radii) });
    },
    aggregateEdges: (visible: Uint8Array) => {
      state.aggregations.push(visible.slice());
      state.calls.push("aggregateEdges");
    },
    releaseEdgeAggregation: () => {
      state.aggregationReleases++;
      state.calls.push("releaseEdgeAggregation");
    },
    translateNodeRange: (lo: number, hi: number, dx: number, dy: number) => {
      state.translations.push({ lo, hi, dx, dy });
      state.calls.push("translateNodeRange");
    },
    syncCards: (entries: readonly CardSyncEntry[]) => {
      state.cards.push([...entries]);
    },
    emit: (event: { type: string; nodeId?: NodeId; reason?: string; visibleCount?: number }) => {
      state.events.push(
        event.type === "lod:change"
          ? `lod:change/${event.visibleCount}`
          : `${event.type}/${event.nodeId}/${event.reason}`,
      );
    },
    setAlpha: reheat,
    setAlphaTarget: reheat,
    setAlphaDecay: reheat,
    bumpSimulationAlpha: reheat,
    simulationController: { setAlpha: reheat },
    restartSimulation: reheat,
  };

  state.host = new Proxy(target, {
    get(object, property, receiver) {
      if (typeof property === "string") state.accessed.push(property);
      return Reflect.get(object, property, receiver);
    },
  }) as unknown as LodHost;

  return state;
}

/**
 * Assert the controller took no route to simulation alpha.
 *
 * Two claims, and the second is the durable one: nothing named a reheat was
 * touched, *and* nothing outside the declared seam was touched at all — so a
 * future route to alpha through some new host method fails here even though
 * this list has never heard of it.
 */
function assertNoReheat(log: Recorder): void {
  const touched = log.accessed.filter((name) => (REHEAT_NAMES as readonly string[]).includes(name));
  assertEquals(touched, [], "LOD must never reach simulation alpha");
  assertEquals(log.reheatCalls, 0);

  const undeclared = [...new Set(log.accessed)].filter((name) => !HOST_SURFACE.includes(name));
  assertEquals(undeclared, [], "LOD must read nothing outside the declared LodHost surface");
}

interface Rig {
  controller: LODController;
  log: Recorder;
}

function rig(
  hierarchy: RetainedHierarchy | null,
  config: Parameters<LODController["setConfig"]>[0] = {},
): Rig {
  const log = recorder(hierarchy);
  const controller = new LODController(log.host);
  controller.setConfig({ enabled: true, ...config }, 0);
  return { controller, log };
}

// =============================================================================
// Cut walk — the exact visible set at a table of zoom levels
// =============================================================================

const ZOOM_TABLE = [0.125, 0.25, 0.5, 1, 2, 4, 8] as const;

Deno.test("cut walk: a cold controller reproduces the geometric cut at every zoom", () => {
  const hierarchy = codeTreeHierarchy();
  const { parent, wellRadius } = hierarchy.columns;

  for (const zoom of ZOOM_TABLE) {
    const { controller, log } = rig(hierarchy);
    log.viewport.scale = zoom;
    controller.evaluateNow(0);

    const expected = computeVisibleCut({
      parent,
      wellRadius,
      zoom,
      expandThreshold: DEFAULT_LOD_CONFIG.expandThreshold,
    });
    assertEquals(
      Array.from(controller.getVisibleNodes()),
      Array.from(expected.visible),
      `visible set at zoom ${zoom}`,
    );
    assertNoReheat(log);
  }
});

Deno.test("cut walk: zooming in through the table tracks the cut step by step", () => {
  const hierarchy = codeTreeHierarchy();
  const { parent, wellRadius } = hierarchy.columns;
  const { controller, log } = rig(hierarchy);

  // Zoom only ever increases, so hysteresis never engages and the cut must
  // equal the geometric one at every step of a single controller's life.
  for (const zoom of ZOOM_TABLE) {
    log.viewport.scale = zoom;
    controller.viewportChanged();
    controller.tick(zoom * 100);

    const expected = computeVisibleCut({
      parent,
      wellRadius,
      zoom,
      expandThreshold: DEFAULT_LOD_CONFIG.expandThreshold,
    });
    assertEquals(
      Array.from(controller.getVisibleNodes()),
      Array.from(expected.visible),
      `visible set after stepping to zoom ${zoom}`,
    );
  }
  assertNoReheat(log);
});

Deno.test("cut walk: hidden nodes are flagged and visible ones cleared", () => {
  // Root with two children; the left one has two of its own.
  const hierarchy = hierarchyOf([-1, 0, 0, 1, 1], (slot) => (slot === 0 ? 400 : 40));
  const { controller, log } = rig(hierarchy);

  log.viewport.scale = 1;
  controller.evaluateNow(0);

  // Root expands (400 >= 96); node 1 does not (40 < 96), so 3 and 4 are hidden.
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2]);
  assert(controller.isCollapsed(1));
  assertEquals(controller.getVisibleAncestor(3), 1);
  assertEquals(controller.getVisibleAncestor(2), 2);

  // The fade has to land before the flag does, or the crossfade is invisible.
  const beforeFade = log.visibility.flatMap((write) => write.hidden);
  assertEquals(beforeFade, []);
  controller.tick(DEFAULT_LOD_CONFIG.transitionMs);
  const afterFade = log.visibility.flatMap((write) => write.hidden);
  assertEquals(afterFade, [3, 4]);
});

// =============================================================================
// Oscillation
// =============================================================================

Deno.test("oscillation: +/-5% zoom jitter across a band boundary changes nothing", () => {
  // Node 1's extent is exactly the expand threshold at zoom 1, and the jitter
  // straddles a zoom quantum boundary, so nothing but hysteresis can hold it.
  assertNotEquals(quantiseZoom(0.95), quantiseZoom(1.05));
  const hierarchy = hierarchyOf([-1, 0, 0, 1, 1], (slot) => (slot === 0 ? 400 : 96));

  const { controller, log } = rig(hierarchy);
  log.viewport.scale = 1;
  controller.evaluateNow(0);

  const settled = Array.from(controller.getVisibleNodes());
  const eventsAtRest = log.events.length;

  for (let i = 0; i < 40; i++) {
    log.viewport.scale = i % 2 === 0 ? 0.95 : 1.05;
    controller.viewportChanged();
    controller.tick(1000 + i * 16);
    assertEquals(Array.from(controller.getVisibleNodes()), settled);
  }
  assertEquals(log.events.length, eventsAtRest, "jitter must produce no LOD events");
  assertNoReheat(log);
});

Deno.test("oscillation: jitter over the code tree produces no events at all", () => {
  const { controller, log } = rig(codeTreeHierarchy());
  log.viewport.scale = 1;
  controller.evaluateNow(0);

  const settled = Array.from(controller.getVisibleNodes());
  const eventsAtRest = log.events.length;
  for (let i = 0; i < 30; i++) {
    log.viewport.scale = i % 2 === 0 ? 0.95 : 1.05;
    controller.viewportChanged();
    controller.tick(1000 + i * 16);
  }
  assertEquals(log.events.length, eventsAtRest);
  assertEquals(Array.from(controller.getVisibleNodes()), settled);
});

// =============================================================================
// Committed lifetime
// =============================================================================

Deno.test("committed lifetime: a state change cannot revert inside the window", () => {
  const commitFrames = 6;
  const hierarchy = hierarchyOf([-1, 0, 0, 1, 1], (slot) => (slot === 0 ? 4000 : 100));
  const { controller, log } = rig(hierarchy, { minBandCommitFrames: commitFrames });

  // Frame 1: extent 100 >= 96, node 1 expands and 3/4 enter the cut.
  log.viewport.scale = 1;
  controller.viewportChanged();
  controller.tick(0);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2, 3, 4]);
  const commitFrame = controller.frame;

  // The metric crosses all the way back: extent 25, well under the exit
  // threshold of 64. It must still be refused for the whole window.
  log.viewport.scale = 0.25;
  controller.viewportChanged();
  for (let elapsed = 1; elapsed < commitFrames; elapsed++) {
    controller.tick(controller.frame * 16);
    assertEquals(controller.frame - commitFrame, elapsed);
    assertEquals(
      Array.from(controller.getVisibleNodes()),
      [0, 1, 2, 3, 4],
      `reverted ${elapsed} frames into a ${commitFrames}-frame window`,
    );
  }

  // The frame the window expires on, it lands.
  controller.tick(controller.frame * 16);
  assertEquals(controller.frame - commitFrame, commitFrames);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2]);
  assertNoReheat(log);
});

Deno.test("committed lifetime: zero frames means the metric applies immediately", () => {
  const hierarchy = hierarchyOf([-1, 0, 0, 1, 1], (slot) => (slot === 0 ? 4000 : 100));
  const { controller, log } = rig(hierarchy, { minBandCommitFrames: 0 });

  log.viewport.scale = 1;
  controller.viewportChanged();
  controller.tick(0);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2, 3, 4]);

  log.viewport.scale = 0.25;
  controller.viewportChanged();
  controller.tick(16);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2]);
});

// =============================================================================
// Policy
// =============================================================================

Deno.test("policy: pass-through collapses a single-child chain out of the cut", () => {
  // 0 -> 1 -> 2 -> 3 -> {4, 5}: three single-child links, then a branch.
  const hierarchy = hierarchyOf([-1, 0, 1, 2, 3, 3], () => 1000);
  const { controller, log } = rig(hierarchy);

  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(
    Array.from(controller.getVisibleNodes()),
    [0, 1, 2, 3, 4, 5],
    "without a policy the whole chain is in the cut",
  );

  const passThroughChains: LodPolicy = (candidate) =>
    candidate.childCount === 1 ? "pass-through" : "default";
  controller.setPolicy(passThroughChains);
  controller.evaluateNow(16);

  // 0, 1 and 2 each carry exactly one child and vanish; their children are
  // considered at their parent's level, so the branch point is the new root.
  assertEquals(Array.from(controller.getVisibleNodes()), [3, 4, 5]);
  assertEquals(controller.getVisibleAncestor(4), 4);
  // A pass-through node has no representative: it was not folded into an
  // ancestor, it was declared absent, and everything above it is absent too.
  assertEquals(controller.getVisibleAncestor(1), -1);
  assertNoReheat(log);
});

Deno.test("policy: expand and collapse override the geometric rule", () => {
  const hierarchy = hierarchyOf([-1, 0, 0, 1, 1], (slot) => (slot === 0 ? 4000 : 10));
  const { controller, log } = rig(hierarchy);
  log.viewport.scale = 1;

  controller.evaluateNow(0);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2], "extent 10 is far below 96");

  controller.setPolicy((candidate) => (candidate.node === 1 ? "expand" : "default"));
  controller.evaluateNow(16);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2, 3, 4]);
  assert(log.events.some((entry) => entry === "node:expand/1/policy"));
});

Deno.test("policy: force-card promotes a node far below the DOM threshold", () => {
  const hierarchy = hierarchyOf([-1, 0, 0], () => 4000);
  const { controller, log } = rig(hierarchy, { minCardLifetimeMs: 0 });
  log.viewport.scale = 1;
  // One pixel of screen radius: well below the 48 px DOM band.
  log.radiusOf = () => 1;

  controller.evaluateNow(0);
  assertEquals(cardedNodes(log), [], "nothing is large enough to card");

  controller.setPolicy((candidate) => (candidate.node === 0 ? "force-card" : "default"));
  controller.evaluateNow(16);
  assertEquals(cardedNodes(log), [0]);
  assert(lastCards(log)[0].priority >= LOD_FORCE_CARD_PRIORITY);
});

Deno.test("policy: force-card reaches a focus leaf that crossed nothing", () => {
  const hierarchy = hierarchyOf([-1, 0, 0], () => 4000);
  const { controller, log } = rig(hierarchy, { minCardLifetimeMs: 0 });
  log.viewport.scale = 1;
  log.radiusOf = () => 1;

  // Node 2 is a leaf, so it never crosses a subtree band; it is a candidate
  // only because it is in the focus set.
  controller.setPolicy((candidate) =>
    candidate.node === 2 && candidate.childCount === 0 ? "force-card" : "default"
  );
  controller.setFocus([2]);
  assertEquals(cardedNodes(log), [2]);
});

Deno.test("policy: a focus node below the DOM threshold is carded", () => {
  const hierarchy = hierarchyOf([-1, 0, 0], () => 4000);
  const { controller, log } = rig(hierarchy, { minCardLifetimeMs: 0 });
  log.viewport.scale = 1;
  log.radiusOf = () => 1;

  controller.evaluateNow(0);
  assertEquals(cardedNodes(log), []);

  controller.setFocus([1]);
  assertEquals(cardedNodes(log), [1]);
  assert(lastCards(log)[0].priority >= LOD_FORCE_CARD_PRIORITY);

  controller.setFocus([]);
  assertEquals(cardedNodes(log), []);
  assertNoReheat(log);
});

Deno.test("policy: dropping the policy drops its sticky verdicts", () => {
  const hierarchy = hierarchyOf([-1, 0, 1, 2], () => 1000);
  const { controller, log } = rig(hierarchy);
  log.viewport.scale = 1;

  controller.setPolicy((candidate) => (candidate.childCount === 1 ? "pass-through" : "default"));
  controller.evaluateNow(0);
  assertEquals(Array.from(controller.getVisibleNodes()), [3]);

  controller.setPolicy(null);
  controller.evaluateNow(16);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2, 3]);
});

/** Cards declared by the most recent sync, excluding prefetch-only offers. */
function lastCards(log: Recorder): CardSyncEntry[] {
  const last = log.cards.at(-1) ?? [];
  return last.filter((entry) => entry.prefetchOnly !== true);
}

function cardedNodes(log: Recorder): number[] {
  return lastCards(log).map((entry) => entry.node).sort((a, b) => a - b);
}

// =============================================================================
// Cards
// =============================================================================

Deno.test("cards: the DOM band has its own hysteresis and a prefetch ring", () => {
  const hierarchy = hierarchyOf([-1, 0, 0], () => 4000);
  const { controller, log } = rig(hierarchy, { minCardLifetimeMs: 0 });

  // Radius 48 in graph units, so screen radius tracks zoom directly.
  log.radiusOf = () => 48;
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(cardedNodes(log), [0, 1, 2], "48 px is exactly the DOM threshold");

  // 0.75 zoom lands at 36 px: out of the entry band, still above the exit.
  log.viewport.scale = 0.75;
  controller.evaluateNow(100);
  assertEquals(cardedNodes(log), [0, 1, 2], "hysteresis holds the cards");

  // 0.5 zoom lands at 24 px: below the exit threshold, but inside the
  // prefetch ring at 48 / 2 = 24.
  log.viewport.scale = 0.5;
  controller.evaluateNow(200);
  assertEquals(cardedNodes(log), []);
  assertEquals(
    (log.cards.at(-1) ?? []).filter((entry) => entry.prefetchOnly === true).length,
    3,
  );
});

Deno.test("cards: the minimum lifetime holds a card that has shrunk away", () => {
  const hierarchy = hierarchyOf([-1, 0], () => 4000);
  const { controller, log } = rig(hierarchy, { minCardLifetimeMs: 400 });
  log.radiusOf = () => 48;

  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(cardedNodes(log), [0, 1]);

  log.viewport.scale = 0.25;
  controller.evaluateNow(300);
  assertEquals(cardedNodes(log), [0, 1], "still inside the lifetime floor");

  controller.evaluateNow(500);
  assertEquals(cardedNodes(log), [], "the floor has expired");
});

Deno.test("cards: the budget keeps the largest and weight breaks ties", () => {
  const hierarchy = hierarchyOf([-1, 0, 0, 0], () => 4000);
  const { controller, log } = rig(hierarchy, { maxCards: 2, minCardLifetimeMs: 0 });
  log.viewport.scale = 1;
  log.radiusOf = (node) => (node === 3 ? 96 : 48);
  log.weightOf = (node) => (node === 2 ? 1 : 0);

  controller.evaluateNow(0);
  // 3 is twice the size; 1 and 2 tie on size, so the heavier one survives.
  assertEquals(cardedNodes(log), [2, 3]);
});

// =============================================================================
// Budget
// =============================================================================

Deno.test("budget: maxVisibleNodes folds the smallest subtrees first", () => {
  // Root with two children: 1 has three descendants, 2 has one.
  const hierarchy = hierarchyOf(
    [-1, 0, 0, 1, 1, 1, 2],
    (slot) => [4000, 400, 200, 100, 100, 100, 100][slot],
  );
  const { controller, log } = rig(hierarchy, { maxVisibleNodes: 6 });
  log.viewport.scale = 1;
  controller.evaluateNow(0);

  // Roots plus 1 and 2 is 3 nodes; expanding 1 costs three more (6, at the
  // budget), which leaves no room for 2's single child.
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2, 3, 4, 5]);
  assert(controller.isCollapsed(2));
  assert(log.events.some((entry) => entry === "node:collapse/2/budget"));
});

Deno.test("budget: a generous budget expands everything", () => {
  const hierarchy = hierarchyOf(
    [-1, 0, 0, 1, 1, 1, 2],
    (slot) => [4000, 400, 200, 100, 100, 100, 100][slot],
  );
  const { controller, log } = rig(hierarchy, { maxVisibleNodes: 7 });
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2, 3, 4, 5, 6]);
});

// =============================================================================
// Imperative control and lifecycle
// =============================================================================

Deno.test("imperative: expandNode and collapseNode override geometry and report why", () => {
  const hierarchy = hierarchyOf([-1, 0, 0, 1, 1], (slot) => (slot === 0 ? 4000 : 10));
  const { controller, log } = rig(hierarchy);
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2]);

  controller.expandNode(1, 16);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2, 3, 4]);
  assert(log.events.some((entry) => entry === "node:expand/1/imperative"));

  controller.collapseNode(1, 32);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2]);
  assert(log.events.some((entry) => entry === "node:collapse/1/imperative"));
  assertNoReheat(log);
});

Deno.test("lifecycle: disabling releases the cut and shows everything again", () => {
  const hierarchy = hierarchyOf([-1, 0, 0, 1, 1], (slot) => (slot === 0 ? 4000 : 10));
  const { controller, log } = rig(hierarchy);
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2]);
  assert(controller.hasCut);

  controller.setConfig({ enabled: false }, 0);
  assertEquals(controller.hasCut, false);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2, 3, 4]);
  assertEquals(log.visibility.at(-1)?.hidden, []);
  assertEquals(log.cards.at(-1), []);
});

Deno.test("lifecycle: disabling announces the release like any other transition", () => {
  // A listener mirroring the fold state — a HUD counter, a host reconciling
  // cards — has only the events to go by. A silent release leaves it
  // describing a cut that no longer exists, and nothing arrives later to
  // correct it: the controller is off.
  const hierarchy = hierarchyOf([-1, 0, 0, 1, 1], (slot) => (slot === 0 ? 4000 : 10));
  const { controller, log } = rig(hierarchy);
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assert(controller.hasCut, "the fixture must fold node 1 or the test is vacuous");

  const before = log.events.length;
  controller.setConfig({ enabled: false }, 0);
  const released = log.events.slice(before);
  assert(
    released.includes("node:expand/1/zoom"),
    `the released proxy must announce its expand; got ${JSON.stringify(released)}`,
  );
  assertEquals(
    released.at(-1),
    "lod:change/5",
    "the closing lod:change reports every node visible",
  );

  // Disabling with nothing folded stays silent: there is no transition to
  // announce, and a spurious lod:change would ripple through listeners.
  const idle = rig(hierarchy);
  const idleBefore = idle.log.events.length;
  idle.controller.setConfig({ enabled: false }, 0);
  assertEquals(idle.log.events.length, idleBefore, "no cut, no events");
  assertNoReheat(log);
});

// =============================================================================
// Collapsed proxies — mass, render radius and the expand fix-up
// =============================================================================

/**
 * Root 0 over subtree root 1 (children 3, 4) and leaf 2.
 *
 * Slots are depth-first inside node 1's subtree, so a fix-up over it is one
 * contiguous run — the producer contract the hierarchy module asks for.
 */
function proxyFixture(): RetainedHierarchy {
  return hierarchyOf([-1, 0, 0, 1, 1], (slot) => (slot === 0 ? 4000 : 10));
}

Deno.test("proxies: a collapse rolls the subtree's mass onto the proxy and zeroes what it hides", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);

  // Node 1 stands for itself plus 3 and 4; nothing else is folded.
  assertEquals(Array.from(log.mass.at(-1)!), [1, 3, 1, 0, 0]);
  // Total mass is conserved whatever the collapse set, or the visible layout
  // would change scale as you zoom.
  assertEquals(log.mass.at(-1)!.reduce((a, b) => a + b, 0), 5);
  assertNoReheat(log);
});

Deno.test("proxies: a transition's mass roll-up allocates nothing", () => {
  // The roll-up walks the whole slot space, so at 220 000 nodes the three
  // working arrays it used to allocate were 1.9 MB of garbage per zoom step —
  // produced on the interaction path, where a collection pause is a dropped
  // frame. The controller holds one scratch set for the hierarchy's lifetime.
  const hierarchy = proxyFixture();
  const { controller, log } = rig(hierarchy);
  const n = hierarchy.nodeCount;
  log.viewport.scale = 1;

  // Warm: #adopt allocates the scratch, and the first cut is not the steady
  // state this measures.
  controller.evaluateNow(0);
  const uploads = log.mass.length;

  // Records every Uint8Array allocated while two real transitions run.
  //
  // Uint8Array specifically, because the roll-up's collapsed-set membership
  // array is the only nodeCount-long one of that type anywhere on this path —
  // the controller's own per-slot byte columns are allocated once, in #adopt.
  // (Its two stack arrays are Uint32Arrays, which a transition does allocate
  // for other reasons; the "allocates nothing at all" claim belongs to
  // rollUpMass itself and is measured exactly in tests/unit/mass_test.ts. What
  // this test adds is that the controller actually supplies a scratch set.)
  const seen: string[] = [];
  const originals = { Uint8Array: globalThis.Uint8Array };
  for (const name of Object.keys(originals) as (keyof typeof originals)[]) {
    const Original = originals[name];
    // deno-lint-ignore no-explicit-any
    (globalThis as any)[name] = class extends (Original as any) {
      // deno-lint-ignore no-explicit-any
      constructor(...args: any[]) {
        super(...args);
        // deno-lint-ignore no-explicit-any
        seen.push(`${name}(${(this as any).length})`);
      }
    };
  }
  try {
    controller.expandNode(1, 16);
    controller.collapseNode(1, 16);
  } finally {
    for (const name of Object.keys(originals) as (keyof typeof originals)[]) {
      // deno-lint-ignore no-explicit-any
      (globalThis as any)[name] = originals[name];
    }
  }

  // Non-vacuous: both transitions really did re-run the roll-up.
  assertEquals(log.mass.length, uploads + 2);
  assertEquals(
    seen.filter((entry) => entry === `Uint8Array(${n})`),
    [],
    `the roll-up allocated its collapsed-set array again (saw ${seen.join(", ")})`,
  );
});

Deno.test("proxies: expanding restores unit mass everywhere", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 3, 1, 0, 0]);

  controller.expandNode(1, 16);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 1, 1, 1, 1]);
  assertNoReheat(log);
});

Deno.test("proxies: every transition reuses one mass array", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  controller.expandNode(1, 16);
  controller.collapseNode(1, 32);

  assert(log.massIdentities.length >= 3, "expected one upload per transition");
  for (const array of log.massIdentities) {
    assertStrictEquals(
      array,
      log.massIdentities[0],
      "the per-transition path must not allocate a mass array",
    );
  }
});

Deno.test("proxies: a rebuilt hierarchy re-declares mass and radii", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  const uploadsBefore = log.mass.length;
  const proxiesBefore = log.proxies.length;

  // A topology change: the graph drops its hierarchy and derives a new one,
  // and the host has already reset mass and radii against the old slot map.
  log.hierarchy = proxyFixture();
  controller.evaluateNow(16);

  assert(log.mass.length > uploadsBefore, "adopting a hierarchy must re-upload mass");
  assertEquals(Array.from(log.mass.at(-1)!), [1, 3, 1, 0, 0]);
  assert(log.proxies.length > proxiesBefore, "adopting a hierarchy must re-declare proxies");
  assertEquals(log.proxies.at(-1), { slots: [1], radii: [10] });
});

Deno.test("proxies: a rebuilt hierarchy that folds nothing still clears the old rollup", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 3, 1, 0, 0]);

  // Every node is now large enough to expand, so the new cut folds nothing and
  // its diff against a freshly adopted state is empty in every column. The
  // rollup still has to be withdrawn: it named the old slot mapping.
  log.hierarchy = hierarchyOf([-1, 0, 0, 1, 1], () => 4000);
  controller.evaluateNow(16);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 1, 1, 1, 1]);
  assertEquals(log.proxies.at(-1), { slots: [], radii: [] });
});

Deno.test("proxies: the collapsed parent renders at its well radius and gets its own back", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(log.proxies.at(-1), { slots: [1], radii: [10] });

  controller.expandNode(1, 16);
  assertEquals(log.proxies.at(-1), { slots: [], radii: [] });
  assertNoReheat(log);
});

Deno.test("expand fix-up: the subtree is translated by the proxy's drift, before it is revealed", () => {
  const { controller, log } = rig(proxyFixture());
  const positions = new Map<NodeId, { x: number; y: number }>([[1, { x: 10, y: -4 }]]);
  log.positionOf = (node) => positions.get(node) ?? { x: 0, y: 0 };
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(log.translations, [], "a collapse translates nothing");

  // The proxy goes on simulating while 3 and 4 are frozen underneath it.
  positions.set(1, { x: 25, y: 6 });
  const visibilityWrites = log.visibility.length;
  const callsBefore = log.calls.length;
  controller.expandNode(1, 16);

  // Slots 3 and 4 are adjacent, so the whole subtree is one buffer write.
  assertEquals(log.translations, [{ lo: 3, hi: 5, dx: 15, dy: 10 }]);
  // The reveal must follow the translate: a subtree unflagged first is drawn
  // for a frame at the position its parent abandoned.
  const expandCalls = log.calls.slice(callsBefore);
  assertEquals(
    expandCalls.indexOf("translateNodeRange") < expandCalls.indexOf("applyVisibility"),
    true,
    `expected the translate before the reveal, got ${expandCalls.join(" -> ")}`,
  );
  assert(log.visibility.length > visibilityWrites);
  assertNoReheat(log);
});

Deno.test("expand fix-up: a scattered subtree becomes one call per contiguous run", () => {
  // Node 1's children are slots 2 and 4; node 3 belongs to the other root.
  const hierarchy = hierarchyOf([-1, 0, 1, 0, 1], (slot) => (slot === 0 ? 4000 : 10));
  const { controller, log } = rig(hierarchy);
  const positions = new Map<NodeId, { x: number; y: number }>([[1, { x: 0, y: 0 }]]);
  log.positionOf = (node) => positions.get(node) ?? { x: 0, y: 0 };
  log.viewport.scale = 1;
  controller.evaluateNow(0);

  positions.set(1, { x: 3, y: 7 });
  controller.expandNode(1, 16);
  assertEquals(log.translations, [
    { lo: 2, hi: 3, dx: 3, dy: 7 },
    { lo: 4, hi: 5, dx: 3, dy: 7 },
  ]);
});

Deno.test("expand fix-up: a proxy that never moved costs no write at all", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  controller.expandNode(1, 16);
  assertEquals(log.translations, []);
});

Deno.test("proxies: disabling LOD unfolds the proxies and returns every slot to unit mass", () => {
  const { controller, log } = rig(proxyFixture());
  const positions = new Map<NodeId, { x: number; y: number }>([[1, { x: 0, y: 0 }]]);
  log.positionOf = (node) => positions.get(node) ?? { x: 0, y: 0 };
  log.viewport.scale = 1;
  controller.evaluateNow(0);

  positions.set(1, { x: -8, y: 2 });
  controller.setConfig({ enabled: false }, 0);

  assertEquals(log.translations, [{ lo: 3, hi: 5, dx: -8, dy: 2 }]);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 1, 1, 1, 1]);
  assertEquals(log.proxies.at(-1), { slots: [], radii: [] });
});

Deno.test("proxies: a capacity change re-declares the live collapsed set", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  const uploads = log.mass.length;

  // Reallocation hands back a unit-mass buffer, so the rollup has to be
  // rewritten rather than left to whenever the next band is crossed.
  controller.handleNodeCapacityChange(64, 16);
  assert(log.mass.length > uploads);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 3, 1, 0, 0]);
});

// =============================================================================
// Edge aggregation — the seam, and when it fires
// =============================================================================

Deno.test("edges: the first cut aggregates against the visible mask", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);

  // Node 1 stands for its subtree, so 3 and 4 are off the cut and every edge
  // touching them has to be bundled onto it.
  assertEquals(Array.from(log.aggregations.at(-1)!), [1, 1, 1, 0, 0]);
  assertNoReheat(log);
});

Deno.test("edges: aggregation runs on the same transition boundary as mass", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  const before = log.aggregations.length;

  controller.expandNode(1, 16);
  assertEquals(Array.from(log.aggregations.at(-1)!), [1, 1, 1, 1, 1]);
  // Ordering is load-bearing: mass has to be uploaded before the aggregation
  // is, or a bundle arrives at a proxy the shaders still think weighs one node.
  const tail = log.calls.slice(log.calls.lastIndexOf("uploadNodeMass"));
  assertEquals(tail.includes("aggregateEdges"), true);
  assert(log.aggregations.length > before, "an expand must re-aggregate");
});

Deno.test("edges: a transition that changes nothing costs no aggregation", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  controller.expandNode(1, 16);
  const before = log.aggregations.length;

  // An imperative expand always rebuilds the cut, but this one asks for a
  // state the cut is already in: nothing enters, nothing leaves, and there is
  // nothing to re-walk 253 000 edges for.
  controller.expandNode(1, 32);
  assertEquals(log.aggregations.length, before);
});

Deno.test("edges: disabling LOD releases the aggregation", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(log.aggregationReleases, 0);

  controller.setConfig({ enabled: false }, 0);
  assertEquals(log.aggregationReleases, 1, "springs must go back to the source edge list");
  assertNoReheat(log);
});

Deno.test("edges: a node-capacity change leaves the aggregation alone", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  const before = log.aggregations.length;

  // Growing the node buffers reallocates mass and alpha, so those are
  // re-declared — but not the edge buffers, and the aggregation names slots
  // and edge indices that a capacity change preserves. Re-walking the edge
  // array here would be the transition budget spent on nothing.
  controller.handleNodeCapacityChange(64, 16);

  assertEquals(log.aggregations.length, before);
  assertEquals(log.aggregationReleases, 0);
  assertNoReheat(log);
});

Deno.test("lifecycle: no hierarchy means no cut and no host writes", () => {
  const { controller, log } = rig(null);
  assertEquals(controller.evaluateNow(0), false);
  assertEquals(controller.hasCut, false);
  assertEquals(log.visibility, []);
  assertEquals(log.events, []);
  assertEquals(log.cards, [], "a graph with no hierarchy must not be touched at all");
});

// =============================================================================
// Determinism
// =============================================================================

Deno.test("determinism: the same viewport sequence yields the same decision stream", () => {
  const sequence = [0.3, 1.7, 0.9, 4.2, 0.6, 2.5, 1.1, 0.4];

  const run = (): { events: string[]; visible: number[][]; cards: number[][] } => {
    const hierarchy = codeTreeHierarchy();
    const { controller, log } = rig(hierarchy, { minCardLifetimeMs: 0 });
    log.radiusOf = (node) => (node % 7) + 1;
    log.weightOf = (node) => (node % 5) / 4;

    const visible: number[][] = [];
    for (let i = 0; i < sequence.length; i++) {
      log.viewport.scale = sequence[i];
      controller.viewportChanged();
      for (let frame = 0; frame < 10; frame++) {
        controller.tick(i * 1000 + frame * 16);
      }
      visible.push(Array.from(controller.getVisibleNodes()));
    }
    assertNoReheat(log);
    return {
      events: log.events,
      visible,
      cards: log.cards.map((entries) => entries.map((entry) => entry.node)),
    };
  };

  const first = run();
  const second = run();
  assertEquals(first.events, second.events);
  assertEquals(first.visible, second.visible);
  assertEquals(first.cards, second.cards);
  assert(first.events.length > 0, "the sequence must actually decide something");
});

// =============================================================================
// Zoom quantisation and configuration
// =============================================================================

Deno.test("quantiseZoom: monotone, idempotent, and exact on powers of two", () => {
  for (const zoom of [0.125, 0.25, 0.5, 1, 2, 4, 8, 16]) {
    assertEquals(quantiseZoom(zoom), zoom);
  }
  let previous = 0;
  for (let zoom = 0.05; zoom < 20; zoom *= 1.01) {
    const q = quantiseZoom(zoom);
    assert(q >= previous, `quantisation went backwards at ${zoom}`);
    assertEquals(quantiseZoom(q), q, `not idempotent at ${zoom}`);
    previous = q;
  }
  assertEquals(quantiseZoom(0), 0);
  assertEquals(quantiseZoom(-1), 0);
  assertEquals(quantiseZoom(Number.NaN), 0);
});

Deno.test("config: band edges are forced into strict order", () => {
  const inverted = resolveLodConfig({ collapseThreshold: 200, expandThreshold: 96 });
  assert(inverted.collapseThreshold < inverted.expandThreshold);

  const invertedDom = resolveLodConfig({ domExitThreshold: 90, domThreshold: 48 });
  assert(invertedDom.domExitThreshold < invertedDom.domThreshold);
});

Deno.test("config: nonsense falls back and counts are clamped", () => {
  const resolved = resolveLodConfig({
    expandThreshold: Number.NaN,
    prefetchRatio: 0.1,
    maxVisibleNodes: -5,
    maxCards: -1,
    minBandCommitFrames: 2.7,
    transitionMs: -10,
  });
  assertEquals(resolved.expandThreshold, DEFAULT_LOD_CONFIG.expandThreshold);
  assertEquals(resolved.prefetchRatio, 1);
  assertEquals(resolved.maxVisibleNodes, 1);
  assertEquals(resolved.maxCards, 0);
  assertEquals(resolved.minBandCommitFrames, 2);
  assertEquals(resolved.transitionMs, 0);
});

Deno.test("config: a patch keeps every field it does not mention", () => {
  const first = resolveLodConfig({ enabled: true, maxCards: 12 });
  const second = resolveLodConfig({ transitionMs: 50 }, first);
  assertEquals(second.enabled, true);
  assertEquals(second.maxCards, 12);
  assertEquals(second.transitionMs, 50);
});
