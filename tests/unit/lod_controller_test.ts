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

/**
 * A root over `dirs` directories of `perDir` leaves each.
 *
 * Regular on purpose: a mutation test wants the same shape at two sizes, so
 * that what changed between them is the node count and nothing else.
 */
function balancedTree(dirs: number, perDir: number): RetainedHierarchy {
  const parents: number[] = [-1];
  for (let d = 0; d < dirs; d++) parents.push(0);
  for (let d = 0; d < dirs; d++) {
    for (let c = 0; c < perDir; c++) parents.push(1 + d);
  }
  return hierarchyOf(parents, (_slot, subtreeSize) => Math.sqrt(subtreeSize) * 4);
}

/** Extend a visibility shadow to a grown node count, new slots visible. */
function growVisible(shadow: Uint8Array, nodeCount: number): Uint8Array {
  const grown = new Uint8Array(nodeCount).fill(1);
  grown.set(shadow);
  return grown;
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
  /**
   * Visibility as the host actually holds it, kept across writes.
   *
   * The call log above cannot answer "is this slot still flagged hidden" — an
   * absent write looks the same as a clearing one. GraphMother's node-flag
   * shadow survives everything short of a slot being reused, and only the
   * `[lo, hi)` of a write is rewritten, so a controller that assumes a clean
   * buffer instead of declaring one is only visible against a host that
   * remembers. Grows on demand, filled with 1: a fresh graph has no
   * `HIDDEN_LOD` bits anywhere.
   */
  visibleShadow: Uint8Array;
  cards: CardSyncEntry[][];
  events: string[];
  /** Every property name read off the host, in order. */
  accessed: string[];
  reheatCalls: number;
  viewport: { x: number; y: number; scale: number };
  radiusOf: (node: NodeId) => number;
  weightOf: (node: NodeId) => number;
  positionOf: (node: NodeId) => { x: number; y: number };
  /**
   * Positions as the host holds them, moved by every `translateNodeRange`.
   *
   * What the default `positionOf` reads, so a fix-up feeds back into the next
   * one — which is the whole question when folds nest. A test wanting inert
   * positions overrides `positionOf` instead.
   */
  positions: Map<NodeId, { x: number; y: number }>;
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
    visibleShadow: new Uint8Array(hierarchy?.nodeCount ?? 0).fill(1),
    cards: [],
    events: [],
    accessed: [],
    reheatCalls: 0,
    viewport: { x: VIEWPORT.x, y: VIEWPORT.y, scale: VIEWPORT.scale },
    radiusOf: () => 1,
    weightOf: () => 0,
    positionOf: (node) => state.positions.get(node) ?? { x: 0, y: 0 },
    positions: new Map(),
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
      if (hi > state.visibleShadow.length) {
        const grown = new Uint8Array(hi).fill(1);
        grown.set(state.visibleShadow);
        state.visibleShadow = grown;
      }
      state.visibleShadow.set(visible.subarray(lo, hi), lo);
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
      for (let slot = lo; slot < hi; slot++) {
        const at = state.positions.get(slot) ?? { x: 0, y: 0 };
        state.positions.set(slot, { x: at.x + dx, y: at.y + dy });
      }
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

/**
 * One root over six leaves. Slots 1..3 sit under the camera at half the size
 * of slots 4..6, which are a million units away — so a rank that never asks
 * where the camera is awards every card to nodes nobody can see.
 */
function offscreenFixture(): RetainedHierarchy {
  return hierarchyOf([-1, 0, 0, 0, 0, 0, 0], () => 4000);
}

const FAR_AWAY = { x: 1_000_000, y: 0 };

function offscreenRig(config: Parameters<LODController["setConfig"]>[0] = {}): Rig {
  const built = rig(offscreenFixture(), { maxCards: 3, minCardLifetimeMs: 0, ...config });
  built.log.viewport.scale = 1;
  // The root is far too small to card, so the budget is a contest between the
  // two leaf clusters and nothing else.
  built.log.radiusOf = (node) => (node === 0 ? 1 : node >= 4 ? 200 : 50);
  built.log.positionOf = (node) => (node >= 4 ? FAR_AWAY : { x: 0, y: 0 });
  return built;
}

Deno.test("cards: the card set is culled to the camera, whatever the rank says", () => {
  const { controller, log } = offscreenRig();
  controller.evaluateNow(0);

  // 4, 5 and 6 outrank everything on screen four to one and get nothing: a
  // card the user cannot see is DOM, layout and provider work spent on
  // nothing, and at the zoom where every leaf clears the DOM threshold the
  // ring is the only thing left to spend the budget on.
  assertEquals(cardedNodes(log), [1, 2, 3]);
  const offered = (log.cards.at(-1) ?? []).map((entry) => entry.node);
  assertEquals(offered.filter((node) => node >= 4), [], "nor is an unseen node prefetched");
  assertNoReheat(log);
});

Deno.test("cards: panning to the far cluster takes the cards with it", () => {
  const { controller, log } = offscreenRig();
  controller.evaluateNow(0);
  assertEquals(cardedNodes(log), [1, 2, 3]);
  // Every ramp lands first, so the pan is the only thing that can make the
  // tick below re-derive anything: a fade in flight re-syncs cards on its own.
  for (let frame = 1; frame <= 20; frame++) controller.tick(frame * 16);
  assertEquals(cardedNodes(log), [1, 2, 3]);

  // A pan moves no band — both LOD metrics are zoom-only — so nothing but the
  // card ring makes this evaluation worth running, and without it the card set
  // is a function of the graph alone and never follows the camera at all.
  log.viewport.x = FAR_AWAY.x;
  controller.viewportChanged();
  controller.tick(1000);
  assertEquals(cardedNodes(log), [4, 5, 6]);
  assertNoReheat(log);
});

Deno.test("cards: a focused node is carded wherever it is", () => {
  const { controller, log } = offscreenRig();
  controller.evaluateNow(0);
  assertEquals(cardedNodes(log), [1, 2, 3]);

  // Focus and a force-card verdict are declarations of intent, not
  // observations about screen geometry, so the ring does not apply to them.
  controller.setFocus([5]);
  assert(cardedNodes(log).includes(5), `expected 5 to be carded, got ${cardedNodes(log)}`);
});

Deno.test("cards: the ring reaches a prefetch ratio beyond the viewport edge", () => {
  const { controller, log } = rig(offscreenFixture(), { minCardLifetimeMs: 0 });
  log.viewport.scale = 1;
  log.radiusOf = (node) => (node === 0 ? 1 : 50);
  // Half the 1200 px viewport is 600 graph units at zoom 1; the ring reaches
  // prefetchRatio times that, and a node's own render radius reaches further.
  log.positionOf = (node) => ({ x: [0, 900, 1249, 1251, 0, 0, 0][node], y: 0 });

  controller.evaluateNow(0);
  assertEquals(cardedNodes(log), [1, 2, 4, 5, 6]);
});

Deno.test("cards: gaining a card crossfades the sprite out and the card in", () => {
  const hierarchy = hierarchyOf([-1, 0], () => 4000);
  const { controller, log } = rig(hierarchy, { minCardLifetimeMs: 0 });
  log.radiusOf = () => 10;
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(cardedNodes(log), [], "10 px is well below the 48 px DOM threshold");
  assertEquals(controller.crossfade.alphaOf(1), 1);

  // Crossing the DOM band: 10 graph units at zoom 8 is 80 px.
  log.viewport.scale = 8;
  controller.viewportChanged();
  controller.tick(1000);
  assertEquals(cardedNodes(log), [0, 1]);
  // The card mounts transparent over a solid sprite. The two are one node's
  // two representations, so they exchange opacity rather than both being drawn
  // — and the swap is the one transition in the system a user actually watches.
  assertEquals(lastCards(log).map((entry) => entry.opacity), [0, 0]);
  assertEquals(controller.crossfade.alphaOf(1), 1);

  controller.tick(1000 + DEFAULT_LOD_CONFIG.transitionMs / 2);
  assertEquals(lastCards(log).map((entry) => entry.opacity), [0.5, 0.5]);
  assertEquals(controller.crossfade.alphaOf(1), 0.5);

  controller.tick(1000 + DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(lastCards(log).map((entry) => entry.opacity), [1, 1]);
  assertEquals(controller.crossfade.alphaOf(1), 0);

  // Leaving the band reverses both ramps.
  log.viewport.scale = 1;
  controller.viewportChanged();
  controller.tick(2000);
  assertEquals(cardedNodes(log), []);
  assertEquals(controller.crossfade.alphaOf(1), 0, "the sprite comes back over the same ramp");
  controller.tick(2000 + DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(controller.crossfade.alphaOf(1), 1);
  assertNoReheat(log);
});

Deno.test("cards: a carded node fades its sprite but is never hidden or unweighted", () => {
  // The card is where the node is drawn, not a replacement for the node: the
  // simulation still has to move it, or every card in the graph freezes.
  const hierarchy = hierarchyOf([-1, 0], () => 4000);
  const { controller, log } = rig(hierarchy, { minCardLifetimeMs: 0 });
  log.radiusOf = () => 48;
  log.viewport.scale = 1;

  controller.evaluateNow(0);
  for (let frame = 1; frame <= 20; frame++) controller.tick(frame * 16);

  assertEquals(cardedNodes(log), [0, 1]);
  assertEquals(controller.crossfade.alphaOf(1), 0, "the sprite is fully faded");
  assertEquals(log.visibility.flatMap((write) => write.hidden), [], "and never flagged hidden");
  assertEquals(Array.from(log.mass.at(-1)!), [1, 1], "and never loses its mass");
});

Deno.test("cards: a node ranked out of the budget keeps its sprite", () => {
  // Committing the card set before applying the budget would fade out the
  // sprite of a node whose card never mounts, leaving nothing on screen at all.
  const hierarchy = hierarchyOf([-1, 0, 0], () => 4000);
  const { controller, log } = rig(hierarchy, { maxCards: 1, minCardLifetimeMs: 0 });
  log.radiusOf = (node) => 48 + node;
  log.viewport.scale = 1;

  controller.evaluateNow(0);
  for (let frame = 1; frame <= 20; frame++) controller.tick(frame * 16);

  assertEquals(cardedNodes(log), [2], "the largest takes the only card");
  assertEquals(controller.crossfade.alphaOf(2), 0);
  assertEquals(controller.crossfade.alphaOf(0), 1);
  assertEquals(controller.crossfade.alphaOf(1), 1);
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

Deno.test("cards: a suppressed node stops being declared and gets its sprite back", () => {
  // The answer to a card the overlay could not create. The controller has
  // already faded the sprite out under a card that does not exist, so without
  // this the node renders as neither sprite nor card — a hole in the graph.
  const hierarchy = hierarchyOf([-1, 0], () => 4000);
  const { controller, log } = rig(hierarchy, { minCardLifetimeMs: 0 });
  log.radiusOf = () => 48;
  log.viewport.scale = 1;

  controller.evaluateNow(0);
  for (let frame = 1; frame <= 20; frame++) controller.tick(frame * 16);
  assertEquals(cardedNodes(log), [0, 1]);
  assertEquals(controller.crossfade.alphaOf(1), 0, "the carded node's sprite is faded out");

  controller.suppressCard(1, 400);
  controller.tick(400 + DEFAULT_LOD_CONFIG.transitionMs / 2);
  assertEquals(controller.crossfade.alphaOf(1), 0.5, "the sprite comes back over the same ramp");
  controller.tick(400 + DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(controller.crossfade.alphaOf(1), 1, "and comes all the way back");

  // The fade the suppression started is what drives the re-derivation: nothing
  // else would have told the host to stop expecting this card.
  assertEquals(cardedNodes(log), [0], "a suppressed node must not be declared again");
  log.viewport.scale = 2;
  controller.viewportChanged();
  controller.tick(2000);
  assertEquals(cardedNodes(log), [0], "not even after a zoom that would have carded it");
  assertNoReheat(log);
});

Deno.test("cards: clearing the suppressions cards the node again", () => {
  const hierarchy = hierarchyOf([-1, 0], () => 4000);
  const { controller, log } = rig(hierarchy, { minCardLifetimeMs: 0 });
  log.radiusOf = () => 48;
  log.viewport.scale = 1;

  controller.evaluateNow(0);
  controller.suppressCard(1, 100);
  controller.tick(100 + DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(cardedNodes(log), [0]);

  // Registering a working provider is the documented way back, and it must not
  // wait for the camera to move.
  controller.clearCardSuppressions();

  assertEquals(cardedNodes(log), [0, 1]);
});

Deno.test("cards: a topology change lifts the suppressions, because it moves slots", () => {
  // A suppression names a slot; a removal compacts slots. Held across one, it
  // stops the controller declaring a node that never failed — which renders as
  // neither sprite nor card, with nothing to announce it and no act a consumer
  // would think to perform. The controller cannot tell whose slot it is (it has
  // no producer ids), so it forgives on the one signal it does get. The retry
  // that costs is caught by the overlay's quarantine, which *is* keyed by
  // producer id, so a node that really is broken is refused there instead.
  //
  // Deliberately not a change of hierarchy object: adoption already clears the
  // set, and this is the contract of the other entry point.
  const hierarchy = hierarchyOf([-1, 0], () => 4000);
  const { controller, log } = rig(hierarchy, { minCardLifetimeMs: 0 });
  log.radiusOf = () => 48;
  log.viewport.scale = 1;

  controller.evaluateNow(0);
  controller.suppressCard(1, 100);
  controller.tick(100 + DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(cardedNodes(log), [0], "the suppressed node is not declared");

  controller.handleTopologyChange();
  controller.tick(1000);

  assertEquals(
    cardedNodes(log),
    [0, 1],
    "the slot's occupant is in question, so it is a candidate",
  );
  assertNoReheat(log);
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
  // The roll-up lands with the flag, not with the cut — see the physics-timing
  // tests below — so the crossfade has to finish before the mass is final.
  controller.tick(DEFAULT_LOD_CONFIG.transitionMs);

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
  controller.tick(DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 3, 1, 0, 0]);

  controller.expandNode(1, 200);
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
  controller.tick(16 + DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 3, 1, 0, 0]);
  assert(log.proxies.length > proxiesBefore, "adopting a hierarchy must re-declare proxies");
  assertEquals(log.proxies.at(-1), { slots: [1], radii: [10] });
});

Deno.test("proxies: a rebuilt hierarchy that folds nothing still clears the old rollup", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  controller.tick(DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 3, 1, 0, 0]);

  // Every node is now large enough to expand, so the new cut folds nothing and
  // its diff against a freshly adopted state is empty in every column. The
  // rollup still has to be withdrawn: it named the old slot mapping.
  log.hierarchy = hierarchyOf([-1, 0, 0, 1, 1], () => 4000);
  controller.evaluateNow(16);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 1, 1, 1, 1]);
  assertEquals(log.proxies.at(-1), { slots: [], radii: [] });
});

Deno.test("adoption: a rebuilt hierarchy clears the flags the previous cut left on the host", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  controller.tick(DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(Array.from(log.visibleShadow), [1, 1, 1, 0, 0], "the fold must have landed");

  // A topology change rebuilds the hierarchy and resets mass and proxy radii,
  // but it does not touch the node-flag shadow, and neither does a
  // reallocation — it re-uploads that shadow verbatim. The new cut here folds
  // nothing, so its diff against a freshly adopted state is empty in every
  // column and nothing in the transition would ever clear slots 3 and 4.
  log.hierarchy = hierarchyOf([-1, 0, 0, 1, 1], () => 4000);
  controller.evaluateNow(200);

  for (const slot of controller.getVisibleNodes()) {
    assertEquals(
      log.visibleShadow[slot],
      1,
      `slot ${slot} is reported visible but is still flagged hidden on the host`,
    );
  }
  assertEquals(Array.from(log.visibleShadow), [1, 1, 1, 1, 1]);
  assertNoReheat(log);
});

Deno.test("adoption: the host is told the slot space is visible, not assumed to be", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  controller.tick(DEFAULT_LOD_CONFIG.transitionMs);
  const writes = log.visibility.length;

  log.hierarchy = proxyFixture();
  controller.evaluateNow(200);

  // The adoption's own write comes first and spans the whole slot space, so
  // the controller's shadow and the host's flags agree by construction. Every
  // later write in the transition is a diff against that.
  assertEquals(log.visibility[writes], { lo: 0, hi: 5, hidden: [] });
  assertEquals(Array.from(log.visibleShadow), [1, 1, 1, 1, 1]);

  // And the new cut's own fold lands from there, as any first cut would.
  controller.tick(200 + DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(Array.from(log.visibleShadow), [1, 1, 1, 0, 0]);
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

/**
 * Root 0 over A(1) over X(2) over leaves 3 and 4: two interior levels, so a
 * fold can close over a fold. Radii are chosen so the two collapse at different
 * zooms — X at 0.25, A at 0.03125 — and both are back by zoom 1.
 */
function nestedFixture(): RetainedHierarchy {
  const radii = [4000, 1000, 200, 10, 10];
  return hierarchyOf([-1, 0, 1, 2, 2], (slot) => radii[slot]);
}

/** How far a slot was translated in total, across every fix-up so far. */
function totalDx(log: Recorder, slot: NodeId): number {
  let total = 0;
  for (const write of log.translations) {
    if (slot >= write.lo && slot < write.hi) total += write.dx;
  }
  return total;
}

Deno.test("expand fix-up: a fold dissolved under a higher one still owes its drift", () => {
  // The failure this pins: X collapses, drifts as a live proxy, and is then
  // swallowed by A collapsing over it. X leaves the collapsed set without ever
  // entering the visible cut, so a diff keyed on expansion never sees it — and
  // its drift is lost, leaving its leaves detached from it by exactly that
  // much once A comes back.
  const { controller, log } = rig(nestedFixture(), {
    transitionMs: 0,
    minBandCommitFrames: 0,
  });
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(controller.isCollapsed(2), false, "X must start expanded");

  log.viewport.scale = 0.25;
  controller.evaluateNow(1);
  assert(controller.isCollapsed(2), "X must be standing in for its leaves");
  log.positions.set(2, { x: 500, y: 0 });

  log.viewport.scale = 0.03125;
  controller.evaluateNow(2);
  assert(controller.isCollapsed(1), "A must be standing in for X's subtree");
  assertEquals(controller.isCollapsed(2), false, "X's own fold has dissolved into A's");
  log.positions.set(1, { x: 1000, y: 0 });

  log.viewport.scale = 1;
  controller.evaluateNow(3);
  assertEquals(controller.isCollapsed(1), false);

  assertEquals(totalDx(log, 3), 1500, "X's leaves owe both drifts");
  assertEquals(totalDx(log, 4), 1500);
  assertEquals(
    totalDx(log, 2),
    1000,
    "X was a frozen body under A once its own fold dissolved, so it owes only A's",
  );
  assertNoReheat(log);
});

Deno.test("expand fix-up: re-emerging still folded keeps the drift already flushed", () => {
  // The staged variant: A expands at a zoom where X is still too small, so X
  // comes back as a proxy and is immediately re-anchored. The re-anchor must
  // not swallow drift the dissolve already paid out, and must not pay it twice.
  const { controller, log } = rig(nestedFixture(), {
    transitionMs: 0,
    minBandCommitFrames: 0,
  });
  log.viewport.scale = 1;
  controller.evaluateNow(0);

  log.viewport.scale = 0.25;
  controller.evaluateNow(1);
  log.positions.set(2, { x: 500, y: 0 });

  log.viewport.scale = 0.03125;
  controller.evaluateNow(2);
  log.positions.set(1, { x: 1000, y: 0 });

  // Back through the band where X folds on its own account.
  log.viewport.scale = 0.25;
  controller.evaluateNow(3);
  assert(controller.isCollapsed(2), "X must re-emerge still folded");
  assertEquals(controller.isCollapsed(1), false);

  log.viewport.scale = 1;
  controller.evaluateNow(4);
  assertEquals(controller.isCollapsed(2), false);

  assertEquals(totalDx(log, 3), 1500, "X's leaves owe both drifts, once each");
  assertEquals(totalDx(log, 4), 1500);
  assertEquals(totalDx(log, 2), 1000);
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

Deno.test("mutation: a graph that grows under a still camera is re-folded", () => {
  // A streaming producer mutates the graph continuously while the user sits
  // still. Zoom is the controller's only geometric input, so without a
  // topology signal the cut goes on describing the graph as it was and every
  // node streamed in since renders unfolded — which for a producer that never
  // stops is the steady state, not a moment of lag.
  const before = balancedTree(6, 40);
  const { controller, log } = rig(before);
  log.viewport.scale = 2;
  controller.evaluateNow(0);

  // Hiding waits behind the crossfade, so the flags land some frames after the
  // decision that made them.
  let clock = 0;
  const settle = () => {
    for (let frame = 0; frame < 60; frame++) controller.tick(clock += 16);
  };
  const hiddenCount = () => log.visibleShadow.reduce((n, v) => n + (v === 0 ? 1 : 0), 0);
  settle();

  assertEquals(
    Array.from(controller.getVisibleNodes()).length,
    7,
    "root plus six folded directories",
  );
  assertEquals(hiddenCount(), before.nodeCount - 7);

  // The graph grows: same shape, three times the leaves, camera untouched.
  const after = balancedTree(6, 120);
  log.hierarchy = after;
  log.visibleShadow = growVisible(log.visibleShadow, after.nodeCount);

  controller.handleTopologyChange();
  settle();

  assertEquals(
    Array.from(controller.getVisibleNodes()).length,
    7,
    "the cut is the same seven nodes: the new leaves belong to folded subtrees",
  );
  assertEquals(
    hiddenCount(),
    after.nodeCount - 7,
    "every newly streamed node is folded away, not left rendering",
  );
  // Each directory now stands for three times the subtree, so the bubble it
  // presents has to grow with it or the layout reads as the old graph.
  const radii = log.proxies.at(-1)!.radii;
  assertEquals(radii.length, 6);
  assert(
    radii.every((r) => r > before.columns.wellRadius[1]),
    `proxy radii ${radii[0]} did not grow past the pre-mutation ${before.columns.wellRadius[1]}`,
  );
});

// =============================================================================
// Edge aggregation — the seam, and when it fires
// =============================================================================

Deno.test("edges: the first cut aggregates against the applied mask", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);

  // 3 and 4 have left the cut but are still drawn and still integrated, so
  // they still carry their own springs.
  assertEquals(Array.from(log.aggregations.at(-1)!), [1, 1, 1, 1, 1]);

  // Node 1 stands for its subtree, so once they are flagged every edge
  // touching them has to be bundled onto it.
  controller.tick(DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(Array.from(log.aggregations.at(-1)!), [1, 1, 1, 0, 0]);
  assertNoReheat(log);
});

Deno.test("edges: aggregation runs on the same transition boundary as mass", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  controller.tick(DEFAULT_LOD_CONFIG.transitionMs);
  const before = log.aggregations.length;

  controller.expandNode(1, 200);
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

  // Nor does simply advancing the clock: every ramp has landed, so no flag
  // moves and the aggregation still describes the state the host is in.
  controller.tick(200);
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

Deno.test("edges: the aggregation knob gates the pass and takes back what it built", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  controller.tick(DEFAULT_LOD_CONFIG.transitionMs);
  const built = log.aggregations.length;
  assert(built > 0, "the knob defaults on, so the first cut must have aggregated");

  // Turning it off cannot merely stop the pass: the bundles already driving
  // the springs have to go back to the source edge list, or the knob reports a
  // state the host is not in.
  controller.setConfig({ edgeAggregation: false }, 0);
  assertEquals(log.aggregationReleases, 1);

  controller.expandNode(1, 16);
  controller.collapseNode(1, 32);
  controller.tick(32 + DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(log.aggregations.length, built, "a disabled pass must not run");

  // Back on, the transitions taken while it was off are not replayed, so the
  // next one has to rebuild against the mask the host is actually in.
  controller.setConfig({ edgeAggregation: true }, 64);
  controller.evaluateNow(64);
  assertEquals(log.aggregations.length, built + 1);
  assertEquals(Array.from(log.aggregations.at(-1)!), [1, 1, 1, 0, 0]);
  assertNoReheat(log);
});

Deno.test("edges: a node-capacity change leaves the aggregation alone", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  // With the cut's own fade landed, nothing about the applied cut moves here.
  controller.tick(DEFAULT_LOD_CONFIG.transitionMs);
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

// =============================================================================
// Physics timing — the cut moves a fade ahead of the flags, and the physics
// must follow the flags
// =============================================================================

Deno.test("physics timing: a collapse holds mass and springs until the fade lands", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);

  // The cut has moved and the proxy is declared...
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2]);
  assertEquals(log.proxies.at(-1), { slots: [1], radii: [10] });
  // ...but 3 and 4 are still drawn and still integrated. Rolling their mass
  // onto the proxy now leaves them weightless, and aggregating their edges onto
  // it strips every spring they have, for the whole length of the fade: two
  // massless bodies inside the aggregate repulsion of the node standing in for
  // them, which measurably throws them a full link length outwards and then
  // freezes them there.
  assertEquals(Array.from(log.mass.at(-1)!), [1, 1, 1, 1, 1]);
  assertEquals(Array.from(log.aggregations.at(-1)!), [1, 1, 1, 1, 1]);
  assertEquals(log.visibility.flatMap((write) => write.hidden), []);

  controller.tick(DEFAULT_LOD_CONFIG.transitionMs / 2);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 1, 1, 1, 1], "mid-ramp is still mid-ramp");
  assertEquals(Array.from(log.aggregations.at(-1)!), [1, 1, 1, 1, 1]);

  // The frame the flag lands, the physics lands with it.
  controller.tick(DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(log.visibility.at(-1)?.hidden, [3, 4]);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 3, 1, 0, 0]);
  assertEquals(Array.from(log.aggregations.at(-1)!), [1, 1, 1, 0, 0]);
  assertNoReheat(log);
});

Deno.test("physics timing: overlapping collapses each land with their own ramp", () => {
  // Two sibling subtrees, collapsed 100 ms apart inside a 150 ms transition.
  const hierarchy = hierarchyOf([-1, 0, 0, 1, 1, 2, 2], (slot) => (slot === 0 ? 4000 : 200));
  const { controller, log } = rig(hierarchy);
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  assertEquals(Array.from(controller.getVisibleNodes()), [0, 1, 2, 3, 4, 5, 6]);

  controller.collapseNode(1, 0);
  controller.collapseNode(2, 100);

  // The deferred physics is recomputed from the flags rather than replayed
  // from a diff, so a settle that lands only part of the pending set applies
  // only that part: node 1 carries its subtree, node 2 does not yet.
  controller.tick(DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(log.visibility.at(-1)?.hidden, [3, 4]);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 3, 1, 0, 0, 1, 1]);
  assertEquals(Array.from(log.aggregations.at(-1)!), [1, 1, 1, 0, 0, 1, 1]);

  controller.tick(100 + DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 3, 3, 0, 0, 0, 0]);
  assertEquals(Array.from(log.aggregations.at(-1)!), [1, 1, 1, 0, 0, 0, 0]);
  assertNoReheat(log);
});

Deno.test("physics timing: an expand restores mass and springs immediately", () => {
  const { controller, log } = rig(proxyFixture());
  log.viewport.scale = 1;
  controller.evaluateNow(0);
  controller.tick(DEFAULT_LOD_CONFIG.transitionMs);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 3, 1, 0, 0]);

  // The other direction does not defer: a node fading *in* is drawn and
  // integrated from the first frame, so it needs its own mass and its own
  // springs from the first frame, not a fade later.
  controller.expandNode(1, 200);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 1, 1, 1, 1]);
  assertEquals(Array.from(log.aggregations.at(-1)!), [1, 1, 1, 1, 1]);
  assert(controller.crossfade.alphaOf(3) < 1, "the reveal must still be ramping");
});

Deno.test("physics timing: with no transition the physics lands in the same call", () => {
  const { controller, log } = rig(proxyFixture(), { transitionMs: 0 });
  log.viewport.scale = 1;
  const massBefore = log.mass.length;
  const aggregationsBefore = log.aggregations.length;

  controller.evaluateNow(0);

  // Nothing to wait for, so the flag, the roll-up and the aggregation all land
  // on the cut — and each is written once, not once per settle pass.
  assertEquals(log.visibility.at(-1)?.hidden, [3, 4]);
  assertEquals(Array.from(log.mass.at(-1)!), [1, 3, 1, 0, 0]);
  assertEquals(Array.from(log.aggregations.at(-1)!), [1, 1, 1, 0, 0]);
  assertEquals(log.mass.length, massBefore + 1);
  assertEquals(log.aggregations.length, aggregationsBefore + 1);
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
