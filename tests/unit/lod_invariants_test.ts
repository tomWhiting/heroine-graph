/**
 * Semantic-LOD invariant helpers: the assertions the LOD packages will be
 * verified with, verified themselves. Each test pins down both what the
 * helper accepts and what it must reject — a helper that never fails is
 * worse than no helper, since every LOD test would then be green by
 * construction.
 */

import { assertEquals, AssertionError, assertThrows } from "jsr:@std/assert@^1";
import {
  assertFrozen,
  assertVisibleSetMatchesReference,
  computeVisibleCut,
  type PositionSnapshot,
} from "../helpers/invariants.ts";

function snapshot(xs: readonly number[], ys: readonly number[]): PositionSnapshot {
  return { x: Float32Array.from(xs), y: Float32Array.from(ys) };
}

// =============================================================================
// assertFrozen
// =============================================================================

Deno.test("assertFrozen: passes when every listed node is bit-identical", () => {
  const before = snapshot([0, 10, 20, 30], [0, -10, -20, -30]);
  const after = snapshot([0, 10, 20, 30], [0, -10, -20, -30]);
  assertFrozen(before, after, [0, 1, 2, 3]);
  assertFrozen(before, after, []);
});

Deno.test("assertFrozen: rejects any drift, however small", () => {
  const before = snapshot([0, 10, 20], [0, -10, -20]);
  // 2^-20 is exactly one f32 ulp at 10, the smallest drift representable here.
  const drifted = snapshot([0, 10 + 2 ** -20, 20], [0, -10, -20]);
  assertThrows(
    () => assertFrozen(before, drifted, [0, 1, 2]),
    AssertionError,
    "1/3 nodes expected to be frozen moved; first is node 1",
  );
});

Deno.test("assertFrozen: rejects a frozen node that turned non-finite", () => {
  const before = snapshot([0, 10], [0, -10]);
  const broken = snapshot([0, NaN], [0, -10]);
  assertThrows(() => assertFrozen(before, broken, [1]));
});

Deno.test("assertFrozen: ignores nodes outside the index set", () => {
  const before = snapshot([0, 10, 20], [0, -10, -20]);
  // Node 1 (the live one) moved a long way; nodes 0 and 2 are the frozen set.
  const after = snapshot([0, 999, 20], [0, 999, -20]);
  assertFrozen(before, after, [0, 2]);
  assertThrows(() => assertFrozen(before, after, [0, 1, 2]));
});

// =============================================================================
// assertVisibleSetMatchesReference
// =============================================================================

Deno.test("assertVisibleSetMatchesReference: passes inside the tolerance, fails outside", () => {
  // Every offset here is exact in f32, so the boundary case is a real boundary.
  const reference = snapshot([0, 100, 200], [0, 0, 0]);
  const within = snapshot([0.25, 100, 200.5], [0, 0, 0]);
  assertVisibleSetMatchesReference(within, reference, [0, 1, 2], 0.5);

  const beyond = snapshot([0.25, 100, 200.75], [0, 0, 0]);
  assertThrows(
    () => assertVisibleSetMatchesReference(beyond, reference, [0, 1, 2], 0.5),
    AssertionError,
    "1/3 visible nodes diverge from the reference layout by more than 0.5; worst is node 2",
  );
});

Deno.test("assertVisibleSetMatchesReference: only the listed nodes are compared", () => {
  const reference = snapshot([0, 100, 200], [0, 0, 0]);
  // Node 1 is hidden in the actual run and parked far away; it must not count.
  const actual = snapshot([0, -5000, 200], [0, -5000, 0]);
  assertVisibleSetMatchesReference(actual, reference, [0, 2], 0.5);
  assertThrows(() => assertVisibleSetMatchesReference(actual, reference, [0, 1, 2], 0.5));
});

Deno.test("assertVisibleSetMatchesReference: maps onto a compacted reference slot space", () => {
  // Visible slots 1 and 3 of the LOD run correspond to slots 0 and 1 of a
  // reference graph built from only the visible nodes.
  const actual = snapshot([0, 100, 0, 300], [0, 7, 0, 9]);
  const reference = snapshot([100.2, 300.1], [7.1, 9.2]);
  assertVisibleSetMatchesReference(actual, reference, [1, 3], 0.5, [0, 1]);
  // Swapping the mapping compares the wrong pairs and must fail.
  assertThrows(() => assertVisibleSetMatchesReference(actual, reference, [1, 3], 0.5, [1, 0]));
});

Deno.test("assertVisibleSetMatchesReference: rejects mismatched index-set lengths", () => {
  const positions = snapshot([0, 1], [0, 1]);
  assertThrows(() => assertVisibleSetMatchesReference(positions, positions, [0, 1], 1, [0]));
});

Deno.test("assertVisibleSetMatchesReference: a non-finite coordinate is a violation", () => {
  const reference = snapshot([0, 100], [0, 0]);
  const actual = snapshot([0, NaN], [0, 0]);
  assertThrows(() => assertVisibleSetMatchesReference(actual, reference, [0, 1], 1e9));
});

// =============================================================================
// computeVisibleCut
// =============================================================================

/**
 *      0  r=100
 *     / \
 *    1   2      r=40, r=10
 *   / \   \
 *  3   4   5    r=5 (leaves)
 */
const CUT_PARENT = Int32Array.from([-1, 0, 0, 1, 1, 2]);
const CUT_WELL_RADIUS = Float32Array.from([100, 40, 10, 5, 5, 5]);
const CUT_EXPAND_THRESHOLD = 50;

Deno.test("computeVisibleCut: matches the hand-computed cut at each zoom band", () => {
  const cases: readonly {
    zoom: number;
    visible: readonly number[];
    collapsed: readonly number[];
  }[] = [
    // root extent 10 px: nothing expands, the root stands in for everything
    { zoom: 0.1, visible: [0], collapsed: [0] },
    // root exactly on the threshold (50 px) expands; both children are under it
    { zoom: 0.5, visible: [0, 1, 2], collapsed: [1, 2] },
    // node 1 reaches 60 px and expands; node 2 is still 15 px
    { zoom: 1.5, visible: [0, 1, 2, 3, 4], collapsed: [2] },
    // everything above a leaf expands; leaves are visible but never collapsed
    { zoom: 6, visible: [0, 1, 2, 3, 4, 5], collapsed: [] },
  ];

  for (const { zoom, visible, collapsed } of cases) {
    const cut = computeVisibleCut({
      parent: CUT_PARENT,
      wellRadius: CUT_WELL_RADIUS,
      zoom,
      expandThreshold: CUT_EXPAND_THRESHOLD,
    });
    assertEquals([...cut.visible], visible, `visible at zoom ${zoom}`);
    assertEquals([...cut.collapsed], collapsed, `collapsed at zoom ${zoom}`);
  }
});

Deno.test("computeVisibleCut: the visible set grows monotonically with zoom", () => {
  let previous = new Set<number>();
  for (const zoom of [0.05, 0.1, 0.3, 0.5, 1, 1.5, 3, 6, 20]) {
    const cut = computeVisibleCut({
      parent: CUT_PARENT,
      wellRadius: CUT_WELL_RADIUS,
      zoom,
      expandThreshold: CUT_EXPAND_THRESHOLD,
    });
    const current = new Set(cut.visible);
    for (const slot of previous) {
      assertEquals(current.has(slot), true, `node ${slot} vanished at zoom ${zoom}`);
    }
    previous = current;
  }
  assertEquals(previous.size, CUT_PARENT.length);
});

Deno.test("computeVisibleCut: reads Uint32Array roots the same as Int32Array roots", () => {
  const uintParent = Uint32Array.from([0xFFFFFFFF, 0, 0, 1, 1, 2]);
  for (const zoom of [0.1, 0.5, 1.5, 6]) {
    const fromInt = computeVisibleCut({
      parent: CUT_PARENT,
      wellRadius: CUT_WELL_RADIUS,
      zoom,
      expandThreshold: CUT_EXPAND_THRESHOLD,
    });
    const fromUint = computeVisibleCut({
      parent: uintParent,
      wellRadius: CUT_WELL_RADIUS,
      zoom,
      expandThreshold: CUT_EXPAND_THRESHOLD,
    });
    assertEquals([...fromUint.visible], [...fromInt.visible], `visible at zoom ${zoom}`);
    assertEquals([...fromUint.collapsed], [...fromInt.collapsed], `collapsed at zoom ${zoom}`);
  }
});

Deno.test("computeVisibleCut: handles a forest of independent roots", () => {
  // Two repo roots plus an orphan config file — the shape bubble.rs mishandles.
  const parent = Int32Array.from([-1, 0, -1, 2, -1]);
  const wellRadius = Float32Array.from([80, 5, 20, 5, 5]);

  const zoomedOut = computeVisibleCut({ parent, wellRadius, zoom: 0.1, expandThreshold: 50 });
  assertEquals([...zoomedOut.visible], [0, 2, 4]);
  assertEquals([...zoomedOut.collapsed], [0, 2]);

  const middle = computeVisibleCut({ parent, wellRadius, zoom: 1, expandThreshold: 50 });
  assertEquals([...middle.visible], [0, 1, 2, 4]);
  assertEquals([...middle.collapsed], [2]);

  const zoomedIn = computeVisibleCut({ parent, wellRadius, zoom: 4, expandThreshold: 50 });
  assertEquals([...zoomedIn.visible], [0, 1, 2, 3, 4]);
  assertEquals([...zoomedIn.collapsed], []);
});

Deno.test("computeVisibleCut: agrees with an independent ancestor-chain definition", () => {
  // The cut is exactly "every node whose ancestors all expanded". That is
  // checkable without the top-down walk, by climbing each node's parent chain.
  const parent = Int32Array.from([-1, 0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 6]);
  const wellRadius = Float32Array.from([90, 60, 30, 12, 55, 8, 40, 9, 4, 4, 4, 4]);
  const zoom = 1;
  const expandThreshold = 50;

  const cut = computeVisibleCut({ parent, wellRadius, zoom, expandThreshold });
  const expanded = (slot: number): boolean =>
    wellRadius[slot] * zoom >= expandThreshold && parent.some((p) => p === slot);
  const expectedVisible: number[] = [];
  for (let i = 0; i < parent.length; i++) {
    let ancestor = parent[i];
    let visible = true;
    while (ancestor >= 0) {
      if (!expanded(ancestor)) {
        visible = false;
        break;
      }
      ancestor = parent[ancestor];
    }
    if (visible) expectedVisible.push(i);
  }

  assertEquals([...cut.visible], expectedVisible);
  assertEquals([...cut.collapsed], [2, 3]);
});
