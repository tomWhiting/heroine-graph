/**
 * GPU tests for the LOD expand fix-up (WP-G).
 *
 * A collapsed subtree is frozen while the proxy standing for it goes on
 * simulating, so on expand the subtree would reappear wherever its parent used
 * to be — detached from the parent, often by more than its own diameter. The
 * fix is a rigid-body translation by the proxy's drift, applied before the
 * subtree is revealed.
 *
 * The claim under test is therefore *relative*: every descendant comes back at
 * exactly its pre-collapse offset from the proxy. Absolute positions are not
 * the invariant and must not be — the proxy moved on purpose.
 *
 * The controller is driven through a host built over the real simulation
 * buffers, so the writes it issues land in the same ping-pong pair a tick
 * reads. A fix-up that wrote only one orientation would survive a CPU-only
 * test and lose the subtree on the next swap.
 */

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import {
  createSimHarness,
  GPU_SKIP_MESSAGE,
  NODE_FLAG_HIDDEN_LOD,
  probeAdapter,
  requestHarnessDevice,
  type SimHarness,
} from "../helpers/gpu.ts";
import { LODController, type LodHost } from "../../packages/core/src/lod/controller.ts";
import {
  HIERARCHY_ROOT,
  type HierarchyColumns,
  type RetainedHierarchy,
  retainSuppliedHierarchy,
} from "../../packages/core/src/graph/hierarchy.ts";
import type { NodeId, ViewportState } from "../../packages/core/src/types.ts";

const adapter = await probeAdapter();
if (!adapter) {
  console.warn(`[gpu] ${GPU_SKIP_MESSAGE}`);
}

function gpuTest(
  name: string,
  fn: (device: GPUDevice) => Promise<void>,
): void {
  Deno.test({
    name,
    ignore: adapter === null,
    sanitizeResources: false,
    sanitizeOps: false,
    async fn() {
      const device = await requestHarnessDevice(adapter!);
      try {
        await fn(device);
      } finally {
        device.destroy();
      }
    },
  });
}

// =============================================================================
// Fixture
// =============================================================================

/**
 * Root 0, subtree root 1 over leaves 2..7, and four more leaves under the root.
 *
 * Slots inside the subtree are contiguous — the depth-first order producers are
 * asked for — so a correct fix-up is a single buffer write and the test can say
 * so exactly.
 */
const SUBTREE_ROOT = 1;
const DESCENDANTS = [2, 3, 4, 5, 6, 7] as const;
const SIBLINGS = [8, 9, 10, 11] as const;
const NODE_COUNT = 12;

/** Well radius of the root: far above the expand threshold, so it unfolds. */
const ROOT_WELL_RADIUS = 4000;
/** Well radius of the subtree: far below the collapse threshold, so it folds. */
const SUBTREE_WELL_RADIUS = 10;

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
 * The same twelve slots with an extra interior level: root 0 over A(1) over
 * X(2) over leaves 3..7, siblings still under the root.
 *
 * Radii put A and X in different bands, so a fold can close over a live fold —
 * the case where the inner proxy stops standing in for its subtree without ever
 * expanding, and its drift has nowhere to go unless the transition flushes it.
 */
const NESTED_OUTER = 1;
const NESTED_INNER = 2;
const NESTED_LEAVES = [3, 4, 5, 6, 7] as const;

function nestedHierarchy(): RetainedHierarchy {
  const parent = new Uint32Array(NODE_COUNT);
  const depth = new Uint16Array(NODE_COUNT);
  const subtreeSize = new Uint32Array(NODE_COUNT).fill(1);
  const wellRadius = new Float32Array(NODE_COUNT).fill(SUBTREE_WELL_RADIUS);

  parent[0] = HIERARCHY_ROOT;
  parent[NESTED_OUTER] = 0;
  depth[NESTED_OUTER] = 1;
  parent[NESTED_INNER] = NESTED_OUTER;
  depth[NESTED_INNER] = 2;
  for (const slot of NESTED_LEAVES) {
    parent[slot] = NESTED_INNER;
    depth[slot] = 3;
  }
  for (const slot of SIBLINGS) {
    parent[slot] = 0;
    depth[slot] = 1;
  }
  for (let i = NODE_COUNT - 1; i > 0; i--) subtreeSize[parent[i]] += subtreeSize[i];
  wellRadius[0] = ROOT_WELL_RADIUS;
  // A folds at zoom 1/32 and X at 1/4, against the default 96/64 band edges.
  wellRadius[NESTED_OUTER] = 1000;
  wellRadius[NESTED_INNER] = 200;

  const columns: HierarchyColumns = { parent, wellRadius, depth, subtreeSize };
  return retainSuppliedHierarchy(columns, NODE_COUNT);
}

function fixtureHierarchy(): RetainedHierarchy {
  const parent = new Uint32Array(NODE_COUNT);
  const depth = new Uint16Array(NODE_COUNT);
  const subtreeSize = new Uint32Array(NODE_COUNT).fill(1);
  const wellRadius = new Float32Array(NODE_COUNT).fill(SUBTREE_WELL_RADIUS);

  parent[0] = HIERARCHY_ROOT;
  parent[SUBTREE_ROOT] = 0;
  depth[SUBTREE_ROOT] = 1;
  for (const slot of DESCENDANTS) {
    parent[slot] = SUBTREE_ROOT;
    depth[slot] = 2;
  }
  for (const slot of SIBLINGS) {
    parent[slot] = 0;
    depth[slot] = 1;
  }
  for (let i = NODE_COUNT - 1; i > 0; i--) subtreeSize[parent[i]] += subtreeSize[i];
  wellRadius[0] = ROOT_WELL_RADIUS;

  const columns: HierarchyColumns = { parent, wellRadius, depth, subtreeSize };
  return retainSuppliedHierarchy(columns, NODE_COUNT);
}

/**
 * Positions: a tight subtree cluster left of centre and four siblings bunched
 * to its right, so repulsion drives the proxy in a definite direction rather
 * than leaving it balanced at the origin.
 */
function fixturePositions(): { x: Float32Array; y: Float32Array } {
  const x = new Float32Array(NODE_COUNT);
  const y = new Float32Array(NODE_COUNT);
  x[0] = 0;
  y[0] = 0;
  x[SUBTREE_ROOT] = -20;
  y[SUBTREE_ROOT] = 5;
  DESCENDANTS.forEach((slot, k) => {
    const angle = (2 * Math.PI * k) / DESCENDANTS.length;
    x[slot] = -20 + 3 * Math.cos(angle);
    y[slot] = 5 + 3 * Math.sin(angle);
  });
  SIBLINGS.forEach((slot, k) => {
    x[slot] = 30 + 6 * k;
    y[slot] = -10 + 4 * k;
  });
  return { x, y };
}

// =============================================================================
// A LodHost over the real simulation buffers
// =============================================================================

interface FixupRig {
  readonly controller: LODController;
  /** CPU position shadow, as GraphMother's is: refreshed by readback. */
  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly flags: Uint32Array;
  /** Every `translateNodeRange` the controller issued. */
  readonly translations: { lo: number; hi: number }[];
  refreshFromGpu(): Promise<void>;
  /** Move a slot the way a drag or an external write would, GPU included. */
  place(slot: NodeId, x: number, y: number): void;
  /** Point the camera at a new zoom; the next evaluation sees it. */
  setZoom(scale: number): void;
}

function createRig(
  harness: SimHarness,
  hierarchy: RetainedHierarchy,
  config: Parameters<LODController["setConfig"]>[0] = {},
): FixupRig {
  const { x, y } = fixturePositions();
  const flags = new Uint32Array(NODE_COUNT);
  const translations: { lo: number; hi: number }[] = [];
  let scale = VIEWPORT.scale;

  const writeRange = (lo: number, hi: number): void => {
    harness.writePositionRange(lo, x.subarray(lo, hi), y.subarray(lo, hi));
  };

  const host: LodHost = {
    getHierarchy: () => hierarchy,
    getViewport: () => ({ ...VIEWPORT, scale }),
    getNodePosition: (node) => ({ x: x[node], y: y[node] }),
    getNodeRadius: () => 1,
    getNodeTag: () => 0,
    getNodeWeight: () => 0,
    applyVisibility: (lo, hi, visible) => {
      for (let slot = lo; slot < hi; slot++) {
        if (visible[slot] === 0) flags[slot] |= NODE_FLAG_HIDDEN_LOD;
        else flags[slot] &= ~NODE_FLAG_HIDDEN_LOD;
      }
      harness.setNodeFlags(flags);
    },
    uploadNodeAlpha: () => {},
    uploadNodeMass: (mass) => harness.setNodeMass(mass),
    setCollapsedProxies: () => {},
    // This rig asserts the expand fix-up, which is a position write; the
    // aggregated edge set has no bearing on it and is exercised in
    // tests/gpu/lod_edge_bundle_test.ts.
    aggregateEdges: () => {},
    releaseEdgeAggregation: () => {},
    translateNodeRange: (lo, hi, dx, dy) => {
      for (let slot = lo; slot < hi; slot++) {
        x[slot] += dx;
        y[slot] += dy;
      }
      translations.push({ lo, hi });
      writeRange(lo, hi);
    },
    syncCards: () => {},
    emit: () => {},
  };

  const controller = new LODController(host);
  controller.setConfig({ enabled: true, ...config }, 0);

  return {
    controller,
    x,
    y,
    flags,
    translations,
    async refreshFromGpu(): Promise<void> {
      const live = await harness.readPositions();
      x.set(live.x);
      y.set(live.y);
    },
    place(slot: NodeId, px: number, py: number): void {
      x[slot] = px;
      y[slot] = py;
      writeRange(slot, slot + 1);
    },
    setZoom(next: number): void {
      scale = next;
    },
  };
}

/** Offsets of `members` from `root`, in the given order. */
function offsetsFrom(
  x: Float32Array,
  y: Float32Array,
  root: number,
  members: readonly number[],
): { dx: number; dy: number }[] {
  return members.map((slot) => ({ dx: x[slot] - x[root], dy: y[slot] - y[root] }));
}

/** Offsets of every descendant from the proxy, in slot order. */
function offsetsFromProxy(x: Float32Array, y: Float32Array): { dx: number; dy: number }[] {
  return offsetsFrom(x, y, SUBTREE_ROOT, DESCENDANTS);
}

function assertOffsetsMatch(
  actual: { dx: number; dy: number }[],
  expected: { dx: number; dy: number }[],
  tolerance: number,
  members: readonly number[] = DESCENDANTS,
): void {
  for (let k = 0; k < expected.length; k++) {
    assertAlmostEquals(actual[k].dx, expected[k].dx, tolerance, `slot ${members[k]} dx`);
    assertAlmostEquals(actual[k].dy, expected[k].dy, tolerance, `slot ${members[k]} dy`);
  }
}

async function harnessOver(device: GPUDevice): Promise<SimHarness> {
  const { x, y } = fixturePositions();
  return await createSimHarness(device, {
    nodeCount: NODE_COUNT,
    positionsX: x,
    positionsY: y,
    // One edge between two siblings: enough that the pipeline has an edge
    // buffer to upload, and it touches neither the proxy nor anything folded
    // under it. What the springs do to those two is irrelevant — the claim is
    // about offsets from the proxy, wherever the proxy ends up.
    edgeSources: Uint32Array.from([SIBLINGS[2]]),
    edgeTargets: Uint32Array.from([SIBLINGS[3]]),
  }, { centerStrength: 0 });
}

// =============================================================================
// The crossfade window
// =============================================================================

gpuTest(
  "crossfade window: a collapsing subtree simulates exactly as it did before",
  async (device) => {
    // The cut moves a whole fade ahead of the flags, so for `transitionMs` the
    // subtree is still drawn, still dispatched and still integrated. Anything
    // the collapse changes about the physics inside that window acts on bodies
    // that are still there: mass rolled onto the proxy leaves them weightless
    // inside its aggregate repulsion, and the arrangement they are blown into
    // is the one the flag then freezes and the expand fix-up faithfully
    // restores. So the claim is exact rather than a tolerance on a blowout —
    // mid-fade, the simulation must be indistinguishable from one where no
    // collapse happened at all.
    const collapsing = await harnessOver(device);
    const control = await harnessOver(device);
    try {
      const rig = createRig(collapsing, fixtureHierarchy());
      rig.controller.evaluateNow(0);
      for (const slot of DESCENDANTS) {
        assertEquals(
          rig.flags[slot] & NODE_FLAG_HIDDEN_LOD,
          0,
          `slot ${slot} was flagged at cut time, so this test proves nothing`,
        );
      }

      // Nine ticks is one 150 ms crossfade at 60 fps.
      await collapsing.tick(9);
      await control.tick(9);
      const live = await collapsing.readPositions();
      const expected = await control.readPositions();
      assertOffsetsMatch(
        offsetsFromProxy(live.x, live.y),
        offsetsFromProxy(expected.x, expected.y),
        1e-3,
      );
    } finally {
      collapsing.dispose();
      control.dispose();
    }
  },
);

// =============================================================================
// Round trip
// =============================================================================

gpuTest(
  "expand fix-up: a proxy moved by hand carries its whole subtree with it",
  async (device) => {
    const harness = await harnessOver(device);
    try {
      const rig = createRig(harness, fixtureHierarchy());
      rig.controller.evaluateNow(0);
      const before = offsetsFromProxy(rig.x, rig.y);

      // No simulation between collapse and expand: this isolates the transform
      // from the freezing WP-D owns. The proxy is displaced the way a drag would
      // displace it, which is a larger move than a settled sim would produce.
      rig.place(SUBTREE_ROOT, rig.x[SUBTREE_ROOT] + 37.5, rig.y[SUBTREE_ROOT] - 21.25);
      rig.controller.expandNode(SUBTREE_ROOT, 16);

      // The subtree occupies one contiguous slot run, so the whole fix-up is a
      // single buffer write.
      assertEquals(rig.translations, [{ lo: DESCENDANTS[0], hi: DESCENDANTS.at(-1)! + 1 }]);

      const live = await harness.readPositions();
      assertOffsetsMatch(offsetsFromProxy(live.x, live.y), before, 1e-3);
    } finally {
      harness.dispose();
    }
  },
);

gpuTest("expand fix-up: a fold closed over a live fold restores both drifts", async (device) => {
  // Two levels, and the inner one folds first. When A closes over X, X stops
  // being a proxy without ever expanding — so a diff that only knows about
  // expansion never restores it, and the drift X accumulated while it *was* a
  // proxy is simply lost. The tear that leaves is permanent: the fold outlives
  // the simulation's alpha, and LOD may not reheat it.
  const harness = await harnessOver(device);
  try {
    const rig = createRig(harness, nestedHierarchy(), { minBandCommitFrames: 0 });
    rig.controller.evaluateNow(0);
    assertEquals(rig.controller.isCollapsed(NESTED_INNER), false, "X must start expanded");
    const before = offsetsFrom(rig.x, rig.y, NESTED_INNER, NESTED_LEAVES);

    // X folds on its own account and then drifts as a live proxy, its leaves
    // frozen underneath it.
    rig.setZoom(0.25);
    rig.controller.evaluateNow(100);
    rig.controller.tick(1000);
    assert(rig.controller.isCollapsed(NESTED_INNER), "X must be the proxy for its leaves");
    for (const slot of NESTED_LEAVES) {
      assertEquals(rig.flags[slot] & NODE_FLAG_HIDDEN_LOD, NODE_FLAG_HIDDEN_LOD);
    }
    await harness.tick(60);
    await rig.refreshFromGpu();
    const innerDrift = Math.hypot(rig.x[NESTED_INNER] + 20, rig.y[NESTED_INNER] - 5);
    assert(innerDrift > 3, `X must actually have drifted, got ${innerDrift}`);

    // A closes over it. X's own fold dissolves into A's, and its outstanding
    // drift has to be paid to its leaves here or never.
    rig.setZoom(0.03125);
    rig.controller.evaluateNow(2000);
    rig.controller.tick(3000);
    assert(rig.controller.isCollapsed(NESTED_OUTER), "A must be the proxy");
    assertEquals(rig.controller.isCollapsed(NESTED_INNER), false, "X's fold has dissolved");
    await harness.tick(60);
    await rig.refreshFromGpu();

    // Everything comes back at once, so A's fix-up moves the whole subtree —
    // which is only correct because X's leaves are already square with X.
    rig.setZoom(1);
    rig.controller.evaluateNow(4000);
    rig.controller.tick(5000);
    assertEquals(rig.controller.isCollapsed(NESTED_OUTER), false);

    const live = await harness.readPositions();
    assertOffsetsMatch(
      offsetsFrom(live.x, live.y, NESTED_INNER, NESTED_LEAVES),
      before,
      1e-3,
      NESTED_LEAVES,
    );
  } finally {
    harness.dispose();
  }
});

gpuTest("expand fix-up: a proxy that drifted under the simulation carries its subtree", async (
  device,
) => {
  const harness = await harnessOver(device);
  try {
    const rig = createRig(harness, fixtureHierarchy());
    rig.controller.evaluateNow(0);
    const before = offsetsFromProxy(rig.x, rig.y);

    // Land the HIDDEN_LOD flags: they are deferred behind the crossfade, and
    // until they land the subtree is still simulating.
    rig.controller.tick(1000);
    for (const slot of DESCENDANTS) {
      assertEquals(rig.flags[slot] & NODE_FLAG_HIDDEN_LOD, NODE_FLAG_HIDDEN_LOD);
    }

    await harness.tick(60);
    await rig.refreshFromGpu();
    // Larger than the subtree's own radius: the fix-up is answering a real
    // detachment, not rounding.
    assert(
      Math.abs(rig.x[SUBTREE_ROOT] + 20) > 3,
      `the proxy must actually have drifted, got x=${rig.x[SUBTREE_ROOT]}`,
    );

    rig.controller.expandNode(SUBTREE_ROOT, 1016);
    const live = await harness.readPositions();
    assertOffsetsMatch(offsetsFromProxy(live.x, live.y), before, 1e-3);
  } finally {
    harness.dispose();
  }
});
