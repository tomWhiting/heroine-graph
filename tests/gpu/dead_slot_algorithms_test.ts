/**
 * GPU tests for dead-slot masking at the ALGORITHM level.
 *
 * tests/gpu/node_flags_test.ts covers the shared pipeline (N² repulsion,
 * springs, integrate, collision). These tests cover the algorithm plugins
 * whose force passes replace it, using the strongest available assertion:
 * run the same logical graph twice — once packed into slots 0..n-1, once
 * with dead slots wedged between the live nodes — and require the live
 * results to agree. Any unmasked read of a dead slot changes one run and
 * not the other.
 *
 * - Community layout: modulated repulsion must not see a hole at the origin
 *   as a repelling body, and centroid accumulation must not fold a hole's
 *   stale community id into its cluster's centroid or member count.
 * - t-FDP: repulsion masking (main_masked) end-to-end, plus the per-edge
 *   attraction guard — an edge whose endpoint is a dead slot exerts no pull.
 */

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import {
  createAlgorithmSimHarness,
  GPU_SKIP_MESSAGE,
  type HarnessForceAlgorithm,
  type HarnessGraphData,
  loadModuleInliningWgsl,
  NODE_FLAG_DEAD,
  probeAdapter,
} from "../helpers/gpu.ts";

const adapter = await probeAdapter();
if (!adapter) {
  console.warn(`[gpu] ${GPU_SKIP_MESSAGE}`);
}

function gpuTest(name: string, fn: (device: GPUDevice) => Promise<void>): void {
  Deno.test({
    name,
    ignore: adapter === null,
    sanitizeResources: false,
    sanitizeOps: false,
    async fn() {
      const device = await adapter!.requestDevice();
      try {
        await fn(device);
      } finally {
        device.destroy();
      }
    },
  });
}

/**
 * One slot in the interleaved layout. Dead slots carry a position and a
 * community id exactly as a removed node's slot does: the position is what
 * graph.ts zeroes it to (or whatever stale value a test wants to prove is
 * invisible) and the community id is the former occupant's, never cleared.
 */
interface SlotSpec {
  x: number;
  y: number;
  community: number;
  dead: boolean;
}

/**
 * The same logical graph in two slot layouts. `interleaved` keeps the dead
 * slots; `packed` drops them and renumbers, preserving the relative order of
 * the live nodes so force accumulation loops visit them in the same sequence
 * (a prerequisite for bit-identical float sums).
 */
interface SlotLayoutPair {
  interleaved: { graph: HarnessGraphData; communities: Uint32Array; liveSlots: number[] };
  packed: { graph: HarnessGraphData; communities: Uint32Array };
}

/**
 * Builds the interleaved/packed pair from a slot spec and a live-index edge
 * list. Edges are given as index pairs into the LIVE nodes (in spec order)
 * and are mapped into each layout's slot numbering.
 */
function buildSlotLayoutPair(
  slots: SlotSpec[],
  liveEdges: ReadonlyArray<readonly [number, number]>,
): SlotLayoutPair {
  const liveSlots = slots.map((s, i) => (s.dead ? -1 : i)).filter((i) => i >= 0);

  const interleaved: HarnessGraphData = {
    nodeCount: slots.length,
    positionsX: new Float32Array(slots.map((s) => s.x)),
    positionsY: new Float32Array(slots.map((s) => s.y)),
    edgeSources: new Uint32Array(liveEdges.map(([a]) => liveSlots[a])),
    edgeTargets: new Uint32Array(liveEdges.map(([, b]) => liveSlots[b])),
    flags: new Uint32Array(slots.map((s) => (s.dead ? NODE_FLAG_DEAD : 0))),
  };

  const live = liveSlots.map((slot) => slots[slot]);
  const packed: HarnessGraphData = {
    nodeCount: live.length,
    positionsX: new Float32Array(live.map((s) => s.x)),
    positionsY: new Float32Array(live.map((s) => s.y)),
    edgeSources: new Uint32Array(liveEdges.map(([a]) => a)),
    edgeTargets: new Uint32Array(liveEdges.map(([, b]) => b)),
    flags: new Uint32Array(live.length),
  };

  return {
    interleaved: {
      graph: interleaved,
      communities: new Uint32Array(slots.map((s) => s.community)),
      liveSlots,
    },
    packed: {
      graph: packed,
      communities: new Uint32Array(live.map((s) => s.community)),
    },
  };
}

// ---------------------------------------------------------------------------
// Community layout (repulsion_community + cluster_accumulate/attract)
// ---------------------------------------------------------------------------

/** Community layout exposes uploadCommunityIds beyond the ForceAlgorithm API. */
type CommunityAlgorithm = HarnessForceAlgorithm & {
  uploadCommunityIds(
    device: GPUDevice,
    assignments: Uint32Array,
    communityCount: number,
  ): void;
};

function loadCommunityAlgorithm(): Promise<
  { createCommunityLayoutAlgorithm(): CommunityAlgorithm }
> {
  return loadModuleInliningWgsl(
    new URL(
      "../../packages/core/src/simulation/algorithms/community.ts",
      import.meta.url,
    ),
  );
}

/**
 * Two clusters of three nodes each, with three dead slots wedged between
 * them: one at the origin (where graph.ts parks a removed node), one inside
 * cluster 0 and one inside cluster 1. Each hole keeps a live community id,
 * so an unmasked cluster_accumulate would both drag that cluster's centroid
 * toward the hole and inflate the member count the attraction divides by.
 */
const COMMUNITY_SLOTS: SlotSpec[] = [
  { x: -60, y: -20, community: 0, dead: false },
  { x: 0, y: 0, community: 0, dead: true },
  { x: -40, y: 10, community: 0, dead: false },
  { x: 50, y: -30, community: 1, dead: false },
  { x: -55, y: 5, community: 0, dead: true },
  { x: -70, y: 30, community: 0, dead: false },
  { x: 70, y: 15, community: 1, dead: false },
  { x: 55, y: 0, community: 1, dead: true },
  { x: 45, y: 25, community: 1, dead: false },
];

/**
 * Live-index edges: a triangle inside each cluster plus one bridge. Live
 * indices 0, 1, 3 are cluster 0; 2, 4, 5 are cluster 1. Degrees feed the
 * mass weighting in repulsion_community and cluster_attract, so the edge
 * set also exercises the degree upload path over a slot range with holes.
 */
const COMMUNITY_EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 3],
  [0, 3],
  [2, 4],
  [4, 5],
  [2, 5],
  [1, 2],
];

gpuTest(
  "GPU community layout: dead slots are inert and live layout matches the packed graph",
  async (device) => {
    const mod = await loadCommunityAlgorithm();
    const { interleaved, packed } = buildSlotLayoutPair(COMMUNITY_SLOTS, COMMUNITY_EDGES);

    const run = async (
      graph: HarnessGraphData,
      communities: Uint32Array,
    ) => {
      const algorithm = mod.createCommunityLayoutAlgorithm();
      const harness = await createAlgorithmSimHarness(
        device,
        algorithm,
        graph,
        {},
        undefined,
        {
          // Community ids must be uploaded after createBuffers allocates the
          // id buffer and before the bind groups reference it — the same
          // ordering graph.ts uses after Louvain detection.
          onAlgorithmBuffers: () => {
            algorithm.uploadCommunityIds(device, communities, 2);
          },
        },
      );
      try {
        await harness.tick(40);
        return await harness.readPositions();
      } finally {
        harness.dispose();
      }
    };

    const withHoles = await run(interleaved.graph, interleaved.communities);
    const compact = await run(packed.graph, packed.communities);

    // Dead slots never move: their positions come back exactly as uploaded
    for (let slot = 0; slot < COMMUNITY_SLOTS.length; slot++) {
      const spec = COMMUNITY_SLOTS[slot];
      if (!spec.dead) continue;
      assertEquals(withHoles.x[slot], spec.x, `dead slot ${slot} moved in x`);
      assertEquals(withHoles.y[slot], spec.y, `dead slot ${slot} moved in y`);
    }

    // Live nodes land bit-identically to the same graph without the holes
    interleaved.liveSlots.forEach((slot, packedIdx) => {
      assertEquals(
        withHoles.x[slot],
        compact.x[packedIdx],
        `slot ${slot}: dead slots leaked into x`,
      );
      assertEquals(
        withHoles.y[slot],
        compact.y[packedIdx],
        `slot ${slot}: dead slots leaked into y`,
      );
    });

    // Sanity: the layout actually ran (clusters contracted toward their
    // centroids), so the equality above is not comparing two no-ops
    assert(
      interleaved.liveSlots.some(
        (slot) => withHoles.x[slot] !== COMMUNITY_SLOTS[slot].x,
      ),
      "community layout never moved any live node",
    );
  },
);

// ---------------------------------------------------------------------------
// t-FDP (t_fdp main_masked + t_fdp_attraction)
// ---------------------------------------------------------------------------

function loadTFdpAlgorithm(): Promise<{ createTFdpAlgorithm(): HarnessForceAlgorithm }> {
  return loadModuleInliningWgsl(
    new URL(
      "../../packages/core/src/simulation/algorithms/t-fdp.ts",
      import.meta.url,
    ),
  );
}

/** A five-node path with two dead slots wedged into the middle of the chain. */
const T_FDP_SLOTS: SlotSpec[] = [
  { x: -80, y: 0, community: 0, dead: false },
  { x: 0, y: 0, community: 0, dead: true },
  { x: -40, y: 20, community: 0, dead: false },
  { x: 5, y: -10, community: 0, dead: false },
  { x: -10, y: 15, community: 0, dead: true },
  { x: 45, y: 25, community: 0, dead: false },
  { x: 85, y: -5, community: 0, dead: false },
];

/** Path edges over the live nodes, in live-index space. */
const T_FDP_EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
];

async function runTFdp(
  device: GPUDevice,
  mod: { createTFdpAlgorithm(): HarnessForceAlgorithm },
  graph: HarnessGraphData,
): Promise<{ x: Float32Array; y: Float32Array }> {
  const harness = await createAlgorithmSimHarness(
    device,
    mod.createTFdpAlgorithm(),
    graph,
  );
  try {
    await harness.tick(40);
    return await harness.readPositions();
  } finally {
    harness.dispose();
  }
}

/**
 * Tolerance for t-FDP position comparisons. t-FDP handles its own springs,
 * so its per-edge attraction pass accumulates with a CAS loop whose float
 * summation order depends on lane scheduling; two runs of the same graph can
 * differ in the last bits and drift slightly over 40 ticks. The layout spans
 * ~170 units, so 1e-2 is four orders of magnitude below any real leak (a
 * phantom body at the origin moves these nodes by tens of units).
 */
const T_FDP_TOLERANCE = 1e-2;

gpuTest(
  "GPU t-FDP: dead slots are inert and live layout matches the packed graph",
  async (device) => {
    const mod = await loadTFdpAlgorithm();
    const { interleaved, packed } = buildSlotLayoutPair(T_FDP_SLOTS, T_FDP_EDGES);

    const withHoles = await runTFdp(device, mod, interleaved.graph);
    const compact = await runTFdp(device, mod, packed.graph);

    for (let slot = 0; slot < T_FDP_SLOTS.length; slot++) {
      const spec = T_FDP_SLOTS[slot];
      if (!spec.dead) continue;
      assertEquals(withHoles.x[slot], spec.x, `dead slot ${slot} moved in x`);
      assertEquals(withHoles.y[slot], spec.y, `dead slot ${slot} moved in y`);
    }

    interleaved.liveSlots.forEach((slot, packedIdx) => {
      assertAlmostEquals(
        withHoles.x[slot],
        compact.x[packedIdx],
        T_FDP_TOLERANCE,
        `slot ${slot}: dead slots leaked into x`,
      );
      assertAlmostEquals(
        withHoles.y[slot],
        compact.y[packedIdx],
        T_FDP_TOLERANCE,
        `slot ${slot}: dead slots leaked into y`,
      );
    });

    assert(
      interleaved.liveSlots.some(
        (slot) => withHoles.x[slot] !== T_FDP_SLOTS[slot].x,
      ),
      "t-FDP never moved any live node",
    );
  },
);

gpuTest(
  "GPU t-FDP: an edge pointing at a dead slot exerts no attraction",
  async (device) => {
    // The mutation paths cascade-remove a node's edges before flagging its
    // slot dead, so production never produces this state — but the guard in
    // t_fdp_attraction.comp.wgsl (like springs_simple/fa2/linlog) promises
    // that a dangling endpoint is inert, and that promise is what this pins.
    // Without the guard the dangling edge drags live node 0 toward the hole
    // at the origin every tick.
    const mod = await loadTFdpAlgorithm();
    const { interleaved } = buildSlotLayoutPair(T_FDP_SLOTS, T_FDP_EDGES);

    const dangling: HarnessGraphData = {
      ...interleaved.graph,
      edgeSources: Uint32Array.from([...interleaved.graph.edgeSources, 0]),
      edgeTargets: Uint32Array.from([...interleaved.graph.edgeTargets, 1]),
    };

    const clean = await runTFdp(device, mod, interleaved.graph);
    const withDanglingEdge = await runTFdp(device, mod, dangling);

    interleaved.liveSlots.forEach((slot) => {
      assertAlmostEquals(
        withDanglingEdge.x[slot],
        clean.x[slot],
        T_FDP_TOLERANCE,
        `slot ${slot}: dangling edge pulled x`,
      );
      assertAlmostEquals(
        withDanglingEdge.y[slot],
        clean.y[slot],
        T_FDP_TOLERANCE,
        `slot ${slot}: dangling edge pulled y`,
      );
    });

    // The dead endpoint itself also stays put
    assertEquals(withDanglingEdge.x[1], T_FDP_SLOTS[1].x, "dead endpoint moved in x");
    assertEquals(withDanglingEdge.y[1], T_FDP_SLOTS[1].y, "dead endpoint moved in y");
  },
);
