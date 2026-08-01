/**
 * Hit testing: column fast path vs per-node accessor path.
 *
 * `HitTester` scans a provider's typed arrays directly when the provider
 * exposes them, and falls back to the `getNodeIds`/`getNodePosition`/
 * `getNodeRadius` accessors otherwise. Both must answer identically — the
 * fast path is a performance change, not a behaviour change.
 *
 * The small-graph table pins absolute expectations (so a mutation to shared
 * geometry cannot hide behind parity), and the mid-size fixture pins parity
 * across thousands of probes (so a mutation to either loop alone shows).
 *
 * The providers here are built exactly as `GraphMother.updateHitTester`
 * builds them, including the interleaved radius stride.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  createHitTester,
  type EdgeProvider,
  type HitResult,
  type PositionProvider,
} from "../../packages/core/src/interaction/hit_test.ts";
import type { NodeId, Vec2 } from "../../packages/core/src/types.ts";
import { mulberry32 } from "../fixtures/prng.ts";

/** Attribute stride of `parsedGraph.nodeAttributes` (NODE_ATTR_FLOATS). */
const ATTR_STRIDE = 8;

interface Fixture {
  nodeCount: number;
  positionsX: Float32Array;
  positionsY: Float32Array;
  nodeAttributes: Float32Array;
  edgeSources: Uint32Array;
  edgeTargets: Uint32Array;
}

function fixtureFrom(
  nodes: readonly (readonly [number, number, number])[],
  edges: readonly (readonly [number, number])[],
): Fixture {
  const nodeCount = nodes.length;
  const positionsX = new Float32Array(nodeCount);
  const positionsY = new Float32Array(nodeCount);
  const nodeAttributes = new Float32Array(nodeCount * ATTR_STRIDE);
  nodes.forEach(([x, y, radius], i) => {
    positionsX[i] = x;
    positionsY[i] = y;
    nodeAttributes[i * ATTR_STRIDE] = radius;
  });
  return {
    nodeCount,
    positionsX,
    positionsY,
    nodeAttributes,
    edgeSources: Uint32Array.from(edges, (e) => e[0]),
    edgeTargets: Uint32Array.from(edges, (e) => e[1]),
  };
}

/** Random layout, radii spanning the range a code graph produces. */
function randomFixture(seed: number, nodeCount: number, edgeCount: number): Fixture {
  const rng = mulberry32(seed);
  const positionsX = new Float32Array(nodeCount);
  const positionsY = new Float32Array(nodeCount);
  const nodeAttributes = new Float32Array(nodeCount * ATTR_STRIDE);
  for (let i = 0; i < nodeCount; i++) {
    positionsX[i] = (rng() - 0.5) * 2000;
    positionsY[i] = (rng() - 0.5) * 2000;
    nodeAttributes[i * ATTR_STRIDE] = 1 + rng() * 11;
  }
  const edgeSources = new Uint32Array(edgeCount);
  const edgeTargets = new Uint32Array(edgeCount);
  for (let i = 0; i < edgeCount; i++) {
    edgeSources[i] = Math.floor(rng() * nodeCount);
    edgeTargets[i] = Math.floor(rng() * nodeCount);
  }
  return { nodeCount, positionsX, positionsY, nodeAttributes, edgeSources, edgeTargets };
}

/** The accessor-only shape: a generator, a `Vec2` per node, a call per radius. */
function accessorProviders(f: Fixture): { nodes: PositionProvider; edges: EdgeProvider } {
  return {
    nodes: {
      getNodePosition: (nodeId: NodeId): Vec2 | undefined => {
        if (nodeId < 0 || nodeId >= f.nodeCount) return undefined;
        return { x: f.positionsX[nodeId], y: f.positionsY[nodeId] };
      },
      getNodeRadius: (nodeId: NodeId): number | undefined => {
        if (nodeId < 0 || nodeId >= f.nodeCount) return undefined;
        return f.nodeAttributes[nodeId * ATTR_STRIDE];
      },
      getNodeIds: function* () {
        for (let i = 0; i < f.nodeCount; i++) yield i;
      },
      getNodeCount: () => f.nodeCount,
    },
    edges: {
      getEdges: function* () {
        let edgeId = 0;
        for (let i = 0; i < f.edgeSources.length; i++) {
          yield [edgeId++, f.edgeSources[i], f.edgeTargets[i]];
        }
      },
      getEdgeCount: () => f.edgeSources.length,
    },
  };
}

/** The shipped shape: accessors plus the columns the scans actually use. */
function columnProviders(f: Fixture): { nodes: PositionProvider; edges: EdgeProvider } {
  const { nodes, edges } = accessorProviders(f);
  return {
    nodes: {
      ...nodes,
      getNodeColumns: () => ({
        count: f.nodeCount,
        x: f.positionsX,
        y: f.positionsY,
        radii: f.nodeAttributes,
        radiusStride: ATTR_STRIDE,
        radiusOffset: 0,
      }),
    },
    edges: {
      ...edges,
      getEdgeColumns: () => ({
        count: f.edgeSources.length,
        sources: f.edgeSources,
        targets: f.edgeTargets,
      }),
    },
  };
}

function testerFor(providers: { nodes: PositionProvider; edges: EdgeProvider }) {
  const tester = createHitTester();
  tester.setPositionProvider(providers.nodes);
  tester.setEdgeProvider(providers.edges);
  return tester;
}

// Nodes: [x, y, radius]. Hit tolerance is radius + 2.
const SMALL_NODES = [
  [0, 0, 10], // 0
  [30, 0, 5], // 1
  [0, 40, 3], // 2
  [100, 100, 8], // 3
  [-50, 0, 1], // 4
  [200, 0, 20], // 5 — overlaps 6
  [205, 0, 4], // 6
] as const;

// Edges: [source, target].
const SMALL_EDGES = [
  [0, 1], // 0 — horizontal (0,0)-(30,0)
  [0, 2], // 1 — vertical   (0,0)-(0,40)
  [3, 1], // 2 — diagonal  (100,100)-(30,0)
] as const;

interface NodeProbe {
  readonly at: readonly [number, number];
  readonly expect: NodeId | null;
  readonly why: string;
}

const NODE_PROBES: readonly NodeProbe[] = [
  { at: [0, 0], expect: 0, why: "dead centre" },
  { at: [11.9, 0], expect: 0, why: "inside the 12-unit tolerance" },
  { at: [12, 0], expect: 0, why: "exactly on the tolerance boundary (inclusive)" },
  { at: [12.01, 0], expect: null, why: "one hundredth past the boundary" },
  { at: [23, 0], expect: 1, why: "exactly on node 1's boundary" },
  { at: [22.9, 0], expect: null, why: "between two nodes, outside both" },
  { at: [200, 0], expect: 5, why: "centre of the large node of an overlapping pair" },
  { at: [203, 0], expect: 6, why: "overlap: closest centre wins, not the larger disc" },
  { at: [222, 0], expect: 5, why: "large node's boundary, small node out of range" },
  { at: [222.5, 0], expect: null, why: "past the large node's boundary" },
  { at: [-50, 0], expect: 4, why: "smallest radius still hittable" },
  { at: [0, 40], expect: 2, why: "isolated node" },
  { at: [5000, 5000], expect: null, why: "empty space" },
];

interface EdgeProbe {
  readonly at: readonly [number, number];
  readonly expect: number | null;
  readonly why: string;
}

const EDGE_PROBES: readonly EdgeProbe[] = [
  { at: [15, 3], expect: 0, why: "3 units off the horizontal edge" },
  { at: [15, 4.999], expect: 0, why: "a thousandth inside the 5-unit tolerance" },
  { at: [15, 5], expect: null, why: "exactly on the tolerance boundary (exclusive)" },
  { at: [0, 20], expect: 1, why: "on the vertical edge" },
  { at: [0, 2], expect: 1, why: "both edges in range, nearest wins" },
  { at: [-10, 0], expect: null, why: "past a shared endpoint" },
  { at: [65, 50], expect: 2, why: "midpoint of the diagonal edge" },
  { at: [15, -4], expect: 0, why: "other side of the horizontal edge" },
  { at: [5000, 5000], expect: null, why: "empty space" },
];

Deno.test("hit test: small graph node probes, both provider shapes", () => {
  const f = fixtureFrom(SMALL_NODES, SMALL_EDGES);
  for (
    const [label, providers] of [
      ["accessors", accessorProviders(f)],
      ["columns", columnProviders(f)],
    ] as const
  ) {
    const tester = testerFor(providers);
    for (const probe of NODE_PROBES) {
      const hit = tester.hitTestNode(probe.at[0], probe.at[1]);
      assertEquals(
        hit?.nodeId ?? null,
        probe.expect,
        `${label} @ (${probe.at}) — ${probe.why}`,
      );
      if (hit) {
        assertEquals(hit.position.x, f.positionsX[hit.nodeId]);
        assertEquals(hit.position.y, f.positionsY[hit.nodeId]);
      }
    }
  }
});

Deno.test("hit test: small graph edge probes, both provider shapes", () => {
  const f = fixtureFrom(SMALL_NODES, SMALL_EDGES);
  for (
    const [label, providers] of [
      ["accessors", accessorProviders(f)],
      ["columns", columnProviders(f)],
    ] as const
  ) {
    const tester = testerFor(providers);
    for (const probe of EDGE_PROBES) {
      const hit = tester.hitTestEdge(probe.at[0], probe.at[1]);
      assertEquals(
        hit?.edgeId ?? null,
        probe.expect,
        `${label} @ (${probe.at}) — ${probe.why}`,
      );
      if (hit) {
        assertEquals(hit.sourceId, f.edgeSources[hit.edgeId]);
        assertEquals(hit.targetId, f.edgeTargets[hit.edgeId]);
      }
    }
  }
});

Deno.test("hit test: node priority and rect query, both provider shapes", () => {
  const f = fixtureFrom(SMALL_NODES, SMALL_EDGES);
  for (
    const [label, providers] of [
      ["accessors", accessorProviders(f)],
      ["columns", columnProviders(f)],
    ] as const
  ) {
    const tester = testerFor(providers);

    // (5, 1) sits inside node 0 and within tolerance of both edges out of it.
    const both = tester.hitTest(5, 1);
    assertEquals(both?.type, "node", `${label}: nodes win over edges`);
    assertEquals((both as HitResult & { nodeId: NodeId }).nodeId, 0, label);

    // (15, 3) is off every node but on edge 0.
    assertEquals(tester.hitTest(15, 3)?.type, "edge", label);

    assertEquals(tester.findNodesInRect(-1, -1, 31, 1), [0, 1], label);
    assertEquals(tester.findNodesInRect(300, 300, 400, 400), [], label);
    assertEquals(tester.findNodesInRect(-100, -100, 300, 300).length, f.nodeCount, label);
    // Degenerate rect on y = 0: every bound is exact, so it pins all four
    // comparisons as inclusive.
    assertEquals(tester.findNodesInRect(0, 0, 205, 0), [0, 1, 5, 6], label);
  }
});

Deno.test("hit test: column path agrees with accessor path on a mid-size graph", () => {
  const f = randomFixture(0x5EED, 5000, 8000);
  const accessors = testerFor(accessorProviders(f));
  const columns = testerFor(columnProviders(f));

  const rng = mulberry32(0xC0FFEE);
  let nodeHits = 0;
  let edgeHits = 0;

  // Probes: every 50th node centre (guaranteed hits), plus random points
  // across the layout (mostly misses, some grazing hits).
  const probes: [number, number][] = [];
  for (let i = 0; i < f.nodeCount; i += 50) probes.push([f.positionsX[i], f.positionsY[i]]);
  for (let i = 0; i < 400; i++) probes.push([(rng() - 0.5) * 2200, (rng() - 0.5) * 2200]);

  for (const [x, y] of probes) {
    const a = accessors.hitTestNode(x, y);
    const c = columns.hitTestNode(x, y);
    assertEquals(c, a, `node hit @ (${x}, ${y})`);
    if (c) nodeHits++;

    const ae = accessors.hitTestEdge(x, y);
    const ce = columns.hitTestEdge(x, y);
    assertEquals(ce, ae, `edge hit @ (${x}, ${y})`);
    if (ce) edgeHits++;

    assertEquals(columns.hitTest(x, y), accessors.hitTest(x, y), `combined @ (${x}, ${y})`);
  }

  // Guard the guard: a probe set that never hits anything would pass vacuously.
  assert(nodeHits >= 100, `expected node hits, got ${nodeHits}`);
  assert(edgeHits > 0, `expected edge hits, got ${edgeHits}`);

  assertEquals(
    columns.findNodesInRect(-100, -100, 100, 100),
    accessors.findNodesInRect(-100, -100, 100, 100),
  );
});

Deno.test("hit test: slots past the scanned count are invisible to both paths", () => {
  // `parsedGraph` arrays run to buffer capacity while `nodeCount` is the
  // high-water mark, so slot 7 below holds a real coordinate that must not be
  // hit, and edge 3 pointing at it must not be hit either.
  const f = fixtureFrom([...SMALL_NODES, [500, 500, 10]], [...SMALL_EDGES, [0, 7]]);
  f.nodeCount = SMALL_NODES.length;

  for (
    const [label, providers] of [
      ["accessors", accessorProviders(f)],
      ["columns", columnProviders(f)],
    ] as const
  ) {
    const tester = testerFor(providers);
    assertEquals(tester.hitTestNode(500, 500), null, `${label}: node past the count`);
    assertEquals(tester.hitTestEdge(250, 250), null, `${label}: edge to a node past the count`);
    assertEquals(tester.findNodesInRect(490, 490, 510, 510), [], label);
    // The rest of the graph is unaffected.
    assertEquals(tester.hitTestNode(0, 0)?.nodeId, 0, label);
    assertEquals(tester.hitTestEdge(15, 3)?.edgeId, 0, label);
  }
});

Deno.test("hit test: a provider whose columns are unavailable falls back to accessors", () => {
  const f = fixtureFrom(SMALL_NODES, SMALL_EDGES);
  const { nodes, edges } = accessorProviders(f);
  const tester = createHitTester();
  tester.setPositionProvider({ ...nodes, getNodeColumns: () => null });
  tester.setEdgeProvider({ ...edges, getEdgeColumns: () => null });

  for (const probe of NODE_PROBES) {
    assertEquals(tester.hitTestNode(probe.at[0], probe.at[1])?.nodeId ?? null, probe.expect);
  }
  for (const probe of EDGE_PROBES) {
    assertEquals(tester.hitTestEdge(probe.at[0], probe.at[1])?.edgeId ?? null, probe.expect);
  }
});

/**
 * A full-miss scan at Meridian's stated ceiling: every node and every edge is
 * visited and rejected, which is the worst case and also the common one
 * (pointer over empty canvas). Through the shipped provider this cost
 * 12.85 ms/pointermove before the column path; the bound below is generous
 * enough to survive a loaded CI box while still failing on any return to the
 * per-node-allocation scan.
 */
Deno.test({
  name: "hit test: 220K full-miss scan stays far under the pre-fix 12.85 ms",
  ignore: Deno.env.get("GRAPHMOTHER_SKIP_PERF") === "1",
  fn: () => {
    const generationStart = performance.now();
    const f = randomFixture(0xBEEF, 220_000, 253_000);
    const generationMs = performance.now() - generationStart;

    const tester = testerFor(columnProviders(f));

    // Far outside the layout, so nothing is ever within tolerance.
    const probeX = 1e6;
    const probeY = 1e6;

    for (let i = 0; i < 3; i++) {
      assertEquals(tester.hitTestNode(probeX, probeY), null);
      assertEquals(tester.hitTestEdge(probeX, probeY), null);
    }

    // Best of N, not the median: this machine may be sharing cores with
    // anything, and the fastest observed run is the one least polluted by
    // whatever else was scheduled. A regression to the old scan shape shows
    // up in every sample, including the best.
    const nodeSamples: number[] = [];
    const edgeSamples: number[] = [];
    for (let i = 0; i < 7; i++) {
      const nodeStart = performance.now();
      tester.hitTestNode(probeX, probeY);
      const edgeStart = performance.now();
      tester.hitTestEdge(probeX, probeY);
      const end = performance.now();
      nodeSamples.push(edgeStart - nodeStart);
      edgeSamples.push(end - edgeStart);
    }
    const nodeBest = Math.min(...nodeSamples);
    const edgeBest = Math.min(...edgeSamples);
    const total = nodeBest + edgeBest;

    console.log(
      `220K/253K fixture generated in ${generationMs.toFixed(1)} ms; ` +
        `full-miss best of 7: nodes ${nodeBest.toFixed(3)} ms, ` +
        `edges ${edgeBest.toFixed(3)} ms, total ${total.toFixed(3)} ms`,
    );

    assert(total < 4, `full-miss scan took ${total.toFixed(3)} ms, expected < 4 ms`);
  },
});
