/**
 * GPU tests for LOD edge aggregation (WP-K).
 *
 * The bug this fixes is a bug of omission. `springs_simple.comp.wgsl` skips
 * any edge with an inert endpoint, so collapsing a subtree does not transfer
 * its cross-cutting dependency attractions to the proxy standing in for it —
 * it *deletes* them. A collapsed module exerts no pull at all on the modules
 * it imports, and the collapsed layout therefore differs structurally from the
 * expanded one, which is SC-002 broken at the root.
 *
 * Four claims:
 *
 * - **A collapsed pair still pulls.** With every force but the spring turned
 *   off, two proxies joined only by hidden-to-hidden imports do not move at
 *   all today; with the aggregation they move toward each other. That is the
 *   non-zero force the design asks for, measured as displacement rather than
 *   inferred.
 * - **The collapsed layout approximates the expanded one (SC-002).** Bundling
 *   moves the visible frontier much closer to where the fully-expanded run put
 *   it than dropping the edges does.
 * - **SC-005: with no cut, nothing changes.** The un-cut spring pass is
 *   bit-identical to the shipped shader with this work package's weight and
 *   inverse-mass terms textually removed — the baseline is the code under
 *   test minus the change, so it cannot go stale.
 * - **Dispatch tracks the cut, and the list is a hint.** With 90 % of a code
 *   tree hidden the spring pass covers 544 entries rather than all 2 874
 *   edges, and a deliberately stale list — the whole edge array, including
 *   edges into hidden nodes — leaves the layout alone, because the shader
 *   still masks under the gather.
 */

import { assert, assertEquals, assertStrictEquals } from "jsr:@std/assert@^1";
import {
  createSimHarness,
  GPU_SKIP_MESSAGE,
  HARNESS_ALPHA_DECAY,
  loadPipelineModule,
  NODE_FLAG_HIDDEN_LOD,
  probeAdapter,
  requestHarnessDevice,
} from "../helpers/gpu.ts";
import { assertFrozen, countNonFinite, type PositionSnapshot } from "../helpers/invariants.ts";
import { referenceAggregateLodEdges } from "../helpers/edge_aggregation.ts";
import { CODE_TREE_SCALES, type CodeTreeGraph, generateCodeTree } from "../fixtures/code_tree.ts";
import { decodeEdgeAggregation } from "../../packages/core/src/lod/edge_aggregation.ts";
import { HIERARCHY_ROOT } from "../../packages/core/src/graph/hierarchy.ts";
import { validateForceConfig } from "../../packages/core/src/simulation/config.ts";

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
// Fixture: two modules of leaves, joined only by cross-cutting imports
// =============================================================================

/** Leaves per module in {@link twoModules}. */
const LEAVES = 8;

/** Half the opening separation of the two module proxies. */
const HALF_SEPARATION = 120;

/**
 * Root 0, modules 1 and 2, `LEAVES` leaves each, and one import per leaf pair.
 *
 * The modules open far enough apart that the bundle spring is in tension, and
 * the leaves are seeded around their own module so the expanded run has
 * somewhere coherent to settle.
 */
interface TwoModules {
  readonly nodeCount: number;
  readonly parent: Uint32Array;
  readonly positionsX: Float32Array;
  readonly positionsY: Float32Array;
  readonly edgeSources: Uint32Array;
  readonly edgeTargets: Uint32Array;
  readonly depths: Float32Array;
  /** Slots left on screen when both modules collapse. */
  readonly frontier: Uint32Array;
  /** Per-slot flags with the leaves marked hidden. */
  readonly collapsedFlags: Uint32Array;
  /** Per-slot visibility for the aggregation: 1 for the frontier. */
  readonly collapsedVisible: Uint8Array;
  /** Per-slot mass with each module carrying its leaves (rollUpMass's answer). */
  readonly collapsedMass: Float32Array;
}

function twoModules(withContainmentEdges: boolean): TwoModules {
  const nodeCount = 3 + LEAVES * 2;
  const parent = new Uint32Array(nodeCount).fill(HIERARCHY_ROOT);
  const positionsX = new Float32Array(nodeCount);
  const positionsY = new Float32Array(nodeCount);
  const depths = new Float32Array(nodeCount);

  parent[1] = 0;
  parent[2] = 0;
  depths[1] = 1;
  depths[2] = 1;
  positionsX[1] = -HALF_SEPARATION;
  positionsX[2] = HALF_SEPARATION;

  for (let k = 0; k < LEAVES; k++) {
    const angle = (k / LEAVES) * Math.PI * 2;
    for (const module of [1, 2]) {
      const slot = leafSlot(module, k);
      parent[slot] = module;
      depths[slot] = 2;
      positionsX[slot] = positionsX[module] + Math.cos(angle) * 20;
      positionsY[slot] = positionsY[module] + Math.sin(angle) * 20;
    }
  }

  const sources: number[] = [];
  const targets: number[] = [];
  if (withContainmentEdges) {
    sources.push(0, 0);
    targets.push(1, 2);
    for (let k = 0; k < LEAVES; k++) {
      sources.push(1, 2);
      targets.push(leafSlot(1, k), leafSlot(2, k));
    }
  }
  // The cross-cutting imports: leaf-to-leaf, both endpoints hidden by the
  // collapse, which is exactly the set today's spring pass drops.
  for (let k = 0; k < LEAVES; k++) {
    sources.push(leafSlot(1, k));
    targets.push(leafSlot(2, k));
  }

  const collapsedFlags = new Uint32Array(nodeCount);
  const collapsedVisible = new Uint8Array(nodeCount);
  const collapsedMass = new Float32Array(nodeCount);
  collapsedVisible[0] = 1;
  collapsedVisible[1] = 1;
  collapsedVisible[2] = 1;
  collapsedMass[0] = 1;
  collapsedMass[1] = 1 + LEAVES;
  collapsedMass[2] = 1 + LEAVES;
  for (let slot = 3; slot < nodeCount; slot++) collapsedFlags[slot] = NODE_FLAG_HIDDEN_LOD;

  return {
    nodeCount,
    parent,
    positionsX,
    positionsY,
    edgeSources: Uint32Array.from(sources),
    edgeTargets: Uint32Array.from(targets),
    depths,
    frontier: Uint32Array.from([0, 1, 2]),
    collapsedFlags,
    collapsedVisible,
    collapsedMass,
  };
}

/** Slot of module `module`'s `k`th leaf. */
function leafSlot(module: number, k: number): number {
  return 3 + (module - 1) * LEAVES + k;
}

/** The aggregation the reference walk produces for a cut, decoded. */
function aggregate(
  edgeSources: Uint32Array,
  edgeTargets: Uint32Array,
  parent: Uint32Array,
  visible: Uint8Array,
) {
  return decodeEdgeAggregation(
    referenceAggregateLodEdges(edgeSources, edgeTargets, parent, visible),
  );
}

/** Every source edge, as a live-edge list — the stale/identity case. */
function identityEdgeList(edgeCount: number): Uint32Array {
  const list = new Uint32Array(edgeCount);
  for (let e = 0; e < edgeCount; e++) list[e] = e;
  return list;
}

// =============================================================================
// The physics bug: a collapsed pair that pulls on nothing
// =============================================================================

/**
 * Springs alone: no repulsion, no centring gravity, no collision. Any motion a
 * node shows is a spring force and nothing else, so "the bundle exerts a
 * non-zero force" is a displacement rather than an inference.
 */
const SPRING_ONLY = validateForceConfig({
  repulsionStrength: 0,
  centerStrength: 0,
  springStrength: 0.05,
  springLength: 30,
  velocityDecay: 0.4,
});

gpuTest(
  "bundles: a collapsed pair joined only by hidden imports feels a spring at all",
  async (device) => {
    const fixture = twoModules(false);
    const graph = {
      nodeCount: fixture.nodeCount,
      positionsX: fixture.positionsX,
      positionsY: fixture.positionsY,
      edgeSources: fixture.edgeSources,
      edgeTargets: fixture.edgeTargets,
      depths: fixture.depths,
      flags: fixture.collapsedFlags,
      mass: fixture.collapsedMass,
    };

    // Today's behaviour: every import has two hidden endpoints, the spring
    // pass masks all of them, and the two proxies are joined by nothing.
    const dropped = await createSimHarness(device, graph, SPRING_ONLY);
    let before: PositionSnapshot;
    let after: PositionSnapshot;
    try {
      before = await dropped.readPositions();
      await dropped.tick(8);
      after = await dropped.readPositions();
    } finally {
      dropped.dispose();
    }
    assertFrozen(before, after, fixture.frontier);

    // With the aggregation, the same imports pull on the proxies instead.
    const bundled = await createSimHarness(device, graph, SPRING_ONLY);
    try {
      const aggregation = aggregate(
        fixture.edgeSources,
        fixture.edgeTargets,
        fixture.parent,
        fixture.collapsedVisible,
      );
      assertEquals(aggregation.liveEdges.length, 0, "no import has a visible endpoint");
      assertEquals(
        Array.from(aggregation.bundles),
        [1, 2, LEAVES],
        "every import folds into one weighted bundle between the two proxies",
      );

      bundled.setEdgeBundles(aggregation.liveEdges, aggregation.bundles);
      assertEquals(bundled.activeEdgeCount, 0);
      assertEquals(bundled.bundleCount, 1);

      const start = await bundled.readPositions();
      await bundled.tick(8);
      const end = await bundled.readPositions();

      assertEquals(countNonFinite(end.x, end.y), 0);
      const opened = Math.abs(start.x[2] - start.x[1]);
      const closed = Math.abs(end.x[2] - end.x[1]);
      assert(
        closed < opened - 1,
        `the bundle must pull the proxies together: ${opened} -> ${closed}`,
      );
      assertFrozen(start, end, [leafSlot(1, 0), leafSlot(2, 0)]);
    } finally {
      bundled.dispose();
    }
  },
);

gpuTest("bundles: a bundle pulls with the strength of the edges it replaces", async (device) => {
  // The design decision this pins: spring strength scales linearly with the
  // bundle count and is not capped. A cap would leave the proxies' repulsion —
  // which does carry the full mass of the subtrees — unopposed, and move the
  // equilibrium the expanded layout had.
  const fixture = twoModules(false);
  const graph = {
    nodeCount: fixture.nodeCount,
    positionsX: fixture.positionsX,
    positionsY: fixture.positionsY,
    edgeSources: fixture.edgeSources,
    edgeTargets: fixture.edgeTargets,
    depths: fixture.depths,
    flags: fixture.collapsedFlags,
    mass: fixture.collapsedMass,
  };

  const closedBy = async (weight: number): Promise<number> => {
    const harness = await createSimHarness(device, graph, SPRING_ONLY);
    try {
      harness.setEdgeBundles(new Uint32Array(0), Uint32Array.from([1, 2, weight]));
      const start = await harness.readPositions();
      await harness.tick(8);
      const end = await harness.readPositions();
      return Math.abs(start.x[2] - start.x[1]) - Math.abs(end.x[2] - end.x[1]);
    } finally {
      harness.dispose();
    }
  };

  const single = await closedBy(1);
  const bundled = await closedBy(LEAVES);
  assert(single > 0, `a weight-1 bundle must still pull, closed ${single}`);
  assert(
    bundled > single * 4,
    `a weight-${LEAVES} bundle must pull far harder than one edge: ${bundled} vs ${single}`,
  );
});

// =============================================================================
// SC-002: the collapsed layout approximates the expanded one
// =============================================================================

const SC002_CONFIG = validateForceConfig({ springStrength: 0.08, springLength: 40 });
const SC002_TICKS = 60;

gpuTest(
  "bundles: the collapsed frontier lands where the expanded run put it",
  async (device) => {
    const fixture = twoModules(true);
    const base = {
      nodeCount: fixture.nodeCount,
      positionsX: fixture.positionsX,
      positionsY: fixture.positionsY,
      edgeSources: fixture.edgeSources,
      edgeTargets: fixture.edgeTargets,
      depths: fixture.depths,
    };

    const run = async (
      collapsed: boolean,
      bundled: boolean,
    ): Promise<PositionSnapshot> => {
      const harness = await createSimHarness(
        device,
        collapsed ? { ...base, flags: fixture.collapsedFlags, mass: fixture.collapsedMass } : base,
        SC002_CONFIG,
      );
      try {
        if (bundled) {
          const aggregation = aggregate(
            fixture.edgeSources,
            fixture.edgeTargets,
            fixture.parent,
            fixture.collapsedVisible,
          );
          harness.setEdgeBundles(aggregation.liveEdges, aggregation.bundles);
        }
        await harness.tick(SC002_TICKS);
        return await harness.readPositions();
      } finally {
        harness.dispose();
      }
    };

    const expanded = await run(false, false);
    const dropped = await run(true, false);
    const aggregated = await run(true, true);

    assertEquals(countNonFinite(aggregated.x, aggregated.y), 0);

    // The reference for a proxy is its subtree's centre of mass, not the
    // parent node's own position. That is what the mass convention makes it:
    // repulsion is scaled by the source's aggregate mass and each arriving
    // spring is divided by the receiver's, so a proxy accelerates the way the
    // centre of mass of the bodies it replaces does. Node 1's own expanded
    // position is not that point — the cluster deforms around it — and
    // measuring against it would be measuring the rendering choice (§5.4 draws
    // the proxy at its own position) rather than the physics.
    const subtreeCentre = (snapshot: PositionSnapshot, module: number): [number, number] => {
      let x = snapshot.x[module];
      let y = snapshot.y[module];
      for (let k = 0; k < LEAVES; k++) {
        x += snapshot.x[leafSlot(module, k)];
        y += snapshot.y[leafSlot(module, k)];
      }
      return [x / (LEAVES + 1), y / (LEAVES + 1)];
    };

    const divergence = (actual: PositionSnapshot): number => {
      let worst = Math.hypot(actual.x[0] - expanded.x[0], actual.y[0] - expanded.y[0]);
      for (const module of [1, 2]) {
        const [cx, cy] = subtreeCentre(expanded, module);
        const d = Math.hypot(actual.x[module] - cx, actual.y[module] - cy);
        if (!(d <= worst)) worst = d;
      }
      return worst;
    };

    const droppedDivergence = divergence(dropped);
    const aggregatedDivergence = divergence(aggregated);

    // The bug has to be visible for the fix to mean anything: with the imports
    // dropped the two modules drift apart by half their separation again.
    assert(
      droppedDivergence > 15,
      `dropping the imports must visibly change the layout, moved ${droppedDivergence}`,
    );
    // Comparative rather than absolute, because a proxy is a rigid point and
    // the subtree it replaces is not: cluster A elongates toward B under the
    // imports, which no single body can reproduce. What must hold is that
    // bundling recovers most of the difference.
    assert(
      aggregatedDivergence < droppedDivergence * 0.5,
      `bundling must recover the expanded layout: ${aggregatedDivergence} vs ` +
        `${droppedDivergence} without it`,
    );
  },
);

// =============================================================================
// SC-005: with no cut, nothing changes
// =============================================================================

/**
 * A perfect matching: node `2k` joined to node `2k + 1`, and nothing else.
 *
 * Degree one, and that is the point. Spring accumulation is a compare-exchange
 * loop into an accumulator the repulsion pass has already written, so a node
 * receiving two springs sums three terms in an order the GPU does not
 * guarantee — and float addition is not associative. One spring per node makes
 * the sum a fixed two terms, which is what lets the SC-005 comparison be exact
 * rather than tolerant.
 */
function matching(nodeCount: number): {
  nodeCount: number;
  positionsX: Float32Array;
  positionsY: Float32Array;
  edgeSources: Uint32Array;
  edgeTargets: Uint32Array;
  depths: Float32Array;
} {
  const graph = ring(nodeCount);
  const pairs = nodeCount >> 1;
  const edgeSources = new Uint32Array(pairs);
  const edgeTargets = new Uint32Array(pairs);
  for (let k = 0; k < pairs; k++) {
    edgeSources[k] = k * 2;
    edgeTargets[k] = k * 2 + 1;
  }
  return { ...graph, edgeSources, edgeTargets };
}

/** A ring: every node has degree exactly two, and one spring per edge. */
function ring(nodeCount: number): {
  nodeCount: number;
  positionsX: Float32Array;
  positionsY: Float32Array;
  edgeSources: Uint32Array;
  edgeTargets: Uint32Array;
  depths: Float32Array;
} {
  const positionsX = new Float32Array(nodeCount);
  const positionsY = new Float32Array(nodeCount);
  const edgeSources = new Uint32Array(nodeCount);
  const edgeTargets = new Uint32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    const angle = (i / nodeCount) * Math.PI * 2;
    positionsX[i] = Math.cos(angle) * 90;
    positionsY[i] = Math.sin(angle) * 90;
    edgeSources[i] = i;
    edgeTargets[i] = (i + 1) % nodeCount;
  }
  return {
    nodeCount,
    positionsX,
    positionsY,
    edgeSources,
    edgeTargets,
    depths: new Float32Array(nodeCount),
  };
}

const SC005_CONFIG = validateForceConfig({});

/**
 * Ticks the code-tree comparisons run for.
 *
 * Short on purpose. The compare-exchange accumulation makes a hub node's sum
 * order-dependent, and a force layout amplifies any difference exponentially:
 * at 20 ticks this pipeline diverges from a second copy of itself by hundreds
 * of units on this fixture. Five keeps the floor small enough that a real
 * regression is still visible above it.
 */
const SC005_TICKS = 5;

/** How far above the pipeline's own reproducibility floor a run may land. */
const FLOOR_MARGIN = 4;

/** Floor for a fixture that happens to reproduce exactly, so the bar is never 0. */
const ABSOLUTE_FLOOR = 1e-3;

/** Largest distance between two snapshots over `[0, count)`. */
function worstDivergence(
  actual: PositionSnapshot,
  reference: PositionSnapshot,
  count: number,
): number {
  let worst = 0;
  for (let i = 0; i < count; i++) {
    const d = Math.hypot(actual.x[i] - reference.x[i], actual.y[i] - reference.y[i]);
    if (!(d <= worst)) worst = d;
  }
  return worst;
}

const SPRINGS_URL = new URL(
  "../../packages/core/src/simulation/shaders/springs_simple.comp.wgsl",
  import.meta.url,
);

/**
 * The weight and inverse-mass terms this work package added to the shared
 * spring body, and what removing them leaves: the shader as it stood before
 * edge aggregation existed. Applied as text, so the SC-005 baseline is always
 * the code under test minus this change and can never drift into a stale copy.
 */
const BUNDLE_STRIPS: ReadonlyArray<readonly [string, string]> = [
  ["    weight: f32,\n    source_inv_mass: f32,\n    target_inv_mass: f32,\n", ""],
  [
    "let force_magnitude = uniforms.spring_strength * displacement * weight;",
    "let force_magnitude = uniforms.spring_strength * displacement;",
  ],
  [
    "accumulate_force(source_idx, force * source_inv_mass);\n" +
    "    accumulate_force(target_idx, -force * target_inv_mass);",
    "accumulate_force(source_idx, force);\n    accumulate_force(target_idx, -force);",
  ],
  [
    "apply_spring(edge_sources[edge_idx], edge_targets[edge_idx], 1.0, 1.0, 1.0);",
    "apply_spring(edge_sources[edge_idx], edge_targets[edge_idx]);",
  ],
];

/** Text between these markers is the LOD half of the shader, removed wholesale. */
const LOD_SECTION_START = "// --- LOD edge aggregation";
const LOD_SECTION_END = "const NODE_FLAG_DEAD";
const BUNDLED_ENTRY_START = "// Spring pass under a live LOD cut.";

/** The shipped spring shader with everything this package added removed. */
function stripBundling(shipped: string): string {
  const bundledAt = shipped.indexOf(BUNDLED_ENTRY_START);
  assert(bundledAt > 0, "SC-005 baseline is stale: the bundled entry point is gone");
  let source = shipped.slice(0, bundledAt);

  const lodStart = source.indexOf(LOD_SECTION_START);
  const lodEnd = source.indexOf(LOD_SECTION_END);
  assert(lodStart > 0 && lodEnd > lodStart, "SC-005 baseline is stale: the LOD bindings moved");
  source = source.slice(0, lodStart) + source.slice(lodEnd);

  for (const [from, to] of BUNDLE_STRIPS) {
    assert(
      source.includes(from),
      `SC-005 baseline is stale: ${JSON.stringify(from.slice(0, 48))} is no longer in the shader`,
    );
    source = source.replace(from, to);
  }

  // Code references, not mentions: the uniform struct's comment still names
  // the fields the bundled entry point reads.
  for (const survivor of ["* weight", "inv_mass", "node_mass[", "live_edge_idx[", "bundles["]) {
    assert(!source.includes(survivor), `"${survivor}" must not survive in the reference`);
  }
  return source;
}

/**
 * Runs `ticks` steps with the spring pass taken from an arbitrary WGSL source.
 *
 * The *shipped* record function drives every stage — only the spring pipeline
 * and its bind group are swapped — so an edited shader is a baseline rather
 * than a second implementation of the tick. `main`'s auto layout reaches
 * bindings 0-5 in both sources, so both bind groups are the same shape.
 */
async function runWithSpringsShader(
  device: GPUDevice,
  graph: {
    nodeCount: number;
    positionsX: Float32Array;
    positionsY: Float32Array;
    edgeSources: Uint32Array;
    edgeTargets: Uint32Array;
  },
  springsCode: string,
  ticks: number,
): Promise<PositionSnapshot> {
  const mod = await loadPipelineModule();
  const { nodeCount } = graph;
  const edgeCount = graph.edgeSources.length;

  const pipeline = mod.createSimulationPipeline({ device }, {
    maxNodes: nodeCount,
    maxEdges: edgeCount,
  });
  const buffers = mod.createSimulationBuffers(device, nodeCount, edgeCount);
  mod.copyPositionsToSimulation(device, buffers, graph.positionsX, graph.positionsY);
  mod.copyEdgesToSimulation(device, buffers, graph.edgeSources, graph.edgeTargets);

  const springs = device.createComputePipeline({
    label: "Reference Springs",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: springsCode }),
      entryPoint: "main",
    },
  });
  const springsBindGroup = device.createBindGroup({
    layout: springs.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.springUniforms } },
      { binding: 1, resource: { buffer: buffers.positions } },
      { binding: 2, resource: { buffer: buffers.forces } },
      { binding: 3, resource: { buffer: buffers.edgeSources } },
      { binding: 4, resource: { buffer: buffers.edgeTargets } },
      { binding: 5, resource: { buffer: buffers.nodeFlags } },
    ],
  });
  const patched = { ...pipeline, pipelines: { ...pipeline.pipelines, springs } };

  for (let t = 0; t < ticks; t++) {
    mod.updateSimulationUniforms(
      device,
      buffers,
      nodeCount,
      edgeCount,
      Math.pow(1 - HARNESS_ALPHA_DECAY, t + 1),
      SC005_CONFIG,
    );
    const bindGroups = mod.createSimulationBindGroups(device, pipeline, buffers);
    const encoder = device.createCommandEncoder();
    mod.recordSimulationStepWithOptions(
      encoder,
      patched,
      { ...bindGroups, springs: springsBindGroup },
      nodeCount,
      edgeCount,
    );
    device.queue.submit([encoder.finish()]);
    mod.swapSimulationBuffers(buffers);
  }

  const encoder = device.createCommandEncoder();
  mod.copyPositionsToReadback(encoder, buffers);
  device.queue.submit([encoder.finish()]);
  const x = new Float32Array(nodeCount);
  const y = new Float32Array(nodeCount);
  await mod.readbackPositions(buffers, x, y);
  return { x, y };
}

/** Compare two float arrays as raw bit patterns — no tolerance, no NaN escape. */
function assertBitIdentical(actual: Float32Array, expected: Float32Array, label: string): void {
  assertEquals(actual.length, expected.length, `${label}: length`);
  const a = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const b = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  for (let i = 0; i < a.length; i++) {
    if (a[i] === b[i]) continue;
    throw new Error(
      `${label}: element ${i} differs — ${actual[i]} (0x${a[i].toString(16)}) vs ` +
        `${expected[i]} (0x${b[i].toString(16)})`,
    );
  }
}

gpuTest("SC-005: with no cut, the spring pass is bit-identical to the pre-change shader", async (
  device,
) => {
  const shipped = await Deno.readTextFile(SPRINGS_URL);
  const graph = matching(64);

  const actual = await runWithSpringsShader(device, graph, shipped, 40);
  const reference = await runWithSpringsShader(device, graph, stripBundling(shipped), 40);

  assertEquals(countNonFinite(actual.x, actual.y), 0);
  assertBitIdentical(actual.x, reference.x, "positions.x after 40 ticks");
  assertBitIdentical(actual.y, reference.y, "positions.y after 40 ticks");
});

gpuTest("SC-005: an identity aggregation matches the un-aggregated run", async (device) => {
  // The bundled entry point cannot be bit-exact against the un-bundled one and
  // does not claim to be: the two dispatch different thread counts, so the
  // compare-exchange accumulation reaches each node's force in a different
  // order — and that accumulator already holds the repulsion pass's result, so
  // a three-term sum whose order changed is a different f32. The exactness
  // claim belongs to the un-cut path, which the shader-baseline test above
  // carries. What must hold here is that an identity aggregation — every edge
  // live, no bundles — is the same layout to well inside a node radius.
  const graph = ring(64);

  const plain = await createSimHarness(device, graph, SC005_CONFIG);
  let reference: PositionSnapshot;
  try {
    assertEquals(plain.activeEdgeCount, 0, "no aggregation is declared by default");
    assertEquals(plain.bundleCount, 0);
    await plain.tick(SC005_TICKS);
    reference = await plain.readPositions();
  } finally {
    plain.dispose();
  }

  const identity = await createSimHarness(device, graph, SC005_CONFIG);
  try {
    identity.setEdgeBundles(identityEdgeList(graph.edgeSources.length), new Uint32Array(0));
    assertEquals(identity.activeEdgeCount, graph.edgeSources.length);
    await identity.tick(SC005_TICKS);
    const actual = await identity.readPositions();

    assertEquals(countNonFinite(actual.x, actual.y), 0);
    const observed = worstDivergence(actual, reference, graph.nodeCount);
    assert(observed < ABSOLUTE_FLOOR, `the identity aggregation diverged by ${observed}`);
  } finally {
    identity.dispose();
  }
});

gpuTest("activeEdges: an aggregation covering nothing springs nothing", async (device) => {
  // Zero live edges and zero bundles is a legal aggregation — a cut can leave
  // every edge inside a collapsed subtree — and it must dispatch no spring
  // pass at all rather than fall back to springing the whole edge array. The
  // fixture states it the only way it is observable: every node is visible, so
  // the fallback would be plainly visible in the layout.
  const graph = ring(64);

  const empty = await createSimHarness(device, graph, SC005_CONFIG);
  let actual: PositionSnapshot;
  try {
    empty.setEdgeBundles(new Uint32Array(0), new Uint32Array(0));
    assertEquals(empty.activeEdgeCount, 0);
    assertEquals(empty.bundleCount, 0);
    await empty.tick(SC005_TICKS);
    actual = await empty.readPositions();
  } finally {
    empty.dispose();
  }

  const springless = await createSimHarness(
    device,
    graph,
    validateForceConfig({ springStrength: 0 }),
  );
  let reference: PositionSnapshot;
  try {
    await springless.tick(SC005_TICKS);
    reference = await springless.readPositions();
  } finally {
    springless.dispose();
  }

  const observed = worstDivergence(actual, reference, graph.nodeCount);
  assert(observed < ABSOLUTE_FLOOR, `an empty aggregation still sprang something: ${observed}`);
});

// =============================================================================
// activeEdges: dispatch tracks the cut, and the list is only a hint
// =============================================================================

/**
 * Collapses all but every `keepEveryNth` subtree rooted at `cutDepth` — the
 * same shape the active-node dispatch tests use, so the two halves of the
 * feature are measured against the same kind of cut.
 */
function collapseBranches(tree: CodeTreeGraph, cutDepth: number, keepEveryNth: number): {
  flags: Uint32Array;
  visible: Uint8Array;
  parent: Uint32Array;
} {
  const collapseRoot = new Uint8Array(tree.nodeCount);
  let seen = 0;
  for (let slot = 0; slot < tree.nodeCount; slot++) {
    if (tree.depths[slot] !== cutDepth) continue;
    if (seen % keepEveryNth !== 0) collapseRoot[slot] = 1;
    seen++;
  }

  const flags = new Uint32Array(tree.nodeCount);
  const visible = new Uint8Array(tree.nodeCount).fill(1);
  const parent = new Uint32Array(tree.nodeCount);
  for (let slot = 0; slot < tree.nodeCount; slot++) {
    const up = tree.parent[slot];
    parent[slot] = up < 0 ? HIERARCHY_ROOT : up;
    const hidden = up >= 0 && (collapseRoot[up] === 1 || flags[up] === NODE_FLAG_HIDDEN_LOD);
    if (!hidden) continue;
    flags[slot] = NODE_FLAG_HIDDEN_LOD;
    visible[slot] = 0;
  }
  return { flags, visible, parent };
}

gpuTest("activeEdges: with 90% hidden the spring pass covers the cut, not the graph", async (
  device,
) => {
  const tree = generateCodeTree({ ...CODE_TREE_SCALES.small, seed: 31 });
  const cut = collapseBranches(tree, 3, 14);
  const hidden = cut.visible.reduce((n, v) => n + (v === 0 ? 1 : 0), 0);
  assert(
    hidden / tree.nodeCount > 0.88,
    `the fixture must hide ~90% of the tree, hid ${(hidden / tree.nodeCount * 100).toFixed(1)}%`,
  );

  const aggregation = aggregate(tree.edgeSources, tree.edgeTargets, cut.parent, cut.visible);

  // Every entry of the live list is a source edge with both endpoints on
  // screen — this is the set the un-aggregated pass would have masked away
  // one thread at a time.
  for (const edge of aggregation.liveEdges) {
    assertEquals(cut.visible[tree.edgeSources[edge]], 1, `live edge ${edge} names a hidden source`);
    assertEquals(cut.visible[tree.edgeTargets[edge]], 1, `live edge ${edge} names a hidden target`);
  }
  // And every bundle joins two visible slots.
  for (let k = 0; k < aggregation.bundleCount; k++) {
    assertEquals(cut.visible[aggregation.bundles[k * 3]], 1);
    assertEquals(cut.visible[aggregation.bundles[k * 3 + 1]], 1);
  }

  const harness = await createSimHarness(device, {
    nodeCount: tree.nodeCount,
    positionsX: tree.positionsX,
    positionsY: tree.positionsY,
    edgeSources: tree.edgeSources,
    edgeTargets: tree.edgeTargets,
    depths: tree.depths,
    flags: cut.flags,
  }, SC005_CONFIG);
  try {
    const bundleBuffer = harness.edgeBundlesBuffer;
    const liveEdgeBuffer = harness.liveEdgeIndicesBuffer;

    harness.setEdgeBundles(aggregation.liveEdges, aggregation.bundles);
    const dispatched = harness.activeEdgeCount + harness.bundleCount;
    assertEquals(harness.activeEdgeCount, aggregation.liveEdges.length);
    assertEquals(harness.bundleCount, aggregation.bundleCount);
    assert(
      dispatched < harness.edgeCount * 0.5,
      `the cut must cost a fraction of the edge list: ${dispatched} of ${harness.edgeCount}`,
    );

    // Expanding again is a contents write: the counts move, the buffers do not.
    harness.clearEdgeBundles();
    assertEquals(harness.activeEdgeCount, 0);
    assertEquals(harness.bundleCount, 0);
    assertStrictEquals(harness.edgeBundlesBuffer, bundleBuffer);
    assertStrictEquals(harness.liveEdgeIndicesBuffer, liveEdgeBuffer);
  } finally {
    harness.dispose();
  }
});

gpuTest("activeEdges: a stale live-edge list is slower, never wrong", async (device) => {
  const tree = generateCodeTree({ ...CODE_TREE_SCALES.small, seed: 31 });
  const cut = collapseBranches(tree, 3, 14);
  const aggregation = aggregate(tree.edgeSources, tree.edgeTargets, cut.parent, cut.visible);
  const graph = {
    nodeCount: tree.nodeCount,
    positionsX: tree.positionsX,
    positionsY: tree.positionsY,
    edgeSources: tree.edgeSources,
    edgeTargets: tree.edgeTargets,
    depths: tree.depths,
    flags: cut.flags,
  };

  const run = async (liveEdges: Uint32Array): Promise<PositionSnapshot> => {
    const harness = await createSimHarness(device, graph, SC005_CONFIG);
    try {
      harness.setEdgeBundles(liveEdges, aggregation.bundles);
      await harness.tick(SC005_TICKS);
      return await harness.readPositions();
    } finally {
      harness.dispose();
    }
  };

  const exact = await run(aggregation.liveEdges);
  const floor = worstDivergence(await run(aggregation.liveEdges), exact, tree.nodeCount);

  // The list a caller would hold before it caught up with the transition: the
  // whole edge array, most of whose entries now name hidden nodes. The shader
  // masks them under the gather, so the extra threads cost time and change
  // nothing.
  const stale = identityEdgeList(tree.edgeSources.length);
  assert(stale.length > aggregation.liveEdges.length * 5, "the stale list must be far larger");
  const masked = await run(stale);

  assertEquals(countNonFinite(masked.x, masked.y), 0);
  const observed = worstDivergence(masked, exact, tree.nodeCount);
  assert(
    observed <= Math.max(floor * FLOOR_MARGIN, ABSOLUTE_FLOOR),
    `the stale list changed the layout by ${observed}, beyond the pipeline's own ${floor}`,
  );
});
