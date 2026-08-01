/**
 * GPU tests for the active-index simulation dispatch (WP-D).
 *
 * `nodeFlags` bit 2 says a node is out of the picture; `liveIndices` is how
 * the simulation stops paying for it. The force passes dispatch one thread per
 * entry of that list and gather their slot from it, so hiding 90% of a graph
 * turns into 90% of the work not being submitted, rather than 90% of the
 * threads returning early.
 *
 * Four claims, in the order they matter:
 *
 * - **The visible layout is the layout of the visible graph.** After K ticks
 *   with 90% of a code tree hidden, the nodes that stayed on screen sit where
 *   a graph built from only those nodes puts them. This is the whole point:
 *   the mechanism may skip work, but it may not change physics.
 * - **Hidden nodes are inert and frozen.** They exert no force — moving one
 *   somewhere else entirely does not perturb the visible set by a single ULP —
 *   and they do not move, so expanding restores the arrangement they were
 *   collapsed with.
 * - **SC-005: with LOD off the output is bit-identical to slot-order
 *   dispatch.** The reference is the shipped shader with the gather textually
 *   removed, so the baseline is always "the code under test, minus the list".
 * - **Cost tracks the list, and the buffer never moves.** activeCount follows
 *   the flags through collapse and expand while `liveIndices` keeps its
 *   identity, which is what makes a transition cost zero bind-group work.
 */

import { assert, assertEquals, assertStrictEquals } from "jsr:@std/assert@^1";
import {
  createSimHarness,
  GPU_SKIP_MESSAGE,
  HARNESS_ALPHA_DECAY,
  loadActiveSetModule,
  loadPipelineModule,
  NODE_FLAG_HIDDEN_LOD,
  probeAdapter,
  requestHarnessDevice,
  type SimHarness,
} from "../helpers/gpu.ts";
import {
  assertFrozen,
  assertVisibleSetMatchesReference,
  countNonFinite,
  type PositionSnapshot,
} from "../helpers/invariants.ts";
import { CODE_TREE_SCALES, type CodeTreeGraph, generateCodeTree } from "../fixtures/code_tree.ts";
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
// Fixture: a code tree cut at a depth, and the compacted graph of what survives
// =============================================================================

/** A code tree plus the LOD cut applied to it. */
interface CutTree {
  readonly tree: CodeTreeGraph;
  /** Slots left on screen, ascending. */
  readonly visible: Uint32Array;
  /** Slots hidden by the cut, ascending. */
  readonly hidden: Uint32Array;
  /** Per-slot flags with NODE_FLAG_HIDDEN_LOD set on every hidden slot. */
  readonly flags: Uint32Array;
}

/**
 * Collapses all but every `keepEveryNth` subtree rooted at `cutDepth`.
 *
 * The cut a real zoom-out produces: some branches expanded, most standing as a
 * single proxy, the collapse roots themselves still visible. Two properties
 * matter and a simpler mask has neither. Whole subtrees go together, so the
 * surviving edge set is a real graph rather than a shredded one — and the
 * visible slots are SCATTERED, in a dozen runs rather than one prefix, which
 * is what collapsing interior subtrees actually leaves behind. A contiguous
 * visible set would let `live_idx[k] == k` hold by accident and the gather
 * would test as correct while doing nothing.
 */
function collapseBranches(tree: CodeTreeGraph, cutDepth: number, keepEveryNth: number): CutTree {
  const collapseRoot = new Uint8Array(tree.nodeCount);
  let seen = 0;
  for (let slot = 0; slot < tree.nodeCount; slot++) {
    if (tree.depths[slot] !== cutDepth) continue;
    if (seen % keepEveryNth !== 0) collapseRoot[slot] = 1;
    seen++;
  }

  // One forward pass: the generator emits children after their parent, so a
  // node's parent is always already classified.
  const flags = new Uint32Array(tree.nodeCount);
  const visible: number[] = [];
  const hidden: number[] = [];
  for (let slot = 0; slot < tree.nodeCount; slot++) {
    const parent = tree.parent[slot];
    assert(parent < slot, `fixture is not parent-before-child at slot ${slot}`);
    const isHidden = parent >= 0 &&
      (collapseRoot[parent] === 1 || flags[parent] === NODE_FLAG_HIDDEN_LOD);
    if (isHidden) {
      flags[slot] = NODE_FLAG_HIDDEN_LOD;
      hidden.push(slot);
    } else {
      visible.push(slot);
    }
  }
  return {
    tree,
    visible: Uint32Array.from(visible),
    hidden: Uint32Array.from(hidden),
    flags,
  };
}

/** Number of maximal runs of consecutive slots in an ascending index list. */
function contiguousRuns(indices: Uint32Array): number {
  if (indices.length === 0) return 0;
  let runs = 1;
  for (let k = 1; k < indices.length; k++) {
    if (indices[k] !== indices[k - 1] + 1) runs++;
  }
  return runs;
}

/** Graph data for the visible nodes alone, renumbered into a dense slot space. */
interface CompactedGraph {
  nodeCount: number;
  positionsX: Float32Array;
  positionsY: Float32Array;
  edgeSources: Uint32Array;
  edgeTargets: Uint32Array;
  depths: Float32Array;
  /** Dense slot of each entry of the cut's `visible`, in the same order. */
  referenceIndices: Uint32Array;
}

/**
 * Builds the reference graph: only the visible nodes, only the edges with both
 * endpoints visible, renumbered 0..V-1 but in the same ascending slot order.
 *
 * Order matters as much as membership. The all-pairs shaders sum in list
 * order, so a reference that permuted the nodes would differ by float
 * summation alone and the comparison would be measuring rounding.
 */
function compactVisible(cut: CutTree): CompactedGraph {
  const { tree, visible } = cut;
  const dense = new Int32Array(tree.nodeCount).fill(-1);
  for (let k = 0; k < visible.length; k++) dense[visible[k]] = k;

  const positionsX = new Float32Array(visible.length);
  const positionsY = new Float32Array(visible.length);
  const depths = new Float32Array(visible.length);
  for (let k = 0; k < visible.length; k++) {
    positionsX[k] = tree.positionsX[visible[k]];
    positionsY[k] = tree.positionsY[visible[k]];
    depths[k] = tree.depths[visible[k]];
  }

  const sources: number[] = [];
  const targets: number[] = [];
  for (let e = 0; e < tree.edgeSources.length; e++) {
    const src = dense[tree.edgeSources[e]];
    const tgt = dense[tree.edgeTargets[e]];
    if (src < 0 || tgt < 0) continue;
    sources.push(src);
    targets.push(tgt);
  }

  return {
    nodeCount: visible.length,
    positionsX,
    positionsY,
    edgeSources: Uint32Array.from(sources),
    edgeTargets: Uint32Array.from(targets),
    depths,
    referenceIndices: Uint32Array.from(visible, (slot) => dense[slot]),
  };
}

/** Largest distance between two snapshots over `indices` / `referenceIndices`. */
function maxDivergence(
  actual: PositionSnapshot,
  reference: PositionSnapshot,
  indices: ArrayLike<number>,
  referenceIndices: ArrayLike<number>,
): number {
  let worst = 0;
  for (let k = 0; k < indices.length; k++) {
    const d = Math.hypot(
      actual.x[indices[k]] - reference.x[referenceIndices[k]],
      actual.y[indices[k]] - reference.y[referenceIndices[k]],
    );
    if (!(d <= worst)) worst = d;
  }
  return worst;
}

const CUT_CONFIG = validateForceConfig({});
const CUT_TICKS = 40;

async function runTicks(harness: SimHarness, ticks: number): Promise<PositionSnapshot> {
  await harness.tick(ticks);
  return await harness.readPositions();
}

// =============================================================================
// Primary invariant — hiding does not change the visible layout
// =============================================================================

/**
 * The cut fixture the layout invariants run on: a 2 500-node code tree with
 * all but every fourteenth depth-3 subtree collapsed — 90.9% of the slots
 * hidden, leaving 228 visible in 11 separate runs of the slot space.
 */
function primaryCut(): CutTree {
  const cut = collapseBranches(generateCodeTree({ ...CODE_TREE_SCALES.small, seed: 31 }), 3, 14);
  const hiddenFraction = cut.hidden.length / cut.tree.nodeCount;
  assert(
    hiddenFraction > 0.88 && hiddenFraction < 0.94,
    `fixture must hide ~90% of the tree, hid ${(hiddenFraction * 100).toFixed(1)}%`,
  );
  assert(
    contiguousRuns(cut.visible) > 5,
    "the visible set must be scattered, or the gather is untested",
  );
  return cut;
}

/** Runs one cut fixture through the shipped pipeline for `ticks` ticks. */
async function runCutFixture(
  device: GPUDevice,
  graph: {
    nodeCount: number;
    positionsX: Float32Array;
    positionsY: Float32Array;
    edgeSources: Uint32Array;
    edgeTargets: Uint32Array;
    depths: Float32Array;
    flags?: Uint32Array;
  },
  ticks: number,
  expectActive?: number,
): Promise<PositionSnapshot> {
  const harness = await createSimHarness(device, graph, CUT_CONFIG);
  try {
    if (expectActive !== undefined) {
      assertEquals(
        harness.activeCount,
        expectActive,
        "the dispatch list must hold exactly the visible slots",
      );
    }
    return await runTicks(harness, ticks);
  } finally {
    harness.dispose();
  }
}

/** A self-loop: dropped by the spring pass at zero distance, so the run is edge-free. */
const SELF_LOOP = { edgeSources: new Uint32Array([0]), edgeTargets: new Uint32Array([0]) };

gpuTest(
  "active dispatch: with 90% hidden, the visible layout IS the visible graph's layout",
  async (device) => {
    // Edge-free, and therefore exact. The repulsion sum runs over the active
    // list in ascending slot order, and the compacted graph runs over the same
    // nodes in the same order — so if the list mechanism is faithful the two
    // runs agree bit for bit, not merely within a tolerance. That is a far
    // stronger claim than the spring pass can support: its compare-exchange
    // accumulation is not order-reproducible, and over 40 ticks of a chaotic
    // layout that noise grows to tens of units (see the spring-path test
    // below, which is why it runs briefly and compares to a measured floor).
    const cut = primaryCut();
    const reference = compactVisible(cut);

    const collapsed = await runCutFixture(
      device,
      {
        nodeCount: cut.tree.nodeCount,
        positionsX: cut.tree.positionsX,
        positionsY: cut.tree.positionsY,
        ...SELF_LOOP,
        depths: cut.tree.depths,
        flags: cut.flags,
      },
      CUT_TICKS,
      cut.visible.length,
    );

    const visibleOnly = await runCutFixture(device, {
      nodeCount: reference.nodeCount,
      positionsX: reference.positionsX,
      positionsY: reference.positionsY,
      ...SELF_LOOP,
      depths: reference.depths,
    }, CUT_TICKS);

    assertEquals(countNonFinite(collapsed.x, collapsed.y), 0);
    assertVisibleSetMatchesReference(
      collapsed,
      visibleOnly,
      cut.visible,
      0,
      reference.referenceIndices,
    );

    // Control: the same comparison with nothing hidden. The hidden 88.9% then
    // repel the frontier outward and the divergence is enormous — without this
    // the assertion above could be passing on a fixture whose hidden nodes
    // never mattered.
    const expanded = await runCutFixture(device, {
      nodeCount: cut.tree.nodeCount,
      positionsX: cut.tree.positionsX,
      positionsY: cut.tree.positionsY,
      ...SELF_LOOP,
      depths: cut.tree.depths,
    }, CUT_TICKS);
    const control = maxDivergence(expanded, visibleOnly, cut.visible, reference.referenceIndices);
    assert(control > 1, `control is too weak to make the invariant meaningful: ${control}`);
  },
);

gpuTest(
  "active dispatch: the spring pass ignores edges into the hidden set",
  async (device) => {
    // The same invariant carrying the tree's real edges, which is what the
    // hidden-endpoint guard in springs_simple.comp.wgsl exists for: a spring
    // from a visible parent to a collapsed child would drag the parent toward
    // something that is not on screen.
    //
    // Short, and against a measured floor rather than a fixed tolerance. The
    // spring pass accumulates with a compare-exchange loop whose float
    // addition order is not reproducible, so no two runs of ANY configuration
    // agree exactly; the question is only whether hiding moves the visible set
    // further than that irreducible noise does.
    const cut = primaryCut();
    const reference = compactVisible(cut);
    const ticks = 6;

    const withEdges = {
      nodeCount: cut.tree.nodeCount,
      positionsX: cut.tree.positionsX,
      positionsY: cut.tree.positionsY,
      edgeSources: cut.tree.edgeSources,
      edgeTargets: cut.tree.edgeTargets,
      depths: cut.tree.depths,
    };

    const collapsed = await runCutFixture(
      device,
      { ...withEdges, flags: cut.flags },
      ticks,
      cut.visible.length,
    );
    const repeat = await runCutFixture(device, { ...withEdges, flags: cut.flags }, ticks);
    const visibleOnly = await runCutFixture(device, {
      nodeCount: reference.nodeCount,
      positionsX: reference.positionsX,
      positionsY: reference.positionsY,
      edgeSources: reference.edgeSources,
      edgeTargets: reference.edgeTargets,
      depths: reference.depths,
    }, ticks);

    assertEquals(countNonFinite(collapsed.x, collapsed.y), 0);
    assertEquals(countNonFinite(visibleOnly.x, visibleOnly.y), 0);

    const noise = maxDivergence(collapsed, repeat, cut.visible, cut.visible);
    const divergence = maxDivergence(
      collapsed,
      visibleOnly,
      cut.visible,
      reference.referenceIndices,
    );
    const tolerance = Math.max(20 * noise, 1e-2);
    console.log(
      `[gpu] spring path: run-to-run noise ${noise.toExponential(2)}, vs visible-only graph ` +
        `${divergence.toExponential(2)} (tolerance ${tolerance.toExponential(2)})`,
    );
    assertVisibleSetMatchesReference(
      collapsed,
      visibleOnly,
      cut.visible,
      tolerance,
      reference.referenceIndices,
    );

    // Control: with nothing hidden the same comparison must fail by orders of
    // magnitude, or the tolerance above is doing no work.
    const expanded = await runCutFixture(device, withEdges, ticks);
    const control = maxDivergence(expanded, visibleOnly, cut.visible, reference.referenceIndices);
    assert(
      control > 100 * tolerance,
      `control is too weak to make the invariant meaningful: unhidden divergence ` +
        `${control} vs tolerance ${tolerance}`,
    );
  },
);

gpuTest(
  "active dispatch: hidden nodes never move and never reach the visible set",
  async (device) => {
    // Edge-free (one self-loop, which the spring pass drops at zero distance),
    // so the comparison below can be exact. The spring pass accumulates through
    // a compare-exchange loop whose float addition order is not reproducible;
    // its hidden-endpoint guard is covered by the visible-layout invariant
    // above, which compares against a graph that has no such edges at all.
    const cut = collapseBranches(
      generateCodeTree({ ...CODE_TREE_SCALES.small, seed: 47, maxNodes: 1_200 }),
      3,
      12,
    );
    assert(cut.hidden.length > 100, "fixture must hide a substantial subtree");

    const run = async (displaceHidden: boolean): Promise<PositionSnapshot> => {
      const positionsX = cut.tree.positionsX.slice();
      const positionsY = cut.tree.positionsY.slice();
      if (displaceHidden) {
        // Fling every hidden node a long way off. If any of them exerted force
        // — through repulsion or through the integrator reading their slot —
        // the visible set would notice.
        for (const slot of cut.hidden) {
          positionsX[slot] += 5_000;
          positionsY[slot] -= 3_000;
        }
      }
      const harness = await createSimHarness(device, {
        nodeCount: cut.tree.nodeCount,
        positionsX,
        positionsY,
        edgeSources: new Uint32Array([0]),
        edgeTargets: new Uint32Array([0]),
        depths: cut.tree.depths,
        flags: cut.flags,
      }, CUT_CONFIG);
      try {
        const after = await runTicks(harness, 30);
        // Frozen, exactly: every hidden node still holds the coordinates it
        // was uploaded with.
        assertFrozen({ x: positionsX, y: positionsY }, after, cut.hidden);
        return after;
      } finally {
        harness.dispose();
      }
    };

    const inPlace = await run(false);
    const flung = await run(true);

    // Zero force, not "small force": the visible trajectories are bit-identical
    // whether the hidden nodes sit among them or 5 000 units away.
    for (const slot of cut.visible) {
      assertEquals(flung.x[slot], inPlace.x[slot], `hidden node reached visible slot ${slot} (x)`);
      assertEquals(flung.y[slot], inPlace.y[slot], `hidden node reached visible slot ${slot} (y)`);
    }
    assert(
      inPlace.x[cut.visible[1]] !== cut.tree.positionsX[cut.visible[1]],
      "the visible set never moved, so the comparison above is vacuous",
    );
  },
);

// =============================================================================
// SC-005 — with LOD off, the gather is bit-identical to slot-order dispatch
// =============================================================================

const REPULSION_N2_URL = new URL(
  "../../packages/core/src/simulation/shaders/repulsion_n2.comp.wgsl",
  import.meta.url,
);

/**
 * The gather in the shipped shader, and what removing it leaves: the
 * slot-indexed loop the pipeline ran before the active list existed. Applied
 * as text so the SC-005 baseline is always the code under test minus the list,
 * and cannot drift into a stale copy.
 */
const GATHER_STRIPS: ReadonlyArray<readonly [string, string]> = [
  ["if (entry >= uniforms.active_count) {", "if (entry >= uniforms.node_count) {"],
  ["let node_idx = live_idx[entry];", "let node_idx = entry;"],
  [
    "for (var k = 0u; k < uniforms.active_count; k++) {\n        let i = live_idx[k];",
    "for (var i = 0u; i < uniforms.node_count; i++) {",
  ],
  ["@group(0) @binding(5) var<storage, read> live_idx: array<u32>;", ""],
];

/** The shipped repulsion shader with the active-index gather removed. */
function stripGather(shipped: string): string {
  let stripped = shipped;
  for (const [from, to] of GATHER_STRIPS) {
    assert(
      stripped.includes(from),
      `SC-005 baseline is stale: ${JSON.stringify(from.slice(0, 40))} is no longer in the shader`,
    );
    stripped = stripped.replace(from, to);
  }
  assert(!stripped.includes("live_idx["), "no list read may survive in the reference");
  assert(!stripped.includes("var<storage, read> live_idx"), "the binding must be gone too");
  return stripped;
}

/**
 * Runs `ticks` steps of the shipped pipeline with the repulsion pass taken
 * from an arbitrary WGSL source, so a modified shader can serve as a
 * reference. `bindsList` follows the source: an auto layout only declares
 * bindings the chosen entry point can reach, so the reference's bind group
 * must not carry binding 5.
 */
async function runWithRepulsionShader(
  device: GPUDevice,
  graph: {
    nodeCount: number;
    positionsX: Float32Array;
    positionsY: Float32Array;
    flags?: Uint32Array;
  },
  repulsionCode: string,
  bindsList: boolean,
  ticks: number,
): Promise<PositionSnapshot> {
  const mod = await loadPipelineModule();
  const { nodeCount } = graph;

  const pipeline = mod.createSimulationPipeline({ device }, { maxNodes: nodeCount, maxEdges: 1 });
  const buffers = mod.createSimulationBuffers(device, nodeCount, 0);
  mod.copyPositionsToSimulation(device, buffers, graph.positionsX, graph.positionsY);
  // Deliberately no uploadLiveIndices: the buffer keeps the identity list, so
  // a fixture with hidden flags here is the never-refreshed-list case.
  if (graph.flags) device.queue.writeBuffer(buffers.nodeFlags, 0, graph.flags.slice().buffer);

  const repulsion = device.createComputePipeline({
    label: "Reference Repulsion",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: repulsionCode }),
      entryPoint: "main_masked",
    },
  });

  const workgroups = Math.ceil(nodeCount / 256);
  for (let t = 0; t < ticks; t++) {
    mod.updateSimulationUniforms(
      device,
      buffers,
      nodeCount,
      0,
      Math.pow(1 - HARNESS_ALPHA_DECAY, t + 1),
      CUT_CONFIG,
    );
    const bindGroups = mod.createSimulationBindGroups(device, pipeline, buffers);
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: buffers.repulsionUniforms } },
      { binding: 1, resource: { buffer: buffers.positions } },
      { binding: 2, resource: { buffer: buffers.forces } },
      { binding: 3, resource: { buffer: buffers.nodeFlags } },
      { binding: 4, resource: { buffer: buffers.nodeMass } },
    ];
    if (bindsList) entries.push({ binding: 5, resource: { buffer: buffers.liveIndices } });

    const encoder = device.createCommandEncoder();
    mod.recordSimulationStepWithOptions(encoder, pipeline, bindGroups, nodeCount, 0, {
      recordRepulsionPass: (enc) => {
        const pass = enc.beginComputePass({ label: "Repulsion Under Test" });
        pass.setPipeline(repulsion);
        pass.setBindGroup(
          0,
          device.createBindGroup({ layout: repulsion.getBindGroupLayout(0), entries }),
        );
        pass.dispatchWorkgroups(workgroups);
        pass.end();
      },
    });
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

gpuTest(
  "SC-005: with LOD off, 30 ticks are bit-identical to slot-order dispatch",
  async (device) => {
    const shipped = await Deno.readTextFile(REPULSION_N2_URL);
    const stripped = stripGather(shipped);

    // Edge-free: the spring pass's compare-exchange accumulation is not
    // reproducible, and this comparison is exact.
    const tree = generateCodeTree({ seed: 3, maxNodes: 400 });
    const graph = {
      nodeCount: tree.nodeCount,
      positionsX: tree.positionsX,
      positionsY: tree.positionsY,
    };

    const gathered = await runWithRepulsionShader(device, graph, shipped, true, 30);
    const slotOrder = await runWithRepulsionShader(device, graph, stripped, false, 30);
    const gatheredAgain = await runWithRepulsionShader(device, graph, shipped, true, 30);

    // A rig that is not reproducible run-to-run cannot prove bit-identity.
    assertBitIdentical(gatheredAgain.x, gathered.x, "self-consistency x");
    assertBitIdentical(gatheredAgain.y, gathered.y, "self-consistency y");

    assertEquals(countNonFinite(gathered.x, gathered.y), 0);
    assertBitIdentical(gathered.x, slotOrder.x, "SC-005 x");
    assertBitIdentical(gathered.y, slotOrder.y, "SC-005 y");
  },
);

gpuTest(
  "active dispatch: a never-refreshed list is slower, never wrong",
  async (device) => {
    // The list is derived from the flags at one choke point, but the flag bit
    // remains the state and the mask remains in the shader. So a buffer set
    // whose list was never refreshed — the identity list createSimulationBuffers
    // writes — must produce exactly the physics the refreshed one does: the
    // gather then walks every slot, and the mask makes the hidden ones inert.
    // Over-inclusion costs threads; it cannot cost correctness.
    //
    // This is what lets the flag be set by paths that know nothing about the
    // list (and is why the mask cannot be dropped as redundant).
    const shipped = await Deno.readTextFile(REPULSION_N2_URL);
    const cut = collapseBranches(
      generateCodeTree({ ...CODE_TREE_SCALES.small, seed: 61, maxNodes: 900 }),
      3,
      6,
    );
    assert(cut.hidden.length > 200, "fixture must hide a substantial part of the tree");

    const staleList = await runWithRepulsionShader(
      device,
      {
        nodeCount: cut.tree.nodeCount,
        positionsX: cut.tree.positionsX,
        positionsY: cut.tree.positionsY,
        flags: cut.flags,
      },
      shipped,
      true,
      25,
    );

    const derivedList = await runCutFixture(
      device,
      {
        nodeCount: cut.tree.nodeCount,
        positionsX: cut.tree.positionsX,
        positionsY: cut.tree.positionsY,
        ...SELF_LOOP,
        depths: new Float32Array(cut.tree.nodeCount),
        flags: cut.flags,
      },
      25,
      cut.visible.length,
    );

    assertBitIdentical(
      Float32Array.from(cut.visible, (slot) => staleList.x[slot]),
      Float32Array.from(cut.visible, (slot) => derivedList.x[slot]),
      "stale vs derived list x",
    );
    assertBitIdentical(
      Float32Array.from(cut.visible, (slot) => staleList.y[slot]),
      Float32Array.from(cut.visible, (slot) => derivedList.y[slot]),
      "stale vs derived list y",
    );
    assertFrozen(
      { x: cut.tree.positionsX, y: cut.tree.positionsY },
      staleList,
      cut.hidden,
    );
  },
);

// =============================================================================
// The list itself: contents, count, and buffer identity
// =============================================================================

gpuTest("active dispatch: an entirely hidden graph is a valid state", async (device) => {
  // The degenerate end of the cut. An empty list means a zero-workgroup
  // dispatch and a zero-length buffer write, both of which have to be
  // no-ops rather than a panic — the alternative is that collapsing the last
  // visible root takes the renderer down.
  const nodeCount = 64;
  const tree = generateCodeTree({ ...CODE_TREE_SCALES.small, seed: 5, maxNodes: nodeCount });
  const before = { x: tree.positionsX.slice(), y: tree.positionsY.slice() };
  const harness = await createSimHarness(device, {
    nodeCount: tree.nodeCount,
    positionsX: tree.positionsX,
    positionsY: tree.positionsY,
    edgeSources: tree.edgeSources,
    edgeTargets: tree.edgeTargets,
    flags: new Uint32Array(tree.nodeCount).fill(NODE_FLAG_HIDDEN_LOD),
  }, CUT_CONFIG);
  try {
    assertEquals(harness.activeCount, 0);
    await harness.tick(5);
    const after = await harness.readPositions();
    assertFrozen(before, after, Uint32Array.from({ length: tree.nodeCount }, (_, i) => i));
  } finally {
    harness.dispose();
  }
});

gpuTest("active dispatch: a fresh buffer set holds the identity list", async (device) => {
  const mod = await loadPipelineModule();
  const nodeCount = 137; // not a workgroup multiple: the tail must be filled too
  const nodeCapacity = 300;
  const buffers = mod.createSimulationBuffers(device, nodeCount, 0, nodeCapacity);

  assertEquals(buffers.activeCount, nodeCount, "a fresh set dispatches over every node");

  const readback = device.createBuffer({
    size: nodeCapacity * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffers.liveIndices, 0, readback, 0, nodeCapacity * 4);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const list = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  // The whole capacity, not just the current node count: a later add raises
  // activeCount before anything re-derives the list, and the entries it starts
  // reading must already name their own slots.
  for (let i = 0; i < nodeCapacity; i++) {
    assertEquals(list[i], i, `identity list entry ${i}`);
  }
});

gpuTest("active dispatch: collapse and expand move the count, never the buffer", async (device) => {
  const activeSet = await loadActiveSetModule();
  const cut = collapseBranches(
    generateCodeTree({ ...CODE_TREE_SCALES.small, seed: 11, maxNodes: 800 }),
    3,
    12,
  );
  const harness = await createSimHarness(device, {
    nodeCount: cut.tree.nodeCount,
    positionsX: cut.tree.positionsX,
    positionsY: cut.tree.positionsY,
    edgeSources: cut.tree.edgeSources,
    edgeTargets: cut.tree.edgeTargets,
  });
  try {
    const initial = harness.liveIndicesBuffer;
    assertEquals(harness.activeCount, cut.tree.nodeCount);
    await harness.tick(3);

    harness.setNodeFlags(cut.flags);
    assertEquals(harness.activeCount, cut.visible.length, "collapse must shrink the dispatch");
    assertStrictEquals(harness.liveIndicesBuffer, initial, "a collapse must not reallocate");
    await harness.tick(3);

    harness.setNodeFlags(new Uint32Array(cut.tree.nodeCount));
    assertEquals(harness.activeCount, cut.tree.nodeCount, "expand must restore the dispatch");
    assertStrictEquals(harness.liveIndicesBuffer, initial, "an expand must not reallocate");
    await harness.tick(3);

    const after = await harness.readPositions();
    assertEquals(countNonFinite(after.x, after.y), 0);

    // A reallocation is the one thing that does replace it, and it must come
    // back describing the whole graph rather than zero-filled (which would
    // dispatch every thread at slot 0).
    await harness.reallocate();
    assert(harness.liveIndicesBuffer !== initial, "reallocation replaces the buffer");
    await harness.tick(3);
    const grown = await harness.readPositions();
    assertEquals(countNonFinite(grown.x, grown.y), 0);

    // And the derivation the harness and GraphMother share agrees with the cut.
    const derived = new Uint32Array(cut.tree.nodeCount);
    const count = activeSet.deriveActiveIndices(cut.flags, cut.tree.nodeCount, derived);
    assertEquals(Array.from(derived.subarray(0, count)), Array.from(cut.visible));
  } finally {
    harness.dispose();
  }
});
