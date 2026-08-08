/**
 * GPU tests for the prebuilt ping/pong bind-group sets.
 *
 * The position/velocity buffers ping-pong every frame, which is why every
 * bind group referencing them used to be recreated per frame (~14
 * createBindGroup calls, ~800/sec at 60fps). There are only two possible
 * buffer configurations, so both are now built once at allocation time and
 * the per-frame path is an index flip.
 *
 * The state machine under test is BindGroupParitySets/ParitySlot in
 * packages/core/src/simulation/pipeline.ts — the same objects GraphMother
 * delegates to (it keeps no parity state of its own; see the slots declared in
 * api/graph.ts). The harness drives that class rather than reimplementing it,
 * so a mutation to the shipped mechanism fails these tests. Its bookkeeping in
 * isolation — flip sequencing, reallocation at odd parity, the staleness
 * invariant — is covered without a GPU in tests/unit/parity_sets_test.ts.
 *
 * An off-by-one in the indexing would otherwise be almost invisible: the
 * simulation would keep producing plausible-looking motion while reading the
 * buffer it is writing. These tests therefore run several ticks against the
 * pre-optimization reference — bind groups rebuilt from the live buffers
 * after every swap — and compare the resulting layouts, bit for bit wherever
 * the pipeline is exactly reproducible (see CONFIG_EXACT) and within a slack
 * far below any parity error otherwise. Two cases reallocate every simulation
 * buffer mid-run at odd parity, and a final test asserts the steady-state tick
 * loop allocates no bind groups at all.
 *
 * Requires a WebGPU adapter (Deno flag --unstable-webgpu, wired into
 * `deno task test`). When no adapter is available the tests are skipped.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  createAlgorithmSimHarness,
  createSimHarness,
  GPU_SKIP_MESSAGE,
  HARNESS_STORAGE_BUFFERS_PER_STAGE,
  type HarnessForceAlgorithm,
  loadModuleInliningWgsl,
  probeAdapter,
  requestHarnessDevice,
  type SimHarness,
} from "../helpers/gpu.ts";
import { countNonFinite } from "../helpers/invariants.ts";
import { type CodeTreeGraph, generateCodeTree } from "../fixtures/code_tree.ts";
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
      // Production's limits, not the WebGPU defaults: a bare requestDevice()
      // caps maxStorageBuffersPerShaderStage at 8, which invalidates
      // Barnes-Hut's 10-buffer Karras tree layout. The resulting invalid bind
      // group poisons the encoder and the whole tick is discarded, so the
      // simulation looks inert instead of erroring.
      const device = await requestHarnessDevice(adapter!);
      try {
        await fn(device);
      } finally {
        device.destroy();
      }
    },
  });
}

/** Enough ticks that both parities are exercised repeatedly. */
const TICKS = 5;

/**
 * Force config the cases run under. The library defaults leave several
 * algorithms nearly static over five ticks, which would make the comparison
 * vacuous; assertEvolved enforces that it is not.
 */
const CONFIG = { repulsionStrength: -50, theta: 0.8 };

function fixture(): CodeTreeGraph {
  return generateCodeTree({ seed: 4242, maxNodes: 200, maxDepth: 5 });
}

/**
 * Config that silences the per-edge passes without removing the edges.
 *
 * Every per-edge pass (the shared spring pass and the algorithms' native
 * attraction) accumulates into `forces` with atomics, and float atomic adds
 * are not associative — two independent runs of the SAME code can differ by
 * an ULP purely from workgroup scheduling, which rules out a bit-identical
 * comparison. Zeroing the spring/attraction strengths keeps the passes
 * dispatched but makes every contribution exactly 0, which IS
 * order-independent, leaving the remaining per-node passes exact.
 */
const CONFIG_EXACT = { ...CONFIG, springStrength: 0, attraction: 0 };

/** Fails if the run left positions untouched, which would make any comparison trivial. */
function assertEvolved(
  positions: { x: Float32Array; y: Float32Array },
  graph: CodeTreeGraph,
  label: string,
): void {
  let moved = 0;
  for (let i = 0; i < positions.x.length; i++) {
    if (positions.x[i] !== graph.positionsX[i] || positions.y[i] !== graph.positionsY[i]) moved++;
  }
  assert(
    moved > graph.nodeCount / 2,
    `${label}: simulation did not evolve (${moved}/${graph.nodeCount} nodes moved) — ` +
      "the parity comparison would be vacuous",
  );
}

/**
 * Asserts the two position sets agree within `tolerance` (0 = bit-identical).
 *
 * The tolerance exists only to absorb atomic accumulation order in the
 * per-edge passes (see edgelessFixture); it is orders of magnitude tighter
 * than a wrong parity index, which reruns a whole tick from a stale buffer
 * and moves nodes by tens of units on this fixture.
 */
function assertPositionsAgree(
  actual: { x: Float32Array; y: Float32Array },
  expected: { x: Float32Array; y: Float32Array },
  tolerance: number,
  label: string,
): void {
  assertEquals(actual.x.length, expected.x.length);
  for (let i = 0; i < actual.x.length; i++) {
    const dx = Math.abs(actual.x[i] - expected.x[i]);
    const dy = Math.abs(actual.y[i] - expected.y[i]);
    assert(
      dx <= tolerance && dy <= tolerance,
      `${label}: node ${i} diverged by (${dx}, ${dy}) — prebuilt parity ` +
        `(${actual.x[i]}, ${actual.y[i]}) vs rebuilt reference ` +
        `(${expected.x[i]}, ${expected.y[i]})`,
    );
  }
}

/** Runs both harnesses for TICKS ticks and asserts they produce the same layout. */
async function assertParityMatchesReference(
  prebuilt: SimHarness,
  reference: SimHarness,
  graph: CodeTreeGraph,
  label: string,
  tolerance: number,
): Promise<void> {
  try {
    await prebuilt.tick(TICKS);
    await reference.tick(TICKS);

    const a = await prebuilt.readPositions();
    const b = await reference.readPositions();

    assertEquals(countNonFinite(a.x, a.y), 0);
    assertEvolved(a, graph, label);
    assertPositionsAgree(a, b, tolerance, label);
  } finally {
    prebuilt.dispose();
    reference.dispose();
  }
}

/**
 * Runs both harnesses through a mid-run reallocation and asserts they still
 * agree.
 *
 * The reallocation deliberately lands at odd parity — the case
 * `buildForBothParities`' `currentParity` parameter exists for, and the one
 * every reallocation path in graph.ts (reallocateNodeBuffers,
 * reallocateEdgeBuffers, ensureAlgorithmCapacity, setForceAlgorithm,
 * initializeCollisionResources) can hit at runtime. If the rebuilt sets landed
 * at the wrong index, the prebuilt harness would run a tick from the buffer it
 * is writing; the ParitySlot staleness check turns that into a thrown error,
 * and any residual divergence shows up in the comparison below.
 */
async function assertParityMatchesAcrossReallocation(
  prebuilt: SimHarness,
  reference: SimHarness,
  graph: CodeTreeGraph,
  label: string,
  tolerance: number,
): Promise<void> {
  try {
    const oddTicks = 3;
    await prebuilt.tick(oddTicks);
    await reference.tick(oddTicks);
    assertEquals(prebuilt.parity, 1, "the reallocation must land while parity is odd");
    assertEquals(reference.parity, 1);

    await prebuilt.reallocate();
    await reference.reallocate();

    await prebuilt.tick(TICKS);
    await reference.tick(TICKS);

    const a = await prebuilt.readPositions();
    const b = await reference.readPositions();

    assertEquals(countNonFinite(a.x, a.y), 0);
    assertEvolved(a, graph, label);
    assertPositionsAgree(a, b, tolerance, label);
  } finally {
    prebuilt.dispose();
    reference.dispose();
  }
}

/**
 * Bit-identity is required under CONFIG_EXACT. With the per-edge passes live,
 * atomic accumulation order lets two runs of identical code drift; measured
 * at ~5e-3 units after five ticks on this fixture, versus the ~40 units a
 * wrong parity index produces (it reruns a whole tick from a stale buffer).
 * The slack sits between the two by three orders of magnitude.
 */
const EXACT = 0;
const ATOMIC_ORDER_SLACK = 0.05;

/**
 * Barnes-Hut's slack. Its tree build aggregates centre-of-mass with atomics
 * and its traversal order depends on the Morton sort, so it is not bit-exact
 * across runs even with the per-edge passes silenced. Measured prebuilt-vs-
 * rebuilt divergence after five ticks: 1.07e-4 at this fixture's 200 nodes and
 * 4.39e-3 at 1008 nodes. 0.05 covers both with an order of magnitude to spare
 * while staying ~3 orders below the tens of units a wrong parity index costs.
 */
const BARNES_HUT_SLACK = 0.05;

/** Bounds refresh cadence for the spatial algorithms (mirrors graph.ts's sync). */
const BOUNDS_SYNC_INTERVAL = 2;

/** Loads an algorithm plugin factory with its .wgsl imports inlined. */
async function loadAlgorithm(
  module: string,
  factory: string,
): Promise<HarnessForceAlgorithm> {
  const mod = await loadModuleInliningWgsl<Record<string, () => HarnessForceAlgorithm>>(
    new URL(`../../packages/core/src/simulation/algorithms/${module}`, import.meta.url),
  );
  return mod[factory]();
}

gpuTest(
  "GPU bind group parity: simulation passes are bit-identical to per-tick rebuilds",
  async (device) => {
    const graph = fixture();
    await assertParityMatchesReference(
      await createSimHarness(device, graph, CONFIG_EXACT),
      await createSimHarness(
        device,
        graph,
        CONFIG_EXACT,
        undefined,
        undefined,
        "rebuild-each-tick",
      ),
      graph,
      "simulation passes (exact config)",
      EXACT,
    );
  },
);

gpuTest(
  "GPU bind group parity: simulation passes match per-tick rebuilds with edges",
  async (device) => {
    // Adds the shared spring pass, which binds positions and accumulates
    // atomically — hence the slack (see edgelessFixture).
    const graph = fixture();
    await assertParityMatchesReference(
      await createSimHarness(device, graph, CONFIG),
      await createSimHarness(device, graph, CONFIG, undefined, undefined, "rebuild-each-tick"),
      graph,
      "simulation passes",
      ATOMIC_ORDER_SLACK,
    );
  },
);

gpuTest(
  "GPU bind group parity: algorithm bind groups are bit-identical to per-tick rebuilds",
  async (device) => {
    // The N² plugin is the one algorithm whose repulsion is exactly
    // reproducible: each thread sums its own row in index order, with no
    // atomics and no tiled partial sums. Under CONFIG_EXACT the surrounding
    // spring pass contributes exactly zero, so the whole run is deterministic
    // and the algorithm's parity indexing can be checked to the last bit.
    const graph = fixture();
    await assertParityMatchesReference(
      await createAlgorithmSimHarness(
        device,
        await loadAlgorithm("n2.ts", "createN2Algorithm"),
        graph,
        CONFIG_EXACT,
      ),
      await createAlgorithmSimHarness(
        device,
        await loadAlgorithm("n2.ts", "createN2Algorithm"),
        graph,
        CONFIG_EXACT,
        undefined,
        { bindGroupMode: "rebuild-each-tick" },
      ),
      graph,
      "n2 repulsion (exact config)",
      EXACT,
    );
  },
);

gpuTest(
  "GPU bind group parity: ForceAtlas2 matches per-tick rebuilds with edges",
  async (device) => {
    // FA2 binds positions in both of its passes (repulsion and its native
    // attraction), so a wrong parity index shows up in the layout immediately.
    const graph = fixture();
    await assertParityMatchesReference(
      await createAlgorithmSimHarness(
        device,
        await loadAlgorithm("force-atlas2.ts", "createForceAtlas2Algorithm"),
        graph,
        CONFIG,
      ),
      await createAlgorithmSimHarness(
        device,
        await loadAlgorithm("force-atlas2.ts", "createForceAtlas2Algorithm"),
        graph,
        CONFIG,
        undefined,
        { bindGroupMode: "rebuild-each-tick" },
      ),
      graph,
      "force-atlas2 passes",
      ATOMIC_ORDER_SLACK,
    );
  },
);

gpuTest(
  "GPU bind group parity: t-FDP matches per-tick rebuilds",
  async (device) => {
    // t-FDP builds its repulsion bind group's entry list conditionally (the
    // nodeFlags binding is only appended when the host supplies the buffer),
    // so both parity variants must take the same branch.
    const graph = fixture();
    await assertParityMatchesReference(
      await createAlgorithmSimHarness(
        device,
        await loadAlgorithm("t-fdp.ts", "createTFdpAlgorithm"),
        graph,
        CONFIG,
      ),
      await createAlgorithmSimHarness(
        device,
        await loadAlgorithm("t-fdp.ts", "createTFdpAlgorithm"),
        graph,
        CONFIG,
        undefined,
        { bindGroupMode: "rebuild-each-tick" },
      ),
      graph,
      "t-fdp passes",
      ATOMIC_ORDER_SLACK,
    );
  },
);

gpuTest(
  "GPU bind group parity: Barnes-Hut matches per-tick rebuilds",
  async (device) => {
    // Barnes-Hut binds positions in its bounds, Morton, tree-build and
    // traversal passes, so a wrong parity index is immediately visible. Its
    // widest pass binds 8 storage buffers, the WebGPU default — on a device
    // below that the pipelines are invalid and every submit is silently
    // discarded, so skip rather than assert on a frozen run.
    if (device.limits.maxStorageBuffersPerShaderStage < HARNESS_STORAGE_BUFFERS_PER_STAGE) {
      console.warn(
        `[gpu] skipping Barnes-Hut parity: device supports only ` +
          `${device.limits.maxStorageBuffersPerShaderStage} storage buffers per stage ` +
          `(needs ${HARNESS_STORAGE_BUFFERS_PER_STAGE})`,
      );
      return;
    }
    const graph = fixture();
    await assertParityMatchesReference(
      await createAlgorithmSimHarness(
        device,
        await loadAlgorithm("barnes-hut.ts", "createBarnesHutAlgorithm"),
        graph,
        CONFIG,
        undefined,
        { boundsSyncInterval: BOUNDS_SYNC_INTERVAL },
      ),
      await createAlgorithmSimHarness(
        device,
        await loadAlgorithm("barnes-hut.ts", "createBarnesHutAlgorithm"),
        graph,
        CONFIG,
        undefined,
        {
          boundsSyncInterval: BOUNDS_SYNC_INTERVAL,
          bindGroupMode: "rebuild-each-tick",
        },
      ),
      graph,
      "barnes-hut passes",
      BARNES_HUT_SLACK,
    );
  },
);

gpuTest(
  "GPU bind group parity: Relativity Atlas in bubble mode matches per-tick rebuilds",
  async (device) => {
    // Relativity Atlas binds positions in five passes, and bubble mode adds a
    // sixth: forest-root separation. Bubble mode is switched on here
    // specifically so that pass dispatches — with it off the root list is empty
    // and the pass is skipped, which would leave the new pass out of the run
    // entirely.
    //
    // What this case establishes is that the pass takes part in a parity run
    // and produces forces the reference agrees with, NOT that its bind group
    // carries the right orientation: the comparison poisons both arms
    // identically, so a bind group captured at one orientation and reused at
    // the other passes it. That defect is the subject of the case below.
    //
    // Its widest pass binds 8 storage buffers, the WebGPU default; below that
    // the pipelines are invalid and every submit is silently discarded.
    if (device.limits.maxStorageBuffersPerShaderStage < HARNESS_STORAGE_BUFFERS_PER_STAGE) {
      console.warn(
        `[gpu] skipping Relativity Atlas parity: device supports only ` +
          `${device.limits.maxStorageBuffersPerShaderStage} storage buffers per stage ` +
          `(needs ${HARNESS_STORAGE_BUFFERS_PER_STAGE})`,
      );
      return;
    }

    const graph = fixture();
    // A forest of singleton roots: the pass needs two or more roots to
    // dispatch, and it reads only positions, wells, flags and this list, so a
    // list is all the topology it takes to exercise the bind group.
    const roots = new Uint32Array(graph.nodeCount);
    for (let i = 0; i < graph.nodeCount; i++) roots[i] = i;

    const mod = await loadModuleInliningWgsl<
      {
        createRelativityAtlasAlgorithm(): HarnessForceAlgorithm;
        uploadRelativityAtlasBubbleRoots(
          device: GPUDevice,
          buffers: { destroy(): void },
          roots: Uint32Array,
        ): void;
      }
    >(
      new URL(
        "../../packages/core/src/simulation/algorithms/relativity-atlas.ts",
        import.meta.url,
      ),
    );
    const withRoots = (bindGroupMode?: "rebuild-each-tick"): Promise<SimHarness> =>
      createAlgorithmSimHarness(
        device,
        mod.createRelativityAtlasAlgorithm(),
        graph,
        { ...CONFIG, relativityBubbleMode: true },
        undefined,
        {
          boundsSyncInterval: BOUNDS_SYNC_INTERVAL,
          onAlgorithmBuffers: (algoBuffers) =>
            mod.uploadRelativityAtlasBubbleRoots(device, algoBuffers, roots),
          ...(bindGroupMode ? { bindGroupMode } : {}),
        },
      );

    await assertParityMatchesReference(
      await withRoots(),
      await withRoots("rebuild-each-tick"),
      graph,
      "relativity-atlas bubble passes",
      ATOMIC_ORDER_SLACK,
    );
  },
);

/**
 * One ping-pong orientation: the position/force pair a bind group is built
 * against.
 *
 * The two orientations below hold DIFFERENT positions and separate force
 * buffers, which is what the parity-versus-reference comparison cannot do —
 * there both arms see the same positions, so a bind group that reads the wrong
 * one is invisible.
 */
interface Orientation {
  positions: GPUBuffer;
  forces: GPUBuffer;
  readForces(): Promise<Float32Array>;
  destroy(): void;
}

/** Allocate an orientation holding `positions` (interleaved x,y) and zero forces. */
function createOrientation(device: GPUDevice, positions: Float32Array): Orientation {
  const nodeCount = positions.length / 2;
  const positionBuffer = device.createBuffer({
    size: positions.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(positionBuffer, 0, positions);
  const forces = device.createBuffer({
    size: nodeCount * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(forces, 0, new Float32Array(nodeCount * 2));

  return {
    positions: positionBuffer,
    forces,
    async readForces(): Promise<Float32Array> {
      const readback = device.createBuffer({
        size: nodeCount * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(forces, 0, readback, 0, nodeCount * 8);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      readback.destroy();
      return out;
    },
    destroy(): void {
      positionBuffer.destroy();
      forces.destroy();
    },
  };
}

gpuTest(
  "GPU bind group parity: the bubble root pass binds the orientation it was built for",
  async (device) => {
    // A bind group captured once and reused at both orientations reads the
    // buffer the simulation is writing and accumulates into the one it is
    // reading — a real defect that the parity-versus-reference comparison
    // cannot see, because it poisons both arms identically. So this case does
    // not compare two runs: it builds the algorithm's bind groups at two
    // orientations whose positions DISAGREE, dispatches the second, and checks
    // that the forces landed in the second orientation's buffer and describe
    // the second orientation's geometry.
    if (device.limits.maxStorageBuffersPerShaderStage < HARNESS_STORAGE_BUFFERS_PER_STAGE) {
      console.warn(
        `[gpu] skipping bubble root orientation: device supports only ` +
          `${device.limits.maxStorageBuffersPerShaderStage} storage buffers per stage ` +
          `(needs ${HARNESS_STORAGE_BUFFERS_PER_STAGE})`,
      );
      return;
    }

    const mod = await loadModuleInliningWgsl<
      {
        createRelativityAtlasAlgorithm(): HarnessForceAlgorithm;
        uploadRelativityAtlasBubbleRoots(
          device: GPUDevice,
          buffers: { destroy(): void },
          roots: Uint32Array,
        ): void;
      }
    >(
      new URL(
        "../../packages/core/src/simulation/algorithms/relativity-atlas.ts",
        import.meta.url,
      ),
    );

    const algorithm = mod.createRelativityAtlasAlgorithm();
    const pipelines = algorithm.createPipelines({ device });
    const nodeCount = 2;
    const buffers = algorithm.createBuffers(device, nodeCount);
    // Two roots with overlapping wells: the only pair the root pass can act on,
    // and with no edges uploaded the CSR is empty, so no other pass has
    // anything to contribute. Gravity and the density grid are switched off in
    // the config below, leaving root separation as the sole force in the run.
    mod.uploadRelativityAtlasBubbleRoots(device, buffers, Uint32Array.from([0, 1]));
    device.queue.writeBuffer(
      (buffers as unknown as { wellRadius: GPUBuffer }).wellRadius,
      0,
      new Float32Array([100, 100]),
    );

    // The pair overlaps along x at one orientation and along y at the other, so
    // the axis the forces come out on names the positions buffer that was read.
    const even = createOrientation(device, new Float32Array([-30, 0, 30, 0]));
    const odd = createOrientation(device, new Float32Array([0, -30, 0, 30]));
    const edgeStub = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const forceConfig = validateForceConfig({
      ...CONFIG,
      relativityBubbleMode: true,
      relativityDensityRepulsion: 0,
      centerStrength: 0,
    });
    const contextFor = (orientation: Orientation) => ({
      device,
      positions: orientation.positions,
      forces: orientation.forces,
      nodeCount,
      edgeCount: 0,
      forceConfig,
      bounds: { minX: -100, minY: -100, maxX: 100, maxY: 100 },
      edgeSources: edgeStub,
      edgeTargets: edgeStub,
    });

    try {
      // Built in this order so that a cache keyed on nothing — the shape of the
      // defect — hands the even orientation's group back for the odd one.
      algorithm.createBindGroups(device, pipelines, contextFor(even), buffers);
      const oddBindGroups = algorithm.createBindGroups(
        device,
        pipelines,
        contextFor(odd),
        buffers,
      );

      // Uniforms are per-frame state written for the orientation about to run,
      // exactly as graph.ts writes them before recording the tick.
      algorithm.updateUniforms(device, buffers, contextFor(odd));
      const encoder = device.createCommandEncoder();
      algorithm.recordRepulsionPass(encoder, pipelines, oddBindGroups, nodeCount);
      device.queue.submit([encoder.finish()]);

      const oddForces = await odd.readForces();
      const evenForces = await even.readForces();
      console.log(
        `[gpu] bubble root orientation: odd F0 = (${oddForces[0].toFixed(3)}, ` +
          `${oddForces[1].toFixed(3)}), even F0 = (${evenForces[0].toFixed(3)}, ` +
          `${evenForces[1].toFixed(3)})`,
      );

      // The odd orientation stacks the pair on the y axis, so separation is a
      // y force. An x force here would mean the pass read the even positions.
      assert(
        oddForces[1] < -1 && oddForces[3] > 1,
        `root pass did not separate the orientation it was dispatched with: ` +
          `F0 = (${oddForces[0]}, ${oddForces[1]}), F1 = (${oddForces[2]}, ${oddForces[3]})`,
      );
      assert(
        Math.abs(oddForces[0]) < 1e-4 && Math.abs(oddForces[2]) < 1e-4,
        `root pass produced force on the axis of the OTHER orientation's positions: ` +
          `F0.x = ${oddForces[0]}, F1.x = ${oddForces[2]}`,
      );
      for (let i = 0; i < evenForces.length; i++) {
        assertEquals(
          evenForces[i],
          0,
          `the orientation that was not dispatched was written to at component ${i}`,
        );
      }
    } finally {
      even.destroy();
      odd.destroy();
      edgeStub.destroy();
      buffers.destroy();
    }
  },
);

gpuTest(
  "GPU bind group parity: simulation buffers reallocated at odd parity stay in sync",
  async (device) => {
    // Reallocation is the path no test used to reach: every simulation buffer
    // is replaced mid-run and both parity variants must be rebuilt against the
    // new buffers at the parity the run has already reached.
    const graph = fixture();
    await assertParityMatchesAcrossReallocation(
      await createSimHarness(device, graph, CONFIG_EXACT),
      await createSimHarness(
        device,
        graph,
        CONFIG_EXACT,
        undefined,
        undefined,
        "rebuild-each-tick",
      ),
      graph,
      "simulation passes across reallocation (exact config)",
      EXACT,
    );
  },
);

gpuTest(
  "GPU bind group parity: algorithm bind groups survive reallocation at odd parity",
  async (device) => {
    // Same, for the algorithm's own bind groups: graph.ts rebuilds them from
    // the replaced simulation buffers even when the algorithm's own buffers
    // are untouched (reallocateNodeBuffers).
    const graph = fixture();
    await assertParityMatchesAcrossReallocation(
      await createAlgorithmSimHarness(
        device,
        await loadAlgorithm("n2.ts", "createN2Algorithm"),
        graph,
        CONFIG_EXACT,
      ),
      await createAlgorithmSimHarness(
        device,
        await loadAlgorithm("n2.ts", "createN2Algorithm"),
        graph,
        CONFIG_EXACT,
        undefined,
        { bindGroupMode: "rebuild-each-tick" },
      ),
      graph,
      "n2 repulsion across reallocation (exact config)",
      EXACT,
    );
  },
);

gpuTest(
  "GPU bind group parity: the steady-state tick loop creates no bind groups",
  async (device) => {
    // The point of the change: after setup, ticking must not allocate. Counted
    // by intercepting the device method the optimization removes from the
    // per-frame path.
    const graph = fixture();
    const original = device.createBindGroup.bind(device);
    let created = 0;
    device.createBindGroup = (descriptor: GPUBindGroupDescriptor): GPUBindGroup => {
      created++;
      return original(descriptor);
    };

    try {
      const prebuilt = await createAlgorithmSimHarness(
        device,
        await loadAlgorithm("force-atlas2.ts", "createForceAtlas2Algorithm"),
        graph,
        CONFIG,
      );
      const reference = await createAlgorithmSimHarness(
        device,
        await loadAlgorithm("force-atlas2.ts", "createForceAtlas2Algorithm"),
        graph,
        CONFIG,
        undefined,
        { bindGroupMode: "rebuild-each-tick" },
      );
      try {
        created = 0;
        await prebuilt.tick(TICKS);
        assertEquals(created, 0, "prebuilt parity path allocated bind groups while ticking");

        created = 0;
        await reference.tick(TICKS);
        assert(
          created >= TICKS,
          `reference path should rebuild every tick, but only created ${created} bind groups`,
        );
      } finally {
        prebuilt.dispose();
        reference.dispose();
      }
    } finally {
      device.createBindGroup = original;
    }
  },
);
