/**
 * GPU tests for the prebuilt ping/pong bind-group sets.
 *
 * The position/velocity buffers ping-pong every frame, which is why every
 * bind group referencing them used to be recreated per frame (~14
 * createBindGroup calls, ~800/sec at 60fps). There are only two possible
 * buffer configurations, so both are now built once at allocation time and
 * the per-frame path is an index flip (GraphMother.advanceFrameParity).
 *
 * An off-by-one in that indexing would be almost invisible: the simulation
 * would keep producing plausible-looking motion while reading the buffer it
 * is writing. These tests therefore run several ticks against the
 * pre-optimization reference — bind groups rebuilt from the live buffers
 * after every swap — and compare the resulting layouts, bit for bit wherever
 * the pipeline is exactly reproducible (see CONFIG_EXACT) and within a slack
 * far below any parity error otherwise. A final test asserts the
 * steady-state tick loop allocates no bind groups at all.
 *
 * Requires a WebGPU adapter (Deno flag --unstable-webgpu, wired into
 * `deno task test`). When no adapter is available the tests are skipped.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  createAlgorithmSimHarness,
  createSimHarness,
  GPU_SKIP_MESSAGE,
  type HarnessForceAlgorithm,
  loadModuleInliningWgsl,
  probeAdapter,
  type SimHarness,
} from "../helpers/gpu.ts";
import { countNonFinite } from "../helpers/invariants.ts";
import { type CodeTreeGraph, generateCodeTree } from "../fixtures/code_tree.ts";

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
 * Bit-identity is required under CONFIG_EXACT. With the per-edge passes live,
 * atomic accumulation order lets two runs of identical code drift; measured
 * at ~5e-3 units after five ticks on this fixture, versus the ~40 units a
 * wrong parity index produces (it reruns a whole tick from a stale buffer).
 * The slack sits between the two by three orders of magnitude.
 */
const EXACT = 0;
const ATOMIC_ORDER_SLACK = 0.05;

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
