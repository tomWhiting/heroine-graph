/**
 * GPU integration tests: drive the real simulation compute pipeline
 * headlessly (no canvas, no renderer) on fixture graphs and assert
 * physical invariants of the output.
 *
 * Requires a WebGPU adapter (Deno flag --unstable-webgpu, wired into
 * `deno task test`). When no adapter is available the tests are skipped
 * with an explanatory message rather than failing.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  createSimHarness,
  GPU_SKIP_MESSAGE,
  probeAdapter,
  requestHarnessDevice,
} from "../helpers/gpu.ts";
import {
  countNonFinite,
  kineticEnergy,
  maxRadius,
  meanParentChildDistance,
  radialOrderingScore,
} from "../helpers/invariants.ts";
import { generateCodeTree } from "../fixtures/code_tree.ts";

const adapter = await probeAdapter();
if (!adapter) {
  console.warn(`[gpu] ${GPU_SKIP_MESSAGE}`);
}

/**
 * GPU tests share one device and disable the resource/op sanitizers:
 * Deno's WebGPU bindings keep adapter/device handles alive at process
 * scope, which the sanitizer would misreport as test-local leaks.
 */
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

gpuTest(
  "GPU simulation: code-tree layout stays finite, bounded, and cools; pinned node fixed",
  async (device) => {
    const tree = generateCodeTree({
      seed: 2026,
      maxDepth: 4,
      minChildren: 2,
      maxChildren: 5,
      maxNodes: 300,
      crossEdgeRatio: 0.15,
    });

    const harness = await createSimHarness(
      device,
      tree,
      { pinnedNode: 0, centerX: 0, centerY: 0 },
    );

    try {
      // Kinetic energy sampled from position deltas across single ticks
      await harness.tick(19);
      const p19 = await harness.readPositions();
      await harness.tick(1);
      const p20 = await harness.readPositions();
      const energyAt20 = kineticEnergy(p19.x, p19.y, p20.x, p20.y);

      await harness.tick(179);
      const p199 = await harness.readPositions();
      await harness.tick(1);
      const p200 = await harness.readPositions();
      const energyAt200 = kineticEnergy(p199.x, p199.y, p200.x, p200.y);

      assertEquals(harness.tickCount, 200);

      // Invariant: no NaN/Inf anywhere
      assertEquals(countNonFinite(p20.x, p20.y), 0, "non-finite positions at tick 20");
      assertEquals(countNonFinite(p200.x, p200.y), 0, "non-finite positions at tick 200");

      // Invariant: layout stays bounded (fixture starts inside r ~ 175;
      // a stable simulation must not fling nodes to extreme radii)
      const radius = maxRadius(p200.x, p200.y);
      assert(radius < 5000, `layout exploded: max radius ${radius}`);

      // Invariant: the simulation cools — energy at tick 200 is below tick 20
      assert(energyAt20 > 0, "simulation never moved");
      assert(
        energyAt200 < energyAt20,
        `not cooling: KE(200)=${energyAt200} >= KE(20)=${energyAt20}`,
      );

      // Invariant: pinned node is held exactly at the configured center
      assertEquals(p200.x[0], 0, "pinned node drifted in x");
      assertEquals(p200.y[0], 0, "pinned node drifted in y");

      // Structural metrics: exercised (and logged) for future physics
      // tuning; only sanity-bounded for now.
      const parentDist = meanParentChildDistance(p200.x, p200.y, tree.parent);
      const radialScore = radialOrderingScore(p200.x, p200.y, tree.parent);
      assert(
        Number.isFinite(parentDist) && parentDist > 0,
        `degenerate parent-child distance: ${parentDist}`,
      );
      assert(radialScore >= 0 && radialScore <= 1);
      console.log(
        `[gpu] tree(${tree.nodeCount}n/${tree.edgeCount}e) tick 200: ` +
          `KE20=${energyAt20.toFixed(1)} KE200=${energyAt200.toFixed(1)} ` +
          `maxR=${radius.toFixed(1)} parentDist=${parentDist.toFixed(1)} ` +
          `radialOrder=${radialScore.toFixed(2)}`,
      );
    } finally {
      harness.dispose();
    }
  },
);

gpuTest(
  "GPU simulation: spring pulls a stretched pair together, repulsion keeps it apart",
  async (device) => {
    // Two nodes far beyond the spring rest length (30): the net force
    // must contract the pair, but repulsion must keep it clearly separated.
    const harness = await createSimHarness(device, {
      nodeCount: 2,
      positionsX: new Float32Array([-150, 150]),
      positionsY: new Float32Array([0, 0]),
      edgeSources: new Uint32Array([0]),
      edgeTargets: new Uint32Array([1]),
    }, { centerStrength: 0 });

    try {
      await harness.tick(150);
      const { x, y } = await harness.readPositions();

      assertEquals(countNonFinite(x, y), 0);
      const distance = Math.hypot(x[1] - x[0], y[1] - y[0]);
      assert(distance < 300, `spring never contracted the pair: d=${distance}`);
      assert(distance > 1, `pair collapsed to a point: d=${distance}`);
    } finally {
      harness.dispose();
    }
  },
);

gpuTest(
  "GPU simulation: pinned node stays fixed while free nodes move",
  async (device) => {
    const tree = generateCodeTree({ seed: 11, maxDepth: 3, maxNodes: 50 });
    const harness = await createSimHarness(
      device,
      tree,
      { pinnedNode: 0, centerX: 42, centerY: -17 },
    );

    try {
      await harness.tick(30);
      const { x, y } = await harness.readPositions();

      // Pinned node snaps to and holds the configured center exactly
      assertEquals(x[0], 42);
      assertEquals(y[0], -17);

      // Free nodes actually moved from their seeded positions
      let moved = 0;
      for (let i = 1; i < tree.nodeCount; i++) {
        if (x[i] !== tree.positionsX[i] || y[i] !== tree.positionsY[i]) moved++;
      }
      assert(moved > 0, "no free node moved in 30 ticks");
    } finally {
      harness.dispose();
    }
  },
);
