/**
 * GPU tests for force correctness:
 *
 * - Per-edge force accumulation is race-free: a hub with hundreds of
 *   contiguous edges receives the full sum of its spring forces (the old
 *   non-atomic `forces[idx] += f` lost a degree-proportional fraction —
 *   whole SIMD waves read the same stale value and only one write survived).
 * - FA2-style adaptive speed: a stiff spring pair that oscillates forever
 *   under plain integration converges once per-node swing/traction speed
 *   control is enabled.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import { createSimHarness, GPU_SKIP_MESSAGE, probeAdapter } from "../helpers/gpu.ts";
import { countNonFinite } from "../helpers/invariants.ts";

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

gpuTest(
  "GPU forces: hub spring accumulation loses no edge contributions",
  async (device) => {
    // Star graph engineered for maximal write contention: one hub as the
    // source of 256 contiguous edges, so every SIMD wave in the spring pass
    // hammers the same force accumulator. All leaves sit on a 200-radius
    // arc on one side, so the hub's expected net force is large and every
    // lost update shows up as missing displacement.
    //
    // With repulsion, gravity, and velocity clamping neutralized, one tick
    // from rest moves the hub by exactly its accumulated force
    // (v = F * alpha * dt; x += v * dt; alpha = 1, dt = 1).
    const leafCount = 256;
    const nodeCount = leafCount + 1;
    const positionsX = new Float32Array(nodeCount);
    const positionsY = new Float32Array(nodeCount);
    const edgeSources = new Uint32Array(leafCount);
    const edgeTargets = new Uint32Array(leafCount);

    for (let i = 0; i < leafCount; i++) {
      // Quarter arc from -45 deg to +45 deg, radius 200
      const angle = (-Math.PI / 4) + (Math.PI / 2) * (i / (leafCount - 1));
      positionsX[i + 1] = Math.fround(200 * Math.cos(angle));
      positionsY[i + 1] = Math.fround(200 * Math.sin(angle));
      edgeSources[i] = 0; // hub is the source of every edge
      edgeTargets[i] = i + 1;
    }

    const springStrength = 0.1;
    const springLength = 30;

    // Expected hub force, computed from the exact f32 positions the GPU sees
    let expectedX = 0;
    let expectedY = 0;
    for (let i = 1; i < nodeCount; i++) {
      const dx = positionsX[i];
      const dy = positionsY[i];
      const dist = Math.hypot(dx, dy);
      const magnitude = springStrength * (dist - springLength);
      expectedX += (dx / dist) * magnitude;
      expectedY += (dy / dist) * magnitude;
    }
    const expectedMag = Math.hypot(expectedX, expectedY);

    const harness = await createSimHarness(
      device,
      { nodeCount, positionsX, positionsY, edgeSources, edgeTargets },
      {
        springStrength,
        springLength,
        repulsionStrength: 0,
        centerStrength: 0,
        maxVelocity: 1e6,
      },
      0, // alphaDecay 0: alpha stays exactly 1
    );

    try {
      await harness.tick(1);
      const { x, y } = await harness.readPositions();
      assertEquals(countNonFinite(x, y), 0);

      const gotX = x[0] - 0;
      const gotY = y[0] - 0;
      const err = Math.hypot(gotX - expectedX, gotY - expectedY) / expectedMag;
      // f32 summation-order noise is ~1e-6 relative; the old racy
      // accumulation lost ~90%+ of the hub force here.
      assert(
        err < 1e-3,
        `hub force lost contributions: got (${gotX.toFixed(3)}, ${gotY.toFixed(3)}), ` +
          `expected (${expectedX.toFixed(3)}, ${expectedY.toFixed(3)}), rel err ${err}`,
      );

      // Leaves have one edge each (no contention): each moved toward the hub
      // by its own spring force ~ 0.1 * (200 - 30) = 17
      const leafDisp = Math.hypot(x[1] - positionsX[1], y[1] - positionsY[1]);
      assert(
        Math.abs(leafDisp - 17) < 0.2,
        `leaf displacement wrong: ${leafDisp}, expected ~17`,
      );
    } finally {
      harness.dispose();
    }
  },
);

/**
 * Runs the oscillation-prone stiff-spring pair: a stiff spring (k = 3)
 * between two nodes is unstable under the plain semi-implicit integrator at
 * alpha = 1 — the pair overshoots equilibrium every tick and oscillates at
 * the velocity cap forever. FA2-style swing/traction speed control detects
 * the force sign flips and kills the oscillation.
 */
async function runStiffPair(
  device: GPUDevice,
  adaptiveSpeed: boolean,
  adaptiveSpeedOverride?: boolean,
): Promise<{ maxStep: number; distance: number }> {
  const harness = await createSimHarness(
    device,
    {
      nodeCount: 2,
      positionsX: new Float32Array([-50, 50]),
      positionsY: new Float32Array([0, 0]),
      edgeSources: new Uint32Array([0]),
      edgeTargets: new Uint32Array([1]),
    },
    {
      springStrength: 3,
      springLength: 30,
      repulsionStrength: 0,
      centerStrength: 0,
      velocityDecay: 0.1,
      maxVelocity: 1000,
      adaptiveSpeed,
      adaptiveSpeedStrength: 1.0,
    },
    0, // alphaDecay 0: alpha stays 1, no cooling to mask the oscillation
    adaptiveSpeedOverride,
  );

  try {
    await harness.tick(400);
    const before = await harness.readPositions();
    await harness.tick(1);
    const after = await harness.readPositions();

    assertEquals(countNonFinite(after.x, after.y), 0);

    let maxStep = 0;
    for (let i = 0; i < 2; i++) {
      const step = Math.hypot(after.x[i] - before.x[i], after.y[i] - before.y[i]);
      if (step > maxStep) maxStep = step;
    }
    const distance = Math.hypot(after.x[1] - after.x[0], after.y[1] - after.y[0]);
    return { maxStep, distance };
  } finally {
    harness.dispose();
  }
}

gpuTest(
  "GPU adaptive speed: oscillation-prone stiff spring converges with it on, not off",
  async (device) => {
    const off = await runStiffPair(device, false);
    const on = await runStiffPair(device, true);

    // Without adaptive speed the pair is still bouncing hard after 400 ticks
    assert(
      off.maxStep > 1,
      `expected sustained oscillation with adaptiveSpeed off, step=${off.maxStep}`,
    );

    // With adaptive speed the pair has converged: per-tick displacement
    // below epsilon and the spring sits near its rest length
    assert(
      on.maxStep < 0.05,
      `adaptiveSpeed failed to converge the pair: step=${on.maxStep}`,
    );
    assert(
      Math.abs(on.distance - 30) < 5,
      `converged distance far from rest length: d=${on.distance}`,
    );

    console.log(
      `[gpu] adaptive speed: off step=${off.maxStep.toFixed(3)} d=${off.distance.toFixed(1)} | ` +
        `on step=${on.maxStep.toFixed(4)} d=${on.distance.toFixed(2)}`,
    );
  },
);

gpuTest(
  "GPU adaptive speed: per-algorithm override wins over forceConfig.adaptiveSpeed",
  async (device) => {
    // The path ForceAlgorithm.prefersAdaptiveSpeed rides: graph.ts forwards
    // it to updateSimulationUniforms as adaptiveSpeedOverride, which must win
    // over forceConfig.adaptiveSpeed in both directions.
    const forcedOn = await runStiffPair(device, false, true);
    const forcedOff = await runStiffPair(device, true, false);

    // config off + override true: adaptive speed engages and converges the pair
    assert(
      forcedOn.maxStep < 0.05,
      `override=true did not engage adaptive speed: step=${forcedOn.maxStep}`,
    );
    assert(
      Math.abs(forcedOn.distance - 30) < 5,
      `override=true converged distance far from rest length: d=${forcedOn.distance}`,
    );

    // config on + override false: adaptive speed suppressed, pair keeps bouncing
    assert(
      forcedOff.maxStep > 1,
      `override=false failed to suppress adaptive speed: step=${forcedOff.maxStep}`,
    );

    console.log(
      `[gpu] adaptive speed override: forcedOn step=${forcedOn.maxStep.toFixed(4)} ` +
        `d=${forcedOn.distance.toFixed(2)} | forcedOff step=${forcedOff.maxStep.toFixed(3)}`,
    );
  },
);
