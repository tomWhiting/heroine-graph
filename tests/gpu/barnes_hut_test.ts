/**
 * GPU tests for the Barnes-Hut (Karras) repair (findings #0, #7, #8, #11,
 * #12, #15):
 *
 * - Gold standard: single-step Barnes-Hut repulsion forces on a ~5K-node
 *   code-tree fixture must statistically match the N² reference shader.
 *   This is end-to-end over the whole pipeline — Morton codes, the radix
 *   sort (digit-major layout + stable ranks), Karras topology (fixed split
 *   search), and the race-free multi-pass bottom-up mass aggregation. The
 *   old broken sort/aggregation produced garbage masses and wildly wrong
 *   forces here.
 * - Coincident nodes: exactly-overlapping distinct nodes must receive a
 *   deterministic golden-angle separation impulse instead of being silently
 *   skipped (old behavior: permanently fused).
 * - No collapse: a full simulation run must cool (kinetic energy decays)
 *   and keep a bounding radius comparable to the N² reference instead of
 *   collapsing to the center.
 * - Scale (task #8): above 131,072 particles the radix sort's prefix scan used
 *   to refuse outright, so recordRepulsionPass returned early and Barnes-Hut
 *   silently produced no forces at all. The 220K case below is the Meridian
 *   target scale and asserts both that the pipeline runs and that it still
 *   agrees with direct summation.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  createAlgorithmSimHarness,
  GPU_SKIP_MESSAGE,
  type HarnessForceAlgorithm,
  loadModuleInliningWgsl,
  probeAdapter,
  requestHarnessDevice,
} from "../helpers/gpu.ts";
import { countNonFinite, kineticEnergy, maxRadius } from "../helpers/invariants.ts";
import { CODE_TREE_SCALES, generateCodeTree } from "../fixtures/code_tree.ts";
import { mulberry32 } from "../fixtures/prng.ts";
import {
  type FullForceConfig,
  validateForceConfig,
} from "../../packages/core/src/simulation/config.ts";

const adapter = await probeAdapter();
if (!adapter) {
  console.warn(`[gpu] ${GPU_SKIP_MESSAGE}`);
}

// The traversal pass is Barnes-Hut's widest bind group at 8 storage buffers,
// which is the WebGPU default, so every conformant adapter clears this.
const REQUIRED_STORAGE_BUFFERS = 8;
const limitSupported = adapter !== null &&
  adapter.limits.maxStorageBuffersPerShaderStage >= REQUIRED_STORAGE_BUFFERS;
if (adapter && !limitSupported) {
  console.warn(
    `[gpu] adapter supports fewer than ${REQUIRED_STORAGE_BUFFERS} storage buffers per stage ` +
      "— Barnes-Hut tests skipped",
  );
}

function gpuTest(name: string, fn: (device: GPUDevice) => Promise<void>): void {
  Deno.test({
    name,
    ignore: !limitSupported,
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

function loadBarnesHut(): Promise<HarnessForceAlgorithm> {
  return loadModuleInliningWgsl<{ createBarnesHutAlgorithm(): HarnessForceAlgorithm }>(
    new URL(
      "../../packages/core/src/simulation/algorithms/barnes-hut.ts",
      import.meta.url,
    ),
  ).then((m) => m.createBarnesHutAlgorithm());
}

function loadN2(): Promise<HarnessForceAlgorithm> {
  return loadModuleInliningWgsl<{ createN2Algorithm(): HarnessForceAlgorithm }>(
    new URL("../../packages/core/src/simulation/algorithms/n2.ts", import.meta.url),
  ).then((m) => m.createN2Algorithm());
}

function computeBounds(xs: Float32Array, ys: Float32Array) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] < minX) minX = xs[i];
    if (xs[i] > maxX) maxX = xs[i];
    if (ys[i] < minY) minY = ys[i];
    if (ys[i] > maxY) maxY = ys[i];
  }
  return { minX, minY, maxX, maxY };
}

/**
 * Runs exactly one repulsion pass of the given algorithm over static
 * positions and reads the resulting per-node force vectors back.
 */
async function runRepulsionOnce(
  device: GPUDevice,
  algorithm: HarnessForceAlgorithm,
  xs: Float32Array,
  ys: Float32Array,
  forceConfig: FullForceConfig,
): Promise<Float32Array> {
  const n = xs.length;

  const positionData = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    positionData[2 * i] = xs[i];
    positionData[2 * i + 1] = ys[i];
  }

  const positions = device.createBuffer({
    size: n * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(positions, 0, positionData);

  // Zero-initialized per WebGPU spec; repulsion shaders accumulate (+=)
  const forces = device.createBuffer({
    size: n * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const context = {
    device,
    positions,
    forces,
    nodeCount: n,
    edgeCount: 0,
    forceConfig,
    bounds: computeBounds(xs, ys),
  };

  const pipelines = algorithm.createPipelines({ device });
  const buffers = algorithm.createBuffers(device, n);
  const bindGroups = algorithm.createBindGroups(device, pipelines, context, buffers);
  algorithm.updateUniforms(device, buffers, context);

  const encoder = device.createCommandEncoder();
  algorithm.recordRepulsionPass(encoder, pipelines, bindGroups, n);
  const staging = device.createBuffer({
    size: n * 8,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  encoder.copyBufferToBuffer(forces, 0, staging, 0, n * 8);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();

  staging.destroy();
  buffers.destroy();
  positions.destroy();
  forces.destroy();
  return result;
}

/**
 * Compares Barnes-Hut against the N² reference on a code-tree fixture of
 * the given size and asserts theta-appropriate statistical agreement.
 */
async function assertMatchesN2(device: GPUDevice, maxNodes: number): Promise<void> {
  const graph = generateCodeTree({ seed: 42, maxNodes, maxDepth: 7, maxChildren: 8 });
  const n = graph.nodeCount;

  const forceConfig = validateForceConfig({
    repulsionStrength: -50,
    theta: 0.5,
    repulsionDistanceMin: 1,
    repulsionDistanceMax: 0, // no cutoff: BH applies it per-cell, N² per-pair
  });

  const bh = await loadBarnesHut();
  const n2 = await loadN2();
  const bhForces = await runRepulsionOnce(
    device,
    bh,
    graph.positionsX,
    graph.positionsY,
    forceConfig,
  );
  const n2Forces = await runRepulsionOnce(
    device,
    n2,
    graph.positionsX,
    graph.positionsY,
    forceConfig,
  );

  for (const f of [bhForces, n2Forces]) {
    for (let i = 0; i < f.length; i++) {
      assert(Number.isFinite(f[i]), `non-finite force component at ${i}`);
    }
  }

  // Mean reference magnitude, for the relative-error denominator floor
  // (nodes near the field null have |F| ~ 0 and unbounded relative error)
  let magSumN2 = 0;
  let magSumBh = 0;
  for (let i = 0; i < n; i++) {
    magSumN2 += Math.hypot(n2Forces[2 * i], n2Forces[2 * i + 1]);
    magSumBh += Math.hypot(bhForces[2 * i], bhForces[2 * i + 1]);
  }
  const meanMag = magSumN2 / n;
  assert(meanMag > 0, "N² reference produced zero forces — broken rig");

  const relErrors = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const dx = bhForces[2 * i] - n2Forces[2 * i];
    const dy = bhForces[2 * i + 1] - n2Forces[2 * i + 1];
    const ref = Math.max(
      Math.hypot(n2Forces[2 * i], n2Forces[2 * i + 1]),
      0.05 * meanMag,
    );
    relErrors[i] = Math.hypot(dx, dy) / ref;
  }
  const sorted = relErrors.slice().sort();
  const median = sorted[Math.floor(n / 2)];
  const p95 = sorted[Math.floor(n * 0.95)];
  const magRatio = magSumBh / magSumN2;

  console.log(
    `[gpu] BH vs N² (n=${n}): median relErr=${median.toFixed(4)} ` +
      `p95=${p95.toFixed(4)} magRatio=${magRatio.toFixed(4)}`,
  );

  // theta = 0.5 keeps the multipole approximation within a few percent;
  // the broken sort/aggregation produced errors of ~1 (missing subtrees)
  // and magnitude ratios far from 1.
  assert(median < 0.05, `median relative error too high: ${median}`);
  assert(p95 < 0.15, `p95 relative error too high: ${p95}`);
  assert(
    magRatio > 0.9 && magRatio < 1.1,
    `aggregate force magnitude off: BH/N² = ${magRatio}`,
  );
}

gpuTest(
  "GPU Barnes-Hut: forces match N² reference on 5K-node tree (full radix path)",
  async (device) => {
    await assertMatchesN2(device, 5000);
  },
);

gpuTest(
  "GPU Barnes-Hut: forces match N² reference on 800-node tree (simple-sort path)",
  async (device) => {
    await assertMatchesN2(device, 800);
  },
);

gpuTest(
  "GPU Barnes-Hut: exactly-coincident nodes receive a deterministic separation impulse",
  async (device) => {
    // Nodes 0 and 1 are exactly stacked; 2 and 3 are far away. The old
    // traversal skipped everything within 0.01 units, so stacked pairs
    // received zero repulsion forever.
    const xs = new Float32Array([5, 5, 200, -200]);
    const ys = new Float32Array([5, 5, 0, 0]);
    const forceConfig = validateForceConfig({
      repulsionStrength: -50,
      theta: 0.5,
      repulsionDistanceMin: 1,
      repulsionDistanceMax: 0,
    });

    const bh = await loadBarnesHut();
    const forces = await runRepulsionOnce(device, bh, xs, ys, forceConfig);
    assertEquals(countNonFinite(forces, new Float32Array(forces.length)), 0);

    // Each of the stacked pair sees one extra unit of coincident mass:
    // impulse magnitude = k * m / min_distance^2 = 50
    const mag0 = Math.hypot(forces[0], forces[1]);
    const mag1 = Math.hypot(forces[2], forces[3]);
    assert(mag0 > 40, `coincident node 0 got no separation impulse: |F|=${mag0}`);
    assert(mag1 > 40, `coincident node 1 got no separation impulse: |F|=${mag1}`);

    // Golden-angle directions are per-index, so the pair pushes apart in
    // clearly different directions (not the same vector)
    const dot = (forces[0] * forces[2] + forces[1] * forces[3]) / (mag0 * mag1);
    assert(dot < 0.9, `coincident pair pushed in the same direction: dot=${dot}`);

    // A far-away isolated node must NOT receive a spurious impulse — only
    // genuine (~k*M/d^2) long-range repulsion
    const mag2 = Math.hypot(forces[4], forces[5]);
    assert(mag2 < 1, `isolated node got a spurious impulse: |F|=${mag2}`);
  },
);

gpuTest(
  "GPU Barnes-Hut: layout cools and does not collapse to center",
  async (device) => {
    const graph = generateCodeTree({ seed: 5, maxNodes: 1200, maxDepth: 5 });
    const config = { repulsionStrength: -50, theta: 0.8 };

    const runLayout = async (algorithm: HarnessForceAlgorithm, bounds: boolean) => {
      const harness = await createAlgorithmSimHarness(
        device,
        algorithm,
        graph,
        config,
        undefined,
        bounds ? { boundsSyncInterval: 10 } : {},
      );
      try {
        await harness.tick(5);
        const early0 = await harness.readPositions();
        await harness.tick(1);
        const early1 = await harness.readPositions();
        await harness.tick(193);
        const late0 = await harness.readPositions();
        await harness.tick(1);
        const late1 = await harness.readPositions();
        return { early0, early1, late0, late1 };
      } finally {
        harness.dispose();
      }
    };

    const bhRun = await runLayout(await loadBarnesHut(), true);
    const n2Run = await runLayout(await loadN2(), false);

    assertEquals(countNonFinite(bhRun.late1.x, bhRun.late1.y), 0);

    // Kinetic energy decays as the simulation cools
    const keEarly = kineticEnergy(
      bhRun.early0.x,
      bhRun.early0.y,
      bhRun.early1.x,
      bhRun.early1.y,
    );
    const keLate = kineticEnergy(
      bhRun.late0.x,
      bhRun.late0.y,
      bhRun.late1.x,
      bhRun.late1.y,
    );
    assert(
      keLate < 0.05 * keEarly,
      `kinetic energy failed to decay: early=${keEarly}, late=${keLate}`,
    );

    // Bounding radius stays comparable to the N² reference layout instead
    // of collapsing to the center (the reported Barnes-Hut symptom)
    const rBh = maxRadius(bhRun.late1.x, bhRun.late1.y);
    const rN2 = maxRadius(n2Run.late1.x, n2Run.late1.y);
    const rInitial = maxRadius(graph.positionsX, graph.positionsY);

    console.log(
      `[gpu] BH collapse check: rBH=${rBh.toFixed(1)} rN2=${rN2.toFixed(1)} ` +
        `rInitial=${rInitial.toFixed(1)} keEarly=${keEarly.toFixed(1)} keLate=${keLate.toFixed(3)}`,
    );

    assert(
      rBh > 0.5 * rN2 && rBh < 2.5 * rN2,
      `BH bounding radius diverged from N² reference: BH=${rBh}, N²=${rN2}`,
    );
    assert(
      rBh > 0.2 * rInitial,
      `layout collapsed toward center: final=${rBh}, initial=${rInitial}`,
    );
  },
);

/**
 * Direct-summation repulsion on ONE particle, replicating
 * `compute_repulsion` in barnes_hut_binary.comp.wgsl term for term (unit
 * masses, `max_distance` disabled). O(N) per call, which is why the 220K test
 * references a sample of particles rather than all of them.
 *
 * Returns the force and the mass found at (essentially) this particle's own
 * position — the shader's `coincident_mass`, which the caller checks is just
 * the particle's own leaf, since the golden-angle separation impulse for
 * genuinely stacked bodies is not modelled here.
 */
function referenceRepulsion(
  xs: Float32Array,
  ys: Float32Array,
  i: number,
  strength: number,
  minDistance: number,
): { fx: number; fy: number; coincidentMass: number } {
  const minDistSq = minDistance * minDistance;
  const xi = xs[i];
  const yi = ys[i];
  let fx = 0;
  let fy = 0;
  let coincidentMass = 0;
  for (let j = 0; j < xs.length; j++) {
    const dx = xi - xs[j];
    const dy = yi - ys[j];
    const distSq = dx * dx + dy * dy;
    // COINCIDENT_DIST_SQ; the particle's own leaf always lands here
    if (distSq < 0.0001) {
      coincidentMass += 1;
      continue;
    }
    const safeSq = Math.max(distSq, minDistSq);
    const scale = strength / (safeSq * Math.sqrt(safeSq));
    fx += dx * scale;
    fy += dy * scale;
  }
  return { fx, fy, coincidentMass };
}

gpuTest(
  "GPU Barnes-Hut: 220K particles (above the old 131,072 cap) run and stay accurate",
  async (device) => {
    // CODE_TREE_SCALES.large is the Meridian target scale: 860 sort workgroups,
    // so the radix sort takes the hierarchical phase-2 path. Before task #8
    // recordRadixSort refused here and recordRepulsionPass returned before
    // touching the force buffer — which is exactly what the non-zero assertion
    // below catches, since a refusal leaves every force at zero and would
    // otherwise sail past a finiteness check.
    const graph = generateCodeTree(CODE_TREE_SCALES.large);
    const n = graph.nodeCount;
    assert(n > 131072, `fixture must exceed the old cap, got ${n}`);

    const repulsionStrength = -50;
    const repulsionDistanceMin = 1;
    const forceConfig = validateForceConfig({
      repulsionStrength,
      theta: 0.5,
      repulsionDistanceMin,
      repulsionDistanceMax: 0,
    });

    const bh = await loadBarnesHut();
    const forces = await runRepulsionOnce(
      device,
      bh,
      graph.positionsX,
      graph.positionsY,
      forceConfig,
    );

    assertEquals(
      countNonFinite(forces, new Float32Array(forces.length)),
      0,
      "Barnes-Hut produced non-finite forces at 220K particles",
    );

    // Every particle in a 220K repulsive cloud is pushed by something. An
    // all-zero buffer is the signature of a pipeline that refused to run.
    let moved = 0;
    for (let i = 0; i < n; i++) {
      if (forces[2 * i] !== 0 || forces[2 * i + 1] !== 0) moved++;
    }
    assert(
      moved === n,
      `${n - moved} of ${n} particles received no force — the tree pass did not run`,
    );

    // Direct summation on a seeded sample. The traversal reads the whole tree,
    // so a sample of particles still exercises every level of it.
    const sampleRng = mulberry32(88);
    const sampleCount = 192;
    const relErrors = new Float64Array(sampleCount);
    let magSumRef = 0;
    let magSumBh = 0;
    const samples = new Uint32Array(sampleCount);
    for (let s = 0; s < sampleCount; s++) {
      samples[s] = Math.floor(sampleRng() * n);
    }

    const refFx = new Float64Array(sampleCount);
    const refFy = new Float64Array(sampleCount);
    for (let s = 0; s < sampleCount; s++) {
      const i = samples[s];
      const ref = referenceRepulsion(
        graph.positionsX,
        graph.positionsY,
        i,
        Math.abs(repulsionStrength),
        repulsionDistanceMin,
      );
      assertEquals(
        ref.coincidentMass,
        1,
        `sampled particle ${i} has a stacked neighbour; the reference does not ` +
          `model the golden-angle separation impulse`,
      );
      refFx[s] = ref.fx;
      refFy[s] = ref.fy;
      magSumRef += Math.hypot(ref.fx, ref.fy);
      magSumBh += Math.hypot(forces[2 * i], forces[2 * i + 1]);
    }
    assert(magSumRef > 0, "direct summation produced zero forces — broken rig");
    const meanMag = magSumRef / sampleCount;

    for (let s = 0; s < sampleCount; s++) {
      const i = samples[s];
      const dx = forces[2 * i] - refFx[s];
      const dy = forces[2 * i + 1] - refFy[s];
      // Same denominator floor as the 5K comparison: particles near the field
      // null have |F| ~ 0 and unbounded relative error.
      const denom = Math.max(Math.hypot(refFx[s], refFy[s]), 0.05 * meanMag);
      relErrors[s] = Math.hypot(dx, dy) / denom;
    }
    const sorted = relErrors.slice().sort();
    const median = sorted[Math.floor(sampleCount / 2)];
    const p95 = sorted[Math.floor(sampleCount * 0.95)];
    const magRatio = magSumBh / magSumRef;

    console.log(
      `[gpu] BH vs direct summation (n=${n}, ${sampleCount} samples): ` +
        `median relErr=${median.toFixed(4)} p95=${p95.toFixed(4)} ` +
        `magRatio=${magRatio.toFixed(4)}`,
    );

    // theta = 0.5 tolerances, as in the 5K comparison. A dropped block offset
    // in the hierarchical scan mis-sorts whole Morton runs, which scrambles
    // the tree and drives these far past 1.
    assert(median < 0.05, `median relative error too high: ${median}`);
    assert(p95 < 0.15, `p95 relative error too high: ${p95}`);
    assert(
      magRatio > 0.9 && magRatio < 1.1,
      `aggregate force magnitude off: BH/direct = ${magRatio}`,
    );
  },
);

Deno.test({
  name: "GPU Barnes-Hut: a default-limit device builds every pipeline",
  ignore: adapter === null,
  sanitizeResources: false,
  sanitizeOps: false,
  async fn() {
    // The failure this guards is silent: a bind group layout over more storage
    // buffers than the device supplies is INVALID, binding it poisons the
    // compute pass, the pass poisons the encoder, and submit() drops the whole
    // frame — springs and integration with it. The simulation then looks inert
    // rather than broken, which is what made Barnes-Hut appear to produce zero
    // motion in the parity harness, whose device had the default limit.
    //
    // The flat tree_links / node_attrs layouts brought the widest pass down to
    // exactly the default of 8, so the correct assertion is now that a
    // default-limit device is served rather than refused. `assertAlgorithmSup-
    // portedOnDevice`'s refusal path is covered without a GPU in
    // tests/unit/algorithm_device_limits_test.ts, which is the only way to
    // reach it: wgpu clamps a below-default requiredLimits request back up to
    // the default, so no real device reports fewer than 8.
    const device = await adapter!.requestDevice({
      requiredLimits: { maxStorageBuffersPerShaderStage: 8 },
    });
    try {
      assertEquals(device.limits.maxStorageBuffersPerShaderStage, 8);
      const bh = await loadBarnesHut();
      const pipelines = bh.createPipelines({ device });
      assert(pipelines !== undefined, "every Barnes-Hut pipeline must build at the default limit");
    } finally {
      device.destroy();
    }
  },
});
