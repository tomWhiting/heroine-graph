/**
 * GPU tests for finding #2: FA2/LinLog disabled their native attraction
 * but kept the FA2-calibrated 1/d repulsion, so the aggregate radial
 * pressure drowned local structure and every graph inflated into a
 * structureless uniform disc.
 *
 * With the native attraction passes restored (FA2: F = w*d, LinLog:
 * F = w*log(1+d)) and per-node adaptive speed engaged, a graph with
 * planted community structure must come out spatially clustered:
 *
 * - mean intra-cluster distance clearly below mean inter-cluster distance
 *   (the uniform-disc failure scores ~1 because labels are uncorrelated
 *   with position), and
 * - grid-cell density dispersion far above the uniform-disc signature
 *   (~1, Poisson).
 *
 * Fixture scale and thresholds were calibrated against a faithful
 * emulation of the pre-fix pipeline (repulsion-only algorithm pass +
 * the racy non-atomic Hooke spring pass from commit ccb5182, no adaptive
 * speed) so that each assertion demonstrably FAILS on the pre-fix code:
 *
 * - FA2, 1000 nodes:    fixed ratio ~0.06 | pre-fix ~0.66-0.70 (threshold 0.3)
 * - LinLog, dense inter: fixed ratio ~0.38 | pre-fix ~0.67-0.78 (threshold 0.55)
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  createAlgorithmSimHarness,
  GPU_SKIP_MESSAGE,
  type HarnessForceAlgorithm,
  loadModuleInliningWgsl,
  probeAdapter,
} from "../helpers/gpu.ts";
import {
  clusterSeparation,
  countNonFinite,
  densityDispersionIndex,
} from "../helpers/invariants.ts";
import { generateClusteredGraph } from "../fixtures/clustered_graph.ts";
import { mulberry32 } from "../fixtures/prng.ts";

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

/** Density dispersion of a seeded uniform disc with the given node count —
 * the "no structure" signature the layout must clearly exceed. */
function uniformDiscDispersion(nodeCount: number, seed: number): number {
  const rng = mulberry32(seed);
  const xs = new Float32Array(nodeCount);
  const ys = new Float32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    const r = Math.sqrt(rng());
    const theta = 2 * Math.PI * rng();
    xs[i] = r * Math.cos(theta);
    ys[i] = r * Math.sin(theta);
  }
  return densityDispersionIndex(xs, ys);
}

async function runClusteredLayout(
  device: GPUDevice,
  algorithm: HarnessForceAlgorithm,
  ticks: number,
  alphaDecay?: number,
  fixture: { nodesPerCluster?: number; interEdgesPerPair?: number } = {},
): Promise<{ ratio: number; dispersion: number; uniformDispersion: number }> {
  const graph = generateClusteredGraph({ seed: 7, ...fixture });

  // Sanity: initial positions are uncorrelated with cluster labels — the
  // structure asserted below must be produced by the algorithm.
  const initial = clusterSeparation(graph.positionsX, graph.positionsY, graph.labels);
  assert(
    initial.ratio > 0.85,
    `fixture broken: initial positions already clustered (ratio=${initial.ratio})`,
  );

  // Default force config: this tests the production calibration, not a
  // hand-picked one.
  const harness = await createAlgorithmSimHarness(device, algorithm, graph, {}, alphaDecay);
  try {
    await harness.tick(ticks);
    const { x, y } = await harness.readPositions();
    assertEquals(countNonFinite(x, y), 0);

    const { intra, inter, ratio } = clusterSeparation(x, y, graph.labels);
    const dispersion = densityDispersionIndex(x, y);
    const uniformDispersion = uniformDiscDispersion(graph.nodeCount, 99);

    console.log(
      `[gpu] ${algorithm.info.id} clustered layout: intra=${intra.toFixed(1)} ` +
        `inter=${inter.toFixed(1)} ratio=${ratio.toFixed(3)} ` +
        `dispersion=${dispersion.toFixed(2)} uniform=${uniformDispersion.toFixed(2)}`,
    );
    return { ratio, dispersion, uniformDispersion };
  } finally {
    harness.dispose();
  }
}

gpuTest(
  "GPU ForceAtlas2: clustered graph develops spatial cluster structure (not a uniform disc)",
  async (device) => {
    const mod = await loadModuleInliningWgsl<
      { createForceAtlas2Algorithm(): HarnessForceAlgorithm }
    >(
      new URL(
        "../../packages/core/src/simulation/algorithms/force-atlas2.ts",
        import.meta.url,
      ),
    );
    // Harness default alpha schedule reaches rest in ~300 ticks.
    // 4 clusters x 250 = 1000 nodes: scale matters. At 160 nodes the
    // pre-fix pipeline still scored ~0.35 because the disc is small
    // enough for the sparse inter-cluster Hooke springs to matter; at
    // 1000 nodes its aggregate 1/d radial pressure dominates and it
    // scores ~0.66-0.70, while the fixed pipeline scores ~0.06.
    const { ratio, dispersion, uniformDispersion } = await runClusteredLayout(
      device,
      mod.createForceAtlas2Algorithm(),
      400,
      undefined,
      { nodesPerCluster: 250 },
    );

    // Uniform-disc failure mode scores ratio ~1, the pre-fix pipeline
    // ~0.66-0.70. Genuine cluster separation lands near 0.06; 0.3 leaves
    // headroom for noise while still failing the pre-fix code.
    assert(
      ratio < 0.3,
      `FA2 shows no cluster structure: intra/inter ratio ${ratio} (uniform disc ~1)`,
    );

    // Uniform disc has Poisson cell counts (dispersion ~1). Clusters with
    // empty space between them push it far higher.
    assert(
      dispersion > 3 * uniformDispersion,
      `FA2 density is uniform-disc-like: dispersion ${dispersion} vs uniform ${uniformDispersion}`,
    );
  },
);

gpuTest(
  "GPU LinLog: clustered graph develops spatial cluster structure (not a uniform disc)",
  async (device) => {
    const mod = await loadModuleInliningWgsl<
      { createLinLogAlgorithm(): HarnessForceAlgorithm }
    >(
      new URL(
        "../../packages/core/src/simulation/algorithms/linlog.ts",
        import.meta.url,
      ),
    );
    // LinLog's bounded log(1+d) attraction against 1/d repulsion settles at
    // a much larger spatial scale than FA2, so the layout needs a slower
    // cooling schedule to reach it (~600 ticks to rest instead of ~300) and
    // the fixture must stay small (convergence from the seeded disc slows
    // sharply with node count — ratio ~0.9 at 400 nodes on this schedule).
    //
    // Discrimination comes from density instead: 8 inter-cluster edges per
    // pair exercise LinLog's defining property (Noack: distances reflect
    // community DENSITY, not path length). The pre-fix Hooke springs pull
    // the well-connected clusters toward rest length and score ~0.67-0.78;
    // native log attraction still separates them at ~0.38.
    const { ratio, dispersion, uniformDispersion } = await runClusteredLayout(
      device,
      mod.createLinLogAlgorithm(),
      700,
      0.0114,
      { interEdgesPerPair: 8 },
    );

    // Pre-fix pipeline scores ~0.67-0.78 here; the fixed one ~0.38.
    assert(
      ratio < 0.55,
      `LinLog shows no cluster structure: intra/inter ratio ${ratio} (uniform disc ~1)`,
    );
    assert(
      dispersion > 3 * uniformDispersion,
      `LinLog density is uniform-disc-like: dispersion ${dispersion} vs uniform ${uniformDispersion}`,
    );
  },
);
