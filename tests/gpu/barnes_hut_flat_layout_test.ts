/**
 * GPU tests for Barnes-Hut's flat multi-region tree buffers.
 *
 * `tree_links` holds left_child, right_child and parent as three concatenated
 * regions and `node_attrs` holds node_mass and node_size as two, so that the
 * traversal pass binds 8 storage buffers rather than 10 and Barnes-Hut is
 * selectable on a default-limit adapter. Both region maps are strided by the
 * ALLOCATED particle capacity, published to the shaders as the single
 * `node_capacity` uniform word (see treeLinksElements / nodeAttrsElements in
 * algorithms/barnes-hut.ts).
 *
 * A base computed from the wrong quantity — the live node count, a hard-coded
 * stride, the wrong region — is invisible whenever capacity happens to equal
 * the node count, which is what every other Barnes-Hut test runs at. These
 * tests attack exactly that:
 *
 * - Region shift: the same fixture, run at three different allocated
 *   capacities, must produce BIT-IDENTICAL forces. Every region base moves
 *   with the capacity, so a base derived from anything else lands in the wrong
 *   region for at least one of them.
 * - Root mass: a probe far from a tight cluster of K bodies must feel exactly
 *   k*K/d², i.e. the aggregated subtree mass equals the sum of the particle
 *   masses. A mass region reading into the size region (or into left_child's
 *   bit patterns) does not sum to K.
 * - Symmetry: a fixture symmetric under reflection in both axes must produce
 *   forces symmetric under the same reflections. Region bases are index
 *   arithmetic, and getting one wrong makes a node's attributes depend on its
 *   position in the buffer rather than on its geometry.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  GPU_SKIP_MESSAGE,
  type HarnessForceAlgorithm,
  loadModuleInliningWgsl,
  probeAdapter,
  requestHarnessDevice,
} from "../helpers/gpu.ts";
import {
  type FullForceConfig,
  validateForceConfig,
} from "../../packages/core/src/simulation/config.ts";
import { generateCodeTree } from "../fixtures/code_tree.ts";

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

function loadBarnesHut(): Promise<HarnessForceAlgorithm> {
  return loadModuleInliningWgsl<{ createBarnesHutAlgorithm(): HarnessForceAlgorithm }>(
    new URL(
      "../../packages/core/src/simulation/algorithms/barnes-hut.ts",
      import.meta.url,
    ),
  ).then((m) => m.createBarnesHutAlgorithm());
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
 * One repulsion pass over static positions, with the buffer set allocated for
 * `capacity` particles rather than for exactly `xs.length`.
 *
 * The capacity is the only thing that varies between the runs the region-shift
 * test compares: dispatch widths, tree topology, aggregation pass count and
 * traversal order all follow from the node count, which is held fixed.
 */
async function runRepulsionAtCapacity(
  device: GPUDevice,
  algorithm: HarnessForceAlgorithm,
  xs: Float32Array,
  ys: Float32Array,
  forceConfig: FullForceConfig,
  capacity: number,
): Promise<Float32Array> {
  const n = xs.length;
  assert(capacity >= n, "capacity must cover the fixture");

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
  const buffers = algorithm.createBuffers(device, capacity);
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

const REGION_SHIFT_CONFIG = validateForceConfig({
  repulsionStrength: -50,
  theta: 0.5,
  repulsionDistanceMin: 1,
  repulsionDistanceMax: 0,
});

/**
 * Asserts that allocating the buffer set larger leaves the physics untouched.
 *
 * Bit-identical rather than approximately equal: nothing about the computation
 * changes with capacity, so any difference at all is a region base that moved
 * when it should not have (or one that failed to move when it should have).
 */
async function assertCapacityInvariant(
  device: GPUDevice,
  xs: Float32Array,
  ys: Float32Array,
  capacities: readonly number[],
): Promise<void> {
  const n = xs.length;
  const bh = await loadBarnesHut();

  const baseline = await runRepulsionAtCapacity(
    device,
    bh,
    xs,
    ys,
    REGION_SHIFT_CONFIG,
    n,
  );
  let nonZero = 0;
  for (let i = 0; i < baseline.length; i++) {
    if (baseline[i] !== 0) nonZero++;
  }
  assert(nonZero > 0, "the baseline run produced no forces — broken rig");

  for (const capacity of capacities) {
    const shifted = await runRepulsionAtCapacity(
      device,
      bh,
      xs,
      ys,
      REGION_SHIFT_CONFIG,
      capacity,
    );
    for (let i = 0; i < baseline.length; i++) {
      if (shifted[i] !== baseline[i]) {
        throw new Error(
          `force component ${i} changed with the allocated capacity ` +
            `(n=${n}, capacity=${capacity}): ${baseline[i]} -> ${shifted[i]}. ` +
            "A region base is derived from something other than node_capacity.",
        );
      }
    }
  }
}

gpuTest(
  "GPU Barnes-Hut flat layout: forces are identical at any allocated capacity (simple-sort path)",
  async (device) => {
    const graph = generateCodeTree({ seed: 42, maxNodes: 800, maxDepth: 6, maxChildren: 8 });
    // 4 is the floor createBuffers clamps to; the rest cross the internal-node
    // and tree-node stride boundaries by non-multiples so that a base off by a
    // whole region, and one off by a scaled region, both land somewhere wrong.
    await assertCapacityInvariant(
      device,
      graph.positionsX,
      graph.positionsY,
      [graph.nodeCount + 1, graph.nodeCount + 337, 4 * graph.nodeCount],
    );
  },
);

gpuTest(
  "GPU Barnes-Hut flat layout: forces are identical at any allocated capacity (full radix path)",
  async (device) => {
    // Above 1024 nodes the 8-pass radix sort runs, so the tree is built from
    // the keysA/valuesA bind group rather than keysB/valuesB — a second set of
    // bind-group entries over the same merged buffers.
    const graph = generateCodeTree({ seed: 7, maxNodes: 5000, maxDepth: 7, maxChildren: 8 });
    await assertCapacityInvariant(
      device,
      graph.positionsX,
      graph.positionsY,
      [graph.nodeCount + 1, 3 * graph.nodeCount],
    );
  },
);

gpuTest(
  "GPU Barnes-Hut flat layout: an aggregated subtree carries the sum of its particle masses",
  async (device) => {
    // A tight cluster of CLUSTER unit-mass bodies at the origin and one probe
    // far to the right. Every cluster body sits inside COINCIDENT_DIST_SQ of
    // the others, so the cluster's centre of mass is the origin exactly and
    // the probe's force is k*M/d² for M = the aggregated cluster mass — no
    // matter which level of the tree the theta criterion opens at.
    const cluster = 64;
    const distance = 1000;
    const n = cluster + 1;
    const xs = new Float32Array(n);
    const ys = new Float32Array(n);
    xs[cluster] = distance;

    const strength = 50;
    const forces = await runRepulsionAtCapacity(
      device,
      await loadBarnesHut(),
      xs,
      ys,
      validateForceConfig({
        repulsionStrength: -strength,
        theta: 0.5,
        repulsionDistanceMin: 1,
        repulsionDistanceMax: 0,
      }),
      // Deliberately over-allocated: at capacity === n the mass region base is
      // indistinguishable from several wrong ones.
      4 * n,
    );

    const fx = forces[2 * cluster];
    const fy = forces[2 * cluster + 1];
    const expected = strength * cluster / (distance * distance);
    assertEquals(fy, 0, "a probe on the x axis must feel no y force");
    assert(
      Math.abs(fx - expected) < 1e-4 * expected,
      `probe force implies an aggregate cluster mass of ${fx * distance * distance / strength} ` +
        `bodies, not ${cluster}: |F|=${fx}, expected ${expected}`,
    );
  },
);

gpuTest(
  "GPU Barnes-Hut flat layout: a fixture symmetric in both axes yields symmetric forces",
  async (device) => {
    // Four bodies at (±a, ±b) with a != b, so reflection in x and reflection
    // in y are distinct symmetries of the configuration. The repulsion field
    // inherits both: node 0 at (-a, -b) must feel exactly the negation of
    // node 3 at (a, b), and node 1 at (a, -b) the negation of node 2.
    const a = 120;
    const b = 70;
    const xs = new Float32Array([-a, a, -a, a]);
    const ys = new Float32Array([-b, -b, b, b]);

    const forces = await runRepulsionAtCapacity(
      device,
      await loadBarnesHut(),
      xs,
      ys,
      validateForceConfig({
        repulsionStrength: -50,
        theta: 0.5,
        repulsionDistanceMin: 1,
        repulsionDistanceMax: 0,
      }),
      512,
    );

    for (let i = 0; i < 4; i++) {
      assert(
        Number.isFinite(forces[2 * i]) && Number.isFinite(forces[2 * i + 1]),
        `non-finite force on node ${i}`,
      );
    }

    // Every body sits at the same distances from the same neighbours, so all
    // four magnitudes agree; f32 differences between the summation orders the
    // traversal reaches them in are the only slack allowed.
    const magnitudes = [0, 1, 2, 3].map((i) => Math.hypot(forces[2 * i], forces[2 * i + 1]));
    assert(magnitudes[0] > 0, "the fixture produced no forces — broken rig");
    for (const m of magnitudes) {
      assert(
        Math.abs(m - magnitudes[0]) < 1e-4 * magnitudes[0],
        `force magnitudes are not symmetric: ${magnitudes.join(", ")}`,
      );
    }

    // ...and each body is pushed directly away from the centre.
    for (let i = 0; i < 4; i++) {
      assert(
        Math.sign(forces[2 * i]) === Math.sign(xs[i]) &&
          Math.sign(forces[2 * i + 1]) === Math.sign(ys[i]),
        `node ${i} is not pushed outward: F=(${forces[2 * i]}, ${forces[2 * i + 1]})`,
      );
    }
    for (const [i, j] of [[0, 3], [1, 2]]) {
      assert(
        Math.abs(forces[2 * i] + forces[2 * j]) < 1e-4 * magnitudes[0] &&
          Math.abs(forces[2 * i + 1] + forces[2 * j + 1]) < 1e-4 * magnitudes[0],
        `nodes ${i} and ${j} are reflections but their forces are not: ` +
          `(${forces[2 * i]}, ${forces[2 * i + 1]}) vs (${forces[2 * j]}, ${forces[2 * j + 1]})`,
      );
    }
  },
);
