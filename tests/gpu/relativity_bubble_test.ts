/**
 * GPU tests for Relativity Atlas bubble mode: the nested-bubble separation
 * invariant and the three force-law corrections it rests on.
 *
 * Bubble mode is off by default and, until this file existed, no GPU test set
 * `bubble_mode = 1` at all — tests/gpu/relativity_test.ts explicitly writes 0.
 * Everything asserted here was therefore unexercised code.
 *
 * The invariant under test has three clauses:
 *
 *   containment — a node's well stays inside its parent's;
 *   siblings    — wells of nodes sharing a parent do not overlap;
 *   roots       — wells of forest roots do not overlap.
 *
 * Together they imply that two nodes with no common ancestor are separated,
 * with no spatial structure anywhere: every term is a continuous function of
 * pairwise distance and two radii, which is what distinguishes this from the
 * density grid it is meant to replace.
 */

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import {
  createAlgorithmSimHarness,
  GPU_SKIP_MESSAGE,
  type HarnessForceAlgorithm,
  loadModuleInliningWgsl,
  probeAdapter,
  requestHarnessDevice,
} from "../helpers/gpu.ts";
import { countNonFinite, kineticEnergy } from "../helpers/invariants.ts";
import {
  collectForestRoots,
  deriveHierarchy,
  HIERARCHY_ROOT,
  hierarchyCsrPair,
  type HierarchyDeriver,
  type RetainedHierarchy,
} from "../../packages/core/src/graph/hierarchy.ts";
import { createHeadlessGraph } from "../helpers/headless_graph.ts";
import type { GraphTypedInput } from "../../packages/core/src/types.ts";
import { createWasmEngine } from "../../packages/core/src/wasm/loader.ts";
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
      const device = await requestHarnessDevice(adapter!);
      try {
        await fn(device);
      } finally {
        device.destroy();
      }
    },
  });
}

const SIBLING_WGSL = await Deno.readTextFile(
  new URL(
    "../../packages/core/src/simulation/shaders/relativity_sibling.comp.wgsl",
    import.meta.url,
  ),
);
const BUBBLE_ROOTS_WGSL = await Deno.readTextFile(
  new URL(
    "../../packages/core/src/simulation/shaders/relativity_bubble_roots.comp.wgsl",
    import.meta.url,
  ),
);

function createStorageBuffer(
  device: GPUDevice,
  data: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>,
  extraUsage = 0,
): GPUBuffer {
  const buffer = device.createBuffer({
    size: Math.max(data.byteLength, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
  });
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

/** Run one compute dispatch and read back the vec2 forces buffer. */
async function dispatchAndReadForces(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  entries: GPUBindGroupEntry[],
  forcesBuffer: GPUBuffer,
  nodeCount: number,
  threads: number,
): Promise<Float32Array> {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });
  const readback = device.createBuffer({
    size: nodeCount * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(threads / 256));
  pass.end();
  encoder.copyBufferToBuffer(forcesBuffer, 0, readback, 0, nodeCount * 8);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const forces = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return forces;
}

// ===========================================================================
// Sibling-pass fixtures
// ===========================================================================

/** The 16 fields of SiblingUniforms, in declaration order. */
interface SiblingKnobs {
  repulsionStrength: number;
  minDistance: number;
  maxSiblings: number;
  parentChildMultiplier: number;
  cousinEnabled: number;
  cousinStrength: number;
  phantomEnabled: number;
  phantomMultiplier: number;
  orbitStrength: number;
  tangentialMultiplier: number;
  orbitRadiusBase: number;
  bubbleMode: number;
  orbitScale: number;
}

/**
 * Knobs that leave ONLY the terms under test acting: no cousins, no tangential
 * amplification, no parent→child repulsion. Each case re-enables what it needs.
 */
const QUIET_KNOBS: SiblingKnobs = {
  repulsionStrength: 50,
  minDistance: 1,
  maxSiblings: 100,
  parentChildMultiplier: 0,
  cousinEnabled: 0,
  cousinStrength: 0,
  phantomEnabled: 1,
  phantomMultiplier: 0.5,
  orbitStrength: 1,
  tangentialMultiplier: 1,
  orbitRadiusBase: 25,
  bubbleMode: 1,
  orbitScale: 0.6,
};

interface SiblingCase {
  /** Containment parent per slot, or -1 for a root. Both CSRs derive from it. */
  parent: readonly number[];
  positions: Float32Array<ArrayBuffer>;
  wellRadius: Float32Array<ArrayBuffer>;
  masses?: Float32Array<ArrayBuffer>;
  knobs?: Partial<SiblingKnobs>;
}

/**
 * Dispatch relativity_sibling.comp.wgsl over a tree given as a parent array.
 *
 * Both CSR directions come from the one parent array, so the fixture cannot
 * describe a forward tree and an inverse tree that disagree.
 */
async function runSibling(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  testCase: SiblingCase,
): Promise<Float32Array> {
  const { parent, positions, wellRadius } = testCase;
  const nodeCount = parent.length;
  const knobs = { ...QUIET_KNOBS, ...testCase.knobs };

  const childCounts = new Uint32Array(nodeCount);
  let edgeCount = 0;
  for (const p of parent) {
    if (p >= 0) {
      childCounts[p]++;
      edgeCount++;
    }
  }

  const csrOffsets = new Uint32Array(nodeCount + 1);
  for (let i = 0; i < nodeCount; i++) csrOffsets[i + 1] = csrOffsets[i] + childCounts[i];
  const csrTargets = new Uint32Array(edgeCount);
  const cursor = csrOffsets.slice(0, nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    if (parent[i] >= 0) csrTargets[cursor[parent[i]]++] = i;
  }

  // Inverse CSR: one row per slot, length 0 for a root and 1 otherwise. The
  // offset row and the source row share one buffer; see csrInverseSourcesBase
  // in simulation/algorithms/relativity-atlas.ts.
  const invSourcesBase = nodeCount + 1;
  const csrInverse = new Uint32Array(invSourcesBase + edgeCount);
  let written = 0;
  for (let i = 0; i < nodeCount; i++) {
    csrInverse[i + 1] = csrInverse[i] + (parent[i] >= 0 ? 1 : 0);
    if (parent[i] >= 0) csrInverse[invSourcesBase + written++] = parent[i];
  }

  const uniformData = new ArrayBuffer(64);
  const view = new DataView(uniformData);
  view.setUint32(0, nodeCount, true);
  view.setUint32(4, edgeCount, true);
  view.setFloat32(8, knobs.repulsionStrength, true);
  view.setFloat32(12, knobs.minDistance, true);
  view.setUint32(16, knobs.maxSiblings, true);
  view.setFloat32(20, knobs.parentChildMultiplier, true);
  view.setUint32(24, knobs.cousinEnabled, true);
  view.setFloat32(28, knobs.cousinStrength, true);
  view.setUint32(32, knobs.phantomEnabled, true);
  view.setFloat32(36, knobs.phantomMultiplier, true);
  view.setFloat32(40, knobs.orbitStrength, true);
  view.setFloat32(44, knobs.tangentialMultiplier, true);
  view.setFloat32(48, knobs.orbitRadiusBase, true);
  view.setUint32(52, knobs.bubbleMode, true);
  view.setFloat32(56, knobs.orbitScale, true);
  view.setUint32(60, invSourcesBase, true);
  const uniforms = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniforms, 0, uniformData);

  const posBuf = createStorageBuffer(device, positions);
  const forceBuf = createStorageBuffer(
    device,
    new Float32Array(nodeCount * 2),
    GPUBufferUsage.COPY_SRC,
  );
  const invBuf = createStorageBuffer(device, csrInverse);
  const offBuf = createStorageBuffer(device, csrOffsets);
  const tgtBuf = createStorageBuffer(device, csrTargets);
  const massBuf = createStorageBuffer(
    device,
    testCase.masses ?? new Float32Array(nodeCount).fill(1),
  );
  const wellBuf = createStorageBuffer(device, wellRadius);
  const flagBuf = createStorageBuffer(device, new Uint32Array(nodeCount));

  try {
    return await dispatchAndReadForces(
      device,
      pipeline,
      [
        { binding: 0, resource: { buffer: uniforms } },
        { binding: 1, resource: { buffer: posBuf } },
        { binding: 2, resource: { buffer: forceBuf } },
        { binding: 3, resource: { buffer: invBuf } },
        { binding: 4, resource: { buffer: offBuf } },
        { binding: 5, resource: { buffer: tgtBuf } },
        { binding: 6, resource: { buffer: massBuf } },
        { binding: 7, resource: { buffer: wellBuf } },
        { binding: 8, resource: { buffer: flagBuf } },
      ],
      forceBuf,
      nodeCount,
      nodeCount,
    );
  } finally {
    for (
      const b of [uniforms, posBuf, forceBuf, invBuf, offBuf, tgtBuf, massBuf, wellBuf, flagBuf]
    ) {
      b.destroy();
    }
  }
}

function siblingPipeline(device: GPUDevice): GPUComputePipeline {
  return device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: SIBLING_WGSL }), entryPoint: "main" },
  });
}

// ===========================================================================
// Containment barrier
// ===========================================================================

gpuTest(
  "GPU RA bubble: containment barrier pulls in a child whose well overhangs its parent's",
  async (device) => {
    const pipeline = siblingPipeline(device);

    // Parent well 100 at the origin, child well 50 at distance 55: the child's
    // bubble reaches 105, five units past its parent's. The fixture is chosen so
    // the OLD two-sided shell spring has the opposite sign here — with
    // orbit_scale 0.6 the shell sits at 60, so a child at 55 was pushed further
    // OUT, deepening the violation. A child placed outside the shell would be
    // pulled inward by both laws and prove nothing.
    const forces = await runSibling(device, pipeline, {
      parent: [-1, 0],
      positions: new Float32Array([0, 0, 55, 0]),
      wellRadius: new Float32Array([100, 50]),
    });

    const radial = forces[2]; // +x is away from the parent
    console.log(`[gpu] RA bubble containment: radial force on the child = ${radial.toFixed(4)}`);
    assert(Number.isFinite(radial), `non-finite force on the child: ${radial}`);
    assert(
      radial < 0,
      `child overhanging its parent's well is pushed further out: radial ${radial}`,
    );
    assertAlmostEquals(forces[3], 0, 1e-4, "containment force must be purely radial");
  },
);

gpuTest(
  "GPU RA bubble: the containment force is continuous across the well boundary",
  async (device) => {
    const pipeline = siblingPipeline(device);

    // Parent well 100, child well 50, so containment binds at distance 50
    // exactly. The orbit pull is off so the barrier is the only term acting: a
    // step here is a step in the whole field, which is the defect the density
    // grid was replaced to avoid reintroducing.
    const parentWell = 100;
    const childWell = 50;
    const boundary = parentWell - childWell;
    const samples = 601;
    const start = 30;
    const step = 0.1;

    const radial: number[] = [];
    for (let i = 0; i < samples; i++) {
      const d = start + i * step;
      const forces = await runSibling(device, pipeline, {
        parent: [-1, 0],
        positions: new Float32Array([0, 0, d, 0]),
        wellRadius: new Float32Array([parentWell, childWell]),
        knobs: { orbitStrength: 0 },
      });
      radial.push(forces[2]);
    }

    const atBoundary = radial[Math.round((boundary - start) / step)];
    assertAlmostEquals(
      atBoundary,
      0,
      1e-4,
      `force must vanish at the boundary itself, got ${atBoundary}`,
    );

    let maxStep = 0;
    let range = 0;
    for (let i = 1; i < radial.length; i++) {
      maxStep = Math.max(maxStep, Math.abs(radial[i] - radial[i - 1]));
      range = Math.max(range, Math.abs(radial[i] - radial[0]));
    }
    console.log(
      `[gpu] RA bubble continuity: max adjacent step ${maxStep.toFixed(4)} over range ` +
        `${range.toFixed(2)}`,
    );
    // A quadratic ramp sampled every 0.1 units steps by ~1% of its range; a
    // constant push inside the violation region steps by the whole of it.
    assert(
      maxStep < range * 0.05,
      `discontinuity at the well boundary: max step ${maxStep} against range ${range}`,
    );
  },
);

// ===========================================================================
// Ancestor pairs and mass weighting
// ===========================================================================

gpuTest(
  "GPU RA bubble: no phantom force acts between a node and its own child",
  async (device) => {
    const pipeline = siblingPipeline(device);

    // A child's well is inside its parent's by construction, so the phantom
    // overlap predicate is permanently satisfied on this pair and the overlay
    // would be a constant outward push on exactly what containment holds
    // together. Zeroing the wells makes the overlay geometrically inactive
    // without touching any other term, so the two runs must agree exactly.
    const base = {
      parent: [-1, 0],
      positions: new Float32Array([0, 0, 30, 0]),
      knobs: { orbitStrength: 0, parentChildMultiplier: 0.15 },
    } as const;

    const withWells = await runSibling(device, pipeline, {
      ...base,
      wellRadius: new Float32Array([100, 10]),
    });
    const withoutWells = await runSibling(device, pipeline, {
      ...base,
      wellRadius: new Float32Array([0, 0]),
    });

    console.log(
      `[gpu] RA bubble parent-child: F_parent with wells = ${withWells[0].toFixed(5)}, ` +
        `without = ${withoutWells[0].toFixed(5)}`,
    );
    assert(Math.abs(withWells[0]) > 0, "the plain parent-child repulsion must survive");
    assertAlmostEquals(
      withWells[0],
      withoutWells[0],
      1e-4,
      "the well radii must not change the force on a node from its own child",
    );
    assertAlmostEquals(withWells[1], withoutWells[1], 1e-4);
  },
);

gpuTest(
  "GPU RA bubble: phantom separation is geometric in bubble mode and mass-weighted outside it",
  async (device) => {
    const pipeline = siblingPipeline(device);

    // Two siblings 60 apart with wells of 50 each: overlapping. The phantom's
    // own contribution is isolated by differencing a phantom-on run against a
    // phantom-off one, so every other term — including the mass-weighted 1/d
    // repulsion, which stays mass-weighted in both modes — cancels exactly.
    const fixture = {
      parent: [-1, 0, 0],
      positions: new Float32Array([0, 0, -30, 0, 30, 0]),
      wellRadius: new Float32Array([400, 50, 50]),
    } as const;

    const phantomOf = async (
      masses: Float32Array<ArrayBuffer>,
      bubbleMode: number,
    ): Promise<number> => {
      const on = await runSibling(device, pipeline, {
        ...fixture,
        masses,
        knobs: { orbitStrength: 0, bubbleMode, phantomEnabled: 1, phantomMultiplier: 50 },
      });
      const off = await runSibling(device, pipeline, {
        ...fixture,
        masses,
        knobs: { orbitStrength: 0, bubbleMode, phantomEnabled: 0, phantomMultiplier: 50 },
      });
      return on[2] - off[2];
    };

    const light = new Float32Array([1, 1, 1]);
    const heavy = new Float32Array([1, 50, 50]);

    const bubbleLight = await phantomOf(light, 1);
    const bubbleHeavy = await phantomOf(heavy, 1);
    console.log(
      `[gpu] RA bubble phantom: bubble mode light ${bubbleLight.toFixed(4)}, ` +
        `heavy ${bubbleHeavy.toFixed(4)}`,
    );
    assert(Math.abs(bubbleLight) > 1, `bubble phantom is inert: ${bubbleLight}`);
    // Well radius already encodes subtree size, so multiplying by a mass that
    // compounds geometrically with depth double-counts it and drives the force
    // orders of magnitude past the integrator's velocity cap.
    assertAlmostEquals(
      bubbleHeavy,
      bubbleLight,
      Math.abs(bubbleLight) * 1e-2,
      `bubble phantom still scales with mass: ${bubbleHeavy} vs ${bubbleLight}`,
    );

    // Outside bubble mode the zone radius IS derived from mass, and mass is the
    // only size proxy the pass has there — so the weighting must stay.
    const plainLight = await phantomOf(light, 0);
    const plainHeavy = await phantomOf(heavy, 0);
    console.log(
      `[gpu] RA phantom (non-bubble): light ${plainLight.toFixed(4)}, ` +
        `heavy ${plainHeavy.toFixed(4)}`,
    );
    assert(
      Math.abs(plainHeavy) > Math.abs(plainLight) * 10,
      `non-bubble phantom lost its mass weighting: ${plainHeavy} vs ${plainLight}`,
    );
  },
);

// ===========================================================================
// Forest-root separation
// ===========================================================================

interface RootCase {
  positions: Float32Array<ArrayBuffer>;
  wellRadius: Float32Array<ArrayBuffer>;
  roots: Uint32Array<ArrayBuffer>;
  repulsionStrength?: number;
}

/** Dispatch relativity_bubble_roots.comp.wgsl over a compacted root list. */
async function runBubbleRoots(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  testCase: RootCase,
): Promise<Float32Array> {
  const nodeCount = testCase.wellRadius.length;

  const uniformData = new ArrayBuffer(16);
  const view = new DataView(uniformData);
  view.setUint32(0, nodeCount, true);
  view.setUint32(4, testCase.roots.length, true);
  view.setFloat32(8, testCase.repulsionStrength ?? 50, true);
  view.setFloat32(12, 1, true); // min_distance
  const uniforms = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniforms, 0, uniformData);

  const posBuf = createStorageBuffer(device, testCase.positions);
  const forceBuf = createStorageBuffer(
    device,
    new Float32Array(nodeCount * 2),
    GPUBufferUsage.COPY_SRC,
  );
  const wellBuf = createStorageBuffer(device, testCase.wellRadius);
  const flagBuf = createStorageBuffer(device, new Uint32Array(nodeCount));
  const rootBuf = createStorageBuffer(device, testCase.roots);

  try {
    return await dispatchAndReadForces(
      device,
      pipeline,
      [
        { binding: 0, resource: { buffer: uniforms } },
        { binding: 1, resource: { buffer: posBuf } },
        { binding: 2, resource: { buffer: forceBuf } },
        { binding: 3, resource: { buffer: wellBuf } },
        { binding: 4, resource: { buffer: flagBuf } },
        { binding: 5, resource: { buffer: rootBuf } },
      ],
      forceBuf,
      nodeCount,
      testCase.roots.length,
    );
  } finally {
    for (const b of [uniforms, posBuf, forceBuf, wellBuf, flagBuf, rootBuf]) b.destroy();
  }
}

function bubbleRootsPipeline(device: GPUDevice): GPUComputePipeline {
  return device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: BUBBLE_ROOTS_WGSL }),
      entryPoint: "main",
    },
  });
}

gpuTest(
  "GPU RA bubble roots: overlapping root bubbles push apart and conserve momentum",
  async (device) => {
    const pipeline = bubbleRootsPipeline(device);

    // Two forest roots 120 apart with wells of 100 each: 80 units of overlap.
    // No sibling relation exists between them — this is the pair the sibling
    // pass structurally cannot reach, and until this shader existed only the
    // density grid held them apart.
    const forces = await runBubbleRoots(device, pipeline, {
      positions: new Float32Array([0, 0, 120, 0]),
      wellRadius: new Float32Array([100, 100]),
      roots: Uint32Array.from([0, 1]),
    });

    console.log(
      `[gpu] RA bubble roots: F0 = (${forces[0].toFixed(4)}, ${forces[1].toFixed(4)}), ` +
        `F1 = (${forces[2].toFixed(4)}, ${forces[3].toFixed(4)})`,
    );
    assert(forces[0] < -1, `root 0 is not pushed away from root 1: ${forces[0]}`);
    assert(forces[2] > 1, `root 1 is not pushed away from root 0: ${forces[2]}`);
    assertAlmostEquals(forces[0] + forces[2], 0, 1e-5, "root separation injects net momentum");
    assertAlmostEquals(forces[1] + forces[3], 0, 1e-5);
  },
);

gpuTest(
  "GPU RA bubble roots: a truncated 200-root list receives no net momentum",
  async (device) => {
    const pipeline = bubbleRootsPipeline(device);

    // 200 roots exceeds the 64-interaction budget, so the pass falls back to
    // modulo-group selection. A head-prefix cap instead would push the tail of
    // the list against its head with no recoil, drifting the whole layout every
    // frame — the same defect the sibling pass was fixed for.
    const rootCount = 200;
    const rng = mulberry32(7);
    const positions = new Float32Array(rootCount * 2);
    for (let i = 0; i < rootCount; i++) {
      const angle = 2 * Math.PI * (i / rootCount) + (rng() - 0.5) * 0.02;
      const radius = 300 + (rng() - 0.5) * 20;
      positions[i * 2] = radius * Math.cos(angle);
      positions[i * 2 + 1] = radius * Math.sin(angle);
    }
    const roots = new Uint32Array(rootCount);
    for (let i = 0; i < rootCount; i++) roots[i] = i;

    const forces = await runBubbleRoots(device, pipeline, {
      positions,
      wellRadius: new Float32Array(rootCount).fill(100),
      roots,
    });

    let gross = 0;
    let netX = 0;
    let netY = 0;
    for (let i = 0; i < rootCount; i++) {
      const fx = forces[i * 2];
      const fy = forces[i * 2 + 1];
      const mag = Math.hypot(fx, fy);
      assert(Number.isFinite(mag), `non-finite force on root ${i}`);
      assert(mag > 0, `root ${i} receives no separation force`);
      gross += mag;
      netX += fx;
      netY += fy;
    }

    const net = Math.hypot(netX, netY);
    const mean = gross / rootCount;
    console.log(
      `[gpu] RA bubble roots truncation: mean |F| = ${mean.toFixed(4)}, ` +
        `|sum F| = ${net.toFixed(6)}`,
    );
    assert(
      net < mean * 1e-2,
      `root truncation injects net momentum: |sum F| = ${net}, mean |F| = ${mean}`,
    );
  },
);

/**
 * The shader's own interaction budget, read from its source.
 *
 * The two cases below are stated in terms of it — "one more root than the
 * budget" is what forces exactly two modulo-groups — so raising the budget
 * moves the fixtures with it instead of silently changing what they mean.
 */
const MAX_ROOT_INTERACTIONS = (() => {
  const match = BUBBLE_ROOTS_WGSL.match(/MAX_ROOT_INTERACTIONS:\s*u32\s*=\s*(\d+)u/);
  if (!match) throw new Error("MAX_ROOT_INTERACTIONS not found in the roots shader");
  return Number(match[1]);
})();

gpuTest(
  "GPU RA bubble roots: truncation groups roots by list position, not by node slot",
  async (device) => {
    const pipeline = bubbleRootsPipeline(device);

    // Grouping by list POSITION is what lets the loop stride over its group
    // instead of scanning the list and discarding — the difference between
    // O(R * budget) and O(R^2) wall time. It is also observable: the list holds
    // an arbitrary ascending subset of the slots, so position parity and slot
    // parity disagree, and a pair can be visible under one rule and invisible
    // under the other.
    //
    // One more root than the budget forces exactly two groups. The list skips
    // slot 1, so positions 0 and 2 — one group under the shipped rule — land on
    // slots 0 and 3, which are two groups apart under the old one.
    const rootCount = MAX_ROOT_INTERACTIONS + 1;
    const groups = Math.ceil(rootCount / MAX_ROOT_INTERACTIONS);
    const slotOf = (position: number): number => (position === 0 ? 0 : position + 1);
    const roots = new Uint32Array(rootCount);
    for (let p = 0; p < rootCount; p++) roots[p] = slotOf(p);
    const nodeCount = slotOf(rootCount - 1) + 1;

    const pair = [slotOf(0), slotOf(2)];
    assert(
      0 % groups === 2 % groups && pair[0] % groups !== pair[1] % groups,
      "fixture no longer separates the two grouping rules",
    );

    // The pair overlaps; every other root is parked far enough out that its
    // well cannot reach anything, so the pair's force is the only one in play.
    const positions = new Float32Array(nodeCount * 2);
    const wellRadius = new Float32Array(nodeCount).fill(1);
    for (let p = 1; p < rootCount; p++) {
      const angle = (2 * Math.PI * p) / rootCount;
      positions[slotOf(p) * 2] = 1e5 * Math.cos(angle);
      positions[slotOf(p) * 2 + 1] = 1e5 * Math.sin(angle);
    }
    positions[pair[0] * 2] = 0;
    positions[pair[0] * 2 + 1] = 0;
    positions[pair[1] * 2] = 30;
    positions[pair[1] * 2 + 1] = 0;
    wellRadius[pair[0]] = 100;
    wellRadius[pair[1]] = 100;

    const forces = await runBubbleRoots(device, pipeline, { positions, wellRadius, roots });

    console.log(
      `[gpu] RA bubble roots grouping: F(slot ${pair[0]}) = ${forces[pair[0] * 2].toFixed(4)}, ` +
        `F(slot ${pair[1]}) = ${forces[pair[1] * 2].toFixed(4)}`,
    );
    assert(
      forces[pair[0] * 2] < -1,
      `slots ${pair[0]} and ${pair[1]} share a group by list position but did not interact: ` +
        `force ${forces[pair[0] * 2]}`,
    );
    assertAlmostEquals(
      forces[pair[0] * 2] + forces[pair[1] * 2],
      0,
      1e-5,
      "position grouping must stay mutual",
    );
  },
);

/**
 * Time one dispatch of the root pass over `rootCount` roots, in milliseconds.
 *
 * Dispatches are batched into a single pass and a single submit so the figure
 * is per-dispatch cost rather than per-submit latency, and the best of several
 * timed runs is returned after a discarded warm-up. Best-of, not mean: the
 * noise on a shared GPU is one-sided — scheduling delay can only inflate a
 * measurement — so the minimum is the closest estimate of the real cost and the
 * one that makes the ratio below reproducible.
 */
async function timeRootPass(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  rootCount: number,
  dispatches: number,
): Promise<number> {
  const positions = new Float32Array(rootCount * 2);
  for (let i = 0; i < rootCount; i++) {
    const angle = (2 * Math.PI * i) / rootCount;
    positions[i * 2] = 50 * Math.cos(angle);
    positions[i * 2 + 1] = 50 * Math.sin(angle);
  }
  const roots = new Uint32Array(rootCount);
  for (let i = 0; i < rootCount; i++) roots[i] = i;

  const uniformData = new ArrayBuffer(16);
  const view = new DataView(uniformData);
  view.setUint32(0, rootCount, true);
  view.setUint32(4, rootCount, true);
  view.setFloat32(8, 50, true);
  view.setFloat32(12, 1, true);
  const uniforms = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniforms, 0, uniformData);

  // Every well overlaps every other, so the force branch is taken for every
  // pair the shader chooses to look at: no early-out can flatter the result.
  const posBuf = createStorageBuffer(device, positions);
  const forceBuf = createStorageBuffer(device, new Float32Array(rootCount * 2));
  const wellBuf = createStorageBuffer(device, new Float32Array(rootCount).fill(500));
  const flagBuf = createStorageBuffer(device, new Uint32Array(rootCount));
  const rootBuf = createStorageBuffer(device, roots);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniforms } },
      { binding: 1, resource: { buffer: posBuf } },
      { binding: 2, resource: { buffer: forceBuf } },
      { binding: 3, resource: { buffer: wellBuf } },
      { binding: 4, resource: { buffer: flagBuf } },
      { binding: 5, resource: { buffer: rootBuf } },
    ],
  });

  const run = async (): Promise<void> => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    for (let d = 0; d < dispatches; d++) {
      pass.dispatchWorkgroups(Math.ceil(rootCount / 256));
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  };

  try {
    await run();
    let best = Infinity;
    for (let attempt = 0; attempt < 3; attempt++) {
      const started = performance.now();
      await run();
      best = Math.min(best, (performance.now() - started) / dispatches);
    }
    return best;
  } finally {
    for (const b of [uniforms, posBuf, forceBuf, wellBuf, flagBuf, rootBuf]) b.destroy();
  }
}

/**
 * How much dearer a 32x larger root list may be per dispatch.
 *
 * The pass is O(R * budget): each thread walks its own modulo-group and the
 * group has at most `MAX_ROOT_INTERACTIONS` members whatever R is, so 32x the
 * roots is 32x the threads doing the same work each — which on any GPU with
 * spare occupancy costs barely more than the small list. Scanning the list and
 * filtering inside the loop computes exactly the same forces but touches every
 * entry, and that is 32x the work PER THREAD on top: R = 32768 measures ~35x
 * the small list rather than ~1x.
 *
 * Measured on an M-series adapter: strided 1.1x, scanning 36x. The bound is
 * stated against a measurement of the same shader on the same device rather
 * than an absolute millisecond figure so that a slower adapter shifts both
 * sides together.
 */
const ROOT_SCALING_BOUND = 10;

gpuTest(
  "GPU RA bubble roots: cost per dispatch is flat in the root count",
  async (device) => {
    const pipeline = bubbleRootsPipeline(device);

    // R = N is reachable from a legal input — a supplied hierarchy may declare
    // every slot a root, and validateHierarchyColumns accepts it — so the
    // degenerate case has to be affordable, not merely documented. At the
    // repository's 35K-node target a quadratic scan costs ~45 ms, three whole
    // frame budgets for one of the algorithm's eight passes.
    const small = await timeRootPass(device, pipeline, 1024, 32);
    const large = await timeRootPass(device, pipeline, 32768, 32);
    const ratio = large / small;
    console.log(
      `[gpu] RA bubble roots cost: 1024 roots ${small.toFixed(3)} ms, ` +
        `32768 roots ${large.toFixed(3)} ms (${ratio.toFixed(1)}x)`,
    );
    assert(
      ratio < ROOT_SCALING_BOUND,
      `root separation is superlinear in the root count: 32x the roots cost ` +
        `${ratio.toFixed(1)}x per dispatch (${small.toFixed(3)} ms -> ${large.toFixed(3)} ms)`,
    );
  },
);

// ===========================================================================
// Whole-algorithm behaviour
// ===========================================================================

interface RaBuffersView {
  destroy(): void;
  nodeDepth: GPUBuffer;
  wellRadius: GPUBuffer;
}

interface RaModule {
  createRelativityAtlasAlgorithm(): HarnessForceAlgorithm;
  uploadRelativityAtlasEdges(
    device: GPUDevice,
    buffers: { destroy(): void },
    forwardCSR: { offsets: Uint32Array; indices: Uint32Array },
    inverseCSR: { offsets: Uint32Array; indices: Uint32Array },
    nodeCount: number,
  ): void;
  uploadRelativityAtlasBubbleRoots(
    device: GPUDevice,
    buffers: { destroy(): void },
    roots: Uint32Array,
  ): void;
}

/** A two-root forest of directories and files: the shape a multi-repo graph has. */
interface ForestFixture {
  nodeCount: number;
  roots: number[];
  dirs: number[];
  edgeSources: Uint32Array;
  edgeTargets: Uint32Array;
  containment: Uint32Array;
  positionsX: Float32Array;
  positionsY: Float32Array;
}

function buildForestFixture(
  rootCount: number,
  dirsPerRoot: number,
  filesPerDir: number,
  seed: number,
): ForestFixture {
  const rng = mulberry32(seed);
  const roots: number[] = [];
  const dirs: number[] = [];
  const sources: number[] = [];
  const targets: number[] = [];
  let next = 0;

  for (let r = 0; r < rootCount; r++) {
    const root = next++;
    roots.push(root);
    for (let d = 0; d < dirsPerRoot; d++) {
      const dir = next++;
      dirs.push(dir);
      sources.push(root);
      targets.push(dir);
      for (let f = 0; f < filesPerDir; f++) {
        const file = next++;
        sources.push(dir);
        targets.push(file);
      }
    }
  }

  const containment = new Uint32Array(sources.length * 2);
  for (let e = 0; e < sources.length; e++) {
    containment[e * 2] = sources[e];
    containment[e * 2 + 1] = targets[e];
  }

  // Seeded scatter over a disc: no fixture-encoded structure for the layout to
  // inherit, so the separation asserted below has to be produced by the forces.
  const positionsX = new Float32Array(next);
  const positionsY = new Float32Array(next);
  for (let i = 0; i < next; i++) {
    const angle = rng() * 2 * Math.PI;
    const radius = 400 * Math.sqrt(rng());
    positionsX[i] = radius * Math.cos(angle);
    positionsY[i] = radius * Math.sin(angle);
  }

  return {
    nodeCount: next,
    roots,
    dirs,
    edgeSources: Uint32Array.from(sources),
    edgeTargets: Uint32Array.from(targets),
    containment,
    positionsX,
    positionsY,
  };
}

/** Derive the fixture's hierarchy through the shipped WASM path. */
async function deriveFixtureHierarchy(fixture: ForestFixture): Promise<RetainedHierarchy> {
  const engine = await createWasmEngine() as unknown as HierarchyDeriver & {
    clear(): void;
    addNodesFromPositions(p: Float32Array): void;
  };
  engine.clear();
  engine.addNodesFromPositions(new Float32Array(fixture.nodeCount * 2));
  return deriveHierarchy(engine, fixture.containment, fixture.nodeCount, {
    baseRadius: 10,
    padding: 5,
    rootId: HIERARCHY_ROOT,
  });
}

/**
 * Build the harness the way graph.ts builds bubble mode: containment-only CSR,
 * the well radii, the depths and the forest roots.
 */
async function bubbleHarness(
  device: GPUDevice,
  fixture: ForestFixture,
  hierarchy: RetainedHierarchy,
  config: Record<string, unknown>,
) {
  const mod = await loadModuleInliningWgsl<RaModule>(
    new URL(
      "../../packages/core/src/simulation/algorithms/relativity-atlas.ts",
      import.meta.url,
    ),
  );
  const csr = hierarchyCsrPair(hierarchy);
  const depths = new Float32Array(fixture.nodeCount);
  for (let i = 0; i < fixture.nodeCount; i++) depths[i] = hierarchy.columns.depth[i];

  return await createAlgorithmSimHarness(
    device,
    mod.createRelativityAtlasAlgorithm(),
    {
      nodeCount: fixture.nodeCount,
      positionsX: fixture.positionsX,
      positionsY: fixture.positionsY,
      edgeSources: fixture.edgeSources,
      edgeTargets: fixture.edgeTargets,
      depths,
    },
    {
      relativityBubbleMode: true,
      relativityPhantomZone: true,
      // Off on purpose: this is the claim under test. If the layout separates
      // without it, the 128x128 grid is redundant in bubble mode rather than
      // load-bearing, and the cross-hatch it produces is avoidable.
      relativityDensityRepulsion: 0,
      ...config,
    },
    undefined,
    {
      boundsSyncInterval: 10,
      onAlgorithmBuffers(algoBuffers) {
        mod.uploadRelativityAtlasEdges(
          device,
          algoBuffers,
          { offsets: csr.forward.offsets, indices: csr.forward.targets },
          { offsets: csr.inverse.offsets, indices: csr.inverse.sources },
          fixture.nodeCount,
        );
        mod.uploadRelativityAtlasBubbleRoots(
          device,
          algoBuffers,
          collectForestRoots(hierarchy.columns.parent, fixture.nodeCount),
        );
        const b = algoBuffers as RaBuffersView;
        device.queue.writeBuffer(b.nodeDepth, 0, depths.slice().buffer);
        device.queue.writeBuffer(
          b.wellRadius,
          0,
          hierarchy.columns.wellRadius.slice().buffer,
        );
      },
    },
  );
}

/**
 * Slack allowed on the containment clause: a node's reach may exceed its
 * parent's well by this factor.
 *
 * Held to 5% because containment is the clause the barrier can actually win.
 * The measured worst case on this fixture is 1.003.
 */
const CONTAINMENT_SLACK = 1.05;

/**
 * Floor on how close two same-level bubbles may come, as a fraction of touching.
 *
 * Sibling separation, unlike containment, cannot be exact here, for two reasons
 * that are worth separating.
 *
 * The first is geometric and belongs to the radius formula, not to the forces:
 * the well radius is `sqrt(sum of child areas / (pi * 0.82))`, an asymptotic
 * packing bound that falls short of the exact small-k circle-packing ratio at
 * the branching factors a directory tree has — six equal children need 3.00
 * parent radii and the formula allots 2.70. At k = 6 no arrangement satisfies
 * containment and disjointness simultaneously, so no equilibrium can either.
 *
 * The second is that a C1 barrier is soft by construction near its boundary,
 * which is exactly the property that keeps the force field free of steps. A
 * constant opposing pull — here the always-on log(1 + d) edge attraction — will
 * always penetrate to the depth where the quadratic ramp matches it.
 *
 * So this asserts the property a force model can deliver, bounded penetration,
 * and the sharper user-visible consequence is asserted separately below. The
 * measured range over repeated runs is 0.73 to 0.79.
 */
const SEPARATION_FLOOR = 0.65;

/**
 * Fraction of leaves that must sit nearer their own parent than any other
 * parent at the same level.
 *
 * This is the legibility claim the whole invariant exists to buy: subtree
 * membership readable from position. It is a fraction rather than an absolute
 * because the residual overlap described under {@link SEPARATION_FLOOR} leaves
 * a handful of leaves on the contested boundary between two bubbles, and
 * because the layout is not bit-reproducible — the edge-attraction pass
 * accumulates forces atomically, so summation order varies between runs.
 * Measured over repeated runs: 0.986 to 1.000.
 */
const ASSIGNMENT_FLOOR = 0.95;

/**
 * Floor on the distance between the two forest roots, as a fraction of
 * touching.
 *
 * The fixture starts the two roots on the same spot (see the case below), so
 * this is not "did the pair drift apart" but "did the pass move a fully
 * superimposed pair to almost disjoint". Measured over repeated runs: 0.865 to
 * 0.896 with the pass acting, 0.267 to 0.289 with its force zeroed. The floor
 * sits between those two populations with room on both sides, because the
 * layout is not bit-reproducible — the edge-attraction pass accumulates
 * atomically — and a threshold pinned just under the healthy range would be
 * deciding CI runs by summation order rather than by physics.
 */
const ROOT_SEPARATION_FLOOR = 0.8;

gpuTest(
  "GPU RA bubble: unrelated subtrees stay separated with the density grid off",
  async (device) => {
    const fixture = buildForestFixture(2, 6, 12, 11);

    // The two roots start superimposed, and their subtrees are scattered over
    // the same disc. NOTHING in the model acts between two different trees
    // except the root pass: they share no edge, no parent and therefore no
    // sibling or cousin relation, and the density grid is off. So a pair that
    // ends up apart was moved apart by that pass and by nothing else — which is
    // the causal claim clause 3 needs and which a pair started at a random
    // separation cannot make, since each tree contracting onto its own root
    // leaves them roughly where the seed put them.
    fixture.positionsX[fixture.roots[1]] = fixture.positionsX[fixture.roots[0]];
    fixture.positionsY[fixture.roots[1]] = fixture.positionsY[fixture.roots[0]];

    const hierarchy = await deriveFixtureHierarchy(fixture);
    const { parent, wellRadius } = hierarchy.columns;

    const harness = await bubbleHarness(device, fixture, hierarchy, {});
    try {
      await harness.tick(400);
      const end = await harness.readPositions();
      assertEquals(countNonFinite(end.x, end.y), 0);

      // Clause 1 — containment. Every node's well is inside its parent's, which
      // is what makes a bubble stand for its whole subtree.
      let worst = 0;
      let worstSlot = -1;
      for (let i = 0; i < fixture.nodeCount; i++) {
        const p = parent[i];
        if (p === HIERARCHY_ROOT) continue;
        const reach = Math.hypot(end.x[i] - end.x[p], end.y[i] - end.y[p]) + wellRadius[i];
        const ratio = reach / wellRadius[p];
        if (ratio > worst) {
          worst = ratio;
          worstSlot = i;
        }
      }
      console.log(
        `[gpu] RA bubble containment: worst reach/parent-well = ${worst.toFixed(3)} ` +
          `at slot ${worstSlot}`,
      );
      assert(
        worst <= CONTAINMENT_SLACK,
        `slot ${worstSlot} escapes its parent's well: reach is ${worst.toFixed(3)} of it`,
      );

      // Clause 3 — the forest roots, the pair with no common ancestor at all.
      // They were put on top of each other and only the root pass can have
      // moved them, against mass-weighted gravity pulling both to the centre.
      const [rootA, rootB] = fixture.roots;
      const rootGap = Math.hypot(end.x[rootA] - end.x[rootB], end.y[rootA] - end.y[rootB]) /
        (wellRadius[rootA] + wellRadius[rootB]);
      console.log(`[gpu] RA bubble roots: root pair at ${rootGap.toFixed(3)} of touching`);
      assert(
        rootGap >= ROOT_SEPARATION_FLOOR,
        `forest roots stayed superimposed: ${rootGap.toFixed(3)} of touching`,
      );

      // Clause 2 — same-level bubbles, within a root and across the two.
      let closest = Infinity;
      for (let a = 0; a < fixture.dirs.length; a++) {
        for (let b = a + 1; b < fixture.dirs.length; b++) {
          const i = fixture.dirs[a];
          const j = fixture.dirs[b];
          const d = Math.hypot(end.x[i] - end.x[j], end.y[i] - end.y[j]);
          closest = Math.min(closest, d / (wellRadius[i] + wellRadius[j]));
        }
      }
      console.log(
        `[gpu] RA bubble separation: closest dir pair = ${closest.toFixed(3)} of touching`,
      );
      assert(
        closest >= SEPARATION_FLOOR,
        `directory bubbles interpenetrate: closest pair at ${closest.toFixed(3)} of touching`,
      );

      // The consequence the three clauses exist for, and the one a reader of the
      // picture actually sees: a file sits nearer its own directory than any
      // other, so subtree membership is legible from position alone.
      let files = 0;
      let assigned = 0;
      for (let i = 0; i < fixture.nodeCount; i++) {
        const p = parent[i];
        if (p === HIERARCHY_ROOT || parent[p] === HIERARCHY_ROOT) continue;
        files++;
        const own = Math.hypot(end.x[i] - end.x[p], end.y[i] - end.y[p]);
        let nearest = Infinity;
        for (const dir of fixture.dirs) {
          if (dir === p) continue;
          nearest = Math.min(nearest, Math.hypot(end.x[i] - end.x[dir], end.y[i] - end.y[dir]));
        }
        if (own < nearest) assigned++;
      }
      const assignedFraction = assigned / files;
      console.log(
        `[gpu] RA bubble separation: files nearest their own directory = ` +
          `${assigned}/${files} (${assignedFraction.toFixed(3)})`,
      );
      assert(
        assignedFraction >= ASSIGNMENT_FLOOR,
        `only ${(assignedFraction * 100).toFixed(1)}% of files sit nearest their own directory`,
      );
    } finally {
      harness.dispose();
    }
  },
);

/** `maxVelocity`'s default in simulation/config.ts, the integrator's clamp. */
const VELOCITY_CAP = 50;

/**
 * Ticks allowed before the cap is asserted on.
 *
 * A fixture starts as a 400-unit scatter, so nodes begin hundreds of units
 * outside wells ten units wide and the containment barrier is legitimately at
 * its ceiling. Saturation there is the barrier doing its job; saturation that
 * PERSISTS is a force law with no resolution left, and that is what this
 * window separates. Measured: the peak per-tick step leaves the cap around
 * tick 15 and is under 4 by tick 30.
 */
const SETTLE_TICKS = 50;

gpuTest(
  "GPU RA bubble: separation never saturates the velocity cap, and the layout settles",
  async (device) => {
    const fixture = buildForestFixture(2, 6, 12, 5);
    const hierarchy = await deriveFixtureHierarchy(fixture);

    const harness = await bubbleHarness(device, fixture, hierarchy, {});
    try {
      // Per-tick displacement is the only view of force magnitude a harness has,
      // and it is the one that matters: a node whose net force exceeds the cap
      // moves exactly `maxVelocity * dt` and no further, so a run pinned at the
      // cap is a run whose force law has lost all resolution. That is what a
      // mass-weighted separation force produces — mass compounds as roughly
      // (branching/2)^depth up the tree, so even three levels puts the product
      // orders of magnitude past the cap, every frame clips, and the direction
      // flips frame to frame. Sampled over the opening ticks, where the alpha
      // schedule is hottest and any saturation shows first.
      await harness.tick(SETTLE_TICKS);
      let peakStep = 0;
      let previous = await harness.readPositions();
      for (let tick = 0; tick < 100; tick++) {
        await harness.tick(1);
        const now = await harness.readPositions();
        for (let i = 0; i < fixture.nodeCount; i++) {
          peakStep = Math.max(
            peakStep,
            Math.hypot(now.x[i] - previous.x[i], now.y[i] - previous.y[i]),
          );
        }
        previous = now;
      }
      console.log(
        `[gpu] RA bubble: peak per-tick displacement = ${peakStep.toFixed(2)} ` +
          `against a cap of ${VELOCITY_CAP}`,
      );
      assert(
        peakStep < VELOCITY_CAP * 0.5,
        `bubble separation saturates the velocity cap: peak step ${peakStep}`,
      );

      await harness.tick(150);
      const mid = await harness.readPositions();
      assertEquals(countNonFinite(mid.x, mid.y), 0);

      await harness.tick(100);
      const end = await harness.readPositions();
      assertEquals(countNonFinite(end.x, end.y), 0);

      const lateKE = kineticEnergy(mid.x, mid.y, end.x, end.y, 100) / fixture.nodeCount;
      console.log(`[gpu] RA bubble: late per-node KE = ${lateKE.toExponential(2)}`);
      assert(lateKE < 1e-3, `bubble layout still moving after cooling: per-node KE ${lateKE}`);
    } finally {
      harness.dispose();
    }
  },
);

// ===========================================================================
// The bubble-mode upload path on the assembled instance
// ===========================================================================

/**
 * The one field of the algorithm's buffers these cases read.
 *
 * `rootCount` is CPU-side bookkeeping, not a GPU buffer, and there is no other
 * observable that distinguishes "the roots were uploaded" from "the pass is
 * silently dispatching over nothing" — the root list has no COPY_SRC usage and
 * a missed upload changes no position by any amount a test could name. So the
 * private field is read directly rather than inferred from a layout.
 */
interface BubbleUploadProbe {
  algorithmBuffers: { rootCount: number; maxNodes: number } | null;
}

/**
 * A two-root forest as typed input: `contains` edges plus one cross-cutting
 * edge, with the hierarchy columns supplied because the headless harness runs
 * without a WASM engine.
 */
function typedForestInput(): {
  input: GraphTypedInput;
  nodeCount: number;
  rootCount: number;
} {
  // 0 → {1, 2}, 3 → {4, 5}: two roots, two children each.
  const parent = Uint32Array.from([
    HIERARCHY_ROOT,
    0,
    0,
    HIERARCHY_ROOT,
    3,
    3,
  ]);
  const nodeCount = parent.length;
  const positions = new Float32Array(nodeCount * 2);
  for (let i = 0; i < nodeCount; i++) {
    positions[i * 2] = i * 30;
    positions[i * 2 + 1] = (i % 2) * 30;
  }

  // The last pair is a dependency edge, kind 1 — the shape that reparents a
  // node when the whole edge set is mistaken for the containment tree.
  const edgePairs = Uint32Array.from([0, 1, 0, 2, 3, 4, 3, 5, 5, 1]);
  const edgeKinds = Uint16Array.from([0, 0, 0, 0, 1]);

  return {
    input: {
      nodeCount,
      edgeCount: edgeKinds.length,
      positions,
      edgePairs,
      edgeKinds,
      containmentKind: 0,
      hierarchy: {
        parent,
        wellRadius: Float32Array.from([40, 10, 10, 40, 10, 10]),
        depth: Uint16Array.from([0, 1, 1, 0, 1, 1]),
        subtreeSize: Uint32Array.from([3, 1, 1, 3, 1, 1]),
      },
    },
    nodeCount,
    rootCount: 2,
  };
}

Deno.test({
  name: "GPU RA bubble: enabling bubble mode after load uploads the forest roots",
  ignore: adapter === null,
  sanitizeResources: false,
  sanitizeOps: false,
  async fn() {
    const device = await requestHarnessDevice(adapter!);
    const harness = await createHeadlessGraph(device);
    try {
      const { input, rootCount } = typedForestInput();
      await harness.graph.load(input);
      harness.graph.setForceAlgorithm("relativity-atlas");

      const probe = harness.graph as unknown as BubbleUploadProbe;
      assert(probe.algorithmBuffers !== null, "the algorithm's buffers were never created");

      // Outside bubble mode the pass must not dispatch at all: separating
      // roots is meaningless when the well radii are the uniform fallback.
      assertEquals(
        probe.algorithmBuffers.rootCount,
        0,
        "roots uploaded outside bubble mode",
      );

      // The bubble knobs feed a load-time buffer write rather than the per-tick
      // uniforms, so turning bubble mode on after load has to re-run the whole
      // upload. Missing that leaves the pass reading an empty list and doing
      // nothing, with no error and no visible symptom.
      harness.graph.setForceConfig({ relativityBubbleMode: true });
      assertEquals(
        probe.algorithmBuffers.rootCount,
        rootCount,
        "enabling bubble mode did not re-upload the forest roots",
      );

      // And back off again, or the previous graph's roots outlive the setting.
      harness.graph.setForceConfig({ relativityBubbleMode: false });
      assertEquals(
        probe.algorithmBuffers.rootCount,
        0,
        "disabling bubble mode left the roots uploaded",
      );
    } finally {
      harness.dispose();
      device.destroy();
    }
  },
});

/**
 * A two-tree forest with one dependency edge that contradicts the containment
 * tree.
 *
 * The one edge runs from a leaf of the second tree to a leaf of the first. On
 * the whole edge set it reads as parent→child, so that leaf acquires a second
 * "parent" in the other tree whose well is 10 wide — a containment barrier
 * satisfiable only by sitting on top of it. On the containment forest the edge
 * is what it is: a cross-cutting spring, and nothing more.
 *
 * ONE such edge, not one per leaf. Cross-cutting edges attract under either
 * CSR, and enough of them drag the two trees together whichever parenting rule
 * is in force — measured at six, both configurations leave the leaf nearer its
 * importer, which says something true about the model but nothing about the
 * question here. A single edge leaves the parenting rule as the only thing that
 * can decide where the leaf ends up.
 */
function crossLinkedForestInput(): {
  input: GraphTypedInput;
  /** The first tree's root and the leaf the dependency edge points at. */
  importedLeaf: number;
  /** Its containment parent, and the far root. */
  roots: [number, number];
  /** The second tree's leaf that imports {@link importedLeaf}. */
  importer: number;
} {
  const perTree = 6;
  const rootA = 0;
  const leavesA = Array.from({ length: perTree }, (_, i) => 1 + i);
  const rootB = perTree + 1;
  const leavesB = Array.from({ length: perTree }, (_, i) => rootB + 1 + i);
  const nodeCount = 2 * (perTree + 1);

  const parent = new Uint32Array(nodeCount);
  parent[rootA] = HIERARCHY_ROOT;
  parent[rootB] = HIERARCHY_ROOT;
  for (const leaf of leavesA) parent[leaf] = rootA;
  for (const leaf of leavesB) parent[leaf] = rootB;

  const pairs: number[] = [];
  const kinds: number[] = [];
  for (const leaf of leavesA) {
    pairs.push(rootA, leaf);
    kinds.push(0);
  }
  for (const leaf of leavesB) {
    pairs.push(rootB, leaf);
    kinds.push(0);
  }
  pairs.push(leavesB[0], leavesA[0]);
  kinds.push(1);

  // Seeded scatter over one disc, both trees intermixed: no starting structure
  // for either CSR to inherit.
  const rng = mulberry32(31);
  const positions = new Float32Array(nodeCount * 2);
  for (let i = 0; i < nodeCount; i++) {
    const angle = rng() * 2 * Math.PI;
    const radius = 200 * Math.sqrt(rng());
    positions[i * 2] = radius * Math.cos(angle);
    positions[i * 2 + 1] = radius * Math.sin(angle);
  }

  const wellRadius = new Float32Array(nodeCount).fill(10);
  wellRadius[rootA] = 60;
  wellRadius[rootB] = 60;
  const depth = new Uint16Array(nodeCount).fill(1);
  depth[rootA] = 0;
  depth[rootB] = 0;
  const subtreeSize = new Uint32Array(nodeCount).fill(1);
  subtreeSize[rootA] = perTree + 1;
  subtreeSize[rootB] = perTree + 1;

  return {
    input: {
      nodeCount,
      edgeCount: kinds.length,
      positions,
      edgePairs: Uint32Array.from(pairs),
      edgeKinds: Uint16Array.from(kinds),
      containmentKind: 0,
      hierarchy: { parent, wellRadius, depth, subtreeSize },
    },
    importedLeaf: leavesA[0],
    roots: [rootA, rootB],
    importer: leavesB[0],
  };
}

/**
 * How much nearer its containment parent than its importer the shared leaf must
 * settle.
 *
 * Measured on this fixture: 0.53 with the forest CSR, 1.52 with the whole edge
 * set — the leaf changes which tree it belongs to. The bound is the natural
 * boundary between those two outcomes rather than a tightened version of the
 * healthy figure, because what is being asserted is which tree the node joined,
 * not how snugly it orbits.
 */
const IMPORTED_LEAF_BOUND = 1;

Deno.test({
  name: "GPU RA bubble: bubble mode takes its parent set from the containment forest",
  ignore: adapter === null,
  sanitizeResources: false,
  sanitizeOps: false,
  async fn() {
    const device = await requestHarnessDevice(adapter!);
    const harness = await createHeadlessGraph(device);
    try {
      const { input, importedLeaf, roots, importer } = crossLinkedForestInput();
      await harness.graph.load(input);
      harness.graph.setForceAlgorithm("relativity-atlas");
      // The density grid off, so what holds the trees apart here is the bubble
      // force laws and not the field they exist to replace.
      harness.graph.setForceConfig({
        relativityBubbleMode: true,
        relativityDensityRepulsion: 0,
      });
      harness.graph.startSimulation();

      await harness.simulation.tick(300);
      const end = await harness.simulation.readPositions();
      assertEquals(countNonFinite(end.x, end.y), 0);

      // Every bubble clause — the orbit spring, the containment barrier, the
      // phantom separation — is stated against wellRadius and depth, and those
      // columns are computed over the containment forest. Hand the pass the
      // whole edge set instead and the invariants are stated across two
      // different graphs: this leaf becomes a child of its importer as well,
      // and the barrier holds it inside a well 10 wide in the OTHER tree.
      const toParent = Math.hypot(
        end.x[importedLeaf] - end.x[roots[0]],
        end.y[importedLeaf] - end.y[roots[0]],
      );
      const toImporter = Math.hypot(
        end.x[importedLeaf] - end.x[importer],
        end.y[importedLeaf] - end.y[importer],
      );
      const ratio = toParent / toImporter;
      console.log(
        `[gpu] RA bubble CSR: imported leaf sits ${toParent.toFixed(1)} from its parent and ` +
          `${toImporter.toFixed(1)} from its importer (${ratio.toFixed(2)}x)`,
      );
      assert(
        ratio < IMPORTED_LEAF_BOUND,
        `slot ${importedLeaf} settled nearer the node that imports it (${toImporter.toFixed(1)}) ` +
          `than its own containment parent (${toParent.toFixed(1)}): the sibling pass is ` +
          `deriving parents from the whole edge set, not from the forest the well radii ` +
          `were computed over`,
      );
    } finally {
      harness.dispose();
      device.destroy();
    }
  },
});
