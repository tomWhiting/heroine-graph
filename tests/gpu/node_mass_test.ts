/**
 * GPU tests for aggregate mass (SimulationBuffers.nodeMass), the per-slot
 * multiplier that lets a collapsed subtree's proxy repel like the subtree it
 * stands for.
 *
 * Four claims, in the order they matter:
 *
 * - **SC-005: with LOD off the simulation is bit-identical to before mass
 *   existed.** The reference is the shipped repulsion shader with the mass
 *   multiply textually removed — so the baseline cannot drift away from the
 *   code under test — driven through the same clear/spring/integrate
 *   pipelines for K ticks. Equality is exact, on the raw f32 bit patterns.
 * - **Mass scales repulsion, per node.** Quadrupling one node's mass
 *   quadruples the force it exerts and changes nothing about the force it
 *   receives.
 * - **SC-002: collapsing a subtree does not move what stays visible.** A
 *   collapsed subtree must lay the visible set out exactly as a single node of
 *   the same mass would, and — the physical claim underneath that — as the
 *   expanded subtree itself does.
 * - **Buffer identity is stable.** A collapse or an expand is a contents
 *   write; the GPUBuffer object never changes, so no bind group is ever
 *   rebuilt and the ping-pong parity machine is untouched.
 *
 * Every fixture here is edge-free. The spring pass accumulates through a
 * compare-exchange loop whose float addition order is not reproducible, and
 * these are exact-equality tests.
 */

import {
  assert,
  assertAlmostEquals,
  assertEquals,
  assertNotEquals,
  assertStrictEquals,
} from "jsr:@std/assert@^1";
import {
  createAlgorithmSimHarness,
  createSimHarness,
  GPU_SKIP_MESSAGE,
  HARNESS_ALPHA_DECAY,
  type HarnessForceAlgorithm,
  loadModuleInliningWgsl,
  loadPipelineModule,
  NODE_FLAG_HIDDEN_LOD,
  NODE_FLAG_PINNED,
  probeAdapter,
  requestHarnessDevice,
} from "../helpers/gpu.ts";
import { assertVisibleSetMatchesReference, countNonFinite } from "../helpers/invariants.ts";
import { generateCodeTree } from "../fixtures/code_tree.ts";
import {
  type FullForceConfig,
  validateForceConfig,
} from "../../packages/core/src/simulation/config.ts";
import { NODE_MASS_HIDDEN, NODE_MASS_UNIT } from "../../packages/core/src/lod/mass.ts";

const adapter = await probeAdapter();
if (!adapter) {
  console.warn(`[gpu] ${GPU_SKIP_MESSAGE}`);
}

/** The Karras tree layout needs 10 storage buffers; production asks for that. */
const REQUIRED_STORAGE_BUFFERS = 10;
const barnesHutSupported = adapter !== null &&
  adapter.limits.maxStorageBuffersPerShaderStage >= REQUIRED_STORAGE_BUFFERS;

function gpuTest(
  name: string,
  fn: (device: GPUDevice) => Promise<void>,
  options: { needsBarnesHut?: boolean } = {},
): void {
  Deno.test({
    name,
    ignore: adapter === null || (options.needsBarnesHut === true && !barnesHutSupported),
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

const REPULSION_N2_URL = new URL(
  "../../packages/core/src/simulation/shaders/repulsion_n2.comp.wgsl",
  import.meta.url,
);

/**
 * The mass reads in the shipped shader, and what removing them leaves. Applied
 * as text so the SC-005 baseline is always "the code under test, minus mass" —
 * it cannot drift into a stale copy of an older physics.
 */
const MASS_STRIPS: ReadonlyArray<readonly [string, string]> = [
  [", other_mass: f32)", ")"],
  [" * other_mass /", " /"],
  [", node_mass[i])", ")"],
];

/** The shipped repulsion shader with every per-node mass read removed. */
function stripMass(shipped: string): string {
  let stripped = shipped;
  for (const [from, to] of MASS_STRIPS) {
    assert(
      stripped.includes(from),
      `SC-005 baseline is stale: "${from}" is no longer in the shader`,
    );
    stripped = stripped.replaceAll(from, to);
  }
  assert(!stripped.includes("node_mass["), "no mass read may survive in the reference");
  assert(!stripped.includes("other_mass"), "no mass parameter may survive in the reference");
  return stripped;
}

function loadBarnesHut(): Promise<HarnessForceAlgorithm> {
  return loadModuleInliningWgsl<{ createBarnesHutAlgorithm(): HarnessForceAlgorithm }>(
    new URL("../../packages/core/src/simulation/algorithms/barnes-hut.ts", import.meta.url),
  ).then((m) => m.createBarnesHutAlgorithm());
}

function loadN2(): Promise<HarnessForceAlgorithm> {
  return loadModuleInliningWgsl<{ createN2Algorithm(): HarnessForceAlgorithm }>(
    new URL("../../packages/core/src/simulation/algorithms/n2.ts", import.meta.url),
  ).then((m) => m.createN2Algorithm());
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

interface EdgeFreeGraph {
  nodeCount: number;
  positionsX: Float32Array;
  positionsY: Float32Array;
  flags?: Uint32Array;
  mass?: Float32Array;
}

/**
 * Runs `ticks` simulation steps over an edge-free graph, taking the repulsion
 * pass from an arbitrary WGSL source so a modified shader can serve as a
 * reference. Everything else — clear, integrate, the alpha schedule, the
 * ping-pong swap — is the shipped pipeline.
 */
async function runWithRepulsionShader(
  device: GPUDevice,
  graph: EdgeFreeGraph,
  repulsionCode: string,
  bindsMass: boolean,
  forceConfig: FullForceConfig,
  ticks: number,
): Promise<{ x: Float32Array; y: Float32Array }> {
  const mod = await loadPipelineModule();
  const { nodeCount } = graph;

  const pipeline = mod.createSimulationPipeline({ device }, { maxNodes: nodeCount, maxEdges: 1 });
  const buffers = mod.createSimulationBuffers(device, nodeCount, 0);
  mod.copyPositionsToSimulation(device, buffers, graph.positionsX, graph.positionsY);
  if (graph.flags) device.queue.writeBuffer(buffers.nodeFlags, 0, graph.flags.slice().buffer);
  if (graph.mass) device.queue.writeBuffer(buffers.nodeMass, 0, graph.mass.slice().buffer);

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
      forceConfig,
    );
    // Rebuilt per tick because the ping-pong buffers rotate; correctness over
    // speed, and the parity machine is covered by its own suite.
    const bindGroups = mod.createSimulationBindGroups(device, pipeline, buffers);
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: buffers.repulsionUniforms } },
      { binding: 1, resource: { buffer: buffers.positions } },
      { binding: 2, resource: { buffer: buffers.forces } },
      { binding: 3, resource: { buffer: buffers.nodeFlags } },
    ];
    if (bindsMass) entries.push({ binding: 4, resource: { buffer: buffers.nodeMass } });
    const repulsionBindGroup = device.createBindGroup({
      layout: repulsion.getBindGroupLayout(0),
      entries,
    });

    const encoder = device.createCommandEncoder();
    mod.recordSimulationStepWithOptions(encoder, pipeline, bindGroups, nodeCount, 0, {
      recordRepulsionPass: (enc) => {
        const pass = enc.beginComputePass({ label: "Repulsion Under Test" });
        pass.setPipeline(repulsion);
        pass.setBindGroup(0, repulsionBindGroup);
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

  for (
    const buffer of [
      buffers.positions,
      buffers.positionsOut,
      buffers.velocities,
      buffers.velocitiesOut,
      buffers.forces,
      buffers.prevForces,
      buffers.edgeSources,
      buffers.edgeTargets,
      buffers.clearUniforms,
      buffers.repulsionUniforms,
      buffers.springUniforms,
      buffers.integrationUniforms,
      buffers.nodeFlags,
      buffers.nodeAlpha,
      buffers.nodeMass,
      buffers.nodeDepth,
      buffers.readback,
    ]
  ) {
    buffer.destroy();
  }
  return { x, y };
}

// =============================================================================
// SC-005 — LOD off is bit-identical to the mass-free shader
// =============================================================================

gpuTest("nodeMass: a freshly allocated buffer holds exactly unit mass", async (device) => {
  const mod = await loadPipelineModule();
  const nodeCount = 137; // not a workgroup multiple: the tail must be filled too
  const buffers = mod.createSimulationBuffers(device, nodeCount, 0);

  const readback = device.createBuffer({
    size: nodeCount * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffers.nodeMass, 0, readback, 0, nodeCount * 4);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const mass = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  assertBitIdentical(
    mass,
    new Float32Array(nodeCount).fill(NODE_MASS_UNIT),
    "fresh nodeMass",
  );
});

gpuTest("SC-005: with unit mass, 30 ticks are bit-identical to the mass-free shader", async (
  device,
) => {
  const shipped = await Deno.readTextFile(REPULSION_N2_URL);
  const stripped = stripMass(shipped);
  assertNotEquals(stripped, shipped, "the mass reads must actually have been stripped");

  const tree = generateCodeTree({ seed: 3, maxNodes: 400 });
  const graph: EdgeFreeGraph = {
    nodeCount: tree.nodeCount,
    positionsX: tree.positionsX,
    positionsY: tree.positionsY,
  };
  const config = validateForceConfig({});

  const withMass = await runWithRepulsionShader(device, graph, shipped, true, config, 30);
  const massFree = await runWithRepulsionShader(device, graph, stripped, false, config, 30);
  const withMassAgain = await runWithRepulsionShader(device, graph, shipped, true, config, 30);

  // A rig that is not reproducible run-to-run cannot prove bit-identity.
  assertBitIdentical(withMassAgain.x, withMass.x, "self-consistency x");
  assertBitIdentical(withMassAgain.y, withMass.y, "self-consistency y");

  assertEquals(countNonFinite(withMass.x, withMass.y), 0);
  assertBitIdentical(withMass.x, massFree.x, "SC-005 x");
  assertBitIdentical(withMass.y, massFree.y, "SC-005 y");
});

// =============================================================================
// Mass scales repulsion
// =============================================================================

/** Two nodes on the x axis; returns each node's displacement after one tick. */
async function twoBodyDisplacement(
  device: GPUDevice,
  otherMass: number,
): Promise<{ moved: number; other: number }> {
  const shipped = await Deno.readTextFile(REPULSION_N2_URL);
  const config = validateForceConfig({ centerStrength: 0 });
  const graph: EdgeFreeGraph = {
    nodeCount: 2,
    positionsX: new Float32Array([-5, 5]),
    positionsY: new Float32Array([0, 0]),
    mass: new Float32Array([NODE_MASS_UNIT, otherMass]),
  };
  const { x } = await runWithRepulsionShader(device, graph, shipped, true, config, 1);
  return { moved: -5 - x[0], other: x[1] - 5 };
}

gpuTest("nodeMass scales the force a node exerts, not the force it receives", async (device) => {
  const unit = await twoBodyDisplacement(device, NODE_MASS_UNIT);
  const heavy = await twoBodyDisplacement(device, 4 * NODE_MASS_UNIT);

  assert(unit.moved > 0, `the light node must be pushed away, got ${unit.moved}`);
  // Node 0's force comes entirely from node 1, scaled by node 1's mass.
  assertAlmostEquals(
    heavy.moved / unit.moved,
    4,
    1e-3,
    "quadrupling the source mass must quadruple the force it exerts",
  );
  // Node 1's own force depends on node 0's mass, which did not change.
  assertAlmostEquals(
    heavy.other,
    unit.other,
    1e-6,
    "a node's own mass must not change the force it receives",
  );
});

// =============================================================================
// SC-002 — collapsing a subtree does not move the visible set
// =============================================================================

/** Ring node count in the SC-002 fixtures; these are the visible slots. */
const RING_COUNT = 8;
const RING_RADIUS = 40;
/** Hidden subtree size, and therefore the proxy's rolled-up mass. */
const SUBTREE_SIZE = 16;
const SUBTREE_SPREAD = 0.3;

const RING_INDICES = Uint32Array.from({ length: RING_COUNT }, (_, i) => i);

function ringPositions(): { x: number[]; y: number[] } {
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < RING_COUNT; i++) {
    const angle = (2 * Math.PI * i) / RING_COUNT;
    x.push(RING_RADIUS * Math.cos(angle));
    y.push(RING_RADIUS * Math.sin(angle));
  }
  return { x, y };
}

/**
 * A tight cluster whose centroid is exactly the origin: `SUBTREE_SIZE` points
 * in diametrically opposite pairs, so the proxy that replaces it stands at the
 * centre of mass rather than merely near it.
 */
function clusterPositions(): { x: number[]; y: number[] } {
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < SUBTREE_SIZE / 2; i++) {
    const angle = (Math.PI * i) / (SUBTREE_SIZE / 2);
    const dx = SUBTREE_SPREAD * Math.cos(angle);
    const dy = SUBTREE_SPREAD * Math.sin(angle);
    x.push(dx, -dx);
    y.push(dy, -dy);
  }
  return { x, y };
}

const SC002_CONFIG = validateForceConfig({ centerStrength: 0 });
const SC002_TICKS = 25;

/** The collapsed run: proxy at the centroid carrying the whole subtree's mass. */
function collapsedGraph(): EdgeFreeGraph {
  const ring = ringPositions();
  const cluster = clusterPositions();
  const nodeCount = RING_COUNT + SUBTREE_SIZE;
  const flags = new Uint32Array(nodeCount);
  const mass = new Float32Array(nodeCount).fill(NODE_MASS_UNIT);

  // Slot RING_COUNT is the subtree root, standing at the centroid; the rest of
  // the subtree is hidden and weightless — exactly what rollUpMass produces.
  const x = [...ring.x, 0, ...cluster.x.slice(1)];
  const y = [...ring.y, 0, ...cluster.y.slice(1)];
  mass[RING_COUNT] = SUBTREE_SIZE;
  for (let i = RING_COUNT + 1; i < nodeCount; i++) {
    flags[i] = NODE_FLAG_HIDDEN_LOD | NODE_FLAG_PINNED;
    mass[i] = NODE_MASS_HIDDEN;
  }
  flags[RING_COUNT] = NODE_FLAG_PINNED;

  return {
    nodeCount,
    positionsX: new Float32Array(x),
    positionsY: new Float32Array(y),
    flags,
    mass,
  };
}

gpuTest("SC-002: a collapsed subtree lays out like one node of the same mass", async (device) => {
  const shipped = await Deno.readTextFile(REPULSION_N2_URL);
  const ring = ringPositions();

  // Reference: the subtree is simply not there, replaced by a single node of
  // the same mass at the same place.
  const proxyOnly: EdgeFreeGraph = {
    nodeCount: RING_COUNT + 1,
    positionsX: new Float32Array([...ring.x, 0]),
    positionsY: new Float32Array([...ring.y, 0]),
    flags: Uint32Array.from(
      { length: RING_COUNT + 1 },
      (_, i) => i === RING_COUNT ? NODE_FLAG_PINNED : 0,
    ),
    mass: Float32Array.from(
      { length: RING_COUNT + 1 },
      (_, i) => (i === RING_COUNT ? SUBTREE_SIZE : NODE_MASS_UNIT),
    ),
  };

  const collapsed = await runWithRepulsionShader(
    device,
    collapsedGraph(),
    shipped,
    true,
    SC002_CONFIG,
    SC002_TICKS,
  );
  const reference = await runWithRepulsionShader(
    device,
    proxyOnly,
    shipped,
    true,
    SC002_CONFIG,
    SC002_TICKS,
  );

  assertEquals(countNonFinite(collapsed.x, collapsed.y), 0);
  assertVisibleSetMatchesReference(collapsed, reference, RING_INDICES, 1e-4, RING_INDICES);
});

gpuTest("SC-002: a collapsed subtree lays out like the expanded subtree", async (device) => {
  const shipped = await Deno.readTextFile(REPULSION_N2_URL);
  const ring = ringPositions();
  const cluster = clusterPositions();
  const nodeCount = RING_COUNT + SUBTREE_SIZE;

  // Reference: every subtree node present, visible, unit mass, pinned so the
  // comparison is of the far field alone and not of a cluster blowing apart.
  const expanded: EdgeFreeGraph = {
    nodeCount,
    positionsX: new Float32Array([...ring.x, ...cluster.x]),
    positionsY: new Float32Array([...ring.y, ...cluster.y]),
    flags: Uint32Array.from(
      { length: nodeCount },
      (_, i) => i >= RING_COUNT ? NODE_FLAG_PINNED : 0,
    ),
  };

  const collapsed = await runWithRepulsionShader(
    device,
    collapsedGraph(),
    shipped,
    true,
    SC002_CONFIG,
    SC002_TICKS,
  );
  const reference = await runWithRepulsionShader(
    device,
    expanded,
    shipped,
    true,
    SC002_CONFIG,
    SC002_TICKS,
  );

  // The residual is the multipole error of standing 16 bodies of spread 0.3 at
  // their centroid, seen from 40 units away — parts in 10^4 of the ring radius.
  // A proxy that repelled like a single node would leave the ring ~14 units
  // short of here, four orders of magnitude outside this bound.
  assertVisibleSetMatchesReference(collapsed, reference, RING_INDICES, 0.02, RING_INDICES);
});

// =============================================================================
// Buffer identity
// =============================================================================

gpuTest("nodeMass: collapse and expand never change the buffer identity", async (device) => {
  const tree = generateCodeTree({ seed: 11, maxNodes: 200 });
  const harness = await createSimHarness(device, {
    nodeCount: tree.nodeCount,
    positionsX: tree.positionsX,
    positionsY: tree.positionsY,
    edgeSources: tree.edgeSources,
    edgeTargets: tree.edgeTargets,
  });
  try {
    const initial = harness.nodeMassBuffer;
    await harness.tick(3);

    // Collapse: half the graph rolls into slot 0.
    const collapsed = new Float32Array(tree.nodeCount).fill(NODE_MASS_UNIT);
    const hidden = tree.nodeCount >> 1;
    collapsed[0] = 1 + hidden;
    collapsed.fill(NODE_MASS_HIDDEN, 1, 1 + hidden);
    harness.setNodeMass(collapsed);
    await harness.tick(3);
    assertStrictEquals(harness.nodeMassBuffer, initial, "a collapse must not reallocate nodeMass");

    // Expand again.
    harness.setNodeMass(new Float32Array(tree.nodeCount).fill(NODE_MASS_UNIT));
    await harness.tick(3);
    assertStrictEquals(harness.nodeMassBuffer, initial, "an expand must not reallocate nodeMass");

    const after = await harness.readPositions();
    assertEquals(countNonFinite(after.x, after.y), 0);

    // A reallocation is the one thing that *does* replace it, and it must come
    // back at unit mass rather than zero-filled.
    await harness.reallocate();
    assert(harness.nodeMassBuffer !== initial, "reallocation replaces the buffer");
    await harness.tick(3);
    const grown = await harness.readPositions();
    assertEquals(countNonFinite(grown.x, grown.y), 0);
  } finally {
    harness.dispose();
  }
});

// =============================================================================
// Algorithm plugins
// =============================================================================

/**
 * One repulsion pass of an algorithm plugin over static positions, with the
 * per-node mass buffer optionally withheld so the plugin's fallback is what
 * gets exercised.
 */
async function algorithmForcesOnce(
  device: GPUDevice,
  algorithm: HarnessForceAlgorithm,
  xs: Float32Array,
  ys: Float32Array,
  mass: Float32Array | null,
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
  const forces = device.createBuffer({
    size: n * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const nodeFlags = device.createBuffer({
    size: n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  let nodeMass: GPUBuffer | undefined;
  if (mass) {
    nodeMass = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(nodeMass, 0, mass.slice().buffer);
  }
  const readback = device.createBuffer({
    size: n * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    minX = Math.min(minX, xs[i]);
    maxX = Math.max(maxX, xs[i]);
    minY = Math.min(minY, ys[i]);
    maxY = Math.max(maxY, ys[i]);
  }

  const context = {
    device,
    positions,
    forces,
    nodeCount: n,
    edgeCount: 0,
    forceConfig,
    bounds: { minX: minX - 1, minY: minY - 1, maxX: maxX + 1, maxY: maxY + 1 },
    nodeFlags,
    nodeMass,
  };

  const pipelines = algorithm.createPipelines({ device });
  const algoBuffers = algorithm.createBuffers(device, n);
  const bindGroups = algorithm.createBindGroups(device, pipelines, context, algoBuffers);
  algorithm.updateUniforms(device, algoBuffers, context);

  const encoder = device.createCommandEncoder();
  algorithm.recordRepulsionPass(encoder, pipelines, bindGroups, n);
  encoder.copyBufferToBuffer(forces, 0, readback, 0, n * 8);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  algoBuffers.destroy();
  positions.destroy();
  forces.destroy();
  nodeFlags.destroy();
  nodeMass?.destroy();
  readback.destroy();
  return result;
}

gpuTest("Barnes-Hut consumes nodeMass at the leaves", async (device) => {
  const algorithm = await loadBarnesHut();
  // theta this small forces the traversal down to individual leaves, so the
  // expected forces are the exact Coulomb sums rather than an approximation.
  const config = validateForceConfig({ theta: 0.0001 });
  const k = Math.abs(config.repulsionStrength);
  const xs = new Float32Array([-20, 0, 20]);
  const ys = new Float32Array([0, 0, 0]);

  for (const middleMass of [1, 4]) {
    const forces = await algorithmForcesOnce(
      device,
      algorithm,
      xs,
      ys,
      new Float32Array([NODE_MASS_UNIT, middleMass, NODE_MASS_UNIT]),
      config,
    );

    // Probe 0 is pushed in -x by the middle body (mass m at 20) and by probe 2
    // (unit mass at 40).
    const expected = -k * (middleMass / 400 + 1 / 1600);
    assertAlmostEquals(forces[0], expected, 1e-4, `probe force at middle mass ${middleMass}`);
    assertAlmostEquals(forces[4], -expected, 1e-4, "the configuration is symmetric");
    assertAlmostEquals(forces[1], 0, 1e-4, "no y force on a collinear configuration");

    // The middle body's own leaf is the only mass coincident with it, so it
    // must receive no separation impulse — the subtraction that isolates it
    // has to use the body's own mass, not a hardcoded 1.0.
    assertAlmostEquals(forces[2], 0, 1e-3, `middle-body x force at mass ${middleMass}`);
    assertAlmostEquals(forces[3], 0, 1e-3, `middle-body y force at mass ${middleMass}`);
  }
}, { needsBarnesHut: true });

gpuTest("Barnes-Hut and N² fall back to unit mass, never to zero mass", async (device) => {
  const config = validateForceConfig({ theta: 0.0001 });
  const tree = generateCodeTree({ seed: 5, maxNodes: 300 });
  const unit = new Float32Array(tree.nodeCount).fill(NODE_MASS_UNIT);

  for (const load of [loadN2, loadBarnesHut]) {
    if (load === loadBarnesHut && !barnesHutSupported) continue;
    const algorithm = await load();
    const supplied = await algorithmForcesOnce(
      device,
      algorithm,
      tree.positionsX,
      tree.positionsY,
      unit,
      config,
    );
    const fallback = await algorithmForcesOnce(
      device,
      algorithm,
      tree.positionsX,
      tree.positionsY,
      null,
      config,
    );
    let nonZero = 0;
    for (const f of supplied) {
      if (f !== 0) nonZero++;
    }
    assert(nonZero > 0, `${algorithm.info.id}: the probe must produce forces at all`);
    assertBitIdentical(fallback, supplied, `${algorithm.info.id} fallback mass`);
  }
});

gpuTest("Barnes-Hut: a collapsed subtree lays out like the expanded subtree", async (device) => {
  const algorithm = await loadBarnesHut();
  const ring = ringPositions();
  const cluster = clusterPositions();
  const nodeCount = RING_COUNT + SUBTREE_SIZE;
  const collapsed = collapsedGraph();

  const run = async (graph: EdgeFreeGraph) => {
    const harness = await createAlgorithmSimHarness(
      device,
      algorithm,
      {
        nodeCount: graph.nodeCount,
        positionsX: graph.positionsX,
        positionsY: graph.positionsY,
        // A self-loop: the spring pass skips it at zero distance, so the run
        // stays edge-free in effect without a zero-length buffer write.
        edgeSources: new Uint32Array([0]),
        edgeTargets: new Uint32Array([0]),
        ...(graph.flags ? { flags: graph.flags } : {}),
        ...(graph.mass ? { mass: graph.mass } : {}),
      },
      { centerStrength: 0, theta: 0.0001 },
      HARNESS_ALPHA_DECAY,
      { boundsSyncInterval: 5 },
    );
    try {
      await harness.tick(SC002_TICKS);
      return await harness.readPositions();
    } finally {
      harness.dispose();
    }
  };

  const expanded: EdgeFreeGraph = {
    nodeCount,
    positionsX: new Float32Array([...ring.x, ...cluster.x]),
    positionsY: new Float32Array([...ring.y, ...cluster.y]),
    flags: Uint32Array.from(
      { length: nodeCount },
      (_, i) => i >= RING_COUNT ? NODE_FLAG_PINNED : 0,
    ),
  };

  const collapsedRun = await run(collapsed);
  const expandedRun = await run(expanded);

  assertEquals(countNonFinite(collapsedRun.x, collapsedRun.y), 0);
  assertVisibleSetMatchesReference(collapsedRun, expandedRun, RING_INDICES, 0.05, RING_INDICES);
}, { needsBarnesHut: true });
