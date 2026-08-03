/**
 * GPU tests for the LOD physics guarantees under the ALGORITHM PLUGINS.
 *
 * WP-K gave the shared spring pass an aggregated edge set so that collapsing a
 * subtree transfers its cross-cutting attractions to the proxy instead of
 * deleting them (SC-002). Every algorithm that supplies its own attraction —
 * ForceAtlas2, LinLog, t-FDP, Relativity Atlas — kept the bug: their per-edge
 * kernels skip any edge with a hidden endpoint, so under `setForceAlgorithm`
 * the collapsed layout still diverged structurally from the expanded one.
 *
 * Four claims, one per section:
 *
 * - **SC-005, exactly.** With the LOD terms textually removed, each attraction
 *   shader's un-cut entry point computes the same bits. The baseline is the
 *   code under test minus this change, so it cannot drift into a stale copy.
 * - **SC-005, end to end.** An identity aggregation — every edge live, no
 *   bundles — is bit-identical to no aggregation at all, on a degree-one
 *   fixture where the accumulation order is fixed.
 * - **SC-002.** A collapsed pair joined only by hidden imports still pulls,
 *   and the collapsed frontier lands much closer to where the expanded run put
 *   it than dropping the imports does. Run for every plugin, with the
 *   tolerance argued from that plugin's force law.
 * - **The N² plugin has no phantom bodies.** A removed node's slot sits at the
 *   origin inside the high-water mark; under the old unmasked entry point it
 *   repelled every survivor outward from a point where nothing was drawn.
 */

import { assert, assertEquals, assertStrictEquals } from "jsr:@std/assert@^1";
import {
  createAlgorithmSimHarness,
  GPU_SKIP_MESSAGE,
  type HarnessForceAlgorithm,
  type HarnessGraphData,
  loadModuleInliningWgsl,
  NODE_FLAG_DEAD,
  NODE_FLAG_HIDDEN_LOD,
  probeAdapter,
  requestHarnessDevice,
  type SimHarness,
} from "../helpers/gpu.ts";
import { assertFrozen, countNonFinite, type PositionSnapshot } from "../helpers/invariants.ts";
import { referenceAggregateLodEdges } from "../helpers/edge_aggregation.ts";
import { decodeEdgeAggregation } from "../../packages/core/src/lod/edge_aggregation.ts";
import { HIERARCHY_ROOT } from "../../packages/core/src/graph/hierarchy.ts";
import {
  type FullForceConfig,
  validateForceConfig,
} from "../../packages/core/src/simulation/config.ts";

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
// Section 1 — SC-005, exactly: `main` computes the pre-change bits
// =============================================================================

const SHADER_DIR = new URL("../../packages/core/src/simulation/shaders/", import.meta.url);

/**
 * How to reduce one attraction shader to the code it was before this work
 * package, and how to drive it.
 *
 * The reduction is applied as text to the shipped source, so the SC-005
 * baseline is always "the shader under test, minus the change" — there is no
 * second copy to go stale. `survivors` are the code references that prove the
 * reduction actually happened.
 */
interface AttractionShader {
  readonly name: string;
  readonly file: string;
  /** Bytes of the uniform struct `main` binds. */
  readonly uniformBytes: number;
  /** Fills the uniform buffer for a run of `edgeCount` edges. */
  writeUniforms(view: DataView, edgeCount: number): void;
  /** Does `main` read an `edge_weights` buffer? (t-FDP has none.) */
  readonly bindsEdgeWeights: boolean;
  /** Text between these markers is the LOD binding block, removed wholesale. */
  readonly strips: ReadonlyArray<readonly [string, string]>;
}

/** Marker text bounding the LOD binding block in every attraction shader. */
const LOD_SECTION_START = "// --- LOD edge aggregation";
const LOD_SECTION_END = "const NODE_FLAG_DEAD";
/** Everything from here down is the bundled entry point. */
const BUNDLED_ENTRY_START = "// Attraction under a live LOD cut.";
/** The `inverse_mass` helper, which only the bundled path calls. */
const INVERSE_MASS_START = "// Reciprocal of the mass";
const INVERSE_MASS_END = "// Race-free float accumulation";
/**
 * The paragraphs of `apply_attraction`'s doc comment that explain the LOD
 * terms, removed down to the signature. Prose, not code — but the reduction has
 * to leave a source in which no LOD identifier appears at all, so that the
 * "nothing survived" assertions below can be blunt.
 */
const LOD_DOC_START = "// `count` is how many source edges";
const LOD_DOC_END = "fn apply_attraction(";

/** Code references that must not survive the reduction. */
const FORBIDDEN_IN_BASELINE = [
  "* count",
  "inv_mass",
  "node_mass[",
  "lod_edge_set[",
];

const ATTRACTION_SHADERS: readonly AttractionShader[] = [
  {
    name: "fa2_attraction",
    file: "fa2_attraction.comp.wgsl",
    uniformBytes: 32,
    bindsEdgeWeights: true,
    writeUniforms(view, edgeCount) {
      view.setUint32(0, edgeCount, true); // edge_count
      view.setFloat32(4, 1.0, true); // edge_weight_influence
      view.setUint32(8, 0, true); // flags: linear branch
    },
    strips: [
      [
        "    factor: f32,\n    count: f32,\n    source_inv_mass: f32,\n" +
        "    target_inv_mass: f32,\n",
        "    factor: f32,\n",
      ],
      ["factor * log(1.0 + dist) * count", "factor * log(1.0 + dist)"],
      ["factor * dist * count", "factor * dist"],
      [
        "accumulate_force(src, force * source_inv_mass);\n" +
        "    accumulate_force(tgt, -force * target_inv_mass);",
        "accumulate_force(src, force);\n    accumulate_force(tgt, -force);",
      ],
      [
        "        weight_factor(edge_weights[edge_idx]),\n        1.0,\n        1.0,\n        1.0,\n",
        "        weight_factor(edge_weights[edge_idx]),\n",
      ],
    ],
  },
  {
    name: "linlog_attraction",
    file: "linlog_attraction.comp.wgsl",
    uniformBytes: 48,
    bindsEdgeWeights: true,
    writeUniforms(view, edgeCount) {
      view.setUint32(0, 0, true); // node_count (unused here)
      view.setUint32(4, edgeCount, true); // edge_count
      view.setFloat32(16, 1.0, true); // edge_weight_influence
    },
    strips: [
      [
        "    factor: f32,\n    count: f32,\n    source_inv_mass: f32,\n" +
        "    target_inv_mass: f32,\n",
        "    factor: f32,\n",
      ],
      ["factor * log(1.0 + dist) * count", "factor * log(1.0 + dist)"],
      [
        "accumulate_force(src, force * source_inv_mass);\n" +
        "    accumulate_force(tgt, -force * target_inv_mass);",
        "accumulate_force(src, force);\n    accumulate_force(tgt, -force);",
      ],
      [
        "        weight_factor(edge_weights[edge_idx]),\n        1.0,\n        1.0,\n        1.0,\n",
        "        weight_factor(edge_weights[edge_idx]),\n",
      ],
    ],
  },
  {
    name: "t_fdp_attraction",
    file: "t_fdp_attraction.comp.wgsl",
    uniformBytes: 32,
    bindsEdgeWeights: false,
    writeUniforms(view, edgeCount) {
      view.setUint32(0, edgeCount, true); // edge_count
      view.setFloat32(4, 0.1, true); // alpha
      view.setFloat32(8, 8.0, true); // beta
      view.setFloat32(12, 60.0, true); // dist_scale
    },
    strips: [
      [
        "    count: f32,\n    source_inv_mass: f32,\n    target_inv_mass: f32,\n",
        "",
      ],
      ["(spring_force + t_force) * count", "spring_force + t_force"],
      [
        "accumulate_force(src, force * source_inv_mass);\n" +
        "    accumulate_force(tgt, -force * target_inv_mass);",
        "accumulate_force(src, force);\n    accumulate_force(tgt, -force);",
      ],
      [
        "apply_attraction(edge_sources[edge_idx], edge_targets[edge_idx], 1.0, 1.0, 1.0);",
        "apply_attraction(edge_sources[edge_idx], edge_targets[edge_idx]);",
      ],
    ],
  },
];

/** The shipped shader with everything this work package added removed. */
function stripLod(shipped: string, shader: AttractionShader): string {
  const bundledAt = shipped.indexOf(BUNDLED_ENTRY_START);
  assert(bundledAt > 0, `${shader.name}: SC-005 baseline is stale, main_bundled is gone`);
  let source = shipped.slice(0, bundledAt);

  for (
    const [start, end] of [
      [LOD_SECTION_START, LOD_SECTION_END],
      [INVERSE_MASS_START, INVERSE_MASS_END],
      [LOD_DOC_START, LOD_DOC_END],
    ]
  ) {
    const lo = source.indexOf(start);
    const hi = source.indexOf(end);
    assert(lo > 0 && hi > lo, `${shader.name}: SC-005 baseline is stale, ${start} moved`);
    source = source.slice(0, lo) + source.slice(hi);
  }

  for (const [from, to] of shader.strips) {
    assert(
      source.includes(from),
      `${shader.name}: SC-005 baseline is stale, ${JSON.stringify(from.slice(0, 48))} is gone`,
    );
    source = source.replace(from, to);
  }

  for (const survivor of FORBIDDEN_IN_BASELINE) {
    assert(
      !source.includes(survivor),
      `${shader.name}: "${survivor}" must not survive in the reference`,
    );
  }
  return source;
}

/** Two rings of nodes joined edge-for-edge — hub-free, so degree stays low. */
function ladder(nodeCount: number): {
  positions: Float32Array;
  edgeSources: Uint32Array;
  edgeTargets: Uint32Array;
} {
  const positions = new Float32Array(nodeCount * 2);
  for (let i = 0; i < nodeCount; i++) {
    const angle = (i / nodeCount) * Math.PI * 2;
    positions[i * 2] = Math.cos(angle) * (60 + (i % 3) * 17);
    positions[i * 2 + 1] = Math.sin(angle) * (60 + (i % 5) * 11);
  }
  const pairs = nodeCount >> 1;
  const edgeSources = new Uint32Array(pairs);
  const edgeTargets = new Uint32Array(pairs);
  for (let k = 0; k < pairs; k++) {
    edgeSources[k] = k * 2;
    edgeTargets[k] = k * 2 + 1;
  }
  return { positions, edgeSources, edgeTargets };
}

/**
 * Runs one attraction dispatch from arbitrary WGSL and reads the force buffer.
 *
 * `layout: "auto"` follows the source: `main` reaches bindings 0-6 (or 0-5
 * without edge weights) in both the shipped and the reduced shader, so both
 * bind groups are the same shape.
 */
async function attractionForces(
  device: GPUDevice,
  shader: AttractionShader,
  code: string,
  graph: { positions: Float32Array; edgeSources: Uint32Array; edgeTargets: Uint32Array },
): Promise<Float32Array> {
  const nodeCount = graph.positions.length / 2;
  const edgeCount = graph.edgeSources.length;

  const storage = (label: string, bytes: number): GPUBuffer =>
    device.createBuffer({
      label,
      size: Math.max(bytes, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

  const positions = storage("attraction positions", nodeCount * 8);
  const forces = storage("attraction forces", nodeCount * 8);
  const edgeSources = storage("attraction edge sources", edgeCount * 4);
  const edgeTargets = storage("attraction edge targets", edgeCount * 4);
  const edgeWeights = storage("attraction edge weights", edgeCount * 4);
  const nodeFlags = storage("attraction node flags", nodeCount * 4);
  const uniforms = device.createBuffer({
    label: "attraction uniforms",
    size: shader.uniformBytes,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const readback = device.createBuffer({
    label: "attraction readback",
    size: nodeCount * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  device.queue.writeBuffer(positions, 0, graph.positions.slice().buffer);
  device.queue.writeBuffer(edgeSources, 0, graph.edgeSources.slice().buffer);
  device.queue.writeBuffer(edgeTargets, 0, graph.edgeTargets.slice().buffer);
  device.queue.writeBuffer(edgeWeights, 0, new Float32Array(edgeCount).fill(1).buffer);
  const uniformData = new ArrayBuffer(shader.uniformBytes);
  shader.writeUniforms(new DataView(uniformData), edgeCount);
  device.queue.writeBuffer(uniforms, 0, uniformData);

  const pipeline = device.createComputePipeline({
    label: `${shader.name} under test`,
    layout: "auto",
    compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
  });

  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: uniforms } },
    { binding: 1, resource: { buffer: positions } },
    { binding: 2, resource: { buffer: forces } },
    { binding: 3, resource: { buffer: edgeSources } },
    { binding: 4, resource: { buffer: edgeTargets } },
  ];
  if (shader.bindsEdgeWeights) {
    entries.push({ binding: 5, resource: { buffer: edgeWeights } });
    entries.push({ binding: 6, resource: { buffer: nodeFlags } });
  } else {
    entries.push({ binding: 5, resource: { buffer: nodeFlags } });
  }

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass({ label: shader.name });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
  pass.dispatchWorkgroups(Math.ceil(edgeCount / 256));
  pass.end();
  encoder.copyBufferToBuffer(forces, 0, readback, 0, nodeCount * 8);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  for (
    const buffer of [
      positions,
      forces,
      edgeSources,
      edgeTargets,
      edgeWeights,
      nodeFlags,
      uniforms,
      readback,
    ]
  ) {
    buffer.destroy();
  }
  return result;
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

for (const shader of ATTRACTION_SHADERS) {
  gpuTest(
    `SC-005: ${shader.name} main is bit-identical to the pre-aggregation shader`,
    async (device) => {
      const shipped = await Deno.readTextFile(new URL(shader.file, SHADER_DIR));
      const reduced = stripLod(shipped, shader);
      assert(reduced !== shipped, "the LOD terms must actually have been stripped");

      const graph = ladder(96);
      const actual = await attractionForces(device, shader, shipped, graph);
      const reference = await attractionForces(device, shader, reduced, graph);

      assertEquals(countNonFinite(actual, actual), 0);
      // Non-vacuous: the pass must have produced forces at all.
      assert(actual.some((f) => f !== 0), "the attraction pass produced no force");
      assertBitIdentical(actual, reference, `${shader.name} forces`);
    },
  );
}

// =============================================================================
// Fixture: two modules of leaves, joined only by cross-cutting imports
// =============================================================================

/** Leaves per module. */
const LEAVES = 6;
/** Half the opening separation of the two module proxies. */
const HALF_SEPARATION = 200;

/**
 * Two module roots, `LEAVES` leaves each, containment edges from each module to
 * its own leaves, and an import from every leaf of one module to every leaf of
 * the other.
 *
 * No shared root, and nothing else in the graph: what holds the two modules at
 * a finite distance is the imports against repulsion, and nothing else. Every
 * import is leaf-to-leaf, so collapsing both modules hides BOTH endpoints —
 * exactly the set every per-edge attraction kernel drops — and the whole of the
 * structure the layout has is the thing under test.
 */
interface TwoModules {
  readonly nodeCount: number;
  readonly parent: Uint32Array;
  readonly positionsX: Float32Array;
  readonly positionsY: Float32Array;
  readonly edgeSources: Uint32Array;
  readonly edgeTargets: Uint32Array;
  readonly depths: Float32Array;
  readonly frontier: readonly number[];
  readonly collapsedFlags: Uint32Array;
  readonly collapsedVisible: Uint8Array;
  readonly collapsedMass: Float32Array;
  /** Just the imports, for the "does a collapsed pair pull at all" fixture. */
  readonly importSources: Uint32Array;
  readonly importTargets: Uint32Array;
}

/** Slot of module `module`'s `k`th leaf. */
function leafSlot(module: number, k: number): number {
  return 2 + module * LEAVES + k;
}

function twoModules(): TwoModules {
  const nodeCount = 2 + LEAVES * 2;
  const parent = new Uint32Array(nodeCount).fill(HIERARCHY_ROOT);
  const positionsX = new Float32Array(nodeCount);
  const positionsY = new Float32Array(nodeCount);
  const depths = new Float32Array(nodeCount);

  positionsX[0] = -HALF_SEPARATION;
  positionsX[1] = HALF_SEPARATION;

  for (let k = 0; k < LEAVES; k++) {
    const angle = (k / LEAVES) * Math.PI * 2;
    for (const module of [0, 1]) {
      const slot = leafSlot(module, k);
      parent[slot] = module;
      depths[slot] = 1;
      positionsX[slot] = positionsX[module] + Math.cos(angle) * 25;
      positionsY[slot] = positionsY[module] + Math.sin(angle) * 25;
    }
  }

  const sources: number[] = [];
  const targets: number[] = [];
  for (let k = 0; k < LEAVES; k++) {
    for (const module of [0, 1]) {
      sources.push(module);
      targets.push(leafSlot(module, k));
    }
  }
  const importSources: number[] = [];
  const importTargets: number[] = [];
  for (let i = 0; i < LEAVES; i++) {
    for (let j = 0; j < LEAVES; j++) {
      importSources.push(leafSlot(0, i));
      importTargets.push(leafSlot(1, j));
    }
  }
  sources.push(...importSources);
  targets.push(...importTargets);

  const collapsedFlags = new Uint32Array(nodeCount);
  const collapsedVisible = new Uint8Array(nodeCount);
  const collapsedMass = new Float32Array(nodeCount);
  collapsedVisible[0] = 1;
  collapsedVisible[1] = 1;
  collapsedMass[0] = 1 + LEAVES;
  collapsedMass[1] = 1 + LEAVES;
  for (let slot = 2; slot < nodeCount; slot++) collapsedFlags[slot] = NODE_FLAG_HIDDEN_LOD;

  return {
    nodeCount,
    parent,
    positionsX,
    positionsY,
    edgeSources: Uint32Array.from(sources),
    edgeTargets: Uint32Array.from(targets),
    depths,
    frontier: [0, 1],
    collapsedFlags,
    collapsedVisible,
    collapsedMass,
    importSources: Uint32Array.from(importSources),
    importTargets: Uint32Array.from(importTargets),
  };
}

/** The aggregation the reference walk produces for a cut, decoded. */
function aggregate(
  edgeSources: Uint32Array,
  edgeTargets: Uint32Array,
  fixture: TwoModules,
) {
  return decodeEdgeAggregation(
    referenceAggregateLodEdges(
      edgeSources,
      edgeTargets,
      fixture.parent,
      fixture.collapsedVisible,
    ),
  );
}

// =============================================================================
// The plugins under test
// =============================================================================

/** One algorithm plugin and what its force law implies for the assertions. */
interface PluginCase {
  readonly id: string;
  load(): Promise<HarnessForceAlgorithm>;
  /**
   * How much of the dropped-import divergence bundling must recover, as a
   * fraction. Argued per force law at the use site.
   */
  readonly sc002Recovery: number;
}

function loadPlugin(module: string, factory: string): () => Promise<HarnessForceAlgorithm> {
  return () =>
    loadModuleInliningWgsl<Record<string, () => HarnessForceAlgorithm>>(
      new URL(`../../packages/core/src/simulation/algorithms/${module}`, import.meta.url),
    ).then((m) => m[factory]());
}

/**
 * Recovery fractions: how much of the dropped-import error bundling must
 * remove, as a fraction of it.
 *
 * A bundle applies the plugin's own force law once, at the proxy separation,
 * multiplied by the number of edges it replaces. Superposition is exact — the
 * sum of forces is linear whatever the law is — so what each fraction really
 * measures is how badly the law reacts to being evaluated at one long
 * proxy-to-proxy separation instead of at many slightly different ones, plus
 * the residue of everything else the collapse changed.
 *
 * - **ForceAtlas2** is linear in the separation, so a bundle's pull is the
 *   exact sum of the edges it replaces whenever the subtree is small against
 *   the gap.
 * - **LinLog** is concave (log(1 + d)), so a bundle pulls slightly harder than
 *   the spread-out edges did and settles slightly closer; its bar is the
 *   loosest of the three.
 * - **t-FDP** carries a short-range term (beta·d/(1+d²)) meant to hold
 *   neighbours together, which decays over the proxy separation. Its
 *   long-range linear term aggregates exactly, and that term is what sets the
 *   separation being measured.
 *
 * Measured on this fixture: 0.032 / 0.048 / 0.035. The bars sit about three
 * times above that, which leaves room for driver-level float differences
 * without leaving room for the force to go missing again.
 */
const PLUGINS: readonly PluginCase[] = [
  {
    id: "force-atlas2",
    load: loadPlugin("force-atlas2.ts", "createForceAtlas2Algorithm"),
    sc002Recovery: 0.1,
  },
  {
    id: "linlog",
    load: loadPlugin("linlog.ts", "createLinLogAlgorithm"),
    sc002Recovery: 0.15,
  },
  {
    id: "t-fdp",
    load: loadPlugin("t-fdp.ts", "createTFdpAlgorithm"),
    sc002Recovery: 0.1,
  },
];

/**
 * Attraction and repulsion only: no centring gravity, so any motion the two
 * proxies show comes from the edge set under test.
 */
const PLUGIN_CONFIG: FullForceConfig = validateForceConfig({
  centerStrength: 0,
  repulsionStrength: 200,
  springLength: 40,
  velocityDecay: 0.4,
  collisionEnabled: false,
});

const SC002_TICKS = 60;

// =============================================================================
// Section 2 — SC-005 end to end: an identity aggregation changes nothing
// =============================================================================

/** Every edge, as a live-edge list. */
function identityEdgeList(edgeCount: number): Uint32Array {
  const list = new Uint32Array(edgeCount);
  for (let e = 0; e < edgeCount; e++) list[e] = e;
  return list;
}

/**
 * A perfect matching: node `2k` joined to node `2k + 1`, and nothing else.
 *
 * Degree one, and that is the point. Attraction accumulates through a
 * compare-exchange loop into an accumulator the repulsion pass has already
 * written, so a node receiving two pulls sums three terms in an order the GPU
 * does not guarantee — and float addition is not associative. One pull per node
 * makes the sum a fixed two terms, which removes every source of divergence
 * except the one below.
 */
function matchingGraph(nodeCount: number): HarnessGraphData {
  const { positions, edgeSources, edgeTargets } = ladder(nodeCount);
  const positionsX = new Float32Array(nodeCount);
  const positionsY = new Float32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    positionsX[i] = positions[i * 2];
    positionsY[i] = positions[i * 2 + 1];
  }
  return {
    nodeCount,
    positionsX,
    positionsY,
    edgeSources,
    edgeTargets,
    depths: new Float32Array(nodeCount),
  };
}

const SC005_TICKS = 5;

/**
 * How far an identity aggregation may land from the un-aggregated run.
 *
 * Not zero, and the reason is specific. The bundled entry point multiplies each
 * arriving force by the receiving endpoint's inverse mass, which is exactly 1.0
 * here — but it is a *runtime* 1.0, so the accumulator's `old + force.x * inv`
 * is a contraction candidate the un-bundled path's plain `old + force.x` is
 * not. An fma rounds once where two operations round twice, and the two answers
 * differ in the last bit whenever `old` is non-zero, which it always is: the
 * repulsion pass wrote it first. Section 1 carries the exactness claim, against
 * the shader the change was made to; this section carries the end-to-end one,
 * at a bar three orders of magnitude below a node radius. (Measured worst case
 * on this fixture: 3.4e-4.)
 */
const IDENTITY_AGGREGATION_FLOOR = 1e-3;

/** Largest distance between two snapshots over `[0, count)`. */
function worstDivergence(
  actual: PositionSnapshot,
  reference: PositionSnapshot,
  count: number,
): number {
  let worst = 0;
  for (let i = 0; i < count; i++) {
    const d = Math.hypot(actual.x[i] - reference.x[i], actual.y[i] - reference.y[i]);
    if (!(d <= worst)) worst = d;
  }
  return worst;
}

for (const plugin of PLUGINS) {
  gpuTest(
    `SC-005: ${plugin.id} with an identity aggregation matches the un-aggregated run`,
    async (device) => {
      const algorithm = await plugin.load();
      const graph = matchingGraph(64);

      const run = async (identity: boolean): Promise<PositionSnapshot> => {
        const harness = await createAlgorithmSimHarness(
          device,
          algorithm,
          graph,
          PLUGIN_CONFIG,
        );
        try {
          if (identity) {
            harness.setEdgeBundles(
              identityEdgeList(graph.edgeSources.length),
              new Uint32Array(0),
            );
            assertEquals(harness.activeEdgeCount, graph.edgeSources.length);
            assertEquals(harness.bundleCount, 0);
          } else {
            assertEquals(harness.activeEdgeCount, 0, "no aggregation is declared by default");
          }
          await harness.tick(SC005_TICKS);
          return await harness.readPositions();
        } finally {
          harness.dispose();
        }
      };

      const plain = await run(false);
      const identity = await run(true);

      assertEquals(countNonFinite(identity.x, identity.y), 0);
      // Each path is bit-reproducible against itself, so the bar below measures
      // the two paths' difference and nothing else.
      const observed = worstDivergence(identity, plain, graph.nodeCount);
      assert(
        observed < IDENTITY_AGGREGATION_FLOOR,
        `${plugin.id}: the identity aggregation diverged by ${observed}`,
      );
    },
  );
}

// =============================================================================
// Section 3 — SC-002: the collapsed layout approximates the expanded one
// =============================================================================

for (const plugin of PLUGINS) {
  gpuTest(
    `SC-002: ${plugin.id} transfers a collapsed subtree's imports to its proxy`,
    async (device) => {
      const algorithm = await plugin.load();
      const fixture = twoModules();
      const base: HarnessGraphData = {
        nodeCount: fixture.nodeCount,
        positionsX: fixture.positionsX,
        positionsY: fixture.positionsY,
        edgeSources: fixture.edgeSources,
        edgeTargets: fixture.edgeTargets,
        depths: fixture.depths,
      };

      const run = async (collapsed: boolean, bundled: boolean): Promise<PositionSnapshot> => {
        const harness = await createAlgorithmSimHarness(
          device,
          algorithm,
          collapsed
            ? { ...base, flags: fixture.collapsedFlags, mass: fixture.collapsedMass }
            : base,
          PLUGIN_CONFIG,
        );
        try {
          if (bundled) {
            const aggregation = aggregate(fixture.edgeSources, fixture.edgeTargets, fixture);
            assertEquals(
              Array.from(aggregation.bundles),
              [0, 1, LEAVES * LEAVES],
              "every import folds into one weighted bundle between the two proxies",
            );
            harness.setEdgeBundles(aggregation.liveEdges, aggregation.bundles);
          }
          await harness.tick(SC002_TICKS);
          return await harness.readPositions();
        } finally {
          harness.dispose();
        }
      };

      const expanded = await run(false, false);
      const dropped = await run(true, false);
      const aggregated = await run(true, true);

      assertEquals(countNonFinite(aggregated.x, aggregated.y), 0);

      // A proxy stands for its subtree's centre of attraction, not for its own
      // expanded position: a proxy accelerates the way the centre of mass of
      // the bodies it replaces does, and the module node itself sits wherever
      // the cluster's own internal forces left it.
      const subtreeCentre = (snapshot: PositionSnapshot, module: number): [number, number] => {
        let x = snapshot.x[module];
        let y = snapshot.y[module];
        for (let k = 0; k < LEAVES; k++) {
          x += snapshot.x[leafSlot(module, k)];
          y += snapshot.y[leafSlot(module, k)];
        }
        return [x / (LEAVES + 1), y / (LEAVES + 1)];
      };

      /** Distance between the two subtrees' centres of attraction. */
      const separation = (snapshot: PositionSnapshot, collapsed: boolean): number => {
        if (collapsed) {
          return Math.hypot(snapshot.x[1] - snapshot.x[0], snapshot.y[1] - snapshot.y[0]);
        }
        const [ax, ay] = subtreeCentre(snapshot, 0);
        const [bx, by] = subtreeCentre(snapshot, 1);
        return Math.hypot(bx - ax, by - ay);
      };

      // The imports are the only thing opposing repulsion, so the one number
      // the layout has to preserve across a collapse is how far apart the two
      // modules settle. Everything else in this fixture is internal to a
      // subtree that is no longer on screen.
      const reference = separation(expanded, false);
      const droppedError = Math.abs(separation(dropped, true) - reference);
      const aggregatedError = Math.abs(separation(aggregated, true) - reference);
      console.log(
        `[gpu] SC-002 ${plugin.id}: reference=${reference.toFixed(1)} ` +
          `dropped=${droppedError.toFixed(1)} aggregated=${aggregatedError.toFixed(1)} ` +
          `ratio=${(aggregatedError / droppedError).toFixed(3)}`,
      );

      // The bug has to be visible for the fix to mean anything: with the
      // imports dropped nothing opposes repulsion and the pair flies apart.
      assert(
        droppedError > reference * 0.5,
        `${plugin.id}: dropping the imports must visibly change the layout, ` +
          `separation moved by ${droppedError} against ${reference}`,
      );
      // Comparative rather than absolute, because a proxy is a rigid point and
      // the subtree it replaces is not. What must hold is that bundling
      // recovers most of the difference — see PLUGINS for why the fraction
      // differs per force law.
      assert(
        aggregatedError < droppedError * plugin.sc002Recovery,
        `${plugin.id}: bundling must recover the expanded separation: ` +
          `${aggregatedError} vs ${droppedError} without it ` +
          `(bar: ${droppedError * plugin.sc002Recovery}, reference ${reference})`,
      );
    },
  );
}

gpuTest("bundles: a collapsed pair joined only by hidden imports feels a pull at all", async (
  device,
) => {
  // The narrowest statement of the bug: with only the cross-cutting imports in
  // the graph and no repulsion at all, the two proxies are joined by nothing
  // until the aggregation exists.
  const algorithm = await PLUGINS[0].load();
  const fixture = twoModules();
  const graph: HarnessGraphData = {
    nodeCount: fixture.nodeCount,
    positionsX: fixture.positionsX,
    positionsY: fixture.positionsY,
    edgeSources: fixture.importSources,
    edgeTargets: fixture.importTargets,
    depths: fixture.depths,
    flags: fixture.collapsedFlags,
    mass: fixture.collapsedMass,
  };
  const noRepulsion = validateForceConfig({
    centerStrength: 0,
    repulsionStrength: 0,
    collisionEnabled: false,
    velocityDecay: 0.4,
  });

  const dropped = await createAlgorithmSimHarness(device, algorithm, graph, noRepulsion);
  let before: PositionSnapshot;
  let after: PositionSnapshot;
  try {
    before = await dropped.readPositions();
    await dropped.tick(8);
    after = await dropped.readPositions();
  } finally {
    dropped.dispose();
  }
  assertFrozen(before, after, fixture.frontier);

  const bundled = await createAlgorithmSimHarness(device, algorithm, graph, noRepulsion);
  try {
    const aggregation = aggregate(fixture.importSources, fixture.importTargets, fixture);
    assertEquals(aggregation.liveEdges.length, 0, "no import has a visible endpoint");
    assertEquals(
      Array.from(aggregation.bundles),
      [0, 1, LEAVES * LEAVES],
      "every import folds into one weighted bundle between the two proxies",
    );

    bundled.setEdgeBundles(aggregation.liveEdges, aggregation.bundles);
    const start = await bundled.readPositions();
    await bundled.tick(8);
    const end = await bundled.readPositions();

    assertEquals(countNonFinite(end.x, end.y), 0);
    const opened = Math.abs(start.x[1] - start.x[0]);
    const closed = Math.abs(end.x[1] - end.x[0]);
    assert(
      closed < opened - 1,
      `the bundle must pull the proxies together: ${opened} -> ${closed}`,
    );
    assertFrozen(start, end, [leafSlot(0, 0), leafSlot(1, 0)]);
  } finally {
    bundled.dispose();
  }
});

// =============================================================================
// Section 4 — the N² plugin has no phantom bodies
// =============================================================================

/**
 * One live node far out on the x axis, and one slot at the origin.
 *
 * The origin slot is what a removed node leaves behind: graph.ts zeroes a
 * removed node's position and it stays inside the high-water mark. The live
 * node's distance to it is the whole signal — under the old unmasked entry
 * point it was pushed straight out along +x by a body nobody could see.
 */
function phantomFixture(originFlag: number): HarnessGraphData {
  return {
    nodeCount: 2,
    positionsX: new Float32Array([100, 0]),
    positionsY: new Float32Array([0, 0]),
    // One edge, carrying no force (springStrength is zero below). It exists
    // only because Deno's WebGPU backend panics on a zero-length buffer write,
    // which an edgeless fixture would trigger in copyEdgesToSimulation.
    edgeSources: Uint32Array.from([0]),
    edgeTargets: Uint32Array.from([1]),
    depths: new Float32Array(2),
    flags: Uint32Array.from([0, originFlag]),
  };
}

/** Repulsion and nothing else: no gravity, no springs, no collision. */
const PHANTOM_CONFIG = validateForceConfig({
  centerStrength: 0,
  repulsionStrength: 500,
  springStrength: 0,
  collisionEnabled: false,
});

for (
  const [label, flag] of [
    ["removed", NODE_FLAG_DEAD],
    ["LOD-hidden", NODE_FLAG_HIDDEN_LOD],
  ] as const
) {
  gpuTest(`n2 plugin: a ${label} slot at the origin exerts no force`, async (device) => {
    const algorithm = await loadPlugin("n2.ts", "createN2Algorithm")();

    const run = async (graph: HarnessGraphData, staleList: boolean): Promise<PositionSnapshot> => {
      const harness: SimHarness = await createAlgorithmSimHarness(
        device,
        algorithm,
        graph,
        PHANTOM_CONFIG,
      );
      try {
        // The list a host holds for one frame after a transition: every slot,
        // including the one it has just flagged. The flag mask is meant to be
        // defence in depth over the gather, and this is the only way to say so
        // — with a freshly derived list the phantom is already off the list and
        // the mask is never consulted.
        if (staleList) harness.setActiveIndices(Uint32Array.from([0, 1]));
        await harness.tick(10);
        return await harness.readPositions();
      } finally {
        harness.dispose();
      }
    };

    for (const staleList of [false, true]) {
      const masked = await run(phantomFixture(flag), staleList);
      assertEquals(countNonFinite(masked.x, masked.y), 0);
      // Zero force, not "small force": with nothing else in the graph and no
      // gravity, the only thing that could move slot 0 is the phantom.
      assertFrozen(
        { x: Float32Array.from([100, 0]), y: new Float32Array(2) },
        masked,
        [0],
      );
    }

    // Non-vacuity: the same fixture with the origin slot alive does move it,
    // so the assertions above are measuring the mask rather than a dead rig.
    const live = await run(phantomFixture(0), false);
    assert(
      live.x[0] > 100.5,
      `an unflagged body at the origin must push the live node out, got ${live.x[0]}`,
    );
  });
}

// =============================================================================
// Section 5 — a collapsed proxy repels like the subtree it stands for
// =============================================================================

/**
 * Two bodies on the x axis and nothing else: no edges, no gravity, no
 * collision. How far they push apart is the repulsion law and only that.
 */
function twoBodyFixture(mass: Float32Array): HarnessGraphData {
  return {
    nodeCount: 2,
    positionsX: new Float32Array([-40, 40]),
    positionsY: new Float32Array([0, 0]),
    // One weightless edge; see phantomFixture for why an edgeless fixture is
    // not an option.
    edgeSources: Uint32Array.from([0]),
    edgeTargets: Uint32Array.from([1]),
    depths: new Float32Array(2),
    mass,
  };
}

for (const plugin of PLUGINS) {
  gpuTest(`${plugin.id}: a proxy repels with the mass of the subtree it replaces`, async (
    device,
  ) => {
    // WP-F's guarantee, which these three plugins never had: a collapse
    // replaces a subtree with one body, and if that body pushes like one node
    // the visible layout contracts on collapse and re-inflates on expand — the
    // one thing semantic LOD must not do. Their own degree weighting is a
    // different quantity and cannot stand in for it: an eight-node module and
    // an eight-node star with one hub have the same degree sum and very
    // different extents.
    const algorithm = await plugin.load();

    const pushedTo = async (mass: number): Promise<number> => {
      const harness = await createAlgorithmSimHarness(
        device,
        algorithm,
        twoBodyFixture(Float32Array.from([1, mass])),
        PHANTOM_CONFIG,
      );
      try {
        await harness.tick(10);
        const end = await harness.readPositions();
        return end.x[1] - end.x[0];
      } finally {
        harness.dispose();
      }
    };

    const unit = await pushedTo(1);
    const heavy = await pushedTo(16);
    assert(unit > 0, `${plugin.id}: the two bodies must repel at all, gap ${unit}`);
    assert(
      heavy > unit * 1.2,
      `${plugin.id}: a 16-body proxy must push far harder than one body: ` +
        `${heavy} vs ${unit}`,
    );
  });
}

// =============================================================================
// Section 6 — the two algorithms whose passes are not shaped like the others
// =============================================================================

/** Forward and inverse CSR over an edge list, as graph.ts uploads for RA. */
function buildCsr(
  nodeCount: number,
  sources: Uint32Array,
  targets: Uint32Array,
): {
  forward: { offsets: Uint32Array; indices: Uint32Array };
  inverse: { offsets: Uint32Array; indices: Uint32Array };
} {
  const edgeCount = sources.length;
  const fOff = new Uint32Array(nodeCount + 1);
  const iOff = new Uint32Array(nodeCount + 1);
  for (let e = 0; e < edgeCount; e++) {
    fOff[sources[e] + 1]++;
    iOff[targets[e] + 1]++;
  }
  for (let i = 0; i < nodeCount; i++) {
    fOff[i + 1] += fOff[i];
    iOff[i + 1] += iOff[i];
  }
  const fIdx = new Uint32Array(edgeCount);
  const iIdx = new Uint32Array(edgeCount);
  const fCur = fOff.slice(0, nodeCount);
  const iCur = iOff.slice(0, nodeCount);
  for (let e = 0; e < edgeCount; e++) {
    fIdx[fCur[sources[e]]++] = targets[e];
    iIdx[iCur[targets[e]]++] = sources[e];
  }
  return {
    forward: { offsets: fOff, indices: fIdx },
    inverse: { offsets: iOff, indices: iIdx },
  };
}

/** The Relativity Atlas module's two extra exports the harness has to drive. */
interface RaModule {
  createRelativityAtlasAlgorithm(): HarnessForceAlgorithm;
  uploadRelativityAtlasEdges(
    device: GPUDevice,
    buffers: { destroy(): void },
    forward: { offsets: Uint32Array; indices: Uint32Array },
    inverse: { offsets: Uint32Array; indices: Uint32Array },
    nodeCount: number,
  ): void;
}

gpuTest("relativity-atlas: a collapsed pair joined only by hidden imports feels a pull", async (
  device,
) => {
  // Relativity Atlas runs the FA2 attraction module in its log branch, so the
  // arithmetic is already covered above. What is specific to it — and what this
  // pins — is the wiring: its own uniform layout (the two LOD counts sit where
  // its 16-byte struct used to end), its own bundled bind group, and its own
  // choice of pipeline per frame. A collapse used to delete this pass's pull
  // entirely, and this is the pass that holds a child to its parent.
  const mod = await loadModuleInliningWgsl<RaModule>(
    new URL("../../packages/core/src/simulation/algorithms/relativity-atlas.ts", import.meta.url),
  );
  const algorithm = mod.createRelativityAtlasAlgorithm();
  const fixture = twoModules();
  const csr = buildCsr(fixture.nodeCount, fixture.importSources, fixture.importTargets);
  const graph: HarnessGraphData = {
    nodeCount: fixture.nodeCount,
    positionsX: fixture.positionsX,
    positionsY: fixture.positionsY,
    edgeSources: fixture.importSources,
    edgeTargets: fixture.importTargets,
    depths: fixture.depths,
    flags: fixture.collapsedFlags,
    mass: fixture.collapsedMass,
  };
  // No repulsion, no gravity: whatever the two proxies do is the attraction
  // pass and nothing else.
  const attractionOnly = validateForceConfig({
    centerStrength: 0,
    repulsionStrength: 0,
    collisionEnabled: false,
    velocityDecay: 0.4,
  });

  const run = async (bundled: boolean): Promise<PositionSnapshot> => {
    const harness = await createAlgorithmSimHarness(
      device,
      algorithm,
      graph,
      attractionOnly,
      undefined,
      {
        onAlgorithmBuffers(algoBuffers) {
          mod.uploadRelativityAtlasEdges(
            device,
            algoBuffers,
            csr.forward,
            csr.inverse,
            fixture.nodeCount,
          );
        },
      },
    );
    try {
      if (bundled) {
        const aggregation = aggregate(fixture.importSources, fixture.importTargets, fixture);
        harness.setEdgeBundles(aggregation.liveEdges, aggregation.bundles);
      }
      const start = await harness.readPositions();
      await harness.tick(8);
      const end = await harness.readPositions();
      assertEquals(countNonFinite(end.x, end.y), 0);
      return {
        x: Float32Array.from([Math.abs(start.x[1] - start.x[0]), Math.abs(end.x[1] - end.x[0])]),
        y: end.y,
      };
    } finally {
      harness.dispose();
    }
  };

  const dropped = await run(false);
  assertEquals(dropped.x[1], dropped.x[0], "today the imports pull on nothing");

  const bundled = await run(true);
  assert(
    bundled.x[1] < bundled.x[0] - 1,
    `the bundle must pull the proxies together: ${bundled.x[0]} -> ${bundled.x[1]}`,
  );
});

gpuTest("barnes-hut: the traversal dispatch is the active list", async (device) => {
  // Barnes-Hut is the one plugin whose repulsion is not a slot sweep: the tree
  // is still built over every particle (leaf indices are sorted-particle order,
  // so a compacted build would renumber leaves rather than remove them) and
  // only the TRAVERSAL — the expensive half — is gathered. Omitting a visible
  // particle from the list is the only way to observe that: the tree still
  // contains it, so if the traversal were slot-indexed it would still be
  // pushed.
  const algorithm = await loadPlugin("barnes-hut.ts", "createBarnesHutAlgorithm")();
  const nodeCount = 64;
  const positionsX = new Float32Array(nodeCount);
  const positionsY = new Float32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    const angle = (i / nodeCount) * Math.PI * 2;
    positionsX[i] = Math.cos(angle) * 50;
    positionsY[i] = Math.sin(angle) * 50;
  }
  const graph: HarnessGraphData = {
    nodeCount,
    positionsX,
    positionsY,
    edgeSources: Uint32Array.from([0]),
    edgeTargets: Uint32Array.from([1]),
    depths: new Float32Array(nodeCount),
  };

  const run = async (omit: number | null): Promise<PositionSnapshot> => {
    const harness = await createAlgorithmSimHarness(
      device,
      algorithm,
      graph,
      PHANTOM_CONFIG,
      undefined,
      { boundsSyncInterval: 10 },
    );
    try {
      if (omit !== null) {
        const list: number[] = [];
        for (let i = 0; i < nodeCount; i++) {
          if (i !== omit) list.push(i);
        }
        harness.setActiveIndices(Uint32Array.from(list));
      }
      await harness.tick(6);
      return await harness.readPositions();
    } finally {
      harness.dispose();
    }
  };

  const listed = await run(null);
  assert(
    Math.abs(listed.x[7] - positionsX[7]) > 0.5,
    `the traversal must displace a listed particle, moved ${listed.x[7] - positionsX[7]}`,
  );

  const unlisted = await run(7);
  assertEquals(countNonFinite(unlisted.x, unlisted.y), 0);
  assertFrozen({ x: positionsX, y: positionsY }, unlisted, [7]);
  // Everything else still moved: the omission removed one traversal, not the
  // pass.
  assert(
    Math.abs(unlisted.x[8] - positionsX[8]) > 0.5,
    "the remaining particles must still be traversed",
  );
});

// =============================================================================
// Section 8 — the flat LOD edge set: one buffer, two regions
// =============================================================================

/**
 * Each bundled attraction shader addresses the bundle table at an offset equal
 * to the live-edge count, because the two lists share one storage buffer:
 * apart they put the pass at nine storage buffers against a WebGPU default of
 * eight, which is not slower but unselectable. Every shader carries its own
 * copy of that arithmetic (`bundle_base`), so every shader needs its own proof.
 *
 * The proof is a base shift. `twoModules()` collapsed leaves no edge with two
 * visible endpoints, so an active-edge list naming every source edge is
 * gathered, masked and contributes nothing at all — not a zero added to a sum.
 * Running once with an empty live region and once with a full one therefore
 * changes only where the bundle sits in the buffer, and the two runs must
 * agree BIT FOR BIT: one bundle means one contribution per node, so the
 * force-accumulation CAS loop has nothing to reorder. A base stuck at zero
 * reads live-edge indices as `[source, target, weight]` triples instead, and
 * attracts whichever slots they happen to name.
 *
 * Relativity Atlas is covered by the force-atlas2 case: it runs the same
 * fa2_attraction.comp.wgsl module.
 */
for (const plugin of PLUGINS) {
  gpuTest(
    `flat layout: ${plugin.id} bases the bundle region at the live-edge count`,
    async (device) => {
      const algorithm = await plugin.load();
      const fixture = twoModules();
      const graph: HarnessGraphData = {
        nodeCount: fixture.nodeCount,
        positionsX: fixture.positionsX,
        positionsY: fixture.positionsY,
        edgeSources: fixture.edgeSources,
        edgeTargets: fixture.edgeTargets,
        depths: fixture.depths,
        flags: fixture.collapsedFlags,
        mass: fixture.collapsedMass,
      };
      const aggregation = aggregate(fixture.edgeSources, fixture.edgeTargets, fixture);
      assertEquals(aggregation.liveEdges.length, 0, "the collapse must leave no visible edge");

      const run = async (liveEdges: Uint32Array): Promise<PositionSnapshot> => {
        const harness = await createAlgorithmSimHarness(
          device,
          algorithm,
          graph,
          PLUGIN_CONFIG,
        );
        try {
          harness.setEdgeBundles(liveEdges, aggregation.bundles);
          assertEquals(harness.activeEdgeCount, liveEdges.length);
          assertEquals(harness.bundleCount, 1);
          await harness.tick(8);
          return await harness.readPositions();
        } finally {
          harness.dispose();
        }
      };

      // Every source edge: each has a hidden endpoint under this cut, so the
      // whole list is inert and only the bundle's region base moves.
      const inert = new Uint32Array(fixture.edgeSources.length);
      for (let e = 0; e < inert.length; e++) inert[e] = e;
      assert(inert.length > 0, "the shifted run needs a non-empty live-edge region");

      const based = await run(new Uint32Array(0));
      const shifted = await run(inert);

      assertEquals(countNonFinite(based.x, based.y), 0);
      for (let slot = 0; slot < fixture.nodeCount; slot++) {
        assertStrictEquals(
          shifted.x[slot],
          based.x[slot],
          `slot ${slot} x moved when the bundle region shifted by ${inert.length} words`,
        );
        assertStrictEquals(
          shifted.y[slot],
          based.y[slot],
          `slot ${slot} y moved when the bundle region shifted by ${inert.length} words`,
        );
      }
    },
  );
}
