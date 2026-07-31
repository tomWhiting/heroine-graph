/**
 * GPU tests for Relativity Atlas correctness:
 *
 * - Finding #3 (leaves dragged inward past parents): the gravity shader's
 *   inverse-mass modulation is bounded while depth decay is active, so a
 *   light child's centripetal pull never exceeds its heavy parent's at
 *   equal radius. The unbounded hub-clustering behavior stays available as
 *   an explicit opt-in (depth decay disabled).
 * - Finding #14 (sibling truncation asymmetry): when a sibling list
 *   exceeds max_siblings, the modulo-group truncation keeps pairwise
 *   visibility mutual, so sibling repulsion injects no net momentum into
 *   large sibling groups (the old head-prefix cap pushed late-listed
 *   children without any recoil on early-listed ones).
 * - Findings #3 + #5 (system level): a full Relativity Atlas run over a
 *   code-tree fixture stays finite, keeps children outside their parents
 *   (radial ordering), and settles instead of oscillating.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  createAlgorithmSimHarness,
  GPU_SKIP_MESSAGE,
  type HarnessForceAlgorithm,
  loadModuleInliningWgsl,
  probeAdapter,
} from "../helpers/gpu.ts";
import { countNonFinite, kineticEnergy, radialOrderingScore } from "../helpers/invariants.ts";
import { generateCodeTree } from "../fixtures/code_tree.ts";
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
      // The sibling shader binds 9 storage buffers; the WebGPU default
      // limit is 8. Request the same limit production does
      // (webgpu/context.ts DEFAULT_LIMITS).
      const device = await adapter!.requestDevice({
        requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
      });
      try {
        await fn(device);
      } finally {
        device.destroy();
      }
    },
  });
}

const GRAVITY_WGSL = await Deno.readTextFile(
  new URL(
    "../../packages/core/src/simulation/shaders/relativity_gravity.comp.wgsl",
    import.meta.url,
  ),
);
const SIBLING_WGSL = await Deno.readTextFile(
  new URL(
    "../../packages/core/src/simulation/shaders/relativity_sibling.comp.wgsl",
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
  pass.dispatchWorkgroups(Math.ceil(nodeCount / 256));
  pass.end();
  encoder.copyBufferToBuffer(forcesBuffer, 0, readback, 0, nodeCount * 8);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const forces = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return forces;
}

/**
 * Dispatch relativity_gravity.comp.wgsl over a parent/child pair placed at
 * the SAME radius from the center, returning [|F_parent|, |F_child|].
 */
async function gravityPullPair(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  parentMass: number,
  parentDepth: number,
  depthDecayRate: number,
): Promise<[number, number]> {
  const radius = 150;
  // Same distance from center (0, 0), different directions
  const positions = new Float32Array([radius, 0, 0, radius]);
  const masses = new Float32Array([parentMass, 1.0]);
  const depths = new Float32Array([parentDepth, parentDepth + 1]);
  const flags = new Uint32Array([0, 0]);

  // GravityUniforms: node_count, gravity_strength, center_x, center_y,
  //                  mass_exponent, gravity_curve, gravity_exponent, depth_decay_rate
  const uniformData = new ArrayBuffer(32);
  const view = new DataView(uniformData);
  view.setUint32(0, 2, true);
  view.setFloat32(4, 1.0, true); // gravity_strength (invariant is a ratio)
  view.setFloat32(8, 0, true);
  view.setFloat32(12, 0, true);
  view.setFloat32(16, 0.5, true); // mass_exponent (default)
  view.setUint32(20, 2, true); // gravity_curve: soft (default)
  view.setFloat32(24, 1.0, true); // gravity_exponent
  view.setFloat32(28, depthDecayRate, true);
  const uniforms = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniforms, 0, uniformData);

  const posBuf = createStorageBuffer(device, positions);
  const forceBuf = createStorageBuffer(device, new Float32Array(4), GPUBufferUsage.COPY_SRC);
  const massBuf = createStorageBuffer(device, masses);
  const depthBuf = createStorageBuffer(device, depths);
  const flagBuf = createStorageBuffer(device, flags);

  try {
    const forces = await dispatchAndReadForces(
      device,
      pipeline,
      [
        { binding: 0, resource: { buffer: uniforms } },
        { binding: 1, resource: { buffer: posBuf } },
        { binding: 2, resource: { buffer: forceBuf } },
        { binding: 3, resource: { buffer: massBuf } },
        { binding: 4, resource: { buffer: depthBuf } },
        { binding: 5, resource: { buffer: flagBuf } },
      ],
      forceBuf,
      2,
    );
    return [
      Math.hypot(forces[0], forces[1]),
      Math.hypot(forces[2], forces[3]),
    ];
  } finally {
    for (const b of [uniforms, posBuf, forceBuf, massBuf, depthBuf, flagBuf]) b.destroy();
  }
}

gpuTest(
  "GPU RA gravity: child's centripetal pull never exceeds its parent's at equal radius (hierarchy mode)",
  async (device) => {
    const module = device.createShaderModule({ code: GRAVITY_WGSL });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });

    // Sweep parent masses (a dir with 2 / 50 / 200 files under the default
    // mass formula 1 + 0.5 * children) and depths, always comparing a
    // unit-mass child one level deeper at the SAME radius. This is the
    // reported failure: with unbounded inverse-mass gravity the leaf's pull
    // was ~3.6x the parent's, dragging leaves inward past their parents.
    for (const parentMass of [2, 26, 101]) {
      for (const parentDepth of [0, 1, 3]) {
        const [parentPull, childPull] = await gravityPullPair(
          device,
          pipeline,
          parentMass,
          parentDepth,
          0.7, // default relativityDepthDecay
        );
        assert(
          Number.isFinite(parentPull) && Number.isFinite(childPull) && parentPull > 0,
          `degenerate pulls: parent=${parentPull} child=${childPull}`,
        );
        // The bounded mass factor guarantees ratio <= 1 exactly; allow f32 noise
        assert(
          childPull <= parentPull * (1 + 1e-4),
          `child out-pulls parent (mass=${parentMass}, depth=${parentDepth}): ` +
            `child ${childPull} > parent ${parentPull}`,
        );
      }
    }
  },
);

gpuTest(
  "GPU RA gravity: hub-clustering inverse-mass pull stays available when depth decay is disabled",
  async (device) => {
    const module = device.createShaderModule({ code: GRAVITY_WGSL });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });

    // Opt-in mode (depth_decay_rate >= 1): light nodes ARE pulled harder
    // than hubs — the original design intent for non-hierarchical graphs.
    const [hubPull, leafPull] = await gravityPullPair(device, pipeline, 26, 0, 1.0);
    assert(
      leafPull > hubPull * 2,
      `hub-clustering opt-in lost: leaf pull ${leafPull} vs hub pull ${hubPull}`,
    );
  },
);

gpuTest(
  "GPU RA sibling repulsion: truncated 200-child group receives no net momentum",
  async (device) => {
    const module = device.createShaderModule({ code: SIBLING_WGSL });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });

    // Parent (slot 0) with 200 children on a jittered ring. max_siblings
    // stays at the default 100, so the sibling list is truncated: the old
    // head-prefix cap made children after position 100 push against the
    // first 100 with no recoil — persistent directional drift.
    const childCount = 200;
    const nodeCount = childCount + 1;
    const rng = mulberry32(42);

    const positions = new Float32Array(nodeCount * 2);
    for (let i = 0; i < childCount; i++) {
      const angle = 2 * Math.PI * (i / childCount) + (rng() - 0.5) * 0.02;
      const radius = 50 + (rng() - 0.5) * 8;
      positions[(i + 1) * 2] = radius * Math.cos(angle);
      positions[(i + 1) * 2 + 1] = radius * Math.sin(angle);
    }
    const masses = new Float32Array(nodeCount).fill(1);
    const flags = new Uint32Array(nodeCount);
    const wellRadius = new Float32Array(nodeCount);

    // Forward CSR: parent -> children 1..200; children have none
    const csrOffsets = new Uint32Array(nodeCount + 1);
    const csrTargets = new Uint32Array(childCount);
    for (let i = 0; i < childCount; i++) csrTargets[i] = i + 1;
    for (let i = 1; i <= nodeCount; i++) csrOffsets[i] = childCount;
    // Inverse CSR: parent has no parents, each child <- parent 0
    // offsets = [0, 0, 1, 2, ..., childCount]
    const invOffsets = new Uint32Array(nodeCount + 1);
    for (let i = 2; i <= nodeCount; i++) invOffsets[i] = i - 1;
    const invSources = new Uint32Array(childCount); // all zeros (parent slot)

    // SiblingUniforms (64 bytes). Orbit spring, parent-child repulsion,
    // phantom zones, cousins, and tangential amplification are all
    // disabled so ONLY pairwise sibling repulsion acts — any nonzero net
    // force is momentum injected by asymmetric truncation.
    const uniformData = new ArrayBuffer(64);
    const view = new DataView(uniformData);
    view.setUint32(0, nodeCount, true); // node_count
    view.setUint32(4, childCount, true); // edge_count
    view.setFloat32(8, 50, true); // repulsion_strength
    view.setFloat32(12, 1, true); // min_distance
    view.setUint32(16, 100, true); // max_siblings (default; 200 > 100 -> truncation)
    view.setFloat32(20, 0, true); // parent_child_multiplier off
    view.setUint32(24, 0, true); // cousin_enabled off
    view.setFloat32(28, 0, true); // cousin_strength
    view.setUint32(32, 0, true); // phantom_enabled off
    view.setFloat32(36, 0, true); // phantom_multiplier
    view.setFloat32(40, 0, true); // orbit_strength off
    view.setFloat32(44, 1, true); // tangential_multiplier 1 = plain pairwise
    view.setFloat32(48, 25, true); // orbit_radius_base
    view.setUint32(52, 0, true); // bubble_mode off
    view.setFloat32(56, 0.6, true); // orbit_scale
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
    const invOffBuf = createStorageBuffer(device, invOffsets);
    const invSrcBuf = createStorageBuffer(device, invSources);
    const offBuf = createStorageBuffer(device, csrOffsets);
    const tgtBuf = createStorageBuffer(device, csrTargets);
    const massBuf = createStorageBuffer(device, masses);
    const wellBuf = createStorageBuffer(device, wellRadius);
    const flagBuf = createStorageBuffer(device, flags);

    try {
      const forces = await dispatchAndReadForces(
        device,
        pipeline,
        [
          { binding: 0, resource: { buffer: uniforms } },
          { binding: 1, resource: { buffer: posBuf } },
          { binding: 2, resource: { buffer: forceBuf } },
          { binding: 3, resource: { buffer: invOffBuf } },
          { binding: 4, resource: { buffer: invSrcBuf } },
          { binding: 5, resource: { buffer: offBuf } },
          { binding: 6, resource: { buffer: tgtBuf } },
          { binding: 7, resource: { buffer: massBuf } },
          { binding: 8, resource: { buffer: wellBuf } },
          { binding: 9, resource: { buffer: flagBuf } },
        ],
        forceBuf,
        nodeCount,
      );

      // Every child must still interact with its modulo-group (truncation
      // samples the whole list, it does not silence anyone)
      let grossMagnitude = 0;
      let netX = 0;
      let netY = 0;
      for (let i = 1; i < nodeCount; i++) {
        const fx = forces[i * 2];
        const fy = forces[i * 2 + 1];
        const mag = Math.hypot(fx, fy);
        assert(Number.isFinite(mag), `non-finite force on child ${i}`);
        assert(mag > 0, `child ${i} receives zero sibling repulsion`);
        grossMagnitude += mag;
        netX += fx;
        netY += fy;
      }

      // Mutual pairwise visibility means every computed pair cancels
      // exactly (Newton's third law); only f32 summation noise remains.
      // The old asymmetric prefix produced |net| comparable to the mean
      // per-child force.
      const netMagnitude = Math.hypot(netX, netY);
      const meanMagnitude = grossMagnitude / childCount;
      console.log(
        `[gpu] RA sibling truncation: mean |F| = ${meanMagnitude.toFixed(4)}, ` +
          `|sum F| = ${netMagnitude.toFixed(6)}`,
      );
      assert(
        netMagnitude < meanMagnitude * 1e-2,
        `sibling truncation injects net momentum: |sum F| = ${netMagnitude}, ` +
          `mean |F| = ${meanMagnitude}`,
      );
    } finally {
      for (
        const b of [
          uniforms,
          posBuf,
          forceBuf,
          invOffBuf,
          invSrcBuf,
          offBuf,
          tgtBuf,
          massBuf,
          wellBuf,
          flagBuf,
        ]
      ) {
        b.destroy();
      }
    }
  },
);

/** Structural view of the RA buffers the integration test uploads into. */
interface RaBuffersView {
  destroy(): void;
  nodeDepth: GPUBuffer;
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
}

/** Build forward/inverse CSR from edge lists (slot-indexed). */
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

gpuTest(
  "GPU Relativity Atlas: code tree stays finite, radially ordered, and settles",
  async (device) => {
    const mod = await loadModuleInliningWgsl<RaModule>(
      new URL(
        "../../packages/core/src/simulation/algorithms/relativity-atlas.ts",
        import.meta.url,
      ),
    );
    const algorithm = mod.createRelativityAtlasAlgorithm();

    const graph = generateCodeTree({ seed: 3, maxNodes: 300 });
    const csr = buildCsr(graph.nodeCount, graph.edgeSources, graph.edgeTargets);

    const harness = await createAlgorithmSimHarness(
      device,
      algorithm,
      {
        nodeCount: graph.nodeCount,
        positionsX: graph.positionsX,
        positionsY: graph.positionsY,
        edgeSources: graph.edgeSources,
        edgeTargets: graph.edgeTargets,
        depths: graph.depths,
      },
      {}, // production default calibration
      undefined,
      {
        // graph.ts computes bounds every frame, activating RA's density
        // phase (global spacing pressure) — mirror that here.
        boundsSyncInterval: 10,
        onAlgorithmBuffers(algoBuffers) {
          // Mirror graph.ts uploadAlgorithmEdgeData: CSR for sibling
          // repulsion + BFS depths for depth-decayed gravity.
          mod.uploadRelativityAtlasEdges(
            device,
            algoBuffers,
            csr.forward,
            csr.inverse,
            graph.nodeCount,
          );
          device.queue.writeBuffer(
            (algoBuffers as RaBuffersView).nodeDepth,
            0,
            graph.depths.slice().buffer,
          );
        },
      },
    );

    try {
      // Run to rest (harness alpha schedule reaches alphaMin in ~300 ticks)
      await harness.tick(360);
      const mid = await harness.readPositions();
      assertEquals(countNonFinite(mid.x, mid.y), 0);

      // Settling (finding #5): with bounded log(1+d) attraction and
      // adaptive speed, late-run motion must be negligible. The old
      // unbounded F=d pull kept deep leaves saturating the velocity cap.
      await harness.tick(40);
      const end = await harness.readPositions();
      assertEquals(countNonFinite(end.x, end.y), 0);
      const lateKE = kineticEnergy(mid.x, mid.y, end.x, end.y, 40) / graph.nodeCount;
      console.log(`[gpu] RA code tree: late per-node KE = ${lateKE.toExponential(2)}`);
      assert(
        lateKE < 1e-3,
        `layout still moving after cooling: per-node KE ${lateKE}`,
      );

      // Hierarchy unfolds outward (finding #3): mean distance from the
      // root must increase strictly with depth. The reported "leaves
      // dragged inward past their parents" failure inverts this — the
      // unbounded inverse-mass gravity pulled unit-mass leaves toward the
      // center ~3.6x harder than their parents.
      const rootX = end.x[0];
      const rootY = end.y[0];
      const depthRadius = new Map<number, { sum: number; n: number }>();
      for (let i = 0; i < graph.nodeCount; i++) {
        const d = graph.depths[i];
        const entry = depthRadius.get(d) ?? { sum: 0, n: 0 };
        entry.sum += Math.hypot(end.x[i] - rootX, end.y[i] - rootY);
        entry.n++;
        depthRadius.set(d, entry);
      }
      const levels = [...depthRadius.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([d, e]) => ({ depth: d, meanR: e.sum / e.n, n: e.n }));
      console.log(
        `[gpu] RA code tree: mean radius from root by depth: ` +
          levels.map((l) => `d${l.depth}=${l.meanR.toFixed(1)}(n=${l.n})`).join(" "),
      );
      assert(levels.length >= 3, "fixture too shallow to test hierarchy unfolding");
      for (let i = 1; i < levels.length; i++) {
        assert(
          levels[i].meanR > levels[i - 1].meanR,
          `depth ${levels[i].depth} sits inside depth ${levels[i - 1].depth}: ` +
            `${levels[i].meanR} <= ${levels[i - 1].meanR} — leaves dragged inward`,
        );
      }

      // Per-pair radial ordering from the root as a secondary tripwire
      // (local ring jitter keeps this well below 1; the inward-collapse
      // failure mode drives it toward 0.3).
      const score = radialOrderingScore(end.x, end.y, graph.parent, rootX, rootY);
      console.log(`[gpu] RA code tree: radial ordering score (root) = ${score.toFixed(3)}`);
      assert(
        score > 0.55,
        `children collapse inside parents: radial ordering ${score}`,
      );
    } finally {
      harness.dispose();
    }
  },
);
