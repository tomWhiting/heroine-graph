/**
 * GPU tests for the t-FDP algorithm (t-distribution repulsion kernel):
 *
 * - The anti-collapse force floor engages ONLY near zero distance. The old
 *   unconditional max(raw, 0.3*kr) made every pair repel at constant
 *   magnitude at essentially every distance (with gamma=2 the raw kernel
 *   peaks at ~0.325*kr), nullifying the t-kernel's long-range decay and
 *   inflating the layout without bound.
 * - Distances are normalized by distScale so the kernel is meaningfully
 *   above zero at typical world spacing (raw world units put all its mass
 *   below ~2 units and the layout collapsed instead).
 * - Coincident nodes separate (deterministic golden-angle tiebreaker).
 * - Dead slots neither exert nor receive repulsion (main_masked path).
 * - Layout extent on a code-tree fixture stabilizes instead of inflating.
 *
 * Drives the real TFdpAlgorithm plugin (pipelines, bind groups, uniform
 * packing) with a minimal gradient-descent integrator, since t-FDP's
 * force passes are algorithm-plugin code the core pipeline harness does
 * not dispatch.
 */

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import {
  GPU_SKIP_MESSAGE,
  loadModuleInliningWgsl,
  NODE_FLAG_DEAD,
  probeAdapter,
} from "../helpers/gpu.ts";
import { countNonFinite, maxRadius, meanEdgeLength } from "../helpers/invariants.ts";
import { generateCodeTree } from "../fixtures/code_tree.ts";
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
      const device = await adapter!.requestDevice();
      try {
        await fn(device);
      } finally {
        device.destroy();
      }
    },
  });
}

/**
 * Structural subset of the ForceAlgorithm interface (types.ts cannot be
 * imported statically without dragging .wgsl-importing modules into the
 * test's module graph — see tests/helpers/gpu.ts module doc).
 */
interface ForceAlgorithmLike {
  createPipelines(context: unknown): unknown;
  createBuffers(device: GPUDevice, maxNodes: number): { destroy(): void };
  createBindGroups(
    device: GPUDevice,
    pipelines: unknown,
    context: unknown,
    algorithmBuffers: unknown,
  ): unknown;
  updateUniforms(device: GPUDevice, algorithmBuffers: unknown, context: unknown): void;
  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: unknown,
    bindGroups: unknown,
    nodeCount: number,
  ): void;
}

interface TFdpModule {
  createTFdpAlgorithm(): ForceAlgorithmLike;
}

function loadTFdpModule(): Promise<TFdpModule> {
  return loadModuleInliningWgsl<TFdpModule>(
    new URL(
      "../../packages/core/src/simulation/algorithms/t-fdp.ts",
      import.meta.url,
    ),
  );
}

/**
 * Minimal position update: gradient descent with a per-tick step cap
 * (stand-in for the velocity-capped integrator), then clears forces for
 * the next tick.
 */
const INTEGRATE_WGSL = `
struct IntegrateUniforms {
    node_count: u32,
    step: f32,
    max_step: f32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: IntegrateUniforms;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= uniforms.node_count) {
        return;
    }

    var delta = forces[i] * uniforms.step;
    let len = length(delta);
    if (len > uniforms.max_step) {
        delta *= uniforms.max_step / len;
    }
    positions[i] += delta;
    forces[i] = vec2<f32>(0.0, 0.0);
}
`;

interface TFdpHarnessGraph {
  nodeCount: number;
  positionsX: Float32Array;
  positionsY: Float32Array;
  edgeSources?: Uint32Array;
  edgeTargets?: Uint32Array;
  /** Per-slot flags; providing them exercises the main_masked entry point */
  flags?: Uint32Array;
}

interface TFdpHarness {
  /** Advance repulsion + attraction + integrate by `steps` ticks */
  tick(steps: number): Promise<void>;
  /** Dispatch the force passes once (no integration) and read raw forces */
  computeForces(): Promise<{ fx: Float32Array; fy: Float32Array }>;
  readPositions(): Promise<{ x: Float32Array; y: Float32Array }>;
  dispose(): void;
}

async function createTFdpHarness(
  device: GPUDevice,
  graph: TFdpHarnessGraph,
  config: Partial<FullForceConfig> = {},
  integrator: { step?: number; maxStep?: number } = {},
): Promise<TFdpHarness> {
  const mod = await loadTFdpModule();
  const { nodeCount } = graph;
  const edgeCount = graph.edgeSources?.length ?? 0;
  const forceConfig = validateForceConfig(config);

  const interleaved = new Float32Array(nodeCount * 2);
  for (let i = 0; i < nodeCount; i++) {
    interleaved[i * 2] = graph.positionsX[i];
    interleaved[i * 2 + 1] = graph.positionsY[i];
  }

  const positions = device.createBuffer({
    label: "t-FDP test positions",
    size: nodeCount * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(positions, 0, interleaved);

  const forces = device.createBuffer({
    label: "t-FDP test forces",
    size: nodeCount * 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  // createBindGroups requires edge buffers even for edgeless graphs
  const edgeSources = device.createBuffer({
    label: "t-FDP test edge sources",
    size: Math.max(edgeCount, 1) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const edgeTargets = device.createBuffer({
    label: "t-FDP test edge targets",
    size: Math.max(edgeCount, 1) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  if (graph.edgeSources && graph.edgeTargets && edgeCount > 0) {
    device.queue.writeBuffer(edgeSources, 0, graph.edgeSources.slice().buffer);
    device.queue.writeBuffer(edgeTargets, 0, graph.edgeTargets.slice().buffer);
  }

  let nodeFlags: GPUBuffer | undefined;
  if (graph.flags) {
    nodeFlags = device.createBuffer({
      label: "t-FDP test node flags",
      size: nodeCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(nodeFlags, 0, graph.flags.slice().buffer);
  }

  const readback = device.createBuffer({
    label: "t-FDP test readback",
    size: nodeCount * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const algorithm = mod.createTFdpAlgorithm();
  const pipelines = algorithm.createPipelines({ device });
  const algoBuffers = algorithm.createBuffers(device, nodeCount);
  const context = {
    device,
    positions,
    forces,
    nodeCount,
    edgeCount,
    forceConfig,
    edgeSources,
    edgeTargets,
    nodeFlags,
  };
  const bindGroups = algorithm.createBindGroups(device, pipelines, context, algoBuffers);
  algorithm.updateUniforms(device, algoBuffers, context);

  const integratePipeline = device.createComputePipeline({
    label: "t-FDP test integrate",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: INTEGRATE_WGSL }),
      entryPoint: "main",
    },
  });
  const integrateUniforms = device.createBuffer({
    label: "t-FDP test integrate uniforms",
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const integrateData = new ArrayBuffer(16);
  const view = new DataView(integrateData);
  view.setUint32(0, nodeCount, true);
  view.setFloat32(4, integrator.step ?? 1, true);
  view.setFloat32(8, integrator.maxStep ?? 10, true);
  device.queue.writeBuffer(integrateUniforms, 0, integrateData);

  const integrateBindGroup = device.createBindGroup({
    label: "t-FDP test integrate bind group",
    layout: integratePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: integrateUniforms } },
      { binding: 1, resource: { buffer: positions } },
      { binding: 2, resource: { buffer: forces } },
    ],
  });

  async function readInterleaved(source: GPUBuffer): Promise<Float32Array> {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, nodeCount * 8);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return data;
  }

  return {
    async tick(steps: number): Promise<void> {
      const encoder = device.createCommandEncoder();
      for (let s = 0; s < steps; s++) {
        algorithm.recordRepulsionPass(encoder, pipelines, bindGroups, nodeCount);
        const pass = encoder.beginComputePass({ label: "t-FDP test integrate" });
        pass.setPipeline(integratePipeline);
        pass.setBindGroup(0, integrateBindGroup);
        pass.dispatchWorkgroups(Math.ceil(nodeCount / 256));
        pass.end();
      }
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    },

    async computeForces(): Promise<{ fx: Float32Array; fy: Float32Array }> {
      const encoder = device.createCommandEncoder();
      algorithm.recordRepulsionPass(encoder, pipelines, bindGroups, nodeCount);
      device.queue.submit([encoder.finish()]);
      const data = await readInterleaved(forces);
      const fx = new Float32Array(nodeCount);
      const fy = new Float32Array(nodeCount);
      for (let i = 0; i < nodeCount; i++) {
        fx[i] = data[i * 2];
        fy[i] = data[i * 2 + 1];
      }
      return { fx, fy };
    },

    async readPositions(): Promise<{ x: Float32Array; y: Float32Array }> {
      const data = await readInterleaved(positions);
      const x = new Float32Array(nodeCount);
      const y = new Float32Array(nodeCount);
      for (let i = 0; i < nodeCount; i++) {
        x[i] = data[i * 2];
        y[i] = data[i * 2 + 1];
      }
      return { x, y };
    },

    dispose(): void {
      algoBuffers.destroy();
      for (
        const buffer of [
          positions,
          forces,
          edgeSources,
          edgeTargets,
          readback,
          integrateUniforms,
        ]
      ) {
        buffer.destroy();
      }
      nodeFlags?.destroy();
    },
  };
}

/** Repulsion magnitude on node 0 of an edgeless pair at the given distance. */
async function pairRepulsion(device: GPUDevice, distance: number): Promise<number> {
  const harness = await createTFdpHarness(device, {
    nodeCount: 2,
    positionsX: new Float32Array([0, distance]),
    positionsY: new Float32Array([0, 0]),
  });
  try {
    const { fx, fy } = await harness.computeForces();
    // Equal and opposite (momentum conservation)
    assertAlmostEquals(fx[0], -fx[1], 1e-5);
    assertAlmostEquals(fy[0], -fy[1], 1e-5);
    return Math.hypot(fx[0], fy[0]);
  } finally {
    harness.dispose();
  }
}

gpuTest(
  "GPU t-FDP: repulsion follows the t-kernel, force floor only near zero",
  async (device) => {
    // Defaults: gamma=2, kr = 1/0.1 = 10, distScale = 2*springLength = 60.
    // At dist == distScale (normalized d = 1): F = kr * 1 / (1+1)^2 = 2.5,
    // well clear of the near-zero floor zone.
    const atScale = await pairRepulsion(device, 60);
    assertAlmostEquals(atScale, 2.5, 1e-3);

    // Long range (d = 10): F = kr * 10 / 101^2 ~ 0.0098. The old
    // unconditional floor pinned this at 0.3 * kr = 3.0.
    const far = await pairRepulsion(device, 600);
    assert(far < 0.05, `long-range repulsion did not decay: ${far}`);

    // Very long range (d = 100): essentially zero
    const veryFar = await pairRepulsion(device, 6000);
    assert(veryFar < 1e-3, `very-long-range repulsion did not decay: ${veryFar}`);

    // Overlap zone (2 world units): the anti-collapse floor engages, so
    // near-stacked nodes still get a meaningful separating push
    const overlapping = await pairRepulsion(device, 2);
    assert(overlapping > 1.0, `overlap separation force too weak: ${overlapping}`);

    console.log(
      `[gpu] t-FDP kernel: d=1 -> ${atScale.toFixed(3)}, d=10 -> ${far.toFixed(4)}, ` +
        `d=100 -> ${veryFar.toExponential(2)}, overlap -> ${overlapping.toFixed(3)}`,
    );
  },
);

gpuTest(
  "GPU t-FDP: coincident nodes separate deterministically",
  async (device) => {
    // Two nodes stacked at the same position: the pair delta gives no
    // direction to repel along, so the golden-angle tiebreaker must break
    // the symmetry (the old shader skipped coincident pairs entirely).
    // Flags provided (all live) so this exercises the main_masked path.
    const harness = await createTFdpHarness(device, {
      nodeCount: 2,
      positionsX: new Float32Array([10, 10]),
      positionsY: new Float32Array([-5, -5]),
      flags: new Uint32Array([0, 0]),
    });
    try {
      await harness.tick(20);
      const { x, y } = await harness.readPositions();
      assertEquals(countNonFinite(x, y), 0);
      const dist = Math.hypot(x[1] - x[0], y[1] - y[0]);
      assert(dist > 2, `coincident nodes failed to separate: dist=${dist}`);
    } finally {
      harness.dispose();
    }
  },
);

gpuTest(
  "GPU t-FDP: dead slots neither exert nor receive repulsion",
  async (device) => {
    // Identical live pair, dead slot parked at two different positions.
    // If masking works the dead slot's position is invisible: live forces
    // match exactly across runs and the dead slot accumulates zero force.
    const run = async (deadX: number, deadY: number) => {
      const harness = await createTFdpHarness(device, {
        nodeCount: 3,
        positionsX: new Float32Array([-50, deadX, 50]),
        positionsY: new Float32Array([0, deadY, 0]),
        flags: new Uint32Array([0, NODE_FLAG_DEAD, 0]),
      });
      try {
        return await harness.computeForces();
      } finally {
        harness.dispose();
      }
    };

    const near = await run(-45, 3); // right next to node 0
    const far = await run(200, -100);

    // Dead slot receives nothing
    assertEquals(near.fx[1], 0, "dead slot received force (x)");
    assertEquals(near.fy[1], 0, "dead slot received force (y)");
    assertEquals(far.fx[1], 0, "dead slot received force (x)");
    assertEquals(far.fy[1], 0, "dead slot received force (y)");

    // Live forces are nonzero and bit-identical regardless of the dead
    // slot's position (it exerts nothing)
    assert(Math.hypot(near.fx[0], near.fy[0]) > 0, "live pair exerts no repulsion");
    for (const i of [0, 2]) {
      assertEquals(near.fx[i], far.fx[i], `dead slot leaked force into node ${i} (x)`);
      assertEquals(near.fy[i], far.fy[i], `dead slot leaked force into node ${i} (y)`);
    }
  },
);

gpuTest(
  "GPU t-FDP: layout extent stabilizes on a code tree (no unbounded inflation)",
  async (device) => {
    // With the old unconditional force floor every pair repelled at
    // constant magnitude regardless of distance, so total outward push
    // grew with N and never decayed — the layout inflated indefinitely
    // (extent roughly doubling from tick 200 to tick 400 under a capped
    // step). With the t-kernel restored, repulsion decays as 1/d^3 while
    // edge attraction grows with distance, so the extent converges.
    const tree = generateCodeTree({ maxNodes: 300 });
    const harness = await createTFdpHarness(device, tree);

    try {
      await harness.tick(200);
      const mid = await harness.readPositions();
      assertEquals(countNonFinite(mid.x, mid.y), 0);
      let cx = 0;
      let cy = 0;
      for (let i = 0; i < tree.nodeCount; i++) {
        cx += mid.x[i] / tree.nodeCount;
        cy += mid.y[i] / tree.nodeCount;
      }
      const r200 = maxRadius(mid.x, mid.y, cx, cy);

      await harness.tick(200);
      const end = await harness.readPositions();
      assertEquals(countNonFinite(end.x, end.y), 0);
      cx = 0;
      cy = 0;
      for (let i = 0; i < tree.nodeCount; i++) {
        cx += end.x[i] / tree.nodeCount;
        cy += end.y[i] / tree.nodeCount;
      }
      const r400 = maxRadius(end.x, end.y, cx, cy);

      // Bounded: not still inflating between tick 200 and tick 400
      assert(
        r400 < 1.35 * r200,
        `layout still inflating: r200=${r200.toFixed(1)}, r400=${r400.toFixed(1)}`,
      );
      // ...and not collapsed to a point (the pre-normalization failure
      // mode: kernel ~0 at world spacing, attraction wins everywhere)
      assert(r400 > 60, `layout collapsed: r400=${r400.toFixed(1)}`);
      const edgeLen = meanEdgeLength(end.x, end.y, tree.edgeSources, tree.edgeTargets);
      assert(edgeLen > 5, `edges collapsed: mean length ${edgeLen.toFixed(2)}`);

      console.log(
        `[gpu] t-FDP extent: r200=${r200.toFixed(1)}, r400=${r400.toFixed(1)}, ` +
          `ratio=${(r400 / r200).toFixed(3)}, meanEdge=${edgeLen.toFixed(1)}`,
      );
    } finally {
      harness.dispose();
    }
  },
);
