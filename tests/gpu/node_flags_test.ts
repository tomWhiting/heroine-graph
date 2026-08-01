/**
 * GPU tests for NODE_FLAG_DEAD / NODE_FLAG_PINNED / NODE_FLAG_HIDDEN_LOD
 * semantics:
 *
 * - A dead slot exerts no force on anything and never moves — the live
 *   nodes' trajectories are bit-identical regardless of where the dead
 *   slot's (stale) position happens to sit.
 * - A pinned node holds its exact position under strong spring forces
 *   while still repelling neighbors.
 * - Collision resolution never displaces a pinned node (it still pushes
 *   overlapping neighbors away), and dead slots neither move nor push.
 * - A hidden node produces no fragments in the node render pipeline, and is
 *   out of the simulation as well: it exerts no force, receives none, is
 *   frozen by integration and never enters the collision set — without
 *   disturbing the dead/pinned behaviour above.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import { assertFrozen } from "../helpers/invariants.ts";
import {
  createSimHarness,
  DEAD_SLOT_RADIUS,
  GPU_SKIP_MESSAGE,
  loadModuleInliningWgsl,
  NODE_FLAG_DEAD,
  NODE_FLAG_HIDDEN_LOD,
  NODE_FLAG_PINNED,
  probeAdapter,
  requestHarnessDevice,
  runCollision,
} from "../helpers/gpu.ts";

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

gpuTest(
  "GPU flags: dead slot exerts no force and never moves",
  async (device) => {
    // Two live nodes with an edge, plus one dead slot. Run the identical
    // simulation twice with the dead slot parked at different positions —
    // once dead-center between the live pair (where a phantom node would
    // repel both) and once far off to the side. If the dead mask works,
    // the dead slot's position is invisible: live trajectories match bit
    // for bit, and the dead slot itself never moves.
    const run = async (deadX: number, deadY: number) => {
      const harness = await createSimHarness(device, {
        nodeCount: 3,
        positionsX: new Float32Array([-100, deadX, 100]),
        positionsY: new Float32Array([0, deadY, 0]),
        edgeSources: new Uint32Array([0]),
        edgeTargets: new Uint32Array([2]),
        flags: new Uint32Array([0, NODE_FLAG_DEAD, 0]),
      });
      try {
        await harness.tick(50);
        return await harness.readPositions();
      } finally {
        harness.dispose();
      }
    };

    const centered = await run(0, 0);
    const offside = await run(37, 91);

    // Dead slot never moves (position carried through the ping-pong intact)
    assertEquals(centered.x[1], 0, "dead slot moved in x");
    assertEquals(centered.y[1], 0, "dead slot moved in y");
    assertEquals(offside.x[1], 37, "dead slot moved in x");
    assertEquals(offside.y[1], 91, "dead slot moved in y");

    // Dead slot exerts no force: live trajectories are identical no matter
    // where the dead slot sits
    for (const i of [0, 2]) {
      assertEquals(
        centered.x[i],
        offside.x[i],
        `dead slot position leaked into node ${i} (x)`,
      );
      assertEquals(
        centered.y[i],
        offside.y[i],
        `dead slot position leaked into node ${i} (y)`,
      );
    }

    // Sanity: the live pair actually moved (simulation ran)
    assert(
      centered.x[0] !== -100 || centered.x[2] !== 100,
      "live nodes never moved",
    );
  },
);

gpuTest(
  "GPU flags: pinned node stays exactly put under strong forces and still repels",
  async (device) => {
    // Node 0 is pinned at (10, 20) via NODE_FLAG_PINNED. A stretched edge
    // to distant node 2 pulls on it hard every tick; unpinned it would race
    // toward node 2. Node 1 sits right next to the pinned node with no edge:
    // it must be pushed away by the pinned node's repulsion.
    const harness = await createSimHarness(
      device,
      {
        nodeCount: 3,
        positionsX: new Float32Array([10, 13, 510]),
        positionsY: new Float32Array([20, 20, 20]),
        edgeSources: new Uint32Array([0]),
        edgeTargets: new Uint32Array([2]),
        flags: new Uint32Array([NODE_FLAG_PINNED, 0, 0]),
      },
      { centerStrength: 0 },
    );

    try {
      await harness.tick(60);
      const { x, y } = await harness.readPositions();

      // Pinned node held its externally-written position exactly
      assertEquals(x[0], 10, "pinned node drifted in x");
      assertEquals(y[0], 20, "pinned node drifted in y");

      // The pinned node still repels: node 1 started 3 units away and must
      // have been pushed well clear
      const dist1 = Math.hypot(x[1] - 10, y[1] - 20);
      assert(dist1 > 5, `neighbor was not repelled by pinned node: d=${dist1}`);

      // The spring still acts on the free endpoint: node 2 was pulled toward
      // the pinned node
      assert(x[2] < 510, `spring to pinned node never pulled free end: x=${x[2]}`);
    } finally {
      harness.dispose();
    }
  },
);

// ---------------------------------------------------------------------------
// Collision pass (post-integration position resolution)
// ---------------------------------------------------------------------------

for (const variant of ["main", "tiled"] as const) {
  gpuTest(
    `GPU collision (${variant}): pinned node is never displaced but still pushes; dead slots inert`,
    async (device) => {
      // Node 0 pinned at origin, node 1 overlapping it at (3, 0), both
      // radius 5 (min separation 10, overlap 7). Node 2 is a dead slot
      // wedged between them — it must neither move nor push.
      // Expected: node 1 displaced by overlap * 0.5 * strength(0.7) = 2.45
      // (exactly the two-node result), node 0 bit-exact at the origin.
      const result = await runCollision(device, {
        positions: new Float32Array([0, 0, 3, 0, 2, 0]),
        sizes: new Float32Array([5, 5, DEAD_SLOT_RADIUS]),
        flags: new Uint32Array([NODE_FLAG_PINNED, 0, NODE_FLAG_DEAD]),
        variant,
      });

      // Pinned node: bit-exact hold
      assertEquals(result[0], 0, "pinned node displaced in x");
      assertEquals(result[1], 0, "pinned node displaced in y");

      // Dead slot: bit-exact hold
      assertEquals(result[4], 2, "dead slot displaced in x");
      assertEquals(result[5], 0, "dead slot displaced in y");

      // Live node: pushed away by the pinned node's full half-overlap,
      // unaffected by the dead slot
      const expectedX = 3 + (10 - 3) * 0.5 * 0.7;
      assert(
        Math.abs(result[2] - expectedX) < 1e-4,
        `live node not pushed by pinned node: x=${result[2]}, expected ${expectedX}`,
      );
      assertEquals(result[3], 0, "live node displaced in y");
    },
  );
}

gpuTest(
  "GPU collision (grid): pinned node is never displaced but still pushes; dead slots inert",
  async (device) => {
    // Same scenario as the O(n^2) variants, driven through the spatial-hash
    // grid pipeline. The dead slot keeps a live-looking radius so the grid
    // path must exclude it via NODE_FLAG_DEAD alone (not the size sentinel).
    const result = await runCollision(device, {
      positions: new Float32Array([0, 0, 3, 0, 2, 0]),
      sizes: new Float32Array([5, 5, 5]),
      flags: new Uint32Array([NODE_FLAG_PINNED, 0, NODE_FLAG_DEAD]),
      variant: "grid",
      bounds: { minX: -10, minY: -10, maxX: 10, maxY: 10 },
    });

    // Pinned node: bit-exact hold
    assertEquals(result[0], 0, "pinned node displaced in x");
    assertEquals(result[1], 0, "pinned node displaced in y");

    // Dead slot: bit-exact hold despite overlapping both live nodes
    assertEquals(result[4], 2, "dead slot displaced in x");
    assertEquals(result[5], 0, "dead slot displaced in y");

    // Live node: pushed only by the pinned node's half-overlap. If the dead
    // slot leaked into the cell lists it would add its own push and this
    // value would come out ~3.15 higher.
    const expectedX = 3 + (10 - 3) * 0.5 * 0.7;
    assert(
      Math.abs(result[2] - expectedX) < 1e-4,
      `live node not pushed by pinned node alone: x=${result[2]}, expected ${expectedX}`,
    );
    assertEquals(result[3], 0, "live node displaced in y");
  },
);

// ---------------------------------------------------------------------------
// Hidden nodes (NODE_FLAG_HIDDEN_LOD)
// ---------------------------------------------------------------------------

gpuTest(
  "GPU flags: a hidden node exerts no force and never moves",
  async (device) => {
    // WP-A deliberately left the simulation ignoring bit 2 and asserted it
    // inert. WP-D changes that: a hidden node is out of the simulation, not
    // merely out of the picture. It is excluded from the active-index list the
    // force passes dispatch over, and frozen by the integrate early-out — which
    // is what lets a collapsed subtree be expanded again into the arrangement
    // it was collapsed with.
    //
    // Same structure as the dead-slot test, which is the point: hiding must be
    // as invisible to the rest of the layout as a hole is. Two live nodes with
    // an edge, plus a hidden node parked once between them (where a body would
    // repel both) and once far off to the side.
    const run = async (hiddenX: number, hiddenY: number) => {
      const harness = await createSimHarness(
        device,
        {
          nodeCount: 3,
          positionsX: new Float32Array([-100, hiddenX, 100]),
          positionsY: new Float32Array([0, hiddenY, 0]),
          edgeSources: new Uint32Array([0]),
          edgeTargets: new Uint32Array([2]),
          flags: new Uint32Array([0, NODE_FLAG_HIDDEN_LOD, 0]),
        },
        { centerStrength: 0 },
      );
      try {
        await harness.tick(50);
        return await harness.readPositions();
      } finally {
        harness.dispose();
      }
    };

    const centered = await run(0, 0);
    const offside = await run(37, 91);

    // Frozen: the position is carried through the ping-pong untouched.
    assertEquals(centered.x[1], 0, "hidden node moved in x");
    assertEquals(centered.y[1], 0, "hidden node moved in y");
    assertEquals(offside.x[1], 37, "hidden node moved in x");
    assertEquals(offside.y[1], 91, "hidden node moved in y");

    // No force exerted: where the hidden node sits cannot reach the live pair.
    for (const i of [0, 2]) {
      assertEquals(centered.x[i], offside.x[i], `hidden node position leaked into node ${i} (x)`);
      assertEquals(centered.y[i], offside.y[i], `hidden node position leaked into node ${i} (y)`);
    }

    // Sanity: the live pair actually moved, so the assertions above are not
    // passing on a simulation that did nothing.
    assert(centered.x[0] !== -100 || centered.x[2] !== 100, "live nodes never moved");
  },
);

gpuTest(
  "GPU flags: hiding a node mid-run freezes it where it stood",
  async (device) => {
    // The transition case the steady-state test above cannot reach: a node that
    // was moving when the cut hid it. It must stop dead — not drift, and not
    // oscillate between the two ping-pong buffers' last values, which is the
    // failure mode of compacting the integrate dispatch along with the force
    // passes.
    const harness = await createSimHarness(
      device,
      {
        nodeCount: 3,
        positionsX: new Float32Array([-100, 0, 100]),
        positionsY: new Float32Array([0, 5, 0]),
        edgeSources: new Uint32Array([0]),
        edgeTargets: new Uint32Array([2]),
      },
      { centerStrength: 0 },
    );
    try {
      await harness.tick(20);
      const moving = await harness.readPositions();
      assert(moving.y[1] !== 5, "node 1 must be in motion before it is hidden");

      harness.setNodeFlags(new Uint32Array([0, NODE_FLAG_HIDDEN_LOD, 0]));
      assertEquals(harness.activeCount, 2, "the hidden slot must leave the dispatch list");

      // An odd tick count so a two-buffer oscillation cannot alias back to the
      // value it was frozen at.
      await harness.tick(21);
      const frozen = await harness.readPositions();
      assertFrozen(moving, frozen, [1]);

      // The rest of the graph keeps simulating.
      assert(
        frozen.x[0] !== moving.x[0] || frozen.x[2] !== moving.x[2],
        "the visible nodes stopped moving too",
      );
    } finally {
      harness.dispose();
    }
  },
);

for (const variant of ["main", "tiled", "grid"] as const) {
  gpuTest(
    `GPU collision (${variant}): a hidden node is neither displaced nor a pusher`,
    async (device) => {
      // Node 0 pinned at the origin, node 1 overlapping it at (3, 0), and a
      // HIDDEN node at (2, 0) with a perfectly live radius — so the only thing
      // that can keep it out of the collision set is the flag. Node 1 must be
      // pushed by the pinned node alone, exactly as in the dead-slot case; if
      // the hidden node leaked in it would add its own push and land node 1
      // further out.
      const result = await runCollision(device, {
        positions: new Float32Array([0, 0, 3, 0, 2, 0]),
        sizes: new Float32Array([5, 5, 5]),
        flags: new Uint32Array([NODE_FLAG_PINNED, 0, NODE_FLAG_HIDDEN_LOD]),
        variant,
        bounds: { minX: -10, minY: -10, maxX: 10, maxY: 10 },
      });

      assertEquals(result[0], 0, "pinned node displaced in x");
      assertEquals(result[1], 0, "pinned node displaced in y");
      assertEquals(result[4], 2, "hidden node displaced in x");
      assertEquals(result[5], 0, "hidden node displaced in y");

      const expectedX = 3 + (10 - 3) * 0.5 * 0.7;
      assert(
        Math.abs(result[2] - expectedX) < 1e-4,
        `live node not pushed by the pinned node alone: x=${result[2]}, expected ${expectedX}`,
      );
      assertEquals(result[3], 0, "live node displaced in y");
    },
  );
}

gpuTest(
  "GPU collision: the hidden bit does not disturb dead or pinned handling",
  async (device) => {
    // The dead/pinned scenario with every node ALSO flagged hidden. The three
    // bits are independent, and the masks that read two of them must not
    // reorder into one that catches the third by accident.
    const result = await runCollision(device, {
      positions: new Float32Array([0, 0, 3, 0, 2, 0]),
      sizes: new Float32Array([5, 5, DEAD_SLOT_RADIUS]),
      flags: new Uint32Array([
        NODE_FLAG_PINNED | NODE_FLAG_HIDDEN_LOD,
        NODE_FLAG_HIDDEN_LOD,
        NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD,
      ]),
      variant: "main",
    });

    // Everything is inert now, so nothing may move at all.
    assertEquals(result[0], 0, "pinned+hidden node displaced in x");
    assertEquals(result[1], 0, "pinned+hidden node displaced in y");
    assertEquals(result[2], 3, "hidden node displaced in x");
    assertEquals(result[3], 0, "hidden node displaced in y");
    assertEquals(result[4], 2, "dead+hidden slot displaced in x");
    assertEquals(result[5], 0, "dead+hidden slot displaced in y");
  },
);

// ---------------------------------------------------------------------------
// Hidden nodes in the render pipeline
// ---------------------------------------------------------------------------

/**
 * Structural view of renderer/pipelines/nodes.ts, which imports .wgsl and so
 * cannot be referenced with `import type` (see tests/helpers/gpu.ts module doc).
 */
type NodeRenderPipelineHandle = { readonly __brand?: "NodeRenderPipeline" };
interface NodesModule {
  createNodeRenderPipeline(
    context: { device: GPUDevice },
    config?: { maxNodes?: number; sampleCount?: number; format?: GPUTextureFormat },
  ): NodeRenderPipelineHandle;
  createNodeBindGroup(
    device: GPUDevice,
    pipeline: NodeRenderPipelineHandle,
    positions: GPUBuffer,
    nodeAttrs: GPUBuffer,
    nodeFlags: GPUBuffer,
    nodeAlpha: GPUBuffer,
  ): GPUBindGroup;
  createViewportBindGroup(
    device: GPUDevice,
    pipeline: NodeRenderPipelineHandle,
    viewportUniformBuffer: GPUBuffer,
  ): GPUBindGroup;
  createRenderConfigBindGroup(
    device: GPUDevice,
    pipeline: NodeRenderPipelineHandle,
    renderConfigBuffer: GPUBuffer,
  ): GPUBindGroup;
  renderNodes(
    pass: GPURenderPassEncoder,
    pipeline: NodeRenderPipelineHandle,
    viewportBindGroup: GPUBindGroup,
    nodeBindGroup: GPUBindGroup,
    renderConfigBindGroup: GPUBindGroup,
    nodeCount: number,
  ): void;
}

/** Square render target; 64 * 4 bytes = 256 = WebGPU's bytesPerRow alignment. */
const RENDER_SIZE = 64;
/** Node radius in graph units, and px-per-graph-unit, giving an 8 px disc. */
const NODE_RADIUS = 5;
const PX_PER_UNIT = 1.6;

/**
 * Draws two nodes through the real node render pipeline — one centred in the
 * left half of the target, one in the right — and returns the peak coverage
 * (destination alpha) found in each half.
 */
async function renderTwoNodes(
  device: GPUDevice,
  flags: Uint32Array,
): Promise<{ left: number; right: number }> {
  const mod = await loadModuleInliningWgsl<NodesModule>(
    new URL("../../packages/core/src/renderer/pipelines/nodes.ts", import.meta.url),
  );
  const pipeline = mod.createNodeRenderPipeline({ device }, {
    maxNodes: 2,
    format: "rgba8unorm",
  });

  // Clip-space x -0.5 / +0.5 => pixel columns 16 and 48 of a 64-wide target.
  const positionsData = new Float32Array([-0.5, 0, 0.5, 0]);
  const attrsData = new Float32Array(2 * 8);
  for (let i = 0; i < 2; i++) {
    attrsData[i * 8] = NODE_RADIUS;
    attrsData[i * 8 + 1] = 1; // color r
    attrsData[i * 8 + 2] = 1; // color g
    attrsData[i * 8 + 3] = 1; // color b
  }

  const positions = device.createBuffer({
    size: positionsData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(positions, 0, positionsData.slice().buffer);
  const nodeAttrs = device.createBuffer({
    size: attrsData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(nodeAttrs, 0, attrsData.slice().buffer);
  const nodeFlags = device.createBuffer({
    size: flags.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(nodeFlags, 0, flags.slice().buffer);

  // Crossfade alpha, all opaque: this test is about the hide flag, so every
  // node renders at full strength or not at all.
  const nodeAlpha = device.createBuffer({
    size: flags.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(nodeAlpha, 0, new Float32Array(flags.length).fill(1));

  // ViewportUniforms: identity graph->clip transform, so the positions above
  // are already clip coordinates.
  const viewportData = new Float32Array(20);
  viewportData[0] = 1; // col0.x
  viewportData[5] = 1; // col1.y
  viewportData[10] = 1; // col2.z
  viewportData[12] = RENDER_SIZE; // screen_size.x
  viewportData[13] = RENDER_SIZE; // screen_size.y
  viewportData[14] = PX_PER_UNIT; // scale
  viewportData[15] = 1 / PX_PER_UNIT; // inv_scale
  viewportData[16] = 1; // dpr
  const viewportUniforms = device.createBuffer({
    size: viewportData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(viewportUniforms, 0, viewportData.slice().buffer);

  // RenderConfig: 80 bytes, all zeros = borders and birth pulse disabled.
  const renderConfig = device.createBuffer({
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const target = device.createTexture({
    size: { width: RENDER_SIZE, height: RENDER_SIZE },
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const pixelBuffer = device.createBuffer({
    size: RENDER_SIZE * RENDER_SIZE * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: target.createView(),
      // Alpha 0: any non-zero alpha in the readback is a fragment we wrote.
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  mod.renderNodes(
    pass,
    pipeline,
    mod.createViewportBindGroup(device, pipeline, viewportUniforms),
    mod.createNodeBindGroup(device, pipeline, positions, nodeAttrs, nodeFlags, nodeAlpha),
    mod.createRenderConfigBindGroup(device, pipeline, renderConfig),
    2,
  );
  pass.end();
  encoder.copyTextureToBuffer(
    { texture: target },
    { buffer: pixelBuffer, bytesPerRow: RENDER_SIZE * 4, rowsPerImage: RENDER_SIZE },
    { width: RENDER_SIZE, height: RENDER_SIZE },
  );
  device.queue.submit([encoder.finish()]);

  await pixelBuffer.mapAsync(GPUMapMode.READ);
  const pixels = new Uint8Array(pixelBuffer.getMappedRange().slice(0));
  pixelBuffer.unmap();

  const half = RENDER_SIZE / 2;
  let left = 0;
  let right = 0;
  for (let y = 0; y < RENDER_SIZE; y++) {
    for (let x = 0; x < RENDER_SIZE; x++) {
      const alpha = pixels[(y * RENDER_SIZE + x) * 4 + 3];
      if (x < half) {
        if (alpha > left) left = alpha;
      } else if (alpha > right) right = alpha;
    }
  }

  positions.destroy();
  nodeAttrs.destroy();
  nodeFlags.destroy();
  nodeAlpha.destroy();
  viewportUniforms.destroy();
  renderConfig.destroy();
  target.destroy();
  pixelBuffer.destroy();
  return { left, right };
}

gpuTest(
  "GPU render: a hidden node produces no fragments while its neighbour still draws",
  async (device) => {
    // Control: neither node hidden, both halves covered. This is what makes the
    // hidden assertions below meaningful — without it, a broken fixture that
    // draws nothing at all would pass.
    const both = await renderTwoNodes(device, new Uint32Array([0, 0]));
    assert(both.left > 0, "control: left node did not render");
    assert(both.right > 0, "control: right node did not render");

    // Hiding the left node must clear exactly the left half. Nothing else about
    // the node changed — same position, same radius, same colour.
    const leftHidden = await renderTwoNodes(
      device,
      new Uint32Array([NODE_FLAG_HIDDEN_LOD, 0]),
    );
    assertEquals(leftHidden.left, 0, "hidden node still produced fragments");
    assertEquals(leftHidden.right, both.right, "visible node's coverage changed");

    // Mirrored, so the result tracks the flag rather than the instance index.
    const rightHidden = await renderTwoNodes(
      device,
      new Uint32Array([0, NODE_FLAG_HIDDEN_LOD]),
    );
    assertEquals(rightHidden.right, 0, "hidden node still produced fragments");
    assertEquals(rightHidden.left, both.left, "visible node's coverage changed");
  },
);
