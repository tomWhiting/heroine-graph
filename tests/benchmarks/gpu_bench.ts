/**
 * GPU pipeline benchmarks — real measurements of the actual library.
 *
 * Two families of benches, both driving the same code paths GraphMother
 * runs per frame, on a headless WebGPU device (no canvas):
 *
 * - "sim tick": one full simulation step through the real compute
 *   pipeline (uniform upload, force clear, N² repulsion, springs,
 *   integration, ping-pong swap, bind-group selection). Graphs are
 *   hub-heavy code trees from the shared fixture — the project's target
 *   workload.
 *
 * - "node render": one render pass of the real instanced node pipeline
 *   (renderer/pipelines/nodes.ts) drawing N SDF circles into an offscreen
 *   1920x1080 target.
 *
 * Numbers are honest wall-clock times including the per-frame CPU work the
 * library actually does (encode, submit, bind-group selection); they are not
 * idealized GPU-only timings. But they are **await-amortized**: K operations
 * are submitted per queue fence and the time divided by K, because the fence
 * itself costs ~13-14 ms under Deno and otherwise dominates everything (see
 * ./amortize.ts). The printed `fence ≤` column bounds the residual share of
 * that cost in each number; unamortized it was 90%+, which is what made the
 * previous version of this file report a 17x spread as "everything is 15 ms".
 *
 * This file does not use `Deno.bench`: its reported time is per iteration and
 * there is no way to declare that one iteration covered K operations, which
 * is exactly the transformation the fence floor forces. It runs its own
 * calibrate/warm/sample loop and prints a table instead. `deno task bench`
 * still executes it (and then reports zero registered benches).
 *
 * Run with: deno task bench
 * Requires a WebGPU adapter; prints why and stops without one.
 */

import {
  createSimHarness,
  GPU_SKIP_MESSAGE,
  loadModuleInliningWgsl,
  probeAdapter,
  requestHarnessDevice,
  waitForQueue,
} from "../helpers/gpu.ts";
import { CODE_TREE_SCALES, generateCodeTree } from "../fixtures/code_tree.ts";
import { countNonFinite, maxRadius, type PositionSnapshot } from "../helpers/invariants.ts";
import { measureAmortized, measureAwaitFloorMs } from "./amortize.ts";
import { type BenchRow, formatBenchTable } from "./report.ts";

// =============================================================================
// Structural types for the node render pipeline module
// (renderer/pipelines/nodes.ts imports .wgsl, so it is loaded via
// loadModuleInliningWgsl and cannot be referenced with `import type`;
// see tests/helpers/gpu.ts module doc.)
// =============================================================================

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

// =============================================================================
// Workloads
// =============================================================================

/**
 * Reference figures are the Phase 3 design's measured envelope (§4), taken on
 * an Apple M1 Max under Deno 2.9.4 headless WebGPU. They are printed as a
 * delta and never asserted: this bench must stay runnable on other hardware.
 */
const SIM_WORKLOADS = [
  { scale: CODE_TREE_SCALES.small, referenceMs: 0.91 },
  { scale: CODE_TREE_SCALES.medium, referenceMs: 15.67 },
  { scale: CODE_TREE_SCALES.large, referenceMs: 461.36 },
] as const;

const RENDER_WORKLOADS = [
  { nodeCount: 35_000, referenceMs: 0.73 },
  { nodeCount: 220_000, referenceMs: 1.28 },
  { nodeCount: 500_000, referenceMs: 2.22 },
] as const;

/** The design's envelope is stated at 1080p; matching it keeps the deltas meaningful. */
const RENDER_WIDTH = 1920;
const RENDER_HEIGHT = 1080;

/** Node attribute floats per node, mirroring graph_state.ts NODE_ATTR_FLOATS. */
const NODE_ATTR_FLOATS = 8;

/** Fraction of the shorter viewport axis the seeded disc is fitted into. */
const RENDER_DISC_FILL = 0.9;

/**
 * Edge of the square read back to prove the render pass rasterized. 64 px
 * makes `bytesPerRow` exactly the 256-byte copy alignment.
 */
const RASTER_PROBE_SIZE = 64;

// =============================================================================
// Run
// =============================================================================

const adapter = await probeAdapter();

if (!adapter) {
  console.warn(`[bench] ${GPU_SKIP_MESSAGE}`);
} else {
  const device = await requestHarnessDevice(adapter);
  const awaitFloorMs = await measureAwaitFloorMs(device);
  const rows: BenchRow[] = [];

  for (const { scale, referenceMs } of SIM_WORKLOADS) {
    const tree = generateCodeTree(scale);
    const harness = await createSimHarness(device, tree);
    const workload = `sim tick, N²: ${count(tree.nodeCount)} nodes / ${
      count(tree.edgeCount)
    } edges`;
    try {
      // SimHarness.tick(K) is already K submits behind a single fence.
      const timing = await measureAmortized((units) => harness.tick(units), awaitFloorMs);
      assertLayoutSurvived(await harness.readPositions(), workload);
      rows.push({ workload, timing, referenceMs });
    } finally {
      harness.dispose();
    }
  }

  const nodesModule = await loadModuleInliningWgsl<NodesModule>(
    new URL("../../packages/core/src/renderer/pipelines/nodes.ts", import.meta.url),
  );

  const target = device.createTexture({
    label: "Bench Render Target",
    size: { width: RENDER_WIDTH, height: RENDER_HEIGHT },
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const targetView = target.createView();

  for (const { nodeCount, referenceMs } of RENDER_WORKLOADS) {
    const frame = createNodeRenderFrame(device, nodesModule, targetView, nodeCount);
    const workload = `node render: ${count(nodeCount)} nodes @ ${RENDER_WIDTH}x${RENDER_HEIGHT}`;
    try {
      await assertFrameRasterized(device, target, frame, workload);
      const timing = await measureAmortized(async (units) => {
        for (let i = 0; i < units; i++) frame.submit();
        await waitForQueue(device);
      }, awaitFloorMs);
      rows.push({ workload, timing, referenceMs });
    } finally {
      frame.destroy();
    }
  }

  target.destroy();

  console.log(`\nawait floor (waitForQueue on an idle queue): ${awaitFloorMs.toFixed(2)} ms\n`);
  console.log(formatBenchTable(rows));
  console.log(
    "\nreference = Phase 3 design §4 (Apple M1 Max, Deno 2.9.4); informational, not asserted\n",
  );
}

// =============================================================================
// Node render workload
// =============================================================================

/** One preconfigured node render frame plus the GPU resources it owns. */
interface NodeRenderFrame {
  /** Encodes and submits one render pass. Does not fence the queue. */
  submit(): void;
  destroy(): void;
}

/**
 * Builds the real node render pipeline over a phyllotaxis disc of `nodeCount`
 * nodes, matching the library's default seeding, fitted to the render target.
 */
function createNodeRenderFrame(
  device: GPUDevice,
  nodesModule: NodesModule,
  targetView: GPUTextureView,
  nodeCount: number,
): NodeRenderFrame {
  const pipeline = nodesModule.createNodeRenderPipeline({ device }, {
    maxNodes: nodeCount,
    format: "rgba8unorm",
  });

  const discRadius = Math.sqrt(nodeCount) * 10;
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const positionsData = new Float32Array(nodeCount * 2);
  const attrsData = new Float32Array(nodeCount * NODE_ATTR_FLOATS);
  for (let i = 0; i < nodeCount; i++) {
    const angle = i * goldenAngle;
    const r = Math.sqrt(i / nodeCount) * discRadius;
    positionsData[i * 2] = r * Math.cos(angle);
    positionsData[i * 2 + 1] = r * Math.sin(angle);
    const attr = i * NODE_ATTR_FLOATS;
    attrsData[attr] = 5; // radius
    attrsData[attr + 1] = 0.3; // color r
    attrsData[attr + 2] = 0.5; // color g
    attrsData[attr + 3] = 0.9; // color b
    // selected / hovered / birth_time / tex_index stay 0
  }

  const positions = device.createBuffer({
    label: "Bench Node Positions",
    size: positionsData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(positions, 0, positionsData);

  const nodeAttrs = device.createBuffer({
    label: "Bench Node Attributes",
    size: attrsData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(nodeAttrs, 0, attrsData);

  // Node state flags; zero-filled = every node live and visible.
  const nodeFlags = device.createBuffer({
    label: "Bench Node Flags",
    size: nodeCount * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // ViewportUniforms (node.vert.wgsl): 3 vec4 transform columns, screen_size,
  // scale (px per graph unit), inv_scale, dpr, padding. The transform maps
  // graph units straight to clip space, so each axis carries the aspect ratio.
  const pxPerUnit = (RENDER_DISC_FILL * Math.min(RENDER_WIDTH, RENDER_HEIGHT)) / (2 * discRadius);
  const viewportData = new Float32Array(20);
  viewportData[0] = (2 * pxPerUnit) / RENDER_WIDTH; // col0.x
  viewportData[5] = (2 * pxPerUnit) / RENDER_HEIGHT; // col1.y
  viewportData[10] = 1; // col2.z
  viewportData[12] = RENDER_WIDTH; // screen_size.x
  viewportData[13] = RENDER_HEIGHT; // screen_size.y
  viewportData[14] = pxPerUnit; // scale
  viewportData[15] = 1 / pxPerUnit; // inv_scale
  viewportData[16] = 1; // dpr
  const viewportUniforms = device.createBuffer({
    label: "Bench Viewport Uniforms",
    size: viewportData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(viewportUniforms, 0, viewportData);

  // RenderConfig (node.frag.wgsl): 80 bytes; all zeros = borders and
  // birth-pulse disabled, which is the steady-state render path.
  const renderConfig = device.createBuffer({
    label: "Bench Render Config",
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const viewportBindGroup = nodesModule.createViewportBindGroup(device, pipeline, viewportUniforms);
  const nodeBindGroup = nodesModule.createNodeBindGroup(
    device,
    pipeline,
    positions,
    nodeAttrs,
    nodeFlags,
  );
  const renderConfigBindGroup = nodesModule.createRenderConfigBindGroup(
    device,
    pipeline,
    renderConfig,
  );

  return {
    submit(): void {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: targetView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      nodesModule.renderNodes(
        pass,
        pipeline,
        viewportBindGroup,
        nodeBindGroup,
        renderConfigBindGroup,
        nodeCount,
      );
      pass.end();
      device.queue.submit([encoder.finish()]);
    },
    destroy(): void {
      positions.destroy();
      nodeAttrs.destroy();
      nodeFlags.destroy();
      viewportUniforms.destroy();
      renderConfig.destroy();
    },
  };
}

// =============================================================================
// Credibility guards
//
// A pipeline that silently does nothing benchmarks as very fast and reads as
// an improvement. Both families are therefore checked for having done work
// before their number is believed.
// =============================================================================

/** Fails unless the simulation left a finite, non-degenerate layout behind. */
function assertLayoutSurvived(positions: PositionSnapshot, workload: string): void {
  const nonFinite = countNonFinite(positions.x, positions.y);
  if (nonFinite > 0) {
    throw new Error(`${workload}: ${nonFinite} non-finite coordinates after the run`);
  }
  if (maxRadius(positions.x, positions.y) === 0) {
    throw new Error(`${workload}: layout collapsed onto a single point`);
  }
}

/**
 * Fails unless one frame leaves fragments at the centre of the target, where
 * the seeded disc is densest.
 */
async function assertFrameRasterized(
  device: GPUDevice,
  target: GPUTexture,
  frame: NodeRenderFrame,
  workload: string,
): Promise<void> {
  const bytesPerRow = RASTER_PROBE_SIZE * 4;
  const readback = device.createBuffer({
    label: "Bench Raster Probe",
    size: bytesPerRow * RASTER_PROBE_SIZE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    frame.submit();
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      {
        texture: target,
        origin: {
          x: (RENDER_WIDTH - RASTER_PROBE_SIZE) / 2,
          y: (RENDER_HEIGHT - RASTER_PROBE_SIZE) / 2,
        },
      },
      { buffer: readback, bytesPerRow },
      { width: RASTER_PROBE_SIZE, height: RASTER_PROBE_SIZE },
    );
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(readback.getMappedRange());
    let lit = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i] | pixels[i + 1] | pixels[i + 2]) lit++;
    }
    readback.unmap();
    if (lit === 0) {
      throw new Error(`${workload}: render pass produced no fragments at the frame centre`);
    }
  } finally {
    readback.destroy();
  }
}

function count(value: number): string {
  return value.toLocaleString("en-US");
}
