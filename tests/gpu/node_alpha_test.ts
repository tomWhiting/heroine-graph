/**
 * GPU tests for the per-node render-alpha buffer (SimulationBuffers.nodeAlpha,
 * bound at @group(1) @binding(3) of the node render pipeline).
 *
 * Two things must hold:
 *
 * - **With no crossfade in flight the buffer is all-1.0 and the multiply is
 *   the identity.** A freshly allocated buffer is checked by readback, and a
 *   node rendered at alpha 1.0 must produce bit-exactly the unblended, fully
 *   opaque node colour — the pixels the pipeline produced before the buffer
 *   existed.
 * - **Alpha is per node and blends.** A node at 0.5 composites at half
 *   strength over the background, a node at 0 contributes nothing, and in both
 *   cases its neighbour's pixels are byte-identical to the all-opaque run.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  GPU_SKIP_MESSAGE,
  loadModuleInliningWgsl,
  loadPipelineModule,
  probeAdapter,
  requestHarnessDevice,
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
/** Opaque background the nodes composite over. */
const BACKGROUND: GPUColorDict = { r: 0, g: 0, b: 0, a: 1 };

/**
 * Draws two white nodes through the real node render pipeline — one centred in
 * the left half of the target, one in the right — over an opaque black
 * background, and returns the raw RGBA bytes.
 */
async function renderTwoNodes(
  device: GPUDevice,
  alphas: Float32Array,
): Promise<Uint8Array> {
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

  const storage = (data: Float32Array | Uint32Array): GPUBuffer => {
    const buffer = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, data.slice().buffer);
    return buffer;
  };
  const positions = storage(positionsData);
  const nodeAttrs = storage(attrsData);
  const nodeFlags = storage(new Uint32Array(2)); // all live, none hidden
  const nodeAlpha = storage(alphas);

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
      clearValue: BACKGROUND,
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

  for (
    const resource of [
      positions,
      nodeAttrs,
      nodeFlags,
      nodeAlpha,
      viewportUniforms,
      renderConfig,
      target,
      pixelBuffer,
    ]
  ) {
    resource.destroy();
  }
  return pixels;
}

/** Red channel at the centre of the left / right node. */
function centres(pixels: Uint8Array): { left: number; right: number } {
  const row = RENDER_SIZE / 2;
  const index = (x: number): number => (row * RENDER_SIZE + x) * 4;
  return { left: pixels[index(16)], right: pixels[index(48)] };
}

/** Bytes of one vertical half of the target. */
function half(pixels: Uint8Array, side: "left" | "right"): Uint8Array {
  const mid = RENDER_SIZE / 2;
  const out = new Uint8Array(RENDER_SIZE * mid * 4);
  let w = 0;
  for (let y = 0; y < RENDER_SIZE; y++) {
    const x0 = side === "left" ? 0 : mid;
    for (let x = x0; x < x0 + mid; x++) {
      const p = (y * RENDER_SIZE + x) * 4;
      out[w++] = pixels[p];
      out[w++] = pixels[p + 1];
      out[w++] = pixels[p + 2];
      out[w++] = pixels[p + 3];
    }
  }
  return out;
}

gpuTest(
  "GPU alpha: a freshly allocated nodeAlpha buffer is entirely 1.0",
  async (device) => {
    const mod = await loadPipelineModule();
    const nodeCount = 37; // not a multiple of anything; every slot must be set
    const buffers = mod.createSimulationBuffers(device, nodeCount, 0);

    const readback = device.createBuffer({
      size: nodeCount * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffers.nodeAlpha, 0, readback, 0, nodeCount * 4);
    device.queue.submit([encoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const alphas = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();

    assertEquals(
      Array.from(alphas),
      new Array(nodeCount).fill(1),
      "nodeAlpha was not initialized to fully opaque",
    );
  },
);

gpuTest(
  "GPU alpha: at 1.0 the node renders bit-exactly opaque",
  async (device) => {
    // The identity property SC-005 rests on: with no crossfade in flight the
    // fragment shader's multiply changes nothing, so a fully covered pixel of a
    // white node is the unblended white it was before the buffer existed.
    const pixels = await renderTwoNodes(device, new Float32Array([1, 1]));
    const { left, right } = centres(pixels);
    assertEquals(left, 255, "opaque node did not composite at full strength");
    assertEquals(right, 255, "opaque node did not composite at full strength");
  },
);

gpuTest(
  "GPU alpha: 0.5 blends at half strength and does not touch its neighbour",
  async (device) => {
    const opaque = await renderTwoNodes(device, new Float32Array([1, 1]));
    const faded = await renderTwoNodes(device, new Float32Array([0.5, 1]));

    // White node over black background at alpha 0.5 => the 8-bit rounding of
    // 127.5. Either neighbour is correct; 255 (no blend) and 0 (no draw) are not.
    const { left, right } = centres(faded);
    assert(
      left === 127 || left === 128,
      `node at alpha 0.5 composited at ${left}, expected ~127`,
    );
    assertEquals(right, 255, "the un-faded node was dimmed");

    // Alpha is strictly per node: every byte of the other half is unchanged.
    assertEquals(
      half(faded, "right"),
      half(opaque, "right"),
      "fading one node changed the other node's pixels",
    );
  },
);

gpuTest(
  "GPU alpha: 0.0 removes the node entirely and leaves the background intact",
  async (device) => {
    const opaque = await renderTwoNodes(device, new Float32Array([1, 1]));
    const gone = await renderTwoNodes(device, new Float32Array([0, 1]));

    const background = await renderTwoNodes(device, new Float32Array([0, 0]));
    assertEquals(
      half(gone, "left"),
      half(background, "left"),
      "a node at alpha 0 still contributed to the frame",
    );
    assertEquals(
      half(gone, "right"),
      half(opaque, "right"),
      "fading one node out changed the other node's pixels",
    );

    // Sanity: the fixture really does draw something at full alpha, so the
    // equality above is not two blank halves agreeing.
    assertEquals(centres(opaque).left, 255);
    assertEquals(centres(gone).left, 0);
  },
);
