/**
 * GPU tests for the simple contour layer's multi-threshold rendering.
 *
 * Regression coverage: all thresholds used to share ONE uniform buffer,
 * and queue.writeBuffer executes at call time while encoded render
 * passes execute at the frame's single queue.submit — so every pass
 * sampled only the LAST threshold written. The layer now keeps one
 * uniform buffer + bind group per threshold.
 *
 * Verified two ways against the real layer + pipeline + shaders:
 * 1. Readback of the per-threshold uniform buffers after a full
 *    encode+submit — each must still hold its own threshold.
 * 2. Readback of the render target: a horizontal density gradient must
 *    produce a visible isoline band at EVERY configured threshold (the
 *    broken path drew only the last threshold's isoline, N times).
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  GPU_SKIP_MESSAGE,
  loadModuleInliningWgsl,
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

// Structural types for the contour layer module (cannot be referenced
// via `import type`: deno test's loader would follow it into the
// .wgsl-importing pipeline module — see tests/helpers/gpu.ts module doc).
interface ContourLayerLike {
  enabled: boolean;
  setRenderContext(
    ctx: { densityTextureView: GPUTextureView; maxDensity: number },
  ): void;
  setConfig(config: Record<string, unknown>): void;
  getThresholdUniformBuffers(): readonly GPUBuffer[];
  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void;
  destroy(): void;
}
interface ContourModule {
  createSimpleContourLayer(
    id: string,
    context: unknown,
    config?: Record<string, unknown>,
  ): ContourLayerLike;
}

function loadContourModule(): Promise<ContourModule> {
  return loadModuleInliningWgsl<ContourModule>(
    new URL("../../packages/core/src/layers/contour/simple-layer.ts", import.meta.url),
  );
}

const WIDTH = 64; // 64 * 4 bytes = 256 = WebGPU bytesPerRow alignment
const HEIGHT = 16;
const FORMAT: GPUTextureFormat = "rgba8unorm";

/** Density texture whose red channel ramps 0 -> 1 across x. */
function createGradientDensityTexture(device: GPUDevice): GPUTexture {
  const texture = device.createTexture({
    label: "Contour Test Density Gradient",
    size: { width: WIDTH, height: HEIGHT },
    format: FORMAT,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const data = new Uint8Array(WIDTH * HEIGHT * 4);
  for (let y = 0; y < HEIGHT; y++) {
    for (let x = 0; x < WIDTH; x++) {
      data[(y * WIDTH + x) * 4] = Math.round((x / (WIDTH - 1)) * 255);
      data[(y * WIDTH + x) * 4 + 3] = 255;
    }
  }
  device.queue.writeTexture(
    { texture },
    data,
    { bytesPerRow: WIDTH * 4, rowsPerImage: HEIGHT },
    { width: WIDTH, height: HEIGHT },
  );
  return texture;
}

async function readbackBuffer(
  device: GPUDevice,
  src: GPUBuffer,
  size: number,
): Promise<ArrayBuffer> {
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(src, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return copy;
}

gpuTest(
  "GPU contour: every threshold keeps its own uniforms and draws its own isoline",
  async (device) => {
    const mod = await loadContourModule();
    const context = { device, format: FORMAT };
    const thresholds = [0.2, 0.8];
    // strokeWidth 4 -> density band halfwidth 0.04 (+0.0075 feather):
    // wide enough to cover a few texel columns of the 64-wide gradient.
    const layer = mod.createSimpleContourLayer("contour-test", context, {
      enabled: true,
      thresholds,
      strokeWidth: 4,
      strokeColor: "#ff0000",
      opacity: 1,
    });

    const density = createGradientDensityTexture(device);
    layer.setRenderContext({
      densityTextureView: density.createView(),
      maxDensity: 1,
    });

    const target = device.createTexture({
      label: "Contour Test Target",
      size: { width: WIDTH, height: HEIGHT },
      format: FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const targetView = target.createView();

    // One frame, one submit — exactly the api/graph.ts frame flow the
    // shared-buffer bug depended on.
    const encoder = device.createCommandEncoder();
    const clear = encoder.beginRenderPass({
      colorAttachments: [{
        view: targetView,
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        storeOp: "store",
      }],
    });
    clear.end();
    layer.render(encoder, targetView);

    const pixelBuffer = device.createBuffer({
      size: WIDTH * HEIGHT * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyTextureToBuffer(
      { texture: target },
      { buffer: pixelBuffer, bytesPerRow: WIDTH * 4, rowsPerImage: HEIGHT },
      { width: WIDTH, height: HEIGHT },
    );
    device.queue.submit([encoder.finish()]);

    // 1. Each per-threshold uniform buffer must retain its own threshold
    //    (offset 6 floats in: color vec4, thickness, feather, threshold).
    const uniformBuffers = layer.getThresholdUniformBuffers();
    assertEquals(uniformBuffers.length, thresholds.length);
    for (let i = 0; i < thresholds.length; i++) {
      const floats = new Float32Array(await readbackBuffer(device, uniformBuffers[i], 32));
      assertEquals(
        floats[6],
        Math.fround(thresholds[i]),
        `uniform buffer ${i} must hold threshold ${thresholds[i]} after submit`,
      );
      assertEquals(floats[7], 1, "maxDensity uniform");
    }

    // 2. The rendered frame must contain an isoline band at EVERY
    //    threshold. Density at column x is x/63, so the 0.2 isoline
    //    falls near column 13 and the 0.8 isoline near column 50; the
    //    midpoint (density ~0.5) must stay empty.
    await pixelBuffer.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(pixelBuffer.getMappedRange().slice(0));
    pixelBuffer.unmap();

    const row = 8; // gradient is constant in y; sample one interior row
    const alphaAt = (x: number) => pixels[(row * WIDTH + x) * 4 + 3];
    const maxAlpha = (x0: number, x1: number) => {
      let max = 0;
      for (let x = x0; x <= x1; x++) max = Math.max(max, alphaAt(x));
      return max;
    };

    assert(
      maxAlpha(10, 16) > 0,
      "isoline for threshold 0.2 missing — earlier thresholds must not be clobbered by later uniform writes",
    );
    assert(maxAlpha(47, 54) > 0, "isoline for threshold 0.8 missing");
    assertEquals(
      maxAlpha(25, 40),
      0,
      "no isoline expected between the two thresholds",
    );

    layer.destroy();
    density.destroy();
    target.destroy();
    pixelBuffer.destroy();
  },
);

gpuTest(
  "GPU contour: threshold buffer pool tracks config changes",
  async (device) => {
    const mod = await loadContourModule();
    const context = { device, format: FORMAT };
    const layer = mod.createSimpleContourLayer("contour-pool-test", context, {
      enabled: true,
      thresholds: [0.2, 0.4, 0.6, 0.8],
    });

    const density = createGradientDensityTexture(device);
    layer.setRenderContext({
      densityTextureView: density.createView(),
      maxDensity: 1,
    });

    const target = device.createTexture({
      size: { width: WIDTH, height: HEIGHT },
      format: FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const targetView = target.createView();

    const renderOnce = () => {
      const encoder = device.createCommandEncoder();
      layer.render(encoder, targetView);
      device.queue.submit([encoder.finish()]);
    };

    renderOnce();
    assertEquals(layer.getThresholdUniformBuffers().length, 4);
    const before = [...layer.getThresholdUniformBuffers()];

    // Unchanged config: buffers are reused, not reallocated per frame
    renderOnce();
    const after = layer.getThresholdUniformBuffers();
    for (let i = 0; i < before.length; i++) {
      assert(before[i] === after[i], `buffer ${i} must be reused across frames`);
    }

    // Shrinking the threshold list shrinks the pool
    layer.setConfig({ thresholds: [0.5] });
    renderOnce();
    assertEquals(layer.getThresholdUniformBuffers().length, 1);
    const floats = new Float32Array(
      await readbackBuffer(device, layer.getThresholdUniformBuffers()[0], 32),
    );
    assertEquals(floats[6], 0.5);

    layer.destroy();
    density.destroy();
    target.destroy();
  },
);
