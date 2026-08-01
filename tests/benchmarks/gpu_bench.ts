/**
 * GPU pipeline benchmarks — real measurements of the actual library.
 *
 * Two families of benches, both driving the same code paths GraphMother
 * runs per frame, on a headless WebGPU device (no canvas):
 *
 * - "sim tick": one full simulation step through the real compute
 *   pipeline (uniform upload, force clear, N² repulsion, springs,
 *   integration, ping-pong swap, bind-group rebuild), awaited to GPU
 *   completion. Graphs are hub-heavy code trees from the shared fixture —
 *   the project's target workload.
 *
 * - "node render": one render pass of the real instanced node pipeline
 *   (renderer/pipelines/nodes.ts) drawing N SDF circles into an offscreen
 *   1024x1024 target, awaited to GPU completion.
 *
 * Numbers are honest wall-clock times including the per-frame CPU work
 * the library actually does (encode, submit, bind-group rebuilds); they
 * are not idealized GPU-only timings.
 *
 * Run with: deno task bench
 * Requires a WebGPU adapter; registers nothing (and says so) without one.
 */

import {
  createSimHarness,
  GPU_SKIP_MESSAGE,
  loadModuleInliningWgsl,
  probeAdapter,
  requestHarnessDevice,
  waitForQueue,
} from "../helpers/gpu.ts";
import { generateCodeTree } from "../fixtures/code_tree.ts";

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
// Setup
// =============================================================================

const adapter = await probeAdapter();

if (!adapter) {
  console.warn(`[bench] ${GPU_SKIP_MESSAGE}`);
} else {
  const device = await requestHarnessDevice(adapter);

  // ---------------------------------------------------------------------------
  // Simulation tick benches (real compute pipeline, code-tree workload)
  // ---------------------------------------------------------------------------

  const SIM_SIZES = [2_500, 10_000, 35_000];

  for (const maxNodes of SIM_SIZES) {
    const tree = generateCodeTree({
      seed: 42,
      maxDepth: 16,
      minChildren: 2,
      maxChildren: 5,
      maxNodes,
      crossEdgeRatio: 0.15,
    });
    const harness = await createSimHarness(device, tree);

    Deno.bench({
      name: `sim tick: code tree ${tree.nodeCount} nodes / ${tree.edgeCount} edges`,
      group: "simulation",
      async fn() {
        await harness.tick(1);
      },
    });
  }

  // ---------------------------------------------------------------------------
  // Node render benches (real instanced render pipeline, offscreen target)
  // ---------------------------------------------------------------------------

  const nodesModule = await loadModuleInliningWgsl<NodesModule>(
    new URL("../../packages/core/src/renderer/pipelines/nodes.ts", import.meta.url),
  );

  const TARGET_SIZE = 1024;
  const target = device.createTexture({
    label: "Bench Render Target",
    size: { width: TARGET_SIZE, height: TARGET_SIZE },
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const targetView = target.createView();

  const RENDER_SIZES = [10_000, 100_000, 500_000];

  for (const nodeCount of RENDER_SIZES) {
    const pipeline = nodesModule.createNodeRenderPipeline({ device }, {
      maxNodes: nodeCount,
      format: "rgba8unorm",
    });

    // Phyllotaxis disc of nodes, matching the library's default seeding.
    const discRadius = Math.sqrt(nodeCount) * 10;
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    const positionsData = new Float32Array(nodeCount * 2);
    const attrsData = new Float32Array(nodeCount * 8); // NODE_ATTR_FLOATS
    for (let i = 0; i < nodeCount; i++) {
      const angle = i * goldenAngle;
      const r = Math.sqrt(i / nodeCount) * discRadius;
      positionsData[i * 2] = r * Math.cos(angle);
      positionsData[i * 2 + 1] = r * Math.sin(angle);
      attrsData[i * 8] = 5; // radius
      attrsData[i * 8 + 1] = 0.3; // color r
      attrsData[i * 8 + 2] = 0.5; // color g
      attrsData[i * 8 + 3] = 0.9; // color b
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

    // ViewportUniforms (node.vert.wgsl): 3 vec4 transform columns,
    // screen_size, scale (px per graph unit), inv_scale, dpr, padding.
    const clipScale = 0.9 / discRadius; // fit the disc into clip space
    const pxPerUnit = clipScale * (TARGET_SIZE / 2);
    const viewportData = new Float32Array(20);
    viewportData[0] = clipScale; // col0.x
    viewportData[5] = clipScale; // col1.y
    viewportData[10] = 1; // col2.z
    viewportData[12] = TARGET_SIZE; // screen_size.x
    viewportData[13] = TARGET_SIZE; // screen_size.y
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

    const viewportBindGroup = nodesModule.createViewportBindGroup(
      device,
      pipeline,
      viewportUniforms,
    );
    const nodeBindGroup = nodesModule.createNodeBindGroup(device, pipeline, positions, nodeAttrs);
    const renderConfigBindGroup = nodesModule.createRenderConfigBindGroup(
      device,
      pipeline,
      renderConfig,
    );

    Deno.bench({
      name: `node render: ${nodeCount.toLocaleString("en-US")} nodes @ 1024x1024`,
      group: "render",
      async fn() {
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
        await waitForQueue(device);
      },
    });
  }
}
