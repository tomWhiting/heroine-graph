/**
 * GPU tests for Relativity Atlas pipeline wiring.
 *
 * Regression: the shared fa2_attraction.comp.wgsl gained a node_flags
 * binding at @group(0) @binding(6) for dead-slot masking, but RA's
 * attraction bind group layout was not updated, so createComputePipeline
 * failed validation ("binding 6 is not available in the pipeline layout")
 * and Relativity Atlas was entirely broken at createPipelines. WebGPU
 * reports this asynchronously (an uncaptured error, not a JS throw), so
 * these tests wrap the full pipeline + bind group creation and dispatch
 * in an explicit validation error scope.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  createAlgorithmSimHarness,
  GPU_SKIP_MESSAGE,
  type HarnessForceAlgorithm,
  loadModuleInliningWgsl,
  probeAdapter,
  requestHarnessDevice,
} from "../helpers/gpu.ts";
import { countNonFinite } from "../helpers/invariants.ts";
import { generateCodeTree } from "../fixtures/code_tree.ts";

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
      // RA's sibling pass binds 9 storage buffers; requestHarnessDevice asks
      // for production's 10 rather than the base WebGPU limit of 8.
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
  "GPU RelativityAtlas: pipelines and bind groups pass WebGPU validation and ticks produce finite positions",
  async (device) => {
    const mod = await loadModuleInliningWgsl<
      { createRelativityAtlasAlgorithm(): HarnessForceAlgorithm }
    >(
      new URL(
        "../../packages/core/src/simulation/algorithms/relativity-atlas.ts",
        import.meta.url,
      ),
    );

    const graph = generateCodeTree({ seed: 3, maxNodes: 200 });

    // Pipeline/bind-group validation errors surface asynchronously, so
    // capture them explicitly — without this scope a broken layout only
    // logs an uncaptured-error warning and the test would pass.
    device.pushErrorScope("validation");

    const harness = await createAlgorithmSimHarness(
      device,
      mod.createRelativityAtlasAlgorithm(),
      graph,
    );
    try {
      // Covers every RA phase recorded by recordRepulsionPass (mass init on
      // the first tick, then the steady-state passes including the shared
      // fa2_attraction dispatch that requires the node_flags binding).
      await harness.tick(5);

      const validationError = await device.popErrorScope();
      assert(
        validationError === null,
        `RelativityAtlas pipeline failed WebGPU validation: ${validationError?.message}`,
      );

      const { x, y } = await harness.readPositions();
      assertEquals(countNonFinite(x, y), 0);
    } finally {
      harness.dispose();
    }
  },
);
