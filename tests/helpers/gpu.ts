/**
 * Headless GPU Simulation Harness
 *
 * Drives the real simulation pipeline from packages/core/src/simulation
 * against a fixture graph on a headless WebGPU device — no canvas, no
 * renderer, no HeroineGraph instance. This is the rig physics changes are
 * verified (and later tuned) against.
 *
 * Deno cannot load the bare `.wgsl` imports the core package uses (they
 * are resolved by esbuild in browser builds), so `loadPipelineModule`
 * inlines the shader sources as string constants and imports the
 * transformed module via a data: URL. The module cannot be referenced
 * even with `import type` (deno test's loader follows type-only imports
 * into the .wgsl-importing files), so the signatures used here are
 * declared locally as structural types.
 *
 * @module
 */

import {
  type FullForceConfig,
  validateForceConfig,
} from "../../packages/core/src/simulation/config.ts";

/** Opaque handle: SimulationPipeline from pipeline.ts */
type SimulationPipeline = { readonly __brand?: "SimulationPipeline" };
/** Opaque handle: SimulationBindGroups from pipeline.ts */
type SimulationBindGroups = { readonly __brand?: "SimulationBindGroups" };

/**
 * Structural subset of pipeline.ts's SimulationBuffers: the GPU buffers
 * the harness uploads to or destroys directly.
 */
interface SimulationBuffers {
  positions: GPUBuffer;
  positionsOut: GPUBuffer;
  velocities: GPUBuffer;
  velocitiesOut: GPUBuffer;
  forces: GPUBuffer;
  edgeSources: GPUBuffer;
  edgeTargets: GPUBuffer;
  clearUniforms: GPUBuffer;
  repulsionUniforms: GPUBuffer;
  springUniforms: GPUBuffer;
  integrationUniforms: GPUBuffer;
  nodeFlags: GPUBuffer;
  nodeDepth: GPUBuffer;
  readback: GPUBuffer;
  nodeCount: number;
}

/**
 * Signatures of the pipeline.ts exports the harness calls. Kept in sync
 * by hand; a drift shows up immediately as a runtime failure in the GPU
 * tests.
 */
interface PipelineModule {
  createSimulationPipeline(
    context: unknown,
    config?: { maxNodes?: number; maxEdges?: number },
  ): SimulationPipeline;
  createSimulationBuffers(
    device: GPUDevice,
    nodeCount: number,
    edgeCount: number,
  ): SimulationBuffers;
  createSimulationBindGroups(
    device: GPUDevice,
    pipeline: SimulationPipeline,
    buffers: SimulationBuffers,
  ): SimulationBindGroups;
  copyPositionsToSimulation(
    device: GPUDevice,
    buffers: SimulationBuffers,
    positionsX: Float32Array,
    positionsY: Float32Array,
  ): void;
  copyEdgesToSimulation(
    device: GPUDevice,
    buffers: SimulationBuffers,
    edgeSources: Uint32Array,
    edgeTargets: Uint32Array,
  ): void;
  updateSimulationUniforms(
    device: GPUDevice,
    buffers: SimulationBuffers,
    nodeCount: number,
    edgeCount: number,
    alpha: number,
    forceConfig?: FullForceConfig,
  ): void;
  recordSimulationStep(
    encoder: GPUCommandEncoder,
    pipeline: SimulationPipeline,
    bindGroups: SimulationBindGroups,
    nodeCount: number,
    edgeCount: number,
  ): void;
  swapSimulationBuffers(buffers: SimulationBuffers): void;
  copyPositionsToReadback(encoder: GPUCommandEncoder, buffers: SimulationBuffers): void;
  readbackPositions(
    buffers: SimulationBuffers,
    targetX: Float32Array,
    targetY: Float32Array,
  ): Promise<void>;
}

let pipelineModulePromise: Promise<PipelineModule> | undefined;

/**
 * Loads pipeline.ts with its .wgsl imports inlined as string constants.
 * Cached: repeated calls return the same module instance.
 */
export function loadPipelineModule(): Promise<PipelineModule> {
  pipelineModulePromise ??= importInliningWgsl(
    new URL("../../packages/core/src/simulation/pipeline.ts", import.meta.url),
  );
  return pipelineModulePromise;
}

async function importInliningWgsl(moduleUrl: URL): Promise<PipelineModule> {
  const source = await Deno.readTextFile(moduleUrl);

  // Replace `import NAME from "./x.wgsl";` with the shader text inline.
  const inlined = source.replace(
    /import\s+(\w+)\s+from\s+"([^"]+\.wgsl)";/g,
    (_match, name: string, specifier: string) => {
      const wgsl = Deno.readTextFileSync(new URL(specifier, moduleUrl));
      return `const ${name} = ${JSON.stringify(wgsl)};`;
    },
  );

  // Rewrite remaining relative imports to absolute file: URLs so they
  // resolve from the data: URL module (same URLs as static imports, so
  // module identity — e.g. DEFAULT_FORCE_CONFIG — is shared).
  const rewritten = inlined.replace(
    /(from\s+)"(\.\.?\/[^"]+)"/g,
    (_match, prefix: string, specifier: string) =>
      `${prefix}"${new URL(specifier, moduleUrl).href}"`,
  );

  const dataUrl = "data:application/typescript;charset=utf-8," +
    encodeURIComponent(rewritten);
  return await import(dataUrl) as PipelineModule;
}

/**
 * Probe for a WebGPU adapter. Returns null (never throws) when WebGPU is
 * unavailable — e.g. missing --unstable-webgpu or no GPU on the machine —
 * so tests can skip gracefully.
 */
export async function probeAdapter(): Promise<GPUAdapter | null> {
  if (typeof navigator === "undefined" || !navigator.gpu) return null;
  try {
    return await navigator.gpu.requestAdapter();
  } catch {
    return null;
  }
}

/** Human-readable reason to print when GPU tests are skipped. */
export const GPU_SKIP_MESSAGE = "WebGPU adapter unavailable — GPU integration tests skipped. " +
  "Run via `deno task test` (passes --unstable-webgpu) on a machine with a GPU.";

/**
 * Graph data consumed by the harness (slot-indexed typed arrays, as
 * produced by tests/fixtures/code_tree.ts).
 */
export interface HarnessGraphData {
  nodeCount: number;
  positionsX: Float32Array;
  positionsY: Float32Array;
  edgeSources: Uint32Array;
  edgeTargets: Uint32Array;
  /** Optional BFS depths for hierarchical settling (f32 per node) */
  depths?: Float32Array;
}

/**
 * A running headless simulation over one fixture graph.
 */
export interface SimHarness {
  readonly nodeCount: number;
  readonly edgeCount: number;
  /** Ticks advanced so far */
  readonly tickCount: number;
  /** Advance the simulation by `steps` ticks */
  tick(steps: number): Promise<void>;
  /** Read current node positions back to the CPU */
  readPositions(): Promise<{ x: Float32Array; y: Float32Array }>;
  /** Release GPU buffers owned by this harness (device is caller-owned) */
  dispose(): void;
}

/**
 * d3-convention cooling used by the harness alpha schedule
 * (~300 ticks to rest). Deliberately independent of the library's
 * controller defaults so harness assertions do not shift when those
 * defaults are retuned.
 */
export const HARNESS_ALPHA_DECAY = 0.0228;

/**
 * Creates a simulation harness on the given device: builds the real
 * compute pipelines, uploads the fixture graph, and steps the simulation
 * exactly the way HeroineGraph does (uniforms -> record -> submit ->
 * ping-pong swap -> bind group rebuild).
 */
export async function createSimHarness(
  device: GPUDevice,
  graph: HarnessGraphData,
  config: Partial<FullForceConfig> = {},
  alphaDecay: number = HARNESS_ALPHA_DECAY,
): Promise<SimHarness> {
  const mod = await loadPipelineModule();
  const { nodeCount } = graph;
  const edgeCount = graph.edgeSources.length;
  const forceConfig = validateForceConfig(config);

  // createSimulationPipeline only touches context.device.
  const pipeline = mod.createSimulationPipeline({ device }, {
    maxNodes: nodeCount,
    maxEdges: edgeCount,
  });

  const buffers = mod.createSimulationBuffers(device, nodeCount, edgeCount);
  mod.copyPositionsToSimulation(device, buffers, graph.positionsX, graph.positionsY);
  mod.copyEdgesToSimulation(device, buffers, graph.edgeSources, graph.edgeTargets);
  if (graph.depths) {
    device.queue.writeBuffer(buffers.nodeDepth, 0, graph.depths.slice().buffer);
  }

  let bindGroups = mod.createSimulationBindGroups(device, pipeline, buffers);
  let tickCount = 0;

  return {
    nodeCount,
    edgeCount,
    get tickCount() {
      return tickCount;
    },

    async tick(steps: number): Promise<void> {
      for (let s = 0; s < steps; s++) {
        const alpha = Math.pow(1 - alphaDecay, tickCount + 1);
        mod.updateSimulationUniforms(
          device,
          buffers,
          nodeCount,
          edgeCount,
          alpha,
          forceConfig,
        );
        const encoder = device.createCommandEncoder();
        mod.recordSimulationStep(encoder, pipeline, bindGroups, nodeCount, edgeCount);
        device.queue.submit([encoder.finish()]);
        // Ping-pong swap invalidates the bind groups' buffer references;
        // rebuild them exactly as HeroineGraph.tickSimulation does.
        mod.swapSimulationBuffers(buffers);
        bindGroups = mod.createSimulationBindGroups(device, pipeline, buffers);
        tickCount++;
      }
      await device.queue.onSubmittedWorkDone();
    },

    async readPositions(): Promise<{ x: Float32Array; y: Float32Array }> {
      const encoder = device.createCommandEncoder();
      mod.copyPositionsToReadback(encoder, buffers);
      device.queue.submit([encoder.finish()]);
      const x = new Float32Array(nodeCount);
      const y = new Float32Array(nodeCount);
      await mod.readbackPositions(buffers, x, y);
      return { x, y };
    },

    dispose(): void {
      for (
        const buffer of [
          buffers.positions,
          buffers.positionsOut,
          buffers.velocities,
          buffers.velocitiesOut,
          buffers.forces,
          buffers.edgeSources,
          buffers.edgeTargets,
          buffers.clearUniforms,
          buffers.repulsionUniforms,
          buffers.springUniforms,
          buffers.integrationUniforms,
          buffers.nodeFlags,
          buffers.nodeDepth,
          buffers.readback,
        ]
      ) {
        buffer.destroy();
      }
    },
  };
}
