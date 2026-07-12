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

/**
 * Node flag bits, mirroring pipeline.ts (which cannot be imported statically
 * here because of its .wgsl imports — see module doc above).
 */
export const NODE_FLAG_DEAD = 1;
export const NODE_FLAG_PINNED = 2;
/** Collision node_sizes sentinel for dead slots, mirroring pipeline.ts. */
export const DEAD_SLOT_RADIUS = -1;

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
  prevForces: GPUBuffer;
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
    adaptiveSpeedOverride?: boolean,
  ): void;
  recordSimulationStep(
    encoder: GPUCommandEncoder,
    pipeline: SimulationPipeline,
    bindGroups: SimulationBindGroups,
    nodeCount: number,
    edgeCount: number,
  ): void;
  recordSimulationStepWithOptions(
    encoder: GPUCommandEncoder,
    pipeline: SimulationPipeline,
    bindGroups: SimulationBindGroups,
    nodeCount: number,
    edgeCount: number,
    options?: {
      recordRepulsionPass?: ((encoder: GPUCommandEncoder) => void) | undefined;
      skipSprings?: boolean;
    },
  ): void;
  swapSimulationBuffers(buffers: SimulationBuffers): void;
  copyPositionsToReadback(encoder: GPUCommandEncoder, buffers: SimulationBuffers): void;
  readbackPositions(
    buffers: SimulationBuffers,
    targetX: Float32Array,
    targetY: Float32Array,
  ): Promise<void>;
}

/**
 * Structural subset of collision.ts's exports used by the GPU tests.
 * Opaque pipeline/buffer handles are passed straight back into the module.
 */
interface CollisionModule {
  createCollisionPipeline(context: unknown): {
    resolve: GPUComputePipeline;
    resolveTiled: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
  };
  createCollisionBuffers(device: GPUDevice, maxNodes: number): {
    uniforms: GPUBuffer;
    nodeSizes: GPUBuffer;
    fallbackNodeFlags: GPUBuffer;
    maxNodes: number;
  };
  createCollisionBindGroup(
    device: GPUDevice,
    pipeline: unknown,
    collisionBuffers: unknown,
    positions: GPUBuffer,
    nodeFlags?: GPUBuffer,
  ): { bindGroup: GPUBindGroup };
  updateCollisionUniforms(
    device: GPUDevice,
    collisionBuffers: unknown,
    nodeCount: number,
    forceConfig: FullForceConfig,
  ): void;
  uploadNodeSizes(
    device: GPUDevice,
    collisionBuffers: unknown,
    nodeSizes: Float32Array,
  ): void;
  recordCollisionPass(
    encoder: GPUCommandEncoder,
    pipeline: unknown,
    bindGroup: unknown,
    nodeCount: number,
    iterations?: number,
    useTiled?: boolean,
  ): void;
  destroyCollisionBuffers(buffers: unknown): void;
  createGridCollisionPipeline(context: unknown): {
    clearCells: GPUComputePipeline;
    buildLists: GPUComputePipeline;
    resolveGrid: GPUComputePipeline;
    gridLayout: GPUBindGroupLayout;
  };
  createGridCollisionBuffers(device: GPUDevice, maxNodes: number): {
    gridUniforms: GPUBuffer;
    cellHead: GPUBuffer;
    nodeNext: GPUBuffer;
    nodeCell: GPUBuffer;
    fallbackNodeFlags: GPUBuffer;
    maxNodes: number;
    maxCells: number;
  };
  createGridCollisionBindGroups(
    device: GPUDevice,
    pipeline: unknown,
    gridBuffers: unknown,
    nodeSizes: GPUBuffer,
    positions: GPUBuffer,
    nodeFlags?: GPUBuffer,
  ): { grid: GPUBindGroup };
  updateGridCollisionUniforms(
    device: GPUDevice,
    gridBuffers: unknown,
    nodeCount: number,
    forceConfig: FullForceConfig,
    bounds: { minX: number; minY: number; maxX: number; maxY: number },
    maxRadius: number,
  ): void;
  recordGridCollisionPass(
    encoder: GPUCommandEncoder,
    pipeline: unknown,
    bindGroups: unknown,
    gridBuffers: unknown,
    nodeCount: number,
    iterations?: number,
  ): void;
  destroyGridCollisionBuffers(buffers: unknown): void;
}

let pipelineModulePromise: Promise<PipelineModule> | undefined;
let collisionModulePromise: Promise<CollisionModule> | undefined;
const inlinedModuleCache = new Map<string, Promise<unknown>>();

/**
 * Loads an arbitrary module with its .wgsl imports inlined as string
 * constants (see module doc). Used by tests that drive algorithm plugins
 * (e.g. algorithms/t-fdp.ts) directly. Cached per URL.
 */
export function loadModuleInliningWgsl<T>(moduleUrl: URL): Promise<T> {
  let promise = inlinedModuleCache.get(moduleUrl.href);
  if (!promise) {
    promise = importInliningWgsl<T>(moduleUrl);
    inlinedModuleCache.set(moduleUrl.href, promise);
  }
  return promise as Promise<T>;
}

/**
 * Loads pipeline.ts with its .wgsl imports inlined as string constants.
 * Cached: repeated calls return the same module instance.
 */
export function loadPipelineModule(): Promise<PipelineModule> {
  pipelineModulePromise ??= importInliningWgsl<PipelineModule>(
    new URL("../../packages/core/src/simulation/pipeline.ts", import.meta.url),
  );
  return pipelineModulePromise;
}

/**
 * Loads collision.ts with its .wgsl imports inlined as string constants.
 * Cached: repeated calls return the same module instance.
 */
export function loadCollisionModule(): Promise<CollisionModule> {
  collisionModulePromise ??= importInliningWgsl<CollisionModule>(
    new URL("../../packages/core/src/simulation/collision.ts", import.meta.url),
  );
  return collisionModulePromise;
}

// Matches `import NAME from "./x.wgsl";` and the Vite-style
// `import NAME from "./x.wgsl?raw";` used by the layer modules.
const WGSL_IMPORT_PATTERN = /import\s+(\w+)\s+from\s+"([^"]+\.wgsl(?:\?raw)?)";/g;
const RELATIVE_IMPORT_PATTERN = /(from\s+)"(\.\.?\/[^"]+)"/g;

async function importInliningWgsl<T>(moduleUrl: URL): Promise<T> {
  const dataUrl = await dataUrlInliningWgsl(moduleUrl);
  return await import(dataUrl) as T;
}

/**
 * Whether a module (or any module it reaches through relative .ts
 * imports) contains .wgsl imports Deno cannot load. Memoized; the
 * in-progress `false` doubles as a cycle guard.
 */
const wgslSubtreeCache = new Map<string, boolean>();
function subtreeImportsWgsl(moduleUrl: URL): boolean {
  const cached = wgslSubtreeCache.get(moduleUrl.href);
  if (cached !== undefined) return cached;
  wgslSubtreeCache.set(moduleUrl.href, false);

  const source = Deno.readTextFileSync(moduleUrl);
  let needs = new RegExp(WGSL_IMPORT_PATTERN.source).test(source);
  if (!needs) {
    for (const match of source.matchAll(new RegExp(RELATIVE_IMPORT_PATTERN.source, "g"))) {
      const specifier = match[2];
      if (specifier.endsWith(".ts") && subtreeImportsWgsl(new URL(specifier, moduleUrl))) {
        needs = true;
        break;
      }
    }
  }
  wgslSubtreeCache.set(moduleUrl.href, needs);
  return needs;
}

const dataUrlCache = new Map<string, Promise<string>>();
function dataUrlInliningWgsl(moduleUrl: URL): Promise<string> {
  let promise = dataUrlCache.get(moduleUrl.href);
  if (!promise) {
    promise = buildDataUrlInliningWgsl(moduleUrl);
    dataUrlCache.set(moduleUrl.href, promise);
  }
  return promise;
}

async function buildDataUrlInliningWgsl(moduleUrl: URL): Promise<string> {
  const source = await Deno.readTextFile(moduleUrl);

  // Replace .wgsl imports with the shader text inline.
  const inlined = source.replace(
    new RegExp(WGSL_IMPORT_PATTERN.source, "g"),
    (_match, name: string, specifier: string) => {
      const path = specifier.replace(/\?raw$/, "");
      const wgsl = Deno.readTextFileSync(new URL(path, moduleUrl));
      return `const ${name} = ${JSON.stringify(wgsl)};`;
    },
  );

  // Relative imports whose subtree itself imports .wgsl must be inlined
  // recursively (they cannot load as plain files); all others rewrite to
  // absolute file: URLs so they resolve from the data: URL module (same
  // URLs as static imports, so module identity — e.g.
  // DEFAULT_FORCE_CONFIG — is shared).
  const replacements = new Map<string, string>();
  for (const match of inlined.matchAll(new RegExp(RELATIVE_IMPORT_PATTERN.source, "g"))) {
    const specifier = match[2];
    if (replacements.has(specifier)) continue;
    const childUrl = new URL(specifier, moduleUrl);
    replacements.set(
      specifier,
      specifier.endsWith(".ts") && subtreeImportsWgsl(childUrl)
        ? await dataUrlInliningWgsl(childUrl)
        : childUrl.href,
    );
  }

  const rewritten = inlined.replace(
    new RegExp(RELATIVE_IMPORT_PATTERN.source, "g"),
    (_match, prefix: string, specifier: string) => `${prefix}"${replacements.get(specifier)}"`,
  );

  return "data:application/typescript;charset=utf-8," +
    encodeURIComponent(rewritten);
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
  /**
   * Optional per-slot state flags (u32 per node; bit 0 = NODE_FLAG_DEAD,
   * bit 1 = NODE_FLAG_PINNED). Uploaded after buffer initialization.
   */
  flags?: Uint32Array;
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
  adaptiveSpeedOverride?: boolean,
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
  if (graph.flags) {
    device.queue.writeBuffer(buffers.nodeFlags, 0, graph.flags.slice().buffer);
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
          adaptiveSpeedOverride,
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
          buffers.prevForces,
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

/** Opaque handles for algorithm plugin pipelines/buffers/bind groups. */
type AlgorithmPipelinesHandle = { readonly __brand?: "AlgorithmPipelines" };
type AlgorithmBindGroupsHandle = { readonly __brand?: "AlgorithmBindGroups" };

/**
 * Structural subset of AlgorithmRenderContext (algorithms/types.ts) the
 * harness constructs. Declared locally: the algorithms modules import
 * .wgsl and cannot be referenced even with `import type` (see module doc).
 */
interface HarnessAlgorithmContext {
  device: GPUDevice;
  positions: GPUBuffer;
  forces: GPUBuffer;
  nodeCount: number;
  edgeCount: number;
  forceConfig: FullForceConfig;
  bounds?: { minX: number; minY: number; maxX: number; maxY: number } | undefined;
  edgeSources?: GPUBuffer;
  edgeTargets?: GPUBuffer;
  edgeSourcesData?: Uint32Array | undefined;
  edgeTargetsData?: Uint32Array | undefined;
  nodeFlags?: GPUBuffer | undefined;
}

/**
 * Structural subset of the ForceAlgorithm interface (algorithms/types.ts)
 * the harness drives — load implementations via loadModuleInliningWgsl.
 */
export interface HarnessForceAlgorithm {
  readonly info: { readonly id: string };
  readonly handlesGravity: boolean;
  readonly handlesSprings?: boolean;
  readonly prefersAdaptiveSpeed?: boolean;
  createPipelines(context: { device: GPUDevice }): AlgorithmPipelinesHandle;
  createBuffers(device: GPUDevice, maxNodes: number): { destroy(): void };
  createBindGroups(
    device: GPUDevice,
    pipelines: AlgorithmPipelinesHandle,
    context: HarnessAlgorithmContext,
    algorithmBuffers: { destroy(): void },
  ): AlgorithmBindGroupsHandle;
  updateUniforms(
    device: GPUDevice,
    algorithmBuffers: { destroy(): void },
    context: HarnessAlgorithmContext,
  ): void;
  recordRepulsionPass(
    encoder: GPUCommandEncoder,
    pipelines: AlgorithmPipelinesHandle,
    bindGroups: AlgorithmBindGroupsHandle,
    nodeCount: number,
  ): void;
}

/**
 * Creates a simulation harness that runs a pluggable force algorithm the
 * way HeroineGraph does: the algorithm's recordRepulsionPass replaces the
 * default N² repulsion, the shared spring pass is skipped when the
 * algorithm handles its own attraction, integration gravity is suppressed
 * when the algorithm applies its own, and prefersAdaptiveSpeed is
 * forwarded as the adaptive-speed override. Algorithm bind groups are
 * rebuilt after every ping-pong swap, mirroring
 * HeroineGraph.swapAndRebuildBindGroups.
 */
export async function createAlgorithmSimHarness(
  device: GPUDevice,
  algorithm: HarnessForceAlgorithm,
  graph: HarnessGraphData,
  config: Partial<FullForceConfig> = {},
  alphaDecay: number = HARNESS_ALPHA_DECAY,
  options: {
    /**
     * When set, the harness maintains context.bounds for spatial algorithms
     * (Barnes-Hut, density field): computed from the fixture positions up
     * front and refreshed from a GPU readback every N ticks, mirroring
     * graph.ts's periodically-synced CPU bounds.
     */
    boundsSyncInterval?: number;
    /**
     * Called once right after the algorithm's buffers are created, before
     * bind groups. Lets tests upload algorithm-specific data (e.g. CSR
     * edges + BFS depths for Relativity Atlas) the way graph.ts does.
     */
    onAlgorithmBuffers?: (algoBuffers: { destroy(): void }) => void;
  } = {},
): Promise<SimHarness> {
  const mod = await loadPipelineModule();
  const { nodeCount } = graph;
  const edgeCount = graph.edgeSources.length;
  const forceConfig = validateForceConfig(config);

  const computeBounds = (
    xs: Float32Array,
    ys: Float32Array,
  ): NonNullable<HarnessAlgorithmContext["bounds"]> => {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < xs.length; i++) {
      if (!Number.isFinite(xs[i]) || !Number.isFinite(ys[i])) continue;
      if (xs[i] < minX) minX = xs[i];
      if (xs[i] > maxX) maxX = xs[i];
      if (ys[i] < minY) minY = ys[i];
      if (ys[i] > maxY) maxY = ys[i];
    }
    if (minX > maxX) {
      return { minX: -1, minY: -1, maxX: 1, maxY: 1 };
    }
    // Pad: positions drift between syncs; Morton codes clamp at the bounds
    const pad = 0.05 * Math.max(maxX - minX, maxY - minY) + 1;
    return { minX: minX - pad, minY: minY - pad, maxX: maxX + pad, maxY: maxY + pad };
  };

  let currentBounds = options.boundsSyncInterval !== undefined
    ? computeBounds(graph.positionsX, graph.positionsY)
    : undefined;

  // When the algorithm handles gravity itself, suppress integration gravity
  // to avoid double-applying center pull (mirrors graph.ts).
  const effectiveForceConfig = algorithm.handlesGravity
    ? { ...forceConfig, centerStrength: 0 }
    : forceConfig;

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
  if (graph.flags) {
    device.queue.writeBuffer(buffers.nodeFlags, 0, graph.flags.slice().buffer);
  }

  const algoPipelines = algorithm.createPipelines({ device });
  const algoBuffers = algorithm.createBuffers(device, nodeCount);
  options.onAlgorithmBuffers?.(algoBuffers);

  const makeContext = (): HarnessAlgorithmContext => ({
    device,
    positions: buffers.positions,
    forces: buffers.forces,
    nodeCount,
    edgeCount,
    forceConfig,
    bounds: currentBounds,
    edgeSources: buffers.edgeSources,
    edgeTargets: buffers.edgeTargets,
    edgeSourcesData: graph.edgeSources,
    edgeTargetsData: graph.edgeTargets,
    nodeFlags: buffers.nodeFlags,
  });

  let bindGroups = mod.createSimulationBindGroups(device, pipeline, buffers);
  let algoBindGroups = algorithm.createBindGroups(
    device,
    algoPipelines,
    makeContext(),
    algoBuffers,
  );
  let tickCount = 0;

  return {
    nodeCount,
    edgeCount,
    get tickCount() {
      return tickCount;
    },

    async tick(steps: number): Promise<void> {
      for (let s = 0; s < steps; s++) {
        // Refresh bounds from a position readback every syncInterval ticks
        // (initial bounds come from the fixture positions).
        const syncInterval = options.boundsSyncInterval;
        if (
          syncInterval !== undefined && tickCount > 0 &&
          tickCount % syncInterval === 0
        ) {
          const encoder = device.createCommandEncoder();
          mod.copyPositionsToReadback(encoder, buffers);
          device.queue.submit([encoder.finish()]);
          const x = new Float32Array(nodeCount);
          const y = new Float32Array(nodeCount);
          await mod.readbackPositions(buffers, x, y);
          currentBounds = computeBounds(x, y);
        }

        const alpha = Math.pow(1 - alphaDecay, tickCount + 1);
        mod.updateSimulationUniforms(
          device,
          buffers,
          nodeCount,
          edgeCount,
          alpha,
          effectiveForceConfig,
          algorithm.prefersAdaptiveSpeed,
        );
        algorithm.updateUniforms(device, algoBuffers, makeContext());
        const encoder = device.createCommandEncoder();
        mod.recordSimulationStepWithOptions(
          encoder,
          pipeline,
          bindGroups,
          nodeCount,
          edgeCount,
          {
            recordRepulsionPass: (enc) => {
              algorithm.recordRepulsionPass(enc, algoPipelines, algoBindGroups, nodeCount);
            },
            skipSprings: algorithm.handlesSprings ?? false,
          },
        );
        device.queue.submit([encoder.finish()]);
        // Ping-pong swap invalidates the bind groups' buffer references;
        // rebuild both sim and algorithm bind groups as HeroineGraph does.
        mod.swapSimulationBuffers(buffers);
        bindGroups = mod.createSimulationBindGroups(device, pipeline, buffers);
        algoBindGroups = algorithm.createBindGroups(
          device,
          algoPipelines,
          makeContext(),
          algoBuffers,
        );
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
      algoBuffers.destroy();
      for (
        const buffer of [
          buffers.positions,
          buffers.positionsOut,
          buffers.velocities,
          buffers.velocitiesOut,
          buffers.forces,
          buffers.prevForces,
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
