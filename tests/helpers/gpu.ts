/**
 * Headless GPU Simulation Harness
 *
 * Drives the real simulation pipeline from packages/core/src/simulation
 * against a fixture graph on a headless WebGPU device — no canvas, no
 * renderer, no GraphMother instance. This is the rig physics changes are
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
 * Structural view of pipeline.ts's ParitySlot — one parity-indexed group of
 * bind groups. See {@link BindGroupParitySetsHandle}.
 */
export interface ParitySlotHandle<T> {
  readonly label: string;
  readonly built: boolean;
  readonly current: T | null;
  rebuild(build: (view: SimulationBuffers) => T): void;
  clear(): void;
}

/**
 * Structural view of pipeline.ts's BindGroupParitySets — the production
 * ping-pong parity state machine. The harness drives the real class (rather
 * than keeping its own parity counter) so the GPU tests exercise the code
 * GraphMother ships.
 */
export interface BindGroupParitySetsHandle {
  readonly buffers: SimulationBuffers | null;
  readonly parity: 0 | 1;
  slot<T>(label: string): ParitySlotHandle<T>;
  advance(): void;
  clearAll(): void;
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
  buildForBothParities<T>(
    buffers: SimulationBuffers,
    currentParity: 0 | 1,
    build: (view: SimulationBuffers) => T,
  ): readonly [T, T];
  BindGroupParitySets: new (
    buffersRef: () => SimulationBuffers | null,
  ) => BindGroupParitySetsHandle;
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
    apply: GPUComputePipeline;
    bindGroupLayout: GPUBindGroupLayout;
    applyLayout: GPUBindGroupLayout;
  };
  createCollisionBuffers(device: GPUDevice, maxNodes: number): {
    uniforms: GPUBuffer;
    nodeSizes: GPUBuffer;
    displacements: GPUBuffer;
    fallbackNodeFlags: GPUBuffer;
    maxNodes: number;
  };
  createCollisionBindGroup(
    device: GPUDevice,
    pipeline: unknown,
    collisionBuffers: unknown,
    positions: GPUBuffer,
    nodeFlags?: GPUBuffer,
  ): { bindGroup: GPUBindGroup; applyBindGroup: GPUBindGroup };
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
    apply: GPUComputePipeline;
    gridLayout: GPUBindGroupLayout;
    applyLayout: GPUBindGroupLayout;
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
    displacements: GPUBuffer,
    positions: GPUBuffer,
    nodeFlags?: GPUBuffer,
  ): { grid: GPUBindGroup; apply: GPUBindGroup };
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

/** Collision code path driven by {@link runCollision}. */
export type CollisionVariant = "main" | "tiled" | "grid";

/** One collision run over a hand-built position buffer. */
export interface CollisionRunInput {
  /** Node positions, interleaved x,y */
  positions: Float32Array;
  /** Radius per node; DEAD_SLOT_RADIUS marks a dead slot */
  sizes: Float32Array;
  /** Which pipeline to drive */
  variant: CollisionVariant;
  /**
   * Per-node state flags (NODE_FLAG_DEAD / NODE_FLAG_PINNED).
   * Defaults to all-live, all-unpinned.
   */
  flags?: Uint32Array;
  /** Collision resolution iterations (default 1) */
  iterations?: number;
  /** Force config overrides (collisionStrength, collisionRadiusMultiplier) */
  config?: Partial<FullForceConfig>;
  /**
   * Grid variant only: bounds used to size the spatial hash. Defaults to the
   * bounding box of `positions`.
   */
  bounds?: { minX: number; minY: number; maxX: number; maxY: number };
}

/**
 * Drives the real collision pipeline over a hand-built position buffer and
 * returns the resolved positions (interleaved x,y). Every call builds and
 * destroys its own GPU resources, so repeated calls with identical input are
 * genuinely independent runs.
 */
export async function runCollision(
  device: GPUDevice,
  input: CollisionRunInput,
): Promise<Float32Array> {
  const mod = await loadCollisionModule();
  const { positions, sizes, variant } = input;
  const nodeCount = sizes.length;
  const iterations = input.iterations ?? 1;
  const flags = input.flags ?? new Uint32Array(nodeCount);
  const forceConfig = validateForceConfig(input.config ?? {});

  const positionBuffer = device.createBuffer({
    label: "Collision Run Positions",
    size: positions.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const flagsBuffer = device.createBuffer({
    label: "Collision Run Node Flags",
    size: flags.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const readback = device.createBuffer({
    label: "Collision Run Readback",
    size: positions.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  device.queue.writeBuffer(positionBuffer, 0, positions.slice().buffer);
  device.queue.writeBuffer(flagsBuffer, 0, flags.slice().buffer);

  // The grid path reuses the O(n^2) module's nodeSizes and displacements
  // buffers, exactly as graph.ts does.
  const buffers = mod.createCollisionBuffers(device, nodeCount);
  mod.uploadNodeSizes(device, buffers, sizes);

  const encoder = device.createCommandEncoder();
  let disposeVariant: () => void;

  if (variant === "grid") {
    const pipeline = mod.createGridCollisionPipeline({ device });
    const gridBuffers = mod.createGridCollisionBuffers(device, nodeCount);
    const bindGroups = mod.createGridCollisionBindGroups(
      device,
      pipeline,
      gridBuffers,
      buffers.nodeSizes,
      buffers.displacements,
      positionBuffer,
      flagsBuffer,
    );
    let maxRadius = 0;
    for (const size of sizes) {
      if (size > maxRadius) maxRadius = size;
    }
    mod.updateGridCollisionUniforms(
      device,
      gridBuffers,
      nodeCount,
      forceConfig,
      input.bounds ?? boundsOf(positions),
      maxRadius,
    );
    mod.recordGridCollisionPass(
      encoder,
      pipeline,
      bindGroups,
      gridBuffers,
      nodeCount,
      iterations,
    );
    disposeVariant = () => mod.destroyGridCollisionBuffers(gridBuffers);
  } else {
    const pipeline = mod.createCollisionPipeline({ device });
    mod.updateCollisionUniforms(device, buffers, nodeCount, forceConfig);
    const bindGroup = mod.createCollisionBindGroup(
      device,
      pipeline,
      buffers,
      positionBuffer,
      flagsBuffer,
    );
    mod.recordCollisionPass(
      encoder,
      pipeline,
      bindGroup,
      nodeCount,
      iterations,
      variant === "tiled",
    );
    disposeVariant = () => {};
  }

  encoder.copyBufferToBuffer(positionBuffer, 0, readback, 0, positions.byteLength);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  disposeVariant();
  mod.destroyCollisionBuffers(buffers);
  positionBuffer.destroy();
  flagsBuffer.destroy();
  readback.destroy();
  return result;
}

/** Bounding box of an interleaved x,y position array. */
function boundsOf(
  positions: Float32Array,
): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < positions.length; i += 2) {
    if (positions[i] < minX) minX = positions[i];
    if (positions[i] > maxX) maxX = positions[i];
    if (positions[i + 1] < minY) minY = positions[i + 1];
    if (positions[i + 1] > maxY) maxY = positions[i + 1];
  }
  return { minX, minY, maxX, maxY };
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
 * Storage buffers per compute stage the production device asks for
 * (DEFAULT_REQUIRED_LIMITS in packages/core/src/webgpu/context.ts).
 *
 * `adapter.requestDevice()` with no requiredLimits gives the WebGPU *default*
 * of 8 even when the adapter supports far more, which silently invalidates
 * Barnes-Hut's 10-buffer Karras tree layout and Relativity Atlas's 9-buffer
 * sibling pass. An invalid bind group poisons the compute pass, which poisons
 * the command encoder, so the whole tick's command buffer is discarded and the
 * simulation looks inert rather than erroring. Every GPU test therefore takes
 * its device from {@link requestHarnessDevice}.
 */
export const HARNESS_STORAGE_BUFFERS_PER_STAGE = 10;

/**
 * Request a device with production's limits, clamped to what the adapter can
 * actually supply (so a weaker adapter still yields a device rather than a
 * rejected request — the tests that need 10 check `device.limits` themselves).
 */
export function requestHarnessDevice(adapter: GPUAdapter): Promise<GPUDevice> {
  return adapter.requestDevice({
    requiredLimits: {
      maxStorageBuffersPerShaderStage: Math.min(
        HARNESS_STORAGE_BUFFERS_PER_STAGE,
        adapter.limits.maxStorageBuffersPerShaderStage,
      ),
    },
  });
}

/**
 * Wait for all work submitted to the device's queue to complete.
 *
 * Deno 2.6.x never resolves `queue.onSubmittedWorkDone()` (even on an empty
 * queue), so awaiting it deadlocks every multi-tick GPU test. `mapAsync`
 * does resolve, and queue execution is in submission order, so mapping a
 * fence buffer that a just-submitted copy wrote to gives the same
 * "everything before this is done" guarantee.
 */
export async function waitForQueue(device: GPUDevice): Promise<void> {
  const src = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_SRC });
  const fence = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(src, 0, fence, 0, 4);
    device.queue.submit([encoder.finish()]);
    await fence.mapAsync(GPUMapMode.READ);
    fence.unmap();
  } finally {
    src.destroy();
    fence.destroy();
  }
}

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
  /** Which ping-pong orientation the buffers are in right now */
  readonly parity: 0 | 1;
  /** Advance the simulation by `steps` ticks */
  tick(steps: number): Promise<void>;
  /**
   * Replace every simulation buffer with a freshly allocated set carrying the
   * current positions, and rebuild the bind groups — the harness equivalent of
   * GraphMother.reallocateNodeBuffers (which likewise zeroes velocities and
   * forces). Exercises `buildForBothParities` at whatever parity the run has
   * reached, which is the case the `currentParity` parameter exists for.
   */
  reallocate(): Promise<void>;
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
 * How a harness obtains the bind groups for each tick.
 *
 * - `"prebuilt-parity"` (default) mirrors production: both ping-pong parities
 *   are built once up front and the tick loop only flips an index.
 * - `"rebuild-each-tick"` recreates every bind group after each swap. This is
 *   the pre-optimization behaviour, kept solely as the reference
 *   implementation the parity-equivalence test compares against; nothing else
 *   should select it.
 */
export type HarnessBindGroupMode = "prebuilt-parity" | "rebuild-each-tick";

/**
 * Creates a simulation harness on the given device: builds the real
 * compute pipelines, uploads the fixture graph, and steps the simulation
 * exactly the way GraphMother does (uniforms -> record -> submit ->
 * ping-pong swap -> parity flip).
 */
export async function createSimHarness(
  device: GPUDevice,
  graph: HarnessGraphData,
  config: Partial<FullForceConfig> = {},
  alphaDecay: number = HARNESS_ALPHA_DECAY,
  adaptiveSpeedOverride?: boolean,
  bindGroupMode: HarnessBindGroupMode = "prebuilt-parity",
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

  /**
   * Allocates a complete simulation buffer set holding `positionsX/Y`.
   * Order matters: copyPositionsToSimulation zeroes nodeFlags, so the
   * fixture's flags are uploaded after it.
   */
  const allocate = (positionsX: Float32Array, positionsY: Float32Array): SimulationBuffers => {
    const fresh = mod.createSimulationBuffers(device, nodeCount, edgeCount);
    mod.copyPositionsToSimulation(device, fresh, positionsX, positionsY);
    mod.copyEdgesToSimulation(device, fresh, graph.edgeSources, graph.edgeTargets);
    if (graph.depths) {
      device.queue.writeBuffer(fresh.nodeDepth, 0, graph.depths.slice().buffer);
    }
    if (graph.flags) {
      device.queue.writeBuffer(fresh.nodeFlags, 0, graph.flags.slice().buffer);
    }
    return fresh;
  };

  let buffers = allocate(graph.positionsX, graph.positionsY);

  // Production path: the real parity state machine from pipeline.ts. Both
  // ping-pong orientations are built once and the tick loop only flips the
  // index (BindGroupParitySets.advance), exactly as GraphMother does.
  const paritySets = new mod.BindGroupParitySets(() => buffers);
  const simSlot = paritySets.slot<SimulationBindGroups>("simulation");
  const rebuildSimSlot = (): void =>
    simSlot.rebuild((view) => mod.createSimulationBindGroups(device, pipeline, view));
  rebuildSimSlot();

  // Reference path (bindGroupMode === "rebuild-each-tick"): rebuilt from the
  // live buffers after every swap, deliberately bypassing the parity indexing
  // it exists to check.
  let rebuiltBindGroups = mod.createSimulationBindGroups(device, pipeline, buffers);
  const currentBindGroups = (): SimulationBindGroups =>
    bindGroupMode === "prebuilt-parity" ? simSlot.current! : rebuiltBindGroups;
  let tickCount = 0;

  const readPositions = async (): Promise<{ x: Float32Array; y: Float32Array }> => {
    const encoder = device.createCommandEncoder();
    mod.copyPositionsToReadback(encoder, buffers);
    device.queue.submit([encoder.finish()]);
    const x = new Float32Array(nodeCount);
    const y = new Float32Array(nodeCount);
    await mod.readbackPositions(buffers, x, y);
    return { x, y };
  };

  return {
    nodeCount,
    edgeCount,
    get tickCount() {
      return tickCount;
    },
    get parity() {
      return paritySets.parity;
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
        mod.recordSimulationStep(encoder, pipeline, currentBindGroups(), nodeCount, edgeCount);
        device.queue.submit([encoder.finish()]);
        paritySets.advance();
        if (bindGroupMode === "rebuild-each-tick") {
          rebuiltBindGroups = mod.createSimulationBindGroups(device, pipeline, buffers);
        }
        tickCount++;
      }
      await waitForQueue(device);
    },

    async reallocate(): Promise<void> {
      const { x, y } = await readPositions();
      const old = buffers;
      buffers = allocate(x, y);
      destroySimulationBufferSet(old);
      rebuildSimSlot();
      if (bindGroupMode === "rebuild-each-tick") {
        rebuiltBindGroups = mod.createSimulationBindGroups(device, pipeline, buffers);
      }
    },

    readPositions,

    dispose(): void {
      destroySimulationBufferSet(buffers);
    },
  };
}

/** Releases every GPU buffer in a harness-owned SimulationBuffers. */
function destroySimulationBufferSet(buffers: SimulationBuffers): void {
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
 * way GraphMother does: the algorithm's recordRepulsionPass replaces the
 * default N² repulsion, the shared spring pass is skipped when the
 * algorithm handles its own attraction, integration gravity is suppressed
 * when the algorithm applies its own, and prefersAdaptiveSpeed is
 * forwarded as the adaptive-speed override. Simulation and algorithm bind
 * groups are built once per ping-pong parity and selected by index, mirroring
 * GraphMother.advanceFrameParity.
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
    /** See {@link HarnessBindGroupMode}; defaults to the production path. */
    bindGroupMode?: HarnessBindGroupMode;
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

  /**
   * Allocates a complete simulation buffer set holding `positionsX/Y`.
   * Order matters: copyPositionsToSimulation zeroes nodeFlags, so the
   * fixture's flags are uploaded after it.
   */
  const allocate = (positionsX: Float32Array, positionsY: Float32Array): SimulationBuffers => {
    const fresh = mod.createSimulationBuffers(device, nodeCount, edgeCount);
    mod.copyPositionsToSimulation(device, fresh, positionsX, positionsY);
    mod.copyEdgesToSimulation(device, fresh, graph.edgeSources, graph.edgeTargets);
    if (graph.depths) {
      device.queue.writeBuffer(fresh.nodeDepth, 0, graph.depths.slice().buffer);
    }
    if (graph.flags) {
      device.queue.writeBuffer(fresh.nodeFlags, 0, graph.flags.slice().buffer);
    }
    return fresh;
  };

  let buffers = allocate(graph.positionsX, graph.positionsY);

  const algoPipelines = algorithm.createPipelines({ device });
  const algoBuffers = algorithm.createBuffers(device, nodeCount);
  options.onAlgorithmBuffers?.(algoBuffers);

  const makeContext = (view: SimulationBuffers): HarnessAlgorithmContext => ({
    device,
    positions: view.positions,
    forces: view.forces,
    nodeCount,
    edgeCount,
    forceConfig,
    bounds: currentBounds,
    edgeSources: view.edgeSources,
    edgeTargets: view.edgeTargets,
    edgeSourcesData: graph.edgeSources,
    edgeTargetsData: graph.edgeTargets,
    nodeFlags: view.nodeFlags,
  });

  const bindGroupMode = options.bindGroupMode ?? "prebuilt-parity";
  // Production path: the real parity state machine from pipeline.ts, holding
  // both the simulation passes' bind groups and the algorithm's own; the tick
  // loop only flips the index (BindGroupParitySets.advance).
  const paritySets = new mod.BindGroupParitySets(() => buffers);
  const simSlot = paritySets.slot<SimulationBindGroups>("simulation");
  const algoSlot = paritySets.slot<AlgorithmBindGroupsHandle>(`algorithm:${algorithm.info.id}`);
  const rebuildSlots = (): void => {
    simSlot.rebuild((view) => mod.createSimulationBindGroups(device, pipeline, view));
    algoSlot.rebuild((view) =>
      algorithm.createBindGroups(device, algoPipelines, makeContext(view), algoBuffers)
    );
  };
  rebuildSlots();

  // Reference path (see HarnessBindGroupMode): rebuilt from the live buffers
  // after every swap, deliberately bypassing the parity indexing it exists to
  // check.
  let rebuiltBindGroups = mod.createSimulationBindGroups(device, pipeline, buffers);
  let rebuiltAlgoBindGroups = algorithm.createBindGroups(
    device,
    algoPipelines,
    makeContext(buffers),
    algoBuffers,
  );
  const currentBindGroups = (): SimulationBindGroups =>
    bindGroupMode === "prebuilt-parity" ? simSlot.current! : rebuiltBindGroups;
  const currentAlgoBindGroups = (): AlgorithmBindGroupsHandle =>
    bindGroupMode === "prebuilt-parity" ? algoSlot.current! : rebuiltAlgoBindGroups;
  let tickCount = 0;

  const readPositions = async (): Promise<{ x: Float32Array; y: Float32Array }> => {
    const encoder = device.createCommandEncoder();
    mod.copyPositionsToReadback(encoder, buffers);
    device.queue.submit([encoder.finish()]);
    const x = new Float32Array(nodeCount);
    const y = new Float32Array(nodeCount);
    await mod.readbackPositions(buffers, x, y);
    return { x, y };
  };

  return {
    nodeCount,
    edgeCount,
    get tickCount() {
      return tickCount;
    },
    get parity() {
      return paritySets.parity;
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
        // buffers is always in the current parity's orientation, so this
        // context matches the bind groups selected below.
        algorithm.updateUniforms(device, algoBuffers, makeContext(buffers));
        const encoder = device.createCommandEncoder();
        mod.recordSimulationStepWithOptions(
          encoder,
          pipeline,
          currentBindGroups(),
          nodeCount,
          edgeCount,
          {
            recordRepulsionPass: (enc) => {
              algorithm.recordRepulsionPass(
                enc,
                algoPipelines,
                currentAlgoBindGroups(),
                nodeCount,
              );
            },
            skipSprings: algorithm.handlesSprings ?? false,
          },
        );
        device.queue.submit([encoder.finish()]);
        paritySets.advance();
        if (bindGroupMode === "rebuild-each-tick") {
          rebuiltBindGroups = mod.createSimulationBindGroups(device, pipeline, buffers);
          rebuiltAlgoBindGroups = algorithm.createBindGroups(
            device,
            algoPipelines,
            makeContext(buffers),
            algoBuffers,
          );
        }
        tickCount++;
      }
      await waitForQueue(device);
    },

    async reallocate(): Promise<void> {
      const { x, y } = await readPositions();
      const old = buffers;
      buffers = allocate(x, y);
      destroySimulationBufferSet(old);
      // Mirrors graph.ts: the algorithm's own buffers survive, but every bind
      // group referencing a replaced simulation buffer is rebuilt for both
      // parities before the next tick.
      rebuildSlots();
      if (bindGroupMode === "rebuild-each-tick") {
        rebuiltBindGroups = mod.createSimulationBindGroups(device, pipeline, buffers);
        rebuiltAlgoBindGroups = algorithm.createBindGroups(
          device,
          algoPipelines,
          makeContext(buffers),
          algoBuffers,
        );
      }
    },

    readPositions,

    dispose(): void {
      algoBuffers.destroy();
      destroySimulationBufferSet(buffers);
    },
  };
}
