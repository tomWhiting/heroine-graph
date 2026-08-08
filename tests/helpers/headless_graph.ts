/**
 * Headless GraphMother Harness
 *
 * Constructs the real {@link GraphMother} on a headless WebGPU device, so a
 * test can drive the whole instance — load, mutate, fold, card — rather than
 * one pipeline at a time. `tests/helpers/gpu.ts` is the harness for the
 * opposite job: it deliberately bypasses GraphMother to test a compute pass in
 * isolation. Reach for this one when the thing under test is an *interaction*
 * between subsystems that only exists on the assembled object — the LOD
 * controller's slot-indexed state surviving a batch removal being the case it
 * was built for.
 *
 * Three substitutions make that possible, and each is narrow enough to leave
 * the code under test untouched:
 *
 *  - **The canvas** is a plain object exposing the six members GraphMother
 *    reads (dimensions, listeners, bounding rect, style, parent). GraphMother
 *    never calls a canvas *method* that draws.
 *  - **The canvas GPU context** is a stub whose `getCurrentTexture` throws.
 *    Nothing here presents a frame — `renderFrame` is the one caller, and a
 *    headless test has no surface to present to. A test that reaches it has
 *    found a real bug, so the throw is deliberate rather than a silent no-op.
 *  - **The device** comes from `requestHarnessDevice`, so it carries the same
 *    8-storage-buffer limit production asks for.
 *
 * The `.wgsl` imports are handled the same way `gpu.ts` handles them: through
 * `loadModuleInliningWgsl`, which inlines shader sources and imports the
 * rewritten module from a `data:` URL. That is why `GraphMother` is typed
 * structurally here instead of imported — deno test's loader follows even a
 * type-only import into the `.wgsl`-importing files.
 *
 * @module
 */

import { loadModuleInliningWgsl, probeAdapter, requestHarnessDevice } from "./gpu.ts";
import type {
  EventMap,
  GraphInput,
  GraphTypedInput,
  NodeId,
  ViewportState,
} from "../../packages/core/src/types.ts";
import type { CardProvider, DomOverlayConfig } from "../../packages/core/src/overlay/types.ts";
import type { LodConfig } from "../../packages/core/src/lod/config.ts";

/**
 * Structural view of the GraphMother surface the harness exposes.
 *
 * Deliberately a subset: what is listed here is what headless tests have had a
 * use for, and adding a member is the cheap part. Everything is the real
 * method on the real instance — nothing is reimplemented.
 */
export interface HeadlessGraph {
  load(data: GraphInput | GraphTypedInput): Promise<void>;
  removeNodesBatch(
    ids: (NodeId | string)[],
  ): Promise<{ removedCount: number; nodeSlotRemap: Map<number, number> }>;
  getExternalId(node: NodeId): string | number | undefined;
  getNodeId(externalId: string | number): NodeId | undefined;
  readonly nodeCount: number;
  setLodConfig(config: Partial<LodConfig>): void;
  /**
   * Also the only public act that forces an LOD evaluation without a frame,
   * which is what makes the controller drivable here: the harness never
   * presents, so `renderFrame`'s tick never runs.
   */
  setLodFocus(nodes: Iterable<NodeId>): void;
  isCollapsed(node: NodeId): boolean;
  expandNode(node: NodeId): void;
  getVisibleNodes(): NodeId[];
  setDomOverlay(config: Partial<DomOverlayConfig>): void;
  setCardProvider(provider: CardProvider<unknown> | null): void;
  syncDomCards(entries: readonly { node: NodeId; priority: number }[]): void;
  setScale(scale: number): void;
  getViewport(): ViewportState;
  on<K extends keyof EventMap>(type: K, handler: (event: EventMap[K]) => void): void;
  off<K extends keyof EventMap>(type: K, handler: (event: EventMap[K]) => void): void;
  dispose(): void;
}

/** A headless GraphMother and the resources it must outlive. */
export interface HeadlessHarness {
  readonly graph: HeadlessGraph;
  readonly device: GPUDevice;
  /** The stub canvas, so a test can resize it or read what was written. */
  readonly canvas: HeadlessCanvas;
  /** Releases the graph. The device is caller-owned, as in `gpu.ts`. */
  dispose(): void;
}

/**
 * The canvas surface GraphMother actually touches.
 *
 * Written out as an interface rather than an `any` so that a member added to
 * the real usage shows up here as a type error the next time this file is
 * checked, instead of as an undefined at runtime.
 */
export interface HeadlessCanvas {
  width: number;
  height: number;
  clientWidth: number;
  clientHeight: number;
  style: Record<string, string>;
  parentElement: HTMLElement | null;
  addEventListener(): void;
  removeEventListener(): void;
  getBoundingClientRect(): { left: number; top: number; width: number; height: number };
}

/** Frame callbacks the shimmed `requestAnimationFrame` has accepted. */
const pendingFrames = new Map<number, FrameRequestCallback>();
let nextFrameHandle = 1;

/**
 * Install `requestAnimationFrame`/`cancelAnimationFrame` if the runtime has
 * none, which Deno does not.
 *
 * The shim accepts callbacks and never fires them on its own. That is not a
 * convenience — it is the accurate model of this environment: nothing is being
 * presented, exactly as in a browser tab that is never painted, and the render
 * loop's own pause/resume path is built for precisely that state. Firing them
 * would drive `renderFrame` into `getCurrentTexture`, which has no surface to
 * return. A test that wants frames asks for them through {@link pumpFrames}.
 *
 * Idempotent, and never replaces a real implementation.
 */
function installFrameShim(): void {
  const target = globalThis as {
    requestAnimationFrame?: (cb: FrameRequestCallback) => number;
    cancelAnimationFrame?: (handle: number) => void;
  };
  if (typeof target.requestAnimationFrame === "function") return;
  target.requestAnimationFrame = (callback: FrameRequestCallback): number => {
    const handle = nextFrameHandle++;
    pendingFrames.set(handle, callback);
    return handle;
  };
  target.cancelAnimationFrame = (handle: number): void => {
    pendingFrames.delete(handle);
  };
}

/**
 * Run every frame callback the render loop is currently waiting on, once.
 *
 * Only useful to a test that has arranged for the frame not to reach the
 * renderer — everything the LOD and overlay paths need is reachable from
 * `syncDomCards` and the mutation methods without a frame at all.
 *
 * @param timestampMs - Value passed to the callbacks as the frame time
 * @returns How many callbacks ran
 */
export function pumpFrames(timestampMs: number): number {
  const due = [...pendingFrames.entries()];
  pendingFrames.clear();
  for (const [, callback] of due) callback(timestampMs);
  return due.length;
}

/** Build the stub canvas at a given CSS size. */
export function headlessCanvas(width = 800, height = 600): HeadlessCanvas {
  return {
    width,
    height,
    clientWidth: width,
    clientHeight: height,
    style: {},
    parentElement: null,
    addEventListener() {},
    removeEventListener() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width, height }),
  };
}

let graphModulePromise:
  | Promise<{ GraphMother: new (config: unknown) => HeadlessGraph }>
  | undefined;

/**
 * Loads api/graph.ts with its (transitive) .wgsl imports inlined. Cached, so
 * repeated harnesses in one test file share one module instance.
 */
function loadGraphModule(): Promise<{ GraphMother: new (config: unknown) => HeadlessGraph }> {
  graphModulePromise ??= loadModuleInliningWgsl(
    new URL("../../packages/core/src/api/graph.ts", import.meta.url),
  );
  return graphModulePromise;
}

export interface HeadlessOptions {
  readonly width?: number;
  readonly height?: number;
  /**
   * Element the DOM overlay mounts into. Supply one (linkedom's document body
   * serves) to exercise cards; omit it and the overlay is simply never enabled.
   */
  readonly overlayHost?: HTMLElement;
}

/**
 * Create a headless GraphMother on `device`.
 *
 * The device is caller-owned so a test file can share one across cases, which
 * is what `probeAdapter` + {@link requestHarnessDevice} are for.
 */
export async function createHeadlessGraph(
  device: GPUDevice,
  options: HeadlessOptions = {},
): Promise<HeadlessHarness> {
  installFrameShim();
  const { GraphMother } = await loadGraphModule();
  const canvas = headlessCanvas(options.width, options.height);

  const gpuContext = {
    // `adapter` is only read for diagnostics; the device is the load-bearing
    // half and is the one the caller supplied.
    adapter: null,
    device,
    context: {
      configure() {},
      unconfigure() {},
      getCurrentTexture(): never {
        throw new Error(
          "headless harness has no presentable surface — a test reached renderFrame",
        );
      },
    },
    format: "bgra8unorm" as GPUTextureFormat,
    canvas,
    limits: device.limits,
  };

  const graph = new GraphMother({
    gpuContext,
    wasmEngine: null,
    canvas,
    debug: false,
  });

  return {
    graph,
    device,
    canvas,
    dispose(): void {
      graph.dispose();
    },
  };
}

/**
 * Skip guard for headless-GraphMother tests, mirroring the GPU tests'.
 * Returns null (never throws) when no adapter is available.
 */
export async function headlessDevice(): Promise<GPUDevice | null> {
  const adapter = await probeAdapter();
  if (!adapter) return null;
  return await requestHarnessDevice(adapter);
}
