/**
 * HeroineGraph Factory Function
 *
 * Creates and initializes a HeroineGraph instance with all required
 * subsystems (WebGPU, WASM, rendering, simulation).
 *
 * @module
 */

import { HeroineGraphError, ErrorCode, wrapAsync } from "../errors.ts";
import { checkWebGPU, hasWebGPU } from "../webgpu/check.ts";
import { createGPUContext, type GPUContext, type GPUContextOptions } from "../webgpu/context.ts";
import { loadWasmModule, createWasmEngine, isWasmLoaded } from "../wasm/loader.ts";
import { HeroineGraph, type HeroineGraphConfig } from "./graph.ts";
import type { GraphConfig } from "../types.ts";

/**
 * Options for creating a HeroineGraph instance
 */
export interface CreateHeroineGraphOptions {
  /** Canvas element or selector to render into */
  canvas: HTMLCanvasElement | string;

  /** WASM module URL (optional, uses default if not provided) */
  wasmUrl?: string;

  /** GPU context options */
  gpu?: Partial<GPUContextOptions>;

  /** Graph configuration */
  config?: Partial<GraphConfig>;

  /** Enable debug mode */
  debug?: boolean;
}

/**
 * Result of initialization
 */
export interface InitResult {
  /** The created HeroineGraph instance */
  graph: HeroineGraph;
  /** GPU context */
  gpuContext: GPUContext;
  /** WebGPU capabilities info */
  capabilities: {
    maxTextureSize: number;
    maxComputeWorkgroupSize: number;
    maxBufferSize: number;
  };
}

/**
 * Creates a HeroineGraph instance
 *
 * This is the main entry point for using the library. It handles:
 * - WebGPU availability check and initialization
 * - WASM module loading
 * - GPU context creation
 * - Render pipeline setup
 *
 * @param options - Creation options
 * @returns Promise resolving to HeroineGraph instance
 *
 * @example
 * ```typescript
 * const graph = await createHeroineGraph({
 *   canvas: '#graph-canvas',
 *   config: {
 *     simulation: { alphaDecay: 0.02 }
 *   }
 * });
 *
 * await graph.load({
 *   nodes: [{ id: 'a' }, { id: 'b' }],
 *   edges: [{ source: 'a', target: 'b' }]
 * });
 * ```
 */
export async function createHeroineGraph(
  options: CreateHeroineGraphOptions,
): Promise<HeroineGraph> {
  const { canvas, wasmUrl: _wasmUrl, gpu = {}, config = {}, debug = false } = options;

  // Resolve canvas element
  const canvasElement = resolveCanvas(canvas);

  // Check WebGPU availability
  const webgpuStatus = await checkWebGPU();
  if (!webgpuStatus.supported) {
    throw new HeroineGraphError(
      ErrorCode.WEBGPU_NOT_SUPPORTED,
      webgpuStatus.error || "WebGPU is not supported in this browser",
    );
  }

  // Load WASM module if not already loaded
  if (!isWasmLoaded()) {
    await wrapAsync(
      () => loadWasmModule(),
      ErrorCode.WASM_LOAD_FAILED,
      "Failed to load WASM module",
    );
  }

  // Create GPU context
  const gpuContext = await wrapAsync(
    () => createGPUContext({ canvas: canvasElement, ...gpu }),
    ErrorCode.WEBGPU_ADAPTER_FAILED,
    "Failed to create GPU context",
  );

  // Create WASM engine
  const wasmEngine = await createWasmEngine();

  // Get device capabilities
  const { device } = gpuContext;
  const limits = device.limits;

  const capabilities = {
    maxTextureSize: limits.maxTextureDimension2D,
    maxComputeWorkgroupSize: limits.maxComputeWorkgroupSizeX,
    maxBufferSize: limits.maxBufferSize,
  };

  if (debug) {
    console.log("HeroineGraph initialized with capabilities:", capabilities);
  }

  // Create HeroineGraph instance
  const graphConfig: HeroineGraphConfig = {
    gpuContext,
    wasmEngine,
    canvas: canvasElement,
    config,
    debug,
  };

  const graph = new HeroineGraph(graphConfig);

  return graph;
}

/**
 * Resolve canvas element from selector or element
 *
 * @param canvas - Canvas element or selector
 * @returns HTMLCanvasElement
 */
function resolveCanvas(canvas: HTMLCanvasElement | string): HTMLCanvasElement {
  if (typeof canvas === "string") {
    const element = document.querySelector(canvas);
    if (!element) {
      throw new HeroineGraphError(
        ErrorCode.CANVAS_NOT_FOUND,
        `Canvas element not found: ${canvas}`,
      );
    }
    if (!(element instanceof HTMLCanvasElement)) {
      throw new HeroineGraphError(
        ErrorCode.CANVAS_NOT_FOUND,
        `Element is not a canvas: ${canvas}`,
      );
    }
    return element;
  }
  return canvas;
}

/**
 * Quick check if HeroineGraph can be used in this environment
 *
 * @returns Promise resolving to true if supported
 */
export async function isSupported(): Promise<boolean> {
  return hasWebGPU();
}

/**
 * Get detailed support information
 *
 * @returns Promise resolving to support details
 */
export async function getSupportInfo(): Promise<{
  supported: boolean;
  webgpu: boolean;
  wasm: boolean;
  reason?: string | undefined;
}> {
  const webgpuStatus = await checkWebGPU();

  // Check WASM support
  const wasmSupported =
    typeof WebAssembly !== "undefined" &&
    typeof WebAssembly.instantiate === "function";

  return {
    supported: webgpuStatus.supported && wasmSupported,
    webgpu: webgpuStatus.supported,
    wasm: wasmSupported,
    reason: webgpuStatus.error,
  };
}

/**
 * Default WASM module URL
 */
export const DEFAULT_WASM_URL = "/heroine_graph_wasm_bg.wasm";

/**
 * Version information
 */
export const VERSION = {
  major: 0,
  minor: 1,
  patch: 0,
  toString() {
    return `${this.major}.${this.minor}.${this.patch}`;
  },
};
