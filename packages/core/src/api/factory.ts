/**
 * GraphMother Factory Function
 *
 * Creates and initializes a GraphMother instance with all required
 * subsystems (WebGPU, WASM, rendering, simulation).
 *
 * @module
 */

import { ErrorCode, GraphMotherError, wrapAsync } from "../errors.ts";
import { checkWebGPU, hasWebGPU } from "../webgpu/check.ts";
import { createGPUContext, type GPUContext, type GPUContextOptions } from "../webgpu/context.ts";
import { createWasmEngine, isWasmLoaded, loadWasmModule } from "../wasm/loader.ts";
import { GraphMother, type GraphMotherConfig } from "./graph.ts";
import type { GraphConfig } from "../types.ts";

/**
 * Options for creating a GraphMother instance
 */
export interface CreateGraphMotherOptions {
  /** Canvas element or selector to render into */
  canvas: HTMLCanvasElement | string;

  /**
   * URL of the .wasm binary (optional). When omitted the wasm-bindgen glue
   * resolves the binary relative to itself, which is what bundlers expect.
   */
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
  /** The created GraphMother instance */
  graph: GraphMother;
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
 * Creates a GraphMother instance
 *
 * This is the main entry point for using the library. It handles:
 * - WebGPU availability check and initialization
 * - WASM module loading
 * - GPU context creation
 * - Render pipeline setup
 *
 * @param options - Creation options
 * @returns Promise resolving to GraphMother instance
 *
 * @example
 * ```typescript
 * const graph = await createGraphMother({
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
export async function createGraphMother(
  options: CreateGraphMotherOptions,
): Promise<GraphMother> {
  const {
    canvas,
    wasmUrl,
    gpu = {},
    config = {},
    debug = false,
  } = options;

  // Resolve canvas element
  const canvasElement = resolveCanvas(canvas);

  // Check WebGPU availability
  const webgpuStatus = await checkWebGPU();
  if (!webgpuStatus.supported) {
    throw new GraphMotherError(
      ErrorCode.WEBGPU_NOT_SUPPORTED,
      webgpuStatus.error || "WebGPU is not supported in this browser",
    );
  }

  // Load WASM module if not already loaded
  if (!isWasmLoaded()) {
    await wrapAsync(
      () => loadWasmModule(wasmUrl),
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
    console.log("GraphMother initialized with capabilities:", capabilities);
  }

  // Create GraphMother instance
  const graphConfig: GraphMotherConfig = {
    gpuContext,
    wasmEngine,
    canvas: canvasElement,
    config,
    debug,
  };

  const graph = new GraphMother(graphConfig);

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
      throw new GraphMotherError(
        ErrorCode.CANVAS_NOT_FOUND,
        `Canvas element not found: ${canvas}`,
      );
    }
    if (!(element instanceof HTMLCanvasElement)) {
      throw new GraphMotherError(
        ErrorCode.CANVAS_NOT_FOUND,
        `Element is not a canvas: ${canvas}`,
      );
    }
    return element;
  }
  return canvas;
}

/**
 * Quick check if GraphMother can be used in this environment
 *
 * @returns Promise resolving to true if supported
 */
export function isSupported(): boolean {
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
  const wasmSupported = typeof WebAssembly !== "undefined" &&
    typeof WebAssembly.instantiate === "function";

  return {
    supported: webgpuStatus.supported && wasmSupported,
    webgpu: webgpuStatus.supported,
    wasm: wasmSupported,
    reason: webgpuStatus.error,
  };
}

/**
 * Version information
 */
export const VERSION = {
  major: 0,
  minor: 2,
  patch: 1,
  toString() {
    return `${this.major}.${this.minor}.${this.patch}`;
  },
};
