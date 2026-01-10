/**
 * WASM Module Loader
 *
 * Handles loading and initialization of the Heroine Graph WASM module.
 * Provides zero-copy views for GPU buffer uploads.
 */

import type { HeroineGraphWasm } from "@heroine-graph/wasm";
import { ErrorCode, HeroineGraphError } from "../errors.ts";

/** WASM module state */
interface WasmState {
  /** The initialized WASM module */
  module: typeof import("@heroine-graph/wasm") | null;
  /** Whether initialization is in progress */
  loading: boolean;
  /** Promise for pending initialization */
  loadPromise: Promise<typeof import("@heroine-graph/wasm")> | null;
}

const state: WasmState = {
  module: null,
  loading: false,
  loadPromise: null,
};

/**
 * Load and initialize the WASM module.
 *
 * This function is idempotent - calling it multiple times returns the same module.
 * The WASM module is loaded lazily on first use.
 *
 * @returns Promise resolving to the WASM module exports
 * @throws HeroineGraphError if WASM loading fails
 */
export function loadWasmModule(): Promise<typeof import("@heroine-graph/wasm")> {
  // Already loaded
  if (state.module) {
    return Promise.resolve(state.module);
  }

  // Loading in progress
  if (state.loadPromise) {
    return state.loadPromise;
  }

  state.loading = true;
  state.loadPromise = (async () => {
    try {
      // Dynamic import of the WASM module
      const wasmModule = await import("@heroine-graph/wasm");

      // Initialize the module (calls wasm_bindgen start function)
      await wasmModule.default();

      state.module = wasmModule;
      state.loading = false;
      return wasmModule;
    } catch (error) {
      state.loading = false;
      state.loadPromise = null;

      const message = error instanceof Error ? error.message : String(error);
      throw new HeroineGraphError(
        ErrorCode.WASM_LOAD_FAILED,
        `Failed to load WASM module: ${message}`,
        { originalError: error },
      );
    }
  })();

  return state.loadPromise;
}

/**
 * Create a new HeroineGraphWasm engine instance.
 *
 * @returns Promise resolving to a new engine instance
 */
export async function createWasmEngine(): Promise<HeroineGraphWasm> {
  const module = await loadWasmModule();
  return new module.HeroineGraphWasm();
}

/**
 * Create a new HeroineGraphWasm engine with pre-allocated capacity.
 *
 * Use this when you know the approximate graph size upfront
 * to avoid reallocations.
 *
 * @param nodeCapacity Expected number of nodes
 * @param edgeCapacity Expected number of edges
 * @returns Promise resolving to a new engine instance
 */
export async function createWasmEngineWithCapacity(
  nodeCapacity: number,
  edgeCapacity: number,
): Promise<HeroineGraphWasm> {
  const module = await loadWasmModule();
  return module.HeroineGraphWasm.withCapacity(nodeCapacity, edgeCapacity);
}

/**
 * Check if the WASM module has been loaded.
 *
 * @returns True if the module is loaded and ready
 */
export function isWasmLoaded(): boolean {
  return state.module !== null;
}

/**
 * Get the WASM module if loaded.
 *
 * @returns The module or null if not loaded
 */
export function getWasmModule(): typeof import("@heroine-graph/wasm") | null {
  return state.module;
}

/**
 * WASM memory management utilities.
 *
 * These helpers handle the complexity of zero-copy data transfer
 * between JavaScript and WASM.
 */
export const WasmMemory = {
  /**
   * Create a Float32Array view into WASM memory.
   *
   * IMPORTANT: The returned view is invalidated if WASM memory grows.
   * Use immediately for GPU uploads, do not store long-term.
   *
   * @param wasmMemory The WASM memory object
   * @param ptr Pointer to the start of the data
   * @param length Number of f32 elements
   * @returns Float32Array view
   */
  viewFloat32(wasmMemory: WebAssembly.Memory, ptr: number, length: number): Float32Array {
    return new Float32Array(wasmMemory.buffer, ptr, length);
  },

  /**
   * Create a Uint32Array view into WASM memory.
   *
   * @param wasmMemory The WASM memory object
   * @param ptr Pointer to the start of the data
   * @param length Number of u32 elements
   * @returns Uint32Array view
   */
  viewUint32(wasmMemory: WebAssembly.Memory, ptr: number, length: number): Uint32Array {
    return new Uint32Array(wasmMemory.buffer, ptr, length);
  },

  /**
   * Track WASM memory buffer for change detection.
   *
   * WASM memory can grow, which changes the underlying ArrayBuffer.
   * Use this to detect when views need to be recreated.
   */
  createBufferTracker(wasmMemory: WebAssembly.Memory): {
    hasChanged(): boolean;
    update(): void;
  } {
    let cachedBuffer: ArrayBuffer = wasmMemory.buffer;

    return {
      hasChanged(): boolean {
        return wasmMemory.buffer !== cachedBuffer;
      },
      update(): void {
        cachedBuffer = wasmMemory.buffer;
      },
    };
  },
};
