/**
 * Graph Store for Svelte
 *
 * Provides reactive access to the GraphMother instance and common operations.
 *
 * @module
 */

import { onDestroy } from "svelte";
import type { EdgeId, GraphConfig, GraphInput, GraphMother, NodeId, Vec2 } from "@graphmother/core";
import { createGraphMother, isSupported } from "@graphmother/core";

/**
 * Options for createGraphStore
 */
export interface GraphStoreOptions {
  /** Initial graph configuration */
  config?: Partial<GraphConfig>;
  /** Enable debug mode */
  debug?: boolean;
  /** Initial graph data */
  initialData?: GraphInput;
}

/**
 * Create a reactive graph store
 *
 * The returned object is a Svelte 5 runes object, not a Svelte 4 store: read
 * its state through the plain properties (`store.isReady`), never with the `$`
 * store prefix. Destructuring the state getters would snapshot them and lose
 * reactivity, so destructure only the methods.
 *
 * When called during a component's initialization the store registers an
 * `onDestroy` hook that disposes the graph automatically. Created outside a
 * component (a module-level singleton, say), it cannot register that hook —
 * call `dispose()` yourself.
 *
 * @example
 * ```svelte
 * <script lang="ts">
 *   import { createGraphStore } from '@graphmother/svelte';
 *
 *   // Created during component init: disposal is registered automatically.
 *   const store = createGraphStore();
 *   // Methods are safe to destructure; state is read off `store`.
 *   const { initialize, load } = store;
 *
 *   let canvasEl: HTMLCanvasElement | undefined = $state();
 *
 *   $effect(() => {
 *     if (canvasEl) {
 *       initialize(canvasEl);
 *     }
 *   });
 * </script>
 *
 * <canvas bind:this={canvasEl}></canvas>
 * {#if store.isReady}
 *   <button onclick={() => load(myData)}>Load Data</button>
 * {/if}
 * ```
 *
 * Outside a component context, dispose explicitly:
 *
 * ```ts
 * const store = createGraphStore();
 * await store.initialize(canvas);
 * // ... later, when you are done with it:
 * store.dispose();
 * ```
 */
export function createGraphStore(options: GraphStoreOptions = {}) {
  const { config, debug = false, initialData } = options;

  // Reactive state using Svelte 5 runes
  let graph = $state<GraphMother | null>(null);
  let isReady = $state(false);
  let isLoading = $state(false);
  let error = $state<Error | null>(null);
  const supported = isSupported();

  // Bumped by dispose() to invalidate an in-flight initialization; the stale
  // init disposes its instance instead of leaking a WebGPU device.
  let initGeneration = 0;

  // Initialize graph with canvas
  async function initialize(canvas: HTMLCanvasElement): Promise<void> {
    if (graph) {
      return; // Already initialized
    }

    const generation = ++initGeneration;

    try {
      const graphInstance = await createGraphMother({
        canvas,
        config,
        debug,
      });

      // Disposed while awaiting: discard the late instance
      if (generation !== initGeneration) {
        graphInstance.dispose();
        return;
      }

      graph = graphInstance;
      isReady = true;
      error = null;

      // Load initial data if provided
      if (initialData) {
        isLoading = true;
        await graphInstance.load(initialData);
        isLoading = false;
      }
    } catch (err) {
      if (generation !== initGeneration) return;
      error = err instanceof Error ? err : new Error(String(err));
    }
  }

  // Load data
  async function load(data: GraphInput): Promise<void> {
    if (!graph) {
      throw new Error("Graph not initialized");
    }

    isLoading = true;
    try {
      await graph.load(data);
      error = null;
    } catch (err) {
      error = err instanceof Error ? err : new Error(String(err));
      throw err;
    } finally {
      isLoading = false;
    }
  }

  // Dispose
  function dispose(): void {
    // Invalidate any in-flight initialization so it disposes its instance
    initGeneration++;
    if (graph) {
      graph.dispose();
      graph = null;
      isReady = false;
    }
  }

  // Auto-dispose when the store is created during a component's
  // initialization. `onDestroy` throws when there is no component context
  // (module-level or plain-script usage), in which case the caller owns
  // disposal — see the doc comment above.
  try {
    onDestroy(dispose);
  } catch {
    // No component context; caller must call dispose() explicitly.
  }

  // Selection methods
  function selectNodes(nodeIds: NodeId[]): void {
    graph?.selectNodes(nodeIds);
  }

  function selectEdges(edgeIds: EdgeId[]): void {
    graph?.selectEdges(edgeIds);
  }

  function clearSelection(): void {
    graph?.clearSelection();
  }

  function getSelectedNodes(): NodeId[] {
    return graph?.getSelectedNodes() ?? [];
  }

  function getSelectedEdges(): EdgeId[] {
    return graph?.getSelectedEdges() ?? [];
  }

  // Viewport methods
  function pan(delta: Vec2): void {
    graph?.pan(delta.x, delta.y);
  }

  function zoom(scale: number, center?: Vec2): void {
    graph?.zoom(scale, center);
  }

  function fitToView(padding?: number): void {
    graph?.fitToView(padding);
  }

  function resetView(): void {
    graph?.fitToView();
  }

  // Node operations
  function setNodePosition(nodeId: NodeId, position: Vec2): void {
    graph?.setNodePosition(nodeId, position.x, position.y);
  }

  function pinNode(nodeId: NodeId): void {
    graph?.pinNode(nodeId);
  }

  function unpinNode(nodeId: NodeId): void {
    graph?.unpinNode(nodeId);
  }

  return {
    // State (use getter functions for reactivity)
    get graph() {
      return graph;
    },
    get isReady() {
      return isReady;
    },
    get isLoading() {
      return isLoading;
    },
    get error() {
      return error;
    },
    get isSupported() {
      return supported;
    },

    // Methods
    initialize,
    load,
    dispose,
    selectNodes,
    selectEdges,
    clearSelection,
    getSelectedNodes,
    getSelectedEdges,
    pan,
    zoom,
    fitToView,
    resetView,
    setNodePosition,
    pinNode,
    unpinNode,
  };
}

export type GraphStore = ReturnType<typeof createGraphStore>;
