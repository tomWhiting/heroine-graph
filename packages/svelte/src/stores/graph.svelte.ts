/**
 * Graph Store for Svelte
 *
 * Provides reactive access to the HeroineGraph instance and common operations.
 *
 * @module
 */

import type {
  GraphConfig,
  GraphInput,
  HeroineGraph,
  NodeId,
  EdgeId,
  Vec2,
} from "@heroine-graph/core";
import { createHeroineGraph, isSupported } from "@heroine-graph/core";

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
 * @example
 * ```svelte
 * <script lang="ts">
 *   import { createGraphStore } from '@heroine-graph/svelte';
 *
 *   const graphStore = createGraphStore();
 *   const { graph, isReady, initialize, load } = graphStore;
 *
 *   let canvasEl: HTMLCanvasElement;
 *
 *   $effect(() => {
 *     if (canvasEl) {
 *       initialize(canvasEl);
 *     }
 *   });
 * </script>
 *
 * <canvas bind:this={canvasEl} />
 * {#if $isReady}
 *   <button onclick={() => load(myData)}>Load Data</button>
 * {/if}
 * ```
 */
export function createGraphStore(options: GraphStoreOptions = {}) {
  const { config, debug = false, initialData } = options;

  // Reactive state using Svelte 5 runes
  let graph = $state<HeroineGraph | null>(null);
  let isReady = $state(false);
  let isLoading = $state(false);
  let error = $state<Error | null>(null);
  const supported = isSupported();

  // Initialize graph with canvas
  async function initialize(canvas: HTMLCanvasElement): Promise<void> {
    if (graph) {
      return; // Already initialized
    }

    try {
      const graphInstance = await createHeroineGraph({
        canvas,
        config,
        debug,
      });

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
    if (graph) {
      graph.dispose();
      graph = null;
      isReady = false;
    }
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
    get graph() { return graph; },
    get isReady() { return isReady; },
    get isLoading() { return isLoading; },
    get error() { return error; },
    get isSupported() { return supported; },

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
