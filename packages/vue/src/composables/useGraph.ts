/**
 * useGraph Composable
 *
 * Provides access to the HeroineGraph instance and common graph operations.
 *
 * @module
 */

import { ref, shallowRef, onUnmounted, type Ref, type ShallowRef } from "vue";
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
 * Options for the useGraph composable
 */
export interface UseGraphOptions {
  /** Initial graph configuration */
  config?: Partial<GraphConfig>;
  /** Enable debug mode */
  debug?: boolean;
  /** Initial graph data */
  initialData?: GraphInput;
}

/**
 * Return value of the useGraph composable
 */
export interface UseGraphReturn {
  /** The HeroineGraph instance (null until initialized) */
  graph: ShallowRef<HeroineGraph | null>;
  /** Whether the graph is initialized and ready */
  isReady: Ref<boolean>;
  /** Whether the graph is currently loading data */
  isLoading: Ref<boolean>;
  /** Any error that occurred during initialization or loading */
  error: Ref<Error | null>;
  /** Whether WebGPU is supported */
  isSupported: Ref<boolean>;

  // Methods
  /** Initialize the graph with a canvas element */
  initialize: (canvas: HTMLCanvasElement) => Promise<void>;
  /** Load graph data */
  load: (data: GraphInput) => Promise<void>;
  /** Dispose of the graph instance */
  dispose: () => void;

  // Selection
  /** Select nodes by ID */
  selectNodes: (nodeIds: NodeId[]) => void;
  /** Select edges by ID */
  selectEdges: (edgeIds: EdgeId[]) => void;
  /** Clear all selections */
  clearSelection: () => void;
  /** Get selected node IDs */
  getSelectedNodes: () => NodeId[];
  /** Get selected edge IDs */
  getSelectedEdges: () => EdgeId[];

  // Viewport
  /** Pan the viewport by delta */
  pan: (delta: Vec2) => void;
  /** Zoom to a specific scale */
  zoom: (scale: number, center?: Vec2) => void;
  /** Fit the graph to the viewport */
  fitToView: (padding?: number) => void;
  /** Reset the viewport to initial state */
  resetView: () => void;

  // Node operations
  /** Set position for a node */
  setNodePosition: (nodeId: NodeId, position: Vec2) => void;
  /** Pin a node (prevent simulation from moving it) */
  pinNode: (nodeId: NodeId) => void;
  /** Unpin a node */
  unpinNode: (nodeId: NodeId) => void;
}

/**
 * Composable for managing a HeroineGraph instance
 *
 * @example
 * ```vue
 * <script setup lang="ts">
 * import { ref, onMounted } from 'vue';
 * import { useGraph } from '@heroine-graph/vue';
 *
 * const canvasRef = ref<HTMLCanvasElement | null>(null);
 * const {
 *   graph,
 *   isReady,
 *   initialize,
 *   load,
 *   selectNodes,
 *   fitToView,
 * } = useGraph();
 *
 * onMounted(async () => {
 *   if (canvasRef.value) {
 *     await initialize(canvasRef.value);
 *     await load({
 *       nodes: [{ id: 1 }, { id: 2 }],
 *       edges: [{ source: 1, target: 2 }],
 *     });
 *   }
 * });
 * </script>
 *
 * <template>
 *   <canvas ref="canvasRef" />
 * </template>
 * ```
 */
export function useGraph(options: UseGraphOptions = {}): UseGraphReturn {
  const { config, debug = false, initialData } = options;

  const graph = shallowRef<HeroineGraph | null>(null);
  const isReady = ref(false);
  const isLoading = ref(false);
  const error = ref<Error | null>(null);
  const supported = ref(isSupported());

  // Initialize graph with canvas
  async function initialize(canvas: HTMLCanvasElement): Promise<void> {
    if (graph.value) {
      return; // Already initialized
    }

    try {
      const graphInstance = await createHeroineGraph({
        canvas,
        config,
        debug,
      });

      graph.value = graphInstance;
      isReady.value = true;
      error.value = null;

      // Load initial data if provided
      if (initialData) {
        isLoading.value = true;
        await graphInstance.load(initialData);
        isLoading.value = false;
      }
    } catch (err) {
      error.value = err instanceof Error ? err : new Error(String(err));
    }
  }

  // Load data
  async function load(data: GraphInput): Promise<void> {
    const g = graph.value;
    if (!g) {
      throw new Error("Graph not initialized");
    }

    isLoading.value = true;
    try {
      await g.load(data);
      error.value = null;
    } catch (err) {
      error.value = err instanceof Error ? err : new Error(String(err));
      throw err;
    } finally {
      isLoading.value = false;
    }
  }

  // Dispose
  function dispose(): void {
    if (graph.value) {
      graph.value.dispose();
      graph.value = null;
      isReady.value = false;
    }
  }

  // Selection methods
  function selectNodes(nodeIds: NodeId[]): void {
    graph.value?.selectNodes(nodeIds);
  }

  function selectEdges(edgeIds: EdgeId[]): void {
    graph.value?.selectEdges(edgeIds);
  }

  function clearSelection(): void {
    graph.value?.clearSelection();
  }

  function getSelectedNodes(): NodeId[] {
    return graph.value?.getSelectedNodes() ?? [];
  }

  function getSelectedEdges(): EdgeId[] {
    return graph.value?.getSelectedEdges() ?? [];
  }

  // Viewport methods
  function pan(delta: Vec2): void {
    graph.value?.pan(delta.x, delta.y);
  }

  function zoom(scale: number, center?: Vec2): void {
    graph.value?.zoom(scale, center);
  }

  function fitToView(padding?: number): void {
    graph.value?.fitToView(padding);
  }

  function resetView(): void {
    graph.value?.fitToView();
  }

  // Node operations
  function setNodePosition(nodeId: NodeId, position: Vec2): void {
    graph.value?.setNodePosition(nodeId, position.x, position.y);
  }

  function pinNode(nodeId: NodeId): void {
    graph.value?.pinNode(nodeId);
  }

  function unpinNode(nodeId: NodeId): void {
    graph.value?.unpinNode(nodeId);
  }

  // Cleanup on unmount
  onUnmounted(() => {
    if (graph.value) {
      graph.value.dispose();
      graph.value = null;
    }
  });

  return {
    graph,
    isReady,
    isLoading,
    error,
    isSupported: supported,
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

export default useGraph;
