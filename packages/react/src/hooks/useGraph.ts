"use client";

/**
 * useGraph Hook
 *
 * Provides access to the GraphMother instance and common graph operations.
 *
 * @module
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { EdgeId, GraphConfig, GraphInput, GraphMother, NodeId, Vec2 } from "@graphmother/core";
import { createGraphMother, isSupported } from "@graphmother/core";

/**
 * Options for the useGraph hook
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
 * Return value of the useGraph hook
 */
export interface UseGraphReturn {
  /** The GraphMother instance (null until initialized) */
  graph: GraphMother | null;
  /** Whether the graph is initialized and ready */
  isReady: boolean;
  /** Whether the graph is currently loading data */
  isLoading: boolean;
  /** Any error that occurred during initialization or loading */
  error: Error | null;
  /** Whether WebGPU is supported */
  isSupported: boolean;

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
 * Hook for managing a GraphMother instance
 *
 * @example
 * ```tsx
 * import { useGraph } from '@graphmother/react';
 *
 * function GraphComponent() {
 *   const canvasRef = useRef<HTMLCanvasElement>(null);
 *   const {
 *     graph,
 *     isReady,
 *     initialize,
 *     load,
 *     selectNodes,
 *     fitToView,
 *   } = useGraph();
 *
 *   useEffect(() => {
 *     if (canvasRef.current) {
 *       initialize(canvasRef.current);
 *     }
 *   }, [initialize]);
 *
 *   useEffect(() => {
 *     if (isReady) {
 *       load({
 *         nodes: [{ id: 1 }, { id: 2 }],
 *         edges: [{ source: 1, target: 2 }],
 *       });
 *     }
 *   }, [isReady, load]);
 *
 *   return <canvas ref={canvasRef} />;
 * }
 * ```
 */
export function useGraph(options: UseGraphOptions = {}): UseGraphReturn {
  const { config, debug = false, initialData } = options;

  const graphRef = useRef<GraphMother | null>(null);
  // In-flight initialization, if any. Prevents StrictMode's double-invoked
  // effects (or any concurrent callers) from creating two instances.
  const initPromiseRef = useRef<Promise<void> | null>(null);
  // Bumped by dispose()/unmount to invalidate an in-flight initialization;
  // the stale init disposes its instance instead of leaking a WebGPU device.
  const initEpochRef = useRef(0);
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [supported] = useState(() => isSupported());

  // Initialize graph with canvas
  const initialize = useCallback(
    async (canvas: HTMLCanvasElement) => {
      if (graphRef.current) {
        return; // Already initialized
      }
      if (initPromiseRef.current) {
        return initPromiseRef.current; // Initialization already in flight
      }

      const epoch = initEpochRef.current;
      const initPromise = (async () => {
        try {
          // `config` is spread in only when present: the core options type is
          // exactOptionalPropertyTypes-strict and rejects `config: undefined`.
          const graph = await createGraphMother({
            canvas,
            ...(config ? { config } : {}),
            debug,
          });

          // Disposed/unmounted while awaiting: discard the late instance
          if (epoch !== initEpochRef.current) {
            graph.dispose();
            return;
          }

          graphRef.current = graph;
          setIsReady(true);
          setError(null);

          // Load initial data if provided
          if (initialData) {
            setIsLoading(true);
            await graph.load(initialData);
            setIsLoading(false);
          }
        } catch (err) {
          if (epoch === initEpochRef.current) {
            setError(err instanceof Error ? err : new Error(String(err)));
          }
        } finally {
          if (epoch === initEpochRef.current) {
            initPromiseRef.current = null;
          }
        }
      })();

      initPromiseRef.current = initPromise;
      return initPromise;
    },
    [config, debug, initialData],
  );

  // Load data
  const load = useCallback(async (data: GraphInput) => {
    const graph = graphRef.current;
    if (!graph) {
      throw new Error("Graph not initialized");
    }

    setIsLoading(true);
    try {
      await graph.load(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Dispose
  const dispose = useCallback(() => {
    // Invalidate any in-flight initialization so it disposes its instance
    initEpochRef.current++;
    initPromiseRef.current = null;
    if (graphRef.current) {
      graphRef.current.dispose();
      graphRef.current = null;
      setIsReady(false);
    }
  }, []);

  // Selection methods
  const selectNodes = useCallback((nodeIds: NodeId[]) => {
    graphRef.current?.selectNodes(nodeIds);
  }, []);

  const selectEdges = useCallback((edgeIds: EdgeId[]) => {
    graphRef.current?.selectEdges(edgeIds);
  }, []);

  const clearSelection = useCallback(() => {
    graphRef.current?.clearSelection();
  }, []);

  const getSelectedNodes = useCallback(() => {
    return graphRef.current?.getSelectedNodes() ?? [];
  }, []);

  const getSelectedEdges = useCallback(() => {
    return graphRef.current?.getSelectedEdges() ?? [];
  }, []);

  // Viewport methods
  const pan = useCallback((delta: Vec2) => {
    graphRef.current?.pan(delta.x, delta.y);
  }, []);

  const zoom = useCallback((scale: number, center?: Vec2) => {
    graphRef.current?.zoom(scale, center);
  }, []);

  const fitToView = useCallback((padding?: number) => {
    graphRef.current?.fitToView(padding);
  }, []);

  const resetView = useCallback(() => {
    graphRef.current?.fitToView();
  }, []);

  // Node operations
  const setNodePosition = useCallback((nodeId: NodeId, position: Vec2) => {
    graphRef.current?.setNodePosition(nodeId, position.x, position.y);
  }, []);

  const pinNode = useCallback((nodeId: NodeId) => {
    graphRef.current?.pinNode(nodeId);
  }, []);

  const unpinNode = useCallback((nodeId: NodeId) => {
    graphRef.current?.unpinNode(nodeId);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Invalidate any in-flight initialization so it disposes its instance
      initEpochRef.current++;
      initPromiseRef.current = null;
      if (graphRef.current) {
        graphRef.current.dispose();
        graphRef.current = null;
      }
    };
  }, []);

  return {
    graph: graphRef.current,
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
