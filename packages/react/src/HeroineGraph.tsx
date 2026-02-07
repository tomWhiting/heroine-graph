/**
 * HeroineGraph React Component
 *
 * A React wrapper for the Heroine Graph visualization library.
 * Handles WebGPU initialization, lifecycle management, and event forwarding.
 *
 * @module
 */

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import type {
  GraphConfig,
  GraphInput,
  HeroineGraph as HeroineGraphCore,
  NodeClickEvent,
  NodeDoubleClickEvent,
  NodeDragEndEvent,
  NodeDragMoveEvent,
  NodeDragStartEvent,
  NodeHoverEnterEvent,
  NodeHoverLeaveEvent,
  EdgeClickEvent,
  EdgeHoverEnterEvent,
  EdgeHoverLeaveEvent,
  SelectionChangeEvent,
  ViewportChangeEvent,
  SimulationTickEvent,
  SimulationEndEvent,
  BackgroundClickEvent,
} from "@heroine-graph/core";
import { createHeroineGraph, isSupported } from "@heroine-graph/core";

/**
 * Props for the HeroineGraph component
 */
export interface HeroineGraphProps {
  /** Graph data to display */
  data?: GraphInput;
  /** Graph configuration */
  config?: Partial<GraphConfig>;
  /** Width of the canvas (default: 100%) */
  width?: string | number;
  /** Height of the canvas (default: 100%) */
  height?: string | number;
  /** CSS class name for the container */
  className?: string;
  /** Inline styles for the container */
  style?: React.CSSProperties;
  /** Enable debug mode */
  debug?: boolean;

  // Event callbacks
  onReady?: (graph: HeroineGraphCore) => void;
  onError?: (error: Error) => void;
  onNodeClick?: (event: NodeClickEvent) => void;
  onNodeDoubleClick?: (event: NodeDoubleClickEvent) => void;
  onNodeHoverEnter?: (event: NodeHoverEnterEvent) => void;
  onNodeHoverLeave?: (event: NodeHoverLeaveEvent) => void;
  onNodeDragStart?: (event: NodeDragStartEvent) => void;
  onNodeDragMove?: (event: NodeDragMoveEvent) => void;
  onNodeDragEnd?: (event: NodeDragEndEvent) => void;
  onEdgeClick?: (event: EdgeClickEvent) => void;
  onEdgeHoverEnter?: (event: EdgeHoverEnterEvent) => void;
  onEdgeHoverLeave?: (event: EdgeHoverLeaveEvent) => void;
  onSelectionChange?: (event: SelectionChangeEvent) => void;
  onViewportChange?: (event: ViewportChangeEvent) => void;
  onSimulationTick?: (event: SimulationTickEvent) => void;
  onSimulationEnd?: (event: SimulationEndEvent) => void;
  onBackgroundClick?: (event: BackgroundClickEvent) => void;
}

/**
 * Ref handle for the HeroineGraph component
 */
export interface HeroineGraphRef {
  /** Get the underlying HeroineGraph instance */
  getGraph: () => HeroineGraphCore | null;
  /** Get the canvas element */
  getCanvas: () => HTMLCanvasElement | null;
}

/**
 * HeroineGraph React Component
 *
 * Renders a high-performance graph visualization using WebGPU.
 *
 * @example
 * ```tsx
 * import { HeroineGraph } from '@heroine-graph/react';
 *
 * function App() {
 *   const graphRef = useRef<HeroineGraphRef>(null);
 *
 *   const data = {
 *     nodes: [
 *       { id: 1, label: 'Node 1' },
 *       { id: 2, label: 'Node 2' },
 *     ],
 *     edges: [
 *       { source: 1, target: 2 },
 *     ],
 *   };
 *
 *   return (
 *     <HeroineGraph
 *       ref={graphRef}
 *       data={data}
 *       onNodeClick={(e) => console.log('Clicked:', e.nodeId)}
 *       style={{ width: '100%', height: '600px' }}
 *     />
 *   );
 * }
 * ```
 */
export const HeroineGraph = forwardRef<HeroineGraphRef, HeroineGraphProps>(
  function HeroineGraph(props, ref) {
    const {
      data,
      config,
      width = "100%",
      height = "100%",
      className,
      style,
      debug = false,
      onReady,
      onError,
      onNodeClick,
      onNodeDoubleClick,
      onNodeHoverEnter,
      onNodeHoverLeave,
      onNodeDragStart,
      onNodeDragMove,
      onNodeDragEnd,
      onEdgeClick,
      onEdgeHoverEnter,
      onEdgeHoverLeave,
      onSelectionChange,
      onViewportChange,
      onSimulationTick,
      onSimulationEnd,
      onBackgroundClick,
    } = props;

    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const graphRef = useRef<HeroineGraphCore | null>(null);
    const [isInitialized, setIsInitialized] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    // Store callback refs to avoid re-init on callback identity change
    const onReadyRef = useRef(onReady);
    onReadyRef.current = onReady;
    const onErrorRef = useRef(onError);
    onErrorRef.current = onError;

    // Expose ref handle
    useImperativeHandle(ref, () => ({
      getGraph: () => graphRef.current,
      getCanvas: () => canvasRef.current,
    }));

    // Initialize graph
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      let mounted = true;

      async function init() {
        try {
          // Check WebGPU support
          if (!isSupported()) {
            throw new Error(
              "WebGPU is not supported in this browser. Please use a browser with WebGPU support."
            );
          }

          // Create graph instance
          const graph = await createHeroineGraph({
            canvas,
            config,
            debug,
          });

          if (!mounted) {
            graph.dispose();
            return;
          }

          graphRef.current = graph;
          setIsInitialized(true);
          onReadyRef.current?.(graph);
        } catch (err) {
          const error = err instanceof Error ? err : new Error(String(err));
          setError(error);
          onErrorRef.current?.(error);
        }
      }

      init();

      return () => {
        mounted = false;
        if (graphRef.current) {
          graphRef.current.dispose();
          graphRef.current = null;
        }
        setIsInitialized(false);
      };
    }, [config, debug]);

    // Register event handlers
    useEffect(() => {
      const graph = graphRef.current;
      if (!graph || !isInitialized) return;

      const handlers: Array<{ event: string; handler: (e: unknown) => void }> = [];

      if (onNodeClick) {
        handlers.push({ event: "node:click", handler: onNodeClick as (e: unknown) => void });
      }
      if (onNodeDoubleClick) {
        handlers.push({ event: "node:doubleclick", handler: onNodeDoubleClick as (e: unknown) => void });
      }
      if (onNodeHoverEnter) {
        handlers.push({ event: "node:hoverenter", handler: onNodeHoverEnter as (e: unknown) => void });
      }
      if (onNodeHoverLeave) {
        handlers.push({ event: "node:hoverleave", handler: onNodeHoverLeave as (e: unknown) => void });
      }
      if (onNodeDragStart) {
        handlers.push({ event: "node:dragstart", handler: onNodeDragStart as (e: unknown) => void });
      }
      if (onNodeDragMove) {
        handlers.push({ event: "node:dragmove", handler: onNodeDragMove as (e: unknown) => void });
      }
      if (onNodeDragEnd) {
        handlers.push({ event: "node:dragend", handler: onNodeDragEnd as (e: unknown) => void });
      }
      if (onEdgeClick) {
        handlers.push({ event: "edge:click", handler: onEdgeClick as (e: unknown) => void });
      }
      if (onEdgeHoverEnter) {
        handlers.push({ event: "edge:hoverenter", handler: onEdgeHoverEnter as (e: unknown) => void });
      }
      if (onEdgeHoverLeave) {
        handlers.push({ event: "edge:hoverleave", handler: onEdgeHoverLeave as (e: unknown) => void });
      }
      if (onSelectionChange) {
        handlers.push({ event: "selection:change", handler: onSelectionChange as (e: unknown) => void });
      }
      if (onViewportChange) {
        handlers.push({ event: "viewport:change", handler: onViewportChange as (e: unknown) => void });
      }
      if (onSimulationTick) {
        handlers.push({ event: "simulation:tick", handler: onSimulationTick as (e: unknown) => void });
      }
      if (onSimulationEnd) {
        handlers.push({ event: "simulation:end", handler: onSimulationEnd as (e: unknown) => void });
      }
      if (onBackgroundClick) {
        handlers.push({ event: "background:click", handler: onBackgroundClick as (e: unknown) => void });
      }

      // Register all handlers
      for (const { event, handler } of handlers) {
        graph.on(event as Parameters<typeof graph.on>[0], handler as Parameters<typeof graph.on>[1]);
      }

      // Cleanup
      return () => {
        for (const { event, handler } of handlers) {
          graph.off(event as Parameters<typeof graph.off>[0], handler as Parameters<typeof graph.off>[1]);
        }
      };
    }, [
      isInitialized,
      onNodeClick,
      onNodeDoubleClick,
      onNodeHoverEnter,
      onNodeHoverLeave,
      onNodeDragStart,
      onNodeDragMove,
      onNodeDragEnd,
      onEdgeClick,
      onEdgeHoverEnter,
      onEdgeHoverLeave,
      onSelectionChange,
      onViewportChange,
      onSimulationTick,
      onSimulationEnd,
      onBackgroundClick,
    ]);

    // Load data when it changes
    useEffect(() => {
      const graph = graphRef.current;
      if (!graph || !isInitialized || !data) return;

      graph.load(data).catch((err) => {
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
        onErrorRef.current?.(error);
      });
    }, [data, isInitialized]);

    // Handle resize
    const handleResize = useCallback(() => {
      if (graphRef.current && containerRef.current) {
        graphRef.current.resize();
      }
    }, []);

    useEffect(() => {
      const container = containerRef.current;
      if (!container) return;

      const resizeObserver = new ResizeObserver(handleResize);
      resizeObserver.observe(container);

      return () => {
        resizeObserver.disconnect();
      };
    }, [handleResize]);

    // Container styles
    const containerStyle: React.CSSProperties = {
      width: typeof width === "number" ? `${width}px` : width,
      height: typeof height === "number" ? `${height}px` : height,
      position: "relative",
      ...style,
    };

    if (error) {
      return (
        <div
          ref={containerRef}
          className={className}
          style={{
            ...containerStyle,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "#1a1a1a",
            color: "#ff6b6b",
            fontFamily: "system-ui, sans-serif",
            padding: "20px",
            textAlign: "center",
          }}
        >
          <div>
            <div style={{ fontSize: "1.2em", marginBottom: "8px" }}>
              Failed to initialize graph
            </div>
            <div style={{ opacity: 0.8, fontSize: "0.9em" }}>{error.message}</div>
          </div>
        </div>
      );
    }

    return (
      <div ref={containerRef} className={className} style={containerStyle}>
        <canvas
          ref={canvasRef}
          style={{
            width: "100%",
            height: "100%",
            display: "block",
          }}
        />
      </div>
    );
  }
);

export default HeroineGraph;
