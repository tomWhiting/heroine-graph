<!--
  GraphMother Svelte Component

  A Svelte wrapper for the GraphMother visualization library.
  Handles WebGPU initialization, lifecycle management, and event forwarding.

  @module
-->
<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import type {
    GraphConfig,
    GraphInput,
    GraphMother as GraphMotherCore,
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
  } from "@graphmother/core";
  import { createGraphMother, isSupported } from "@graphmother/core";

  /**
   * Props
   */
  interface Props {
    /** Graph data to display */
    data?: GraphInput;
    /** Graph configuration */
    config?: Partial<GraphConfig>;
    /** Width of the canvas (default: 100%) */
    width?: string | number;
    /** Height of the canvas (default: 100%) */
    height?: string | number;
    /** CSS class name for the container */
    class?: string;
    /** Enable debug mode */
    debug?: boolean;
    // Event callback props
    onready?: (graph: GraphMotherCore) => void;
    onerror?: (error: Error) => void;
    onnodeClick?: (event: NodeClickEvent) => void;
    onnodeDoubleClick?: (event: NodeDoubleClickEvent) => void;
    onnodeHoverEnter?: (event: NodeHoverEnterEvent) => void;
    onnodeHoverLeave?: (event: NodeHoverLeaveEvent) => void;
    onnodeDragStart?: (event: NodeDragStartEvent) => void;
    onnodeDragMove?: (event: NodeDragMoveEvent) => void;
    onnodeDragEnd?: (event: NodeDragEndEvent) => void;
    onedgeClick?: (event: EdgeClickEvent) => void;
    onedgeHoverEnter?: (event: EdgeHoverEnterEvent) => void;
    onedgeHoverLeave?: (event: EdgeHoverLeaveEvent) => void;
    onselectionChange?: (event: SelectionChangeEvent) => void;
    onviewportChange?: (event: ViewportChangeEvent) => void;
    onsimulationTick?: (event: SimulationTickEvent) => void;
    onsimulationEnd?: (event: SimulationEndEvent) => void;
    onbackgroundClick?: (event: BackgroundClickEvent) => void;
  }

  let {
    data = undefined,
    config = undefined,
    width = "100%",
    height = "100%",
    class: className = "",
    debug = false,
    onready = undefined,
    onerror = undefined,
    onnodeClick = undefined,
    onnodeDoubleClick = undefined,
    onnodeHoverEnter = undefined,
    onnodeHoverLeave = undefined,
    onnodeDragStart = undefined,
    onnodeDragMove = undefined,
    onnodeDragEnd = undefined,
    onedgeClick = undefined,
    onedgeHoverEnter = undefined,
    onedgeHoverLeave = undefined,
    onselectionChange = undefined,
    onviewportChange = undefined,
    onsimulationTick = undefined,
    onsimulationEnd = undefined,
    onbackgroundClick = undefined,
  }: Props = $props();

  // State
  // Both element refs are `$state` so `bind:this` updates are reactive: the
  // canvas lives inside an {:else} branch and is created/destroyed as the
  // error state toggles.
  let containerEl = $state<HTMLDivElement | undefined>(undefined);
  let canvasEl = $state<HTMLCanvasElement | undefined>(undefined);
  let graph = $state<GraphMotherCore | null>(null);
  let isInitialized = $state(false);
  let error = $state<Error | null>(null);

  // Computed styles
  const containerStyle = $derived(
    `width: ${typeof width === "number" ? `${width}px` : width}; ` +
    `height: ${typeof height === "number" ? `${height}px` : height}; ` +
    `position: relative;`
  );

  // Expose graph instance
  export function getGraph(): GraphMotherCore | null {
    return graph;
  }

  export function getCanvas(): HTMLCanvasElement | null {
    return canvasEl ?? null;
  }

  // Event handler registration
  function registerEventHandlers(g: GraphMotherCore) {
    g.on("node:click", (e) => onnodeClick?.(e));
    g.on("node:dblclick", (e) => onnodeDoubleClick?.(e));
    g.on("node:hoverenter", (e) => onnodeHoverEnter?.(e));
    g.on("node:hoverleave", (e) => onnodeHoverLeave?.(e));
    g.on("node:dragstart", (e) => onnodeDragStart?.(e));
    g.on("node:dragmove", (e) => onnodeDragMove?.(e));
    g.on("node:dragend", (e) => onnodeDragEnd?.(e));
    g.on("edge:click", (e) => onedgeClick?.(e));
    g.on("edge:hoverenter", (e) => onedgeHoverEnter?.(e));
    g.on("edge:hoverleave", (e) => onedgeHoverLeave?.(e));
    g.on("selection:change", (e) => onselectionChange?.(e));
    g.on("viewport:change", (e) => onviewportChange?.(e));
    g.on("simulation:tick", (e) => onsimulationTick?.(e));
    g.on("simulation:end", (e) => onsimulationEnd?.(e));
    g.on("background:click", (e) => onbackgroundClick?.(e));
  }

  // ResizeObserver
  let resizeObserver: ResizeObserver | null = null;

  // Bumped whenever the component is destroyed, so an in-flight initialization
  // can detect it is stale and dispose its instance instead of leaking a
  // WebGPU device and its render loop. Mirrors the Vue wrapper's
  // initGeneration counter and the React hook's initEpochRef.
  let initGeneration = 0;

  onMount(async () => {
    if (!canvasEl) return;

    const generation = ++initGeneration;

    try {
      // Check WebGPU support
      if (!isSupported()) {
        throw new Error(
          "WebGPU is not supported in this browser. Please use a browser with WebGPU support."
        );
      }

      // Create graph instance
      const graphInstance = await createGraphMother({
        canvas: canvasEl,
        config,
        debug,
      });

      // Destroyed while awaiting: discard the late instance rather than
      // leaking it (onDestroy already ran and saw graph === null).
      if (generation !== initGeneration) {
        graphInstance.dispose();
        return;
      }

      graph = graphInstance;
      isInitialized = true;

      // Register event handlers
      registerEventHandlers(graphInstance);

      // Emit ready event
      onready?.(graphInstance);

      // Data loading is handled by the $effect below

      // Setup resize observer
      if (containerEl) {
        resizeObserver = new ResizeObserver(() => {
          if (graph) {
            graph.resize();
          }
        });
        resizeObserver.observe(containerEl);
      }
    } catch (err) {
      if (generation !== initGeneration) return;
      const e = err instanceof Error ? err : new Error(String(err));
      error = e;
      onerror?.(e);
    }
  });

  onDestroy(() => {
    // Invalidate any in-flight initialization; the awaiting init disposes the
    // instance it is about to receive.
    initGeneration++;
    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }
    if (graph) {
      graph.dispose();
      graph = null;
    }
    isInitialized = false;
  });

  // Watch for data changes using $effect (also handles initial load)
  $effect(() => {
    if (!graph || !isInitialized || !data) return;

    graph.load(data).catch((err) => {
      const e = err instanceof Error ? err : new Error(String(err));
      error = e;
      onerror?.(e);
    });
  });
</script>

<div bind:this={containerEl} class={className} style={containerStyle}>
  {#if error}
    <div
      style="
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 100%;
        background-color: #1a1a1a;
        color: #ff6b6b;
        font-family: system-ui, sans-serif;
        padding: 20px;
        text-align: center;
      "
    >
      <div>
        <div style="font-size: 1.2em; margin-bottom: 8px;">
          Failed to initialize graph
        </div>
        <div style="opacity: 0.8; font-size: 0.9em;">{error.message}</div>
      </div>
    </div>
  {:else}
    <canvas
      bind:this={canvasEl}
      style="width: 100%; height: 100%; display: block;"
    ></canvas>
  {/if}
</div>
