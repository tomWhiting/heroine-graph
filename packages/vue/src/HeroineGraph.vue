<!--
  GraphMother Vue Component

  A Vue wrapper for the GraphMother visualization library.
  Handles WebGPU initialization, lifecycle management, and event forwarding.

  @module
-->
<template>
  <div
    ref="containerRef"
    :class="className"
    :style="containerStyle"
  >
    <div
      v-if="error"
      :style="errorStyle"
    >
      <div style="font-size: 1.2em; margin-bottom: 8px;">
        Failed to initialize graph
      </div>
      <div style="opacity: 0.8; font-size: 0.9em;">{{ error.message }}</div>
    </div>
    <canvas
      v-else
      ref="canvasRef"
      style="width: 100%; height: 100%; display: block;"
    />
  </div>
</template>

<script setup lang="ts">
import {
  ref,
  computed,
  watch,
  nextTick,
  onMounted,
  onUnmounted,
  type CSSProperties,
} from "vue";
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
 * Props for the GraphMother component
 */
export interface GraphMotherProps {
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
  /** Enable debug mode */
  debug?: boolean;
}

const props = withDefaults(defineProps<GraphMotherProps>(), {
  width: "100%",
  height: "100%",
  debug: false,
});

const emit = defineEmits<{
  ready: [graph: GraphMotherCore];
  error: [error: Error];
  nodeClick: [event: NodeClickEvent];
  nodeDoubleClick: [event: NodeDoubleClickEvent];
  nodeHoverEnter: [event: NodeHoverEnterEvent];
  nodeHoverLeave: [event: NodeHoverLeaveEvent];
  nodeDragStart: [event: NodeDragStartEvent];
  nodeDragMove: [event: NodeDragMoveEvent];
  nodeDragEnd: [event: NodeDragEndEvent];
  edgeClick: [event: EdgeClickEvent];
  edgeHoverEnter: [event: EdgeHoverEnterEvent];
  edgeHoverLeave: [event: EdgeHoverLeaveEvent];
  selectionChange: [event: SelectionChangeEvent];
  viewportChange: [event: ViewportChangeEvent];
  simulationTick: [event: SimulationTickEvent];
  simulationEnd: [event: SimulationEndEvent];
  backgroundClick: [event: BackgroundClickEvent];
}>();

// Template refs
const containerRef = ref<HTMLDivElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);

// State
const graph = ref<GraphMotherCore | null>(null);
const isInitialized = ref(false);
const error = ref<Error | null>(null);

// Expose graph instance to parent
defineExpose({
  /** Get the underlying GraphMother instance */
  getGraph: () => graph.value,
  /** Get the canvas element */
  getCanvas: () => canvasRef.value,
  /**
   * Reload the current data prop into the graph.
   *
   * The data prop is watched by reference only (deep watching is O(nodes+edges)
   * per evaluation, prohibitive at large graph sizes). After mutating the data
   * object in place, call this to push the changes to the graph.
   */
  reload,
});

// Computed styles
const containerStyle = computed<CSSProperties>(() => ({
  width: typeof props.width === "number" ? `${props.width}px` : props.width,
  height: typeof props.height === "number" ? `${props.height}px` : props.height,
  position: "relative",
}));

const errorStyle = computed<CSSProperties>(() => ({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  width: "100%",
  height: "100%",
  backgroundColor: "#1a1a1a",
  color: "#ff6b6b",
  fontFamily: "system-ui, sans-serif",
  padding: "20px",
  textAlign: "center",
}));

// Event handler registration and cleanup
const registeredHandlers: Array<{ event: string; handler: (e: unknown) => void }> = [];

function registerEventHandlers(graphInstance: GraphMotherCore) {
  const handlers: Array<[string, (e: unknown) => void]> = [
    ["node:click", (e) => emit("nodeClick", e as NodeClickEvent)],
    ["node:dblclick", (e) => emit("nodeDoubleClick", e as NodeDoubleClickEvent)],
    ["node:hoverenter", (e) => emit("nodeHoverEnter", e as NodeHoverEnterEvent)],
    ["node:hoverleave", (e) => emit("nodeHoverLeave", e as NodeHoverLeaveEvent)],
    ["node:dragstart", (e) => emit("nodeDragStart", e as NodeDragStartEvent)],
    ["node:dragmove", (e) => emit("nodeDragMove", e as NodeDragMoveEvent)],
    ["node:dragend", (e) => emit("nodeDragEnd", e as NodeDragEndEvent)],
    ["edge:click", (e) => emit("edgeClick", e as EdgeClickEvent)],
    ["edge:hoverenter", (e) => emit("edgeHoverEnter", e as EdgeHoverEnterEvent)],
    ["edge:hoverleave", (e) => emit("edgeHoverLeave", e as EdgeHoverLeaveEvent)],
    ["selection:change", (e) => emit("selectionChange", e as SelectionChangeEvent)],
    ["viewport:change", (e) => emit("viewportChange", e as ViewportChangeEvent)],
    ["simulation:tick", (e) => emit("simulationTick", e as SimulationTickEvent)],
    ["simulation:end", (e) => emit("simulationEnd", e as SimulationEndEvent)],
    ["background:click", (e) => emit("backgroundClick", e as BackgroundClickEvent)],
  ];

  for (const [event, handler] of handlers) {
    graphInstance.on(event as Parameters<typeof graphInstance.on>[0], handler as Parameters<typeof graphInstance.on>[1]);
    registeredHandlers.push({ event, handler });
  }
}

// Bumped whenever the current graph is (or is about to be) destroyed, so an
// in-flight initialization can detect it is stale and dispose its instance
// instead of leaking a WebGPU device.
let initGeneration = 0;

async function initGraph(): Promise<void> {
  const canvas = canvasRef.value;
  if (!canvas) return;

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
      canvas,
      config: props.config,
      debug: props.debug,
    });

    // Destroyed/re-initialized while awaiting: discard the late instance
    if (generation !== initGeneration) {
      graphInstance.dispose();
      return;
    }

    graph.value = graphInstance;
    isInitialized.value = true;

    // Register event handlers
    registerEventHandlers(graphInstance);

    // Emit ready event
    emit("ready", graphInstance);

    // Load initial data if provided
    if (props.data) {
      await graphInstance.load(props.data);
    }
  } catch (err) {
    if (generation !== initGeneration) return;
    const e = err instanceof Error ? err : new Error(String(err));
    error.value = e;
    emit("error", e);
  }
}

function destroyGraph(): void {
  initGeneration++;
  if (graph.value) {
    for (const { event, handler } of registeredHandlers) {
      graph.value.off(event as Parameters<typeof graph.value.off>[0], handler as Parameters<typeof graph.value.off>[1]);
    }
    registeredHandlers.length = 0;
    graph.value.dispose();
    graph.value = null;
  }
  isInitialized.value = false;
}

/** Reload the current data prop into the graph (see defineExpose docs). */
async function reload(): Promise<void> {
  if (!graph.value || !isInitialized.value || !props.data) return;

  try {
    await graph.value.load(props.data);
  } catch (err) {
    const e = err instanceof Error ? err : new Error(String(err));
    error.value = e;
    emit("error", e);
  }
}

// Initialize graph
onMounted(initGraph);

// Cleanup on unmount
onUnmounted(destroyGraph);

// Watch for data changes by reference only. Deep watching would traverse
// every node/edge on each evaluation and reload the whole graph on any
// nested field mutation; consumers replace the data object (or call the
// exposed reload()) instead.
watch(
  () => props.data,
  async (newData) => {
    if (!graph.value || !isInitialized.value || !newData) return;

    try {
      await graph.value.load(newData);
    } catch (err) {
      const e = err instanceof Error ? err : new Error(String(err));
      error.value = e;
      emit("error", e);
    }
  }
);

// Config is consumed at creation time by the core, so re-initialize the
// graph when the config prop is replaced (matches the React wrapper).
watch(
  () => props.config,
  async () => {
    destroyGraph();
    error.value = null;
    // Wait for the canvas to re-render if the error template was showing
    await nextTick();
    await initGraph();
  }
);

// Handle resize
let resizeObserver: ResizeObserver | null = null;

onMounted(() => {
  const container = containerRef.value;
  if (!container) return;

  resizeObserver = new ResizeObserver(() => {
    if (graph.value) {
      graph.value.resize();
    }
  });
  resizeObserver.observe(container);
});

onUnmounted(() => {
  if (resizeObserver) {
    resizeObserver.disconnect();
    resizeObserver = null;
  }
});
</script>
