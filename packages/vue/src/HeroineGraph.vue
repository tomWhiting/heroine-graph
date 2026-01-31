<!--
  HeroineGraph Vue Component

  A Vue wrapper for the Heroine Graph visualization library.
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
  onMounted,
  onUnmounted,
  type CSSProperties,
} from "vue";
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
  /** Enable debug mode */
  debug?: boolean;
}

const props = withDefaults(defineProps<HeroineGraphProps>(), {
  width: "100%",
  height: "100%",
  debug: false,
});

const emit = defineEmits<{
  ready: [graph: HeroineGraphCore];
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
const graph = ref<HeroineGraphCore | null>(null);
const isInitialized = ref(false);
const error = ref<Error | null>(null);

// Expose graph instance to parent
defineExpose({
  /** Get the underlying HeroineGraph instance */
  getGraph: () => graph.value,
  /** Get the canvas element */
  getCanvas: () => canvasRef.value,
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

// Event handler registration
function registerEventHandlers(graphInstance: HeroineGraphCore) {
  graphInstance.on("node:click", (e) => emit("nodeClick", e));
  graphInstance.on("node:doubleclick", (e) => emit("nodeDoubleClick", e));
  graphInstance.on("node:hoverenter", (e) => emit("nodeHoverEnter", e));
  graphInstance.on("node:hoverleave", (e) => emit("nodeHoverLeave", e));
  graphInstance.on("node:dragstart", (e) => emit("nodeDragStart", e));
  graphInstance.on("node:dragmove", (e) => emit("nodeDragMove", e));
  graphInstance.on("node:dragend", (e) => emit("nodeDragEnd", e));
  graphInstance.on("edge:click", (e) => emit("edgeClick", e));
  graphInstance.on("edge:hoverenter", (e) => emit("edgeHoverEnter", e));
  graphInstance.on("edge:hoverleave", (e) => emit("edgeHoverLeave", e));
  graphInstance.on("selection:change", (e) => emit("selectionChange", e));
  graphInstance.on("viewport:change", (e) => emit("viewportChange", e));
  graphInstance.on("simulation:tick", (e) => emit("simulationTick", e));
  graphInstance.on("simulation:end", (e) => emit("simulationEnd", e));
  graphInstance.on("background:click", (e) => emit("backgroundClick", e));
}

// Initialize graph
onMounted(async () => {
  const canvas = canvasRef.value;
  if (!canvas) return;

  try {
    // Check WebGPU support
    if (!isSupported()) {
      throw new Error(
        "WebGPU is not supported in this browser. Please use a browser with WebGPU support."
      );
    }

    // Create graph instance
    const graphInstance = await createHeroineGraph({
      canvas,
      config: props.config,
      debug: props.debug,
    });

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
    const e = err instanceof Error ? err : new Error(String(err));
    error.value = e;
    emit("error", e);
  }
});

// Cleanup on unmount
onUnmounted(() => {
  if (graph.value) {
    graph.value.dispose();
    graph.value = null;
  }
});

// Watch for data changes
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
