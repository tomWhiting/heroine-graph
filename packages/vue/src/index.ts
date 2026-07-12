/**
 * @graphmother/vue
 *
 * Vue wrapper for Heroine Graph - high-performance WebGPU graph visualization.
 *
 * @module
 */

// Main component
export { default as HeroineGraph } from "./HeroineGraph.vue";
export type { HeroineGraphProps } from "./HeroineGraph.vue";

// Composables
export { useGraph } from "./composables/useGraph";
export type { UseGraphOptions, UseGraphReturn } from "./composables/useGraph";

export { useSimulation } from "./composables/useSimulation";
export type { UseSimulationOptions, UseSimulationReturn } from "./composables/useSimulation";

// Re-export core types for convenience
export type {
  BackgroundClickEvent,
  EdgeClickEvent,
  EdgeHoverEnterEvent,
  EdgeHoverLeaveEvent,
  EdgeId,
  EdgeInput,
  ForceConfig,
  GraphConfig,
  GraphInput,
  NodeClickEvent,
  NodeDoubleClickEvent,
  NodeDragEndEvent,
  NodeDragMoveEvent,
  NodeDragStartEvent,
  NodeHoverEnterEvent,
  NodeHoverLeaveEvent,
  NodeId,
  NodeInput,
  SelectionChangeEvent,
  SimulationEndEvent,
  SimulationStatus,
  SimulationTickEvent,
  Vec2,
  ViewportChangeEvent,
} from "@graphmother/core";
