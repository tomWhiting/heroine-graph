/**
 * @heroine-graph/vue
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
  GraphInput,
  GraphConfig,
  NodeInput,
  EdgeInput,
  NodeId,
  EdgeId,
  Vec2,
  SimulationStatus,
  ForceConfig,
  NodeClickEvent,
  NodeDoubleClickEvent,
  NodeDragStartEvent,
  NodeDragMoveEvent,
  NodeDragEndEvent,
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
