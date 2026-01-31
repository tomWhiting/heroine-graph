/**
 * @heroine-graph/svelte
 *
 * Svelte wrapper for Heroine Graph - high-performance WebGPU graph visualization.
 *
 * @module
 */

// Main component
export { default as HeroineGraph } from "./HeroineGraph.svelte";

// Stores
export { createGraphStore } from "./stores/graph.svelte";
export type { GraphStoreOptions, GraphStore } from "./stores/graph.svelte";

export { createSimulationStore } from "./stores/simulation.svelte";
export type { SimulationStoreOptions, SimulationStore } from "./stores/simulation.svelte";

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
