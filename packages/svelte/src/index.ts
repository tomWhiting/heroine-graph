/**
 * @graphmother/svelte
 *
 * Svelte wrapper for Heroine Graph - high-performance WebGPU graph visualization.
 *
 * @module
 */

// Main component
export { default as HeroineGraph } from "./HeroineGraph.svelte";

// Stores
export { createGraphStore } from "./stores/graph.svelte";
export type { GraphStore, GraphStoreOptions } from "./stores/graph.svelte";

export { createSimulationStore } from "./stores/simulation.svelte";
export type { SimulationStore, SimulationStoreOptions } from "./stores/simulation.svelte";

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
