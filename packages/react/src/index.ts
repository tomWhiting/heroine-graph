/**
 * @graphmother/react
 *
 * React wrapper for Heroine Graph - high-performance WebGPU graph visualization.
 *
 * @module
 */

// Main component
export { default, HeroineGraph } from "./HeroineGraph";
export type { HeroineGraphProps, HeroineGraphRef } from "./HeroineGraph";

// Hooks
export { useGraph } from "./hooks/useGraph";
export type { UseGraphOptions, UseGraphReturn } from "./hooks/useGraph";

export { useSimulation } from "./hooks/useSimulation";
export type { UseSimulationOptions, UseSimulationReturn } from "./hooks/useSimulation";

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
