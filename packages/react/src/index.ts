/**
 * @heroine-graph/react
 *
 * React wrapper for Heroine Graph - high-performance WebGPU graph visualization.
 *
 * @module
 */

// Main component
export { HeroineGraph, default } from "./HeroineGraph";
export type { HeroineGraphProps, HeroineGraphRef } from "./HeroineGraph";

// Hooks
export { useGraph } from "./hooks/useGraph";
export type { UseGraphOptions, UseGraphReturn } from "./hooks/useGraph";

export { useSimulation } from "./hooks/useSimulation";
export type { UseSimulationOptions, UseSimulationReturn } from "./hooks/useSimulation";

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
