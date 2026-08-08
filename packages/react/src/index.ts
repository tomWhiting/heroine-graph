/**
 * @graphmother/react
 *
 * React wrapper for GraphMother - high-performance WebGPU graph visualization.
 *
 * @module
 */

// Main component
export { default, GraphMother } from "./GraphMother";
export type { GraphMotherProps, GraphMotherRef } from "./GraphMother";

// Hooks
export { useGraph } from "./hooks/useGraph";
export type { UseGraphOptions, UseGraphReturn } from "./hooks/useGraph";

export { useSimulation } from "./hooks/useSimulation";
export type { UseSimulationOptions, UseSimulationReturn } from "./hooks/useSimulation";

// DOM cards
export { GraphCards, useGraphCards } from "./cards";
export type { CardProviderHost, GraphCardsProps } from "./cards";

// The attribute a card's own element carries to become that card's drag handle
export { CARD_DRAG_HANDLE_ATTRIBUTE } from "@graphmother/core";

// Re-export core types for convenience
export type {
  BackgroundClickEvent,
  CardErrorEvent,
  CardNode,
  CardProviderHook,
  DomOverlayConfig,
  EdgeClickEvent,
  EdgeHoverEnterEvent,
  EdgeHoverLeaveEvent,
  EdgeId,
  EdgeInput,
  ForceConfig,
  GraphConfig,
  GraphInput,
  LiveCard,
  LodConfig,
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
