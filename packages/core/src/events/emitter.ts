/**
 * Event Emitter for Heroine Graph
 *
 * A type-safe event emitter supporting typed events with
 * subscribe/unsubscribe and one-time listeners.
 */

import type { EventMap, GraphEvent, EventHandler } from "../types.ts";

/**
 * Type-safe event emitter for graph events.
 */
export class EventEmitter {
  private readonly listeners: Map<keyof EventMap, Set<EventHandler<GraphEvent>>>;
  private readonly onceListeners: Map<keyof EventMap, Set<EventHandler<GraphEvent>>>;

  constructor() {
    this.listeners = new Map();
    this.onceListeners = new Map();
  }

  /**
   * Subscribe to an event.
   *
   * @param type Event type
   * @param handler Event handler function
   * @returns Unsubscribe function
   */
  on<K extends keyof EventMap>(type: K, handler: EventHandler<EventMap[K]>): () => void {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    set.add(handler as EventHandler<GraphEvent>);

    // Return unsubscribe function
    return () => {
      this.off(type, handler);
    };
  }

  /**
   * Subscribe to an event (one-time).
   *
   * The handler will be automatically removed after the first call.
   *
   * @param type Event type
   * @param handler Event handler function
   * @returns Unsubscribe function
   */
  once<K extends keyof EventMap>(type: K, handler: EventHandler<EventMap[K]>): () => void {
    let set = this.onceListeners.get(type);
    if (!set) {
      set = new Set();
      this.onceListeners.set(type, set);
    }
    set.add(handler as EventHandler<GraphEvent>);

    return () => {
      set?.delete(handler as EventHandler<GraphEvent>);
    };
  }

  /**
   * Unsubscribe from an event.
   *
   * @param type Event type
   * @param handler Event handler function
   */
  off<K extends keyof EventMap>(type: K, handler: EventHandler<EventMap[K]>): void {
    const set = this.listeners.get(type);
    if (set) {
      set.delete(handler as EventHandler<GraphEvent>);
      if (set.size === 0) {
        this.listeners.delete(type);
      }
    }

    const onceSet = this.onceListeners.get(type);
    if (onceSet) {
      onceSet.delete(handler as EventHandler<GraphEvent>);
      if (onceSet.size === 0) {
        this.onceListeners.delete(type);
      }
    }
  }

  /**
   * Emit an event.
   *
   * @param event The event to emit
   */
  emit<K extends keyof EventMap>(event: EventMap[K]): void {
    const type = event.type as K;

    // Call regular listeners
    const listeners = this.listeners.get(type);
    if (listeners) {
      for (const handler of listeners) {
        try {
          handler(event);
        } catch (error) {
          console.error(`[HeroineGraph] Error in event handler for "${type}":`, error);
        }
      }
    }

    // Call and remove once listeners
    const onceListeners = this.onceListeners.get(type);
    if (onceListeners) {
      for (const handler of onceListeners) {
        try {
          handler(event);
        } catch (error) {
          console.error(`[HeroineGraph] Error in once handler for "${type}":`, error);
        }
      }
      this.onceListeners.delete(type);
    }
  }

  /**
   * Remove all listeners for a specific event type.
   *
   * @param type Event type
   */
  removeAllListeners<K extends keyof EventMap>(type: K): void {
    this.listeners.delete(type);
    this.onceListeners.delete(type);
  }

  /**
   * Remove all listeners for all event types.
   */
  clear(): void {
    this.listeners.clear();
    this.onceListeners.clear();
  }

  /**
   * Get the number of listeners for an event type.
   *
   * @param type Event type
   * @returns Number of listeners
   */
  listenerCount<K extends keyof EventMap>(type: K): number {
    const regular = this.listeners.get(type)?.size ?? 0;
    const once = this.onceListeners.get(type)?.size ?? 0;
    return regular + once;
  }

  /**
   * Check if there are any listeners for an event type.
   *
   * @param type Event type
   * @returns True if there are listeners
   */
  hasListeners<K extends keyof EventMap>(type: K): boolean {
    return this.listenerCount(type) > 0;
  }

  /**
   * Get all event types that have listeners.
   *
   * @returns Array of event types
   */
  eventTypes(): (keyof EventMap)[] {
    const types = new Set<keyof EventMap>();
    for (const type of this.listeners.keys()) {
      types.add(type);
    }
    for (const type of this.onceListeners.keys()) {
      types.add(type);
    }
    return Array.from(types);
  }
}

/**
 * Create a new event emitter instance.
 *
 * @returns A new EventEmitter
 */
export function createEventEmitter(): EventEmitter {
  return new EventEmitter();
}

/**
 * Create a timestamp for events.
 *
 * @returns Current time in milliseconds (performance.now)
 */
export function createTimestamp(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

/**
 * Helper to create event objects with timestamp.
 */
export const Events = {
  /**
   * Create a node click event.
   */
  nodeClick(
    nodeId: number,
    position: { x: number; y: number },
    originalEvent: PointerEvent
  ): EventMap["node:click"] {
    return {
      type: "node:click",
      timestamp: createTimestamp(),
      nodeId,
      position,
      originalEvent,
    };
  },

  /**
   * Create a node double-click event.
   */
  nodeDoubleClick(
    nodeId: number,
    position: { x: number; y: number },
    originalEvent: PointerEvent
  ): EventMap["node:dblclick"] {
    return {
      type: "node:dblclick",
      timestamp: createTimestamp(),
      nodeId,
      position,
      originalEvent,
    };
  },

  /**
   * Create a node hover enter event.
   */
  nodeHoverEnter(
    nodeId: number,
    position: { x: number; y: number }
  ): EventMap["node:hoverenter"] {
    return {
      type: "node:hoverenter",
      timestamp: createTimestamp(),
      nodeId,
      position,
    };
  },

  /**
   * Create a node hover leave event.
   */
  nodeHoverLeave(nodeId: number): EventMap["node:hoverleave"] {
    return {
      type: "node:hoverleave",
      timestamp: createTimestamp(),
      nodeId,
    };
  },

  /**
   * Create a node drag start event.
   */
  nodeDragStart(
    nodeId: number,
    position: { x: number; y: number }
  ): EventMap["node:dragstart"] {
    return {
      type: "node:dragstart",
      timestamp: createTimestamp(),
      nodeId,
      position,
    };
  },

  /**
   * Create a node drag move event.
   */
  nodeDragMove(
    nodeId: number,
    position: { x: number; y: number },
    delta: { x: number; y: number }
  ): EventMap["node:dragmove"] {
    return {
      type: "node:dragmove",
      timestamp: createTimestamp(),
      nodeId,
      position,
      delta,
    };
  },

  /**
   * Create a node drag end event.
   */
  nodeDragEnd(
    nodeId: number,
    position: { x: number; y: number }
  ): EventMap["node:dragend"] {
    return {
      type: "node:dragend",
      timestamp: createTimestamp(),
      nodeId,
      position,
    };
  },

  /**
   * Create an edge click event.
   */
  edgeClick(
    edgeId: number,
    position: { x: number; y: number },
    originalEvent: PointerEvent
  ): EventMap["edge:click"] {
    return {
      type: "edge:click",
      timestamp: createTimestamp(),
      edgeId,
      position,
      originalEvent,
    };
  },

  /**
   * Create an edge hover enter event.
   */
  edgeHoverEnter(
    edgeId: number,
    position: { x: number; y: number }
  ): EventMap["edge:hoverenter"] {
    return {
      type: "edge:hoverenter",
      timestamp: createTimestamp(),
      edgeId,
      position,
    };
  },

  /**
   * Create an edge hover leave event.
   */
  edgeHoverLeave(edgeId: number): EventMap["edge:hoverleave"] {
    return {
      type: "edge:hoverleave",
      timestamp: createTimestamp(),
      edgeId,
    };
  },

  /**
   * Create a viewport change event.
   */
  viewportChange(
    viewport: EventMap["viewport:change"]["viewport"]
  ): EventMap["viewport:change"] {
    return {
      type: "viewport:change",
      timestamp: createTimestamp(),
      viewport,
    };
  },

  /**
   * Create a simulation tick event.
   */
  simulationTick(alpha: number, iteration: number): EventMap["simulation:tick"] {
    return {
      type: "simulation:tick",
      timestamp: createTimestamp(),
      alpha,
      iteration,
    };
  },

  /**
   * Create a simulation end event.
   */
  simulationEnd(iterations: number): EventMap["simulation:end"] {
    return {
      type: "simulation:end",
      timestamp: createTimestamp(),
      iterations,
    };
  },

  /**
   * Create a selection change event.
   */
  selectionChange(
    selectedNodes: readonly number[],
    selectedEdges: readonly number[]
  ): EventMap["selection:change"] {
    return {
      type: "selection:change",
      timestamp: createTimestamp(),
      selectedNodes,
      selectedEdges,
    };
  },

  /**
   * Create a background click event.
   */
  backgroundClick(
    position: { x: number; y: number },
    originalEvent: PointerEvent
  ): EventMap["background:click"] {
    return {
      type: "background:click",
      timestamp: createTimestamp(),
      position,
      originalEvent,
    };
  },
};
