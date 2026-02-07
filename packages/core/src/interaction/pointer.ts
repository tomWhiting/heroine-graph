/**
 * Pointer Event Handling
 *
 * Unified handling of mouse and touch events for graph interaction.
 * Translates DOM events into graph-space coordinates and interaction intents.
 *
 * @module
 */

import type { Vec2 } from "../types.ts";
import type { Viewport } from "../viewport/viewport.ts";

/**
 * Pointer event types relevant to graph interaction.
 */
export type PointerEventType =
  | "pointerdown"
  | "pointermove"
  | "pointerup"
  | "pointercancel"
  | "wheel";

/**
 * Normalized pointer event data.
 */
export interface NormalizedPointerEvent {
  /** Event type */
  readonly type: PointerEventType;
  /** Position in screen/canvas coordinates */
  readonly screenPosition: Vec2;
  /** Position in graph coordinates */
  readonly graphPosition: Vec2;
  /** Pointer ID for multi-touch tracking */
  readonly pointerId: number;
  /** Whether this is a primary pointer (left mouse / first touch) */
  readonly isPrimary: boolean;
  /** Mouse button (0 = left, 1 = middle, 2 = right) */
  readonly button: number;
  /** Modifier keys */
  readonly modifiers: {
    readonly shift: boolean;
    readonly ctrl: boolean;
    readonly alt: boolean;
    readonly meta: boolean;
  };
  /** Wheel delta for scroll events */
  readonly wheelDelta?: Vec2;
  /** Original DOM event */
  readonly originalEvent: PointerEvent | WheelEvent;
  /** Timestamp */
  readonly timestamp: number;
}

/**
 * Callback for pointer events.
 */
export type PointerEventCallback = (event: NormalizedPointerEvent) => void;

/**
 * Pointer event manager configuration.
 */
export interface PointerManagerConfig {
  /** Canvas element to attach listeners to */
  canvas: HTMLCanvasElement;
  /** Viewport for coordinate transforms */
  viewport: Viewport;
  /** Whether to prevent default browser behavior */
  preventDefault?: boolean;
  /** Whether to stop event propagation */
  stopPropagation?: boolean;
}

/**
 * Manages pointer events for a canvas element.
 */
export class PointerManager {
  private readonly canvas: HTMLCanvasElement;
  private readonly viewport: Viewport;
  private readonly config: Required<Omit<PointerManagerConfig, "canvas" | "viewport">>;

  private callbacks: Map<PointerEventType, Set<PointerEventCallback>> = new Map();
  private boundHandlers: Map<string, EventListener> = new Map();
  private activePointers: Map<number, Vec2> = new Map();

  constructor(config: PointerManagerConfig) {
    this.canvas = config.canvas;
    this.viewport = config.viewport;
    this.config = {
      preventDefault: config.preventDefault ?? true,
      stopPropagation: config.stopPropagation ?? false,
    };

    this.setupEventListeners();
  }

  /**
   * Subscribe to a pointer event type.
   */
  on(type: PointerEventType, callback: PointerEventCallback): void {
    if (!this.callbacks.has(type)) {
      this.callbacks.set(type, new Set());
    }
    this.callbacks.get(type)!.add(callback);
  }

  /**
   * Unsubscribe from a pointer event type.
   */
  off(type: PointerEventType, callback: PointerEventCallback): void {
    this.callbacks.get(type)?.delete(callback);
  }

  /**
   * Get currently active pointers.
   */
  getActivePointers(): Map<number, Vec2> {
    return new Map(this.activePointers);
  }

  /**
   * Check if a pointer is currently active.
   */
  isPointerActive(pointerId: number): boolean {
    return this.activePointers.has(pointerId);
  }

  /**
   * Dispose of the pointer manager.
   */
  dispose(): void {
    // Remove all event listeners
    for (const [eventName, handler] of this.boundHandlers) {
      this.canvas.removeEventListener(eventName, handler);
    }
    this.boundHandlers.clear();
    this.callbacks.clear();
    this.activePointers.clear();
  }

  private setupEventListeners(): void {
    const addListener = (
      eventName: string,
      handler: (e: Event) => void,
      options?: AddEventListenerOptions,
    ): void => {
      const boundHandler = handler.bind(this);
      this.canvas.addEventListener(eventName, boundHandler, options);
      this.boundHandlers.set(eventName, boundHandler);
    };

    addListener("pointerdown", this.handlePointerDown);
    addListener("pointermove", this.handlePointerMove);
    addListener("pointerup", this.handlePointerUp);
    addListener("pointercancel", this.handlePointerCancel);
    addListener("pointerleave", this.handlePointerLeave);
    // Wheel events need passive: false to allow preventDefault() for zoom handling
    addListener("wheel", this.handleWheel, { passive: false });

    // Prevent context menu on right-click
    const contextMenuHandler = (e: Event) => e.preventDefault();
    this.canvas.addEventListener("contextmenu", contextMenuHandler);
    this.boundHandlers.set("contextmenu", contextMenuHandler);
  }

  private handlePointerDown = (e: Event): void => {
    const event = e as PointerEvent;
    this.processEvent(event);

    const screenPos = this.getScreenPosition(event);
    this.activePointers.set(event.pointerId, screenPos);

    // Capture pointer for reliable tracking outside canvas
    this.canvas.setPointerCapture(event.pointerId);

    this.emit("pointerdown", this.normalizePointerEvent(event, "pointerdown"));
  };

  private handlePointerMove = (e: Event): void => {
    const event = e as PointerEvent;
    this.processEvent(event);

    if (this.activePointers.has(event.pointerId)) {
      const screenPos = this.getScreenPosition(event);
      this.activePointers.set(event.pointerId, screenPos);
    }

    this.emit("pointermove", this.normalizePointerEvent(event, "pointermove"));
  };

  private handlePointerUp = (e: Event): void => {
    const event = e as PointerEvent;
    this.processEvent(event);

    this.activePointers.delete(event.pointerId);

    // Release pointer capture
    if (this.canvas.hasPointerCapture(event.pointerId)) {
      this.canvas.releasePointerCapture(event.pointerId);
    }

    this.emit("pointerup", this.normalizePointerEvent(event, "pointerup"));
  };

  private handlePointerCancel = (e: Event): void => {
    const event = e as PointerEvent;
    this.processEvent(event);

    this.activePointers.delete(event.pointerId);

    this.emit(
      "pointercancel",
      this.normalizePointerEvent(event, "pointercancel"),
    );
  };

  private handlePointerLeave = (e: Event): void => {
    const event = e as PointerEvent;
    // Only clear if not captured (dragging outside canvas)
    if (!this.canvas.hasPointerCapture(event.pointerId)) {
      this.activePointers.delete(event.pointerId);
    }
  };

  private handleWheel = (e: Event): void => {
    const event = e as WheelEvent;
    this.processEvent(event);

    this.emit("wheel", this.normalizeWheelEvent(event));
  };

  private processEvent(event: Event): void {
    if (this.config.preventDefault) {
      event.preventDefault();
    }
    if (this.config.stopPropagation) {
      event.stopPropagation();
    }
  }

  private getScreenPosition(event: PointerEvent | WheelEvent): Vec2 {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  }

  private normalizePointerEvent(
    event: PointerEvent,
    type: PointerEventType,
  ): NormalizedPointerEvent {
    const screenPosition = this.getScreenPosition(event);
    const graphPosition = this.viewport.screenToGraph(
      screenPosition.x,
      screenPosition.y,
    );

    return {
      type,
      screenPosition,
      graphPosition,
      pointerId: event.pointerId,
      isPrimary: event.isPrimary,
      button: event.button,
      modifiers: {
        shift: event.shiftKey,
        ctrl: event.ctrlKey,
        alt: event.altKey,
        meta: event.metaKey,
      },
      originalEvent: event,
      timestamp: event.timeStamp,
    };
  }

  private normalizeWheelEvent(event: WheelEvent): NormalizedPointerEvent {
    const screenPosition = this.getScreenPosition(event);
    const graphPosition = this.viewport.screenToGraph(
      screenPosition.x,
      screenPosition.y,
    );

    return {
      type: "wheel",
      screenPosition,
      graphPosition,
      pointerId: 0,
      isPrimary: true,
      button: 0,
      modifiers: {
        shift: event.shiftKey,
        ctrl: event.ctrlKey,
        alt: event.altKey,
        meta: event.metaKey,
      },
      wheelDelta: {
        x: event.deltaX,
        y: event.deltaY,
      },
      originalEvent: event,
      timestamp: event.timeStamp,
    };
  }

  private emit(type: PointerEventType, event: NormalizedPointerEvent): void {
    const callbacks = this.callbacks.get(type);
    if (callbacks) {
      for (const callback of callbacks) {
        try {
          callback(event);
        } catch (error) {
          console.error(`[PointerManager] Error in ${type} callback:`, error);
        }
      }
    }
  }
}

/**
 * Create a pointer manager for a canvas.
 */
export function createPointerManager(
  config: PointerManagerConfig,
): PointerManager {
  return new PointerManager(config);
}
