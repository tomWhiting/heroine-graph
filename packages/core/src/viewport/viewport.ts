/**
 * Viewport Management
 *
 * Handles pan, zoom, and coordinate transforms for the graph view.
 */

import type { BoundingBox, Vec2, ViewportState } from "../types.ts";
import { EventEmitter, Events } from "../events/emitter.ts";

/**
 * Configuration for the viewport.
 */
export interface ViewportConfig {
  /** Minimum zoom level */
  readonly minScale: number;
  /** Maximum zoom level */
  readonly maxScale: number;
  /** Zoom speed multiplier */
  readonly zoomSpeed: number;
  /** Animation duration in milliseconds */
  readonly animationDuration: number;
}

/**
 * Default viewport configuration.
 */
export const DEFAULT_VIEWPORT_CONFIG: ViewportConfig = {
  minScale: 0.01,
  maxScale: 100,
  zoomSpeed: 1.0,
  animationDuration: 300,
};

/**
 * Viewport class for managing pan/zoom state.
 */
export class Viewport {
  private x: number;
  private y: number;
  private scale: number;
  private width: number;
  private height: number;
  private readonly config: ViewportConfig;
  private readonly emitter: EventEmitter;

  // Animation state
  private animating: boolean;
  private animationFrame: number | null;

  constructor(
    width: number,
    height: number,
    config: Partial<ViewportConfig> = {},
    emitter?: EventEmitter,
  ) {
    this.x = 0;
    this.y = 0;
    this.scale = 1;
    this.width = width;
    this.height = height;
    this.config = { ...DEFAULT_VIEWPORT_CONFIG, ...config };
    this.emitter = emitter ?? new EventEmitter();
    this.animating = false;
    this.animationFrame = null;
  }

  /**
   * Get the current viewport state.
   */
  getState(): ViewportState {
    return {
      x: this.x,
      y: this.y,
      scale: this.scale,
      width: this.width,
      height: this.height,
      minScale: this.config.minScale,
      maxScale: this.config.maxScale,
    };
  }

  /**
   * Get the current viewport state (property accessor).
   */
  get state(): ViewportState {
    return this.getState();
  }

  /**
   * Pan the viewport by a delta in graph units.
   */
  pan(dx: number, dy: number): void {
    this.x += dx;
    this.y += dy;
    this.emitChange();
  }

  /**
   * Pan the viewport by a delta in screen pixels.
   * Converts screen pixels to graph units by dividing by scale.
   * Same sign convention as pan() - positive dx moves viewport right.
   */
  panScreen(dx: number, dy: number): void {
    this.x += dx / this.scale;
    this.y += dy / this.scale;
    this.emitChange();
  }

  /**
   * Pan to center on a position in graph units.
   */
  panTo(x: number, y: number, animate: boolean = false): void {
    if (animate) {
      this.animateTo(x, y, this.scale);
    } else {
      this.x = x;
      this.y = y;
      this.emitChange();
    }
  }

  /**
   * Zoom by a factor, optionally centered on a screen position.
   */
  zoom(factor: number, centerX?: number, centerY?: number): void {
    const newScale = this.clampScale(this.scale * factor);
    const actualFactor = newScale / this.scale;

    if (actualFactor === 1) return;

    // If no center provided, zoom from canvas center
    const cx = centerX ?? this.width / 2;
    const cy = centerY ?? this.height / 2;

    // Convert screen center to graph coordinates before zoom
    const graphX = this.screenToGraphX(cx);
    const graphY = this.screenToGraphY(cy);

    // Apply zoom
    this.scale = newScale;

    // Adjust pan so the point under the cursor stays fixed
    // After changing scale, the graph point would appear at newScreen position.
    // We need to adjust pan to bring it back to the original cursor position.
    const newScreenX = this.graphToScreenX(graphX);
    const newScreenY = this.graphToScreenY(graphY);

    // The adjustment should move the viewport so the point returns to cursor position
    // Subtracting because we need to compensate for where the point moved to
    this.x -= (cx - newScreenX) / this.scale;
    this.y -= (cy - newScreenY) / this.scale;

    this.emitChange();
  }

  /**
   * Set an absolute zoom level.
   */
  zoomTo(scale: number, animate: boolean = false): void {
    const newScale = this.clampScale(scale);
    if (animate) {
      this.animateTo(this.x, this.y, newScale);
    } else {
      this.scale = newScale;
      this.emitChange();
    }
  }

  /**
   * Set the scale (alias for zoomTo).
   */
  setScale(scale: number): void {
    this.zoomTo(scale, false);
  }

  /**
   * Set the center position.
   */
  setCenter(x: number, y: number): void {
    this.panTo(x, y, false);
  }

  /**
   * Fit the viewport to show all content.
   */
  fitToView(bounds: BoundingBox, padding: number = 50, animate: boolean = false): void {
    const { minX, minY, maxX, maxY } = bounds;
    const contentWidth = maxX - minX;
    const contentHeight = maxY - minY;

    if (contentWidth <= 0 || contentHeight <= 0) return;

    const paddedWidth = this.width - padding * 2;
    const paddedHeight = this.height - padding * 2;

    const scaleX = paddedWidth / contentWidth;
    const scaleY = paddedHeight / contentHeight;
    const newScale = this.clampScale(Math.min(scaleX, scaleY));

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    if (animate) {
      this.animateTo(centerX, centerY, newScale);
    } else {
      this.x = centerX;
      this.y = centerY;
      this.scale = newScale;
      this.emitChange();
    }
  }

  /**
   * Resize the viewport.
   */
  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.emitChange();
  }

  /**
   * Convert screen coordinates to graph coordinates.
   */
  screenToGraph(screenX: number, screenY: number): Vec2 {
    return {
      x: this.screenToGraphX(screenX),
      y: this.screenToGraphY(screenY),
    };
  }

  /**
   * Convert graph coordinates to screen coordinates.
   */
  graphToScreen(graphX: number, graphY: number): Vec2 {
    return {
      x: this.graphToScreenX(graphX),
      y: this.graphToScreenY(graphY),
    };
  }

  /**
   * Convert screen X to graph X.
   */
  screenToGraphX(screenX: number): number {
    return (screenX - this.width / 2) / this.scale + this.x;
  }

  /**
   * Convert screen Y to graph Y.
   */
  screenToGraphY(screenY: number): number {
    return (screenY - this.height / 2) / this.scale + this.y;
  }

  /**
   * Convert graph X to screen X.
   */
  graphToScreenX(graphX: number): number {
    return (graphX - this.x) * this.scale + this.width / 2;
  }

  /**
   * Convert graph Y to screen Y.
   */
  graphToScreenY(graphY: number): number {
    return (graphY - this.y) * this.scale + this.height / 2;
  }

  /**
   * Get the visible bounds in graph coordinates.
   */
  getVisibleBounds(): BoundingBox {
    return {
      minX: this.screenToGraphX(0),
      minY: this.screenToGraphY(0),
      maxX: this.screenToGraphX(this.width),
      maxY: this.screenToGraphY(this.height),
    };
  }

  /**
   * Check if a graph point is visible.
   */
  isPointVisible(x: number, y: number, margin: number = 0): boolean {
    const screenX = this.graphToScreenX(x);
    const screenY = this.graphToScreenY(y);
    return (
      screenX >= -margin &&
      screenX <= this.width + margin &&
      screenY >= -margin &&
      screenY <= this.height + margin
    );
  }

  /**
   * Get the scale factor for a given zoom level.
   */
  getScaleAtZoom(zoom: number): number {
    return 2 ** zoom;
  }

  /**
   * Get the zoom level for a given scale.
   */
  getZoomAtScale(scale: number): number {
    return Math.log2(scale);
  }

  /**
   * Stop any running animation.
   */
  stopAnimation(): void {
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    this.animating = false;
  }

  /**
   * Dispose of the viewport.
   */
  dispose(): void {
    this.stopAnimation();
  }

  /**
   * Check if currently animating.
   */
  isAnimating(): boolean {
    return this.animating;
  }

  /**
   * Clamp scale to valid range.
   */
  private clampScale(scale: number): number {
    return Math.max(this.config.minScale, Math.min(this.config.maxScale, scale));
  }

  /**
   * Emit a viewport change event.
   */
  private emitChange(): void {
    this.emitter.emit(Events.viewportChange(this.getState()));
  }

  /**
   * Animate to a target position and scale.
   */
  private animateTo(targetX: number, targetY: number, targetScale: number): void {
    this.stopAnimation();
    this.animating = true;

    const startX = this.x;
    const startY = this.y;
    const startScale = this.scale;
    const startTime = performance.now();
    const duration = this.config.animationDuration;

    const animate = (currentTime: number): void => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Ease-out cubic
      const t = 1 - (1 - progress) ** 3;

      this.x = startX + (targetX - startX) * t;
      this.y = startY + (targetY - startY) * t;
      this.scale = startScale + (targetScale - startScale) * t;

      this.emitChange();

      if (progress < 1) {
        this.animationFrame = requestAnimationFrame(animate);
      } else {
        this.animating = false;
        this.animationFrame = null;
      }
    };

    this.animationFrame = requestAnimationFrame(animate);
  }
}

/**
 * Options for createViewport when using canvas element.
 */
export interface CreateViewportOptions extends Partial<ViewportConfig> {
  /** Callback when viewport changes */
  onViewportChange?: (state: ViewportState) => void;
}

/**
 * Create a new viewport instance.
 *
 * Can be called with:
 * - createViewport(canvas, options) - creates viewport from canvas dimensions
 * - createViewport(width, height, config, emitter) - creates with explicit dimensions
 */
export function createViewport(
  canvasOrWidth: HTMLCanvasElement | number,
  optionsOrHeight?: CreateViewportOptions | number,
  config?: Partial<ViewportConfig>,
  emitter?: EventEmitter,
): Viewport {
  if (typeof canvasOrWidth === "number") {
    // Legacy signature: (width, height, config, emitter)
    return new Viewport(canvasOrWidth, optionsOrHeight as number, config, emitter);
  }

  // New signature: (canvas, options)
  const canvas = canvasOrWidth;
  const options = optionsOrHeight as CreateViewportOptions | undefined;
  // Use CSS dimensions (clientWidth/clientHeight) to match pointer event coordinates
  // from getBoundingClientRect(). canvas.width/height may include devicePixelRatio
  // which causes coordinate mismatch on high-DPI displays.
  const width = canvas.clientWidth || canvas.width || 800;
  const height = canvas.clientHeight || canvas.height || 600;

  // Create emitter if onViewportChange callback is provided
  let viewportEmitter: EventEmitter | undefined;
  if (options?.onViewportChange) {
    viewportEmitter = new EventEmitter();
    viewportEmitter.on("viewport:change", (event: unknown) => {
      const viewportEvent = event as { state: ViewportState };
      options.onViewportChange!(viewportEvent.state);
    });
  }

  return new Viewport(width, height, options, viewportEmitter);
}
