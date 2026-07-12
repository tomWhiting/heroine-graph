/**
 * Shared Layer Types
 *
 * The `Layer` interface lives here — free of any GPU pipeline or shader
 * imports — so CPU-side modules (LayerManager, tests) can import it
 * without dragging `.wgsl` imports into their module graph.
 *
 * @module
 */

/**
 * Layer interface for the layer system
 */
export interface Layer {
  /** Unique layer ID */
  readonly id: string;
  /** Layer type */
  readonly type: string;
  /** Whether layer is enabled */
  enabled: boolean;
  /** Render order (higher = on top) */
  order: number;
  /** Render the layer */
  render(encoder: GPUCommandEncoder, targetView: GPUTextureView): void;
  /** Resize layer resources */
  resize(width: number, height: number): void;
  /** Destroy layer resources */
  destroy(): void;
}
