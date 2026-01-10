/**
 * Layer Manager
 *
 * Manages multiple visualization layers, their visibility, ordering,
 * and coordinated rendering.
 *
 * @module
 */

import type { Layer } from "./heatmap/layer.ts";

/**
 * Layer manager configuration
 */
export interface LayerManagerConfig {
  /** Maximum number of layers allowed */
  maxLayers?: number;
}

/**
 * Default layer manager configuration
 */
export const DEFAULT_LAYER_MANAGER_CONFIG: Required<LayerManagerConfig> = {
  maxLayers: 16,
};

/**
 * Layer registration info
 */
export interface LayerInfo {
  /** Layer ID */
  id: string;
  /** Layer type */
  type: string;
  /** Whether layer is enabled */
  enabled: boolean;
  /** Render order */
  order: number;
}

/**
 * Layer manager for coordinating visualization layers
 */
export class LayerManager {
  private layers: Map<string, Layer> = new Map();
  private config: Required<LayerManagerConfig>;
  private sortedLayers: Layer[] = [];
  private orderDirty = true;

  constructor(config: LayerManagerConfig = {}) {
    this.config = { ...DEFAULT_LAYER_MANAGER_CONFIG, ...config };
  }

  /**
   * Add a layer
   */
  addLayer(layer: Layer): void {
    if (this.layers.size >= this.config.maxLayers) {
      throw new Error(
        `Maximum layer count (${this.config.maxLayers}) exceeded`
      );
    }

    if (this.layers.has(layer.id)) {
      throw new Error(`Layer with ID "${layer.id}" already exists`);
    }

    this.layers.set(layer.id, layer);
    this.orderDirty = true;
  }

  /**
   * Remove a layer by ID
   */
  removeLayer(id: string): boolean {
    const layer = this.layers.get(id);
    if (!layer) return false;

    layer.destroy();
    this.layers.delete(id);
    this.orderDirty = true;
    return true;
  }

  /**
   * Get a layer by ID
   */
  getLayer<T extends Layer = Layer>(id: string): T | undefined {
    return this.layers.get(id) as T | undefined;
  }

  /**
   * Check if a layer exists
   */
  hasLayer(id: string): boolean {
    return this.layers.has(id);
  }

  /**
   * Get all layer IDs
   */
  getLayerIds(): string[] {
    return Array.from(this.layers.keys());
  }

  /**
   * Get layer info for all layers
   */
  getLayerInfo(): LayerInfo[] {
    return Array.from(this.layers.values()).map((layer) => ({
      id: layer.id,
      type: layer.type,
      enabled: layer.enabled,
      order: layer.order,
    }));
  }

  /**
   * Enable a layer
   */
  enableLayer(id: string): boolean {
    const layer = this.layers.get(id);
    if (!layer) return false;

    layer.enabled = true;
    return true;
  }

  /**
   * Disable a layer
   */
  disableLayer(id: string): boolean {
    const layer = this.layers.get(id);
    if (!layer) return false;

    layer.enabled = false;
    return true;
  }

  /**
   * Toggle a layer's visibility
   */
  toggleLayer(id: string): boolean {
    const layer = this.layers.get(id);
    if (!layer) return false;

    layer.enabled = !layer.enabled;
    return layer.enabled;
  }

  /**
   * Set layer visibility
   */
  setLayerVisible(id: string, visible: boolean): boolean {
    const layer = this.layers.get(id);
    if (!layer) return false;

    layer.enabled = visible;
    return true;
  }

  /**
   * Check if a layer is visible
   */
  isLayerVisible(id: string): boolean {
    const layer = this.layers.get(id);
    return layer?.enabled ?? false;
  }

  /**
   * Set layer render order
   */
  setLayerOrder(id: string, order: number): boolean {
    const layer = this.layers.get(id);
    if (!layer) return false;

    layer.order = order;
    this.orderDirty = true;
    return true;
  }

  /**
   * Get layers sorted by render order
   */
  getSortedLayers(): Layer[] {
    if (this.orderDirty) {
      this.sortedLayers = Array.from(this.layers.values())
        .filter((layer) => layer.enabled)
        .sort((a, b) => a.order - b.order);
      this.orderDirty = false;
    }
    return this.sortedLayers;
  }

  /**
   * Render all enabled layers in order
   * @param encoder - GPU command encoder
   * @param targetView - Target texture view
   * @param skipLayers - Optional array of layer IDs to skip
   */
  render(encoder: GPUCommandEncoder, targetView: GPUTextureView, skipLayers?: string[]): void {
    const layers = this.getSortedLayers();
    const skipSet = skipLayers ? new Set(skipLayers) : null;

    for (const layer of layers) {
      if (skipSet && skipSet.has(layer.id)) {
        continue;
      }
      layer.render(encoder, targetView);
    }
  }

  /**
   * Resize all layers
   */
  resize(width: number, height: number): void {
    for (const layer of this.layers.values()) {
      layer.resize(width, height);
    }
  }

  /**
   * Destroy all layers
   */
  destroy(): void {
    for (const layer of this.layers.values()) {
      layer.destroy();
    }
    this.layers.clear();
    this.sortedLayers = [];
  }

  /**
   * Get layer count
   */
  get count(): number {
    return this.layers.size;
  }

  /**
   * Get enabled layer count
   */
  get enabledCount(): number {
    let count = 0;
    for (const layer of this.layers.values()) {
      if (layer.enabled) count++;
    }
    return count;
  }
}

/**
 * Create a layer manager
 */
export function createLayerManager(
  config?: LayerManagerConfig
): LayerManager {
  return new LayerManager(config);
}
