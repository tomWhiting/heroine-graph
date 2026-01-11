/**
 * Value Stream
 *
 * A single value stream that maps numeric values to colors for visualization.
 *
 * @module
 */

import type {
  BlendMode,
  ColorStop,
  StreamBulkData,
  StreamDataPoint,
  StreamInfo,
  ValueColorScale,
  ValueStreamConfig,
} from "./types.ts";

/**
 * Default configuration values
 */
const DEFAULTS = {
  blendMode: "additive" as BlendMode,
  enabled: true,
  opacity: 1.0,
};

/**
 * ValueStream class
 *
 * Manages a single named value stream with color mapping.
 * Acts as a "dumb pipe" - values in, colors out.
 */
export class ValueStream {
  readonly id: string;
  readonly name: string;

  private colorScale: ValueColorScale;
  private blendMode: BlendMode;
  private enabled: boolean;
  private opacity: number;

  /** Sparse map of nodeIndex -> value */
  private values: Map<number, number> = new Map();

  /** Cached color output (invalidated on data change) */
  private colorCache: Map<number, [number, number, number, number]> | null = null;

  constructor(config: ValueStreamConfig) {
    this.id = config.id;
    this.name = config.name ?? config.id;
    this.colorScale = config.colorScale;
    this.blendMode = config.blendMode ?? DEFAULTS.blendMode;
    this.enabled = config.enabled ?? DEFAULTS.enabled;
    this.opacity = config.opacity ?? DEFAULTS.opacity;
  }

  /**
   * Set values for multiple nodes
   */
  setData(data: StreamDataPoint[]): void {
    for (const point of data) {
      this.values.set(point.nodeIndex, point.value);
    }
    this.colorCache = null;
  }

  /**
   * Set values from bulk data (more efficient for large updates)
   */
  setBulkData(data: StreamBulkData): void {
    const { indices, values } = data;
    const len = Math.min(indices.length, values.length);
    for (let i = 0; i < len; i++) {
      this.values.set(
        typeof indices[i] === "number" ? indices[i] : indices[i],
        typeof values[i] === "number" ? values[i] : values[i],
      );
    }
    this.colorCache = null;
  }

  /**
   * Set a single node's value
   */
  setValue(nodeIndex: number, value: number): void {
    this.values.set(nodeIndex, value);
    this.colorCache = null;
  }

  /**
   * Clear a node's value
   */
  clearValue(nodeIndex: number): void {
    this.values.delete(nodeIndex);
    this.colorCache = null;
  }

  /**
   * Clear all values
   */
  clear(): void {
    this.values.clear();
    this.colorCache = null;
  }

  /**
   * Get a node's raw value
   */
  getValue(nodeIndex: number): number | undefined {
    return this.values.get(nodeIndex);
  }

  /**
   * Check if a node has a value
   */
  hasValue(nodeIndex: number): boolean {
    return this.values.has(nodeIndex);
  }

  /**
   * Get all node indices with values
   */
  getNodeIndices(): number[] {
    return Array.from(this.values.keys());
  }

  /**
   * Get the color for a node based on its value
   * Returns [r, g, b, a] in 0-1 range, or null if no value set
   */
  getColor(nodeIndex: number): [number, number, number, number] | null {
    if (!this.enabled) return null;

    const value = this.values.get(nodeIndex);
    if (value === undefined) return null;

    // Check cache
    if (this.colorCache?.has(nodeIndex)) {
      return this.colorCache.get(nodeIndex)!;
    }

    // Compute color
    const color = this.interpolateColor(value);

    // Apply opacity
    color[3] *= this.opacity;

    // Cache result
    if (!this.colorCache) {
      this.colorCache = new Map();
    }
    this.colorCache.set(nodeIndex, color);

    return color;
  }

  /**
   * Get colors for all nodes with values
   * Returns Map<nodeIndex, [r, g, b, a]>
   */
  getAllColors(): Map<number, [number, number, number, number]> {
    if (!this.enabled) return new Map();

    const result = new Map<number, [number, number, number, number]>();
    for (const nodeIndex of this.values.keys()) {
      const color = this.getColor(nodeIndex);
      if (color) {
        result.set(nodeIndex, color);
      }
    }
    return result;
  }

  /**
   * Interpolate color based on value and color scale
   */
  private interpolateColor(value: number): [number, number, number, number] {
    const { domain, stops } = this.colorScale;

    // Normalize value to 0-1 based on domain
    const normalized = Math.max(0, Math.min(1, (value - domain[0]) / (domain[1] - domain[0])));

    // Find surrounding color stops
    let lower: ColorStop = stops[0];
    let upper: ColorStop = stops[stops.length - 1];

    for (let i = 0; i < stops.length - 1; i++) {
      if (normalized >= stops[i].position && normalized <= stops[i + 1].position) {
        lower = stops[i];
        upper = stops[i + 1];
        break;
      }
    }

    // Interpolate between stops
    const range = upper.position - lower.position;
    const t = range > 0 ? (normalized - lower.position) / range : 0;

    return [
      lower.color[0] + t * (upper.color[0] - lower.color[0]),
      lower.color[1] + t * (upper.color[1] - lower.color[1]),
      lower.color[2] + t * (upper.color[2] - lower.color[2]),
      lower.color[3] + t * (upper.color[3] - lower.color[3]),
    ];
  }

  /**
   * Update color scale
   */
  setColorScale(scale: ValueColorScale): void {
    this.colorScale = scale;
    this.colorCache = null;
  }

  /**
   * Get current color scale
   */
  getColorScale(): ValueColorScale {
    return this.colorScale;
  }

  /**
   * Set blend mode
   */
  setBlendMode(mode: BlendMode): void {
    this.blendMode = mode;
  }

  /**
   * Get blend mode
   */
  getBlendMode(): BlendMode {
    return this.blendMode;
  }

  /**
   * Enable the stream
   */
  enable(): void {
    this.enabled = true;
  }

  /**
   * Disable the stream
   */
  disable(): void {
    this.enabled = false;
    this.colorCache = null;
  }

  /**
   * Toggle stream enabled state
   */
  toggle(): boolean {
    this.enabled = !this.enabled;
    if (!this.enabled) {
      this.colorCache = null;
    }
    return this.enabled;
  }

  /**
   * Check if enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Set opacity
   */
  setOpacity(opacity: number): void {
    this.opacity = Math.max(0, Math.min(1, opacity));
    this.colorCache = null;
  }

  /**
   * Get opacity
   */
  getOpacity(): number {
    return this.opacity;
  }

  /**
   * Get stream info
   */
  getInfo(): StreamInfo {
    return {
      id: this.id,
      name: this.name,
      enabled: this.enabled,
      blendMode: this.blendMode,
      domain: this.colorScale.domain,
      nodeCount: this.values.size,
    };
  }

  /**
   * Number of nodes with values
   */
  get size(): number {
    return this.values.size;
  }
}
