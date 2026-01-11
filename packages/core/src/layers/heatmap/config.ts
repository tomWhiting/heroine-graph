/**
 * Heatmap Layer Configuration
 *
 * Configuration types and defaults for the heatmap visualization layer.
 *
 * @module
 */

import type { ColorScaleName } from "./colorscale.ts";

/**
 * Data source for heatmap intensity
 * - 'density': All nodes contribute equally (default)
 * - string: ID of a value stream - nodes contribute based on stream values
 */
export type HeatmapDataSource = "density" | string;

/**
 * Heatmap layer configuration
 */
export interface HeatmapConfig {
  /** Enable/disable the heatmap layer */
  enabled?: boolean;
  /** Splat radius in graph units */
  radius?: number;
  /** Base intensity per node (multiplied by per-node intensity when using stream) */
  intensity?: number;
  /** Minimum density for color mapping */
  minDensity?: number;
  /** Maximum density for color mapping (auto if undefined) */
  maxDensity?: number;
  /** Global opacity of heatmap overlay */
  opacity?: number;
  /** Color scale to use */
  colorScale?: ColorScaleName;
  /** Resolution scale for density texture (0.5 = half resolution) */
  resolutionScale?: number;
  /**
   * Data source for per-node intensity.
   * - 'density' (default): All nodes contribute equally (intensity = 1.0)
   * - streamId: Use values from a value stream (nodes with values contribute, others don't)
   */
  dataSource?: HeatmapDataSource;
}

/**
 * Default heatmap configuration
 */
export const DEFAULT_HEATMAP_CONFIG: Required<HeatmapConfig> = {
  enabled: false,
  radius: 50.0,
  intensity: 0.1,
  minDensity: 0.0,
  maxDensity: 1.0,
  opacity: 0.7,
  colorScale: "viridis",
  resolutionScale: 1.0,
  dataSource: "density",
};

/**
 * Merge user config with defaults
 */
export function mergeHeatmapConfig(
  config: HeatmapConfig = {},
): Required<HeatmapConfig> {
  return { ...DEFAULT_HEATMAP_CONFIG, ...config };
}

/**
 * Validate heatmap configuration
 */
export function validateHeatmapConfig(config: HeatmapConfig): string[] {
  const errors: string[] = [];

  if (config.radius !== undefined && config.radius <= 0) {
    errors.push("Radius must be positive");
  }

  if (config.intensity !== undefined && config.intensity < 0) {
    errors.push("Intensity must be non-negative");
  }

  if (config.opacity !== undefined && (config.opacity < 0 || config.opacity > 1)) {
    errors.push("Opacity must be between 0 and 1");
  }

  if (
    config.resolutionScale !== undefined &&
    (config.resolutionScale <= 0 || config.resolutionScale > 2)
  ) {
    errors.push("Resolution scale must be between 0 and 2");
  }

  return errors;
}
