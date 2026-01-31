/**
 * Contour Layer Configuration
 *
 * Configuration types and defaults for the contour visualization layer.
 * Contours render iso-lines at specified density thresholds.
 *
 * @module
 */

/**
 * Data source for contour layer
 * - 'density': Use uniform node density (default)
 * - string: ID of a value stream - contours follow stream value thresholds
 */
export type ContourDataSource = "density" | string;

/**
 * Contour layer configuration
 */
export interface ContourConfig {
  /** Enable/disable the contour layer */
  enabled?: boolean;
  /** Density thresholds for iso-lines (0.0-1.0) */
  thresholds?: number[];
  /** Line stroke width in pixels */
  strokeWidth?: number;
  /** Line stroke color (CSS color string) */
  strokeColor?: string;
  /** Global opacity of contour overlay */
  opacity?: number;
  /** Resolution scale for contour texture (0.5 = half resolution) */
  resolutionScale?: number;
  /** Whether to smooth the contour lines */
  smooth?: boolean;
  /**
   * Data source for contour values.
   * - 'density' (default): Contours based on uniform node density
   * - streamId: Contours based on stream value thresholds
   */
  dataSource?: ContourDataSource;
}

/**
 * Default contour configuration
 */
export const DEFAULT_CONTOUR_CONFIG: Required<ContourConfig> = {
  enabled: false,
  thresholds: [0.2, 0.4, 0.6, 0.8],
  strokeWidth: 1.5,
  strokeColor: "#333333",
  opacity: 0.8,
  resolutionScale: 1.0,
  smooth: true,
  dataSource: "density",
};

/**
 * Merge user config with defaults
 */
export function mergeContourConfig(
  config: ContourConfig = {},
): Required<ContourConfig> {
  return {
    ...DEFAULT_CONTOUR_CONFIG,
    ...config,
    // Deep copy thresholds array
    thresholds: config.thresholds ? [...config.thresholds] : [...DEFAULT_CONTOUR_CONFIG.thresholds],
  };
}

/**
 * Validate contour configuration
 */
export function validateContourConfig(config: ContourConfig): string[] {
  const errors: string[] = [];

  if (config.thresholds !== undefined) {
    if (config.thresholds.length === 0) {
      errors.push("At least one threshold is required");
    }
    for (const t of config.thresholds) {
      if (t < 0 || t > 1) {
        errors.push(`Threshold ${t} must be between 0 and 1`);
      }
    }
  }

  if (config.strokeWidth !== undefined && config.strokeWidth <= 0) {
    errors.push("Stroke width must be positive");
  }

  if (
    config.opacity !== undefined &&
    (config.opacity < 0 || config.opacity > 1)
  ) {
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

// Re-export parseColor from shared utilities for backwards compatibility
export { parseColorToRGBA as parseColor } from "../../utils/color.ts";
