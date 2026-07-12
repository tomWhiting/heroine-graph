/**
 * Metaball Layer Configuration
 *
 * Configuration types and defaults for the metaball visualization layer.
 * Metaballs render smooth blob-like shapes around node clusters.
 *
 * @module
 */

/**
 * Data source for metaball intensity
 * - 'density': All nodes contribute equally (default)
 * - string: ID of a value stream - nodes contribute based on stream values
 */
export type MetaballDataSource = "density" | string;

/**
 * Metaball layer configuration
 */
export interface MetaballConfig {
  /** Enable/disable the metaball layer */
  enabled?: boolean;
  /** SDF threshold for boundary (0-1) */
  threshold?: number;
  /** Smooth union blend radius in pixels */
  blendRadius?: number;
  /** Fill color (CSS color string) */
  fillColor?: string;
  /** Global opacity of metaball overlay */
  opacity?: number;
  /** Node influence radius in graph units */
  nodeRadius?: number;
  /** Whether to show outline only */
  outlineOnly?: boolean;
  /** Outline width in pixels (when outlineOnly is true) */
  outlineWidth?: number;
  /**
   * Data source for per-node intensity.
   * - 'density' (default): All nodes contribute equally
   * - streamId: Use values from a value stream (nodes with higher values = larger blobs)
   */
  dataSource?: MetaballDataSource;
}

/**
 * Maximum node count the metaball layer will render.
 *
 * The SDF fragment shader (sdf.frag.wgsl) iterates every node for every
 * pixel of a fullscreen triangle — O(nodeCount * pixels) — with only an
 * in-loop AABB early-out. Beyond roughly this many nodes a fullscreen
 * pass takes long enough to trigger GPU watchdog timeouts (TDR/device
 * loss) on common hardware; at the project's 35K-node target it is a
 * guaranteed device kill. Until spatial acceleration (tiling/BVH) lands,
 * the layer skips rendering above this bound and logs a warning.
 */
export const MAX_METABALL_NODES = 2000;

/**
 * Default metaball configuration
 */
export const DEFAULT_METABALL_CONFIG: Required<MetaballConfig> = {
  enabled: false,
  threshold: 0.5,
  blendRadius: 30.0,
  fillColor: "#6366f1",
  opacity: 0.3,
  nodeRadius: 50.0,
  outlineOnly: false,
  outlineWidth: 2.0,
  dataSource: "density",
};

/**
 * Merge user config with defaults
 */
export function mergeMetaballConfig(
  config: MetaballConfig = {},
): Required<MetaballConfig> {
  return { ...DEFAULT_METABALL_CONFIG, ...config };
}

/**
 * Validate metaball configuration
 */
export function validateMetaballConfig(config: MetaballConfig): string[] {
  const errors: string[] = [];

  if (
    config.threshold !== undefined &&
    (config.threshold < 0 || config.threshold > 1)
  ) {
    errors.push("Threshold must be between 0 and 1");
  }

  if (config.blendRadius !== undefined && config.blendRadius < 0) {
    errors.push("Blend radius must be non-negative");
  }

  if (
    config.opacity !== undefined &&
    (config.opacity < 0 || config.opacity > 1)
  ) {
    errors.push("Opacity must be between 0 and 1");
  }

  if (config.nodeRadius !== undefined && config.nodeRadius <= 0) {
    errors.push("Node radius must be positive");
  }

  if (config.outlineWidth !== undefined && config.outlineWidth < 0) {
    errors.push("Outline width must be non-negative");
  }

  return errors;
}

// Re-export parseColor from shared utilities for backwards compatibility
export { parseColorToRGBA as parseMetaballColor } from "../../utils/color.ts";
