/**
 * Metaball Layer Configuration
 *
 * Configuration types and defaults for the metaball visualization layer.
 * Metaballs render smooth blob-like shapes around node clusters.
 *
 * @module
 */

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
}

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
};

/**
 * Merge user config with defaults
 */
export function mergeMetaballConfig(
  config: MetaballConfig = {}
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

/**
 * Parse CSS color to RGBA values (0-1 range)
 */
export function parseMetaballColor(
  color: string
): [number, number, number, number] {
  // Handle hex colors
  if (color.startsWith("#")) {
    const hex = color.slice(1);
    if (hex.length === 3) {
      const r = parseInt(hex[0] + hex[0], 16) / 255;
      const g = parseInt(hex[1] + hex[1], 16) / 255;
      const b = parseInt(hex[2] + hex[2], 16) / 255;
      return [r, g, b, 1.0];
    } else if (hex.length === 6) {
      const r = parseInt(hex.slice(0, 2), 16) / 255;
      const g = parseInt(hex.slice(2, 4), 16) / 255;
      const b = parseInt(hex.slice(4, 6), 16) / 255;
      return [r, g, b, 1.0];
    } else if (hex.length === 8) {
      const r = parseInt(hex.slice(0, 2), 16) / 255;
      const g = parseInt(hex.slice(2, 4), 16) / 255;
      const b = parseInt(hex.slice(4, 6), 16) / 255;
      const a = parseInt(hex.slice(6, 8), 16) / 255;
      return [r, g, b, a];
    }
  }

  // Default to indigo
  return [0.388, 0.4, 0.945, 1.0];
}
