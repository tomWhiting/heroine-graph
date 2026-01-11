/**
 * Value Stream Types
 *
 * Types for the value stream system that maps numeric values to visual heat colors.
 *
 * @module
 */

/**
 * Color stop for interpolated color scales
 */
export interface ColorStop {
  /** Position in the scale (0-1) */
  position: number;
  /** RGBA color values (0-1 range) */
  color: [number, number, number, number];
}

/**
 * Color scale configuration for a value stream
 */
export interface ValueColorScale {
  /** Domain: [min, max] value range */
  domain: [number, number];
  /** Color stops for interpolation */
  stops: ColorStop[];
}

/**
 * Blend mode for combining multiple streams
 */
export type BlendMode = "additive" | "multiply" | "max" | "replace";

/**
 * Value stream definition
 */
export interface ValueStreamConfig {
  /** Unique stream identifier */
  id: string;
  /** Human-readable name */
  name?: string;
  /** Color scale configuration */
  colorScale: ValueColorScale;
  /** Blend mode when combining with other streams */
  blendMode?: BlendMode;
  /** Whether this stream is active */
  enabled?: boolean;
  /** Opacity multiplier (0-1) */
  opacity?: number;
}

/**
 * Data point for a value stream
 */
export interface StreamDataPoint {
  /** Node index */
  nodeIndex: number;
  /** Numeric value */
  value: number;
}

/**
 * Bulk data for a value stream (Float32Array for performance)
 */
export interface StreamBulkData {
  /** Node indices (Int32Array or number[]) */
  indices: Int32Array | number[];
  /** Values for each node (same length as indices) */
  values: Float32Array | number[];
}

/**
 * Information about a defined stream
 */
export interface StreamInfo {
  id: string;
  name: string;
  enabled: boolean;
  blendMode: BlendMode;
  domain: [number, number];
  nodeCount: number;
}

/**
 * Preset color scales for common use cases
 */
export const VALUE_COLOR_PRESETS: Record<string, ColorStop[]> = {
  /** Red for errors/danger */
  error: [
    { position: 0, color: [0, 0, 0, 0] },
    { position: 0.5, color: [0.8, 0.2, 0.1, 0.5] },
    { position: 1, color: [1, 0.1, 0.05, 1] },
  ],
  /** Yellow for warnings */
  warning: [
    { position: 0, color: [0, 0, 0, 0] },
    { position: 0.5, color: [0.9, 0.7, 0.1, 0.5] },
    { position: 1, color: [1, 0.9, 0.2, 1] },
  ],
  /** Green for success/health */
  success: [
    { position: 0, color: [0, 0, 0, 0] },
    { position: 0.5, color: [0.2, 0.7, 0.3, 0.5] },
    { position: 1, color: [0.3, 0.9, 0.4, 1] },
  ],
  /** Blue for info/activity */
  activity: [
    { position: 0, color: [0, 0, 0, 0] },
    { position: 0.5, color: [0.2, 0.5, 0.9, 0.5] },
    { position: 1, color: [0.3, 0.7, 1, 1] },
  ],
  /** Purple for coverage/importance */
  importance: [
    { position: 0, color: [0, 0, 0, 0] },
    { position: 0.5, color: [0.5, 0.3, 0.8, 0.5] },
    { position: 1, color: [0.7, 0.4, 1, 1] },
  ],
  /** Heat: cold (blue) to hot (red) */
  heat: [
    { position: 0, color: [0.2, 0.4, 0.8, 0.3] },
    { position: 0.33, color: [0.2, 0.8, 0.4, 0.5] },
    { position: 0.66, color: [0.9, 0.8, 0.2, 0.7] },
    { position: 1, color: [1, 0.2, 0.1, 1] },
  ],
};

/**
 * Create a color scale from a preset name
 */
export function createColorScaleFromPreset(
  preset: keyof typeof VALUE_COLOR_PRESETS,
  domain: [number, number] = [0, 1],
): ValueColorScale {
  const stops = VALUE_COLOR_PRESETS[preset];
  if (!stops) {
    throw new Error(`Unknown color preset: ${preset}`);
  }
  return { domain, stops: [...stops] };
}

/**
 * Create a simple two-color gradient scale
 */
export function createGradientScale(
  fromColor: [number, number, number, number],
  toColor: [number, number, number, number],
  domain: [number, number] = [0, 1],
): ValueColorScale {
  return {
    domain,
    stops: [
      { position: 0, color: fromColor },
      { position: 1, color: toColor },
    ],
  };
}
