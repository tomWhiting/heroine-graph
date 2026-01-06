/**
 * Heatmap Color Scales
 *
 * Provides predefined color gradients for heatmap visualization.
 * Creates 1D GPU textures for efficient shader-based color lookup.
 *
 * @module
 */

import type { GPUContext } from "../../webgpu/context.ts";

/**
 * RGBA color as 4 floats [0-1]
 */
export type ColorRGBA = [number, number, number, number];

/**
 * Color stop in a gradient
 */
export interface ColorStop {
  /** Position in gradient [0-1] */
  position: number;
  /** RGBA color */
  color: ColorRGBA;
}

/**
 * Predefined color scale names
 */
export type ColorScaleName =
  | "viridis"
  | "plasma"
  | "inferno"
  | "magma"
  | "turbo"
  | "hot"
  | "cool"
  | "blues"
  | "greens"
  | "reds"
  | "grayscale";

/**
 * Color scale texture resources
 */
export interface ColorScaleTexture {
  /** The 1D texture containing the color gradient */
  texture: GPUTexture;
  /** Texture view for shader binding */
  view: GPUTextureView;
  /** Sampler for interpolated lookups */
  sampler: GPUSampler;
  /** Name of the color scale */
  name: ColorScaleName | "custom";
  /** Destroy resources */
  destroy: () => void;
}

/**
 * Predefined color gradients (based on matplotlib/d3 color scales)
 */
export const COLOR_SCALES: Record<ColorScaleName, ColorStop[]> = {
  viridis: [
    { position: 0.0, color: [0.267, 0.004, 0.329, 1.0] },
    { position: 0.25, color: [0.282, 0.140, 0.458, 1.0] },
    { position: 0.5, color: [0.127, 0.566, 0.551, 1.0] },
    { position: 0.75, color: [0.369, 0.789, 0.383, 1.0] },
    { position: 1.0, color: [0.993, 0.906, 0.144, 1.0] },
  ],
  plasma: [
    { position: 0.0, color: [0.050, 0.030, 0.528, 1.0] },
    { position: 0.25, color: [0.416, 0.090, 0.643, 1.0] },
    { position: 0.5, color: [0.718, 0.214, 0.475, 1.0] },
    { position: 0.75, color: [0.951, 0.506, 0.146, 1.0] },
    { position: 1.0, color: [0.940, 0.975, 0.131, 1.0] },
  ],
  inferno: [
    { position: 0.0, color: [0.001, 0.000, 0.014, 1.0] },
    { position: 0.25, color: [0.341, 0.062, 0.429, 1.0] },
    { position: 0.5, color: [0.735, 0.216, 0.330, 1.0] },
    { position: 0.75, color: [0.973, 0.558, 0.201, 1.0] },
    { position: 1.0, color: [0.988, 0.998, 0.645, 1.0] },
  ],
  magma: [
    { position: 0.0, color: [0.001, 0.000, 0.014, 1.0] },
    { position: 0.25, color: [0.316, 0.071, 0.485, 1.0] },
    { position: 0.5, color: [0.716, 0.215, 0.475, 1.0] },
    { position: 0.75, color: [0.973, 0.462, 0.498, 1.0] },
    { position: 1.0, color: [0.987, 0.991, 0.750, 1.0] },
  ],
  turbo: [
    { position: 0.0, color: [0.190, 0.072, 0.232, 1.0] },
    { position: 0.2, color: [0.085, 0.532, 0.829, 1.0] },
    { position: 0.4, color: [0.133, 0.846, 0.549, 1.0] },
    { position: 0.6, color: [0.631, 0.926, 0.248, 1.0] },
    { position: 0.8, color: [0.978, 0.636, 0.144, 1.0] },
    { position: 1.0, color: [0.712, 0.110, 0.092, 1.0] },
  ],
  hot: [
    { position: 0.0, color: [0.0, 0.0, 0.0, 1.0] },
    { position: 0.33, color: [1.0, 0.0, 0.0, 1.0] },
    { position: 0.66, color: [1.0, 1.0, 0.0, 1.0] },
    { position: 1.0, color: [1.0, 1.0, 1.0, 1.0] },
  ],
  cool: [
    { position: 0.0, color: [0.0, 1.0, 1.0, 1.0] },
    { position: 1.0, color: [1.0, 0.0, 1.0, 1.0] },
  ],
  blues: [
    { position: 0.0, color: [0.969, 0.984, 1.0, 1.0] },
    { position: 0.5, color: [0.392, 0.671, 0.847, 1.0] },
    { position: 1.0, color: [0.031, 0.188, 0.420, 1.0] },
  ],
  greens: [
    { position: 0.0, color: [0.969, 0.988, 0.961, 1.0] },
    { position: 0.5, color: [0.455, 0.769, 0.463, 1.0] },
    { position: 1.0, color: [0.0, 0.392, 0.0, 1.0] },
  ],
  reds: [
    { position: 0.0, color: [1.0, 0.961, 0.941, 1.0] },
    { position: 0.5, color: [0.984, 0.416, 0.290, 1.0] },
    { position: 1.0, color: [0.502, 0.0, 0.0, 1.0] },
  ],
  grayscale: [
    { position: 0.0, color: [0.0, 0.0, 0.0, 1.0] },
    { position: 1.0, color: [1.0, 1.0, 1.0, 1.0] },
  ],
};

/**
 * Interpolate between color stops at a given position
 */
function interpolateColorStops(stops: ColorStop[], position: number): ColorRGBA {
  // Clamp position
  const t = Math.max(0, Math.min(1, position));

  // Find surrounding stops
  let lower = stops[0];
  let upper = stops[stops.length - 1];

  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i].position && t <= stops[i + 1].position) {
      lower = stops[i];
      upper = stops[i + 1];
      break;
    }
  }

  // Interpolate
  const range = upper.position - lower.position;
  const localT = range > 0 ? (t - lower.position) / range : 0;

  return [
    lower.color[0] + (upper.color[0] - lower.color[0]) * localT,
    lower.color[1] + (upper.color[1] - lower.color[1]) * localT,
    lower.color[2] + (upper.color[2] - lower.color[2]) * localT,
    lower.color[3] + (upper.color[3] - lower.color[3]) * localT,
  ];
}

/**
 * Generate color scale data as Float32Array
 */
export function generateColorScaleData(
  stops: ColorStop[],
  resolution: number = 256
): Float32Array {
  const data = new Float32Array(resolution * 4);

  for (let i = 0; i < resolution; i++) {
    const t = i / (resolution - 1);
    const color = interpolateColorStops(stops, t);

    data[i * 4 + 0] = color[0];
    data[i * 4 + 1] = color[1];
    data[i * 4 + 2] = color[2];
    data[i * 4 + 3] = color[3];
  }

  return data;
}

/**
 * Create a color scale texture from predefined name
 */
export function createColorScaleTexture(
  context: GPUContext,
  name: ColorScaleName,
  resolution: number = 256
): ColorScaleTexture {
  const stops = COLOR_SCALES[name];
  return createCustomColorScaleTexture(context, stops, resolution, name);
}

/**
 * Create a color scale texture from custom color stops
 */
export function createCustomColorScaleTexture(
  context: GPUContext,
  stops: ColorStop[],
  resolution: number = 256,
  name: ColorScaleName | "custom" = "custom"
): ColorScaleTexture {
  const { device } = context;

  // Generate color data as RGBA8 (0-255)
  const colorDataFloat = generateColorScaleData(stops, resolution);
  const colorData = new Uint8Array(resolution * 4);
  for (let i = 0; i < resolution * 4; i++) {
    colorData[i] = Math.round(colorDataFloat[i] * 255);
  }

  // Create 1D texture with filterable format
  const texture = device.createTexture({
    label: `Color Scale Texture (${name})`,
    size: { width: resolution },
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    dimension: "1d",
  });

  // Upload data
  device.queue.writeTexture(
    { texture },
    colorData,
    { bytesPerRow: resolution * 4 },
    { width: resolution }
  );

  // Create view
  const view = texture.createView({
    label: `Color Scale View (${name})`,
    dimension: "1d",
  });

  // Create sampler with linear interpolation
  const sampler = device.createSampler({
    label: `Color Scale Sampler (${name})`,
    magFilter: "linear",
    minFilter: "linear",
    addressModeU: "clamp-to-edge",
  });

  return {
    texture,
    view,
    sampler,
    name,
    destroy: () => texture.destroy(),
  };
}

/**
 * Get all available color scale names
 */
export function getColorScaleNames(): ColorScaleName[] {
  return Object.keys(COLOR_SCALES) as ColorScaleName[];
}
