/**
 * Color Parsing Utilities
 *
 * Unified color parsing for the entire library. All color string/object
 * parsing should use these functions - no reinventing the wheel.
 *
 * @module
 */

/**
 * RGBA color as 4 floats [0-1]
 */
export type ColorRGBA = [number, number, number, number];

/**
 * RGB color as 3 floats [0-1]
 */
export type ColorRGB = [number, number, number];

/**
 * Color object with named properties
 */
export interface RgbaColor {
  r: number;
  g: number;
  b: number;
  a?: number;
}

/**
 * Input types that can be parsed as a color
 */
export type ColorInput = string | RgbaColor | ColorRGBA | ColorRGB;

/**
 * Default colors for fallback
 */
export const DEFAULT_COLORS = {
  black: [0, 0, 0, 1] as ColorRGBA,
  white: [1, 1, 1, 1] as ColorRGBA,
  gray: [0.5, 0.5, 0.5, 1] as ColorRGBA,
  darkGray: [0.2, 0.2, 0.2, 1] as ColorRGBA,
  transparent: [0, 0, 0, 0] as ColorRGBA,
} as const;

/**
 * Parse a CSS color string to RGBA values (0-1 range).
 *
 * Supports:
 * - Hex: #RGB, #RRGGBB, #RRGGBBAA
 * - RGB: rgb(r, g, b)
 * - RGBA: rgba(r, g, b, a)
 *
 * @param color - CSS color string
 * @param fallback - Fallback color if parsing fails (default: dark gray)
 * @returns RGBA color as [r, g, b, a] with values 0-1
 *
 * @example
 * parseColorToRGBA("#ff0000")     // [1, 0, 0, 1]
 * parseColorToRGBA("#f00")        // [1, 0, 0, 1]
 * parseColorToRGBA("#ff000080")   // [1, 0, 0, 0.5]
 * parseColorToRGBA("rgb(255,0,0)") // [1, 0, 0, 1]
 */
export function parseColorToRGBA(
  color: string,
  fallback: ColorRGBA = DEFAULT_COLORS.darkGray,
): ColorRGBA {
  if (!color || typeof color !== "string") {
    return [...fallback] as ColorRGBA;
  }

  const trimmed = color.trim();

  // Handle hex colors
  if (trimmed.startsWith("#")) {
    const hex = trimmed.slice(1);

    if (hex.length === 3) {
      // #RGB
      return [
        parseInt(hex[0] + hex[0], 16) / 255,
        parseInt(hex[1] + hex[1], 16) / 255,
        parseInt(hex[2] + hex[2], 16) / 255,
        1.0,
      ];
    }

    if (hex.length === 6) {
      // #RRGGBB
      return [
        parseInt(hex.slice(0, 2), 16) / 255,
        parseInt(hex.slice(2, 4), 16) / 255,
        parseInt(hex.slice(4, 6), 16) / 255,
        1.0,
      ];
    }

    if (hex.length === 8) {
      // #RRGGBBAA
      return [
        parseInt(hex.slice(0, 2), 16) / 255,
        parseInt(hex.slice(2, 4), 16) / 255,
        parseInt(hex.slice(4, 6), 16) / 255,
        parseInt(hex.slice(6, 8), 16) / 255,
      ];
    }
  }

  // Handle rgb/rgba
  const rgbMatch = trimmed.match(
    /rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+))?\s*\)/,
  );
  if (rgbMatch) {
    return [
      parseInt(rgbMatch[1]) / 255,
      parseInt(rgbMatch[2]) / 255,
      parseInt(rgbMatch[3]) / 255,
      rgbMatch[4] !== undefined ? parseFloat(rgbMatch[4]) : 1.0,
    ];
  }

  // Handle named colors
  const namedColor = NAMED_COLORS[trimmed.toLowerCase()];
  if (namedColor) {
    return namedColor;
  }

  // Unknown format, return fallback
  return [...fallback] as ColorRGBA;
}

/**
 * Common named colors (CSS basic colors)
 */
export const NAMED_COLORS: Record<string, ColorRGBA> = {
  red: [1, 0, 0, 1],
  green: [0, 0.5, 0, 1],
  blue: [0, 0, 1, 1],
  yellow: [1, 1, 0, 1],
  orange: [1, 0.647, 0, 1],
  purple: [0.5, 0, 0.5, 1],
  cyan: [0, 1, 1, 1],
  magenta: [1, 0, 1, 1],
  white: [1, 1, 1, 1],
  black: [0, 0, 0, 1],
  gray: [0.5, 0.5, 0.5, 1],
  grey: [0.5, 0.5, 0.5, 1],
  pink: [1, 0.753, 0.796, 1],
  brown: [0.647, 0.165, 0.165, 1],
  lime: [0, 1, 0, 1],
  navy: [0, 0, 0.5, 1],
  teal: [0, 0.5, 0.5, 1],
  olive: [0.5, 0.5, 0, 1],
  maroon: [0.5, 0, 0, 1],
  silver: [0.753, 0.753, 0.753, 1],
  aqua: [0, 1, 1, 1],
  fuchsia: [1, 0, 1, 1],
};

/**
 * Parse a CSS color string to RGB values (0-1 range).
 * Alpha channel is discarded.
 *
 * @param color - CSS color string
 * @param fallback - Fallback color if parsing fails
 * @returns RGB color as [r, g, b] with values 0-1
 */
export function parseColorToRGB(
  color: string,
  fallback: ColorRGB = [0.2, 0.2, 0.2],
): ColorRGB {
  const rgba = parseColorToRGBA(color, [...fallback, 1] as ColorRGBA);
  return [rgba[0], rgba[1], rgba[2]];
}

/**
 * Parse any color input (string, object, or array) to RGBA.
 *
 * @param color - Color in any supported format
 * @param fallback - Fallback color if parsing fails
 * @returns RGBA color as [r, g, b, a] with values 0-1
 */
export function parseColor(
  color: ColorInput | undefined | null,
  fallback: ColorRGBA = DEFAULT_COLORS.darkGray,
): ColorRGBA {
  if (color === undefined || color === null) {
    return [...fallback] as ColorRGBA;
  }

  // String input
  if (typeof color === "string") {
    return parseColorToRGBA(color, fallback);
  }

  // Array input (already normalized)
  if (Array.isArray(color)) {
    if (color.length === 4) {
      return color as ColorRGBA;
    }
    if (color.length === 3) {
      return [color[0], color[1], color[2], 1.0];
    }
    return [...fallback] as ColorRGBA;
  }

  // Object input (RgbaColor)
  if (typeof color === "object" && "r" in color && "g" in color && "b" in color) {
    // Check if values are 0-1 or 0-255 range
    const isNormalized = color.r <= 1 && color.g <= 1 && color.b <= 1;
    if (isNormalized) {
      return [color.r, color.g, color.b, color.a ?? 1.0];
    }
    // Assume 0-255 range
    return [color.r / 255, color.g / 255, color.b / 255, color.a ?? 1.0];
  }

  return [...fallback] as ColorRGBA;
}

/**
 * Convert RGBA color to hex string.
 *
 * @param color - RGBA color as [r, g, b, a] with values 0-1
 * @param includeAlpha - Whether to include alpha in output
 * @returns Hex color string (#RRGGBB or #RRGGBBAA)
 */
export function colorToHex(color: ColorRGBA | ColorRGB, includeAlpha = false): string {
  const r = Math.round(color[0] * 255)
    .toString(16)
    .padStart(2, "0");
  const g = Math.round(color[1] * 255)
    .toString(16)
    .padStart(2, "0");
  const b = Math.round(color[2] * 255)
    .toString(16)
    .padStart(2, "0");

  if (includeAlpha && color.length === 4) {
    const a = Math.round(color[3] * 255)
      .toString(16)
      .padStart(2, "0");
    return `#${r}${g}${b}${a}`;
  }

  return `#${r}${g}${b}`;
}

/**
 * Interpolate between two colors.
 *
 * @param a - First color
 * @param b - Second color
 * @param t - Interpolation factor (0-1)
 * @returns Interpolated color
 */
export function lerpColor(a: ColorRGBA, b: ColorRGBA, t: number): ColorRGBA {
  const clampedT = Math.max(0, Math.min(1, t));
  return [
    a[0] + (b[0] - a[0]) * clampedT,
    a[1] + (b[1] - a[1]) * clampedT,
    a[2] + (b[2] - a[2]) * clampedT,
    a[3] + (b[3] - a[3]) * clampedT,
  ];
}

/**
 * Apply alpha to a color.
 *
 * @param color - Input color
 * @param alpha - Alpha value (0-1)
 * @returns Color with applied alpha
 */
export function withAlpha(color: ColorRGBA | ColorRGB, alpha: number): ColorRGBA {
  return [color[0], color[1], color[2], alpha];
}
