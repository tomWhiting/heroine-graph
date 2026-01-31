/**
 * Shared Utilities
 *
 * Common utility functions used across the library.
 * Import from here instead of reimplementing.
 *
 * @module
 */

export {
  colorToHex,
  DEFAULT_COLORS,
  lerpColor,
  NAMED_COLORS,
  parseColor,
  parseColorToRGB,
  parseColorToRGBA,
  withAlpha,
} from "./color.ts";
export type { ColorInput, ColorRGB, ColorRGBA, RgbaColor } from "./color.ts";
