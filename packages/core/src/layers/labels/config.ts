/**
 * Label Layer Configuration
 *
 * Defines configuration options for the label visualization layer.
 *
 * @module
 */

/**
 * Label layer configuration
 */
export interface LabelConfig {
  /** Whether the layer is visible */
  visible: boolean;
  /** Font size in pixels (default: 14) */
  fontSize: number;
  /** Text color as CSS color string (default: "#1f2937") */
  fontColor: string;
  /** Minimum zoom level to show any labels (default: 0.3) */
  minZoom: number;
  /** Maximum number of visible labels (default: 1000) */
  maxLabels: number;
  /** Label priority mode: 'importance' uses node importance, 'degree' uses node degree */
  priority: "importance" | "degree";
  /** Padding between labels in pixels (default: 4) */
  labelPadding: number;
  /** Vertical offset from node center in pixels (default: 10) */
  verticalOffset: number;
  /** Background color for labels, null for transparent (default: null) */
  backgroundColor: string | null;
  /** Background padding in pixels (default: 2) */
  backgroundPadding: number;
  /** Background corner radius in pixels (default: 2) */
  backgroundRadius: number;
}

/**
 * Default label configuration
 */
export const DEFAULT_LABEL_CONFIG: LabelConfig = {
  visible: false,
  fontSize: 14,
  fontColor: "#1f2937",
  minZoom: 0.3,
  maxLabels: 1000,
  priority: "importance",
  labelPadding: 4,
  verticalOffset: 10,
  backgroundColor: null,
  backgroundPadding: 2,
  backgroundRadius: 2,
};

/**
 * Parse CSS color to RGBA values (0-1 range)
 */
export function parseColor(color: string): [number, number, number, number] {
  // Handle hex colors
  if (color.startsWith("#")) {
    const hex = color.slice(1);
    if (hex.length === 3) {
      const r = parseInt(hex[0] + hex[0], 16) / 255;
      const g = parseInt(hex[1] + hex[1], 16) / 255;
      const b = parseInt(hex[2] + hex[2], 16) / 255;
      return [r, g, b, 1.0];
    }
    if (hex.length === 6) {
      const r = parseInt(hex.slice(0, 2), 16) / 255;
      const g = parseInt(hex.slice(2, 4), 16) / 255;
      const b = parseInt(hex.slice(4, 6), 16) / 255;
      return [r, g, b, 1.0];
    }
    if (hex.length === 8) {
      const r = parseInt(hex.slice(0, 2), 16) / 255;
      const g = parseInt(hex.slice(2, 4), 16) / 255;
      const b = parseInt(hex.slice(4, 6), 16) / 255;
      const a = parseInt(hex.slice(6, 8), 16) / 255;
      return [r, g, b, a];
    }
  }

  // Handle rgb/rgba
  const rgbMatch = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
  if (rgbMatch) {
    const r = parseInt(rgbMatch[1]) / 255;
    const g = parseInt(rgbMatch[2]) / 255;
    const b = parseInt(rgbMatch[3]) / 255;
    const a = rgbMatch[4] ? parseFloat(rgbMatch[4]) : 1.0;
    return [r, g, b, a];
  }

  // Default to black
  return [0, 0, 0, 1.0];
}
