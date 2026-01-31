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

// Re-export parseColor from shared utilities for backwards compatibility
export { parseColorToRGBA as parseColor } from "../../utils/color.ts";
