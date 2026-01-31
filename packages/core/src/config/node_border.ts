/**
 * Node Border Configuration
 *
 * Configuration for node border rendering (outline around nodes).
 *
 * @module
 */

/**
 * Node border configuration
 */
export interface NodeBorderConfig {
  /** Whether borders are enabled */
  enabled: boolean;
  /** Border width in pixels */
  width: number;
  /** Border color (CSS color string or hex) */
  color: string;
}

/**
 * Default node border configuration
 */
export const DEFAULT_NODE_BORDER_CONFIG: NodeBorderConfig = {
  enabled: false,
  width: 2.0,
  color: "#000000",
};

/**
 * Merge user config with defaults
 */
export function mergeNodeBorderConfig(
  config: Partial<NodeBorderConfig> = {},
): NodeBorderConfig {
  return { ...DEFAULT_NODE_BORDER_CONFIG, ...config };
}
