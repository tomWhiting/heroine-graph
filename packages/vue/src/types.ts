/**
 * Shared types for the GraphMother Vue wrapper.
 *
 * @module
 */

import type { GraphConfig, GraphInput } from "@graphmother/core";

/**
 * Props for the GraphMother component
 */
export interface GraphMotherProps {
  /** Graph data to display */
  data?: GraphInput;
  /** Graph configuration */
  config?: Partial<GraphConfig>;
  /** Width of the canvas (default: 100%) */
  width?: string | number;
  /** Height of the canvas (default: 100%) */
  height?: string | number;
  /** CSS class name for the container */
  className?: string;
  /** Enable debug mode */
  debug?: boolean;
}
