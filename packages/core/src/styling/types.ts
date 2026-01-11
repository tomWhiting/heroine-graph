/**
 * Type-Based Styling Types
 *
 * Type definitions for the type-based styling system.
 *
 * @module
 */

/**
 * Style for a node type
 */
export interface NodeTypeStyle {
  /** Fill color (CSS color string or hex) */
  color?: string;
  /** Node radius multiplier (1.0 = default) */
  size?: number;
  /** Border color (CSS color string or hex) */
  borderColor?: string;
  /** Border width in pixels */
  borderWidth?: number;
}

/**
 * Style for an edge type
 */
export interface EdgeTypeStyle {
  /** Edge color (CSS color string or hex) */
  color?: string;
  /** Edge width multiplier (1.0 = default) */
  width?: number;
  /** Edge opacity (0-1) */
  opacity?: number;
}

/**
 * Map of type names to node styles
 */
export type NodeTypeStyleMap = Record<string, NodeTypeStyle>;

/**
 * Map of type names to edge styles
 */
export type EdgeTypeStyleMap = Record<string, EdgeTypeStyle>;

/**
 * Resolved style for a node (all fields required)
 */
export interface ResolvedNodeStyle {
  /** RGBA color (0-1 range) */
  color: [number, number, number, number];
  /** Node radius */
  size: number;
}

/**
 * Resolved style for an edge (all fields required)
 */
export interface ResolvedEdgeStyle {
  /** RGBA color (0-1 range) */
  color: [number, number, number, number];
  /** Edge width */
  width: number;
}
