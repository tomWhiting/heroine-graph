/**
 * Styling Module
 *
 * Type-based styling system for nodes and edges.
 *
 * @module
 */

export type {
  EdgeTypeStyle,
  EdgeTypeStyleMap,
  NodeTypeStyle,
  NodeTypeStyleMap,
  ResolvedEdgeStyle,
  ResolvedNodeStyle,
} from "./types.ts";

export { createTypeStyleManager, TypeStyleManager } from "./type_styles.ts";
