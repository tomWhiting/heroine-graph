/**
 * Type Style Manager
 *
 * Manages visual styles for node and edge types.
 * Provides efficient lookup of styles by type name.
 *
 * @module
 */

import type {
  EdgeTypeStyle,
  EdgeTypeStyleMap,
  NodeTypeStyle,
  NodeTypeStyleMap,
  ResolvedEdgeStyle,
  ResolvedNodeStyle,
} from "./types.ts";

/**
 * Default node style
 */
const DEFAULT_NODE_STYLE: ResolvedNodeStyle = {
  color: [0.5, 0.5, 0.5, 1.0],
  size: 1.0,
};

/**
 * Default edge style
 */
const DEFAULT_EDGE_STYLE: ResolvedEdgeStyle = {
  color: [0.5, 0.5, 0.5, 0.4],
  width: 1.0,
};

/**
 * Parse a CSS color string to RGBA (0-1 range)
 */
function parseColor(color: string): [number, number, number, number] {
  // Handle hex colors
  if (color.startsWith("#")) {
    const hex = color.slice(1);
    if (hex.length === 3) {
      // Short hex (#RGB)
      const r = parseInt(hex[0] + hex[0], 16) / 255;
      const g = parseInt(hex[1] + hex[1], 16) / 255;
      const b = parseInt(hex[2] + hex[2], 16) / 255;
      return [r, g, b, 1.0];
    } else if (hex.length === 6) {
      // Long hex (#RRGGBB)
      const r = parseInt(hex.slice(0, 2), 16) / 255;
      const g = parseInt(hex.slice(2, 4), 16) / 255;
      const b = parseInt(hex.slice(4, 6), 16) / 255;
      return [r, g, b, 1.0];
    } else if (hex.length === 8) {
      // Hex with alpha (#RRGGBBAA)
      const r = parseInt(hex.slice(0, 2), 16) / 255;
      const g = parseInt(hex.slice(2, 4), 16) / 255;
      const b = parseInt(hex.slice(4, 6), 16) / 255;
      const a = parseInt(hex.slice(6, 8), 16) / 255;
      return [r, g, b, a];
    }
  }

  // Handle rgb/rgba
  const rgbaMatch = color.match(/rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+))?\s*\)/i);
  if (rgbaMatch) {
    const r = parseInt(rgbaMatch[1], 10) / 255;
    const g = parseInt(rgbaMatch[2], 10) / 255;
    const b = parseInt(rgbaMatch[3], 10) / 255;
    const a = rgbaMatch[4] ? parseFloat(rgbaMatch[4]) : 1.0;
    return [r, g, b, a];
  }

  // Handle named colors (basic set)
  const namedColors: Record<string, [number, number, number, number]> = {
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
  };

  const lower = color.toLowerCase();
  if (namedColors[lower]) {
    return namedColors[lower];
  }

  // Fallback to default gray
  return [0.5, 0.5, 0.5, 1.0];
}

/**
 * Type Style Manager
 *
 * Stores and resolves styles for node and edge types.
 */
export class TypeStyleManager {
  private nodeStyles: Map<string, NodeTypeStyle> = new Map();
  private edgeStyles: Map<string, EdgeTypeStyle> = new Map();

  // Cached resolved styles
  private nodeStyleCache: Map<string, ResolvedNodeStyle> = new Map();
  private edgeStyleCache: Map<string, ResolvedEdgeStyle> = new Map();

  /**
   * Set styles for multiple node types
   */
  setNodeTypeStyles(styles: NodeTypeStyleMap): void {
    this.nodeStyles.clear();
    this.nodeStyleCache.clear();
    for (const [type, style] of Object.entries(styles)) {
      this.nodeStyles.set(type, style);
    }
  }

  /**
   * Set style for a single node type
   */
  setNodeTypeStyle(type: string, style: NodeTypeStyle): void {
    this.nodeStyles.set(type, style);
    this.nodeStyleCache.delete(type);
  }

  /**
   * Get the raw style for a node type
   */
  getNodeTypeStyle(type: string): NodeTypeStyle | undefined {
    return this.nodeStyles.get(type);
  }

  /**
   * Set styles for multiple edge types
   */
  setEdgeTypeStyles(styles: EdgeTypeStyleMap): void {
    this.edgeStyles.clear();
    this.edgeStyleCache.clear();
    for (const [type, style] of Object.entries(styles)) {
      this.edgeStyles.set(type, style);
    }
  }

  /**
   * Set style for a single edge type
   */
  setEdgeTypeStyle(type: string, style: EdgeTypeStyle): void {
    this.edgeStyles.set(type, style);
    this.edgeStyleCache.delete(type);
  }

  /**
   * Get the raw style for an edge type
   */
  getEdgeTypeStyle(type: string): EdgeTypeStyle | undefined {
    return this.edgeStyles.get(type);
  }

  /**
   * Resolve the complete style for a node type
   * Returns cached result if available
   */
  resolveNodeStyle(type: string | undefined): ResolvedNodeStyle {
    // No type = default style
    if (!type) return DEFAULT_NODE_STYLE;

    // Check cache
    const cached = this.nodeStyleCache.get(type);
    if (cached) return cached;

    // Resolve style
    const typeStyle = this.nodeStyles.get(type);
    if (!typeStyle) return DEFAULT_NODE_STYLE;

    const resolved: ResolvedNodeStyle = {
      color: typeStyle.color ? parseColor(typeStyle.color) : DEFAULT_NODE_STYLE.color,
      size: typeStyle.size ?? DEFAULT_NODE_STYLE.size,
    };

    // Cache and return
    this.nodeStyleCache.set(type, resolved);
    return resolved;
  }

  /**
   * Resolve the complete style for an edge type
   * Returns cached result if available
   */
  resolveEdgeStyle(type: string | undefined): ResolvedEdgeStyle {
    // No type = default style
    if (!type) return DEFAULT_EDGE_STYLE;

    // Check cache
    const cached = this.edgeStyleCache.get(type);
    if (cached) return cached;

    // Resolve style
    const typeStyle = this.edgeStyles.get(type);
    if (!typeStyle) return DEFAULT_EDGE_STYLE;

    const resolved: ResolvedEdgeStyle = {
      color: typeStyle.color ? parseColor(typeStyle.color) : DEFAULT_EDGE_STYLE.color,
      width: typeStyle.width ?? DEFAULT_EDGE_STYLE.width,
    };

    // Apply opacity if specified
    if (typeStyle.opacity !== undefined) {
      resolved.color[3] = typeStyle.opacity;
    }

    // Cache and return
    this.edgeStyleCache.set(type, resolved);
    return resolved;
  }

  /**
   * Clear all type styles
   */
  clear(): void {
    this.nodeStyles.clear();
    this.edgeStyles.clear();
    this.nodeStyleCache.clear();
    this.edgeStyleCache.clear();
  }

  /**
   * Check if any node type styles are defined
   */
  hasNodeStyles(): boolean {
    return this.nodeStyles.size > 0;
  }

  /**
   * Check if any edge type styles are defined
   */
  hasEdgeStyles(): boolean {
    return this.edgeStyles.size > 0;
  }

  /**
   * Get all defined node types
   */
  getNodeTypes(): string[] {
    return Array.from(this.nodeStyles.keys());
  }

  /**
   * Get all defined edge types
   */
  getEdgeTypes(): string[] {
    return Array.from(this.edgeStyles.keys());
  }
}

/**
 * Create a new TypeStyleManager
 */
export function createTypeStyleManager(): TypeStyleManager {
  return new TypeStyleManager();
}
