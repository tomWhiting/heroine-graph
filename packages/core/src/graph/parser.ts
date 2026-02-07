/**
 * GraphInput Parser
 *
 * Parses GraphInput format (arrays of node/edge objects) into internal
 * representation for GPU processing.
 *
 * @module
 */

import type { GraphInput } from "../types.ts";
import { ErrorCode, HeroineGraphError } from "../errors.ts";
import { createIdMap, type IdLike, type IdMap } from "./id_map.ts";
import { parseColorToRGB, type RgbaColor } from "../utils/color.ts";

// Re-export RgbaColor for backwards compatibility
export type { RgbaColor } from "../utils/color.ts";

/**
 * Parsed graph data ready for GPU upload
 */
export interface ParsedGraph {
  /** Number of nodes */
  nodeCount: number;
  /** Number of edges */
  edgeCount: number;

  /** Node ID to index mapping (user IDs can be string or number) */
  nodeIdMap: IdMap<IdLike>;
  /** Edge ID to index mapping (user IDs can be string or number) */
  edgeIdMap: IdMap<IdLike>;

  /** Node positions X (Float32Array) */
  positionsX: Float32Array;
  /** Node positions Y (Float32Array) */
  positionsY: Float32Array;

  /** Node attributes (6 floats per node: radius, r, g, b, selected, hovered) */
  nodeAttributes: Float32Array;

  /** Edge source indices (Uint32Array) */
  edgeSources: Uint32Array;
  /** Edge target indices (Uint32Array) */
  edgeTargets: Uint32Array;

  /** Edge attributes (8 floats per edge: width, r, g, b, selected, hovered, curvature, opacity) */
  edgeAttributes: Float32Array;

  /** Original node metadata (for lookup) */
  nodeMetadata: Map<number, Record<string, unknown>>;
  /** Original edge metadata (for lookup) */
  edgeMetadata: Map<number, Record<string, unknown>>;

  /** Node types (for type-based styling) */
  nodeTypes?: string[] | undefined;
  /** Edge types (for type-based styling) */
  edgeTypes?: string[] | undefined;
}

/**
 * Parser configuration
 */
export interface ParserConfig {
  /** Default node radius */
  defaultNodeRadius?: number;
  /** Default node color (RGBA with values 0-1) */
  defaultNodeColor?: RgbaColor;
  /** Default edge width */
  defaultEdgeWidth?: number;
  /** Default edge color (RGBA with values 0-1) */
  defaultEdgeColor?: RgbaColor;
  /** Validate node/edge references */
  validateReferences?: boolean;
}

/**
 * Default parser configuration
 */
export const DEFAULT_PARSER_CONFIG: Required<ParserConfig> = {
  defaultNodeRadius: 5,
  defaultNodeColor: { r: 0.4, g: 0.6, b: 0.9, a: 1.0 },
  defaultEdgeWidth: 1,
  defaultEdgeColor: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 },
  validateReferences: true,
};

/**
 * Parse a color string or object to RGB values (0-1 range)
 */
function parseColor(
  color: string | RgbaColor | undefined,
  defaultColor: RgbaColor,
): [number, number, number] {
  if (!color) {
    return [defaultColor.r, defaultColor.g, defaultColor.b];
  }

  if (typeof color === "string") {
    // Use shared utility for string parsing
    return parseColorToRGB(color, [defaultColor.r, defaultColor.g, defaultColor.b]);
  }

  // Object with r, g, b properties
  return [color.r, color.g, color.b];
}

/**
 * Parses GraphInput into GPU-ready format
 *
 * @param input - Graph input data
 * @param config - Parser configuration
 * @returns Parsed graph data
 */
export function parseGraphInput(
  input: GraphInput,
  config: ParserConfig = {},
): ParsedGraph {
  const finalConfig = { ...DEFAULT_PARSER_CONFIG, ...config };
  const { nodes, edges } = input;

  const nodeCount = nodes.length;
  const edgeCount = edges.length;

  // Create ID mappings (accepts string or number IDs)
  const nodeIdMap = createIdMap<IdLike>();
  const edgeIdMap = createIdMap<IdLike>();

  // Allocate arrays
  const positionsX = new Float32Array(nodeCount);
  const positionsY = new Float32Array(nodeCount);
  const nodeAttributes = new Float32Array(nodeCount * 6);
  const edgeSources = new Uint32Array(edgeCount);
  const edgeTargets = new Uint32Array(edgeCount);
  // 8 floats per edge: width, r, g, b, selected, hovered, curvature, opacity
  const edgeAttributes = new Float32Array(edgeCount * 8);

  // Metadata storage
  const nodeMetadata = new Map<number, Record<string, unknown>>();
  const edgeMetadata = new Map<number, Record<string, unknown>>();

  // Type storage (for type-based styling)
  const nodeTypes: string[] = new Array(nodeCount);
  const edgeTypes: string[] = new Array(edgeCount);
  let hasNodeTypes = false;
  let hasEdgeTypes = false;

  // Parse nodes
  for (let i = 0; i < nodeCount; i++) {
    const node = nodes[i];
    if (nodeIdMap.has(node.id)) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_GRAPH_DATA,
        `Duplicate node ID: "${node.id}" at index ${i}`,
      );
    }
    const idx = nodeIdMap.add(node.id);

    // Position (default to 0,0 - will be randomized later)
    positionsX[idx] = node.x ?? 0;
    positionsY[idx] = node.y ?? 0;

    // Attributes
    const attrBase = idx * 6;
    nodeAttributes[attrBase] = node.radius ?? finalConfig.defaultNodeRadius;
    const [r, g, b] = parseColor(node.color, finalConfig.defaultNodeColor);
    nodeAttributes[attrBase + 1] = r;
    nodeAttributes[attrBase + 2] = g;
    nodeAttributes[attrBase + 3] = b;
    nodeAttributes[attrBase + 4] = 0; // selected
    nodeAttributes[attrBase + 5] = 0; // hovered

    // Store metadata (accessed via index signature)
    const nodeMetadataValue = node["metadata"] as Record<string, unknown> | undefined;
    if (nodeMetadataValue) {
      nodeMetadata.set(idx, nodeMetadataValue);
    }

    // Extract type for type-based styling (top-level or metadata.type)
    const nodeType = (node["type"] as string | undefined)
      ?? (nodeMetadataValue?.["type"] as string | undefined);
    if (nodeType) {
      nodeTypes[idx] = nodeType;
      hasNodeTypes = true;
    }
  }

  // Parse edges
  for (let i = 0; i < edgeCount; i++) {
    const edge = edges[i];

    // Get or generate edge ID (id comes from index signature on EdgeInput)
    const edgeId = (edge["id"] as IdLike | undefined) ?? `edge_${i}`;
    const idx = edgeIdMap.add(edgeId);

    // Resolve source/target to indices
    const sourceIdx = nodeIdMap.get(edge.source);
    const targetIdx = nodeIdMap.get(edge.target);

    if (finalConfig.validateReferences) {
      if (sourceIdx === undefined) {
        throw new HeroineGraphError(
          ErrorCode.INVALID_GRAPH_DATA,
          `Edge ${edgeId}: source node "${edge.source}" not found`,
        );
      }
      if (targetIdx === undefined) {
        throw new HeroineGraphError(
          ErrorCode.INVALID_GRAPH_DATA,
          `Edge ${edgeId}: target node "${edge.target}" not found`,
        );
      }
    }

    edgeSources[idx] = sourceIdx ?? 0;
    edgeTargets[idx] = targetIdx ?? 0;

    // Attributes (8 floats per edge)
    const attrBase = idx * 8;
    edgeAttributes[attrBase] = edge.width ?? finalConfig.defaultEdgeWidth;
    const [r, g, b] = parseColor(edge.color, finalConfig.defaultEdgeColor);
    edgeAttributes[attrBase + 1] = r;
    edgeAttributes[attrBase + 2] = g;
    edgeAttributes[attrBase + 3] = b;
    edgeAttributes[attrBase + 4] = 0; // selected
    edgeAttributes[attrBase + 5] = 0; // hovered
    edgeAttributes[attrBase + 6] = 0; // curvature (default: straight)
    edgeAttributes[attrBase + 7] = 1.0; // opacity (default: fully visible)

    // Store metadata (accessed via index signature)
    const edgeMetadataValue = edge["metadata"] as Record<string, unknown> | undefined;
    if (edgeMetadataValue) {
      edgeMetadata.set(idx, edgeMetadataValue);
    }

    // Extract type for type-based styling
    const edgeType = edge["type"] as string | undefined;
    if (edgeType) {
      edgeTypes[idx] = edgeType;
      hasEdgeTypes = true;
    }
  }

  return {
    nodeCount,
    edgeCount,
    nodeIdMap,
    edgeIdMap,
    positionsX,
    positionsY,
    nodeAttributes,
    edgeSources,
    edgeTargets,
    edgeAttributes,
    nodeMetadata,
    edgeMetadata,
    // Only include types if any were found
    nodeTypes: hasNodeTypes ? nodeTypes : undefined,
    edgeTypes: hasEdgeTypes ? edgeTypes : undefined,
  };
}

/**
 * Validates GraphInput structure
 *
 * @param input - Input to validate
 * @returns Validation result with error messages
 */
export function validateGraphInput(input: unknown): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (!input || typeof input !== "object") {
    return { valid: false, errors: ["Input must be an object"] };
  }

  const obj = input as Record<string, unknown>;
  const nodes = obj["nodes"];
  const edges = obj["edges"];

  // Check nodes array
  if (!Array.isArray(nodes)) {
    errors.push("Missing or invalid 'nodes' array");
  } else {
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      if (!node || typeof node !== "object") {
        errors.push(`Node at index ${i} must be an object`);
        continue;
      }
      const n = node as Record<string, unknown>;
      if (!("id" in n)) {
        errors.push(`Node at index ${i} missing 'id'`);
      }
    }
  }

  // Check edges array
  if (!Array.isArray(edges)) {
    errors.push("Missing or invalid 'edges' array");
  } else {
    for (let i = 0; i < edges.length; i++) {
      const edge = edges[i];
      if (!edge || typeof edge !== "object") {
        errors.push(`Edge at index ${i} must be an object`);
        continue;
      }
      const e = edge as Record<string, unknown>;
      if (!("source" in e)) {
        errors.push(`Edge at index ${i} missing 'source'`);
      }
      if (!("target" in e)) {
        errors.push(`Edge at index ${i} missing 'target'`);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Create edge indices buffer from source/target arrays
 * Packs source and target indices into a single buffer for the edge shader
 *
 * @param sources - Edge source indices
 * @param targets - Edge target indices
 * @returns Interleaved edge indices buffer
 */
export function createEdgeIndicesBuffer(
  sources: Uint32Array,
  targets: Uint32Array,
): Uint32Array {
  const edgeCount = sources.length;
  const buffer = new Uint32Array(edgeCount * 2);

  for (let i = 0; i < edgeCount; i++) {
    buffer[i * 2] = sources[i];
    buffer[i * 2 + 1] = targets[i];
  }

  return buffer;
}
