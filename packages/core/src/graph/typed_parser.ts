/**
 * GraphTypedInput Parser
 *
 * Parses GraphTypedInput format (typed arrays) for maximum performance
 * with minimal memory allocation. Ideal for large graphs (100K+ nodes).
 *
 * @module
 */

import type { GraphTypedInput, NodeId, EdgeId } from "../types.ts";
import { HeroineGraphError, ErrorCode } from "../errors.ts";
import { createIdMap, type IdMap } from "./id_map.ts";
import type { ParsedGraph } from "./parser.ts";

/**
 * Typed parser configuration
 */
export interface TypedParserConfig {
  /** Default node radius */
  defaultNodeRadius?: number;
  /** Default node color RGB (0-1) */
  defaultNodeColor?: [number, number, number];
  /** Default edge width */
  defaultEdgeWidth?: number;
  /** Default edge color RGB (0-1) */
  defaultEdgeColor?: [number, number, number];
  /** Generate sequential IDs if not provided */
  generateIds?: boolean;
}

/**
 * Default typed parser configuration
 */
export const DEFAULT_TYPED_PARSER_CONFIG: Required<TypedParserConfig> = {
  defaultNodeRadius: 5,
  defaultNodeColor: [0.4, 0.6, 0.9],
  defaultEdgeWidth: 1,
  defaultEdgeColor: [0.5, 0.5, 0.5],
  generateIds: true,
};

/**
 * Parses GraphTypedInput into GPU-ready format
 *
 * This parser is optimized for large graphs where data is already
 * in typed array format. It minimizes allocations by referencing
 * or copying directly from input arrays.
 *
 * @param input - Graph typed input data
 * @param config - Parser configuration
 * @returns Parsed graph data
 */
export function parseGraphTypedInput(
  input: GraphTypedInput,
  config: TypedParserConfig = {},
): ParsedGraph {
  const finalConfig = { ...DEFAULT_TYPED_PARSER_CONFIG, ...config };

  // Validate required fields
  if (!input.nodeCount || input.nodeCount < 0) {
    throw new HeroineGraphError(
      ErrorCode.INVALID_GRAPH_DATA,
      "nodeCount must be a positive integer",
    );
  }

  const nodeCount = input.nodeCount;
  const edgeCount = input.edgeCount ?? 0;

  // Create ID maps
  const nodeIdMap = createIdMap<NodeId>();
  const edgeIdMap = createIdMap<EdgeId>();

  // Generate or use provided node IDs
  if (input.nodeIds) {
    for (let i = 0; i < nodeCount; i++) {
      nodeIdMap.add(input.nodeIds[i]);
    }
  } else if (finalConfig.generateIds) {
    for (let i = 0; i < nodeCount; i++) {
      nodeIdMap.add(`n${i}`);
    }
  }

  // Generate or use provided edge IDs
  if (input.edgeIds) {
    for (let i = 0; i < edgeCount; i++) {
      edgeIdMap.add(input.edgeIds[i]);
    }
  } else if (finalConfig.generateIds) {
    for (let i = 0; i < edgeCount; i++) {
      edgeIdMap.add(`e${i}`);
    }
  }

  // Positions - copy or create
  let positionsX: Float32Array;
  let positionsY: Float32Array;

  if (input.positions) {
    // Deinterleave if provided as [x0, y0, x1, y1, ...]
    positionsX = new Float32Array(nodeCount);
    positionsY = new Float32Array(nodeCount);
    for (let i = 0; i < nodeCount; i++) {
      positionsX[i] = input.positions[i * 2];
      positionsY[i] = input.positions[i * 2 + 1];
    }
  } else {
    // Will be initialized to 0, needs randomization
    positionsX = new Float32Array(nodeCount);
    positionsY = new Float32Array(nodeCount);
  }

  // Node attributes
  const nodeAttributes = new Float32Array(nodeCount * 6);
  const [defR, defG, defB] = finalConfig.defaultNodeColor;

  if (input.nodeRadii) {
    for (let i = 0; i < nodeCount; i++) {
      const base = i * 6;
      nodeAttributes[base] = input.nodeRadii[i];
    }
  } else {
    for (let i = 0; i < nodeCount; i++) {
      nodeAttributes[i * 6] = finalConfig.defaultNodeRadius;
    }
  }

  if (input.nodeColors) {
    // Colors as [r0, g0, b0, r1, g1, b1, ...]
    for (let i = 0; i < nodeCount; i++) {
      const base = i * 6;
      const colorBase = i * 3;
      nodeAttributes[base + 1] = input.nodeColors[colorBase];
      nodeAttributes[base + 2] = input.nodeColors[colorBase + 1];
      nodeAttributes[base + 3] = input.nodeColors[colorBase + 2];
    }
  } else {
    for (let i = 0; i < nodeCount; i++) {
      const base = i * 6;
      nodeAttributes[base + 1] = defR;
      nodeAttributes[base + 2] = defG;
      nodeAttributes[base + 3] = defB;
    }
  }

  // Selection/hover state initialized to 0
  // (already 0 from Float32Array constructor)

  // Edge data
  let edgeSources: Uint32Array;
  let edgeTargets: Uint32Array;

  if (input.edgePairs) {
    // Deinterleave [src0, tgt0, src1, tgt1, ...]
    edgeSources = new Uint32Array(edgeCount);
    edgeTargets = new Uint32Array(edgeCount);
    for (let i = 0; i < edgeCount; i++) {
      edgeSources[i] = input.edgePairs[i * 2];
      edgeTargets[i] = input.edgePairs[i * 2 + 1];
    }
  } else {
    edgeSources = new Uint32Array(edgeCount);
    edgeTargets = new Uint32Array(edgeCount);
  }

  // Edge attributes
  const edgeAttributes = new Float32Array(edgeCount * 6);
  const [defER, defEG, defEB] = finalConfig.defaultEdgeColor;

  if (input.edgeWidths) {
    for (let i = 0; i < edgeCount; i++) {
      edgeAttributes[i * 6] = input.edgeWidths[i];
    }
  } else {
    for (let i = 0; i < edgeCount; i++) {
      edgeAttributes[i * 6] = finalConfig.defaultEdgeWidth;
    }
  }

  if (input.edgeColors) {
    for (let i = 0; i < edgeCount; i++) {
      const base = i * 6;
      const colorBase = i * 3;
      edgeAttributes[base + 1] = input.edgeColors[colorBase];
      edgeAttributes[base + 2] = input.edgeColors[colorBase + 1];
      edgeAttributes[base + 3] = input.edgeColors[colorBase + 2];
    }
  } else {
    for (let i = 0; i < edgeCount; i++) {
      const base = i * 6;
      edgeAttributes[base + 1] = defER;
      edgeAttributes[base + 2] = defEG;
      edgeAttributes[base + 3] = defEB;
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
    nodeMetadata: new Map(),
    edgeMetadata: new Map(),
  };
}

/**
 * Validates GraphTypedInput structure
 *
 * @param input - Input to validate
 * @returns Validation result
 */
export function validateGraphTypedInput(input: unknown): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (!input || typeof input !== "object") {
    return { valid: false, errors: ["Input must be an object"] };
  }

  const obj = input as Record<string, unknown>;

  // Check nodeCount
  if (typeof obj.nodeCount !== "number" || obj.nodeCount < 0) {
    errors.push("nodeCount must be a non-negative number");
  }

  const nodeCount = (obj.nodeCount as number) || 0;
  const edgeCount = (obj.edgeCount as number) || 0;

  // Validate array lengths
  if (obj.positions instanceof Float32Array) {
    if (obj.positions.length !== nodeCount * 2) {
      errors.push(
        `positions length (${obj.positions.length}) must be nodeCount * 2 (${nodeCount * 2})`,
      );
    }
  }

  if (obj.nodeRadii instanceof Float32Array) {
    if (obj.nodeRadii.length !== nodeCount) {
      errors.push(
        `nodeRadii length (${obj.nodeRadii.length}) must equal nodeCount (${nodeCount})`,
      );
    }
  }

  if (obj.nodeColors instanceof Float32Array) {
    if (obj.nodeColors.length !== nodeCount * 3) {
      errors.push(
        `nodeColors length (${obj.nodeColors.length}) must be nodeCount * 3 (${nodeCount * 3})`,
      );
    }
  }

  if (obj.edgePairs instanceof Uint32Array) {
    if (obj.edgePairs.length !== edgeCount * 2) {
      errors.push(
        `edgePairs length (${obj.edgePairs.length}) must be edgeCount * 2 (${edgeCount * 2})`,
      );
    }
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Create a GraphTypedInput from raw arrays
 *
 * Helper function to construct typed input from various sources.
 *
 * @param nodeCount - Number of nodes
 * @param edgeCount - Number of edges
 * @param positions - Optional positions [x0, y0, x1, y1, ...]
 * @param edgePairs - Optional edges [src0, tgt0, src1, tgt1, ...]
 * @returns GraphTypedInput
 */
export function createTypedInput(
  nodeCount: number,
  edgeCount: number,
  positions?: Float32Array | number[],
  edgePairs?: Uint32Array | number[],
): GraphTypedInput {
  return {
    nodeCount,
    edgeCount,
    positions: positions instanceof Float32Array
      ? positions
      : positions
        ? new Float32Array(positions)
        : undefined,
    edgePairs: edgePairs instanceof Uint32Array
      ? edgePairs
      : edgePairs
        ? new Uint32Array(edgePairs)
        : undefined,
  };
}

/**
 * Merge multiple typed inputs into one
 *
 * Useful for combining graphs from multiple sources.
 *
 * @param inputs - Array of typed inputs to merge
 * @returns Merged typed input
 */
export function mergeTypedInputs(inputs: GraphTypedInput[]): GraphTypedInput {
  if (inputs.length === 0) {
    return { nodeCount: 0 };
  }

  if (inputs.length === 1) {
    return inputs[0];
  }

  // Calculate totals
  let totalNodes = 0;
  let totalEdges = 0;
  const nodeOffsets: number[] = [];

  for (const input of inputs) {
    nodeOffsets.push(totalNodes);
    totalNodes += input.nodeCount;
    totalEdges += input.edgeCount ?? 0;
  }

  // Merge positions
  const positions = new Float32Array(totalNodes * 2);
  let posOffset = 0;
  for (const input of inputs) {
    if (input.positions) {
      positions.set(input.positions, posOffset);
    }
    posOffset += input.nodeCount * 2;
  }

  // Merge edge pairs (adjusting indices)
  const edgePairs = new Uint32Array(totalEdges * 2);
  let edgeOffset = 0;
  for (let i = 0; i < inputs.length; i++) {
    const input = inputs[i];
    const nodeOffset = nodeOffsets[i];
    const edgeCount = input.edgeCount ?? 0;

    if (input.edgePairs) {
      for (let j = 0; j < edgeCount * 2; j++) {
        edgePairs[edgeOffset + j] = input.edgePairs[j] + nodeOffset;
      }
    }
    edgeOffset += edgeCount * 2;
  }

  return {
    nodeCount: totalNodes,
    edgeCount: totalEdges,
    positions,
    edgePairs,
  };
}
