/**
 * GraphTypedInput Parser
 *
 * Parses GraphTypedInput format (typed arrays) for maximum performance
 * with minimal memory allocation. Ideal for large graphs (100K+ nodes).
 *
 * @module
 */

import type { GraphTypedInput } from "../types.ts";
import { ErrorCode, GraphMotherError } from "../errors.ts";
import { createIdMap, type IdLike } from "./id_map.ts";
import { clampWeight, type ParsedGraph } from "./parser.ts";
import { NODE_ATTR_FLOATS } from "../api/graph_state.ts";
import { CONTAINMENT_EDGE_TYPE } from "./hierarchy.ts";

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
 * Resolve the edge-pair column, honouring the deprecated `edges` alias.
 *
 * `edges` is documented as an alias for `edgePairs` and is accepted by the
 * frame contract, but the parser used to read `edgePairs` only — a producer
 * sending `edges` got a silently all-zeros edge list, which renders as every
 * node bound to node 0. Honouring the alias makes the documented contract true
 * without breaking those producers; supplying both is rejected, because there
 * is no way to know which one was meant.
 */
function resolveEdgePairs(input: GraphTypedInput): Uint32Array | undefined {
  if (input.edgePairs && input.edges) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      "GraphTypedInput carries both `edgePairs` and its deprecated alias `edges`; " +
        "supply exactly one",
    );
  }
  return input.edgePairs ?? input.edges;
}

/**
 * Map the per-edge kind column onto the edge-type strings the rest of the
 * pipeline selects containment with.
 *
 * Only the containment kind is retained: the producer's kind table is opaque to
 * core and nothing in v1 reads any other value, so materialising it would be
 * storage no consumer touches.
 *
 * @returns Edge types, or `undefined` when the input carries no usable kinds.
 */
function edgeTypesFromKinds(
  input: GraphTypedInput,
  edgeCount: number,
): string[] | undefined {
  const { edgeKinds, containmentKind } = input;
  if (!edgeKinds) {
    if (containmentKind !== undefined) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        "GraphTypedInput.containmentKind requires an edgeKinds column",
      );
    }
    return undefined;
  }
  if (containmentKind === undefined) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      "GraphTypedInput.edgeKinds requires containmentKind, naming which kind means " +
        "containment; without it core cannot tell a containment edge from a dependency",
    );
  }
  if (edgeKinds.length !== edgeCount) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `edgeKinds length (${edgeKinds.length}) must equal edgeCount (${edgeCount})`,
    );
  }

  const edgeTypes: string[] = new Array(edgeCount);
  for (let i = 0; i < edgeCount; i++) {
    if (edgeKinds[i] === containmentKind) {
      edgeTypes[i] = CONTAINMENT_EDGE_TYPE;
    }
  }
  return edgeTypes;
}

/**
 * Adopt the semantic LOD columns from the input.
 *
 * Lengths are checked and a mismatch throws: these columns are read by slot
 * with no bounds signal of their own, so a short column silently reports 0 for
 * every node past its end — a wrong policy decision that looks exactly like a
 * producer that supplied no column at all.
 *
 * `tag` is adopted by reference, since `Uint16Array` already is the storage
 * format. `weight` is copied, because clamping it into 0..1 must not mutate the
 * caller's array.
 */
function semanticColumns(
  input: GraphTypedInput,
  nodeCount: number,
): { nodeTags?: Uint16Array | undefined; nodeWeights?: Float32Array | undefined } {
  const { tag, weight } = input;
  if (tag && tag.length !== nodeCount) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `tag length (${tag.length}) must equal nodeCount (${nodeCount})`,
    );
  }
  if (weight && weight.length !== nodeCount) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `weight length (${weight.length}) must equal nodeCount (${nodeCount})`,
    );
  }

  let nodeWeights: Float32Array | undefined;
  if (weight) {
    nodeWeights = new Float32Array(nodeCount);
    for (let i = 0; i < nodeCount; i++) nodeWeights[i] = clampWeight(weight[i]);
  }
  return { nodeTags: tag, nodeWeights };
}

/**
 * Adopt the per-slot metadata columns from the input.
 *
 * Entries are retained by reference, keyed by slot, exactly as the object
 * parser retains `node.metadata` — this is what card labels and `contentRef`
 * read, and what `getNode().metadata` reports. Holes (absent entries) are
 * simply nodes without metadata. Styling is NOT read from here on the typed
 * path: `nodeRadii`/`nodeColors`/`edgeWidths`/`edgeColors` are the styling
 * columns, and a second channel for the same knob would disagree silently.
 *
 * Lengths are checked because the arrays are read by slot: a short column
 * would misattribute every entry after the gap to the wrong node.
 */
function metadataColumns(
  input: GraphTypedInput,
  nodeCount: number,
  edgeCount: number,
): {
  nodeMetadata: Map<number, Record<string, unknown>>;
  edgeMetadata: Map<number, Record<string, unknown>>;
} {
  if (input.nodeMetadata && input.nodeMetadata.length !== nodeCount) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `nodeMetadata length (${input.nodeMetadata.length}) must equal nodeCount (${nodeCount})`,
    );
  }
  if (input.edgeMetadata && input.edgeMetadata.length !== edgeCount) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `edgeMetadata length (${input.edgeMetadata.length}) must equal edgeCount (${edgeCount})`,
    );
  }

  const nodeMetadata = new Map<number, Record<string, unknown>>();
  if (input.nodeMetadata) {
    for (let i = 0; i < nodeCount; i++) {
      const entry = input.nodeMetadata[i];
      if (entry) nodeMetadata.set(i, entry as Record<string, unknown>);
    }
  }
  const edgeMetadata = new Map<number, Record<string, unknown>>();
  if (input.edgeMetadata) {
    for (let i = 0; i < edgeCount; i++) {
      const entry = input.edgeMetadata[i];
      if (entry) edgeMetadata.set(i, entry as Record<string, unknown>);
    }
  }
  return { nodeMetadata, edgeMetadata };
}

/**
 * Parses GraphTypedInput into GPU-ready format
 *
 * This parser is optimized for large graphs where data is already
 * in typed array format. It minimizes allocations by referencing
 * or copying directly from input arrays.
 *
 * `revision` is accepted and ignored (reserved for snapshot reconciliation).
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
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      "nodeCount must be a positive integer",
    );
  }

  const nodeCount = input.nodeCount;
  const edgeCount = input.edgeCount ?? 0;
  const inputEdgePairs = resolveEdgePairs(input);
  const edgeTypes = edgeTypesFromKinds(input, edgeCount);
  const semantics = semanticColumns(input, nodeCount);
  const metadata = metadataColumns(input, nodeCount, edgeCount);

  // Create ID maps (accepts string or number IDs)
  const nodeIdMap = createIdMap<IdLike>();
  const edgeIdMap = createIdMap<IdLike>();

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
  const nodeAttributes = new Float32Array(nodeCount * NODE_ATTR_FLOATS);
  const [defR, defG, defB] = finalConfig.defaultNodeColor;

  if (input.nodeRadii) {
    for (let i = 0; i < nodeCount; i++) {
      const base = i * NODE_ATTR_FLOATS;
      nodeAttributes[base] = input.nodeRadii[i];
    }
  } else {
    for (let i = 0; i < nodeCount; i++) {
      nodeAttributes[i * NODE_ATTR_FLOATS] = finalConfig.defaultNodeRadius;
    }
  }

  if (input.nodeColors) {
    // Colors as [r0, g0, b0, r1, g1, b1, ...]
    for (let i = 0; i < nodeCount; i++) {
      const base = i * NODE_ATTR_FLOATS;
      const colorBase = i * 3;
      nodeAttributes[base + 1] = input.nodeColors[colorBase];
      nodeAttributes[base + 2] = input.nodeColors[colorBase + 1];
      nodeAttributes[base + 3] = input.nodeColors[colorBase + 2];
    }
  } else {
    for (let i = 0; i < nodeCount; i++) {
      const base = i * NODE_ATTR_FLOATS;
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

  if (inputEdgePairs) {
    // Deinterleave [src0, tgt0, src1, tgt1, ...]
    edgeSources = new Uint32Array(edgeCount);
    edgeTargets = new Uint32Array(edgeCount);
    for (let i = 0; i < edgeCount; i++) {
      edgeSources[i] = inputEdgePairs[i * 2];
      edgeTargets[i] = inputEdgePairs[i * 2 + 1];
    }
  } else {
    edgeSources = new Uint32Array(edgeCount);
    edgeTargets = new Uint32Array(edgeCount);
  }

  // Edge attributes (8 floats per edge: width, r, g, b, selected, hovered, curvature, reserved)
  const edgeAttributes = new Float32Array(edgeCount * 8);
  const [defER, defEG, defEB] = finalConfig.defaultEdgeColor;

  if (input.edgeWidths) {
    for (let i = 0; i < edgeCount; i++) {
      edgeAttributes[i * 8] = input.edgeWidths[i];
    }
  } else {
    for (let i = 0; i < edgeCount; i++) {
      edgeAttributes[i * 8] = finalConfig.defaultEdgeWidth;
    }
  }

  if (input.edgeColors) {
    for (let i = 0; i < edgeCount; i++) {
      const base = i * 8;
      const colorBase = i * 3;
      edgeAttributes[base + 1] = input.edgeColors[colorBase];
      edgeAttributes[base + 2] = input.edgeColors[colorBase + 1];
      edgeAttributes[base + 3] = input.edgeColors[colorBase + 2];
    }
  } else {
    for (let i = 0; i < edgeCount; i++) {
      const base = i * 8;
      edgeAttributes[base + 1] = defER;
      edgeAttributes[base + 2] = defEG;
      edgeAttributes[base + 3] = defEB;
    }
  }

  // Initialize curvature and reserved to 0 (already zero-initialized)

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
    nodeMetadata: metadata.nodeMetadata,
    edgeMetadata: metadata.edgeMetadata,
    edgeTypes,
    hierarchy: input.hierarchy,
    ...semantics,
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
  const nodeCountVal = obj["nodeCount"];
  const edgeCountVal = obj["edgeCount"];
  const positionsVal = obj["positions"];
  const nodeRadiiVal = obj["nodeRadii"];
  const nodeColorsVal = obj["nodeColors"];
  const edgesVal = obj["edges"];
  // `edges` is the deprecated alias for `edgePairs`; validate whichever is
  // present, and flag the ambiguous case the parser rejects.
  const edgePairsVal = obj["edgePairs"] ?? edgesVal;
  const edgeKindsVal = obj["edgeKinds"];
  const containmentKindVal = obj["containmentKind"];

  if (obj["edgePairs"] !== undefined && edgesVal !== undefined) {
    errors.push("supply either edgePairs or its deprecated alias edges, not both");
  }

  // Check nodeCount
  if (typeof nodeCountVal !== "number" || nodeCountVal < 0) {
    errors.push("nodeCount must be a non-negative number");
  }

  const nodeCount = (nodeCountVal as number) || 0;
  const edgeCount = (edgeCountVal as number) || 0;

  // Validate array lengths
  if (positionsVal instanceof Float32Array) {
    if (positionsVal.length !== nodeCount * 2) {
      errors.push(
        `positions length (${positionsVal.length}) must be nodeCount * 2 (${nodeCount * 2})`,
      );
    }
  }

  if (nodeRadiiVal instanceof Float32Array) {
    if (nodeRadiiVal.length !== nodeCount) {
      errors.push(
        `nodeRadii length (${nodeRadiiVal.length}) must equal nodeCount (${nodeCount})`,
      );
    }
  }

  if (nodeColorsVal instanceof Float32Array) {
    if (nodeColorsVal.length !== nodeCount * 3) {
      errors.push(
        `nodeColors length (${nodeColorsVal.length}) must be nodeCount * 3 (${nodeCount * 3})`,
      );
    }
  }

  const tagVal = obj["tag"];
  if (tagVal instanceof Uint16Array && tagVal.length !== nodeCount) {
    errors.push(`tag length (${tagVal.length}) must equal nodeCount (${nodeCount})`);
  }

  const weightVal = obj["weight"];
  if (weightVal instanceof Float32Array && weightVal.length !== nodeCount) {
    errors.push(`weight length (${weightVal.length}) must equal nodeCount (${nodeCount})`);
  }

  const nodeMetadataVal = obj["nodeMetadata"];
  if (Array.isArray(nodeMetadataVal) && nodeMetadataVal.length !== nodeCount) {
    errors.push(
      `nodeMetadata length (${nodeMetadataVal.length}) must equal nodeCount (${nodeCount})`,
    );
  }

  const edgeMetadataVal = obj["edgeMetadata"];
  if (Array.isArray(edgeMetadataVal) && edgeMetadataVal.length !== edgeCount) {
    errors.push(
      `edgeMetadata length (${edgeMetadataVal.length}) must equal edgeCount (${edgeCount})`,
    );
  }

  if (edgePairsVal instanceof Uint32Array) {
    if (edgePairsVal.length !== edgeCount * 2) {
      errors.push(
        `edgePairs length (${edgePairsVal.length}) must be edgeCount * 2 (${edgeCount * 2})`,
      );
    }
  }

  if (edgeKindsVal instanceof Uint16Array) {
    if (edgeKindsVal.length !== edgeCount) {
      errors.push(
        `edgeKinds length (${edgeKindsVal.length}) must equal edgeCount (${edgeCount})`,
      );
    }
    if (typeof containmentKindVal !== "number") {
      errors.push("edgeKinds requires containmentKind naming which kind means containment");
    }
  } else if (containmentKindVal !== undefined) {
    errors.push("containmentKind requires an edgeKinds column");
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
 * Useful for combining graphs from multiple sources. Only node count,
 * positions and edge pairs are merged — ids, styling columns, metadata,
 * `edgeKinds`/`containmentKind` and a supplied `hierarchy` are dropped, so the
 * result's hierarchy is derived from the merged edges rather than inherited.
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
