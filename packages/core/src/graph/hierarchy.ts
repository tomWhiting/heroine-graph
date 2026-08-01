/**
 * Containment hierarchy — the retained, algorithm-independent tree.
 *
 * A code graph is a containment tree (repo → directory → file → symbol) with
 * cross-cutting dependency edges laid over it. Four per-slot columns describe
 * that tree, and four independent consumers read them: the nested-bubble
 * layout, semantic LOD, hit-test broad phase, and DOM-overlay culling.
 *
 * The columns are **supplied-or-computed**. A producer that already knows the
 * tree (an indexer walking a filesystem) ships {@link HierarchyColumns} with
 * the graph; anything else has them derived in WASM at load. Both paths land in
 * the same retained {@link RetainedHierarchy}, so no consumer can tell them
 * apart. The WASM path is not a legacy fallback — it is what keeps a supplied
 * hierarchy optional and what recomputes when a producer's bubble constants go
 * stale.
 *
 * # Slot-ordering producer contract
 *
 * Producers **SHOULD** emit node slots in depth-first order, so that every
 * subtree occupies a contiguous slot range. Consumers that translate a whole
 * subtree (LOD expand fix-up) then become one contiguous buffer write instead
 * of one write per node. Core does not verify and does not reorder: slot order
 * is the producer's, and a non-DFS graph is correct, merely slower on those
 * paths.
 *
 * @module
 */

import { ErrorCode, GraphMotherError } from "../errors.ts";

/**
 * `parent[i]` value marking a forest root: slot `i` has no containment parent.
 *
 * Multiple roots are legal and expected — a repo root plus a handful of orphan
 * config files is a forest, not a tree.
 */
export const HIERARCHY_ROOT = 0xFFFFFFFF;

/**
 * Largest representable depth.
 *
 * {@link HierarchyColumns.depth} is a `Uint16Array` (2 bytes/node rather than
 * 4; depth is read on the LOD hot path at 220K). Depths beyond this saturate,
 * which requires a containment chain 65 535 levels deep — three orders of
 * magnitude past any real filesystem.
 */
export const MAX_HIERARCHY_DEPTH = 0xFFFF;

/** Number of concatenated columns in the WASM `computeBubbleDataFromEdges` result. */
export const HIERARCHY_WASM_COLUMNS = 4;

/**
 * Containment hierarchy, one entry per node slot.
 *
 * Precomputed by the producer, or derived in WASM as fallback. Every array is
 * exactly `nodeCount` long and indexed by GPU slot.
 */
export interface HierarchyColumns {
  /** Containment parent slot, or {@link HIERARCHY_ROOT}. Forests are legal. */
  readonly parent: Uint32Array;
  /** Bubble radius: bottom-up over the subtree. A layout force parameter, not a geometric bound. */
  readonly wellRadius: Float32Array;
  /** Distance from the slot's own root; roots are 0. Saturates at {@link MAX_HIERARCHY_DEPTH}. */
  readonly depth: Uint16Array;
  /** Subtree node count including the slot itself; leaves are 1. */
  readonly subtreeSize: Uint32Array;
  /** Optional `[x0, y0, x1, y1, ...]`. Core derives from live positions when absent. */
  readonly centroid?: Float32Array | undefined;
}

/**
 * Parent→children CSR over the dense slot space.
 *
 * The children of slot `n` are `children[offsets[n] .. offsets[n + 1]]`, in
 * ascending slot order. `offsets` has `nodeCount + 1` entries.
 */
export interface ChildrenCsr {
  readonly offsets: Uint32Array;
  readonly children: Uint32Array;
}

/** Where a retained hierarchy came from. */
export type HierarchySource = "supplied" | "derived";

/**
 * The hierarchy as the graph retains it: columns plus the downward index.
 *
 * `parent[]` walks up (lowest visible ancestor, edge re-targeting); the CSR
 * walks down (the LOD visible-cut descent). Both are needed and the CSR is
 * derived from `parent[]`, so supplied and derived hierarchies index
 * identically.
 */
export interface RetainedHierarchy {
  readonly nodeCount: number;
  readonly columns: HierarchyColumns;
  readonly children: ChildrenCsr;
  readonly source: HierarchySource;
}

/**
 * The WASM surface hierarchy derivation needs.
 *
 * Structural, not nominal, so the derivation is testable against the real
 * module or a stub without importing the whole engine type.
 */
export interface HierarchyDeriver {
  /**
   * Returns `4 * nodeBound` floats: `[wellRadius…, depth…, parent…, subtreeSize…]`,
   * with `-1` marking a forest root in the parent column.
   */
  computeBubbleDataFromEdges(
    containmentEdges: Uint32Array,
    rootId: number,
    baseRadius: number,
    padding: number,
  ): Float32Array;
}

/** Bubble radius constants handed to the WASM derivation. */
export interface HierarchyDerivationConfig {
  /** Base radius for leaf slots. */
  readonly baseRadius: number;
  /** Padding added to internal-node radii. */
  readonly padding: number;
  /** Slot to root first, or {@link HIERARCHY_ROOT} to let the edges decide. */
  readonly rootId?: number | undefined;
}

/**
 * Edge type naming containment. The typed fast path maps its
 * `containmentKind` index onto this string so both input paths converge.
 */
export const CONTAINMENT_EDGE_TYPE = "contains";

/**
 * The force-config fields the well-radius GPU upload is derived from.
 *
 * Structural, so it matches `FullForceConfig` without importing it.
 */
export interface BubbleUploadConfig {
  readonly relativityBubbleMode: boolean;
  readonly relativityBubbleBaseRadius: number;
  readonly relativityBubblePadding: number;
}

/**
 * Whether a force-config change invalidates the well-radius upload.
 *
 * Almost every force parameter reaches the GPU through the per-tick uniform
 * buffer, so setting it is enough. These three do not: they decide what a
 * load-time buffer write contains. Enabling bubble mode after load therefore
 * used to flip a shader flag over an all-zero radius buffer — the setting
 * appeared to apply and changed nothing.
 */
export function bubbleUploadChanged(
  previous: BubbleUploadConfig,
  next: BubbleUploadConfig,
): boolean {
  return previous.relativityBubbleMode !== next.relativityBubbleMode ||
    previous.relativityBubbleBaseRadius !== next.relativityBubbleBaseRadius ||
    previous.relativityBubblePadding !== next.relativityBubblePadding;
}

/**
 * Select the containment edges the hierarchy is derived from.
 *
 * When the graph carries typed edges and *any* of them is
 * {@link CONTAINMENT_EDGE_TYPE}, only those are used: a cross-cutting
 * dependency edge (an import, a test reference) otherwise reparents its target
 * and skews every column. A graph with no containment typing has no way to
 * distinguish them, so every edge is treated as containment.
 *
 * @returns Interleaved `[parent0, child0, parent1, child1, ...]` slot pairs.
 */
export function selectContainmentEdges(
  edgeSources: Uint32Array,
  edgeTargets: Uint32Array,
  edgeCount: number,
  edgeTypes: readonly (string | undefined)[],
): Uint32Array {
  let hasContainment = false;
  for (let i = 0; i < edgeCount; i++) {
    if (edgeTypes[i] === CONTAINMENT_EDGE_TYPE) {
      hasContainment = true;
      break;
    }
  }

  const pairs = new Uint32Array(edgeCount * 2);
  let kept = 0;
  for (let i = 0; i < edgeCount; i++) {
    if (hasContainment && edgeTypes[i] !== CONTAINMENT_EDGE_TYPE) continue;
    pairs[kept * 2] = edgeSources[i];
    pairs[kept * 2 + 1] = edgeTargets[i];
    kept++;
  }
  return pairs.subarray(0, kept * 2);
}

/**
 * Decode the WASM `[wellRadius…, depth…, parent…, subtreeSize…]` block.
 *
 * The integer columns arrive `f32`-encoded (one tree walk, one copy across the
 * boundary); slot indices and subtree sizes round-trip exactly through `f32` up
 * to 2^24, three orders of magnitude above the 220K target. `-1` in the parent
 * column becomes {@link HIERARCHY_ROOT}.
 *
 * The column stride is the WASM engine's node bound, which may exceed the
 * caller's `nodeCount` when the engine holds trailing free slots. It is read
 * from the block length rather than assumed equal to `nodeCount`: assuming
 * would silently shift every column but the first. In that case a parent
 * pointing past `nodeCount` is truncated to a root, since the referenced slot
 * is not part of the retained graph.
 */
export function decodeHierarchyColumns(
  data: Float32Array,
  nodeCount: number,
): HierarchyColumns {
  const stride = Math.floor(data.length / HIERARCHY_WASM_COLUMNS);
  if (stride < nodeCount || data.length % HIERARCHY_WASM_COLUMNS !== 0) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `Bubble data has ${data.length} floats, expected a multiple of ` +
        `${HIERARCHY_WASM_COLUMNS} at least ${nodeCount * HIERARCHY_WASM_COLUMNS} long ` +
        `(${HIERARCHY_WASM_COLUMNS} columns x ${nodeCount} slots)`,
    );
  }

  const wellRadius = new Float32Array(nodeCount);
  const depth = new Uint16Array(nodeCount);
  const parent = new Uint32Array(nodeCount);
  const subtreeSize = new Uint32Array(nodeCount);

  for (let i = 0; i < nodeCount; i++) {
    wellRadius[i] = data[i];
    const d = data[stride + i];
    depth[i] = d >= MAX_HIERARCHY_DEPTH ? MAX_HIERARCHY_DEPTH : d;
    // Outside the retained slot space is a root: -1 is the Rust root sentinel,
    // and anything at or above nodeCount is a trailing engine slot.
    const p = data[stride * 2 + i];
    parent[i] = p >= 0 && p < nodeCount ? p : HIERARCHY_ROOT;
    subtreeSize[i] = data[stride * 3 + i];
  }

  return { parent, wellRadius, depth, subtreeSize };
}

/**
 * Build the parent→children CSR from `parent[]`.
 *
 * Counting sort, O(nodeCount). Children come out in ascending slot order, so
 * the index is a pure function of `parent[]` — a supplied hierarchy and a
 * derived one produce byte-identical CSRs.
 */
export function buildChildrenCsr(
  parent: Uint32Array,
  nodeCount: number,
): ChildrenCsr {
  const offsets = new Uint32Array(nodeCount + 1);
  let childTotal = 0;
  for (let i = 0; i < nodeCount; i++) {
    const p = parent[i];
    if (p === HIERARCHY_ROOT) continue;
    offsets[p + 1]++;
    childTotal++;
  }
  for (let i = 0; i < nodeCount; i++) {
    offsets[i + 1] += offsets[i];
  }

  const children = new Uint32Array(childTotal);
  const cursor = offsets.slice(0, nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    const p = parent[i];
    if (p === HIERARCHY_ROOT) continue;
    children[cursor[p]++] = i;
  }

  return { offsets, children };
}

/**
 * Reject supplied columns that cannot describe a forest over `nodeCount` slots.
 *
 * Checked, all O(nodeCount): column lengths; parents in range; no self-parent;
 * `depth[c] === depth[parent[c]] + 1` (which also rules out cycles, since no
 * cycle can carry strictly increasing depths); and subtree sizes summing over
 * the roots to exactly `nodeCount`.
 *
 * Supplied columns are load-bearing for layout, LOD and hit testing, and a
 * wrong one fails silently and far from its cause — so this throws rather than
 * falling back to derivation.
 *
 * @throws GraphMotherError when the columns are not a well-formed forest.
 */
export function validateHierarchyColumns(
  columns: HierarchyColumns,
  nodeCount: number,
): void {
  const lengths: ReadonlyArray<readonly [string, number]> = [
    ["parent", columns.parent.length],
    ["wellRadius", columns.wellRadius.length],
    ["depth", columns.depth.length],
    ["subtreeSize", columns.subtreeSize.length],
  ];
  for (const [name, length] of lengths) {
    if (length !== nodeCount) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        `hierarchy.${name} has ${length} entries, expected nodeCount (${nodeCount})`,
      );
    }
  }
  if (columns.centroid && columns.centroid.length !== nodeCount * 2) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `hierarchy.centroid has ${columns.centroid.length} entries, expected nodeCount * 2 ` +
        `(${nodeCount * 2})`,
    );
  }

  let rootSubtreeTotal = 0;
  for (let i = 0; i < nodeCount; i++) {
    const p = columns.parent[i];
    if (p === HIERARCHY_ROOT) {
      if (columns.depth[i] !== 0) {
        throw new GraphMotherError(
          ErrorCode.INVALID_GRAPH_DATA,
          `hierarchy: root slot ${i} has depth ${columns.depth[i]}, expected 0`,
        );
      }
      rootSubtreeTotal += columns.subtreeSize[i];
      continue;
    }
    if (p >= nodeCount) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        `hierarchy: slot ${i} has parent ${p}, out of range for nodeCount ${nodeCount} ` +
          `(use HIERARCHY_ROOT = 0xFFFFFFFF for a root)`,
      );
    }
    if (p === i) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        `hierarchy: slot ${i} is its own parent`,
      );
    }
    if (columns.depth[i] !== columns.depth[p] + 1) {
      throw new GraphMotherError(
        ErrorCode.INVALID_GRAPH_DATA,
        `hierarchy: slot ${i} has depth ${columns.depth[i]} under parent ${p} of depth ` +
          `${columns.depth[p]}; a containment cycle or an inconsistent depth column`,
      );
    }
  }

  if (rootSubtreeTotal !== nodeCount) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `hierarchy: subtree sizes over the forest roots sum to ${rootSubtreeTotal}, ` +
        `expected nodeCount (${nodeCount})`,
    );
  }
}

/**
 * Validate supplied columns and pair them with their children CSR.
 *
 * @throws GraphMotherError when the columns are not a well-formed forest.
 */
export function retainSuppliedHierarchy(
  columns: HierarchyColumns,
  nodeCount: number,
): RetainedHierarchy {
  validateHierarchyColumns(columns, nodeCount);
  return {
    nodeCount,
    columns,
    children: buildChildrenCsr(columns.parent, nodeCount),
    source: "supplied",
  };
}

/**
 * Derive the hierarchy in WASM from containment-only edges.
 *
 * The Rust side produces a spanning forest over the whole dense slot space:
 * every component root gets a real tree, so orphan slots are not flattened to
 * depth 0 and left invisible to every tree-walk consumer.
 */
export function deriveHierarchy(
  deriver: HierarchyDeriver,
  containmentEdges: Uint32Array,
  nodeCount: number,
  config: HierarchyDerivationConfig,
): RetainedHierarchy {
  const data = deriver.computeBubbleDataFromEdges(
    containmentEdges,
    config.rootId ?? HIERARCHY_ROOT,
    config.baseRadius,
    config.padding,
  );
  const columns = decodeHierarchyColumns(data, nodeCount);
  return {
    nodeCount,
    columns,
    children: buildChildrenCsr(columns.parent, nodeCount),
    source: "derived",
  };
}
