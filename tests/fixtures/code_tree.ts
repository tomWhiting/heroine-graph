/**
 * Deterministic Code-Tree Fixture Generator
 *
 * Produces code-repository-shaped graphs: a hierarchical containment tree
 * (dir -> file -> symbol) plus cross-cutting dependency edges (imports).
 * This mirrors the library's primary workload (CLAUDE.md: Meridian-style
 * repo graphs) at parameterized scale.
 *
 * Everything is derived from a seeded PRNG, so the same options always
 * produce byte-identical graphs — suitable both for unit tests and as the
 * rig future physics work is tuned against.
 *
 * @module
 */

import type { EdgeInput, GraphInput, NodeInput } from "../../packages/core/src/types.ts";
import { mulberry32, randInt } from "./prng.ts";

/**
 * Options for code-tree generation
 */
export interface CodeTreeOptions {
  /** PRNG seed (default: 1) */
  seed?: number;
  /** Maximum tree depth; root is depth 0 (default: 4) */
  maxDepth?: number;
  /** Minimum children per internal node (default: 2) */
  minChildren?: number;
  /** Maximum children per internal node (default: 5) */
  maxChildren?: number;
  /** Hard cap on total node count (default: 500) */
  maxNodes?: number;
  /** Cross-cutting "imports" edges as a fraction of node count (default: 0.1) */
  crossEdgeRatio?: number;
  /** Scale of the initial seeded position disc (default: 10) */
  spread?: number;
}

/**
 * Generated code-tree graph with both API-shaped input and raw typed arrays.
 *
 * Node ids are "n{slot}" in slot order, so parser indices equal generator
 * slots and the typed arrays can be fed straight to the GPU harness.
 */
export interface CodeTreeGraph {
  /** GraphInput-shaped data for the parsers / full API */
  input: GraphInput;
  nodeCount: number;
  edgeCount: number;
  /** Leading edges [0, containmentEdgeCount) are "contains"; the rest are "imports" */
  containmentEdgeCount: number;
  /** Parent slot per node; -1 for the root */
  parent: Int32Array;
  /** BFS depth per node (root = 0), f32 for direct GPU upload */
  depths: Float32Array;
  /** Deterministic initial positions (seeded disc) */
  positionsX: Float32Array;
  positionsY: Float32Array;
  /** Edge endpoints as slot indices */
  edgeSources: Uint32Array;
  edgeTargets: Uint32Array;
}

/**
 * Node type by depth: directories down to maxDepth-2, files at maxDepth-1,
 * symbols at maxDepth.
 */
function nodeTypeForDepth(depth: number, maxDepth: number): string {
  if (depth >= maxDepth) return "symbol";
  if (depth === maxDepth - 1) return "file";
  return "dir";
}

/**
 * Generates a deterministic code-tree-shaped graph.
 */
export function generateCodeTree(options: CodeTreeOptions = {}): CodeTreeGraph {
  const seed = options.seed ?? 1;
  const maxDepth = options.maxDepth ?? 4;
  const minChildren = options.minChildren ?? 2;
  const maxChildren = options.maxChildren ?? 5;
  const maxNodes = options.maxNodes ?? 500;
  const crossEdgeRatio = options.crossEdgeRatio ?? 0.1;
  const spread = options.spread ?? 10;

  const rng = mulberry32(seed);

  // --- Containment tree (BFS so slot order groups siblings contiguously,
  // matching how real code-tree loaders lay out their edge buffers) ---
  const parentList: number[] = [-1];
  const depthList: number[] = [0];
  const queue: number[] = [0];

  while (queue.length > 0 && parentList.length < maxNodes) {
    const slot = queue.shift()!;
    const depth = depthList[slot];
    if (depth >= maxDepth) continue;

    const childCount = randInt(rng, minChildren, maxChildren);
    for (let c = 0; c < childCount && parentList.length < maxNodes; c++) {
      const childSlot = parentList.length;
      parentList.push(slot);
      depthList.push(depth + 1);
      queue.push(childSlot);
    }
  }

  const nodeCount = parentList.length;
  const parent = Int32Array.from(parentList);
  const depths = Float32Array.from(depthList);

  // --- Deterministic initial positions: uniform disc scaled by node count ---
  const discRadius = spread * Math.sqrt(nodeCount);
  const positionsX = new Float32Array(nodeCount);
  const positionsY = new Float32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    const r = discRadius * Math.sqrt(rng());
    const theta = 2 * Math.PI * rng();
    positionsX[i] = r * Math.cos(theta);
    positionsY[i] = r * Math.sin(theta);
  }

  // --- Edges: containment first, then cross-cutting imports ---
  const containmentEdgeCount = nodeCount - 1;
  const crossEdgeCount = nodeCount > 1 ? Math.round(crossEdgeRatio * nodeCount) : 0;
  const edgeCount = containmentEdgeCount + crossEdgeCount;

  const edgeSources = new Uint32Array(edgeCount);
  const edgeTargets = new Uint32Array(edgeCount);

  for (let i = 1; i < nodeCount; i++) {
    edgeSources[i - 1] = parent[i];
    edgeTargets[i - 1] = i;
  }
  for (let k = 0; k < crossEdgeCount; k++) {
    const src = randInt(rng, 0, nodeCount - 1);
    let tgt = randInt(rng, 0, nodeCount - 2);
    if (tgt >= src) tgt += 1; // never self-referencing
    edgeSources[containmentEdgeCount + k] = src;
    edgeTargets[containmentEdgeCount + k] = tgt;
  }

  // --- GraphInput view (ids "n{slot}" so parser indices == slots) ---
  const nodes: NodeInput[] = new Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    nodes[i] = {
      id: `n${i}`,
      x: positionsX[i],
      y: positionsY[i],
      type: nodeTypeForDepth(depthList[i], maxDepth),
    };
  }

  const edges: EdgeInput[] = new Array(edgeCount);
  for (let e = 0; e < edgeCount; e++) {
    edges[e] = {
      source: `n${edgeSources[e]}`,
      target: `n${edgeTargets[e]}`,
      type: e < containmentEdgeCount ? "contains" : "imports",
    };
  }

  return {
    input: { nodes, edges },
    nodeCount,
    edgeCount,
    containmentEdgeCount,
    parent,
    depths,
    positionsX,
    positionsY,
    edgeSources,
    edgeTargets,
  };
}
