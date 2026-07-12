/**
 * Deterministic Clustered Graph Fixture Generator
 *
 * Produces a graph with planted community structure: k clusters, each
 * internally dense (spanning chain plus extra random intra-cluster edges),
 * connected to each other by only a handful of inter-cluster edges.
 *
 * Initial positions are seeded uniformly on a disc with NO correlation to
 * cluster membership, so any spatial cluster separation in a layout must
 * be produced by the force algorithm — this is the fixture used to detect
 * the "uniform circle with no structure" failure mode.
 *
 * Everything is derived from a seeded PRNG, so the same options always
 * produce byte-identical graphs.
 *
 * @module
 */

import { mulberry32, randInt } from "./prng.ts";

/**
 * Options for clustered graph generation
 */
export interface ClusteredGraphOptions {
  /** PRNG seed (default: 1) */
  seed?: number;
  /** Number of clusters (default: 4) */
  clusterCount?: number;
  /** Nodes per cluster (default: 40) */
  nodesPerCluster?: number;
  /** Extra random intra-cluster edges per node beyond the spanning chain (default: 2) */
  extraIntraEdgesPerNode?: number;
  /** Inter-cluster edges per cluster pair (default: 1) */
  interEdgesPerPair?: number;
  /** Scale of the initial seeded position disc (default: 10) */
  spread?: number;
}

/**
 * Generated clustered graph as slot-indexed typed arrays, feedable straight
 * to the GPU harness.
 */
export interface ClusteredGraph {
  nodeCount: number;
  edgeCount: number;
  clusterCount: number;
  /** Cluster label per node slot */
  labels: Uint32Array;
  /** Deterministic initial positions (seeded disc, uncorrelated with labels) */
  positionsX: Float32Array;
  positionsY: Float32Array;
  /** Edge endpoints as slot indices */
  edgeSources: Uint32Array;
  edgeTargets: Uint32Array;
}

/**
 * Generates a deterministic clustered graph.
 */
export function generateClusteredGraph(options: ClusteredGraphOptions = {}): ClusteredGraph {
  const seed = options.seed ?? 1;
  const clusterCount = options.clusterCount ?? 4;
  const nodesPerCluster = options.nodesPerCluster ?? 40;
  const extraIntraEdgesPerNode = options.extraIntraEdgesPerNode ?? 2;
  const interEdgesPerPair = options.interEdgesPerPair ?? 1;
  const spread = options.spread ?? 10;

  const rng = mulberry32(seed);
  const nodeCount = clusterCount * nodesPerCluster;

  const labels = new Uint32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    labels[i] = Math.floor(i / nodesPerCluster);
  }

  // Initial positions: uniform disc, uncorrelated with cluster membership.
  const discRadius = spread * Math.sqrt(nodeCount);
  const positionsX = new Float32Array(nodeCount);
  const positionsY = new Float32Array(nodeCount);
  for (let i = 0; i < nodeCount; i++) {
    const r = discRadius * Math.sqrt(rng());
    const theta = 2 * Math.PI * rng();
    positionsX[i] = r * Math.cos(theta);
    positionsY[i] = r * Math.sin(theta);
  }

  const sources: number[] = [];
  const targets: number[] = [];

  // Intra-cluster edges: spanning chain (guarantees connectivity) plus
  // extra random links for internal cohesion.
  for (let c = 0; c < clusterCount; c++) {
    const base = c * nodesPerCluster;
    for (let i = 1; i < nodesPerCluster; i++) {
      sources.push(base + i - 1);
      targets.push(base + i);
    }
    for (let i = 0; i < nodesPerCluster; i++) {
      for (let k = 0; k < extraIntraEdgesPerNode; k++) {
        let other = randInt(rng, 0, nodesPerCluster - 2);
        if (other >= i) other += 1; // never self-referencing
        sources.push(base + i);
        targets.push(base + other);
      }
    }
  }

  // Sparse inter-cluster edges between every cluster pair.
  for (let a = 0; a < clusterCount; a++) {
    for (let b = a + 1; b < clusterCount; b++) {
      for (let k = 0; k < interEdgesPerPair; k++) {
        sources.push(a * nodesPerCluster + randInt(rng, 0, nodesPerCluster - 1));
        targets.push(b * nodesPerCluster + randInt(rng, 0, nodesPerCluster - 1));
      }
    }
  }

  return {
    nodeCount,
    edgeCount: sources.length,
    clusterCount,
    labels,
    positionsX,
    positionsY,
    edgeSources: Uint32Array.from(sources),
    edgeTargets: Uint32Array.from(targets),
  };
}
