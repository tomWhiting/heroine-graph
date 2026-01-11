/**
 * Graph Algorithms WASM API Contract
 *
 * This file defines the TypeScript interface for the graph algorithms
 * exposed by the WASM module. These are the methods added to HeroineGraphWasm.
 *
 * @module heroine-graph-wasm/algorithms
 */

// =============================================================================
// Enums & Types
// =============================================================================

export type CommunityAlgorithm = 'louvain' | 'leiden';
export type HullType = 'convex' | 'concave';
export type CentralityType =
  | 'pagerank'
  | 'betweenness'
  | 'closeness'
  | 'eigenvector'
  | 'degree'
  | 'katz';
export type ComponentType = 'weak' | 'strong';

// =============================================================================
// Configuration Interfaces
// =============================================================================

export interface CommunityDetectionConfig {
  /** Algorithm to use (default: 'louvain') */
  algorithm?: CommunityAlgorithm;
  /** Resolution parameter - higher values produce more communities (default: 1.0) */
  resolution?: number;
  /** Whether to use edge weights in computation (default: false) */
  weighted?: boolean;
  /** Maximum optimization iterations (default: 100) */
  maxIterations?: number;
  /** Stop when modularity gain is below this threshold (default: 0.0001) */
  minModularityGain?: number;
}

export interface HullComputationConfig {
  /** Type of hull to compute (default: 'convex') */
  hullType?: HullType;
  /** Concavity parameter for concave hulls - higher = tighter (default: 2.0) */
  concavity?: number;
  /** Fallback circle radius for communities with 1-2 nodes (default: 10.0) */
  fallbackRadius?: number;
}

export interface CentralityConfig {
  /** Type of centrality to compute */
  type: CentralityType;
  /** Whether to normalize scores to [0, 1] (default: true) */
  normalized?: boolean;
  /** Maximum iterations for iterative algorithms (default: 100) */
  maxIterations?: number;
  /** Convergence tolerance (default: 1e-6) */
  tolerance?: number;
  /** PageRank damping factor (default: 0.85) */
  damping?: number;
}

export interface BoundaryPhysicsConfig {
  /** Whether boundary collision is enabled (default: true) */
  enabled?: boolean;
  /** Repulsion force strength 0.0-1.0 (default: 0.5) */
  repulsionStrength?: number;
  /** Velocity damping per frame (default: 0.9) */
  damping?: number;
  /** Maximum displacement per update (default: 10.0) */
  maxDisplacement?: number;
}

// =============================================================================
// Result Interfaces
// =============================================================================

export interface Community {
  /** Unique community identifier */
  id: number;
  /** Node IDs belonging to this community */
  members: Uint32Array;
  /** Community's contribution to total modularity */
  modularity: number;
}

export interface CommunityAssignment {
  /** Map from node ID to community ID */
  nodeToCommunity: Map<number, number>;
  /** Full community details */
  communities: Community[];
  /** Overall modularity score of the partition */
  totalModularity: number;
  /** Algorithm that was used */
  algorithm: CommunityAlgorithm;
}

export interface CommunityBoundary {
  /** ID of the community this bounds */
  communityId: number;
  /** Type of hull computed */
  hullType: HullType;
  /** Polygon vertices as [x0, y0, x1, y1, ...] */
  vertices: Float32Array;
  /** Center point [x, y] */
  centroid: [number, number];
}

export interface CentralityResult {
  /** Type of centrality computed */
  type: CentralityType;
  /** Map from node ID to score */
  scores: Map<number, number>;
  /** Minimum score across all nodes */
  min: number;
  /** Maximum score across all nodes */
  max: number;
  /** Mean score across all nodes */
  mean: number;
}

/** Bulk format for efficient GPU upload */
export interface CentralityResultBulk {
  /** Type of centrality computed */
  type: CentralityType;
  /** Node IDs in order */
  nodeIds: Uint32Array;
  /** Scores corresponding to nodeIds */
  scores: Float32Array;
  /** Minimum score */
  min: number;
  /** Maximum score */
  max: number;
  /** Mean score */
  mean: number;
}

export interface Component {
  /** Component identifier */
  id: number;
  /** Node IDs in this component */
  members: Uint32Array;
}

export interface ComponentResult {
  /** Type of connectivity analyzed */
  type: ComponentType;
  /** List of all components */
  components: Component[];
  /** Map from node ID to component ID */
  nodeToComponent: Map<number, number>;
}

export interface BoundaryPhysicsResult {
  /** Node IDs that were displaced */
  nodeIds: Uint32Array;
  /** X displacement for each node */
  displacementsX: Float32Array;
  /** Y displacement for each node */
  displacementsY: Float32Array;
  /** Whether any boundaries still overlap */
  hasOverlaps: boolean;
  /** Current physics iteration count */
  iteration: number;
}

// =============================================================================
// Progress Callback
// =============================================================================

export interface AlgorithmProgress {
  /** Current phase of the algorithm */
  phase: string;
  /** Progress within phase (0.0 - 1.0) */
  progress: number;
  /** Optional status message */
  message?: string;
}

export type ProgressCallback = (progress: AlgorithmProgress) => void;

// =============================================================================
// API Methods (added to HeroineGraphWasm)
// =============================================================================

export interface GraphAlgorithmsAPI {
  // -------------------------------------------------------------------------
  // Community Detection (FR-001 to FR-006)
  // -------------------------------------------------------------------------

  /**
   * Detect communities in the graph using Louvain or Leiden algorithm.
   *
   * @param config - Configuration options
   * @param onProgress - Optional progress callback for large graphs
   * @returns Community assignments for all nodes
   *
   * @example
   * const result = graph.detectCommunities({ resolution: 1.0 });
   * console.log(`Found ${result.communities.length} communities`);
   */
  detectCommunities(
    config?: CommunityDetectionConfig,
    onProgress?: ProgressCallback
  ): CommunityAssignment;

  /**
   * Get the community ID for a specific node.
   *
   * @param nodeId - The node to query
   * @returns Community ID or undefined if node doesn't exist
   */
  getNodeCommunity(nodeId: number): number | undefined;

  // -------------------------------------------------------------------------
  // Hull Computation (FR-007 to FR-011)
  // -------------------------------------------------------------------------

  /**
   * Compute boundary hulls for all communities.
   *
   * @param communities - Community assignment from detectCommunities()
   * @param config - Hull computation options
   * @returns Array of boundary polygons, one per community
   *
   * @example
   * const communities = graph.detectCommunities();
   * const hulls = graph.computeHulls(communities, { hullType: 'concave' });
   */
  computeHulls(
    communities: CommunityAssignment,
    config?: HullComputationConfig
  ): CommunityBoundary[];

  /**
   * Compute hull for a single community.
   *
   * @param nodeIds - Node IDs in the community
   * @param config - Hull computation options
   * @returns Boundary polygon
   */
  computeHull(nodeIds: Uint32Array, config?: HullComputationConfig): CommunityBoundary;

  // -------------------------------------------------------------------------
  // Boundary Physics (FR-012 to FR-016)
  // -------------------------------------------------------------------------

  /**
   * Initialize boundary physics state.
   *
   * @param boundaries - Community boundaries from computeHulls()
   * @param config - Physics configuration
   */
  initBoundaryPhysics(
    boundaries: CommunityBoundary[],
    config?: BoundaryPhysicsConfig
  ): void;

  /**
   * Update boundary physics for one frame.
   *
   * @returns Displacement vectors to apply to nodes
   *
   * @example
   * const result = graph.updateBoundaryPhysics();
   * if (result.hasOverlaps) {
   *   // Apply displacements to node positions
   * }
   */
  updateBoundaryPhysics(): BoundaryPhysicsResult;

  /**
   * Configure boundary physics.
   *
   * @param config - New configuration (partial updates allowed)
   */
  setBoundaryPhysicsConfig(config: Partial<BoundaryPhysicsConfig>): void;

  /**
   * Check if boundary physics is currently enabled.
   */
  isBoundaryPhysicsEnabled(): boolean;

  // -------------------------------------------------------------------------
  // Centrality Measures (FR-017 to FR-022)
  // -------------------------------------------------------------------------

  /**
   * Compute centrality scores for all nodes.
   *
   * @param config - Centrality configuration (type is required)
   * @param onProgress - Optional progress callback
   * @returns Centrality scores for all nodes
   *
   * @example
   * const result = graph.computeCentrality({ type: 'pagerank' });
   * const mostImportant = [...result.scores.entries()]
   *   .sort((a, b) => b[1] - a[1])[0];
   */
  computeCentrality(
    config: CentralityConfig,
    onProgress?: ProgressCallback
  ): CentralityResult;

  /**
   * Compute centrality and return in bulk format for GPU upload.
   *
   * @param config - Centrality configuration
   * @returns Bulk format with TypedArrays
   */
  computeCentralityBulk(config: CentralityConfig): CentralityResultBulk;

  // -------------------------------------------------------------------------
  // Connected Components (FR-023 to FR-025)
  // -------------------------------------------------------------------------

  /**
   * Find connected components in the graph.
   *
   * @returns Component assignments for all nodes
   *
   * @example
   * const result = graph.getConnectedComponents();
   * console.log(`Graph has ${result.components.length} connected components`);
   */
  getConnectedComponents(): ComponentResult;

  /**
   * Find strongly connected components in a directed graph.
   *
   * @returns SCC assignments for all nodes
   */
  getStronglyConnectedComponents(): ComponentResult;

  /**
   * Get the component ID for a specific node.
   *
   * @param nodeId - The node to query
   * @param type - Type of connectivity
   * @returns Component ID or undefined if node doesn't exist
   */
  getNodeComponent(nodeId: number, type?: ComponentType): number | undefined;
}
