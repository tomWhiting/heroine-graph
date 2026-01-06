/**
 * Hit Testing via WASM R-tree
 *
 * Provides efficient spatial queries for node and edge hit testing
 * using the WASM-based R-tree spatial index.
 *
 * @module
 */

import type { NodeId, EdgeId, Vec2 } from "../types.ts";

/**
 * Hit test result for a node.
 */
export interface NodeHitResult {
  /** Type discriminator */
  readonly type: "node";
  /** The hit node ID */
  readonly nodeId: NodeId;
  /** Distance from query point to node center */
  readonly distance: number;
  /** The node's position */
  readonly position: Vec2;
}

/**
 * Hit test result for an edge.
 */
export interface EdgeHitResult {
  /** Type discriminator */
  readonly type: "edge";
  /** The hit edge ID */
  readonly edgeId: EdgeId;
  /** Distance from query point to edge line */
  readonly distance: number;
  /** Source node ID */
  readonly sourceId: NodeId;
  /** Target node ID */
  readonly targetId: NodeId;
}

/**
 * Combined hit test result.
 */
export type HitResult = NodeHitResult | EdgeHitResult;

/**
 * Hit tester configuration.
 */
export interface HitTesterConfig {
  /** Default hit radius for nodes (in graph units) */
  nodeHitRadius?: number;
  /** Default hit radius for edges (in graph units) */
  edgeHitRadius?: number;
  /** Prioritize nodes over edges when both are hit */
  prioritizeNodes?: boolean;
}

/**
 * Default hit tester configuration.
 */
export const DEFAULT_HIT_TESTER_CONFIG: Required<HitTesterConfig> = {
  nodeHitRadius: 20,
  edgeHitRadius: 5,
  prioritizeNodes: true,
};

/**
 * Interface for WASM spatial queries.
 * This matches the HeroineGraphWasm API.
 */
export interface SpatialQueryEngine {
  /** Find nearest node within distance */
  findNearestNodeWithin(x: number, y: number, maxDistance: number): number | undefined;
  /** Find all nodes in rectangle */
  findNodesInRect(minX: number, minY: number, maxX: number, maxY: number): number[];
  /** Rebuild spatial index */
  rebuildSpatialIndex(): void;
}

/**
 * Node position provider interface.
 */
export interface PositionProvider {
  /** Get node position by ID */
  getNodePosition(nodeId: NodeId): Vec2 | undefined;
  /** Get node radius by ID (optional - uses config default if not provided) */
  getNodeRadius?(nodeId: NodeId): number | undefined;
  /** Get all node IDs */
  getNodeIds(): Iterable<NodeId>;
  /** Get node count */
  getNodeCount(): number;
}

/**
 * Edge data provider interface.
 */
export interface EdgeProvider {
  /** Get all edges as [edgeId, sourceId, targetId] tuples */
  getEdges(): Iterable<[EdgeId, NodeId, NodeId]>;
  /** Get edge count */
  getEdgeCount(): number;
}

/**
 * Hit tester for graph elements.
 *
 * Uses a WASM R-tree for efficient node hit testing.
 * Edge hit testing uses line-point distance calculations.
 */
export class HitTester {
  private readonly config: Required<HitTesterConfig>;
  private spatialEngine: SpatialQueryEngine | null = null;
  private positionProvider: PositionProvider | null = null;
  private edgeProvider: EdgeProvider | null = null;

  constructor(config: HitTesterConfig = {}) {
    this.config = { ...DEFAULT_HIT_TESTER_CONFIG, ...config };
  }

  /**
   * Set the spatial query engine (WASM).
   */
  setSpatialEngine(engine: SpatialQueryEngine): void {
    this.spatialEngine = engine;
  }

  /**
   * Set the position provider.
   */
  setPositionProvider(provider: PositionProvider): void {
    this.positionProvider = provider;
  }

  /**
   * Set the edge provider.
   */
  setEdgeProvider(provider: EdgeProvider): void {
    this.edgeProvider = provider;
  }

  /**
   * Test for node hit at a position.
   *
   * @param graphX X position in graph coordinates
   * @param graphY Y position in graph coordinates
   * @param hitRadius Optional override for hit radius
   * @returns Node hit result or null if no hit
   */
  hitTestNode(
    graphX: number,
    graphY: number,
    hitRadius?: number
  ): NodeHitResult | null {
    const radius = hitRadius ?? this.config.nodeHitRadius;

    // Try WASM spatial index first (most efficient)
    if (this.spatialEngine) {
      const nodeId = this.spatialEngine.findNearestNodeWithin(graphX, graphY, radius);
      if (nodeId !== undefined) {
        const position = this.positionProvider?.getNodePosition(nodeId);
        if (position) {
          const dx = graphX - position.x;
          const dy = graphY - position.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          return {
            type: "node",
            nodeId,
            distance,
            position,
          };
        }
      }
    }

    // Fallback: brute force search if no spatial engine
    if (this.positionProvider && !this.spatialEngine) {
      return this.bruteForceNodeHitTest(graphX, graphY, radius);
    }

    return null;
  }

  /**
   * Test for edge hit at a position.
   *
   * @param graphX X position in graph coordinates
   * @param graphY Y position in graph coordinates
   * @param hitRadius Optional override for hit radius
   * @returns Edge hit result or null if no hit
   */
  hitTestEdge(
    graphX: number,
    graphY: number,
    hitRadius?: number
  ): EdgeHitResult | null {
    if (!this.edgeProvider || !this.positionProvider) {
      return null;
    }

    const radius = hitRadius ?? this.config.edgeHitRadius;
    let closestEdge: EdgeHitResult | null = null;
    let closestDistance = radius;

    for (const [edgeId, sourceId, targetId] of this.edgeProvider.getEdges()) {
      const sourcePos = this.positionProvider.getNodePosition(sourceId);
      const targetPos = this.positionProvider.getNodePosition(targetId);

      if (!sourcePos || !targetPos) continue;

      const distance = this.pointToLineDistance(
        graphX,
        graphY,
        sourcePos.x,
        sourcePos.y,
        targetPos.x,
        targetPos.y
      );

      if (distance < closestDistance) {
        closestDistance = distance;
        closestEdge = {
          type: "edge",
          edgeId,
          distance,
          sourceId,
          targetId,
        };
      }
    }

    return closestEdge;
  }

  /**
   * Test for any hit (node or edge) at a position.
   *
   * @param graphX X position in graph coordinates
   * @param graphY Y position in graph coordinates
   * @param nodeRadius Optional override for node hit radius
   * @param edgeRadius Optional override for edge hit radius
   * @returns Hit result or null if no hit
   */
  hitTest(
    graphX: number,
    graphY: number,
    nodeRadius?: number,
    edgeRadius?: number
  ): HitResult | null {
    const nodeHit = this.hitTestNode(graphX, graphY, nodeRadius);
    const edgeHit = this.hitTestEdge(graphX, graphY, edgeRadius);

    if (nodeHit && edgeHit) {
      // Both hit - return based on priority
      if (this.config.prioritizeNodes) {
        return nodeHit;
      }
      // Return whichever is closer
      return nodeHit.distance < edgeHit.distance ? nodeHit : edgeHit;
    }

    return nodeHit ?? edgeHit ?? null;
  }

  /**
   * Find all nodes in a rectangular region.
   *
   * @param minX Minimum X coordinate
   * @param minY Minimum Y coordinate
   * @param maxX Maximum X coordinate
   * @param maxY Maximum Y coordinate
   * @returns Array of node IDs in the region
   */
  findNodesInRect(
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): NodeId[] {
    if (this.spatialEngine) {
      return this.spatialEngine.findNodesInRect(minX, minY, maxX, maxY);
    }

    // Fallback: brute force
    if (this.positionProvider) {
      const results: NodeId[] = [];
      for (const nodeId of this.positionProvider.getNodeIds()) {
        const pos = this.positionProvider.getNodePosition(nodeId);
        if (pos && pos.x >= minX && pos.x <= maxX && pos.y >= minY && pos.y <= maxY) {
          results.push(nodeId);
        }
      }
      return results;
    }

    return [];
  }

  /**
   * Rebuild the spatial index.
   * Call this after bulk position updates.
   */
  rebuildIndex(): void {
    this.spatialEngine?.rebuildSpatialIndex();
  }

  /**
   * Brute force node hit test (fallback when no spatial engine).
   */
  private bruteForceNodeHitTest(
    graphX: number,
    graphY: number,
    maxRadius: number
  ): NodeHitResult | null {
    if (!this.positionProvider) return null;

    let closestNode: NodeHitResult | null = null;
    let closestDist = Infinity;

    for (const nodeId of this.positionProvider.getNodeIds()) {
      const pos = this.positionProvider.getNodePosition(nodeId);
      if (!pos) continue;

      const dx = graphX - pos.x;
      const dy = graphY - pos.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      // Use per-node radius if available, otherwise use maxRadius
      // Add a small tolerance (2 units) for easier clicking
      const nodeRadius = this.positionProvider.getNodeRadius?.(nodeId) ?? maxRadius;
      const hitRadius = nodeRadius + 2;

      if (dist <= hitRadius && dist < closestDist) {
        closestDist = dist;
        closestNode = {
          type: "node",
          nodeId,
          distance: dist,
          position: pos,
        };
      }
    }

    return closestNode;
  }

  /**
   * Calculate distance from a point to a line segment.
   */
  private pointToLineDistance(
    px: number,
    py: number,
    x1: number,
    y1: number,
    x2: number,
    y2: number
  ): number {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lengthSq = dx * dx + dy * dy;

    if (lengthSq === 0) {
      // Line segment is a point
      const ddx = px - x1;
      const ddy = py - y1;
      return Math.sqrt(ddx * ddx + ddy * ddy);
    }

    // Project point onto line, clamped to segment
    const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lengthSq));
    const nearestX = x1 + t * dx;
    const nearestY = y1 + t * dy;

    const distX = px - nearestX;
    const distY = py - nearestY;
    return Math.sqrt(distX * distX + distY * distY);
  }
}

/**
 * Create a hit tester instance.
 */
export function createHitTester(config?: HitTesterConfig): HitTester {
  return new HitTester(config);
}
