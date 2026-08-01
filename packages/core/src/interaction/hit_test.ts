/**
 * Hit Testing
 *
 * Provides spatial queries for node and edge hit testing.
 *
 * Queries brute-force over the position provider (the CPU shadow of GPU
 * positions), which is the freshest CPU-side data available — node positions
 * live on the GPU and are read back periodically. An index would have to be
 * re-synced and rebuilt on every readback, which costs more than the
 * per-pointer-event scan it would replace.
 *
 * A provider that can expose its backing typed arrays ({@link NodeColumns},
 * {@link EdgeColumns}) is scanned through them directly; the per-node accessor
 * interface remains for providers that cannot.
 *
 * @module
 */

import type { EdgeId, NodeId, Vec2 } from "../types.ts";

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
 * Column-oriented node data for the zero-allocation scan path.
 *
 * Node `i` (for `0 <= i < count`) is at `(x[i], y[i])` with radius
 * `radii[i * radiusStride + radiusOffset]`. The provider must keep `count`
 * within the extent of every array; a scan does not re-check bounds per node.
 */
export interface NodeColumns {
  /** Number of leading slots to scan */
  readonly count: number;
  /** Node X positions */
  readonly x: Float32Array;
  /** Node Y positions */
  readonly y: Float32Array;
  /** Array holding per-node radii, possibly interleaved with other attributes */
  readonly radii: Float32Array;
  /** Distance in elements between consecutive nodes' radii */
  readonly radiusStride: number;
  /** Element offset of the radius within a node's stride */
  readonly radiusOffset: number;
  /**
   * Per-slot state flags, one word per node.
   *
   * A slot is skipped entirely when `(flags[i] & skipMask) !== 0`. This is what
   * keeps the pointer honest about what is on screen: a node the renderer culls
   * keeps its position and its full-size hit disc, so without the mask a click
   * inside a collapsed bubble resolves to some hidden descendant behind it
   * rather than to the bubble the user aimed at. Omit both fields to scan every
   * slot.
   */
  readonly flags?: Uint32Array | undefined;
  /** Flag bits that take a slot out of hit testing; see {@link NodeColumns.flags} */
  readonly skipMask?: number | undefined;
}

/**
 * Column-oriented edge data for the zero-allocation scan path.
 *
 * Edge `i` (for `0 <= i < count`) runs from node `sources[i]` to node
 * `targets[i]` and reports `i` as its edge ID.
 */
export interface EdgeColumns {
  /** Number of leading edges to scan */
  readonly count: number;
  /** Source node index per edge */
  readonly sources: Uint32Array;
  /** Target node index per edge */
  readonly targets: Uint32Array;
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
  /**
   * Whether a node can be hit at all, for the accessor path.
   *
   * The column path's {@link NodeColumns.flags} mask expressed per node, so
   * both paths answer the same question. Omitted, every node is hittable.
   */
  isNodeHittable?(nodeId: NodeId): boolean;
  /**
   * Optional fast path: the backing columns, scanned directly.
   *
   * A scan over `getNodeIds` + `getNodePosition` costs an iterator step, a
   * `Vec2` allocation and two more calls per node, which dominates the
   * arithmetic at graph scale. When this returns columns they are used
   * instead, and must describe the same nodes as the per-node accessors.
   * Returning `null` (e.g. before a graph is loaded) falls back to those.
   */
  getNodeColumns?(): NodeColumns | null;
}

/**
 * Edge data provider interface.
 */
export interface EdgeProvider {
  /** Get all edges as [edgeId, sourceId, targetId] tuples */
  getEdges(): Iterable<[EdgeId, NodeId, NodeId]>;
  /** Get edge count */
  getEdgeCount(): number;
  /**
   * Optional fast path over the backing columns; see
   * {@link PositionProvider.getNodeColumns}. Only used when the position
   * provider also supplies columns, since an edge scan needs both.
   */
  getEdgeColumns?(): EdgeColumns | null;
}

/** Whether slot `i` is masked out of the scan by {@link NodeColumns.flags}. */
function isSkipped(columns: NodeColumns, i: number): boolean {
  const { flags, skipMask } = columns;
  return flags !== undefined && skipMask !== undefined && (flags[i] & skipMask) !== 0;
}

/**
 * Hit tester for graph elements.
 *
 * Node hit testing scans the position provider.
 * Edge hit testing uses line-point distance calculations.
 */
export class HitTester {
  private readonly config: Required<HitTesterConfig>;
  private positionProvider: PositionProvider | null = null;
  private edgeProvider: EdgeProvider | null = null;

  constructor(config: HitTesterConfig = {}) {
    this.config = { ...DEFAULT_HIT_TESTER_CONFIG, ...config };
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
    hitRadius?: number,
  ): NodeHitResult | null {
    const radius = hitRadius ?? this.config.nodeHitRadius;

    if (this.positionProvider) {
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
    hitRadius?: number,
  ): EdgeHitResult | null {
    if (!this.edgeProvider || !this.positionProvider) {
      return null;
    }

    const radius = hitRadius ?? this.config.edgeHitRadius;
    const nodeColumns = this.positionProvider.getNodeColumns?.() ?? null;
    const edgeColumns = this.edgeProvider.getEdgeColumns?.() ?? null;
    if (nodeColumns && edgeColumns) {
      return this.columnEdgeHitTest(graphX, graphY, radius, nodeColumns, edgeColumns);
    }

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
        targetPos.y,
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
    edgeRadius?: number,
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
    maxY: number,
  ): NodeId[] {
    if (!this.positionProvider) return [];

    const results: NodeId[] = [];
    const columns = this.positionProvider.getNodeColumns?.() ?? null;
    if (columns) {
      const { count, x, y } = columns;
      for (let i = 0; i < count; i++) {
        if (isSkipped(columns, i)) continue;
        const px = x[i];
        const py = y[i];
        if (px >= minX && px <= maxX && py >= minY && py <= maxY) {
          results.push(i);
        }
      }
      return results;
    }

    for (const nodeId of this.positionProvider.getNodeIds()) {
      if (this.positionProvider.isNodeHittable?.(nodeId) === false) continue;
      const pos = this.positionProvider.getNodePosition(nodeId);
      if (pos && pos.x >= minX && pos.x <= maxX && pos.y >= minY && pos.y <= maxY) {
        results.push(nodeId);
      }
    }
    return results;
  }

  /**
   * Node hit test over the provider's columns: no iterator, no per-node
   * allocation, no per-node accessor calls.
   */
  private columnNodeHitTest(
    graphX: number,
    graphY: number,
    columns: NodeColumns,
  ): NodeHitResult | null {
    const { count, x, y, radii, radiusStride, radiusOffset } = columns;

    let closestId = -1;
    let closestDist = Infinity;

    for (let i = 0; i < count; i++) {
      if (isSkipped(columns, i)) continue;
      const dx = graphX - x[i];
      const dy = graphY - y[i];
      const dist = Math.sqrt(dx * dx + dy * dy);

      // Same 2-unit click tolerance as the generic path below.
      const hitRadius = radii[i * radiusStride + radiusOffset] + 2;

      if (dist <= hitRadius && dist < closestDist) {
        closestDist = dist;
        closestId = i;
      }
    }

    if (closestId < 0) return null;
    return {
      type: "node",
      nodeId: closestId,
      distance: closestDist,
      position: { x: x[closestId], y: y[closestId] },
    };
  }

  /**
   * Edge hit test over the providers' columns; see {@link columnNodeHitTest}.
   */
  private columnEdgeHitTest(
    graphX: number,
    graphY: number,
    maxRadius: number,
    nodes: NodeColumns,
    edges: EdgeColumns,
  ): EdgeHitResult | null {
    const { x, y, count: nodeCount } = nodes;
    const { count, sources, targets } = edges;

    let closestId = -1;
    let closestDistance = maxRadius;

    for (let i = 0; i < count; i++) {
      const sourceId = sources[i];
      const targetId = targets[i];
      // The position arrays commonly extend past `count` (capacity beyond the
      // high-water mark), so an endpoint past it reads a real but meaningless
      // coordinate rather than producing NaN. Skip it, as the accessor path
      // does when `getNodePosition` returns undefined.
      if (sourceId >= nodeCount || targetId >= nodeCount) continue;

      const distance = this.pointToLineDistance(
        graphX,
        graphY,
        x[sourceId],
        y[sourceId],
        x[targetId],
        y[targetId],
      );

      if (distance < closestDistance) {
        closestDistance = distance;
        closestId = i;
      }
    }

    if (closestId < 0) return null;
    return {
      type: "edge",
      edgeId: closestId,
      distance: closestDistance,
      sourceId: sources[closestId],
      targetId: targets[closestId],
    };
  }

  /**
   * Brute force node hit test over the position provider.
   */
  private bruteForceNodeHitTest(
    graphX: number,
    graphY: number,
    maxRadius: number,
  ): NodeHitResult | null {
    if (!this.positionProvider) return null;

    const columns = this.positionProvider.getNodeColumns?.() ?? null;
    if (columns) return this.columnNodeHitTest(graphX, graphY, columns);

    let closestNode: NodeHitResult | null = null;
    let closestDist = Infinity;

    for (const nodeId of this.positionProvider.getNodeIds()) {
      if (this.positionProvider.isNodeHittable?.(nodeId) === false) continue;
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
    y2: number,
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
