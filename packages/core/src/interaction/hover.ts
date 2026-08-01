/**
 * Hover Tracking
 *
 * Owns the node/edge hover state machine that sits between pointer motion and
 * hover events. Two properties are load-bearing at graph scale, because every
 * evaluation is a full brute-force scan (see `hit_test.ts`):
 *
 * - Pointer moves are **coalesced**: a browser delivers `pointermove` far more
 *   often than it paints, and only the last position of a frame can be under
 *   the pointer when that frame is shown.
 * - The edge scan runs **only when something is listening for edge hover**. It
 *   is the larger half of the cost and it runs in the common case — pointer
 *   over empty canvas, no node hit.
 *
 * @module
 */

import type { EdgeId, NodeId, Vec2 } from "../types.ts";

/**
 * Everything the tracker needs from its host: the scans, the coordinate
 * transform, the edge-scan gate, and the four transition callbacks.
 */
export interface HoverTargets {
  /** Node under the pointer, or null. */
  nodeAt(screenX: number, screenY: number): NodeId | null;
  /** Edge under the pointer, or null. Called only when {@link edgeHoverWanted} is true. */
  edgeAt(screenX: number, screenY: number): EdgeId | null;
  /** Screen to graph coordinates, for the position carried by enter events. */
  toGraph(screenX: number, screenY: number): Vec2;
  /** Whether any consumer observes edge hover; false skips the edge scan entirely. */
  edgeHoverWanted(): boolean;
  onNodeEnter(nodeId: NodeId, position: Vec2): void;
  onNodeLeave(nodeId: NodeId): void;
  onEdgeEnter(edgeId: EdgeId, position: Vec2): void;
  onEdgeLeave(edgeId: EdgeId): void;
}

/**
 * Defers a coalesced hover evaluation. Invoked at most once per pending
 * evaluation; the callback may run synchronously.
 */
export type HoverScheduler = (run: () => void) => void;

/**
 * Coalesce to one evaluation per animation frame where there is a frame clock,
 * and evaluate immediately where there is not (headless embedders and tests).
 */
export const defaultHoverScheduler: HoverScheduler = (run) => {
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(() => run());
  } else {
    run();
  }
};

/**
 * Node/edge hover state machine.
 */
export class HoverTracker {
  private readonly targets: HoverTargets;
  private readonly schedule: HoverScheduler;

  private hoveredNodeId: NodeId | null = null;
  private hoveredEdgeId: EdgeId | null = null;

  private pendingX = 0;
  private pendingY = 0;
  private hasPending = false;
  private scheduled = false;

  constructor(targets: HoverTargets, schedule: HoverScheduler = defaultHoverScheduler) {
    this.targets = targets;
    this.schedule = schedule;
  }

  /** Currently hovered node, or null. */
  get node(): NodeId | null {
    return this.hoveredNodeId;
  }

  /** Currently hovered edge, or null. */
  get edge(): EdgeId | null {
    return this.hoveredEdgeId;
  }

  /**
   * Record the pointer position. The evaluation is coalesced: any moves
   * arriving before the scheduled evaluation runs replace this one.
   */
  move(screenX: number, screenY: number): void {
    this.pendingX = screenX;
    this.pendingY = screenY;
    this.hasPending = true;

    if (this.scheduled) return;
    this.scheduled = true;
    this.schedule(() => this.flush());
  }

  /**
   * Drop hover state and any pending evaluation without emitting transitions.
   * For when the hovered slots stop meaning what they meant (graph reload,
   * disposal).
   */
  reset(): void {
    this.hoveredNodeId = null;
    this.hoveredEdgeId = null;
    this.hasPending = false;
  }

  private flush(): void {
    this.scheduled = false;
    if (!this.hasPending) return;
    this.hasPending = false;
    this.evaluate(this.pendingX, this.pendingY);
  }

  private evaluate(screenX: number, screenY: number): void {
    const nodeId = this.targets.nodeAt(screenX, screenY);
    const position = this.targets.toGraph(screenX, screenY);

    if (nodeId !== this.hoveredNodeId) {
      // Leave is reported while the tracker still reads as hovering the old
      // target, and enter once it reads as hovering the new one.
      if (this.hoveredNodeId !== null) this.targets.onNodeLeave(this.hoveredNodeId);
      this.hoveredNodeId = nodeId;
      if (nodeId !== null) this.targets.onNodeEnter(nodeId, position);
    }

    // A node under the pointer occludes edges, so the scan is pointless there.
    const edgeId = nodeId === null && this.targets.edgeHoverWanted()
      ? this.targets.edgeAt(screenX, screenY)
      : null;

    if (edgeId !== this.hoveredEdgeId) {
      if (this.hoveredEdgeId !== null) this.targets.onEdgeLeave(this.hoveredEdgeId);
      this.hoveredEdgeId = edgeId;
      if (edgeId !== null) this.targets.onEdgeEnter(edgeId, position);
    }
  }
}
