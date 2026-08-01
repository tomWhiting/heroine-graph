/**
 * Node Drag
 *
 * One gesture with two entry points: a pointer on a node's sprite, and a
 * pointer on the drag handle of the DOM card standing in for that node. Both
 * must take the same pin, write the same position, emit the same three events
 * and leave the node pinned where it was dropped — so both drive this, and
 * there is one implementation to get right rather than two to keep in step.
 *
 * Everything the drag touches arrives through {@link NodeDragHost}, including
 * the clock, so a gesture replays to the same event stream every time and the
 * two entry points are comparable in a test with no canvas and no DOM.
 *
 * @module
 */

import type {
  NodeDragEndEvent,
  NodeDragMoveEvent,
  NodeDragStartEvent,
  NodeId,
  Vec2,
} from "../types.ts";

/** What a drag needs from the graph, and nothing more. */
export interface NodeDragHost {
  /**
   * Pin or unpin a node. A drag pins at the start and never unpins: a node
   * dropped somewhere stays where it was dropped, and releasing it again is
   * the caller's decision, not the gesture's.
   */
  setPinned(node: NodeId, pinned: boolean): void;
  /** Write a node's graph-space position. */
  setPosition(node: NodeId, x: number, y: number): void;
  emit(event: NodeDragStartEvent | NodeDragMoveEvent | NodeDragEndEvent): void;
  /** Clock for event timestamps. */
  now(): number;
}

/**
 * The drag state machine.
 *
 * `move` and `end` name their node so a second entry point cannot drive a
 * gesture that belongs to the first: an event for a node that is not the one
 * being dragged is dropped rather than acted on.
 */
export class NodeDragController {
  readonly #host: NodeDragHost;
  #node: NodeId | null = null;
  #position: Vec2 | null = null;

  constructor(host: NodeDragHost) {
    this.#host = host;
  }

  /** The node being dragged, or `null`. */
  get node(): NodeId | null {
    return this.#node;
  }

  /** The last position the drag wrote, or `null` when none is in flight. */
  get position(): Vec2 | null {
    return this.#position;
  }

  /**
   * Start dragging `node`, which is at `position`.
   *
   * A gesture still in flight is ended first. Pointer capture makes that
   * unreachable from either entry point, but a drag that outlived its input —
   * a card released mid-gesture, a pointer the host never reported up — would
   * otherwise leave a node pinned with no `node:dragend` ever emitted.
   */
  begin(node: NodeId, position: Vec2): void {
    if (this.#node !== null) this.end(this.#node, this.#position ?? position);

    this.#node = node;
    this.#position = { x: position.x, y: position.y };
    this.#host.setPinned(node, true);
    this.#host.emit({
      type: "node:dragstart",
      timestamp: this.#host.now(),
      nodeId: node,
      position,
    });
  }

  /** Move the dragged node to `position`. */
  move(node: NodeId, position: Vec2): void {
    const previous = this.#position;
    if (this.#node !== node || previous === null) return;

    const delta: Vec2 = { x: position.x - previous.x, y: position.y - previous.y };
    this.#position = { x: position.x, y: position.y };
    this.#host.setPosition(node, position.x, position.y);
    this.#host.emit({
      type: "node:dragmove",
      timestamp: this.#host.now(),
      nodeId: node,
      position,
      delta,
    });
  }

  /** Finish the drag. The node keeps the pin the gesture took. */
  end(node: NodeId, position: Vec2): void {
    if (this.#node !== node) return;

    this.#node = null;
    this.#position = null;
    this.#host.emit({
      type: "node:dragend",
      timestamp: this.#host.now(),
      nodeId: node,
      position,
    });
  }

  /**
   * Drop the gesture without a `node:dragend`.
   *
   * For the press that turns out to have been a click: nothing was dragged, so
   * there is no drag to have ended, and the caller undoes the pin itself.
   */
  cancel(): void {
    this.#node = null;
    this.#position = null;
  }
}
