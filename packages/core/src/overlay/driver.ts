/**
 * Card Driver
 *
 * Owns the DOM lifecycle of the card containers and is the single place the
 * {@link CardProvider} contract is enforced: mount exactly once, placement
 * writes confined to the four core-owned style properties, one typed change
 * per update, release strictly before detach, and prefetch advisory,
 * abortable and capped at once per node per LOD epoch.
 *
 * It deliberately knows nothing about zoom, thresholds, eviction budgets or
 * pooling — the overlay layer decides *which* nodes are carded and *where*
 * they sit, then drives them through here.
 *
 * @module
 */

import type { NodeId } from "../types.ts";
import type { CardChange, CardNode, CardProvider, CardStateChange } from "./types.ts";

/**
 * Where core has decided a card sits, in the coordinate space of the driver's
 * host element.
 */
export interface CardPlacement {
  /** Left edge in CSS pixels. */
  readonly x: number;
  /** Top edge in CSS pixels. */
  readonly y: number;
  readonly width: number;
  readonly height: number;
  /** 0..1 crossfade opacity. Core-owned, so it raises no `CardChange`. */
  readonly opacity: number;
}

export interface CardDriverOptions<TState> {
  /**
   * Element card containers are appended to. The driver only adds and removes
   * containers it created itself; siblings the caller put there are untouched.
   */
  readonly host: HTMLElement;
  readonly provider: CardProvider<TState>;
  /**
   * Supplies the element handed to `provider.mount`. Defaults to a fresh
   * `div` from the host's document; the overlay layer substitutes a pooled
   * element. Whatever it returns must have no children.
   */
  readonly createContainer?: () => HTMLElement;
  /** Called once a container has been detached, after `provider.release`. */
  readonly recycleContainer?: (container: HTMLElement) => void;
  /** Class applied to every card container. */
  readonly cardClassName?: string;
}

/**
 * Singleton change records. `place` runs per frame for every card, so the
 * common no-allocation path matters; the records carry no per-card data.
 */
const POSITION_CHANGE: CardChange = { kind: "position" };
const SIZE_CHANGE: CardChange = { kind: "size" };

interface MountedCard<TState> {
  readonly node: CardNode;
  readonly container: HTMLElement;
  readonly state: TState;
  placement: CardPlacement;
}

/**
 * Take a container out of normal flow, once, at mount. `transform` cannot
 * place a statically-positioned element at all, so this is the precondition
 * for placement rather than styling the consumer might want a say in.
 */
function anchorContainer(container: HTMLElement): void {
  const style = container.style;
  style.position = "absolute";
  style.left = "0";
  style.top = "0";
}

/** Write the four style properties core owns, and nothing else. */
function applyPlacement(container: HTMLElement, placement: CardPlacement): void {
  const style = container.style;
  style.transform = `translate(${placement.x}px, ${placement.y}px)`;
  style.width = `${placement.width}px`;
  style.height = `${placement.height}px`;
  style.opacity = String(placement.opacity);
}

export class CardDriver<TState = unknown> {
  private readonly host: HTMLElement;
  private readonly provider: CardProvider<TState>;
  private readonly createContainer: () => HTMLElement;
  private readonly recycleContainer: ((container: HTMLElement) => void) | undefined;
  private readonly cardClassName: string | undefined;

  private readonly cards = new Map<NodeId, MountedCard<TState>>();
  /** Live prefetch controllers, including those of nodes that went on to mount. */
  private readonly pendingPrefetch = new Map<NodeId, AbortController>();
  /** Nodes already offered to `provider.prefetch` in the current LOD epoch. */
  private readonly prefetchedThisEpoch = new Set<NodeId>();

  constructor(options: CardDriverOptions<TState>) {
    this.host = options.host;
    this.provider = options.provider;
    this.createContainer = options.createContainer ??
      (() => this.host.ownerDocument.createElement("div"));
    this.recycleContainer = options.recycleContainer;
    this.cardClassName = options.cardClassName;
  }

  /** Number of mounted cards. */
  get size(): number {
    return this.cards.size;
  }

  /** Whether this node currently has a mounted card. */
  has(nodeId: NodeId): boolean {
    return this.cards.has(nodeId);
  }

  /**
   * The node whose card container is `element`, or `undefined`.
   *
   * A scan rather than a reverse index: it runs once per pointer gesture over
   * at most `maxCards` entries, and a second map would be one more thing that
   * can disagree with this one.
   */
  nodeForContainer(element: Element): NodeId | undefined {
    for (const [nodeId, card] of this.cards) {
      if (card.container === element) return nodeId;
    }
    return undefined;
  }

  /**
   * Offer a node to `provider.prefetch`. Ignored if the node was already
   * offered in this epoch, still has a live prefetch, or is already carded,
   * which is what makes the call safe to issue for every node in the ring on
   * every evaluation.
   */
  prefetch(node: CardNode): void {
    if (this.provider.prefetch === undefined) return;
    if (this.pendingPrefetch.has(node.id) || this.prefetchedThisEpoch.has(node.id)) return;

    const controller = new AbortController();
    this.pendingPrefetch.set(node.id, controller);
    this.prefetchedThisEpoch.add(node.id);
    this.provider.prefetch(node, controller.signal);
  }

  /**
   * Abort the prefetch of a node that left the ring without being carded.
   * The node stays spent for this epoch: prefetch is advisory, and re-offering
   * it on every threshold wobble is exactly the churn the budget prevents.
   */
  cancelPrefetch(nodeId: NodeId): void {
    const controller = this.pendingPrefetch.get(nodeId);
    if (controller === undefined) return;
    this.pendingPrefetch.delete(nodeId);
    controller.abort();
  }

  /**
   * Start a new LOD epoch: abort every prefetch that has not produced a card
   * and re-arm the once-per-node budget. Mounted cards and their in-flight
   * prefetches are untouched — their content is still on screen.
   */
  beginEpoch(): void {
    for (const [nodeId, controller] of this.pendingPrefetch) {
      if (this.cards.has(nodeId)) continue;
      this.pendingPrefetch.delete(nodeId);
      controller.abort();
    }
    this.prefetchedThisEpoch.clear();
  }

  /**
   * Create a card: attach and place an empty container, then hand it to
   * `provider.mount` exactly once.
   *
   * @throws if the node already has a card — double-mount would break the
   * provider's "exactly once" guarantee silently, and there is no caller for
   * which it is meaningful.
   */
  mount(node: CardNode, placement: CardPlacement): void {
    if (this.cards.has(node.id)) {
      throw new Error(`Card for node ${node.id} is already mounted`);
    }

    const container = this.createContainer();
    if (this.cardClassName !== undefined) container.className = this.cardClassName;
    anchorContainer(container);
    applyPlacement(container, placement);
    this.host.appendChild(container);

    const state = this.provider.mount(container, node);
    this.cards.set(node.id, { node, container, state, placement });
  }

  /**
   * Re-place a mounted card. Writes only the core-owned style properties and
   * reports at most one `position` and one `size` change; an unchanged
   * placement produces no DOM write and no callback. Ignored for nodes with
   * no mounted card.
   */
  place(nodeId: NodeId, placement: CardPlacement): void {
    const card = this.cards.get(nodeId);
    if (card === undefined) return;

    const previous = card.placement;
    const moved = placement.x !== previous.x || placement.y !== previous.y;
    const resized = placement.width !== previous.width ||
      placement.height !== previous.height;
    if (!moved && !resized && placement.opacity === previous.opacity) return;

    card.placement = placement;
    applyPlacement(card.container, placement);

    if (moved) {
      this.provider.update?.(card.container, card.node, POSITION_CHANGE, card.state);
    }
    if (resized) {
      this.provider.update?.(card.container, card.node, SIZE_CHANGE, card.state);
    }
  }

  /**
   * Report a data or interaction change to a mounted card. Placement changes
   * are not accepted here: they are core's to derive, via {@link place}.
   * Ignored for nodes with no mounted card.
   */
  notify(nodeId: NodeId, change: CardStateChange): void {
    const card = this.cards.get(nodeId);
    if (card === undefined) return;
    this.provider.update?.(card.container, card.node, change, card.state);
  }

  /**
   * Tear a card down: `provider.release` runs while the container is still in
   * the DOM, then the container is detached and offered back to the pool.
   * Ignored for nodes with no mounted card.
   */
  release(nodeId: NodeId): void {
    const card = this.cards.get(nodeId);
    if (card === undefined) return;

    this.cards.delete(nodeId);
    this.cancelPrefetch(nodeId);
    this.provider.release?.(card.container, card.node, card.state);
    card.container.remove();
    this.recycleContainer?.(card.container);
  }

  /** Release every card and abort every prefetch. */
  releaseAll(): void {
    for (const nodeId of [...this.cards.keys()]) this.release(nodeId);
    for (const controller of this.pendingPrefetch.values()) controller.abort();
    this.pendingPrefetch.clear();
    this.prefetchedThisEpoch.clear();
  }
}
