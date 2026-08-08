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
 * Being the single enforcement point makes it the single *containment* point
 * too. A provider callback is consumer code running inside core's per-frame
 * path: uncontained, one card that throws in `mount` strands a styled,
 * attached container in the overlay forever and takes the whole render callback
 * down with it on every frame after. So every call into the provider is
 * wrapped, the card is left in a defined state, and the failure is handed to
 * {@link CardDriverOptions.onProviderError} — or logged, when no one asked for
 * it. Containment here is never silence.
 *
 * @module
 */

import type { CardProviderHook, NodeId } from "../types.ts";
import { Errors, type GraphMotherError } from "../errors.ts";
import type { CardChange, CardNode, CardProvider, CardSize, CardStateChange } from "./types.ts";

/**
 * Whether a throw from this hook is evidence that the provider cannot render
 * this node at all — the question every quarantine decision turns on.
 *
 * True for the two hooks a card cannot exist without. `mount` failed to build
 * one; `update` had one and could not keep it, and `place` calls `update` on
 * every frame of camera motion, so re-offering the node is a throw per frame.
 * Retrying either is one failure per node per evaluation for as long as the
 * graph is open.
 *
 * False for the other two, and the distinction is the whole point of asking.
 * `prefetch` is advisory and abortable: the driver already spends the node for
 * the epoch, and a speculative fetch that raised — a cache miss, a bad URL —
 * says nothing about whether `mount` would have worked. `release` runs after a
 * card the provider built and core displayed, so a throw there is a teardown
 * defect, not an inability to render. Treating either as forfeiting the card
 * costs the node its content for the life of the graph, with no event once the
 * first has been reported and no way back short of replacing the provider.
 *
 * Exported because the answer has to be the same in the overlay, which stops
 * offering the node, and in the graph, which stops declaring it — two copies of
 * this rule would silently disagree, and a node the two disagree about is drawn
 * as neither sprite nor card.
 *
 * @param hook - Which provider callback threw
 */
export function cardFailureForfeitsCard(hook: CardProviderHook): boolean {
  return hook === "mount" || hook === "update";
}

/**
 * Where core has decided a card sits, in the coordinate space of the driver's
 * host element.
 */
export interface CardPlacement {
  /** Left edge, in the container's (camera-scaled) coordinate space. */
  readonly x: number;
  /** Top edge, in the container's (camera-scaled) coordinate space. */
  readonly y: number;
  /** Layout width in CSS pixels — the card's natural size, not its rendered one. */
  readonly width: number;
  /** Layout height in CSS pixels. */
  readonly height: number;
  /**
   * Counter-scale applied to the card, cancelling the camera so the card is
   * laid out at {@link CardPlacement.width} and rendered at the size the
   * overlay chose. See `cardCounterScale`.
   */
  readonly scale: number;
  /** 0..1 crossfade opacity. Core-owned, so it raises no `CardChange`. */
  readonly opacity: number;
}

/**
 * One contained provider failure.
 *
 * Reported after the driver has finished putting the card back into a defined
 * state, so a handler sees settled state rather than a teardown in progress.
 */
export interface CardProviderFailure {
  /** The node whose card failed; still a live view of its slot. */
  readonly node: CardNode;
  /** Which provider callback threw. */
  readonly hook: CardProviderHook;
  /** The throw, wrapped with the node, the hook and the contract rules. */
  readonly error: GraphMotherError;
  /** The value the provider threw, unwrapped. */
  readonly cause: unknown;
  /**
   * Whether the node lost its card. False when there was none to lose: a
   * `mount` that never completed, or an advisory `prefetch`.
   */
  readonly released: boolean;
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
  /**
   * Called once per contained provider failure. Omit it and the driver logs
   * the wrapped error instead — a swallowed exception is a defect, so the
   * quiet path is a decision the caller has to make explicitly.
   */
  readonly onProviderError?: (failure: CardProviderFailure) => void;
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
 *
 * The origin goes with it: placement puts the card's *top-left* at `x`/`y`, so
 * the default `50% 50%` origin would offset every scaled card by half the
 * difference between its layout and rendered size.
 */
function anchorContainer(container: HTMLElement): void {
  const style = container.style;
  style.position = "absolute";
  style.left = "0";
  style.top = "0";
  style.transformOrigin = "0 0";
}

/**
 * Write the four style properties core owns, and nothing else.
 *
 * The transform composes as `translate ∘ scale`, so the scale runs first and
 * the translation is in container units either way — which is what lets
 * placement express the offset without knowing the counter-scale.
 */
function applyPlacement(container: HTMLElement, placement: CardPlacement): void {
  const style = container.style;
  style.transform = `translate(${placement.x}px, ${placement.y}px) scale(${placement.scale})`;
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
  private readonly onProviderError: ((failure: CardProviderFailure) => void) | undefined;

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
    this.onProviderError = options.onProviderError;
  }

  /**
   * Announce a contained failure.
   *
   * Called only once the card is back in a defined state, so a handler is free
   * to drive the driver again from inside it.
   */
  private report(
    node: CardNode,
    hook: CardProviderHook,
    cause: unknown,
    released: boolean,
  ): void {
    const error = Errors.cardProviderFailed(hook, node.id, node.externalId, cause);
    const handler = this.onProviderError;
    if (handler === undefined) {
      error.log();
      return;
    }
    handler({ node, hook, error, cause, released });
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
    try {
      this.provider.prefetch(node, controller.signal);
    } catch (cause) {
      // The offer is abandoned exactly as one the node walked away from is,
      // and the node stays spent for the epoch: a prefetch that threw is no
      // more worth repeating than one that was ignored.
      this.cancelPrefetch(node.id);
      this.report(node, "prefetch", cause, false);
    }
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
   * Ask the provider how big this card needs to be, contained.
   *
   * Separate from {@link CardDriver.mount} because the answer is needed
   * *before* the container exists — the box is what the container is created
   * at, and what the card is laid out in for the rest of its life.
   *
   * @returns the provider's box, or `undefined` for "no opinion" — which a
   * provider without the hook, a provider that returned nothing, and a
   * provider that threw all give alike. A measurement is not evidence the
   * provider cannot render the node, so unlike `mount` this one does not cost
   * the card; the caller falls back to the default box and carries on.
   */
  measure(node: CardNode): CardSize | undefined {
    if (this.provider.size === undefined) return undefined;
    try {
      return this.provider.size(node);
    } catch (cause) {
      this.report(node, "size", cause, false);
      return undefined;
    }
  }

  /**
   * Create a card: attach and place an empty container, then hand it to
   * `provider.mount` exactly once.
   *
   * A provider that throws leaves no trace: the container is detached again and
   * offered back to the pool, which drops it if the half-built card left
   * children behind — the pool's existing dirty check is already exactly the
   * right rule here, so containment introduces no new discipline. The card is
   * never registered, so nothing later tries to place or release it.
   *
   * @returns whether the card exists. A caller keeping its own record of the
   * card must roll that record back on `false`.
   * @throws if the node already has a card — double-mount would break the
   * provider's "exactly once" guarantee silently, and there is no caller for
   * which it is meaningful.
   */
  mount(node: CardNode, placement: CardPlacement): boolean {
    if (this.cards.has(node.id)) {
      throw new Error(`Card for node ${node.id} is already mounted`);
    }

    const container = this.createContainer();
    if (this.cardClassName !== undefined) container.className = this.cardClassName;
    anchorContainer(container);
    applyPlacement(container, placement);
    this.host.appendChild(container);

    let state: TState;
    try {
      state = this.provider.mount(container, node);
    } catch (cause) {
      container.remove();
      this.recycleContainer?.(container);
      this.report(node, "mount", cause, false);
      return false;
    }
    this.cards.set(node.id, { node, container, state, placement });
    return true;
  }

  /**
   * Re-place a mounted card. Writes only the core-owned style properties and
   * reports at most one `position` and one `size` change; an unchanged
   * placement produces no DOM write and no callback. Ignored for nodes with
   * no mounted card.
   *
   * A `size` change is reported against the card's *rendered* extent rather
   * than its layout box, which is fixed at mount. That is the one of the two a
   * provider can act on, and it is silent through the clamped regime — where
   * the counter-scale changes on every zoom step precisely so that the rendered
   * size does not.
   */
  place(nodeId: NodeId, placement: CardPlacement): void {
    const card = this.cards.get(nodeId);
    if (card === undefined) return;

    const previous = card.placement;
    const moved = placement.x !== previous.x || placement.y !== previous.y;
    const resized = placement.width * placement.scale !== previous.width * previous.scale ||
      placement.height * placement.scale !== previous.height * previous.scale;
    const rescaled = placement.scale !== previous.scale;
    if (!moved && !resized && !rescaled && placement.opacity === previous.opacity) return;

    card.placement = placement;
    applyPlacement(card.container, placement);

    if (moved && !this.update(nodeId, card, POSITION_CHANGE)) return;
    if (resized) this.update(nodeId, card, SIZE_CHANGE);
  }

  /**
   * Report a data or interaction change to a mounted card. Placement changes
   * are not accepted here: they are core's to derive, via {@link place}.
   * Ignored for nodes with no mounted card.
   */
  notify(nodeId: NodeId, change: CardStateChange): void {
    const card = this.cards.get(nodeId);
    if (card === undefined) return;
    this.update(nodeId, card, change);
  }

  /**
   * Report one change to a mounted card, and tear the card down if that throws.
   *
   * Releasing rather than reporting and carrying on: a provider whose `update`
   * threw is in a state it did not intend, and `place` runs for every card on
   * every frame of camera motion — so "leave it mounted" means throwing again
   * sixty times a second against a card that is already wrong on screen.
   *
   * @returns whether the card is still mounted
   */
  private update(nodeId: NodeId, card: MountedCard<TState>, change: CardChange): boolean {
    try {
      this.provider.update?.(card.container, card.node, change, card.state);
      return true;
    } catch (cause) {
      this.release(nodeId);
      this.report(card.node, "update", cause, true);
      return false;
    }
  }

  /**
   * Tear a card down: `provider.release` runs while the container is still in
   * the DOM, then the container is detached and offered back to the pool.
   * Ignored for nodes with no mounted card.
   *
   * Detaching happens whether or not `release` threw. A provider that failed to
   * clean up has lost nothing it can still use — the card is already gone from
   * the driver's map — and the alternative is a live, pointer-catching card
   * wedged on screen over a node that no longer has one.
   */
  release(nodeId: NodeId): void {
    const card = this.cards.get(nodeId);
    if (card === undefined) return;

    this.cards.delete(nodeId);
    this.cancelPrefetch(nodeId);
    let failure: unknown;
    let failed = false;
    try {
      this.provider.release?.(card.container, card.node, card.state);
    } catch (cause) {
      failure = cause;
      failed = true;
    }
    card.container.remove();
    this.recycleContainer?.(card.container);
    if (failed) this.report(card.node, "release", failure, true);
  }

  /** Release every card and abort every prefetch. */
  releaseAll(): void {
    for (const nodeId of [...this.cards.keys()]) this.release(nodeId);
    for (const controller of this.pendingPrefetch.values()) controller.abort();
    this.pendingPrefetch.clear();
    this.prefetchedThisEpoch.clear();
  }
}
