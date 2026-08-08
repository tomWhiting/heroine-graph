/**
 * Card Store — the bridge from the provider contract to a declarative renderer
 *
 * {@link CardProvider} is imperative by necessity: `mount` has to hand back
 * per-card state and `release` has to run at a defined point in the DOM
 * lifecycle, neither of which an event can express. A declarative renderer
 * wants the opposite — a list of what exists, and a subscription that fires
 * when the list changes. This module is the whole of the translation, and it
 * lives in core rather than in a framework package because it is stated
 * entirely in the library's own terms: a provider in, a subscribable list of
 * mounted cards out. The React, Vue and Svelte wrappers each need the same
 * list, and none of them needs a different one.
 *
 * The two rules that make the naive version wrong are both here:
 *
 * 1. **`release` must leave the container empty synchronously.** Containers are
 *    pooled, and `CardContainerPool.recycle` drops a container that still has
 *    children rather than leaking one card's content into the next. Every
 *    framework unmounts asynchronously, so the store detaches its own host
 *    element itself and lets the framework tear down into a node that is
 *    already off-screen.
 * 2. **Placement must not reach the renderer.** `CardDriver.place` runs for
 *    every card on every frame of camera motion; a store that republished on it
 *    would re-render every card sixty times a second for a transform core has
 *    already written. Geometry is read off {@link CardNode}, which is a live
 *    view, at whatever rate the consumer actually wants it.
 *
 * @module
 */

import type { IdLike } from "../graph/id_map.ts";
import type { CardNode, CardProvider } from "./types.ts";

/**
 * One mounted card, as a renderer sees it.
 *
 * Replaced by a new object whenever anything here changes, so object identity
 * is the change signal — a version counter beside it would be a second answer
 * to the same question, free to disagree with the first.
 */
export interface LiveCard {
  /**
   * Stable identity for the card, captured at mount.
   *
   * The producer's identifier rather than the slot: slots are recycled, so a
   * batch removal can move a different node into a carded slot, and a renderer
   * keyed on the slot would carry one node's component state onto another's
   * content. Frozen at mount because {@link CardNode.externalId} reads through
   * to whatever the slot holds *now*, which is exactly the value that must not
   * change under a rendered card.
   */
  readonly key: IdLike;
  /** Live view of the node, for content, position, size and pinning. */
  readonly node: CardNode;
  /**
   * Element to render into. Core owns the container above it; this is the
   * consumer's, and stays intact after release so an asynchronous unmount has
   * something to tear down.
   */
  readonly host: HTMLElement;
  readonly selected: boolean;
  readonly hovered: boolean;
}

/** A card provider paired with the subscription its cards are published on. */
export interface CardStore {
  /**
   * Register this with `GraphMother.setCardProvider`. It never throws, so the
   * only failures a consumer of this store can have are in its own renderer,
   * which only that framework's error boundary can contain.
   */
  readonly provider: CardProvider<unknown>;
  /** Subscribe to card-set changes; call the result to unsubscribe. */
  subscribe(listener: () => void): () => void;
  /**
   * The mounted cards. Identity-stable between changes, which is what a
   * `useSyncExternalStore`-style subscription requires of it.
   */
  getSnapshot(): readonly LiveCard[];
}

/** Mutable half of a {@link LiveCard}, kept as the provider's per-card state. */
interface StoreCard {
  card: LiveCard;
}

/**
 * Create a card store.
 *
 * One per graph. The store holds no graph reference of its own — it learns
 * which cards exist by being registered as the provider, and forgets them the
 * same way.
 */
export function createCardStore(): CardStore {
  const cards = new Map<HTMLElement, StoreCard>();
  const listeners = new Set<() => void>();
  let snapshot: readonly LiveCard[] = [];
  let stale = false;

  const publish = (): void => {
    stale = true;
    for (const listener of listeners) listener();
  };

  const provider: CardProvider<StoreCard> = {
    mount(container: HTMLElement, node: CardNode): StoreCard {
      const host = container.ownerDocument.createElement("div");
      // `display: contents` rather than a full-size box: the container is
      // already laid out at the card's size, and a wrapper of its own would be
      // one more element between core's geometry and the consumer's content —
      // visible to their CSS, and one more thing to keep in sync.
      host.style.display = "contents";
      container.appendChild(host);

      const entry: StoreCard = {
        card: { key: node.externalId, node, host, selected: false, hovered: false },
      };
      cards.set(host, entry);
      publish();
      return entry;
    },

    update(_container: HTMLElement, _node: CardNode, change, entry: StoreCard): void {
      switch (change.kind) {
        case "data":
          // Nothing on the record changes — the node is a live view — but the
          // renderer has to be told that what it reads through it has moved on.
          entry.card = { ...entry.card };
          break;
        case "selection":
          if (entry.card.selected === change.selected) return;
          entry.card = { ...entry.card, selected: change.selected };
          break;
        case "hover":
          if (entry.card.hovered === change.hovered) return;
          entry.card = { ...entry.card, hovered: change.hovered };
          break;
        case "position":
        case "size":
          // Deliberately silent, and load-bearing: see the module doc. Core has
          // already placed the container, and a renderer that wants live
          // geometry reads `node.position()` / `node.size()`.
          return;
      }
      publish();
    },

    release(_container: HTMLElement, _node: CardNode, entry: StoreCard): void {
      cards.delete(entry.card.host);
      // Synchronous, and the reason this module exists: the container must be
      // empty by the time this returns or the pool drops it, and no framework
      // unmounts synchronously. The host itself stays whole, so the teardown
      // that arrives later removes children from a node it still recognises.
      entry.card.host.remove();
      publish();
    },
  };

  return {
    provider,
    subscribe(listener: () => void): () => void {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    getSnapshot(): readonly LiveCard[] {
      if (stale) {
        snapshot = [...cards.values()].map((entry) => entry.card);
        stale = false;
      }
      return snapshot;
    },
  };
}
