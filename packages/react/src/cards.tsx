"use client";

/**
 * DOM Card Adapter
 *
 * The React half of the card overlay. Core promotes a node to a real DOM
 * element and decides where that element sits and how big it is; everything
 * inside it belongs to the consumer, and this is how a React consumer gets to
 * put a component there instead of writing the imperative bridge by hand.
 *
 * The division of labour is the point, and it is core's, not this file's: this
 * adapter never positions, sizes, orders or evicts a card. It renders content
 * into hosts core created, in the order core published them. Anything here that
 * looked like layout would be a second answer to a question core has already
 * answered, and the two would drift apart the first time the camera moved.
 *
 * Everything except the JSX lives in `createCardStore` in core, so the same
 * bridge serves the Vue and Svelte wrappers unchanged and the part that is
 * genuinely React — a portal per card, and an error boundary around it — is all
 * that is left here.
 *
 * @module
 */

import {
  Component,
  type ReactNode,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { createPortal } from "react-dom";
import type { CardNode, CardProvider, CardSize, LiveCard } from "@graphmother/core";
import { createCardStore } from "@graphmother/core";

/**
 * The one thing the card bridge needs from a graph.
 *
 * Structural rather than the `GraphMother` class, so a test — or a wrapper that
 * owns the graph itself — can stand in for it without a WebGPU device.
 */
export interface CardProviderHost {
  setCardProvider(provider: CardProvider<unknown> | null): void;
}

/**
 * Subscribe to the cards a graph currently has mounted.
 *
 * Registers a card provider on `graph` for as long as the calling component is
 * mounted, and returns the live card list. Pass `null` while the graph is still
 * initialising — the `onReady` callback of the `GraphMother` component is the
 * usual source — and the hook simply waits.
 *
 * The returned array changes identity only when a card is added, removed, or
 * changes selection, hover or data. It deliberately does *not* change when a
 * card moves or resizes: core re-places every card on every frame of camera
 * motion, and re-rendering the whole card set at that rate would spend the
 * budget the overlay exists to protect. Read `card.node.position()` and
 * `card.node.size()` for live geometry.
 */
export function useGraphCards(
  graph: CardProviderHost | null | undefined,
  size?: (node: CardNode) => CardSize | undefined,
): readonly LiveCard[] {
  // Read through a ref because the store is created once and the box is decided
  // before any DOM exists: a consumer defining `size` inline would otherwise pin
  // every card to whatever the first render happened to close over. The ref is
  // filled by an effect declared ahead of the registration below, so it is
  // current before the provider is ever reachable.
  const sizeRef = useRef(size);
  useEffect(() => {
    sizeRef.current = size;
  }, [size]);

  const [store] = useState(() =>
    createCardStore({ size: (node: CardNode) => sizeRef.current?.(node) })
  );

  useEffect(() => {
    if (!graph) return;
    graph.setCardProvider(store.provider);
    return () => {
      graph.setCardProvider(null);
    };
  }, [graph, store]);

  return useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
}

export interface GraphCardsProps {
  /** The graph whose cards to render. `null` until it has initialised. */
  graph: CardProviderHost | null | undefined;
  /** Content for one card. Called on every render of the card set. */
  render: (card: LiveCard) => ReactNode;
  /**
   * A card component threw while rendering.
   *
   * Distinct from the graph's `card:error` event, which reports a *provider*
   * failure core contained. The provider here is core's own store and never
   * throws, so the failure a React consumer actually has is their own component,
   * and only a React error boundary can see it.
   */
  onCardError?: (error: Error, card: LiveCard) => void;
  /** Rendered in place of a card whose content threw. Nothing, by default. */
  fallback?: ReactNode;
  /**
   * How big each card needs to be, in CSS pixels; omit for the default box.
   *
   * Asked once per card, immediately before it mounts, and fixed for that card's
   * life — so the content never reflows under the user, and past the mount scale
   * the card renders at exactly this size however far the camera zooms in. A
   * component with its own layout needs this: 200x120 is a caption.
   *
   * It cannot be derived from the rendered component, because the box is decided
   * before React has rendered anything into the card.
   */
  size?: (node: CardNode) => CardSize | undefined;
}

/**
 * Render a React component into every card the graph has mounted.
 *
 * Renders nothing itself: each card is a portal into the host element core
 * created for it, so the cards live in core's overlay container — above the
 * canvas, carrying the camera transform — while their components stay in the
 * calling component's tree, with its context, its state and its hooks.
 *
 * @example
 * ```tsx
 * <GraphCards
 *   graph={graph}
 *   render={({ node }) => <article>{node.label ?? String(node.externalId)}</article>}
 * />
 * ```
 */
export function GraphCards(props: GraphCardsProps): ReactNode {
  const cards = useGraphCards(props.graph, props.size);
  return cards.map((card) =>
    createPortal(
      <CardBoundary card={card} onCardError={props.onCardError} fallback={props.fallback}>
        {props.render(card)}
      </CardBoundary>,
      card.host,
      // Keyed on the producer's identifier, never on the slot: slots are
      // recycled, so a batch removal can hand a carded slot to a different
      // node, and a key that changed hands would carry one node's component
      // state onto another node's content.
      String(card.key),
    )
  );
}

interface CardBoundaryProps {
  readonly card: LiveCard;
  readonly onCardError: ((error: Error, card: LiveCard) => void) | undefined;
  readonly fallback: ReactNode;
  readonly children: ReactNode;
}

interface CardBoundaryState {
  readonly failed: boolean;
}

/**
 * Contain one card's render errors.
 *
 * Without it a single card component throwing unmounts the tree it was rendered
 * from — which, for the usual arrangement, is the tree holding the canvas, so
 * one bad card takes the graph with it. React can only be stopped by a class
 * component, which is why this is one.
 *
 * It cannot contain everything: an error thrown from an effect after commit is
 * outside a boundary's reach, and there is no React mechanism that would put it
 * inside one.
 */
class CardBoundary extends Component<CardBoundaryProps, CardBoundaryState> {
  override state: CardBoundaryState = { failed: false };

  static getDerivedStateFromError(): CardBoundaryState {
    return { failed: true };
  }

  override componentDidCatch(error: Error): void {
    this.props.onCardError?.(error, this.props.card);
  }

  override render(): ReactNode {
    return this.state.failed ? this.props.fallback ?? null : this.props.children;
  }
}
