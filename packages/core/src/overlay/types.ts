/**
 * DOM Card Overlay — Consumer Contract
 *
 * At high zoom a handful of nodes stop being useful as GPU sprites: the user
 * wants selectable text, real links and browser find-in-page. The overlay
 * promotes those nodes to DOM elements. The split of responsibility is the
 * whole contract: **core owns where a card is and how big it is; the consumer
 * owns everything inside it.**
 *
 * Callbacks rather than events, for two things the event emitter cannot do:
 * `mount` must return per-card state (a framework root, a cleanup closure),
 * and that state must be torn down at a defined point in the DOM lifecycle.
 *
 * Every rule stated on these types is enforced by {@link CardDriver} and
 * asserted in `tests/unit/card_contract_test.ts`.
 *
 * @module
 */

import type { NodeId, Vec2 } from "../types.ts";
import type { IdLike } from "../graph/id_map.ts";

/** Card box in CSS pixels. Core computes it; the provider must not change it. */
export interface CardSize {
  readonly width: number;
  readonly height: number;
}

/**
 * Live view of one carded node, handed to every {@link CardProvider} callback.
 *
 * The object is stable for the lifetime of the card, so a provider may retain
 * it in its own state; its accessors read through to current graph data.
 */
export interface CardNode {
  /**
   * GPU slot index. Only meaningful inside the owning GraphMother instance,
   * and recycled when nodes are removed — key consumer records on
   * {@link CardNode.externalId} instead.
   */
  readonly id: NodeId;
  /** The identifier the producer supplied for this node. */
  readonly externalId: IdLike;
  /** Display label, when the producer supplied one. */
  readonly label: string | undefined;
  /** Producer-defined tag index. Core compares it but never interprets it. */
  readonly tag: number;
  /** Producer-defined importance in 0..1. Core uses it only for tie-breaks. */
  readonly weight: number;
  /** Containment depth; 0 at a hierarchy root. */
  readonly depth: number;
  /**
   * Opaque producer content reference — a content hash, a path, whatever the
   * producer chose. Core never reads, parses or dereferences it: GraphMother
   * has no transport and never fetches card content.
   */
  readonly contentRef: string | undefined;
  /** Current graph-space position, from the existing position readback. */
  position(): Vec2;
  /** Current card box. Core owns it; this is a readout, not a setter. */
  size(): CardSize;
  /** Pin the underlying node, freezing it against simulation forces. */
  pin(): void;
  /** Release a pin taken by {@link CardNode.pin}. */
  unpin(): void;
  isPinned(): boolean;
}

/** Card moved or resized because core re-placed it. */
export type CardPlacementChange = { kind: "position" } | { kind: "size" };

/** Card's underlying node data or interaction state changed. */
export type CardStateChange =
  | { kind: "data" }
  | { kind: "selection"; selected: boolean }
  | { kind: "hover"; hovered: boolean };

/** Everything core reports to a mounted card through {@link CardProvider.update}. */
export type CardChange = CardPlacementChange | CardStateChange;

/**
 * Consumer-supplied card renderer. Register one to control card content;
 * register none and {@link createDefaultCardProvider} is used instead.
 *
 * @typeParam TState - Whatever `mount` needs to hand back to `update`/`release`.
 */
export interface CardProvider<TState = unknown> {
  /**
   * Advisory notice that a node entered the prefetch ring. No DOM exists yet
   * and `mount` may never follow, so this must not have visible side effects.
   * Fires at most once per node per LOD epoch.
   *
   * `signal` aborts when the node leaves the ring without being carded, when
   * its card is released, or when the overlay is torn down. Abandon the work
   * then — core will not call back about it.
   */
  prefetch?(node: CardNode, signal: AbortSignal): void;

  /**
   * The card box this node needs, in CSS pixels. Return `undefined` to take
   * {@link DEFAULT_CARD_SIZE}.
   *
   * Asked once, immediately before {@link CardProvider.mount}, and fixed for
   * the card's whole life: the card is laid out in this box from then on, so
   * content never reflows under the user's pointer, and the box is what the
   * camera counter-scales *to* — past the scale a card mounted at it renders
   * at exactly this size, pixel for pixel, however far the camera zooms in.
   * A card that has to hold a working control, rather than a caption, is the
   * reason this hook exists: 200x120 is a label, not an editor.
   *
   * `node.size()` is not meaningful here. It reads the box this call is
   * deciding, and answers with the default until the answer exists.
   *
   * A throw costs the node the size it asked for, not its card: the default
   * is a usable answer, and forfeiting the content over a measurement would
   * cost more than it saved.
   */
  size?(node: CardNode): CardSize | undefined;

  /**
   * Called exactly once per card, with an empty container that core has
   * already attached, positioned and sized.
   *
   * Everything inside `container` belongs to the provider. Core writes only
   * `transform`, `width`, `height` and `opacity` on the container element
   * itself and never touches its children.
   *
   * A child carrying the `data-graphmother-drag` attribute becomes the card's
   * drag handle: a pointer gesture starting on it moves the node exactly as
   * dragging the node's sprite on the canvas does. Declare none and the card
   * does not drag, which is what keeps its text selectable.
   */
  mount(container: HTMLElement, node: CardNode): TState;

  /** Called once per change core makes to a mounted card. */
  update?(container: HTMLElement, node: CardNode, change: CardChange, state: TState): void;

  /**
   * Called while the container is still in the DOM, so teardown can read
   * layout, and always before core detaches it.
   *
   * Containers are pooled and reused, so removing the children this provider
   * added is the provider's job — core will not do it.
   */
  release?(container: HTMLElement, node: CardNode, state: TState): void;
}

/** Overlay-wide settings. */
export interface DomOverlayConfig {
  /** Master switch, independent of whether semantic LOD is enabled. */
  enabled: boolean;
  /**
   * Element the card container is appended to. Defaults to the canvas's
   * parent element, which is what an embedding adapter already owns.
   */
  host?: HTMLElement;
  /** Rasterize cards back to sprites once a gesture settles. */
  rasterizeOnSettle: boolean;
  /** Forward `wheel` events over a card to the canvas zoom handler. */
  forwardWheel: boolean;
  /** Class applied to each card container, for consumer styling. */
  className?: string;
  /**
   * Ceiling on simultaneously mounted cards (FR-009).
   *
   * Shared with `LodConfig.maxCards`: the controller ranks and truncates the
   * declared set, the overlay budgets what it is handed. Two independently
   * settable copies would silently disagree, so `setLodConfig` forwards it.
   */
  maxCards: number;
  /**
   * Anti-flicker floor on how long a card stays once it exists, in ms
   * (FR-009). Shared with `LodConfig.minCardLifetimeMs` for the same reason
   * as {@link DomOverlayConfig.maxCards}.
   */
  minCardLifetimeMs: number;
}

/** Defaults for {@link DomOverlayConfig}: off, with both behaviours armed. */
export const DEFAULT_DOM_OVERLAY_CONFIG: DomOverlayConfig = {
  enabled: false,
  rasterizeOnSettle: true,
  forwardWheel: true,
  maxCards: 150,
  minCardLifetimeMs: 400,
};
