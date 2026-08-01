/**
 * DOM Card Overlay Module
 *
 * The consumer-facing card contract, the driver that enforces it, the default
 * renderer used when no provider is registered, the slot ↔ producer-id mapping
 * cards key themselves on, and the overlay engine that decides which cards
 * exist and where they sit.
 *
 * Deliberately free of GPU imports: the overlay is a peer of the layer system,
 * not a `Layer` (a `Layer` renders into a `GPUCommandEncoder`, which a DOM
 * overlay never does).
 *
 * @module
 */

export { CardDriver } from "./driver.ts";
export type { CardDriverOptions, CardPlacement } from "./driver.ts";

export {
  CARD_DRAG_HANDLE_ATTRIBUTE,
  DEFAULT_CARD_SIZE,
  DEFAULT_GESTURE_IDLE_MS,
  DEFAULT_MAX_CARDS,
  DEFAULT_MIN_CARD_LIFETIME_MS,
  DomCardOverlay,
} from "./overlay.ts";
export type {
  CardDragSink,
  CardNodeSource,
  CardSyncEntry,
  DomCardOverlayOptions,
  OverlayTimers,
} from "./overlay.ts";

export { CardContainerPool, DEFAULT_MAX_IDLE_CONTAINERS } from "./pool.ts";

export { cardPlacementAt, formatCssMatrix, overlayMatrix, projectByMatrix } from "./projection.ts";
export type { CssMatrix } from "./projection.ts";

export { createDefaultCardProvider } from "./default_card.ts";
export type { DefaultCardState } from "./default_card.ts";

export { externalIdForSlot, slotForExternalId } from "./identity.ts";
export type { NodeIdentitySource } from "./identity.ts";

export { DEFAULT_DOM_OVERLAY_CONFIG } from "./types.ts";
export type {
  CardChange,
  CardNode,
  CardPlacementChange,
  CardProvider,
  CardSize,
  CardStateChange,
  DomOverlayConfig,
} from "./types.ts";
