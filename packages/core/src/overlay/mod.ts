/**
 * DOM Card Overlay Module
 *
 * The consumer-facing card contract, the driver that enforces it, the default
 * renderer used when no provider is registered, and the slot ↔ producer-id
 * mapping cards key themselves on.
 *
 * Deliberately free of GPU imports: the overlay is a peer of the layer system,
 * not a `Layer` (a `Layer` renders into a `GPUCommandEncoder`, which a DOM
 * overlay never does).
 *
 * @module
 */

export { CardDriver } from "./driver.ts";
export type { CardDriverOptions, CardPlacement } from "./driver.ts";

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
