/**
 * Card Container Pool
 *
 * Mounting a card costs an element creation, a style write and a layout;
 * zooming across a threshold band does that for tens of cards at once, and a
 * user working at the boundary crosses it repeatedly. The pool makes the
 * second crossing free by keeping released containers around instead of
 * dropping them on the floor.
 *
 * The one rule that makes pooling safe is the contract's, not the pool's:
 * everything inside a container belongs to the provider, and the provider
 * removes it in `release`. So the pool never touches children — it *checks*
 * for them, and quietly drops an element a provider left dirty rather than
 * handing the leftovers to the next card.
 *
 * @module
 */

/** Idle containers kept per overlay when the caller supplies no limit. */
export const DEFAULT_MAX_IDLE_CONTAINERS = 32;

/**
 * A recycling bin of detached card containers.
 *
 * Deliberately unaware of cards, nodes and placement: {@link CardDriver} owns
 * the DOM lifecycle and calls in here through its `createContainer` /
 * `recycleContainer` hooks.
 */
export class CardContainerPool {
  readonly #document: Document;
  readonly #maxIdle: number;
  readonly #idle: HTMLElement[] = [];

  /**
   * @param document - Document new containers are created in
   * @param maxIdle - Idle containers to keep; extras are dropped for the GC
   */
  constructor(document: Document, maxIdle: number = DEFAULT_MAX_IDLE_CONTAINERS) {
    this.#document = document;
    this.#maxIdle = Math.max(0, Math.trunc(maxIdle));
  }

  /** Containers currently held for reuse. */
  get idleCount(): number {
    return this.#idle.length;
  }

  /**
   * Take a container for a new card: a recycled one when the pool has any,
   * otherwise a fresh element.
   *
   * Core-owned presentation is (re)established here rather than at recycle
   * time, so a container is in a known state whichever path produced it. The
   * class name is cleared because the driver only *sets* it when the overlay
   * is configured with one, and a stale class from a previous configuration
   * would otherwise survive the round trip.
   */
  acquire(): HTMLElement {
    const container = this.#idle.pop() ?? this.#document.createElement("div");
    container.className = "";
    // Cards are the interactive part of an otherwise click-through overlay:
    // the container above the canvas is `pointer-events: none`, each card
    // switches it back on so text selection, links and find-in-page work.
    container.style.pointerEvents = "auto";
    return container;
  }

  /**
   * Offer a detached container back for reuse.
   *
   * A container that still has children is dropped instead: its provider did
   * not clean up, and reusing it would leak that content into an unrelated
   * card. Core removing the children itself would be the same contract
   * violation from the other direction (SC-004).
   */
  recycle(container: HTMLElement): void {
    if (container.childNodes.length > 0) return;
    if (this.#idle.length >= this.#maxIdle) return;
    this.#idle.push(container);
  }

  /** Drop every idle container. */
  clear(): void {
    this.#idle.length = 0;
  }
}
