/**
 * The render radius a collapsed proxy borrows, and giving it back.
 *
 * A collapsed parent is drawn as one instance at its well radius (Phase 3
 * §5.4 v1), which means overwriting the radius the producer gave it. That
 * overwrite has to be undone — on expand, when LOD is switched off, and, the
 * hard one, when the graph's topology changes underneath a live collapse.
 *
 * **Why the producer id and not the slot.** `removeNodes` compacts the slot
 * space, so between a collapse and the topology change that ends it a node can
 * move. A radius restored to the slot it was saved against would then resize
 * whatever node inherited that slot, and leave the node it belonged to
 * permanently inflated. Saving the producer id alongside the radius makes the
 * restore follow the node instead of the index.
 *
 * @module
 */

import type { IdLike } from "../graph/id_map.ts";

/** The graph readings and the one write this table needs. */
export interface ProxyRadiusHost {
  /** Current render radius of a slot. */
  radiusOf(slot: number): number;
  /** Producer id of a slot, or `undefined` when it holds no live node. */
  externalIdOf(slot: number): IdLike | undefined;
  /** Current slot of a producer id, or `undefined` when the node is gone. */
  slotOf(externalId: IdLike): number | undefined;
  /** Write a slot's render radius. @returns whether the value actually moved */
  setRadius(slot: number, radius: number): boolean;
}

/** What a slot has to be given back when it stops standing in for a subtree. */
interface SavedRadius {
  readonly externalId: IdLike | undefined;
  readonly radius: number;
}

/**
 * The set of slots currently rendering as collapsed proxies, and the radius
 * each of them is owed.
 */
export class ProxyRadiusTable {
  readonly #saved = new Map<number, SavedRadius>();

  /** How many slots are currently inflated. */
  get size(): number {
    return this.#saved.size;
  }

  /**
   * Make `proxies[0, count)` the complete set of collapsed proxies, at
   * `radii[k]`, giving every slot that has left the set its own radius back.
   *
   * Complete rather than incremental on purpose: a caller cannot leak an
   * inflated radius by forgetting to name an expansion.
   *
   * @returns the slots whose radius moved, unordered
   */
  declare(
    proxies: ArrayLike<number>,
    count: number,
    radii: ArrayLike<number>,
    host: ProxyRadiusHost,
  ): number[] {
    const declared = new Set<number>();
    for (let k = 0; k < count; k++) declared.add(proxies[k]);

    const changed: number[] = [];
    for (const [slot, saved] of this.#saved) {
      if (declared.has(slot)) continue;
      this.#saved.delete(slot);
      if (host.setRadius(slot, saved.radius)) changed.push(slot);
    }
    for (let k = 0; k < count; k++) {
      const slot = proxies[k];
      if (!this.#saved.has(slot)) {
        this.#saved.set(slot, {
          externalId: host.externalIdOf(slot),
          radius: host.radiusOf(slot),
        });
      }
      if (host.setRadius(slot, radii[k])) changed.push(slot);
    }
    return changed;
  }

  /**
   * Empty the table, restoring each saved radius to wherever its node is
   * *now* — which is not necessarily the slot it was saved against.
   *
   * A node the mutation deleted has no slot and is simply dropped; the radius
   * it was owed belongs to nothing.
   *
   * @returns the slots whose radius moved, unordered
   */
  release(host: ProxyRadiusHost): number[] {
    const changed: number[] = [];
    for (const saved of this.#saved.values()) {
      if (saved.externalId === undefined) continue;
      const slot = host.slotOf(saved.externalId);
      if (slot === undefined) continue;
      if (host.setRadius(slot, saved.radius)) changed.push(slot);
    }
    this.#saved.clear();
    return changed;
  }
}
