/**
 * Proxy render-radius bookkeeping.
 *
 * A collapsed parent renders at its well radius, which means overwriting the
 * radius the producer gave it. Everything asserted here is about giving that
 * radius back, including in the case the design is actually hard: a topology
 * change that compacts the slot space while a collapse is live. A restore
 * addressed to a stale slot resizes the wrong node *and* leaves the right one
 * inflated forever, and neither is visible in a screenshot.
 */

import { assertEquals } from "jsr:@std/assert@^1";
import {
  type ProxyRadiusHost,
  ProxyRadiusTable,
} from "../../packages/core/src/lod/proxy_radius.ts";
import type { IdLike } from "../../packages/core/src/graph/id_map.ts";

/**
 * A slot space with radii and producer ids, and a `remove` that compacts the
 * way `GraphMother.removeNodes` does — survivors shift down, ids travel with
 * them.
 */
class Slots implements ProxyRadiusHost {
  radii: number[];
  ids: IdLike[];

  constructor(radii: readonly number[]) {
    this.radii = [...radii];
    this.ids = radii.map((_, slot) => `node-${slot}`);
  }

  radiusOf(slot: number): number {
    return this.radii[slot];
  }
  externalIdOf(slot: number): IdLike | undefined {
    return this.ids[slot];
  }
  slotOf(externalId: IdLike): number | undefined {
    const slot = this.ids.indexOf(externalId);
    return slot === -1 ? undefined : slot;
  }
  setRadius(slot: number, radius: number): boolean {
    if (slot < 0 || slot >= this.radii.length || this.radii[slot] === radius) return false;
    this.radii[slot] = radius;
    return true;
  }

  /** Drop `slots` and shift every survivor down, ids included. */
  remove(slots: readonly number[]): void {
    const gone = new Set(slots);
    this.radii = this.radii.filter((_, slot) => !gone.has(slot));
    this.ids = this.ids.filter((_, slot) => !gone.has(slot));
  }
}

function declare(
  table: ProxyRadiusTable,
  slots: Slots,
  proxies: readonly number[],
  radii: readonly number[],
): number[] {
  return table.declare(proxies, proxies.length, radii, slots).sort((a, b) => a - b);
}

Deno.test("proxy radius: a declared proxy takes the well radius and reports the change", () => {
  const slots = new Slots([5, 6, 7]);
  const table = new ProxyRadiusTable();

  assertEquals(declare(table, slots, [1], [90]), [1]);
  assertEquals(slots.radii, [5, 90, 7]);
  assertEquals(table.size, 1);
});

Deno.test("proxy radius: a slot dropped from the set gets its own radius back", () => {
  const slots = new Slots([5, 6, 7]);
  const table = new ProxyRadiusTable();

  declare(table, slots, [0, 2], [90, 91]);
  assertEquals(slots.radii, [90, 6, 91]);

  assertEquals(declare(table, slots, [2], [91]), [0]);
  assertEquals(slots.radii, [5, 6, 91]);
  assertEquals(table.size, 1);
});

Deno.test("proxy radius: re-declaring the same proxy never saves the inflated value", () => {
  const slots = new Slots([5, 6, 7]);
  const table = new ProxyRadiusTable();

  // The well radius grows as the subtree moves; the *original* must survive
  // every re-declaration, or the node ratchets up and never comes back.
  declare(table, slots, [1], [90]);
  declare(table, slots, [1], [120]);
  declare(table, slots, [1], [150]);
  assertEquals(slots.radii, [5, 150, 7]);

  declare(table, slots, [], []);
  assertEquals(slots.radii, [5, 6, 7]);
  assertEquals(table.size, 0);
});

Deno.test("proxy radius: an unchanged radius is not reported as a write", () => {
  const slots = new Slots([5, 6, 7]);
  const table = new ProxyRadiusTable();

  // Declaring a node at the radius it already has costs no queue operation.
  assertEquals(declare(table, slots, [1], [6]), []);
  assertEquals(slots.radii, [5, 6, 7]);
});

Deno.test("proxy radius: release restores everything and empties the table", () => {
  const slots = new Slots([5, 6, 7]);
  const table = new ProxyRadiusTable();

  declare(table, slots, [0, 2], [90, 91]);
  assertEquals(table.release(slots).sort((a, b) => a - b), [0, 2]);
  assertEquals(slots.radii, [5, 6, 7]);
  assertEquals(table.size, 0);
});

Deno.test("proxy radius: release follows the node through a slot compaction", () => {
  const slots = new Slots([5, 6, 7, 8]);
  const table = new ProxyRadiusTable();

  declare(table, slots, [2, 3], [90, 91]);
  assertEquals(slots.radii, [5, 6, 90, 91]);

  // Slot 0 is removed, so the two proxies are now at slots 1 and 2. A restore
  // by saved slot would put 7 on the node that used to be node-1 and leave the
  // real owner at 90 for the rest of its life.
  slots.remove([0]);
  assertEquals(slots.radii, [6, 90, 91]);

  table.release(slots);
  assertEquals(slots.radii, [6, 7, 8]);
});

Deno.test("proxy radius: a proxy the mutation deleted is dropped, not misapplied", () => {
  const slots = new Slots([5, 6, 7]);
  const table = new ProxyRadiusTable();

  declare(table, slots, [1], [90]);
  slots.remove([1]);
  assertEquals(slots.radii, [5, 7]);

  assertEquals(table.release(slots), []);
  assertEquals(slots.radii, [5, 7], "a dead node's radius belongs to nothing");
  assertEquals(table.size, 0);
});

Deno.test("proxy radius: a slot with no producer id is still restored in place", () => {
  const slots = new Slots([5, 6, 7]);
  slots.ids[1] = undefined as unknown as IdLike;
  const table = new ProxyRadiusTable();

  declare(table, slots, [1], [90]);
  assertEquals(slots.radii, [5, 90, 7]);

  // `release` cannot chase an id that does not exist, but `declare` addresses
  // slots directly, so the ordinary expand path is unaffected.
  assertEquals(declare(table, slots, [], []), [1]);
  assertEquals(slots.radii, [5, 6, 7]);
});
