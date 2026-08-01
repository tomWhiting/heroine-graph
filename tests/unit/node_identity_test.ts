/**
 * Slot ↔ producer-identifier resolution.
 *
 * These two functions are the whole of `GraphMother.getExternalId` /
 * `GraphMother.getNodeId` (the methods add a null-state guard and nothing
 * else), so the round-trip, the recycled-slot hazard and the never-throws
 * promise are pinned here rather than behind a GPU device.
 *
 * The states are built exactly as `load()` builds them — parser output fed to
 * `MutableGraphState.fromParsedGraph` — and mutated exactly as the removal
 * path mutates them: free the slot, then drop the id from the map.
 */

import { assertEquals } from "jsr:@std/assert@^1";
import { MutableGraphState } from "../../packages/core/src/api/graph_state.ts";
import { parseGraphInput } from "../../packages/core/src/graph/parser.ts";
import { externalIdForSlot, slotForExternalId } from "../../packages/core/src/overlay/identity.ts";
import { microTreeGraph, triangleGraph } from "../fixtures/tiny_graphs.ts";

function loadedState(input = microTreeGraph()): MutableGraphState {
  return MutableGraphState.fromParsedGraph(parseGraphInput(input));
}

/** Mirror of GraphMother's node removal: free the slot, then unmap the id. */
function removeNode(gs: MutableGraphState, slot: number): void {
  const id = gs.nodeIdMap.getId(slot);
  if (id === undefined) throw new Error(`slot ${slot} holds no node`);
  gs.freeNodeSlot(slot);
  gs.nodeIdMap.remove(id);
}

Deno.test("identity: every loaded node round-trips slot -> id -> slot", () => {
  const gs = loadedState();
  const expected = ["root", "src", "readme", "main", "util"];

  assertEquals(gs.nodeHighWater, expected.length);
  for (let slot = 0; slot < expected.length; slot++) {
    const externalId = externalIdForSlot(gs, slot);
    assertEquals(externalId, expected[slot]);
    assertEquals(slotForExternalId(gs, externalId!), slot);
  }
});

Deno.test("identity: numeric producer ids survive the round-trip untouched", () => {
  const gs = loadedState({
    nodes: [{ id: 10 }, { id: 0 }, { id: -3 }],
    edges: [],
  });

  assertEquals(externalIdForSlot(gs, 0), 10);
  assertEquals(externalIdForSlot(gs, 1), 0);
  assertEquals(externalIdForSlot(gs, 2), -3);
  // A producer id of 0 or 1 must not be confused with the slot of the same value
  assertEquals(slotForExternalId(gs, 0), 1);
  assertEquals(slotForExternalId(gs, 10), 0);
  assertEquals(slotForExternalId(gs, -3), 2);
  // ...and the string spelling of a numeric id is a different id
  assertEquals(slotForExternalId(gs, "10"), undefined);
});

Deno.test("identity: out-of-range and non-integer slots resolve to undefined, never throw", () => {
  const gs = loadedState(triangleGraph());

  assertEquals(gs.nodeHighWater, 3);
  assertEquals(externalIdForSlot(gs, -1), undefined);
  assertEquals(externalIdForSlot(gs, 3), undefined);
  assertEquals(externalIdForSlot(gs, 1_000_000), undefined);
  assertEquals(externalIdForSlot(gs, 1.5), undefined);
  assertEquals(externalIdForSlot(gs, Number.NaN), undefined);
  assertEquals(externalIdForSlot(gs, Number.POSITIVE_INFINITY), undefined);
});

Deno.test("identity: unknown producer ids resolve to undefined", () => {
  const gs = loadedState();

  assertEquals(slotForExternalId(gs, "nope"), undefined);
  assertEquals(slotForExternalId(gs, 0), undefined);
  // Map internals must not leak through as identifiers
  assertEquals(slotForExternalId(gs, "toString"), undefined);
});

Deno.test("identity: a removed node's slot and id both go dead", () => {
  const gs = loadedState();
  removeNode(gs, 2); // "readme"

  // The stale NodeId a card was holding must resolve to nothing, not throw
  assertEquals(externalIdForSlot(gs, 2), undefined);
  assertEquals(slotForExternalId(gs, "readme"), undefined);

  // Neighbouring slots are unaffected
  assertEquals(externalIdForSlot(gs, 1), "src");
  assertEquals(externalIdForSlot(gs, 3), "main");
});

Deno.test("identity: a recycled slot resolves to its new occupant only", () => {
  const gs = loadedState();
  removeNode(gs, 2);

  const slot = gs.allocateNodeSlot();
  assertEquals(slot, 2, "removal frees slot 2 for LIFO reuse");
  gs.nodeIdMap.set("license", slot);

  assertEquals(externalIdForSlot(gs, 2), "license");
  assertEquals(slotForExternalId(gs, "license"), 2);
  // The evicted id does not come back with the slot
  assertEquals(slotForExternalId(gs, "readme"), undefined);
});

Deno.test("identity: nodeFreeSet is authoritative even if the id map disagrees", () => {
  const gs = loadedState();
  // Half-applied removal: the slot is freed but the id map has not caught up.
  // Resolving either direction here would hand out a slot the simulation has
  // already marked dead.
  gs.freeNodeSlot(4);

  assertEquals(gs.nodeIdMap.getId(4), "util", "precondition: id map still maps the slot");
  assertEquals(externalIdForSlot(gs, 4), undefined);
  assertEquals(slotForExternalId(gs, "util"), undefined);
});

Deno.test("identity: a slot at or beyond the high-water mark is not live", () => {
  const gs = loadedState();
  // An id map entry can outlive the slot space it was allocated in (a reload
  // shrinks nodeHighWater); high-water is the bound that matters.
  gs.nodeIdMap.set("ghost", 99);

  assertEquals(slotForExternalId(gs, "ghost"), undefined);
  assertEquals(externalIdForSlot(gs, 99), undefined);
});
