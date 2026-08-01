/**
 * MutableGraphState slot allocation and node-flag invariants.
 *
 * Node slot reuse must be pure LIFO with no tail pruning so it stays in
 * lockstep with two other allocators fed the same add/remove sequence:
 * - the IdMap adopting slots via set(id, slot) (single free-list owner)
 * - petgraph StableGraph's vacancy list in the WASM engine (NodeId == slot)
 *
 * nodeFlagsShadow is the single CPU-side authority for the GPU nodeFlags
 * buffer. Its three bits (dead / pinned / hidden) have independent owners, so
 * every write must be a masked edit of the existing word — the recomposition
 * pattern it replaced dropped whichever bits the writer did not know about.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import { MutableGraphState, NODE_ATTR_FLOATS } from "../../packages/core/src/api/graph_state.ts";
import { createIdMap, type IdLike } from "../../packages/core/src/graph/id_map.ts";
import type { ParsedGraph } from "../../packages/core/src/graph/parser.ts";
import { NODE_FLAG_DEAD, NODE_FLAG_HIDDEN_LOD, NODE_FLAG_PINNED } from "../helpers/gpu.ts";

function makeParsed(nodeCount: number): ParsedGraph {
  const nodeIdMap = createIdMap<IdLike>();
  for (let i = 0; i < nodeCount; i++) nodeIdMap.add(`n${i}`);
  return {
    nodeCount,
    edgeCount: 0,
    nodeIdMap,
    edgeIdMap: createIdMap(),
    positionsX: new Float32Array(nodeCount).fill(1),
    positionsY: new Float32Array(nodeCount).fill(1),
    nodeAttributes: new Float32Array(nodeCount * NODE_ATTR_FLOATS).fill(5),
    edgeSources: new Uint32Array(0),
    edgeTargets: new Uint32Array(0),
    edgeAttributes: new Float32Array(0),
    nodeMetadata: new Map(),
    edgeMetadata: new Map(),
  };
}

Deno.test("MutableGraphState: freed slots are reused LIFO with no tail pruning", () => {
  const gs = MutableGraphState.fromParsedGraph(makeParsed(10));
  assertEquals(gs.nodeHighWater, 10);

  gs.freeNodeSlot(5);
  gs.freeNodeSlot(9); // tail slot — must NOT shrink the high-water mark

  // petgraph StableGraph keeps freed tail indices in its vacancy list, so
  // the high-water mark must not shrink and the free list must not be pruned
  assertEquals(gs.nodeHighWater, 10);
  assertEquals(gs.nodeCount, 8);
  assertEquals(gs.nodeFreeList, [5, 9]);

  // LIFO reuse: most recently freed first, then older, then fresh tail
  assertEquals(gs.allocateNodeSlot(), 9);
  assertEquals(gs.allocateNodeSlot(), 5);
  assertEquals(gs.allocateNodeSlot(), 10);
  assertEquals(gs.nodeHighWater, 11);
  assertEquals(gs.nodeCount, 11);
  assertEquals(gs.nodeFreeList.length, 0);
});

Deno.test("MutableGraphState: freeNodeSlot zeroes the slot's shadow data", () => {
  const gs = MutableGraphState.fromParsedGraph(makeParsed(4));
  gs.freeNodeSlot(2);

  assertEquals(gs.positionsX[2], 0);
  assertEquals(gs.positionsY[2], 0);
  for (let k = 0; k < NODE_ATTR_FLOATS; k++) {
    assertEquals(gs.nodeAttributes[2 * NODE_ATTR_FLOATS + k], 0);
  }
  assert(gs.nodeFreeSet.has(2));
});

Deno.test("MutableGraphState: ID map stays aligned with slot allocation across remove/add", () => {
  // Regression for the free-list divergence: after removing slot 5 then the
  // tail slot 9, the old high-water pruning made allocateNodeSlot() return 5
  // while the IdMap's own free list handed out 9 — mapping an added node's ID
  // to a slot its data was never written to.
  const gs = MutableGraphState.fromParsedGraph(makeParsed(10));

  // Mirror graph.ts removeNode: drop the ID mapping, then free the slot
  for (const slot of [5, 9]) {
    gs.nodeIdMap.remove(gs.nodeIdMap.getId(slot)!);
    gs.freeNodeSlot(slot);
  }

  // Mirror graph.ts addNode: allocate the slot, ID map adopts it via set()
  const slotX = gs.allocateNodeSlot();
  gs.nodeIdMap.set("X", slotX);
  assertEquals(slotX, 9);
  assertEquals(gs.nodeIdMap.get("X"), slotX);
  assertEquals(gs.nodeIdMap.getId(slotX), "X");

  const slotY = gs.allocateNodeSlot();
  gs.nodeIdMap.set("Y", slotY);
  assertEquals(slotY, 5);
  assertEquals(gs.nodeIdMap.get("Y"), slotY);

  // All live IDs resolve to unique slots below the high-water mark
  const seen = new Set<number>();
  for (const [index] of gs.nodeIdMap.entries()) {
    assert(index < gs.nodeHighWater, `index ${index} >= highWater`);
    assert(!seen.has(index), `duplicate slot ${index}`);
    seen.add(index);
  }
  assertEquals(seen.size, gs.nodeCount);
});

Deno.test("MutableGraphState: a fresh state is all-live, all-visible, unpinned", () => {
  const gs = MutableGraphState.fromParsedGraph(makeParsed(4));

  // Must match the zero-filled GPU nodeFlags buffer created alongside it
  assertEquals(gs.nodeFlagsShadow.length, gs.nodeCapacity);
  for (let i = 0; i < gs.nodeHighWater; i++) assertEquals(gs.nodeFlagsShadow[i], 0);
});

Deno.test("MutableGraphState: each flag write preserves the bits it does not own", () => {
  const gs = MutableGraphState.fromParsedGraph(makeParsed(4));

  // A liveness write preserves the pin bit
  gs.setNodeFlagBits(0, NODE_FLAG_PINNED, true);
  gs.setNodeFlagBits(0, NODE_FLAG_DEAD, true);
  assertEquals(gs.nodeFlagsShadow[0], NODE_FLAG_DEAD | NODE_FLAG_PINNED);

  // A pin write preserves the hide bit
  gs.setNodeFlagBits(1, NODE_FLAG_HIDDEN_LOD, true);
  gs.setNodeFlagBits(1, NODE_FLAG_PINNED, true);
  assertEquals(gs.nodeFlagsShadow[1], NODE_FLAG_HIDDEN_LOD | NODE_FLAG_PINNED);

  // A hide write preserves both the dead and the pin bit
  gs.setNodeFlagBits(2, NODE_FLAG_DEAD, true);
  gs.setNodeFlagBits(2, NODE_FLAG_PINNED, true);
  gs.setNodeFlagBits(2, NODE_FLAG_HIDDEN_LOD, true);
  assertEquals(
    gs.nodeFlagsShadow[2],
    NODE_FLAG_DEAD | NODE_FLAG_PINNED | NODE_FLAG_HIDDEN_LOD,
  );

  // Clearing is equally narrow: unhiding leaves dead and pinned standing
  gs.setNodeFlagBits(2, NODE_FLAG_HIDDEN_LOD, false);
  assertEquals(gs.nodeFlagsShadow[2], NODE_FLAG_DEAD | NODE_FLAG_PINNED);

  // Neighbours are untouched throughout
  assertEquals(gs.nodeFlagsShadow[3], 0);
});

Deno.test("MutableGraphState: setNodeFlagBits reports whether the word changed", () => {
  const gs = MutableGraphState.fromParsedGraph(makeParsed(2));

  // The return value is what lets callers skip a redundant GPU write
  assert(gs.setNodeFlagBits(0, NODE_FLAG_HIDDEN_LOD, true), "first hide must report a change");
  assert(!gs.setNodeFlagBits(0, NODE_FLAG_HIDDEN_LOD, true), "re-hiding must report no change");
  assert(gs.setNodeFlagBits(0, NODE_FLAG_HIDDEN_LOD, false), "unhide must report a change");
  assert(!gs.setNodeFlagBits(0, NODE_FLAG_HIDDEN_LOD, false), "re-showing must report no change");
});

Deno.test("MutableGraphState: resetNodeFlags drops the previous occupant's state", () => {
  const gs = MutableGraphState.fromParsedGraph(makeParsed(2));

  // Slot reassignment (graph.ts writeNodeSlotLiveness) is the one write that
  // must NOT preserve bits: pin and visibility belonged to a node that is gone.
  gs.setNodeFlagBits(0, NODE_FLAG_PINNED | NODE_FLAG_HIDDEN_LOD, true);
  gs.resetNodeFlags(0, NODE_FLAG_DEAD);
  assertEquals(gs.nodeFlagsShadow[0], NODE_FLAG_DEAD);

  gs.resetNodeFlags(0, 0);
  assertEquals(gs.nodeFlagsShadow[0], 0);
});

Deno.test("MutableGraphState: growNodeCapacity carries the flag shadow across", () => {
  const gs = MutableGraphState.fromParsedGraph(makeParsed(4));
  gs.setNodeFlagBits(1, NODE_FLAG_PINNED, true);
  gs.setNodeFlagBits(3, NODE_FLAG_HIDDEN_LOD, true);

  gs.growNodeCapacity(gs.nodeCapacity * 2);

  assertEquals(gs.nodeFlagsShadow.length, gs.nodeCapacity);
  assertEquals(gs.nodeFlagsShadow[1], NODE_FLAG_PINNED);
  assertEquals(gs.nodeFlagsShadow[3], NODE_FLAG_HIDDEN_LOD);
  assertEquals(gs.nodeFlagsShadow[gs.nodeCapacity - 1], 0);
});
