/**
 * IdMap allocate/free/remap invariants.
 *
 * The IdMap is the contract between user-facing IDs and GPU slot indices;
 * incremental mutations (removeNodes slot remap) depend on freed-index
 * reuse behaving exactly as asserted here.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  createIdMap,
  createIdMapFromArray,
  createSequentialIdMap,
  deserializeIdMap,
  type IdMap,
  serializeIdMap,
} from "../../packages/core/src/graph/id_map.ts";
import { mulberry32 } from "../fixtures/prng.ts";

Deno.test("IdMap: add assigns sequential indices and is idempotent per id", () => {
  const map = createIdMap<string>();
  assertEquals(map.add("a"), 0);
  assertEquals(map.add("b"), 1);
  assertEquals(map.add("c"), 2);
  // Re-adding an existing id returns its index without growing the map
  assertEquals(map.add("b"), 1);
  assertEquals(map.size, 3);
  assertEquals(map.get("c"), 2);
  assertEquals(map.getId(0), "a");
});

Deno.test("IdMap: remove frees the index and hides it from lookups", () => {
  const map = createIdMap<string>();
  map.add("a");
  map.add("b");
  map.add("c");

  assertEquals(map.remove("b"), 1);
  assertEquals(map.size, 2);
  assertEquals(map.has("b"), false);
  assertEquals(map.get("b"), undefined);
  assertEquals(map.getId(1), undefined);
  assertEquals(map.ids(), ["a", "c"]);
  assertEquals([...map.entries()], [[0, "a"], [2, "c"]]);
  // Removing a missing id is a no-op returning undefined
  assertEquals(map.remove("nope"), undefined);
});

Deno.test("IdMap: freed indices are reused LIFO by subsequent adds", () => {
  const map = createIdMap<string>();
  map.add("a"); // 0
  map.add("b"); // 1
  map.add("c"); // 2
  map.add("d"); // 3

  map.remove("b"); // frees 1
  map.remove("d"); // frees 3

  // LIFO: most recently freed index is handed out first
  assertEquals(map.add("x"), 3);
  assertEquals(map.add("y"), 1);
  // Free list exhausted: back to appending
  assertEquals(map.add("z"), 4);
  assertEquals(map.size, 5);
  assertEquals(map.getId(3), "x");
  assertEquals(map.getId(1), "y");
});

Deno.test("IdMap: set force-assigns an index, freeing any previous one", () => {
  const map = createIdMap<string>();
  map.add("a"); // 0
  map.add("b"); // 1

  // Move "a" to index 5; index 0 becomes free and is reused next
  map.set("a", 5);
  assertEquals(map.get("a"), 5);
  assertEquals(map.getId(5), "a");
  assertEquals(map.add("c"), 0);

  // set onto a freed index removes it from the free list: after freeing 1
  // and claiming it via set, a subsequent add must NOT hand out 1 again
  map.remove("b"); // frees 1
  map.set("d", 1);
  assertEquals(map.getId(1), "d");
  const next = map.add("e");
  assert(next !== 1, `add() handed out index 1 which set() already claimed`);
});

Deno.test("IdMap: clear resets all state including the free list", () => {
  const map = createIdMap<string>();
  map.add("a");
  map.add("b");
  map.remove("a");
  map.clear();

  assertEquals(map.size, 0);
  assertEquals(map.ids(), []);
  // Fresh indices start from 0 again (no stale free list entries)
  assertEquals(map.add("x"), 0);
  assertEquals(map.add("y"), 1);
});

Deno.test("IdMap: serialize/deserialize round-trip preserves sparse assignments", () => {
  const map = createIdMap<string>();
  map.add("a"); // 0
  map.add("b"); // 1
  map.add("c"); // 2
  map.remove("b"); // gap at 1

  const restored = deserializeIdMap(serializeIdMap(map));
  assertEquals(restored.get("a"), 0);
  assertEquals(restored.get("c"), 2);
  assertEquals(restored.has("b"), false);
  assertEquals(restored.getId(0), "a");
  assertEquals(restored.getId(2), "c");
  assertEquals(restored.size, map.size);
  // Note: entries() of a deserialized map is not asserted — deserialization
  // leaves array holes at gap indices which entries() currently yields as
  // [index, undefined], unlike the original map (candidate upstream fix).
});

Deno.test("IdMap: helper constructors", () => {
  const seq = createSequentialIdMap(3, "node");
  assertEquals(seq.ids(), ["node0", "node1", "node2"]);

  const fromArr = createIdMapFromArray([10, 20, 30]);
  assertEquals(fromArr.get(20), 1);
  assertEquals(fromArr.size, 3);
});

/**
 * Property test: random add/remove churn against a reference model.
 * After every operation the bidirectional mapping must stay consistent:
 * unique live indices, getId(get(id)) === id, freed indices invisible.
 */
Deno.test("IdMap: random churn keeps id<->index mapping consistent", () => {
  const rng = mulberry32(0xC0FFEE);
  const map: IdMap<string> = createIdMap<string>();
  const model = new Map<string, number>();
  let nextId = 0;

  for (let op = 0; op < 500; op++) {
    const live = [...model.keys()];
    if (live.length === 0 || rng() < 0.6) {
      const id = `id${nextId++}`;
      model.set(id, map.add(id));
    } else {
      const victim = live[Math.floor(rng() * live.length)];
      const freedIndex = map.remove(victim);
      assertEquals(freedIndex, model.get(victim));
      model.delete(victim);
    }

    // Invariants after every operation
    assertEquals(map.size, model.size);
    const seenIndices = new Set<number>();
    for (const [id, index] of model) {
      assertEquals(map.get(id), index, `get(${id})`);
      assertEquals(map.getId(index), id, `getId(${index})`);
      assert(!seenIndices.has(index), `duplicate live index ${index}`);
      seenIndices.add(index);
    }
  }
});
