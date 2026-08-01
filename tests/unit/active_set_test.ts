/**
 * Unit tests for the active-set derivation
 * (packages/core/src/simulation/active_set.ts).
 *
 * The list is what the force passes dispatch over, so a slot this function
 * omits is a node that silently stops moving, and a slot it emits out of order
 * changes the summation order of every all-pairs shader — which is the one
 * thing SC-005's bit-exact comparison cannot survive. Both properties are
 * asserted directly rather than inferred from a GPU run.
 *
 * active_set.ts reaches pipeline.ts for the flag constants and so cannot be
 * imported statically here (see tests/helpers/gpu.ts module doc); it is loaded
 * through the same inlining loader the GPU tests use. No GPU is touched: these
 * are pure array functions.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  loadActiveSetModule,
  NODE_FLAG_DEAD,
  NODE_FLAG_HIDDEN_LOD,
  NODE_FLAG_PINNED,
} from "../helpers/gpu.ts";
import { generateCodeTree } from "../fixtures/code_tree.ts";
import { mulberry32 } from "../fixtures/prng.ts";

const { deriveActiveIndices, activeIndicesUnchanged } = await loadActiveSetModule();

/** The derived list as a plain array, for readable assertions. */
function derive(flags: Uint32Array, slotCount = flags.length): number[] {
  const out = new Uint32Array(flags.length);
  const count = deriveActiveIndices(flags, slotCount, out);
  return Array.from(out.subarray(0, count));
}

Deno.test("deriveActiveIndices: with no flags set, every slot is active in slot order", () => {
  assertEquals(derive(new Uint32Array(6)), [0, 1, 2, 3, 4, 5]);
});

Deno.test("deriveActiveIndices: dead and hidden slots drop out, pinned ones do not", () => {
  const flags = new Uint32Array([
    0,
    NODE_FLAG_DEAD,
    NODE_FLAG_PINNED,
    NODE_FLAG_HIDDEN_LOD,
    NODE_FLAG_PINNED | NODE_FLAG_HIDDEN_LOD,
    NODE_FLAG_DEAD | NODE_FLAG_HIDDEN_LOD,
    0,
  ]);
  // A pinned node still exerts force on its neighbours, so it stays in the
  // dispatch; only the two inert bits remove a slot.
  assertEquals(derive(flags), [0, 2, 6]);
});

Deno.test("deriveActiveIndices: the result is strictly ascending", () => {
  const random = mulberry32(7);
  const flags = new Uint32Array(500);
  for (let i = 0; i < flags.length; i++) {
    flags[i] = random() < 0.4 ? NODE_FLAG_HIDDEN_LOD : (random() < 0.1 ? NODE_FLAG_DEAD : 0);
  }
  const list = derive(flags);
  assert(list.length > 0 && list.length < flags.length, "the fixture must hide some but not all");
  for (let k = 1; k < list.length; k++) {
    assert(list[k] > list[k - 1], `entry ${k} (${list[k]}) does not follow ${list[k - 1]}`);
  }
});

Deno.test("deriveActiveIndices: slotCount bounds the scan at the high-water mark", () => {
  // Slots at or above nodeHighWater have never been assigned; their flag word
  // is zero, which would read as "active" and dispatch threads at coordinates
  // no node owns.
  const flags = new Uint32Array(8);
  assertEquals(derive(flags, 3), [0, 1, 2]);
});

Deno.test("deriveActiveIndices: the count is the return value, not the array length", () => {
  const out = new Uint32Array(5).fill(0xEEEE);
  const flags = new Uint32Array([0, NODE_FLAG_DEAD, 0, NODE_FLAG_HIDDEN_LOD, 0]);
  assertEquals(deriveActiveIndices(flags, 5, out), 3);
  assertEquals(Array.from(out), [0, 2, 4, 0xEEEE, 0xEEEE]);
});

Deno.test("deriveActiveIndices: an all-inert graph derives an empty list", () => {
  const flags = new Uint32Array(4).fill(NODE_FLAG_HIDDEN_LOD);
  assertEquals(derive(flags), []);
});

Deno.test("deriveActiveIndices: a reused output array does not leak the previous list", () => {
  const out = new Uint32Array(6);
  const first = deriveActiveIndices(new Uint32Array(6), 6, out);
  assertEquals(first, 6);

  const flags = new Uint32Array(6);
  flags[0] = NODE_FLAG_HIDDEN_LOD;
  flags[1] = NODE_FLAG_HIDDEN_LOD;
  const second = deriveActiveIndices(flags, 6, out);
  assertEquals(Array.from(out.subarray(0, second)), [2, 3, 4, 5]);
});

Deno.test("activeIndicesUnchanged: agrees with deriveActiveIndices on the code tree", () => {
  const tree = generateCodeTree({ seed: 21, maxNodes: 3_000 });
  const flags = new Uint32Array(tree.nodeCount);
  const out = new Uint32Array(tree.nodeCount);
  let count = deriveActiveIndices(flags, tree.nodeCount, out);
  assert(activeIndicesUnchanged(flags, tree.nodeCount, out, count));

  // Collapse every node below depth 3: the shape an LOD cut produces.
  for (let i = 0; i < tree.nodeCount; i++) {
    if (tree.depths[i] > 3) flags[i] = NODE_FLAG_HIDDEN_LOD;
  }
  assert(
    !activeIndicesUnchanged(flags, tree.nodeCount, out, count),
    "hiding most of the tree must be reported as a change",
  );

  count = deriveActiveIndices(flags, tree.nodeCount, out);
  assert(count > 0 && count < tree.nodeCount, `implausible cut size ${count}`);
  assert(activeIndicesUnchanged(flags, tree.nodeCount, out, count));

  // Pinning changes flags without changing the active set — the case that
  // makes the comparison worth doing at all.
  for (let i = 0; i < tree.nodeCount; i += 17) flags[i] |= NODE_FLAG_PINNED;
  assert(activeIndicesUnchanged(flags, tree.nodeCount, out, count));
});

Deno.test("activeIndicesUnchanged: a stale count is reported as a change either way", () => {
  const flags = new Uint32Array(5);
  const out = Uint32Array.from([0, 1, 2, 3, 4]);
  assert(!activeIndicesUnchanged(flags, 5, out, 4), "a short count must not read as unchanged");
  assert(!activeIndicesUnchanged(flags, 4, out, 5), "a long count must not read as unchanged");
});

Deno.test("activeIndicesUnchanged: a list holding the wrong slot is reported as a change", () => {
  const flags = new Uint32Array([0, NODE_FLAG_HIDDEN_LOD, 0]);
  assert(!activeIndicesUnchanged(flags, 3, Uint32Array.from([0, 1]), 2));
  assert(activeIndicesUnchanged(flags, 3, Uint32Array.from([0, 2]), 2));
});
