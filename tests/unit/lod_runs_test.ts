/**
 * Contiguous slot runs.
 *
 * The shared primitive behind every LOD upload: whether a transition costs one
 * queue operation or one per node comes down to this function finding the runs
 * a depth-first producer hands it. Both the mass/radius path in `GraphMother`
 * and the expand fix-up in `LODController` go through it, so a defect here is a
 * defect in both.
 */

import { assertEquals } from "jsr:@std/assert@^1";
import { forEachSlotRun } from "../../packages/core/src/lod/runs.ts";

/** Every run the visitor was handed, as `[lo, hi)` pairs. */
function runsOf(slots: readonly number[]): [number, number][] {
  const runs: [number, number][] = [];
  forEachSlotRun(slots, slots.length, (lo, hi) => runs.push([lo, hi]));
  return runs;
}

Deno.test("slot runs: a depth-first subtree is exactly one range", () => {
  assertEquals(runsOf([4, 5, 6, 7, 8]), [[4, 9]]);
});

Deno.test("slot runs: a scattered set becomes one range per run", () => {
  assertEquals(runsOf([0, 1, 4, 9, 10, 11]), [[0, 2], [4, 5], [9, 12]]);
});

Deno.test("slot runs: a single slot is a one-wide range", () => {
  assertEquals(runsOf([7]), [[7, 8]]);
});

Deno.test("slot runs: isolated slots never merge", () => {
  assertEquals(runsOf([0, 2, 4]), [[0, 1], [2, 3], [4, 5]]);
});

Deno.test("slot runs: an empty list visits nothing", () => {
  assertEquals(runsOf([]), []);
});

Deno.test("slot runs: only the first `count` entries are read", () => {
  // Callers pass a reused scratch buffer with a live prefix, so anything past
  // `count` is last transition's data and must not reach the visitor.
  const scratch = [3, 4, 999, 1000];
  const runs: [number, number][] = [];
  forEachSlotRun(scratch, 2, (lo, hi) => runs.push([lo, hi]));
  assertEquals(runs, [[3, 5]]);
});

Deno.test("slot runs: a typed-array view is accepted as-is", () => {
  const slots = Uint32Array.from([2, 3, 4, 20]);
  const runs: [number, number][] = [];
  forEachSlotRun(slots, slots.length, (lo, hi) => runs.push([lo, hi]));
  assertEquals(runs, [[2, 5], [20, 21]]);
});
