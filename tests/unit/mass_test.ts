/**
 * Unit tests for the collapsed-subtree mass rollup (rollUpMass in
 * packages/core/src/lod/mass.ts).
 *
 * The rollup is what stops a collapse from changing the layout of what stays
 * visible: the proxy must weigh exactly what its hidden subtree weighed, and
 * the hidden nodes must weigh nothing so the total is unchanged. Everything
 * below is a statement of one of those two halves, or of the walk that has to
 * get the *lowest visible* ancestor right when collapse sets nest.
 *
 * Pure TS — the module touches no GPU and no shader.
 */

import { assertAlmostEquals, assertEquals, assertThrows } from "jsr:@std/assert@^1";
import {
  commitNodeMass,
  NODE_MASS_HIDDEN,
  NODE_MASS_UNIT,
  rollUpMass,
} from "../../packages/core/src/lod/mass.ts";
import {
  buildChildrenCsr,
  type ChildrenCsr,
  HIERARCHY_ROOT,
} from "../../packages/core/src/graph/hierarchy.ts";
import { GraphMotherError } from "../../packages/core/src/errors.ts";
import { generateCodeTree } from "../fixtures/code_tree.ts";

/** A parent column plus its CSR, from a `parent[slot]` list (-1 = root). */
function tree(parents: readonly number[]): { parent: Uint32Array; children: ChildrenCsr } {
  const parent = Uint32Array.from(parents, (p) => (p < 0 ? HIERARCHY_ROOT : p));
  return { parent, children: buildChildrenCsr(parent, parent.length) };
}

/**
 *      0
 *    /   \
 *   1     2
 *  / \    |
 * 3   4   5
 *         |
 *         6
 */
const SAMPLE = [-1, 0, 0, 1, 1, 2, 5];

function sum(mass: Float32Array): number {
  let total = 0;
  for (const m of mass) total += m;
  return total;
}

/** Slots reported hidden, i.e. holding NODE_MASS_HIDDEN. */
function hiddenSlots(mass: Float32Array): number[] {
  const hidden: number[] = [];
  for (let i = 0; i < mass.length; i++) {
    if (mass[i] === NODE_MASS_HIDDEN) hidden.push(i);
  }
  return hidden;
}

Deno.test("rollUpMass: an empty collapse set leaves every slot at unit mass", () => {
  const { parent, children } = tree(SAMPLE);
  const mass = rollUpMass(parent, children, []);
  assertEquals(Array.from(mass), [1, 1, 1, 1, 1, 1, 1]);
});

Deno.test("rollUpMass: a collapsed subtree's mass moves into its proxy", () => {
  const { parent, children } = tree(SAMPLE);
  // Collapse node 1, which hides its two leaves.
  const mass = rollUpMass(parent, children, [1]);
  assertEquals(Array.from(mass), [1, 3, 1, 0, 0, 1, 1]);
  assertEquals(hiddenSlots(mass), [3, 4]);
});

Deno.test("rollUpMass: collapsing a deep chain carries every descendant up", () => {
  const { parent, children } = tree(SAMPLE);
  // Node 2 -> 5 -> 6: collapsing 2 hides two generations.
  const mass = rollUpMass(parent, children, [2]);
  assertEquals(Array.from(mass), [1, 1, 3, 1, 1, 0, 0]);
});

Deno.test("rollUpMass: nested collapse sets roll to the lowest VISIBLE ancestor", () => {
  const { parent, children } = tree(SAMPLE);
  // 5 is inside 2's subtree, so collapsing both must not leave mass at 5:
  // 5 is itself hidden and its subtree passes through it to 2.
  const nested = rollUpMass(parent, children, [2, 5]);
  assertEquals(Array.from(nested), [1, 1, 3, 1, 1, 0, 0]);

  // ...which is exactly what collapsing the outer node alone produces.
  const outerOnly = rollUpMass(parent, children, [2]);
  assertEquals(Array.from(nested), Array.from(outerOnly));

  // Collapsing the whole forest at the root leaves all the mass at the root.
  const atRoot = rollUpMass(parent, children, [0, 1, 2, 5]);
  assertEquals(Array.from(atRoot), [7, 0, 0, 0, 0, 0, 0]);
});

Deno.test("rollUpMass: a collapsed leaf hides nothing and keeps unit mass", () => {
  const { parent, children } = tree(SAMPLE);
  const mass = rollUpMass(parent, children, [3, 6]);
  assertEquals(Array.from(mass), [1, 1, 1, 1, 1, 1, 1]);
  assertEquals(hiddenSlots(mass), []);
});

Deno.test("rollUpMass: forests collapse independently, roots included", () => {
  //  0        3
  //  |       / \
  //  1      4   5
  //  |
  //  2
  const { parent, children } = tree([-1, 0, 1, -1, 3, 3]);
  const mass = rollUpMass(parent, children, [0, 4]);
  assertEquals(Array.from(mass), [3, 0, 0, 1, 1, 1]);

  const other = rollUpMass(parent, children, [3]);
  assertEquals(Array.from(other), [1, 1, 1, 3, 0, 0]);
});

Deno.test("rollUpMass: total mass is conserved for every collapse set", () => {
  const { parent, children } = tree(SAMPLE);
  const nodeCount = parent.length;
  // Every subset of the 7 slots — 128 collapse sets, including the nested and
  // degenerate ones. Total mass is the invariant that makes a collapse
  // layout-neutral, so it must hold for all of them, not a sampled few.
  for (let bits = 0; bits < 1 << nodeCount; bits++) {
    const collapsed: number[] = [];
    for (let i = 0; i < nodeCount; i++) {
      if (bits & (1 << i)) collapsed.push(i);
    }
    const mass = rollUpMass(parent, children, collapsed);
    assertEquals(
      sum(mass),
      nodeCount,
      `collapse set {${collapsed.join(",")}} does not conserve mass`,
    );
  }
});

Deno.test("rollUpMass: conserves mass on a code-tree fixture at every depth cut", () => {
  const graph = generateCodeTree({ seed: 7, maxNodes: 600 });
  const parent = Uint32Array.from(graph.parent, (p) => (p < 0 ? HIERARCHY_ROOT : p));
  const children = buildChildrenCsr(parent, graph.nodeCount);

  for (let depth = 0; depth <= 4; depth++) {
    const collapsed: number[] = [];
    for (let i = 0; i < graph.nodeCount; i++) {
      if (graph.depths[i] === depth) collapsed.push(i);
    }
    const mass = rollUpMass(parent, children, collapsed);
    // Float32 accumulation of a few hundred unit adds is exact, but state the
    // tolerance rather than relying on it.
    assertAlmostEquals(
      sum(mass),
      graph.nodeCount,
      1e-3,
      `collapsing every depth-${depth} node does not conserve mass`,
    );

    // Every hidden slot has a collapsed ancestor, and every proxy weighs
    // exactly one more than the descendants it hides.
    for (const root of collapsed) {
      let descendants = 0;
      const stack = [root];
      while (stack.length > 0) {
        const slot = stack.pop()!;
        for (let c = children.offsets[slot]; c < children.offsets[slot + 1]; c++) {
          descendants++;
          stack.push(children.children[c]);
        }
      }
      assertEquals(mass[root], descendants + 1, `proxy ${root} carries the wrong mass`);
    }
  }
});

Deno.test("rollUpMass: reusing an output array does not leak the previous transition", () => {
  const { parent, children } = tree(SAMPLE);
  const out = new Float32Array(parent.length);

  const collapsed = rollUpMass(parent, children, [1], out);
  assertEquals(collapsed, out, "the supplied array must be the one returned");
  assertEquals(Array.from(out), [1, 3, 1, 0, 0, 1, 1]);

  // Expanding again must return every slot to unit mass through the same
  // array — a proxy left heavy is a layout that never re-inflates.
  rollUpMass(parent, children, [], out);
  assertEquals(Array.from(out), [1, 1, 1, 1, 1, 1, 1]);
});

Deno.test("rollUpMass: a longer output array is filled only over the slot space", () => {
  const { parent, children } = tree(SAMPLE);
  // Capacity beyond nodeCount, as a GPU buffer shadow has.
  const out = new Float32Array(10).fill(-1);
  rollUpMass(parent, children, [1], out);
  assertEquals(Array.from(out.subarray(0, 7)), [1, 3, 1, 0, 0, 1, 1]);
  assertEquals(Array.from(out.subarray(7)), [-1, -1, -1], "padding must be left alone");
});

Deno.test("rollUpMass: malformed input throws instead of producing wrong masses", () => {
  const { parent, children } = tree(SAMPLE);

  assertThrows(
    () => rollUpMass(parent, children, [7]),
    GraphMotherError,
    "collapsed root 7",
  );
  assertThrows(
    () => rollUpMass(parent, children, [-1]),
    GraphMotherError,
    "collapsed root -1",
  );
  assertThrows(
    () => rollUpMass(parent, children, [1], new Float32Array(6)),
    GraphMotherError,
    "at least nodeCount",
  );
  assertThrows(
    () => rollUpMass(parent, buildChildrenCsr(parent.subarray(0, 5), 5), [1]),
    GraphMotherError,
    "offsets, expected nodeCount + 1",
  );
});

// =============================================================================
// Upload range (commitNodeMass — the diff behind GraphMother.uploadNodeMass)
// =============================================================================

/** A shadow of `length` slots, all at unit mass, as a fresh GPU buffer holds. */
function unitShadow(length: number): Float32Array {
  return new Float32Array(length).fill(NODE_MASS_UNIT);
}

Deno.test("commitNodeMass: an unchanged upload reports an empty range", () => {
  const shadow = unitShadow(8);
  assertEquals(commitNodeMass(shadow, unitShadow(8)), { lo: 0, hi: 0 });
  assertEquals(commitNodeMass(shadow, unitShadow(5)), { lo: 0, hi: 0 });
});

Deno.test("commitNodeMass: the range spans exactly the changed slots", () => {
  const shadow = unitShadow(8);
  const next = unitShadow(8);
  next[3] = 4;
  next[5] = NODE_MASS_HIDDEN;

  assertEquals(commitNodeMass(shadow, next), { lo: 3, hi: 6 });
  assertEquals(Array.from(shadow), [1, 1, 1, 4, 1, 0, 1, 1]);

  // Committing the same masses again is a no-op: the shadow now agrees.
  assertEquals(commitNodeMass(shadow, next), { lo: 0, hi: 0 });
});

Deno.test("commitNodeMass: a shorter upload leaves the capacity tail alone", () => {
  const shadow = unitShadow(8);
  shadow[7] = 99; // capacity padding the caller does not describe
  const next = unitShadow(4);
  next[0] = 3;

  assertEquals(commitNodeMass(shadow, next), { lo: 0, hi: 1 });
  assertEquals(shadow[7], 99);
});

Deno.test("commitNodeMass: expanding back to unit mass is itself a diff", () => {
  const shadow = unitShadow(6);
  const collapsed = Float32Array.from([1, 4, 0, 0, 0, 1]);
  assertEquals(commitNodeMass(shadow, collapsed), { lo: 1, hi: 5 });

  const expanded = commitNodeMass(shadow, unitShadow(6));
  assertEquals(expanded, { lo: 1, hi: 5 }, "a proxy left heavy never re-inflates");
  assertEquals(Array.from(shadow), [1, 1, 1, 1, 1, 1]);
});

Deno.test("commitNodeMass: rejects masses that would poison the layout", () => {
  assertThrows(
    () => commitNodeMass(unitShadow(4), Float32Array.from([1, NaN, 1, 1])),
    GraphMotherError,
    "slot 1 is NaN",
  );
  assertThrows(
    () => commitNodeMass(unitShadow(4), Float32Array.from([1, 1, -2, 1])),
    GraphMotherError,
    "slot 2 is -2",
  );
  assertThrows(
    () => commitNodeMass(unitShadow(4), Float32Array.from([1, 1, 1, Infinity])),
    GraphMotherError,
    "slot 3 is Infinity",
  );
  assertThrows(
    () => commitNodeMass(unitShadow(4), unitShadow(5)),
    GraphMotherError,
    "exceeding the shadow length",
  );
});

Deno.test("commitNodeMass: a rejected upload leaves the shadow untouched", () => {
  // The shadow must keep describing what the GPU actually holds, or the next
  // upload diffs against a lie and skips slots the GPU never received.
  const shadow = unitShadow(4);
  assertThrows(
    () => commitNodeMass(shadow, Float32Array.from([2, 3, NaN, 5])),
    GraphMotherError,
  );
  assertEquals(Array.from(shadow), [1, 1, 1, 1]);
});
