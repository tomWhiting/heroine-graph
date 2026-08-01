/**
 * Semantic LOD columns (`tag`, `weight`) through both parser paths.
 *
 * These two columns are the only thing a producer can say to the LOD policy,
 * and core never interprets them — so what has to be pinned down is that they
 * arrive intact and indexed by the right slot, that a wrong-length column is
 * rejected rather than silently reading 0 past its end, and that a graph
 * supplying neither pays no allocation for them.
 */

import { assert, assertEquals, assertThrows } from "jsr:@std/assert@^1";
import { parseGraphInput } from "../../packages/core/src/graph/parser.ts";
import {
  parseGraphTypedInput,
  validateGraphTypedInput,
} from "../../packages/core/src/graph/typed_parser.ts";
import { GraphMotherError } from "../../packages/core/src/errors.ts";

// =============================================================================
// Typed path
// =============================================================================

Deno.test("typed parser: tag and weight land on their slots", () => {
  const parsed = parseGraphTypedInput({
    nodeCount: 3,
    tag: Uint16Array.from([7, 0, 65535]),
    weight: Float32Array.from([0, 0.25, 1]),
  });

  assertEquals(Array.from(parsed.nodeTags ?? []), [7, 0, 65535]);
  assertEquals(Array.from(parsed.nodeWeights ?? []), [0, 0.25, 1]);
});

Deno.test("typed parser: weight is clamped without mutating the caller's array", () => {
  const weight = Float32Array.from([-3, 2, Number.NaN]);
  const parsed = parseGraphTypedInput({ nodeCount: 3, weight });

  assertEquals(Array.from(parsed.nodeWeights ?? []), [0, 1, 0]);
  assert(Number.isNaN(weight[2]), "the input array must not be rewritten");
  assertEquals(weight[0], -3);
});

Deno.test("typed parser: a wrong-length column is rejected", () => {
  assertThrows(
    () => parseGraphTypedInput({ nodeCount: 3, tag: Uint16Array.from([1, 2]) }),
    GraphMotherError,
    "tag length (2) must equal nodeCount (3)",
  );
  assertThrows(
    () => parseGraphTypedInput({ nodeCount: 2, weight: Float32Array.from([1, 1, 1]) }),
    GraphMotherError,
    "weight length (3) must equal nodeCount (2)",
  );
});

Deno.test("typed parser: no columns supplied means no columns allocated", () => {
  const parsed = parseGraphTypedInput({ nodeCount: 1000 });
  assertEquals(parsed.nodeTags, undefined);
  assertEquals(parsed.nodeWeights, undefined);
});

Deno.test("validateGraphTypedInput: reports column length mismatches", () => {
  const result = validateGraphTypedInput({
    nodeCount: 4,
    tag: Uint16Array.from([1]),
    weight: Float32Array.from([1, 1]),
  });
  assertEquals(result.valid, false);
  assertEquals(result.errors, [
    "tag length (1) must equal nodeCount (4)",
    "weight length (2) must equal nodeCount (4)",
  ]);

  const ok = validateGraphTypedInput({
    nodeCount: 2,
    tag: Uint16Array.from([1, 2]),
    weight: Float32Array.from([0, 1]),
  });
  assertEquals(ok.errors, []);
});

// =============================================================================
// Object path
// =============================================================================

Deno.test("parser: per-node tag and weight are indexed by assigned slot", () => {
  const parsed = parseGraphInput({
    nodes: [
      { id: "a", tag: 3, weight: 0.5 },
      { id: "b" },
      { id: "c", tag: 9, weight: 2 },
    ],
    edges: [],
  });

  const slotOf = (id: string): number => parsed.nodeIdMap.get(id)!;
  assertEquals(parsed.nodeTags?.[slotOf("a")], 3);
  assertEquals(parsed.nodeTags?.[slotOf("b")], 0);
  assertEquals(parsed.nodeTags?.[slotOf("c")], 9);
  assertEquals(parsed.nodeWeights?.[slotOf("a")], 0.5);
  assertEquals(parsed.nodeWeights?.[slotOf("c")], 1, "out of range clamps to 1");
});

Deno.test("parser: a column appears only when some node supplies it", () => {
  const neither = parseGraphInput({ nodes: [{ id: "a" }, { id: "b" }], edges: [] });
  assertEquals(neither.nodeTags, undefined);
  assertEquals(neither.nodeWeights, undefined);

  const tagOnly = parseGraphInput({ nodes: [{ id: "a" }, { id: "b", tag: 1 }], edges: [] });
  assertEquals(tagOnly.nodeTags?.length, 2);
  assertEquals(tagOnly.nodeWeights, undefined);
});

Deno.test("parser: a non-finite tag reads as the neutral value", () => {
  const parsed = parseGraphInput({
    nodes: [{ id: "a", tag: Number.POSITIVE_INFINITY, weight: Number.NaN }],
    edges: [],
  });
  assertEquals(parsed.nodeTags?.[0], 0);
  assertEquals(parsed.nodeWeights?.[0], 0);
});
