/**
 * GraphTypedInput parser round-trip tests.
 */

import { assertAlmostEquals, assertEquals, assertThrows } from "jsr:@std/assert@^1";
import {
  createTypedInput,
  mergeTypedInputs,
  parseGraphTypedInput,
  validateGraphTypedInput,
} from "../../packages/core/src/graph/typed_parser.ts";
import { NODE_ATTR_FLOATS } from "../../packages/core/src/api/graph_state.ts";
import { HeroineGraphError } from "../../packages/core/src/errors.ts";
import { generateCodeTree } from "../fixtures/code_tree.ts";

Deno.test("parseGraphTypedInput: deinterleaves positions and edge pairs", () => {
  const parsed = parseGraphTypedInput({
    nodeCount: 3,
    edgeCount: 2,
    positions: new Float32Array([1, 2, 3, 4, 5, 6]),
    edgePairs: new Uint32Array([0, 1, 1, 2]),
  });

  assertEquals(parsed.nodeCount, 3);
  assertEquals(parsed.edgeCount, 2);
  assertEquals([...parsed.positionsX], [1, 3, 5]);
  assertEquals([...parsed.positionsY], [2, 4, 6]);
  assertEquals([...parsed.edgeSources], [0, 1]);
  assertEquals([...parsed.edgeTargets], [1, 2]);

  // Generated sequential ids
  assertEquals(parsed.nodeIdMap.get("n0"), 0);
  assertEquals(parsed.nodeIdMap.get("n2"), 2);
  assertEquals(parsed.edgeIdMap.get("e1"), 1);
});

Deno.test("parseGraphTypedInput: applies defaults and provided attribute arrays", () => {
  const withDefaults = parseGraphTypedInput({ nodeCount: 2, edgeCount: 1 });
  assertEquals(withDefaults.nodeAttributes[0], 5); // default radius
  assertAlmostEquals(withDefaults.nodeAttributes[1], 0.4); // default color r
  assertEquals(withDefaults.edgeAttributes[0], 1); // default edge width
  assertAlmostEquals(withDefaults.edgeAttributes[1], 0.5); // default edge color r

  const explicit = parseGraphTypedInput({
    nodeCount: 2,
    edgeCount: 1,
    nodeRadii: new Float32Array([7, 9]),
    nodeColors: new Float32Array([1, 0, 0, 0, 0, 1]),
    edgeWidths: new Float32Array([3]),
    edgeColors: new Float32Array([0, 1, 0]),
  });
  assertEquals(explicit.nodeAttributes[0], 7);
  assertEquals(explicit.nodeAttributes[1 * NODE_ATTR_FLOATS], 9);
  assertEquals(explicit.nodeAttributes[1], 1); // node 0 red channel
  assertEquals(explicit.nodeAttributes[1 * NODE_ATTR_FLOATS + 3], 1); // node 1 blue channel
  assertEquals(explicit.edgeAttributes[0], 3); // width
  assertEquals(explicit.edgeAttributes[2], 1); // green channel
});

Deno.test("parseGraphTypedInput: honors provided node ids", () => {
  const parsed = parseGraphTypedInput({
    nodeCount: 2,
    nodeIds: ["alpha", "beta"],
  });
  assertEquals(parsed.nodeIdMap.get("alpha"), 0);
  assertEquals(parsed.nodeIdMap.get("beta"), 1);
});

Deno.test("parseGraphTypedInput: zero nodeCount throws", () => {
  assertThrows(() => parseGraphTypedInput({ nodeCount: 0 }), HeroineGraphError);
});

Deno.test("parseGraphTypedInput: agrees with object-parser on a seeded fixture", () => {
  const tree = generateCodeTree({ seed: 9, maxDepth: 3, maxNodes: 40 });

  // Build the equivalent typed input from the fixture's raw arrays
  const positions = new Float32Array(tree.nodeCount * 2);
  for (let i = 0; i < tree.nodeCount; i++) {
    positions[i * 2] = tree.positionsX[i];
    positions[i * 2 + 1] = tree.positionsY[i];
  }
  const edgePairs = new Uint32Array(tree.edgeCount * 2);
  for (let e = 0; e < tree.edgeCount; e++) {
    edgePairs[e * 2] = tree.edgeSources[e];
    edgePairs[e * 2 + 1] = tree.edgeTargets[e];
  }

  const parsed = parseGraphTypedInput({
    nodeCount: tree.nodeCount,
    edgeCount: tree.edgeCount,
    positions,
    edgePairs,
  });

  assertEquals([...parsed.positionsX], [...tree.positionsX]);
  assertEquals([...parsed.positionsY], [...tree.positionsY]);
  assertEquals([...parsed.edgeSources], [...tree.edgeSources]);
  assertEquals([...parsed.edgeTargets], [...tree.edgeTargets]);
});

Deno.test("validateGraphTypedInput: catches array-length mismatches", () => {
  assertEquals(validateGraphTypedInput(null).valid, false);
  assertEquals(validateGraphTypedInput({ nodeCount: -1 }).valid, false);

  const bad = validateGraphTypedInput({
    nodeCount: 3,
    edgeCount: 2,
    positions: new Float32Array(4), // needs 6
    nodeRadii: new Float32Array(2), // needs 3
    nodeColors: new Float32Array(6), // needs 9
    edgePairs: new Uint32Array(2), // needs 4
  });
  assertEquals(bad.valid, false);
  assertEquals(bad.errors.length, 4);

  const good = validateGraphTypedInput({
    nodeCount: 2,
    edgeCount: 1,
    positions: new Float32Array(4),
    edgePairs: new Uint32Array(2),
  });
  assertEquals(good.valid, true);
});

Deno.test("mergeTypedInputs: offsets edge indices by preceding node counts", () => {
  const a = createTypedInput(2, 1, [0, 0, 10, 0], [0, 1]);
  const b = createTypedInput(3, 2, [20, 0, 30, 0, 40, 0], [0, 1, 1, 2]);

  const merged = mergeTypedInputs([a, b]);
  assertEquals(merged.nodeCount, 5);
  assertEquals(merged.edgeCount, 3);
  // Second input's node indices shifted by 2
  assertEquals([...(merged.edgePairs ?? [])], [0, 1, 2, 3, 3, 4]);
  assertEquals(merged.positions?.[6], 30); // b's second node x lands at slot 3

  // Identity cases
  assertEquals(mergeTypedInputs([]).nodeCount, 0);
  assertEquals(mergeTypedInputs([a]), a);
});
