/**
 * GraphInput parser round-trip tests.
 */

import { assertAlmostEquals, assertEquals, assertThrows } from "jsr:@std/assert@^1";
import {
  createEdgeIndicesBuffer,
  parseGraphInput,
  validateGraphInput,
} from "../../packages/core/src/graph/parser.ts";
import { NODE_ATTR_FLOATS } from "../../packages/core/src/api/graph_state.ts";
import { GraphMotherError } from "../../packages/core/src/errors.ts";
import { microTreeGraph, triangleGraph } from "../fixtures/tiny_graphs.ts";
import { generateCodeTree } from "../fixtures/code_tree.ts";

Deno.test("parseGraphInput: round-trips ids, positions, and explicit attributes", () => {
  const parsed = parseGraphInput(triangleGraph());

  assertEquals(parsed.nodeCount, 3);
  assertEquals(parsed.edgeCount, 3);

  // IDs map to insertion-order indices, both directions
  assertEquals(parsed.nodeIdMap.get("a"), 0);
  assertEquals(parsed.nodeIdMap.get("b"), 1);
  assertEquals(parsed.nodeIdMap.get("c"), 2);
  assertEquals(parsed.nodeIdMap.getId(2), "c");

  // Positions round-trip
  assertEquals([...parsed.positionsX], [0, 100, 50]);
  assertEquals([...parsed.positionsY], [0, 0, 80]);

  // Node attributes: [radius, r, g, b, selected, hovered, ...]
  const b = 1 * NODE_ATTR_FLOATS;
  assertEquals(parsed.nodeAttributes[b], 20); // radius
  assertAlmostEquals(parsed.nodeAttributes[b + 1], 0); // #00ff00 red
  assertAlmostEquals(parsed.nodeAttributes[b + 2], 1); // green
  assertAlmostEquals(parsed.nodeAttributes[b + 3], 0); // blue
  assertEquals(parsed.nodeAttributes[b + 4], 0); // selected
  assertEquals(parsed.nodeAttributes[b + 5], 0); // hovered

  // Edge endpoints resolved to node indices
  assertEquals([...parsed.edgeSources], [0, 1, 2]);
  assertEquals([...parsed.edgeTargets], [1, 2, 0]);

  // Edge attributes: [width, r, g, b, selected, hovered, curvature, opacity]
  assertEquals(parsed.edgeAttributes[0], 2); // width of "ab"
  assertAlmostEquals(parsed.edgeAttributes[1], 1); // #ffffff
  assertEquals(parsed.edgeAttributes[6], 0); // straight
  assertEquals(parsed.edgeAttributes[7], 1); // fully visible

  // Explicit edge ids kept; missing edge id generated from index
  assertEquals(parsed.edgeIdMap.get("ab"), 0);
  assertEquals(parsed.edgeIdMap.get("edge_2"), 2);
});

Deno.test("parseGraphInput: defaults applied when attributes omitted", () => {
  const parsed = parseGraphInput({
    nodes: [{ id: "solo" }],
    edges: [],
  });

  assertEquals(parsed.positionsX[0], 0);
  assertEquals(parsed.positionsY[0], 0);
  assertEquals(parsed.nodeAttributes[0], 5); // default radius
  assertAlmostEquals(parsed.nodeAttributes[1], 0.4); // default color r
  assertAlmostEquals(parsed.nodeAttributes[2], 0.6); // g
  assertAlmostEquals(parsed.nodeAttributes[3], 0.9); // b
  assertEquals(parsed.nodeTypes, undefined);
  assertEquals(parsed.edgeTypes, undefined);
});

Deno.test("parseGraphInput: extracts node types (top-level and metadata) and edge types", () => {
  const parsed = parseGraphInput(microTreeGraph());

  assertEquals(parsed.nodeTypes?.[0], "dir");
  assertEquals(parsed.nodeTypes?.[2], "file"); // from metadata.type
  assertEquals(parsed.edgeTypes?.[0], "contains");
  assertEquals(parsed.edgeTypes?.[4], "imports");

  // Metadata stored per node index
  assertEquals(parsed.nodeMetadata.get(2)?.["lang"], "md");
});

Deno.test("parseGraphInput: duplicate node id throws", () => {
  assertThrows(
    () =>
      parseGraphInput({
        nodes: [{ id: "dup" }, { id: "dup" }],
        edges: [],
      }),
    GraphMotherError,
    "Duplicate node ID",
  );
});

Deno.test("parseGraphInput: dangling edge reference throws unless validation disabled", () => {
  const input = {
    nodes: [{ id: "a" }],
    edges: [{ source: "a", target: "ghost" }],
  };

  assertThrows(() => parseGraphInput(input), GraphMotherError, "not found");

  const lax = parseGraphInput(input, { validateReferences: false });
  assertEquals(lax.edgeTargets[0], 0); // dangling target mapped to 0
});

Deno.test("parseGraphInput: seeded fixture round-trips slot-for-slot", () => {
  const tree = generateCodeTree({ seed: 42, maxDepth: 3, maxNodes: 60, crossEdgeRatio: 0.2 });
  const parsed = parseGraphInput(tree.input);

  assertEquals(parsed.nodeCount, tree.nodeCount);
  assertEquals(parsed.edgeCount, tree.edgeCount);

  // Fixture ids are "n{slot}", so parser indices must equal fixture slots
  for (let i = 0; i < tree.nodeCount; i++) {
    assertEquals(parsed.nodeIdMap.get(`n${i}`), i);
    assertAlmostEquals(parsed.positionsX[i], tree.positionsX[i], 1e-5);
    assertAlmostEquals(parsed.positionsY[i], tree.positionsY[i], 1e-5);
  }
  assertEquals([...parsed.edgeSources], [...tree.edgeSources]);
  assertEquals([...parsed.edgeTargets], [...tree.edgeTargets]);
});

Deno.test("validateGraphInput: reports structural problems", () => {
  assertEquals(validateGraphInput(null).valid, false);
  assertEquals(validateGraphInput("nope").valid, false);

  const missingArrays = validateGraphInput({});
  assertEquals(missingArrays.valid, false);
  assertEquals(missingArrays.errors.length, 2);

  const badMembers = validateGraphInput({
    nodes: [{}, { id: "ok" }],
    edges: [{ source: "a" }],
  });
  assertEquals(badMembers.valid, false);
  assertEquals(badMembers.errors, [
    "Node at index 0 missing 'id'",
    "Edge at index 0 missing 'target'",
  ]);

  assertEquals(validateGraphInput(triangleGraph()).valid, true);
});

Deno.test("createEdgeIndicesBuffer: interleaves source/target pairs", () => {
  const buffer = createEdgeIndicesBuffer(
    new Uint32Array([0, 1, 2]),
    new Uint32Array([1, 2, 0]),
  );
  assertEquals([...buffer], [0, 1, 1, 2, 2, 0]);
});
