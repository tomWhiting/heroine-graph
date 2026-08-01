/**
 * Containment hierarchy: derivation, retention, supplied-vs-computed parity.
 *
 * The hierarchy is the keystone of semantic LOD — layout, the LOD cut, the
 * hit-test broad phase and DOM culling all read the same four columns — so
 * these tests pin both the shape of the columns and the ground truth they
 * describe, against the `code_tree` fixture that already emits `parent` and
 * `depths`.
 */

import { assert, assertEquals, assertThrows } from "jsr:@std/assert@^1";
import {
  bubbleUploadChanged,
  buildChildrenCsr,
  CONTAINMENT_EDGE_TYPE,
  decodeHierarchyColumns,
  deriveHierarchy,
  HIERARCHY_ROOT,
  type HierarchyColumns,
  type HierarchyDeriver,
  MAX_HIERARCHY_DEPTH,
  retainSuppliedHierarchy,
  selectContainmentEdges,
  validateHierarchyColumns,
} from "../../packages/core/src/graph/hierarchy.ts";
import { GraphMotherError } from "../../packages/core/src/errors.ts";
import { parseGraphInput } from "../../packages/core/src/graph/parser.ts";
import { createWasmEngine } from "../../packages/core/src/wasm/loader.ts";
import { CODE_TREE_SCALES, type CodeTreeGraph, generateCodeTree } from "../fixtures/code_tree.ts";

const BUBBLE = { baseRadius: 10, padding: 5 } as const;

/** A WASM engine is process-wide; one instance serves every test. */
let sharedDeriver: HierarchyDeriver | null = null;
async function deriver(): Promise<HierarchyDeriver> {
  if (!sharedDeriver) sharedDeriver = await createWasmEngine();
  return sharedDeriver;
}

/**
 * Feed slots and containment pairs into the engine and derive.
 *
 * The engine's node bound is what sets the WASM column stride, so the slots
 * must be populated before deriving even though only the edges shape the tree.
 */
async function derive(
  nodeCount: number,
  containmentPairs: Uint32Array,
): Promise<ReturnType<typeof deriveHierarchy>> {
  const engine = await deriver() as HierarchyDeriver & {
    clear(): void;
    addNodesFromPositions(p: Float32Array): void;
  };
  engine.clear();
  engine.addNodesFromPositions(new Float32Array(nodeCount * 2));
  return deriveHierarchy(engine, containmentPairs, nodeCount, {
    baseRadius: BUBBLE.baseRadius,
    padding: BUBBLE.padding,
    rootId: HIERARCHY_ROOT,
  });
}

/** The fixture's containment edges as interleaved `[parent, child, ...]` pairs. */
function containmentPairs(tree: CodeTreeGraph): Uint32Array {
  const pairs = new Uint32Array(tree.containmentEdgeCount * 2);
  for (let e = 0; e < tree.containmentEdgeCount; e++) {
    pairs[e * 2] = tree.edgeSources[e];
    pairs[e * 2 + 1] = tree.edgeTargets[e];
  }
  return pairs;
}

// ===========================================================================
// Containment edge selection
// ===========================================================================

Deno.test("selectContainmentEdges: keeps only contains edges when the graph is typed", () => {
  const sources = new Uint32Array([0, 0, 3, 1]);
  const targets = new Uint32Array([1, 2, 1, 3]);
  const types = [CONTAINMENT_EDGE_TYPE, CONTAINMENT_EDGE_TYPE, "imports", CONTAINMENT_EDGE_TYPE];

  const pairs = selectContainmentEdges(sources, targets, 4, types);

  assertEquals(Array.from(pairs), [0, 1, 0, 2, 1, 3]);
});

Deno.test("selectContainmentEdges: keeps every edge when nothing is typed as containment", () => {
  const sources = new Uint32Array([0, 0]);
  const targets = new Uint32Array([1, 2]);

  const pairs = selectContainmentEdges(sources, targets, 2, ["imports", undefined]);

  assertEquals(Array.from(pairs), [0, 1, 0, 2]);
});

// ===========================================================================
// Column decoding and the children CSR
// ===========================================================================

Deno.test("decodeHierarchyColumns: maps the -1 root sentinel and saturates depth", () => {
  // Two slots: 0 is a root at depth 0, 1 is its child at an absurd depth.
  const data = new Float32Array([
    12.5,
    10, // wellRadius
    0,
    MAX_HIERARCHY_DEPTH + 5, // depth
    -1,
    0, // parent
    2,
    1, // subtreeSize
  ]);

  const columns = decodeHierarchyColumns(data, 2);

  assertEquals(Array.from(columns.wellRadius), [12.5, 10]);
  assertEquals(Array.from(columns.parent), [HIERARCHY_ROOT, 0]);
  assertEquals(Array.from(columns.depth), [0, MAX_HIERARCHY_DEPTH]);
  assertEquals(Array.from(columns.subtreeSize), [2, 1]);
});

Deno.test("decodeHierarchyColumns: reads the stride from the block, not from nodeCount", () => {
  // Engine node bound 3, caller only retains the first 2 slots. Reading the
  // columns at a stride of 2 would misplace depth, parent and subtreeSize.
  // Slot 1's parent is slot 2, which is outside the retained range and must
  // therefore decode as a root rather than as a dangling reference.
  const data = new Float32Array([
    12.5,
    10,
    10, // wellRadius (stride 3)
    0,
    1,
    1, // depth
    -1,
    2,
    0, // parent
    3,
    1,
    1, // subtreeSize
  ]);

  const columns = decodeHierarchyColumns(data, 2);

  assertEquals(Array.from(columns.depth), [0, 1]);
  assertEquals(Array.from(columns.parent), [HIERARCHY_ROOT, HIERARCHY_ROOT]);
  assertEquals(Array.from(columns.subtreeSize), [3, 1]);
});

Deno.test("decodeHierarchyColumns: rejects a block too short for the slot count", () => {
  assertThrows(
    () => decodeHierarchyColumns(new Float32Array(6), 2),
    GraphMotherError,
  );
});

Deno.test("buildChildrenCsr: indexes children in ascending slot order", () => {
  //   0 → {1, 3}, 1 → {2}, 4 is a second root
  const parent = new Uint32Array([HIERARCHY_ROOT, 0, 1, 0, HIERARCHY_ROOT]);

  const csr = buildChildrenCsr(parent, 5);

  assertEquals(Array.from(csr.offsets), [0, 2, 3, 3, 3, 3]);
  assertEquals(Array.from(csr.children), [1, 3, 2]);
  assertEquals(Array.from(csr.children.subarray(csr.offsets[0], csr.offsets[1])), [1, 3]);
  assertEquals(Array.from(csr.children.subarray(csr.offsets[1], csr.offsets[2])), [2]);
});

// ===========================================================================
// Supplied-column validation
// ===========================================================================

/** A valid two-level forest: 0 → {1, 2}, 3 alone. */
function validColumns(): HierarchyColumns {
  return {
    parent: new Uint32Array([HIERARCHY_ROOT, 0, 0, HIERARCHY_ROOT]),
    wellRadius: new Float32Array([20, 10, 10, 10]),
    depth: new Uint16Array([0, 1, 1, 0]),
    subtreeSize: new Uint32Array([3, 1, 1, 1]),
  };
}

Deno.test("validateHierarchyColumns: accepts a well-formed forest", () => {
  validateHierarchyColumns(validColumns(), 4);
});

Deno.test("validateHierarchyColumns: rejects a column of the wrong length", () => {
  const columns = { ...validColumns(), depth: new Uint16Array([0, 1, 1]) };
  assertThrows(() => validateHierarchyColumns(columns, 4), GraphMotherError, "depth");
});

Deno.test("validateHierarchyColumns: rejects an out-of-range parent", () => {
  const columns = validColumns();
  columns.parent[1] = 9;
  assertThrows(() => validateHierarchyColumns(columns, 4), GraphMotherError, "out of range");
});

Deno.test("validateHierarchyColumns: rejects a self-parent", () => {
  const columns = validColumns();
  columns.parent[1] = 1;
  assertThrows(() => validateHierarchyColumns(columns, 4), GraphMotherError, "its own parent");
});

Deno.test("validateHierarchyColumns: rejects a containment cycle via inconsistent depths", () => {
  // 1 → 2 → 1 is a cycle; no depth assignment can satisfy depth[c] = depth[p]+1.
  const columns: HierarchyColumns = {
    parent: new Uint32Array([HIERARCHY_ROOT, 2, 1, HIERARCHY_ROOT]),
    wellRadius: new Float32Array([10, 10, 10, 10]),
    depth: new Uint16Array([0, 1, 2, 0]),
    subtreeSize: new Uint32Array([1, 2, 1, 1]),
  };
  assertThrows(() => validateHierarchyColumns(columns, 4), GraphMotherError, "cycle");
});

Deno.test("validateHierarchyColumns: rejects subtree sizes that do not cover the slot space", () => {
  const columns = validColumns();
  columns.subtreeSize[0] = 2;
  assertThrows(() => validateHierarchyColumns(columns, 4), GraphMotherError, "subtree sizes");
});

Deno.test("retainSuppliedHierarchy: rejects columns that are not a well-formed forest", () => {
  // Producer-supplied columns are load-bearing for layout, LOD and hit
  // testing; a wrong one fails silently and far from its cause, so retention
  // must not be a bare pass-through.
  const columns = validColumns();
  columns.parent[2] = 9;
  assertThrows(() => retainSuppliedHierarchy(columns, 4), GraphMotherError, "out of range");
});

Deno.test("validateHierarchyColumns: rejects a centroid of the wrong length", () => {
  const columns = { ...validColumns(), centroid: new Float32Array(4) };
  assertThrows(() => validateHierarchyColumns(columns, 4), GraphMotherError, "centroid");
});

// ===========================================================================
// WASM derivation against the fixture's ground truth
// ===========================================================================

for (const scale of ["small", "medium"] as const) {
  Deno.test(`deriveHierarchy: reproduces the ${scale} code-tree fixture's parents and depths`, async () => {
    const tree = generateCodeTree(CODE_TREE_SCALES[scale]);
    const hierarchy = await derive(tree.nodeCount, containmentPairs(tree));
    const { parent, depth, subtreeSize, wellRadius } = hierarchy.columns;

    assertEquals(hierarchy.nodeCount, tree.nodeCount);
    assertEquals(hierarchy.source, "derived");

    for (let i = 0; i < tree.nodeCount; i++) {
      const expectedParent = tree.parent[i] < 0 ? HIERARCHY_ROOT : tree.parent[i];
      assertEquals(parent[i], expectedParent, `parent[${i}]`);
      assertEquals(depth[i], tree.depths[i], `depth[${i}]`);
    }

    // The fixture is a single tree rooted at slot 0.
    assertEquals(subtreeSize[0], tree.nodeCount);

    // subtreeSize[p] == 1 + sum of its children's subtree sizes.
    const summed = new Uint32Array(tree.nodeCount).fill(1);
    for (let i = 0; i < tree.nodeCount; i++) {
      if (parent[i] !== HIERARCHY_ROOT) summed[parent[i]] += subtreeSize[i];
    }
    assertEquals(summed, subtreeSize);

    // A parent's bubble encloses each of its children's.
    for (let i = 0; i < tree.nodeCount; i++) {
      if (parent[i] === HIERARCHY_ROOT) continue;
      assert(
        wellRadius[parent[i]] > wellRadius[i],
        `wellRadius[${parent[i]}] must exceed its child ${i}`,
      );
    }
  });
}

Deno.test("deriveHierarchy: selection through the object parser feeds only containment edges", async () => {
  // The fixture's cross-cutting "imports" edges must not reparent anything.
  const tree = generateCodeTree(CODE_TREE_SCALES.small);
  const parsed = parseGraphInput(tree.input);

  const selected = selectContainmentEdges(
    parsed.edgeSources,
    parsed.edgeTargets,
    parsed.edgeCount,
    parsed.edgeTypes ?? [],
  );
  assertEquals(selected.length, tree.containmentEdgeCount * 2);

  const hierarchy = await derive(tree.nodeCount, selected);
  for (let i = 0; i < tree.nodeCount; i++) {
    const expectedParent = tree.parent[i] < 0 ? HIERARCHY_ROOT : tree.parent[i];
    assertEquals(hierarchy.columns.parent[i], expectedParent, `parent[${i}]`);
  }
});

Deno.test("deriveHierarchy: every component of a forest gets a real tree", async () => {
  // A repo root plus an orphan config component, which is the shape a real
  // repository graph has. Before forest handling every slot outside the
  // largest component was flattened to depth 0 and a bare base radius, which
  // made it invisible to any tree-walk consumer.
  const tree = generateCodeTree(CODE_TREE_SCALES.small);
  const n = tree.nodeCount;
  const orphanRoot = n;
  const pairs = Uint32Array.from([
    ...containmentPairs(tree),
    orphanRoot,
    n + 1,
    orphanRoot,
    n + 2,
  ]);

  const hierarchy = await derive(n + 3, pairs);
  const { parent, depth, subtreeSize, wellRadius } = hierarchy.columns;

  assertEquals(parent[orphanRoot], HIERARCHY_ROOT);
  assertEquals(depth[orphanRoot], 0);
  assertEquals(parent[n + 1], orphanRoot);
  assertEquals(parent[n + 2], orphanRoot);
  assertEquals(depth[n + 1], 1);
  assertEquals(depth[n + 2], 1);
  assertEquals(subtreeSize[orphanRoot], 3);
  assert(
    wellRadius[orphanRoot] > BUBBLE.baseRadius,
    `orphan component root must get a real bubble radius, got ${wellRadius[orphanRoot]}`,
  );

  // The main tree is untouched, and the two roots partition the slot space.
  assertEquals(subtreeSize[0], n);
  let rootTotal = 0;
  for (let i = 0; i < n + 3; i++) {
    if (parent[i] === HIERARCHY_ROOT) rootTotal += subtreeSize[i];
  }
  assertEquals(rootTotal, n + 3);
});

Deno.test("deriveHierarchy: isolated slots are singleton roots, not depth-0 debris", async () => {
  const hierarchy = await derive(5, Uint32Array.from([0, 1]));
  const { parent, depth, subtreeSize } = hierarchy.columns;

  for (const slot of [2, 3, 4]) {
    assertEquals(parent[slot], HIERARCHY_ROOT);
    assertEquals(depth[slot], 0);
    assertEquals(subtreeSize[slot], 1);
    assertEquals(hierarchy.children.offsets[slot], hierarchy.children.offsets[slot + 1]);
  }
});

// ===========================================================================
// Supplied-or-computed parity
// ===========================================================================

Deno.test("retainSuppliedHierarchy: supplied columns produce the same retained hierarchy as derivation", async () => {
  const tree = generateCodeTree(CODE_TREE_SCALES.small);
  const derived = await derive(tree.nodeCount, containmentPairs(tree));

  // A producer that precomputed the same tree ships exactly these columns.
  const supplied = retainSuppliedHierarchy(derived.columns, tree.nodeCount);

  assertEquals(supplied.source, "supplied");
  assertEquals(supplied.nodeCount, derived.nodeCount);
  assertEquals(supplied.columns.parent, derived.columns.parent);
  assertEquals(supplied.columns.depth, derived.columns.depth);
  assertEquals(supplied.columns.wellRadius, derived.columns.wellRadius);
  assertEquals(supplied.columns.subtreeSize, derived.columns.subtreeSize);
  assertEquals(supplied.children.offsets, derived.children.offsets);
  assertEquals(supplied.children.children, derived.children.children);
});

Deno.test("retainSuppliedHierarchy: the children CSR reproduces the fixture's child sets", async () => {
  const tree = generateCodeTree(CODE_TREE_SCALES.small);
  const { children } = await derive(tree.nodeCount, containmentPairs(tree));

  const expected: number[][] = Array.from({ length: tree.nodeCount }, () => []);
  for (let i = 0; i < tree.nodeCount; i++) {
    if (tree.parent[i] >= 0) expected[tree.parent[i]].push(i);
  }

  for (let slot = 0; slot < tree.nodeCount; slot++) {
    const actual = Array.from(
      children.children.subarray(children.offsets[slot], children.offsets[slot + 1]),
    );
    assertEquals(actual, expected[slot], `children of slot ${slot}`);
  }
});

// ===========================================================================
// F15 — the bubble knobs are not per-tick uniforms
// ===========================================================================
//
// `setForceConfig({ relativityBubbleMode: true })` after load was a silent
// no-op: it never re-ran the upload that fills the well-radius buffer.
// `GraphMother` needs a canvas and a GPU device and cannot be constructed in a
// unit test, so the decision the fix turns on is covered here directly; the
// wiring in `setForceConfig` is a single call guarded by this predicate.

Deno.test("bubbleUploadChanged: detects each knob that only reaches the GPU at upload time", () => {
  const base = {
    relativityBubbleMode: false,
    relativityBubbleBaseRadius: 10,
    relativityBubblePadding: 5,
  };

  assertEquals(bubbleUploadChanged(base, { ...base }), false);
  assertEquals(bubbleUploadChanged(base, { ...base, relativityBubbleMode: true }), true);
  assertEquals(bubbleUploadChanged(base, { ...base, relativityBubbleBaseRadius: 11 }), true);
  assertEquals(bubbleUploadChanged(base, { ...base, relativityBubblePadding: 6 }), true);
  // Turning it back off must re-upload too, or stale radii survive the change.
  assertEquals(
    bubbleUploadChanged({ ...base, relativityBubbleMode: true }, base),
    true,
  );
});
