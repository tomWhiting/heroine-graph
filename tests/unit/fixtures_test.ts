/**
 * Fixture infrastructure self-tests: the seeded PRNG and code-tree
 * generator must be deterministic and structurally sound, since every
 * other test (including future physics tuning) builds on them.
 */

import { assert, assertEquals, assertNotEquals } from "jsr:@std/assert@^1";
import { mulberry32, randInt } from "../fixtures/prng.ts";
import { generateCodeTree } from "../fixtures/code_tree.ts";
import {
  countNonFinite,
  kineticEnergy,
  maxRadius,
  meanEdgeLength,
  meanParentChildDistance,
  radialOrderingScore,
} from "../helpers/invariants.ts";

Deno.test("mulberry32: deterministic per seed, uniform in [0, 1)", () => {
  const a = mulberry32(1234);
  const b = mulberry32(1234);
  const c = mulberry32(5678);

  const seqA = Array.from({ length: 20 }, () => a());
  const seqB = Array.from({ length: 20 }, () => b());
  const seqC = Array.from({ length: 20 }, () => c());

  assertEquals(seqA, seqB);
  assertNotEquals(seqA, seqC);
  for (const v of seqA) {
    assert(v >= 0 && v < 1, `out of range: ${v}`);
  }

  const rng = mulberry32(1);
  for (let i = 0; i < 100; i++) {
    const n = randInt(rng, 3, 7);
    assert(n >= 3 && n <= 7 && Number.isInteger(n), `randInt out of range: ${n}`);
  }
});

Deno.test("generateCodeTree: identical options produce identical graphs", () => {
  const options = { seed: 77, maxDepth: 4, maxNodes: 200, crossEdgeRatio: 0.15 };
  const a = generateCodeTree(options);
  const b = generateCodeTree(options);

  assertEquals(a.nodeCount, b.nodeCount);
  assertEquals([...a.parent], [...b.parent]);
  assertEquals([...a.positionsX], [...b.positionsX]);
  assertEquals([...a.edgeSources], [...b.edgeSources]);
  assertEquals([...a.edgeTargets], [...b.edgeTargets]);
  assertEquals(a.input, b.input);

  const different = generateCodeTree({ ...options, seed: 78 });
  assertNotEquals([...a.positionsX], [...different.positionsX]);
});

Deno.test("generateCodeTree: containment tree is structurally valid", () => {
  const tree = generateCodeTree({ seed: 5, maxDepth: 4, maxNodes: 300, crossEdgeRatio: 0.2 });

  assert(tree.nodeCount > 1 && tree.nodeCount <= 300);
  assertEquals(tree.containmentEdgeCount, tree.nodeCount - 1);
  assertEquals(tree.edgeCount, tree.edgeSources.length);

  // Root has no parent; every other node's parent precedes it (BFS order)
  // and sits exactly one level up.
  assertEquals(tree.parent[0], -1);
  assertEquals(tree.depths[0], 0);
  for (let i = 1; i < tree.nodeCount; i++) {
    const p = tree.parent[i];
    assert(p >= 0 && p < i, `node ${i} has invalid parent ${p}`);
    assertEquals(tree.depths[i], tree.depths[p] + 1);
  }

  // Containment edges mirror the parent array; cross edges never self-loop
  for (let e = 0; e < tree.containmentEdgeCount; e++) {
    assertEquals(tree.edgeSources[e], tree.parent[tree.edgeTargets[e]]);
  }
  for (let e = tree.containmentEdgeCount; e < tree.edgeCount; e++) {
    assertNotEquals(tree.edgeSources[e], tree.edgeTargets[e]);
    assert(tree.edgeSources[e] < tree.nodeCount);
    assert(tree.edgeTargets[e] < tree.nodeCount);
  }

  // GraphInput view is consistent with the typed arrays
  assertEquals(tree.input.nodes.length, tree.nodeCount);
  assertEquals(tree.input.edges.length, tree.edgeCount);
  assertEquals(tree.input.nodes[0].id, "n0");
  assertEquals(tree.input.edges[0]["type"], "contains");
  assertEquals(tree.input.edges[tree.edgeCount - 1]["type"], "imports");
});

Deno.test("generateCodeTree: node types follow depth (dir -> file -> symbol)", () => {
  const tree = generateCodeTree({ seed: 3, maxDepth: 3, maxNodes: 100 });
  for (let i = 0; i < tree.nodeCount; i++) {
    const depth = tree.depths[i];
    const type = tree.input.nodes[i]["type"];
    if (depth === 3) assertEquals(type, "symbol");
    else if (depth === 2) assertEquals(type, "file");
    else assertEquals(type, "dir");
  }
});

Deno.test("invariant helpers: metrics behave on known configurations", () => {
  // Two nodes at distance 3-4-5, one parent-child pair
  const xs = new Float32Array([0, 3]);
  const ys = new Float32Array([0, 4]);
  const parent = Int32Array.from([-1, 0]);

  assertEquals(countNonFinite(xs, ys), 0);
  assertEquals(countNonFinite(new Float32Array([NaN, Infinity]), ys), 2);
  assertEquals(maxRadius(xs, ys), 5);
  assertEquals(meanParentChildDistance(xs, ys, parent), 5);
  assertEquals(
    meanEdgeLength(xs, ys, new Uint32Array([0]), new Uint32Array([1])),
    5,
  );
  // Child at radius 5, parent at radius 0: perfectly radially ordered
  assertEquals(radialOrderingScore(xs, ys, parent), 1);
  // Swap center so the child is inside the parent: score drops to 0
  assertEquals(radialOrderingScore(xs, ys, parent, 3, 4), 0);

  // Kinetic energy of a single node moving (3, 4) in one tick: 0.5 * 25
  const energy = kineticEnergy(
    new Float32Array([0]),
    new Float32Array([0]),
    new Float32Array([3]),
    new Float32Array([4]),
  );
  assertEquals(energy, 12.5);

  // Stationary nodes carry zero energy
  assertEquals(kineticEnergy(xs, ys, xs, ys), 0);
});
