/**
 * The code-graph example's producer.
 *
 * `examples/mission-control/src/repo.ts` stands in for a real indexer, which
 * means it has to honour the same contracts one would: a well-formed forest
 * that `validateHierarchyColumns` accepts, depth-first slot order so subtrees
 * are contiguous ranges, `wellRadius` computed with the formula the WASM
 * derivation uses, and columns whose lengths and ranges match what the typed
 * parser will assert at load.
 *
 * These are all silent-failure shapes: a producer that gets any of them wrong
 * either throws deep inside `load()` or, worse, lays out a tree nobody asked
 * for. They are cheap to check here and impossible to see in a screenshot.
 */

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import {
  HIERARCHY_ROOT,
  validateHierarchyColumns,
} from "../../packages/core/src/graph/hierarchy.ts";
import {
  BUBBLE_BASE_RADIUS,
  BUBBLE_PADDING,
  generateRepo,
  NODE_KIND,
  rankLabelCandidates,
  REPO_SCALES,
  type RepoGraph,
} from "../../examples/mission-control/src/repo.ts";

/** Sizes big enough to exercise nesting without making the suite slow. */
const SIZES = [1, 2, 25, 400, 2_500];

Deno.test("repo: node count is hit exactly, at every size", () => {
  for (const nodeCount of SIZES) {
    const repo = generateRepo({ nodeCount });
    assertEquals(repo.nodeCount, nodeCount, `requested ${nodeCount}`);
    assertEquals(repo.input.nodeCount, nodeCount);
    assertEquals(repo.kind.length, nodeCount);
    assertEquals(repo.path.length, nodeCount);
    assertEquals(repo.name.length, nodeCount);
    assertEquals(repo.weight.length, nodeCount);
    assertEquals(repo.positions.length, nodeCount * 2);
  }
});

Deno.test("repo: the supplied hierarchy is a well-formed forest", () => {
  for (const nodeCount of SIZES) {
    const repo = generateRepo({ nodeCount });
    // Throws on any malformed column; this is the exact check core runs.
    validateHierarchyColumns(repo.hierarchy, nodeCount);
    assertEquals(repo.hierarchy.parent[0], HIERARCHY_ROOT, "slot 0 is the repository root");
    assertEquals(repo.hierarchy.subtreeSize[0], nodeCount, "the root's subtree is everything");
  }
});

Deno.test("repo: slots are in DFS pre-order, so subtrees are contiguous ranges", () => {
  const repo = generateRepo({ nodeCount: 2_500 });
  const { parent, subtreeSize, depth } = repo.hierarchy;

  for (let slot = 1; slot < repo.nodeCount; slot++) {
    assert(parent[slot] < slot, `slot ${slot} precedes its parent ${parent[slot]}`);
    assertEquals(depth[slot], depth[parent[slot]] + 1);
  }

  // Pre-order means the subtree of `slot` is exactly [slot, slot + size).
  for (let slot = 0; slot < repo.nodeCount; slot++) {
    const end = slot + subtreeSize[slot];
    assert(end <= repo.nodeCount);
    for (let descendant = slot + 1; descendant < end; descendant++) {
      let walk = parent[descendant];
      while (walk !== HIERARCHY_ROOT && walk > slot) walk = parent[walk];
      assertEquals(
        walk,
        slot,
        `slot ${descendant} is inside the range of ${slot} but not under it`,
      );
    }
  }
});

Deno.test("repo: the first child of a node is the next slot", () => {
  // `main.ts` reads a node's fold state off `slot + 1`, which is only its first
  // child under pre-order emission.
  const repo = generateRepo({ nodeCount: 900 });
  const { parent } = repo.hierarchy;
  let checked = 0;
  for (let slot = 0; slot < repo.nodeCount; slot++) {
    if (repo.childCount[slot] === 0) continue;
    assertEquals(parent[slot + 1], slot, `slot ${slot + 1} should be the first child of ${slot}`);
    checked++;
  }
  assert(checked > 50, "the fixture should contain plenty of internal nodes");
});

Deno.test("repo: wellRadius follows the WASM bubble formula", () => {
  const repo = generateRepo({ nodeCount: 1_200 });
  const { parent, wellRadius } = repo.hierarchy;

  const childArea = new Float64Array(repo.nodeCount);
  for (let slot = repo.nodeCount - 1; slot >= 0; slot--) {
    const p = parent[slot];
    if (p !== HIERARCHY_ROOT) {
      childArea[p] += Math.PI * wellRadius[slot] * wellRadius[slot];
    }
  }

  for (let slot = 0; slot < repo.nodeCount; slot++) {
    if (repo.childCount[slot] === 0) {
      assertEquals(wellRadius[slot], BUBBLE_BASE_RADIUS, `leaf ${slot}`);
      continue;
    }
    const enclosing = Math.sqrt(childArea[slot] / (Math.PI * 0.82));
    const expected = Math.max(enclosing, BUBBLE_BASE_RADIUS) + BUBBLE_PADDING;
    assertAlmostEquals(wellRadius[slot], expected, 1e-3, `internal node ${slot}`);
  }
});

Deno.test("repo: a parent bubble encloses each of its children", () => {
  const repo = generateRepo({ nodeCount: 1_200 });
  const { parent, wellRadius } = repo.hierarchy;
  for (let slot = 1; slot < repo.nodeCount; slot++) {
    const p = parent[slot];
    assert(
      wellRadius[p] >= wellRadius[slot],
      `bubble ${p} (${wellRadius[p]}) is smaller than its child ${slot} (${wellRadius[slot]})`,
    );
  }
});

Deno.test("repo: kinds nest the way a repository does", () => {
  const repo = generateRepo({ nodeCount: 2_500 });
  const { parent } = repo.hierarchy;

  assertEquals(repo.kind[0], NODE_KIND.repo);
  for (let slot = 1; slot < repo.nodeCount; slot++) {
    const kind = repo.kind[slot];
    const parentKind = repo.kind[parent[slot]];
    switch (kind) {
      case NODE_KIND.repo:
        throw new Error(`slot ${slot} is a second repository root`);
      case NODE_KIND.symbol:
        assertEquals(parentKind, NODE_KIND.file, `symbol ${slot} must live in a file`);
        break;
      default:
        assert(
          parentKind === NODE_KIND.dir || parentKind === NODE_KIND.repo,
          `slot ${slot} of kind ${kind} hangs off kind ${parentKind}`,
        );
    }
  }

  // Symbols are the bottom of the tree; nothing nests inside one.
  for (let slot = 0; slot < repo.nodeCount; slot++) {
    if (repo.kind[slot] === NODE_KIND.symbol) {
      assertEquals(repo.childCount[slot], 0, `symbol ${slot} has children`);
    }
  }
  assert(repo.kindCounts[NODE_KIND.symbol] > 0, "a code graph without symbols is not one");
  assert(repo.kindCounts[NODE_KIND.dir] > 0);
});

Deno.test("repo: paths spell out the containment chain", () => {
  const repo = generateRepo({ nodeCount: 600 });
  const { parent } = repo.hierarchy;
  for (let slot = 1; slot < repo.nodeCount; slot++) {
    const separator = repo.kind[slot] === NODE_KIND.symbol ? "#" : "/";
    assertEquals(repo.path[slot], `${repo.path[parent[slot]]}${separator}${repo.name[slot]}`);
  }
  // Sibling names must differ, or two nodes share a path — and paths are the
  // external identifiers the typed input is loaded with.
  const seen = new Set(repo.path);
  assertEquals(seen.size, repo.nodeCount, "paths are not unique");
});

Deno.test("repo: the typed input carries the columns core needs", () => {
  const repo = generateRepo({ nodeCount: 800 });
  const input = repo.input;

  assertEquals(input.tag?.length, repo.nodeCount);
  assertEquals(input.weight?.length, repo.nodeCount);
  assertEquals(input.nodeRadii?.length, repo.nodeCount);
  assertEquals(input.nodeColors?.length, repo.nodeCount * 3);
  assertEquals(input.nodeIds?.length, repo.nodeCount);
  assertEquals(input.edgeKinds?.length, repo.edgeCount);
  assertEquals(input.edgePairs?.length, repo.edgeCount * 2);
  assertEquals(input.edgeColors?.length, repo.edgeCount * 3);
  assertEquals(input.containmentKind, 0);
  assertEquals(input.hierarchy, repo.hierarchy);

  const tag = input.tag as Uint16Array;
  const weight = input.weight as Float32Array;
  for (let slot = 0; slot < repo.nodeCount; slot++) {
    assertEquals(tag[slot], repo.kind[slot], "the tag column is the kind column");
    assert(weight[slot] >= 0 && weight[slot] <= 1, `weight ${weight[slot]} is outside 0..1`);
  }
});

Deno.test("repo: containment edges come first and match the parent column", () => {
  const repo = generateRepo({ nodeCount: 800 });
  const pairs = repo.input.edgePairs as Uint32Array;
  const kinds = repo.input.edgeKinds as Uint16Array;
  const { parent } = repo.hierarchy;

  assertEquals(repo.containmentEdgeCount, repo.nodeCount - 1);
  for (let slot = 1; slot < repo.nodeCount; slot++) {
    const edge = slot - 1;
    assertEquals(kinds[edge], 0, "containment edges carry the containment kind");
    assertEquals(pairs[edge * 2], parent[slot]);
    assertEquals(pairs[edge * 2 + 1], slot);
  }
});

Deno.test("repo: imports run between leaves, never to themselves", () => {
  const repo = generateRepo({ nodeCount: 2_500 });
  const pairs = repo.input.edgePairs as Uint32Array;
  const kinds = repo.input.edgeKinds as Uint16Array;

  let imports = 0;
  for (let edge = repo.containmentEdgeCount; edge < repo.edgeCount; edge++) {
    const source = pairs[edge * 2];
    const target = pairs[edge * 2 + 1];
    assertEquals(kinds[edge], 1, "an import must not be typed as containment");
    assert(source !== target, `edge ${edge} is a self-import`);
    for (const endpoint of [source, target]) {
      assert(endpoint < repo.nodeCount, `endpoint ${endpoint} is out of range`);
      const kind = repo.kind[endpoint];
      assert(
        kind === NODE_KIND.file || kind === NODE_KIND.symbol,
        `endpoint ${endpoint} of kind ${kind} cannot import`,
      );
    }
    imports++;
  }
  assert(imports > 0, "the fixture should contain imports");
});

Deno.test("repo: generation is a pure function of its seed", () => {
  const columns = (repo: RepoGraph) => [
    Array.from(repo.hierarchy.parent),
    Array.from(repo.kind),
    Array.from(repo.weight),
    repo.path,
  ];

  assertEquals(
    columns(generateRepo({ nodeCount: 500, seed: 7 })),
    columns(generateRepo({ nodeCount: 500, seed: 7 })),
  );

  const a = generateRepo({ nodeCount: 500, seed: 7 });
  const b = generateRepo({ nodeCount: 500, seed: 8 });
  assert(
    a.path.join("\n") !== b.path.join("\n"),
    "two seeds produced the same repository",
  );
});

Deno.test("repo: label candidates exclude symbols and are ranked by weight", () => {
  const repo = generateRepo({ nodeCount: 2_500 });
  const ranked = rankLabelCandidates(repo);

  assertEquals(ranked.length, repo.nodeCount - repo.kindCounts[NODE_KIND.symbol]);
  for (const slot of ranked) {
    assert(repo.kind[slot] !== NODE_KIND.symbol, `slot ${slot} is a symbol`);
  }

  // Descending by weight, so any prefix is the most important nodes — which is
  // the whole reason the density knob can take one.
  for (let i = 1; i < ranked.length; i++) {
    assert(
      repo.weight[ranked[i - 1]] >= repo.weight[ranked[i]],
      `candidates are not ranked at position ${i}`,
    );
  }
  assertEquals(ranked[0], 0, "the repository root outranks everything");

  // Ties broken by slot, so the ranking does not depend on sort stability.
  assertEquals(Array.from(ranked), Array.from(rankLabelCandidates(repo)));
});

Deno.test("repo: the offered scales are the ones the envelope is stated at", () => {
  assertEquals(REPO_SCALES.map((scale) => scale.nodeCount), [2_500, 35_000, 220_000]);
  assertEquals(REPO_SCALES.map((scale) => scale.id), ["small", "medium", "large"]);
});
