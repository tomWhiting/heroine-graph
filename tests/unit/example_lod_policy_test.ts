/**
 * The code-graph example's LOD policy.
 *
 * The policy is the only place where the demo overrides core's geometric rule,
 * so what it does *not* override matters as much as what it does. Both of its
 * rules are structural claims about code trees rather than about pixels, which
 * is precisely why they cannot be checked by looking at the screen: a
 * single-child directory folded away leaves no trace, and a repository root
 * that folded would look like a working "too small to read" decision.
 */

import { assertEquals } from "jsr:@std/assert@^1";
import type { LodCandidate, LodContext } from "../../packages/core/src/lod/policy.ts";
import { codeGraphLodPolicy } from "../../examples/mission-control/src/lod_policy.ts";
import { NODE_KIND } from "../../examples/mission-control/src/repo.ts";

const CONTEXT: LodContext = {
  zoom: 1,
  expand: 96,
  collapse: 64,
  dom: 48,
  focus: new Set<number>(),
};

function candidate(patch: Partial<LodCandidate>): LodCandidate {
  return {
    node: 1,
    depth: 3,
    childCount: 4,
    subtreeSize: 40,
    wellRadius: 30,
    screenRadius: 30,
    onScreen: true,
    tag: NODE_KIND.dir,
    weight: 0.5,
    ...patch,
  };
}

Deno.test("policy: a repository root never stands in for its repository", () => {
  assertEquals(codeGraphLodPolicy(candidate({ depth: 0, tag: NODE_KIND.repo }), CONTEXT), "expand");
  // Even when geometry says it is far too small to read.
  assertEquals(
    codeGraphLodPolicy(
      candidate({ depth: 0, tag: NODE_KIND.repo, screenRadius: 0.5, onScreen: false }),
      CONTEXT,
    ),
    "expand",
  );
});

Deno.test("policy: a single-child directory is dropped from the cut", () => {
  assertEquals(
    codeGraphLodPolicy(candidate({ tag: NODE_KIND.dir, childCount: 1 }), CONTEXT),
    "pass-through",
  );
});

Deno.test("policy: a directory that branches is left to geometry", () => {
  for (const childCount of [0, 2, 3, 17]) {
    assertEquals(
      codeGraphLodPolicy(candidate({ tag: NODE_KIND.dir, childCount }), CONTEXT),
      "default",
      `childCount ${childCount}`,
    );
  }
});

Deno.test("policy: only directories pass through", () => {
  // A file with one symbol is not a redundant link in a chain — the symbol and
  // the file it lives in are different things, and dropping the file would
  // reparent the symbol onto a directory.
  for (const tag of [NODE_KIND.file, NODE_KIND.symbol, NODE_KIND.repo]) {
    assertEquals(
      codeGraphLodPolicy(candidate({ tag, childCount: 1 }), CONTEXT),
      "default",
      `tag ${tag}`,
    );
  }
});

Deno.test("policy: the root rule outranks the chain rule", () => {
  // A repository whose only top-level entry is one directory is still a
  // repository, and folding it away would leave nothing to look at.
  assertEquals(
    codeGraphLodPolicy(candidate({ depth: 0, tag: NODE_KIND.dir, childCount: 1 }), CONTEXT),
    "expand",
  );
});

Deno.test("policy: nothing else is overridden", () => {
  const decisions = new Set<string>();
  for (const tag of [NODE_KIND.repo, NODE_KIND.dir, NODE_KIND.file, NODE_KIND.symbol]) {
    for (const depth of [1, 2, 9]) {
      for (const childCount of [0, 2, 5]) {
        decisions.add(codeGraphLodPolicy(candidate({ tag, depth, childCount }), CONTEXT));
      }
    }
  }
  assertEquals([...decisions], ["default"]);
});
