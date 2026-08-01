/**
 * Every mutation entry point drops the state derived from the old topology.
 *
 * The hierarchy, the rolled-up masses, the proxy radii and the LOD edge
 * aggregation all name slots and edge indices of the graph they were computed
 * against. A mutation moves both, and none of it self-heals: the bundled spring
 * pass keeps driving springs from an edge list that now describes different
 * edges, and a new node's edges are dropped by `aggregate_edges` for as long as
 * the cut holds. The invalidation used to live inside `uploadAlgorithmEdgeData`,
 * which the singular paths — and the plural ones that loop over them — never
 * reach.
 *
 * This is asserted against the source because `GraphMother` cannot be
 * constructed headless (it requires an HTMLCanvasElement), so there is no seam
 * at which to observe the call. The invariant is structural anyway: what makes
 * it hold is that no mutation path exists which skips the call, and that is a
 * statement about the set of entry points, not about one execution. Replace
 * this with a behavioural test the day a GraphMother can be built over the
 * headless device.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";

const SOURCE = await Deno.readTextFile(
  new URL("../../packages/core/src/api/graph.ts", import.meta.url),
);

/** Methods only: the interfaces above the class declare `addNode`, `removeNode` too. */
const CLASS_SOURCE = SOURCE.slice(SOURCE.indexOf("\nexport class GraphMother"));

const INVALIDATE = "this.invalidateTopologyDerived()";

/**
 * The body of a method of the `GraphMother` class, braces included.
 *
 * The opening brace of a body is the first one followed by a newline: a return
 * type such as `Promise<{ removedCount: number }>` keeps its braces on the
 * signature line, so it cannot be mistaken for the body.
 */
function methodBody(name: string): string {
  const signature = new RegExp(`\\n  (?:private |public )?(?:async )?${name}\\(`);
  const start = CLASS_SOURCE.search(signature);
  assert(start >= 0, `no method named ${name} in graph.ts`);

  const open = CLASS_SOURCE.indexOf("{\n", start);
  assert(open > start, `no body found for ${name}`);

  let depth = 0;
  for (let i = open; i < CLASS_SOURCE.length; i++) {
    const c = CLASS_SOURCE[i];
    if (c === "{") depth++;
    else if (c === "}") {
      depth--;
      if (depth === 0) return CLASS_SOURCE.slice(open, i + 1);
    }
  }
  throw new Error(`unterminated body for ${name}`);
}

/** A string each body must contain, proving the extraction found the right one. */
const SENTINELS: Record<string, string> = {
  addNode: '"node:add"',
  addNodes: "flushNodeBuffersToGPU()",
  removeNode: '"node:remove"',
  removeNodesBatch: "nodeSlotRemap",
  addEdge: '"edge:add"',
  removeEdgeByIndex: "freeEdgeSlot(index)",
  uploadAlgorithmEdgeData: "generateForwardCSR()",
};

Deno.test("mutation invalidation: the body extractor finds the method it was asked for", () => {
  for (const [name, sentinel] of Object.entries(SENTINELS)) {
    const body = methodBody(name);
    assert(body.includes(sentinel), `${name}'s body does not contain ${sentinel}`);
    assert(body.startsWith("{\n") && body.endsWith("}"), `${name}'s body is not brace-delimited`);
  }
});

Deno.test("mutation invalidation: every singular and batch mutation invalidates", () => {
  for (const name of Object.keys(SENTINELS)) {
    assert(
      methodBody(name).includes(INVALIDATE),
      `${name} mutates the graph without calling invalidateTopologyDerived()`,
    );
  }
});

Deno.test("mutation invalidation: the plural paths reach a singular one that invalidates", () => {
  // These are per-item loops, not batch implementations: their invalidation is
  // the singular call's, once per item.
  assert(methodBody("removeNodes").includes("this.removeNode(ids[0])"));
  assert(methodBody("removeNodes").includes("this.removeNodesBatch(ids)"));
  assert(methodBody("addEdges").includes("await this.addEdge(edge)"));
  assert(methodBody("removeEdges").includes("await this.removeEdge(id)"));
  assert(methodBody("removeEdge").includes("this.removeEdgeByIndex(slot)"));
});

Deno.test("mutation invalidation: loading a graph drops the outgoing graph's derived state", () => {
  assert(methodBody("load").includes(INVALIDATE));
});

Deno.test("mutation invalidation: the invalidation still drops all four", () => {
  const body = methodBody("invalidateTopologyDerived");
  for (
    const statement of [
      "this.hierarchy = null",
      "this.resetNodeMass()",
      "this.resetLodProxyRadii()",
      "this.releaseLodEdgeAggregation()",
    ]
  ) {
    assert(body.includes(statement), `invalidateTopologyDerived no longer runs ${statement}`);
  }
});

Deno.test("mutation invalidation: nothing else assigns the retained hierarchy", () => {
  // A second assignment site is a second answer to "is the cache stale?", which
  // is how the invalidation came to be reachable from only one path.
  const assignments = SOURCE.match(/this\.hierarchy = /g) ?? [];
  assertEquals(assignments.length, 2, "expected exactly the lazy build and the invalidation");
});
