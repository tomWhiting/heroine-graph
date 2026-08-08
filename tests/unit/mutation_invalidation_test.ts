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
 * This is asserted against the source because the invariant is structural:
 * what makes it hold is that no entry point exists which reaches the
 * invalidation without settling first, and that is a statement about the set of
 * paths rather than about any one execution. No suite of examples can establish
 * it, because the next unsettled method to be written is the one no example
 * covers.
 *
 * What this oracle CANNOT see is whether the call it finds does anything: empty
 * `beginTopologyChange`'s body and every assertion here still holds. That half
 * is covered behaviourally, on a real assembled instance, by "a mutation pays
 * out the drift a live fold owes its subtree" in
 * `tests/gpu/headless_graph_test.ts`. The two are complements and neither is
 * sufficient — this one enumerates the paths, that one proves the call has an
 * effect. Emptying the method makes exactly that test go red.
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

const SETTLE = "this.beginTopologyChange()";

/**
 * The deliberate exception: `load` replaces every position a translate would
 * move, so the drift is not lost there but moot, and paying it would be buffer
 * writes spent on positions about to be overwritten.
 */
const EXEMPT = "load";

/** One method of the class: how it was declared, and its body. */
interface Method {
  /** Whether the declaration carries `private`, i.e. is unreachable from a host. */
  hidden: boolean;
  body: string;
}

/**
 * Every method of `GraphMother`, by name.
 *
 * Enumerated rather than named, because the claim being tested is about the
 * *set* of paths into the invalidation: a hand-written list of entry points
 * asserts only that the listed ones are correct and is silent about the one
 * somebody adds next, which is precisely how a path comes to be missed. The
 * declaration pattern is anchored to two-space indentation, so nothing nested
 * inside a body — an object literal's methods, a `for (` — can be mistaken for
 * a declaration.
 */
function classMethods(): Map<string, Method> {
  const declaration =
    /\n {2}(private |public |protected )?(?:static )?(?:async )?([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^>]*>)?\(/g;
  const declarations = [...CLASS_SOURCE.matchAll(declaration)];
  const methods = new Map<string, Method>();
  for (let d = 0; d < declarations.length; d++) {
    const match = declarations[d];
    // Bounded by the next declaration, so a bodiless overload signature adopts
    // no body rather than the following method's.
    const limit = declarations[d + 1]?.index ?? CLASS_SOURCE.length;
    const open = CLASS_SOURCE.indexOf("{\n", match.index);
    if (open < 0 || open > limit) continue;
    let depth = 0;
    for (let i = open; i < CLASS_SOURCE.length; i++) {
      const c = CLASS_SOURCE[i];
      if (c === "{") depth++;
      else if (c === "}" && --depth === 0) {
        methods.set(match[2], {
          hidden: match[1] === "private ",
          body: CLASS_SOURCE.slice(open, i + 1),
        });
        break;
      }
    }
  }
  return methods;
}

const METHODS = classMethods();

/** The methods `name`'s body calls on `this`. */
function callees(name: string): string[] {
  const body = METHODS.get(name)!.body;
  return [...METHODS.keys()].filter(
    (other) => other !== name && new RegExp(`this\\.${other}\\(`).test(body),
  );
}

/**
 * Whether some path from `name` reaches the invalidation without passing the
 * settle first.
 *
 * A method that settles closes every path below it, which is what lets an entry
 * point cover the private helpers it delegates the invalidation to. Recursion
 * through a cycle answers false: a cycle adds no path that its entry did not
 * already offer.
 */
function unsettledPathToInvalidation(name: string, seen = new Set<string>()): boolean {
  if (seen.has(name)) return false;
  seen.add(name);
  const body = METHODS.get(name)!.body;
  if (body.includes(SETTLE)) return false;
  if (body.includes(INVALIDATE)) return true;
  return callees(name).some((callee) => unsettledPathToInvalidation(callee, seen));
}

Deno.test("mutation invalidation: the enumerator agrees with the by-name extractor", () => {
  // The oracle below is only worth anything if it sees the real bodies: an
  // enumerator that silently found nothing would pass every assertion in it.
  for (const name of Object.keys(SENTINELS)) {
    const found = METHODS.get(name);
    assert(found !== undefined, `the enumerator missed ${name}`);
    assertEquals(found.body, methodBody(name), `the two extractors disagree about ${name}`);
  }
  assert(METHODS.size > 100, `only ${METHODS.size} methods enumerated; the pattern has drifted`);
});

Deno.test("mutation invalidation: no reachable path drops the derived state unsettled", () => {
  // Every path that reaches the invalidation loses the outstanding fold drift,
  // whether or not it moved a slot: the hierarchy goes, and the anchors that
  // measured the drift go with it at the next adoption. So the obligation is
  // owed by the invalidation's callers rather than by mutation as such, and it
  // is checked over everything a host can call plus everything nothing calls.
  const callers = new Map<string, number>();
  for (const name of METHODS.keys()) {
    for (const callee of callees(name)) callers.set(callee, (callers.get(callee) ?? 0) + 1);
  }

  const offenders: string[] = [];
  for (const [name, method] of METHODS) {
    const reachable = !method.hidden || !callers.has(name);
    if (!reachable || name === EXEMPT) continue;
    if (unsettledPathToInvalidation(name)) offenders.push(name);
  }
  assertEquals(
    offenders,
    [],
    "these reach invalidateTopologyDerived without settling the outstanding fold drift first",
  );

  // Vacuity guard: the oracle is a search for the absence of something, so it
  // has to be shown finding the paths that do exist. `load` is one standing
  // example; the rest are made by taking each settle back out and requiring the
  // path it was covering to reappear, which is the only way to know the pass
  // above came from the call and not from the search failing to look.
  assert(unsettledPathToInvalidation(EXEMPT), "load must still be the one exempt path");
  assert(!methodBody(EXEMPT).includes(SETTLE));
  for (const name of ["addNode", "removeNodesBatch", "addEdgesBatch", "setForceAlgorithm"]) {
    const settled = METHODS.get(name)!;
    METHODS.set(name, { ...settled, body: settled.body.replaceAll(SETTLE, "") });
    const found = unsettledPathToInvalidation(name);
    METHODS.set(name, settled);
    assert(found, `${name} would pass this oracle even without its settle`);
  }
});

Deno.test("mutation invalidation: the settle precedes what it protects", () => {
  // Presence is not enough: once a compaction has run, reading a proxy's
  // position reads whichever node moved into its slot, so a flush issued from
  // there translates subtrees by an arbitrary distance.
  for (const [name, method] of METHODS) {
    const settle = method.body.indexOf(SETTLE);
    if (settle < 0) continue;
    const invalidate = method.body.indexOf(INVALIDATE);
    if (invalidate >= 0) {
      assert(settle < invalidate, `${name} settles after it has already invalidated`);
    }
    for (const callee of callees(name)) {
      if (!unsettledPathToInvalidation(callee, new Set())) continue;
      const call = method.body.indexOf(`this.${callee}(`);
      assert(settle < call, `${name} settles after calling ${callee}, which invalidates`);
    }
  }

  const batch = methodBody("removeNodesBatch");
  assert(
    batch.indexOf(SETTLE) < batch.indexOf("nodeSlotRemap.set("),
    "removeNodesBatch settles the drift after it has already moved the slots",
  );
});

Deno.test("mutation invalidation: a compaction hands its slot remap to the LOD controller", () => {
  // The focus set is host-declared slot numbers, so it is the one piece of LOD
  // state a rebuilt hierarchy cannot re-derive; it moves with the same map the
  // pins move with, or it follows whichever node inherited the slot.
  assert(methodBody("removeNodesBatch").includes("remapSlots(nodeSlotRemap)"));
  assert(methodBody("load").includes("remapSlots(new Map())"));
});

Deno.test("mutation invalidation: nothing else assigns the retained hierarchy", () => {
  // A second assignment site is a second answer to "is the cache stale?", which
  // is how the invalidation came to be reachable from only one path.
  const assignments = SOURCE.match(/this\.hierarchy = /g) ?? [];
  assertEquals(assignments.length, 2, "expected exactly the lazy build and the invalidation");
});
