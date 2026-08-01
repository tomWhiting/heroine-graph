/**
 * One authority for the per-slot columns a card reads.
 *
 * `parsedGraph` is the load-time parse and `MutableGraphState` is what every
 * mutation writes; a batch removal compacts the slot space and rebuilds the
 * metadata map against the new indices. While the two were separate objects a
 * card on the survivor of a removal rendered its *neighbour's* label and, worse,
 * fetched its neighbour's `contentRef` — no error, no warning, just the wrong
 * document. The producer's semantic columns have the same shape and the same
 * hazard, with no `MutableGraphState` counterpart at all.
 *
 * The state is built as `load()` builds it (real parser, real
 * `fromParsedGraph`, real aliasing) and the compaction body is replayed from
 * `GraphMother.removeNodesBatch` — the method needs a GPU device, its slot
 * bookkeeping does not.
 */

import { assertEquals, assertNotStrictEquals, assertStrictEquals } from "jsr:@std/assert@^1";
import {
  aliasParsedGraphToState,
  compactNodeColumn,
  growSlotColumn,
  MutableGraphState,
} from "../../packages/core/src/api/graph_state.ts";
import { parseGraphInput } from "../../packages/core/src/graph/parser.ts";
import type { ParsedGraph } from "../../packages/core/src/graph/parser.ts";

/** Four nodes carrying everything a card renders, plus the LOD policy columns. */
function loaded(): { parsed: ParsedGraph; gs: MutableGraphState } {
  const parsed = parseGraphInput({
    nodes: [
      { id: "A", metadata: { label: "Alpha", contentRef: "a.md" }, tag: 1, weight: 0.1 },
      { id: "B", metadata: { label: "Bravo", contentRef: "b.md" }, tag: 2, weight: 0.2 },
      { id: "C", metadata: { label: "Charlie", contentRef: "c.md" }, tag: 3, weight: 0.3 },
      { id: "D", metadata: { label: "Delta", contentRef: "d.md" }, tag: 4, weight: 0.4 },
    ],
    edges: [{ source: "A", target: "B", metadata: { kind: "import" } }],
  });
  const gs = MutableGraphState.fromParsedGraph(parsed);
  // load() aliases immediately after building the state.
  aliasParsedGraphToState(parsed, gs);
  return { parsed, gs };
}

/**
 * The slot bookkeeping of `removeNodesBatch` for a node-only graph: survivors
 * shift down, the metadata map is rebuilt against the write index, and the
 * producer columns are moved by the same remap. Returns the remap.
 */
function compact(
  parsed: ParsedGraph,
  gs: MutableGraphState,
  removed: readonly number[],
): Map<number, number> {
  const dead = new Set(removed);
  const remap = new Map<number, number>();
  const survivingMeta = new Map<number, Record<string, unknown>>();
  const prevHighWater = gs.nodeHighWater;

  let write = 0;
  for (let read = 0; read < prevHighWater; read++) {
    if (dead.has(read)) continue;
    const meta = gs.nodeMetadata.get(read);
    if (meta) survivingMeta.set(write, meta);
    remap.set(read, write);
    write++;
  }

  gs.nodeCount = write;
  gs.nodeHighWater = write;
  gs.nodeMetadata = survivingMeta;
  compactNodeColumn(parsed.nodeTags, remap, prevHighWater);
  compactNodeColumn(parsed.nodeWeights, remap, prevHighWater);
  aliasParsedGraphToState(parsed, gs);
  return remap;
}

Deno.test("metadata: the parse and the state share one map from the first alias", () => {
  const { parsed, gs } = loaded();

  assertStrictEquals(parsed.nodeMetadata, gs.nodeMetadata);
  assertStrictEquals(parsed.edgeMetadata, gs.edgeMetadata);
  assertEquals(parsed.nodeMetadata.get(0)?.["label"], "Alpha");
});

Deno.test("metadata: a card on a survivor reads its own label after a compaction", () => {
  const { parsed, gs } = loaded();

  // Removing A and C moves B to slot 0 and D to slot 1.
  compact(parsed, gs, [0, 2]);

  // What createCardNodeSource reads, for the node now in slot 0.
  assertEquals(parsed.nodeMetadata.get(0), { label: "Bravo", contentRef: "b.md" });
  assertEquals(parsed.nodeMetadata.get(1), { label: "Delta", contentRef: "d.md" });
  assertEquals(parsed.nodeMetadata.get(2), undefined);
  assertStrictEquals(parsed.nodeMetadata, gs.nodeMetadata);
});

Deno.test("metadata: tags and weights follow the slots they describe", () => {
  const { parsed, gs } = loaded();
  assertEquals([...parsed.nodeTags!], [1, 2, 3, 4]);

  compact(parsed, gs, [0, 2]);

  // getNodeTag / getNodeWeight index these directly by slot.
  assertEquals([...parsed.nodeTags!.subarray(0, 2)], [2, 4]);
  assertEquals(
    [...parsed.nodeWeights!.subarray(0, 2)].map((w) => Math.round(w * 10) / 10),
    [0.2, 0.4],
  );
  // Nothing readable is left above the last survivor.
  assertEquals([...parsed.nodeTags!.subarray(2)], [0, 0]);
  assertEquals([...parsed.nodeWeights!.subarray(2)], [0, 0]);
});

Deno.test("metadata: a column the input never carried stays absent through a compaction", () => {
  const parsed = parseGraphInput({ nodes: [{ id: "A" }, { id: "B" }], edges: [] });
  const gs = MutableGraphState.fromParsedGraph(parsed);
  aliasParsedGraphToState(parsed, gs);

  compact(parsed, gs, [0]);

  assertEquals(parsed.nodeTags, undefined);
  assertEquals(parsed.nodeWeights, undefined);
});

Deno.test("metadata: a semantic column grows to hold a slot added after load", () => {
  const tags = Uint16Array.of(7, 8);

  // Fits: the same array, written in place.
  assertStrictEquals(growSlotColumn(tags, 1, Uint16Array), tags);

  const grown = growSlotColumn(tags, 900, Uint16Array);
  assertNotStrictEquals(grown, tags);
  assertEquals(grown.length >= 901, true);
  assertEquals([...grown.subarray(0, 2)], [7, 8]);
  assertEquals(grown[900], 0);

  // Absent columns materialise on demand.
  const fresh = growSlotColumn(undefined, 3, Float32Array);
  assertEquals(fresh.length >= 4, true);
  assertEquals([...fresh.subarray(0, 4)], [0, 0, 0, 0]);
});
