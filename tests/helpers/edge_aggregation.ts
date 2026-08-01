/**
 * Reference edge aggregator.
 *
 * The shipping aggregation walk lives in Rust
 * (`packages/wasm/src/layout/edge_aggregation.rs`, unit-tested by `cargo
 * test`) and reaches TypeScript through a wasm-bindgen method that only exists
 * once `deno task build:wasm` has run. The TypeScript half of the feature —
 * the decode, the spring dispatch, the bundle render — is seam-typed against
 * {@link EdgeAggregator} precisely so it can be exercised without that build.
 *
 * This is that seam's reference implementation: the same three outcomes per
 * edge (internal, live, crossing), the same unordered bundle key, the same
 * flat encoding. Deliberately naive — one walk up the parent chain per
 * endpoint and a `Map` for the dedupe — because its job is to be obviously
 * right, not fast. It is the definition the Rust version is checked against by
 * eye; the Rust tests assert the same properties independently.
 *
 * @module
 */

import { HIERARCHY_ROOT } from "../../packages/core/src/graph/hierarchy.ts";
import {
  EDGE_AGGREGATION_HEADER,
  EDGE_BUNDLE_STRIDE,
  type EdgeAggregator,
} from "../../packages/core/src/lod/edge_aggregation.ts";

/** Slot with no visible ancestor on its root path. */
const NONE = -1;

/**
 * The lowest ancestor of every slot that is visible, or {@link NONE}.
 *
 * A visible slot resolves to itself. Walks up from each slot independently, so
 * it is O(nodeCount * depth) — fine for a fixture, not for 220 000 slots.
 */
export function lowestVisibleAncestors(parent: Uint32Array, visible: Uint8Array): Int32Array {
  const nodeCount = parent.length;
  const ancestor = new Int32Array(nodeCount).fill(NONE);
  for (let slot = 0; slot < nodeCount; slot++) {
    let cursor = slot;
    for (let step = 0; step <= nodeCount; step++) {
      if (visible[cursor] === 1) {
        ancestor[slot] = cursor;
        break;
      }
      const up = parent[cursor];
      if (up === HIERARCHY_ROOT || up >= nodeCount) break;
      cursor = up;
    }
  }
  return ancestor;
}

/**
 * Aggregate `edgeSources`/`edgeTargets` against a visible cut, in the flat
 * encoding the WASM method returns:
 * `[liveCount, bundleCount, liveEdges…, (source, target, weight)…]`.
 */
export function referenceAggregateLodEdges(
  edgeSources: Uint32Array,
  edgeTargets: Uint32Array,
  parent: Uint32Array,
  visible: Uint8Array,
): Uint32Array {
  const nodeCount = parent.length;
  const ancestor = lowestVisibleAncestors(parent, visible);

  const liveEdges: number[] = [];
  const weights = new Map<number, number>();
  for (let edge = 0; edge < edgeSources.length; edge++) {
    const source = edgeSources[edge];
    const target = edgeTargets[edge];
    if (source >= nodeCount || target >= nodeCount) continue;
    if (visible[source] === 1 && visible[target] === 1) {
      liveEdges.push(edge);
      continue;
    }
    const a = ancestor[source];
    const b = ancestor[target];
    if (a === NONE || b === NONE || a === b) continue;
    const key = Math.min(a, b) * nodeCount + Math.max(a, b);
    weights.set(key, (weights.get(key) ?? 0) + 1);
  }

  const keys = [...weights.keys()].sort((p, q) => p - q);
  const data = new Uint32Array(
    EDGE_AGGREGATION_HEADER + liveEdges.length + keys.length * EDGE_BUNDLE_STRIDE,
  );
  data[0] = liveEdges.length;
  data[1] = keys.length;
  data.set(liveEdges, EDGE_AGGREGATION_HEADER);
  let out = EDGE_AGGREGATION_HEADER + liveEdges.length;
  for (const key of keys) {
    data[out++] = Math.floor(key / nodeCount);
    data[out++] = key % nodeCount;
    data[out++] = weights.get(key)!;
  }
  return data;
}

/** The reference walk behind the {@link EdgeAggregator} seam. */
export const referenceAggregator: EdgeAggregator = {
  aggregateLodEdges: referenceAggregateLodEdges,
};
