/**
 * Edge aggregation for the semantic-LOD cut — the CPU half.
 *
 * The walk itself is in WASM (`packages/wasm/src/layout/edge_aggregation.rs`):
 * it is the one genuinely O(E) step of a transition, and a TypeScript version
 * of it is the thing most likely to blow the 4 ms budget. This module is what
 * surrounds it — decoding the flat result, turning bundles into edge-render
 * instances, and remembering which source edges the cut hid so they can be
 * shown again.
 *
 * Two consumers, one pass:
 *
 * - **The spring pass** dispatches over the decoded live-edge list plus the
 *   bundles. This is the physics fix: springs skip any edge with a hidden
 *   endpoint, so without a bundle a collapsed module exerts no pull at all on
 *   the modules it imports and the collapsed layout differs structurally from
 *   the expanded one.
 * - **The edge render** draws the bundles through the ordinary edge pipeline,
 *   width proportional to how many edges each stands for, while the source
 *   edges that lead into hidden nodes are taken to zero opacity.
 *
 * The source edge arrays are never re-targeted. Expanding is undone by
 * discarding the aggregation, not by unpicking a mutation.
 *
 * @module
 */

import { ErrorCode, GraphMotherError } from "../errors.ts";

/** `u32`s the WASM result puts before the live-edge list. */
export const EDGE_AGGREGATION_HEADER = 2;

/** `u32`s per bundle in the WASM result: source, target, weight. */
export const EDGE_BUNDLE_STRIDE = 3;

/**
 * Largest multiple of the base edge width a bundle is drawn at.
 *
 * Width is proportional to the bundle count as FR-004 asks, but only up to
 * here: a directory pair joined by 5 000 imports drawn at 5 000× width is a
 * filled viewport, not a graph. The physics is deliberately *not* capped the
 * same way — see the spring shader, where a cap would move the equilibrium the
 * expanded layout had.
 */
export const BUNDLE_MAX_WIDTH_SCALE = 8;

/** Opacity written to a source edge the cut hides. */
export const EDGE_OPACITY_HIDDEN = 0;

/**
 * The WASM surface edge aggregation needs.
 *
 * Structural, not nominal, so the glue is testable against the real module or
 * a stub without importing the whole engine type — the same seam
 * `HierarchyDeriver` uses.
 */
export interface EdgeAggregator {
  /**
   * Returns `[liveCount, bundleCount, liveEdges…, (source, target, weight)…]`.
   */
  aggregateLodEdges(
    edgeSources: Uint32Array,
    edgeTargets: Uint32Array,
    parent: Uint32Array,
    visible: Uint8Array,
  ): Uint32Array;
}

/**
 * The aggregated edge set for one cut.
 *
 * Both members are views into the WASM result, not copies: they are uploaded
 * and then dropped, and a transition that allocated a megabyte to hand it over
 * would be paying twice for the copy the boundary already made.
 */
export interface EdgeAggregation {
  /** Indices of the source edges with both endpoints in the cut, ascending. */
  readonly liveEdges: Uint32Array;
  /** `[source, target, weight]` per bundle, ascending by `(source, target)`. */
  readonly bundles: Uint32Array;
  /** Number of bundles, i.e. `bundles.length / EDGE_BUNDLE_STRIDE`. */
  readonly bundleCount: number;
}

/**
 * Decode the flat WASM aggregation result.
 *
 * @throws GraphMotherError if the header disagrees with the payload length,
 *   which means the two sides of the encoding have drifted apart — a silent
 *   mis-slice would upload one list as another.
 */
export function decodeEdgeAggregation(data: Uint32Array): EdgeAggregation {
  if (data.length < EDGE_AGGREGATION_HEADER) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `Edge aggregation result has ${data.length} entries, expected at least the ` +
        `${EDGE_AGGREGATION_HEADER}-word header`,
    );
  }
  const liveCount = data[0];
  const bundleCount = data[1];
  const expected = EDGE_AGGREGATION_HEADER + liveCount + bundleCount * EDGE_BUNDLE_STRIDE;
  if (data.length !== expected) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `Edge aggregation result has ${data.length} entries, expected ${expected} for ` +
        `${liveCount} live edges and ${bundleCount} bundles`,
    );
  }

  const bundleBase = EDGE_AGGREGATION_HEADER + liveCount;
  return {
    liveEdges: data.subarray(EDGE_AGGREGATION_HEADER, bundleBase),
    bundles: data.subarray(bundleBase),
    bundleCount,
  };
}

/**
 * Run the aggregation, checking the arguments the WASM side cannot.
 *
 * `visible[slot]` is 1 for a slot in the cut. Endpoints outside the slot space
 * are dropped rather than rejected: an edge naming a slot the hierarchy does
 * not describe has no ancestor to bundle onto.
 *
 * @throws GraphMotherError if the columns disagree about the slot space, since
 *   the aggregation would then be computed against a cut it does not describe.
 */
export function aggregateEdges(
  aggregator: EdgeAggregator,
  edgeSources: Uint32Array,
  edgeTargets: Uint32Array,
  parent: Uint32Array,
  visible: Uint8Array,
): EdgeAggregation {
  if (visible.length < parent.length) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `Edge aggregation visibility covers ${visible.length} slots, fewer than the ` +
        `hierarchy's ${parent.length}`,
    );
  }
  if (edgeSources.length !== edgeTargets.length) {
    throw new GraphMotherError(
      ErrorCode.INVALID_GRAPH_DATA,
      `Edge aggregation received ${edgeSources.length} sources and ${edgeTargets.length} ` +
        `targets`,
    );
  }
  return decodeEdgeAggregation(
    aggregator.aggregateLodEdges(edgeSources, edgeTargets, parent, visible),
  );
}

/** Render instance rows for a bundle set: slot pairs and attribute rows. */
export interface BundleInstances {
  /** `[source, target]` per bundle, for the edge pipeline's index buffer. */
  readonly indices: Uint32Array;
  /** `EDGE_ATTR_FLOATS` per bundle, for the edge pipeline's attribute buffer. */
  readonly attributes: Float32Array;
  /** Bundles written; both arrays are scratch and may be longer. */
  readonly count: number;
}

/** Scratch a caller reuses across transitions to keep the path allocation-free. */
export interface BundleInstanceScratch {
  indices: Uint32Array;
  attributes: Float32Array;
}

/**
 * Style every bundle shares. Bundles are structural, not semantic — they stand
 * for a set of edges of possibly different types — so they are drawn in one
 * neutral style rather than borrowing one member's.
 */
export interface BundleStyle {
  /** Width of a weight-1 bundle, in the same units as an edge's own width. */
  readonly width: number;
  /** Linear RGB, each 0..1. */
  readonly color: readonly [number, number, number];
  /** Multiplier on the edge pipeline's opacity channel, 0..1. */
  readonly opacity: number;
}

/**
 * Build the edge-pipeline instance rows for a bundle set.
 *
 * `attributeFloats` is the renderer's per-edge attribute stride; the first
 * four entries are width and colour and the last is opacity, matching what the
 * ordinary edge upload writes, so a bundle is an edge instance in every
 * respect but where its data comes from.
 *
 * Both arrays are grown in place on `scratch` and returned as views, so a
 * transition that fits inside the previous one allocates nothing.
 */
export function buildBundleInstances(
  bundles: Uint32Array,
  bundleCount: number,
  style: BundleStyle,
  attributeFloats: number,
  scratch: BundleInstanceScratch,
): BundleInstances {
  if (scratch.indices.length < bundleCount * 2) {
    scratch.indices = new Uint32Array(bundleCount * 2);
  }
  if (scratch.attributes.length < bundleCount * attributeFloats) {
    scratch.attributes = new Float32Array(bundleCount * attributeFloats);
  }

  const { indices, attributes } = scratch;
  const [r, g, b] = style.color;
  for (let k = 0; k < bundleCount; k++) {
    const base = k * EDGE_BUNDLE_STRIDE;
    indices[k * 2] = bundles[base];
    indices[k * 2 + 1] = bundles[base + 1];

    const attr = k * attributeFloats;
    attributes[attr] = style.width * Math.min(bundles[base + 2], BUNDLE_MAX_WIDTH_SCALE);
    attributes[attr + 1] = r;
    attributes[attr + 2] = g;
    attributes[attr + 3] = b;
    attributes[attr + 4] = 0; // selected
    attributes[attr + 5] = 0; // hovered
    attributes[attr + 6] = 0; // curvature
    attributes[attr + 7] = style.opacity;
  }

  return {
    indices: indices.subarray(0, bundleCount * 2),
    attributes: attributes.subarray(0, bundleCount * attributeFloats),
    count: bundleCount,
  };
}

/** The one edge-attribute reading and write the opacity mask needs. */
export interface EdgeOpacityHost {
  /** Current opacity of a source edge. */
  opacityOf(edge: number): number;
  /** Write a source edge's opacity. @returns whether the value actually moved */
  setOpacity(edge: number, opacity: number): boolean;
}

/**
 * The source edges the cut currently hides, and the opacity each is owed.
 *
 * An edge into a collapsed subtree must stop being drawn — its bundle is drawn
 * instead — but opacity is also a consumer-facing channel (`setEdgeOpacity`,
 * highlight dimming), so the value it had is saved rather than assumed. The
 * shape mirrors `ProxyRadiusTable`, which does the same job for the
 * render radius a proxy borrows (`lod/proxy_radius.ts`).
 *
 * Keyed on edge index rather than on an edge identity: unlike the node case
 * there is no producer id to follow, so a topology change invalidates the
 * table wholesale — which is exactly when the aggregation is recomputed
 * anyway.
 */
export class EdgeOpacityMask {
  readonly #saved = new Map<number, number>();

  /** How many source edges are currently forced transparent. */
  get size(): number {
    return this.#saved.size;
  }

  /**
   * Hide every source edge outside `liveEdges`, and give back the opacity of
   * every edge that is in it again.
   *
   * `liveEdges` is ascending and covers `[0, edgeCount)` of the source edges;
   * complete rather than incremental, so a caller cannot leak a transparent
   * edge by forgetting to name an expansion.
   *
   * @returns the edges whose opacity moved, ascending
   */
  apply(
    liveEdges: Uint32Array,
    edgeCount: number,
    host: EdgeOpacityHost,
  ): number[] {
    const changed: number[] = [];
    let cursor = 0;
    for (let edge = 0; edge < edgeCount; edge++) {
      const live = cursor < liveEdges.length && liveEdges[cursor] === edge;
      if (live) cursor++;

      if (live) {
        const saved = this.#saved.get(edge);
        if (saved === undefined) continue;
        this.#saved.delete(edge);
        if (host.setOpacity(edge, saved)) changed.push(edge);
        continue;
      }

      if (!this.#saved.has(edge)) this.#saved.set(edge, host.opacityOf(edge));
      if (host.setOpacity(edge, EDGE_OPACITY_HIDDEN)) changed.push(edge);
    }
    return changed;
  }

  /**
   * Empty the table, giving every edge its opacity back.
   *
   * @param host - Where to write, or `null` to forget without restoring, for
   *   the case where the edges the table names no longer exist
   * @returns the edges whose opacity moved, ascending
   */
  release(host: EdgeOpacityHost | null): number[] {
    const changed: number[] = [];
    if (host !== null) {
      for (const edge of [...this.#saved.keys()].sort((a, b) => a - b)) {
        if (host.setOpacity(edge, this.#saved.get(edge)!)) changed.push(edge);
      }
    }
    this.#saved.clear();
    return changed;
  }
}
