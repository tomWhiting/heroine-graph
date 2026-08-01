/**
 * Unit tests for the CPU half of LOD edge aggregation (WP-K).
 *
 * The walk itself is Rust and is tested by `cargo test`. What is tested here
 * is everything the walk's result passes through on the way to the GPU: the
 * decode of the flat encoding, the argument checks that stop a cut and a
 * hierarchy describing different slot spaces from being aggregated together,
 * the bundle render rows, and the opacity mask that hides the source edges a
 * bundle stands for and gives them back on expand.
 */

import { assert, assertEquals, assertThrows } from "jsr:@std/assert@^1";
import {
  aggregateEdges,
  buildBundleInstances,
  BUNDLE_MAX_WIDTH_SCALE,
  type BundleInstanceScratch,
  type BundleStyle,
  decodeEdgeAggregation,
  EDGE_BUNDLE_STRIDE,
  EDGE_OPACITY_HIDDEN,
  type EdgeOpacityHost,
  EdgeOpacityMask,
} from "../../packages/core/src/lod/edge_aggregation.ts";
import { GraphMotherError } from "../../packages/core/src/errors.ts";
import { HIERARCHY_ROOT } from "../../packages/core/src/graph/hierarchy.ts";
import { referenceAggregator } from "../helpers/edge_aggregation.ts";

// =============================================================================
// Fixture: root 0, modules 1 and 2, four leaves each
// =============================================================================

const NODE_COUNT = 11;

/** `parent[]` for root 0 → modules 1, 2 → leaves 3..6 and 7..10. */
function moduleParents(): Uint32Array {
  const parent = new Uint32Array(NODE_COUNT).fill(HIERARCHY_ROOT);
  parent[1] = 0;
  parent[2] = 0;
  for (let leaf = 3; leaf <= 6; leaf++) parent[leaf] = 1;
  for (let leaf = 7; leaf <= 10; leaf++) parent[leaf] = 2;
  return parent;
}

/** Root and both modules on screen, every leaf folded away. */
function collapsedModules(): Uint8Array {
  const visible = new Uint8Array(NODE_COUNT);
  visible[0] = 1;
  visible[1] = 1;
  visible[2] = 1;
  return visible;
}

// =============================================================================
// Decode
// =============================================================================

Deno.test("decode: the header slices the two lists", () => {
  //  2 live edges, 2 bundles.
  const data = Uint32Array.from([2, 2, 5, 9, /**/ 0, 1, 4, /**/ 1, 2, 7]);
  const aggregation = decodeEdgeAggregation(data);

  assertEquals(Array.from(aggregation.liveEdges), [5, 9]);
  assertEquals(Array.from(aggregation.bundles), [0, 1, 4, 1, 2, 7]);
  assertEquals(aggregation.bundleCount, 2);
});

Deno.test("decode: an empty aggregation is a bare header", () => {
  const aggregation = decodeEdgeAggregation(Uint32Array.from([0, 0]));
  assertEquals(aggregation.liveEdges.length, 0);
  assertEquals(aggregation.bundleCount, 0);
});

Deno.test("decode: a header disagreeing with the payload is rejected", () => {
  // Says three bundles, carries one.
  assertThrows(
    () => decodeEdgeAggregation(Uint32Array.from([0, 3, 1, 2, 4])),
    GraphMotherError,
  );
  assertThrows(() => decodeEdgeAggregation(new Uint32Array(1)), GraphMotherError);
});

// =============================================================================
// The seam
// =============================================================================

Deno.test("aggregate: crossing edges bundle onto the visible ancestors", () => {
  const parent = moduleParents();
  const visible = collapsedModules();
  // Four imports between the two modules' leaves, one written backwards, plus
  // one edge from a leaf to the visible root and one wholly inside module 1.
  const sources = Uint32Array.from([3, 4, 5, 8, 3, 4]);
  const targets = Uint32Array.from([7, 8, 9, 6, 0, 5]);

  const aggregation = aggregateEdges(referenceAggregator, sources, targets, parent, visible);

  assertEquals(Array.from(aggregation.bundles), [0, 1, 1, /**/ 1, 2, 4]);
  assertEquals(aggregation.bundleCount, 2);
  // The leaf-to-leaf edge inside module 1 has both endpoints under one proxy
  // and pulls on nothing that is still on screen.
  assertEquals(aggregation.liveEdges.length, 0);
});

Deno.test("aggregate: with nothing hidden every edge is live and nothing bundles", () => {
  const parent = moduleParents();
  const sources = Uint32Array.from([3, 4, 5]);
  const targets = Uint32Array.from([7, 8, 9]);

  const aggregation = aggregateEdges(
    referenceAggregator,
    sources,
    targets,
    parent,
    new Uint8Array(NODE_COUNT).fill(1),
  );

  assertEquals(Array.from(aggregation.liveEdges), [0, 1, 2]);
  assertEquals(aggregation.bundleCount, 0);
});

Deno.test("aggregate: a cut smaller than the hierarchy is rejected", () => {
  const parent = moduleParents();
  assertThrows(
    () =>
      aggregateEdges(
        referenceAggregator,
        new Uint32Array(0),
        new Uint32Array(0),
        parent,
        new Uint8Array(NODE_COUNT - 1),
      ),
    GraphMotherError,
  );
});

Deno.test("aggregate: mismatched source and target arrays are rejected", () => {
  assertThrows(
    () =>
      aggregateEdges(
        referenceAggregator,
        Uint32Array.from([0, 1]),
        Uint32Array.from([1]),
        moduleParents(),
        collapsedModules(),
      ),
    GraphMotherError,
  );
});

// =============================================================================
// Bundle render instances
// =============================================================================

// Colours and opacity are exact in f32, so the assertions read the value written.
const STYLE: BundleStyle = { width: 2, color: [0.25, 0.5, 0.75], opacity: 0.5 };
const ATTR_FLOATS = 8;

function scratch(): BundleInstanceScratch {
  return { indices: new Uint32Array(0), attributes: new Float32Array(0) };
}

Deno.test("bundle instances: endpoints, style and width proportional to weight", () => {
  const bundles = Uint32Array.from([0, 1, 1, /**/ 1, 2, 3]);
  const instances = buildBundleInstances(bundles, 2, STYLE, ATTR_FLOATS, scratch());

  assertEquals(Array.from(instances.indices), [0, 1, 1, 2]);
  assertEquals(instances.count, 2);
  assertEquals(instances.attributes[0], STYLE.width, "a weight-1 bundle is one edge wide");
  assertEquals(instances.attributes[ATTR_FLOATS], STYLE.width * 3, "width tracks the weight");
  assertEquals(
    Array.from(instances.attributes.subarray(1, 4)),
    [...STYLE.color],
    "colour is the bundle style's, not any member edge's",
  );
  assertEquals(instances.attributes[7], STYLE.opacity);
  assertEquals(instances.attributes[4], 0, "a bundle is never selected");
  assertEquals(instances.attributes[5], 0, "a bundle is never hovered");
});

Deno.test("bundle instances: width saturates so a huge bundle stays a line", () => {
  const weight = BUNDLE_MAX_WIDTH_SCALE * 1000;
  const instances = buildBundleInstances(
    Uint32Array.from([0, 1, weight]),
    1,
    STYLE,
    ATTR_FLOATS,
    scratch(),
  );
  assertEquals(instances.attributes[0], STYLE.width * BUNDLE_MAX_WIDTH_SCALE);
});

Deno.test("bundle instances: scratch is reused and never shrinks", () => {
  const reused = scratch();
  const large = new Uint32Array(30 * EDGE_BUNDLE_STRIDE);
  for (let k = 0; k < 30; k++) {
    large[k * EDGE_BUNDLE_STRIDE] = k;
    large[k * EDGE_BUNDLE_STRIDE + 1] = k + 1;
    large[k * EDGE_BUNDLE_STRIDE + 2] = 1;
  }
  buildBundleInstances(large, 30, STYLE, ATTR_FLOATS, reused);
  const grownIndices = reused.indices;
  const grownAttributes = reused.attributes;

  const small = buildBundleInstances(large.subarray(0, EDGE_BUNDLE_STRIDE), 1, STYLE, 8, reused);
  assert(reused.indices === grownIndices, "a smaller transition must not reallocate");
  assert(reused.attributes === grownAttributes, "a smaller transition must not reallocate");
  assertEquals(small.indices.length, 2, "the view is exactly the transition's size");
  assertEquals(small.attributes.length, ATTR_FLOATS);
});

// =============================================================================
// Source-edge opacity mask
// =============================================================================

/** An edge-attribute host over a plain opacity array. */
function opacityHost(opacity: Float32Array): EdgeOpacityHost {
  return {
    opacityOf: (edge) => opacity[edge],
    setOpacity: (edge, value) => {
      if (opacity[edge] === value) return false;
      opacity[edge] = value;
      return true;
    },
  };
}

Deno.test("opacity mask: hides exactly the edges outside the live list", () => {
  const opacity = new Float32Array(6).fill(1);
  const mask = new EdgeOpacityMask();

  const changed = mask.apply(Uint32Array.from([1, 4]), 6, opacityHost(opacity));

  assertEquals(Array.from(opacity), [0, 1, 0, 0, 1, 0]);
  assertEquals(changed, [0, 2, 3, 5], "changed edges come back ascending, for run coalescing");
  assertEquals(mask.size, 4);
});

Deno.test("opacity mask: expanding gives back the consumer's own opacity", () => {
  // A consumer that dimmed two edges before any LOD cut existed.
  const opacity = Float32Array.from([1, 0.25, 1, 0.5, 1, 1]);
  const mask = new EdgeOpacityMask();

  mask.apply(new Uint32Array(0), 6, opacityHost(opacity));
  assertEquals(Array.from(opacity), [0, 0, 0, 0, 0, 0]);

  mask.release(opacityHost(opacity));
  assertEquals(Array.from(opacity), [1, 0.25, 1, 0.5, 1, 1]);
  assertEquals(mask.size, 0);
});

Deno.test("opacity mask: a shrinking hidden set restores incrementally", () => {
  const opacity = Float32Array.from([1, 0.75, 1, 1]);
  const mask = new EdgeOpacityMask();
  const host = opacityHost(opacity);

  mask.apply(Uint32Array.from([0]), 4, host);
  assertEquals(Array.from(opacity), [1, EDGE_OPACITY_HIDDEN, 0, 0]);

  // Edge 1 comes back into the cut; edges 2 and 3 stay folded away.
  const changed = mask.apply(Uint32Array.from([0, 1]), 4, host);
  assertEquals(Array.from(opacity), [1, 0.75, 0, 0]);
  assertEquals(changed, [1]);
  assertEquals(mask.size, 2);
});

Deno.test("opacity mask: a second apply does not re-save the hidden value", () => {
  const opacity = Float32Array.from([0.625, 1]);
  const mask = new EdgeOpacityMask();
  const host = opacityHost(opacity);

  mask.apply(new Uint32Array(0), 2, host);
  mask.apply(new Uint32Array(0), 2, host);
  mask.release(host);

  assertEquals(
    Array.from(opacity),
    [0.625, 1],
    "re-saving would have recorded zero as the owed opacity",
  );
});

Deno.test("opacity mask: releasing with no host forgets without writing", () => {
  const opacity = Float32Array.from([1, 1]);
  const mask = new EdgeOpacityMask();
  const host = opacityHost(opacity);

  mask.apply(new Uint32Array(0), 2, host);
  // The edges the table names are gone with the graph they belonged to.
  assertEquals(mask.release(null), []);
  assertEquals(mask.size, 0);
  assertEquals(Array.from(opacity), [0, 0]);
});
