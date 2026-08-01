/**
 * Label level of detail.
 *
 * Three promises the GPU label set makes once semantic LOD and DOM cards are
 * in play, none of which the layer can be screenshotted for:
 *
 *  - a node the cut hides, or one already showing a card, gets no label —
 *    the first would be text for something that is not drawn, the second
 *    would be a second copy of the card's own title underneath it;
 *  - selection is ranked by on-screen size rather than by the producer's
 *    static priority, so zooming into a region promotes that region's labels
 *    and a collapsed bubble — inflated to its well radius in the same
 *    attribute row the sprite is drawn from — outranks the leaves it stands
 *    for (US4-AS3);
 *  - the set does not oscillate at a fixed camera.
 *
 * All of it runs against `LabelManager` directly with a stub atlas: the layer
 * around it is GPU plumbing, and the decisions under test are pure.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import { type BMFontChar, type FontAtlas } from "../../packages/core/src/layers/labels/atlas.ts";
import {
  type LabelData,
  LabelManager,
  type LabelManagerConfig,
  type LabelNodeSource,
} from "../../packages/core/src/layers/labels/manager.ts";

// -----------------------------------------------------------------------------
// Harness
// -----------------------------------------------------------------------------

/**
 * A monospaced stand-in for the shipped MSDF atlas: `measureText` reads
 * `info.size`, `chars` and `kernings` and nothing else, and the GPU handles
 * are only ever touched by the layer.
 */
function stubAtlas(): FontAtlas {
  const chars = new Map<number, BMFontChar>();
  for (let code = 32; code < 127; code++) {
    chars.set(code, {
      id: code,
      x: 0,
      y: 0,
      width: 8,
      height: 12,
      xoffset: 0,
      yoffset: 0,
      xadvance: 21,
      page: 0,
      chnl: 15,
    });
  }
  return {
    info: { face: "stub", size: 42, bold: 0, italic: 0, charset: [], unicode: 1 },
    common: { lineHeight: 50, base: 40, scaleW: 512, scaleH: 512, pages: 1, packed: 0 },
    chars,
    kernings: new Map<string, number>(),
    distanceRange: 4,
    destroy: () => {},
  } as unknown as FontAtlas;
}

function manager(config: Partial<LabelManagerConfig> = {}): LabelManager {
  // minZoom 0 so a test may choose any camera; the shipped default would
  // return nothing below 0.3 and hide the ranking under test.
  const instance = new LabelManager({ minZoom: 0, ...config });
  instance.setFontAtlas(stubAtlas());
  return instance;
}

function label(
  nodeId: number,
  x: number,
  y: number,
  priority: number,
  text = `n${nodeId}`,
): LabelData {
  return { nodeId, text, x, y, priority };
}

/**
 * The per-node readings the graph feeds the labels, in the shape it feeds
 * them: live positions, a live radius from the attribute row, and the union of
 * the LOD cut and the card set as one suppression predicate.
 */
function source(
  labels: readonly LabelData[],
  radii: Map<number, number>,
  hidden: Set<number> = new Set(),
  carded: Set<number> = new Set(),
): LabelNodeSource {
  const positions = new Map(labels.map((entry) => [entry.nodeId, entry]));
  return {
    getX: (id) => positions.get(id)?.x ?? 0,
    getY: (id) => positions.get(id)?.y ?? 0,
    getRadius: (id) => radii.get(id) ?? 0,
    isSuppressed: (id) => hidden.has(id) || carded.has(id),
  };
}

/** Node ids of the labels that survived, in selection order. */
function selected(
  instance: LabelManager,
  scale: number,
  nodes?: LabelNodeSource,
): number[] {
  return instance
    .getVisibleLabels(0, 0, scale, 800, 600, nodes)
    .map((visible) => visible.label.nodeId);
}

// -----------------------------------------------------------------------------
// Suppression
// -----------------------------------------------------------------------------

Deno.test("label LOD: nodes hidden by the cut and nodes holding a card get no label", () => {
  const instance = manager({ maxLabels: 10, gridCellSize: 1 });
  const labels = [label(0, -40, 0, 0.5), label(1, 0, 0, 0.5), label(2, 40, 0, 0.5)];
  instance.setLabels(labels);
  const radii = new Map([[0, 1], [1, 1], [2, 1]]);

  assertEquals(
    selected(instance, 1, source(labels, radii)).sort(),
    [0, 1, 2],
    "nothing is suppressed with a full cut and no cards",
  );
  assertEquals(
    selected(instance, 1, source(labels, radii, new Set([1]))).sort(),
    [0, 2],
    "a node the cut hides is not labelled",
  );
  assertEquals(
    selected(instance, 1, source(labels, radii, new Set(), new Set([2]))).sort(),
    [0, 1],
    "a node showing a card is not labelled underneath it",
  );
});

Deno.test("label LOD: suppression frees the budget for the next candidate", () => {
  // Otherwise suppression would only blank a label rather than hand its place
  // to a node that can use it — the budget is the scarce resource, not the
  // pixel area.
  const instance = manager({ maxLabels: 1, gridCellSize: 1 });
  const labels = [label(0, 0, 0, 0.9), label(1, 40, 0, 0.1)];
  instance.setLabels(labels);
  const radii = new Map([[0, 1], [1, 1]]);

  assertEquals(selected(instance, 1, source(labels, radii)), [0]);
  assertEquals(selected(instance, 1, source(labels, radii, new Set([0]))), [1]);
});

// -----------------------------------------------------------------------------
// Screen-space priority (US4-AS3)
// -----------------------------------------------------------------------------

Deno.test("label LOD: zoom promotes the label that is larger on screen", () => {
  const instance = manager({ maxLabels: 1, gridCellSize: 1 });
  // A small, important node against a large, unimportant one.
  const labels = [label(0, -20, 0, 0.9), label(1, 20, 0, 0.1)];
  instance.setLabels(labels);
  const nodes = source(labels, new Map([[0, 1], [1, 2]]));

  assertEquals(
    selected(instance, 0.1, nodes),
    [0],
    "wide camera: both are sub-pixel, so the producer's priority decides",
  );
  assertEquals(
    selected(instance, 10, nodes),
    [1],
    "close camera: the node that is bigger on screen takes the budget",
  );
});

Deno.test("label LOD: a collapsed proxy outranks the leaves it stands for", () => {
  // The proxy's radius is its well radius, written into the same attribute row
  // the sprite is drawn from, so the label path prefers bubbles by reading the
  // radius the renderer already uses — it never learns what a bubble is.
  const instance = manager({ maxLabels: 1, gridCellSize: 1 });
  const labels = [label(0, 0, 0, 0.1, "src"), label(1, 30, 0, 0.9, "leaf")];
  instance.setLabels(labels);

  assertEquals(
    selected(instance, 1, source(labels, new Map([[0, 3], [1, 3]]))),
    [1],
    "expanded: equal radii, so the more important leaf wins",
  );
  assertEquals(
    selected(instance, 1, source(labels, new Map([[0, 40], [1, 3]]))),
    [0],
    "collapsed: the inflated proxy wins",
  );
});

Deno.test("label LOD: a label with no radius keeps its static ordering", () => {
  // Backward compatibility for producers that never supplied a radius: with
  // no screen term at all, selection is priority order exactly as before.
  const instance = manager({ maxLabels: 2, gridCellSize: 1 });
  instance.setLabels([
    label(0, -40, 0, 0.1),
    label(1, 0, 0, 0.9),
    label(2, 40, 0, 0.5),
  ]);

  assertEquals(selected(instance, 4), [1, 2]);
});

// -----------------------------------------------------------------------------
// Anti-flicker
// -----------------------------------------------------------------------------

Deno.test("label LOD: the label set is identical across frames at a fixed camera", () => {
  // Nothing about a frame boundary is an input to selection, so re-culling at
  // the same camera must be a pure function — a set that reshuffles between
  // frames reads as flicker even when its size never changes.
  const instance = manager({ maxLabels: 4, gridCellSize: 50 });
  const labels: LabelData[] = [];
  const radii = new Map<number, number>();
  for (let i = 0; i < 12; i++) {
    labels.push(label(i, (i % 4) * 6 - 9, Math.floor(i / 4) * 6 - 6, (i % 3) / 3));
    radii.set(i, 1 + (i % 5));
  }
  instance.setLabels(labels);
  const nodes = source(labels, radii);

  const first = selected(instance, 3, nodes);
  assert(first.length > 0, "the fixture must actually place labels");
  assert(first.length < labels.length, "collision must actually bind");
  for (let frame = 1; frame < 8; frame++) {
    assertEquals(selected(instance, 3, nodes), first, `frame ${frame} differs`);
  }
});

Deno.test("label LOD: selection order is total, so equal ranks do not reshuffle", () => {
  const instance = manager({ maxLabels: 2, gridCellSize: 1 });
  const labels = [label(0, -40, 0, 0.5), label(1, 0, 0, 0.5), label(2, 40, 0, 0.5)];
  instance.setLabels(labels);
  const nodes = source(labels, new Map([[0, 2], [1, 2], [2, 2]]));

  assertEquals(selected(instance, 1, nodes), [0, 1], "ties break on the lower node id");
});
