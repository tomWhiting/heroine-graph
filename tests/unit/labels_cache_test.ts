/**
 * Label layout cache tests.
 *
 * The labels layer must recompute culling/measurement/glyph generation
 * only when the camera moved beyond a threshold or the labels/positions
 * changed — not every rendered frame. shouldRebuildLabels is the pure
 * decision function; LabelManager provides the version counter and the
 * position fingerprint that feed it.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  LABEL_REBUILD_PAN_FRACTION,
  LABEL_REBUILD_ZOOM_FRACTION,
  type LabelViewState,
  shouldRebuildLabels,
} from "../../packages/core/src/layers/labels/cache.ts";
import { type LabelData, LabelManager } from "../../packages/core/src/layers/labels/manager.ts";

function state(overrides: Partial<LabelViewState> = {}): LabelViewState {
  return {
    viewportX: 100,
    viewportY: 50,
    scale: 2,
    canvasWidth: 800,
    canvasHeight: 600,
    labelsVersion: 1,
    positionFingerprint: 12345.5,
    nodeStateVersion: 7,
    ...overrides,
  };
}

Deno.test("shouldRebuildLabels: no cache -> rebuild", () => {
  assert(shouldRebuildLabels(null, state()));
});

Deno.test("shouldRebuildLabels: identical state -> cached", () => {
  assertEquals(shouldRebuildLabels(state(), state()), false);
});

Deno.test("shouldRebuildLabels: label set or position changes force a rebuild", () => {
  assert(shouldRebuildLabels(state(), state({ labelsVersion: 2 })));
  assert(shouldRebuildLabels(state(), state({ positionFingerprint: 12345.75 })));
});

Deno.test("shouldRebuildLabels: an LOD cut or card-set change forces a rebuild", () => {
  // The cut and the card set decide which nodes get a GPU label at all, and
  // neither moves a node — so nothing else in the view state would notice.
  assert(shouldRebuildLabels(state(), state({ nodeStateVersion: 8 })));
});

Deno.test("shouldRebuildLabels: canvas resize forces a rebuild", () => {
  assert(shouldRebuildLabels(state(), state({ canvasWidth: 801 })));
  assert(shouldRebuildLabels(state(), state({ canvasHeight: 601 })));
});

Deno.test("shouldRebuildLabels: small pan is cached, large pan rebuilds", () => {
  const prev = state();
  // Viewport extent in graph units: 800 / 2 = 400 wide, 600 / 2 = 300 tall
  const panLimitX = 400 * LABEL_REBUILD_PAN_FRACTION;

  assertEquals(
    shouldRebuildLabels(prev, state({ viewportX: 100 + panLimitX * 0.5 })),
    false,
    "pan below threshold must reuse cached glyphs",
  );
  assert(
    shouldRebuildLabels(prev, state({ viewportX: 100 + panLimitX * 2 })),
    "pan beyond threshold must re-cull",
  );
  assert(
    shouldRebuildLabels(prev, state({ viewportY: 50 + 300 * LABEL_REBUILD_PAN_FRACTION * 2 })),
  );
});

Deno.test("shouldRebuildLabels: small zoom is cached, larger zoom rebuilds", () => {
  const prev = state();
  const within = 2 * (1 + LABEL_REBUILD_ZOOM_FRACTION * 0.5);
  const beyond = 2 * (1 + LABEL_REBUILD_ZOOM_FRACTION * 2);

  assertEquals(shouldRebuildLabels(prev, state({ scale: within })), false);
  assert(shouldRebuildLabels(prev, state({ scale: beyond })));
  assert(shouldRebuildLabels(prev, state({ scale: 2 * (1 - LABEL_REBUILD_ZOOM_FRACTION * 2) })));
});

function label(nodeId: number, x: number, y: number, text = `n${nodeId}`): LabelData {
  return { nodeId, text, x, y, priority: 1 };
}

Deno.test("LabelManager: version bumps on every label/config/atlas mutation", () => {
  const manager = new LabelManager();
  const v0 = manager.version;

  manager.setLabels([label(0, 0, 0)]);
  const v1 = manager.version;
  assert(v1 > v0, "setLabels must bump version");

  manager.addLabel(label(1, 10, 10));
  const v2 = manager.version;
  assert(v2 > v1, "addLabel must bump version");

  manager.removeLabel(1);
  const v3 = manager.version;
  assert(v3 > v2, "removeLabel must bump version");

  manager.setConfig({ fontSize: 18 });
  const v4 = manager.version;
  assert(v4 > v3, "setConfig must bump version");

  manager.clear();
  assert(manager.version > v4, "clear must bump version");
});

Deno.test("LabelManager: position fingerprint is stable for static positions", () => {
  const manager = new LabelManager();
  manager.setLabels([label(0, 1, 2), label(1, 3, 4), label(2, -5, 6)]);

  assertEquals(manager.positionFingerprint(), manager.positionFingerprint());
});

Deno.test("LabelManager: position fingerprint changes when a node moves", () => {
  const manager = new LabelManager();
  manager.setLabels([label(0, 1, 2), label(1, 3, 4), label(2, -5, 6)]);

  const positions = new Map<number, { x: number; y: number }>([
    [0, { x: 1, y: 2 }],
    [1, { x: 3, y: 4 }],
    [2, { x: -5, y: 6 }],
  ]);
  const provider = {
    getX: (id: number) => positions.get(id)!.x,
    getY: (id: number) => positions.get(id)!.y,
  };

  const before = manager.positionFingerprint(provider);
  positions.set(1, { x: 3.25, y: 4 });
  const after = manager.positionFingerprint(provider);

  assert(before !== after, "moving a sampled node must change the fingerprint");
});

Deno.test("LabelManager: fingerprint order-weighting distinguishes swapped positions", () => {
  const manager = new LabelManager();
  // Same priority keeps sort stable; two labels with swapped coordinates
  // must not fingerprint identically (a plain sum would collide).
  manager.setLabels([label(0, 1, 2), label(1, 3, 4)]);

  const a = {
    getX: (id: number) => (id === 0 ? 1 : 3),
    getY: (id: number) => (id === 0 ? 2 : 4),
  };
  const b = {
    getX: (id: number) => (id === 0 ? 3 : 1),
    getY: (id: number) => (id === 0 ? 4 : 2),
  };

  assert(
    manager.positionFingerprint(a) !== manager.positionFingerprint(b),
    "swapping two nodes' positions must change the fingerprint",
  );
});
