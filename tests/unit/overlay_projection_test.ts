/**
 * Overlay projection — SC-002, proved arithmetically.
 *
 * A DOM card must sit on the pixel the GPU would have rasterized the sprite
 * on. That claim is normally checked by looking at a screenshot; here it is
 * checked by arithmetic, because the two paths are two derivations of the same
 * camera and can be compared without rendering anything:
 *
 *   GPU:  world -> `graphToClipMatrix` -> NDC -> device pixels -> CSS pixels
 *   DOM:  world -> the CSS `matrix()` the overlay writes -> CSS pixels
 *
 * The CSS side goes through the *formatted string*, not the in-memory matrix,
 * so any precision the formatter drops shows up here rather than on screen.
 *
 * No DOM, no GPU, no rendering: everything under test is pure.
 */

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import {
  cardCounterScale,
  cardPlacementAt,
  type CssMatrix,
  formatCssMatrix,
  overlayMatrix,
  projectByMatrix,
} from "../../packages/core/src/overlay/projection.ts";
import { graphToClipMatrix, transformPoint } from "../../packages/core/src/viewport/transforms.ts";
import type { Vec2, ViewportState } from "../../packages/core/src/types.ts";

function viewport(overrides: Partial<ViewportState> = {}): ViewportState {
  return {
    x: 0,
    y: 0,
    scale: 1,
    width: 1920,
    height: 1080,
    minScale: 0.01,
    maxScale: 100,
    ...overrides,
  };
}

/** Read a `matrix(...)` value back, the way the browser's parser would. */
function parseCssMatrix(css: string): CssMatrix {
  const inner = css.match(/^matrix\((.*)\)$/);
  assert(inner !== null, `not a CSS matrix: ${css}`);
  const parts = inner[1].split(",").map((part) => Number(part.trim()));
  assertEquals(parts.length, 6, `expected six components: ${css}`);
  for (const part of parts) assert(Number.isFinite(part), `non-finite component in ${css}`);
  return { a: parts[0], b: parts[1], c: parts[2], d: parts[3], e: parts[4], f: parts[5] };
}

/**
 * Where the GPU puts a world point, in CSS pixels: clip space through the
 * viewport transform, into the drawing buffer, then back out of the device
 * pixel ratio the canvas was scaled by.
 */
function gpuScreenPosition(state: ViewportState, point: Vec2, dpr: number): Vec2 {
  const clip = transformPoint(graphToClipMatrix(state), point);
  const deviceX = ((clip.x + 1) / 2) * (state.width * dpr);
  const deviceY = ((1 - clip.y) / 2) * (state.height * dpr);
  return { x: deviceX / dpr, y: deviceY / dpr };
}

const VIEWPORTS: readonly ViewportState[] = [
  viewport(),
  viewport({ scale: 0.01 }),
  viewport({ scale: 0.37, x: -1234.5, y: 987.25 }),
  viewport({ scale: 2.5, x: 4096, y: -2048 }),
  viewport({ scale: 17.75, x: 512.125, y: 512.125 }),
  viewport({ scale: 100 }),
  viewport({ width: 640, height: 360, scale: 0.8, x: 33, y: -77 }),
  viewport({ width: 3840, height: 2160, scale: 6.25, x: -10000, y: 10000 }),
];

const DEVICE_PIXEL_RATIOS = [1, 1.5, 2, 3] as const;

const POINTS: readonly Vec2[] = [
  { x: 0, y: 0 },
  { x: 1, y: -1 },
  { x: 250.5, y: -412.25 },
  { x: -8192, y: 4096 },
  { x: 12345.75, y: -9876.5 },
];

Deno.test("overlay projection: container transform lands within 0.5 px of the GPU (SC-002)", () => {
  let worstRelative = 0;
  for (const state of VIEWPORTS) {
    // Through the string, so formatting is part of what is being asserted.
    const css = parseCssMatrix(formatCssMatrix(overlayMatrix(state)));
    for (const point of POINTS) {
      const dom = projectByMatrix(css, point);
      for (const dpr of DEVICE_PIXEL_RATIOS) {
        const gpu = gpuScreenPosition(state, point, dpr);
        const dx = Math.abs(dom.x - gpu.x);
        const dy = Math.abs(dom.y - gpu.y);
        worstRelative = Math.max(
          worstRelative,
          dx / Math.max(1, Math.abs(gpu.x)),
          dy / Math.max(1, Math.abs(gpu.y)),
        );
        assert(
          dx <= 0.5 && dy <= 0.5,
          `scale ${state.scale} dpr ${dpr} point (${point.x}, ${point.y}): ` +
            `DOM (${dom.x}, ${dom.y}) vs GPU (${gpu.x}, ${gpu.y})`,
        );
      }
    }
  }
  // Nothing above spends the budget: the residue is the f32 rounding of the
  // extra clip-space multiply on the GPU side, which is proportional to the
  // screen coordinate (0.06 px at zoom 100, a megapixel from the origin) and
  // is a few ulps in relative terms. A real divergence would be structural,
  // and would blow this by orders of magnitude before it blew the 0.5 px.
  assert(
    worstRelative < 1e-6,
    `container transform and GPU projection disagree by ${worstRelative} relative`,
  );
});

Deno.test("overlay projection: formatting a matrix loses nothing", () => {
  for (const state of VIEWPORTS) {
    const matrix = overlayMatrix(state);
    assertEquals(
      parseCssMatrix(formatCssMatrix(matrix)),
      matrix,
      "a rounded scale term multiplies the graph coordinate, so it may not be rounded",
    );
  }
});

Deno.test("overlay projection: the matrix is the viewport camera", () => {
  // Pure zoom + pan, no rotation or shear: the off-diagonal terms stay zero
  // and the diagonal is the zoom, which is what lets a card be sized in graph
  // units and inherit its apparent size from the camera.
  const matrix = overlayMatrix(viewport({ scale: 3, x: 100, y: -50, width: 800, height: 600 }));
  assertEquals(matrix.a, 3);
  assertEquals(matrix.d, 3);
  assertEquals(matrix.b, 0);
  assertEquals(matrix.c, 0);
  // Graph (100, -50) is the camera centre, so it maps to the canvas centre.
  assertEquals(projectByMatrix(matrix, { x: 100, y: -50 }), { x: 400, y: 300 });
});

Deno.test("overlay projection: a card is centred on its node", () => {
  const placement = cardPlacementAt({ x: 40, y: -20 }, 200, 120, 1, 0.75);
  assertEquals(placement, {
    x: -60,
    y: -80,
    width: 200,
    height: 120,
    scale: 1,
    opacity: 0.75,
  });

  // The centre of the *rendered* box is the anchor, at every size and every
  // counter-scale — the layout box is not what the user sees.
  for (const [width, height] of [[10, 10], [200, 120], [0, 0]] as const) {
    for (const scale of [1, 0.5, 0.125, 4]) {
      const box = cardPlacementAt({ x: 7, y: 9 }, width, height, scale, 1);
      assertEquals(box.x + (width * scale) / 2, 7);
      assertEquals(box.y + (height * scale) / 2, 9);
    }
  }
});

Deno.test("overlay projection: the counter-scale caps a card at its own size", () => {
  // A card mounts at scale 8 — the ordinary case, since a node cards when its
  // screen radius crosses a threshold well above its world radius.
  const mount = 8;
  const natural = 200;
  const rendered = (camera: number) => natural * camera * cardCounterScale(camera, mount);

  // At and above the mount scale the card is at its natural size, exactly, and
  // the ceiling does not sag however far the camera keeps going.
  assertEquals(rendered(8), natural);
  for (const camera of [8.0001, 16, 1e3, 1e6]) {
    assertAlmostEquals(rendered(camera), natural, 1e-9, `capped at camera ${camera}`);
  }

  // Below it the card tracks the camera, in proportion.
  assertEquals(rendered(4), natural / 2);
  assertEquals(rendered(1), natural / 8);
  assertEquals(rendered(0.5), natural / 16);

  // Continuous across the join: no jump at the moment the ceiling engages.
  assertAlmostEquals(rendered(8 - 1e-9), rendered(8), 1e-6);
});

Deno.test("overlay projection: the counter-scale never yields an unusable transform", () => {
  // A degenerate camera scale must not put NaN or Infinity into a style
  // property — the browser drops the whole transform and the card lands at the
  // container origin, which reads as every card stacked in one corner.
  for (const camera of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    const scale = cardCounterScale(camera, 4);
    assert(Number.isFinite(scale), `finite for camera ${camera}`);
    assert(scale > 0, `positive for camera ${camera}`);
  }
  for (const mount of [0, -1, Number.NaN]) {
    assert(Number.isFinite(cardCounterScale(2, mount)), `finite for mount ${mount}`);
  }
});
