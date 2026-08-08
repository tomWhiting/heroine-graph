/**
 * Overlay Projection — Graph Space to CSS
 *
 * The DOM overlay rests entirely on one identity that already holds in
 * `viewport/transforms.ts`: `graphToClipMatrix(v) = screenToClip ∘
 * graphToScreenMatrix(v)`. A CSS `matrix()` built from `graphToScreenMatrix`
 * therefore lands a DOM element on the same pixel the GPU rasterizes, and the
 * overlay never grows a second camera to keep in sync with the first.
 *
 * One consequence shapes the rest of the overlay. The container carries the
 * zoom, so a card *inside* it is positioned in graph units and is moved along
 * with every sprite: one transform write per frame moves every card.
 *
 * Size is the part that cannot simply ride the camera. A card is real DOM with
 * real text, inputs and hit targets, so it needs two things at once that pull
 * in opposite directions: it must keep the apparent size of the sprite it
 * replaced (no jump at the swap), and it must eventually stand at its own CSS
 * size (or its text is magnified, its layout is computed at the wrong width,
 * and it grows without bound as the camera keeps zooming). Both hold if the
 * card is *laid out* at its natural CSS size and given a counter-scale — see
 * {@link cardCounterScale}.
 *
 * Everything here is pure and DOM-free, so the ≤0.5 px claim (SC-002) is
 * provable arithmetically — no browser, no rendering, no GPU.
 *
 * @module
 */

import type { Vec2, ViewportState } from "../types.ts";
import type { CardPlacement } from "./driver.ts";
import { graphToScreenMatrix } from "../viewport/transforms.ts";

/**
 * The six components of a CSS 2D transform, in `matrix(a, b, c, d, e, f)`
 * order — i.e. the affine map `[x y] -> [a·x + c·y + e, b·x + d·y + f]`.
 */
export interface CssMatrix {
  readonly a: number;
  readonly b: number;
  readonly c: number;
  readonly d: number;
  readonly e: number;
  readonly f: number;
}

/**
 * The container transform for a viewport state: the camera, expressed as a CSS
 * matrix.
 *
 * The components are read out of {@link graphToScreenMatrix}, which is
 * column-major and `Float32Array`-backed, so the overlay inherits exactly the
 * rounding the shader sees rather than recomputing the camera at a different
 * precision.
 */
export function overlayMatrix(viewport: ViewportState): CssMatrix {
  const m = graphToScreenMatrix(viewport);
  return { a: m[0], b: m[1], c: m[3], d: m[4], e: m[6], f: m[7] };
}

/**
 * Render a matrix as a CSS `transform` value.
 *
 * Nothing is rounded. `a` and `d` multiply a graph coordinate, so trimming
 * them trades an unbounded position error for cosmetics, and the SC-002 budget
 * is meant to be spent on device-pixel rounding in the compositor rather than
 * here. `String` only reaches exponent notation below 1e-6 or above 1e21, and
 * CSS accepts that notation anyway.
 */
export function formatCssMatrix(m: CssMatrix): string {
  return `matrix(${m.a}, ${m.b}, ${m.c}, ${m.d}, ${m.e}, ${m.f})`;
}

/** Apply a matrix to a point, in the same convention CSS applies it. */
export function projectByMatrix(m: CssMatrix, point: Vec2): Vec2 {
  return {
    x: m.a * point.x + m.c * point.y + m.e,
    y: m.b * point.x + m.d * point.y + m.f,
  };
}

/**
 * The scale a card's own transform must carry to sit correctly inside the
 * camera-scaled container.
 *
 * A card is laid out at its natural CSS size, so this is the whole of what
 * makes its *rendered* size behave. Composed with the container's camera scale
 * `S`, a card renders at `natural × S × cardCounterScale(S, S₀)`, which is
 * `natural × min(1, S / S₀)` — two regimes with one expression:
 *
 * - **`S ≥ S₀` (in the DOM band).** Net scale is exactly 1: the card renders at
 *   its natural CSS size, pixel for pixel, however far the camera keeps
 *   zooming. This is the ceiling. Without it a card is a magnified picture of
 *   a small card — 13px text drawn 8px-per-pixel, borders eight pixels thick,
 *   layout computed at an eighth of the width it appears to occupy — and it
 *   inflates without bound as the camera goes in. A card that has to hold a
 *   working input has to reach its own size and stop.
 * - **`S < S₀` (zooming back out, before the card is dropped).** Net scale is
 *   `S / S₀`, so the card shrinks with the camera exactly as the sprite it
 *   replaced does. This is what keeps the swap free of a jump in *both*
 *   directions, which is the property the graph-unit box was there to provide.
 *
 * `S₀` is the camera scale the card mounted at, and a card mounts on crossing
 * into the DOM band, so `S ≥ S₀` is the ordinary case and the natural size is
 * the size a card is seen at almost all of the time.
 *
 * @param cameraScale - Current camera scale `S`
 * @param mountScale - Camera scale when the card mounted, `S₀`
 */
export function cardCounterScale(cameraScale: number, mountScale: number): number {
  // A non-positive or non-finite camera scale has no sane counter-scale, and
  // dividing by it would put NaN into a style property, which silently drops
  // the whole transform. Fall back to unscaled: wrong size, still on screen.
  const larger = Math.max(cameraScale, mountScale);
  if (!Number.isFinite(larger) || larger <= 0) return 1;
  return 1 / larger;
}

/**
 * Placement of a card centred on a graph-space anchor.
 *
 * `x`/`y` are in graph units, because that is the coordinate space *inside* the
 * transformed container. `width`/`height` are in CSS pixels, because that is
 * the box the card is laid out in; `scale` reconciles the two. See the module
 * doc and {@link cardCounterScale}.
 *
 * @param anchor - Graph-space position of the node the card stands for
 * @param width - Card layout width in CSS pixels
 * @param height - Card layout height in CSS pixels
 * @param scale - Counter-scale from {@link cardCounterScale}
 * @param opacity - Crossfade opacity, 0..1
 */
export function cardPlacementAt(
  anchor: Vec2,
  width: number,
  height: number,
  scale: number,
  opacity: number,
): CardPlacement {
  // The card occupies `width * scale` graph units, so that — not the layout
  // box — is what has to be centred on the node.
  return {
    x: anchor.x - (width * scale) / 2,
    y: anchor.y - (height * scale) / 2,
    width,
    height,
    scale,
    opacity,
  };
}
