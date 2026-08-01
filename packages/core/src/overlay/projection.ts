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
 * zoom, so a card *inside* it is positioned and sized in graph units and is
 * scaled along with every sprite: one transform write per frame moves every
 * card, and a card keeps the apparent size of the node it replaced instead of
 * jumping when the swap happens.
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
 * Placement of a card centred on a graph-space anchor.
 *
 * Expressed in graph units, because that is the coordinate space *inside* the
 * transformed container — see the module doc.
 *
 * @param anchor - Graph-space position of the node the card stands for
 * @param width - Card width in graph units
 * @param height - Card height in graph units
 * @param opacity - Crossfade opacity, 0..1
 */
export function cardPlacementAt(
  anchor: Vec2,
  width: number,
  height: number,
  opacity: number,
): CardPlacement {
  return {
    x: anchor.x - width / 2,
    y: anchor.y - height / 2,
    width,
    height,
    opacity,
  };
}
