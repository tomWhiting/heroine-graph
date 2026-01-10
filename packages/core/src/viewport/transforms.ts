/**
 * Coordinate Transform Utilities
 *
 * Provides matrix operations and coordinate transformations for
 * converting between screen space, graph space, and clip space.
 */

import type { BoundingBox, Vec2, ViewportState } from "../types.ts";

/**
 * 3x3 transformation matrix in column-major order.
 *
 * | m0 m3 m6 |
 * | m1 m4 m7 |
 * | m2 m5 m8 |
 */
export type Matrix3 = Float32Array;

/**
 * Create an identity matrix.
 */
export function identity(): Matrix3 {
  return new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
}

/**
 * Create a translation matrix.
 */
export function translate(tx: number, ty: number): Matrix3 {
  return new Float32Array([1, 0, 0, 0, 1, 0, tx, ty, 1]);
}

/**
 * Create a scale matrix.
 */
export function scale(sx: number, sy: number = sx): Matrix3 {
  return new Float32Array([sx, 0, 0, 0, sy, 0, 0, 0, 1]);
}

/**
 * Create a rotation matrix.
 */
export function rotate(radians: number): Matrix3 {
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  return new Float32Array([c, s, 0, -s, c, 0, 0, 0, 1]);
}

/**
 * Multiply two matrices.
 */
export function multiply(a: Matrix3, b: Matrix3): Matrix3 {
  const result = new Float32Array(9);

  result[0] = a[0] * b[0] + a[3] * b[1] + a[6] * b[2];
  result[1] = a[1] * b[0] + a[4] * b[1] + a[7] * b[2];
  result[2] = a[2] * b[0] + a[5] * b[1] + a[8] * b[2];

  result[3] = a[0] * b[3] + a[3] * b[4] + a[6] * b[5];
  result[4] = a[1] * b[3] + a[4] * b[4] + a[7] * b[5];
  result[5] = a[2] * b[3] + a[5] * b[4] + a[8] * b[5];

  result[6] = a[0] * b[6] + a[3] * b[7] + a[6] * b[8];
  result[7] = a[1] * b[6] + a[4] * b[7] + a[7] * b[8];
  result[8] = a[2] * b[6] + a[5] * b[7] + a[8] * b[8];

  return result;
}

/**
 * Transform a point by a matrix.
 */
export function transformPoint(m: Matrix3, p: Vec2): Vec2 {
  return {
    x: m[0] * p.x + m[3] * p.y + m[6],
    y: m[1] * p.x + m[4] * p.y + m[7],
  };
}

/**
 * Invert a matrix.
 */
export function invert(m: Matrix3): Matrix3 | null {
  const det = m[0] * (m[4] * m[8] - m[7] * m[5]) -
    m[3] * (m[1] * m[8] - m[7] * m[2]) +
    m[6] * (m[1] * m[5] - m[4] * m[2]);

  if (Math.abs(det) < 1e-10) {
    return null;
  }

  const invDet = 1 / det;
  const result = new Float32Array(9);

  result[0] = (m[4] * m[8] - m[7] * m[5]) * invDet;
  result[1] = (m[7] * m[2] - m[1] * m[8]) * invDet;
  result[2] = (m[1] * m[5] - m[4] * m[2]) * invDet;
  result[3] = (m[6] * m[5] - m[3] * m[8]) * invDet;
  result[4] = (m[0] * m[8] - m[6] * m[2]) * invDet;
  result[5] = (m[3] * m[2] - m[0] * m[5]) * invDet;
  result[6] = (m[3] * m[7] - m[6] * m[4]) * invDet;
  result[7] = (m[6] * m[1] - m[0] * m[7]) * invDet;
  result[8] = (m[0] * m[4] - m[3] * m[1]) * invDet;

  return result;
}

/**
 * Create a graph-to-screen transformation matrix.
 *
 * Transforms from graph coordinates to screen pixels.
 */
export function graphToScreenMatrix(viewport: ViewportState): Matrix3 {
  // 1. Translate to center on viewport position
  // 2. Scale by zoom level
  // 3. Translate to screen center
  const t1 = translate(-viewport.x, -viewport.y);
  const s = scale(viewport.scale, viewport.scale);
  const t2 = translate(viewport.width / 2, viewport.height / 2);

  return multiply(t2, multiply(s, t1));
}

/**
 * Create a screen-to-graph transformation matrix.
 *
 * Transforms from screen pixels to graph coordinates.
 */
export function screenToGraphMatrix(viewport: ViewportState): Matrix3 {
  const forward = graphToScreenMatrix(viewport);
  const inverse = invert(forward);
  return inverse ?? identity();
}

/**
 * Create a graph-to-clip transformation matrix for shaders.
 *
 * Clip space is [-1, 1] for both axes (WebGPU NDC).
 */
export function graphToClipMatrix(viewport: ViewportState): Matrix3 {
  // Graph -> Screen -> Clip
  // Screen: (0, 0) to (width, height)
  // Clip: (-1, -1) to (1, 1)

  const graphToScreen = graphToScreenMatrix(viewport);

  // Screen to clip
  const sx = 2 / viewport.width;
  const sy = -2 / viewport.height; // Flip Y for WebGPU
  const screenToClip = new Float32Array([sx, 0, 0, 0, sy, 0, -1, 1, 1]);

  return multiply(screenToClip, graphToScreen);
}

/**
 * Convert screen coordinates to graph coordinates.
 */
export function screenToGraph(viewport: ViewportState, screen: Vec2): Vec2 {
  return {
    x: (screen.x - viewport.width / 2) / viewport.scale + viewport.x,
    y: (screen.y - viewport.height / 2) / viewport.scale + viewport.y,
  };
}

/**
 * Convert graph coordinates to screen coordinates.
 */
export function graphToScreen(viewport: ViewportState, graph: Vec2): Vec2 {
  return {
    x: (graph.x - viewport.x) * viewport.scale + viewport.width / 2,
    y: (graph.y - viewport.y) * viewport.scale + viewport.height / 2,
  };
}

/**
 * Calculate the visible bounds in graph coordinates.
 */
export function getVisibleBounds(viewport: ViewportState): BoundingBox {
  const topLeft = screenToGraph(viewport, { x: 0, y: 0 });
  const bottomRight = screenToGraph(viewport, {
    x: viewport.width,
    y: viewport.height,
  });

  return {
    minX: topLeft.x,
    minY: topLeft.y,
    maxX: bottomRight.x,
    maxY: bottomRight.y,
  };
}

/**
 * Calculate scale to fit bounds within viewport.
 */
export function fitBoundsScale(
  bounds: BoundingBox,
  width: number,
  height: number,
  padding: number = 0,
): number {
  const contentWidth = bounds.maxX - bounds.minX;
  const contentHeight = bounds.maxY - bounds.minY;

  if (contentWidth <= 0 || contentHeight <= 0) {
    return 1;
  }

  const availableWidth = width - padding * 2;
  const availableHeight = height - padding * 2;

  return Math.min(availableWidth / contentWidth, availableHeight / contentHeight);
}

/**
 * Calculate center point of bounds.
 */
export function boundsCenter(bounds: BoundingBox): Vec2 {
  return {
    x: (bounds.minX + bounds.maxX) / 2,
    y: (bounds.minY + bounds.maxY) / 2,
  };
}

/**
 * Check if a point is within bounds.
 */
export function pointInBounds(point: Vec2, bounds: BoundingBox): boolean {
  return (
    point.x >= bounds.minX &&
    point.x <= bounds.maxX &&
    point.y >= bounds.minY &&
    point.y <= bounds.maxY
  );
}

/**
 * Expand bounds by a margin.
 */
export function expandBounds(bounds: BoundingBox, margin: number): BoundingBox {
  return {
    minX: bounds.minX - margin,
    minY: bounds.minY - margin,
    maxX: bounds.maxX + margin,
    maxY: bounds.maxY + margin,
  };
}

/**
 * Calculate distance from a point to the edge of bounds.
 */
export function distanceToBounds(point: Vec2, bounds: BoundingBox): number {
  const dx = Math.max(bounds.minX - point.x, 0, point.x - bounds.maxX);
  const dy = Math.max(bounds.minY - point.y, 0, point.y - bounds.maxY);
  return Math.sqrt(dx * dx + dy * dy);
}
