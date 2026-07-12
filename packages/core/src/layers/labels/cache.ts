/**
 * Label Layout Cache Decision
 *
 * Pure CPU-side logic deciding whether the labels layer must re-run
 * culling, text measurement, and glyph generation for a frame, or can
 * re-render the previously generated glyph instances.
 *
 * Glyph instances are stored in graph space and transformed by the
 * viewport uniforms (updated every frame, 48 bytes), so a pure pan keeps
 * already-generated glyphs pixel-exact — a rebuild is only needed when
 * the visible label SET may have changed (camera moved far enough for
 * labels to enter/leave the viewport), when zoom changes the pixel-to-
 * graph conversion baked into glyph offsets, or when the labels or the
 * node positions feeding them changed.
 *
 * @module
 */

/**
 * The inputs the cached glyph layout depends on
 */
export interface LabelViewState {
  /** Viewport center X in graph space */
  viewportX: number;
  /** Viewport center Y in graph space */
  viewportY: number;
  /** Zoom scale (pixels per graph unit) */
  scale: number;
  /** Canvas width in pixels */
  canvasWidth: number;
  /** Canvas height in pixels */
  canvasHeight: number;
  /** Monotonic counter bumped when the label set or label config changes */
  labelsVersion: number;
  /** Fingerprint of (sampled) node positions feeding the labels */
  positionFingerprint: number;
}

/**
 * Re-cull when the camera pans more than this fraction of the viewport
 * extent (labels already generated stay pixel-exact under pan; only the
 * culled-in set goes slightly stale near the viewport edges).
 */
export const LABEL_REBUILD_PAN_FRACTION = 0.1;

/**
 * Re-generate glyphs when the zoom scale changes by more than this
 * relative amount (glyph offsets bake in the pixel-to-graph conversion).
 */
export const LABEL_REBUILD_ZOOM_FRACTION = 0.01;

/**
 * Decide whether label culling/measurement/glyph generation must re-run.
 *
 * @param prev - View state the cached glyphs were generated for (null = no cache)
 * @param next - Current frame's view state
 */
export function shouldRebuildLabels(
  prev: LabelViewState | null,
  next: LabelViewState,
): boolean {
  if (!prev) return true;

  if (prev.labelsVersion !== next.labelsVersion) return true;
  if (prev.positionFingerprint !== next.positionFingerprint) return true;
  if (prev.canvasWidth !== next.canvasWidth || prev.canvasHeight !== next.canvasHeight) {
    return true;
  }

  // Zoom: relative scale change beyond threshold
  if (Math.abs(next.scale - prev.scale) > Math.abs(prev.scale) * LABEL_REBUILD_ZOOM_FRACTION) {
    return true;
  }

  // Pan: camera center moved beyond a fraction of the viewport extent
  // (in graph units)
  const extentX = next.canvasWidth / next.scale;
  const extentY = next.canvasHeight / next.scale;
  if (Math.abs(next.viewportX - prev.viewportX) > extentX * LABEL_REBUILD_PAN_FRACTION) {
    return true;
  }
  if (Math.abs(next.viewportY - prev.viewportY) > extentY * LABEL_REBUILD_PAN_FRACTION) {
    return true;
  }

  return false;
}
