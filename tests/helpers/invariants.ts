/**
 * Layout Invariant Metrics
 *
 * Pure functions over position arrays used to assert physical sanity of
 * simulation output (finiteness, boundedness, cooling) and structural
 * quality of hierarchical layouts (parent-child distances, radial
 * ordering). Exported as reusable infrastructure: future physics work
 * (nested-bubble algorithm tuning) should assert against these same
 * metrics.
 *
 * @module
 */

/**
 * Number of non-finite (NaN or +/-Inf) coordinates across both arrays.
 */
export function countNonFinite(xs: Float32Array, ys: Float32Array): number {
  let bad = 0;
  for (let i = 0; i < xs.length; i++) {
    if (!Number.isFinite(xs[i])) bad++;
    if (!Number.isFinite(ys[i])) bad++;
  }
  return bad;
}

/**
 * Largest distance of any node from the given center.
 */
export function maxRadius(xs: Float32Array, ys: Float32Array, cx = 0, cy = 0): number {
  let max = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - cx;
    const dy = ys[i] - cy;
    const r = Math.sqrt(dx * dx + dy * dy);
    if (r > max) max = r;
  }
  return max;
}

/**
 * Total kinetic energy estimated from per-node position deltas between two
 * consecutive ticks (unit mass, velocity ~= delta / dt). Works with
 * position-only readback; velocity buffers never leave the GPU.
 */
export function kineticEnergy(
  prevX: Float32Array,
  prevY: Float32Array,
  curX: Float32Array,
  curY: Float32Array,
  dt = 1,
): number {
  let energy = 0;
  for (let i = 0; i < prevX.length; i++) {
    const vx = (curX[i] - prevX[i]) / dt;
    const vy = (curY[i] - prevY[i]) / dt;
    energy += 0.5 * (vx * vx + vy * vy);
  }
  return energy;
}

/**
 * Mean Euclidean distance between each node and its parent.
 * Nodes with parent < 0 (roots) are skipped. Returns 0 if no pairs.
 */
export function meanParentChildDistance(
  xs: Float32Array,
  ys: Float32Array,
  parent: Int32Array,
): number {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < parent.length; i++) {
    const p = parent[i];
    if (p < 0) continue;
    const dx = xs[i] - xs[p];
    const dy = ys[i] - ys[p];
    sum += Math.sqrt(dx * dx + dy * dy);
    count++;
  }
  return count > 0 ? sum / count : 0;
}

/**
 * Mean Euclidean length of the given edges. Returns 0 if no edges.
 */
export function meanEdgeLength(
  xs: Float32Array,
  ys: Float32Array,
  sources: Uint32Array,
  targets: Uint32Array,
): number {
  const count = sources.length;
  if (count === 0) return 0;
  let sum = 0;
  for (let e = 0; e < count; e++) {
    const dx = xs[sources[e]] - xs[targets[e]];
    const dy = ys[sources[e]] - ys[targets[e]];
    sum += Math.sqrt(dx * dx + dy * dy);
  }
  return sum / count;
}

/**
 * Fraction of parent-child pairs where the child sits at a larger radius
 * from the center than its parent (in [0, 1]).
 *
 * A hierarchy laid out as nested rings scores near 1; the reported
 * "leaves dragged inward past their parents" failure scores near 0.
 * Not asserted strictly yet — exported as the tuning metric for the
 * nested-bubble work. Returns 1 if there are no pairs.
 */
export function radialOrderingScore(
  xs: Float32Array,
  ys: Float32Array,
  parent: Int32Array,
  cx = 0,
  cy = 0,
): number {
  let ordered = 0;
  let count = 0;
  for (let i = 0; i < parent.length; i++) {
    const p = parent[i];
    if (p < 0) continue;
    const childR = Math.hypot(xs[i] - cx, ys[i] - cy);
    const parentR = Math.hypot(xs[p] - cx, ys[p] - cy);
    if (childR >= parentR) ordered++;
    count++;
  }
  return count > 0 ? ordered / count : 1;
}

/**
 * Mean pairwise distance within clusters (same label) and across clusters
 * (different labels), plus their ratio.
 *
 * When labels are uncorrelated with position (the uniform-disc failure
 * mode) intra ~= inter and the ratio is ~1. A layout with real spatial
 * cluster separation drives the ratio well below 1. O(N^2) — intended for
 * fixture-sized graphs.
 */
export function clusterSeparation(
  xs: Float32Array,
  ys: Float32Array,
  labels: Uint32Array,
): { intra: number; inter: number; ratio: number } {
  let intraSum = 0;
  let intraCount = 0;
  let interSum = 0;
  let interCount = 0;
  for (let i = 0; i < xs.length; i++) {
    for (let j = i + 1; j < xs.length; j++) {
      const d = Math.hypot(xs[i] - xs[j], ys[i] - ys[j]);
      if (labels[i] === labels[j]) {
        intraSum += d;
        intraCount++;
      } else {
        interSum += d;
        interCount++;
      }
    }
  }
  const intra = intraCount > 0 ? intraSum / intraCount : 0;
  const inter = interCount > 0 ? interSum / interCount : 0;
  return { intra, inter, ratio: inter > 0 ? intra / inter : 1 };
}

/**
 * Index of dispersion (variance / mean) of node counts over equal-size
 * grid cells inside the layout's bounding disc.
 *
 * Cells are sized so a uniform layout puts ~`targetPerCell` nodes in each,
 * and only cells whose center lies inside the disc (centroid, radius = max
 * node radius) are counted, so empty space outside a circular layout does
 * not inflate the statistic. A uniform disc scores ~1 (Poisson); a layout
 * with dense clusters and empty gaps scores far above 1.
 */
export function densityDispersionIndex(
  xs: Float32Array,
  ys: Float32Array,
  targetPerCell = 4,
): number {
  const n = xs.length;
  if (n === 0) return 0;

  let cx = 0;
  let cy = 0;
  for (let i = 0; i < n; i++) {
    cx += xs[i];
    cy += ys[i];
  }
  cx /= n;
  cy /= n;

  const radius = maxRadius(xs, ys, cx, cy);
  if (radius === 0) return 0;

  // Cell edge such that a uniform disc yields ~targetPerCell nodes per cell
  const cell = radius * Math.sqrt((Math.PI * targetPerCell) / n);
  const gridSize = Math.ceil((2 * radius) / cell);

  const counts = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const gx = Math.min(gridSize - 1, Math.floor((xs[i] - (cx - radius)) / cell));
    const gy = Math.min(gridSize - 1, Math.floor((ys[i] - (cy - radius)) / cell));
    const key = gy * gridSize + gx;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }

  // Enumerate all cells inside the disc, including empty ones
  let cellCount = 0;
  let sum = 0;
  let sumSq = 0;
  for (let gy = 0; gy < gridSize; gy++) {
    for (let gx = 0; gx < gridSize; gx++) {
      const centerX = (gx + 0.5) * cell - radius;
      const centerY = (gy + 0.5) * cell - radius;
      if (centerX * centerX + centerY * centerY > radius * radius) continue;
      const c = counts.get(gy * gridSize + gx) ?? 0;
      cellCount++;
      sum += c;
      sumSq += c * c;
    }
  }
  if (cellCount === 0 || sum === 0) return 0;

  const mean = sum / cellCount;
  const variance = sumSq / cellCount - mean * mean;
  return variance / mean;
}
