/**
 * Contiguous slot runs.
 *
 * An LOD transition touches a set of slots and then has to get them to the GPU.
 * One `writeBuffer` per slot is what the pre-LOD visibility path did and what
 * the Phase 3 design exists to delete — 35 000 queue operations on a single
 * frame. One write spanning the whole set is not a substitute either: it would
 * carry along every slot in between, which is safe for a CPU-authoritative
 * buffer and silently destructive for one the GPU also writes.
 *
 * Maximal runs are the answer to both. A producer that emits slots depth-first
 * — the ordering `graph/hierarchy.ts` asks for — gives every subtree a
 * contiguous slot range, so a whole collapse or expand becomes a single write.
 *
 * @module
 */

/**
 * Visit the maximal contiguous runs of an ascending slot list as half-open
 * ranges.
 *
 * `slots[0, count)` must be ascending with no repeats; a caller holding an
 * unordered set sorts it first. Nothing is visited for an empty list.
 */
export function forEachSlotRun(
  slots: ArrayLike<number>,
  count: number,
  visit: (lo: number, hi: number) => void,
): void {
  if (count === 0) return;

  let lo = slots[0];
  let hi = lo + 1;
  for (let k = 1; k < count; k++) {
    if (slots[k] === hi) {
      hi++;
      continue;
    }
    visit(lo, hi);
    lo = slots[k];
    hi = lo + 1;
  }
  visit(lo, hi);
}
