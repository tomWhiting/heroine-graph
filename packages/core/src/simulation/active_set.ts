/**
 * The active set: which node slots the simulation actually dispatches over.
 *
 * `nodeFlags` is the state — one authoritative CPU shadow (`nodeFlagsShadow`
 * on MutableGraphState), with liveness on bit 0 and the semantic-LOD cut on
 * bit 2. This module turns that state into the *dispatch* form the force
 * shaders consume: a contiguous list of the slots that are neither dead nor
 * hidden, which the shaders index instead of walking every slot and masking.
 *
 * Keeping the two apart is the point. Masking has a hard ceiling — every
 * thread still launches and still reads the flag — while a list makes the cost
 * proportional to what is on screen. Deriving the list from the flags rather
 * than maintaining it alongside them means there is exactly one writer of
 * visibility and no way for the two to disagree about a slot.
 *
 * @module
 */

import { NODE_FLAGS_INERT } from "./pipeline.ts";

/**
 * Write the slots of `flags[0, slotCount)` that are neither dead nor hidden
 * into `out`, ascending, and return how many there are.
 *
 * Ascending order is load-bearing rather than incidental: the all-pairs
 * shaders sum in list order, so it is what keeps the result bit-identical to
 * the pre-list shaders' slot-order sum when nothing is hidden (SC-005).
 *
 * `out` is caller-owned and reused across transitions — this runs on every
 * flag change, and at 220 000 slots an allocation per change is the whole
 * transition budget. Only `[0, returned count)` is written; the tail keeps
 * whatever it held, which is why the count is the return value and never
 * `out.length`.
 *
 * @param flags - Per-slot flag words (the `nodeFlagsShadow` authority)
 * @param slotCount - Slots to consider, normally `nodeHighWater`
 * @param out - Destination, at least `slotCount` long
 * @returns Number of active slots written
 */
export function deriveActiveIndices(
  flags: Uint32Array,
  slotCount: number,
  out: Uint32Array,
): number {
  const end = Math.min(slotCount, flags.length, out.length);
  let count = 0;
  for (let slot = 0; slot < end; slot++) {
    if ((flags[slot] & NODE_FLAGS_INERT) !== 0) continue;
    out[count++] = slot;
  }
  return count;
}

/**
 * Whether `out[0, count)` already equals what {@link deriveActiveIndices}
 * would write for these flags.
 *
 * Lets a caller skip the queue write when a flag change did not move the
 * active set — pinning and unpinning, or a cut that re-evaluated to the same
 * frontier. Same scan cost as deriving, no upload and no buffer traffic.
 */
export function activeIndicesUnchanged(
  flags: Uint32Array,
  slotCount: number,
  out: Uint32Array,
  count: number,
): boolean {
  const end = Math.min(slotCount, flags.length);
  let seen = 0;
  for (let slot = 0; slot < end; slot++) {
    if ((flags[slot] & NODE_FLAGS_INERT) !== 0) continue;
    if (seen >= count || out[seen] !== slot) return false;
    seen++;
  }
  return seen === count;
}
