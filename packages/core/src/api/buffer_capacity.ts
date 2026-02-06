/**
 * Buffer Capacity Management
 *
 * GPU buffers are fixed-size at creation. We over-allocate with headroom
 * and track count vs capacity separately. When count exceeds capacity,
 * buffers are reallocated at 2x growth.
 *
 * @module
 */

/**
 * Buffer capacity tracking
 */
export interface BufferCapacity {
  /** Logical count of active items */
  count: number;
  /** Allocated GPU buffer capacity (always >= count) */
  capacity: number;
}

/** Calculate initial capacity with headroom */
export function initialCapacity(count: number, multiplier = 2.0): number {
  return Math.max(Math.ceil(count * multiplier), 256);
}

/** Check if a count fits in existing capacity */
export function fitsInCapacity(count: number, capacity: number): boolean {
  return count <= capacity;
}

/** Calculate new capacity on overflow (2x growth) */
export function growCapacity(required: number, current: number): number {
  let newCap = current;
  while (newCap < required) {
    newCap = Math.max(newCap * 2, 256);
  }
  return newCap;
}
