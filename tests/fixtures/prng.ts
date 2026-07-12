/**
 * Seeded PRNG for Deterministic Fixtures
 *
 * mulberry32: small, fast, 32-bit state. The same seed always yields the
 * same sequence, so fixture graphs are reproducible across runs and machines.
 *
 * @module
 */

/**
 * Random number generator function returning floats in [0, 1).
 */
export type Rng = () => number;

/**
 * Creates a mulberry32 PRNG from a 32-bit seed.
 *
 * @param seed - Any integer seed
 * @returns Function producing deterministic floats in [0, 1)
 */
export function mulberry32(seed: number): Rng {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Deterministic integer in [min, max] inclusive.
 */
export function randInt(rng: Rng, min: number, max: number): number {
  return min + Math.floor(rng() * (max - min + 1));
}
