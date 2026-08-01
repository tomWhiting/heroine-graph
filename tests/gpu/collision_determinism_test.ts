/**
 * GPU tests for collision resolution determinism and iteration semantics.
 *
 * Collision runs as two dispatches per iteration: a resolve pass that reads
 * positions and writes per-node displacements, then an apply pass that folds
 * them in. Resolving in place instead — reading positions[j] while writing
 * positions[i] in the same dispatch — is a cross-workgroup data race: whether
 * a thread sees a neighbour's pre- or post-displacement position depends on
 * how the driver schedules workgroups.
 *
 * WHAT THESE TESTS CAN AND CANNOT CATCH — read before trusting them:
 *
 * The repeat-run and mirror-symmetry cases are INVARIANT GUARDS, not reliable
 * regression tests for the race. They state properties an in-place resolve is
 * free to violate, but whether it does is up to the driver's scheduling on the
 * day. Measured against a shader reverted to the pre-fix in-place resolve on
 * this machine: the two `main` cases failed in 2 of 3 consecutive suite runs
 * and passed in the third; the `tiled` and `grid` cases never failed; an
 * earlier reviewer's single run saw all of them pass. The mirror-symmetry case
 * in particular cannot detect an in-place resolve even in principle — the left
 * half is the right half at a constant index offset and the halves never
 * interact, so any index-monotone execution order preserves the mirror. Keep
 * them (a racy build is unsound regardless of what one driver does on one
 * run), but do not read a green run as proof the race is gone.
 *
 * The Jacobi-semantics case at the bottom is the one with teeth: it recomputes
 * the expected multi-iteration result on the CPU under the semantics the
 * two-pass structure promises (resolve everything against one snapshot, then
 * apply), so it fails on any change that alters how iterations compose —
 * including hoisting the apply dispatch out of the iteration loop, which
 * silently discards all but the first iteration and which every other test
 * here passes (verified: that mutant fails this case alone, 11 others green).
 * It is not a race detector either: at 24 nodes the whole fixture is one
 * workgroup, and the in-place mutant above passes it.
 *
 * Fixtures are large enough to span several 256-thread workgroups — a race
 * confined to one workgroup would be masked by lockstep execution.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  type CollisionVariant,
  DEAD_SLOT_RADIUS,
  GPU_SKIP_MESSAGE,
  NODE_FLAG_DEAD,
  NODE_FLAG_PINNED,
  probeAdapter,
  requestHarnessDevice,
  runCollision,
} from "../helpers/gpu.ts";
import { mulberry32 } from "../fixtures/prng.ts";

const adapter = await probeAdapter();
if (!adapter) {
  console.warn(`[gpu] ${GPU_SKIP_MESSAGE}`);
}

function gpuTest(name: string, fn: (device: GPUDevice) => Promise<void>): void {
  Deno.test({
    name,
    ignore: adapter === null,
    sanitizeResources: false,
    sanitizeOps: false,
    async fn() {
      const device = await requestHarnessDevice(adapter!);
      try {
        await fn(device);
      } finally {
        device.destroy();
      }
    },
  });
}

const VARIANTS: CollisionVariant[] = ["main", "tiled", "grid"];

/** Node radius used by every fixture here (min separation 10 between nodes). */
const RADIUS = 5;

/**
 * The grid variant cannot be bitwise reproducible: its per-cell lists are
 * built with atomicExchange, so the traversal order — and therefore the
 * summation order of the displacement terms — varies between runs. That
 * reorders the last bits of a float sum; it never changes which neighbours a
 * node sees. Measured run-to-run spread on this fixture is ~2e-5, and a
 * position read across the resolve/apply boundary would shift nodes by whole
 * units, so 1e-3 separates the two cleanly.
 */
const GRID_TOLERANCE = 1e-3;

/**
 * Slack between the GPU run and the CPU reference below.
 *
 * The reference evaluates the same expressions in the same order with f32
 * rounding at every step, so the only residual is arithmetic the CPU cannot
 * mirror exactly: the driver is free to contract `a*b + c` into an fma, and
 * sqrt/divide are only required to be correctly rounded per operation.
 * Measured worst-case divergence on this fixture after 3 iterations: 1.9e-6,
 * i.e. ~500x inside this tolerance. The 1-iteration answer differs from the
 * 3-iteration answer by 4.5 units, ~4500x outside it, so the tolerance
 * separates float noise from any change to iteration semantics by three
 * orders of magnitude on both sides.
 */
const REFERENCE_TOLERANCE = 1e-3;

/**
 * `count` nodes scattered over a `span`-wide box centred on (centerX, 0),
 * dense enough that most nodes overlap several neighbours.
 */
function overlappingCluster(
  count: number,
  seed: number,
  span: number,
  centerX: number,
): Float32Array {
  const rng = mulberry32(seed);
  const positions = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    positions[i * 2] = centerX + (rng() - 0.5) * span;
    positions[i * 2 + 1] = (rng() - 0.5) * span;
  }
  return positions;
}

/** Largest absolute difference between two position arrays. */
function maxDelta(a: Float32Array, b: Float32Array): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > worst) worst = d;
  }
  return worst;
}

for (const variant of VARIANTS) {
  gpuTest(
    `GPU collision (${variant}): repeated runs over identical input agree`,
    async (device) => {
      // 600 nodes (3 workgroups) packed at ~4x overlap density, resolved over
      // 3 iterations. Node 5 is pinned and node 7 is a dead slot: both must
      // hold their exact positions through every iteration, which is also the
      // check that a stale displacement never survives into the apply pass.
      const nodeCount = 600;
      const positions = overlappingCluster(nodeCount, 0x5EED, 150, 0);
      const sizes = new Float32Array(nodeCount).fill(RADIUS);
      sizes[7] = DEAD_SLOT_RADIUS;
      const flags = new Uint32Array(nodeCount);
      flags[5] = NODE_FLAG_PINNED;
      flags[7] = NODE_FLAG_DEAD;

      const runs: Float32Array[] = [];
      for (let run = 0; run < 3; run++) {
        runs.push(
          await runCollision(device, {
            positions,
            sizes,
            flags,
            variant,
            iterations: 3,
          }),
        );
      }

      // Sanity: the pass actually moved nodes (an inert pipeline would pass
      // every equality check below).
      assert(
        maxDelta(runs[0], positions) > RADIUS,
        "collision resolved nothing — fixture is not overlapping",
      );

      const tolerance = variant === "grid" ? GRID_TOLERANCE : 0;
      for (let run = 1; run < runs.length; run++) {
        const delta = maxDelta(runs[0], runs[run]);
        assert(
          delta <= tolerance,
          `run ${run} diverged from run 0 by ${delta} (tolerance ${tolerance})`,
        );
      }

      for (const result of runs) {
        assertEquals(result[10], positions[10], "pinned node displaced in x");
        assertEquals(result[11], positions[11], "pinned node displaced in y");
        assertEquals(result[14], positions[14], "dead slot displaced in x");
        assertEquals(result[15], positions[15], "dead slot displaced in y");
      }
    },
  );

  gpuTest(
    `GPU collision (${variant}): mirror-symmetric input resolves symmetrically`,
    async (device) => {
      // Two identical clusters mirrored about the y axis, far enough apart
      // that neither half touches the other (gap 100 >> max separation 10).
      // Node i in the right half and node HALF + i in the left half see the
      // same neighbours in the same order, so their displacements must mirror
      // each other exactly. They sit in different workgroups, so under an
      // in-place resolve their neighbours' update state can differ.
      const half = 320;
      const right = overlappingCluster(half, 0xC0FFEE, 100, 80);
      const positions = new Float32Array(half * 4);
      for (let i = 0; i < half; i++) {
        positions[i * 2] = right[i * 2];
        positions[i * 2 + 1] = right[i * 2 + 1];
        positions[(half + i) * 2] = -right[i * 2];
        positions[(half + i) * 2 + 1] = right[i * 2 + 1];
      }
      const sizes = new Float32Array(half * 2).fill(RADIUS);

      const result = await runCollision(device, {
        positions,
        sizes,
        variant,
        iterations: 3,
      });

      const tolerance = variant === "grid" ? GRID_TOLERANCE : 0;
      let moved = 0;
      for (let i = 0; i < half; i++) {
        const dx = Math.abs(result[(half + i) * 2] + result[i * 2]);
        const dy = Math.abs(result[(half + i) * 2 + 1] - result[i * 2 + 1]);
        assert(
          dx <= tolerance && dy <= tolerance,
          `node ${i} and its mirror ${half + i} resolved asymmetrically: ` +
            `dx=${dx}, dy=${dy} (tolerance ${tolerance})`,
        );
        moved += Math.abs(result[i * 2] - positions[i * 2]);
      }

      // Sanity: the halves were actually resolved, not left untouched
      // (mean |dx| over the right half above one unit).
      assert(moved > half, `collision barely moved anything: ${moved}`);
    },
  );
}

// ---------------------------------------------------------------------------
// Iteration semantics
// ---------------------------------------------------------------------------

/** Collision parameters the reference case runs under (no defaults involved). */
const REF_STRENGTH = 1.0;
const REF_MULTIPLIER = 1.0;
const REF_EPSILON = 0.0001;

/**
 * CPU reference for one resolve pass of collision.comp.wgsl's `main` entry
 * point, evaluated in f32 in the shader's iteration order (j ascending).
 *
 * Returns the per-node displacement, exactly what the resolve dispatch writes
 * into the displacements buffer.
 */
function referenceDisplacements(
  positions: Float32Array,
  sizes: Float32Array,
  flags: Uint32Array,
): Float32Array {
  const f = Math.fround;
  const n = sizes.length;
  const disp = new Float32Array(n * 2);

  for (let i = 0; i < n; i++) {
    if ((flags[i] & (NODE_FLAG_DEAD | NODE_FLAG_PINNED)) !== 0) continue;
    if (sizes[i] < 0) continue;
    const ri = f(sizes[i] * REF_MULTIPLIER);
    let dx = 0;
    let dy = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      if ((flags[j] & NODE_FLAG_DEAD) !== 0) continue;
      if (sizes[j] < 0) continue;
      const rj = f(sizes[j] * REF_MULTIPLIER);

      const deltaX = f(positions[i * 2] - positions[j * 2]);
      const deltaY = f(positions[i * 2 + 1] - positions[j * 2 + 1]);
      const distSq = f(f(deltaX * deltaX) + f(deltaY * deltaY));
      const dist = f(Math.sqrt(distSq));
      const minDist = f(ri + rj);

      if (dist < minDist && dist > REF_EPSILON) {
        const overlap = f(minDist - dist);
        const push = f(f(overlap * 0.5) * REF_STRENGTH);
        dx = f(dx + f(f(deltaX / dist) * push));
        dy = f(dy + f(f(deltaY / dist) * push));
      }
      // The coincident-node branch is deliberately unreachable in this
      // fixture (no two nodes share a position), so it needs no mirror here.
    }
    disp[i * 2] = dx;
    disp[i * 2 + 1] = dy;
  }
  return disp;
}

/**
 * CPU reference for `iterations` full collision iterations under Jacobi
 * semantics: every displacement in an iteration is computed against the same
 * position snapshot, then all of them are applied at once.
 */
function referenceCollision(
  positions: Float32Array,
  sizes: Float32Array,
  flags: Uint32Array,
  iterations: number,
): Float32Array {
  const f = Math.fround;
  const current = Float32Array.from(positions);
  for (let iter = 0; iter < iterations; iter++) {
    const disp = referenceDisplacements(current, sizes, flags);
    for (let i = 0; i < current.length; i++) {
      // The apply shader skips exactly-zero displacements rather than adding
      // them, so an untouched node's coordinate is bit-preserved.
      if (disp[i] !== 0) current[i] = f(current[i] + disp[i]);
    }
  }
  return current;
}

gpuTest(
  "GPU collision (main): multi-iteration output matches the Jacobi CPU reference",
  async (device) => {
    // Small and low-amplitude on purpose: the reference evaluates the same
    // arithmetic in f32 but cannot control whether the driver contracts
    // a*b+c into an fma, so keeping coordinates around ±20 keeps the residual
    // well under REFERENCE_TOLERANCE.
    //
    // What this pins that nothing else does: run the same fixture at
    // iterations=1 and the positions differ from the 3-iteration answer by
    // whole units (asserted below), so any change that drops or reorders
    // iterations — hoisting the apply dispatch out of the loop, applying
    // displacements from a stale iteration — lands orders of magnitude
    // outside the tolerance.
    const nodeCount = 24;
    const positions = overlappingCluster(nodeCount, 0xBEEF, 40, 0);
    const sizes = new Float32Array(nodeCount).fill(RADIUS);
    sizes[7] = DEAD_SLOT_RADIUS;
    const flags = new Uint32Array(nodeCount);
    flags[3] = NODE_FLAG_PINNED;
    flags[7] = NODE_FLAG_DEAD;

    const config = {
      collisionStrength: REF_STRENGTH,
      collisionRadiusMultiplier: REF_MULTIPLIER,
    };
    const iterations = 3;

    const actual = await runCollision(device, {
      positions,
      sizes,
      flags,
      variant: "main",
      iterations,
      config,
    });
    const expected = referenceCollision(positions, sizes, flags, iterations);
    const single = referenceCollision(positions, sizes, flags, 1);

    // The iterations must actually compose, or matching the reference proves
    // nothing about iteration handling.
    assert(
      maxDelta(expected, single) > 1,
      `3 iterations barely differ from 1 (${maxDelta(expected, single)}) — ` +
        "the fixture cannot detect dropped iterations",
    );

    const delta = maxDelta(actual, expected);
    assert(
      delta <= REFERENCE_TOLERANCE,
      `collision diverged from the Jacobi reference by ${delta} ` +
        `(tolerance ${REFERENCE_TOLERANCE}); 1-iteration output would differ by ` +
        `${maxDelta(actual, single)}`,
    );

    // Pinned node and dead slot are bit-preserved by the reference too.
    assertEquals(actual[6], positions[6], "pinned node displaced in x");
    assertEquals(actual[7], positions[7], "pinned node displaced in y");
    assertEquals(actual[14], positions[14], "dead slot displaced in x");
    assertEquals(actual[15], positions[15], "dead slot displaced in y");
  },
);
