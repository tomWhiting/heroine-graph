/**
 * GPU tests for collision resolution determinism.
 *
 * Collision runs as two dispatches per iteration: a resolve pass that reads
 * positions and writes per-node displacements, then an apply pass that folds
 * them in. Resolving in place instead — reading positions[j] while writing
 * positions[i] in the same dispatch — is a cross-workgroup data race: whether
 * a thread sees a neighbour's pre- or post-displacement position depends on
 * how the driver schedules workgroups. These tests pin down the two
 * observable consequences of that race:
 *
 * - repeated runs over identical input must produce identical output;
 * - a mirror-symmetric input must resolve to a mirror-symmetric output
 *   (under the race, the two halves are scheduled independently, so the
 *   mirror nodes can read differently-updated neighbours).
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
      const device = await adapter!.requestDevice();
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
