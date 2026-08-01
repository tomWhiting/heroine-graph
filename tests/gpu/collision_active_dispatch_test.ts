/**
 * GPU tests for collision dispatch under an LOD cut.
 *
 * WP-D made the force passes cost what is visible; the collision passes kept
 * paying full N. They masked IMMOVABLE slots, so a 90 %-hidden graph was
 * *correct* — it just resolved ten thousand nodes to move a thousand, every
 * iteration, three dispatches deep.
 *
 * The resolve passes now dispatch over the active-index list. The apply pass
 * deliberately does not: its body is a load, a compare and an add, and it is
 * what makes the shortened resolve dispatch safe — a slot no list entry names
 * keeps whatever displacement it was left with, and apply drops every
 * IMMOVABLE slot on the flags rather than trusting the list.
 *
 * Three claims:
 *
 * - **The omitted entries contributed nothing.** Resolving the cut lands every
 *   node exactly where resolving the whole graph did, for all three code paths.
 * - **The dispatch really is the list.** Drop a visible, overlapping node from
 *   the list and it stops being displaced. This is the gather observed directly
 *   rather than inferred from timing.
 * - **Hidden nodes stay out of it.** They are stacked on each other and on
 *   visible nodes, and they do not move — including on the frame after they
 *   were hidden, when the displacement buffer still holds the nudge the
 *   previous cut computed for them.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  type CollisionVariant,
  GPU_SKIP_MESSAGE,
  NODE_FLAG_HIDDEN_LOD,
  probeAdapter,
  requestHarnessDevice,
  runCollision,
} from "../helpers/gpu.ts";

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

/** Overlapping pairs, `PAIRS` of them, of which every `VISIBLE_STRIDE`th is on screen. */
const PAIRS = 50;
const VISIBLE_STRIDE = 10;
/** Radius of every node, and how far apart a pair's two nodes start. */
const RADIUS = 10;
const OVERLAP_GAP = 6;
/** Distance between one pair and the next, wide enough that pairs never interact. */
const PAIR_PITCH = 120;

interface CutFixture {
  readonly positions: Float32Array;
  readonly sizes: Float32Array;
  readonly flags: Uint32Array;
  /** Ascending slots the cut leaves on screen. */
  readonly visible: Uint32Array;
  /** Ascending slots the cut hides. */
  readonly hidden: readonly number[];
}

/**
 * `PAIRS` overlapping pairs on a line, every `VISIBLE_STRIDE`th of them visible
 * and the rest hidden by the cut.
 *
 * Every pair overlaps by `2 * RADIUS - OVERLAP_GAP`, so every node in the
 * dispatch has something to push against and a node that is *not* dispatched
 * visibly fails to move. Hidden pairs sit at the same overlaps as visible ones,
 * which is what makes "they did not move" a statement about the cut rather than
 * about the geometry.
 *
 * The visible pairs are spread through the slot space rather than packed at the
 * front, and that is load-bearing: with a contiguous cut the active list is the
 * identity list over its own length, and a sweep that indexed by entry instead
 * of gathering `live_idx[entry]` would produce the same answer. Interleaving
 * makes the two disagree on every entry but the first.
 */
function overlappingPairs(): CutFixture {
  const nodeCount = PAIRS * 2;
  const positions = new Float32Array(nodeCount * 2);
  const sizes = new Float32Array(nodeCount).fill(RADIUS);
  const flags = new Uint32Array(nodeCount);
  const visible: number[] = [];
  const hidden: number[] = [];

  for (let p = 0; p < PAIRS; p++) {
    const x = p * PAIR_PITCH;
    positions[p * 4] = x;
    positions[p * 4 + 1] = 0;
    positions[p * 4 + 2] = x + OVERLAP_GAP;
    positions[p * 4 + 3] = 0;
    for (const slot of [p * 2, p * 2 + 1]) {
      if (p % VISIBLE_STRIDE === 0) {
        visible.push(slot);
      } else {
        flags[slot] = NODE_FLAG_HIDDEN_LOD;
        hidden.push(slot);
      }
    }
  }

  return { positions, sizes, flags, visible: Uint32Array.from(visible), hidden };
}

const VARIANTS: readonly CollisionVariant[] = ["main", "tiled", "grid"];

/**
 * How far two runs of the same variant may land apart.
 *
 * Zero for the two O(n²) paths: the inner sweep visits the same slots in the
 * same ascending order whether it walks the list or the whole slot space, so
 * the accumulated displacement is the same float. The grid path cannot promise
 * that — its per-cell lists are built by atomic prepend, so the order in which
 * a cell's occupants are summed depends on how the driver schedules the build —
 * and a shorter build dispatch reorders them. A tenth of a node radius is far
 * inside "the same layout".
 */
function toleranceFor(variant: CollisionVariant): number {
  return variant === "grid" ? RADIUS * 0.1 : 0;
}

/** Largest per-component distance between two interleaved position arrays. */
function worstShift(a: Float32Array, b: Float32Array, slots: Iterable<number>): number {
  let worst = 0;
  for (const slot of slots) {
    const d = Math.hypot(a[slot * 2] - b[slot * 2], a[slot * 2 + 1] - b[slot * 2 + 1]);
    if (!(d <= worst)) worst = d;
  }
  return worst;
}

for (const variant of VARIANTS) {
  gpuTest(`collision (${variant}): a 90 % cut resolves to the same layout`, async (device) => {
    const fixture = overlappingPairs();
    assert(
      fixture.hidden.length / (PAIRS * 2) >= 0.9,
      `the fixture must hide 90 % of the graph, hid ${fixture.hidden.length}`,
    );

    const run = (liveIndices?: Uint32Array) =>
      runCollision(device, {
        positions: fixture.positions,
        sizes: fixture.sizes,
        flags: fixture.flags,
        variant,
        iterations: 4,
        ...(liveIndices ? { liveIndices } : {}),
      });

    // The identity dispatch: every slot swept, inert ones masked one thread at
    // a time. This is what the passes did before the list reached them.
    const whole = await run();
    // The cut dispatch: one thread per visible slot.
    const cut = await run(fixture.visible);

    const observed = worstShift(whole, cut, fixture.visible);
    assert(
      observed <= toleranceFor(variant),
      `${variant}: shortening the dispatch moved the visible layout by ${observed}`,
    );

    // Non-vacuous: the visible nodes did separate, so the comparison above is
    // not two copies of "nothing happened".
    const separated = Math.abs(cut[2] - cut[0]);
    assert(
      separated > OVERLAP_GAP + 1,
      `${variant}: the visible pair must have separated, gap is ${separated}`,
    );

    // Hidden nodes are out of the collision set entirely: they overlap exactly
    // as hard as the visible pairs do and must not move a float.
    for (const slot of fixture.hidden) {
      assertEquals(
        cut[slot * 2],
        fixture.positions[slot * 2],
        `${variant}: hidden slot ${slot} was displaced (x)`,
      );
      assertEquals(
        cut[slot * 2 + 1],
        fixture.positions[slot * 2 + 1],
        `${variant}: hidden slot ${slot} was displaced (y)`,
      );
    }
  });

  gpuTest(`collision (${variant}): the resolve dispatch is the list`, async (device) => {
    // A list that omits a visible, overlapping node. Deliberately out of
    // contract — the list is meant to hold every non-inert slot — because it is
    // the only way to observe the gather itself rather than the flag mask: if
    // the resolve pass still swept every slot, the omitted node would separate
    // from its partner exactly as it does when it is listed.
    const fixture = overlappingPairs();
    const omitted = fixture.visible[0];
    const partner = fixture.visible[1];
    const short = fixture.visible.filter((slot) => slot !== omitted);

    const listed = await runCollision(device, {
      positions: fixture.positions,
      sizes: fixture.sizes,
      flags: fixture.flags,
      variant,
      iterations: 4,
      liveIndices: fixture.visible,
    });
    const unlisted = await runCollision(device, {
      positions: fixture.positions,
      sizes: fixture.sizes,
      flags: fixture.flags,
      variant,
      iterations: 4,
      liveIndices: Uint32Array.from(short),
    });

    assert(
      Math.abs(listed[omitted * 2] - fixture.positions[omitted * 2]) > 0.5,
      `${variant}: the listed node must be displaced, or the omission proves nothing`,
    );
    assertEquals(
      unlisted[omitted * 2],
      fixture.positions[omitted * 2],
      `${variant}: an omitted slot must not be resolved (x)`,
    );
    assertEquals(
      unlisted[omitted * 2 + 1],
      fixture.positions[omitted * 2 + 1],
      `${variant}: an omitted slot must not be resolved (y)`,
    );

    // What happens to its partner is where the three paths differ, and the
    // difference is the design rather than an accident:
    //
    // - `main` sweeps the LIST in its inner loop and the grid path builds its
    //   cell lists from the LIST, so an unlisted slot stops pushing as well as
    //   stops being pushed. That is the whole point — it is how the cost
    //   becomes O(active²) rather than O(active·N) — and it is why the list is
    //   derived from the flag shadow rather than being a knob.
    // - `resolve_tiled` loads its tiles by SLOT (it cannot gather: every thread
    //   in a workgroup has to reach the same barriers loading the same tile),
    //   so an unlisted slot is still a tile occupant and still pushes.
    const partnerMoved = Math.abs(unlisted[partner * 2] - fixture.positions[partner * 2]) > 0.5;
    if (variant === "tiled") {
      assert(partnerMoved, "tiled: the tile loop is slot-indexed, so the partner still separates");
    } else {
      assert(
        !partnerMoved,
        `${variant}: the inner sweep is the list, so an unlisted partner exerts nothing`,
      );
    }
  });
}

gpuTest("collision: a node hidden after being displaced is not nudged again", async (device) => {
  // The frame after a transition is the only place the apply pass's flag mask
  // can matter, and it is a real frame: the displacement buffer is scratch that
  // lives across frames, the resolve pass no longer writes every slot, and a
  // node hidden this frame still holds the displacement the last cut computed
  // for it. Without the mask, apply would fold that stale nudge into a node the
  // simulation has frozen — a hidden subtree twitching once on collapse.
  const fixture = overlappingPairs();
  const hiddenNow = fixture.visible[0];
  const stillVisible = fixture.visible.filter((slot) => slot !== hiddenNow);
  const hiddenFlags = fixture.flags.slice();
  hiddenFlags[hiddenNow] = NODE_FLAG_HIDDEN_LOD;

  for (const variant of VARIANTS) {
    // One frame with everything visible: both members of the first pair pick up
    // a non-zero displacement.
    const afterFirst = await runCollision(device, {
      positions: fixture.positions,
      sizes: fixture.sizes,
      flags: fixture.flags,
      variant,
      iterations: 1,
      liveIndices: fixture.visible,
    });
    const displaced = afterFirst[hiddenNow * 2];
    assert(
      Math.abs(displaced - fixture.positions[hiddenNow * 2]) > 0.5,
      `${variant}: the first frame must displace the node, or the stale value is zero`,
    );

    // Second frame, same buffers: the node is hidden and off the list.
    const afterSecond = await runCollision(device, {
      positions: fixture.positions,
      sizes: fixture.sizes,
      flags: fixture.flags,
      variant,
      iterations: 1,
      liveIndices: fixture.visible,
      then: { flags: hiddenFlags, liveIndices: Uint32Array.from(stillVisible) },
    });

    assertEquals(
      afterSecond[hiddenNow * 2],
      displaced,
      `${variant}: a newly hidden node was moved by its stale displacement (x)`,
    );
    assertEquals(
      afterSecond[hiddenNow * 2 + 1],
      afterFirst[hiddenNow * 2 + 1],
      `${variant}: a newly hidden node was moved by its stale displacement (y)`,
    );
  }
});
