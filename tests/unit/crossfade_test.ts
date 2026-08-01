/**
 * CrossfadeScheduler contract tests.
 *
 * The scheduler is the only writer of the per-node alpha buffer, so the
 * properties pinned down here are the ones a viewer would notice if they broke:
 * a fade never reverses direction on its own, it lands on exactly 0 or 1 (a
 * node left at 0.997 stays faintly drawn forever), it takes exactly
 * `transitionMs`, concurrent fades started at different times do not
 * contaminate each other, and a node re-targeted mid-fade turns around from
 * where it actually is rather than snapping to an endpoint.
 *
 * Everything here is driven by an explicit clock, so the tests assert exact
 * values rather than tolerances.
 */

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import {
  CrossfadeScheduler,
  NODE_ALPHA_OPAQUE,
  NODE_ALPHA_TRANSPARENT,
} from "../../packages/core/src/lod/crossfade.ts";

const TRANSITION_MS = 150;

function scheduler(capacity: number): CrossfadeScheduler {
  const s = new CrossfadeScheduler();
  s.resize(capacity);
  return s;
}

/** One recorded device.queue.writeBuffer call. */
interface Write {
  offsetBytes: number;
  values: number[];
}

/**
 * Minimal GPUDevice stand-in: the scheduler only ever reaches
 * `device.queue.writeBuffer`, so recording that one call is the whole
 * observable surface of {@link CrossfadeScheduler.flush}.
 */
function recordingDevice(): { device: GPUDevice; writes: Write[] } {
  const writes: Write[] = [];
  const queue = {
    // dataOffset/size are element counts (the scheduler passes a TypedArray).
    writeBuffer(
      _buffer: GPUBuffer,
      offsetBytes: number,
      data: Float32Array,
      dataOffset: number,
      size: number,
    ): void {
      writes.push({
        offsetBytes,
        values: Array.from(data.subarray(dataOffset, dataOffset + size)),
      });
    },
  };
  return { device: { queue } as unknown as GPUDevice, writes };
}

const FAKE_BUFFER = {} as GPUBuffer;

// =============================================================================
// Monotonicity
// =============================================================================

Deno.test("crossfade: a fade-out never moves back up", () => {
  const s = scheduler(1);
  s.fadeOut([0], 0, TRANSITION_MS);

  let previous = NODE_ALPHA_OPAQUE;
  for (let t = 0; t <= TRANSITION_MS + 20; t++) {
    s.advance(t);
    const alpha = s.alphaOf(0);
    assert(alpha <= previous, `alpha rose at t=${t}: ${previous} -> ${alpha}`);
    assert(alpha >= 0 && alpha <= 1, `alpha left [0,1] at t=${t}: ${alpha}`);
    previous = alpha;
  }
  assert(previous === NODE_ALPHA_TRANSPARENT);
});

Deno.test("crossfade: a fade-in never moves back down", () => {
  const s = scheduler(1);
  s.fadeOut([0], 0, 0); // start from fully transparent
  assertEquals(s.alphaOf(0), NODE_ALPHA_TRANSPARENT);

  s.fadeIn([0], 0, TRANSITION_MS);
  let previous = NODE_ALPHA_TRANSPARENT;
  for (let t = 0; t <= TRANSITION_MS + 20; t++) {
    s.advance(t);
    const alpha = s.alphaOf(0);
    assert(alpha >= previous, `alpha fell at t=${t}: ${previous} -> ${alpha}`);
    previous = alpha;
  }
  assertEquals(previous, NODE_ALPHA_OPAQUE);
});

Deno.test("crossfade: a clock that steps backwards cannot rewind a fade", () => {
  const s = scheduler(1);
  s.fadeOut([0], 1000, TRANSITION_MS);
  s.advance(1075);
  const midpoint = s.alphaOf(0);

  s.advance(1000);
  assertEquals(s.alphaOf(0), midpoint, "backwards clock rewound the ramp");
});

// =============================================================================
// Termination
// =============================================================================

Deno.test("crossfade: a fade terminates at exactly 0 or 1 and clears itself", () => {
  const s = scheduler(2);
  s.fadeOut([0], 0, TRANSITION_MS);
  s.fadeOut([1], 0, TRANSITION_MS);
  assertEquals(s.activeCount, 2);

  // One tick short of the end nothing has landed yet.
  s.advance(TRANSITION_MS - 1);
  assertEquals(s.activeCount, 2);
  assert(s.alphaOf(0) > NODE_ALPHA_TRANSPARENT);

  assertEquals(s.advance(TRANSITION_MS), true);
  assertEquals(s.alphaOf(0), NODE_ALPHA_TRANSPARENT);
  assertEquals(s.alphaOf(1), NODE_ALPHA_TRANSPARENT);
  assertEquals(s.activeCount, 0, "landed fades were not dropped");

  // Once landed, further ticks are inert.
  assertEquals(s.advance(TRANSITION_MS * 10), false);
  assertEquals(s.alphaOf(0), NODE_ALPHA_TRANSPARENT);
});

Deno.test("crossfade: overshooting the end lands on the endpoint, not past it", () => {
  const s = scheduler(1);
  s.fadeOut([0], 0, TRANSITION_MS);
  s.advance(TRANSITION_MS * 4);
  assertEquals(s.alphaOf(0), NODE_ALPHA_TRANSPARENT);
  assertEquals(s.activeCount, 0);
});

Deno.test("crossfade: transitionMs <= 0 applies immediately", () => {
  const s = scheduler(1);
  s.fadeOut([0], 0, 0);
  assertEquals(s.alphaOf(0), NODE_ALPHA_TRANSPARENT);
  assertEquals(s.activeCount, 0);

  s.fadeIn([0], 0, -5);
  assertEquals(s.alphaOf(0), NODE_ALPHA_OPAQUE);
  assertEquals(s.activeCount, 0);
});

// =============================================================================
// transitionMs
// =============================================================================

Deno.test("crossfade: a full fade is linear over exactly transitionMs", () => {
  const s = scheduler(1);
  s.fadeOut([0], 1000, TRANSITION_MS);

  for (const [elapsed, expected] of [[0, 1], [37.5, 0.75], [75, 0.5], [112.5, 0.25]] as const) {
    s.advance(1000 + elapsed);
    assertAlmostEquals(s.alphaOf(0), expected, 1e-6, `wrong alpha at +${elapsed}ms`);
  }

  s.advance(1000 + TRANSITION_MS);
  assertEquals(s.alphaOf(0), NODE_ALPHA_TRANSPARENT);
});

Deno.test("crossfade: transitionMs is honoured per call, not globally", () => {
  const s = scheduler(2);
  s.fadeOut([0], 0, 100);
  s.fadeOut([1], 0, 400);

  s.advance(100);
  assertEquals(s.alphaOf(0), NODE_ALPHA_TRANSPARENT, "short fade did not finish on time");
  assertAlmostEquals(s.alphaOf(1), 0.75, 1e-6, "long fade ran on the short duration");

  s.advance(400);
  assertEquals(s.alphaOf(1), NODE_ALPHA_TRANSPARENT);
});

// =============================================================================
// Concurrency
// =============================================================================

Deno.test("crossfade: fades started at different times stay independent", () => {
  const s = scheduler(4);
  // Node 3 starts transparent so it has somewhere to fade in from.
  s.fadeOut([3], 0, 0);

  s.fadeOut([0], 0, TRANSITION_MS);
  s.advance(50);
  s.fadeOut([1], 50, TRANSITION_MS);
  s.fadeIn([3], 50, TRANSITION_MS);
  s.advance(100);
  s.fadeOut([2], 100, TRANSITION_MS);

  // At t=125: node 0 is 125/150 through, node 1 is 75/150, node 2 is 25/150,
  // node 3 has risen 75/150 of the way back to opaque.
  s.advance(125);
  assertAlmostEquals(s.alphaOf(0), 1 - 125 / 150, 1e-6);
  assertAlmostEquals(s.alphaOf(1), 1 - 75 / 150, 1e-6);
  assertAlmostEquals(s.alphaOf(2), 1 - 25 / 150, 1e-6);
  assertAlmostEquals(s.alphaOf(3), 75 / 150, 1e-6);
  assertEquals(s.activeCount, 4);

  // Each lands exactly transitionMs after its own start, not after the last.
  s.advance(150);
  assertEquals(s.alphaOf(0), NODE_ALPHA_TRANSPARENT);
  assertEquals(s.activeCount, 3);
  s.advance(200);
  assertEquals(s.alphaOf(1), NODE_ALPHA_TRANSPARENT);
  assertEquals(s.alphaOf(3), NODE_ALPHA_OPAQUE);
  assertEquals(s.activeCount, 1);
  s.advance(250);
  assertEquals(s.alphaOf(2), NODE_ALPHA_TRANSPARENT);
  assertEquals(s.activeCount, 0);
});

Deno.test("crossfade: untouched slots keep their alpha while neighbours fade", () => {
  const s = scheduler(3);
  s.fadeOut([1], 0, TRANSITION_MS);
  for (let t = 0; t <= TRANSITION_MS; t += 10) {
    s.advance(t);
    assertEquals(s.alphaOf(0), NODE_ALPHA_OPAQUE);
    assertEquals(s.alphaOf(2), NODE_ALPHA_OPAQUE);
  }
});

// =============================================================================
// Re-targeting mid-fade
// =============================================================================

Deno.test("crossfade: re-targeting mid-fade reverses from the current value", () => {
  const s = scheduler(1);
  s.fadeOut([0], 0, TRANSITION_MS);
  s.advance(75);
  assertAlmostEquals(s.alphaOf(0), 0.5, 1e-6);

  // The turn itself must not move the value.
  s.fadeIn([0], 75, TRANSITION_MS);
  assertAlmostEquals(s.alphaOf(0), 0.5, 1e-6, "re-target snapped the alpha");

  // And it must rise from there at the same rate, reaching opaque after half
  // a transition rather than a whole one.
  s.advance(112.5);
  assertAlmostEquals(s.alphaOf(0), 0.75, 1e-6);
  s.advance(150);
  assertEquals(s.alphaOf(0), NODE_ALPHA_OPAQUE);
  assertEquals(s.activeCount, 0);
});

Deno.test("crossfade: re-targeting without ticking first still branches off the live value", () => {
  const ticked = scheduler(1);
  ticked.fadeOut([0], 0, TRANSITION_MS);
  ticked.advance(75);
  ticked.fadeIn([0], 75, TRANSITION_MS);

  const untouched = scheduler(1);
  untouched.fadeOut([0], 0, TRANSITION_MS);
  untouched.fadeIn([0], 75, TRANSITION_MS);

  assertEquals(untouched.alphaOf(0), ticked.alphaOf(0));
  for (const t of [90, 112.5, 130, 150]) {
    ticked.advance(t);
    untouched.advance(t);
    assertEquals(untouched.alphaOf(0), ticked.alphaOf(0), `diverged at t=${t}`);
  }
});

Deno.test("crossfade: re-issuing the target a fade already has does not restart it", () => {
  const s = scheduler(1);
  s.fadeOut([0], 0, TRANSITION_MS);
  // A controller re-evaluating the band every frame must not stall the ramp.
  for (let t = 0; t < TRANSITION_MS; t += 10) {
    s.advance(t);
    s.fadeOut([0], t, TRANSITION_MS);
  }
  s.advance(TRANSITION_MS);
  assertEquals(s.alphaOf(0), NODE_ALPHA_TRANSPARENT, "repeated scheduling stalled the fade");
});

Deno.test("crossfade: fading toward the value a slot already holds is a no-op", () => {
  const s = scheduler(1);
  s.fadeIn([0], 0, TRANSITION_MS);
  assertEquals(s.activeCount, 0);
  assertEquals(s.alphaOf(0), NODE_ALPHA_OPAQUE);
});

// =============================================================================
// Capacity
// =============================================================================

Deno.test("crossfade: growing preserves live values and opens new slots opaque", () => {
  const s = scheduler(2);
  s.fadeOut([0], 0, 0);
  s.fadeOut([1], 0, TRANSITION_MS);
  s.advance(75);
  const mid = s.alphaOf(1);

  s.resize(5);
  assertEquals(s.capacity, 5);
  assertEquals(s.alphaOf(0), NODE_ALPHA_TRANSPARENT);
  assertEquals(s.alphaOf(1), mid);
  for (const slot of [2, 3, 4]) assertEquals(s.alphaOf(slot), NODE_ALPHA_OPAQUE);
});

Deno.test("crossfade: slots re-entering the live range come back opaque", () => {
  const s = scheduler(4);
  s.fadeOut([3], 0, 0);
  assertEquals(s.alphaOf(3), NODE_ALPHA_TRANSPARENT);

  s.resize(2);
  s.resize(4);
  assertEquals(s.alphaOf(3), NODE_ALPHA_OPAQUE, "stale alpha survived a capacity round-trip");
});

Deno.test("crossfade: shrinking drops fades on slots that left the range", () => {
  const s = scheduler(4);
  s.fadeOut([1, 3], 0, TRANSITION_MS);
  assertEquals(s.activeCount, 2);

  s.resize(2);
  assertEquals(s.activeCount, 1);
  s.advance(TRANSITION_MS);
  assertEquals(s.alphaOf(1), NODE_ALPHA_TRANSPARENT);
});

Deno.test("crossfade: slots outside the live range are ignored", () => {
  const s = scheduler(2);
  s.fadeOut([-1, 2, 99, 1.5], 0, TRANSITION_MS);
  assertEquals(s.activeCount, 0);
  assertEquals(s.alphaOf(5), NODE_ALPHA_OPAQUE);
});

Deno.test("crossfade: reset cancels every fade and restores full opacity", () => {
  const s = scheduler(3);
  s.fadeOut([0, 1, 2], 0, TRANSITION_MS);
  s.advance(75);

  s.reset();
  assertEquals(s.activeCount, 0);
  for (const slot of [0, 1, 2]) assertEquals(s.alphaOf(slot), NODE_ALPHA_OPAQUE);
  assertEquals(s.advance(1000), false);
});

// =============================================================================
// Upload
// =============================================================================

Deno.test("crossfade: flush writes one contiguous range covering the changed slots", () => {
  const { device, writes } = recordingDevice();
  const s = scheduler(8);
  s.fadeOut([2, 5], 0, TRANSITION_MS);

  s.advance(75);
  assertEquals(s.flush(device, FAKE_BUFFER), true);
  assertEquals(writes.length, 1);
  // Slots 2..5 inclusive: the two fading nodes and the untouched pair between.
  assertEquals(writes[0].offsetBytes, 2 * 4);
  assertEquals(writes[0].values.length, 4);
  assertAlmostEquals(writes[0].values[0], 0.5, 1e-6);
  assertEquals(writes[0].values[1], NODE_ALPHA_OPAQUE);
  assertEquals(writes[0].values[2], NODE_ALPHA_OPAQUE);
  assertAlmostEquals(writes[0].values[3], 0.5, 1e-6);
});

Deno.test("crossfade: an idle frame produces no upload", () => {
  const { device, writes } = recordingDevice();
  const s = scheduler(4);

  assertEquals(s.advance(16), false);
  assertEquals(s.flush(device, FAKE_BUFFER), false);
  assertEquals(writes.length, 0);

  // A landed fade flushes once and then goes quiet again.
  s.fadeOut([1], 16, TRANSITION_MS);
  s.advance(16 + TRANSITION_MS);
  assertEquals(s.flush(device, FAKE_BUFFER), true);
  assertEquals(writes.length, 1);

  s.advance(1000);
  assertEquals(s.flush(device, FAKE_BUFFER), false);
  assertEquals(writes.length, 1);
});

Deno.test("crossfade: reset uploads the whole live range", () => {
  const { device, writes } = recordingDevice();
  const s = scheduler(4);
  s.reset();

  assertEquals(s.flush(device, FAKE_BUFFER), true);
  assertEquals(writes[0].offsetBytes, 0);
  assertEquals(writes[0].values, [1, 1, 1, 1]);
});

Deno.test("crossfade: a capacity change alone does not schedule an upload", () => {
  const { device, writes } = recordingDevice();
  const s = scheduler(4);

  // The caller has just (re)created the GPU buffer fully opaque; re-uploading
  // the same value would duplicate that write.
  s.resize(64);
  assertEquals(s.flush(device, FAKE_BUFFER), false);
  assertEquals(writes.length, 0);
});

Deno.test("crossfade: shrinking clamps a pending upload to the live range", () => {
  const { device, writes } = recordingDevice();
  const s = scheduler(8);
  s.fadeOut([1, 6], 0, 0);

  s.resize(4);
  assertEquals(s.flush(device, FAKE_BUFFER), true);
  assertEquals(writes[0].offsetBytes, 1 * 4);
  assertEquals(writes[0].values.length, 3, "upload ran past the live range");
});
