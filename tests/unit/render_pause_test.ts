/**
 * Host-driven render pause.
 *
 * A host measured 180 presented frames in 1.5s while the graph canvas was
 * occluded — covered by other UI, page still visible — against 107 while it was
 * on screen. Nothing in the engine can see that: `document.hidden` is false and
 * requestAnimationFrame keeps firing, so the pause has to be something the host
 * drives. The two axes (host intent, page visibility) are held apart so a tab
 * switch cannot lift a pause the host asked for.
 *
 * The gate is exercised directly and the loop through a fake frame clock; the
 * `GraphMother` methods on top of them are three delegating one-liners over a
 * constructor that needs a real GPU device, so the wiring that a unit test can
 * still pin — that the visibility handler goes through the gate rather than
 * touching the loop — is asserted against the source, as `event_coverage_test`
 * does for emit sites.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import { createRenderLoop } from "../../packages/core/src/renderer/render_loop.ts";
import { RenderPauseGate } from "../../packages/core/src/renderer/render_pause.ts";

// -----------------------------------------------------------------------------
// Harness
// -----------------------------------------------------------------------------

/** A gate target that records the transitions it was driven through. */
function recorder(): { calls: string[]; pause: () => void; resume: () => void } {
  const calls: string[] = [];
  return {
    calls,
    pause: () => calls.push("pause"),
    resume: () => calls.push("resume"),
  };
}

interface FrameClock {
  /** Run every callback queued right now, timestamped from the real clock. */
  flush: () => void;
  /** Callbacks waiting on a frame. */
  readonly pending: number;
  restore: () => void;
}

/**
 * Replaces the animation-frame globals with a manually pumped queue.
 *
 * Restoration is exact — `hover_test` asserts requestAnimationFrame is absent
 * in this runtime, so leaving a stub behind would break an unrelated file.
 */
function fakeFrames(): FrameClock {
  const globals = globalThis as {
    requestAnimationFrame?: typeof requestAnimationFrame;
    cancelAnimationFrame?: typeof cancelAnimationFrame;
  };
  const priorRequest = globals.requestAnimationFrame;
  const priorCancel = globals.cancelAnimationFrame;

  const queued = new Map<number, FrameRequestCallback>();
  let nextId = 1;

  globals.requestAnimationFrame = (callback: FrameRequestCallback): number => {
    const id = nextId++;
    queued.set(id, callback);
    return id;
  };
  globals.cancelAnimationFrame = (id: number): void => {
    queued.delete(id);
  };

  return {
    flush(): void {
      // Snapshot first: a loop callback schedules its successor, which belongs
      // to the next flush, not this one.
      const due = [...queued.values()];
      queued.clear();
      for (const callback of due) callback(performance.now());
    },
    get pending(): number {
      return queued.size;
    },
    restore(): void {
      if (priorRequest) globals.requestAnimationFrame = priorRequest;
      else delete globals.requestAnimationFrame;
      if (priorCancel) globals.cancelAnimationFrame = priorCancel;
      else delete globals.cancelAnimationFrame;
    },
  };
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// -----------------------------------------------------------------------------
// Gate arbitration
// -----------------------------------------------------------------------------

Deno.test("render pause: repeat pauses and resumes drive one transition each", () => {
  const target = recorder();
  const gate = new RenderPauseGate(target);

  assertEquals(gate.isPaused, false, "a fresh gate presents");

  gate.pauseByHost();
  gate.pauseByHost();
  assertEquals(target.calls, ["pause"], "pausing twice suspends once");
  assertEquals(gate.isPaused, true);
  assertEquals(gate.isPausedByHost, true);

  gate.resumeByHost();
  gate.resumeByHost();
  assertEquals(target.calls, ["pause", "resume"], "resuming twice resumes once");
  assertEquals(gate.isPaused, false);
  assertEquals(gate.isPausedByHost, false);
});

Deno.test("render pause: a visibility change never lifts a host pause", () => {
  const target = recorder();
  const gate = new RenderPauseGate(target);

  gate.pauseByHost();
  gate.setHidden(true);
  gate.setHidden(false);

  assertEquals(gate.isPaused, true, "the host asked for this pause and still holds it");
  assertEquals(target.calls, ["pause"], "coming back to a visible tab presented nothing");
});

Deno.test("render pause: visibility drives the loop when the host has not paused", () => {
  const target = recorder();
  const gate = new RenderPauseGate(target);

  gate.setHidden(true);
  assertEquals(gate.isPaused, true);
  assertEquals(gate.isPausedByHost, false, "a hidden page is not host intent");

  gate.setHidden(false);
  assertEquals(gate.isPaused, false);
  assertEquals(target.calls, ["pause", "resume"]);
});

Deno.test("render pause: a host resume while hidden leaves frames suspended", () => {
  const target = recorder();
  const gate = new RenderPauseGate(target);

  gate.pauseByHost();
  gate.setHidden(true);
  gate.resumeByHost();

  assertEquals(gate.isPaused, true, "the page is still hidden");
  assertEquals(target.calls, ["pause"]);

  gate.setHidden(false);
  assertEquals(gate.isPaused, false, "the last holder released");
  assertEquals(target.calls, ["pause", "resume"]);
});

// -----------------------------------------------------------------------------
// Render loop
// -----------------------------------------------------------------------------

Deno.test("render loop: pause stops presenting and resume draws the next frame", () => {
  const frames = fakeFrames();
  try {
    let rendered = 0;
    const loop = createRenderLoop(() => rendered++);

    loop.start();
    frames.flush(); // seeds the timing origin
    frames.flush();
    assertEquals(rendered, 1);

    loop.pause();
    assertEquals(loop.isPaused, true);
    assertEquals(loop.isRunning, true, "pausing is not stopping");
    assertEquals(frames.pending, 0, "the frame in flight was cancelled");

    frames.flush();
    assertEquals(rendered, 1, "nothing presents while paused");

    loop.resume();
    frames.flush();
    assertEquals(rendered, 2, "the first frame back draws rather than reseeding timing");
    assertEquals(loop.isPaused, false);
  } finally {
    frames.restore();
  }
});

Deno.test("render loop: pausing and resuming twice is idempotent", () => {
  const frames = fakeFrames();
  try {
    let rendered = 0;
    const loop = createRenderLoop(() => rendered++);
    loop.start();
    frames.flush();

    loop.pause();
    loop.pause();
    assertEquals(frames.pending, 0);

    loop.resume();
    loop.resume();
    assertEquals(frames.pending, 1, "a second resume must not queue a second loop");

    frames.flush();
    assertEquals(rendered, 1, "one frame per flush, so only one loop is live");
    frames.flush();
    assertEquals(rendered, 2);
  } finally {
    frames.restore();
  }
});

Deno.test("render loop: a load that starts a paused loop leaves it paused", () => {
  const frames = fakeFrames();
  try {
    let rendered = 0;
    const loop = createRenderLoop(() => rendered++);

    // The host pauses before any graph is loaded; `load` then starts the loop.
    loop.pause();
    loop.start();

    assertEquals(loop.isRunning, true);
    assertEquals(frames.pending, 0, "starting while paused arms nothing");
    frames.flush();
    assertEquals(rendered, 0);

    loop.resume();
    frames.flush(); // seeds timing, since no frame ever ran
    frames.flush();
    assertEquals(rendered, 1, "the host's resume is what starts presentation");
  } finally {
    frames.restore();
  }
});

Deno.test("render loop: the pause latch survives a stop and start", () => {
  const frames = fakeFrames();
  try {
    let rendered = 0;
    const loop = createRenderLoop(() => rendered++);
    loop.start();
    frames.flush();
    loop.pause();

    // dispose/device-loss stop, then a fresh load starting the loop again.
    loop.stop();
    assertEquals(frames.pending, 0, "a paused loop leaves no callback behind on stop");
    loop.start();

    assertEquals(loop.isPaused, true);
    frames.flush();
    assertEquals(rendered, 0, "a restart is not a resume");
  } finally {
    frames.restore();
  }
});

Deno.test("render loop: the paused span counts as neither frame time nor elapsed", async () => {
  const frames = fakeFrames();
  try {
    let delta = 0;
    let elapsed = 0;
    const loop = createRenderLoop((deltaSeconds, stats) => {
      delta = deltaSeconds;
      elapsed = stats.elapsed;
    });

    loop.start();
    frames.flush();
    frames.flush();

    loop.pause();
    await delay(60);
    loop.resume();
    frames.flush();

    assert(delta < 0.05, `first frame back saw a ${delta}s delta, not a paused-span jump`);
    assert(elapsed < 30, `elapsed was ${elapsed}ms, so the 60ms suspension leaked into it`);
  } finally {
    frames.restore();
  }
});

Deno.test("render loop: requestFrame still draws a single frame while paused", () => {
  const frames = fakeFrames();
  try {
    let rendered = 0;
    const loop = createRenderLoop(() => rendered++);
    loop.start();
    frames.flush();
    loop.pause();

    loop.requestFrame();
    assertEquals(rendered, 1, "a paused host can still ask for one frame");
    assertEquals(frames.pending, 0, "and it does not restart the loop");
  } finally {
    frames.restore();
  }
});

// -----------------------------------------------------------------------------
// Wiring
// -----------------------------------------------------------------------------

Deno.test("render pause: the visibility handler goes through the gate", async () => {
  const source = await Deno.readTextFile(
    new URL("../../packages/core/src/api/graph.ts", import.meta.url),
  );
  const handler = source.slice(
    source.indexOf("private setupVisibilityChangeHandler()"),
    source.indexOf('document.addEventListener("visibilitychange"'),
  );
  assert(handler.length > 0, "setupVisibilityChangeHandler not found");

  assert(
    handler.includes("this.renderPause.setHidden(document.hidden)"),
    "visibility must arbitrate through the gate",
  );
  assert(
    !handler.includes("this.renderLoop."),
    "driving the loop directly would let a tab switch lift a host pause",
  );
});
