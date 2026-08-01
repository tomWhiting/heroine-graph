/**
 * The code-graph example's HUD view model.
 *
 * The readout is the only evidence a viewer has that LOD is doing anything, so
 * its arithmetic has to be right in exactly the cases a screenshot cannot show:
 * a rate averaged over a window rather than one frame, a fold count that is a
 * net of two event streams and must never go negative, and a tick interval that
 * does not report the length of a pause as a slow tick.
 *
 * The model takes its clock as an argument, so all of this is exact.
 */

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import { formatCount, formatMs, HudModel } from "../../examples/mission-control/src/hud.ts";

Deno.test("hud: a load resets everything derived from the previous dataset", () => {
  const hud = new HudModel();
  hud.loaded(100, 120);
  hud.lodChanged(40);
  hud.nodeFolded();
  hud.cardsMounted(7);

  hud.loaded(2_500, 2_874);
  const snapshot = hud.snapshot();
  assertEquals(snapshot.nodes, 2_500);
  assertEquals(snapshot.edges, 2_874);
  assertEquals(snapshot.folded, 0);
  assertEquals(snapshot.cards, 0);
  // Before the cut has been evaluated, everything is on screen.
  assertEquals(snapshot.visible, 2_500);
});

Deno.test("hud: fps is published once per window, not per frame", () => {
  const hud = new HudModel();
  let now = 1_000;
  hud.frame(now); // establishes the window; no rate yet

  // 31 frames at 16ms is 496ms — inside the window, so nothing is published.
  for (let i = 0; i < 31; i++) {
    now += 16;
    hud.frame(now);
  }
  assertEquals(hud.snapshot().fps, 0);

  // The frame that crosses 500ms publishes 32 frames over 512ms.
  now += 16;
  hud.frame(now);
  assertAlmostEquals(hud.snapshot().fps, (32 * 1000) / 512, 1e-9);
});

Deno.test("hud: the fps window restarts from the publishing frame", () => {
  const hud = new HudModel();
  let now = 0;
  hud.frame(now);
  for (let i = 0; i < 60; i++) {
    now += 10;
    hud.frame(now);
  }
  const first = hud.snapshot().fps;
  assert(first > 0);

  // A second identical window must report the same rate — if the window start
  // were not reset, the divisor would keep growing and the rate would decay.
  for (let i = 0; i < 60; i++) {
    now += 10;
    hud.frame(now);
  }
  assertAlmostEquals(hud.snapshot().fps, first, 1e-9);
});

Deno.test("hud: the first tick establishes a baseline and reports nothing", () => {
  const hud = new HudModel();
  hud.tick(1_000);
  assertEquals(hud.snapshot().tickMs, 0);

  hud.tick(1_016);
  assertAlmostEquals(hud.snapshot().tickMs, 16, 1e-9);
});

Deno.test("hud: a clock that starts at zero still establishes both baselines", () => {
  // `performance.now()` is 0 at page load, so a model that used 0 as "no
  // baseline yet" would silently discard the first window and the first tick
  // pair — on exactly the frames a viewer is watching the demo start up.
  const hud = new HudModel();
  hud.frame(0);
  for (let i = 1; i <= 51; i++) hud.frame(i * 10);
  assertAlmostEquals(hud.snapshot().fps, (51 * 1000) / 510, 1e-9);

  const ticks = new HudModel();
  ticks.tick(0);
  ticks.tick(25);
  assertAlmostEquals(ticks.snapshot().tickMs, 25, 1e-9);
});

Deno.test("hud: the tick interval is smoothed toward the true rate", () => {
  const hud = new HudModel();
  let now = 0;
  for (let i = 0; i < 200; i++) {
    now += 20;
    hud.tick(now);
  }
  assertAlmostEquals(hud.snapshot().tickMs, 20, 1e-6);

  // A single long frame must move the readout without dominating it.
  hud.tick(now + 500);
  const after = hud.snapshot().tickMs;
  assert(after > 20 && after < 200, `one spike moved the readout to ${after}`);
});

Deno.test("hud: a pause does not surface as one enormous tick", () => {
  const hud = new HudModel();
  hud.tick(0);
  hud.tick(20);
  const before = hud.snapshot().tickMs;

  hud.simulationStopped();
  // Ten seconds later the user presses resume.
  hud.tick(10_000);
  assertEquals(hud.snapshot().tickMs, before, "the pause leaked into the interval");

  hud.tick(10_020);
  assertAlmostEquals(hud.snapshot().tickMs, before, 1e-9);
});

Deno.test("hud: the fold count is a net that never goes negative", () => {
  const hud = new HudModel();
  hud.loaded(100, 100);

  hud.nodeFolded();
  hud.nodeFolded();
  hud.nodeFolded();
  assertEquals(hud.snapshot().folded, 3);

  hud.nodeUnfolded();
  assertEquals(hud.snapshot().folded, 2);

  // Core expands nodes that were never counted as folded — anything hidden by
  // the cut at load, for one — and a negative readout would be nonsense.
  for (let i = 0; i < 10; i++) hud.nodeUnfolded();
  assertEquals(hud.snapshot().folded, 0);
});

Deno.test("hud: lod:change drives the visible count", () => {
  const hud = new HudModel();
  hud.loaded(35_000, 40_000);
  hud.lodChanged(1_204);
  assertEquals(hud.snapshot().visible, 1_204);
  hud.lodChanged(0);
  assertEquals(hud.snapshot().visible, 0);
});

Deno.test("hud: counts are formatted at a fixed width per magnitude", () => {
  assertEquals(formatCount(0), "0");
  assertEquals(formatCount(999), "999");
  assertEquals(formatCount(1_000), "1.0K");
  assertEquals(formatCount(12_400), "12.4K");
  assertEquals(formatCount(999_999), "1000.0K");
  assertEquals(formatCount(1_000_000), "1.00M");
  assertEquals(formatCount(220_000), "220.0K");
  assertEquals(formatCount(2_499.6), "2.5K");
});

Deno.test("hud: durations keep a decimal only where it is legible", () => {
  assertEquals(formatMs(0), "—");
  assertEquals(formatMs(-1), "—");
  assertEquals(formatMs(4.26), "4.3ms");
  assertEquals(formatMs(9.99), "10.0ms");
  assertEquals(formatMs(16.4), "16ms");
  assertEquals(formatMs(1_500), "1500ms");
});
