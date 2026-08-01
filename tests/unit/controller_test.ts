/**
 * Simulation controller unit tests: alpha cooling schedule.
 *
 * Guards the fix for the alphaDecay default: the documented contract is
 * "cools to rest after ~300 ticks" (d3-force convention), while the old
 * default (0.0002) kept the simulation at near-full temperature for
 * ~34,500 ticks (~10 minutes at 60fps). The long settle is still available
 * as an explicit preset.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  calculateAlphaDecay,
  createSimulationController,
  DEFAULT_SIMULATION_CONFIG,
  LONG_SETTLE_SIMULATION_CONFIG,
} from "../../packages/core/src/simulation/controller.ts";

/** Ticks until alpha snaps to 0 (never: returns limit). */
function ticksToRest(alphaDecay: number, limit: number): number {
  const controller = createSimulationController({ alphaDecay });
  controller.start();
  for (let t = 1; t <= limit; t++) {
    controller.tick();
    if (controller.state.alpha === 0) return t;
  }
  return limit;
}

Deno.test("default alphaDecay cools the simulation to rest in ~300 ticks", () => {
  const ticks = ticksToRest(DEFAULT_SIMULATION_CONFIG.alphaDecay, 1000);
  assert(
    ticks >= 250 && ticks <= 350,
    `expected rest after ~300 ticks (d3 convention), got ${ticks}`,
  );
});

Deno.test("default alphaDecay matches the d3-force ~300-iteration convention", () => {
  // decay = 1 - alphaMin^(1/300) ~= 0.0228
  const d3Decay = calculateAlphaDecay(300);
  const relError = Math.abs(DEFAULT_SIMULATION_CONFIG.alphaDecay - d3Decay) / d3Decay;
  assert(
    relError < 0.05,
    `default decay ${DEFAULT_SIMULATION_CONFIG.alphaDecay} is not ~${d3Decay.toFixed(4)}`,
  );
});

Deno.test("long-settle preset keeps the old ~34,500-tick schedule", () => {
  assertEquals(LONG_SETTLE_SIMULATION_CONFIG.alphaDecay, 0.0002);
  // After 300 ticks the long-settle schedule must still be nearly hot —
  // that is the point of the preset.
  const controller = createSimulationController(LONG_SETTLE_SIMULATION_CONFIG);
  controller.start();
  for (let t = 0; t < 300; t++) controller.tick();
  assert(
    controller.state.alpha > 0.9,
    `long-settle cooled too fast: alpha=${controller.state.alpha} after 300 ticks`,
  );
});

Deno.test("alpha snaps to exactly 0 below alphaMin so equilibrium jitter dies", () => {
  const controller = createSimulationController();
  controller.start();
  for (let t = 0; t < 1000; t++) controller.tick();
  assertEquals(controller.state.alpha, 0);
});

/**
 * "end" announces a settle exactly once per cool-down.
 *
 * The tick loop deliberately never stops (alpha only controls movement), so
 * settling is an event, not a state transition: it must fire on the tick that
 * snaps alpha to 0, not fire again on the next thousand idle ticks, and be
 * re-armed by everything that reheats — restart, and alpha pushed back above
 * alphaMin. Downstream "fit the camera when the layout settles" logic depends
 * on all three properties.
 */
Deno.test("end fires once per settle and re-arms on reheat", () => {
  const controller = createSimulationController();
  const ends: number[] = [];
  controller.events.on("end", ({ tickCount }) => ends.push(tickCount));

  controller.start();
  for (let t = 0; t < 1000; t++) controller.tick();
  assertEquals(ends.length, 1, "one settle, one end");
  assertEquals(controller.state.alpha, 0);
  assert(
    ends[0] < 1000,
    "end must fire on the settling tick, not after the loop",
  );

  controller.restart();
  for (let t = 0; t < 1000; t++) controller.tick();
  assertEquals(ends.length, 2, "restart re-arms the settle event");

  controller.setAlpha(0.5);
  for (let t = 0; t < 1000; t++) controller.tick();
  assertEquals(ends.length, 3, "setAlpha above alphaMin re-arms the settle event");

  controller.setAlpha(0.0001);
  for (let t = 0; t < 1000; t++) controller.tick();
  assertEquals(ends.length, 3, "setAlpha below alphaMin must not re-arm");
});
