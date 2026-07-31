/**
 * Force configuration validation: clamping, snapping, and cross-parameter
 * constraints. These bounds are what keeps GPU uniforms in shader-safe
 * ranges, so they are pinned exactly.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  DEFAULT_FORCE_CONFIG,
  forceConfigBuilder,
  mergeForceConfig,
  validateForceConfig,
} from "../../packages/core/src/simulation/config.ts";

Deno.test("validateForceConfig: empty input returns defaults, unmutated", () => {
  const result = validateForceConfig({});
  assertEquals(result, DEFAULT_FORCE_CONFIG);

  // Validation must not mutate its input
  const input = { theta: 99 };
  validateForceConfig(input);
  assertEquals(input.theta, 99);
});

Deno.test("validateForceConfig: clamps core force parameters", () => {
  const result = validateForceConfig({
    theta: 99,
    repulsionDistanceMax: -5,
    repulsionDistanceMin: 0,
    springStrength: -1,
    springLength: 0,
    velocityDecay: 3,
    maxVelocity: 0,
    collisionStrength: 2,
    collisionRadiusMultiplier: 0,
    collisionIterations: 2.9,
  });

  assertEquals(result.theta, 2.0);
  assertEquals(result.repulsionDistanceMax, 0); // 0 = no limit
  assertEquals(result.repulsionDistanceMin, 0.1);
  assertEquals(result.springStrength, 0);
  assertEquals(result.springLength, 1);
  assertEquals(result.velocityDecay, 1);
  assertEquals(result.maxVelocity, 1);
  assertEquals(result.collisionStrength, 1);
  assertEquals(result.collisionRadiusMultiplier, 0.1);
  assertEquals(result.collisionIterations, 2); // floored

  assertEquals(validateForceConfig({ theta: 0 }).theta, 0.1);
});

Deno.test("validateForceConfig: snaps densityGridSize to a power of two in [32, 512]", () => {
  assertEquals(validateForceConfig({ densityGridSize: 100 }).densityGridSize, 128);
  assertEquals(validateForceConfig({ densityGridSize: 33 }).densityGridSize, 32);
  assertEquals(validateForceConfig({ densityGridSize: 20 }).densityGridSize, 32);
  assertEquals(validateForceConfig({ densityGridSize: 10_000 }).densityGridSize, 512);
  assertEquals(validateForceConfig({ densityGridSize: 128 }).densityGridSize, 128);
});

Deno.test("validateForceConfig: clamps Relativity Atlas parameters", () => {
  const result = validateForceConfig({
    relativityBaseMass: 0,
    relativityChildMassFactor: 2,
    relativityMassExponent: 5,
    relativityMaxSiblings: 3.7,
    relativityDepthDecay: 1.5,
    relativityTangentialMultiplier: 0,
    relativityOrbitRadius: 1000,
  });

  assertEquals(result.relativityBaseMass, 0.1);
  assertEquals(result.relativityChildMassFactor, 1);
  assertEquals(result.relativityMassExponent, 2);
  assertEquals(result.relativityMaxSiblings, 10);
  assertEquals(result.relativityDepthDecay, 1);
  assertEquals(result.relativityTangentialMultiplier, 1);
  assertEquals(result.relativityOrbitRadius, 200);
});

Deno.test("validateForceConfig: enforces the t-FDP paper constraint alpha*(1+beta) < 1", () => {
  // Defaults already satisfy it and must pass through untouched
  const defaults = validateForceConfig({});
  assertEquals(defaults.tFdpAlpha, 0.1);
  assertEquals(defaults.tFdpBeta, 8.0);

  // Violating combination: beta is reduced until the constraint holds
  const adjusted = validateForceConfig({ tFdpAlpha: 0.2, tFdpBeta: 8 });
  assertEquals(adjusted.tFdpAlpha, 0.2);
  assert(
    adjusted.tFdpAlpha * (1 + adjusted.tFdpBeta) < 1,
    `constraint violated: ${adjusted.tFdpAlpha} * (1 + ${adjusted.tFdpBeta}) >= 1`,
  );

  // Gamma clamped to >= 1 (shader pow() stability)
  assertEquals(validateForceConfig({ tFdpGamma: 0.2 }).tFdpGamma, 1.0);
  assertEquals(validateForceConfig({ tFdpGamma: 9 }).tFdpGamma, 5.0);
});

Deno.test("mergeForceConfig: overlays partials onto defaults", () => {
  const merged = mergeForceConfig({ springLength: 55 });
  assertEquals(merged.springLength, 55);
  assertEquals(merged.springStrength, DEFAULT_FORCE_CONFIG.springStrength);
});

Deno.test("forceConfigBuilder: builds an independent config", () => {
  const config = forceConfigBuilder()
    .repulsion(-80, 2000, 2)
    .spring(0.2, 40)
    .center(0.05, 10, -10)
    .theta(1.1)
    .decay(0.3)
    .build();

  assertEquals(config.repulsionStrength, -80);
  assertEquals(config.repulsionDistanceMax, 2000);
  assertEquals(config.repulsionDistanceMin, 2);
  assertEquals(config.springStrength, 0.2);
  assertEquals(config.springLength, 40);
  assertEquals(config.centerStrength, 0.05);
  assertEquals(config.centerX, 10);
  assertEquals(config.centerY, -10);
  assertEquals(config.theta, 1.1);
  assertEquals(config.velocityDecay, 0.3);

  // build() returns a copy — later builder use cannot mutate it
  assertEquals(DEFAULT_FORCE_CONFIG.theta, 0.8);
});

Deno.test("bubble mode implies phantom zones", () => {
  // The wellRadius bubble mode computes is only read behind the shaders'
  // phantom_enabled gate, so the two flags must not be independently settable.
  assert(validateForceConfig({ relativityBubbleMode: true }).relativityPhantomZone);
  assert(mergeForceConfig({ relativityBubbleMode: true }).relativityPhantomZone);

  // ...and the coupling is one-way: phantom zones stand alone.
  assertEquals(
    validateForceConfig({ relativityPhantomZone: true }).relativityBubbleMode,
    false,
  );
  assertEquals(validateForceConfig({}).relativityPhantomZone, false);
});
