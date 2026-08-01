/**
 * The code-graph example's knob-to-config mapping.
 *
 * Five controls stand between the user and the library, and each one has to
 * produce a configuration core will accept without clamping it back. The two
 * properties worth pinning are the ones a browser would not show: that the
 * detail slider runs the right way round (right means more detail, which is a
 * *lower* expand threshold), and that the hysteresis band survives every
 * position of it — a band that inverts is one core silently repairs, and a
 * repaired band is not the band the demo claims to demonstrate.
 */

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import { DEFAULT_LOD_CONFIG, resolveLodConfig } from "../../packages/core/src/lod/config.ts";
import {
  DEFAULT_KNOBS,
  KNOB_RANGES,
  type KnobState,
  LABEL_CANDIDATE_CEILING,
  LABEL_CANDIDATE_RATIO,
  labelCandidateLimit,
  labelConfigFromKnobs,
  lodConfigFromKnobs,
} from "../../examples/code-graph/src/knobs.ts";

function knobs(patch: Partial<KnobState> = {}): KnobState {
  return { ...DEFAULT_KNOBS, ...patch };
}

Deno.test("knobs: the detail slider raises detail to the right", () => {
  const coarse = lodConfigFromKnobs(knobs({ detail: 0 }));
  const middle = lodConfigFromKnobs(knobs({ detail: 0.5 }));
  const fine = lodConfigFromKnobs(knobs({ detail: 1 }));

  const expand = (config: { expandThreshold?: number }) => config.expandThreshold ?? NaN;
  assert(
    expand(coarse) > expand(middle) && expand(middle) > expand(fine),
    `thresholds are not monotonic: ${expand(coarse)}, ${expand(middle)}, ${expand(fine)}`,
  );
  // The fine end must still unfold on something bigger than a single sprite,
  // and the coarse end must not be so high that nothing ever unfolds.
  assert(expand(fine) >= 16, "the fine end asks for sub-sprite subtrees");
  assert(expand(coarse) <= 1_000, "the coarse end never unfolds anything");
});

Deno.test("knobs: the hysteresis band survives every slider position", () => {
  for (let step = 0; step <= 20; step++) {
    const detail = step / 20;
    const patch = lodConfigFromKnobs(knobs({ detail }));
    const expand = patch.expandThreshold ?? NaN;
    const collapse = patch.collapseThreshold ?? NaN;

    assert(collapse < expand, `band inverted at detail ${detail}`);
    // `resolveLodConfig` clamps an inverted or degenerate band; if it changed
    // anything, the demo is not driving the band it claims to.
    const resolved = resolveLodConfig(patch);
    assertAlmostEquals(resolved.expandThreshold, expand, 1e-6, `expand at detail ${detail}`);
    assertAlmostEquals(resolved.collapseThreshold, collapse, 1e-6, `collapse at detail ${detail}`);
    assert(
      collapse / expand < 0.9,
      `band at detail ${detail} is too narrow to damp anything (${collapse / expand})`,
    );
  }
});

Deno.test("knobs: a detail value outside the slider range is clamped, not extrapolated", () => {
  const low = lodConfigFromKnobs(knobs({ detail: -5 }));
  const high = lodConfigFromKnobs(knobs({ detail: 5 }));
  assertEquals(low.expandThreshold, lodConfigFromKnobs(knobs({ detail: 0 })).expandThreshold);
  assertEquals(high.expandThreshold, lodConfigFromKnobs(knobs({ detail: 1 })).expandThreshold);

  const nonsense = lodConfigFromKnobs(knobs({ detail: Number.NaN }));
  assertEquals(nonsense.expandThreshold, low.expandThreshold);
});

Deno.test("knobs: the enable switch and the card budget pass straight through", () => {
  assertEquals(lodConfigFromKnobs(knobs({ lodEnabled: false })).enabled, false);
  assertEquals(lodConfigFromKnobs(knobs({ lodEnabled: true })).enabled, true);

  assertEquals(lodConfigFromKnobs(knobs({ cardBudget: 42 })).maxCards, 42);
  assertEquals(lodConfigFromKnobs(knobs({ cardBudget: 0 })).maxCards, 0);
  assertEquals(lodConfigFromKnobs(knobs({ cardBudget: -10 })).maxCards, 0);
  assertEquals(lodConfigFromKnobs(knobs({ cardBudget: 12.7 })).maxCards, 12);
});

Deno.test("knobs: the whole slider range produces a configuration core keeps", () => {
  const { cardBudget } = KNOB_RANGES;
  for (let value = cardBudget.min; value <= cardBudget.max; value += cardBudget.step) {
    const patch = lodConfigFromKnobs(knobs({ cardBudget: value }));
    assertEquals(resolveLodConfig(patch).maxCards, value);
  }
  // Everything the demo does not set keeps core's default.
  const resolved = resolveLodConfig(lodConfigFromKnobs(DEFAULT_KNOBS));
  assertEquals(resolved.domThreshold, DEFAULT_LOD_CONFIG.domThreshold);
  assertEquals(resolved.minBandCommitFrames, DEFAULT_LOD_CONFIG.minBandCommitFrames);
  assertEquals(resolved.edgeAggregation, DEFAULT_LOD_CONFIG.edgeAggregation);
});

Deno.test("knobs: label density maps to the layer's ceiling", () => {
  assertEquals(labelConfigFromKnobs(knobs({ labelDensity: 300 })).maxLabels, 300);
  assertEquals(labelConfigFromKnobs(knobs({ labelDensity: 0 })).maxLabels, 0);
  assertEquals(labelConfigFromKnobs(knobs({ labelDensity: -1 })).maxLabels, 0);
});

Deno.test("knobs: the candidate set stays a bounded multiple of what is drawn", () => {
  assertEquals(labelCandidateLimit(knobs({ labelDensity: 0 })), 0);
  assertEquals(
    labelCandidateLimit(knobs({ labelDensity: 100 })),
    100 * LABEL_CANDIDATE_RATIO,
  );
  // The per-frame ranking cost is what the ceiling exists to bound, so it must
  // bite before the ratio can run away with a large density.
  assertEquals(
    labelCandidateLimit(knobs({ labelDensity: KNOB_RANGES.labelDensity.max })),
    LABEL_CANDIDATE_CEILING,
  );
  assert(
    KNOB_RANGES.labelDensity.max * LABEL_CANDIDATE_RATIO > LABEL_CANDIDATE_CEILING,
    "the ceiling is unreachable, so it bounds nothing",
  );
  assert(labelCandidateLimit(knobs({ labelDensity: 100 })) >= 100, "fewer candidates than labels");
});

Deno.test("knobs: the demo starts with LOD on, running, at the default scale", () => {
  assertEquals(DEFAULT_KNOBS.lodEnabled, true);
  assertEquals(DEFAULT_KNOBS.running, true);
  assertEquals(DEFAULT_KNOBS.scale, "small");
  assert(DEFAULT_KNOBS.cardBudget > 0, "cards off by default would hide the feature");
  assert(DEFAULT_KNOBS.labelDensity > 0, "labels off by default would hide the feature");
});
