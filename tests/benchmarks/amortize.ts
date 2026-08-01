/**
 * Await-amortized GPU timing.
 *
 * `queue.onSubmittedWorkDone()` never resolves under Deno, so the test
 * harness fences with a `mapAsync` round-trip instead ({@link waitForQueue}),
 * and that round-trip costs ~13-14 ms no matter how much work it waits on.
 * Timing one GPU operation per await therefore measures the fence rather than
 * the operation: measured on this machine, a 2 500-node simulation tick reads
 * 14.7 ms and a 35 000-node one reads 29.4 ms — a 17x spread compressed to
 * 2x, which is not enough resolution to notice a large regression.
 *
 * The fix is to submit K operations per await and divide by K. K is
 * calibrated per workload so one batch lands near `targetBatchMs`, which
 * bounds the fence's share of the reported per-unit figure; that bound is
 * returned as {@link PerUnitTiming.floorShare} — an upper bound on how much
 * the reported number overstates the true per-unit cost.
 *
 * @module
 */

import { waitForQueue } from "../helpers/gpu.ts";

/** One amortized measurement of a repeatable GPU operation. */
export interface PerUnitTiming {
  /** Operations submitted per awaited batch (K). */
  readonly batchUnits: number;
  /** Mean wall-clock time of one batch, ms. */
  readonly batchMs: number;
  /** Mean wall-clock time of one operation, ms. */
  readonly perUnitMs: number;
  /** Fastest per-operation time across the sampled batches, ms. */
  readonly bestPerUnitMs: number;
  /** Upper bound on the fraction of `perUnitMs` that is residual fence cost. */
  readonly floorShare: number;
  /** Batches timed after calibration and warm-up. */
  readonly samples: number;
}

/** Tuning for {@link measureAmortized}. */
export interface AmortizeOptions {
  /** Wall-clock the calibrator aims a batch at, ms (default 400). */
  readonly targetBatchMs?: number;
  /** Ceiling on operations per batch (default 2048). */
  readonly maxBatchUnits?: number;
  /** Batches timed after warm-up (default 5). */
  readonly samples?: number;
}

/** Batch size the first calibration round aims at, ms. */
const CALIBRATION_BATCH_MS = 120;
const DEFAULT_TARGET_BATCH_MS = 400;
const DEFAULT_MAX_BATCH_UNITS = 2048;
const DEFAULT_SAMPLES = 5;

/**
 * Floor on the calibrator's per-unit estimate, so an operation that costs
 * less than the timer can resolve still yields a finite batch size.
 */
const MIN_UNIT_MS = 1e-3;

/**
 * Measures the fence cost that {@link measureAmortized} amortizes away:
 * the median wall-clock of {@link waitForQueue} on an idle queue.
 */
export async function measureAwaitFloorMs(device: GPUDevice, samples = 16): Promise<number> {
  await waitForQueue(device);
  const times: number[] = [];
  for (let i = 0; i < samples; i++) {
    const start = performance.now();
    await waitForQueue(device);
    times.push(performance.now() - start);
  }
  times.sort((a, b) => a - b);
  return times[times.length >> 1];
}

/**
 * Times one repeatable GPU operation by submitting K of them per await.
 *
 * `runBatch(units)` must submit exactly `units` operations and await the
 * queue exactly once (`SimHarness.tick(units)` is one such batch).
 * `awaitFloorMs` comes from {@link measureAwaitFloorMs}; it is used only to
 * pick K, never to adjust the reported result.
 */
export async function measureAmortized(
  runBatch: (units: number) => Promise<void>,
  awaitFloorMs: number,
  options: AmortizeOptions = {},
): Promise<PerUnitTiming> {
  const targetBatchMs = options.targetBatchMs ?? DEFAULT_TARGET_BATCH_MS;
  const maxBatchUnits = options.maxBatchUnits ?? DEFAULT_MAX_BATCH_UNITS;
  const samples = options.samples ?? DEFAULT_SAMPLES;

  // Two calibration rounds: one unit to get an order of magnitude, then a
  // ~CALIBRATION_BATCH_MS batch to refine it. Subtracting the idle fence cost
  // over-corrects slightly (the fence partly overlaps queued work), which can
  // only make K larger than needed — never smaller, so it cannot inflate the
  // reported figure.
  let units = 1;
  for (const goalMs of [CALIBRATION_BATCH_MS, targetBatchMs]) {
    const batchMs = await timeBatch(runBatch, units);
    const perUnitMs = Math.max((batchMs - awaitFloorMs) / units, MIN_UNIT_MS);
    units = clamp(Math.ceil(goalMs / perUnitMs), 1, maxBatchUnits);
  }

  // Warm at the final batch size, then sample.
  await runBatch(units);

  let totalMs = 0;
  let bestBatchMs = Infinity;
  for (let sample = 0; sample < samples; sample++) {
    const batchMs = await timeBatch(runBatch, units);
    totalMs += batchMs;
    if (batchMs < bestBatchMs) bestBatchMs = batchMs;
  }

  const batchMs = totalMs / samples;
  return {
    batchUnits: units,
    batchMs,
    perUnitMs: batchMs / units,
    bestPerUnitMs: bestBatchMs / units,
    floorShare: awaitFloorMs / batchMs,
    samples,
  };
}

async function timeBatch(
  runBatch: (units: number) => Promise<void>,
  units: number,
): Promise<number> {
  const start = performance.now();
  await runBatch(units);
  return performance.now() - start;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
