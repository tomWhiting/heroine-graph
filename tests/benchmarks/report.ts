/**
 * Benchmark table formatting.
 *
 * Renders {@link PerUnitTiming} results as a fixed-width table with the
 * amortization parameters visible (batch size, residual fence share) so a
 * reader can tell a real measurement from one still sitting on the await
 * floor, and — where a reference figure is supplied — the deviation from it.
 *
 * @module
 */

import type { PerUnitTiming } from "./amortize.ts";

/** One measured workload, ready to print. */
export interface BenchRow {
  /** What was measured, e.g. "sim tick, N²: 35,000 nodes". */
  readonly workload: string;
  readonly timing: PerUnitTiming;
  /**
   * Expected ms/unit for this workload, for the delta column. These are the
   * Phase 3 design's measured envelope (Apple M1 Max, Deno 2.9.4 headless
   * WebGPU); the bench prints the deviation and asserts nothing, since the
   * figures are machine-specific.
   */
  readonly referenceMs?: number;
}

const HEADERS = [
  "workload",
  "ms/unit",
  "best",
  "batch K",
  "batch ms",
  "fence ≤",
  "reference",
  "Δ",
] as const;

/** Columns after the workload name are right-aligned. */
const FIRST_RIGHT_ALIGNED_COLUMN = 1;

/**
 * Formats rows as a fixed-width table (no trailing newline).
 */
export function formatBenchTable(rows: readonly BenchRow[]): string {
  const body = rows.map((row) => {
    const { timing, referenceMs } = row;
    return [
      row.workload,
      timing.perUnitMs.toFixed(3),
      timing.bestPerUnitMs.toFixed(3),
      String(timing.batchUnits),
      timing.batchMs.toFixed(1),
      `${(timing.floorShare * 100).toFixed(1)}%`,
      referenceMs === undefined ? "—" : referenceMs.toFixed(2),
      referenceMs === undefined ? "—" : formatSignedPercent(timing.perUnitMs / referenceMs - 1),
    ];
  });

  const widths = HEADERS.map((header, column) =>
    body.reduce((width, cells) => Math.max(width, cells[column].length), header.length)
  );

  const line = (cells: readonly string[]): string =>
    cells
      .map((cell, column) =>
        column < FIRST_RIGHT_ALIGNED_COLUMN
          ? cell.padEnd(widths[column])
          : cell.padStart(widths[column])
      )
      .join("  ")
      .trimEnd();

  return [
    line(HEADERS),
    line(widths.map((width) => "-".repeat(width))),
    ...body.map(line),
  ].join("\n");
}

function formatSignedPercent(fraction: number): string {
  const percent = fraction * 100;
  return `${percent >= 0 ? "+" : ""}${percent.toFixed(1)}%`;
}
