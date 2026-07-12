#!/usr/bin/env -S deno run --allow-run --allow-read --allow-write --allow-env
/**
 * Generates the npm .d.ts output for @graphmother/core.
 *
 * - Runs the TypeScript compiler resolved from node_modules (installed by
 *   `deno install`, pinned by deno.lock) — never an unpinned "latest".
 * - Prepends a `/// <reference types="@webgpu/types" />` directive to the
 *   entry declaration file so npm consumers get the GPU* globals the public
 *   API depends on without needing skipLibCheck or manual wiring
 *   (@webgpu/types is a runtime dependency of the package).
 *
 * Run standalone or via scripts/publish.ts. Requires `deno install` to have
 * been run at the repo root first.
 */

import { join } from "jsr:@std/path";

const ROOT = new URL("..", import.meta.url).pathname;
const CORE_DIR = join(ROOT, "packages/core");
const ENTRY_DTS = join(CORE_DIR, "dist/mod.d.ts");
const WEBGPU_TYPES_REFERENCE = '/// <reference types="@webgpu/types" />\n';

export async function buildCoreTypes(): Promise<void> {
  // tsc is resolved from node_modules (deno.lock-pinned via `deno install`).
  const command = new Deno.Command("deno", {
    // Constraint matches packages/core devDependencies; deno.lock pins the
    // exact version. A versionless specifier would drift to latest (TS 7
    // removes baseUrl and breaks tsconfig.npm.json) if the lock entry
    // ever disappeared.
    args: ["run", "-A", "npm:typescript@^5.7.0/tsc", "--project", "tsconfig.npm.json"],
    cwd: CORE_DIR,
    stdout: "inherit",
    stderr: "inherit",
  });
  const { code } = await command.output();
  if (code !== 0) {
    throw new Error(`tsc failed (exit ${code})`);
  }

  // Make the published declarations self-sufficient for npm consumers.
  const dts = await Deno.readTextFile(ENTRY_DTS);
  if (!dts.startsWith(WEBGPU_TYPES_REFERENCE)) {
    await Deno.writeTextFile(ENTRY_DTS, WEBGPU_TYPES_REFERENCE + dts);
  }
  console.log(`  Types emitted: ${ENTRY_DTS} (with @webgpu/types reference)`);
}

if (import.meta.main) {
  try {
    await buildCoreTypes();
  } catch (error) {
    console.error("[ERROR]", error);
    Deno.exit(1);
  }
}
