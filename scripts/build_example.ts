#!/usr/bin/env -S deno run --allow-run --allow-read --allow-write --allow-env
/**
 * Builds examples/code-graph into a static directory.
 *
 * There is no dev server. `deno task example:code-graph` runs this once and
 * then serves the output with @std/http/file-server, so what you look at is
 * the built artefact and nothing watches or rewrites it. Edit a source file
 * and run the task again.
 *
 * esbuild rather than a bundler config, because the example needs exactly
 * three things beyond plain ESM bundling, all of them flags: WGSL as text,
 * `@graphmother/wasm` pointed at the wasm-pack output, and the font atlas
 * copied in beside the page.
 */

import { copy, emptyDir } from "jsr:@std/fs@^1";
import { join } from "jsr:@std/path@^1";

const ROOT = new URL("..", import.meta.url).pathname;
const EXAMPLE = join(ROOT, "examples/code-graph");
const OUT = join(EXAMPLE, "dist");

// The wasm-pack output, NOT the bundled copy under the repo's dist/: that copy
// is refreshed only by `deno task bundle`, so pointing at it would make
// `deno task build:wasm` look like it had no effect and leave the example
// running a stale binary missing newly exported functions.
const WASM_DIR = join(ROOT, "packages/wasm/pkg");

async function run(cmd: string[], cwd: string, label: string): Promise<void> {
  const command = new Deno.Command(cmd[0], {
    args: cmd.slice(1),
    cwd,
    stdout: "inherit",
    stderr: "inherit",
  });
  const { code } = await command.output();
  if (code !== 0) throw new Error(`${label} failed (exit ${code})`);
}

async function main(): Promise<void> {
  await emptyDir(OUT);

  await run([
    "deno",
    "run",
    "-A",
    "npm:esbuild@0.24/esbuild",
    "main.ts",
    "--bundle",
    "--format=esm",
    "--platform=browser",
    "--target=es2022",
    "--sourcemap",
    "--outfile=dist/main.js",
    "--loader:.wgsl=text",
    `--alias:@graphmother/wasm=${join(WASM_DIR, "graphmother_wasm.js")}`,
  ], EXAMPLE, "esbuild");

  // The generated glue resolves the binary as `new URL(..., import.meta.url)`,
  // which after bundling is the built module — so the .wasm has to sit beside
  // it, not where wasm-pack left it.
  await copy(
    join(WASM_DIR, "graphmother_wasm_bg.wasm"),
    join(OUT, "graphmother_wasm_bg.wasm"),
  );

  // The label atlas is requested relative to the page (see
  // DEFAULT_FONT_ATLAS_JSON in packages/core/src/layers/labels/atlas.ts).
  await copy(join(ROOT, "dist/assets"), join(OUT, "assets"));

  // The page loads TypeScript in the repo and JavaScript once built.
  const html = await Deno.readTextFile(join(EXAMPLE, "index.html"));
  await Deno.writeTextFile(join(OUT, "index.html"), html.replace("./main.ts", "./main.js"));

  console.log(`  Built ${OUT}`);
}

if (import.meta.main) {
  await main();
}
