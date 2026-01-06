#!/usr/bin/env -S deno run --allow-run --allow-read --allow-write
/**
 * Heroine Graph - Build Script
 *
 * Orchestrates the build process for all packages:
 * 1. Build WASM module
 * 2. Type-check TypeScript packages
 * 3. Build framework wrappers
 */

import { join } from "jsr:@std/path";

const ROOT_DIR = new URL("..", import.meta.url).pathname;
const WASM_DIR = join(ROOT_DIR, "packages/wasm");
const CORE_DIR = join(ROOT_DIR, "packages/core");
const REACT_DIR = join(ROOT_DIR, "packages/react");
const VUE_DIR = join(ROOT_DIR, "packages/vue");
const SVELTE_DIR = join(ROOT_DIR, "packages/svelte");

interface BuildOptions {
  release: boolean;
  simd: boolean;
  skipWasm: boolean;
  skipFrameworks: boolean;
}

function parseArgs(): BuildOptions {
  const args = Deno.args;
  return {
    release: !args.includes("--dev"),
    simd: args.includes("--simd"),
    skipWasm: args.includes("--skip-wasm"),
    skipFrameworks: args.includes("--skip-frameworks"),
  };
}

async function runCommand(
  cmd: string[],
  cwd: string,
  description: string
): Promise<void> {
  console.log(`\n[BUILD] ${description}...`);
  console.log(`        Running: ${cmd.join(" ")}`);

  const command = new Deno.Command(cmd[0], {
    args: cmd.slice(1),
    cwd,
    stdout: "inherit",
    stderr: "inherit",
  });

  const { code } = await command.output();

  if (code !== 0) {
    throw new Error(`${description} failed with exit code ${code}`);
  }

  console.log(`        Done!`);
}

async function buildWasm(options: BuildOptions): Promise<void> {
  const args = ["./build.sh"];
  if (!options.release) {
    args.push("--dev");
  }
  if (options.simd) {
    args.push("--simd");
  }

  await runCommand(["bash", ...args], WASM_DIR, "Building WASM module");
}

async function typeCheckCore(): Promise<void> {
  await runCommand(
    ["deno", "check", "mod.ts"],
    CORE_DIR,
    "Type-checking @heroine-graph/core"
  );
}

async function buildReact(): Promise<void> {
  await runCommand(
    ["npm", "run", "build"],
    REACT_DIR,
    "Building @heroine-graph/react"
  );
}

async function buildVue(): Promise<void> {
  await runCommand(
    ["npm", "run", "build"],
    VUE_DIR,
    "Building @heroine-graph/vue"
  );
}

async function buildSvelte(): Promise<void> {
  await runCommand(
    ["npm", "run", "build"],
    SVELTE_DIR,
    "Building @heroine-graph/svelte"
  );
}

async function bundleCore(): Promise<void> {
  const distDir = join(ROOT_DIR, "dist");

  // Create dist directory if it doesn't exist
  try {
    await Deno.mkdir(distDir, { recursive: true });
  } catch {
    // Directory exists
  }

  // Bundle core for browser using esbuild
  // This handles .wgsl imports properly
  await runCommand(
    [
      "npx",
      "esbuild",
      join(CORE_DIR, "mod.ts"),
      "--bundle",
      "--format=esm",
      "--platform=browser",
      "--target=es2022",
      `--outfile=${join(distDir, "heroine-graph.esm.js")}`,
      "--loader:.wgsl=text",
      "--external:@heroine-graph/wasm",
    ],
    ROOT_DIR,
    "Bundling @heroine-graph/core for browser"
  );
}

async function main(): Promise<void> {
  const options = parseArgs();

  console.log("=".repeat(60));
  console.log("Heroine Graph - Build");
  console.log("=".repeat(60));
  console.log(`Release: ${options.release}`);
  console.log(`SIMD: ${options.simd}`);
  console.log(`Skip WASM: ${options.skipWasm}`);
  console.log(`Skip Frameworks: ${options.skipFrameworks}`);

  const startTime = performance.now();

  try {
    // Step 1: Build WASM
    if (!options.skipWasm) {
      await buildWasm(options);
    }

    // Step 2: Type-check core
    await typeCheckCore();

    // Step 3: Bundle core for browser
    await bundleCore();

    // Step 4: Build framework wrappers
    if (!options.skipFrameworks) {
      await Promise.all([buildReact(), buildVue(), buildSvelte()]);
    }

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    console.log("\n" + "=".repeat(60));
    console.log(`Build completed in ${elapsed}s`);
    console.log("=".repeat(60));
  } catch (error) {
    console.error("\n[ERROR]", error);
    Deno.exit(1);
  }
}

main();
