#!/usr/bin/env -S deno run --allow-run --allow-read --allow-write --allow-env
/**
 * GraphMother - Build Script
 *
 * Orchestrates the build process for all packages:
 * 1. Install the lockfile-pinned toolchain (deno install)
 * 2. Build WASM module
 * 3. Type-check TypeScript packages
 * 4. Bundle core for browser
 * 5. Build framework wrappers (react only — see note below)
 *
 * All tools (esbuild, tsc, tsup) resolve from node_modules, pinned by
 * deno.lock, so the build is reproducible in a clean checkout.
 */

import { join } from "jsr:@std/path";
import { buildCoreTypes } from "./build_core_types.ts";

const ROOT_DIR = new URL("..", import.meta.url).pathname;
const WASM_DIR = join(ROOT_DIR, "packages/wasm");
const CORE_DIR = join(ROOT_DIR, "packages/core");
const REACT_DIR = join(ROOT_DIR, "packages/react");

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
  description: string,
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

async function installToolchain(): Promise<void> {
  await runCommand(
    ["deno", "install"],
    ROOT_DIR,
    "Installing lockfile-pinned toolchain",
  );
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
    "Type-checking @graphmother/core",
  );
}

async function buildReact(): Promise<void> {
  // tsup config lives in packages/react/package.json ("tsup" key)
  await runCommand(
    ["deno", "run", "-A", "npm:tsup@^8.3.0/tsup"],
    REACT_DIR,
    "Building @graphmother/react",
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

  // Bundle core for browser using esbuild (pinned via deno.lock)
  // This handles .wgsl imports properly
  await runCommand(
    [
      "deno",
      "run",
      "-A",
      "npm:esbuild@^0.24.0/esbuild",
      join(CORE_DIR, "mod.ts"),
      "--bundle",
      "--format=esm",
      "--platform=browser",
      "--target=es2022",
      `--outfile=${join(distDir, "graphmother.esm.js")}`,
      "--loader:.wgsl=text",
      "--external:@graphmother/wasm",
    ],
    ROOT_DIR,
    "Bundling @graphmother/core for browser",
  );
}

async function main(): Promise<void> {
  const options = parseArgs();

  console.log("=".repeat(60));
  console.log("GraphMother - Build");
  console.log("=".repeat(60));
  console.log(`Release: ${options.release}`);
  console.log(`SIMD: ${options.simd}`);
  console.log(`Skip WASM: ${options.skipWasm}`);
  console.log(`Skip Frameworks: ${options.skipFrameworks}`);

  const startTime = performance.now();

  try {
    // Step 1: Install pinned toolchain
    await installToolchain();

    // Step 2: Build WASM
    if (!options.skipWasm) {
      await buildWasm(options);
    }

    // Step 3: Type-check core
    await typeCheckCore();

    // Step 4: Bundle core for browser
    await bundleCore();

    // Step 5: Generate core .d.ts
    console.log("\n[BUILD] Generating @graphmother/core declarations...");
    await buildCoreTypes();

    // Step 6: Build framework wrappers.
    // vue/svelte are skipped: their dts builds fail at the source level
    // (vue needs a .vue-aware bundler + shim; svelte needs a tsconfig.json).
    if (!options.skipFrameworks) {
      await buildReact();
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
