#!/usr/bin/env -S deno run --allow-run --allow-read --allow-write --allow-env
/**
 * Graph Mother - Build & Publish Script
 *
 * Builds all packages and publishes them to npm in dependency order:
 *   1. @graphmother/wasm   (no deps)
 *   2. @graphmother/core   (depends on wasm)
 *   3. @graphmother/react  (depends on core)
 *   4. @graphmother/vue    (depends on core)
 *   5. @graphmother/svelte (depends on core)
 *
 * Usage:
 *   deno task publish              # build + publish all
 *   deno task publish --dry-run    # build + dry-run (no actual publish)
 *   deno task publish --skip-build # publish only (assumes already built)
 *   deno task publish --tag beta   # publish with a dist-tag
 */

import { join } from "jsr:@std/path";

const ROOT = new URL("..", import.meta.url).pathname;

interface Options {
  dryRun: boolean;
  skipBuild: boolean;
  tag: string | null;
}

function parseArgs(): Options {
  const args = Deno.args;
  const tagIndex = args.indexOf("--tag");
  return {
    dryRun: args.includes("--dry-run"),
    skipBuild: args.includes("--skip-build"),
    tag: tagIndex !== -1 ? args[tagIndex + 1] ?? null : null,
  };
}

async function run(cmd: string[], cwd: string, label: string): Promise<void> {
  console.log(`\n  [${label}] ${cmd.join(" ")}`);
  const command = new Deno.Command(cmd[0], {
    args: cmd.slice(1),
    cwd,
    stdout: "inherit",
    stderr: "inherit",
  });
  const { code } = await command.output();
  if (code !== 0) {
    throw new Error(`${label} failed (exit ${code})`);
  }
}

async function checkAuth(): Promise<void> {
  console.log("\nChecking npm auth...");
  const command = new Deno.Command("bun", {
    args: ["x", "npm", "whoami"],
    stdout: "piped",
    stderr: "piped",
  });
  const { code, stdout } = await command.output();
  if (code !== 0) {
    console.error(
      "  Not logged in to npm. Run:\n\n" +
        "    echo '//registry.npmjs.org/:_authToken=YOUR_TOKEN' >> ~/.npmrc\n\n" +
        "  Or: bunx npm login\n"
    );
    Deno.exit(1);
  }
  const user = new TextDecoder().decode(stdout).trim();
  console.log(`  Authenticated as: ${user}`);
}

async function buildAll(): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log("  STEP 1: BUILD");
  console.log("=".repeat(60));

  // Build WASM
  await run(
    ["bash", "./build.sh", "--release"],
    join(ROOT, "packages/wasm"),
    "wasm"
  );

  // Type-check core
  await run(
    ["deno", "check", "mod.ts"],
    join(ROOT, "packages/core"),
    "check"
  );

  // Bundle core (esbuild)
  const distDir = join(ROOT, "packages/core/dist");
  try {
    await Deno.mkdir(distDir, { recursive: true });
  } catch { /* exists */ }

  await run(
    [
      "bunx", "esbuild",
      "mod.ts",
      "--bundle", "--format=esm", "--platform=browser", "--target=es2022",
      `--outfile=dist/heroine-graph.esm.js`,
      "--loader:.wgsl=text",
      "--external:@graphmother/wasm",
    ],
    join(ROOT, "packages/core"),
    "bundle"
  );

  // Generate .d.ts for core
  await run(
    ["bunx", "tsc", "--project", "tsconfig.npm.json"],
    join(ROOT, "packages/core"),
    "types"
  );

  // Build framework wrappers in parallel
  await Promise.all([
    run(["bun", "run", "build"], join(ROOT, "packages/react"), "react"),
    run(["bun", "run", "build"], join(ROOT, "packages/vue"), "vue"),
    run(["bun", "run", "build"], join(ROOT, "packages/svelte"), "svelte"),
  ]);
}

async function publishPackage(
  dir: string,
  name: string,
  opts: Options
): Promise<void> {
  const args = ["publish", "--access", "public"];
  if (opts.dryRun) args.push("--dry-run");
  if (opts.tag) args.push("--tag", opts.tag);

  await run(["bun", ...args], dir, name);
}

async function publishAll(opts: Options): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log(`  STEP 2: PUBLISH${opts.dryRun ? " (dry-run)" : ""}`);
  console.log("=".repeat(60));

  // Publish in dependency order
  await publishPackage(
    join(ROOT, "packages/wasm/pkg"),
    "@graphmother/wasm",
    opts
  );
  await publishPackage(
    join(ROOT, "packages/core"),
    "@graphmother/core",
    opts
  );

  // Framework wrappers can publish in parallel (all depend on core only)
  await Promise.all([
    publishPackage(join(ROOT, "packages/react"), "@graphmother/react", opts),
    publishPackage(join(ROOT, "packages/vue"), "@graphmother/vue", opts),
    publishPackage(join(ROOT, "packages/svelte"), "@graphmother/svelte", opts),
  ]);
}

async function main(): Promise<void> {
  const opts = parseArgs();

  console.log("=".repeat(60));
  console.log("  Graph Mother - Publish");
  console.log("=".repeat(60));
  console.log(`  Dry run:    ${opts.dryRun}`);
  console.log(`  Skip build: ${opts.skipBuild}`);
  console.log(`  Tag:        ${opts.tag ?? "(latest)"}`);

  const start = performance.now();

  try {
    // Check auth (skip for dry-run since bun dry-run doesn't need it... but
    // it's still nice to verify early)
    if (!opts.dryRun) {
      await checkAuth();
    }

    if (!opts.skipBuild) {
      await buildAll();
    }

    await publishAll(opts);

    const elapsed = ((performance.now() - start) / 1000).toFixed(2);
    console.log("\n" + "=".repeat(60));
    console.log(`  Done in ${elapsed}s`);
    console.log("=".repeat(60));
  } catch (error) {
    console.error("\n[ERROR]", error);
    Deno.exit(1);
  }
}

main();
