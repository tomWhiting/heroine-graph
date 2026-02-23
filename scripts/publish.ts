#!/usr/bin/env -S deno run --allow-run --allow-read --allow-write --allow-env
/**
 * Graph Mother - Build & Publish Script
 *
 * Builds packages and publishes them to npm in dependency order:
 *   1. @graphmother/wasm   (no deps)
 *   2. @graphmother/core   (depends on wasm)
 *   3. @graphmother/react  (depends on core)
 *
 * The build toolchain (tsc, esbuild, tsup) is resolved from node_modules,
 * which `deno install` materializes from deno.lock — every tool version is
 * pinned by the lockfile, so the pipeline is reproducible in a clean
 * checkout. No unpinned "latest" downloads.
 *
 * The vue and svelte wrappers are NOT built or published: their builds are
 * broken at the source level (packages/vue needs a .vue-aware bundler +
 * shim for dts emit; packages/svelte needs a tsconfig.json for
 * svelte-package's dts step). See the skip notice printed during build.
 *
 * bun is used ONLY as the npm registry client for the final publish step
 * (Deno has no npm-registry publisher); all build steps run through Deno.
 *
 * Usage:
 *   deno task publish              # build + publish all
 *   deno task publish --dry-run    # build + dry-run (no actual publish)
 *   deno task publish --skip-build # publish only (assumes already built)
 *   deno task publish --tag beta   # publish with a dist-tag
 */

import { join } from "jsr:@std/path";
import { buildCoreTypes } from "./build_core_types.ts";

const ROOT = new URL("..", import.meta.url).pathname;

/** Package directories that get published (in dependency order). */
const PUBLISH_DIRS: ReadonlyArray<readonly [dir: string, name: string]> = [
  ["packages/wasm/pkg", "@graphmother/wasm"],
  ["packages/core", "@graphmother/core"],
  ["packages/react", "@graphmother/react"],
];

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
    args: ["pm", "whoami"],
    stdout: "piped",
    stderr: "piped",
  });
  const { code, stdout } = await command.output();
  if (code !== 0) {
    console.error(
      "  Not logged in to npm. Run:\n\n" +
        "    echo '//registry.npmjs.org/:_authToken=YOUR_TOKEN' >> ~/.npmrc\n",
    );
    Deno.exit(1);
  }
  const user = new TextDecoder().decode(stdout).trim();
  console.log(`  Authenticated as: ${user}`);
}

/**
 * Copies the repo-root LICENSE into a package directory so the published
 * tarball actually ships the MIT license text (npm silently skips missing
 * "files" entries; none of the packages have their own LICENSE file).
 */
async function copyLicense(packageDir: string): Promise<void> {
  await Deno.copyFile(join(ROOT, "LICENSE"), join(packageDir, "LICENSE"));
}

async function buildAll(): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log("  STEP 1: BUILD");
  console.log("=".repeat(60));

  // Materialize the lockfile-pinned toolchain (typescript, esbuild, tsup,
  // @webgpu/types, ...) into node_modules. deno.lock pins every version,
  // so this is reproducible in a clean checkout.
  await run(["deno", "install"], ROOT, "install");

  // Build WASM
  await run(
    ["bash", "./build.sh", "--release"],
    join(ROOT, "packages/wasm"),
    "wasm",
  );

  // Type-check core
  await run(
    ["deno", "check", "mod.ts"],
    join(ROOT, "packages/core"),
    "check",
  );

  // Bundle core (esbuild, resolved from node_modules via deno.lock)
  const distDir = join(ROOT, "packages/core/dist");
  try {
    await Deno.mkdir(distDir, { recursive: true });
  } catch { /* exists */ }

  await run(
    [
      "deno",
      "run",
      "-A",
      "npm:esbuild@^0.24.0/esbuild",
      "mod.ts",
      "--bundle",
      "--format=esm",
      "--platform=browser",
      "--target=es2022",
      "--outfile=dist/graphmother.esm.js",
      "--loader:.wgsl=text",
      "--external:@graphmother/wasm",
    ],
    join(ROOT, "packages/core"),
    "bundle",
  );

  // Generate .d.ts for core (pinned tsc + @webgpu/types reference injection)
  console.log("\n  [types] deno run -A scripts/build_core_types.ts");
  await buildCoreTypes();

  // Build the react wrapper (tsup config lives in packages/react/package.json)
  await run(
    ["deno", "run", "-A", "npm:tsup@^8.3.0/tsup"],
    join(ROOT, "packages/react"),
    "react",
  );

  console.log(
    "\n  [skip] @graphmother/vue and @graphmother/svelte are not built:\n" +
      "         their dts builds fail at the source level (vue needs a\n" +
      "         .vue-aware bundler + shim; svelte needs a tsconfig.json).\n" +
      "         They stay unpublished until those are fixed.",
  );
}

async function publishPackage(
  dir: string,
  name: string,
  opts: Options,
): Promise<void> {
  await copyLicense(dir);

  const args = ["publish", "--access", "public"];
  if (opts.dryRun) args.push("--dry-run");
  if (opts.tag) args.push("--tag", opts.tag);

  // bun is the npm registry client only; all builds happen through Deno.
  await run(["bun", ...args], dir, name);
}

async function publishAll(opts: Options): Promise<void> {
  console.log("\n" + "=".repeat(60));
  console.log(`  STEP 2: PUBLISH${opts.dryRun ? " (dry-run)" : ""}`);
  console.log("=".repeat(60));

  // Publish in dependency order
  for (const [dir, name] of PUBLISH_DIRS) {
    await publishPackage(join(ROOT, dir), name, opts);
  }
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

if (import.meta.main) {
  main();
}
