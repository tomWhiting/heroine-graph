#!/usr/bin/env -S deno run --allow-run --allow-read --allow-write --allow-env
/**
 * GraphMother - Build & Publish Script
 *
 * Bumps versions, builds packages, and publishes them to npm in dependency order:
 *   1. @graphmother/wasm   (no deps)
 *   2. @graphmother/core   (depends on wasm)
 *   3. @graphmother/react  (depends on core)
 *
 * The build toolchain (tsc, esbuild, tsup) is resolved from node_modules,
 * which `deno install` materializes from deno.lock — every tool version is
 * pinned by the lockfile, so the pipeline is reproducible in a clean
 * checkout. No unpinned "latest" downloads.
 *
 * All three are npm workspace members (root package.json), so each one's
 * dependency on the package below it resolves to the sibling directory rather
 * than to the registry. Without that, no new minor could ever be released: the
 * install step would demand from npm the exact version the run exists to put
 * there.
 *
 * The vue and svelte wrappers are NOT built or published: their builds are
 * broken at the source level (packages/vue needs a .vue-aware bundler +
 * shim for dts emit; packages/svelte needs a tsconfig.json for
 * svelte-package's dts step). See the skip notice printed during build.
 *
 * npm is used only as the registry client for the final publish step
 * (Deno has no npm-registry publisher); all build steps run through Deno.
 *
 * Usage:
 *   deno task publish                  # build + publish wasm and core
 *   deno task publish --version 0.2.0  # set an explicit version
 *   deno task publish --version minor  # bump major, minor, or patch
 *   deno task publish --dry-run        # build + dry-run (no actual publish)
 *   deno task publish --skip-build     # publish only (assumes already built)
 *   deno task publish --tag beta       # publish with a dist-tag
 */

import { join } from "jsr:@std/path";
import { buildCoreTypes } from "./build_core_types.ts";

const ROOT = new URL("..", import.meta.url).pathname;

/** Package directories that get published (in dependency order). */
const PUBLISH_DIRS: ReadonlyArray<readonly [dir: string, name: string]> = [
  ["packages/wasm", "@graphmother/wasm"],
  ["packages/core", "@graphmother/core"],
  ["packages/react", "@graphmother/react"],
];

interface Options {
  dryRun: boolean;
  skipBuild: boolean;
  tag: string | null;
  version: string | null;
}

function parseArgs(): Options {
  const args = Deno.args;
  const tagIndex = args.indexOf("--tag");
  const versionIndex = args.indexOf("--version");
  return {
    dryRun: args.includes("--dry-run"),
    skipBuild: args.includes("--skip-build"),
    tag: tagIndex !== -1 ? args[tagIndex + 1] ?? null : null,
    version: versionIndex !== -1 ? args[versionIndex + 1] ?? null : null,
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
  const command = new Deno.Command("npm", {
    args: ["whoami"],
    stdout: "piped",
    stderr: "piped",
  });
  const { code, stdout } = await command.output();
  if (code !== 0) {
    console.error(
      "  Not logged in to npm. Run:\n\n    npm login\n",
    );
    Deno.exit(1);
  }
  const user = new TextDecoder().decode(stdout).trim();
  console.log(`  Authenticated as: ${user}`);
}

function bumpVersion(current: string, bump: string): string {
  if (bump.includes(".")) return bump;

  const [major, minor, patch] = current.split(".").map(Number);
  switch (bump) {
    case "major":
      return `${major + 1}.0.0`;
    case "minor":
      return `${major}.${minor + 1}.0`;
    case "patch":
      return `${major}.${minor}.${patch + 1}`;
    default:
      throw new Error(
        `Invalid version bump: "${bump}". Use major, minor, patch, or an explicit semver like 0.2.0`,
      );
  }
}

function getCurrentVersion(): string {
  const cargoPath = join(ROOT, "packages/wasm/Cargo.toml");
  const cargo = Deno.readTextFileSync(cargoPath);
  const match = cargo.match(/^version\s*=\s*"([^"]+)"/m);
  if (!match) throw new Error("Could not read version from Cargo.toml");
  return match[1];
}

/** Wrapper packages that are versioned in lockstep but not published here. */
const WRAPPER_PKG_PATHS: readonly string[] = [
  "packages/react/package.json",
  "packages/vue/package.json",
  "packages/svelte/package.json",
];

/** Rewrites the `"version"` field of a deno.json without reformatting the file. */
function setDenoJsonVersion(relPath: string, version: string): void {
  const path = join(ROOT, relPath);
  const source = Deno.readTextFileSync(path);
  const updated = source.replace(/^(\s*"version"\s*:\s*)"[^"]+"/m, `$1"${version}"`);
  if (updated === source) {
    throw new Error(`Could not find a "version" field to update in ${relPath}`);
  }
  Deno.writeTextFileSync(path, updated);
  console.log(`    ${relPath} -> ${version}`);
}

function setVersions(version: string): void {
  console.log(`\n  Bumping packages to ${version}`);

  // Cargo.toml is the source of truth (getCurrentVersion reads it).
  const cargoPath = join(ROOT, "packages/wasm/Cargo.toml");
  const cargo = Deno.readTextFileSync(cargoPath);
  Deno.writeTextFileSync(
    cargoPath,
    cargo.replace(/^(version\s*=\s*)"[^"]+"/m, `$1"${version}"`),
  );
  console.log(`    Cargo.toml -> ${version}`);

  // The npm manifest for the wasm package is hand-maintained here rather than
  // taken from wasm-pack's generated pkg/package.json, because deno.json links
  // `@graphmother/wasm` to this directory and a link target that only exists
  // after a build would stop every deno command in the repo dead.
  const wasmPkgPath = join(ROOT, "packages/wasm/package.json");
  const wasmPkg = JSON.parse(Deno.readTextFileSync(wasmPkgPath));
  wasmPkg.version = version;
  Deno.writeTextFileSync(wasmPkgPath, JSON.stringify(wasmPkg, null, 2) + "\n");
  console.log(`    wasm/package.json -> ${version}`);

  const [major, minor] = version.split(".");

  const corePkgPath = join(ROOT, "packages/core/package.json");
  const corePkg = JSON.parse(Deno.readTextFileSync(corePkgPath));
  corePkg.version = version;
  corePkg.dependencies["@graphmother/wasm"] = `^${major}.${minor}.0`;
  Deno.writeTextFileSync(corePkgPath, JSON.stringify(corePkg, null, 2) + "\n");
  console.log(`    core/package.json -> ${version}`);
  console.log(`    @graphmother/wasm dependency -> ^${major}.${minor}.0`);

  // Deno manifests carry their own version fields; keep them from drifting.
  setDenoJsonVersion("deno.json", version);
  setDenoJsonVersion("packages/core/deno.json", version);

  // The runtime VERSION constant ships inside the bundle; a manual bump that
  // misses it (or this script skipping it) leaves the library reporting the
  // previous release.
  const factoryPath = join(ROOT, "packages/core/src/api/factory.ts");
  const factory = Deno.readTextFileSync(factoryPath);
  const [vMajor, vMinor, vPatch] = version.split(".");
  const updatedFactory = factory.replace(
    /(export const VERSION = \{\n  major: )\d+(,\n  minor: )\d+(,\n  patch: )\d+/,
    `$1${vMajor}$2${vMinor}$3${vPatch}`,
  );
  if (updatedFactory === factory) {
    throw new Error("Could not find the VERSION literal in packages/core/src/api/factory.ts");
  }
  Deno.writeTextFileSync(factoryPath, updatedFactory);
  console.log(`    core/src/api/factory.ts VERSION -> ${version}`);

  // The framework wrappers are versioned in lockstep with core so that a
  // published wrapper always requests a core it was built against. They are
  // not in PUBLISH_DIRS (see the module docstring) but must not drift.
  for (const relPath of WRAPPER_PKG_PATHS) {
    const path = join(ROOT, relPath);
    const pkg = JSON.parse(Deno.readTextFileSync(path));
    pkg.version = version;
    if (pkg.dependencies?.["@graphmother/core"]) {
      pkg.dependencies["@graphmother/core"] = `^${version}`;
    }
    Deno.writeTextFileSync(path, JSON.stringify(pkg, null, 2) + "\n");
    console.log(`    ${relPath} -> ${version} (@graphmother/core -> ^${version})`);
  }

  // wasm-pack derives packages/wasm/pkg/package.json from Cargo.toml during build.
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

  // WASM first, and not only for its own sake: core depends on
  // `@graphmother/wasm` at the version this very run publishes, and the
  // install below resolves that from the npm workspace member at
  // packages/wasm — whose built output has to be on disk by then. wasm-pack
  // and deno are all this step needs; nothing here wants node_modules.
  await run(
    ["bash", "./build.sh", "--release"],
    join(ROOT, "packages/wasm"),
    "wasm",
  );

  // Materialize the lockfile-pinned toolchain (typescript, esbuild,
  // @webgpu/types, ...) into node_modules. The repo has a root package.json,
  // so Deno resolves `npm:` specifiers through node_modules rather than its
  // own cache — the bundle and .d.ts steps below both need this to have run.
  await run(["deno", "install"], ROOT, "install");

  // Type-check core
  await run(
    ["deno", "check", "mod.ts"],
    join(ROOT, "packages/core"),
    "check",
  );

  // Bundle core (esbuild, resolved from node_modules via deno.lock)
  const distDir = join(ROOT, "packages/core/dist");
  try {
    await Deno.remove(distDir, { recursive: true });
  } catch (error) {
    if (!(error instanceof Deno.errors.NotFound)) {
      throw error;
    }
  }
  await Deno.mkdir(distDir, { recursive: true });

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
      "--minify",
      "--sourcemap",
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

  // The React wrapper, bundle and .d.ts in one tsup pass. It is built last
  // because its dts step resolves @graphmother/core through the workspace
  // link, which reads the declarations emitted just above.
  await run(
    ["deno", "run", "-A", "npm:tsup@^8.3.0/tsup"],
    join(ROOT, "packages/react"),
    "react",
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

  // npm is the registry client only; all builds happen through Deno.
  await run(["npm", ...args], dir, name);
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
  const currentVersion = getCurrentVersion();
  const targetVersion = opts.version ? bumpVersion(currentVersion, opts.version) : currentVersion;

  console.log("=".repeat(60));
  console.log("  GraphMother - Publish");
  console.log("=".repeat(60));
  console.log(`  Current:    ${currentVersion}`);
  console.log(`  Target:     ${targetVersion}`);
  console.log(`  Dry run:    ${opts.dryRun}`);
  console.log(`  Skip build: ${opts.skipBuild}`);
  console.log(`  Tag:        ${opts.tag ?? "(latest)"}`);

  const start = performance.now();

  try {
    // Verify authentication before making a real registry change.
    if (!opts.dryRun) {
      await checkAuth();
    }

    if (opts.version) {
      setVersions(targetVersion);
    }

    if (!opts.skipBuild) {
      await buildAll();
    }

    await publishAll(opts);

    const elapsed = ((performance.now() - start) / 1000).toFixed(2);
    console.log("\n" + "=".repeat(60));
    console.log(`  Published @graphmother v${targetVersion} in ${elapsed}s`);
    console.log("=".repeat(60));
  } catch (error) {
    console.error("\n[ERROR]", error);
    Deno.exit(1);
  }
}

if (import.meta.main) {
  main();
}
