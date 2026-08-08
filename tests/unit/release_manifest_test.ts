/**
 * The release manifests have to agree with each other, and the publish
 * pipeline has to be able to build without the registry already carrying the
 * release it is about to make.
 *
 * Both have gone wrong before. 0.3.0 shipped a core that pinned a wasm whose
 * API it had outgrown, because the version lived in six files and only five
 * moved. And a new minor could not be built at all for a while: `deno install`
 * resolved core's dependency on `@graphmother/wasm` from npm, at the exact
 * version the run existed to publish.
 *
 * These are static-manifest checks; they say nothing about whether the built
 * artefacts work. Their job is to fail the moment two files disagree, which is
 * the moment the disagreement is cheap.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import { join } from "jsr:@std/path@^1";

const ROOT = new URL("../../", import.meta.url).pathname;

function readText(relPath: string): string {
  return Deno.readTextFileSync(join(ROOT, relPath));
}

/** Only the manifest fields this file has an opinion about. */
interface Manifest {
  name?: string;
  version?: string;
  main?: string;
  types?: string;
  files?: string[];
  workspaces?: string[];
  dependencies?: Record<string, string>;
}

function readJson(relPath: string): Manifest {
  return JSON.parse(readText(relPath));
}

/** The version every other manifest is measured against (see scripts/publish.ts). */
function cargoVersion(): string {
  const match = readText("packages/wasm/Cargo.toml").match(/^version\s*=\s*"([^"]+)"/m);
  assert(match, "packages/wasm/Cargo.toml has no version field");
  return match[1];
}

/** Every manifest that restates the release version, and where it keeps it. */
const VERSION_SITES: ReadonlyArray<readonly [relPath: string, read: () => string]> = [
  ["packages/wasm/package.json", () => readJson("packages/wasm/package.json").version!],
  ["packages/core/package.json", () => readJson("packages/core/package.json").version!],
  ["packages/core/deno.json", () => readJson("packages/core/deno.json").version!],
  ["deno.json", () => readJson("deno.json").version!],
  ["packages/react/package.json", () => readJson("packages/react/package.json").version!],
  ["packages/vue/package.json", () => readJson("packages/vue/package.json").version!],
  [
    "packages/svelte/package.json",
    () => readJson("packages/svelte/package.json").version!,
  ],
];

Deno.test("every manifest states the same release version", () => {
  const expected = cargoVersion();
  for (const [relPath, read] of VERSION_SITES) {
    assertEquals(read(), expected, `${relPath} disagrees with Cargo.toml`);
  }
});

Deno.test("the VERSION the bundle reports is the version being shipped", () => {
  const [major, minor, patch] = cargoVersion().split(".");
  const factory = readText("packages/core/src/api/factory.ts");
  const match = factory.match(
    /export const VERSION = \{\s*major: (\d+),\s*minor: (\d+),\s*patch: (\d+)/,
  );
  assert(match, "packages/core/src/api/factory.ts has no VERSION literal");
  assertEquals([match[1], match[2], match[3]], [major, minor, patch]);
});

Deno.test("core asks for a wasm range this repo's wasm actually satisfies", () => {
  const core = readJson("packages/core/package.json");
  const range = core.dependencies?.["@graphmother/wasm"];
  assert(range, "packages/core/package.json no longer depends on @graphmother/wasm");

  const caret = range.match(/^\^(\d+)\.(\d+)\.\d+$/);
  assert(caret, `expected a caret range like ^0.4.0, saw ${range}`);
  const [major, minor] = cargoVersion().split(".");
  assertEquals(
    [caret[1], caret[2]],
    [major, minor],
    `core would resolve ${range} from npm, not the wasm in this repo`,
  );
});

Deno.test("the wasm core depends on is a workspace member, not a registry lookup", () => {
  // Without this, `deno install` during a release goes to npm for the very
  // version the release is publishing, and the build fails before it starts.
  const root = readJson("package.json");
  const workspaces = root.workspaces;
  assert(workspaces, "root package.json declares no npm workspaces");

  const wasmName = readJson("packages/wasm/package.json").name;
  assertEquals(wasmName, "@graphmother/wasm");
  assert(
    workspaces.includes("packages/wasm"),
    `packages/wasm is not an npm workspace member (saw ${JSON.stringify(workspaces)})`,
  );
});

Deno.test("the wasm tarball would ship every file it promises", () => {
  const pkg = readJson("packages/wasm/package.json");
  const files = pkg.files!;
  const entry = pkg.main!;
  const types = pkg.types!;

  // `files` is a whitelist: npm drops a listed path that does not exist and
  // says nothing, so a rename in build.sh publishes a tarball missing its
  // module and consumers get the failure instead.
  for (const relPath of [...files, entry, types]) {
    const path = join(ROOT, "packages/wasm", relPath);
    assert(
      Deno.statSync(path).isFile,
      `packages/wasm/${relPath} is promised by package.json but is not a file`,
    );
  }

  assert(files.includes(entry), `files does not include main (${entry})`);
  assert(files.includes(types), `files does not include types (${types})`);
});
