/**
 * Unit tests for the storage-buffer device-limit guards, and for the
 * declarations they read.
 *
 * On a device that cannot supply an algorithm's storage buffers the bind group
 * LAYOUT is invalid, binding it poisons the compute pass, the pass poisons the
 * command encoder, and submit() discards the entire frame — clear-forces,
 * springs and integration included. The observable symptom is a frozen
 * simulation with WebGPU validation spam, which is exactly what made
 * "Barnes-Hut is inert" look like a physics bug.
 *
 * The guards turn that into a named failure (createPipelines) and a skipped
 * candidate (registry selection). The last test here guards the other half:
 * that no built-in algorithm needs more than the WebGPU default of 8 in the
 * first place, so the guards never have to fire on a conformant adapter. No
 * GPU needed — all of it reads `device.limits` and static info.
 */

import { assert, assertEquals, assertThrows } from "jsr:@std/assert@^1";
import {
  assertAlgorithmSupportedOnDevice,
  supportsAlgorithmOnDevice,
} from "../../packages/core/src/simulation/algorithms/types.ts";
import type { ForceAlgorithmInfo } from "../../packages/core/src/simulation/algorithms/types.ts";
import { ForceAlgorithmRegistry } from "../../packages/core/src/simulation/algorithms/registry.ts";
import { loadModuleInliningWgsl } from "../helpers/gpu.ts";
import { GraphMotherError } from "../../packages/core/src/errors.ts";

/** A device stub exposing only the limit the guards consult. */
function deviceWithLimit(maxStorageBuffersPerShaderStage: number): Pick<GPUDevice, "limits"> {
  return { limits: { maxStorageBuffersPerShaderStage } as GPUSupportedLimits };
}

function info(
  id: string,
  minStorageBuffersPerShaderStage?: number,
): ForceAlgorithmInfo {
  return {
    id: id as ForceAlgorithmInfo["id"],
    name: id,
    description: "",
    minNodes: 0,
    maxNodes: -1,
    complexity: "",
    ...(minStorageBuffersPerShaderStage !== undefined ? { minStorageBuffersPerShaderStage } : {}),
  };
}

/** Registry entry: only `info` is consulted by selection. */
function stub(id: string, minBuffers?: number) {
  return { info: info(id, minBuffers) } as unknown as Parameters<
    ForceAlgorithmRegistry["register"]
  >[0];
}

Deno.test("supportsAlgorithmOnDevice: compares the declared requirement against the device", () => {
  assert(supportsAlgorithmOnDevice(info("n2"), deviceWithLimit(8)), "no requirement always fits");
  assert(supportsAlgorithmOnDevice(info("barnes-hut", 10), deviceWithLimit(10)));
  assert(supportsAlgorithmOnDevice(info("barnes-hut", 10), deviceWithLimit(31)));
  assertEquals(supportsAlgorithmOnDevice(info("barnes-hut", 10), deviceWithLimit(8)), false);
  assertEquals(supportsAlgorithmOnDevice(info("barnes-hut", 10), deviceWithLimit(9)), false);
  // Relativity Atlas sits between the two: 9 is enough for it, not for BH.
  assert(supportsAlgorithmOnDevice(info("relativity-atlas", 9), deviceWithLimit(9)));
});

Deno.test("assertAlgorithmSupportedOnDevice: names the limit, the requirement and the fix", () => {
  assertAlgorithmSupportedOnDevice(info("barnes-hut", 10), deviceWithLimit(10));

  const error = assertThrows(
    () => assertAlgorithmSupportedOnDevice(info("barnes-hut", 10), deviceWithLimit(8)),
    GraphMotherError,
    "maxStorageBuffersPerShaderStage",
  );
  assert(error.message.includes("10"), "the requirement must be in the message");
  assert(error.message.includes("8"), "the device's actual limit must be in the message");
  assert(error.suggestion !== undefined, "the error must say what to do instead");
});

Deno.test("registry: large-graph selection skips algorithms the device cannot run", () => {
  const registry = new ForceAlgorithmRegistry();
  registry.register(stub("n2"));
  registry.register(stub("barnes-hut", 10));

  // A capable device gets the intended pick...
  assertEquals(registry.getRecommended(20_000, deviceWithLimit(10))?.info.id, "barnes-hut");
  // ...an 8-limit adapter falls through to a working algorithm instead of
  // selecting the one whose every submit would be discarded.
  assertEquals(registry.getRecommended(20_000, deviceWithLimit(8))?.info.id, "n2");
  assertEquals(registry.getRecommended(200_000, deviceWithLimit(8))?.info.id, "n2");

  // Without a device the caller gets the historical behaviour.
  assertEquals(registry.getRecommended(20_000)?.info.id, "barnes-hut");
});

Deno.test("registry: the last-resort fallback also respects the device", () => {
  const registry = new ForceAlgorithmRegistry();
  // Only over-limit algorithms registered: an 8-limit device must get nothing
  // rather than a guaranteed-frozen one.
  registry.register(stub("barnes-hut", 10));
  registry.register(stub("relativity-atlas", 9));

  assertEquals(registry.getRecommended(100, deviceWithLimit(8)), undefined);
  assertEquals(registry.getRecommended(100, deviceWithLimit(9))?.info.id, "relativity-atlas");
});

/** The WebGPU default, and the floor every conformant adapter reports. */
const WEBGPU_DEFAULT_STORAGE_BUFFERS = 8;

/**
 * The built-in algorithm infos, loaded the way the GPU harnesses load anything
 * that reaches a `.wgsl` import: deno test's loader cannot follow those, so the
 * module is transformed and imported through a data: URL. No GPU is touched —
 * `info` is static data and the guards read only `device.limits`.
 */
async function builtinInfos(): Promise<readonly ForceAlgorithmInfo[]> {
  const mod = await loadModuleInliningWgsl<{
    initializeBuiltinAlgorithms(): void;
    getAlgorithmRegistry(): { listInfo(): ForceAlgorithmInfo[] };
  }>(new URL("../../packages/core/src/simulation/algorithms/mod.ts", import.meta.url));
  mod.initializeBuiltinAlgorithms();
  return mod.getAlgorithmRegistry().listInfo();
}

Deno.test("every built-in algorithm fits a default-limit adapter", async () => {
  const device = deviceWithLimit(WEBGPU_DEFAULT_STORAGE_BUFFERS);
  const infos = await builtinInfos();
  assert(infos.length > 0, "the built-ins must actually be registered");

  for (const algorithmInfo of infos) {
    const declared = algorithmInfo.minStorageBuffersPerShaderStage;
    assert(
      declared === undefined || declared <= WEBGPU_DEFAULT_STORAGE_BUFFERS,
      `${algorithmInfo.id} declares ${declared} storage buffers per stage, past the ` +
        `default of ${WEBGPU_DEFAULT_STORAGE_BUFFERS}: it would be unselectable on every ` +
        "adapter reporting only the default",
    );
    assert(
      supportsAlgorithmOnDevice(algorithmInfo, device),
      `${algorithmInfo.id} is not selectable on a default-limit adapter`,
    );
  }
});

Deno.test("the algorithms sitting exactly on the limit declare it", async () => {
  const infos = new Map((await builtinInfos()).map((i) => [i.id as string, i]));

  // Each of these reached 8 by merging buffers a single pass reads together
  // into one flat multi-region binding, and none has headroom left. The
  // declaration is the tripwire: a ninth binding must fail here rather than
  // silently cost the algorithm every default-limit adapter.
  for (const id of ["force-atlas2", "linlog", "relativity-atlas", "barnes-hut"]) {
    const algorithmInfo = infos.get(id);
    assert(algorithmInfo !== undefined, `${id} must be registered`);
    assertEquals(
      algorithmInfo.minStorageBuffersPerShaderStage,
      WEBGPU_DEFAULT_STORAGE_BUFFERS,
      `${id} must declare its widest layout as exactly ${WEBGPU_DEFAULT_STORAGE_BUFFERS}`,
    );
  }
});
