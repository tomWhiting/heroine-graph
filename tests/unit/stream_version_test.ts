/**
 * Unit tests for StreamManager.version.
 *
 * The layer intensity cache (layers/stream_intensity.ts) skips its
 * O(nodeCount) recompute and buffer upload whenever this counter is unchanged,
 * so a mutation path that fails to advance it would silently serve stale
 * heatmap/metaball data forever. Every mutator is exercised here.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import { createStreamManager } from "../../packages/core/src/streams/stream_manager.ts";
import type { ValueStreamConfig } from "../../packages/core/src/streams/types.ts";

function streamConfig(id: string): ValueStreamConfig {
  return {
    id,
    colorScale: {
      domain: [0, 10],
      stops: [
        { position: 0, color: [0, 0, 0, 0] },
        { position: 1, color: [1, 0, 0, 1] },
      ],
    },
  };
}

Deno.test("StreamManager: every mutation path advances version", () => {
  const manager = createStreamManager();
  let previous = manager.version;

  /** Runs a mutation and asserts it moved the counter forward. */
  const expectBump = (label: string, mutate: () => void): void => {
    mutate();
    assert(
      manager.version > previous,
      `${label} did not advance version (still ${manager.version})`,
    );
    previous = manager.version;
  };

  expectBump("defineStream", () => manager.defineStream(streamConfig("errors")));
  expectBump("setStreamData", () => manager.setStreamData("errors", [{ nodeIndex: 0, value: 5 }]));
  expectBump("setStreamBulkData", () =>
    manager.setStreamBulkData("errors", {
      indices: new Int32Array([1, 2]),
      values: new Float32Array([3, 4]),
    }));
  expectBump("clearStreamData", () => manager.clearStreamData("errors"));
  expectBump("disableStream", () => manager.disableStream("errors"));
  expectBump("enableStream", () => manager.enableStream("errors"));
  expectBump("toggleStream", () => manager.toggleStream("errors"));
  // graph.ts mutates opacity / blend mode / colour scale through the stream
  // object and then calls invalidateCache; that contract is what makes the
  // counter sufficient.
  expectBump("invalidateCache", () => manager.invalidateCache());
  expectBump("removeStream", () => manager.removeStream("errors"));
  expectBump("clear", () => {
    manager.defineStream(streamConfig("latency"));
    manager.clear();
  });
});

Deno.test("StreamManager: read-only access leaves version untouched", () => {
  const manager = createStreamManager();
  manager.defineStream(streamConfig("errors"));
  manager.setStreamData("errors", [{ nodeIndex: 0, value: 5 }]);

  const before = manager.version;
  manager.getStream("errors");
  manager.hasStream("errors");
  manager.getStreamInfo();
  manager.getStreamIds();
  manager.computeBlendedColors(4);
  manager.getAffectedNodeIndices();

  assertEquals(manager.version, before);
});

Deno.test("StreamManager: no-op mutations on missing streams leave version untouched", () => {
  const manager = createStreamManager();
  const before = manager.version;

  // These return false / do nothing rather than throwing, so they must not
  // invalidate anything either.
  assertEquals(manager.removeStream("missing"), false);
  assertEquals(manager.enableStream("missing"), false);
  assertEquals(manager.disableStream("missing"), false);
  assertEquals(manager.toggleStream("missing"), false);
  manager.clearStreamData("missing");

  assertEquals(manager.version, before);
});
