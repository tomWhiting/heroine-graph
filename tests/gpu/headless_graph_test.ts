/**
 * Headless GraphMother — the assembled instance.
 *
 * Everything here needs the *whole* object rather than one subsystem: the
 * question is what a mutation does to state other subsystems hold about it.
 * `removeNodesBatch` compacts GPU slots, and both the card overlay and the LOD
 * controller key their own state by slot, so a batch removal is the one event
 * that can silently put a surviving node behind another node's record.
 *
 * The unit tests in `tests/unit/dom_overlay_test.ts` cover the overlay's half
 * against a stand-in graph. This file is the other half: the real compaction,
 * on the real instance, so the stand-in's assumption — that a slot can come to
 * hold a different producer id — is checked against the code that actually
 * reassigns slots rather than against a test's imitation of it.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import { parseHTML } from "https://esm.sh/linkedom@0.18.11/worker";
import {
  createHeadlessGraph,
  headlessDevice,
  type HeadlessHarness,
} from "../helpers/headless_graph.ts";
import { GPU_SKIP_MESSAGE } from "../helpers/gpu.ts";
import type { NodeId } from "../../packages/core/src/types.ts";
import type { CardNode, CardProvider } from "../../packages/core/src/overlay/types.ts";

const device = await headlessDevice();

/** A chain of `count` nodes, each contained by the one before it. */
function chain(count: number) {
  const nodes = Array.from({ length: count }, (_, i) => ({ id: `src/mod${i}.ts` }));
  const edges = Array.from({ length: count - 1 }, (_, i) => ({
    source: `src/mod${i}.ts`,
    target: `src/mod${i + 1}.ts`,
    type: "contains",
  }));
  return { nodes, edges };
}

/** Records which producer id each mount/release was for. */
interface Recorder extends CardProvider<{ readonly externalId: string }> {
  readonly mounts: string[];
  readonly releases: string[];
}

function recorder(): Recorder {
  const mounts: string[] = [];
  const releases: string[] = [];
  return {
    mounts,
    releases,
    mount(_container: HTMLElement, node: CardNode) {
      const externalId = String(node.externalId);
      mounts.push(externalId);
      return { externalId };
    },
    release(_container: HTMLElement, _node: CardNode, state: { readonly externalId: string }) {
      releases.push(state.externalId);
    },
  };
}

function domHost(): HTMLElement {
  const { document } = parseHTML("<html><body><div id='host'></div></body></html>");
  return document.getElementById("host") as unknown as HTMLElement;
}

Deno.test({
  name: "headless: GraphMother loads and mutates without a canvas",
  ignore: device === null,
  fn: async () => {
    assert(device !== null, GPU_SKIP_MESSAGE);
    let harness: HeadlessHarness | undefined;
    try {
      harness = await createHeadlessGraph(device);
      const { graph } = harness;
      await graph.load(chain(8));
      assertEquals(graph.nodeCount, 8);

      // The identity mapping is the contract cards and consumers key on.
      const slot = graph.getNodeId("src/mod3.ts");
      assert(slot !== undefined);
      assertEquals(graph.getExternalId(slot), "src/mod3.ts");
    } finally {
      harness?.dispose();
    }
  },
});

Deno.test({
  name: "headless: a batch removal compacts slots, so a survivor changes hands",
  ignore: device === null,
  fn: async () => {
    assert(device !== null, GPU_SKIP_MESSAGE);
    let harness: HeadlessHarness | undefined;
    try {
      harness = await createHeadlessGraph(device);
      const { graph } = harness;
      await graph.load(chain(8));

      const before = new Map<NodeId, string>();
      for (let slot = 0; slot < 8; slot++) {
        before.set(slot, String(graph.getExternalId(slot)));
      }

      // Remove from the front, so every survivor shifts down.
      const { removedCount, nodeSlotRemap } = await graph.removeNodesBatch([
        "src/mod0.ts",
        "src/mod1.ts",
      ]);
      assertEquals(removedCount, 2);
      assertEquals(graph.nodeCount, 6);

      // This is the premise the overlay unit tests stand on. If compaction ever
      // stopped reassigning slots, those tests would still pass while testing
      // nothing — so it is asserted here, against the real thing.
      const changedHands = [...before.keys()].filter((slot) =>
        slot < 6 && graph.getExternalId(slot) !== before.get(slot)
      );
      assert(
        changedHands.length > 0,
        "a batch removal must move survivors into slots other nodes held",
      );

      // And the remap the caller is handed agrees with where they actually went.
      for (const [oldSlot, newSlot] of nodeSlotRemap) {
        assertEquals(
          graph.getExternalId(newSlot),
          before.get(oldSlot),
          `slot ${oldSlot} -> ${newSlot} must carry its node`,
        );
      }
    } finally {
      harness?.dispose();
    }
  },
});

Deno.test({
  name: "headless: a card whose slot changed hands is rebound, lifetime floor and all",
  ignore: device === null,
  fn: async () => {
    assert(device !== null, GPU_SKIP_MESSAGE);
    let harness: HeadlessHarness | undefined;
    try {
      const host = domHost();
      harness = await createHeadlessGraph(device, { overlayHost: host });
      const { graph } = harness;
      await graph.load(chain(8));

      const provider = recorder();
      // The floor must be long enough to outlive the whole test. It exists to
      // stop a *correct* card flickering, and the defect being pinned here is
      // precisely that it was also holding an incorrect one — so a short floor
      // would let the ordinary eviction path clean up and the test would pass
      // without the guard ever running.
      graph.setDomOverlay({ enabled: true, host, minCardLifetimeMs: 600_000 });
      graph.setCardProvider(provider as CardProvider<unknown>);

      // Card a node in the middle. Removing the two below it shifts every
      // survivor down by two, so this slot ends up holding a *different, live*
      // node — the case the guard is for. (A card whose old slot goes dead is
      // the easy case: it is unrequested, so eviction takes it anyway.)
      const carded = graph.getNodeId("src/mod2.ts");
      assert(carded !== undefined);
      graph.syncDomCards([{ node: carded, priority: 1 }]);
      assertEquals(provider.mounts, ["src/mod2.ts"]);

      await graph.removeNodesBatch(["src/mod0.ts", "src/mod1.ts"]);

      // Premise check: this test is worthless unless the slot really did change
      // hands to another live node.
      const nowAtSlot = graph.getExternalId(carded);
      assert(
        nowAtSlot !== undefined && nowAtSlot !== "src/mod2.ts",
        `slot ${carded} must now hold a different live node, holds ${String(nowAtSlot)}`,
      );

      // A host re-syncs the same slot, because by screen size it is still worth
      // carding. Nothing in that request says the occupant changed.
      graph.syncDomCards([{ node: carded, priority: 1 }]);

      assertEquals(
        provider.releases,
        ["src/mod2.ts"],
        "the card for the node that left must be released, floor notwithstanding",
      );
      assertEquals(
        provider.mounts,
        ["src/mod2.ts", String(nowAtSlot)],
        "and the slot re-cards for whoever holds it now, in the same sync",
      );
    } finally {
      harness?.dispose();
    }
  },
});
