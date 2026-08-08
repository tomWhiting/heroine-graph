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
import type { CardErrorEvent, NodeId } from "../../packages/core/src/types.ts";
import type { CardNode, CardProvider } from "../../packages/core/src/overlay/types.ts";
import { ErrorCode } from "../../packages/core/src/errors.ts";
import { HIERARCHY_ROOT } from "../../packages/core/src/graph/hierarchy.ts";

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

/**
 * One root containing `weights.length` leaves, with the hierarchy supplied.
 *
 * Supplied rather than derived because the harness has no WASM engine, and the
 * LOD controller does nothing at all without a hierarchy — so this is what makes
 * the *controller* the caller of `syncDomCards` here, rather than the test.
 *
 * `weight` is the producer's own ranking column, and the controller adds it to
 * the screen radius when it ranks cards. Every node here has the same radius, so
 * the weights are what decide which node wins a card budget, and they decide it
 * exactly the way a consumer's own importance column would.
 */
function cardTree(weights: readonly number[]) {
  const leaves = weights.length;
  const n = leaves + 1;
  const parent = new Uint32Array(n);
  parent[0] = HIERARCHY_ROOT;
  const depth = new Uint16Array(n);
  const subtreeSize = new Uint32Array(n).fill(1);
  subtreeSize[0] = n;
  // The root's bubble is large enough to sit above any expand threshold a test
  // sets, so the cut always reaches the leaves.
  const wellRadius = new Float32Array(n).fill(10);
  wellRadius[0] = 200;
  const positions = new Float32Array(n * 2);
  const weight = new Float32Array(n);
  const nodeIds: string[] = ["src"];
  const edgePairs = new Uint32Array(leaves * 2);
  const edgeKinds = new Uint16Array(leaves);

  for (let slot = 1; slot < n; slot++) {
    parent[slot] = 0;
    depth[slot] = 1;
    // Well inside the viewport, so the card ring's cull keeps every leaf.
    positions[slot * 2] = (slot - leaves / 2) * 20;
    positions[slot * 2 + 1] = 30;
    weight[slot] = weights[slot - 1];
    nodeIds.push(`src/f${slot - 1}.ts`);
    edgePairs[(slot - 1) * 2] = 0;
    edgePairs[(slot - 1) * 2 + 1] = slot;
  }

  return {
    nodeCount: n,
    edgeCount: leaves,
    positions,
    edgePairs,
    edgeKinds,
    containmentKind: 0,
    hierarchy: { parent, wellRadius, depth, subtreeSize },
    nodeIds,
    weight,
  };
}

/** Thresholds low enough that every leaf in `cardTree` is card-sized. */
const CARD_EVERYTHING = {
  enabled: true,
  expandThreshold: 1,
  collapseThreshold: 0.5,
  domThreshold: 1,
  domExitThreshold: 0.5,
  minCardLifetimeMs: 0,
  minBandCommitFrames: 0,
} as const;

/** A provider that throws in `mount` for one producer id and works for the rest. */
function providerFailingFor(externalId: string, mounts: string[]): CardProvider<unknown> {
  return {
    mount(_container: HTMLElement, node: CardNode) {
      mounts.push(String(node.externalId));
      if (String(node.externalId) === externalId) throw new Error("card content failed");
      return { externalId: String(node.externalId) };
    },
  } as CardProvider<unknown>;
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

/**
 * A chain of `count` nodes with the containment hierarchy supplied rather than
 * derived. The harness builds GraphMother with `wasmEngine: null`, so nothing
 * derives a hierarchy from containment edges — and without one the LOD cut has
 * no tree to walk and folds nothing. Supplying the columns is the shipped path
 * for exactly that case (a native indexer precomputing them offline), so this
 * exercises real code rather than working around its absence.
 */
function chainWithHierarchy(count: number) {
  const parent = new Uint32Array(count);
  const depth = new Uint16Array(count);
  const subtreeSize = new Uint32Array(count);
  const wellRadius = new Float32Array(count);
  const positions = new Float32Array(count * 2);
  const edgePairs = new Uint32Array((count - 1) * 2);
  const edgeKinds = new Uint16Array(count - 1);

  parent[0] = HIERARCHY_ROOT;
  for (let i = 0; i < count; i++) {
    if (i > 0) {
      parent[i] = i - 1;
      edgePairs[(i - 1) * 2] = i - 1;
      edgePairs[(i - 1) * 2 + 1] = i;
    }
    depth[i] = i;
    subtreeSize[i] = count - i;
    // Descending with depth, so a shallower node is the wider bubble and the
    // cut has a reason to prefer folding deeper ones.
    wellRadius[i] = 10 + (count - i) * 20;
    positions[i * 2] = i * 60;
    positions[i * 2 + 1] = 0;
  }

  return {
    nodeCount: count,
    edgeCount: count - 1,
    positions,
    edgePairs,
    edgeKinds,
    containmentKind: 0,
    nodeIds: Array.from({ length: count }, (_, i) => `src/mod${i}.ts`),
    hierarchy: { parent, wellRadius, depth, subtreeSize },
  };
}

Deno.test({
  name: "headless: a mutation pays out the drift a live fold owes its subtree",
  ignore: device === null,
  fn: async () => {
    assert(device !== null, GPU_SKIP_MESSAGE);
    let harness: HeadlessHarness | undefined;
    try {
      harness = await createHeadlessGraph(device);
      const { graph } = harness;
      await graph.load(chainWithHierarchy(8));

      graph.setScale(1);
      graph.setLodConfig({ enabled: true });
      harness.evaluateLod(0);

      // Fold a subtree. Its descendants are frozen from here on; the proxy is
      // the only one of them the simulation still moves.
      const proxy = graph.getNodeId("src/mod2.ts");
      const descendant = graph.getNodeId("src/mod5.ts");
      assert(proxy !== undefined && descendant !== undefined);
      graph.collapseNode(proxy);
      assert(graph.isCollapsed(proxy), "the subtree must actually be folded");

      const proxyBefore = graph.getNodePosition(proxy);
      const before = graph.getNodePosition(descendant);
      assert(proxyBefore !== undefined && before !== undefined);

      // Drift: the proxy goes on simulating while the subtree does not, so it
      // ends up somewhere its descendants have not followed to.
      const dx = 137;
      const dy = -91;
      graph.setNodePosition(proxy, proxyBefore.x + dx, proxyBefore.y + dy);

      // Any mutation that drops the derived topology ends both the hierarchy
      // and the slot space the debt is expressible in, so it has to be paid
      // first. Adding an edge is as ordinary as a mutation gets.
      await graph.addEdgesBatch([{ source: "src/mod0.ts", target: "src/mod7.ts" }]);

      // The assertion is on the settle's *effect*. Asserting that it was called
      // passes against a settle that does nothing — which is precisely what the
      // static oracle in tests/unit/mutation_invalidation_test.ts can see and
      // this cannot: empty `beginTopologyChange` and only this test goes red.
      const after = graph.getNodePosition(descendant);
      assert(after !== undefined);
      assertEquals(
        [Math.round(after.x - before.x), Math.round(after.y - before.y)],
        [dx, dy],
        "the folded subtree must be carried by the drift its proxy accumulated",
      );
    } finally {
      harness?.dispose();
    }
  },
});

Deno.test({
  name: "headless: a card provider that throws does not take the graph with it",
  ignore: device === null,
  fn: async () => {
    assert(device !== null, GPU_SKIP_MESSAGE);
    let harness: HeadlessHarness | undefined;
    try {
      const host = domHost();
      harness = await createHeadlessGraph(device, { overlayHost: host });
      const { graph } = harness;
      await graph.load(chain(8));

      // Uncontained, the throw below leaves the overlay through `syncDomCards`
      // — and, once the LOD controller is the caller, through its tick at the
      // top of the render callback, which the render loop catches and logs. The
      // frame then aborts before edges, nodes, compute submit and position
      // readback: one bad card freezes the graph while the loop spins.
      const mounts: string[] = [];
      graph.setDomOverlay({ enabled: true, host });
      graph.setCardProvider({
        mount(_container: HTMLElement, node: CardNode) {
          mounts.push(String(node.externalId));
          if (node.externalId === "src/mod3.ts") throw new Error("card content failed");
          return { externalId: String(node.externalId) };
        },
      } as CardProvider<unknown>);

      const errors: CardErrorEvent[] = [];
      graph.on("card:error", (event) => errors.push(event));

      const bad = graph.getNodeId("src/mod3.ts");
      const good = graph.getNodeId("src/mod4.ts");
      assert(bad !== undefined && good !== undefined);

      graph.syncDomCards([{ node: bad, priority: 2 }, { node: good, priority: 1 }]);

      assertEquals(errors.length, 1, "the failure must reach the consumer as an event");
      assertEquals(errors[0].hook, "mount");
      assertEquals(errors[0].externalId, "src/mod3.ts");
      assertEquals(errors[0].nodeId, bad);
      assertEquals(errors[0].released, false, "there was never a card to lose");
      assertEquals(errors[0].error.code, ErrorCode.CARD_PROVIDER_FAILED);

      assertEquals(
        mounts,
        ["src/mod3.ts", "src/mod4.ts"],
        "the rest of the same sync must still be carded",
      );

      // And the graph is still a graph: the next sync neither retries the
      // broken node nor loses the working one.
      graph.syncDomCards([{ node: bad, priority: 2 }, { node: good, priority: 1 }]);
      assertEquals(mounts, ["src/mod3.ts", "src/mod4.ts"]);
      assertEquals(errors.length, 1);
      assertEquals(graph.getExternalId(good), "src/mod4.ts");
    } finally {
      harness?.dispose();
    }
  },
});

Deno.test({
  name: "headless: a failed card stops being declared, instead of becoming a hole",
  ignore: device === null,
  fn: async () => {
    assert(device !== null, GPU_SKIP_MESSAGE);
    let harness: HeadlessHarness | undefined;
    try {
      const host = domHost();
      harness = await createHeadlessGraph(device, { overlayHost: host });
      const { graph } = harness;
      // Descending weights, so which node the controller ranks first is decided
      // by the producer's column rather than by anything incidental.
      await graph.load(cardTree([1, 0.5, 0, 0, 0]));

      const mounts: string[] = [];
      graph.setDomOverlay({ enabled: true, host });
      graph.setCardProvider(providerFailingFor("src/f0.ts", mounts));
      const errors: CardErrorEvent[] = [];
      graph.on("card:error", (event) => errors.push(event));

      // One card, so what the failed node costs is visible: while the
      // controller goes on declaring it, the card budget is spent on a card the
      // overlay refuses to mount and no other node can have one. The sprite side
      // of the same defect — a node faded out under a card that will never exist
      // — is not observable from outside, and is the reason this matters.
      graph.setLodConfig({ ...CARD_EVERYTHING, maxCards: 1 });
      graph.setLodFocus([]);

      assertEquals(mounts, ["src/f0.ts"], "the top-ranked node is carded, and its mount throws");
      assertEquals(errors.length, 1);
      assertEquals(errors[0].hook, "mount");
      assertEquals(errors[0].externalId, "src/f0.ts");

      // A second evaluation, driven exactly as the first one was.
      graph.setLodFocus([]);

      assertEquals(
        mounts,
        ["src/f0.ts", "src/f1.ts"],
        "the card the failed node cannot use must go to the node ranked behind it",
      );
      assertEquals(errors.length, 1, "and the broken provider is never asked again");
    } finally {
      harness?.dispose();
    }
  },
});

Deno.test({
  name: "headless: a new provider takes back the cards the old one failed for",
  ignore: device === null,
  fn: async () => {
    assert(device !== null, GPU_SKIP_MESSAGE);
    let harness: HeadlessHarness | undefined;
    try {
      const host = domHost();
      harness = await createHeadlessGraph(device, { overlayHost: host });
      const { graph } = harness;
      await graph.load(cardTree([1, 0.5, 0, 0, 0]));

      const mounts: string[] = [];
      graph.setDomOverlay({ enabled: true, host });
      graph.setCardProvider(providerFailingFor("src/f0.ts", mounts));

      graph.setLodConfig({ ...CARD_EVERYTHING, maxCards: 1 });
      graph.setLodFocus([]);
      assertEquals(mounts, ["src/f0.ts"]);

      // The documented way back, and it has to reach both halves: the overlay
      // stopped offering the node, the controller stopped declaring it, and a
      // node only one of them has forgiven is still not carded. Nothing else
      // happens here — no evaluation is forced, no camera moves — so a card can
      // only appear because registering the provider re-declared the set.
      const recovered: string[] = [];
      graph.setCardProvider({
        mount(_container: HTMLElement, node: CardNode) {
          recovered.push(String(node.externalId));
          return { externalId: String(node.externalId) };
        },
      } as CardProvider<unknown>);

      assertEquals(
        recovered,
        ["src/f0.ts"],
        "the node the old provider failed for must take its card back in this call",
      );
    } finally {
      harness?.dispose();
    }
  },
});

Deno.test({
  name: "headless: a quarantined slot cards its new occupant after a real compaction",
  ignore: device === null,
  fn: async () => {
    assert(device !== null, GPU_SKIP_MESSAGE);
    let harness: HeadlessHarness | undefined;
    try {
      const host = domHost();
      harness = await createHeadlessGraph(device, { overlayHost: host });
      const { graph } = harness;
      await graph.load(chain(8));

      // The provider fails for a producer id, as a real one does — its content
      // belongs to a node. Nothing about the slot index is what it objects to.
      const mounts: string[] = [];
      graph.setDomOverlay({ enabled: true, host });
      graph.setCardProvider(providerFailingFor("src/mod2.ts", mounts));

      const quarantined = graph.getNodeId("src/mod2.ts");
      assert(quarantined !== undefined);
      graph.syncDomCards([{ node: quarantined, priority: 1 }]);
      assertEquals(mounts, ["src/mod2.ts"], "the failing node is tried once");

      await graph.removeNodesBatch(["src/mod0.ts", "src/mod1.ts"]);

      // Premise check: the quarantine is keyed by slot, so this test says
      // nothing unless the slot really did change hands to another live node.
      const nowAtSlot = graph.getExternalId(quarantined);
      assert(
        nowAtSlot !== undefined && nowAtSlot !== "src/mod2.ts",
        `slot ${quarantined} must now hold a different live node, holds ${String(nowAtSlot)}`,
      );

      graph.syncDomCards([{ node: quarantined, priority: 1 }]);

      assertEquals(
        mounts,
        ["src/mod2.ts", String(nowAtSlot)],
        "a survivor that landed in a quarantined slot never failed and must be carded",
      );
    } finally {
      harness?.dispose();
    }
  },
});

Deno.test({
  name: "headless: an advisory prefetch throw does not cost the node its card",
  ignore: device === null,
  fn: async () => {
    assert(device !== null, GPU_SKIP_MESSAGE);
    let harness: HeadlessHarness | undefined;
    try {
      const host = domHost();
      harness = await createHeadlessGraph(device, { overlayHost: host });
      const { graph } = harness;
      await graph.load(chainWithHierarchy(8));
      graph.setScale(1);
      graph.setLodConfig({ enabled: true });
      // Before the controller has a cut it knows of no slots, and giving up on
      // a node is a no-op — so a failure raised here would be discarded and the
      // test would pass whatever the rule below said.
      harness.evaluateLod(0);

      // `prefetch` is an offer, not a commitment: it creates no DOM and the
      // overlay is free to ignore its outcome. A provider that cannot warm a
      // node up may still be perfectly able to mount it, so a throw here must
      // cost the node nothing. `mount` and `update` are the hooks that were
      // supposed to produce the card, and only those forfeit it.
      const mounts: string[] = [];
      const target = graph.getNodeId("src/mod4.ts");
      assert(target !== undefined);

      graph.setDomOverlay({ enabled: true, host });
      graph.setCardProvider({
        prefetch() {
          throw new Error("warming failed");
        },
        mount(_container: HTMLElement, node: CardNode) {
          mounts.push(String(node.externalId));
          return { externalId: String(node.externalId) };
        },
      } as unknown as CardProvider<unknown>);

      graph.syncDomCards([{ node: target, priority: 1, prefetchOnly: true }]);
      assertEquals(mounts, [], "a prefetch offer creates no DOM");

      // Focus is the LOD controller's own route to a card, so it passes through
      // the suppression set the failure would have added to. Declaring the card
      // through `syncDomCards` instead would bypass the controller entirely and
      // pass whether or not the node had been given up on.
      graph.setLodFocus([target]);

      assertEquals(
        mounts,
        ["src/mod4.ts"],
        "the node must still be cardable after an advisory hook threw",
      );
    } finally {
      harness?.dispose();
    }
  },
});
