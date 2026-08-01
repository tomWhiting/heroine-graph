/**
 * Card state changes and LOD epoch turnover.
 *
 * Two halves of the same seam. `DomCardOverlay.notify` is how everything core
 * knows about a node other than where it sits — selected, hovered, data
 * changed — reaches a mounted card; without it the `CardStateChange` half of
 * the provider contract is declared and undeliverable. `beginEpoch` is how the
 * prefetch budget is re-armed when the cut moves; without a turnover a node
 * whose prefetch was cancelled is spent for the overlay's lifetime and can
 * never be warmed again.
 *
 * Both are asserted against a recording provider, because both are promises
 * made to consumer code core cannot see.
 *
 * linkedom supplies the DOM, as in `dom_overlay_test.ts`: the repo has no
 * browser runner and neither path needs more than element creation,
 * attachment and inline styles.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import { parseHTML } from "https://esm.sh/linkedom@0.18.11/worker";
import {
  type CardNodeSource,
  type CardSyncEntry,
  DomCardOverlay,
} from "../../packages/core/src/overlay/overlay.ts";
import type {
  CardChange,
  CardProvider,
  CardStateChange,
} from "../../packages/core/src/overlay/types.ts";
import type { NodeId, Vec2, ViewportState } from "../../packages/core/src/types.ts";

// -----------------------------------------------------------------------------
// Harness
// -----------------------------------------------------------------------------

interface Dom {
  readonly host: HTMLElement;
  readonly canvas: HTMLCanvasElement;
}

function createDom(): Dom {
  const { document } = parseHTML(
    "<html><body><div id='host'><canvas id='canvas'></canvas></div></body></html>",
  );
  return {
    host: document.getElementById("host") as unknown as HTMLElement,
    canvas: document.getElementById("canvas") as unknown as HTMLCanvasElement,
  };
}

/** One `provider.update` call, flattened to what the assertions care about. */
interface Update {
  readonly nodeId: NodeId;
  readonly change: CardChange;
}

/** One `provider.prefetch` offer, with the signal core handed it. */
interface Offer {
  readonly nodeId: NodeId;
  readonly signal: AbortSignal;
}

interface Recorder extends CardProvider<{ readonly nodeId: NodeId }> {
  readonly updates: Update[];
  readonly offers: Offer[];
  /** Updates for one node, in order, excluding the core-owned placement ones. */
  stateChanges(nodeId: NodeId): CardStateChange[];
  offersFor(nodeId: NodeId): Offer[];
}

function recorder(): Recorder {
  const updates: Update[] = [];
  const offers: Offer[] = [];
  return {
    updates,
    offers,
    stateChanges(nodeId) {
      return updates
        .filter((u) => u.nodeId === nodeId)
        .map((u) => u.change)
        .filter((c): c is CardStateChange => c.kind !== "position" && c.kind !== "size");
    },
    offersFor(nodeId) {
      return offers.filter((o) => o.nodeId === nodeId);
    },
    prefetch(node, signal) {
      offers.push({ nodeId: node.id, signal });
    },
    mount(_container, node) {
      return { nodeId: node.id };
    },
    update(_container, node, change) {
      updates.push({ nodeId: node.id, change });
    },
  };
}

/** The nine per-node readings the overlay projects a card from. */
class Graph implements CardNodeSource {
  readonly #positions = new Map<NodeId, Vec2>();
  readonly #pinned = new Set<NodeId>();

  constructor(nodeIds: readonly NodeId[]) {
    for (const id of nodeIds) this.#positions.set(id, { x: id * 10, y: id * -5 });
  }

  externalId(node: NodeId): string | undefined {
    return this.#positions.has(node) ? `src/mod${node}.ts` : undefined;
  }
  label(node: NodeId): string | undefined {
    return `mod${node}.ts`;
  }
  tag(): number {
    return 0;
  }
  weight(): number {
    return 0;
  }
  depth(): number {
    return 0;
  }
  contentRef(): string | undefined {
    return undefined;
  }
  position(node: NodeId): Vec2 {
    return this.#positions.get(node) ?? { x: 0, y: 0 };
  }
  isPinned(node: NodeId): boolean {
    return this.#pinned.has(node);
  }
  setPinned(node: NodeId, pinned: boolean): void {
    if (pinned) this.#pinned.add(node);
    else this.#pinned.delete(node);
  }
}

const VIEWPORT: ViewportState = {
  x: 0,
  y: 0,
  scale: 1,
  width: 800,
  height: 600,
  minScale: 0.01,
  maxScale: 100,
};

interface Harness {
  readonly overlay: DomCardOverlay;
  readonly provider: Recorder;
}

function harness(): Harness {
  const dom = createDom();
  const provider = recorder();
  const overlay = new DomCardOverlay({
    canvas: dom.canvas,
    viewport: () => VIEWPORT,
    nodes: new Graph([1, 2, 3, 4]),
    // No anti-flicker floor: these tests move cards in and out of the set on
    // purpose, and holding them would only test the floor again.
    minCardLifetimeMs: 0,
  });
  overlay.setProvider(provider);
  overlay.setConfig({ enabled: true });
  return { overlay, provider };
}

function entry(node: NodeId, options: Partial<CardSyncEntry> = {}): CardSyncEntry {
  return { node, priority: 1, ...options };
}

/** A node in the prefetch ring: warmed, but not carded. */
function ringEntry(node: NodeId): CardSyncEntry {
  return entry(node, { prefetchOnly: true });
}

// -----------------------------------------------------------------------------
// Card state changes
// -----------------------------------------------------------------------------

Deno.test("overlay: a mounted card is told when its node is selected and deselected", () => {
  const h = harness();
  h.overlay.syncCards([entry(1)]);
  assertEquals(h.overlay.cardCount, 1);

  h.overlay.notify(1, { kind: "selection", selected: true });
  h.overlay.notify(1, { kind: "selection", selected: false });

  assertEquals(h.provider.stateChanges(1), [
    { kind: "selection", selected: true },
    { kind: "selection", selected: false },
  ]);
});

Deno.test("overlay: a mounted card is told when its node is hovered and unhovered", () => {
  const h = harness();
  h.overlay.syncCards([entry(1)]);

  h.overlay.notify(1, { kind: "hover", hovered: true });
  h.overlay.notify(1, { kind: "hover", hovered: false });

  assertEquals(h.provider.stateChanges(1), [
    { kind: "hover", hovered: true },
    { kind: "hover", hovered: false },
  ]);
});

Deno.test("overlay: a state change reaches only the card it names", () => {
  const h = harness();
  h.overlay.syncCards([entry(1), entry(2)]);
  assertEquals(h.overlay.cardCount, 2);

  h.overlay.notify(2, { kind: "data" });

  assertEquals(h.provider.stateChanges(1), []);
  assertEquals(h.provider.stateChanges(2), [{ kind: "data" }]);
});

Deno.test("overlay: notifying an uncarded node is a no-op, not an error", () => {
  const h = harness();
  h.overlay.syncCards([entry(1), ringEntry(2)]);

  // Never carded, carded-then-released, and not in the graph at all: a caller
  // fanning a selection change out over a set must not have to filter it.
  h.overlay.notify(2, { kind: "selection", selected: true });
  h.overlay.notify(99, { kind: "selection", selected: true });
  h.overlay.syncCards([]);
  h.overlay.notify(1, { kind: "selection", selected: true });

  assertEquals(h.provider.updates.filter((u) => u.change.kind === "selection"), []);
});

// -----------------------------------------------------------------------------
// Prefetch epoch turnover
// -----------------------------------------------------------------------------

Deno.test("overlay: a cancelled prefetch is re-offered once the epoch turns", () => {
  const h = harness();

  h.overlay.syncCards([ringEntry(3)]);
  assertEquals(h.provider.offersFor(3).length, 1, "entering the ring warms the node once");
  const first = h.provider.offersFor(3)[0];
  assert(first !== undefined);
  assertEquals(first.signal.aborted, false);

  // Leaving the ring uncarded abandons the fetch and spends the node.
  h.overlay.syncCards([]);
  assertEquals(first.signal.aborted, true, "leaving the ring aborts the fetch");
  h.overlay.syncCards([ringEntry(3)]);
  assertEquals(
    h.provider.offersFor(3).length,
    1,
    "re-entering inside the same epoch must not re-offer: that is the budget",
  );

  // The cut moved, so the previous epoch's spend no longer stands.
  h.overlay.beginEpoch();
  h.overlay.syncCards([ringEntry(3)]);
  assertEquals(
    h.provider.offersFor(3).length,
    2,
    "a new epoch re-arms the node, or prefetch works exactly once per graph",
  );

  const second = h.provider.offersFor(3)[1];
  assert(second !== undefined);
  assertEquals(second.signal.aborted, false, "the re-offer carries a live signal");
});

Deno.test("overlay: turning the epoch abandons warm-up that never became a card", () => {
  const h = harness();

  h.overlay.syncCards([entry(1), ringEntry(4)]);
  const carded = h.provider.offersFor(1);
  const warmed = h.provider.offersFor(4);
  assertEquals(carded.length, 0, "a node that took a card is never offered a prefetch");
  assertEquals(warmed.length, 1);

  h.overlay.beginEpoch();

  const offer = warmed[0];
  assert(offer !== undefined);
  assertEquals(offer.signal.aborted, true, "the fetch is abandoned with the epoch that wanted it");
  assertEquals(h.overlay.cardCount, 1, "cards outlive the epoch: their content is on screen");
});
