/**
 * DOM card contract.
 *
 * The card contract is a set of promises made to a consumer whose code core
 * cannot see, so every one of them is asserted here against a recording
 * provider: mount exactly once, placement writes confined to the four
 * core-owned style properties, one typed change per update, release strictly
 * before detach, and prefetch advisory, abortable and capped at once per node
 * per LOD epoch.
 *
 * The load-bearing one is "core never touches container children" (SC-004):
 * a consumer's card content — a framework subtree, selected text, an open
 * find-in-page match — must survive every reposition core performs.
 *
 * linkedom supplies the DOM; the repo has no browser test runner and the
 * driver needs nothing beyond element creation, attachment and inline styles.
 */

import { assert, assertEquals, assertStrictEquals, assertThrows } from "jsr:@std/assert@^1";
// linkedom's `/worker` entry point, by full specifier so the test needs no
// import-map or node_modules entry. It is the only DOM implementation Deno can
// reach that models inline `style` — which is exactly what the driver writes.
import { parseHTML } from "https://esm.sh/linkedom@0.18.11/worker";
import { CardDriver, type CardPlacement } from "../../packages/core/src/overlay/driver.ts";
import { createDefaultCardProvider } from "../../packages/core/src/overlay/default_card.ts";
import type {
  CardChange,
  CardNode,
  CardProvider,
  CardSize,
} from "../../packages/core/src/overlay/types.ts";
import type { Vec2 } from "../../packages/core/src/types.ts";

/**
 * linkedom implements the DOM surface the driver uses but types it with its
 * own classes, so the boundary is re-typed structurally here. Nothing below
 * depends on anything lib.dom declares that linkedom does not implement.
 */
function createHost(): HTMLElement {
  const { document } = parseHTML("<html><body><div id='overlay'></div></body></html>");
  return document.getElementById("overlay") as unknown as HTMLElement;
}

const CARD_SIZE: CardSize = { width: 200, height: 120 };

function makeCardNode(id: number, overrides: Partial<CardNode> = {}): CardNode {
  let pinned = false;
  const position: Vec2 = { x: id * 10, y: id * 20 };
  return {
    id,
    externalId: `src/mod${id}.ts`,
    label: `mod${id}.ts`,
    tag: 3,
    weight: 0.5,
    depth: 2,
    contentRef: `sha256:${id}`,
    position: () => position,
    size: () => CARD_SIZE,
    pin: () => {
      pinned = true;
    },
    unpin: () => {
      pinned = false;
    },
    isPinned: () => pinned,
    ...overrides,
  };
}

function placement(overrides: Partial<CardPlacement> = {}): CardPlacement {
  return { x: 10, y: 20, width: 200, height: 120, scale: 1, opacity: 1, ...overrides };
}

interface Call {
  readonly hook: "prefetch" | "mount" | "update" | "release";
  readonly nodeId: number;
  readonly change?: CardChange;
  /** Whether the container was still in the DOM when the hook ran. */
  readonly connected?: boolean;
}

interface RecordingProvider extends CardProvider<{ readonly nodeId: number }> {
  readonly calls: Call[];
  readonly signals: Map<number, AbortSignal>;
  /** Hooks, by name, for the counts assertions to read. */
  count(hook: Call["hook"]): number;
}

function recordingProvider(): RecordingProvider {
  const calls: Call[] = [];
  const signals = new Map<number, AbortSignal>();
  return {
    calls,
    signals,
    count(hook) {
      return calls.filter((c) => c.hook === hook).length;
    },
    prefetch(node, signal) {
      calls.push({ hook: "prefetch", nodeId: node.id });
      signals.set(node.id, signal);
    },
    mount(container, node) {
      calls.push({ hook: "mount", nodeId: node.id, connected: container.isConnected });
      return { nodeId: node.id };
    },
    update(container, node, change) {
      calls.push({
        hook: "update",
        nodeId: node.id,
        change,
        connected: container.isConnected,
      });
    },
    release(container, node) {
      calls.push({ hook: "release", nodeId: node.id, connected: container.isConnected });
    },
  };
}

Deno.test("card contract: mount runs exactly once, on an attached and placed container", () => {
  const host = createHost();
  const provider = recordingProvider();
  const driver = new CardDriver({ host, provider, cardClassName: "gm-card" });
  const node = makeCardNode(7);

  driver.mount(node, placement());

  assertEquals(provider.count("mount"), 1);
  assertEquals(provider.calls[0].connected, true, "mount sees the container in the DOM");
  assertEquals(driver.size, 1);
  assert(driver.has(7));

  const container = host.children[0] as unknown as HTMLElement;
  assertEquals(container.className, "gm-card");
  assertEquals(container.style.transform, "translate(10px, 20px) scale(1)");
  assertEquals(container.style.width, "200px");
  assertEquals(container.style.height, "120px");
  assertEquals(container.style.opacity, "1");

  // Re-placing, re-styling and re-notifying must never re-run mount
  driver.place(7, placement({ x: 40 }));
  driver.notify(7, { kind: "data" });
  assertEquals(provider.count("mount"), 1);

  // A second mount of the same node is a caller bug, not a silent no-op
  assertThrows(() => driver.mount(node, placement()), Error, "already mounted");
  assertEquals(provider.count("mount"), 1);
});

Deno.test("card contract: core never touches container children (SC-004)", () => {
  const host = createHost();
  const provider = recordingProvider();
  const driver = new CardDriver({ host, provider });
  const node = makeCardNode(1);

  driver.mount(node, placement());
  const container = host.children[0] as unknown as HTMLElement;

  // Stand in for consumer-owned content: a framework subtree, a text
  // selection, an open find-in-page match.
  const content = '<p class="body">export function main() {}</p>';
  container.innerHTML = content;

  driver.place(1, placement({ x: 512, y: 640 }));
  driver.place(1, placement({ x: 512, y: 640, width: 320, height: 200 }));
  driver.place(1, placement({ x: 512, y: 640, width: 320, height: 200, opacity: 0.25 }));
  driver.notify(1, { kind: "hover", hovered: true });
  driver.notify(1, { kind: "selection", selected: true });
  driver.notify(1, { kind: "data" });

  assertEquals(container.innerHTML, content, "consumer content survives every core write");
  // ...while core's own four properties did move
  assertEquals(container.style.transform, "translate(512px, 640px) scale(1)");
  assertEquals(container.style.width, "320px");
  assertEquals(container.style.height, "200px");
  assertEquals(container.style.opacity, "0.25");
});

Deno.test("card contract: place reports one typed change per axis, and nothing when idle", () => {
  const host = createHost();
  const provider = recordingProvider();
  const driver = new CardDriver({ host, provider });

  driver.mount(makeCardNode(2), placement());
  provider.calls.length = 0;

  driver.place(2, placement({ x: 99 }));
  assertEquals(provider.calls.map((c) => c.change), [{ kind: "position" }]);

  provider.calls.length = 0;
  driver.place(2, placement({ x: 99, width: 300 }));
  assertEquals(provider.calls.map((c) => c.change), [{ kind: "size" }]);

  provider.calls.length = 0;
  driver.place(2, placement({ x: 1, y: 2, width: 10, height: 20 }));
  assertEquals(provider.calls.map((c) => c.change), [{ kind: "position" }, { kind: "size" }]);

  // An unchanged placement is the per-frame steady state: no callback at all
  provider.calls.length = 0;
  driver.place(2, placement({ x: 1, y: 2, width: 10, height: 20 }));
  assertEquals(provider.calls, []);

  // Opacity is core-owned; it moves the crossfade without raising a change
  driver.place(2, placement({ x: 1, y: 2, width: 10, height: 20, opacity: 0.5 }));
  assertEquals(provider.calls, []);

  // Unmounted nodes are ignored rather than fabricating a card
  driver.place(404, placement());
  driver.notify(404, { kind: "data" });
  assertEquals(provider.calls, []);
});

Deno.test("card contract: notify forwards interaction state verbatim", () => {
  const host = createHost();
  const provider = recordingProvider();
  const driver = new CardDriver({ host, provider });

  driver.mount(makeCardNode(3), placement());
  provider.calls.length = 0;

  driver.notify(3, { kind: "hover", hovered: true });
  driver.notify(3, { kind: "selection", selected: false });
  driver.notify(3, { kind: "data" });

  assertEquals(provider.calls.map((c) => c.change), [
    { kind: "hover", hovered: true },
    { kind: "selection", selected: false },
    { kind: "data" },
  ]);
});

Deno.test("card contract: release runs before the container leaves the DOM", () => {
  const host = createHost();
  const provider = recordingProvider();
  const recycled: HTMLElement[] = [];
  const driver = new CardDriver({
    host,
    provider,
    recycleContainer: (container) => recycled.push(container),
  });

  driver.mount(makeCardNode(4), placement());
  const container = host.children[0] as unknown as HTMLElement;
  provider.calls.length = 0;

  driver.release(4);

  assertEquals(provider.calls.length, 1);
  assertEquals(provider.calls[0].hook, "release");
  assertEquals(
    provider.calls[0].connected,
    true,
    "release must be able to read layout, so it runs before detach",
  );
  assertEquals(container.isConnected, false, "container is detached once release returns");
  assertEquals(host.children.length, 0);
  assertEquals(driver.size, 0);
  assertEquals(recycled.length, 1);
  assertStrictEquals(recycled[0], container);

  // Releasing again is a no-op, not a second release callback
  driver.release(4);
  assertEquals(provider.count("release"), 1);
});

Deno.test("card contract: prefetch fires at most once per node per epoch", () => {
  const host = createHost();
  const provider = recordingProvider();
  const driver = new CardDriver({ host, provider });
  const node = makeCardNode(5);

  driver.prefetch(node);
  driver.prefetch(node);
  driver.prefetch(node);
  assertEquals(provider.count("prefetch"), 1, "repeat offers within an epoch are dropped");

  // Cancelling does not re-arm the node: prefetch is advisory, and a
  // threshold wobble must not turn into repeated fetches.
  driver.cancelPrefetch(5);
  driver.prefetch(node);
  assertEquals(provider.count("prefetch"), 1);

  // A new epoch does re-arm it
  driver.beginEpoch();
  driver.prefetch(node);
  assertEquals(provider.count("prefetch"), 2);
});

Deno.test("card contract: prefetch signals abort on cancel, release and epoch end", () => {
  const host = createHost();
  const provider = recordingProvider();
  const driver = new CardDriver({ host, provider });

  // (a) node leaves the ring without being carded
  driver.prefetch(makeCardNode(10));
  const cancelled = provider.signals.get(10)!;
  assertEquals(cancelled.aborted, false);
  driver.cancelPrefetch(10);
  assertEquals(cancelled.aborted, true);

  // (b) the card is released
  const carded = makeCardNode(11);
  driver.prefetch(carded);
  driver.mount(carded, placement());
  const mounted = provider.signals.get(11)!;
  assertEquals(mounted.aborted, false, "mounting must not abort the fetch feeding the card");
  driver.release(11);
  assertEquals(mounted.aborted, true);

  // (c) the epoch ends — but a mounted card's content is still on screen
  const pending = makeCardNode(12);
  const onScreen = makeCardNode(13);
  driver.prefetch(pending);
  driver.prefetch(onScreen);
  driver.mount(onScreen, placement());
  driver.beginEpoch();
  assertEquals(provider.signals.get(12)!.aborted, true);
  assertEquals(provider.signals.get(13)!.aborted, false);
});

Deno.test("card contract: releaseAll tears down every card and every prefetch", () => {
  const host = createHost();
  const provider = recordingProvider();
  const driver = new CardDriver({ host, provider });

  for (const id of [20, 21, 22]) driver.mount(makeCardNode(id), placement());
  driver.prefetch(makeCardNode(23));
  assertEquals(host.children.length, 3);

  driver.releaseAll();

  assertEquals(provider.count("release"), 3);
  assertEquals(driver.size, 0);
  assertEquals(host.children.length, 0);
  assertEquals(provider.signals.get(23)!.aborted, true);
});

Deno.test("card contract: a provider without optional hooks is driven without error", () => {
  const host = createHost();
  const mounted: number[] = [];
  const driver = new CardDriver<null>({
    host,
    provider: {
      mount(_container, node) {
        mounted.push(node.id);
        return null;
      },
    },
  });

  driver.prefetch(makeCardNode(30));
  driver.mount(makeCardNode(30), placement());
  driver.place(30, placement({ x: 5 }));
  driver.notify(30, { kind: "data" });
  driver.release(30);

  assertEquals(mounted, [30]);
  assertEquals(host.children.length, 0);
});

Deno.test("default card renderer: renders the label and the tag/weight/depth readouts", () => {
  const host = createHost();
  const driver = new CardDriver({ host, provider: createDefaultCardProvider() });

  driver.mount(makeCardNode(40), placement());
  const container = host.children[0] as unknown as HTMLElement;

  assertEquals(container.textContent, "mod40.tstag3weight0.50depth2");
  assertEquals(container.children.length, 1, "the renderer adds exactly one root child");

  // Falls back to the producer identifier when the node carries no label
  const unlabelled = makeCardNode(41, { label: undefined });
  driver.mount(unlabelled, placement());
  const second = host.children[1] as unknown as HTMLElement;
  assert(second.textContent?.startsWith("src/mod41.ts"));
});

Deno.test("default card renderer: selection outranks hover and data refreshes readouts", () => {
  const host = createHost();
  const driver = new CardDriver({ host, provider: createDefaultCardProvider() });
  // A live view, as the real CardNode is: `data` must re-read, not replay.
  const live = { weight: 0.5 };
  const node: CardNode = {
    ...makeCardNode(42),
    get weight() {
      return live.weight;
    },
  };

  driver.mount(node, placement());
  const root = (host.children[0] as unknown as HTMLElement).children[0] as unknown as HTMLElement;
  const idle = root.style.borderColor;

  driver.notify(42, { kind: "hover", hovered: true });
  const hovered = root.style.borderColor;
  assert(hovered !== idle);

  driver.notify(42, { kind: "selection", selected: true });
  const selected = root.style.borderColor;
  assert(selected !== hovered);

  // Hover ending while still selected must not clear the selected border
  driver.notify(42, { kind: "hover", hovered: false });
  assertEquals(root.style.borderColor, selected);

  driver.notify(42, { kind: "selection", selected: false });
  assertEquals(root.style.borderColor, idle);

  live.weight = 0.875;
  driver.notify(42, { kind: "data" });
  assertEquals(
    (host.children[0] as unknown as HTMLElement).textContent,
    "mod42.tstag3weight0.88depth2",
  );
});

Deno.test("default card renderer: release removes the children it added", () => {
  const host = createHost();
  const pooled: HTMLElement[] = [];
  const driver = new CardDriver({
    host,
    provider: createDefaultCardProvider(),
    recycleContainer: (container) => pooled.push(container),
  });

  driver.mount(makeCardNode(43), placement());
  const container = host.children[0] as unknown as HTMLElement;
  assertEquals(container.children.length, 1);

  driver.release(43);

  // Containers are pooled, so a provider that leaves children behind leaks
  // them into the next card mounted on the same element.
  assertStrictEquals(pooled[0], container);
  assertEquals(container.children.length, 0);
  assertEquals(container.innerHTML, "");
});
