/**
 * Card store — the bridge from the provider contract to a declarative renderer.
 *
 * Three claims, and each of them is a thing the obvious implementation gets
 * wrong:
 *
 *  - `release` leaves the container empty *synchronously*, because the pool
 *    drops a container a provider left dirty and no framework unmounts
 *    synchronously;
 *  - placement changes do not republish, because `CardDriver.place` runs for
 *    every card on every frame of camera motion and feeding that into a
 *    declarative renderer would re-render the whole card set at frame rate;
 *  - cards are keyed on the producer's identifier, not the slot, because slots
 *    are recycled and a key that changed hands would carry one node's component
 *    state onto another node's content.
 *
 * Driven through the real {@link CardDriver} and {@link CardContainerPool} over
 * linkedom, so what is asserted is the store's behaviour in the arrangement it
 * actually runs in rather than against a hand-written caller.
 */

import { assert, assertEquals, assertStrictEquals } from "jsr:@std/assert@^1";
import { parseHTML } from "https://esm.sh/linkedom@0.18.11/worker";
import { createCardStore } from "../../packages/core/src/overlay/card_store.ts";
import { CardDriver, type CardPlacement } from "../../packages/core/src/overlay/driver.ts";
import { CardContainerPool } from "../../packages/core/src/overlay/pool.ts";
import type { CardNode, CardSize } from "../../packages/core/src/overlay/types.ts";
import type { Vec2 } from "../../packages/core/src/types.ts";

const CARD_SIZE: CardSize = { width: 200, height: 120 };

function createHost(): HTMLElement {
  const { document } = parseHTML("<html><body><div id='overlay'></div></body></html>");
  return document.getElementById("overlay") as unknown as HTMLElement;
}

function makeCardNode(id: number, externalId: string = `src/mod${id}.ts`): CardNode {
  let pinned = false;
  const position: Vec2 = { x: id * 10, y: id * 20 };
  return {
    id,
    externalId,
    label: externalId,
    tag: 0,
    weight: 0,
    depth: 1,
    contentRef: undefined,
    position: () => position,
    size: () => CARD_SIZE,
    pin: () => {
      pinned = true;
    },
    unpin: () => {
      pinned = false;
    },
    isPinned: () => pinned,
  };
}

function placement(overrides: Partial<CardPlacement> = {}): CardPlacement {
  return { x: 10, y: 20, width: 200, height: 120, scale: 1, opacity: 1, ...overrides };
}

/** The store, wired through a real driver over a real pool. */
function rig(): {
  store: ReturnType<typeof createCardStore>;
  driver: CardDriver<unknown>;
  pool: CardContainerPool;
  host: HTMLElement;
  /** Containers the driver handed the store, in mount order. */
  containers: HTMLElement[];
} {
  const host = createHost();
  const pool = new CardContainerPool(host.ownerDocument);
  const store = createCardStore();
  const containers: HTMLElement[] = [];
  const driver = new CardDriver<unknown>({
    host,
    provider: store.provider,
    createContainer: () => {
      const container = pool.acquire();
      containers.push(container);
      return container;
    },
    recycleContainer: (element) => pool.recycle(element),
  });
  return { store, driver, pool, host, containers };
}

Deno.test("card store: release leaves the container empty so the pool can recycle it", () => {
  const { store, driver, pool, containers } = rig();
  const node = makeCardNode(1);

  assertEquals(driver.mount(node, placement()), true);
  const card = store.getSnapshot()[0];
  // Stands in for the subtree a framework renders into the host and tears down
  // on its own schedule, which is always later than this.
  card.host.appendChild(card.host.ownerDocument.createElement("p"));

  driver.release(node.id);

  assertEquals(
    containers[0].childNodes.length,
    0,
    "the container must be empty by the time release returns",
  );
  assertEquals(pool.idleCount, 1, "so the pool takes it rather than dropping it");
  assertEquals(store.getSnapshot().length, 0);
  assertEquals(card.host.childNodes.length, 1, "and the host stays whole for a later unmount");
});

Deno.test("card store: placement changes do not republish", () => {
  // Load-bearing for frame cost: `place` runs per card per frame of camera
  // motion, so republishing here would re-render every card at frame rate.
  const { store, driver } = rig();
  const node = makeCardNode(1);
  driver.mount(node, placement());

  const mounted = store.getSnapshot();
  driver.place(node.id, placement({ x: 40 }));
  driver.place(node.id, placement({ x: 40, y: 90 }));
  driver.place(node.id, placement({ x: 40, y: 90, scale: 0.5 }));
  assertStrictEquals(store.getSnapshot(), mounted, "a moved or resized card is not a new card");

  driver.notify(node.id, { kind: "selection", selected: true });
  const selected = store.getSnapshot();
  assert(selected !== mounted, "but a selection change is");
  assertEquals(selected[0].selected, true);
});

Deno.test("card store: a repeated state change costs no republish", () => {
  const { store, driver } = rig();
  const node = makeCardNode(1);
  driver.mount(node, placement());

  driver.notify(node.id, { kind: "hover", hovered: true });
  const hovered = store.getSnapshot();
  driver.notify(node.id, { kind: "hover", hovered: true });

  assertStrictEquals(store.getSnapshot(), hovered);
  assertEquals(hovered[0].hovered, true);
});

Deno.test("card store: a data change republishes even though nothing on the record moves", () => {
  // The card reads its content through the live `CardNode`, so nothing on the
  // record changes — but a renderer that was not told would keep showing what
  // it read last.
  const { store, driver } = rig();
  const node = makeCardNode(1);
  driver.mount(node, placement());

  const before = store.getSnapshot();
  driver.notify(node.id, { kind: "data" });

  assert(store.getSnapshot() !== before);
  assertStrictEquals(store.getSnapshot()[0].node, node, "and it is still the same node");
});

Deno.test("card store: cards are keyed on externalId, so a slot changing hands is a new card", () => {
  // A batch removal compacts slots, so a carded slot can come to hold a
  // different node; the overlay releases and re-mounts it in one sync. A store
  // keyed on the slot would hand the new node the old one's component state.
  const { store, driver } = rig();

  driver.mount(makeCardNode(2, "src/mod2.ts"), placement());
  const first = store.getSnapshot()[0];

  driver.release(2);
  driver.mount(makeCardNode(2, "src/mod9.ts"), placement());
  const second = store.getSnapshot()[0];

  assertEquals(first.key, "src/mod2.ts");
  assertEquals(second.key, "src/mod9.ts");
  assert(first.host !== second.host, "and it renders into a host of its own");
});

Deno.test("card store: subscribers are called on every republish and stop on unsubscribe", () => {
  const { store, driver } = rig();
  let calls = 0;
  const unsubscribe = store.subscribe(() => {
    calls++;
  });

  const node = makeCardNode(1);
  driver.mount(node, placement());
  driver.notify(node.id, { kind: "selection", selected: true });
  assertEquals(calls, 2);

  unsubscribe();
  driver.release(node.id);
  assertEquals(calls, 2, "an unsubscribed listener hears nothing");
});

Deno.test("card store: the host adds no layout box between the container and the content", () => {
  // The container is already laid out at the card's size and carries the
  // counter-scale; a wrapper with a box of its own would be one more element
  // between core's geometry and the consumer's CSS.
  const { store, driver } = rig();
  driver.mount(makeCardNode(1), placement());

  assertEquals(store.getSnapshot()[0].host.style.display, "contents");
});
