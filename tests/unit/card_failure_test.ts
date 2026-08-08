/**
 * Card provider failure containment.
 *
 * A provider is consumer code running inside core's per-frame path. Uncontained,
 * one card that throws in `mount` strands a styled, attached container in the
 * overlay forever, and the throw travels out through the LOD controller's tick
 * into the render callback — which the render loop catches and logs, aborting
 * every frame before edges, nodes, compute submit and position readback. One bad
 * card freezes the whole graph while the loop spins.
 *
 * So the driver contains all four hooks, and what is asserted here is that
 * containment leaves the card in a *defined* state rather than merely surviving:
 * no orphan container, no phantom registration, no card wedged on screen, and
 * never a silent catch — a failure with no handler is logged.
 *
 * linkedom supplies the DOM, as in `card_contract_test.ts`.
 */

import { assert, assertEquals, assertStrictEquals } from "jsr:@std/assert@^1";
import { parseHTML } from "https://esm.sh/linkedom@0.18.11/worker";
import {
  CardDriver,
  type CardPlacement,
  type CardProviderFailure,
} from "../../packages/core/src/overlay/driver.ts";
import { CardContainerPool } from "../../packages/core/src/overlay/pool.ts";
import type { CardNode, CardProvider, CardSize } from "../../packages/core/src/overlay/types.ts";
import { ErrorCode } from "../../packages/core/src/errors.ts";
import type { Vec2 } from "../../packages/core/src/types.ts";

const CARD_SIZE: CardSize = { width: 200, height: 120 };

/** The code every containment report carries, as `GraphMotherError` renders it. */
const CODE = ErrorCode.CARD_PROVIDER_FAILED;

function createHost(): HTMLElement {
  const { document } = parseHTML("<html><body><div id='overlay'></div></body></html>");
  return document.getElementById("overlay") as unknown as HTMLElement;
}

function makeCardNode(id: number): CardNode {
  let pinned = false;
  const position: Vec2 = { x: id * 10, y: id * 20 };
  return {
    id,
    externalId: `src/mod${id}.ts`,
    label: `mod${id}.ts`,
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

/** What the provider is asked to do, and which hook it should throw from. */
interface BrokenOptions {
  readonly throwIn: ReadonlySet<"prefetch" | "mount" | "update" | "release">;
  /** Whether `mount` puts a child in the container before it gives up. */
  readonly dirty?: boolean;
}

interface BrokenProvider extends CardProvider<{ readonly nodeId: number }> {
  readonly calls: string[];
}

/**
 * A provider that breaks the contract on demand.
 *
 * `mount` optionally leaves a child behind before throwing, which is the
 * realistic shape of the failure: a provider gets half way through building its
 * content and then hits the error.
 */
function brokenProvider(options: BrokenOptions): BrokenProvider {
  const calls: string[] = [];
  const fail = (hook: "prefetch" | "mount" | "update" | "release"): void => {
    if (options.throwIn.has(hook)) throw new Error(`provider failed in ${hook}`);
  };
  return {
    calls,
    prefetch(node) {
      calls.push(`prefetch:${node.id}`);
      fail("prefetch");
    },
    mount(container, node) {
      calls.push(`mount:${node.id}`);
      if (options.dirty === true) {
        container.appendChild(container.ownerDocument.createElement("p"));
      }
      fail("mount");
      return { nodeId: node.id };
    },
    update(_container, node) {
      calls.push(`update:${node.id}`);
      fail("update");
    },
    release(container, node) {
      calls.push(`release:${node.id}`);
      fail("release");
      while (container.firstChild !== null) container.firstChild.remove();
    },
  };
}

/** A driver over a real pool, with every failure recorded. */
function harness(provider: CardProvider<{ readonly nodeId: number }>): {
  host: HTMLElement;
  pool: CardContainerPool;
  driver: CardDriver<{ readonly nodeId: number }>;
  failures: CardProviderFailure[];
} {
  const host = createHost();
  const pool = new CardContainerPool(host.ownerDocument);
  const failures: CardProviderFailure[] = [];
  const driver = new CardDriver<{ readonly nodeId: number }>({
    host,
    provider,
    createContainer: () => pool.acquire(),
    recycleContainer: (element) => pool.recycle(element),
    onProviderError: (failure) => failures.push(failure),
  });
  return { host, pool, driver, failures };
}

Deno.test("card failure: a provider that throws in mount does not throw at the caller", () => {
  const { driver } = harness(brokenProvider({ throwIn: new Set(["mount"]) }));

  // No assertThrows wrapper: the claim is that the call *returns*, and a throw
  // here fails the test by escaping, which is exactly the defect.
  const mounted = driver.mount(makeCardNode(1), placement());

  assertEquals(mounted, false, "a mount that never completed must report itself as such");
});

Deno.test("card failure: a failed mount leaves no orphan container", () => {
  const { host, pool, driver } = harness(
    brokenProvider({ throwIn: new Set(["mount"]), dirty: true }),
  );

  driver.mount(makeCardNode(1), placement());

  assertEquals(host.children.length, 0, "the container must not be left attached to the overlay");
  assertEquals(driver.size, 0);
  assertEquals(driver.has(1), false, "a card that never mounted must not be registered");
  assertEquals(
    pool.idleCount,
    0,
    "the half-built container still holds the provider's child, so the pool must drop it",
  );
});

Deno.test("card failure: a clean mount after a failed one is unaffected", () => {
  const { host, driver, failures } = harness({
    mount(container, node) {
      if (node.id === 1) throw new Error("provider failed in mount");
      container.appendChild(container.ownerDocument.createElement("p"));
      return { nodeId: node.id };
    },
  });

  driver.mount(makeCardNode(1), placement());
  const mounted = driver.mount(makeCardNode(2), placement());

  assertEquals(mounted, true);
  assertEquals(driver.size, 1);
  assertEquals(host.children.length, 1, "exactly the surviving card is attached");
  assertEquals(failures.length, 1);
});

Deno.test("card failure: a provider that throws in release still gets its container detached", () => {
  const { host, pool, driver, failures } = harness(
    brokenProvider({ throwIn: new Set(["release"]), dirty: true }),
  );
  const node = makeCardNode(1);

  assertEquals(driver.mount(node, placement()), true);
  driver.release(node.id);

  assertEquals(host.children.length, 0, "a card whose teardown threw must still leave the screen");
  assertEquals(driver.has(1), false);
  assertEquals(failures.length, 1);
  assertEquals(failures[0].hook, "release");
  assertEquals(failures[0].released, true);
  assertEquals(pool.idleCount, 0, "the container kept the dead card's content, so it is dropped");

  // And the next card gets a container with nothing of the dead one in it.
  assertEquals(driver.mount(makeCardNode(2), placement()), true);
  assertEquals(host.children[0].children.length, 1);
});

Deno.test("card failure: a provider that throws in update is released, not left wedged", () => {
  const provider = brokenProvider({ throwIn: new Set(["update"]) });
  const { host, driver, failures } = harness(provider);
  const node = makeCardNode(1);

  assertEquals(driver.mount(node, placement()), true);
  driver.place(node.id, placement({ x: 40 }));

  assertEquals(driver.has(1), false, "an update that threw leaves the card in an undefined state");
  assertEquals(host.children.length, 0);

  // `place` runs per card per frame of camera motion, so a card left mounted
  // would throw again immediately — this is the assertion that the frame cost
  // of a broken card is bounded.
  const before = provider.calls.length;
  driver.place(node.id, placement({ x: 80 }));
  assertEquals(provider.calls.length, before, "the released card must not be called again");
  assertEquals(failures.length, 1);
});

Deno.test("card failure: every failure is reported once, with its hook and its node", () => {
  const provider = brokenProvider({
    throwIn: new Set(["prefetch", "mount", "update", "release"]),
  });
  const { driver, failures } = harness(provider);

  driver.prefetch(makeCardNode(7));

  driver.mount(makeCardNode(8), placement());

  // A provider that only throws from `update` and `release`, so the mount that
  // has to precede them succeeds.
  const late = harness(brokenProvider({ throwIn: new Set(["update", "release"]) }));
  const node = makeCardNode(9);
  assertEquals(late.driver.mount(node, placement()), true);
  late.driver.notify(node.id, { kind: "selection", selected: true });

  assertEquals(failures.map((f) => f.hook), ["prefetch", "mount"]);
  assertEquals(failures.map((f) => f.node.id), [7, 8]);
  assertEquals(
    failures.map((f) => f.node.externalId),
    ["src/mod7.ts", "src/mod8.ts"],
  );
  assertEquals(failures.map((f) => f.released), [false, false]);

  // The update failure tears the card down, and the release it runs is the
  // broken one too — so both hooks report, in the order they actually ran.
  assertEquals(late.failures.map((f) => f.hook), ["release", "update"]);
  assertEquals(late.failures.map((f) => f.released), [true, true]);

  for (const failure of [...failures, ...late.failures]) {
    assertEquals(failure.error.code, ErrorCode.CARD_PROVIDER_FAILED);
    assert(failure.cause instanceof Error, "the raw throw must reach the handler");
    assertStrictEquals(failure.error.context.originalError, failure.cause);
  }
});

Deno.test("card failure: the wrapped error names the hook that actually threw", () => {
  // `error.message` is public surface — it is what `card:error` carries into a
  // consumer's telemetry and what the no-handler path logs — and it is built at
  // a different call site for each hook. Pinning only `failure.hook` leaves the
  // message free to name a callback that never ran, and `mount` is the value it
  // would be wrong about least visibly, since it is the one every other test
  // exercises.
  const early = harness(brokenProvider({ throwIn: new Set(["prefetch"]) }));
  early.driver.prefetch(makeCardNode(7));

  const late = harness(brokenProvider({ throwIn: new Set(["update", "release"]) }));
  const node = makeCardNode(9);
  assertEquals(late.driver.mount(node, placement()), true);
  late.driver.notify(node.id, { kind: "hover", hovered: true });

  // The update failure tears the card down through the broken release, so both
  // report — in the order they ran, which is what makes this a two-hook case
  // rather than two one-hook cases.
  const expected: readonly [CardProviderFailure, string][] = [
    [early.failures[0], "prefetch"],
    [late.failures[0], "release"],
    [late.failures[1], "update"],
  ];
  assertEquals(expected.length, 3, "each hook under test must have reported exactly once");

  for (const [failure, hook] of expected) {
    assertEquals(failure.hook, hook);
    assertEquals(
      failure.error.context["hook"],
      hook,
      `the error context must name ${hook}, not whichever hook was hard-coded`,
    );
    assert(
      failure.error.message.includes(`${hook}()`),
      `the message must name ${hook}(), got: ${failure.error.message}`,
    );
  }
});

Deno.test("card failure: a failure with no handler is logged, never swallowed", () => {
  const host = createHost();
  const driver = new CardDriver<{ readonly nodeId: number }>({
    host,
    provider: brokenProvider({ throwIn: new Set(["mount"]) }),
  });

  const logs: string[] = [];
  const original = console.error;
  console.error = (...args: unknown[]): void => {
    logs.push(args.map(String).join(" "));
  };
  try {
    driver.mount(makeCardNode(7), placement());
  } finally {
    console.error = original;
  }

  // The stack goes to the console too and repeats the message, so the report
  // itself is identified by the code line rather than by the text it shares.
  const reports = logs.filter((line) => line.startsWith(`[GraphMother Error ${CODE}]`));
  assertEquals(reports.length, 1, "a contained failure must be reported exactly once");
  assert(reports[0].includes("mount()"), "the report must name the hook that threw");
  assert(reports[0].includes("src/mod7.ts"), "the report must name the node");
});

Deno.test("card failure: the logged report names a non-mount hook too", () => {
  // The mount case above is satisfied by any implementation that always says
  // "mount", which is exactly the defect worth catching here: the log is the
  // only signal a consumer who registered no handler ever gets, so a report
  // naming the wrong callback sends them to read the wrong function.
  const host = createHost();
  const driver = new CardDriver<{ readonly nodeId: number }>({
    host,
    provider: brokenProvider({ throwIn: new Set(["release"]) }),
  });
  const node = makeCardNode(7);
  assertEquals(driver.mount(node, placement()), true);

  const logs: string[] = [];
  const original = console.error;
  console.error = (...args: unknown[]): void => {
    logs.push(args.map(String).join(" "));
  };
  try {
    driver.release(node.id);
  } finally {
    console.error = original;
  }

  const reports = logs.filter((line) => line.startsWith(`[GraphMother Error ${CODE}]`));
  assertEquals(reports.length, 1);
  // Anchored on "threw in": the suggestion the report carries names hooks too,
  // so a bare substring would pass on text that has nothing to do with the throw.
  assert(
    reports[0].includes("threw in release()"),
    `the report must name release(), got: ${reports[0]}`,
  );
  assert(
    !reports[0].includes("threw in mount()"),
    "and must not name a hook that did not throw",
  );
});
