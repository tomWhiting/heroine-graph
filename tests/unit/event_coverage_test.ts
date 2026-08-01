/**
 * Every declared event can actually be fired.
 *
 * `background:click` shipped as a typed event with a factory, three framework
 * wrappers forwarding it and a demo listening for it — and no emit site
 * anywhere, so the documented "click the background to clear" did nothing at
 * all. It is a defect class rather than a slip: `simulation:tick` and
 * `simulation:end` had the same shape and were fixed the same way. A consumer
 * cannot tell a never-fired event from an event that has not happened yet, so
 * nothing surfaces it but this.
 *
 * The check is over the source because emitting these needs a canvas and real
 * pointer events; what it pins is that a call site exists at all, not that it
 * fires at the right moment. The pointer sequence itself is browser-smoke work.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";

const CORE = new URL("../../packages/core/src/", import.meta.url);

async function read(path: string): Promise<string> {
  return await Deno.readTextFile(new URL(path, CORE));
}

const TYPES = await read("types.ts");
const EMITTER = await read("events/emitter.ts");

/** Every key of the public `EventMap`. */
function declaredEventTypes(): string[] {
  const map = TYPES.match(/export interface EventMap \{(.*?)\n\}/s);
  assert(map, "EventMap not found in types.ts");
  const keys = [...map[1].matchAll(/"([^"]+)":/g)].map((m) => m[1]);
  assert(keys.length > 20, `expected the full event map, found ${keys.length} keys`);
  return keys;
}

/** Factory name to the event type it builds, from the `Events` object. */
function factoryEventTypes(): Map<string, string> {
  const object = EMITTER.match(/export const Events = \{(.*?)\n\};/s);
  assert(object, "Events object not found in emitter.ts");
  const factories = new Map<string, string>();
  for (const match of object[1].matchAll(/\n {2}(\w+)\(/g)) {
    const tail = object[1].slice(match.index! + match[0].length);
    const type = tail.match(/type: "([^"]+)"/);
    if (type) factories.set(match[1], type[1]);
  }
  return factories;
}

/** Core, minus the two files that only declare events. */
async function emittingSource(dir: URL = CORE): Promise<string> {
  const parts: string[] = [];
  for await (const entry of Deno.readDir(dir)) {
    const path = new URL(entry.name + (entry.isDirectory ? "/" : ""), dir);
    if (entry.isDirectory) {
      parts.push(await emittingSource(path));
    } else if (
      entry.name.endsWith(".ts") && entry.name !== "types.ts" && entry.name !== "emitter.ts"
    ) {
      parts.push(await Deno.readTextFile(path));
    }
  }
  return parts.join("\n");
}

const SOURCE = await emittingSource();
const FACTORIES = factoryEventTypes();

/** Whether anything builds this event, by factory or by literal. */
function isEmitted(eventType: string): boolean {
  if (SOURCE.includes(`type: "${eventType}"`)) return true;
  for (const [factory, built] of FACTORIES) {
    if (built === eventType && SOURCE.includes(`Events.${factory}(`)) return true;
  }
  return false;
}

/**
 * Declared events with no producer, as of this lane's close.
 *
 * Not an accepted state — an outstanding finding, handed on rather than fixed
 * here because deciding when an edge click fires (and whether it displaces the
 * background click on the same release) is a decision about the interaction
 * model, not a bug fix. The assertion below fails when one is fixed, so the
 * list can only shrink.
 */
const UNEMITTED = ["edge:click"];

Deno.test("events: every declared event type has a producer", () => {
  const missing = declaredEventTypes().filter((t) => !isEmitted(t) && !UNEMITTED.includes(t));
  assertEquals(missing, [], "declared events that nothing can ever emit");
});

Deno.test("events: the known-unemitted list has not gone stale", () => {
  const fixed = UNEMITTED.filter((t) => isEmitted(t));
  assertEquals(fixed, [], "these now have a producer and must leave UNEMITTED");
});

Deno.test("events: background:click fires on a click, not on a pan or a node click", () => {
  const graph = Deno.readTextFileSync(new URL("api/graph.ts", CORE));
  const pointerUp = graph.slice(graph.indexOf('this.pointerManager.on("pointerup"'));
  const emit = pointerUp.indexOf("Events.backgroundClick(");
  assert(emit > 0, "background:click has no emit site in the pointerup handler");

  const guard = pointerUp.slice(0, emit);
  assert(guard.includes("nodeId === null"), "a click that hit a node would also emit");
  assert(guard.includes("isClickDistance("), "a pan would also emit");
});
