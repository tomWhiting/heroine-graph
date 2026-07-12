/**
 * Type style resolution semantics.
 *
 * These tests pin the CORRECT contract for style resolution:
 *  - resolving one type must never mutate the shared defaults, so an
 *    opacity-only edge style cannot bleed its alpha into the default
 *    style or into other types;
 *  - unregistered types resolve to the documented defaults (the
 *    "untyped preservation" contract applyTypeStyles builds on: it can
 *    only preserve per-element attributes if resolution of other types
 *    leaves defaults untouched).
 */

import { assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1";
import {
  createTypeStyleManager,
  TypeStyleManager,
} from "../../packages/core/src/styling/type_styles.ts";

Deno.test("resolveNodeStyle: undefined and unregistered types get the default style", () => {
  const manager = createTypeStyleManager();
  manager.setNodeTypeStyles({ dir: { color: "#ff0000", size: 2 } });

  const untyped = manager.resolveNodeStyle(undefined);
  assertEquals(untyped.color, [0.5, 0.5, 0.5, 1.0]);
  assertEquals(untyped.size, 1.0);

  const unregistered = manager.resolveNodeStyle("mystery");
  assertEquals(unregistered.color, [0.5, 0.5, 0.5, 1.0]);
  assertEquals(unregistered.size, 1.0);
});

Deno.test("resolveNodeStyle: registered type resolves color and size", () => {
  const manager = createTypeStyleManager();
  manager.setNodeTypeStyles({
    dir: { color: "#ff0000", size: 2 },
    file: { size: 0.5 }, // no color: falls back to default color
  });

  const dir = manager.resolveNodeStyle("dir");
  assertEquals(dir.color, [1, 0, 0, 1]);
  assertEquals(dir.size, 2);

  const file = manager.resolveNodeStyle("file");
  assertEquals(file.color, [0.5, 0.5, 0.5, 1.0]);
  assertEquals(file.size, 0.5);
});

Deno.test("resolveEdgeStyle: registered type resolves color, width, opacity", () => {
  const manager = createTypeStyleManager();
  manager.setEdgeTypeStyles({
    imports: { color: "#00ff00", width: 3, opacity: 0.8 },
  });

  const style = manager.resolveEdgeStyle("imports");
  assertEquals(style.color[0], 0);
  assertEquals(style.color[1], 1);
  assertEquals(style.color[2], 0);
  assertAlmostEquals(style.color[3], 0.8); // opacity overrides parsed alpha
  assertEquals(style.width, 3);
});

Deno.test("resolveEdgeStyle: opacity-only style must not mutate the shared default", () => {
  const manager = createTypeStyleManager();
  // "faint" sets ONLY opacity — its resolved color falls back to the
  // default edge color, and applying the opacity must not write through
  // into the module-level DEFAULT_EDGE_STYLE.
  manager.setEdgeTypeStyles({ faint: { opacity: 0.05 } });

  const faint = manager.resolveEdgeStyle("faint");
  assertEquals(faint.color[0], 0.5);
  assertAlmostEquals(faint.color[3], 0.05);

  // The default style (untyped edges) must still have its documented
  // alpha of 0.4 — not the 0.05 leaked from "faint".
  const untyped = manager.resolveEdgeStyle(undefined);
  assertEquals(untyped.color, [0.5, 0.5, 0.5, 0.4]);

  // Unregistered types resolve to the same untouched default.
  const unregistered = manager.resolveEdgeStyle("unknown-type");
  assertEquals(unregistered.color, [0.5, 0.5, 0.5, 0.4]);

  // And a completely fresh manager must also see pristine defaults
  // (DEFAULT_EDGE_STYLE is module-level, shared across instances).
  const fresh = new TypeStyleManager();
  assertEquals(fresh.resolveEdgeStyle(undefined).color, [0.5, 0.5, 0.5, 0.4]);
});

Deno.test("resolveEdgeStyle: opacity-only styles of different types stay independent", () => {
  const manager = createTypeStyleManager();
  manager.setEdgeTypeStyles({
    ghost: { opacity: 0.1 },
    bold: { opacity: 0.9 },
  });

  const ghost = manager.resolveEdgeStyle("ghost");
  const bold = manager.resolveEdgeStyle("bold");
  assertAlmostEquals(ghost.color[3], 0.1);
  assertAlmostEquals(bold.color[3], 0.9);

  // Re-resolving (cached path) must return the same values — the two
  // types cannot alias one underlying color array.
  assertAlmostEquals(manager.resolveEdgeStyle("ghost").color[3], 0.1);
  assertAlmostEquals(manager.resolveEdgeStyle("bold").color[3], 0.9);
});

Deno.test("TypeStyleManager: setting a single style invalidates only its cache entry", () => {
  const manager = createTypeStyleManager();
  manager.setNodeTypeStyles({ dir: { color: "#ff0000" } });

  assertEquals(manager.resolveNodeStyle("dir").color, [1, 0, 0, 1]);
  manager.setNodeTypeStyle("dir", { color: "#0000ff" });
  assertEquals(manager.resolveNodeStyle("dir").color, [0, 0, 1, 1]);

  manager.setEdgeTypeStyle("imports", { width: 5 });
  assertEquals(manager.resolveEdgeStyle("imports").width, 5);
  manager.setEdgeTypeStyle("imports", { width: 7 });
  assertEquals(manager.resolveEdgeStyle("imports").width, 7);
});

Deno.test("TypeStyleManager: bookkeeping (has/get/clear/types)", () => {
  const manager = createTypeStyleManager();
  assertEquals(manager.hasNodeStyles(), false);
  assertEquals(manager.hasEdgeStyles(), false);

  manager.setNodeTypeStyles({ dir: {}, file: {} });
  manager.setEdgeTypeStyles({ contains: {} });
  assertEquals(manager.hasNodeStyles(), true);
  assertEquals(manager.hasEdgeStyles(), true);
  assertEquals(manager.getNodeTypes().sort(), ["dir", "file"]);
  assertEquals(manager.getEdgeTypes(), ["contains"]);
  assertEquals(manager.getNodeTypeStyle("dir"), {});
  assertEquals(manager.getNodeTypeStyle("nope"), undefined);

  manager.clear();
  assertEquals(manager.hasNodeStyles(), false);
  assertEquals(manager.getEdgeTypes(), []);
});
