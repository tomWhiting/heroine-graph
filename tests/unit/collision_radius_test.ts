/**
 * The collision radius column, and what a collapsed proxy may not do to it.
 *
 * A proxy borrows the attribute row's radius to render its bubble, which is the
 * one place a render value and a physics value share storage. Reading that row
 * straight into `nodeSizes` gives the bubble a physical presence it was never
 * meant to have — a directory folded at wellRadius 800 shoves every node within
 * 800 units away — and, because the maximum sizes the spatial-hash cell, one
 * inflated read used to coarsen the grid for the rest of the session.
 *
 * The table here is the real `ProxyRadiusTable` driven exactly as
 * `setCollapsedProxies` drives it, so the lookup under test is the one the
 * shipped path uses.
 */

import { assertEquals } from "jsr:@std/assert@^1";
import {
  collisionRadiusColumn,
  DEFAULT_COLLISION_RADIUS,
  NODE_ATTR_FLOATS,
} from "../../packages/core/src/api/graph_state.ts";
import {
  type ProxyRadiusHost,
  ProxyRadiusTable,
} from "../../packages/core/src/lod/proxy_radius.ts";
import type { IdLike } from "../../packages/core/src/graph/id_map.ts";
import { DEAD_SLOT_RADIUS } from "../helpers/gpu.ts";

/** An attribute shadow and the host reads/writes the proxy table performs on it. */
class Attributes implements ProxyRadiusHost {
  readonly rows: Float32Array;

  constructor(radii: readonly number[]) {
    this.rows = new Float32Array(radii.length * NODE_ATTR_FLOATS);
    radii.forEach((r, slot) => {
      this.rows[slot * NODE_ATTR_FLOATS] = r;
    });
  }

  get count(): number {
    return this.rows.length / NODE_ATTR_FLOATS;
  }
  radiusOf(slot: number): number {
    return this.rows[slot * NODE_ATTR_FLOATS];
  }
  externalIdOf(slot: number): IdLike | undefined {
    return `node-${slot}`;
  }
  slotOf(externalId: IdLike): number | undefined {
    const slot = Number(String(externalId).slice("node-".length));
    return Number.isInteger(slot) ? slot : undefined;
  }
  setRadius(slot: number, radius: number): boolean {
    const index = slot * NODE_ATTR_FLOATS;
    if (this.rows[index] === radius) return false;
    this.rows[index] = radius;
    return true;
  }
}

const NOTHING_DEAD = () => false;

function column(attrs: Attributes, proxies: ProxyRadiusTable, isDead: (slot: number) => boolean) {
  return collisionRadiusColumn(attrs.rows, attrs.count, {
    isDead,
    proxyRadius: (slot) => proxies.savedRadiusOf(slot),
    deadRadius: DEAD_SLOT_RADIUS,
  });
}

Deno.test("collision radii: a proxy collides at its own radius, not at its bubble", () => {
  const attrs = new Attributes([5, 4, 6, 5]);
  const proxies = new ProxyRadiusTable();
  proxies.declare(Uint32Array.of(0), 1, Float32Array.of(800), attrs);

  // The render row really is inflated — this is the state the reader faces.
  assertEquals(attrs.radiusOf(0), 800);

  const { sizes, maxRadius } = column(attrs, proxies, NOTHING_DEAD);
  assertEquals([...sizes], [5, 4, 6, 5]);
  assertEquals(maxRadius, 6);
});

Deno.test("collision radii: the maximum comes back down when the proxy expands", () => {
  const attrs = new Attributes([5, 4, 6, 5]);
  const proxies = new ProxyRadiusTable();

  proxies.declare(Uint32Array.of(0), 1, Float32Array.of(800), attrs);
  assertEquals(column(attrs, proxies, NOTHING_DEAD).maxRadius, 6);

  // Expanding restores the row; the next flush must not still be sizing the
  // grid for a bubble that is no longer drawn.
  proxies.release(attrs);
  const expanded = column(attrs, proxies, NOTHING_DEAD);
  assertEquals(attrs.radiusOf(0), 5);
  assertEquals([...expanded.sizes], [5, 4, 6, 5]);
  assertEquals(expanded.maxRadius, 6);
});

Deno.test("collision radii: a nested collapse leaves no slot standing at a bubble radius", () => {
  const attrs = new Attributes([5, 4, 6, 5]);
  const proxies = new ProxyRadiusTable();

  // Two proxies at once, then a re-declare that keeps only the second: the
  // complete-set contract means slot 0 is restored by the same call.
  proxies.declare(Uint32Array.of(0, 2), 2, Float32Array.of(800, 120), attrs);
  assertEquals([...column(attrs, proxies, NOTHING_DEAD).sizes], [5, 4, 6, 5]);

  proxies.declare(Uint32Array.of(2), 1, Float32Array.of(120), attrs);
  const { sizes, maxRadius } = column(attrs, proxies, NOTHING_DEAD);
  assertEquals([...sizes], [5, 4, 6, 5]);
  assertEquals(maxRadius, 6);
});

Deno.test("collision radii: dead slots carry the sentinel and never set the maximum", () => {
  const attrs = new Attributes([5, 4, 900, 5]);
  const proxies = new ProxyRadiusTable();
  const dead = (slot: number) => slot === 2;

  const { sizes, maxRadius } = column(attrs, proxies, dead);
  assertEquals([...sizes], [5, 4, DEAD_SLOT_RADIUS, 5]);
  assertEquals(maxRadius, 5);
});

Deno.test("collision radii: a live slot with no radius set gets the default", () => {
  const attrs = new Attributes([0, 0]);
  const proxies = new ProxyRadiusTable();

  const { sizes, maxRadius } = column(attrs, proxies, NOTHING_DEAD);
  assertEquals([...sizes], [DEFAULT_COLLISION_RADIUS, DEFAULT_COLLISION_RADIUS]);
  assertEquals(maxRadius, DEFAULT_COLLISION_RADIUS);
});
