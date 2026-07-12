/**
 * LayerManager render-list membership tests.
 *
 * Regression coverage for the cached-render-list bug: layers whose
 * `enabled` flag is mutated directly (bypassing the manager, as
 * graph.ts's enableHeatmapLayer does) must still join/leave the render
 * list returned by getSortedLayers().
 */

import { assertEquals } from "jsr:@std/assert@^1";
import { createLayerManager } from "../../packages/core/src/layers/manager.ts";
import type { Layer } from "../../packages/core/src/layers/types.ts";

function makeLayer(id: string, order = 0, enabled = true): Layer {
  return {
    id,
    type: "test",
    enabled,
    order,
    render() {},
    resize() {},
    destroy() {},
  };
}

function ids(layers: Layer[]): string[] {
  return layers.map((l) => l.id);
}

Deno.test("LayerManager: re-enabling a layer directly re-adds it to the render list", () => {
  const manager = createLayerManager();
  const heatmap = makeLayer("heatmap", 0);
  const labels = makeLayer("labels", 100);
  manager.addLayer(heatmap);
  manager.addLayer(labels);

  assertEquals(ids(manager.getSortedLayers()), ["heatmap", "labels"]);

  // Disable via the manager, then render a frame (caches the list)
  manager.disableLayer("heatmap");
  assertEquals(ids(manager.getSortedLayers()), ["labels"]);

  // Re-enable by mutating the layer directly — the graph.ts re-enable
  // path — without any manager call that would dirty a cache.
  heatmap.enabled = true;
  assertEquals(
    ids(manager.getSortedLayers()),
    ["heatmap", "labels"],
    "layer re-enabled via direct mutation must rejoin the render list",
  );
});

Deno.test("LayerManager: disabling a layer directly removes it from the render list", () => {
  const manager = createLayerManager();
  const heatmap = makeLayer("heatmap", 0);
  manager.addLayer(heatmap);
  assertEquals(ids(manager.getSortedLayers()), ["heatmap"]);

  heatmap.enabled = false;
  assertEquals(ids(manager.getSortedLayers()), []);
});

Deno.test("LayerManager: getSortedLayers sorts by order and respects setLayerOrder", () => {
  const manager = createLayerManager();
  manager.addLayer(makeLayer("c", 30));
  manager.addLayer(makeLayer("a", 10));
  manager.addLayer(makeLayer("b", 20));

  assertEquals(ids(manager.getSortedLayers()), ["a", "b", "c"]);

  manager.setLayerOrder("a", 40);
  assertEquals(ids(manager.getSortedLayers()), ["b", "c", "a"]);
});

Deno.test("LayerManager: toggle/enable/disable via manager still work", () => {
  const manager = createLayerManager();
  const layer = makeLayer("x", 0);
  manager.addLayer(layer);

  assertEquals(manager.toggleLayer("x"), false);
  assertEquals(ids(manager.getSortedLayers()), []);
  assertEquals(manager.enableLayer("x"), true);
  assertEquals(ids(manager.getSortedLayers()), ["x"]);
  assertEquals(manager.disableLayer("x"), true);
  assertEquals(ids(manager.getSortedLayers()), []);
  assertEquals(manager.isLayerVisible("x"), false);
});

Deno.test("LayerManager: removeLayer drops the layer from the render list", () => {
  const manager = createLayerManager();
  manager.addLayer(makeLayer("x", 0));
  manager.addLayer(makeLayer("y", 1));
  assertEquals(ids(manager.getSortedLayers()), ["x", "y"]);

  manager.removeLayer("x");
  assertEquals(ids(manager.getSortedLayers()), ["y"]);
});
