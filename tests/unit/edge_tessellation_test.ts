/**
 * Edge tessellation decision tests.
 *
 * getEdgeVertexCount is the single source of truth shared by
 * renderEdges (pipelines/edges.ts) and CommandOrchestrator.recordRenderPass
 * (commands.ts): a mismatch either truncates curved edges to their first
 * segment or draws no-op vertices. Also guards the WGSL contract: the
 * vertex shader must clamp segments to >= 1 and must degenerate the
 * extra segment quads for straight edges instead of re-drawing the full
 * quad `segments` times.
 */

import { assert, assertEquals } from "jsr:@std/assert@^1";
import {
  getEdgeVertexCount,
  VERTICES_PER_EDGE_SEGMENT,
} from "../../packages/core/src/renderer/edge_tessellation.ts";

Deno.test("getEdgeVertexCount: straight-only pipeline draws one quad", () => {
  assertEquals(getEdgeVertexCount({ enabled: false, segments: 19 }), 6);
  assertEquals(getEdgeVertexCount({ enabled: false, segments: 1 }), 6);
});

Deno.test("getEdgeVertexCount: curve-enabled pipeline draws 6 * segments", () => {
  assertEquals(getEdgeVertexCount({ enabled: true, segments: 19 }), 114);
  assertEquals(getEdgeVertexCount({ enabled: true, segments: 1 }), 6);
  assertEquals(getEdgeVertexCount({ enabled: true, segments: 32 }), 192);
});

Deno.test("getEdgeVertexCount: clamps segments to >= 1, matching the shader", () => {
  // edge.vert.wgsl uses max(curve_config.segments, 1u)
  assertEquals(getEdgeVertexCount({ enabled: true, segments: 0 }), VERTICES_PER_EDGE_SEGMENT);
});

const shaderDir = new URL(
  "../../packages/core/src/renderer/shaders/",
  import.meta.url,
);

Deno.test("edge.vert.wgsl: straight edges degenerate the extra segment quads", async () => {
  const source = await Deno.readTextFile(new URL("edge.vert.wgsl", shaderDir));
  // Straight edges must emit only the first quad when the pipeline
  // dispatches 6 * segments vertices — otherwise every straight edge is
  // overdrawn `segments` times and the AA fringe composites per repeat.
  assert(
    source.includes("!is_curved && vertex_idx >= 6u"),
    "straight edges must exit for vertex_idx >= 6 (segment quads beyond the first)",
  );
});

Deno.test("edge.frag.wgsl: flow layer1 color mix factor is clamped to [0,1]", async () => {
  const source = await Deno.readTextFile(new URL("edge.frag.wgsl", shaderDir));
  // color_a * intensity * brightness can exceed 1 (e.g. 'warning' preset
  // peaks at 2.0); mix() with t > 1 extrapolates past the flow color.
  assert(
    source.includes("mix(final_color, flow1.rgb, clamp(flow1.a, 0.0, 1.0))"),
    "layer1 colored flow must clamp the mix factor to [0,1]",
  );
});
