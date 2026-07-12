/**
 * Edge Tessellation Decision
 *
 * Single source of truth for how many vertices each edge instance needs.
 * Both the canonical render helper (pipelines/edges.ts renderEdges) and
 * the CommandOrchestrator (commands.ts) must draw the same vertex count,
 * matching the segment math in edge.vert.wgsl — a mismatch either
 * truncates curved edges to their first segment or issues no-op vertices.
 *
 * Kept free of shader imports so CPU-side unit tests can import it.
 *
 * @module
 */

/**
 * The subset of CurvedEdgeConfig the tessellation decision depends on
 */
export interface EdgeTessellationConfig {
  /** Whether curved edge rendering is enabled for the pipeline */
  enabled: boolean;
  /** Number of tessellation segments per curved edge */
  segments: number;
}

/** Vertices per quad (2 triangles) */
export const VERTICES_PER_EDGE_SEGMENT = 6;

/**
 * Vertices to draw per edge instance.
 *
 * Straight-only pipelines draw one quad (6 vertices). Curve-enabled
 * pipelines draw 6 * segments so the vertex shader can tessellate curved
 * edges; per-edge straight edges degenerate the extra segment quads in
 * the shader. Mirrors the shader's `max(curve_config.segments, 1u)`.
 */
export function getEdgeVertexCount(config: EdgeTessellationConfig): number {
  return config.enabled
    ? VERTICES_PER_EDGE_SEGMENT * Math.max(1, config.segments)
    : VERTICES_PER_EDGE_SEGMENT;
}
