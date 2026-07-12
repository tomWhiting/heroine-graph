/**
 * Hand-Written Tiny Fixture Graphs
 *
 * Small, fully-explicit graphs for exact-value assertions (parser
 * round-trips, styling, edge indexing). Returned as fresh objects each
 * call so tests cannot contaminate each other via shared references.
 *
 * @module
 */

import type { GraphInput } from "../../packages/core/src/types.ts";

/**
 * Three nodes, three edges, explicit positions/colors/radii/widths.
 */
export function triangleGraph(): GraphInput {
  return {
    nodes: [
      { id: "a", x: 0, y: 0, color: "#ff0000", radius: 10 },
      { id: "b", x: 100, y: 0, color: "#00ff00", radius: 20 },
      { id: "c", x: 50, y: 80, color: "#0000ff", radius: 30 },
    ],
    edges: [
      { id: "ab", source: "a", target: "b", width: 2, color: "#ffffff" },
      { id: "bc", source: "b", target: "c", width: 4 },
      { source: "c", target: "a" }, // no id: parser must generate one
    ],
  };
}

/**
 * Minimal code-tree: root dir containing a dir and a file, the dir
 * containing two files, plus one cross-cutting import. Types on nodes
 * (top-level and via metadata) and edges exercise typed styling paths.
 */
export function microTreeGraph(): GraphInput {
  return {
    nodes: [
      { id: "root", type: "dir" },
      { id: "src", type: "dir" },
      { id: "readme", metadata: { type: "file", lang: "md" } },
      { id: "main", type: "file" },
      { id: "util", type: "file" },
    ],
    edges: [
      { id: "e0", source: "root", target: "src", type: "contains" },
      { id: "e1", source: "root", target: "readme", type: "contains" },
      { id: "e2", source: "src", target: "main", type: "contains" },
      { id: "e3", source: "src", target: "util", type: "contains" },
      { id: "e4", source: "main", target: "util", type: "imports" },
    ],
  };
}
