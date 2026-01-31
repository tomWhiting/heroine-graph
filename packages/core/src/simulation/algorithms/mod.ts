/**
 * Force Algorithm Module
 *
 * Provides pluggable force calculation algorithms for the simulation.
 *
 * @module
 */

// Import for use in initializeBuiltinAlgorithms
import { getAlgorithmRegistry as _getRegistry } from "./registry.ts";
import { createN2Algorithm as _createN2 } from "./n2.ts";
import { createBarnesHutAlgorithm as _createBarnesHut } from "./barnes-hut.ts";
import { createForceAtlas2Algorithm as _createForceAtlas2 } from "./force-atlas2.ts";
import { createDensityFieldAlgorithm as _createDensityField } from "./density-field.ts";
import { createRelativityAtlasAlgorithm as _createRelativityAtlas } from "./relativity-atlas.ts";

// Types
export type {
  AlgorithmBindGroups,
  AlgorithmBuffers,
  AlgorithmPipelines,
  AlgorithmRenderContext,
  ForceAlgorithm,
  ForceAlgorithmInfo,
  ForceAlgorithmType,
} from "./types.ts";
export { EmptyAlgorithmBuffers } from "./types.ts";

// Registry
export {
  createAlgorithmRegistry,
  ForceAlgorithmRegistry,
  getAlgorithmRegistry,
} from "./registry.ts";

// Algorithms
export { createN2Algorithm, N2ForceAlgorithm } from "./n2.ts";
export { BarnesHutForceAlgorithm, createBarnesHutAlgorithm } from "./barnes-hut.ts";
export { createForceAtlas2Algorithm, ForceAtlas2Algorithm } from "./force-atlas2.ts";
export { createDensityFieldAlgorithm, DensityFieldAlgorithm } from "./density-field.ts";
export {
  createRelativityAtlasAlgorithm,
  RelativityAtlasAlgorithm,
  uploadRelativityAtlasEdges,
} from "./relativity-atlas.ts";

/**
 * Initialize the global algorithm registry with built-in algorithms
 */
export function initializeBuiltinAlgorithms(): void {
  const registry = _getRegistry();

  // Register N² algorithm
  if (!registry.has("n2")) {
    registry.register(_createN2());
  }

  // Register Barnes-Hut algorithm
  if (!registry.has("barnes-hut")) {
    registry.register(_createBarnesHut());
  }

  // Register ForceAtlas2 algorithm
  if (!registry.has("force-atlas2")) {
    registry.register(_createForceAtlas2());
  }

  // Register Density Field algorithm
  if (!registry.has("density")) {
    registry.register(_createDensityField());
  }

  // Register Relativity Atlas algorithm
  if (!registry.has("relativity-atlas")) {
    registry.register(_createRelativityAtlas());
  }
}
