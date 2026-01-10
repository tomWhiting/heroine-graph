/**
 * Force Algorithm Registry
 *
 * Central registry for available force algorithms. Allows runtime selection
 * of different force calculation strategies.
 *
 * @module
 */

import type { ForceAlgorithm, ForceAlgorithmInfo, ForceAlgorithmType } from "./types.ts";

/**
 * Force algorithm registry
 */
export class ForceAlgorithmRegistry {
  private algorithms = new Map<ForceAlgorithmType, ForceAlgorithm>();

  /**
   * Register a force algorithm
   *
   * @param algorithm - Algorithm to register
   */
  register(algorithm: ForceAlgorithm): void {
    this.algorithms.set(algorithm.info.id, algorithm);
  }

  /**
   * Get an algorithm by ID
   *
   * @param id - Algorithm ID
   * @returns Algorithm or undefined
   */
  get(id: ForceAlgorithmType): ForceAlgorithm | undefined {
    return this.algorithms.get(id);
  }

  /**
   * Check if an algorithm is registered
   *
   * @param id - Algorithm ID
   * @returns true if registered
   */
  has(id: ForceAlgorithmType): boolean {
    return this.algorithms.has(id);
  }

  /**
   * List all registered algorithms
   *
   * @returns Array of registered algorithms
   */
  list(): ForceAlgorithm[] {
    return Array.from(this.algorithms.values());
  }

  /**
   * List algorithm info for all registered algorithms
   *
   * @returns Array of algorithm info
   */
  listInfo(): ForceAlgorithmInfo[] {
    return this.list().map((a) => a.info);
  }

  /**
   * Get the recommended algorithm for a given node count
   *
   * Selection logic:
   * - < 5,000 nodes: N² (simple and fast enough)
   * - 5,000 - 50,000 nodes: Barnes-Hut (good balance)
   * - > 50,000 nodes: Density-based (fastest for large graphs)
   *
   * @param nodeCount - Number of nodes
   * @returns Recommended algorithm or undefined if none suitable
   */
  getRecommended(nodeCount: number): ForceAlgorithm | undefined {
    // Try to find the best algorithm for this node count
    if (nodeCount < 5000) {
      return this.get("n2") ?? this.getAnyAvailable();
    }
    if (nodeCount < 50000) {
      return this.get("barnes-hut") ?? this.get("n2") ?? this.getAnyAvailable();
    }
    // Large graphs
    return (
      this.get("density") ??
      this.get("barnes-hut") ??
      this.get("n2") ??
      this.getAnyAvailable()
    );
  }

  /**
   * Get any available algorithm (fallback)
   *
   * @returns First available algorithm or undefined
   */
  private getAnyAvailable(): ForceAlgorithm | undefined {
    const first = this.algorithms.values().next();
    return first.done ? undefined : first.value;
  }
}

/**
 * Global algorithm registry instance
 */
let globalRegistry: ForceAlgorithmRegistry | null = null;

/**
 * Get the global algorithm registry
 *
 * @returns Global registry instance
 */
export function getAlgorithmRegistry(): ForceAlgorithmRegistry {
  if (!globalRegistry) {
    globalRegistry = new ForceAlgorithmRegistry();
  }
  return globalRegistry;
}

/**
 * Create a new algorithm registry
 *
 * @returns New registry instance
 */
export function createAlgorithmRegistry(): ForceAlgorithmRegistry {
  return new ForceAlgorithmRegistry();
}
