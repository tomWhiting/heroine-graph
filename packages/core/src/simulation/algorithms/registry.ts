/**
 * Force Algorithm Registry
 *
 * Central registry for available force algorithms. Allows runtime selection
 * of different force calculation strategies.
 *
 * @module
 */

import { supportsAlgorithmOnDevice } from "./types.ts";
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
   * Pass `device` whenever one is available: Barnes-Hut needs 10 storage
   * buffers per compute stage (the WebGPU default is 8), and on a device that
   * cannot supply them its pipelines are invalid — which discards every
   * command buffer they are recorded into, freezing the simulation instead of
   * degrading it. With a device in hand, unsupported algorithms are skipped
   * and the next candidate is returned.
   *
   * @param nodeCount - Number of nodes
   * @param device - Device the algorithm will run on, when known
   * @returns Recommended algorithm or undefined if none suitable
   */
  getRecommended(
    nodeCount: number,
    device?: Pick<GPUDevice, "limits">,
  ): ForceAlgorithm | undefined {
    const pick = (id: ForceAlgorithmType): ForceAlgorithm | undefined => {
      const algorithm = this.algorithms.get(id);
      if (!algorithm) return undefined;
      if (device && !supportsAlgorithmOnDevice(algorithm.info, device)) return undefined;
      return algorithm;
    };

    // Try to find the best algorithm for this node count
    if (nodeCount < 5000) {
      return pick("n2") ?? this.getAnyAvailable(device);
    }
    if (nodeCount < 50000) {
      return pick("barnes-hut") ?? pick("n2") ?? this.getAnyAvailable(device);
    }
    // Large graphs
    return (
      pick("density") ??
        pick("barnes-hut") ??
        pick("n2") ??
        this.getAnyAvailable(device)
    );
  }

  /**
   * Get any available algorithm (fallback), skipping any the device cannot run.
   *
   * @returns First usable algorithm or undefined
   */
  private getAnyAvailable(device?: Pick<GPUDevice, "limits">): ForceAlgorithm | undefined {
    for (const algorithm of this.algorithms.values()) {
      if (!device || supportsAlgorithmOnDevice(algorithm.info, device)) return algorithm;
    }
    return undefined;
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
