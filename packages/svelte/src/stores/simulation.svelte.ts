/**
 * Simulation Store for Svelte
 *
 * Provides reactive control over the force simulation.
 *
 * @module
 */

import type { HeroineGraph, SimulationStatus, ForceConfig } from "@heroine-graph/core";
import type { GraphStore } from "./graph.svelte";

/**
 * Options for createSimulationStore
 */
export interface SimulationStoreOptions {
  /** The graph store to control */
  graphStore: GraphStore;
  /** Initial force configuration */
  initialConfig?: Partial<ForceConfig>;
}

/**
 * Create a reactive simulation store
 *
 * @example
 * ```svelte
 * <script lang="ts">
 *   import { createGraphStore, createSimulationStore } from '@heroine-graph/svelte';
 *
 *   const graphStore = createGraphStore();
 *   const simStore = createSimulationStore({ graphStore });
 *
 *   const { isRunning, alpha, start, stop } = simStore;
 * </script>
 *
 * <button onclick={() => isRunning ? stop() : start()}>
 *   {isRunning ? 'Stop' : 'Start'}
 * </button>
 * <div>Energy: {(alpha * 100).toFixed(1)}%</div>
 * ```
 */
export function createSimulationStore(options: SimulationStoreOptions) {
  const { graphStore, initialConfig } = options;

  // Reactive state using Svelte 5 runes
  let status = $state<SimulationStatus>("stopped");
  let alpha = $state(0);
  let forceConfig = $state<ForceConfig | null>(null);
  let isRunning = $derived(status === "running");

  // Event handlers
  let tickHandler: ((event: { alpha: number }) => void) | null = null;
  let endHandler: (() => void) | null = null;
  let currentGraph: HeroineGraph | null = null;

  function setupEventListeners(g: HeroineGraph) {
    // Cleanup previous listeners
    cleanupEventListeners();

    currentGraph = g;

    // Get initial state
    status = g.getSimulationStatus();
    forceConfig = g.getForceConfig();

    // Subscribe to simulation events
    tickHandler = (event: { alpha: number }) => {
      alpha = event.alpha;
      status = g.getSimulationStatus();
    };

    endHandler = () => {
      status = "stopped";
      alpha = 0;
    };

    g.on("simulation:tick", tickHandler);
    g.on("simulation:end", endHandler);

    // Apply initial config if provided
    if (initialConfig) {
      g.setForceConfig(initialConfig);
      forceConfig = g.getForceConfig();
    }
  }

  function cleanupEventListeners() {
    if (currentGraph) {
      if (tickHandler) {
        currentGraph.off("simulation:tick", tickHandler);
      }
      if (endHandler) {
        currentGraph.off("simulation:end", endHandler);
      }
    }
    tickHandler = null;
    endHandler = null;
    currentGraph = null;
  }

  // Watch for graph changes using $effect
  $effect(() => {
    const g = graphStore.graph;
    if (g && g !== currentGraph) {
      setupEventListeners(g);
    } else if (!g && currentGraph) {
      cleanupEventListeners();
    }
  });

  // Methods
  function start(): void {
    graphStore.graph?.startSimulation();
    status = "running";
  }

  function stop(): void {
    graphStore.graph?.stopSimulation();
    status = "stopped";
  }

  function pause(): void {
    graphStore.graph?.pauseSimulation();
    status = "paused";
  }

  function resume(): void {
    graphStore.graph?.resumeSimulation();
    status = "running";
  }

  function restart(): void {
    graphStore.graph?.restartSimulation();
    status = "running";
  }

  function setForceConfig(config: Partial<ForceConfig>): void {
    graphStore.graph?.setForceConfig(config);
    forceConfig = graphStore.graph?.getForceConfig() ?? null;
  }

  function setAlpha(newAlpha: number): void {
    graphStore.graph?.setSimulationAlpha(newAlpha);
    alpha = newAlpha;
  }

  return {
    // State (use getter functions for reactivity)
    get status() { return status; },
    get isRunning() { return isRunning; },
    get alpha() { return alpha; },
    get forceConfig() { return forceConfig; },

    // Methods
    start,
    stop,
    pause,
    resume,
    restart,
    setForceConfig,
    setAlpha,
    dispose: cleanupEventListeners,
  };
}

export type SimulationStore = ReturnType<typeof createSimulationStore>;
