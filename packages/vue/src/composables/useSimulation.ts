/**
 * useSimulation Composable
 *
 * Provides control over the force simulation.
 *
 * @module
 */

import { ref, watch, onUnmounted, type Ref, type ShallowRef } from "vue";
import type { HeroineGraph, SimulationStatus, ForceConfig } from "@heroine-graph/core";

/**
 * Options for the useSimulation composable
 */
export interface UseSimulationOptions {
  /** The HeroineGraph instance to control */
  graph: ShallowRef<HeroineGraph | null>;
  /** Initial force configuration */
  initialConfig?: Partial<ForceConfig>;
}

/**
 * Return value of the useSimulation composable
 */
export interface UseSimulationReturn {
  /** Current simulation status */
  status: Ref<SimulationStatus>;
  /** Whether the simulation is currently running */
  isRunning: Ref<boolean>;
  /** Current alpha value (0-1, energy level) */
  alpha: Ref<number>;
  /** Current force configuration */
  forceConfig: Ref<ForceConfig | null>;

  // Methods
  /** Start the simulation */
  start: () => void;
  /** Stop the simulation */
  stop: () => void;
  /** Pause the simulation */
  pause: () => void;
  /** Resume the simulation */
  resume: () => void;
  /** Restart the simulation (reheat) */
  restart: (alpha?: number) => void;
  /** Update force configuration */
  setForceConfig: (config: Partial<ForceConfig>) => void;
  /** Set simulation alpha (energy level) */
  setAlpha: (alpha: number) => void;
}

/**
 * Composable for controlling the force simulation
 *
 * @example
 * ```vue
 * <script setup lang="ts">
 * import { useGraph, useSimulation } from '@heroine-graph/vue';
 *
 * const { graph, isReady } = useGraph();
 * const {
 *   isRunning,
 *   alpha,
 *   start,
 *   stop,
 *   setForceConfig,
 * } = useSimulation({ graph });
 * </script>
 *
 * <template>
 *   <div>
 *     <button @click="isRunning ? stop() : start()">
 *       {{ isRunning ? 'Stop' : 'Start' }}
 *     </button>
 *     <div>Energy: {{ (alpha * 100).toFixed(1) }}%</div>
 *     <input
 *       type="range"
 *       min="0"
 *       max="100"
 *       @input="(e) => setForceConfig({ repulsion: Number((e.target as HTMLInputElement).value) })"
 *     />
 *   </div>
 * </template>
 * ```
 */
export function useSimulation(options: UseSimulationOptions): UseSimulationReturn {
  const { graph, initialConfig } = options;

  const status = ref<SimulationStatus>("stopped");
  const alpha = ref(0);
  const forceConfig = ref<ForceConfig | null>(null);
  const isRunning = ref(false);

  // Event handlers
  let tickHandler: ((event: { alpha: number }) => void) | null = null;
  let endHandler: (() => void) | null = null;

  function setupEventListeners(g: HeroineGraph) {
    // Get initial state
    status.value = g.getSimulationStatus();
    forceConfig.value = g.getForceConfig();
    isRunning.value = status.value === "running";

    // Subscribe to simulation events
    tickHandler = (event: { alpha: number }) => {
      alpha.value = event.alpha;
      status.value = g.getSimulationStatus();
      isRunning.value = status.value === "running";
    };

    endHandler = () => {
      status.value = "stopped";
      alpha.value = 0;
      isRunning.value = false;
    };

    g.on("simulation:tick", tickHandler);
    g.on("simulation:end", endHandler);

    // Apply initial config if provided
    if (initialConfig) {
      g.setForceConfig(initialConfig);
      forceConfig.value = g.getForceConfig();
    }
  }

  function cleanupEventListeners(g: HeroineGraph) {
    if (tickHandler) {
      g.off("simulation:tick", tickHandler);
      tickHandler = null;
    }
    if (endHandler) {
      g.off("simulation:end", endHandler);
      endHandler = null;
    }
  }

  // Watch for graph changes
  watch(
    graph,
    (newGraph, oldGraph) => {
      if (oldGraph) {
        cleanupEventListeners(oldGraph);
      }
      if (newGraph) {
        setupEventListeners(newGraph);
      }
    },
    { immediate: true }
  );

  // Cleanup on unmount
  onUnmounted(() => {
    if (graph.value) {
      cleanupEventListeners(graph.value);
    }
  });

  // Methods
  function start(): void {
    graph.value?.startSimulation();
    status.value = "running";
    isRunning.value = true;
  }

  function stop(): void {
    graph.value?.stopSimulation();
    status.value = "stopped";
    isRunning.value = false;
  }

  function pause(): void {
    graph.value?.pauseSimulation();
    status.value = "paused";
    isRunning.value = false;
  }

  function resume(): void {
    graph.value?.resumeSimulation();
    status.value = "running";
    isRunning.value = true;
  }

  function restart(newAlpha?: number): void {
    graph.value?.restartSimulation(newAlpha);
    status.value = "running";
    isRunning.value = true;
  }

  function setForceConfig(config: Partial<ForceConfig>): void {
    graph.value?.setForceConfig(config);
    forceConfig.value = graph.value?.getForceConfig() ?? null;
  }

  function setAlpha(newAlpha: number): void {
    graph.value?.setSimulationAlpha(newAlpha);
    alpha.value = newAlpha;
  }

  return {
    status,
    isRunning,
    alpha,
    forceConfig,
    start,
    stop,
    pause,
    resume,
    restart,
    setForceConfig,
    setAlpha,
  };
}

export default useSimulation;
