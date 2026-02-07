/**
 * useSimulation Hook
 *
 * Provides control over the force simulation.
 *
 * @module
 */

import { useCallback, useEffect, useState } from "react";
import type { HeroineGraph, SimulationStatus, ForceConfig } from "@heroine-graph/core";

/**
 * Options for the useSimulation hook
 */
export interface UseSimulationOptions {
  /** The HeroineGraph instance to control */
  graph: HeroineGraph | null;
  /** Initial force configuration */
  initialConfig?: Partial<ForceConfig>;
}

/**
 * Return value of the useSimulation hook
 */
export interface UseSimulationReturn {
  /** Current simulation status */
  status: SimulationStatus;
  /** Whether the simulation is currently running */
  isRunning: boolean;
  /** Current alpha value (0-1, energy level) */
  alpha: number;
  /** Current force configuration */
  forceConfig: ForceConfig | null;

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
  restart: () => void;
  /** Update force configuration */
  setForceConfig: (config: Partial<ForceConfig>) => void;
  /** Set simulation alpha (energy level) */
  setAlpha: (alpha: number) => void;
}

/**
 * Hook for controlling the force simulation
 *
 * @example
 * ```tsx
 * import { useGraph, useSimulation } from '@heroine-graph/react';
 *
 * function SimulationControls() {
 *   const { graph, isReady } = useGraph();
 *   const {
 *     isRunning,
 *     alpha,
 *     start,
 *     stop,
 *     setForceConfig,
 *   } = useSimulation({ graph });
 *
 *   return (
 *     <div>
 *       <button onClick={isRunning ? stop : start}>
 *         {isRunning ? 'Stop' : 'Start'}
 *       </button>
 *       <div>Energy: {(alpha * 100).toFixed(1)}%</div>
 *       <input
 *         type="range"
 *         min="0"
 *         max="100"
 *         onChange={(e) => setForceConfig({ repulsion: Number(e.target.value) })}
 *       />
 *     </div>
 *   );
 * }
 * ```
 */
export function useSimulation(options: UseSimulationOptions): UseSimulationReturn {
  const { graph, initialConfig } = options;

  const [status, setStatus] = useState<SimulationStatus>("stopped");
  const [alpha, setAlphaState] = useState(0);
  const [forceConfig, setForceConfigState] = useState<ForceConfig | null>(null);

  // Track simulation state
  useEffect(() => {
    if (!graph) return;

    // Get initial state
    setStatus(graph.getSimulationStatus());
    setForceConfigState(graph.getForceConfig());

    // Subscribe to simulation events
    const handleTick = (event: { alpha: number }) => {
      setAlphaState(event.alpha);
      setStatus(graph.getSimulationStatus());
    };

    const handleEnd = () => {
      setStatus("stopped");
      setAlphaState(0);
    };

    graph.on("simulation:tick", handleTick);
    graph.on("simulation:end", handleEnd);

    // Apply initial config if provided
    if (initialConfig) {
      graph.setForceConfig(initialConfig);
      setForceConfigState(graph.getForceConfig());
    }

    return () => {
      graph.off("simulation:tick", handleTick);
      graph.off("simulation:end", handleEnd);
    };
  }, [graph, initialConfig]);

  // Methods
  const start = useCallback(() => {
    graph?.startSimulation();
    setStatus("running");
  }, [graph]);

  const stop = useCallback(() => {
    graph?.stopSimulation();
    setStatus("stopped");
  }, [graph]);

  const pause = useCallback(() => {
    graph?.pauseSimulation();
    setStatus("paused");
  }, [graph]);

  const resume = useCallback(() => {
    graph?.resumeSimulation();
    setStatus("running");
  }, [graph]);

  const restart = useCallback(
    () => {
      graph?.restartSimulation();
      setStatus("running");
    },
    [graph]
  );

  const setForceConfig = useCallback(
    (config: Partial<ForceConfig>) => {
      graph?.setForceConfig(config);
      setForceConfigState(graph?.getForceConfig() ?? null);
    },
    [graph]
  );

  const setAlpha = useCallback(
    (newAlpha: number) => {
      graph?.setSimulationAlpha(newAlpha);
      setAlphaState(newAlpha);
    },
    [graph]
  );

  return {
    status,
    isRunning: status === "running",
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
