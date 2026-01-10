/**
 * Simulation Controller
 *
 * High-level control of the force-directed layout simulation with
 * start/pause/stop/restart functionality.
 *
 * @module
 */

import type { SimulationStatus } from "../types.ts";

/**
 * Simulation state
 */
export interface SimulationState {
  /** Current status */
  status: SimulationStatus;
  /** Current alpha (cooling factor) */
  alpha: number;
  /** Target alpha for restart */
  alphaTarget: number;
  /** Minimum alpha (simulation stops below this) */
  alphaMin: number;
  /** Alpha decay rate per tick */
  alphaDecay: number;
  /** Number of ticks (iterations) run */
  tickCount: number;
  /** Whether simulation is warming up */
  isWarmingUp: boolean;
}

/**
 * Simulation event types
 */
export type SimulationEventType = "tick" | "end" | "pause" | "resume" | "restart";

/**
 * Simulation event data by type
 */
export interface SimulationEventData {
  tick: { alpha: number; tickCount: number };
  end: { tickCount: number };
  pause: { alpha: number };
  resume: { alpha: number };
  restart: { alpha: number };
}

/**
 * Simulation event handler
 */
export type SimulationEventHandler<T extends SimulationEventType> = (
  data: SimulationEventData[T],
) => void;

/**
 * Simple event emitter for simulation events
 */
export interface SimulationEventEmitter {
  on<T extends SimulationEventType>(type: T, handler: SimulationEventHandler<T>): () => void;
  off<T extends SimulationEventType>(type: T, handler: SimulationEventHandler<T>): void;
  emit<T extends SimulationEventType>(type: T, data: SimulationEventData[T]): void;
}

/**
 * Simulation controller configuration
 */
export interface SimulationControllerConfig {
  /** Initial alpha value */
  alpha?: number;
  /** Target alpha (usually 0 for cool-down) */
  alphaTarget?: number;
  /** Minimum alpha before stopping */
  alphaMin?: number;
  /** Alpha decay rate (0-1, smaller = slower cooling) */
  alphaDecay?: number;
  /** Velocity decay (damping) */
  velocityDecay?: number;
  /** Maximum iterations per tick (for catch-up) */
  maxIterationsPerTick?: number;
  /** Warm-up ticks before rendering */
  warmUpTicks?: number;
}

/**
 * Default simulation configuration
 *
 * Note: alphaTarget defaults to 0.1 for continuous simulation mode.
 * This keeps the simulation responsive to changes and dragging.
 * Set to 0.0 if you want the simulation to fully stop after cooling down.
 */
export const DEFAULT_SIMULATION_CONFIG: Required<SimulationControllerConfig> = {
  alpha: 1.0,
  alphaTarget: 0.1, // Non-zero for continuous simulation
  alphaMin: 0.001,
  alphaDecay: 0.0228, // ~300 iterations to cool down
  velocityDecay: 0.4,
  maxIterationsPerTick: 1,
  warmUpTicks: 0,
};

/**
 * Simulation controller interface
 */
export interface SimulationController {
  /** Current simulation state */
  readonly state: SimulationState;
  /** Whether simulation is currently running */
  readonly isRunning: boolean;
  /** Event emitter for simulation events */
  readonly events: SimulationEventEmitter;

  /** Start the simulation */
  start: () => void;
  /** Pause the simulation (can be resumed) */
  pause: () => void;
  /** Stop the simulation and reset */
  stop: () => void;
  /** Restart the simulation from alpha = 1 */
  restart: () => void;
  /** Run a single tick manually */
  tick: () => boolean;
  /** Set alpha value directly */
  setAlpha: (alpha: number) => void;
  /** Set alpha target */
  setAlphaTarget: (target: number) => void;
  /** Update configuration */
  setConfig: (config: Partial<SimulationControllerConfig>) => void;
  /** Get current configuration */
  getConfig: () => Required<SimulationControllerConfig>;
}

/**
 * Creates a simulation controller
 *
 * @param config - Controller configuration
 * @returns Simulation controller
 */
export function createSimulationController(
  config: SimulationControllerConfig = {},
): SimulationController {
  const finalConfig: Required<SimulationControllerConfig> = {
    ...DEFAULT_SIMULATION_CONFIG,
    ...config,
  };

  // Internal event handlers storage
  const handlers = new Map<SimulationEventType, Set<SimulationEventHandler<SimulationEventType>>>();

  // Create simulation event emitter
  const events: SimulationEventEmitter = {
    on<T extends SimulationEventType>(type: T, handler: SimulationEventHandler<T>): () => void {
      let set = handlers.get(type);
      if (!set) {
        set = new Set();
        handlers.set(type, set);
      }
      set.add(handler as SimulationEventHandler<SimulationEventType>);
      return () => events.off(type, handler);
    },
    off<T extends SimulationEventType>(type: T, handler: SimulationEventHandler<T>): void {
      const set = handlers.get(type);
      if (set) {
        set.delete(handler as SimulationEventHandler<SimulationEventType>);
      }
    },
    emit<T extends SimulationEventType>(type: T, data: SimulationEventData[T]): void {
      const set = handlers.get(type);
      if (set) {
        for (const handler of set) {
          try {
            (handler as SimulationEventHandler<T>)(data);
          } catch (error) {
            console.error(`[Simulation] Error in ${type} handler:`, error);
          }
        }
      }
    },
  };

  // Internal state
  const state: SimulationState = {
    status: "idle",
    alpha: finalConfig.alpha,
    alphaTarget: finalConfig.alphaTarget,
    alphaMin: finalConfig.alphaMin,
    alphaDecay: finalConfig.alphaDecay,
    tickCount: 0,
    isWarmingUp: finalConfig.warmUpTicks > 0,
  };

  /**
   * Perform one simulation tick
   * @returns true if simulation should continue
   */
  function tick(): boolean {
    if (state.status !== "running") {
      return false;
    }

    state.tickCount++;

    // Alpha decay toward target
    state.alpha += (state.alphaTarget - state.alpha) * state.alphaDecay;

    // Check for warm-up completion
    if (state.isWarmingUp && state.tickCount >= finalConfig.warmUpTicks) {
      state.isWarmingUp = false;
    }

    // Emit tick event
    events.emit("tick", {
      alpha: state.alpha,
      tickCount: state.tickCount,
    });

    // Note: We do NOT stop the simulation when alpha < alphaMin.
    // The simulation should always run - alpha just controls movement intensity.
    // At low alpha, forces are computed but movements are negligible (equilibrium).
    // This ensures dragging a node immediately affects the simulation.

    return true;
  }

  /**
   * Start the simulation
   */
  function start(): void {
    if (state.status === "running") return;

    state.status = "running";

    if (state.tickCount === 0) {
      // Fresh start
      state.alpha = finalConfig.alpha;
      state.isWarmingUp = finalConfig.warmUpTicks > 0;
    }
  }

  /**
   * Pause the simulation
   */
  function pause(): void {
    if (state.status !== "running") return;

    state.status = "paused";
    events.emit("pause", { alpha: state.alpha });
  }

  /**
   * Stop the simulation and reset
   */
  function stop(): void {
    state.status = "stopped";
    state.tickCount = 0;
    state.alpha = 0;
    state.isWarmingUp = false;
  }

  /**
   * Restart the simulation
   */
  function restart(): void {
    state.status = "running";
    state.alpha = finalConfig.alpha;
    state.alphaTarget = finalConfig.alphaTarget;
    state.tickCount = 0;
    state.isWarmingUp = finalConfig.warmUpTicks > 0;
    events.emit("restart", { alpha: state.alpha });
  }

  /**
   * Set alpha directly
   */
  function setAlpha(alpha: number): void {
    state.alpha = Math.max(0, Math.min(1, alpha));
  }

  /**
   * Set alpha target
   */
  function setAlphaTarget(target: number): void {
    state.alphaTarget = Math.max(0, Math.min(1, target));
  }

  /**
   * Update configuration
   */
  function setConfig(newConfig: Partial<SimulationControllerConfig>): void {
    Object.assign(finalConfig, newConfig);

    // Update state from config
    if (newConfig.alphaMin !== undefined) {
      state.alphaMin = newConfig.alphaMin;
    }
    if (newConfig.alphaDecay !== undefined) {
      state.alphaDecay = newConfig.alphaDecay;
    }
  }

  /**
   * Get current configuration
   */
  function getConfig(): Required<SimulationControllerConfig> {
    return { ...finalConfig };
  }

  return {
    get state() {
      return { ...state };
    },
    get isRunning() {
      return state.status === "running";
    },
    events,
    start,
    pause,
    stop,
    restart,
    tick,
    setAlpha,
    setAlphaTarget,
    setConfig,
    getConfig,
  };
}

/**
 * Calculate alpha decay rate for a target number of iterations
 *
 * @param iterations - Number of iterations to cool down
 * @param alphaMin - Minimum alpha value
 * @returns Alpha decay rate
 */
export function calculateAlphaDecay(
  iterations: number,
  alphaMin: number = 0.001,
): number {
  // Solve: alphaMin = 1 * (1 - decay)^iterations
  // decay = 1 - alphaMin^(1/iterations)
  return 1 - alphaMin ** (1 / iterations);
}
