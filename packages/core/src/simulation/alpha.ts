/**
 * Alpha Decay and Convergence Logic
 *
 * Manages the simulation "temperature" (alpha) which controls how much
 * force is applied each tick. Alpha decays over time, causing the
 * simulation to gradually settle into a stable configuration.
 *
 * @module
 */

/**
 * Alpha decay parameters
 */
export interface AlphaConfig {
  /** Initial alpha value (0-1) */
  initial: number;
  /** Minimum alpha (simulation stops below this) */
  min: number;
  /** Target alpha (usually 0 for cool-down) */
  target: number;
  /** Decay rate per tick (smaller = slower cooling) */
  decay: number;
}

/**
 * Default alpha configuration
 */
export const DEFAULT_ALPHA_CONFIG: AlphaConfig = {
  initial: 1.0,
  min: 0.001,
  target: 0.0,
  decay: 0.0228, // ~300 iterations to cool down
};

/**
 * Alpha manager for simulation temperature control
 */
export interface AlphaManager {
  /** Current alpha value */
  readonly alpha: number;
  /** Whether simulation has converged */
  readonly hasConverged: boolean;
  /** Perform one decay step */
  tick: () => number;
  /** Reset alpha to initial value */
  reset: () => void;
  /** Set alpha directly */
  set: (alpha: number) => void;
  /** Set target alpha */
  setTarget: (target: number) => void;
  /** Reheat the simulation */
  reheat: (alpha?: number) => void;
  /** Update configuration */
  setConfig: (config: Partial<AlphaConfig>) => void;
}

/**
 * Creates an alpha manager
 *
 * @param config - Alpha configuration
 * @returns Alpha manager
 */
export function createAlphaManager(
  config: Partial<AlphaConfig> = {},
): AlphaManager {
  const finalConfig = { ...DEFAULT_ALPHA_CONFIG, ...config };

  let alpha = finalConfig.initial;
  let target = finalConfig.target;

  return {
    get alpha() {
      return alpha;
    },

    get hasConverged() {
      return alpha < finalConfig.min;
    },

    tick(): number {
      // Exponential decay toward target
      alpha += (target - alpha) * finalConfig.decay;

      // Clamp to minimum
      if (alpha < finalConfig.min) {
        alpha = 0;
      }

      return alpha;
    },

    reset(): void {
      alpha = finalConfig.initial;
      target = finalConfig.target;
    },

    set(newAlpha: number): void {
      alpha = Math.max(0, Math.min(1, newAlpha));
    },

    setTarget(newTarget: number): void {
      target = Math.max(0, Math.min(1, newTarget));
    },

    reheat(newAlpha: number = finalConfig.initial): void {
      alpha = newAlpha;
    },

    setConfig(newConfig: Partial<AlphaConfig>): void {
      Object.assign(finalConfig, newConfig);
    },
  };
}

/**
 * Calculate alpha decay rate for a desired number of iterations
 *
 * Solves for decay rate given: alphaMin = alphaInitial * (1 - decay)^iterations
 *
 * @param iterations - Desired number of iterations until convergence
 * @param alphaInitial - Initial alpha (default 1.0)
 * @param alphaMin - Minimum alpha (default 0.001)
 * @returns Decay rate
 */
export function calculateDecayRate(
  iterations: number,
  alphaInitial: number = 1.0,
  alphaMin: number = 0.001,
): number {
  // Solve: alphaMin = alphaInitial * (1 - decay)^iterations
  // (1 - decay)^iterations = alphaMin / alphaInitial
  // 1 - decay = (alphaMin / alphaInitial)^(1/iterations)
  // decay = 1 - (alphaMin / alphaInitial)^(1/iterations)
  return 1 - Math.pow(alphaMin / alphaInitial, 1 / iterations);
}

/**
 * Calculate number of iterations for a given decay rate
 *
 * @param decay - Decay rate
 * @param alphaInitial - Initial alpha (default 1.0)
 * @param alphaMin - Minimum alpha (default 0.001)
 * @returns Number of iterations until convergence
 */
export function calculateIterations(
  decay: number,
  alphaInitial: number = 1.0,
  alphaMin: number = 0.001,
): number {
  // iterations = log(alphaMin / alphaInitial) / log(1 - decay)
  return Math.log(alphaMin / alphaInitial) / Math.log(1 - decay);
}

/**
 * Convergence detector using kinetic energy
 */
export interface ConvergenceDetector {
  /** Record a sample of total kinetic energy */
  record: (kineticEnergy: number) => void;
  /** Check if simulation has converged based on energy */
  hasConverged: () => boolean;
  /** Reset the detector */
  reset: () => void;
  /** Get current average energy */
  readonly averageEnergy: number;
}

/**
 * Creates a convergence detector based on kinetic energy
 *
 * @param threshold - Energy threshold for convergence
 * @param windowSize - Number of samples to average
 * @returns Convergence detector
 */
export function createConvergenceDetector(
  threshold: number = 0.0001,
  windowSize: number = 10,
): ConvergenceDetector {
  const samples: number[] = [];
  let sum = 0;

  return {
    record(kineticEnergy: number): void {
      samples.push(kineticEnergy);
      sum += kineticEnergy;

      // Maintain window size
      if (samples.length > windowSize) {
        sum -= samples.shift()!;
      }
    },

    hasConverged(): boolean {
      if (samples.length < windowSize) return false;
      return sum / windowSize < threshold;
    },

    reset(): void {
      samples.length = 0;
      sum = 0;
    },

    get averageEnergy(): number {
      return samples.length > 0 ? sum / samples.length : 0;
    },
  };
}

/**
 * Adaptive alpha controller that adjusts based on energy
 */
export interface AdaptiveAlphaController {
  /** Update alpha based on current kinetic energy */
  update: (kineticEnergy: number) => number;
  /** Get current alpha */
  readonly alpha: number;
  /** Reset controller */
  reset: () => void;
}

/**
 * Creates an adaptive alpha controller
 *
 * When energy is high (lots of movement), keeps alpha high.
 * When energy drops, allows alpha to decay.
 *
 * @param config - Alpha configuration
 * @returns Adaptive controller
 */
export function createAdaptiveAlphaController(
  config: Partial<AlphaConfig> = {},
): AdaptiveAlphaController {
  const finalConfig = { ...DEFAULT_ALPHA_CONFIG, ...config };
  let alpha = finalConfig.initial;
  let lastEnergy = Infinity;

  return {
    update(kineticEnergy: number): number {
      // If energy increased significantly, reheat
      if (kineticEnergy > lastEnergy * 1.5 && alpha < 0.5) {
        alpha = Math.min(1, alpha + 0.1);
      } else {
        // Normal decay
        alpha += (finalConfig.target - alpha) * finalConfig.decay;
      }

      // Clamp
      if (alpha < finalConfig.min) {
        alpha = 0;
      }

      lastEnergy = kineticEnergy;
      return alpha;
    },

    get alpha() {
      return alpha;
    },

    reset(): void {
      alpha = finalConfig.initial;
      lastEnergy = Infinity;
    },
  };
}
