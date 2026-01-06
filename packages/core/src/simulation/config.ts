/**
 * Force Configuration API
 *
 * Provides configuration interfaces and defaults for force-directed layout
 * simulation parameters including repulsion, attraction, centering, and
 * collision forces.
 *
 * @module
 */

import type { ForceConfig } from "../types.ts";

/**
 * Extended force configuration with all parameters
 */
export interface FullForceConfig extends ForceConfig {
  // Repulsion (charge/many-body force)
  /** Repulsion strength (negative = repel) */
  repulsionStrength: number;
  /** Maximum repulsion distance (nodes beyond this don't repel) */
  repulsionDistanceMax: number;
  /** Minimum repulsion distance (prevents singularities) */
  repulsionDistanceMin: number;
  /** Barnes-Hut theta (0.5-1.5, higher = faster but less accurate) */
  theta: number;

  // Attraction (link/spring force)
  /** Spring strength (stiffness) */
  springStrength: number;
  /** Spring rest length (natural length) */
  springLength: number;
  /** Spring length variation by node degree */
  springLengthByDegree: boolean;

  // Centering force
  /** Centering force strength */
  centerStrength: number;
  /** Center X coordinate */
  centerX: number;
  /** Center Y coordinate */
  centerY: number;

  // Collision force
  /** Enable collision detection */
  collisionEnabled: boolean;
  /** Collision radius multiplier */
  collisionRadiusMultiplier: number;
  /** Collision strength (0-1) */
  collisionStrength: number;
  /** Collision iterations per tick */
  collisionIterations: number;

  // Integration
  /** Velocity damping (0-1, higher = more damping) */
  velocityDecay: number;
  /** Maximum velocity (prevents instability) */
  maxVelocity: number;
  /** Time step for integration */
  timeStep: number;
}

/**
 * Default force configuration values
 */
export const DEFAULT_FORCE_CONFIG: FullForceConfig = {
  // Repulsion
  repulsionStrength: -30,
  repulsionDistanceMax: 1000,
  repulsionDistanceMin: 1,
  theta: 0.8,

  // Attraction
  springStrength: 0.1,
  springLength: 30,
  springLengthByDegree: false,

  // Centering
  centerStrength: 0.01,
  centerX: 0,
  centerY: 0,

  // Collision
  collisionEnabled: false,
  collisionRadiusMultiplier: 1.0,
  collisionStrength: 0.7,
  collisionIterations: 1,

  // Integration
  velocityDecay: 0.4,
  maxVelocity: 50,
  timeStep: 1.0,
};

/**
 * Preset configurations for common use cases
 */
export const FORCE_PRESETS = {
  /** Default balanced configuration */
  default: { ...DEFAULT_FORCE_CONFIG },

  /** Fast layout with less accuracy */
  fast: {
    ...DEFAULT_FORCE_CONFIG,
    theta: 1.2,
    springStrength: 0.2,
    velocityDecay: 0.3,
  },

  /** High quality layout (slower) */
  quality: {
    ...DEFAULT_FORCE_CONFIG,
    theta: 0.5,
    springStrength: 0.05,
    velocityDecay: 0.5,
  },

  /** Clustered layout (stronger repulsion between groups) */
  clustered: {
    ...DEFAULT_FORCE_CONFIG,
    repulsionStrength: -50,
    springLength: 50,
    centerStrength: 0.005,
  },

  /** Dense layout (nodes closer together) */
  dense: {
    ...DEFAULT_FORCE_CONFIG,
    repulsionStrength: -15,
    springLength: 20,
  },

  /** Sparse layout (nodes spread apart) */
  sparse: {
    ...DEFAULT_FORCE_CONFIG,
    repulsionStrength: -50,
    springLength: 80,
    repulsionDistanceMax: 2000,
  },

  /** Radial layout (strong centering) */
  radial: {
    ...DEFAULT_FORCE_CONFIG,
    centerStrength: 0.1,
    repulsionStrength: -20,
  },
} as const;

/**
 * Force configuration builder
 */
export interface ForceConfigBuilder {
  /** Set repulsion parameters */
  repulsion: (
    strength: number,
    distanceMax?: number,
    distanceMin?: number,
  ) => ForceConfigBuilder;
  /** Set spring (attraction) parameters */
  spring: (strength: number, length?: number) => ForceConfigBuilder;
  /** Set centering force */
  center: (strength: number, x?: number, y?: number) => ForceConfigBuilder;
  /** Enable collision with parameters */
  collision: (strength?: number, radiusMultiplier?: number) => ForceConfigBuilder;
  /** Set Barnes-Hut theta */
  theta: (theta: number) => ForceConfigBuilder;
  /** Set velocity decay */
  decay: (velocityDecay: number) => ForceConfigBuilder;
  /** Build the configuration */
  build: () => FullForceConfig;
}

/**
 * Creates a force configuration builder
 *
 * @param base - Base configuration to start from
 * @returns Force configuration builder
 */
export function forceConfigBuilder(
  base: Partial<FullForceConfig> = {},
): ForceConfigBuilder {
  const config: FullForceConfig = { ...DEFAULT_FORCE_CONFIG, ...base };

  const builder: ForceConfigBuilder = {
    repulsion(strength, distanceMax, distanceMin) {
      config.repulsionStrength = strength;
      if (distanceMax !== undefined) config.repulsionDistanceMax = distanceMax;
      if (distanceMin !== undefined) config.repulsionDistanceMin = distanceMin;
      return builder;
    },

    spring(strength, length) {
      config.springStrength = strength;
      if (length !== undefined) config.springLength = length;
      return builder;
    },

    center(strength, x, y) {
      config.centerStrength = strength;
      if (x !== undefined) config.centerX = x;
      if (y !== undefined) config.centerY = y;
      return builder;
    },

    collision(strength = 0.7, radiusMultiplier = 1.0) {
      config.collisionEnabled = true;
      config.collisionStrength = strength;
      config.collisionRadiusMultiplier = radiusMultiplier;
      return builder;
    },

    theta(theta) {
      config.theta = theta;
      return builder;
    },

    decay(velocityDecay) {
      config.velocityDecay = velocityDecay;
      return builder;
    },

    build() {
      return { ...config };
    },
  };

  return builder;
}

/**
 * Validates force configuration values
 *
 * @param config - Configuration to validate
 * @returns Validated configuration with clamped values
 */
export function validateForceConfig(
  config: Partial<FullForceConfig>,
): FullForceConfig {
  const result = { ...DEFAULT_FORCE_CONFIG, ...config };

  // Clamp theta to valid range
  result.theta = Math.max(0.1, Math.min(2.0, result.theta));

  // Ensure positive distances
  result.repulsionDistanceMax = Math.max(1, result.repulsionDistanceMax);
  result.repulsionDistanceMin = Math.max(0.1, result.repulsionDistanceMin);

  // Ensure valid spring parameters
  result.springStrength = Math.max(0, result.springStrength);
  result.springLength = Math.max(1, result.springLength);

  // Clamp velocity decay
  result.velocityDecay = Math.max(0, Math.min(1, result.velocityDecay));

  // Ensure positive max velocity
  result.maxVelocity = Math.max(1, result.maxVelocity);

  // Clamp collision parameters
  result.collisionStrength = Math.max(0, Math.min(1, result.collisionStrength));
  result.collisionRadiusMultiplier = Math.max(0.1, result.collisionRadiusMultiplier);
  result.collisionIterations = Math.max(1, Math.floor(result.collisionIterations));

  return result;
}

/**
 * Convert force config to GPU uniform buffer data
 *
 * @param config - Force configuration
 * @returns Float32Array for uniform buffer
 */
export function forceConfigToUniformData(config: FullForceConfig): Float32Array {
  // Layout: 16 floats (64 bytes, aligned to 16-byte boundary)
  return new Float32Array([
    config.repulsionStrength,
    config.repulsionDistanceMax,
    config.repulsionDistanceMin,
    config.theta,
    config.springStrength,
    config.springLength,
    config.centerStrength,
    0, // padding
    config.centerX,
    config.centerY,
    config.velocityDecay,
    config.maxVelocity,
    config.timeStep,
    config.collisionEnabled ? 1.0 : 0.0,
    config.collisionStrength,
    config.collisionRadiusMultiplier,
  ]);
}

/**
 * Merge partial config with defaults
 *
 * @param partial - Partial configuration
 * @returns Full configuration
 */
export function mergeForceConfig(
  partial: Partial<FullForceConfig>,
): FullForceConfig {
  return { ...DEFAULT_FORCE_CONFIG, ...partial };
}
