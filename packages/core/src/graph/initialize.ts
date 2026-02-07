/**
 * Initial Position Randomization
 *
 * Provides strategies for initializing node positions when not provided.
 * Good initial positions can significantly reduce simulation time.
 *
 * @module
 */

/**
 * Initialization strategy type
 */
export type InitializationStrategy =
  | "random"
  | "grid"
  | "circle"
  | "spiral"
  | "phyllotaxis";

/**
 * Initialization configuration
 */
export interface InitializeConfig {
  /** Strategy for position initialization */
  strategy?: InitializationStrategy;
  /** Center X coordinate */
  centerX?: number;
  /** Center Y coordinate */
  centerY?: number;
  /** Spread radius */
  radius?: number;
  /** Random seed for reproducibility (0 = random) */
  seed?: number;
}

/**
 * Default initialization configuration
 */
export const DEFAULT_INITIALIZE_CONFIG: Required<InitializeConfig> = {
  strategy: "phyllotaxis",
  centerX: 0,
  centerY: 0,
  radius: 100,
  seed: 0,
};

/**
 * Simple seeded random number generator (Mulberry32)
 */
function mulberry32(seed: number): () => number {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Initialize positions with random placement
 *
 * @param positionsX - X positions array to fill
 * @param positionsY - Y positions array to fill
 * @param config - Initialization configuration
 */
export function initializeRandom(
  positionsX: Float32Array,
  positionsY: Float32Array,
  config: InitializeConfig = {},
): void {
  const finalConfig = { ...DEFAULT_INITIALIZE_CONFIG, ...config };
  const { centerX, centerY, radius, seed } = finalConfig;
  const count = positionsX.length;

  const random = seed > 0 ? mulberry32(seed) : Math.random;

  for (let i = 0; i < count; i++) {
    // Random angle and radius for circular distribution
    const angle = random() * 2 * Math.PI;
    const r = Math.sqrt(random()) * radius; // sqrt for uniform distribution

    positionsX[i] = centerX + r * Math.cos(angle);
    positionsY[i] = centerY + r * Math.sin(angle);
  }
}

/**
 * Initialize positions in a grid pattern
 *
 * @param positionsX - X positions array to fill
 * @param positionsY - Y positions array to fill
 * @param config - Initialization configuration
 */
export function initializeGrid(
  positionsX: Float32Array,
  positionsY: Float32Array,
  config: InitializeConfig = {},
): void {
  const finalConfig = { ...DEFAULT_INITIALIZE_CONFIG, ...config };
  const { centerX, centerY, radius } = finalConfig;
  const count = positionsX.length;

  // Calculate grid dimensions
  const cols = Math.ceil(Math.sqrt(count));
  const spacing = (radius * 2) / Math.max(cols - 1, 1);

  const startX = centerX - radius;
  const startY = centerY - radius;

  for (let i = 0; i < count; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);

    positionsX[i] = startX + col * spacing;
    positionsY[i] = startY + row * spacing;
  }
}

/**
 * Initialize positions in a circle
 *
 * @param positionsX - X positions array to fill
 * @param positionsY - Y positions array to fill
 * @param config - Initialization configuration
 */
export function initializeCircle(
  positionsX: Float32Array,
  positionsY: Float32Array,
  config: InitializeConfig = {},
): void {
  const finalConfig = { ...DEFAULT_INITIALIZE_CONFIG, ...config };
  const { centerX, centerY, radius } = finalConfig;
  const count = positionsX.length;

  const angleStep = (2 * Math.PI) / count;

  for (let i = 0; i < count; i++) {
    const angle = i * angleStep;
    positionsX[i] = centerX + radius * Math.cos(angle);
    positionsY[i] = centerY + radius * Math.sin(angle);
  }
}

/**
 * Initialize positions in a spiral pattern
 *
 * @param positionsX - X positions array to fill
 * @param positionsY - Y positions array to fill
 * @param config - Initialization configuration
 */
export function initializeSpiral(
  positionsX: Float32Array,
  positionsY: Float32Array,
  config: InitializeConfig = {},
): void {
  const finalConfig = { ...DEFAULT_INITIALIZE_CONFIG, ...config };
  const { centerX, centerY, radius } = finalConfig;
  const count = positionsX.length;

  // Archimedean spiral
  const turns = 3; // Number of turns
  const maxAngle = turns * 2 * Math.PI;

  for (let i = 0; i < count; i++) {
    const t = count > 1 ? i / (count - 1) : 0;
    const angle = t * maxAngle;
    const r = t * radius;

    positionsX[i] = centerX + r * Math.cos(angle);
    positionsY[i] = centerY + r * Math.sin(angle);
  }
}

/**
 * Initialize positions using phyllotaxis (sunflower) pattern
 *
 * Provides very good initial positions that minimize overlap and
 * distribute nodes evenly in a circular area.
 *
 * @param positionsX - X positions array to fill
 * @param positionsY - Y positions array to fill
 * @param config - Initialization configuration
 */
export function initializePhyllotaxis(
  positionsX: Float32Array,
  positionsY: Float32Array,
  config: InitializeConfig = {},
): void {
  const finalConfig = { ...DEFAULT_INITIALIZE_CONFIG, ...config };
  const { centerX, centerY, radius } = finalConfig;
  const count = positionsX.length;

  // Golden angle in radians
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  for (let i = 0; i < count; i++) {
    const angle = i * goldenAngle;
    // Fermat spiral for even distribution
    const r = Math.sqrt(i / count) * radius;

    positionsX[i] = centerX + r * Math.cos(angle);
    positionsY[i] = centerY + r * Math.sin(angle);
  }
}

/**
 * Initialize positions using the specified strategy
 *
 * @param positionsX - X positions array to fill
 * @param positionsY - Y positions array to fill
 * @param config - Initialization configuration
 */
export function initializePositions(
  positionsX: Float32Array,
  positionsY: Float32Array,
  config: InitializeConfig = {},
): void {
  const finalConfig = { ...DEFAULT_INITIALIZE_CONFIG, ...config };

  switch (finalConfig.strategy) {
    case "random":
      initializeRandom(positionsX, positionsY, finalConfig);
      break;
    case "grid":
      initializeGrid(positionsX, positionsY, finalConfig);
      break;
    case "circle":
      initializeCircle(positionsX, positionsY, finalConfig);
      break;
    case "spiral":
      initializeSpiral(positionsX, positionsY, finalConfig);
      break;
    case "phyllotaxis":
    default:
      initializePhyllotaxis(positionsX, positionsY, finalConfig);
      break;
  }
}

/**
 * Check if positions need initialization (all zeros)
 *
 * @param positionsX - X positions to check
 * @param positionsY - Y positions to check
 * @returns True if positions are all zero
 */
export function needsInitialization(
  positionsX: Float32Array,
  positionsY: Float32Array,
): boolean {
  const count = positionsX.length;

  // Check first 100 nodes or all if fewer
  const checkCount = Math.min(count, 100);

  let allZeroX = true;
  let allZeroY = true;

  for (let i = 0; i < checkCount; i++) {
    if (positionsX[i] !== 0) allZeroX = false;
    if (positionsY[i] !== 0) allZeroY = false;
    if (!allZeroX && !allZeroY) return false;
  }

  return allZeroX && allZeroY;
}

/**
 * Add jitter to existing positions
 *
 * Useful for restarting simulation or escaping local minima.
 *
 * @param positionsX - X positions to jitter
 * @param positionsY - Y positions to jitter
 * @param amount - Maximum jitter amount
 * @param seed - Random seed (0 = random)
 */
export function addJitter(
  positionsX: Float32Array,
  positionsY: Float32Array,
  amount: number = 1,
  seed: number = 0,
): void {
  const random = seed > 0 ? mulberry32(seed) : Math.random;
  const count = positionsX.length;

  for (let i = 0; i < count; i++) {
    positionsX[i] += (random() - 0.5) * 2 * amount;
    positionsY[i] += (random() - 0.5) * 2 * amount;
  }
}
