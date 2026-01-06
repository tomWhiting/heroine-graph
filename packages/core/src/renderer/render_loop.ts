/**
 * Main Render Loop
 *
 * Manages the animation frame loop with precise timing for consistent
 * frame rates and performance monitoring.
 *
 * @module
 */

/**
 * Frame timing statistics
 */
export interface FrameStats {
  /** Current FPS (frames per second) */
  fps: number;
  /** Frame time in milliseconds */
  frameTime: number;
  /** Average frame time over recent frames */
  avgFrameTime: number;
  /** Number of frames rendered */
  frameCount: number;
  /** Time since render loop started (ms) */
  elapsed: number;
  /** GPU time if available (ms) */
  gpuTime?: number;
}

/**
 * Render callback function type
 */
export type RenderCallback = (deltaTime: number, stats: FrameStats) => void;

/**
 * Render loop configuration
 */
export interface RenderLoopConfig {
  /** Target FPS (0 for uncapped) */
  targetFps?: number;
  /** Number of frames to average for FPS calculation */
  fpsAverageFrames?: number;
  /** Enable performance monitoring */
  enableStats?: boolean;
  /** Callback for stats updates (called every statsInterval ms) */
  onStats?: (stats: FrameStats) => void;
  /** Stats callback interval in milliseconds */
  statsInterval?: number;
}

/**
 * Default render loop configuration
 */
export const DEFAULT_RENDER_LOOP_CONFIG: {
  targetFps: number;
  fpsAverageFrames: number;
  enableStats: boolean;
  onStats?: ((stats: FrameStats) => void) | undefined;
  statsInterval: number;
} = {
  targetFps: 0, // Uncapped by default
  fpsAverageFrames: 60,
  enableStats: true,
  statsInterval: 1000,
};

/**
 * Render loop state
 */
export interface RenderLoop {
  /** Whether the loop is running */
  readonly isRunning: boolean;
  /** Current frame statistics */
  readonly stats: FrameStats;
  /** Start the render loop */
  start: () => void;
  /** Stop the render loop */
  stop: () => void;
  /** Request a single frame render (when paused) */
  requestFrame: () => void;
  /** Update configuration */
  setConfig: (config: Partial<RenderLoopConfig>) => void;
}

/**
 * Creates a render loop with frame timing
 *
 * @param renderCallback - Function called each frame with delta time
 * @param config - Render loop configuration
 * @returns Render loop controller
 */
export function createRenderLoop(
  renderCallback: RenderCallback,
  config: RenderLoopConfig = {},
): RenderLoop {
  const finalConfig = { ...DEFAULT_RENDER_LOOP_CONFIG, ...config };

  // State
  let isRunning = false;
  let animationFrameId: number | null = null;
  let lastFrameTime = 0;
  let startTime = 0;
  let frameCount = 0;
  let lastStatsTime = 0;

  // Frame time buffer for averaging
  const frameTimes: number[] = [];
  let frameTimeIndex = 0;

  // Stats
  const stats: FrameStats = {
    fps: 0,
    frameTime: 0,
    avgFrameTime: 0,
    frameCount: 0,
    elapsed: 0,
  };

  // Target frame time for capping
  const getTargetFrameTime = () =>
    finalConfig.targetFps > 0 ? 1000 / finalConfig.targetFps : 0;

  /**
   * Main render loop function
   */
  function loop(currentTime: number): void {
    if (!isRunning) return;

    // Schedule next frame first for consistent timing
    animationFrameId = requestAnimationFrame(loop);

    // Initialize timing on first frame
    if (lastFrameTime === 0) {
      lastFrameTime = currentTime;
      startTime = currentTime;
      lastStatsTime = currentTime;
      return;
    }

    // Calculate delta time
    const deltaTime = currentTime - lastFrameTime;

    // Frame rate limiting
    const targetFrameTime = getTargetFrameTime();
    if (targetFrameTime > 0 && deltaTime < targetFrameTime) {
      return; // Skip frame if under target time
    }

    lastFrameTime = currentTime;
    frameCount++;

    // Update frame time buffer
    if (finalConfig.enableStats) {
      frameTimes[frameTimeIndex] = deltaTime;
      frameTimeIndex = (frameTimeIndex + 1) % finalConfig.fpsAverageFrames;

      // Calculate average frame time
      const validFrames = Math.min(frameCount, finalConfig.fpsAverageFrames);
      let sum = 0;
      for (let i = 0; i < validFrames; i++) {
        sum += frameTimes[i] || 0;
      }
      const avgFrameTime = validFrames > 0 ? sum / validFrames : 0;

      // Update stats
      stats.frameTime = deltaTime;
      stats.avgFrameTime = avgFrameTime;
      stats.fps = avgFrameTime > 0 ? 1000 / avgFrameTime : 0;
      stats.frameCount = frameCount;
      stats.elapsed = currentTime - startTime;

      // Call stats callback at interval
      if (
        finalConfig.onStats &&
        currentTime - lastStatsTime >= finalConfig.statsInterval
      ) {
        finalConfig.onStats({ ...stats });
        lastStatsTime = currentTime;
      }
    }

    // Call render callback
    try {
      renderCallback(deltaTime / 1000, stats);
    } catch (error) {
      console.error("Error in render callback:", error);
      // Don't stop the loop on errors - let the application decide
    }
  }

  /**
   * Start the render loop
   */
  function start(): void {
    if (isRunning) return;

    isRunning = true;
    lastFrameTime = 0;
    animationFrameId = requestAnimationFrame(loop);
  }

  /**
   * Stop the render loop
   */
  function stop(): void {
    isRunning = false;
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
  }

  /**
   * Request a single frame render (when paused)
   */
  function requestFrame(): void {
    if (isRunning) return;

    const now = performance.now();
    const deltaTime =
      lastFrameTime > 0 ? now - lastFrameTime : 16.67; // Default to ~60fps
    lastFrameTime = now;
    frameCount++;

    stats.frameTime = deltaTime;
    stats.frameCount = frameCount;

    try {
      renderCallback(deltaTime / 1000, stats);
    } catch (error) {
      console.error("Error in render callback:", error);
    }
  }

  /**
   * Update configuration
   */
  function setConfig(newConfig: Partial<RenderLoopConfig>): void {
    Object.assign(finalConfig, newConfig);
  }

  return {
    get isRunning() {
      return isRunning;
    },
    get stats() {
      return { ...stats };
    },
    start,
    stop,
    requestFrame,
    setConfig,
  };
}

/**
 * GPU timing helper using timestamp queries (if available)
 */
export interface GPUTimer {
  /** Start timing a GPU operation */
  begin: (encoder: GPUCommandEncoder, label?: string) => void;
  /** End timing and get result promise */
  end: (encoder: GPUCommandEncoder) => Promise<number>;
  /** Check if GPU timing is supported */
  readonly isSupported: boolean;
}

/**
 * Creates a GPU timer for measuring GPU operation times
 *
 * Note: Timestamp queries require the "timestamp-query" feature to be enabled
 *
 * @param device - GPU device
 * @returns GPU timer or null if not supported
 */
export function createGPUTimer(device: GPUDevice): GPUTimer | null {
  // Check if timestamp queries are supported
  if (!device.features.has("timestamp-query")) {
    return null;
  }

  // Create query set for timestamps (2 queries: start and end)
  const querySet = device.createQuerySet({
    type: "timestamp",
    count: 2,
  });

  // Create buffer to read back results
  const resolveBuffer = device.createBuffer({
    size: 16, // 2 x 8-byte timestamps
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });

  const readbackBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  return {
    isSupported: true,

    begin(encoder: GPUCommandEncoder, _label?: string): void {
      encoder.writeTimestamp(querySet, 0);
    },

    async end(encoder: GPUCommandEncoder): Promise<number> {
      encoder.writeTimestamp(querySet, 1);
      encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);

      // Copy to mappable buffer
      encoder.copyBufferToBuffer(resolveBuffer, 0, readbackBuffer, 0, 16);

      // Submit and wait
      await readbackBuffer.mapAsync(GPUMapMode.READ);
      const data = new BigUint64Array(readbackBuffer.getMappedRange());
      const startTime = data[0];
      const endTime = data[1];
      readbackBuffer.unmap();

      // Convert to milliseconds (timestamps are in nanoseconds)
      return Number(endTime - startTime) / 1_000_000;
    },
  };
}

/**
 * Frame pacing helper for consistent frame delivery
 */
export interface FramePacer {
  /** Wait until next frame should be rendered */
  waitForNextFrame: () => Promise<void>;
  /** Reset timing */
  reset: () => void;
}

/**
 * Creates a frame pacer for smooth, consistent frame timing
 *
 * @param targetFps - Target frames per second
 * @returns Frame pacer
 */
export function createFramePacer(targetFps: number = 60): FramePacer {
  const targetFrameTime = 1000 / targetFps;
  let lastFrameTime = 0;

  return {
    async waitForNextFrame(): Promise<void> {
      const now = performance.now();
      const elapsed = now - lastFrameTime;
      const remaining = targetFrameTime - elapsed;

      if (remaining > 1) {
        // Use setTimeout for longer waits
        await new Promise((resolve) => setTimeout(resolve, remaining - 1));
      }

      // Spin-wait for precise timing (last millisecond)
      while (performance.now() - lastFrameTime < targetFrameTime) {
        // Spin
      }

      lastFrameTime = performance.now();
    },

    reset(): void {
      lastFrameTime = performance.now();
    },
  };
}
