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
  /**
   * Whether the loop has been started and not stopped. Independent of
   * `isPaused`: frames are presented only while `isRunning && !isPaused`.
   */
  readonly isRunning: boolean;
  /**
   * Whether presentation is suspended. A latch, not a lifecycle state — it
   * survives `stop`/`start`, so a loop paused before it was ever started stays
   * paused when something else starts it.
   */
  readonly isPaused: boolean;
  /** Current frame statistics */
  readonly stats: FrameStats;
  /** Start the render loop */
  start: () => void;
  /** Stop the render loop */
  stop: () => void;
  /** Suspend presentation, keeping frame timing and stats continuous */
  pause: () => void;
  /** Resume presentation suspended by `pause` */
  resume: () => void;
  /** Request a single frame render (when not presenting) */
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
  let isPaused = false;
  let animationFrameId: number | null = null;
  let lastFrameTime = 0;
  let startTime = 0;
  let frameCount = 0;
  let lastStatsTime = 0;
  /** Timestamp of the `pause` that owns the current suspension. */
  let pausedAt = 0;

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
  const getTargetFrameTime = () => finalConfig.targetFps > 0 ? 1000 / finalConfig.targetFps : 0;

  /**
   * Main render loop function
   */
  function loop(currentTime: number): void {
    if (!isRunning || isPaused) return;

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
   * Cancel the frame this loop has in flight, if any
   */
  function cancelPendingFrame(): void {
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
  }

  /**
   * Start the render loop
   *
   * Honours an outstanding `pause`: starting a paused loop arms it without
   * presenting, so whoever paused stays in control of when frames resume.
   */
  function start(): void {
    if (isRunning) return;

    isRunning = true;
    lastFrameTime = 0;
    if (!isPaused) animationFrameId = requestAnimationFrame(loop);
  }

  /**
   * Stop the render loop
   */
  function stop(): void {
    isRunning = false;
    cancelPendingFrame();
  }

  /**
   * Suspend presentation
   *
   * Unlike `stop`, this preserves the loop's timing origin so `elapsed` and the
   * frame-time average survive the suspension instead of restarting.
   */
  function pause(): void {
    if (isPaused) return;

    isPaused = true;
    pausedAt = performance.now();
    cancelPendingFrame();
  }

  /**
   * Resume presentation suspended by `pause`
   *
   * The paused interval is neither a frame time nor elapsed run time, so it is
   * discounted from both rather than landing on the first frame back as a
   * multi-second delta.
   */
  function resume(): void {
    if (!isPaused) return;

    isPaused = false;
    // A loop that never ran a frame has no timing origin to preserve; its first
    // frame will seed one, as after `start`.
    if (lastFrameTime !== 0) {
      const now = performance.now();
      const suspended = now - pausedAt;
      startTime += suspended;
      lastStatsTime += suspended;
      lastFrameTime = now;
    }
    if (isRunning && animationFrameId === null) {
      animationFrameId = requestAnimationFrame(loop);
    }
  }

  /**
   * Request a single frame render (when not presenting)
   */
  function requestFrame(): void {
    if (isRunning && !isPaused) return;

    const now = performance.now();
    const deltaTime = lastFrameTime > 0 ? now - lastFrameTime : 16.67; // Default to ~60fps
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
    get isPaused() {
      return isPaused;
    },
    get stats() {
      return { ...stats };
    },
    start,
    stop,
    pause,
    resume,
    requestFrame,
    setConfig,
  };
}

/**
 * GPU timing helper using timestamp queries (if available)
 *
 * Uses the current WebGPU API: timestamps are recorded via the
 * `timestampWrites` member of a render/compute pass descriptor
 * (`encoder.writeTimestamp` was removed from the spec). The timer never
 * finishes or submits the caller's encoder — the caller owns submission.
 *
 * Usage:
 * ```ts
 * const timer = createGPUTimer(device);
 * if (timer) {
 *   const pass = encoder.beginComputePass({ timestampWrites: timer.timestampWrites });
 *   // ... encode timed work ...
 *   pass.end();
 *   timer.resolve(encoder);
 *   device.queue.submit([encoder.finish()]); // caller submits
 *   const gpuMs = await timer.read();
 * }
 * ```
 */
export interface GPUTimer {
  /**
   * Timestamp-writes descriptor to attach to a single render or compute
   * pass descriptor; records the pass's begin and end times.
   */
  readonly timestampWrites: GPUComputePassTimestampWrites & GPURenderPassTimestampWrites;
  /**
   * Encode query resolution into the caller's encoder. Call after the
   * timed pass has ended, before the caller finishes/submits the encoder.
   */
  resolve: (encoder: GPUCommandEncoder) => void;
  /**
   * Read back the elapsed GPU time in milliseconds. Call after the
   * caller has submitted the encoder that `resolve` recorded into.
   */
  read: () => Promise<number>;
  /** Destroy GPU resources */
  destroy?: () => void;
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

  // Create query set for timestamps (2 queries: pass begin and end)
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

    timestampWrites: {
      querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    },

    resolve(encoder: GPUCommandEncoder): void {
      encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      // Copy to mappable buffer; the caller finishes and submits.
      encoder.copyBufferToBuffer(resolveBuffer, 0, readbackBuffer, 0, 16);
    },

    async read(): Promise<number> {
      // Wait for GPU to complete, then read back
      await readbackBuffer.mapAsync(GPUMapMode.READ);
      const data = new BigUint64Array(readbackBuffer.getMappedRange());
      const startTime = data[0];
      const endTime = data[1];
      readbackBuffer.unmap();

      // Convert to milliseconds (timestamps are in nanoseconds)
      return Number(endTime - startTime) / 1_000_000;
    },

    destroy(): void {
      querySet.destroy();
      resolveBuffer.destroy();
      readbackBuffer.destroy();
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
