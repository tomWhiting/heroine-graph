/**
 * WebGPU Context Management
 *
 * Handles WebGPU device/adapter initialization and lifecycle.
 */

import { ErrorCode, Errors, HeroineGraphError } from "../errors.ts";

/**
 * GPU context configuration options.
 */
export interface GPUContextOptions {
  /** Canvas element for rendering */
  readonly canvas: HTMLCanvasElement;
  /** Preferred power mode */
  readonly powerPreference?: "high-performance" | "low-power";
  /** Enable WebGPU debugging */
  readonly debug?: boolean;
  /** Alpha mode for canvas */
  readonly alphaMode?: GPUCanvasAlphaMode;
  /** Preferred texture format (auto-detected if not specified) */
  readonly format?: GPUTextureFormat;
  /** Callback when the GPU device is lost */
  readonly onDeviceLost?: DeviceLostCallback;
}

/**
 * Callback for device loss events.
 */
export type DeviceLostCallback = (reason: string, message: string) => void;

/**
 * Initialized GPU context containing all WebGPU resources.
 */
export interface GPUContext {
  /** The WebGPU adapter */
  readonly adapter: GPUAdapter;
  /** The WebGPU device */
  readonly device: GPUDevice;
  /** The canvas context for rendering */
  readonly context: GPUCanvasContext;
  /** The preferred texture format */
  readonly format: GPUTextureFormat;
  /** The canvas element */
  readonly canvas: HTMLCanvasElement;
  /** Device limits */
  readonly limits: GPUSupportedLimits;
  /** Whether the device has been lost */
  isDeviceLost: boolean;
}

/**
 * Default required limits for Heroine Graph.
 *
 * These are the minimum limits we need for graph visualization.
 */
const DEFAULT_REQUIRED_LIMITS: Record<string, number> = {
  maxStorageBufferBindingSize: 256 * 1024 * 1024, // 256 MB
  maxBufferSize: 256 * 1024 * 1024, // 256 MB
  maxComputeWorkgroupsPerDimension: 65535,
  maxStorageBuffersPerShaderStage: 10, // Integration shader needs 10 storage buffers
};

/**
 * Initialize WebGPU and create a GPU context.
 *
 * @param options Configuration options
 * @returns Promise resolving to the GPU context
 * @throws HeroineGraphError if WebGPU initialization fails
 */
export async function createGPUContext(options: GPUContextOptions): Promise<GPUContext> {
  const { canvas, powerPreference = "high-performance", debug = false, alphaMode = "opaque" } =
    options;

  // Check WebGPU availability
  if (!navigator.gpu) {
    throw Errors.webgpuNotSupported("navigator.gpu is not available");
  }

  // Request adapter
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference,
  });

  if (!adapter) {
    throw Errors.adapterFailed();
  }

  // Determine required limits - use Record for mutable limits object
  const requiredLimits: Record<string, number> = {};
  const unsupportedLimits: string[] = [];

  for (const [key, value] of Object.entries(DEFAULT_REQUIRED_LIMITS)) {
    const adapterLimit = (adapter.limits as unknown as Record<string, number>)[key];

    if (typeof adapterLimit === "number") {
      if (adapterLimit >= value) {
        requiredLimits[key] = value;
      } else {
        // Use adapter's maximum if our requirement is too high
        requiredLimits[key] = adapterLimit;
        if (adapterLimit < value * 0.5) {
          // Only warn if significantly lower
          unsupportedLimits.push(`${key}: ${adapterLimit} < ${value}`);
        }
      }
    }
  }

  // Request device with features
  const requiredFeatures: GPUFeatureName[] = [];

  // Add timestamp query if available (useful for profiling)
  if (adapter.features.has("timestamp-query")) {
    requiredFeatures.push("timestamp-query");
  }

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({
      requiredLimits,
      requiredFeatures,
    });
  } catch (error) {
    throw new HeroineGraphError(
      ErrorCode.WEBGPU_DEVICE_FAILED,
      `Failed to create GPU device: ${error instanceof Error ? error.message : String(error)}`,
      { originalError: error, unsupportedLimits },
    );
  }

  // Set up error handling
  device.addEventListener("uncapturederror", (event) => {
    const gpuEvent = event as GPUUncapturedErrorEvent;
    console.error("[HeroineGraph] Uncaptured WebGPU error:", gpuEvent.error);
    if (debug) {
      console.error("Error details:", gpuEvent.error.message);
    }
  });

  // Handle device loss (handler registered once, updates context + calls user callback)

  // Get canvas context - cast to GPUCanvasContext since getContext returns generic type
  const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
  if (!context) {
    device.destroy();
    throw new HeroineGraphError(
      ErrorCode.CANVAS_NOT_FOUND,
      "Failed to get WebGPU context from canvas",
      {},
      "Ensure the canvas element supports WebGPU",
    );
  }

  // Determine format
  const format = options.format ?? navigator.gpu.getPreferredCanvasFormat();

  // Configure canvas
  context.configure({
    device,
    format,
    alphaMode,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  const gpuContext: GPUContext = {
    adapter,
    device,
    context,
    format,
    canvas,
    limits: device.limits,
    isDeviceLost: false,
  };

  // Handle device loss
  device.lost.then((info) => {
    gpuContext.isDeviceLost = true;
    console.error("[HeroineGraph] GPU device lost:", info.message);
    console.error("Reason:", info.reason);
    if (options.onDeviceLost) {
      options.onDeviceLost(info.reason ?? "unknown", info.message);
    }
  });

  return gpuContext;
}

/**
 * Destroy a GPU context and release resources.
 *
 * @param ctx The GPU context to destroy
 */
export function destroyGPUContext(ctx: GPUContext): void {
  ctx.context.unconfigure();
  ctx.device.destroy();
}

/**
 * Resize the GPU context canvas.
 *
 * Call this when the canvas size changes.
 *
 * @param ctx The GPU context
 * @param width New width in pixels
 * @param height New height in pixels
 */
export function resizeGPUContext(ctx: GPUContext, width: number, height: number): void {
  const dpr = globalThis.devicePixelRatio || 1;
  const scaledWidth = Math.floor(width * dpr);
  const scaledHeight = Math.floor(height * dpr);

  if (ctx.canvas.width !== scaledWidth || ctx.canvas.height !== scaledHeight) {
    ctx.canvas.width = scaledWidth;
    ctx.canvas.height = scaledHeight;
  }
}

/**
 * Get the current render texture from the canvas.
 *
 * @param ctx The GPU context
 * @returns The current texture for rendering
 */
export function getCurrentTexture(ctx: GPUContext): GPUTexture {
  return ctx.context.getCurrentTexture();
}

/**
 * Create a depth texture for the canvas.
 *
 * @param ctx The GPU context
 * @returns A depth texture matching the canvas size
 */
export function createDepthTexture(ctx: GPUContext): GPUTexture {
  return ctx.device.createTexture({
    size: {
      width: ctx.canvas.width,
      height: ctx.canvas.height,
    },
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
    label: "Depth Texture",
  });
}

/**
 * Calculate the maximum number of nodes the GPU can handle.
 *
 * @param ctx The GPU context
 * @returns Estimated maximum node count
 */
export function estimateMaxNodes(ctx: GPUContext): number {
  // Each node needs: 2 floats for position, 2 for velocity = 16 bytes
  // Plus attributes, selection state, etc. = ~32 bytes total
  const bytesPerNode = 32;
  const maxBufferSize = ctx.limits.maxStorageBufferBindingSize;
  return Math.floor(maxBufferSize / bytesPerNode);
}
