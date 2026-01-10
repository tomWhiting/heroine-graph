/**
 * WebGPU Availability Check
 *
 * Provides functions to check if WebGPU is available and functional.
 */

/**
 * Result of WebGPU availability check.
 */
export interface WebGPUStatus {
  /** WebGPU is available and functional */
  readonly supported: boolean;
  /** Detailed error message if not supported */
  readonly error?: string;
  /** GPU adapter info if available */
  readonly adapterInfo?: {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
  };
  /** Supported limits */
  readonly limits?: {
    readonly maxBufferSize: number;
    readonly maxStorageBufferBindingSize: number;
    readonly maxComputeWorkgroupsPerDimension: number;
    readonly maxComputeInvocationsPerWorkgroup: number;
  };
}

/**
 * Check if WebGPU is available and functional.
 *
 * This function performs a complete check including:
 * 1. navigator.gpu existence
 * 2. Adapter request
 * 3. Device request
 * 4. Basic capability check
 *
 * @returns Promise resolving to WebGPU status
 */
export async function checkWebGPU(): Promise<WebGPUStatus> {
  // Check if WebGPU API exists
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return {
      supported: false,
      error: "WebGPU API not available. This browser does not support WebGPU.",
    };
  }

  const gpu = navigator.gpu;
  if (!gpu) {
    return {
      supported: false,
      error: "navigator.gpu is null. WebGPU may be disabled.",
    };
  }

  // Request adapter
  let adapter: GPUAdapter | null;
  try {
    adapter = await gpu.requestAdapter({
      powerPreference: "high-performance",
    });
  } catch (error) {
    return {
      supported: false,
      error: `Failed to request GPU adapter: ${
        error instanceof Error ? error.message : String(error)
      }`,
    };
  }

  if (!adapter) {
    return {
      supported: false,
      error: "No suitable GPU adapter found. " +
        "This may occur if no compatible GPU is available or GPU drivers need updating.",
    };
  }

  // Get adapter info (adapter.info is a property in modern WebGPU spec)
  const info = adapter.info;
  const adapterInfo = {
    vendor: info?.vendor || "unknown",
    architecture: info?.architecture || "unknown",
    device: info?.device || "unknown",
    description: info?.description || "unknown",
  };

  // Request device
  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({
      requiredLimits: {
        // Request limits we need for large graphs
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
  } catch (error) {
    return {
      supported: false,
      error: `Failed to request GPU device: ${
        error instanceof Error ? error.message : String(error)
      }`,
      adapterInfo,
    };
  }

  // Get limits
  const limits = {
    maxBufferSize: device.limits.maxBufferSize,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
    maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
  };

  // Clean up test device
  device.destroy();

  return {
    supported: true,
    adapterInfo,
    limits,
  };
}

/**
 * Quick check if WebGPU might be available.
 *
 * This is a synchronous check that only verifies the API exists,
 * not that it's functional. Use checkWebGPU() for a complete check.
 *
 * @returns True if navigator.gpu exists
 */
export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator && navigator.gpu !== null;
}

/**
 * Get a human-readable description of WebGPU support.
 *
 * @param status The WebGPU status from checkWebGPU()
 * @returns Human-readable status description
 */
export function describeWebGPUStatus(status: WebGPUStatus): string {
  if (!status.supported) {
    return `WebGPU not supported: ${status.error}`;
  }

  const parts = ["WebGPU supported"];

  if (status.adapterInfo) {
    const { vendor, device } = status.adapterInfo;
    if (vendor !== "unknown" || device !== "unknown") {
      parts.push(`GPU: ${vendor} ${device}`);
    }
  }

  if (status.limits) {
    const maxNodes = Math.floor(status.limits.maxBufferSize / (4 * 4)); // 4 floats per node
    parts.push(`Max nodes: ~${(maxNodes / 1_000_000).toFixed(1)}M`);
  }

  return parts.join(", ");
}
