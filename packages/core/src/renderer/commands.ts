/**
 * GPU Command Encoder Orchestration
 *
 * Coordinates GPU command encoding for compute passes (simulation)
 * and render passes (visualization) in the correct order.
 *
 * @module
 */

import type { GPUContext } from "../webgpu/context.ts";
import type { NodeRenderPipeline } from "./pipelines/nodes.ts";
import type { EdgeRenderPipeline } from "./pipelines/edges.ts";
import { toArrayBuffer } from "../webgpu/buffer_utils.ts";

/**
 * Clear color configuration
 */
export interface ClearColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

/**
 * Default clear color (light gray background)
 */
export const DEFAULT_CLEAR_COLOR: ClearColor = {
  r: 0.95,
  g: 0.95,
  b: 0.95,
  a: 1.0,
};

/**
 * Render pass configuration
 */
export interface RenderPassConfig {
  /** Clear color for the frame */
  clearColor?: ClearColor;
  /** Whether to clear depth buffer */
  clearDepth?: boolean;
  /** Load operation for color attachment */
  loadOp?: GPULoadOp;
  /** Store operation for color attachment */
  storeOp?: GPUStoreOp;
}

/**
 * Frame rendering context passed to render functions
 */
export interface FrameContext {
  /** GPU device */
  device: GPUDevice;
  /** Current frame's command encoder */
  encoder: GPUCommandEncoder;
  /** Current texture to render to */
  texture: GPUTexture;
  /** Current texture view */
  textureView: GPUTextureView;
  /** Frame number */
  frameNumber: number;
  /** Delta time since last frame (seconds) */
  deltaTime: number;
}

/**
 * Bind groups for a frame
 */
export interface FrameBindGroups {
  /** Viewport uniform bind group */
  viewport: GPUBindGroup;
  /** Node data bind group */
  nodes: GPUBindGroup;
  /** Edge data bind group */
  edges: GPUBindGroup;
}

/**
 * GPU command orchestrator
 */
export interface CommandOrchestrator {
  /** Begin a new frame */
  beginFrame: (deltaTime: number) => FrameContext;
  /** End the current frame and submit commands */
  endFrame: (context: FrameContext) => void;
  /** Record render commands for graph visualization */
  recordRenderPass: (
    context: FrameContext,
    bindGroups: FrameBindGroups,
    nodeCount: number,
    edgeCount: number,
  ) => void;
  /** Record simulation compute commands */
  recordComputePass: (
    context: FrameContext,
    computeCallback: (pass: GPUComputePassEncoder) => void,
  ) => void;
  /** Update clear color */
  setClearColor: (color: ClearColor) => void;
  /** Get frame statistics */
  readonly stats: CommandStats;
}

/**
 * Command statistics
 */
export interface CommandStats {
  /** Number of draw calls per frame */
  drawCalls: number;
  /** Number of compute dispatches per frame */
  computeDispatches: number;
  /** Number of buffer updates per frame */
  bufferUpdates: number;
}

/**
 * Command orchestrator configuration
 */
export interface CommandOrchestratorConfig {
  /** Initial clear color */
  clearColor?: ClearColor;
  /** Node render pipeline */
  nodePipeline: NodeRenderPipeline;
  /** Edge render pipeline */
  edgePipeline: EdgeRenderPipeline;
}

/**
 * Creates a GPU command orchestrator
 *
 * @param gpuContext - GPU context
 * @param config - Orchestrator configuration
 * @returns Command orchestrator
 */
export function createCommandOrchestrator(
  gpuContext: GPUContext,
  config: CommandOrchestratorConfig,
): CommandOrchestrator {
  const { device, context } = gpuContext;
  const { nodePipeline, edgePipeline } = config;

  let clearColor = config.clearColor ?? DEFAULT_CLEAR_COLOR;
  let frameNumber = 0;

  // Statistics tracking
  const stats: CommandStats = {
    drawCalls: 0,
    computeDispatches: 0,
    bufferUpdates: 0,
  };

  /**
   * Begin a new frame
   */
  function beginFrame(deltaTime: number): FrameContext {
    frameNumber++;

    // Reset per-frame stats
    stats.drawCalls = 0;
    stats.computeDispatches = 0;
    stats.bufferUpdates = 0;

    // Get current texture from swap chain
    const texture = context.getCurrentTexture();
    const textureView = texture.createView();

    // Create command encoder for this frame
    const encoder = device.createCommandEncoder({
      label: `Frame ${frameNumber} Commands`,
    });

    return {
      device,
      encoder,
      texture,
      textureView,
      frameNumber,
      deltaTime,
    };
  }

  /**
   * End the current frame and submit commands
   */
  function endFrame(context: FrameContext): void {
    const commandBuffer = context.encoder.finish();
    device.queue.submit([commandBuffer]);
  }

  /**
   * Record render commands for graph visualization
   */
  function recordRenderPass(
    frameContext: FrameContext,
    bindGroups: FrameBindGroups,
    nodeCount: number,
    edgeCount: number,
  ): void {
    const { encoder, textureView } = frameContext;

    // Create render pass
    const renderPass = encoder.beginRenderPass({
      label: "Graph Render Pass",
      colorAttachments: [
        {
          view: textureView,
          clearValue: clearColor,
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    // Render edges first (below nodes)
    if (edgeCount > 0) {
      renderPass.setPipeline(edgePipeline.pipeline);
      renderPass.setBindGroup(0, bindGroups.viewport);
      renderPass.setBindGroup(1, bindGroups.edges);
      renderPass.draw(6, edgeCount);
      stats.drawCalls++;
    }

    // Render nodes
    if (nodeCount > 0) {
      renderPass.setPipeline(nodePipeline.pipeline);
      renderPass.setBindGroup(0, bindGroups.viewport);
      renderPass.setBindGroup(1, bindGroups.nodes);
      renderPass.draw(6, nodeCount);
      stats.drawCalls++;
    }

    renderPass.end();
  }

  /**
   * Record simulation compute commands
   */
  function recordComputePass(
    frameContext: FrameContext,
    computeCallback: (pass: GPUComputePassEncoder) => void,
  ): void {
    const { encoder } = frameContext;

    const computePass = encoder.beginComputePass({
      label: "Simulation Compute Pass",
    });

    computeCallback(computePass);

    computePass.end();
  }

  /**
   * Update clear color
   */
  function setClearColor(color: ClearColor): void {
    clearColor = color;
  }

  return {
    beginFrame,
    endFrame,
    recordRenderPass,
    recordComputePass,
    setClearColor,
    get stats() {
      return { ...stats };
    },
  };
}

/**
 * Creates a render pass descriptor for simple rendering
 *
 * @param textureView - Texture view to render to
 * @param clearColor - Color to clear the framebuffer with
 * @param loadOp - Whether to clear or load existing content
 * @returns Render pass descriptor
 */
export function createRenderPassDescriptor(
  textureView: GPUTextureView,
  clearColor: ClearColor = DEFAULT_CLEAR_COLOR,
  loadOp: GPULoadOp = "clear",
): GPURenderPassDescriptor {
  return {
    colorAttachments: [
      {
        view: textureView,
        clearValue: clearColor,
        loadOp,
        storeOp: "store",
      },
    ],
  };
}

/**
 * Creates a compute pass descriptor
 *
 * @param label - Debug label for the pass
 * @returns Compute pass descriptor
 */
export function createComputePassDescriptor(
  label?: string,
): GPUComputePassDescriptor {
  return label !== undefined ? { label } : {};
}

/**
 * Helper to dispatch compute workgroups
 *
 * @param pass - Compute pass encoder
 * @param pipeline - Compute pipeline
 * @param bindGroups - Bind groups to set
 * @param workgroupCount - Number of workgroups [x, y, z]
 */
export function dispatchCompute(
  pass: GPUComputePassEncoder,
  pipeline: GPUComputePipeline,
  bindGroups: GPUBindGroup[],
  workgroupCount: [number, number?, number?],
): void {
  pass.setPipeline(pipeline);

  for (let i = 0; i < bindGroups.length; i++) {
    pass.setBindGroup(i, bindGroups[i]);
  }

  const [x, y = 1, z = 1] = workgroupCount;
  pass.dispatchWorkgroups(x, y, z);
}

/**
 * Calculate workgroup count for a given element count
 *
 * @param elementCount - Number of elements to process
 * @param workgroupSize - Size of each workgroup
 * @returns Number of workgroups needed
 */
export function calculateWorkgroups(
  elementCount: number,
  workgroupSize: number = 256,
): number {
  return Math.ceil(elementCount / workgroupSize);
}

/**
 * Buffer update helper with proper staging
 */
export interface BufferUpdater {
  /** Queue a buffer update */
  update: (buffer: GPUBuffer, data: ArrayBufferView, offset?: number) => void;
  /** Flush all pending updates */
  flush: () => void;
}

/**
 * Creates a buffer updater for efficient batched updates
 *
 * @param device - GPU device
 * @returns Buffer updater
 */
export function createBufferUpdater(device: GPUDevice): BufferUpdater {
  const pendingUpdates: Array<{
    buffer: GPUBuffer;
    data: ArrayBufferView;
    offset: number;
  }> = [];

  return {
    update(buffer: GPUBuffer, data: ArrayBufferView, offset: number = 0): void {
      pendingUpdates.push({ buffer, data, offset });
    },

    flush(): void {
      for (const { buffer, data, offset } of pendingUpdates) {
        // Convert to ArrayBuffer to satisfy BufferSource type
        const arrayBuffer = toArrayBuffer(
          data as Float32Array | Uint32Array | Uint8Array | Int32Array | Uint16Array | Int16Array,
        );
        device.queue.writeBuffer(buffer, offset, arrayBuffer);
      }
      pendingUpdates.length = 0;
    },
  };
}
