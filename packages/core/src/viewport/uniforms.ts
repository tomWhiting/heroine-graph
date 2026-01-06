/**
 * Viewport Uniform Buffer for Shaders
 *
 * Manages the GPU uniform buffer containing viewport transformation
 * data for use in vertex and fragment shaders.
 */

import type { ViewportState } from "../types.ts";
import { graphToClipMatrix, Matrix3 } from "./transforms.ts";

/**
 * Viewport uniform data for shaders.
 *
 * Layout matches WGSL struct (16-byte aligned):
 * ```wgsl
 * struct ViewportUniforms {
 *     transform: mat3x3<f32>,  // 48 bytes (3 x vec4 due to padding)
 *     screen_size: vec2<f32>,  // 8 bytes
 *     scale: f32,              // 4 bytes
 *     inv_scale: f32,          // 4 bytes
 *     padding: vec2<f32>,      // 8 bytes for alignment
 * }
 * ```
 * Total: 72 bytes, aligned to 16 = 80 bytes
 */
export const VIEWPORT_UNIFORM_SIZE = 80;

/**
 * Manages the viewport uniform buffer.
 */
export class ViewportUniformBuffer {
  private readonly _device: GPUDevice;
  private readonly _buffer: GPUBuffer;
  private readonly data: Float32Array;
  private dirty: boolean;

  constructor(device: GPUDevice) {
    this._device = device;
    this.dirty = true;

    // mat3x3 is stored as 3 x vec4 (with padding) = 12 floats
    // + screen_size (2) + scale (1) + inv_scale (1) + padding (4)
    this.data = new Float32Array(20);

    this._buffer = device.createBuffer({
      size: VIEWPORT_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "ViewportUniforms",
    });
  }

  /**
   * Get the GPU buffer.
   */
  get buffer(): GPUBuffer {
    return this._buffer;
  }

  /**
   * Update the uniform data from viewport state.
   * Signature: update(device, state, width, height) or update(state)
   */
  update(
    deviceOrState: GPUDevice | ViewportState,
    state?: ViewportState,
    width?: number,
    height?: number
  ): void {
    // Handle both signatures
    let viewport: ViewportState;
    if (state !== undefined) {
      // Called as update(device, state, width, height)
      viewport = {
        ...state,
        width: width ?? state.width,
        height: height ?? state.height,
      };
    } else {
      // Called as update(state)
      viewport = deviceOrState as ViewportState;
    }

    const transform = graphToClipMatrix(viewport);

    // mat3x3 stored as 3 x vec4 (column-major, each column padded to vec4)
    // Column 0
    this.data[0] = transform[0];
    this.data[1] = transform[1];
    this.data[2] = transform[2];
    this.data[3] = 0; // padding

    // Column 1
    this.data[4] = transform[3];
    this.data[5] = transform[4];
    this.data[6] = transform[5];
    this.data[7] = 0; // padding

    // Column 2
    this.data[8] = transform[6];
    this.data[9] = transform[7];
    this.data[10] = transform[8];
    this.data[11] = 0; // padding

    // screen_size
    this.data[12] = viewport.width;
    this.data[13] = viewport.height;

    // scale and inv_scale
    this.data[14] = viewport.scale;
    this.data[15] = 1.0 / viewport.scale;

    // padding (already zero)

    // Write to GPU immediately
    this._device.queue.writeBuffer(this._buffer, 0, this.data);
    this.dirty = false;
  }

  /**
   * Upload to GPU if data has changed.
   */
  upload(): void {
    if (this.dirty) {
      this._device.queue.writeBuffer(this._buffer, 0, this.data);
      this.dirty = false;
    }
  }

  /**
   * Force upload to GPU.
   */
  forceUpload(): void {
    this._device.queue.writeBuffer(this._buffer, 0, this.data);
    this.dirty = false;
  }

  /**
   * Get the GPU buffer (method form).
   */
  getBuffer(): GPUBuffer {
    return this._buffer;
  }

  /**
   * Create a bind group entry.
   */
  createBindGroupEntry(binding: number): GPUBindGroupEntry {
    return {
      binding,
      resource: { buffer: this._buffer },
    };
  }

  /**
   * Destroy the buffer.
   */
  destroy(): void {
    this._buffer.destroy();
  }
}

/**
 * Create a viewport uniform buffer.
 */
export function createViewportUniformBuffer(device: GPUDevice): ViewportUniformBuffer {
  return new ViewportUniformBuffer(device);
}

/**
 * WGSL struct definition for viewport uniforms.
 *
 * Include this in shaders that need viewport data.
 */
export const VIEWPORT_UNIFORM_WGSL = `
struct ViewportUniforms {
    // Graph-to-clip transformation matrix
    // Stored as 3 x vec4 due to WGSL alignment requirements
    transform_col0: vec4<f32>,
    transform_col1: vec4<f32>,
    transform_col2: vec4<f32>,

    // Screen dimensions in pixels
    screen_size: vec2<f32>,

    // Zoom scale factor
    scale: f32,

    // Inverse scale (1.0 / scale)
    inv_scale: f32,

    // Padding for alignment
    _padding: vec2<f32>,
}

// Helper function to apply the transformation
fn transform_point(viewport: ViewportUniforms, pos: vec2<f32>) -> vec2<f32> {
    let col0 = viewport.transform_col0.xyz;
    let col1 = viewport.transform_col1.xyz;
    let col2 = viewport.transform_col2.xyz;

    let x = col0.x * pos.x + col1.x * pos.y + col2.x;
    let y = col0.y * pos.x + col1.y * pos.y + col2.y;

    return vec2<f32>(x, y);
}

// Transform a graph position to clip space
fn graph_to_clip(viewport: ViewportUniforms, graph_pos: vec2<f32>) -> vec4<f32> {
    let clip_xy = transform_point(viewport, graph_pos);
    return vec4<f32>(clip_xy, 0.0, 1.0);
}

// Convert screen pixels to graph units (for size calculations)
fn screen_to_graph_size(viewport: ViewportUniforms, screen_size: f32) -> f32 {
    return screen_size * viewport.inv_scale;
}

// Convert graph units to screen pixels
fn graph_to_screen_size(viewport: ViewportUniforms, graph_size: f32) -> f32 {
    return graph_size * viewport.scale;
}
`;

/**
 * Bind group layout entry for viewport uniforms.
 */
export const VIEWPORT_BIND_GROUP_LAYOUT_ENTRY: GPUBindGroupLayoutEntry = {
  binding: 0,
  visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
  buffer: { type: "uniform" },
};
