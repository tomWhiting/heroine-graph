/**
 * Ping-Pong Buffer Manager
 *
 * Generic double-buffering implementation for compute shader operations.
 * Allows read from one buffer while writing to another, then swap.
 */

import { toArrayBuffer } from "../../webgpu/buffer_utils.ts";

/**
 * Configuration for ping-pong buffers.
 */
export interface PingPongBufferConfig {
  /** Buffer size in bytes */
  readonly size: number;
  /** Buffer usage flags */
  readonly usage: GPUBufferUsageFlags;
  /** Label for debugging */
  readonly label: string;
}

/**
 * A pair of buffers for ping-pong double buffering.
 */
export interface BufferPair {
  /** The read buffer (input to compute) */
  readonly read: GPUBuffer;
  /** The write buffer (output from compute) */
  readonly write: GPUBuffer;
}

/**
 * Manages a pair of GPU buffers for ping-pong operations.
 *
 * This is the fundamental pattern for compute shaders that update data:
 * 1. Bind buffer A as read, buffer B as write
 * 2. Run compute shader
 * 3. Swap: now buffer B is read, buffer A is write
 * 4. Repeat
 */
export class PingPongBuffer {
  private readonly device: GPUDevice;
  private readonly config: PingPongBufferConfig;
  private bufferA: GPUBuffer;
  private bufferB: GPUBuffer;
  private readIsA: boolean;

  constructor(device: GPUDevice, config: PingPongBufferConfig) {
    this.device = device;
    this.config = config;
    this.readIsA = true;

    this.bufferA = device.createBuffer({
      size: config.size,
      usage: config.usage,
      label: `${config.label}_A`,
    });

    this.bufferB = device.createBuffer({
      size: config.size,
      usage: config.usage,
      label: `${config.label}_B`,
    });
  }

  /**
   * Get the current read buffer.
   */
  getReadBuffer(): GPUBuffer {
    return this.readIsA ? this.bufferA : this.bufferB;
  }

  /**
   * Get the current write buffer.
   */
  getWriteBuffer(): GPUBuffer {
    return this.readIsA ? this.bufferB : this.bufferA;
  }

  /**
   * Get both buffers as a pair.
   */
  getBufferPair(): BufferPair {
    return {
      read: this.getReadBuffer(),
      write: this.getWriteBuffer(),
    };
  }

  /**
   * Swap read and write buffers.
   */
  swap(): void {
    this.readIsA = !this.readIsA;
  }

  /**
   * Write data to the read buffer.
   *
   * Use this for initial data upload.
   */
  writeToRead(data: ArrayBuffer | Float32Array | Uint32Array, offset: number = 0): void {
    const bufferData = data instanceof ArrayBuffer ? data : toArrayBuffer(data);
    this.device.queue.writeBuffer(this.getReadBuffer(), offset, bufferData);
  }

  /**
   * Copy data between the buffers.
   */
  copyReadToWrite(encoder: GPUCommandEncoder, size?: number): void {
    const copySize = size ?? this.config.size;
    encoder.copyBufferToBuffer(
      this.getReadBuffer(),
      0,
      this.getWriteBuffer(),
      0,
      copySize
    );
  }

  /**
   * Get the buffer size.
   */
  getSize(): number {
    return this.config.size;
  }

  /**
   * Resize the buffers.
   *
   * Creates new buffers and optionally copies existing data.
   */
  resize(newSize: number, copyData: boolean = true): void {
    const newBufferA = this.device.createBuffer({
      size: newSize,
      usage: this.config.usage,
      label: `${this.config.label}_A`,
    });

    const newBufferB = this.device.createBuffer({
      size: newSize,
      usage: this.config.usage,
      label: `${this.config.label}_B`,
    });

    if (copyData) {
      const copySize = Math.min(this.config.size, newSize);
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.bufferA, 0, newBufferA, 0, copySize);
      encoder.copyBufferToBuffer(this.bufferB, 0, newBufferB, 0, copySize);
      this.device.queue.submit([encoder.finish()]);
    }

    this.bufferA.destroy();
    this.bufferB.destroy();

    this.bufferA = newBufferA;
    this.bufferB = newBufferB;
    (this.config as { size: number }).size = newSize;
  }

  /**
   * Create bind group entries for the current state.
   */
  createBindGroupEntries(readBinding: number, writeBinding: number): GPUBindGroupEntry[] {
    return [
      { binding: readBinding, resource: { buffer: this.getReadBuffer() } },
      { binding: writeBinding, resource: { buffer: this.getWriteBuffer() } },
    ];
  }

  /**
   * Destroy the buffers.
   */
  destroy(): void {
    this.bufferA.destroy();
    this.bufferB.destroy();
  }
}

/**
 * Create a ping-pong buffer for f32 arrays.
 */
export function createFloat32PingPong(
  device: GPUDevice,
  elementCount: number,
  label: string
): PingPongBuffer {
  return new PingPongBuffer(device, {
    size: elementCount * Float32Array.BYTES_PER_ELEMENT,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
    label,
  });
}

/**
 * Create a ping-pong buffer for i32 arrays (used for atomic operations).
 */
export function createInt32PingPong(
  device: GPUDevice,
  elementCount: number,
  label: string
): PingPongBuffer {
  return new PingPongBuffer(device, {
    size: elementCount * Int32Array.BYTES_PER_ELEMENT,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
    label,
  });
}

/**
 * Create a ping-pong buffer for u32 arrays.
 */
export function createUint32PingPong(
  device: GPUDevice,
  elementCount: number,
  label: string
): PingPongBuffer {
  return new PingPongBuffer(device, {
    size: elementCount * Uint32Array.BYTES_PER_ELEMENT,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
    label,
  });
}
