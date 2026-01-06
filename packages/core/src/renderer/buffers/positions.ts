/**
 * Position Buffer Manager
 *
 * Manages GPU buffers for node positions in Structure of Arrays (SoA) format.
 * Uses ping-pong double buffering for compute shader updates.
 */

import { HeroineGraphError, ErrorCode } from "../../errors.ts";

/**
 * Configuration for position buffers.
 */
export interface PositionBufferConfig {
  /** Initial capacity in number of nodes */
  readonly initialCapacity: number;
  /** Growth factor when resizing (e.g., 1.5 = 50% growth) */
  readonly growthFactor: number;
  /** Label for debugging */
  readonly label?: string;
}

/**
 * Default configuration for position buffers.
 */
export const DEFAULT_POSITION_BUFFER_CONFIG: PositionBufferConfig = {
  initialCapacity: 1024,
  growthFactor: 1.5,
  label: "PositionBuffer",
};

/**
 * Manages position buffers in SoA format for GPU operations.
 *
 * Provides separate X and Y buffers for cache-friendly access patterns
 * and SIMD operations in compute shaders.
 */
export class PositionBufferManager {
  private readonly device: GPUDevice;
  private readonly config: PositionBufferConfig;

  /** X positions buffer (read) */
  private posXRead: GPUBuffer;
  /** Y positions buffer (read) */
  private posYRead: GPUBuffer;
  /** X positions buffer (write) */
  private posXWrite: GPUBuffer;
  /** Y positions buffer (write) */
  private posYWrite: GPUBuffer;

  /** Current capacity in number of elements */
  private capacity: number;
  /** Current number of active elements */
  private count: number;

  constructor(device: GPUDevice, config: Partial<PositionBufferConfig> = {}) {
    this.device = device;
    this.config = { ...DEFAULT_POSITION_BUFFER_CONFIG, ...config };
    this.capacity = this.config.initialCapacity;
    this.count = 0;

    // Create initial buffers
    const bufferSize = this.capacity * Float32Array.BYTES_PER_ELEMENT;

    this.posXRead = this.createBuffer(bufferSize, "posX_read");
    this.posYRead = this.createBuffer(bufferSize, "posY_read");
    this.posXWrite = this.createBuffer(bufferSize, "posX_write");
    this.posYWrite = this.createBuffer(bufferSize, "posY_write");
  }

  /**
   * Create a GPU storage buffer.
   */
  private createBuffer(size: number, label: string): GPUBuffer {
    return this.device.createBuffer({
      size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.VERTEX,
      label: `${this.config.label}_${label}`,
    });
  }

  /**
   * Ensure buffers have capacity for the given count.
   */
  private ensureCapacity(required: number): void {
    if (required <= this.capacity) {
      return;
    }

    // Calculate new capacity
    let newCapacity = this.capacity;
    while (newCapacity < required) {
      newCapacity = Math.ceil(newCapacity * this.config.growthFactor);
    }

    // Create new buffers
    const newSize = newCapacity * Float32Array.BYTES_PER_ELEMENT;

    const newPosXRead = this.createBuffer(newSize, "posX_read");
    const newPosYRead = this.createBuffer(newSize, "posY_read");
    const newPosXWrite = this.createBuffer(newSize, "posX_write");
    const newPosYWrite = this.createBuffer(newSize, "posY_write");

    // Copy existing data if any
    if (this.count > 0) {
      const copySize = this.count * Float32Array.BYTES_PER_ELEMENT;
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.posXRead, 0, newPosXRead, 0, copySize);
      encoder.copyBufferToBuffer(this.posYRead, 0, newPosYRead, 0, copySize);
      this.device.queue.submit([encoder.finish()]);
    }

    // Destroy old buffers
    this.posXRead.destroy();
    this.posYRead.destroy();
    this.posXWrite.destroy();
    this.posYWrite.destroy();

    // Assign new buffers
    this.posXRead = newPosXRead;
    this.posYRead = newPosYRead;
    this.posXWrite = newPosXWrite;
    this.posYWrite = newPosYWrite;
    this.capacity = newCapacity;
  }

  /**
   * Upload positions from Float32Arrays.
   *
   * @param posX X positions
   * @param posY Y positions
   */
  upload(posX: Float32Array, posY: Float32Array): void {
    if (posX.length !== posY.length) {
      throw new HeroineGraphError(
        ErrorCode.INVALID_POSITIONS,
        `Position arrays must have same length: X=${posX.length}, Y=${posY.length}`
      );
    }

    this.ensureCapacity(posX.length);
    this.count = posX.length;

    this.device.queue.writeBuffer(this.posXRead, 0, posX);
    this.device.queue.writeBuffer(this.posYRead, 0, posY);
  }

  /**
   * Upload positions from a WASM engine.
   *
   * Uses zero-copy views for efficient transfer.
   *
   * @param posXView Float32Array view from WASM
   * @param posYView Float32Array view from WASM
   */
  uploadFromWasm(posXView: Float32Array, posYView: Float32Array): void {
    this.upload(posXView, posYView);
  }

  /**
   * Swap read and write buffers.
   *
   * Call this after a compute pass that writes to the write buffers.
   */
  swap(): void {
    [this.posXRead, this.posXWrite] = [this.posXWrite, this.posXRead];
    [this.posYRead, this.posYWrite] = [this.posYWrite, this.posYRead];
  }

  /**
   * Get the current read buffers.
   */
  getReadBuffers(): { x: GPUBuffer; y: GPUBuffer } {
    return { x: this.posXRead, y: this.posYRead };
  }

  /**
   * Get the current write buffers.
   */
  getWriteBuffers(): { x: GPUBuffer; y: GPUBuffer } {
    return { x: this.posXWrite, y: this.posYWrite };
  }

  /**
   * Get the current node count.
   */
  getCount(): number {
    return this.count;
  }

  /**
   * Get the current capacity.
   */
  getCapacity(): number {
    return this.capacity;
  }

  /**
   * Get buffer size in bytes.
   */
  getBufferSize(): number {
    return this.capacity * Float32Array.BYTES_PER_ELEMENT;
  }

  /**
   * Create bind group entries for reading positions.
   */
  createReadBindGroupEntries(startBinding: number): GPUBindGroupEntry[] {
    return [
      { binding: startBinding, resource: { buffer: this.posXRead } },
      { binding: startBinding + 1, resource: { buffer: this.posYRead } },
    ];
  }

  /**
   * Create bind group entries for writing positions.
   */
  createWriteBindGroupEntries(startBinding: number): GPUBindGroupEntry[] {
    return [
      { binding: startBinding, resource: { buffer: this.posXWrite } },
      { binding: startBinding + 1, resource: { buffer: this.posYWrite } },
    ];
  }

  /**
   * Create bind group layout entries for positions.
   */
  static createLayoutEntries(
    startBinding: number,
    access: "read-only" | "read-write"
  ): GPUBindGroupLayoutEntry[] {
    const storageAccess = access === "read-only" ? "read-only-storage" : "storage";
    return [
      {
        binding: startBinding,
        visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX,
        buffer: { type: storageAccess as GPUBufferBindingType },
      },
      {
        binding: startBinding + 1,
        visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX,
        buffer: { type: storageAccess as GPUBufferBindingType },
      },
    ];
  }

  /**
   * Destroy all buffers.
   */
  destroy(): void {
    this.posXRead.destroy();
    this.posYRead.destroy();
    this.posXWrite.destroy();
    this.posYWrite.destroy();
  }
}
