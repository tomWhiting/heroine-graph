/**
 * WebGPU Buffer Utilities
 *
 * Helper functions for working with GPU buffers and typed arrays.
 */

/**
 * Convert a typed array to an ArrayBuffer suitable for GPU buffer writes.
 *
 * WebGPU's writeBuffer expects BufferSource (ArrayBuffer | ArrayBufferView).
 * TypeScript's type definitions are stricter than what the API accepts,
 * so we use this helper to safely convert typed arrays.
 *
 * @param typedArray - The typed array to convert
 * @returns ArrayBuffer that can be passed to writeBuffer
 */
export function toArrayBuffer(
  typedArray: Float32Array | Uint32Array | Uint8Array | Int32Array | Uint16Array | Int16Array,
): ArrayBuffer {
  // If the typed array's buffer is already the exact size we need, use it directly
  if (
    typedArray.byteOffset === 0 &&
    typedArray.byteLength === typedArray.buffer.byteLength
  ) {
    return typedArray.buffer as ArrayBuffer;
  }
  // Otherwise, slice to get a new ArrayBuffer of the exact size
  return typedArray.buffer.slice(
    typedArray.byteOffset,
    typedArray.byteOffset + typedArray.byteLength,
  ) as ArrayBuffer;
}

/**
 * Write a typed array to a GPU buffer.
 *
 * This is a type-safe wrapper around device.queue.writeBuffer that handles
 * the BufferSource type conversion.
 *
 * @param queue - The GPU queue
 * @param buffer - The GPU buffer to write to
 * @param bufferOffset - Offset in bytes within the buffer
 * @param data - The typed array data to write
 * @param dataOffset - Optional offset in elements within the data array
 * @param size - Optional number of elements to write
 */
export function writeGPUBuffer(
  queue: GPUQueue,
  buffer: GPUBuffer,
  bufferOffset: number,
  data: Float32Array | Uint32Array | Uint8Array | Int32Array,
  dataOffset?: number,
  size?: number,
): void {
  const arrayBuffer = toArrayBuffer(data);
  const bytesPerElement = data.BYTES_PER_ELEMENT;
  const dataByteOffset = (dataOffset ?? 0) * bytesPerElement;
  const dataByteSize = size !== undefined
    ? size * bytesPerElement
    : data.byteLength - dataByteOffset;

  queue.writeBuffer(buffer, bufferOffset, arrayBuffer, dataByteOffset, dataByteSize);
}
