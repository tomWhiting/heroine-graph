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

