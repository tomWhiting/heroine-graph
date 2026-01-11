/**
 * Stream Manager
 *
 * Manages multiple value streams and handles blending their colors.
 *
 * @module
 */

import type {
  BlendMode,
  StreamBulkData,
  StreamDataPoint,
  StreamInfo,
  ValueStreamConfig,
} from "./types.ts";
import { ValueStream } from "./value_stream.ts";

/**
 * Configuration for the stream manager
 */
export interface StreamManagerConfig {
  /** Maximum number of streams allowed */
  maxStreams?: number;
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: Required<StreamManagerConfig> = {
  maxStreams: 16,
};

/**
 * StreamManager class
 *
 * Manages multiple value streams, handles blending, and provides
 * the final computed colors for nodes.
 */
export class StreamManager {
  private streams: Map<string, ValueStream> = new Map();
  private config: Required<StreamManagerConfig>;

  /** Cached blended colors (invalidated on any stream change) */
  private blendedColorCache: Float32Array | null = null;
  private lastNodeCount = 0;

  constructor(config: StreamManagerConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Define a new value stream
   */
  defineStream(config: ValueStreamConfig): ValueStream {
    if (this.streams.size >= this.config.maxStreams) {
      throw new Error(`Maximum stream count (${this.config.maxStreams}) exceeded`);
    }

    if (this.streams.has(config.id)) {
      throw new Error(`Stream with ID "${config.id}" already exists`);
    }

    const stream = new ValueStream(config);
    this.streams.set(config.id, stream);
    this.invalidateCache();
    return stream;
  }

  /**
   * Get a stream by ID
   */
  getStream(id: string): ValueStream | undefined {
    return this.streams.get(id);
  }

  /**
   * Check if a stream exists
   */
  hasStream(id: string): boolean {
    return this.streams.has(id);
  }

  /**
   * Remove a stream
   */
  removeStream(id: string): boolean {
    const removed = this.streams.delete(id);
    if (removed) {
      this.invalidateCache();
    }
    return removed;
  }

  /**
   * Set data for a stream
   */
  setStreamData(streamId: string, data: StreamDataPoint[]): void {
    const stream = this.streams.get(streamId);
    if (!stream) {
      throw new Error(`Stream "${streamId}" not found`);
    }
    stream.setData(data);
    this.invalidateCache();
  }

  /**
   * Set bulk data for a stream (more efficient for large updates)
   */
  setStreamBulkData(streamId: string, data: StreamBulkData): void {
    const stream = this.streams.get(streamId);
    if (!stream) {
      throw new Error(`Stream "${streamId}" not found`);
    }
    stream.setBulkData(data);
    this.invalidateCache();
  }

  /**
   * Clear all data from a stream
   */
  clearStreamData(streamId: string): void {
    const stream = this.streams.get(streamId);
    if (stream) {
      stream.clear();
      this.invalidateCache();
    }
  }

  /**
   * Enable a stream
   */
  enableStream(id: string): boolean {
    const stream = this.streams.get(id);
    if (!stream) return false;
    stream.enable();
    this.invalidateCache();
    return true;
  }

  /**
   * Disable a stream
   */
  disableStream(id: string): boolean {
    const stream = this.streams.get(id);
    if (!stream) return false;
    stream.disable();
    this.invalidateCache();
    return true;
  }

  /**
   * Toggle a stream
   */
  toggleStream(id: string): boolean {
    const stream = this.streams.get(id);
    if (!stream) return false;
    const result = stream.toggle();
    this.invalidateCache();
    return result;
  }

  /**
   * Get info for all streams
   */
  getStreamInfo(): StreamInfo[] {
    return Array.from(this.streams.values()).map((s) => s.getInfo());
  }

  /**
   * Get all stream IDs
   */
  getStreamIds(): string[] {
    return Array.from(this.streams.keys());
  }

  /**
   * Get number of streams
   */
  get count(): number {
    return this.streams.size;
  }

  /**
   * Get number of enabled streams
   */
  get enabledCount(): number {
    let count = 0;
    for (const stream of this.streams.values()) {
      if (stream.isEnabled()) count++;
    }
    return count;
  }

  /**
   * Compute blended colors for all nodes
   *
   * Returns a Float32Array of RGBA values (4 floats per node)
   * that can be passed directly to setNodeColors().
   *
   * Nodes without any stream values will have [0, 0, 0, 0] (transparent).
   *
   * @param nodeCount Total number of nodes in the graph
   * @param baseColors Optional base colors to blend on top of
   */
  computeBlendedColors(nodeCount: number, baseColors?: Float32Array): Float32Array {
    // Check cache
    if (this.blendedColorCache && this.lastNodeCount === nodeCount) {
      return this.blendedColorCache;
    }

    // Create output array
    const result = new Float32Array(nodeCount * 4);

    // If base colors provided, copy them
    if (baseColors && baseColors.length === nodeCount * 4) {
      result.set(baseColors);
    }

    // Collect enabled streams
    const enabledStreams: ValueStream[] = [];
    for (const stream of this.streams.values()) {
      if (stream.isEnabled() && stream.size > 0) {
        enabledStreams.push(stream);
      }
    }

    if (enabledStreams.length === 0) {
      this.blendedColorCache = result;
      this.lastNodeCount = nodeCount;
      return result;
    }

    // Process each enabled stream
    for (const stream of enabledStreams) {
      const colors = stream.getAllColors();
      const blendMode = stream.getBlendMode();

      for (const [nodeIndex, color] of colors) {
        if (nodeIndex >= nodeCount) continue;

        const offset = nodeIndex * 4;
        this.blendColor(result, offset, color, blendMode);
      }
    }

    // Cache result
    this.blendedColorCache = result;
    this.lastNodeCount = nodeCount;

    return result;
  }

  /**
   * Get all node indices that have values in any enabled stream
   */
  getAffectedNodeIndices(): Set<number> {
    const indices = new Set<number>();
    for (const stream of this.streams.values()) {
      if (stream.isEnabled()) {
        for (const idx of stream.getNodeIndices()) {
          indices.add(idx);
        }
      }
    }
    return indices;
  }

  /**
   * Blend a color into the result array at the given offset
   */
  private blendColor(
    result: Float32Array,
    offset: number,
    color: [number, number, number, number],
    blendMode: BlendMode,
  ): void {
    const [sr, sg, sb, sa] = color;
    const dr = result[offset];
    const dg = result[offset + 1];
    const db = result[offset + 2];
    const da = result[offset + 3];

    switch (blendMode) {
      case "additive":
        // Add colors (clamped to 1)
        result[offset] = Math.min(1, dr + sr * sa);
        result[offset + 1] = Math.min(1, dg + sg * sa);
        result[offset + 2] = Math.min(1, db + sb * sa);
        result[offset + 3] = Math.min(1, da + sa);
        break;

      case "multiply":
        // Multiply colors
        if (da > 0) {
          result[offset] = dr * (1 - sa) + dr * sr * sa;
          result[offset + 1] = dg * (1 - sa) + dg * sg * sa;
          result[offset + 2] = db * (1 - sa) + db * sb * sa;
          result[offset + 3] = Math.min(1, da + sa * (1 - da));
        } else {
          result[offset] = sr;
          result[offset + 1] = sg;
          result[offset + 2] = sb;
          result[offset + 3] = sa;
        }
        break;

      case "max":
        // Take maximum of each channel
        result[offset] = Math.max(dr, sr);
        result[offset + 1] = Math.max(dg, sg);
        result[offset + 2] = Math.max(db, sb);
        result[offset + 3] = Math.max(da, sa);
        break;

      case "replace":
        // Simply replace
        result[offset] = sr;
        result[offset + 1] = sg;
        result[offset + 2] = sb;
        result[offset + 3] = sa;
        break;
    }
  }

  /**
   * Invalidate the blended color cache
   */
  invalidateCache(): void {
    this.blendedColorCache = null;
  }

  /**
   * Clear all streams and their data
   */
  clear(): void {
    this.streams.clear();
    this.invalidateCache();
  }

  /**
   * Destroy the manager and release resources
   */
  destroy(): void {
    this.clear();
  }
}

/**
 * Create a stream manager
 */
export function createStreamManager(config?: StreamManagerConfig): StreamManager {
  return new StreamManager(config);
}
