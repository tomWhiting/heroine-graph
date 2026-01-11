/**
 * Streams Module
 *
 * Provides a value stream system for mapping numeric data to visual heat colors.
 * Multiple streams can be defined and blended together for rich data visualization.
 *
 * @module
 */

// Types
export type {
  BlendMode,
  ColorStop,
  StreamBulkData,
  StreamDataPoint,
  StreamInfo,
  ValueColorScale,
  ValueStreamConfig,
} from "./types.ts";

// Type utilities
export { createColorScaleFromPreset, createGradientScale, VALUE_COLOR_PRESETS } from "./types.ts";

// ValueStream class
export { ValueStream } from "./value_stream.ts";

// StreamManager
export type { StreamManagerConfig } from "./stream_manager.ts";
export { createStreamManager, StreamManager } from "./stream_manager.ts";
