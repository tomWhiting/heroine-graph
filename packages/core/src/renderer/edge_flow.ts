/**
 * Edge Flow Configuration
 *
 * PWM-based flow animation for edges with dual-layer support.
 * Provides presets for common flow effects like particles, waves, sparks, etc.
 *
 * @module
 */

import type { EdgeFlowConfig, EdgeFlowLayerConfig, EdgeFlowWaveShape } from "../types.ts";

/**
 * Default disabled layer
 */
export const DISABLED_FLOW_LAYER: EdgeFlowLayerConfig = {
  enabled: false,
  pulseWidth: 0.1,
  pulseCount: 1,
  speed: 0.5,
  waveShape: "sine",
  brightness: 1.0,
  fade: 0.0,
  color: null,
};

/**
 * Default edge flow configuration (disabled)
 */
export const DEFAULT_EDGE_FLOW_CONFIG: EdgeFlowConfig = {
  layer1: { ...DISABLED_FLOW_LAYER },
  layer2: { ...DISABLED_FLOW_LAYER },
};

/**
 * Flow presets for common effects
 */
export const EDGE_FLOW_PRESETS = {
  /** No flow animation */
  none: DEFAULT_EDGE_FLOW_CONFIG,

  /** Flowing particles along edges */
  particles: {
    layer1: {
      enabled: true,
      pulseWidth: 0.05,
      pulseCount: 3,
      speed: 0.3,
      waveShape: "sine" as EdgeFlowWaveShape,
      brightness: 2.0,
      fade: 0.3,
      color: null,
    },
    layer2: { ...DISABLED_FLOW_LAYER },
  },

  /** Smooth sine wave pulses */
  waves: {
    layer1: {
      enabled: true,
      pulseWidth: 0.3,
      pulseCount: 2,
      speed: 0.2,
      waveShape: "sine" as EdgeFlowWaveShape,
      brightness: 1.5,
      fade: 0.5,
      color: null,
    },
    layer2: { ...DISABLED_FLOW_LAYER },
  },

  /** Digital data stream effect */
  dataStream: {
    layer1: {
      enabled: true,
      pulseWidth: 0.02,
      pulseCount: 6,
      speed: 0.8,
      waveShape: "square" as EdgeFlowWaveShape,
      brightness: 3.0,
      fade: 0.1,
      color: null,
    },
    layer2: { ...DISABLED_FLOW_LAYER },
  },

  /** Fast bright sparks */
  sparks: {
    layer1: {
      enabled: true,
      pulseWidth: 0.01,
      pulseCount: 4,
      speed: 1.5,
      waveShape: "triangle" as EdgeFlowWaveShape,
      brightness: 4.0,
      fade: 0.0,
      color: null,
    },
    layer2: { ...DISABLED_FLOW_LAYER },
  },

  /** Warning/alert pulsing effect */
  warning: {
    layer1: {
      enabled: true,
      pulseWidth: 0.5,
      pulseCount: 1,
      speed: 0.5,
      waveShape: "sine" as EdgeFlowWaveShape,
      brightness: 2.5,
      fade: 0.7,
      color: [1.0, 0.3, 0.1, 0.8] as const,
    },
    layer2: { ...DISABLED_FLOW_LAYER },
  },

  /** Dual layer: smooth base + fast highlights */
  dualLayer: {
    layer1: {
      enabled: true,
      pulseWidth: 0.25,
      pulseCount: 2,
      speed: 0.15,
      waveShape: "sine" as EdgeFlowWaveShape,
      brightness: 1.3,
      fade: 0.6,
      color: null,
    },
    layer2: {
      enabled: true,
      pulseWidth: 0.02,
      pulseCount: 5,
      speed: 1.2,
      waveShape: "triangle" as EdgeFlowWaveShape,
      brightness: 3.5,
      fade: 0.0,
      color: [1.0, 1.0, 1.0, 0.5] as const,
    },
  },

  /** Energy flow effect */
  energy: {
    layer1: {
      enabled: true,
      pulseWidth: 0.15,
      pulseCount: 3,
      speed: 0.4,
      waveShape: "sine" as EdgeFlowWaveShape,
      brightness: 2.0,
      fade: 0.4,
      color: [0.3, 0.8, 1.0, 0.9] as const,
    },
    layer2: {
      enabled: true,
      pulseWidth: 0.03,
      pulseCount: 6,
      speed: 0.8,
      waveShape: "sine" as EdgeFlowWaveShape,
      brightness: 3.0,
      fade: 0.0,
      color: [1.0, 1.0, 1.0, 0.3] as const,
    },
  },
} as const satisfies Record<string, EdgeFlowConfig>;

/**
 * Flow preset names
 */
export type EdgeFlowPreset = keyof typeof EDGE_FLOW_PRESETS;

/**
 * Get a flow preset by name
 */
export function getFlowPreset(name: EdgeFlowPreset): EdgeFlowConfig {
  return EDGE_FLOW_PRESETS[name];
}

/**
 * Create a custom flow configuration
 */
export function createEdgeFlowConfig(
  layer1: Partial<EdgeFlowLayerConfig>,
  layer2?: Partial<EdgeFlowLayerConfig>,
): EdgeFlowConfig {
  return {
    layer1: { ...DISABLED_FLOW_LAYER, ...layer1, enabled: true },
    layer2: layer2
      ? { ...DISABLED_FLOW_LAYER, ...layer2, enabled: true }
      : { ...DISABLED_FLOW_LAYER },
  };
}

/**
 * Convert wave shape to numeric value for shader
 * 0 = square, 0.5 = triangle, 1 = sine
 */
export function waveShapeToFloat(shape: EdgeFlowWaveShape): number {
  switch (shape) {
    case "square":
      return 0.0;
    case "triangle":
      return 0.5;
    case "sine":
      return 1.0;
  }
}

/**
 * Edge flow uniform buffer size (aligned to 16 bytes)
 * Layer 1: 12 floats (enabled, pulseWidth, pulseCount, speed, waveShape, brightness, fade, colorR, colorG, colorB, colorA, hasColor)
 * Layer 2: 12 floats
 * Time: 4 floats (time, _pad, _pad, _pad)
 * Total: 28 floats = 112 bytes, rounded to 128 for alignment
 */
export const EDGE_FLOW_UNIFORM_SIZE = 128;

/**
 * Write edge flow config to a uniform buffer
 */
export function writeEdgeFlowUniforms(
  data: ArrayBuffer,
  config: EdgeFlowConfig,
  time: number,
): void {
  const view = new DataView(data);
  let offset = 0;

  // Layer 1 (48 bytes = 12 floats)
  view.setFloat32(offset, config.layer1.enabled ? 1.0 : 0.0, true);
  offset += 4;
  view.setFloat32(offset, config.layer1.pulseWidth, true);
  offset += 4;
  view.setFloat32(offset, config.layer1.pulseCount, true);
  offset += 4;
  view.setFloat32(offset, config.layer1.speed, true);
  offset += 4;
  view.setFloat32(offset, waveShapeToFloat(config.layer1.waveShape), true);
  offset += 4;
  view.setFloat32(offset, config.layer1.brightness, true);
  offset += 4;
  view.setFloat32(offset, config.layer1.fade, true);
  offset += 4;
  // Color (RGBA) + hasColor flag
  if (config.layer1.color) {
    view.setFloat32(offset, config.layer1.color[0], true);
    offset += 4;
    view.setFloat32(offset, config.layer1.color[1], true);
    offset += 4;
    view.setFloat32(offset, config.layer1.color[2], true);
    offset += 4;
    view.setFloat32(offset, config.layer1.color[3], true);
    offset += 4;
    view.setFloat32(offset, 1.0, true); // hasColor = true
    offset += 4;
  } else {
    view.setFloat32(offset, 0.0, true);
    offset += 4;
    view.setFloat32(offset, 0.0, true);
    offset += 4;
    view.setFloat32(offset, 0.0, true);
    offset += 4;
    view.setFloat32(offset, 0.0, true);
    offset += 4;
    view.setFloat32(offset, 0.0, true); // hasColor = false
    offset += 4;
  }

  // Layer 2 (48 bytes = 12 floats)
  view.setFloat32(offset, config.layer2.enabled ? 1.0 : 0.0, true);
  offset += 4;
  view.setFloat32(offset, config.layer2.pulseWidth, true);
  offset += 4;
  view.setFloat32(offset, config.layer2.pulseCount, true);
  offset += 4;
  view.setFloat32(offset, config.layer2.speed, true);
  offset += 4;
  view.setFloat32(offset, waveShapeToFloat(config.layer2.waveShape), true);
  offset += 4;
  view.setFloat32(offset, config.layer2.brightness, true);
  offset += 4;
  view.setFloat32(offset, config.layer2.fade, true);
  offset += 4;
  // Color (RGBA) + hasColor flag
  if (config.layer2.color) {
    view.setFloat32(offset, config.layer2.color[0], true);
    offset += 4;
    view.setFloat32(offset, config.layer2.color[1], true);
    offset += 4;
    view.setFloat32(offset, config.layer2.color[2], true);
    offset += 4;
    view.setFloat32(offset, config.layer2.color[3], true);
    offset += 4;
    view.setFloat32(offset, 1.0, true); // hasColor = true
    offset += 4;
  } else {
    view.setFloat32(offset, 0.0, true);
    offset += 4;
    view.setFloat32(offset, 0.0, true);
    offset += 4;
    view.setFloat32(offset, 0.0, true);
    offset += 4;
    view.setFloat32(offset, 0.0, true);
    offset += 4;
    view.setFloat32(offset, 0.0, true); // hasColor = false
    offset += 4;
  }

  // Time (16 bytes = 4 floats)
  view.setFloat32(offset, time, true);
  offset += 4;
  view.setFloat32(offset, 0.0, true); // _pad1
  offset += 4;
  view.setFloat32(offset, 0.0, true); // _pad2
  offset += 4;
  view.setFloat32(offset, 0.0, true); // _pad3
}
