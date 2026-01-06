/**
 * Heroine Graph - Core Library
 *
 * High-performance graph visualization using WebGPU.
 *
 * @module
 */

// =============================================================================
// Types
// =============================================================================

export type {
  // Identifiers
  NodeId,
  EdgeId,

  // Primitives
  Vec2,
  BoundingBox,
  Color,
  ColorScale,

  // Node types
  NodeMetadata,
  NodeState,
  Node,

  // Edge types
  EdgeMetadata,
  EdgeState,
  Edge,

  // Simulation types
  SimulationStatus,
  ForceConfig,
  SimulationState,

  // Layer types
  LayerType,
  HeatmapLayerConfig,
  ContourLayerConfig,
  MetaballLayerConfig,
  LabelLayerConfig,
  LayerConfig,
  Layer,

  // Viewport types
  ViewportState,

  // Configuration
  GraphConfig,

  // Input types
  NodeInput,
  EdgeInput,
  GraphInput,
  GraphTypedInput,

  // Event types
  GraphEvent,
  NodeClickEvent,
  NodeDoubleClickEvent,
  NodeHoverEnterEvent,
  NodeHoverLeaveEvent,
  NodeDragStartEvent,
  NodeDragMoveEvent,
  NodeDragEndEvent,
  EdgeClickEvent,
  EdgeHoverEnterEvent,
  EdgeHoverLeaveEvent,
  ViewportChangeEvent,
  SimulationTickEvent,
  SimulationEndEvent,
  SelectionChangeEvent,
  BackgroundClickEvent,
  HeroineGraphEvent,
  EventHandler,
  EventMap,
} from "./src/types.ts";

// =============================================================================
// Errors
// =============================================================================

export { HeroineGraphError, ErrorCode, Errors, assert, wrapAsync } from "./src/errors.ts";

// =============================================================================
// WebGPU
// =============================================================================

export { checkWebGPU, hasWebGPU, describeWebGPUStatus } from "./src/webgpu/check.ts";
export type { WebGPUStatus } from "./src/webgpu/check.ts";

export {
  createGPUContext,
  destroyGPUContext,
  resizeGPUContext,
  getCurrentTexture,
  createDepthTexture,
  estimateMaxNodes,
} from "./src/webgpu/context.ts";
export type { GPUContext, GPUContextOptions } from "./src/webgpu/context.ts";

// =============================================================================
// WASM
// =============================================================================

export {
  loadWasmModule,
  createWasmEngine,
  createWasmEngineWithCapacity,
  isWasmLoaded,
  getWasmModule,
  WasmMemory,
} from "./src/wasm/loader.ts";

// =============================================================================
// Events
// =============================================================================

export { EventEmitter, createEventEmitter, createTimestamp, Events } from "./src/events/emitter.ts";

// =============================================================================
// Viewport
// =============================================================================

export { Viewport, createViewport, DEFAULT_VIEWPORT_CONFIG } from "./src/viewport/viewport.ts";
export type { ViewportConfig } from "./src/viewport/viewport.ts";

export {
  identity,
  translate,
  scale,
  rotate,
  multiply,
  transformPoint,
  invert,
  graphToScreenMatrix,
  screenToGraphMatrix,
  graphToClipMatrix,
  screenToGraph,
  graphToScreen,
  getVisibleBounds,
  fitBoundsScale,
  boundsCenter,
  pointInBounds,
  expandBounds,
  distanceToBounds,
} from "./src/viewport/transforms.ts";
export type { Matrix3 } from "./src/viewport/transforms.ts";

export {
  ViewportUniformBuffer,
  createViewportUniformBuffer,
  VIEWPORT_UNIFORM_SIZE,
  VIEWPORT_UNIFORM_WGSL,
  VIEWPORT_BIND_GROUP_LAYOUT_ENTRY,
} from "./src/viewport/uniforms.ts";

// =============================================================================
// Buffers
// =============================================================================

export {
  PositionBufferManager,
  DEFAULT_POSITION_BUFFER_CONFIG,
} from "./src/renderer/buffers/positions.ts";
export type { PositionBufferConfig } from "./src/renderer/buffers/positions.ts";

export {
  PingPongBuffer,
  createFloat32PingPong,
  createInt32PingPong,
  createUint32PingPong,
} from "./src/renderer/buffers/pingpong.ts";
export type { PingPongBufferConfig, BufferPair } from "./src/renderer/buffers/pingpong.ts";

export {
  UniformBuffer,
  createSimulationUniformBuffer,
  createViewportUniformBuffer as createViewportUniformBuffer2,
  forceConfigToUniforms,
  viewportStateToUniforms,
  SIMULATION_UNIFORMS_SIZE,
  VIEWPORT_UNIFORMS_SIZE,
  DEFAULT_SIMULATION_UNIFORMS,
} from "./src/renderer/buffers/uniforms.ts";
export type {
  SimulationUniforms,
  ViewportUniforms,
} from "./src/renderer/buffers/uniforms.ts";

export {
  EdgeBufferManager,
  edgePairsToCSR,
  DEFAULT_EDGE_BUFFER_CONFIG,
} from "./src/renderer/buffers/edges.ts";
export type { EdgeBufferConfig, CSREdgeData } from "./src/renderer/buffers/edges.ts";

// =============================================================================
// Render Pipelines
// =============================================================================

export {
  createNodeRenderPipeline,
  createNodeBindGroup,
  createViewportBindGroup as createNodeViewportBindGroup,
  renderNodes,
  DEFAULT_NODE_PIPELINE_CONFIG,
} from "./src/renderer/pipelines/nodes.ts";
export type { NodePipelineConfig, NodeRenderPipeline } from "./src/renderer/pipelines/nodes.ts";

export {
  createEdgeRenderPipeline,
  createEdgeBindGroup,
  createEdgeViewportBindGroup,
  renderEdges,
  DEFAULT_EDGE_PIPELINE_CONFIG,
} from "./src/renderer/pipelines/edges.ts";
export type { EdgePipelineConfig, EdgeRenderPipeline } from "./src/renderer/pipelines/edges.ts";

// =============================================================================
// Render Loop
// =============================================================================

export {
  createRenderLoop,
  createGPUTimer,
  createFramePacer,
  DEFAULT_RENDER_LOOP_CONFIG,
} from "./src/renderer/render_loop.ts";
export type {
  RenderLoop,
  RenderCallback,
  RenderLoopConfig,
  FrameStats,
  GPUTimer,
  FramePacer,
} from "./src/renderer/render_loop.ts";

// =============================================================================
// GPU Commands
// =============================================================================

export {
  createCommandOrchestrator,
  createRenderPassDescriptor,
  createComputePassDescriptor,
  dispatchCompute,
  calculateWorkgroups,
  createBufferUpdater,
  DEFAULT_CLEAR_COLOR,
} from "./src/renderer/commands.ts";
export type {
  CommandOrchestrator,
  CommandStats,
  CommandOrchestratorConfig,
  ClearColor,
  RenderPassConfig,
  FrameContext,
  FrameBindGroups,
  BufferUpdater,
} from "./src/renderer/commands.ts";

// =============================================================================
// Simulation
// =============================================================================

export {
  createSimulationController,
  calculateAlphaDecay,
  DEFAULT_SIMULATION_CONFIG,
} from "./src/simulation/controller.ts";
export type {
  SimulationController,
  SimulationControllerConfig,
  SimulationState as SimulationControllerState,
  SimulationEvents,
} from "./src/simulation/controller.ts";

export {
  createSimulationPipeline,
  recordSimulationStep,
  createSimulationBindGroups,
  DEFAULT_SIMULATION_PIPELINE_CONFIG,
} from "./src/simulation/pipeline.ts";
export type {
  SimulationPipeline,
  SimulationPipelineConfig,
  SimulationBuffers,
  SimulationBindGroups,
} from "./src/simulation/pipeline.ts";

export {
  createAlphaManager,
  calculateDecayRate,
  calculateIterations,
  createConvergenceDetector,
  createAdaptiveAlphaController,
  DEFAULT_ALPHA_CONFIG,
} from "./src/simulation/alpha.ts";
export type {
  AlphaManager,
  AlphaConfig,
  ConvergenceDetector,
  AdaptiveAlphaController,
} from "./src/simulation/alpha.ts";

export {
  forceConfigBuilder,
  validateForceConfig,
  forceConfigToUniformData,
  mergeForceConfig,
  DEFAULT_FORCE_CONFIG,
  FORCE_PRESETS,
} from "./src/simulation/config.ts";
export type { FullForceConfig, ForceConfigBuilder } from "./src/simulation/config.ts";

// =============================================================================
// Graph Data
// =============================================================================

export {
  parseGraphInput,
  validateGraphInput,
  createEdgeIndicesBuffer,
  DEFAULT_PARSER_CONFIG,
} from "./src/graph/parser.ts";
export type { ParsedGraph, ParserConfig } from "./src/graph/parser.ts";

export {
  parseGraphTypedInput,
  validateGraphTypedInput,
  createTypedInput,
  mergeTypedInputs,
  DEFAULT_TYPED_PARSER_CONFIG,
} from "./src/graph/typed_parser.ts";
export type { TypedParserConfig } from "./src/graph/typed_parser.ts";

export {
  createIdMap,
  createSequentialIdMap,
  createIdMapFromArray,
  serializeIdMap,
  deserializeIdMap,
  mapIdMap,
} from "./src/graph/id_map.ts";
export type { IdMap, IdLike } from "./src/graph/id_map.ts";

export {
  initializePositions,
  initializeRandom,
  initializeGrid,
  initializeCircle,
  initializeSpiral,
  initializePhyllotaxis,
  needsInitialization,
  addJitter,
  DEFAULT_INITIALIZE_CONFIG,
} from "./src/graph/initialize.ts";
export type { InitializeConfig, InitializationStrategy } from "./src/graph/initialize.ts";

// =============================================================================
// Main API
// =============================================================================

export {
  createHeroineGraph,
  isSupported,
  getSupportInfo,
  DEFAULT_WASM_URL,
  VERSION,
} from "./src/api/factory.ts";
export type { CreateHeroineGraphOptions, InitResult } from "./src/api/factory.ts";

export { HeroineGraph } from "./src/api/graph.ts";
export type { HeroineGraphConfig } from "./src/api/graph.ts";
