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
  BackgroundClickEvent,
  BoundingBox,
  Color,
  ColorScale,
  ContourLayerConfig,
  Edge,
  EdgeClickEvent,
  // Edge flow types
  EdgeFlowConfig,
  EdgeFlowLayerConfig,
  EdgeFlowWaveShape,
  EdgeHoverEnterEvent,
  EdgeHoverLeaveEvent,
  EdgeId,
  EdgeInput,
  // Edge types
  EdgeMetadata,
  EdgeState,
  EventHandler,
  EventMap,
  ForceConfig,
  // Configuration
  GraphConfig,
  // Event types
  GraphEvent,
  GraphInput,
  GraphTypedInput,
  HeatmapLayerConfig,
  HeroineGraphEvent,
  LabelLayerConfig,
  Layer,
  LayerConfig,
  // Layer types
  LayerType,
  MetaballLayerConfig,
  Node,
  NodeClickEvent,
  NodeDoubleClickEvent,
  NodeDragEndEvent,
  NodeDragMoveEvent,
  NodeDragStartEvent,
  NodeHoverEnterEvent,
  NodeHoverLeaveEvent,
  // Identifiers
  NodeId,
  // Input types
  NodeInput,
  // Node types
  NodeMetadata,
  NodeState,
  SelectionChangeEvent,
  SimulationEndEvent,
  SimulationState,
  // Simulation types
  SimulationStatus,
  SimulationTickEvent,
  // Primitives
  Vec2,
  ViewportChangeEvent,
  // Viewport types
  ViewportState,
} from "./src/types.ts";

// =============================================================================
// Errors
// =============================================================================

export { assert, ErrorCode, Errors, HeroineGraphError, wrapAsync } from "./src/errors.ts";

// =============================================================================
// WebGPU
// =============================================================================

export { checkWebGPU, describeWebGPUStatus, hasWebGPU } from "./src/webgpu/check.ts";
export type { WebGPUStatus } from "./src/webgpu/check.ts";

export {
  createDepthTexture,
  createGPUContext,
  destroyGPUContext,
  estimateMaxNodes,
  getCurrentTexture,
  resizeGPUContext,
} from "./src/webgpu/context.ts";
export type { GPUContext, GPUContextOptions } from "./src/webgpu/context.ts";

// =============================================================================
// WASM
// =============================================================================

export {
  createWasmEngine,
  createWasmEngineWithCapacity,
  getWasmModule,
  isWasmLoaded,
  loadWasmModule,
  WasmMemory,
} from "./src/wasm/loader.ts";

// =============================================================================
// Events
// =============================================================================

export { createEventEmitter, createTimestamp, EventEmitter, Events } from "./src/events/emitter.ts";

// =============================================================================
// Viewport
// =============================================================================

export { createViewport, DEFAULT_VIEWPORT_CONFIG, Viewport } from "./src/viewport/viewport.ts";
export type { ViewportConfig } from "./src/viewport/viewport.ts";

export {
  boundsCenter,
  distanceToBounds,
  expandBounds,
  fitBoundsScale,
  getVisibleBounds,
  graphToClipMatrix,
  graphToScreen,
  graphToScreenMatrix,
  identity,
  invert,
  multiply,
  pointInBounds,
  rotate,
  scale,
  screenToGraph,
  screenToGraphMatrix,
  transformPoint,
  translate,
} from "./src/viewport/transforms.ts";
export type { Matrix3 } from "./src/viewport/transforms.ts";

export {
  createViewportUniformBuffer,
  VIEWPORT_BIND_GROUP_LAYOUT_ENTRY,
  VIEWPORT_UNIFORM_SIZE,
  VIEWPORT_UNIFORM_WGSL,
  ViewportUniformBuffer,
} from "./src/viewport/uniforms.ts";

// =============================================================================
// Buffers
// =============================================================================

export {
  DEFAULT_POSITION_BUFFER_CONFIG,
  PositionBufferManager,
} from "./src/renderer/buffers/positions.ts";
export type { PositionBufferConfig } from "./src/renderer/buffers/positions.ts";

export {
  createFloat32PingPong,
  createInt32PingPong,
  createUint32PingPong,
  PingPongBuffer,
} from "./src/renderer/buffers/pingpong.ts";
export type { BufferPair, PingPongBufferConfig } from "./src/renderer/buffers/pingpong.ts";

export {
  createSimulationUniformBuffer,
  createViewportUniformBuffer as createViewportUniformBuffer2,
  DEFAULT_SIMULATION_UNIFORMS,
  forceConfigToUniforms,
  SIMULATION_UNIFORMS_SIZE,
  UniformBuffer,
  VIEWPORT_UNIFORMS_SIZE,
  viewportStateToUniforms,
} from "./src/renderer/buffers/uniforms.ts";
export type { SimulationUniforms, ViewportUniforms } from "./src/renderer/buffers/uniforms.ts";

export {
  DEFAULT_EDGE_BUFFER_CONFIG,
  EdgeBufferManager,
  edgePairsToCSR,
} from "./src/renderer/buffers/edges.ts";
export type { CSREdgeData, EdgeBufferConfig } from "./src/renderer/buffers/edges.ts";

// =============================================================================
// Render Pipelines
// =============================================================================

export {
  createNodeBindGroup,
  createNodeRenderPipeline,
  createViewportBindGroup as createNodeViewportBindGroup,
  DEFAULT_NODE_PIPELINE_CONFIG,
  renderNodes,
} from "./src/renderer/pipelines/nodes.ts";
export type { NodePipelineConfig, NodeRenderPipeline } from "./src/renderer/pipelines/nodes.ts";

export {
  createEdgeBindGroup,
  createEdgeRenderPipeline,
  createEdgeViewportBindGroup,
  DEFAULT_EDGE_PIPELINE_CONFIG,
  renderEdges,
} from "./src/renderer/pipelines/edges.ts";
export type { EdgePipelineConfig, EdgeRenderPipeline } from "./src/renderer/pipelines/edges.ts";

export {
  createEdgeFlowConfig,
  DEFAULT_EDGE_FLOW_CONFIG,
  DISABLED_FLOW_LAYER,
  EDGE_FLOW_PRESETS,
  getFlowPreset,
  waveShapeToFloat,
} from "./src/renderer/edge_flow.ts";
export type { EdgeFlowPreset } from "./src/renderer/edge_flow.ts";

// =============================================================================
// Render Loop
// =============================================================================

export {
  createFramePacer,
  createGPUTimer,
  createRenderLoop,
  DEFAULT_RENDER_LOOP_CONFIG,
} from "./src/renderer/render_loop.ts";
export type {
  FramePacer,
  FrameStats,
  GPUTimer,
  RenderCallback,
  RenderLoop,
  RenderLoopConfig,
} from "./src/renderer/render_loop.ts";

// =============================================================================
// GPU Commands
// =============================================================================

export {
  calculateWorkgroups,
  createBufferUpdater,
  createCommandOrchestrator,
  createComputePassDescriptor,
  createRenderPassDescriptor,
  DEFAULT_CLEAR_COLOR,
  dispatchCompute,
} from "./src/renderer/commands.ts";
export type {
  BufferUpdater,
  ClearColor,
  CommandOrchestrator,
  CommandOrchestratorConfig,
  CommandStats,
  FrameBindGroups,
  FrameContext,
  RenderPassConfig,
} from "./src/renderer/commands.ts";

// =============================================================================
// Simulation
// =============================================================================

export {
  calculateAlphaDecay,
  createSimulationController,
  DEFAULT_SIMULATION_CONFIG,
} from "./src/simulation/controller.ts";
export type {
  SimulationController,
  SimulationControllerConfig,
  SimulationEventData,
  SimulationEventEmitter,
  SimulationEventHandler,
  SimulationEventType,
  SimulationState as SimulationControllerState,
} from "./src/simulation/controller.ts";

export {
  copyPositionsToReadback,
  createSimulationBindGroups,
  createSimulationPipeline,
  DEFAULT_SIMULATION_PIPELINE_CONFIG,
  readbackPositions,
  recordSimulationStep,
} from "./src/simulation/pipeline.ts";
export type {
  SimulationBindGroups,
  SimulationBuffers,
  SimulationPipeline,
  SimulationPipelineConfig,
} from "./src/simulation/pipeline.ts";

export {
  calculateDecayRate,
  calculateIterations,
  createAdaptiveAlphaController,
  createAlphaManager,
  createConvergenceDetector,
  DEFAULT_ALPHA_CONFIG,
} from "./src/simulation/alpha.ts";
export type {
  AdaptiveAlphaController,
  AlphaConfig,
  AlphaManager,
  ConvergenceDetector,
} from "./src/simulation/alpha.ts";

export {
  DEFAULT_FORCE_CONFIG,
  FORCE_PRESETS,
  forceConfigBuilder,
  forceConfigToUniformData,
  mergeForceConfig,
  validateForceConfig,
} from "./src/simulation/config.ts";
export type { ForceConfigBuilder, FullForceConfig } from "./src/simulation/config.ts";

// =============================================================================
// Graph Data
// =============================================================================

export {
  createEdgeIndicesBuffer,
  DEFAULT_PARSER_CONFIG,
  parseGraphInput,
  validateGraphInput,
} from "./src/graph/parser.ts";
export type { ParsedGraph, ParserConfig } from "./src/graph/parser.ts";

export {
  createTypedInput,
  DEFAULT_TYPED_PARSER_CONFIG,
  mergeTypedInputs,
  parseGraphTypedInput,
  validateGraphTypedInput,
} from "./src/graph/typed_parser.ts";
export type { TypedParserConfig } from "./src/graph/typed_parser.ts";

export {
  createIdMap,
  createIdMapFromArray,
  createSequentialIdMap,
  deserializeIdMap,
  mapIdMap,
  serializeIdMap,
} from "./src/graph/id_map.ts";
export type { IdLike, IdMap } from "./src/graph/id_map.ts";

export {
  addJitter,
  DEFAULT_INITIALIZE_CONFIG,
  initializeCircle,
  initializeGrid,
  initializePhyllotaxis,
  initializePositions,
  initializeRandom,
  initializeSpiral,
  needsInitialization,
} from "./src/graph/initialize.ts";
export type { InitializationStrategy, InitializeConfig } from "./src/graph/initialize.ts";

// =============================================================================
// Main API
// =============================================================================

export {
  createHeroineGraph,
  DEFAULT_WASM_URL,
  getSupportInfo,
  isSupported,
  VERSION,
} from "./src/api/factory.ts";
export type { CreateHeroineGraphOptions, InitResult } from "./src/api/factory.ts";

export { HeroineGraph } from "./src/api/graph.ts";
export type { HeroineGraphConfig } from "./src/api/graph.ts";

// =============================================================================
// Interaction
// =============================================================================

export { createHitTester, DEFAULT_HIT_TESTER_CONFIG } from "./src/interaction/hit_test.ts";
export type {
  EdgeHitResult,
  EdgeProvider,
  HitResult,
  HitTester,
  HitTesterConfig,
  NodeHitResult,
  PositionProvider,
  SpatialQueryEngine,
} from "./src/interaction/hit_test.ts";

export { createPointerManager } from "./src/interaction/pointer.ts";
export type {
  NormalizedPointerEvent,
  PointerEventCallback,
  PointerEventType,
  PointerManager,
  PointerManagerConfig,
} from "./src/interaction/pointer.ts";

// =============================================================================
// Layers
// =============================================================================

export {
  createLayerManager,
  DEFAULT_LAYER_MANAGER_CONFIG,
  LayerManager,
} from "./src/layers/manager.ts";
export type { LayerInfo, LayerManagerConfig } from "./src/layers/manager.ts";

export {
  clearDensityTexture,
  COLOR_SCALES,
  createColorScaleTexture,
  createCustomColorScaleTexture,
  createDensityTexture,
  createHeatmapLayer,
  createHeatmapPipeline,
  DEFAULT_COLORMAP_UNIFORMS,
  DEFAULT_DENSITY_TEXTURE_CONFIG,
  DEFAULT_HEATMAP_CONFIG,
  DEFAULT_HEATMAP_UNIFORMS,
  generateColorScaleData,
  getColorScaleNames,
  HeatmapLayer,
  mergeHeatmapConfig,
  validateHeatmapConfig,
} from "./src/layers/mod.ts";
export type {
  ColormapUniforms,
  ColorRGBA,
  ColorScaleName,
  ColorScaleTexture,
  ColorStop,
  DensityTexture,
  DensityTextureConfig,
  HeatmapConfig as HeatmapLayerConfiguration,
  HeatmapPipeline,
  HeatmapRenderContext,
  HeatmapUniforms,
  Layer as VisualizationLayer,
} from "./src/layers/mod.ts";

// =============================================================================
// Value Streams
// =============================================================================

export {
  createColorScaleFromPreset,
  createGradientScale,
  createStreamManager,
  StreamManager,
  VALUE_COLOR_PRESETS,
  ValueStream,
} from "./src/streams/mod.ts";
export type {
  BlendMode,
  ColorStop as StreamColorStop,
  StreamBulkData,
  StreamDataPoint,
  StreamInfo,
  StreamManagerConfig,
  ValueColorScale,
  ValueStreamConfig,
} from "./src/streams/mod.ts";
