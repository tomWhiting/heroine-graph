/**
 * GraphMother - Core Library
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
  // Mutation event types
  EdgeAddEvent,
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
  EdgeRemoveEvent,
  EdgeState,
  EventHandler,
  EventMap,
  ForceConfig,
  // Configuration
  GraphConfig,
  // Event types
  GraphEvent,
  GraphInput,
  GraphMotherEvent,
  GraphMutateEvent,
  GraphTypedInput,
  HeatmapLayerConfig,
  LabelLayerConfig,
  Layer,
  LayerConfig,
  // Layer types
  LayerType,
  MetaballLayerConfig,
  Node,
  NodeAddEvent,
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
  NodeRemoveEvent,
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

export { assert, ErrorCode, Errors, GraphMotherError, wrapAsync } from "./src/errors.ts";

// =============================================================================
// WebGPU
// =============================================================================

export { checkWebGPU, describeWebGPUStatus, hasWebGPU } from "./src/webgpu/check.ts";
export type { WebGPUStatus } from "./src/webgpu/check.ts";

export {
  createGPUContext,
  destroyGPUContext,
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
  createCollisionBindGroup,
  createCollisionBuffers,
  createCollisionPipeline,
  destroyCollisionBuffers,
  recordCollisionPass,
  updateCollisionUniforms,
  uploadNodeSizes,
} from "./src/simulation/collision.ts";
export type {
  CollisionBindGroup,
  CollisionBuffers,
  CollisionPipeline,
} from "./src/simulation/collision.ts";

export {
  DEFAULT_FORCE_CONFIG,
  FORCE_PRESETS,
  forceConfigBuilder,
  mergeForceConfig,
  validateForceConfig,
} from "./src/simulation/config.ts";
export type { ForceConfigBuilder, FullForceConfig } from "./src/simulation/config.ts";

// Force algorithms — the registry and the types naming setForceAlgorithm()'s
// parameter and the plugin surface for custom algorithms.
export {
  createAlgorithmRegistry,
  ForceAlgorithmRegistry,
  getAlgorithmRegistry,
} from "./src/simulation/algorithms/registry.ts";
export type {
  AlgorithmBindGroups,
  AlgorithmBuffers,
  AlgorithmPipelines,
  AlgorithmRenderContext,
  ForceAlgorithm,
  ForceAlgorithmInfo,
  ForceAlgorithmType,
} from "./src/simulation/algorithms/types.ts";

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

export { createGraphMother, getSupportInfo, isSupported, VERSION } from "./src/api/factory.ts";
export type { CreateGraphMotherOptions, InitResult } from "./src/api/factory.ts";

export { GraphMother } from "./src/api/graph.ts";
export type { GraphMotherConfig } from "./src/api/graph.ts";

export { MutableGraphState } from "./src/api/graph_state.ts";
export { growCapacity, initialCapacity } from "./src/api/buffer_capacity.ts";
export type { BufferCapacity } from "./src/api/buffer_capacity.ts";

// =============================================================================
// Interaction
// =============================================================================

export { createHitTester, DEFAULT_HIT_TESTER_CONFIG } from "./src/interaction/hit_test.ts";
export type {
  EdgeColumns,
  EdgeHitResult,
  EdgeProvider,
  HitResult,
  HitTester,
  HitTesterConfig,
  NodeColumns,
  NodeHitResult,
  PositionProvider,
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

// Labels Layer
export {
  DEFAULT_LABEL_CONFIG,
  getGlyph,
  getGlyphUVs,
  getKerning,
  LabelManager,
  LabelsLayer,
  loadDefaultFontAtlas,
  loadFontAtlas,
  measureText,
} from "./src/layers/mod.ts";
export type {
  BMFontChar,
  BMFontData,
  FontAtlas,
  LabelConfig,
  LabelData,
  LabelsRenderContext,
  PositionProvider as LabelPositionProvider,
  VisibleLabel,
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

// =============================================================================
// Type-Based Styling
// =============================================================================

export { createTypeStyleManager, TypeStyleManager } from "./src/styling/mod.ts";
export type {
  EdgeTypeStyle,
  EdgeTypeStyleMap,
  NodeTypeStyle,
  NodeTypeStyleMap,
  ResolvedEdgeStyle,
  ResolvedNodeStyle,
} from "./src/styling/mod.ts";

// =============================================================================
// Semantic LOD
// =============================================================================

export { CrossfadeScheduler, NODE_ALPHA_OPAQUE, NODE_ALPHA_TRANSPARENT } from "./src/lod/mod.ts";

// =============================================================================
// DOM Card Overlay
// =============================================================================

export {
  CardDriver,
  createDefaultCardProvider,
  DEFAULT_DOM_OVERLAY_CONFIG,
  externalIdForSlot,
  slotForExternalId,
} from "./src/overlay/mod.ts";
export type {
  CardChange,
  CardDriverOptions,
  CardNode,
  CardPlacement,
  CardPlacementChange,
  CardProvider,
  CardSize,
  CardStateChange,
  DefaultCardState,
  DomOverlayConfig,
  NodeIdentitySource,
} from "./src/overlay/mod.ts";

// =============================================================================
// Central Configuration Nexus
// =============================================================================

// Import from config nexus for centralized access to all settings.
// Note: Some types (ColorRGBA, ColorScaleName, etc.) are already exported
// above from their original locations for backwards compatibility.
// The config nexus re-exports these for convenience.
export {
  // Node Border
  DEFAULT_NODE_BORDER_CONFIG,
  // Shared Data Source type
  type LayerDataSource,
} from "./src/config/index.ts";
export type { NodeBorderConfig } from "./src/config/index.ts";

// =============================================================================
// Utilities
// =============================================================================

// Color utilities - centralized parsing for the entire library
export {
  colorToHex,
  DEFAULT_COLORS,
  lerpColor,
  NAMED_COLORS,
  parseColor,
  parseColorToRGB,
  parseColorToRGBA,
  withAlpha,
} from "./src/utils/mod.ts";
export type { ColorInput, ColorRGB, RgbaColor } from "./src/utils/mod.ts";
