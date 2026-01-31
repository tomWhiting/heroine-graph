# Tasks: Advanced Visualization Layer System

**Input**: Design documents from `/specs/002-visualization-layers/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and module structure

- [x] T001 Create `packages/core/src/layers/` directory structure
- [x] T002 Create `packages/core/src/styling/` directory structure
- [x] T003 [P] Create `packages/core/src/renderer/buffers/` directory structure
- [x] T004 [P] Add d3-scale, d3-color to dependencies in `deno.json` (for color scales)
  - Note: Uses custom color scale implementation instead of d3

---

## Phase 2: Foundational (GPU Buffer Infrastructure)

**Purpose**: Core GPU buffer infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T005 Create `NodeStyleBuffer` class in `packages/core/src/renderer/buffers/node_style.ts`
  - Buffer layout: 8 floats per node (color RGBA + size + reserved)
  - Methods: `create()`, `update()`, `updateRange()`, `destroy()`
  - Note: Implemented via nodeAttributes buffer + setNodeColors/setNodeSizes API
- [x] T006 [P] Create `EdgeStyleBuffer` class in `packages/core/src/renderer/buffers/edge_style.ts`
  - Buffer layout: 8 floats per edge (color RGBA + width + curvature + reserved)
  - Methods: `create()`, `update()`, `updateRange()`, `destroy()`
  - Note: Implemented via edgeAttributes buffer + setEdgeColors/setEdgeWidths API
- [x] T007 Create style buffer bind group in `packages/core/src/renderer/pipelines/nodes.ts`
  - Add bind group layout for nodeStyleBuffer
  - Update pipeline to read per-node colors/sizes from buffer
- [x] T008 [P] Create style buffer bind group in `packages/core/src/renderer/pipelines/edges.ts`
  - Add bind group layout for edgeStyleBuffer
  - Update pipeline to read per-edge colors/widths from buffer
- [x] T009 Update node vertex shader `packages/core/src/renderer/shaders/node.vert.wgsl`
  - Read per-node color and size from storage buffer
  - Support fallback to uniform defaults
- [x] T010 [P] Update edge vertex shader `packages/core/src/renderer/shaders/edge.vert.wgsl`
  - Read per-edge color and width from storage buffer
  - Support fallback to uniform defaults
- [x] T011 Export new buffer types from `packages/core/mod.ts`

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Per-Item Styling API (Priority: P1) 🎯 MVP

**Goal**: Enable setting colors, sizes, widths for individual nodes/edges via Float32Array

**Independent Test**: Call `setNodeColors(Float32Array)` with test data and verify each node renders with its assigned color

### Implementation for User Story 1

- [x] T012 [US1] Add `setNodeColors(colors: Float32Array)` method in `packages/core/src/api/graph.ts`
  - Validate array length = nodeCount × 4
  - Update NodeStyleBuffer
  - Error message: "Expected {expected} values for {nodeCount} nodes (4 per node), got {actual}"
- [x] T013 [P] [US1] Add `setNodeSizes(sizes: Float32Array)` method in `packages/core/src/api/graph.ts`
  - Validate array length = nodeCount × 1
  - Update NodeStyleBuffer
- [x] T014 [P] [US1] Add `setEdgeColors(colors: Float32Array)` method in `packages/core/src/api/graph.ts`
  - Validate array length = edgeCount × 4
  - Update EdgeStyleBuffer
- [x] T015 [P] [US1] Add `setEdgeWidths(widths: Float32Array)` method in `packages/core/src/api/graph.ts`
  - Validate array length = edgeCount × 1
  - Update EdgeStyleBuffer
- [x] T016 [US1] Implement fallback logic for undefined/NaN values in style buffers
  - Check each value during buffer update
  - Use global defaults when value is NaN or outside valid range
  - Note: Shader fallback via `select(DEFAULT_VALUE, value, value > 0.0)`
- [x] T017 [US1] Update Mission Control example `examples/mission-control/main.ts`
  - Add "Per-Item Styling" panel with random color button
  - Demonstrate setNodeColors/setEdgeColors usage
- [x] T018 [US1] Export VisualizationAPI types from `packages/core/mod.ts`

**Checkpoint**: Per-item styling API fully functional (FR-001 through FR-005)

---

## Phase 4: User Story 2 - Value Stream System (Priority: P1) 🎯 MVP

**Goal**: Enable defining value streams that feed data to visualization layers (heatmap, contours, metaballs)

**Independent Test**: Define stream, push data for specific nodes, configure heatmap layer to use stream, verify heat intensity at node positions corresponds to stream values

**Key Concept**: Value streams are a "dumb pipe" - they store node-to-value mappings that layers consume. NO aggregation is performed by the library (user computes any aggregates in their application code). The stream values drive heat intensity at node positions in the heatmap/contour/metaball layers.

### Implementation for User Story 2

- [x] T019 [US2] Create `ValueStream` class in `packages/core/src/streams/value_stream.ts`
  - Properties: id, name, colorScale (domain/range), blendMode, enabled, opacity
  - Methods: `setData()`, `setBulkData()`, `getValue(nodeIndex)`, `getAllColors()`, `clear()`
  - NO aggregation logic - accepts pre-computed values only
- [x] T020 [US2] Create `StreamManager` class in `packages/core/src/streams/stream_manager.ts`
  - Store multiple streams by ID
  - Methods: `defineStream()`, `setStreamData()`, `removeStream()`, `getStreamInfo()`
  - `computeBlendedColors()` for multi-stream blending
- [x] T021 [US2] Create color scale utilities in `packages/core/src/streams/types.ts`
  - `ValueColorScale` type with domain/stops
  - `createColorScaleFromPreset()`, `createGradientScale()`
  - VALUE_COLOR_PRESETS: error (red), warning (yellow), success (green), activity (blue), importance (purple), heat
- [x] T022 [US2] Add `defineValueStream(config)` method in `packages/core/src/api/graph.ts`
  - Validate config (unique ID)
  - Create and store ValueStream via StreamManager
- [x] T023 [P] [US2] Add `setStreamValues(streamId, data)` method in `packages/core/src/api/graph.ts`
  - Validate streamId exists
  - Push data to stream (no aggregation)
- [x] T024 [P] [US2] Add `setStreamBulkValues(streamId, data)` method for efficient large updates
- [x] T025 [P] [US2] Add `removeValueStream(streamId)`, `enableValueStream()`, `disableValueStream()` methods
- [x] T026 [US2] Export stream types from `packages/core/mod.ts`
  - ValueStreamConfig, StreamDataPoint, StreamBulkData, StreamInfo, BlendMode, ValueColorScale
- [x] T027 [US2] Connect streams to HeatmapLayer as data source
  - HeatmapLayer.setDataSource(streamId | 'density') - use stream values for intensity
  - When stream is source: intensity at (x,y) based on node positions + stream values
  - Default 'density' mode: current behavior (node count per area)
- [x] T028 [US2] Connect streams to ContourLayer as data source
  - ContourLayer.setDataSource(streamId) - contours around stream value thresholds
- [x] T029 [US2] Connect streams to MetaballLayer as data source (optional)
  - MetaballLayer.setDataSource(streamId) - blob intensity from stream values
- [x] T030 [US2] Update Mission Control example `examples/mission-control/main.ts`
  - Add "Value Streams" panel with stream definition
  - Demo stream → layer binding (e.g., "errors" stream → heatmap)
  - Show nodes colored by type, heatmap showing stream values

**Checkpoint**: Value stream system fully functional (FR-006 through FR-010)

**Note**: Per spec clarification (Session 2026-01-12): "Library is a 'dumb pipe': values in → colors out → blend → render." NO built-in aggregation.

---

## Phase 5: User Story 4 - Type-Based Styling (Priority: P2)

**Goal**: Enable defining visual styles by node/edge type without manual Float32Array building

**Independent Test**: Define type styles, load typed nodes, verify each node receives its type's style

### Implementation for User Story 4

- [x] T029 [US4] Create `StyleResolver` class in `packages/core/src/styling/style_resolver.ts`
  - Precedence: per-item > type > global defaults
  - Methods: `resolveNodeStyle(nodeIndex)`, `resolveEdgeStyle(edgeIndex)`
  - Note: Implemented as resolveNodeStyle/resolveEdgeStyle in TypeStyleManager
- [x] T030 [US4] Create `TypeStyleManager` in `packages/core/src/styling/type_styles.ts`
  - Store NodeTypeStyleMap and EdgeTypeStyleMap
  - Methods: `setNodeTypeStyles()`, `setEdgeTypeStyles()`, `getStyleForType()`
- [x] T031 [US4] Add `setNodeTypeStyles(styles: NodeTypeStyleMap)` in `packages/core/src/api/graph.ts`
  - Store type styles in TypeStyleManager
  - Rebuild style buffers from type mappings
- [x] T032 [P] [US4] Add `setEdgeTypeStyles(styles: EdgeTypeStyleMap)` in `packages/core/src/api/graph.ts`
  - Store type styles in TypeStyleManager
  - Rebuild style buffers from type mappings
- [x] T033 [US4] Implement automatic style buffer generation from types
  - Iterate nodes, lookup type, build Float32Array
  - Handle undefined types with global defaults
- [x] T034 [US4] Update Mission Control example `examples/mission-control/main.ts`
  - Add "Type Styling" panel with preset type configurations

**Checkpoint**: Type-based styling fully functional (FR-016 through FR-019)

---

## Phase 6: User Story 3 - Multi-Layer System (Priority: P2)

**Goal**: Enable creating filtered visualization layers with independent visibility

**Independent Test**: Define multiple layers with different filters, toggle visibility, verify render output

### Implementation for User Story 3

- [ ] T035 [US3] Create `VisualizationLayer` class in `packages/core/src/layers/visualization_layer.ts`
  - Properties: id, name, nodeFilter, edgeFilter, visualizations, zOrder, visible
  - Methods: `matchesNode()`, `matchesEdge()`, `setVisible()`
- [ ] T036 [US3] Create `LayerManager` class in `packages/core/src/layers/layer_manager.ts`
  - Store layers sorted by zOrder
  - Methods: `defineLayer()`, `removeLayer()`, `setLayerVisible()`, `getLayerVisibility()`
  - Coordinate filtered rendering
- [ ] T037 [US3] Add `defineLayer(config)` method in `packages/core/src/api/graph.ts`
  - Validate config (unique ID, at least one visualization)
  - Create and store VisualizationLayer
- [ ] T038 [P] [US3] Add `removeLayer(layerId)` method in `packages/core/src/api/graph.ts`
- [ ] T039 [US3] Implement layer-aware render loop in `packages/core/src/renderer/`
  - Sort layers by zOrder (ascending)
  - For each visible layer: apply filters, render requested visualizations
- [ ] T040 [US3] Create filtered index buffers for layer rendering
  - Generate per-layer node indices based on nodeFilter
  - Generate per-layer edge indices based on edgeFilter
- [ ] T041 [US3] Handle nodes appearing in multiple layers
  - Same node rendered multiple times with different treatments
- [ ] T042 [US3] Update Mission Control example `examples/mission-control/main.ts`
  - Add "Layers" panel with layer toggles

**Checkpoint**: Multi-layer system fully functional (FR-011 through FR-015)

---

## Phase 7: User Story 5 - Dual-Layer PWM Edge Flow (Priority: P2)

**Goal**: Enable dual-layer edge flow animation with punch-through effect

**Independent Test**: Enable flow with specific parameters, verify pulse animation matches configuration

### Implementation for User Story 5

- [x] T043 [US5] Enhance `edge.frag.wgsl` with full Cosmograph PWM implementation
  - Wave functions: squareWave, triangleWave, sineWave (sin^4)
  - Two independent flow layers with all parameters
  - Punch-through logic: `sparkPunch = flowValue2 * (1.0 - flow2Fade * 0.7)`
- [x] T044 [US5] Add Layer 2 flow uniforms to edge pipeline
  - Extend flow uniform buffer with layer2 parameters
  - Update bind group layout
- [x] T045 [US5] Update `setEdgeFlow(config)` method in `packages/core/src/api/graph.ts`
  - Accept full EdgeFlowConfig with layer1 and layer2
  - Validate parameter ranges (speed 0.01-2.0, pulseWidth 0.005-0.99, etc.)
- [x] T046 [US5] Add per-layer RGBA color configuration
  - Color uniform with alpha as blend amount
  - Apply tint in fragment shader
- [x] T047 [US5] Update Mission Control example `examples/mission-control/main.ts`
  - Add Layer 1 and Layer 2 flow controls
  - Color pickers for flow tint

**Checkpoint**: Dual-layer PWM edge flow fully functional (FR-020 through FR-023)

---

## Phase 8: User Story 6 - Curved Edges (Priority: P3)

**Goal**: Enable smooth curved edges using conic Bezier curves

**Independent Test**: Enable curved edges globally, verify curve rendering, test per-edge curvature overrides

### Implementation for User Story 6

- [x] T048 [US6] Create `curved_edge.wgsl` shader in `packages/core/src/renderer/shaders/`
  - Implement conicParametricCurve function from research.md
  - Control point: midpoint + perpendicular offset × curvature
  - Tessellation with configurable segment count
  - Note: Implemented directly in edge.vert.wgsl with conic_bezier() and compute_control_point() functions
- [x] T049 [US6] Add curved edge pipeline in `packages/core/src/renderer/pipelines/edges.ts`
  - Toggle between straight and curved rendering
  - Pass tessellation params via uniforms
  - Note: Added CurvedEdgeConfig, curveConfigBuffer, updateCurveConfig(); renderEdges uses segments*6 vertices when curved
- [x] T050 [US6] Add `curvedEdges` configuration to `packages/core/src/api/graph.ts`
  - Config options: curvedEdges, curveSegments, curveWeight, curveControlPointDistance
  - Update via `setConfig()` method
  - Note: Added setCurvedEdges(), getCurvedEdgeConfig(), enableCurvedEdges(), disableCurvedEdges()
- [x] T051 [US6] Add `setEdgeCurvatures(curvatures: Float32Array)` in `packages/core/src/api/graph.ts`
  - Validate array length = edgeCount
  - Upload to edgeStyleBuffer curvature slot
  - Note: Edge buffer expanded from 6 to 8 floats per edge; curvature at offset 6
- [x] T052 [US6] Support negative curvature values for opposite direction bend
  - Note: Supported by shader - positive curves right, negative curves left
- [x] T053 [US6] Update Mission Control example `examples/mission-control/main.ts`
  - Add curved edges toggle and curvature sliders
  - Note: Added Curved Edges panel with enabled toggle, segments, weight, curvature sliders, and randomize button

**Checkpoint**: Curved edges fully functional (FR-024 through FR-027)

---

## Phase 9: User Story 7 - Node Border Configuration (Priority: P3)

**Goal**: Enable customizable node borders (on/off, thickness, color)

**Independent Test**: Toggle border visibility, adjust thickness/color, verify render output

### Implementation for User Story 7

- [x] T054 [US7] Add border uniforms to node pipeline
  - enabled (bool), thickness (float), color (RGBA)
  - Update bind group layout
  - Note: Added RenderConfig uniform struct and renderConfigBindGroupLayout
- [x] T055 [US7] Update `node.frag.wgsl` with border rendering
  - SDF-based border calculation
  - Apply border color at edge of node
  - Note: Made border config uniform-based instead of const
- [x] T056 [US7] Add `setNodeBorder(config)` method in `packages/core/src/api/graph.ts`
  - Config: enabled, thickness, color
  - Update uniforms
  - Note: Also added enableNodeBorder(), disableNodeBorder(), getNodeBorderConfig()
- [ ] T057 [US7] Support per-type border overrides in TypeStyleManager
  - BorderConfig in NodeTypeStyle
  - Override global border settings per type
  - Note: Types exist (borderColor, borderWidth in NodeTypeStyle), but wiring requires expanding per-node buffer layout from 6 to 8+ floats
- [ ] T058 [US7] Implement hover/selection border states
  - Configurable border style for hover and selected states
  - Note: Selection ring exists, border already visible; needs API for custom states
- [x] T059 [US7] Update Mission Control example `examples/mission-control/main.ts`
  - Add border toggle, thickness slider, color picker
  - Note: Added Node Borders panel with enabled toggle, width slider, and color picker

**Checkpoint**: Node border configuration fully functional (FR-028 through FR-030)

---

## Phase 10: User Story 8 - Topographical Contours (Priority: P3)

**Goal**: Enable contour line rendering on density heatmaps

**Independent Test**: Enable contours on heatmap, configure thresholds, verify contour lines appear at correct density values

### Implementation for User Story 8

- [x] T060 [US8] Create `contour.comp.wgsl` shader in `packages/core/src/renderer/shaders/`
  - Implement marching squares algorithm
  - Input: density field texture
  - Output: line segments to storage buffer
  - Note: Implemented in packages/core/src/layers/contour/
- [x] T061 [US8] Create contour pipeline in `packages/core/src/renderer/pipelines/contours.ts`
  - Compute pass for marching squares
  - Render pass for line segments
  - Note: Implemented as ContourLayer with SimpleContourPipeline
- [x] T062 [US8] Create contour vertex buffer for line segments
  - Dynamic size based on threshold count and density field
- [x] T063 [US8] Add `setContours(config)` method in `packages/core/src/api/graph.ts`
  - Config: enabled, thresholds, lineColor, lineThickness
  - Trigger contour regeneration
  - Note: Implemented as enableContour(), disableContour(), setContourConfig()
- [x] T064 [US8] Implement contour dirty flag for efficient updates
  - Regenerate only when density field changes
- [x] T065 [US8] Update Mission Control example `examples/mission-control/main.ts`
  - Add contour toggle, threshold inputs, line color picker

**Checkpoint**: Topographical contours fully functional (FR-031 through FR-034)

---

## Phase 11: User Story 9 - Layer Visibility Toggles (Priority: P3)

**Goal**: Enable quick show/hide toggles for visualization layers

**Independent Test**: Create layers, toggle visibility, verify render state matches visibility settings

### Implementation for User Story 9

- [x] T066 [US9] Add `setLayerVisible(layerId, visible)` method in `packages/core/src/api/graph.ts`
  - Update layer visibility state
  - Trigger re-render
  - Note: Implemented in LayerManager.setLayerVisible()
- [x] T067 [P] [US9] Add `getLayerVisibility()` method in `packages/core/src/api/graph.ts`
  - Return Record<string, boolean> of all layer states
  - Note: Implemented as isLayerVisible() per layer (isHeatmapEnabled, isContourEnabled, etc.)
- [x] T068 [US9] Ensure hidden layers preserve configuration
  - Layer not destroyed, just skipped in render loop
- [x] T069 [US9] Implement fast visibility toggle (<16ms target)
  - No buffer rebuilds on toggle
  - Skip layer in render command encoding

**Checkpoint**: Layer visibility toggles fully functional (FR-035 through FR-037)

---

## Phase 12: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T070 [P] Add serialization for all styling/layer/channel configs
  - `exportConfig()` and `importConfig()` methods (SC-009)
- [ ] T071 [P] Performance testing at scale
  - 100K nodes per-item styling <100ms (SC-001)
  - 50K edges flow animation 60fps (SC-004)
  - 50K edges curved rendering 60fps (SC-005)
- [ ] T072 Code cleanup and consolidation
  - Remove duplicate code across styling systems
  - Ensure consistent error messages
- [ ] T073 Run quickstart.md validation
  - Verify all code samples compile and work
- [ ] T074 Final Mission Control integration test
  - All features accessible and working together

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational - MVP, must complete first
- **US2 (Phase 4)**: Depends on Foundational and US1 (uses per-item styling)
- **US4 (Phase 5)**: Depends on Foundational and US1 (builds on per-item API)
- **US3 (Phase 6)**: Depends on Foundational - can parallel with US4/US5
- **US5 (Phase 7)**: Depends on Foundational - can parallel with US3/US4
- **US6-9 (Phases 8-11)**: Depends on Foundational - can parallel with each other
- **Polish (Phase 12)**: Depends on all desired user stories

### Parallel Opportunities

After Foundational phase completes:
- US3 (Layers), US4 (Type Styling), US5 (Edge Flow) can proceed in parallel
- US6, US7, US8, US9 are independent and can all proceed in parallel
- Within each phase, tasks marked [P] can run in parallel

### MVP Path (Minimum Viable Product)

1. Phase 1: Setup
2. Phase 2: Foundational
3. Phase 3: US1 - Per-Item Styling API ← **MVP Checkpoint**
4. Phase 4: US2 - Diagnostic Channels
5. (Stop here for core diagnostic visualization capability)

---

## Notes

- All shader files use `.wgsl` extension (WebGPU Shading Language)
- Float32Array APIs match Cosmograph patterns per research.md
- Performance targets from SC-001 through SC-010 must be verified
- Mission Control updates in each phase enable manual testing
