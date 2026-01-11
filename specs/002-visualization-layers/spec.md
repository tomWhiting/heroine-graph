# Feature Specification: Advanced Visualization Layer System

**Feature Branch**: `002-visualization-layers`
**Created**: 2026-01-11
**Status**: Draft
**Input**: Advanced Visualization Layer System for heroine-graph: A comprehensive enhancement to the graph visualization library adding configurable diagnostic channels for mapping arbitrary data to visual heat/color, a multi-layer system where nodes can appear on multiple layers with different visual treatments, per-node-type and per-edge-type styling with global defaults and type overrides, topographical contour rendering using marching squares on density fields, dual-layer PWM edge flow animation matching the Cosmograph implementation, curved edges with conic Bezier curves, node border configuration, per-item styling API, and layer visibility toggles.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Per-Item Styling API (Priority: P1)

A library user wants to set colors, sizes, and widths for individual nodes and edges without being constrained to a single global style. They have a Float32Array of RGBA values for each node based on their application's data model and want to apply these directly.

**Why this priority**: This is the foundational capability that all other features build upon. Without per-item styling, type-based styling and value streams cannot function.

**Independent Test**: Can be fully tested by calling `setNodeColors(Float32Array)` with test data and verifying that each node renders with its assigned color.

**Acceptance Scenarios**:

1. **Given** a graph with 100 nodes, **When** the user calls `setNodeColors()` with a Float32Array of 400 values (4 per node), **Then** each node displays its assigned RGBA color
2. **Given** a graph with edges, **When** the user calls `setEdgeColors()` with per-edge RGBA values, **Then** each edge displays its assigned color
3. **Given** a graph with edges, **When** the user calls `setEdgeWidths()` with per-edge width values, **Then** each edge renders at its assigned width
4. **Given** a partially filled Float32Array (some values undefined), **When** the user applies it, **Then** undefined values fall back to global defaults

---

### User Story 2 - Value Stream System (Priority: P1)

A developer building a code analysis tool wants to visualize compiler errors and warnings on a repository graph. They define "error" and "warning" value streams with their own color scales, compute aggregated values for folders in their application code, then push the final values to each stream. The library renders heat intensity and blends multiple streams visually.

**Why this priority**: This is the primary use case driving the feature - visualizing arbitrary numeric data as heat/color. It depends on P1 (per-item styling) but enables the core data visualization workflow.

**Independent Test**: Can be tested by defining a value stream, pushing data for specific nodes, and verifying the visual heat corresponds to the data values.

**Acceptance Scenarios**:

1. **Given** no streams defined, **When** the user calls `defineValueStream({ id: 'errors', colorScale: { domain: [0, 20], range: ['transparent', 'red'] } })`, **Then** a new stream is created and available for data
2. **Given** a defined stream, **When** the user calls `setStreamValues('errors', [{ nodeId: 'a', value: 5 }])`, **Then** node 'a' receives heat intensity proportional to value 5
3. **Given** a folder node, **When** the user pushes a pre-computed aggregate value for that folder, **Then** the folder displays heat based on that value (no library-side aggregation)
4. **Given** multiple streams with data, **When** both are active, **Then** stream colors blend according to their configured blend mode
5. **Given** a stream with a color scale, **When** values are pushed, **Then** each node's color is interpolated based on where its value falls in the scale domain

---

### User Story 3 - Multi-Layer System (Priority: P2)

A user wants to create a complex visualization with three layers: (1) a base layer showing all nodes and edges, (2) a heatmap layer showing only file nodes colored by a value stream, and (3) a metaball layer showing only folder nodes as organic blobs. Each layer has independent visibility toggles.

**Why this priority**: Builds on P1 capabilities to enable sophisticated multi-view visualizations. High impact for advanced users but requires foundational features first.

**Independent Test**: Can be tested by defining multiple layers with different node filters, toggling visibility, and verifying correct render output for each combination.

**Acceptance Scenarios**:

1. **Given** no layers defined, **When** the user calls `defineLayer({ id: 'base', nodeFilter: () => true, visualizations: ['nodes', 'edges'] })`, **Then** a new layer is created containing all nodes and edges
2. **Given** a layer with `nodeFilter: (n) => n.type === 'file'`, **When** the layer is rendered, **Then** only nodes of type 'file' appear
3. **Given** a node that matches filters for two layers, **When** both layers are visible, **Then** the node appears in both layers with each layer's visual treatment
4. **Given** a visible layer, **When** the user calls `setLayerVisible('layerId', false)`, **Then** the layer's contents are hidden
5. **Given** multiple layers, **When** rendered, **Then** layers render in their defined z-order (later layers on top)

---

### User Story 4 - Type-Based Styling Configuration (Priority: P2)

A user wants to define visual styles based on node and edge types without manually building Float32Arrays. They specify `{ file: { color: 'blue', size: 10 }, folder: { color: 'orange', size: 20 } }` and the library automatically applies these styles to matching nodes.

**Why this priority**: Quality-of-life feature that simplifies the per-item styling API. Depends on P1 but significantly improves developer experience.

**Independent Test**: Can be tested by defining type styles, loading a graph with typed nodes, and verifying each node receives its type's style.

**Acceptance Scenarios**:

1. **Given** type styles defined for 'file' and 'folder', **When** nodes are loaded with `type` properties, **Then** each node receives its type's visual style
2. **Given** a node with an undefined type, **When** styles are applied, **Then** the node receives global default styles
3. **Given** type styles and per-item overrides, **When** both are provided, **Then** per-item overrides take precedence over type styles
4. **Given** edge type styles defined, **When** edges have `type` properties, **Then** edges receive their type's color/width

---

### User Story 5 - Dual-Layer PWM Edge Flow Animation (Priority: P2)

A user wants to show directional data flow along edges with animated pulses. They configure two independent layers: Layer 1 with slow, wide sine waves representing baseline traffic, and Layer 2 with fast, narrow triangle-wave sparks representing activity bursts. The sparks "punch through" at their peaks over the baseline.

**Why this priority**: High-impact visual feature that enhances understanding of graph dynamics. Independent of layer system (different meaning of "layer").

**Independent Test**: Can be tested by enabling flow with specific parameters and verifying pulse animation matches configuration (width, count, speed, shape, color).

**Acceptance Scenarios**:

1. **Given** flow disabled, **When** user enables Layer 1 flow with `{ enabled: true, speed: 0.5, pulseWidth: 0.2 }`, **Then** animated pulses travel along edges
2. **Given** flow enabled, **When** user sets `waveShape: 0` (square), **Then** pulses have hard on/off transitions
3. **Given** flow enabled, **When** user sets `waveShape: 1` (sine), **Then** pulses have smooth bell-curve transitions
4. **Given** Layer 1 active, **When** user enables Layer 2 with faster speed, **Then** Layer 2 sparks animate independently over Layer 1
5. **Given** both layers active, **When** Layer 2 pulse peaks, **Then** it visually "punches through" Layer 1 with increased brightness
6. **Given** flow layer with color `[1, 0.5, 0, 0.8]`, **When** rendered, **Then** pulses are tinted orange with 80% blend

---

### User Story 6 - Curved Edges with Conic Bezier Curves (Priority: P3)

A user wants edges to render as smooth curves rather than straight lines for better visual aesthetics and reduced edge overlap. They configure global curve settings and optionally override curvature per-edge for specific visual effects.

**Why this priority**: Visual enhancement that improves graph readability. Lower priority as straight edges are functional.

**Independent Test**: Can be tested by enabling curved edges globally, verifying curve rendering, then testing per-edge curvature overrides.

**Acceptance Scenarios**:

1. **Given** straight edges (default), **When** user sets `curvedEdges: true`, **Then** all edges render as smooth curves
2. **Given** curved edges enabled, **When** user adjusts `curveControlPointDistance: 0.7`, **Then** curves become more pronounced
3. **Given** curved edges enabled, **When** user calls `setEdgeCurvatures(Float32Array)`, **Then** each edge uses its individual curvature value
4. **Given** per-edge curvature of 0, **When** rendered, **Then** that specific edge renders straight despite global curve setting
5. **Given** negative curvature value, **When** rendered, **Then** the curve bends in the opposite direction

---

### User Story 7 - Node Border Configuration (Priority: P3)

A user wants to customize node borders - turning them off entirely for a cleaner look, or adjusting thickness and color for emphasis. They configure borders globally and can override per node type.

**Why this priority**: Visual polish feature. Nodes already render acceptably; this enables customization.

**Independent Test**: Can be tested by toggling border visibility, adjusting thickness/color, and verifying render output.

**Acceptance Scenarios**:

1. **Given** default node rendering (with border), **When** user sets `nodeBorder: { enabled: false }`, **Then** nodes render without borders
2. **Given** borders enabled, **When** user sets `nodeBorder: { thickness: 2, color: 'white' }`, **Then** nodes have 2px white borders
3. **Given** global border settings, **When** user defines type-specific border settings, **Then** type settings override global
4. **Given** hover state, **When** node is hovered, **Then** border can display hover-specific styling

---

### User Story 8 - Topographical Contour Rendering (Priority: P3)

A user wants to visualize density gradients with contour lines, similar to elevation lines on a topographical map. When viewing a heatmap driven by value stream data, concentric contour lines appear around hot spots at configurable threshold levels.

**Why this priority**: Advanced visualization feature. The existing heatmap provides value; contours add interpretive depth.

**Independent Test**: Can be tested by enabling contours on a heatmap layer, configuring threshold levels, and verifying contour lines appear at correct density values.

**Acceptance Scenarios**:

1. **Given** a heatmap with density data, **When** user enables contours with `{ thresholds: [0.2, 0.4, 0.6, 0.8] }`, **Then** contour lines appear at those density levels
2. **Given** contours enabled, **When** user configures line color and thickness, **Then** contours render with those visual properties
3. **Given** multiple density peaks, **When** contours are rendered, **Then** each peak has its own set of concentric contour lines
4. **Given** contours enabled, **When** heatmap data changes, **Then** contour lines update to reflect new density distribution

---

### User Story 9 - Layer Visibility Toggles (Priority: P3)

A user wants quick toggles to show/hide individual visualization layers without reconfiguring them. They can rapidly compare views by toggling layers on and off.

**Why this priority**: Quality-of-life feature for the layer system. Fundamental to layer usability.

**Independent Test**: Can be tested by creating layers, toggling visibility, and verifying render state matches visibility settings.

**Acceptance Scenarios**:

1. **Given** a visible layer, **When** user calls `setLayerVisible('id', false)`, **Then** layer content is immediately hidden
2. **Given** a hidden layer, **When** user calls `setLayerVisible('id', true)`, **Then** layer content reappears
3. **Given** multiple layers, **When** user calls `getLayerVisibility()`, **Then** returns visibility state for all layers
4. **Given** layer visibility changed, **When** checked, **Then** layer configuration is preserved (just hidden, not destroyed)

---

### Edge Cases

- What happens when a value stream is defined but no data is pushed? (Display nothing for that stream)
- How does the system handle a node appearing in zero layers? (Node is not rendered)
- What happens when per-item arrays have wrong length? (Validation error with clear message)
- What happens when flow animation is enabled on a graph with no edges? (No-op, no errors)
- What if contour thresholds are outside the density range? (No contours drawn at invalid thresholds)
- How does layer z-ordering interact with the existing render order? (Layers render after base graph elements)
- What happens when stream values are outside the colorScale domain? (Clamp to domain min/max)

## Requirements *(mandatory)*

### Functional Requirements

**Per-Item Styling API**
- **FR-001**: System MUST provide `setNodeColors(Float32Array)` method accepting 4 values (RGBA) per node
- **FR-002**: System MUST provide `setNodeSizes(Float32Array)` method accepting 1 value per node
- **FR-003**: System MUST provide `setEdgeColors(Float32Array)` method accepting 4 values (RGBA) per edge
- **FR-004**: System MUST provide `setEdgeWidths(Float32Array)` method accepting 1 value per edge
- **FR-005**: System MUST fall back to global defaults when per-item values are undefined or NaN

**Value Stream System**
- **FR-006**: System MUST provide `defineValueStream(config)` method to create named value streams
- **FR-007**: Stream configuration MUST include: id (string), colorScale (domain/range for value-to-color mapping)
- **FR-008**: System MUST provide `setStreamValues(streamId, data)` method to push node-value pairs
- **FR-009**: System MUST support multiple active streams with configurable blend modes
- **FR-010**: System MUST NOT perform any hierarchical aggregation (user provides pre-computed values)

**Multi-Layer System**
- **FR-011**: System MUST provide `defineLayer(config)` method to create visualization layers
- **FR-012**: Layer configuration MUST include: id (string), nodeFilter (function), visualizations (array), zOrder (number)
- **FR-013**: System MUST render nodes matching layer filters with layer-specific visual treatments
- **FR-014**: System MUST allow a single node to appear in multiple layers simultaneously
- **FR-015**: System MUST render layers in ascending z-order

**Type-Based Styling**
- **FR-016**: System MUST provide `setNodeTypeStyles(styleMap)` method for type-based node styling
- **FR-017**: System MUST provide `setEdgeTypeStyles(styleMap)` method for type-based edge styling
- **FR-018**: Type styles MUST apply automatically based on node/edge `type` property
- **FR-019**: Per-item overrides MUST take precedence over type styles, which take precedence over global defaults

**Edge Flow Animation**
- **FR-020**: System MUST support two independent flow animation layers
- **FR-021**: Each flow layer MUST be configurable: enabled, speed, pulseWidth, pulseCount, waveShape, brightness, fade, color
- **FR-022**: Wave shapes MUST include: square (0), triangle (0.5), sine (1.0)
- **FR-023**: When both layers active, Layer 2 MUST visually "punch through" at pulse peaks

**Curved Edges**
- **FR-024**: System MUST provide global `curvedEdges` toggle
- **FR-025**: System MUST provide `curveSegments`, `curveWeight`, `curveControlPointDistance` configuration
- **FR-026**: System MUST provide `setEdgeCurvatures(Float32Array)` for per-edge curvature values
- **FR-027**: Curves MUST use conic (rational quadratic) Bezier interpolation

**Node Border Configuration**
- **FR-028**: System MUST provide `nodeBorder` configuration: enabled, thickness, color
- **FR-029**: Border configuration MUST be overridable per node type
- **FR-030**: Border MUST support hover and selection state styling

**Topographical Contours**
- **FR-031**: System MUST provide contour rendering option for heatmap/density layers
- **FR-032**: Contour configuration MUST include: thresholds (array of density levels), lineColor, lineThickness
- **FR-033**: Contours MUST be generated using marching squares algorithm on density field
- **FR-034**: Contours MUST update when underlying density data changes

**Layer Visibility**
- **FR-035**: System MUST provide `setLayerVisible(layerId, visible)` method
- **FR-036**: System MUST provide `getLayerVisibility()` method returning all layer states
- **FR-037**: Hidden layers MUST preserve their configuration for later re-enabling

### Key Entities

- **ValueStream**: A named data pipeline mapping arbitrary numeric values to visual heat/color. Contains: id, name, colorScale (domain/range), blend mode. No aggregation logic - accepts pre-computed values.
- **VisualizationLayer**: A filtered view of the graph with specific visual treatments. Contains: id, name, nodeFilter, edgeFilter, visualizations, zOrder, visibility state.
- **TypeStyleMap**: A mapping from type identifiers to visual configurations. Contains: type name -> { color, size, border, etc. }
- **FlowLayerConfig**: Configuration for a single edge flow animation layer. Contains: enabled, speed, pulseWidth, pulseCount, waveShape, brightness, fade, color.
- **ContourConfig**: Configuration for topographical contour rendering. Contains: enabled, thresholds, lineColor, lineThickness.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can apply per-item styling to graphs with 100,000+ nodes without perceptible lag (<100ms update time)
- **SC-002**: Value stream data updates reflect visually within one render frame
- **SC-003**: Layer visibility toggles respond instantly (<16ms to hide/show)
- **SC-004**: Edge flow animation maintains 60fps on graphs with 50,000 edges
- **SC-005**: Curved edge rendering maintains 60fps on graphs with 50,000 edges
- **SC-006**: Users can define and visualize 5+ value streams simultaneously
- **SC-007**: Contour generation completes within 50ms for typical density fields
- **SC-008**: Type-based styling reduces code needed by 80% compared to manual per-item arrays for type-uniform graphs
- **SC-009**: All styling configurations are fully serializable and restorable (save/load state)
- **SC-010**: Zero hardcoded data types, colors, or visual mappings - all configurable by library consumer

## Assumptions

- Nodes have an optional `type` property for type-based styling
- Hierarchical relationships are managed by the consuming application, not the library
- The existing heatmap infrastructure can be extended for value streams
- Performance targets assume modern GPU with WebGPU support
- Layer system operates at render level, not data level (same underlying graph data)

## Clarifications

### Session 2026-01-12

- Q: Should the library provide primitive heat/color assignment, leaving channel/aggregation logic to the consuming application? → A: Option B - Generic Value Streams. Library accepts multiple named value streams and blends them visually. NO built-in aggregation - user computes folder/parent sums in their application code and passes final values. Library is a "dumb pipe": values in → colors out → blend → render.

## Out of Scope

- Automatic diagnostic data extraction from source code (user provides data)
- Predefined diagnostic channel configurations (user defines their own)
- Built-in hierarchical aggregation (user computes aggregated values in application code)
- 3D visualization layers
- Animation interpolation between layer states
- Collaborative/multi-user layer configuration
