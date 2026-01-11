# Implementation Plan: Advanced Visualization Layer System

**Branch**: `002-visualization-layers` | **Date**: 2026-01-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-visualization-layers/spec.md`

## Summary

Build a comprehensive visualization enhancement system for heroine-graph that adds configurable value streams for data-driven heat visualization (streams feed layers like heatmap/contours/metaballs - no library-side aggregation), a multi-layer rendering system where nodes can appear on multiple layers with different visual treatments, per-node-type and per-edge-type styling with global defaults and overrides, dual-layer PWM edge flow animation matching Cosmograph, curved edges using conic Bezier curves, node border configuration, per-item styling API via Float32Arrays, and layer visibility toggles.

## Technical Context

**Language/Version**: TypeScript 5.x (Deno 2.x), WGSL shaders (WebGPU)
**Primary Dependencies**: WebGPU API, d3-scale/d3-color (TS), existing heroine-graph core
**Storage**: N/A (in-memory GPU buffers only)
**Testing**: Deno test, visual regression tests via Mission Control example
**Target Platform**: Modern browsers with WebGPU support (Chrome 113+, Firefox 141+, Safari 26+, Edge 113+)
**Project Type**: Monorepo - extending existing `packages/core` library
**Performance Goals**: 60fps @ 100K nodes, <100ms per-item styling updates, <16ms layer visibility toggle
**Constraints**: <500 bytes/node additional memory, maintain existing API compatibility
**Scale/Scope**: Up to 1,000,000 nodes with per-item styling, 5+ simultaneous diagnostic channels

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Consistency Is On Us | PASS | Type styles, per-item overrides, and defaults have clear precedence rules (FR-019) |
| II. Contract, Not Coercion | PASS | Users define their own channels, colors, styles - no hardcoded mappings (SC-010) |
| III. Trust Users, Don't Give Guns | PASS | Wrong-length arrays get validation errors (Edge Case), not silent failures |
| IV. Expose All Controls, Make Defaults Excellent | PASS | Every parameter configurable, defaults work beautifully out of box |
| V. No Silent Failures | PASS | Validation errors with clear messages for wrong array lengths |
| VI. Automation Over Gatekeeping | PASS | Event-driven aggregation, layer filters - no hard limits |
| VII. Low-Level Primitives Over Wrappers | PASS | Float32Array API, raw GPU buffer access, explicit layer configuration |
| VIII. Circular Dependencies Are Real | PASS | Aggregation detects cycles, uses direct value only (Edge Case) |
| IX. Make It Easy to Have Fun | PASS | Type-based styling reduces code by 80% (SC-008) while keeping power |
| X. Build With Love | PASS | Comprehensive feature set with attention to edge cases |

**Gate Status**: PASSED - No constitution violations

## Project Structure

### Documentation (this feature)

```text
specs/002-visualization-layers/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (TypeScript interfaces)
├── checklists/
│   └── requirements.md  # Spec validation checklist
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
packages/core/
├── mod.ts                          # Public API exports (add new types)
├── src/
│   ├── api/
│   │   └── graph.ts                # HeroineGraph class (add styling/channel/layer APIs)
│   ├── types.ts                    # Type definitions (add new config types)
│   ├── renderer/
│   │   ├── edge_flow.ts            # Edge flow config (enhance dual-layer PWM)
│   │   ├── buffers/
│   │   │   ├── node_style.ts       # NEW: Per-node color/size buffers
│   │   │   └── edge_style.ts       # NEW: Per-edge color/width/curvature buffers
│   │   ├── pipelines/
│   │   │   ├── nodes.ts            # Node pipeline (add border config, per-item styling)
│   │   │   ├── edges.ts            # Edge pipeline (add curvature, per-item styling)
│   │   │   └── contours.ts         # NEW: Marching squares contour pipeline
│   │   └── shaders/
│   │       ├── node.vert.wgsl      # Node vertex (add border uniforms)
│   │       ├── node.frag.wgsl      # Node fragment (add border rendering)
│   │       ├── edge.vert.wgsl      # Edge vertex (add curve support)
│   │       ├── edge.frag.wgsl      # Edge fragment (enhance flow animation)
│   │       ├── curved_edge.wgsl    # NEW: Conic Bezier curve shader
│   │       └── contour.wgsl        # NEW: Marching squares shader
│   ├── layers/
│   │   ├── mod.ts                  # Layer system exports
│   │   ├── layer_manager.ts        # Multi-layer rendering coordinator
│   │   ├── visualization_layer.ts  # Layer definition and filtering
│   │   └── heatmap/                # Existing heatmap (extend with stream data source)
│   ├── streams/
│   │   ├── mod.ts                  # Stream system exports
│   │   ├── types.ts                # ValueStreamConfig, ColorStop, BlendMode
│   │   ├── value_stream.ts         # Single stream: stores node→value mappings
│   │   └── stream_manager.ts       # Manages multiple streams, blending
│   └── styling/
│       ├── mod.ts                  # NEW: Styling system exports
│       ├── type_styles.ts          # NEW: Type-based style resolution
│       ├── style_resolver.ts       # NEW: Precedence: per-item > type > global
│       └── color_scales.ts         # NEW: Color scale utilities for channels
│
├── examples/mission-control/
│   ├── index.html                  # Add new control panels
│   └── main.ts                     # Wire up new features

tests/
└── visualization/
    ├── value_stream.test.ts        # NEW: Stream data storage tests (no aggregation)
    ├── stream_to_layer.test.ts     # NEW: Stream → layer binding tests
    ├── layer_system.test.ts        # NEW: Layer filtering tests
    ├── type_styling.test.ts        # NEW: Style precedence tests
    └── contour.test.ts             # NEW: Marching squares tests
```

**Structure Decision**: Extend existing monorepo structure. New features organized into:
- `renderer/buffers/` - GPU buffer management for per-item data
- `layers/` - Layer and channel system (new directory)
- `styling/` - Style resolution logic (new directory)

## Complexity Tracking

> No constitution violations requiring justification.

## Phase 0: Research Findings

### R1: Dual-Layer PWM Edge Flow (Cosmograph Reference)

**Decision**: Port exact Cosmograph implementation with WGSL adaptations

**Rationale**: The Cosmograph fork has a proven, visually appealing implementation that the user has validated. Direct port ensures feature parity.

**Key Implementation Details**:
- Wave shape functions: square (hard on/off), triangle (linear ramp), sine (sin^4 for soft peak)
- Layer combination: `sparkPunch = flowValue2 * (1.0 - flow2Fade * 0.7)`
- Combined opacity: `max(layer1_opacity, sparkPunch)` for punch-through effect
- Per-layer RGBA color with alpha as blend amount

**Source Reference**: `/Users/tom/Developer/projects/graph/src/modules/Lines/draw-curve-line.frag` lines 14-194

### R2: Conic Bezier Curves for Edges

**Decision**: Use rational quadratic (conic) Bezier interpolation with configurable control point distance

**Rationale**: Conic curves provide smooth, visually pleasing curves with a single control point. The Cosmograph implementation is proven.

**Key Implementation Details**:
```wgsl
fn conicParametricCurve(A: vec2f, B: vec2f, CP: vec2f, t: f32, w: f32) -> vec2f {
    let dividend = (1.0 - t) * (1.0 - t) * A + 2.0 * (1.0 - t) * t * w * CP + t * t * B;
    let divisor = (1.0 - t) * (1.0 - t) + 2.0 * (1.0 - t) * t * w + t * t;
    return dividend / divisor;
}
```
- Control point: midpoint + perpendicular offset based on curvature
- Per-edge curvature via Float32Array for individual overrides
- Negative curvature = opposite direction bend

**Source Reference**: `/Users/tom/Developer/projects/graph/src/modules/Lines/draw-curve-line.vert` lines 101-196

### R3: Per-Item Styling API Pattern

**Decision**: Use Float32Array with 4 values per node (RGBA) / 1 value per edge (width) pattern

**Rationale**: GPU-friendly, matches WebGPU buffer expectations, proven in Cosmograph

**Key Implementation Details**:
- `setNodeColors(Float32Array)` - 4 floats per node (RGBA)
- `setNodeSizes(Float32Array)` - 1 float per node
- `setEdgeColors(Float32Array)` - 4 floats per edge (RGBA)
- `setEdgeWidths(Float32Array)` - 1 float per edge
- `setEdgeCurvatures(Float32Array)` - 1 float per edge
- Validation: array length must match node/edge count × values per item
- Fallback: NaN or undefined values use global defaults

### R4: Value Stream System (No Aggregation)

**Decision**: Value streams are a "dumb pipe" - store node-to-value mappings that visualization layers consume. NO library-side aggregation.

**Rationale**: Per spec clarification (Session 2026-01-12): "Library is a 'dumb pipe': values in → colors out → blend → render." User computes aggregates in their application code.

**Key Implementation Details**:
- Stream definition: `{ id, colorScale: { domain, stops }, blendMode, opacity }`
- Data push: `setStreamValues(streamId, [{nodeIndex, value}])` - pre-computed values only
- Streams feed INTO layers (heatmap, contours, metaballs) as data sources
- Layer renders heat/intensity at node positions based on stream values
- Multiple streams blend via configurable blend modes (additive, multiply, max, replace)
- NO hierarchical propagation - user handles folder/parent aggregation externally

### R5: Marching Squares for Contours

**Decision**: Implement marching squares on density field texture in compute shader

**Rationale**: Standard algorithm for contour generation, GPU-parallelizable

**Key Implementation Details**:
- Input: density field texture (from heatmap layer)
- Configurable threshold levels: e.g., [0.2, 0.4, 0.6, 0.8]
- Output: line segments for each threshold level
- Line rendering via existing edge pipeline with custom styling
- Update on density change via dirty flag

### R6: Multi-Layer Rendering Architecture

**Decision**: Layers as filtered views with independent render passes

**Rationale**: Same underlying graph data, different visual treatments per layer

**Key Implementation Details**:
- Layer definition: `{ id, nodeFilter, edgeFilter, visualizations, zOrder, visible }`
- Node filter: `(node) => boolean` - determines which nodes appear
- Visualizations: array of ['nodes', 'edges', 'heatmap', 'contours', 'metaballs']
- Layers render in zOrder (ascending)
- Visibility toggle: immediate show/hide without reconfiguration
- Node can appear in multiple layers with different treatments

## Phase 1: Architecture & Contracts

### Data Model

See [data-model.md](./data-model.md) for entity definitions.

### API Contracts

See [contracts/](./contracts/) directory for TypeScript interfaces.

### Integration Points

1. **HeroineGraph API** (`api/graph.ts`):
   - Add `setNodeColors()`, `setNodeSizes()`, `setEdgeColors()`, `setEdgeWidths()`, `setEdgeCurvatures()`
   - Add `defineValueStream()`, `setStreamValues()`, `removeValueStream()` (no aggregation)
   - Add `defineLayer()`, `setLayerVisible()`, `getLayerVisibility()`, `removeLayer()`
   - Add `setNodeTypeStyles()`, `setEdgeTypeStyles()`
   - Add `setNodeBorder()` configuration
   - Layers can bind to streams: `setHeatmapDataSource(streamId)`, `setContourDataSource(streamId)`

2. **Render Loop** (`renderer/render_loop.ts`):
   - Layer-aware rendering: iterate layers in zOrder
   - Per-layer bind groups for style buffers
   - Contour pass after heatmap when enabled

3. **GPU Buffers**:
   - New: `nodeColorBuffer`, `nodeSizeBuffer` (per-item styling)
   - New: `edgeColorBuffer`, `edgeWidthBuffer`, `edgeCurvatureBuffer`
   - Existing: extend uniform buffers for flow animation
   - Streams: CPU-side storage (node-to-value maps), converted to density when layers render
   - Layers (heatmap/contour): generate their own GPU textures from stream data + node positions

## Implementation Phases

### Phase 1: Per-Item Styling Foundation (P1)
- Implement `setNodeColors()`, `setNodeSizes()` API
- Implement `setEdgeColors()`, `setEdgeWidths()` API
- Create style buffers and bind groups
- Update node/edge shaders to read per-item data
- Add validation for array lengths

### Phase 2: Type-Based Styling (P2)
- Implement `setNodeTypeStyles()`, `setEdgeTypeStyles()` API
- Create style resolver with precedence logic
- Build Float32Arrays from type mappings automatically

### Phase 3: Value Stream System (P1)
- Implement `defineValueStream()`, `setStreamValues()` API
- ValueStream class stores node-to-value mappings (no aggregation)
- StreamManager handles multiple streams with blending
- Connect streams to layers (heatmap, contours, metaballs) as data sources
- Layers render heat/intensity at node positions based on stream values

### Phase 4: Dual-Layer PWM Edge Flow (P2)
- Enhance edge fragment shader with full Cosmograph PWM
- Add Layer 2 parameters and punch-through logic
- Add per-layer RGBA color configuration
- Update flow uniforms and bind groups

### Phase 5: Curved Edges (P3)
- Implement conic Bezier vertex shader
- Add `curvedEdges` toggle and configuration
- Implement `setEdgeCurvatures()` for per-edge control
- Tessellation with configurable segment count

### Phase 6: Node Border Configuration (P3)
- Add border uniforms (enabled, thickness, color)
- Update node fragment shader for border rendering
- Support per-type border overrides
- Hover/selection state styling

### Phase 7: Multi-Layer System (P2)
- Implement LayerManager for layer orchestration
- Implement VisualizationLayer with node/edge filters
- Layer-aware render pass organization
- zOrder sorting and visibility toggles

### Phase 8: Topographical Contours (P3)
- Implement marching squares compute shader
- Contour line rendering pipeline
- Threshold configuration
- Integration with heatmap density field

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Per-item buffer updates slow | Use staging buffers, batch updates |
| Too many layers hurt performance | Lazy rendering, frustum culling per layer |
| Curved edges expensive | LOD tessellation based on zoom level |
| Contour generation slow | Compute shader, dirty flag for updates |

## Definition of Done

- [ ] All 37 functional requirements implemented
- [ ] All 10 success criteria measurable and met
- [ ] All 9 user stories have passing acceptance scenarios
- [ ] Mission Control example demonstrates all features
- [ ] Performance targets met at specified node counts
- [ ] No TypeScript/WGSL compilation warnings
- [ ] Tests pass for all new functionality
