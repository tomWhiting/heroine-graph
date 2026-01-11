# Data Model: Advanced Visualization Layer System

**Feature**: 002-visualization-layers
**Date**: 2026-01-12

## Entity Definitions

### DiagnosticChannel

A named data pipeline that maps arbitrary numeric values to visual heat/color for nodes.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique identifier for the channel |
| name | string | Human-readable display name |
| color | RGBA | Single color for heat visualization |
| colorScale | ColorScale | Optional gradient scale (overrides color) |
| aggregation | AggregationType | How to combine child values: 'sum' \| 'max' \| 'avg' \| 'min' |
| blendMode | BlendMode | How to combine with other channels: 'additive' \| 'multiply' \| 'overlay' |
| enabled | boolean | Whether channel is currently active |

**Relationships**:
- One channel → many ChannelDataPoints
- Multiple channels can be active simultaneously

**State Transitions**:
- Created → Active (when data pushed)
- Active → Disabled (via API)
- Disabled → Active (via API)
- Any → Removed (via removeChannel)

---

### ChannelDataPoint

A single data value for a node within a channel.

| Field | Type | Description |
|-------|------|-------------|
| nodeId | NodeId | The node this data applies to |
| value | number | The raw numeric value |
| aggregatedValue | number | Computed value after hierarchical aggregation |

**Relationships**:
- Belongs to one DiagnosticChannel
- References one Node

---

### VisualizationLayer

A filtered view of the graph with specific visual treatments.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique identifier for the layer |
| name | string | Human-readable display name |
| nodeFilter | FilterFunction | Predicate: (node) => boolean |
| edgeFilter | FilterFunction | Optional predicate: (edge) => boolean |
| visualizations | VisualizationType[] | What to render: nodes, edges, heatmap, contours, metaballs |
| zOrder | number | Render order (higher = on top) |
| visible | boolean | Current visibility state |
| styleOverrides | LayerStyles | Optional style overrides for this layer |

**Relationships**:
- One layer → many nodes (via filter)
- One node can appear in multiple layers
- Layers are independent (no parent-child)

**State Transitions**:
- Created → Visible
- Visible → Hidden (setLayerVisible false)
- Hidden → Visible (setLayerVisible true)
- Any → Removed (via removeLayer)

---

### TypeStyleMap

A mapping from node/edge type identifiers to visual styles.

| Field | Type | Description |
|-------|------|-------------|
| nodeStyles | Map<string, NodeStyle> | Type name → node visual style |
| edgeStyles | Map<string, EdgeStyle> | Type name → edge visual style |

**NodeStyle Fields**:
- color: RGBA
- size: number
- border: BorderConfig
- opacity: number

**EdgeStyle Fields**:
- color: RGBA
- width: number
- curvature: number
- opacity: number

---

### FlowLayerConfig

Configuration for a single edge flow animation layer.

| Field | Type | Description |
|-------|------|-------------|
| enabled | boolean | Whether this flow layer is active |
| speed | number | Animation speed (0.01-2.0) |
| pulseWidth | number | Width of each pulse (0.005-0.99) |
| pulseCount | number | Number of pulses visible (1-8) |
| waveShape | number | Shape: 0=square, 0.5=triangle, 1.0=sine |
| brightness | number | Intensity multiplier (1.0-5.0) |
| fade | number | Edge fade amount (0-1) |
| color | RGBA | Tint color with alpha as blend amount |

---

### ContourConfig

Configuration for topographical contour rendering.

| Field | Type | Description |
|-------|------|-------------|
| enabled | boolean | Whether contours are rendered |
| thresholds | number[] | Density levels for contour lines |
| lineColor | RGBA | Color for contour lines |
| lineThickness | number | Width of contour lines in pixels |

---

### CommunityBoundary

A geometric boundary around a group of nodes (for future integration with 003-graph-algorithms).

| Field | Type | Description |
|-------|------|-------------|
| communityId | string | Identifier of the community |
| hullType | 'convex' \| 'concave' | Type of hull used |
| vertices | Float32Array | Polygon vertices as [x0, y0, x1, y1, ...] |
| nodeIds | Set<NodeId> | Nodes contained in this boundary |

---

## GPU Buffer Layout

### Node Style Buffer

Per-node styling data uploaded to GPU.

```
Buffer: nodeStyleBuffer
Size: nodeCount * 8 floats (32 bytes per node)
Layout per node:
  [0-3]: color RGBA (4 floats)
  [4]:   size (1 float)
  [5-7]: reserved for future use
```

### Edge Style Buffer

Per-edge styling data uploaded to GPU.

```
Buffer: edgeStyleBuffer
Size: edgeCount * 8 floats (32 bytes per edge)
Layout per edge:
  [0-3]: color RGBA (4 floats)
  [4]:   width (1 float)
  [5]:   curvature (1 float)
  [6-7]: reserved for future use
```

### Channel Data Buffer

Diagnostic channel values per node.

```
Buffer: channelDataBuffer
Size: nodeCount * maxChannels floats
Layout: channelData[nodeIndex * maxChannels + channelIndex]
```

---

## Validation Rules

### Array Length Validation

| Method | Expected Length | Error Message |
|--------|-----------------|---------------|
| setNodeColors | nodeCount × 4 | "Expected {expected} values for {nodeCount} nodes (4 per node), got {actual}" |
| setNodeSizes | nodeCount × 1 | "Expected {expected} values for {nodeCount} nodes (1 per node), got {actual}" |
| setEdgeColors | edgeCount × 4 | "Expected {expected} values for {edgeCount} edges (4 per edge), got {actual}" |
| setEdgeWidths | edgeCount × 1 | "Expected {expected} values for {edgeCount} edges (1 per edge), got {actual}" |
| setEdgeCurvatures | edgeCount × 1 | "Expected {expected} values for {edgeCount} edges (1 per edge), got {actual}" |

### Channel Validation

- Channel ID must be unique
- Aggregation type must be one of: sum, max, avg, min
- Blend mode must be one of: additive, multiply, overlay

### Layer Validation

- Layer ID must be unique
- zOrder should be unique (warning if duplicate, uses definition order)
- At least one visualization type required
