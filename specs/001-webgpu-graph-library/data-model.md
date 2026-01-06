# Data Model: Heroine Graph

**Branch**: `001-webgpu-graph-library`
**Date**: 2026-01-06

This document defines the core entities, their attributes, relationships, and state
transitions for the Heroine Graph library.

---

## Entity Relationship Diagram

```
┌─────────────────┐         ┌─────────────────┐
│     Graph       │         │    Viewport     │
│─────────────────│         │─────────────────│
│ nodes[]         │◄────────│ graph           │
│ edges[]         │         │ position        │
│ simulation      │         │ scale           │
│ layers[]        │         │ dimensions      │
└────────┬────────┘         └─────────────────┘
         │
         │ contains
         ▼
┌─────────────────┐         ┌─────────────────┐
│      Node       │◄────────│      Edge       │
│─────────────────│ source/ │─────────────────│
│ id              │ target  │ id              │
│ position        │         │ source          │
│ velocity        │         │ target          │
│ metadata        │         │ metadata        │
│ state           │         │ state           │
└─────────────────┘         └─────────────────┘

┌─────────────────┐         ┌─────────────────┐
│   Simulation    │         │     Layer       │
│─────────────────│         │─────────────────│
│ status          │         │ type            │
│ parameters      │         │ visible         │
│ alpha           │         │ config          │
└─────────────────┘         └─────────────────┘
```

---

## Core Entities

### Graph

The top-level container for all graph data and configuration.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| id | string | Unique identifier | UUID format |
| nodes | Node[] | Array of graph vertices | Non-null |
| edges | Edge[] | Array of connections | Non-null |
| simulation | Simulation | Force simulation state | Non-null |
| layers | Layer[] | Visual effect layers | Non-null, ordered |
| config | GraphConfig | Graph-wide settings | Non-null |

**GraphConfig:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| nodeDefaultRadius | number | 5 | Default node visual size |
| nodeDefaultColor | Color | #6366f1 | Default node fill |
| edgeDefaultWidth | number | 1 | Default edge stroke width |
| edgeDefaultColor | Color | #94a3b8 | Default edge stroke |
| backgroundColor | Color | #ffffff | Canvas background |

---

### Node

A vertex in the graph representing a data point.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| id | NodeId (u32) | Stable unique identifier | Immutable after creation |
| x | f32 | X position in graph space | Any finite value |
| y | f32 | Y position in graph space | Any finite value |
| vx | f32 | X velocity | Internal, not user-facing |
| vy | f32 | Y velocity | Internal, not user-facing |
| metadata | NodeMetadata | User-defined attributes | Optional |
| state | NodeState | Interaction state | Non-null |

**NodeMetadata:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| label | string? | null | Text label for display |
| color | Color? | null | Override node color |
| radius | number? | null | Override node radius |
| group | string? | null | Grouping identifier |
| importance | number | 0.5 | Label priority (0-1) |
| data | Record<string, unknown> | {} | Arbitrary user data |

**NodeState:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| selected | boolean | false | Currently selected |
| hovered | boolean | false | Mouse is over node |
| pinned | boolean | false | Fixed position in simulation |
| hidden | boolean | false | Not rendered |

**State Transitions:**
```
         select()          deselect()
IDLE ──────────────► SELECTED ──────────────► IDLE
  │                     │
  │ hover()            │ drag()
  ▼                     ▼
HOVERED              DRAGGING
  │                     │
  │ unhover()          │ drop()
  ▼                     ▼
IDLE                 SELECTED (pinned=false)
                     or IDLE (if was not selected)
```

---

### Edge

A connection between two nodes.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| id | EdgeId (u32) | Stable unique identifier | Immutable after creation |
| source | NodeId | Source node identifier | Must exist in graph |
| target | NodeId | Target node identifier | Must exist in graph |
| metadata | EdgeMetadata | User-defined attributes | Optional |
| state | EdgeState | Interaction state | Non-null |

**EdgeMetadata:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| weight | number | 1.0 | Force simulation weight |
| color | Color? | null | Override edge color |
| width | number? | null | Override edge width |
| label | string? | null | Text label for display |
| directed | boolean | false | Show arrow indicator |
| data | Record<string, unknown> | {} | Arbitrary user data |

**EdgeState:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| selected | boolean | false | Currently selected |
| hovered | boolean | false | Mouse is over edge |
| hidden | boolean | false | Not rendered |

---

### Simulation

Force-directed layout engine state.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| status | SimulationStatus | Current running state | Enum |
| alpha | f32 | Current simulation energy | 0.0 - 1.0 |
| alphaTarget | f32 | Target alpha for decay | 0.0 - 1.0 |
| alphaDecay | f32 | Rate of energy loss | 0.0 - 1.0 |
| alphaMin | f32 | Stop threshold | 0.0 - 1.0 |
| velocityDecay | f32 | Velocity damping | 0.0 - 1.0 |
| forces | ForceConfig | Force parameters | Non-null |

**SimulationStatus (enum):**
- `stopped`: Not running, alpha = 0
- `running`: Actively simulating
- `paused`: Temporarily halted, alpha preserved
- `cooling`: Running but decaying toward stop

**ForceConfig:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| repulsion | number | -30 | Many-body repulsion strength |
| attraction | number | 1 | Link spring strength |
| gravity | number | 0.1 | Center pull strength |
| centerX | number | 0 | Gravity center X |
| centerY | number | 0 | Gravity center Y |
| linkDistance | number | 30 | Ideal link length |
| theta | number | 0.9 | Barnes-Hut approximation |

---

### Layer

Visual effect layer rendered on top of the graph.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| type | LayerType | Kind of visualization | Enum |
| visible | boolean | Currently rendered | Default: false |
| order | number | Render order (lower = behind) | Integer |
| config | LayerConfig | Type-specific settings | Depends on type |

**LayerType (enum):**
- `heatmap`: Density visualization
- `contour`: Iso-lines at density thresholds
- `metaball`: Smooth blob shapes around clusters
- `labels`: Node text labels

**HeatmapConfig:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| radius | number | 25 | Gaussian kernel radius |
| intensity | number | 1.0 | Color intensity multiplier |
| colorScale | ColorScale | viridis | Color gradient |
| opacity | number | 0.6 | Layer opacity |

**ContourConfig:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| thresholds | number[] | [0.2, 0.4, 0.6, 0.8] | Density values for iso-lines |
| strokeWidth | number | 1 | Line width |
| strokeColor | Color | #000000 | Line color |
| opacity | number | 0.8 | Layer opacity |

**MetaballConfig:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| threshold | number | 0.5 | SDF threshold for boundary |
| blendRadius | number | 20 | Smooth union radius |
| fillColor | Color | #6366f1 | Metaball fill |
| opacity | number | 0.3 | Layer opacity |

**LabelConfig:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| fontFamily | string | Inter | Font family name |
| fontSize | number | 12 | Base font size |
| fontColor | Color | #1f2937 | Text color |
| minZoom | number | 0.5 | Hide labels below this zoom |
| maxLabels | number | 1000 | Maximum visible labels |
| priority | 'importance' \| 'degree' | 'importance' | Label ranking |

---

### Viewport

Camera state for viewing the graph.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| x | f32 | Pan offset X (graph units) | Any finite value |
| y | f32 | Pan offset Y (graph units) | Any finite value |
| scale | f32 | Zoom level | 0.1 - 10.0 (configurable) |
| width | u32 | Canvas width (pixels) | > 0 |
| height | u32 | Canvas height (pixels) | > 0 |
| minScale | f32 | Minimum zoom | Default: 0.1 |
| maxScale | f32 | Maximum zoom | Default: 10.0 |

**Coordinate Transforms:**
```
Screen (pixels) ◄──────────────────► Graph (units)

screenToGraph(sx, sy):
  gx = (sx - width/2) / scale + x
  gy = (sy - height/2) / scale + y

graphToScreen(gx, gy):
  sx = (gx - x) * scale + width/2
  sy = (gy - y) * scale + height/2
```

---

## Type Aliases

```typescript
// Identifiers
type NodeId = number;  // u32, stable across removals
type EdgeId = number;  // u32, stable across removals

// Colors
type Color = string;   // CSS color string: #rrggbb, rgb(), hsl()
type ColorScale = 'viridis' | 'plasma' | 'inferno' | 'magma' | 'turbo' | Color[];

// Vectors
interface Vec2 {
  x: number;
  y: number;
}
```

---

## Validation Rules

### Node Validation

1. `id` MUST be unique within the graph
2. `x` and `y` MUST be finite numbers (not NaN, not Infinity)
3. If `metadata.importance` provided, MUST be in range [0, 1]
4. If `metadata.radius` provided, MUST be > 0

### Edge Validation

1. `id` MUST be unique within the graph
2. `source` MUST reference an existing node
3. `target` MUST reference an existing node
4. Self-loops (`source === target`) ARE allowed
5. Multiple edges between same nodes ARE allowed (multigraph)
6. If `metadata.weight` provided, MUST be finite

### Simulation Validation

1. `alpha`, `alphaTarget`, `alphaDecay`, `alphaMin` MUST be in [0, 1]
2. `velocityDecay` MUST be in [0, 1]
3. `forces.theta` MUST be in (0, 2] (typically 0.5-1.5)

### Layer Validation

1. `order` MUST be unique among visible layers
2. `config` MUST match the structure for the given `type`
3. `heatmap.radius` MUST be > 0
4. `contour.thresholds` MUST be non-empty array of numbers in (0, 1)

---

## Data Format (Input)

### Graph Input Format

```typescript
interface GraphInput {
  nodes: NodeInput[];
  edges: EdgeInput[];
}

interface NodeInput {
  id: string | number;       // Converted to NodeId
  x?: number;                // Initial position (random if omitted)
  y?: number;
  label?: string;
  color?: string;
  radius?: number;
  group?: string;
  importance?: number;
  [key: string]: unknown;    // Additional metadata
}

interface EdgeInput {
  source: string | number;   // Node id reference
  target: string | number;
  weight?: number;
  color?: string;
  width?: number;
  directed?: boolean;
  [key: string]: unknown;    // Additional metadata
}
```

### Typed Array Format (High Performance)

For graphs with 100K+ nodes, use typed arrays:

```typescript
interface GraphTypedInput {
  nodeCount: number;
  edgeCount: number;
  positions?: Float32Array;   // [x0, y0, x1, y1, ...] length = nodeCount * 2
  edges: Uint32Array;         // [src0, tgt0, src1, tgt1, ...] length = edgeCount * 2
  nodeMetadata?: NodeMetadata[];
  edgeMetadata?: EdgeMetadata[];
}
```

---

## GPU Buffer Layout

### Position Buffer (SoA)

```
Buffer: positions_x (Float32Array, nodeCount elements)
Index:  [  0  ][  1  ][  2  ] ... [n-1]
Value:  [ x_0 ][ x_1 ][ x_2 ] ... [x_n-1]

Buffer: positions_y (Float32Array, nodeCount elements)
Index:  [  0  ][  1  ][  2  ] ... [n-1]
Value:  [ y_0 ][ y_1 ][ y_2 ] ... [y_n-1]
```

### Edge Buffer (CSR Format)

```
Buffer: edge_offsets (Uint32Array, nodeCount + 1 elements)
Index:  [  0  ][  1  ][  2  ] ... [n-1][ n ]
Value:  [  0  ][ o_1 ][ o_2 ] ... [o_n-1][edgeCount]

Buffer: edge_targets (Uint32Array, edgeCount elements)
// For node i, neighbors are targets[offsets[i]..offsets[i+1]]
```

### Visual Attributes (Interleaved for Rendering)

```
Buffer: node_attributes (Float32Array, nodeCount * 6 elements)
Per-node: [radius, r, g, b, selected, hovered]
```
