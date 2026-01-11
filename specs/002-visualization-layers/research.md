# Research: Advanced Visualization Layer System

**Feature**: 002-visualization-layers
**Date**: 2026-01-12

## R1: Dual-Layer PWM Edge Flow

### Question
How does Cosmograph implement the dual-layer PWM edge flow animation with punch-through effect?

### Findings

**Source**: `/Users/tom/Developer/projects/graph/src/modules/Lines/draw-curve-line.frag` (lines 14-194)

The Cosmograph implementation uses two independent flow animation layers:

**Layer 1 (Primary/Base Wave)**:
- `linkFlow`: boolean enable
- `linkFlowSpeed`: 0.01-2.0 (default 0.5)
- `linkFlowPulseWidth`: 0.005-0.99 (default 0.15)
- `linkFlowPulseCount`: 1-8 (default 3)
- `linkFlowWaveShape`: 0=square, 0.5=triangle, 1.0=sine (default 1.0)
- `linkFlowBrightness`: 1.0-5.0 (default 1.5)
- `linkFlowFade`: 0-1 (default 0.5)
- `linkFlowColor`: RGBA with alpha as blend amount (default [1,1,1,0])

**Layer 2 (Sparks/Highlights)**:
- Same parameters with `linkFlow2` prefix
- Defaults tuned for sparks: faster speed (1.0), narrower pulse (0.05), triangle wave

**Wave Shape Functions** (GLSL):
```glsl
float squareWave(float x) {
    return (x > 0.25 && x < 0.75) ? 1.0 : 0.0;
}

float triangleWave(float x) {
    return 1.0 - abs(x * 2.0 - 1.0);
}

float sineWave(float x) {
    float s = sin(x * 3.14159265);
    return s * s * s * s;  // sin^4 for pronounced peak
}
```

**Punch-Through Logic**:
```glsl
float sparkPunch = flowValue2 * (1.0 - flow2Fade * 0.7);
float combinedOpacity = max(layer1_opacity, sparkPunch);
```

### Decision
Port the exact Cosmograph implementation to WGSL, adapting syntax but preserving all parameters and logic.

### Alternatives Considered
- Simplified single-layer: Rejected - loses visual richness
- Custom wave shapes: Deferred - can add later via customization

---

## R2: Conic Bezier Curves for Edges

### Question
What curve algorithm should we use for curved edges?

### Findings

**Source**: `/Users/tom/Developer/projects/graph/src/modules/Lines/draw-curve-line.vert` (lines 101-196)

Cosmograph uses rational quadratic (conic) Bezier curves:

**Algorithm**:
```glsl
vec2 conicParametricCurve(vec2 A, vec2 B, vec2 ControlPoint, float t, float w) {
    vec2 dividend = (1.0 - t) * (1.0 - t) * A +
                    2.0 * (1.0 - t) * t * w * ControlPoint +
                    t * t * B;
    float divisor = (1.0 - t) * (1.0 - t) +
                    2.0 * (1.0 - t) * t * w +
                    t * t;
    return dividend / divisor;
}
```

**Control Point Calculation**:
```glsl
vec2 xBasis = b - a;                              // Direction A→B
vec2 yBasis = normalize(perpendicular(xBasis));   // Normal vector
float linkDist = length(xBasis);
float h = curvature;                              // Per-link value
vec2 controlPoint = (a + b) / 2.0 + yBasis * linkDist * h;
```

**Configuration**:
- `curvedLinks`: boolean toggle
- `curvedLinkSegments`: tessellation count (default 19)
- `curvedLinkWeight`: rational curve weight (default 0.8)
- `curvedLinkControlPointDistance`: 0-1 (default 0.5)
- Per-edge curvature via `setLinkCurvatures(Float32Array)`

### Decision
Use conic Bezier with same parameters. Tessellation in vertex shader with configurable segment count.

### Alternatives Considered
- Cubic Bezier: More control points but harder to configure
- Quadratic Bezier: Simpler but less control over curve shape
- Catmull-Rom splines: Better for multi-point paths, overkill for A→B edges

---

## R3: Per-Item Styling API

### Question
What's the best API pattern for per-item (node/edge) styling?

### Findings

**Source**: `/Users/tom/Developer/projects/graph/src/index.ts` (lines 375-579)

Cosmograph uses Float32Array-based APIs:

```typescript
// Per-node
setPointColors(colors: Float32Array): void      // 4 values per node (RGBA)
setPointSizes(sizes: Float32Array): void        // 1 value per node

// Per-edge
setLinkColors(colors: Float32Array): void       // 4 values per edge (RGBA)
setLinkWidths(widths: Float32Array): void       // 1 value per edge
setLinkCurvatures(curvatures: Float32Array): void // 1 value per edge
```

**Type-Based Pattern**:
```typescript
const typeColors = {
    file: [0.35, 0.65, 1.0, 1.0],
    folder: [1.0, 0.6, 0.2, 1.0],
};

// Library builds Float32Array internally
const colors = new Float32Array(nodeCount * 4);
for (let i = 0; i < nodeCount; i++) {
    const type = nodeTypes[i];
    const rgba = typeColors[type] || defaultColor;
    colors.set(rgba, i * 4);
}
```

### Decision
- Primary API: Float32Array for power users
- Convenience API: Type-based styling that builds arrays automatically
- Precedence: per-item > type > global defaults
- Validation: throw on wrong array length with clear error message

### Alternatives Considered
- Object-based API: Easier but slower for large graphs
- Map<nodeId, color>: Flexible but O(n) lookup during render

---

## R4: Diagnostic Channel System

### Question
How should diagnostic data channels work for heat visualization?

### Findings

No direct Cosmograph reference - this is new functionality based on user requirements.

**Design based on user description**:
- Channels are user-defined (no hardcoded "error"/"warning")
- Each channel has: id, color/colorScale, aggregation mode
- Data pushed as node-value pairs
- Hierarchical aggregation for parent nodes

**Aggregation Modes**:
- `sum`: Add child values (error counts)
- `max`: Take maximum (severity levels)
- `avg`: Average of children (percentages)
- `min`: Take minimum (confidence scores)

**Cycle Handling**:
- Detect cycles during aggregation
- If cycle found, use node's direct value only
- Log warning, don't error

**Multi-Channel Blending**:
- `additive`: Colors add (up to white)
- `multiply`: Colors multiply (darken)
- `overlay`: Photoshop-style overlay

### Decision
Implement flexible channel system with all aggregation modes. Cycle detection is required.

---

## R5: Marching Squares

### Question
How should contour lines be generated from density fields?

### Findings

Standard marching squares algorithm:
1. Sample density field at grid points
2. For each cell, determine which corners are above threshold
3. 16 possible configurations → lookup table for line segments
4. Connect segments into contour lines

**GPU Implementation**:
- Compute shader iterates over grid cells
- Output line segments to storage buffer
- Render segments via line primitive

**Configuration**:
- `thresholds`: array of density levels [0.2, 0.4, 0.6, 0.8]
- `lineColor`: RGBA per threshold level
- `lineThickness`: width in pixels

### Decision
Implement in compute shader for GPU parallelism. Output to vertex buffer for line rendering.

---

## R6: Multi-Layer Architecture

### Question
How should the multi-layer system work?

### Findings

Design based on user requirements and standard layer systems:

**Layer Definition**:
```typescript
interface VisualizationLayer {
    id: string;
    name?: string;
    nodeFilter: (node: NodeData) => boolean;
    edgeFilter?: (edge: EdgeData) => boolean;
    visualizations: ('nodes' | 'edges' | 'heatmap' | 'contours' | 'metaballs')[];
    zOrder: number;
    visible: boolean;
    styleOverrides?: Partial<LayerStyles>;
}
```

**Rendering Order**:
1. Sort layers by zOrder (ascending)
2. For each visible layer:
   - Apply node filter to get visible nodes
   - Apply edge filter to get visible edges
   - Render requested visualizations with layer styles

**Visibility Toggle**:
- `setLayerVisible(id, visible)` - immediate effect
- Configuration preserved when hidden

### Decision
Implement layer manager that coordinates filtered rendering with zOrder support.
