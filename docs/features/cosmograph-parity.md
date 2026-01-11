# Cosmograph Feature Parity Spec

This document tracks features from the Cosmograph fork that should be implemented in heroine-graph.

**Source:** `/Users/tom/Developer/projects/graph/`

---

## 1. Edge Flow Animation (Dual-Layer PWM)

**Status:** Partially implemented - needs enhancement

**Source files:**
- `/Users/tom/Developer/projects/graph/src/modules/Lines/draw-curve-line.frag` (lines 14-194)
- `/Users/tom/Developer/projects/graph/src/config.ts` (lines 315-412)
- `/Users/tom/Developer/projects/graph/src/stories/flow-demo/flow-demo.ts`

### Current heroine-graph state:
- Basic dual-layer flow implemented
- Missing: Full per-layer RGBA color controls, proper layer blending

### Required changes:

**Layer 1 (Primary) - Full parameters:**
```typescript
linkFlow: boolean                    // Enable Layer 1
linkFlowSpeed: number               // 0.01 - 2.0 (default: 0.5)
linkFlowPulseWidth: number          // 0.005 - 0.99 (default: 0.15)
linkFlowPulseCount: number          // 1 - 8 (default: 3)
linkFlowWaveShape: number           // 0=square, 0.5=triangle, 1.0=sine (default: 1.0)
linkFlowBrightness: number          // 1.0 - 5.0 (default: 1.5)
linkFlowFade: number                // 0 - 1 (default: 0.5)
linkFlowColor: [r,g,b,a]            // RGBA, alpha = blend amount (default: [1,1,1,0])
```

**Layer 2 (Sparks/Highlights) - Full parameters:**
```typescript
linkFlow2: boolean                  // Enable Layer 2
linkFlow2Speed: number              // (default: 1.0)
linkFlow2PulseWidth: number         // (default: 0.05)
linkFlow2PulseCount: number         // (default: 5)
linkFlow2WaveShape: number          // (default: 0.5 = triangle)
linkFlow2Brightness: number         // (default: 2.0)
linkFlow2Fade: number               // (default: 0.0)
linkFlow2Color: [r,g,b,a]           // (default: [1, 1, 0, 0.8] = yellow)
```

**Wave shape functions (from GLSL):**
```glsl
// Square: Sharp on/off - only middle 50% is "on"
float squareWave(float x) {
  return (x > 0.25 && x < 0.75) ? 1.0 : 0.0;
}

// Triangle: Linear ramp
float triangleWave(float x) {
  return 1.0 - abs(x * 2.0 - 1.0);
}

// Sine: Soft feathered edges with pronounced center peak (sin^4)
float sineWave(float x) {
  float s = sin(x * 3.14159265);
  return s * s * s * s;
}
```

**Layer combination logic:**
- When both layers enabled: Layer 2 "punches through" at pulse peaks
- `sparkPunch = flowValue2 * (1.0 - flow2Fade * 0.7)`
- Combined opacity: `max(layer1_opacity, sparkPunch)`
- Color blending uses smoothstep between layer contributions

### Presets:
1. **Particles**: Thin (0.015), fast (0.2), many (5), bright (2.5)
2. **Waves**: Wide (0.5), slow (0.08), single (1), subtle (1.5)
3. **Data Stream**: Medium (0.1), moderate (0.3), triangle, green tint
4. **Sparks**: Layer 2 only - Tiny (0.02), fast (1.2), square, very bright (4.0)
5. **Warning**: Medium (0.15), square, red tint
6. **Dual Layer**: Layer 1 slow red background + Layer 2 fast yellow sparks

---

## 2. Per-Item Styling API

**Status:** Not implemented

**Source files:**
- `/Users/tom/Developer/projects/graph/src/index.ts` (lines 375-579)

### Required API methods:

```typescript
// Per-node arrays (Float32Array, 4 values per node for colors)
setNodeColors(colors: Float32Array): void      // RGBA per node
setNodeSizes(sizes: Float32Array): void        // Size per node
setNodeShapes(shapes: Float32Array): void      // Shape ID per node

// Per-edge arrays
setEdgeColors(colors: Float32Array): void      // RGBA per edge
setEdgeWidths(widths: Float32Array): void      // Width per edge
setEdgeCurvatures(curvatures: Float32Array): void  // Curvature per edge
setEdgeArrows(arrows: boolean[]): void         // Arrow on/off per edge

// Per-node force multipliers
setNodeRepulsionMultipliers(multipliers: Float32Array): void
```

### Pattern for type-based configuration:
```typescript
// User provides type→value mapping
const typeColors = {
  file: [0.35, 0.65, 1.0, 1.0],      // Blue
  symbol: [0.5, 0.5, 0.5, 1.0],      // Gray
  class: [1.0, 0.6, 0.2, 1.0],       // Orange
};

// Build Float32Array based on node types
const colors = new Float32Array(nodeCount * 4);
for (let i = 0; i < nodeCount; i++) {
  const type = nodeTypes[i];
  const rgba = typeColors[type] || [1, 1, 1, 1];
  colors[i * 4] = rgba[0];
  colors[i * 4 + 1] = rgba[1];
  colors[i * 4 + 2] = rgba[2];
  colors[i * 4 + 3] = rgba[3];
}
graph.setNodeColors(colors);
```

---

## 3. Curved Edges

**Status:** Not implemented

**Source files:**
- `/Users/tom/Developer/projects/graph/src/modules/Lines/draw-curve-line.vert` (lines 45-49, 101-196)
- `/Users/tom/Developer/projects/graph/src/config.ts` (lines 283-300)

### Configuration:
```typescript
curvedLinks: boolean                          // Enable curves globally (default: false)
curvedLinkSegments: number                    // Tessellation (default: 19)
curvedLinkWeight: number                      // Rational curve weight (default: 0.8)
curvedLinkControlPointDistance: number        // 0-1, how far control point is (default: 0.5)
```

### Per-edge curvature:
```typescript
// 0 = straight, 0.25 = default curve, negative = opposite direction
setLinkCurvatures(curvatures: Float32Array): void
```

### Curve algorithm (Conic Bezier):
```glsl
vec2 conicParametricCurve(vec2 A, vec2 B, vec2 ControlPoint, float t, float w) {
  vec2 divident = (1.0 - t)² * A + 2.0 * (1.0 - t) * t * w * ControlPoint + t² * B;
  float divisor = (1.0 - t)² + 2.0 * (1.0 - t) * t * w + t²;
  return divident / divisor;
}

// Control point calculation:
vec2 xBasis = b - a;                          // Direction A→B
vec2 yBasis = normalize(perpendicular(xBasis)); // Normal
float linkDist = length(xBasis);
float h = curvature;                          // Per-link value
vec2 controlPoint = (a + b) / 2.0 + yBasis * linkDist * h;
```

---

## 4. Node Styling

**Status:** Partially implemented (basic colors work)

**Source files:**
- `/Users/tom/Developer/projects/graph/src/config.ts` (lines 117-215)
- `/Users/tom/Developer/projects/graph/src/modules/Points/`

### Required configuration:

**Global defaults:**
```typescript
pointDefaultColor: string | [r,g,b,a]         // Default node color
pointDefaultSize: number                      // Default radius (pixels)
pointOpacity: number                          // Universal opacity multiplier
pointGreyoutColor: string | [r,g,b,a]        // Color when deselected
pointGreyoutOpacity: number                   // Opacity when deselected
pointSizeScale: number                        // Size multiplier
scalePointsOnZoom: boolean                    // Scale with zoom
```

**Ring/border configuration:**
```typescript
renderHoveredPointRing: boolean               // Show ring on hover
hoveredPointRingColor: string | [r,g,b,a]    // Ring color
focusedPointRingColor: string | [r,g,b,a]    // Focused ring color
focusedPointIndex: number | undefined         // Which point to focus
```

**Current heroine-graph state:**
- Nodes have a hardcoded border
- Need: Border on/off toggle, thickness, color configuration

### Node shapes (enum):
```typescript
enum NodeShape {
  Circle = 0,      // Default
  Square = 1,
  Triangle = 2,
  Diamond = 3,
  Pentagon = 4,
  Hexagon = 5,
  Star = 6,
  Cross = 7,
  None = 8
}
```

---

## 5. Edge Styling

**Status:** Partially implemented

**Source files:**
- `/Users/tom/Developer/projects/graph/src/config.ts` (lines 220-428)

### Required configuration:

```typescript
// Basic styling
renderLinks: boolean                          // Show/hide edges
linkDefaultColor: string | [r,g,b,a]         // Default edge color
linkDefaultWidth: number                      // Default width (pixels)
linkOpacity: number                           // Universal opacity multiplier
linkWidthScale: number                        // Width multiplier
scaleLinksOnZoom: boolean                     // Scale with zoom

// Greyed out (when selection active)
linkGreyoutOpacity: number                    // (default: 0.1)

// Hover effects
hoveredLinkColor: string | [r,g,b,a]         // Color on hover
hoveredLinkWidthIncrease: number              // Extra pixels on hover (default: 5)

// Arrows
linkDefaultArrows: boolean                    // Show directional arrows
linkArrowsSizeScale: number                   // Arrow size multiplier

// Distance-based opacity
linkVisibilityDistanceRange: [min, max]       // Distance range (pixels)
linkVisibilityMinTransparency: number         // Min opacity at max distance
```

---

## 6. Heatmap/Density Color Configuration

**Status:** Basic implementation exists, needs enhancement

### Required:
- Custom color stops for heatmap gradient
- Dynamic color mapping based on node properties
- Per-layer color scheme selection

---

## Implementation Priority

1. **High Priority (Core UX):**
   - [ ] Fix edge flow to match dual-layer PWM exactly (Layer 2 punch-through)
   - [ ] Add full RGBA color pickers for flow layers
   - [ ] Add per-item styling API (setNodeColors, setEdgeColors, etc.)
   - [ ] Add node border configuration (on/off, thickness, color)

2. **Medium Priority (Enhanced Features):**
   - [ ] Curved edges with conic Bezier
   - [ ] Per-edge curvature array
   - [ ] Node shapes (SDF-based in fragment shader)
   - [ ] Distance-based edge opacity

3. **Lower Priority (Polish):**
   - [ ] Custom heatmap color gradients
   - [ ] Greyed out styling for deselection
   - [ ] Arrow rendering on edges
   - [ ] Hover effects for edges

---

## File Changes Required

### New files:
- `packages/core/src/renderer/shaders/curved_edge.vert.wgsl` - Bezier curve vertex shader
- `packages/core/src/renderer/shaders/node_shapes.wgsl` - SDF shape functions

### Modified files:
- `packages/core/src/renderer/shaders/edge.frag.wgsl` - Enhanced flow animation
- `packages/core/src/renderer/edge_flow.ts` - Full dual-layer config
- `packages/core/src/api/graph.ts` - Per-item styling API
- `packages/core/src/types.ts` - New config types
- `packages/core/src/renderer/pipelines/edges.ts` - Curved edge support
- `packages/core/src/renderer/pipelines/nodes.ts` - Border/shape config
- `examples/mission-control/index.html` - Enhanced controls
- `examples/mission-control/main.ts` - Control wiring
