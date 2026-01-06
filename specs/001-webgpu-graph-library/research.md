# Research: Heroine Graph - WebGPU Graph Visualization Library

**Branch**: `001-webgpu-graph-library`
**Date**: 2026-01-06
**Status**: Complete

This document consolidates research findings for key technical decisions.

---

## 1. WebGPU Force Simulation Architecture

### Decision: Compute Shaders with Barnes-Hut, Bottom-Up Quadtree, SoA Memory Layout

### Rationale

The force simulation must handle 1M+ nodes at 30fps. The Barnes-Hut algorithm reduces O(n²)
repulsive force calculation to O(n log n) by approximating distant clusters. WebGPU compute
shaders provide the parallel execution model required.

### Implementation Architecture

**Compute Pipeline Stages (per frame):**

1. **Morton Code Generation** (~0.02ms for 12K objects)
   - Convert 2D positions to spatial keys via bit interleaving
   - Enables O(1) sibling detection in tree construction

2. **Parallel Radix Sort** (~0.18ms)
   - Sort nodes by Morton code for spatial locality
   - Use existing WebGPU radix sort implementations

3. **Bottom-Up Quadtree Construction** (~0.02ms)
   - Create leaf nodes at maximum depth
   - Parallel reduction builds internal nodes
   - Achieves 75% GPU occupancy vs 0.006% for top-down

4. **Bounding Box Calculation** (~0.06ms)
   - Walk from leaf toward root computing bounds
   - Use atomic operations for synchronization

5. **Barnes-Hut Force Traversal** (~0.25ms)
   - One thread per node
   - Stack-based tree traversal
   - θ (theta) parameter controls accuracy/speed tradeoff

6. **Attractive Force Calculation**
   - Process edges in CSR format
   - Spring forces between connected nodes

7. **Integration Step**
   - Update velocities with damping
   - Update positions
   - Swap ping-pong buffers

### Buffer Layout

```wgsl
// Position buffers (ping-pong pair) - SoA for coalesced access
@group(0) @binding(0) var<storage, read> pos_x_in: array<f32>;
@group(0) @binding(1) var<storage, read> pos_y_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos_x_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> pos_y_out: array<f32>;

// Velocity buffers (ping-pong pair)
@group(0) @binding(4) var<storage, read> vel_x_in: array<f32>;
@group(0) @binding(5) var<storage, read> vel_y_in: array<f32>;
@group(0) @binding(6) var<storage, read_write> vel_x_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> vel_y_out: array<f32>;

// Force accumulation (quantized for atomic ops)
@group(0) @binding(8) var<storage, read_write> force_x: array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> force_y: array<atomic<i32>>;

// Edge list (CSR format)
@group(1) @binding(0) var<storage, read> edge_offsets: array<u32>;
@group(1) @binding(1) var<storage, read> edge_targets: array<u32>;

// Quadtree (6 * nodeCount nodes)
struct QuadtreeNode {
    boundary: vec4<f32>,     // x_min, y_min, x_max, y_max
    center_of_mass: vec2<f32>,
    mass: f32,
    children: vec4<u32>,     // indices of 4 children
}
@group(2) @binding(0) var<storage, read_write> quadtree: array<QuadtreeNode>;
```

**Critical: Avoid vec3 in arrays** - requires 16-byte alignment, wastes 25% memory.
Use separate arrays or vec2/vec4.

### Atomic Float Workaround

WGSL only supports `atomic<i32>` and `atomic<u32>`. For force accumulation:

```wgsl
const QUANTIZE_FACTOR: f32 = 32768.0;

fn accumulate_force(idx: u32, force: vec2<f32>) {
    atomicAdd(&force_x[idx], i32(force.x * QUANTIZE_FACTOR));
    atomicAdd(&force_y[idx], i32(force.y * QUANTIZE_FACTOR));
}
```

### Alternatives Considered

| Approach | Pros | Cons |
|----------|------|------|
| Top-down quadtree | Simpler logic | No recursion in WGSL, 50x slower |
| Linear quadtree (unsorted) | No sorting pass | Poor cache locality, 3x slower |
| Naive O(n²) | Simple | Impractical beyond 10K nodes |
| CPU simulation | Easier to debug | Can't meet performance targets |

### Sources

- GraphWaGu: First WebGPU graph visualization (github.com/harp-lab/GraphWaGu)
- WebGPU Radix Sort (github.com/kishimisu/WebGPU-Radix-Sort)
- NVIDIA GPU Tree Construction (developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu)
- cosmos.gl architecture (github.com/cosmosgl/cosmos)

---

## 2. Text Rendering (Labels)

### Decision: Multi-channel SDF (MSDF) with Hierarchical Culling

### Rationale

Labels must remain sharp at 10%-1000% zoom (SC-005). SDF text rendering scales infinitely
without generating new textures. MSDF preserves sharp corners that single-channel SDF loses.

### Font Atlas Generation

Use msdf-atlas-gen at build time:

```bash
msdf-atlas-gen -font fonts/Inter.ttf -type msdf -size 48 -pxrange 4 \
  -imageout assets/font-atlas.png -json assets/font-atlas.json
```

**Parameters:**
- `-size 48`: Sufficient for 10x zoom
- `-pxrange 4`: Distance field range, must match shader
- `-type msdf`: Multi-channel for sharp corners

### MSDF Fragment Shader

```wgsl
fn median(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

@fragment
fn label_fragment(input: VertexOutput) -> @location(0) vec4f {
    let msdf = textureSample(font_atlas, font_sampler, input.uv);
    let sd = median(msdf.r, msdf.g, msdf.b);
    let screen_px_distance = px_range * (sd - 0.5);
    let opacity = clamp(screen_px_distance / fwidth(screen_px_distance) + 0.5, 0.0, 1.0);
    return vec4f(text_color.rgb, text_color.a * opacity);
}
```

**Critical settings:**
- Disable mipmaps (SDF handles scaling)
- Linear texture filtering
- Interpret channels as linear data, not sRGB

### Label Culling Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Label Manager                             │
├─────────────────────────────────────────────────────────────┤
│  1. Spatial Index (R-tree)                                  │
│     - Query visible labels by viewport bounds               │
│     - O(log n) lookup for millions of labels                │
│                                                             │
│  2. Priority Queue                                          │
│     - Sort by: node_importance * (1 / screen_size)          │
│     - Process high-priority labels first                    │
│                                                             │
│  3. Collision Grid                                          │
│     - Screen-space occupancy tracking                       │
│     - Insert label bounding boxes                           │
│     - Reject overlapping lower-priority labels              │
│                                                             │
│  4. Zoom-Level Thresholds                                   │
│     - Labels have minZoom/maxZoom attributes                │
│     - Pre-filter before collision detection                 │
└─────────────────────────────────────────────────────────────┘
```

### Alternatives Considered

| Approach | Pros | Cons |
|----------|------|------|
| Canvas 2D text | Simple | Blurry at high zoom, no GPU |
| Bitmap fonts | Fast | Requires atlas per size |
| Single-channel SDF | Smaller atlas | Rounded corners at large sizes |
| Runtime SDF generation | Dynamic fonts | Slower, complex |

### Sources

- msdf-atlas-gen (github.com/Chlumsky/msdf-atlas-gen)
- WebGPU MSDF Sample (webgpu.github.io/webgpu-samples/?sample=textRenderingMsdf)
- glyphon for Rust/wgpu (github.com/grovesNL/glyphon)
- deck.gl TextLayer (deck.gl/docs/api-reference/layers/text-layer)

---

## 3. Visual Layers (Heatmap, Contour, Metaball)

### Decision: Multi-Pass GPU Pipeline with Texture Accumulation

### Heatmap: Gaussian Splatting

**Pipeline:**
1. Render instanced quads (1 per node) to RGBA32F texture
2. Fragment shader evaluates Gaussian: `exp(-d²/2σ²)`
3. Additive blending accumulates density
4. Color-map pass converts density to visible colors

```wgsl
@fragment
fn heatmap_splat(input: VertexOutput) -> @location(0) vec4f {
    let d = length(input.local_pos);  // distance from quad center
    let sigma = uniforms.radius;
    let weight = exp(-d * d / (2.0 * sigma * sigma));
    return vec4f(weight, 0.0, 0.0, 1.0);  // accumulate in R channel
}
```

**Performance:** 100K nodes in <5ms/frame at 512x512 resolution.

### Contours: Parallel Marching Squares

**Three-stage compute pipeline:**

1. **Active Cell Identification**
   - Mark cells crossing isosurface threshold
   - Output cell case indices (0-15)

2. **Parallel Prefix Sum**
   - Transform counts to offsets
   - Each cell knows where to write line segments

3. **Vertex Generation**
   - Generate line vertices using offsets
   - No write conflicts, fully parallel

**Multiple contour levels:** Run pipeline once per isovalue, or batch in single pass.

### Metaballs: SDF with Smooth Minimum

**Quadratic smooth minimum (recommended):**

```wgsl
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

fn metaball_field(p: vec2<f32>) -> f32 {
    var d = 1e10;
    for (var i = 0u; i < num_balls; i++) {
        let ball_d = length(p - centers[i]) - radii[i];
        d = smin(d, ball_d, blend_radius);
    }
    return d;
}
```

**Screen-space rendering:** Evaluate SDF per pixel, threshold for boundary.

### Layer Compositing

**Render order (back to front):**
1. Background
2. Heatmap (additive blend)
3. Contour lines (alpha blend)
4. Metaballs (alpha blend)
5. Edges
6. Nodes
7. Labels

Each layer renders to separate texture, final pass composites.

### Sources

- deck.gl HeatmapLayer (deck.gl/docs/api-reference/aggregation-layers/heatmap-layer)
- WebGPU Marching Cubes (willusher.io/graphics/2024/04/22/webgpu-marching-cubes/)
- Inigo Quilez smooth minimum (iquilezles.org/articles/smin/)
- marching_squares_wgpu crate (crates.io/crates/marching_squares_wgpu)

---

## 4. Rust/WASM Architecture

### Decision: wasm-pack Build, SoA Layout, Zero-Copy Views, petgraph + rstar

### Build Toolchain

```bash
wasm-pack build --target web --release
```

- `--target web`: ES modules, no bundler required
- wasm-pack orchestrates: compile → wasm-bindgen → wasm-opt → npm package

### Data Transfer Strategy

**Problem:** Passing `&[f32]` via wasm-bindgen copies the entire array.

**Solution:** Return views into WASM linear memory:

```rust
use js_sys::Float32Array;

#[wasm_bindgen]
impl GraphEngine {
    /// Zero-copy view for WebGL/WebGPU upload
    /// SAFETY: View invalidated on any Rust allocation
    pub fn get_positions_view(&self) -> Float32Array {
        unsafe { Float32Array::view(&self.positions) }
    }
}
```

**JavaScript side:**
```javascript
const view = engine.get_positions_view();
device.queue.writeBuffer(positionBuffer, 0, view);
// Use immediately, don't store
```

### Graph Data Structure

**Use `petgraph::stable_graph::StableGraph`:**
- Stable indices across node/edge removals
- Critical for exposing indices to JavaScript
- O(|V| + |E|) space complexity

```rust
use petgraph::stable_graph::StableGraph;

#[wasm_bindgen]
pub struct GraphEngine {
    graph: StableGraph<(), f32, Directed>,

    // Positions in SoA for SIMD/cache
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    vel_x: Vec<f32>,
    vel_y: Vec<f32>,

    // Spatial index for hit testing
    rtree: RTree<NodePoint>,

    // Interleaved for GPU upload
    render_buffer: Vec<f32>,
}
```

### Spatial Indexing

**Use `rstar` R*-tree:**

```rust
use rstar::{RTree, AABB, PointDistance};

impl SpatialIndex {
    pub fn nearest(&self, x: f32, y: f32) -> Option<u32> {
        self.tree.nearest_neighbor(&[x, y]).map(|p| p.id)
    }

    pub fn query_rect(&self, bounds: AABB<[f32; 2]>) -> Vec<u32> {
        self.tree.locate_in_envelope(&bounds)
            .map(|p| p.id)
            .collect()
    }
}
```

**Performance:** O(log n) queries, bulk loading for initial build.

### SIMD Optimization

Enable WASM SIMD for vectorized force calculations:

```bash
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web
```

### Memory Growth Handling

```javascript
let cachedBuffer = null;
let positionsView = null;

function getPositions() {
    if (wasm.memory.buffer !== cachedBuffer) {
        cachedBuffer = wasm.memory.buffer;
        positionsView = new Float32Array(
            cachedBuffer,
            engine.positions_ptr(),
            engine.positions_len()
        );
    }
    return positionsView;
}
```

### Sources

- wasm-pack documentation (rustwasm.github.io/docs/wasm-pack/)
- petgraph stable_graph (docs.rs/petgraph/latest/petgraph/stable_graph/)
- rstar R*-tree (docs.rs/rstar/latest/rstar/)
- fdg force-directed layout (github.com/grantshandy/fdg)
- V8 WASM SIMD (v8.dev/features/simd)

---

## 5. Interaction and Hit Testing

### Decision: Hybrid CPU/GPU Approach

### Strategy

1. **Spatial Index (Rust/WASM):** R-tree for O(log n) point queries
2. **GPU Picking (optional):** Render node IDs to offscreen texture, read pixel under cursor
3. **Viewport Culling:** Only query nodes in visible bounds

### Event Flow

```
User Input (mousemove)
    ↓
Screen → Graph coordinate transform (TypeScript)
    ↓
R-tree query via WASM (nearest neighbor)
    ↓
Return node ID to TypeScript
    ↓
Emit hover/select event
```

### Drag Interaction

```typescript
interface DragState {
    nodeId: number | null;
    startPosition: Vec2;
    offset: Vec2;
}

function onDragStart(nodeId: number, screenPos: Vec2) {
    engine.pin_node(nodeId);  // WASM: exclude from simulation
    dragState = { nodeId, startPosition: engine.get_node_position(nodeId), offset };
}

function onDragMove(screenPos: Vec2) {
    const graphPos = viewport.screenToGraph(screenPos);
    engine.set_node_position(dragState.nodeId, graphPos.x, graphPos.y);
}

function onDragEnd() {
    engine.unpin_node(dragState.nodeId);  // Rejoin simulation
    dragState = null;
}
```

---

## Summary of Key Decisions

| Area | Decision | Key Benefit |
|------|----------|-------------|
| Force Simulation | WebGPU compute + Barnes-Hut | O(n log n), GPU parallel |
| Quadtree | Bottom-up construction | 69x faster than top-down |
| Memory Layout | Struct of Arrays (SoA) | SIMD, cache coherent |
| Text Rendering | MSDF + hierarchical culling | Sharp at all zoom levels |
| Heatmaps | Gaussian splatting to texture | Real-time, additive blend |
| Contours | Parallel marching squares | GPU-native, stream compaction |
| Metaballs | SDF with quadratic smin | Smooth blobs, GPU-friendly |
| Graph Structure | petgraph StableGraph | Stable indices for JS interop |
| Spatial Index | rstar R*-tree | O(log n) hit testing |
| Data Transfer | Float32Array views | Zero-copy JS↔WASM |
| Build | wasm-pack --target web | No bundler, ES modules |
