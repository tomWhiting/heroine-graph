# HeroineGraph

High-performance graph visualization library powered by WebGPU.

HeroineGraph renders large graphs using GPU-accelerated force simulation and rendering (measure your own workload with `deno task bench`). It provides interactive exploration with pan, zoom, drag, and selection, plus visualization layers for heatmaps, contours, metaballs, and text labels.

## Features

- **GPU-Accelerated Simulation**: Force-directed layout runs entirely on GPU compute shaders
- **WebGPU Rendering**: Hardware-accelerated node, edge, and label rendering
- **Visualization Layers**: Heatmaps, contours, metaballs for density visualization
- **MSDF Text Labels**: Sharp text at any zoom level using multi-channel signed distance fields
- **Interactive**: Pan, zoom, drag nodes, selection with keyboard modifiers
- **Type-Safe**: Full TypeScript support with comprehensive type definitions

## Browser Support

HeroineGraph requires WebGPU, which is available in:
- Chrome 113+ (April 2023)
- Edge 113+
- Firefox 126+ (behind flag, enabled by default in Firefox Nightly)
- Safari 18+ (macOS Sequoia, iOS 18)

Check support with `getSupportInfo()` before initializing.

## Installation

```bash
# npm
npm install @graphmother/core
```

```typescript
// Deno (resolves the same npm package; there is no JSR publication)
import { createHeroineGraph } from "npm:@graphmother/core";
```

## Quick Start

```typescript
import { createHeroineGraph, getSupportInfo } from "@graphmother/core";

// Check WebGPU support
const support = await getSupportInfo();
if (!support.supported) {
  console.error("WebGPU not supported:", support.reason);
  return;
}

// Get a canvas element
const canvas = document.getElementById("graph-canvas") as HTMLCanvasElement;

// Create the graph instance
const graph = await createHeroineGraph({ canvas });

// Load data
await graph.load({
  nodes: [
    { id: "a", radius: 8, color: "#4285f4" },
    { id: "b", radius: 8, color: "#ea4335" },
    { id: "c", radius: 8, color: "#fbbc04" },
  ],
  edges: [
    { source: "a", target: "b" },
    { source: "b", target: "c" },
    { source: "c", target: "a" },
  ],
});

// The graph automatically starts rendering and simulating
```

## Graph Data Format

### Basic Input

```typescript
interface GraphInput {
  nodes: NodeInput[];
  edges: EdgeInput[];
}

interface NodeInput {
  id: string | number;           // Unique identifier
  x?: number;                    // Initial X position (optional)
  y?: number;                    // Initial Y position (optional)
  radius?: number;               // Node radius in pixels (default: 5)
  color?: string;                // CSS color string (default: "#666")
  metadata?: Record<string, any>; // Custom data
}

interface EdgeInput {
  source: string | number;       // Source node ID
  target: string | number;       // Target node ID
  width?: number;                // Edge width in pixels (default: 1)
  color?: string;                // CSS color string (default: "#999")
  metadata?: Record<string, any>; // Custom data
}
```

### Typed Input (Performance)

For large graphs, use typed arrays for better performance:

```typescript
interface GraphTypedInput {
  nodeCount: number;
  edgeCount?: number;
  positions?: Float32Array;     // Interleaved [x0, y0, x1, y1, ...]
  edgePairs?: Uint32Array;      // Interleaved [src0, tgt0, src1, tgt1, ...]
  edges?: Uint32Array;          // Alias for edgePairs
  nodeIds?: (string | number)[];
  edgeIds?: (string | number)[];
  nodeRadii?: Float32Array;     // 1 value per node
  nodeColors?: Float32Array;    // RGB, 3 values per node (0-1 range)
  edgeWidths?: Float32Array;    // 1 value per edge
  edgeColors?: Float32Array;    // RGB, 3 values per edge (0-1 range)
  nodeMetadata?: NodeMetadata[];
  edgeMetadata?: EdgeMetadata[];
}
```

Example:

```typescript
await graph.load({
  nodeCount: 3,
  positions: new Float32Array([0, 0, 100, 0, 50, 80]),
  edgePairs: new Uint32Array([0, 1, 1, 2]),
  nodeColors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), // RGB per node
});
```

## Interaction

### Viewport Controls

- **Pan**: Click and drag on empty space
- **Zoom**: Mouse wheel or trackpad pinch
- **Select Node**: Click on node
- **Multi-select**: Shift+click to add to selection
- **Drag Node**: Click and drag on a node (pins it in place)

### Programmatic Control

```typescript
// Viewport
graph.pan(dx, dy);              // Pan by delta
graph.zoom(factor);             // Zoom by factor
graph.fitToView();              // Fit all content in view (optional padding)
const viewport = graph.getViewportState();

// Selection
graph.selectNodes([id1, id2]);
graph.clearSelection();
const selected = graph.getSelectedNodes();

// Node manipulation
graph.pinNode(id);              // Fix node position
graph.unpinNode(id);            // Release node
graph.setNodePosition(id, x, y); // Set position directly

// Simulation
graph.startSimulation();
graph.pauseSimulation();
graph.stopSimulation();
graph.restartSimulation();      // Re-heat and resume
```

## Events

```typescript
// Node events
graph.on("node:click", (event) => {
  console.log("Clicked node:", event.nodeId);
});

graph.on("node:hoverenter", (event) => {
  console.log("Hover enter:", event.nodeId);
});

graph.on("node:hoverleave", (event) => {
  console.log("Hover leave:", event.nodeId);
});

graph.on("node:dragstart", (event) => { /* ... */ });
graph.on("node:dragmove", (event) => { /* ... */ });
graph.on("node:dragend", (event) => { /* ... */ });

// Edge events
graph.on("edge:click", (event) => { /* ... */ });
graph.on("edge:hoverenter", (event) => { /* ... */ });
graph.on("edge:hoverleave", (event) => { /* ... */ });

// Viewport events
graph.on("viewport:change", (event) => {
  console.log("Zoom:", event.viewport.scale);
  console.log("Pan:", event.viewport.x, event.viewport.y);
});

// Selection events
graph.on("selection:change", (event) => {
  console.log("Selected nodes:", event.selectedNodes);
  console.log("Selected edges:", event.selectedEdges);
});

// Simulation events
graph.on("simulation:tick", (event) => { /* ... */ });
graph.on("simulation:end", (event) => { /* ... */ });
```

## Visualization Layers

### Heatmap

Visualize node density as a color gradient:

```typescript
// Enable heatmap
graph.enableHeatmap({
  colorScale: "viridis",        // Color palette
  radius: 50,                   // Kernel radius in pixels
  intensity: 1.0,               // Brightness multiplier
  opacity: 0.8,                 // Layer opacity
});

// Update configuration
graph.setHeatmapConfig({
  colorScale: "plasma",
  intensity: 1.5,
});

// Disable
graph.disableHeatmap();
```

Available color scales: `viridis`, `plasma`, `inferno`, `magma`, `cividis`, `turbo`, `spectral`, `coolwarm`, `blues`, `reds`, `greens`, `greys`.

### Contours

Draw density contour lines:

```typescript
graph.enableContour({
  thresholds: [0.2, 0.4, 0.6, 0.8], // Density levels
  strokeWidth: 2,
  strokeColor: "#ffffff",
});

graph.disableContour();
```

### Metaballs

Organic blob visualization around dense regions:

```typescript
graph.enableMetaball({
  threshold: 0.5,               // SDF boundary threshold (0-1)
  blendRadius: 20,              // Smooth union blend radius in pixels
  fillColor: "#4285f4",
  opacity: 0.6,
});

graph.disableMetaball();
```

### Labels

Display text labels on nodes:

```typescript
await graph.enableLabels({
  fontSize: 14,
  fontColor: "#333333",
  maxLabels: 100,               // Maximum visible labels
  priority: "importance",       // "importance" (node field) or "degree"
});

graph.disableLabels();
```

The labels layer loads an MSDF font atlas at runtime by fetching
`./assets/fonts/roboto-msdf.json` and `./assets/fonts/roboto-msdf.png`
relative to the page. These files ship in the npm package under
`node_modules/@graphmother/core/assets/fonts/`; make sure your bundler or
static file setup serves them at `assets/fonts/` next to your page (e.g.
copy them into your public directory).

## Configuration Options

```typescript
const graph = await createHeroineGraph({
  canvas,                        // HTMLCanvasElement or CSS selector
  config: {
    // Visual defaults (GraphConfig)
    nodeDefaultRadius: 5,
    nodeDefaultColor: "#666666",
    edgeDefaultWidth: 1,
    edgeDefaultColor: "#999999",
    backgroundColor: "#ffffff",
  },
  debug: false,                  // Enable debug logging
});
```

Simulation forces are configured on the instance after creation:

```typescript
graph.setForceConfig({
  repulsionStrength: -50,        // Node repulsion (negative = repel)
  springStrength: 0.1,           // Edge stiffness
  springLength: 30,              // Ideal edge length
  centerStrength: 0.01,          // Pull toward center
  collisionEnabled: true,        // Node overlap resolution
});

graph.setAlphaDecay(0.0228);     // Cooling rate (~300 ticks to rest)
```

## Cleanup

```typescript
// Dispose of all GPU resources
graph.dispose();
```

## Running Storybook

The project includes interactive examples in Storybook:

```bash
deno task storybook
```

Then open http://localhost:6006 in your browser.

## Development

```bash
# Type check
deno task check

# Lint
deno task lint

# Format
deno task fmt

# Run tests
deno task test

# Build
deno task build
```

## Architecture

HeroineGraph uses a GPU-first architecture:

1. **WASM Module** (Rust): Graph topology storage, spatial indexing with R-tree
2. **GPU Compute Shaders** (WGSL): Force simulation (repulsion, springs, integration)
3. **GPU Render Pipelines** (WebGPU): Instanced rendering of nodes, edges, labels
4. **Visualization Layers**: Composable density-based visualizations

Data flows through GPU storage buffers that are shared between compute and render passes, minimizing CPU-GPU transfers.

## License

MIT
