# HeroineGraph

High-performance graph visualization library powered by WebGPU.

HeroineGraph renders large graphs (100K+ nodes) at 60fps using GPU-accelerated force simulation and rendering. It provides interactive exploration with pan, zoom, drag, and selection, plus visualization layers for heatmaps, contours, metaballs, and text labels.

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
npm install @heroine-graph/core

# Deno
import { createHeroineGraph } from "jsr:@heroine-graph/core";
```

## Quick Start

```typescript
import { createHeroineGraph, getSupportInfo } from "@heroine-graph/core";

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
  edgeCount: number;
  positionsX: Float32Array;
  positionsY: Float32Array;
  radii: Float32Array;
  colors: Float32Array;         // RGBA, 4 values per node
  edgeSources: Uint32Array;
  edgeTargets: Uint32Array;
  edgeWidths?: Float32Array;
  edgeColors?: Float32Array;    // RGBA, 4 values per edge
}
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
graph.fitToView();              // Fit all content in view
graph.centerOn(x, y);           // Center on coordinates

// Selection
graph.selectNodes([id1, id2]);
graph.clearSelection();
const selected = graph.selectedNodes;

// Node manipulation
graph.pinNode(id);              // Fix node position
graph.unpinNode(id);            // Release node
graph.setNodePosition(id, x, y); // Set position directly

// Simulation
graph.startSimulation();
graph.stopSimulation();
graph.pauseSimulation();
graph.resumeSimulation();
```

## Events

```typescript
// Node events
graph.on("node:click", (event) => {
  console.log("Clicked node:", event.nodeId);
});

graph.on("node:hover:enter", (event) => {
  console.log("Hover enter:", event.nodeId);
});

graph.on("node:hover:leave", (event) => {
  console.log("Hover leave:", event.nodeId);
});

graph.on("node:dragstart", (event) => { /* ... */ });
graph.on("node:dragmove", (event) => { /* ... */ });
graph.on("node:dragend", (event) => { /* ... */ });

// Edge events
graph.on("edge:click", (event) => { /* ... */ });
graph.on("edge:hover:enter", (event) => { /* ... */ });
graph.on("edge:hover:leave", (event) => { /* ... */ });

// Viewport events
graph.on("viewport:change", (event) => {
  console.log("Zoom:", event.state.scale);
  console.log("Pan:", event.state.x, event.state.y);
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
  threshold: 0.5,               // Iso-surface threshold
  colorScale: "viridis",
  opacity: 0.6,
});

graph.disableMetaball();
```

### Labels

Display text labels on nodes:

```typescript
graph.enableLabels({
  fontSize: 14,
  color: "#333333",
  maxLabels: 100,               // Maximum visible labels
  priorityField: "importance",  // Metadata field for priority
});

graph.disableLabels();
```

## Configuration Options

```typescript
const graph = await createHeroineGraph({
  canvas,
  config: {
    // Node defaults
    defaultNodeRadius: 5,
    defaultNodeColor: "#666666",

    // Edge defaults
    defaultEdgeWidth: 1,
    defaultEdgeColor: "#999999",

    // Simulation
    simulation: {
      alpha: 1.0,               // Initial temperature
      alphaMin: 0.001,          // Stop threshold
      alphaDecay: 0.0228,       // Cooling rate
      velocityDecay: 0.4,       // Friction
    },

    // Forces
    forces: {
      charge: -30,              // Node repulsion
      linkDistance: 30,         // Ideal edge length
      linkStrength: 1.0,        // Edge stiffness
      centerStrength: 0.1,      // Pull toward center
    },

    // Viewport
    viewport: {
      minScale: 0.01,
      maxScale: 100,
      panSpeed: 1,
      zoomSpeed: 0.002,
    },
  },
  debug: false,                 // Enable debug logging
});
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
