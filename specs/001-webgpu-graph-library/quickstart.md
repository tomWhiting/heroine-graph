# Quickstart: Heroine Graph

Get a graph visualization running in under 5 minutes.

---

## Prerequisites

- Modern browser with WebGPU support (Chrome 113+, Firefox 141+, Safari 26+, Edge 113+)
- Node.js 18+ or Deno 2.x (for development server)

---

## Installation

### npm / pnpm / yarn

```bash
npm install @heroine-graph/core
```

### Deno

```typescript
import { createHeroineGraph } from "jsr:@heroine-graph/core";
```

### CDN (for quick experiments)

```html
<script type="module">
  import { createHeroineGraph } from "https://esm.sh/@heroine-graph/core";
</script>
```

---

## Basic Usage

### 1. Create a Canvas

```html
<!DOCTYPE html>
<html>
<head>
  <style>
    #graph {
      width: 100%;
      height: 100vh;
    }
  </style>
</head>
<body>
  <canvas id="graph"></canvas>
  <script type="module" src="main.js"></script>
</body>
</html>
```

### 2. Initialize the Graph

```typescript
// main.ts
import { createHeroineGraph, checkWebGPU } from "@heroine-graph/core";

async function main() {
  // Check WebGPU availability
  const status = await checkWebGPU();
  if (!status.supported) {
    console.error("WebGPU not supported:", status.error);
    return;
  }

  // Create the graph
  const graph = await createHeroineGraph({
    canvas: "#graph",
  });

  // Load some data
  graph.load({
    nodes: [
      { id: "a", label: "Node A" },
      { id: "b", label: "Node B" },
      { id: "c", label: "Node C" },
      { id: "d", label: "Node D" },
    ],
    edges: [
      { source: "a", target: "b" },
      { source: "a", target: "c" },
      { source: "b", target: "d" },
      { source: "c", target: "d" },
    ],
  });

  // Start the simulation
  graph.startSimulation();

  // Fit all nodes in view
  graph.fitToView();
}

main();
```

That's it! You should see four nodes with edges, animating into position.

---

## Interactive Features

### Handle Node Clicks

```typescript
graph.on("node:click", (event) => {
  console.log("Clicked node:", event.nodeId);
  const node = graph.getNode(event.nodeId);
  console.log("Node data:", node);
});
```

### Handle Selection

```typescript
graph.on("selection:change", (event) => {
  console.log("Selected nodes:", event.selectedNodes);
  console.log("Selected edges:", event.selectedEdges);
});
```

### Drag Nodes

Node dragging is enabled by default. The node becomes pinned while dragging
and rejoins the simulation when released.

```typescript
graph.on("node:dragstart", (e) => console.log("Started dragging", e.nodeId));
graph.on("node:dragend", (e) => console.log("Finished dragging", e.nodeId));
```

---

## Visual Layers

### Enable Heatmap

```typescript
graph.showLayer("heatmap");
graph.setLayerConfig("heatmap", {
  radius: 30,
  intensity: 1.2,
  colorScale: "viridis",
  opacity: 0.6,
});
```

### Enable Contours

```typescript
graph.showLayer("contour");
graph.setLayerConfig("contour", {
  thresholds: [0.2, 0.4, 0.6],
  strokeWidth: 1,
  strokeColor: "#333333",
});
```

### Enable Labels

```typescript
graph.showLayer("labels");
graph.setLayerConfig("labels", {
  fontSize: 12,
  fontColor: "#1f2937",
  minZoom: 0.5,
  maxLabels: 500,
});
```

---

## Simulation Control

### Pause / Resume

```typescript
graph.pauseSimulation();
// ... later
graph.startSimulation();
```

### Adjust Forces

```typescript
graph.setSimulationConfig({
  repulsion: -50,      // Stronger repulsion
  attraction: 0.5,     // Weaker springs
  gravity: 0.2,        // Stronger center pull
  linkDistance: 50,    // Longer ideal edge length
});

graph.restartSimulation();  // Reheat the simulation
```

### Manual Stepping

```typescript
graph.pauseSimulation();
for (let i = 0; i < 100; i++) {
  graph.tickSimulation();
}
```

---

## Viewport Control

### Pan and Zoom

```typescript
// Pan by delta
graph.pan(100, 50);

// Zoom in 2x at center
graph.zoom(2);

// Zoom at specific point
graph.zoom(1.5, mouseX, mouseY);

// Set absolute zoom
graph.zoomTo(1.0);

// Fit all content
graph.fitToView(padding: 50);

// Fit specific nodes
graph.fitNodes([nodeIdA, nodeIdB], padding: 30);
```

### Coordinate Conversion

```typescript
// Screen pixels to graph units
const graphPos = graph.screenToGraph(event.clientX, event.clientY);

// Graph units to screen pixels
const screenPos = graph.graphToScreen(node.x, node.y);
```

---

## Large Graphs (100K+ nodes)

For large graphs, use typed arrays for better performance:

```typescript
const nodeCount = 100000;
const edgeCount = 500000;

// Generate random positions
const positions = new Float32Array(nodeCount * 2);
for (let i = 0; i < nodeCount; i++) {
  positions[i * 2] = Math.random() * 1000 - 500;
  positions[i * 2 + 1] = Math.random() * 1000 - 500;
}

// Generate random edges (source, target pairs)
const edges = new Uint32Array(edgeCount * 2);
for (let i = 0; i < edgeCount; i++) {
  edges[i * 2] = Math.floor(Math.random() * nodeCount);
  edges[i * 2 + 1] = Math.floor(Math.random() * nodeCount);
}

graph.load({
  nodeCount,
  edgeCount,
  positions,
  edges,
});
```

---

## Framework Integration

### React

```bash
npm install @heroine-graph/react
```

```tsx
import { HeroineGraph } from "@heroine-graph/react";

function App() {
  const data = {
    nodes: [{ id: "a" }, { id: "b" }],
    edges: [{ source: "a", target: "b" }],
  };

  return (
    <HeroineGraph
      data={data}
      onNodeClick={(e) => console.log("Clicked", e.nodeId)}
      style={{ width: "100%", height: "600px" }}
    />
  );
}
```

### Vue

```bash
npm install @heroine-graph/vue
```

```vue
<template>
  <HeroineGraph
    :data="data"
    @node:click="handleNodeClick"
    style="width: 100%; height: 600px"
  />
</template>

<script setup>
import { HeroineGraph } from "@heroine-graph/vue";

const data = {
  nodes: [{ id: "a" }, { id: "b" }],
  edges: [{ source: "a", target: "b" }],
};

function handleNodeClick(event) {
  console.log("Clicked", event.nodeId);
}
</script>
```

### Svelte

```bash
npm install @heroine-graph/svelte
```

```svelte
<script>
  import { HeroineGraph } from "@heroine-graph/svelte";

  const data = {
    nodes: [{ id: "a" }, { id: "b" }],
    edges: [{ source: "a", target: "b" }],
  };

  function handleNodeClick(event) {
    console.log("Clicked", event.detail.nodeId);
  }
</script>

<HeroineGraph
  {data}
  on:node:click={handleNodeClick}
  style="width: 100%; height: 600px"
/>
```

---

## Cleanup

Always dispose when done:

```typescript
// When unmounting or navigating away
graph.dispose();
```

---

## Next Steps

- [API Reference](./contracts/api.ts) - Full API documentation
- [Data Model](./data-model.md) - Entity definitions
- [Examples](../../examples/) - More complex examples

---

## Troubleshooting

### "WebGPU not supported"

1. Check browser version (Chrome 113+, Firefox 141+, Safari 26+)
2. Enable WebGPU in flags if needed (`chrome://flags/#enable-unsafe-webgpu`)
3. Ensure hardware acceleration is enabled
4. Check `checkWebGPU()` result for detailed error

### Graph doesn't appear

1. Ensure canvas has non-zero dimensions (check CSS)
2. Call `graph.fitToView()` after loading data
3. Check browser console for WebGPU errors

### Performance issues

1. Use typed array input for large graphs
2. Disable labels layer when zoomed out
3. Reduce heatmap radius for faster updates
4. Check GPU memory usage in browser dev tools
