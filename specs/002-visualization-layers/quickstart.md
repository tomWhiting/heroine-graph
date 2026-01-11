# Quickstart: Advanced Visualization Layer System

**Feature**: 002-visualization-layers
**Date**: 2026-01-12

## Overview

This guide shows how to use the advanced visualization features in heroine-graph.

## Per-Item Styling

### Setting Node Colors Directly

```typescript
import { HeroineGraph } from '@heroine-graph/core';

const graph = await HeroineGraph.create(canvas);
graph.loadGraph({ nodes, edges });

// Color each node individually (4 floats per node: RGBA)
const colors = new Float32Array(nodes.length * 4);
for (let i = 0; i < nodes.length; i++) {
    colors[i * 4 + 0] = Math.random();     // R
    colors[i * 4 + 1] = Math.random();     // G
    colors[i * 4 + 2] = Math.random();     // B
    colors[i * 4 + 3] = 1.0;               // A
}
graph.setNodeColors(colors);
```

### Type-Based Styling (Easier)

```typescript
// Define styles per node type - library builds Float32Array for you
graph.setNodeTypeStyles({
    file: { color: [0.35, 0.65, 1.0, 1.0], size: 8 },
    folder: { color: [1.0, 0.6, 0.2, 1.0], size: 16 },
    symbol: { color: [0.5, 0.5, 0.5, 1.0], size: 4 },
});

graph.setEdgeTypeStyles({
    import: { color: [0.3, 0.3, 0.8, 0.6], width: 1 },
    export: { color: [0.8, 0.3, 0.3, 0.6], width: 2 },
});
```

## Diagnostic Channels

### Creating a Channel

```typescript
// Define a channel for error counts
graph.defineChannel({
    id: 'errors',
    name: 'Error Count',
    color: [1, 0, 0, 0.8],  // Red
    aggregation: 'sum',      // Folders sum their children's errors
});

// Define a channel for code coverage
graph.defineChannel({
    id: 'coverage',
    name: 'Test Coverage',
    colorScale: {
        domain: [0, 100],
        range: ['#ff0000', '#00ff00'],  // Red to green
    },
    aggregation: 'avg',  // Folders average their children's coverage
});
```

### Pushing Data

```typescript
// Push error data for specific nodes
graph.setChannelData('errors', [
    { nodeId: 'src/auth.ts', value: 5 },
    { nodeId: 'src/api.ts', value: 12 },
    { nodeId: 'src/utils.ts', value: 0 },
]);

// Push coverage data
graph.setChannelData('coverage', [
    { nodeId: 'src/auth.ts', value: 45 },
    { nodeId: 'src/api.ts', value: 78 },
    { nodeId: 'src/utils.ts', value: 92 },
]);
```

## Multi-Layer System

### Defining Layers

```typescript
// Base layer: show all nodes and edges
graph.defineLayer({
    id: 'base',
    name: 'All Nodes',
    nodeFilter: () => true,
    visualizations: ['nodes', 'edges'],
    zOrder: 0,
});

// Heatmap layer: only files, colored by errors
graph.defineLayer({
    id: 'error-heatmap',
    name: 'Error Heatmap',
    nodeFilter: (node) => node.type === 'file',
    visualizations: ['heatmap'],
    zOrder: 1,
});

// Metaball layer: only folders as organic blobs
graph.defineLayer({
    id: 'folder-blobs',
    name: 'Folder Groups',
    nodeFilter: (node) => node.type === 'folder',
    visualizations: ['metaballs'],
    zOrder: 2,
});
```

### Toggling Visibility

```typescript
// Hide/show layers
graph.setLayerVisible('error-heatmap', false);
graph.setLayerVisible('error-heatmap', true);

// Get all layer states
const visibility = graph.getLayerVisibility();
// { base: true, 'error-heatmap': true, 'folder-blobs': true }
```

## Edge Flow Animation

### Basic Flow

```typescript
// Enable flow on Layer 1
graph.setEdgeFlow({
    layer1: {
        enabled: true,
        speed: 0.5,
        pulseWidth: 0.15,
        pulseCount: 3,
        waveShape: 1.0,  // Sine wave
    },
});
```

### Dual-Layer with Punch-Through

```typescript
graph.setEdgeFlow({
    layer1: {
        enabled: true,
        speed: 0.3,
        pulseWidth: 0.2,
        waveShape: 1.0,     // Smooth sine
        color: [0.5, 0.5, 1.0, 0.5],  // Blue tint
    },
    layer2: {
        enabled: true,
        speed: 1.2,
        pulseWidth: 0.05,
        waveShape: 0.5,     // Triangle (sharp)
        brightness: 2.5,
        color: [1.0, 1.0, 0.0, 0.8],  // Yellow sparks
    },
});
```

## Curved Edges

### Enable Globally

```typescript
graph.setConfig({
    curvedEdges: true,
    curveSegments: 19,
    curveWeight: 0.8,
    curveControlPointDistance: 0.5,
});
```

### Per-Edge Curvature

```typescript
// Set curvature for each edge (1 float per edge)
// 0 = straight, positive = curve one way, negative = curve other way
const curvatures = new Float32Array(edges.length);
for (let i = 0; i < edges.length; i++) {
    curvatures[i] = (Math.random() - 0.5) * 0.5;  // Random -0.25 to 0.25
}
graph.setEdgeCurvatures(curvatures);
```

## Node Border Configuration

### Global Border Settings

```typescript
graph.setNodeBorder({
    enabled: true,
    thickness: 2,
    color: [1, 1, 1, 1],  // White border
});
```

### Disable Borders

```typescript
graph.setNodeBorder({ enabled: false });
```

### Per-Type Border Overrides

```typescript
graph.setNodeTypeStyles({
    important: {
        color: [1, 0.8, 0, 1],
        border: {
            enabled: true,
            thickness: 3,
            color: [1, 0, 0, 1],  // Red border for important nodes
        },
    },
});
```

## Topographical Contours

### Enable Contours on Heatmap

```typescript
graph.setContours({
    enabled: true,
    thresholds: [0.2, 0.4, 0.6, 0.8],
    lineColor: [0, 0, 0, 0.5],
    lineThickness: 1,
});
```

## Complete Example

```typescript
import { HeroineGraph } from '@heroine-graph/core';

async function createVisualization() {
    const canvas = document.getElementById('graph') as HTMLCanvasElement;
    const graph = await HeroineGraph.create(canvas);

    // Load data
    await graph.loadGraph({
        nodes: myNodes,
        edges: myEdges,
    });

    // Style by type
    graph.setNodeTypeStyles({
        file: { color: [0.4, 0.6, 1.0, 1.0], size: 8 },
        folder: { color: [1.0, 0.7, 0.3, 1.0], size: 20 },
    });

    // Define error channel
    graph.defineChannel({
        id: 'errors',
        color: [1, 0, 0, 0.8],
        aggregation: 'sum',
    });

    // Push error data
    graph.setChannelData('errors', errorData);

    // Enable flow animation
    graph.setEdgeFlow({
        layer1: { enabled: true, speed: 0.5 },
    });

    // Enable curved edges
    graph.setConfig({ curvedEdges: true });

    // Start simulation
    graph.startSimulation();
}
```
