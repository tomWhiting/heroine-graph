# Quickstart: Graph Algorithms WASM Module

This guide shows how to use the graph algorithms in heroine-graph.

## Basic Usage

### 1. Community Detection

Identify clusters of related nodes:

```typescript
import { HeroineGraphWasm } from 'heroine-graph-wasm';

// Create graph and add nodes/edges
const graph = new HeroineGraphWasm();
// ... populate graph ...

// Detect communities using Louvain algorithm
const communities = graph.detectCommunities({
  algorithm: 'louvain',
  resolution: 1.0,  // Higher = more communities
});

console.log(`Found ${communities.communities.length} communities`);
console.log(`Total modularity: ${communities.totalModularity}`);

// Get community for a specific node
const nodeId = 42;
const communityId = communities.nodeToCommunity.get(nodeId);
console.log(`Node ${nodeId} belongs to community ${communityId}`);
```

### 2. Hull Computation

Create visual boundaries around communities:

```typescript
// After detecting communities...
const hulls = graph.computeHulls(communities, {
  hullType: 'convex',  // or 'concave' for tighter boundaries
  concavity: 2.0,      // only for concave hulls
});

// Render hulls
for (const hull of hulls) {
  // hull.vertices is Float32Array [x0, y0, x1, y1, ...]
  renderPolygon(hull.vertices, {
    fillColor: getCommunityColor(hull.communityId),
    strokeWidth: 2,
  });
}
```

### 3. Boundary Collision Physics

Make community boundaries repel each other:

```typescript
// Initialize physics with boundaries
graph.initBoundaryPhysics(hulls, {
  repulsionStrength: 0.5,
  damping: 0.9,
});

// In your render loop
function animate() {
  // Update physics
  const result = graph.updateBoundaryPhysics();

  // Apply displacements to nodes
  for (let i = 0; i < result.nodeIds.length; i++) {
    const nodeId = result.nodeIds[i];
    const dx = result.displacementsX[i];
    const dy = result.displacementsY[i];

    const [x, y] = graph.getNodePosition(nodeId);
    graph.setNodePosition(nodeId, x + dx, y + dy);
  }

  // Recompute hulls if boundaries moved
  if (result.hasOverlaps) {
    const newHulls = graph.computeHulls(communities);
    graph.initBoundaryPhysics(newHulls);
  }

  requestAnimationFrame(animate);
}
```

### 4. Centrality Measures

Identify important nodes:

```typescript
// PageRank - influence based on incoming links
const pagerank = graph.computeCentrality({
  type: 'pagerank',
  damping: 0.85,
});

// Betweenness - nodes on critical paths
const betweenness = graph.computeCentrality({
  type: 'betweenness',
});

// Use scores to size nodes
for (const [nodeId, score] of pagerank.scores) {
  const normalizedSize = (score - pagerank.min) / (pagerank.max - pagerank.min);
  setNodeSize(nodeId, 5 + normalizedSize * 20);
}
```

### 5. Connected Components

Find disconnected subgraphs:

```typescript
// For undirected connectivity
const components = graph.getConnectedComponents();
console.log(`Graph has ${components.components.length} components`);

// For directed graphs (respects edge direction)
const sccs = graph.getStronglyConnectedComponents();
```

## Complete Example: Visualizing Communities

```typescript
import { HeroineGraphWasm, HeroineGraph } from 'heroine-graph';

async function visualizeCommunities() {
  // Initialize
  const wasm = new HeroineGraphWasm();
  const renderer = await HeroineGraph.create(canvas);

  // Load your graph data
  loadGraphData(wasm);

  // Detect communities
  const communities = wasm.detectCommunities({
    algorithm: 'louvain',
    resolution: 1.2,
  });

  // Color nodes by community
  const colors = new Float32Array(wasm.nodeCount() * 4);
  const palette = generateColorPalette(communities.communities.length);

  for (const [nodeId, communityId] of communities.nodeToCommunity) {
    const color = palette[communityId];
    const offset = nodeId * 4;
    colors[offset + 0] = color.r;
    colors[offset + 1] = color.g;
    colors[offset + 2] = color.b;
    colors[offset + 3] = 1.0;
  }
  renderer.setNodeColors(colors);

  // Compute and render boundaries
  const hulls = wasm.computeHulls(communities, {
    hullType: 'concave',
    concavity: 2.5,
  });

  // Enable boundary physics
  wasm.initBoundaryPhysics(hulls, {
    enabled: true,
    repulsionStrength: 0.3,
  });

  // Render
  renderer.render();
}
```

## Progress Callbacks for Large Graphs

For graphs with 50K+ nodes, use progress callbacks:

```typescript
const communities = graph.detectCommunities(
  { algorithm: 'louvain' },
  (progress) => {
    updateProgressBar(progress.progress * 100);
    console.log(`${progress.phase}: ${progress.message}`);
  }
);
```

## Bulk Format for GPU Upload

For efficient GPU operations, use bulk result formats:

```typescript
// Get centrality in GPU-friendly format
const result = graph.computeCentralityBulk({ type: 'pagerank' });

// Direct upload to GPU buffer
const buffer = device.createBuffer({
  size: result.scores.byteLength,
  usage: GPUBufferUsage.STORAGE,
  mappedAtCreation: true,
});
new Float32Array(buffer.getMappedRange()).set(result.scores);
buffer.unmap();
```

## Configuration Reference

### Community Detection

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| algorithm | 'louvain' \| 'leiden' | 'louvain' | Detection algorithm |
| resolution | number | 1.0 | Higher = more, smaller communities |
| weighted | boolean | false | Use edge weights |
| maxIterations | number | 100 | Max optimization passes |

### Hull Computation

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| hullType | 'convex' \| 'concave' | 'convex' | Hull algorithm |
| concavity | number | 2.0 | Tightness (concave only) |
| fallbackRadius | number | 10.0 | Circle radius for 1-2 nodes |

### Centrality

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| type | CentralityType | required | Algorithm to use |
| normalized | boolean | true | Normalize to [0, 1] |
| damping | number | 0.85 | PageRank damping |

### Boundary Physics

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| enabled | boolean | true | Active state |
| repulsionStrength | number | 0.5 | Force magnitude |
| damping | number | 0.9 | Velocity decay |
| maxDisplacement | number | 10.0 | Max movement per frame |
