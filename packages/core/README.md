# @graphmother/core

WebGPU-accelerated visualization for large graphs.

GraphMother pairs a Rust/WASM graph engine (topology storage, spatial indexing,
hierarchy analysis) with a GPU-resident force simulation and renderer. Node and
edge state lives in GPU storage buffers that are shared between the compute and
render passes, so a running layout costs no per-frame CPU/GPU round trips — the
target workload is code-repository graphs in the tens of thousands of nodes.

- **GPU force simulation** — a suite of interchangeable force algorithms, all
  implemented as WGSL compute shaders.
- **GPU rendering** — instanced nodes and edges, MSDF text labels that stay
  sharp at any zoom.
- **Visualization layers** — heatmap, metaball, contour, and label layers
  composited over the graph.
- **Interaction** — pan, zoom, hover, click, multi-select, node dragging and
  pinning, all hit-tested against the WASM spatial index.
- **Typed end to end** — the published package ships its own `.d.ts` and
  references `@webgpu/types`.

## Requirements

GraphMother requires WebGPU:

| Browser | Version                                       |
| ------- | --------------------------------------------- |
| Chrome  | 113+                                          |
| Edge    | 113+                                          |
| Safari  | 18+ (macOS Sequoia, iOS 18)                   |
| Firefox | 141+ on Windows; behind a flag on macOS/Linux |

There is no Canvas/WebGL fallback. Probe support before you construct anything —
`isSupported()` is a cheap synchronous check for the `navigator.gpu` entry
point, `getSupportInfo()` actually requests an adapter and tells you _why_ it
failed.

## Install

```bash
npm install @graphmother/core
```

```ts
// Deno resolves the same npm package; there is no JSR publication.
import { createGraphMother } from "npm:@graphmother/core";
```

## Quick start

```ts
import { createGraphMother, getSupportInfo } from "@graphmother/core";

const support = await getSupportInfo();
if (!support.supported) {
  throw new Error(`WebGPU unavailable: ${support.reason}`);
}

const canvas = document.querySelector("#graph") as HTMLCanvasElement;
const graph = await createGraphMother({ canvas });

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

graph.on("node:click", (event) => console.log("clicked", event.nodeId));
graph.on("selection:change", (event) => console.log(event.selectedNodes));
graph.on("simulation:end", () => graph.fitToView());

// Later, release every GPU resource the instance owns:
graph.dispose();
```

Loading starts the render loop and the simulation automatically. For large
graphs, `load()` also accepts typed arrays (`positions`, `edgePairs`,
`nodeRadii`, `nodeColors`, …) so you never materialize per-node objects.

### Events

`node:click`, `node:dblclick`, `node:hoverenter`, `node:hoverleave`,
`node:dragstart`, `node:dragmove`, `node:dragend`, `edge:click`,
`edge:hoverenter`, `edge:hoverleave`, `selection:change`, `viewport:change`,
`simulation:tick`, `simulation:end`, `background:click`, plus mutation events
for node/edge add and remove. Subscribe with `graph.on(name, handler)` and
release with `graph.off(name, handler)`.

## Force algorithms

Every layout is a GPU compute pipeline behind one switch:

```ts
graph.setForceAlgorithm("barnes-hut");
graph.setForceConfig({
  repulsionStrength: -50, // negative repels
  springStrength: 0.1,
  springLength: 30,
  centerStrength: 0.01,
  collisionEnabled: true,
});
graph.setAlphaDecay(0.0228); // d3 convention — ~300 ticks to rest
graph.restartSimulation();
```

| id                 | Algorithm        | Use it for                                                         |
| ------------------ | ---------------- | ------------------------------------------------------------------ |
| `n2`               | N² direct        | Exact all-pairs repulsion; small graphs (< ~10K nodes).            |
| `barnes-hut`       | Barnes-Hut       | GPU binary radix tree, O(N log N). General purpose, 5K–100K nodes. |
| `force-atlas2`     | ForceAtlas2      | Network visualization with degree-weighted forces.                 |
| `linlog`           | LinLog           | Logarithmic attraction; emphasizes community structure.            |
| `t-fdp`            | t-FDP            | t-distribution kernel; preserves local neighborhoods.              |
| `density`          | Density field    | O(N) grid approximation for very large graphs (100K+).             |
| `relativity-atlas` | Relativity Atlas | Hierarchical O(N+E) layout for DAGs and deep trees.                |
| `tidy-tree`        | Tidy Tree        | Buchheim tree layout — directory trees, org charts.                |
| `community`        | Community        | Louvain detection plus cluster-aware forces.                       |
| `codebase`         | Codebase         | Containment-aware layout; files cluster inside their directories.  |

Simulation control is on the instance: `startSimulation()`, `pauseSimulation()`,
`stopSimulation()`, `restartSimulation()`, `setSimulationAlpha()`.

Rendering pauses on its own axis: `pauseRendering()`, `resumeRendering()`,
`isRenderingPaused()`. Use it when the host knows the canvas is not being seen
while the page still is — occluded by other UI, in a collapsed panel — which
`document.hidden` cannot report. The simulation keeps running unless paused too,
and only `resumeRendering()` lifts a host pause.

## Layers

```ts
graph.enableHeatmap({ colorScale: "viridis", radius: 50, intensity: 1.0 });
graph.enableMetaball({ threshold: 0.5, blendRadius: 20, fillColor: "#4285f4" });
await graph.enableLabels({ fontSize: 14, maxLabels: 100, priority: "degree" });
```

- **Heatmap** — node density as a color field. Scales: `viridis`, `plasma`,
  `inferno`, `magma`, `cividis`, `turbo`, `spectral`, `coolwarm`, `blues`,
  `reds`, `greens`, `greys`.
- **Metaball** — smooth-union blobs around dense regions.
- **Labels** — MSDF text with priority-based culling. The layer fetches
  `./assets/fonts/roboto-msdf.{json,png}` at runtime; those files ship in the
  package under `assets/fonts/`, so copy them into whatever directory your app
  serves as `assets/fonts/`.
- **Contour** — density iso-lines. **Experimental**: the threshold/marching
  pipeline is still being tuned and its config shape may change.

Each layer has a matching `disable*()` and `set*Config()` method.

## Semantic LOD & DOM cards

```ts
graph.setLodConfig({ enabled: true, domThreshold: 48, maxCards: 150 });
graph.setDomOverlay({ enabled: true });
graph.setCardProvider({
  mount(container, node) {
    container.textContent = node.label ?? String(node.externalId);
    return null;
  },
  release(container) {
    container.replaceChildren();
  },
});
graph.on("card:error", (event) => report(event.externalId, event.hook, event.error));
```

Subtrees fold into their ancestors as the camera pulls out; the largest nodes
are promoted to real DOM as it pushes in. Core owns which nodes are carded,
where each card sits and how big it is — the provider owns everything inside a
card, and must remove its own children in `release`, because containers are
pooled.

The four thresholds are two independent pairs over two different metrics, and
consumer records must be keyed on `node.externalId` rather than on the recycled
GPU slot. Both are in
[docs/lod-and-cards.md](https://github.com/tomWhiting/graphmother/blob/main/docs/lod-and-cards.md),
which is the authority for this surface.

## Framework wrappers

Thin components that own initialization and teardown for you:

- [`@graphmother/react`](https://github.com/tomWhiting/graphmother/tree/main/packages/react)
- [`@graphmother/vue`](https://github.com/tomWhiting/graphmother/tree/main/packages/vue)
- [`@graphmother/svelte`](https://github.com/tomWhiting/graphmother/tree/main/packages/svelte)

## License

MIT
