# @graphmother/svelte

Svelte 5 wrapper for [GraphMother](https://github.com/tomWhiting/graphmother) —
WebGPU-accelerated visualization for large graphs.

The `<GraphMother />` component owns the canvas, the WebGPU device, and the
render loop: it initializes on mount, forwards graph events to callback props,
reloads when `data` changes, resizes with its container, and disposes everything
on destroy.

## Install

```bash
npm install @graphmother/svelte @graphmother/core
```

`svelte >= 5` is a peer dependency — this package is runes-only.

## Usage

```svelte
<script lang="ts">
  import { GraphMother } from "@graphmother/svelte";

  let component: GraphMother;

  const data = {
    nodes: [{ id: 1 }, { id: 2 }],
    edges: [{ source: 1, target: 2 }],
  };
</script>

<GraphMother
  bind:this={component}
  {data}
  height="600px"
  onnodeClick={(e) => console.log("clicked", e.nodeId)}
  onready={(graph) => graph.setForceAlgorithm("barnes-hut")}
/>
```

`component.getGraph()` returns the underlying `GraphMother` instance for
anything the props do not cover.

## Stores

`createGraphStore()` manages an instance against a canvas you own;
`createSimulationStore()` exposes simulation status, alpha, and
start/stop/restart controls. These are runes objects, not Svelte 4 stores — read
state as plain properties (`store.isReady`), never with the `$` prefix, and
destructure only the methods.

```svelte
<script lang="ts">
  import { createGraphStore } from "@graphmother/svelte";

  // Created during component init, so disposal is registered automatically.
  const store = createGraphStore();
  const { initialize, load } = store;

  let canvasEl: HTMLCanvasElement | undefined = $state();

  $effect(() => {
    if (canvasEl) initialize(canvasEl);
  });
</script>

<canvas bind:this={canvasEl}></canvas>
{#if store.isReady}
  <button onclick={() => load(data)}>Load</button>
{/if}
```

Created outside a component context there is no `onDestroy` to hook, so call
`store.dispose()` yourself.

## Full API

See the [`@graphmother/core` README](https://github.com/tomWhiting/graphmother/tree/main/packages/core)
for graph data formats, the force-algorithm list, layers, and events.

## License

MIT
