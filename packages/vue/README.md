# @graphmother/vue

Vue 3 wrapper for [GraphMother](https://github.com/tomWhiting/graphmother) —
WebGPU-accelerated visualization for large graphs.

The `<GraphMother />` component owns the canvas, the WebGPU device, and the
render loop: it initializes on mount, re-emits graph events, reloads when the
`data` prop is replaced, resizes with its container, and disposes everything on
unmount.

## Install

```bash
npm install @graphmother/vue @graphmother/core
```

`vue >= 3.5` is a peer dependency.

## Usage

```vue
<script setup lang="ts">
import { ref } from "vue";
import { GraphMother } from "@graphmother/vue";
import type { GraphInput } from "@graphmother/vue";

const data = ref<GraphInput>({
  nodes: [{ id: 1 }, { id: 2 }],
  edges: [{ source: 1, target: 2 }],
});
</script>

<template>
  <GraphMother
    :data="data"
    height="600px"
    @node-click="(e) => console.log('clicked', e.nodeId)"
    @ready="(graph) => graph.setForceAlgorithm('barnes-hut')"
  />
</template>
```

`data` is watched by reference only — deep watching is O(nodes + edges) per
evaluation, which is prohibitive at scale. Replace the object to reload, or call
the exposed `reload()` after mutating it in place. The component also exposes
`getGraph()` and `getCanvas()` via a template ref.

## Composables

`useGraph()` manages an instance against a canvas you own; `useSimulation()`
exposes simulation status, alpha, and start/stop/restart controls. Both clean up
on unmount.

```ts
const { graph, isReady, initialize, load, fitToView } = useGraph();
const { isRunning, alpha, start, stop } = useSimulation({ graph });
```

## Full API

See the [`@graphmother/core` README](https://github.com/tomWhiting/graphmother/tree/main/packages/core)
for graph data formats, the force-algorithm list, layers, and events.

## License

MIT
