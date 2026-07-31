# @graphmother/react

React wrapper for [GraphMother](https://github.com/tomWhiting/graphmother) —
WebGPU-accelerated visualization for large graphs.

The `<GraphMother />` component owns the canvas, the WebGPU device, and the
render loop: it initializes on mount, forwards graph events to props, reloads
when `data` changes, resizes with its container, and disposes everything on
unmount.

## Install

```bash
npm install @graphmother/react @graphmother/core
```

`react >= 18` is a peer dependency. The component and hooks are client-only and
carry the `"use client"` directive, so they work inside a React Server
Components app without extra wrapping.

## Usage

```tsx
import { useRef } from "react";
import { GraphMother, type GraphMotherRef } from "@graphmother/react";

export function Graph() {
  const ref = useRef<GraphMotherRef>(null);

  return (
    <GraphMother
      ref={ref}
      data={{
        nodes: [{ id: 1 }, { id: 2 }],
        edges: [{ source: 1, target: 2 }],
      }}
      style={{ width: "100%", height: "600px" }}
      onNodeClick={(e) => console.log("clicked", e.nodeId)}
      onReady={(graph) => graph.setForceAlgorithm("barnes-hut")}
    />
  );
}
```

`ref.current.getGraph()` returns the underlying `GraphMother` instance for
anything the props do not cover.

The `config` prop is consumed once, when the graph is created. The component
re-initializes only when the config's _values_ change, so passing an inline
object literal is safe.

## Hooks

`useGraph()` manages an instance against a canvas you own; `useSimulation()`
exposes simulation status, alpha, and start/stop/restart controls.

```tsx
const { graph, isReady, initialize, load, fitToView } = useGraph();
const { isRunning, alpha, start, stop } = useSimulation({ graph });
```

## Full API

See the [`@graphmother/core` README](https://github.com/tomWhiting/graphmother/tree/main/packages/core)
for graph data formats, the force-algorithm list, layers, and events.

## License

MIT
