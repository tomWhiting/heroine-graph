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

## Cards

At high zoom GraphMother promotes the largest nodes to real DOM elements.
`<GraphCards>` renders a React component into each one, through a portal, so the
cards live in core's overlay — above the canvas, carrying the camera transform —
while their components stay in your tree, with its context, state and hooks.

```tsx
const [graph, setGraph] = useState<GraphMother | null>(null);

<GraphMother
  onReady={(instance) => {
    instance.setLodConfig({ enabled: true });
    instance.setDomOverlay({ enabled: true });
    setGraph(instance);
  }}
/>
<GraphCards
  graph={graph}
  render={({ node, selected }) => (
    <article data-selected={selected}>{node.label ?? String(node.externalId)}</article>
  )}
  fallback={<article>could not render this card</article>}
  onCardError={(error, card) => report(card.key, error)}
/>;
```

The adapter never positions, sizes, orders or evicts a card — core owns all of
that, and `node.position()` / `node.size()` report it live. Cards are keyed on
`node.externalId`, never on `node.id`: the latter is a GPU slot, and slots are
recycled when nodes are removed, so a record keyed on one silently starts
describing a different node.

`onCardError` reports _your_ component throwing during render, which only a React
error boundary can catch. A failure in the card provider itself is core's to
contain and arrives as the graph's `card:error` event. Note that a card component
throwing from an effect after commit is outside any boundary's reach, and will
unmount the tree the graph is in.

`useGraphCards(graph)` is the same bridge without the JSX, for a consumer that
wants to lay the cards out itself.

## Full API

See the [`@graphmother/core` README](https://github.com/tomWhiting/graphmother/tree/main/packages/core)
for graph data formats, the force-algorithm list, layers, and events.

## License

MIT
