# GraphMother — Code Graph

A focused code-repository visualiser: one dataset shape, one layout, one detail
model, five knobs. It is the reference consumer of GraphMother's typed input
path, nested-bubble layout, semantic LOD and DOM card overlay, and it is meant
to be read as much as run.

## What it shows

A synthetic repository — `repo → directories → files → symbols`, with
cross-cutting `import` edges laid over the containment tree — loaded through the
**typed fast path** with producer-supplied columns, then laid out and thinned by
the four features that exist for exactly this workload:

| Feature                                                    | What it does here                                                                                                                                                                  |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Typed input** with `hierarchy`, `tag`, `weight`          | The producer ships the containment tree, node kinds and importance as flat slot-indexed columns. Nothing is re-derived at load.                                                    |
| **Nested-bubble layout** (`relativity-atlas`, bubble mode) | Every node has a well radius computed bottom-up from its subtree. Unrelated subtrees cannot overlap, and depth-decaying gravity keeps symbols from being dragged past their files. |
| **Semantic LOD**                                           | Subtrees too small to read fold into a single proxy; the cut unfolds as the camera comes in. A policy adds two things geometry cannot know (below).                                |
| **DOM cards**                                              | At high zoom leaves stop being sprites and become real DOM: selectable path text, and a title row that drags the node exactly as dragging its sprite does.                         |

The demo's LOD policy overrides core's geometric rule in exactly two places, in
`src/lod_policy.ts`:

- a repository root never folds — folded, it is one sprite for the whole graph;
- a directory with a single child is dropped from the cut entirely, because
  `src/main/java/...` chains say nothing their children do not.

### Interaction

- **Click** a node — declares it the LOD focus, so it is carded whatever its
  screen size says. Click the background to clear.
- **Double-click** a node — folds or unfolds its subtree.
- **Drag** a card by its title row, or a sprite directly. Both pin the node.
- **Scroll / drag** the background — zoom and pan.

## Running it

From the repository root:

```bash
deno task build:wasm     # once: builds packages/wasm/pkg
deno task build          # once: builds dist/, including the label font atlas
deno task example:mission-control
```

Then open the printed URL in a WebGPU-capable browser (Chrome or Edge 113+, or
Safari 18+). `dist/` is the example's Vite `publicDir`, which is where the MSDF
font atlas the label layer needs comes from; without it the demo still runs and
logs that labels are unavailable.

To produce a static build:

```bash
cd examples/mission-control && dx vite build
```

`dx` is Deno's `npx` equivalent. Vite 8 needs Deno 2.9 or newer; on an older
Deno it fails to start with a `node:util`/`parseEnv` error.

## The five knobs

| Knob                               | What it changes                                                                                                                                                                                                                                                                                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Repository size**                | Regenerates the dataset at 2 500, 35 000 or 220 000 nodes. 2.5K is the default demo repo; 35K is the Meridian-sized target with symbols; **220K is a stress case** — it takes seconds to build, and it is past the point where the graph is pretty. The sizes match `tests/fixtures/code_tree.ts`, so the demo and the physics harness argue about the same workload. |
| **Semantic LOD** (switch + slider) | The switch is `LodConfig.enabled`. The slider moves the expand and collapse thresholds _together_, keeping the hysteresis band between them: right is finer detail, i.e. a lower screen extent at which a subtree unfolds. The ratio between the two is fixed, because it is a stability property and not a matter of taste.                                          |
| **DOM card budget**                | `LodConfig.maxCards` — how many nodes may be promoted to DOM at once. Set it to 0 to see the same graph with GPU sprites only.                                                                                                                                                                                                                                        |
| **Label density**                  | `LabelConfig.maxLabels` — how many GPU labels are drawn at once. It also sizes the _candidate_ set the label layer ranks each time the nodes move (see the note in `src/knobs.ts`).                                                                                                                                                                                   |
| **Simulation**                     | Pause / resume, and **Reheat**. Reheat is the only thing in this app that raises simulation alpha, and it is wired to an explicit button press: no camera move, LOD transition or card mount reheats the layout.                                                                                                                                                      |

## The readout

Driven by `lod:change`, `node:collapse`, `node:expand` and `simulation:tick`:

`nodes` · `edges` · `visible` (nodes in the current LOD cut) · `folded` (nodes
standing in for a folded subtree) · `cards` (mounted DOM cards) · `tick` ·
`fps`.

`tick` is the smoothed wall-clock interval between `simulation:tick` events. It
is **not** GPU time — the library exposes no timestamp queries — so it reports
the simulation's cadence, not the cost of a dispatch.

## Layout

```
main.ts              wiring only: bootstrap, load, event routing
src/repo.ts          the producer — stands in for a real indexer
src/knobs.ts         what each knob means in library terms
src/lod_policy.ts    the two structural overrides
src/hud.ts           the readout's arithmetic
src/cards.ts         the card renderer
src/panel.ts         the controls and the readout, as DOM
index.html           markup and styles
vite.config.js       WGSL loader, the wasm alias, dist/ as publicDir
```

`repo.ts`, `knobs.ts`, `lod_policy.ts` and `hud.ts` are pure and have unit tests
under `tests/unit/example_*_test.ts`. They import types from the core modules
that declare them rather than from `packages/core/mod.ts`: the barrel is a
bundler entry point (it imports WGSL as text) and cannot be resolved by the Deno
test runner. `main.ts`, `cards.ts` and `panel.ts` use the public barrel, as a
consumer would.

## What this example deliberately does not do

There is one dataset shape, one layout algorithm and no theme picker. Anything
that would demonstrate a feature by adding a control instead of by using it has
been left out; the other examples in `examples/` cover the individual layers.
