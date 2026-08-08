# Changelog

`@graphmother/core` and `@graphmother/wasm` are versioned and published in
lockstep: core depends on the wasm package at the same minor, and a core
release that does not name a matching wasm release is a bug (see 0.3.0).

## Unreleased

### Fixed

- **Bubble mode now separates unrelated subtrees, rather than hoping repulsion
  will.** Three defects stood between the nested-bubble columns and the
  guarantee they exist for. The force passes derived their parent, sibling and
  cousin sets from the graph's _whole_ edge set while `wellRadius` and `depth`
  came from the containment forest — so on a code graph an import edge made a
  file a "parent" of its importers and made those importers "siblings", and
  every separation invariant was stated across two different trees. Repulsion
  and the phantom overlay were mass-weighted in bubble mode, where the well
  radius already encodes subtree size, so size was counted twice and the
  resulting force ran orders of magnitude past the integrator's velocity cap,
  saturating and flipping sign every frame — the jitter that read as a grid
  artifact. And the orbit term pulled children onto a shell that cannot be
  satisfied: evenly spaced children on it sit closer than their own bubbles
  allow at every branching factor of four or more, so it fought sibling
  separation permanently.

  Bubble mode now routes the sibling pass onto the containment forest, drops the
  double-counted mass weighting, keeps only the inward half of the orbit pull,
  and adds a containment barrier that is quadratic in violation depth so both it
  and its gradient vanish at the boundary. A new pass separates forest roots,
  which completes the induction: any two nodes whose lowest common ancestor is
  not a real node now sit inside a chain of nested wells ending in two disjoint
  ones. Outside bubble mode nothing changes.

- **A bubble is now big enough to hold its children.** `wellRadius` sized every
  parent from a single packing efficiency of 0.82 — the density many equal
  circles approach in the limit. Small fan-out is nowhere near that limit and is
  not even monotonic on the way to it: two circles pack at density 0.50, three
  at 0.65, six at 0.67. A directory with two subdirectories was therefore given
  a bubble 28% smaller than the one its own contents require, and no arrangement
  of the subtree could resolve it, because the radius itself was the thing that
  was wrong. Since a code tree branches two and three ways constantly, this
  applied to most of the hierarchy, and it compounded: an under-sized bubble
  under-reports its area to its parent, which is then under-sized in turn.
  Sibling subtrees visibly interpenetrated as a result.

  Parent radii now come from the known optimal packings for fan-out up to 12,
  and from a second bound that summed area cannot express: any two disjoint
  circles of radii `a` and `b` inside a circle of radius `R` force `R >= a + b`,
  so a parent holding two wide subtrees is sized by that pair however little of
  its area they account for. `packing_efficiency` still governs wider fan-out,
  where one constant is a fair estimate.

  Bubbles at low fan-out get wider, so a laid-out graph occupies more space than
  it did. That is the correction, not a side effect: the previous extent was one
  the contents never fit inside.

- **A card is now laid out at its own CSS size and never drawn larger than it.**
  The card box was held in graph units and the overlay container carries the
  camera, so a card was laid out at `size / cameraScale` and magnified back up.
  A card mounts when its node's screen radius crosses `domThreshold`, which puts
  the camera well above scale 1 — so the ordinary case was a card laid out at a
  fraction of the width it appeared to occupy and rendered as a magnified
  picture of itself: text drawn several pixels per pixel, borders equally thick,
  wrapping and `min-width` computed against the wrong number. Zooming further in
  inflated it without bound, since nothing counter-scaled.

  A card is now laid out in the box the caller asked for, and carries a
  counter-scale that cancels the camera. At and above the scale it mounted at it
  renders at exactly its natural size, pixel for pixel, however far the camera
  goes; below that it shrinks with the camera as before, so the swap to and from
  a sprite stays free of a jump in both directions.

  Consumers rendering anything with intrinsic size — an input, a control, a
  framework subtree with its own breakpoints — were the exposed case.

### Breaking

- `cardPlacementAt` takes a `scale` argument before `opacity`, and
  `CardPlacement` carries a `scale`. Both are exported for consumers driving
  `syncDomCards` by hand; a consumer that only registers a `CardProvider` is
  unaffected. `CardPlacement.width`/`height` are now the layout box in CSS
  pixels rather than an extent in graph units — the rendered extent is
  `width * scale`.

## 0.3.2

### Fixed

- **A card whose slot came to hold a different node is now released and
  re-mounted.** Cards are keyed by GPU slot and `removeNodesBatch` compacts
  slots, so a surviving node could move into a slot that was already carded.
  Every `CardNode` accessor reads through to whatever the slot holds now, so
  the card went on standing for a node that had left — one node's content at
  another node's position, with no release, no re-mount, and nothing the
  provider could observe. The check runs ahead of `minCardLifetimeMs`: that
  floor exists to stop a _correct_ card flickering.

  Consumers that key their own per-card state on `externalId` were the exposed
  case, and any consumer calling `removeNodesBatch` while the DOM overlay is
  enabled should take this release.

### Known gap

- `{ kind: "data" }` is declared on `CardChange` and is still emitted by
  nothing; only `hover` and `selection` reach a provider's `update`.

## 0.3.1

### Fixed

- **`core@0.3.0` is broken — do not use it.** It declares
  `@graphmother/wasm: ^0.2.0`, so a fresh install resolves wasm 0.2.1, which
  predates the `aggregateLodEdges` and `computeBubbleData` exports the LOD
  controller calls. The library throws at runtime as soon as LOD evaluates.
  0.3.1 corrects the range to `^0.3.0`.
- The runtime `VERSION` literal is now rewritten by the release script, so a
  build can no longer report the previous release.

## 0.3.0

### Breaking

**Twelve exports were removed from the public surface.** The version was bumped
from 0.2.1 to 0.3.0 for exactly this reason — 0.2.1 is published, so the
removals could not ship under it.

If you are moving from 0.2.x, this is the boundary to read: check your imports
against the current `mod.ts` before upgrading. Nothing was renamed, so an
import that still resolves still means what it meant.

**Upgrade note:** go straight from 0.2.1 to 0.3.1 or later. 0.3.0 itself is
unusable for the dependency reason above.

### Added

- Semantic LOD: `setLodConfig`, `setLodPolicy`, `setLodFocus`, `expandNode`,
  `collapseNode`, `getVisibleNodes`, `getVisibleAncestor`, and the `lod:change`
  / `node:collapse` / `node:expand` events. Folding hides nodes rather than
  removing them and never reheats the simulation, so positions survive a fold
  exactly.
- The DOM card overlay: `setDomOverlay`, `setCardProvider`, `syncDomCards`, and
  the `CardProvider` contract.
- `getExternalId` / `getNodeId`, so a consumer can key its own records on the
  producer's identifier instead of on a recycled GPU slot.

### Fixed

- The LOD cut is re-evaluated when the graph itself changes, not only when the
  camera moves. Zoom was the sole geometric input, so a host mutating a
  standing graph re-evaluated nothing until the camera next moved — under a
  streaming producer that was the steady state, not a moment of lag.
- `maxStorageBuffersPerShaderStage` is requested at 8, the WebGPU default,
  rather than 10. Exceeding the device limit invalidated the bind group layout
  and silently dropped the frame.
- Algorithm correctness across the board — GPU radix sort, Karras tree
  aggregation, ForceAtlas2/LinLog calibration, t-FDP's force floor, Relativity
  Atlas density-field quantisation, per-edge force races in the attraction
  shaders, and the Tidy Tree apportion walk. `alphaDecay` now defaults to
  0.0228 (d3's convention, ~300 ticks) instead of 0.0002.

## 0.2.1 and earlier

No changelog was kept. 0.2.1 is the last release before the LOD surface exists;
`setLodConfig` and everything in its family are absent there.
