# Changelog

`@graphmother/core` and `@graphmother/wasm` are versioned and published in
lockstep: core depends on the wasm package at the same minor, and a core
release that does not name a matching wasm release is a bug (see 0.3.0).

## 0.4.0

A minor rather than a patch because the card surface grows: `createCardStore`,
the `card:error` event and the failure types below are new exports, and
`CardDriver.mount` changes its return type. `@graphmother/wasm` is republished at
0.4.0 with no Rust changes, because the lockstep rule above is what makes
`@graphmother/core@^0.4.0` resolvable against a wasm package at the same minor.

### Added

- **`card:error`, and a defined answer to a card provider that throws.** A
  provider is consumer code running inside core's per-frame path; uncontained,
  one card that threw stranded an attached container in the overlay and took the
  render callback down with it on every subsequent frame — edges, nodes, compute
  submit and position readback all aborted, so one bad card froze the graph while
  the loop spun. All four provider hooks are now contained at the single place
  the contract is enforced, the card is left in a defined state, and the failure
  is announced: emitted as `card:error` when anything is listening, logged when
  nothing is. New: `ErrorCode.CARD_PROVIDER_FAILED`, `Errors.cardProviderFailed`,
  `CardErrorEvent` and its `EventMap` entry, `CardProviderHook`,
  `CardProviderFailure`, `cardFailureForfeitsCard`,
  `CardDriverOptions.onProviderError`, `DomCardOverlayOptions.onCardFailure`,
  `DomCardOverlay.failedCards`, and
  `LODController.suppressCard`/`clearCardSuppressions`. See
  `docs/lod-and-cards.md`.
- **`createCardStore`**, with `CardStore` and `LiveCard`: the card contract as a
  subscribable list, for declarative frameworks that render cards themselves
  rather than into the element `mount` is handed. `@graphmother/react`'s
  `<GraphCards>` is built on it.

### Fixed

- **A broken card no longer leaves a hole in the graph.** The overlay stops
  offering a node whose provider failed, and the LOD controller stops declaring
  it — because the controller had already faded the node's sprite out under a
  card that was never going to exist, so containment inside the overlay alone
  left the node drawn as neither sprite nor card, and consuming a slot of
  `maxCards` no other node could use. `setCardProvider` is the way back and
  reaches both halves.
- **A quarantined slot no longer bans its next occupant.** Cards and the
  quarantine are keyed by GPU slot and `removeNodesBatch` compacts slots, so a
  survivor could land in a slot another node's card had failed in and never be
  carded again — with no event, no diagnostic, and only global acts as a way
  back. The quarantine now records the producer id it was taken against and is
  checked against the slot's current occupant. Consumers calling
  `removeNodesBatch` with the overlay enabled were the exposed case.
- **A `prefetch` or `release` that threw no longer costs the node its card.**
  Both were quarantining, which contradicted the contract they are documented
  under: `prefetch` is advisory and abortable, so a speculative fetch that raised
  says nothing about whether `mount` would have worked, and `release` runs after
  a card the provider built and core displayed. Only `mount` and `update` — the
  two a card cannot exist without, and `update` runs on every frame of camera
  motion — now forfeit it. The rule is exported as `cardFailureForfeitsCard` so
  the overlay and the controller cannot disagree about it.
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
- `CardDriver.mount` returns `boolean` — whether the card exists — rather than
  `void`. A caller keeping its own record of the card must roll that record back
  on `false`. Only consumers driving `CardDriver` directly are affected; the
  overlay is the only caller inside core.
- `Errors.cardProviderFailed` takes a `CardProviderHook` rather than a `string`.
  A caller passing one of the four hook names is unaffected; anything else was
  always producing a message that named a callback core never called.

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
