# Changelog

`@graphmother/core` and `@graphmother/wasm` are versioned and published in
lockstep: core depends on the wasm package at the same minor, and a core
release that does not name a matching wasm release is a bug (see 0.3.0).

## 0.5.0

The card seam finishes: a card can now hold an arbitrary component — an editor, a
form, anything with its own layout and its own scrolling — rather than a caption.
Three things stood in the way, and each was fatal on its own.

A minor rather than a patch because every entry below is a new export.
`@graphmother/wasm` is republished at 0.5.0 with no Rust changes, because the
lockstep rule at the top of this file is what makes `@graphmother/core@^0.5.0`
resolvable against a wasm package at the same minor.

### Added

- **`CardProvider.size`** — the provider says how big its card needs to be.
  Asked once, immediately before `mount`, and fixed for the card's whole life, so
  content never reflows under the user's pointer; it is also the box the camera
  counter-scales *to*, so past the mount scale a card that asked for 480×320 is
  480×320 CSS pixels on screen, pixel for pixel, however far the camera zooms in.
  Previously every card was laid out in a box core picked without knowing what
  was in it — 200×120, which is a caption, not an editor. A throw costs the size
  and not the card: the default is a usable answer, and `cardFailureForfeitsCard`
  already answered `false` for anything that is not `mount` or `update`. New:
  `CardProviderHook` gains `"size"`, and `CardDriver.measure`.
- **`CARD_SCROLL_ATTRIBUTE`** (`data-graphmother-scroll`) — a card region that
  claims the wheel. The overlay forwards every wheel over a card to the canvas so
  that zooming back out from a card works, which meant a scrollable component
  inside a card could never be scrolled: the gesture zoomed the graph and
  `preventDefault` stopped the browser scrolling it. A wheel inside a claimed
  region is now left completely alone — asked before anything is prevented,
  because a region that never sees the event cannot scroll. Declared per
  *element*, following `CARD_DRAG_HANDLE_ATTRIBUTE` exactly, so a card can hold a
  scroll region and still zoom from its own chrome.
- **`GraphMother.holdCard` / `unholdCard` / `getHeldCards`**, and
  `LODController.holdCard` / `unholdCard` / `heldCards` — keep a card open across
  the cut. Focus and `force-card` already card a node whatever its screen size,
  but all three tests are only consulted for nodes *in* the cut, so zooming out
  until an ancestor folded over the node destroyed the card underneath it. That
  is right for a label and wrong for an editor, and no threshold setting avoids
  it, because folding is what zooming out is for. A held card survives the cut;
  the graph folds normally behind it. It deliberately does not expand the node's
  ancestors — that would un-fold a whole directory to keep one card, and the
  damage would scale with fan-out.
- **`CardStoreOptions.size`**, and `size` on `useGraphCards` / `<GraphCards>` —
  the same seam for declarative renderers. It has to be a callback on the store
  rather than something the renderer decides, because the box is fixed before any
  DOM exists and before React has rendered anything into the card; without it
  every store-backed card, which is every React card, was stuck at the default
  box. `createCardStore()` with no arguments is unchanged.
- **`CardSyncEntry.at`** — a placement override, for a card held over a node the
  layout has folded away. Such a node is frozen where the fold began while the
  proxy standing for it has gone on moving, so an uncorrected card drifts off the
  bubble that swallowed it by exactly the distance the expand would later pay
  back. Re-read every sync, and it does not change `CardNode.position`: the card
  is displaced, the node is not.

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
- **`@graphmother/react` is published**, from 0.4.0 on. It has existed in the
  repository for some time and was never in the publish set, so `<GraphMother>`,
  `<GraphCards>` and the hooks were unreachable from npm — including the React
  card adapter this release's notes describe.
- **`createCardStore`**, with `CardStore` and `LiveCard`: the card contract as a
  subscribable list, for declarative frameworks that render cards themselves
  rather than into the element `mount` is handed. `@graphmother/react`'s
  `<GraphCards>` is built on it.

### Fixed

- **A new minor can be released at all.** `deno task publish` began with a
  `deno install`, which resolved core's dependency on `@graphmother/wasm` from
  npm — at the exact version the run existed to put there. Every first release
  of a minor therefore failed before it built anything. packages/wasm and
  packages/core are now npm workspace members, so that dependency resolves to
  the sibling directory. The wasm package publishes from `packages/wasm` with a
  checked-in manifest rather than from the wasm-pack output directory that
  `build.sh` used to patch after the fact.
- **`VERSION` reported 0.3.2 in a 0.4.0 build.** The constant lives in
  `packages/core/src/api/factory.ts` and the 0.4.0 bump was done by hand, which
  is exactly how 0.3.0 went wrong. `tests/unit/release_manifest_test.ts` now
  holds every manifest, the caret range core asks wasm for, the workspace
  membership above, and the wasm tarball's file list against Cargo.toml.
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

- **A semantic-LOD focus set now survives a node removal.** `setLodFocus` takes
  slot numbers, and `removeNodes` compacts the slot space, so a focus declared
  once went on naming the numbers rather than the nodes: after a compaction it
  carded whichever nodes had moved into those slots, and a node the removal
  deleted kept a card standing over a stranger. The focus set is now rewritten
  through the same slot remap the pins already moved with, so a host that
  declared its focus once never has to re-declare it, and a focused node the
  removal deleted is dropped rather than left pointing somewhere.

- **A live fold no longer loses its accumulated drift when the topology
  changes.** A collapsed proxy keeps simulating while the subtree folded into it
  is frozen, and the difference is paid out to the descendants when the fold
  opens. That payment is only expressible against the slot space and the
  hierarchy the fold was measured in, and every path that drops the derived
  topology ends both — so each add, remove or algorithm switch silently
  discarded whatever was outstanding, and the losses accumulated in the
  descendants' positions across a stream of mutations until a folded subtree
  reappeared detached from its parent. Every such path now settles the drift
  first, and a mutation that also moves the slot space settles it before the
  move. A streaming producer mutating under a hot simulation pays one position
  write per live fold per mutation, bounded by the drift since the previous one.

- **A fold is no longer translated by a delta measured across a compaction.**
  Between the moment a host declares a slot remap and the moment the rebuilt
  hierarchy is adopted, a proxy's anchor names the outgoing slot space while its
  position names the new one, so their difference is not a drift and the
  descendants it would move are not the ones that were folded. Every route to a
  fold fix-up now declines across that seam rather than translating subtrees by
  an arbitrary distance — including disabling LOD, which restores every fold and
  reaches it without consulting the hierarchy on the way in.

- **A forest wider than `maxVisibleNodes` is no longer a bare layer of
  proxies.** Roots are admitted unconditionally — nothing sits above them to
  fold into — so a graph with more roots than the ceiling exceeded it before the
  first expansion was even considered, and every expansion after that was
  refused for overrunning a budget already overrun: no zoom level opened
  anything. The cut is now built against the root layer plus
  `LodConfig.rootDetailReserve` of it once the configured ceiling has stopped
  being achievable — which is strictly once the root layer _exceeds_ it, since a
  root layer equal to the ceiling is a cut that honours it exactly. Up to and
  including that point the ceiling binds as before, and `LodChangeEvent.budget`
  reports which of the two regimes a cut came from.

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

- `LodConfig` carries a `rootDetailReserve`, defaulting to `0.25`: how far past
  an unachievable `maxVisibleNodes` the cut may go, as a fraction of the root
  layer. Set it to `0` for a `maxVisibleNodes` that is an absolute cap, at the
  cost of a forest wider than the ceiling never opening. Additive for a caller
  patching the config through `setLodConfig`, breaking for one constructing a
  whole `LodConfig` literal.

- `LodChangeEvent` carries three further members: `totalCount`, the nodes in the
  hierarchy the cut was drawn from; `budget`, the ceiling the cut was actually
  built against; and `hiddenByReason`, the undrawn nodes attributed to what
  folded the proxy standing over them, summing to `totalCount - visibleCount`.
  Additive for a listener, breaking for anything constructing the event — a test
  double, or a host re-emitting it — which must now supply all three.

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
