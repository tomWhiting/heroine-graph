# Semantic LOD and DOM cards

At low zoom a code repository is 35,000 sprites and no structure. At high zoom a
sprite is a coloured dot where the user wants selectable text, a working link
and find-in-page. GraphMother answers both with one mechanism: nodes are folded
into their ancestors as the camera pulls out, and promoted to real DOM elements
as it pushes in.

This is the consumer guide to both. The division of responsibility runs through
all of it and is worth stating once, up front:

> **Core owns which nodes are carded, where each card sits and how big it is.
> The consumer owns everything inside a card.**

## The two metrics

Semantic LOD has four thresholds, and they are two independent pairs over two
different measurements of two different sets of nodes. Reading them as one
ordered ladder is the most common way to mis-tune the system.

| Metric | Measured on | Thresholds | Defaults |
| --- | --- | --- | --- |
| Subtree screen extent — `wellRadius × zoom`, in px | Interior nodes (nodes with children) | `expandThreshold` / `collapseThreshold` | 96 / 64 |
| Leaf screen radius — the node's render radius × zoom, in px | Every node in the cut | `domThreshold` / `domExitThreshold` | 48 / 32 |

The first decides whether a subtree stands for itself as a single bubble or
shows its children. The second decides whether a node is drawn as a GPU sprite
or as a DOM card. A node can be carded without its parent being expanded, and a
subtree can be expanded without anything in it being large enough to card. They
do not form a sequence.

Both metrics are functions of zoom alone. Panning cannot move the cut; it is
worth an evaluation only because the card set is culled to the camera and
because an LOD policy may read `onScreen`.

### Each pair is a band, not a value

A single threshold makes the state a discontinuous function of a noisy metric,
and the noise becomes flicker. So each pair is an entry/exit band: a node
expands at `expandThreshold` and does not collapse again until it falls below
`collapseThreshold`; it takes a card at `domThreshold` and does not give it back
until it falls below `domExitThreshold`. `setLodConfig` clamps an exit threshold
that was set at or above its entry threshold, because a band with no interior
has no defined behaviour.

### What hysteresis cannot answer

Hysteresis answers a camera oscillating across a boundary. It does not answer a
subtree that *resizes* while the camera is perfectly still — which is the
ordinary case under a hot simulation, where well radii and positions are both
moving. That is what `minBandCommitFrames` (default 6) is for: a node's
expansion state holds for that many frames after it changes, whatever the metric
does in the meantime. Imperative calls (`expandNode`, `collapseNode`) and
explicit configuration changes bypass the window, because the user asked.

### The prefetch ring

`prefetchRatio` (default 2) widens the DOM band and the camera rect at once. A
node is offered an advisory, abortable `prefetch` once its leaf screen radius
reaches `domThreshold / prefetchRatio` — 24 px at the defaults — and once it is
inside the viewport rect widened by the same ratio. No DOM exists at that point
and `mount` may never follow.

```ts
graph.setLodConfig({
  enabled: true,
  expandThreshold: 96,
  collapseThreshold: 64,
  domThreshold: 48,
  domExitThreshold: 32,
  prefetchRatio: 2,
  transitionMs: 150,
  maxCards: 150,
  minCardLifetimeMs: 400,
  minBandCommitFrames: 6,
});
```

## The provider contract

A card provider is four callbacks. Only `mount` is required.

```ts
graph.setDomOverlay({ enabled: true });
graph.setCardProvider({
  mount(container, node) {
    const root = document.createElement("article");
    root.style.width = "100%";
    root.style.height = "100%";
    root.textContent = node.label ?? String(node.externalId);
    container.appendChild(root);
    return { root };
  },
  update(container, node, change, state) {
    if (change.kind === "selection") state.root.dataset.selected = String(change.selected);
  },
  release(container, node, state) {
    state.root.remove();
  },
});
```

The rules, in the order they bite:

- **`mount` is called exactly once per card**, with a container core has already
  created, attached, positioned and sized. The container is empty.
- **Core writes four style properties on the container and nothing else**:
  `transform`, `width`, `height` and `opacity`. It never touches the
  container's children. Everything inside belongs to the provider.
- **The provider must remove its own children in `release`.** Containers are
  pooled, and a container handed back with children still in it is dropped
  rather than reused — so a provider that does not clean up costs an element per
  card instead of leaking one card's content into the next.
- **`release` runs while the container is still in the DOM**, so teardown can
  read layout. Core detaches it afterwards.
- **`prefetch` is advisory and abortable.** It fires at most once per node per
  LOD epoch, receives an `AbortSignal` that fires when the node leaves the ring
  or its card is released, and must have no visible side effects.
- **`update` reports one typed change at a time.** `data`, `selection` and
  `hover` arrive when the graph says so; `position` and `size` arrive from the
  per-frame placement pass, so they can arrive on every frame of camera motion.
  Treat them as cheap or ignore them — core has already moved the container.

A child carrying the `data-graphmother-drag` attribute becomes the card's drag
handle: a pointer gesture on it moves the node exactly as dragging its sprite on
the canvas does. It is opt-in because a card whose whole surface dragged the node
would have no selectable text, which is most of the reason cards exist.

Register no provider and a built-in one renders the node's label, tag, weight and
depth — plain DOM, inline styles, no network access of any kind.

A child carrying `data-graphmother-scroll` claims the wheel for itself: a wheel
that lands inside it is neither forwarded to the canvas nor `preventDefault`ed,
so the browser scrolls the region natively instead of the graph zooming. It is
declared per *element*, not per card, so an editor can hold a scroll region and
still zoom the graph from its own chrome.

## How a card is sized

A card is **laid out at its natural CSS size** — whatever `CardProvider.size`
asks for, else `CardSyncEntry.size`, else 200×120 — and given a counter-scale
that cancels the camera. The container
sits inside an overlay element carrying the camera as a CSS transform, so with a
camera scale `S` and the scale `S₀` the card mounted at, the card renders at

    natural × S × cardCounterScale(S, S₀)  =  natural × min(1, S / S₀)

Two regimes fall out of one expression:

- **Zoomed in past the mount scale** (`S ≥ S₀`, the ordinary case, since a card
  mounts on crossing *into* the DOM band): net scale is exactly 1. The card
  renders at its own CSS size, pixel for pixel, however far the camera goes. Its
  text is 13px text, its borders are one pixel, and its layout is computed at the
  width it appears to occupy.
- **Zooming back out before the card is dropped** (`S < S₀`): net scale is
  `S / S₀`, so the card shrinks with the camera exactly as the sprite it replaced
  would have. That is what keeps the swap free of a jump in both directions.

The layout box is fixed at mount and never changes, so a provider's content
never reflows under the user's pointer. `CardNode.size()` reports the size the
card is currently *drawn* at, which agrees with the layout box from the mount
scale upwards.

The provider gets first say. `CardProvider.size` is asked once, immediately
before `mount`, and its answer is the box for the card's whole life — so past
the mount scale a card that asked for 480×320 is 480×320 CSS pixels on screen,
pixel for pixel, however far the camera zooms in. It exists because 200×120 is a
caption, not an editor, and a component that has to lay itself out cannot do so
in a box core picked without knowing what was in it. `node.size()` is not
meaningful inside the hook — it reads the box the call is deciding. A throw
costs the size, not the card: the default is a usable answer.

## Holding a card open

`graph.holdCard(slot)` keeps a node's card alive until `unholdCard` gives it
back, and `getHeldCards()` reports the set.

This is not the same as focus. Focus and a policy's `force-card` verdict already
card a node whatever its screen size, and both outrank the budget — but all
three tests are only ever consulted for nodes *in the cut*, so zooming out until
an ancestor folds over the node destroys the card underneath it. For a label
that is the right answer: the thing it labelled is no longer on screen. For an
editor the user is typing into it is not, and no threshold setting avoids it,
because folding is what zooming out is for.

A held card survives the cut. The graph folds normally behind it and the card
stays open. It does **not** expand the node's ancestors — expanding one gathers
every child of it, so a single open card would undo the zoom-out for its whole
directory, and the damage would scale with fan-out.

A folded subtree stops simulating, so a held node's stored position goes stale by
however far its proxy has drifted since the fold began. The card is offset by
exactly that delta, which keeps it on the bubble that swallowed its node and
leaves it in place at the moment the fold ends. `CardNode.position()` still
reports the node's true graph position: the card is displaced, the node is not.

A hold is worth `LOD_FORCE_CARD_PRIORITY` under the budget, so only more held
cards than `maxCards` can displace one. It survives a compaction —
`remapSlots` carries it to wherever the node moved — and is dropped when the node
is deleted.

## `externalId` versus the node slot — the one thing to get right

`NodeId` is a GPU slot index. It is meaningful only inside one GraphMother
instance, and **it is recycled**. A batch removal compacts slots, so a node that
survives the removal can end up in a slot that previously held a different node.

Consequently:

> Key every record you keep about a node on `CardNode.externalId` — the
> identifier your producer supplied — and never on `CardNode.id`.

A consumer map keyed on the slot silently starts describing a different node
after the first `removeNodesBatch`. Nothing detects this for you; the value is
still a valid slot and still resolves.

Core handles its own half: a card whose slot has changed hands is released and
re-mounted for the slot's new occupant in the same sync, ahead of the
anti-flicker floor, because that floor exists to stop a *correct* card
flickering and has no business holding a wrong one.

The two conversions, when you need them:

```ts
const externalId = graph.getExternalId(slot);   // slot → producer id
const slot = graph.getNodeId("src/mod3.ts");    // producer id → slot
```

`getNodeId` returns a value that is valid only until the next mutation.

## Budget and lifetime

Two knobs bound what the overlay costs:

- `maxCards` (default 150) — the ceiling on simultaneously mounted cards.
- `minCardLifetimeMs` (default 400) — the floor on how long a card stays once it
  exists. A card younger than this cannot be evicted, even by a higher-priority
  newcomer and even past `maxCards`; any overshoot drains within one lifetime. A
  budget honoured by making cards blink is not worth honouring.

Both appear in `LodConfig` *and* in `DomOverlayConfig`, because both sides need
them: the controller ranks and truncates the set it declares, and the overlay
admits and holds what it is handed. They are not two knobs. `setLodConfig`
forwards its resolved values to the overlay, so set them there and the two
cannot disagree.

## When a provider throws

A provider is consumer code running inside core's per-frame path. If it throws
and nothing contains it, the exception leaves the overlay through the LOD
controller's tick at the top of the render callback — and the frame aborts
before edges, nodes, compute submit and position readback. One bad card would
freeze the whole graph while the loop spins.

So core contains all four hooks, and containment is defined rather than merely
survivable:

| Hook | What core does | `released` | Quarantined |
| --- | --- | --- | --- |
| `prefetch` | Abandons the offer; the node stays spent for the epoch | `false` | no |
| `mount` | Detaches the container and offers it back to the pool; the card is never registered | `false` | yes |
| `update` | Releases the card — a provider whose update threw is in a state it did not intend, and `place` runs every frame | `true` | yes |
| `release` | Detaches the container anyway, rather than leaving a live card wedged on screen | `true` | no |

Every failure is announced. If anything is listening for `card:error` the event
is emitted; if nothing is, the wrapped error is logged. Containment is never
silence.

```ts
graph.on("card:error", (event) => {
  event.nodeId;      // GPU slot
  event.externalId;  // your identifier for that node
  event.hook;        // "prefetch" | "mount" | "update" | "release"
  event.error;       // GraphMotherError, code 8001
  event.cause;       // exactly what the provider threw
  event.released;    // whether the node lost a card
});
```

**A node whose provider could not card it is not offered to it again.** A
provider that threw from `mount` or `update` will throw for that node again, and
retrying costs one failure per node per evaluation for as long as the graph is
open — `update` runs on every frame of camera motion, so there it is a throw per
frame. The node returns to being an ordinary sprite: the LOD controller stops
declaring it and fades its sprite back in, so a failed card leaves a node rather
than a hole.

The other two hooks cost the node nothing. `prefetch` is advisory and abortable —
a speculative fetch that raised says nothing about whether `mount` would have
worked, and the node was free to be ignored anyway — so it is reported, spent for
the epoch, and offered a card the moment one is wanted. `release` runs *after* a
card the provider built and core displayed, so it is a teardown defect rather
than an inability to render; the node is carded again the next time it qualifies.

The quarantine lifts on three things:

- `setCardProvider(...)` — a new provider is a new answer to the question the old
  one got wrong. This also clears the controller's suppressions, so nodes that
  still qualify take their cards back immediately.
- `graph.load(...)` — a reload releases every card, and the quarantine goes with
  them.
- **The slot changing hands.** Cards and the quarantine are keyed by GPU slot,
  and `removeNodesBatch` compacts slots, so a survivor can land in a slot some
  other node's card failed in. The quarantine records the producer id it was
  taken against and is checked against the slot's current occupant, so it follows
  the node rather than the index — a survivor is never denied a card for a
  failure that was not its own. On the controller's side, where there are no
  producer ids to compare, any topology change lifts the suppressions instead;
  the retry that costs is refused by the overlay's quarantine, which does have
  them.

A failure in your own *rendering* is a different thing and core cannot see it —
by the time a framework renders into the host element, `mount` has long since
returned. In React, `<GraphCards>` wraps every card in an error boundary for
precisely that case; see `packages/react/README.md`.

## Related reading

- `packages/core/src/overlay/types.ts` — the contract, with the rule behind each
  callback.
- `packages/core/src/overlay/projection.ts` — the sizing derivation above, stated
  arithmetically.
- `packages/core/src/lod/controller.ts` — the six load-bearing properties of the
  state machine, none of which is expressible in a type.
