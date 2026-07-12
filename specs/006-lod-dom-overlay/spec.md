# Feature Specification: Semantic Level-of-Detail & DOM Node Overlay

**Feature Branch**: `006-lod-dom-overlay`
**Created**: 2026-07-13
**Status**: Draft
**Input**: Zoom-dependent level-of-detail rendering where the graph renders collapsed hierarchy bubbles when zoomed out, expands them progressively while zooming in, and — past a close-zoom threshold — individual nodes transform from GPU-rendered sprites into live, interactive HTML DOM elements (cards) that stay position-locked to the simulation. The node does not get a popup or a sidebar; the node itself changes its nature. GPU for mass, DOM for meaning.

## Vision

No existing graph library offers this. DOM-based tools (React Flow et al.) give rich HTML nodes but collapse under ~1K elements; GPU engines (Cosmograph, cosmos.gl, heroine-graph today) scale to 100K+ nodes but render dumb sprites at every zoom level. This feature bridges the two: at 35K nodes (Meridian scale) the GPU renders everything; at reading distance, the handful of visible nodes become real HTML — selectable text, working links, syntax-highlighted previews, screen-reader accessible, ⌘F findable.

The hierarchy that drives semantic LOD is the same containment tree that drives the Nested Bubble layout algorithm (see CLAUDE.md): well radii computed by `computeBubbleDataFromEdges` define both *where* subtrees sit and *when* they expand. One data structure, two consumers.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Semantic Zoom via Bubble Hierarchy (Priority: P1)

A developer views a 35K-node code repository graph. Fully zoomed out, they see ~50 top-level module bubbles — not 35K overlapping dots. Zooming toward one module, its bubble opens into directory bubbles, then files, then symbols. Zooming back out re-collapses them.

**Why this priority**: Biggest visual and performance payoff at target scale; prerequisite mental model for the DOM handoff (it establishes "zoom changes representation").

**Independent Test**: Load the 35K fixture, set zoom levels programmatically, assert the rendered node count and which depth bands are visible at each level.

**Acceptance Scenarios**:

1. **Given** a hierarchical graph with bubble data, **When** zoom is at minimum, **Then** only depth-0/1 nodes render, each drawn at its subtree centroid with size derived from its well radius
2. **Given** a collapsed parent bubble, **When** its screen-space radius exceeds the expansion threshold (`wellRadius * zoom > T_expand`), **Then** its children render and the parent transitions to a hull/outline representation
3. **Given** an expanded subtree, **When** zoom decreases so screen-space radius falls below the collapse threshold (with hysteresis: `T_collapse < T_expand`), **Then** children hide and the parent renders solid again
4. **Given** cross-cutting dependency edges between nodes in collapsed subtrees, **When** both endpoints are hidden, **Then** the edge renders between the lowest visible ancestors (edge aggregation), with width proportional to bundled edge count
5. **Given** any LOD transition, **Then** it animates (fade/scale over ~150ms) — no popping

### User Story 2 - Node-to-DOM Transformation (Priority: P1)

Zoomed close, the eight file nodes in view stop being dots: each becomes an HTML card at the same screen position and apparent size, crossfading in place. The user selects text in a card, clicks a link, scrolls a code preview. Zooming out dissolves cards back into GPU sprites exactly where they were.

**Why this priority**: This is the headline capability — the reason the feature exists.

**Independent Test**: Zoom past `T_dom` on a small fixture; assert overlay container holds one element per visible node, positioned within 0.5px of the GPU-projected position; zoom out and assert elements unmount.

**Acceptance Scenarios**:

1. **Given** zoom crosses `T_dom` upward, **When** a node is inside the viewport, **Then** a DOM element mounts for it, positioned via the shared world→screen transform, and the GPU sprite for that node fades out as the card fades in (crossfade, same center, same apparent size)
2. **Given** mounted cards, **When** the user pans/zooms, **Then** cards track the canvas with no visible lag or drift (single container transform, compositor-only during gesture)
3. **Given** a mounted card, **When** the simulation moves the node (still hot, or dragged), **Then** the card follows the live position, and GPU edges terminate at the card's border rather than its center
4. **Given** zoom crosses `T_dom` downward (with hysteresis band), **Then** cards crossfade back to sprites and unmount, releasing all DOM resources
5. **Given** more than `maxCards` (default 150) nodes qualify, **Then** only the highest-priority nodes (screen-space size, then centrality) get cards; the rest stay sprites
6. **Given** a card is interactive (buttons, text selection), **When** the user drags empty canvas between cards, **Then** the canvas pans; **When** the user drags a card by its handle, **Then** the node moves in the simulation (pin-while-dragging, already GPU-supported via NODE_FLAG_PINNED)

### User Story 3 - Consumer-Rendered Card Content (Priority: P2)

An application developer controls what a card contains. Core never dictates a template: it hands the consumer a container element plus node identity/state and lifecycle events. A React app portals a component in; a Vue app teleports; a server-driven app (Phoenix/LiveView-style, frame) injects server-rendered HTML strings; vanilla JS builds elements directly.

**Why this priority**: The contract is what makes this framework-agnostic — critical for non-React consumers — but a built-in default card (label + type + metadata table) ships first so the feature is demoable without custom code.

**Independent Test**: Register a custom `renderCard` callback; assert it is called with (container, nodeInfo) on mount, `updateCard` on data change, `releaseCard` on unmount; assert the default card renders when no callback is registered.

**Acceptance Scenarios**:

1. **Given** no consumer renderer, **When** a card mounts, **Then** the built-in default card renders (label, type badge, degree, stream values)
2. **Given** a registered card renderer, **When** a node enters DOM range, **Then** core invokes `mount(container, node)` exactly once and never touches the container's children afterwards
3. **Given** server-rendered HTML for a node arrives asynchronously, **When** the consumer sets `container.innerHTML` after mount, **Then** positioning/crossfade behavior is unaffected (core owns the container's transform, never its content)
4. **Given** a card unmounts, **Then** core invokes `release(container, node)` before removing the container, allowing framework cleanup (React root unmount, event listener removal)

### User Story 4 - Text & Label LOD (Priority: P2)

Labels obey the same LOD philosophy: zoomed out, only collapsed-bubble titles render (GPU/MSDF); mid-zoom, a label budget by screen-space size; close-zoom, GPU labels for sprite nodes are suppressed for nodes that have DOM cards (the card renders its own crisp HTML text).

**Acceptance Scenarios**:

1. **Given** far zoom, **Then** only visible (collapsed-ancestor) nodes are label candidates — never hidden descendants
2. **Given** a node with a mounted card, **Then** its GPU label is hidden while the card is alive
3. **Given** the label budget is exceeded, **Then** labels are prioritized by screen-space node size, stable across frames (no flicker)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: An `LODController` MUST map viewport zoom to (a) a visible-depth mask over the bubble hierarchy and (b) a DOM-mode flag, using per-subtree screen-space well radius with hysteresis bands (`T_expand`, `T_collapse`, `T_dom_in`, `T_dom_out`), all configurable
- **FR-002**: LOD visibility MUST be applied GPU-side via the existing `nodeFlags` buffer (a HIDDEN_BY_LOD bit alongside NODE_FLAG_DEAD/PINNED) so masking costs no CPU per-node work per frame; flag updates happen only on LOD-band transitions
- **FR-003**: Collapsed parents MUST render at their subtree centroid with radius derived from well radius; centroid/rollup data comes from the WASM bubble computation (containment-only edges)
- **FR-004**: Edges whose endpoints are both hidden MUST aggregate to the lowest visible ancestors; aggregation is recomputed only on LOD transitions, in WASM
- **FR-005**: A `DOMOverlayLayer` MUST maintain an absolutely-positioned container above the canvas whose CSS transform is derived from the same viewport uniforms the shaders consume — one transform write per frame, children positioned in world coordinates
- **FR-006**: The visible-node set for DOM mode MUST come from a viewport range query on the WASM R-tree spatial index (already wired for hit-testing)
- **FR-007**: Node positions for mounted cards MUST come from the existing position readback path; when the simulation is at rest (alpha snapped to 0), readback MUST pause and cached positions serve
- **FR-008**: Sprite↔card transitions MUST crossfade (configurable duration, default 150ms) with per-node GPU alpha driven through the existing per-node color/alpha channel
- **FR-009**: Card count MUST be bounded (`maxCards`, default 150) with deterministic priority; eviction MUST NOT flicker under small zoom oscillation (hysteresis + minimum card lifetime)
- **FR-010**: The card content contract MUST be three callbacks — `mount(container, node)`, `update(container, node, change)`, `release(container, node)` — with core owning container geometry and consumer owning container content; a default renderer ships in core
- **FR-011**: Pointer events MUST route correctly: card interiors receive DOM events; canvas gestures (pan/zoom/box-select) work between and over non-interactive card regions; dragging a card's handle drives the existing pin+drag GPU path
- **FR-012**: On zoom-gesture end, the overlay MUST re-rasterize (snap container scale to 1 by resizing card layout) so text renders at native crispness
- **FR-013**: All LOD/DOM behavior MUST be disableable independently (semantic zoom without DOM cards; DOM cards without semantic zoom)

### Non-Functional Requirements

- **NFR-001**: At 35K nodes with LOD enabled, steady-state frame time MUST NOT exceed the current no-LOD frame time (masking must pay for itself); LOD-transition frames MAY spike ≤ 4ms extra
- **NFR-002**: Pan/zoom with 150 mounted cards MUST hold 60fps on 2020-era hardware (compositor-only container transform during gestures)
- **NFR-003**: Card mount/unmount MUST NOT allocate per-frame; container elements are pooled
- **NFR-004**: The overlay MUST function identically for React/Vue/Svelte/vanilla/server-rendered consumers — no framework imports in core

### Key Entities

- **LODController**: zoom→representation state machine; owns thresholds, hysteresis, depth masks, transition scheduling
- **BubbleLODData**: per-node well radius, depth, subtree centroid, parent chain (from WASM, containment edges only)
- **DOMOverlayLayer**: container lifecycle, transform sync, card pool, priority/eviction, crossfade orchestration
- **CardContract**: `{ mount, update, release }` consumer callbacks + `NodeCardInfo` (id, type, label, position accessor, streams, pin/drag handle API)
- **EdgeAggregate**: visible-ancestor edge bundle with count/weight rollup

## Success Criteria *(mandatory)*

- **SC-001**: Meridian-scale graph (~35K nodes) fully zoomed out renders ≤ 200 visible nodes and holds 60fps on the reference machine
- **SC-002**: Zooming from overview to a single file's card is one continuous gesture with no popping, no drift between GPU and DOM representations (≤ 0.5px divergence at rest)
- **SC-003**: A user can select and copy text from a node card, follow a link in it, and find card text with browser find
- **SC-004**: A server-driven consumer can populate cards with zero client framework code (innerHTML injection after mount)
- **SC-005**: Disabling the feature reproduces current rendering byte-for-byte (flag bits zero, overlay absent)

## Dependencies & Assumptions

- Depends on branch `005-audit-fixes` landing: NODE_FLAG bit machinery, wired R-tree hit-testing, alpha snap-to-rest, `computeBubbleDataFromEdges` (containment-only bubble data), pin+drag GPU path
- Assumes containment edges are distinguishable from dependency edges in typed input (same requirement as the Nested Bubble algorithm)
- Edge-to-card-border termination requires card extents on the GPU edge path — worst case a per-node half-extent buffer; falls back to center anchoring if deferred
- Nested Bubble layout algorithm (CLAUDE.md roadmap) is NOT a dependency — semantic LOD works over any layout once bubble data exists — but the two are designed to compose

## Out of Scope (this feature)

- The Nested Bubble force layout itself (separate feature)
- Card content beyond the default renderer (syntax highlighting, file previews are consumer land)
- Multi-canvas / offscreen rendering
- Mobile gesture tuning
