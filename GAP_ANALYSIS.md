# Heroine Graph: Gap Analysis Report

**Date**: 2026-01-06  
**Reviewer**: Claude  
**Scope**: Comparison of specs vs implementation

---

## Executive Summary

The core GPU simulation pipeline, rendering, basic interaction, and visual layers (heatmap, contour, metaball) are all working. However, there's a meaningful gap between the API contract in the specs and what's actually implemented, plus two entire user stories (Labels and Framework Wrappers) haven't been started.

---

## 1. WASM Integration: Underutilized

**Status: Partially implemented, not wired up**

The WASM module has solid bones:
- ✅ Graph topology storage with petgraph StableGraph
- ✅ R-tree spatial index (rstar)
- ✅ Zero-copy Float32Array views
- ✅ CSR edge format conversion

### Gaps

- ❌ R-tree not actually used for hit testing—there's a `TODO` comment in `graph.ts` saying "using brute-force fallback"
- ❌ WASM positions never synced from GPU (you do `readbackPositions` to `parsedGraph.positionsX/Y`, but not to the WASM engine)
- ❌ No graph algorithms implemented yet (community detection was mentioned as the goal)

### Recommendation

Either wire up the R-tree for hit testing now, or add a task to do it later. The brute-force approach will hurt at 100K+ nodes.

---

## 2. API Contract vs Implementation

**Status: Significant divergence**

The `contracts/api.ts` defines a comprehensive API. Here's what's missing from `HeroineGraph`:

### Node Operations (spec has 11 methods, ~4 implemented)

| Contract Method | Status |
|----------------|--------|
| `getNode(id)` | ❌ Missing |
| `getNodes()` | ❌ Missing |
| `getNodeCount()` | ✅ Via getter |
| `addNode(node)` | ❌ Missing |
| `addNodes(nodes)` | ❌ Missing |
| `removeNode(id)` | ❌ Missing |
| `updateNode(id, metadata)` | ❌ Missing |
| `setNodePosition(id, x, y)` | ✅ Implemented |
| `pinNode/unpinNode` | ✅ Implemented |
| `hideNode/showNode` | ❌ Missing |

### Edge Operations (spec has 9 methods, 0 implemented)

| Contract Method | Status |
|----------------|--------|
| `getEdge(id)` | ❌ Missing |
| `getEdges()` | ❌ Missing |
| `getEdgeCount()` | ✅ Via getter |
| `addEdge(edge)` | ❌ Missing |
| `removeEdge(id)` | ❌ Missing |
| `updateEdge(id, metadata)` | ❌ Missing |
| `getNodeEdges(nodeId)` | ❌ Missing |
| `getNeighbors(nodeId)` | ❌ Missing |

### Selection

| Contract Method | Status |
|----------------|--------|
| `selectNodes(ids, additive?)` | ⚠️ Partial (no additive param) |
| `selectEdges(ids, additive?)` | ⚠️ Partial (no additive param) |
| `selectAll()` | ❌ Missing |

### Viewport

| Contract Method | Status |
|----------------|--------|
| `pan(dx, dy)` | ✅ Implemented |
| `panTo(x, y, animate?)` | ❌ Missing (no animation) |
| `zoom(factor, cx?, cy?)` | ✅ Implemented |
| `zoomTo(scale, animate?)` | ❌ Missing |
| `fitToView(padding?, animate?)` | ⚠️ Partial (no animation) |
| `fitNodes(nodeIds, padding?, animate?)` | ❌ Missing |
| `screenToGraph/graphToScreen` | ✅ In viewport, not exposed on API |

### Hit Testing

| Contract Method | Status |
|----------------|--------|
| `getNodeAtPosition` | ✅ Implemented |
| `getEdgeAtPosition` | ✅ Implemented |
| `getNodesInRect(x1, y1, x2, y2)` | ❌ Missing |
| `getNearestNode(x, y, maxDist?)` | ❌ Missing |

### Utilities

| Contract Method | Status |
|----------------|--------|
| `getBounds()` | ❌ Missing (calculated internally in fitToView) |
| `getNodesBounds(nodeIds)` | ❌ Missing |
| `exportPositions()` | ❌ Missing |
| `importPositions(positions)` | ❌ Missing |
| `screenshot()` | ❌ Missing |

### Layers

| Contract Method | Status |
|----------------|--------|
| `showLayer(type)/hideLayer(type)` | ⚠️ Different API (`enableHeatmap/disableHeatmap`) |
| `toggleLayer(type)` | ✅ Implemented |
| `setLayerConfig(type, config)` | ⚠️ Per-layer methods instead |
| `setLayerOrder(type, order)` | ❌ Missing |

### Recommendation

Either update the contract to match the current API design, or add the missing methods. The current `enableHeatmap/disableHeatmap` pattern is actually nicer than the generic `showLayer(LayerType)` in the contract—consider updating the contract.

---

## 3. User Story 5: Labels (Complete)

**Status: 100% complete**

Implemented:
- ✅ MSDF font atlas (Roboto, 512x512, 95 characters)
- ✅ MSDF vertex and fragment shaders
- ✅ Label visibility culling and collision detection
- ✅ LabelsLayer class with position provider integration
- ✅ Integration with main render pipeline (renders on top of nodes)

---

## 4. User Story 6: Framework Wrappers (Not Started)

**Status: 0% complete**

Tasks T122-T134 (React, Vue, Svelte wrappers + examples) are all unchecked.

The package directories exist (`packages/react/`, `packages/vue/`, `packages/svelte/`) but appear to be scaffolding only.

### Recommendation

These are straightforward once the core API stabilizes. Don't start until API contract is finalized.

---

## 5. Edge Cases Not Handled

From the spec's edge cases list:

| Edge Case | Status |
|-----------|--------|
| Empty graph (0 nodes) | ⚠️ Probably works but not verified |
| Disconnected components | ✅ Should work (force sim handles it) |
| Nodes at identical positions | ⚠️ Untested |
| Self-loops | ⚠️ Untested |
| WebGPU unavailable | ⚠️ `checkWebGPU` exists but error UX unclear |
| Extremely long edges | ⚠️ Untested |
| Load new data while simulation running | ⚠️ Untested |
| Invalid position data | ⚠️ Untested |
| Tab loses focus | ❌ No visibility change handling |
| Window resize | ⚠️ `resize()` exists but auto-hook unclear |

### Recommendation

Add a `visibilitychange` listener to pause simulation when tab is hidden. This is a quick win for battery/CPU.

---

## 6. Incremental Updates (FR-026)

**Status: Not implemented**

The spec requires: "support incremental updates (add/remove nodes/edges) without full reload"

Currently `load()` replaces everything. There's no `addNode()`, `removeNode()`, `addEdge()`, `removeEdge()`.

This is architecturally tricky because:
- GPU buffers are pre-sized
- WASM graph needs to stay in sync
- Spatial index needs rebuilding
- Simulation needs to incorporate new nodes smoothly

### Recommendation

This is a significant feature. If the use case is "load once, explore", deprioritize. If live graph updates are needed, this needs design work.

---

## 7. Success Criteria Verification

| Criterion | Status |
|-----------|--------|
| SC-001: 30fps @ 500K nodes | ⚠️ Needs benchmark |
| SC-002: First frame <5s @ 1M nodes | ⚠️ Needs benchmark |
| SC-003: <16ms drag latency | ⚠️ Needs measurement |
| SC-004: Layers update per-frame | ✅ Implemented |
| SC-005: Labels sharp 10%-1000% zoom | ✅ MSDF labels implemented |
| SC-006: <500 bytes/node memory | ⚠️ Needs profiling |
| SC-007: Chrome/Firefox/Safari/Edge | ⚠️ Needs testing |
| SC-008: First graph in 15min (docs) | ⚠️ quickstart.md exists but untested |
| SC-009: Wrapper <5KB gzipped | ❌ Wrappers not implemented |
| SC-010: Incremental update <100ms | ❌ Incremental updates not implemented |

### Recommendation

Run T062/T063 benchmarks before claiming performance targets.

---

## 8. Minor Implementation Gaps

1. **Force Configuration** - `setSimulationConfig` not exposed on public API (only internal `simulationController`)

2. **Alpha Target** - `setAlphaTarget()` in contract, only `setAlpha()` implemented

3. **Simulation Tick** - `tickSimulation()` for manual stepping not exposed

4. **Animation** - Viewport methods should support `animate?: boolean` parameter

5. **Event Return Values** - Contract says `on()` returns unsubscribe function, implementation returns `void`

---

## Priority Recommendations

### High Priority (Core functionality gaps)

1. Wire up WASM R-tree for hit testing
2. Add `visibilitychange` handler for tab switching
3. Run performance benchmarks to verify targets

### Medium Priority (API completeness)

4. Decide on API contract vs implementation divergence—pick one and align
5. Add incremental update support if needed for use case
6. Expose missing viewport methods (`panTo`, `zoomTo` with animation)

### Lower Priority (Can defer)

7. Labels (US5) - significant work, defer if not needed
8. Framework wrappers (US6) - wait for API stabilization
9. Screenshot export
10. Rect selection (`getNodesInRect`)

---

## Task Checklist Summary

From `tasks.md`:

| Phase | Status |
|-------|--------|
| Phase 1: Setup | ✅ Complete |
| Phase 2: Foundational | ✅ Complete |
| Phase 3: US1 - Large Graph Rendering | ✅ Complete |
| Phase 4: US2 - Interactive Manipulation | ✅ Complete |
| Phase 5: US3 - Heatmaps | ✅ Complete |
| Phase 6: US4 - Contours & Metaballs | ✅ Complete |
| Phase 7: US5 - Labels | ✅ Complete |
| Phase 8: US6 - Framework Wrappers | ❌ Not started |
| Phase 9: Polish & Edge Cases | ❌ Not started |
