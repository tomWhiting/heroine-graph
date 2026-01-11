# Implementation Plan: Graph Algorithms WASM Module

**Branch**: `003-graph-algorithms-wasm` | **Date**: 2026-01-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-graph-algorithms-wasm/spec.md`

## Summary

Extend the existing Rust/WASM module (`packages/wasm`) with graph algorithms that complement GPU-based rendering. The module adds community detection (Leiden/Louvain), centrality measures (PageRank, betweenness, closeness, eigenvector), hull computation for community boundaries, and boundary collision physics. All algorithms integrate with the existing `GraphEngine` and expose a JavaScript API via wasm-bindgen.

## Technical Context

**Language/Version**: Rust 1.75+ (edition 2024), TypeScript 5.x for bindings
**Primary Dependencies**:
- petgraph 0.8.3 (already in deps - provides PageRank, SCC, connected components)
- rustworkx-core 0.17+ (betweenness, closeness, eigenvector centrality)
- geo 0.32.0 (convex/concave hull computation)
- wasm-bindgen 0.2, js-sys 0.3 (already in deps)
- Custom Louvain implementation (see research.md - external crates have WASM compatibility issues)

**Storage**: N/A (in-memory graph already in GraphEngine)
**Testing**: cargo test, wasm-bindgen-test (already configured)
**Target Platform**: WASM (wasm32-unknown-unknown), browser runtime
**Project Type**: Library module extending existing WASM package
**Performance Goals**:
- Community detection: 100K nodes < 1 second
- Hull computation: 100 communities < 100ms
- Boundary physics: 50 communities < 16ms (60fps)
- PageRank: 50K nodes < 500ms
- Betweenness: 10K nodes < 2 seconds

**Constraints**:
- Bundle size: < 500KB gzipped additional
- Must integrate with existing GraphEngine without breaking API
- All algorithms must be deterministic for identical inputs

**Scale/Scope**: Graphs up to 100K nodes, 100 communities

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Consistency Is On Us | PASS | Deterministic algorithms (SC-007), same input = same output |
| II. Contract, Not Coercion | PASS | User calls algorithms explicitly, no automatic behavior |
| III. Trust Users, Don't Give Guns | PASS | Large graphs may be slow but won't crash; no data loss risk |
| IV. Expose All Controls, Make Defaults Excellent | PASS | All algorithm parameters exposed; sensible defaults provided |
| V. No Silent Failures | PASS | Invalid inputs return errors with context, not empty results |
| VI. Automation Over Gatekeeping | PASS | No hard limits; performance warnings via progress callbacks |
| VII. Low-Level Primitives Over Wrappers | PASS | Direct petgraph/algorithm access; no unnecessary abstraction |
| VIII. Circular Dependencies Are Real | PASS | Algorithms handle cyclic graphs correctly |
| IX. Make It Easy to Have Fun | PASS | Single function calls for each algorithm type |
| X. Build With Love | PASS | Careful implementation with proper edge case handling |

**Gate Result**: PASS - No violations. Proceeding to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/003-graph-algorithms-wasm/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
packages/wasm/
├── Cargo.toml                    # Add new dependencies
├── src/
│   ├── lib.rs                    # Extend HeroineGraphWasm with algorithm methods
│   ├── graph/
│   │   ├── mod.rs
│   │   ├── engine.rs             # Existing - may need accessor methods
│   │   ├── node.rs
│   │   └── edge.rs
│   ├── spatial/
│   │   ├── mod.rs
│   │   └── rtree.rs              # Existing spatial index
│   ├── algorithms/               # NEW - algorithm implementations
│   │   ├── mod.rs                # Module exports
│   │   ├── community.rs          # Leiden/Louvain community detection
│   │   ├── centrality.rs         # Centrality measures
│   │   └── components.rs         # Connected/strongly connected components
│   ├── geometry/                 # NEW - geometric computations
│   │   ├── mod.rs
│   │   └── hull.rs               # Convex/concave hull computation
│   └── physics/                  # NEW - boundary physics
│       ├── mod.rs
│       └── boundary.rs           # Boundary collision/repulsion
└── tests/
    ├── algorithms_test.rs        # Algorithm correctness tests
    ├── geometry_test.rs          # Hull computation tests
    └── physics_test.rs           # Boundary physics tests
```

**Structure Decision**: Extend existing WASM package structure with three new modules (`algorithms/`, `geometry/`, `physics/`) following the established pattern of `graph/` and `spatial/`.

## Complexity Tracking

No violations requiring justification.

---

## Phase Completion Status

### Phase 0: Research ✅
- **Output**: [research.md](./research.md)
- **Key Decision**: Custom Louvain implementation (single-clustering has rayon dependency, not WASM-compatible)
- **Dependencies Confirmed**: rustworkx-core 0.17+, geo 0.32.0

### Phase 1: Design & Contracts ✅
- **Data Model**: [data-model.md](./data-model.md)
- **API Contract**: [contracts/algorithms-api.ts](./contracts/algorithms-api.ts)
- **Quickstart**: [quickstart.md](./quickstart.md)
- **Agent Context**: Updated CLAUDE.md

### Phase 2: Tasks (Pending)
Run `/speckit.tasks` to generate implementation tasks.

### Constitution Re-check (Post Phase 1)

| Principle | Status | Notes |
|-----------|--------|-------|
| VII. Low-Level Primitives | PASS | Custom Louvain keeps us in control, no opinionated wrappers |
| IV. Expose All Controls | PASS | All config parameters documented in API contract |
| V. No Silent Failures | PASS | Progress callbacks for long operations; errors return context |

**Gate Result**: PASS - Design phase complete, ready for task generation.
