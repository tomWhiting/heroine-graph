# Implementation Plan: Heroine Graph - WebGPU Graph Visualization Library

**Branch**: `001-webgpu-graph-library` | **Date**: 2026-01-06 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-webgpu-graph-library/spec.md`

## Summary

Build a high-performance graph visualization library using WebGPU for rendering and compute,
with Rust/WASM for graph algorithms and data structures. The library will render graphs with
up to 1,000,000 nodes at 30+ fps, featuring GPU-accelerated force-directed layout, interactive
node manipulation, and visual layers (heatmaps, contours, metaballs, labels). Framework
wrappers will be provided as thin bindings that expose the full core API.

## Technical Context

**Language/Version**: TypeScript 5.x (Deno 2.x), Rust 1.75+ (WASM target), WGSL shaders
**Primary Dependencies**: WebGPU API, wasm-bindgen, petgraph (Rust), d3-scale/d3-color (TS)
**Storage**: N/A (in-memory only, no persistence)
**Testing**: Deno test (TS), cargo test (Rust), WebGPU conformance tests
**Target Platform**: Modern browsers (Chrome 113+, Firefox 141+, Safari 26+, Edge 113+)
**Project Type**: Monorepo with multiple packages (core, wasm, framework wrappers)
**Performance Goals**: 30fps @ 500K nodes, 60fps @ 100K nodes, <16ms interaction latency
**Constraints**: <500 bytes/node memory, <5KB wrapper bundle size (gzipped)
**Scale/Scope**: Up to 1,000,000 nodes, 4,000,000 edges

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Implementation |
|-----------|--------|----------------|
| I. Consistency Is On Us | ✅ PASS | Same input = same output. Deterministic simulation seeding. Consistent interaction behavior across zoom levels. |
| II. Contract, Not Coercion | ✅ PASS | API accepts user data as-is. No silent modification. User defines node importance/selection/grouping. |
| III. Trust Users, Don't Give Guns | ✅ PASS | Allow loading huge graphs (user's performance choice). Guard against data loss via confirmation events. |
| IV. Expose All Controls, Make Defaults Excellent | ✅ PASS | All simulation parameters configurable. Defaults work beautifully out of box. |
| V. No Silent Failures | ✅ PASS | WebGPU errors surface with context. Shader compilation failures include line numbers. Data validation reports specific issues. |
| VI. Automation Over Gatekeeping | ✅ PASS | No hard limits on node count. Events emitted for perf thresholds. Users define handlers. |
| VII. Low-Level Primitives | ✅ PASS | Core library is the primitive. WebGPU direct. Framework wrappers are thin. |
| VIII. Circular Dependencies | ✅ PASS | Cyclic graphs fully supported. Force simulation handles oscillations. |
| IX. Easy to Have Fun | ✅ PASS | Simple API for common cases. Loading + rendering feels magical. |
| X. Build With Love | ✅ PASS | Every frame polished. Every interaction responsive. |

**Technical Conventions Check**:
- Dependency Management: ✅ Using `cargo add` and `deno add` for latest versions
- Implementation Standards: ✅ No `todo!()` or placeholders—full implementations only
- Quality Gates: ✅ CI enforces warnings, tests, benchmarks

## Project Structure

### Documentation (this feature)

```text
specs/001-webgpu-graph-library/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (TypeScript interfaces)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
packages/
├── core/                    # Main TypeScript library
│   ├── src/
│   │   ├── graph/           # Graph data structures (TS side)
│   │   ├── renderer/        # WebGPU rendering
│   │   │   ├── pipelines/   # Render pipelines (nodes, edges, layers)
│   │   │   ├── shaders/     # WGSL shader source
│   │   │   └── buffers/     # GPU buffer management
│   │   ├── simulation/      # Force simulation orchestration
│   │   ├── interaction/     # Mouse/touch/keyboard handling
│   │   ├── layers/          # Visual layers (heatmap, contour, metaball, labels)
│   │   ├── viewport/        # Pan, zoom, coordinate transforms
│   │   └── events/          # Event system
│   ├── mod.ts               # Public API exports
│   └── deno.json            # Deno config
│
├── wasm/                    # Rust WASM module
│   ├── src/
│   │   ├── lib.rs           # WASM entry point
│   │   ├── graph/           # petgraph-based graph structure
│   │   ├── layout/          # Force calculation (Barnes-Hut)
│   │   ├── spatial/         # Quadtree/R-tree for hit testing
│   │   └── algorithms/      # Graph algorithms (clustering, etc.)
│   ├── Cargo.toml
│   └── build.rs             # WASM build configuration
│
├── react/                   # React wrapper
│   ├── src/
│   │   ├── HeroineGraph.tsx # Main component
│   │   ├── hooks/           # useGraph, useSimulation, etc.
│   │   └── index.ts
│   └── package.json
│
├── vue/                     # Vue wrapper
│   ├── src/
│   │   ├── HeroineGraph.vue
│   │   └── index.ts
│   └── package.json
│
└── svelte/                  # Svelte wrapper
    ├── src/
    │   ├── HeroineGraph.svelte
    │   └── index.ts
    └── package.json

tests/
├── contract/                # API contract tests
├── integration/             # Cross-package integration tests
├── unit/                    # Unit tests per package
├── benchmarks/              # Performance benchmarks
└── visual/                  # Visual regression tests

examples/
├── basic/                   # Minimal example
├── large-graph/             # Million-node stress test
├── interactive/             # Full interactivity demo
└── framework-*/             # Framework-specific examples
```

**Structure Decision**: Monorepo with separate packages. Core library is framework-agnostic
TypeScript + WASM. Framework wrappers are separate npm-publishable packages that depend on
core. This enables:
- Independent versioning of core vs wrappers
- Tree-shaking (users only import what they need)
- Clear separation per Principle VII (Low-Level Primitives)

## Complexity Tracking

> **No constitution violations requiring justification.**

The multi-package structure is justified by explicit requirements (FR-029 through FR-033)
for framework-agnostic core with separate framework wrappers. This is not added complexity—
it's the minimum structure to meet requirements.
