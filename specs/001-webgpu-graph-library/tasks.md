# Tasks: Heroine Graph - WebGPU Graph Visualization Library

**Input**: Design documents from `/specs/001-webgpu-graph-library/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests not explicitly requested in spec. Benchmarks included for performance validation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Based on plan.md monorepo structure:
- **Core TS**: `packages/core/src/`
- **WASM/Rust**: `packages/wasm/src/`
- **React**: `packages/react/src/`
- **Vue**: `packages/vue/src/`
- **Svelte**: `packages/svelte/src/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize monorepo, configure tooling, establish project structure

- [x] T001 Create monorepo root structure with deno.json workspace config at root
- [x] T002 [P] Initialize packages/core/ with deno.json and TypeScript config
- [x] T003 [P] Initialize packages/wasm/ with Cargo.toml (wasm-bindgen, petgraph, rstar)
- [x] T004 [P] Create packages/react/ with package.json and TypeScript config
- [x] T005 [P] Create packages/vue/ with package.json and TypeScript config
- [x] T006 [P] Create packages/svelte/ with package.json and TypeScript config
- [x] T007 Configure deno fmt and deno lint rules in deno.json
- [x] T008 Configure rustfmt.toml and clippy.toml in packages/wasm/
- [x] T009 Create build scripts: wasm-pack build command in packages/wasm/build.sh
- [x] T010 Create examples/ directory structure per plan.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Rust/WASM Foundation

- [x] T011 Implement GraphEngine struct with petgraph StableGraph in packages/wasm/src/graph/mod.rs
- [x] T012 Implement Node struct with position, velocity, metadata in packages/wasm/src/graph/node.rs
- [x] T013 Implement Edge struct with source, target, metadata in packages/wasm/src/graph/edge.rs
- [x] T014 [P] Implement SoA position/velocity storage (Vec<f32>) in packages/wasm/src/graph/buffers.rs
- [x] T015 [P] Implement R-tree spatial index wrapper using rstar in packages/wasm/src/spatial/rtree.rs
- [x] T016 Implement wasm-bindgen exports for GraphEngine in packages/wasm/src/lib.rs
- [x] T017 Implement zero-copy Float32Array view methods for positions in packages/wasm/src/lib.rs
- [x] T018 Build WASM module and verify it loads in browser (wasm-pack build --target web)

### TypeScript Core Foundation

- [x] T019 Copy type definitions from specs contracts to packages/core/src/types.ts
- [x] T020 Implement WASM loader and initialization in packages/core/src/wasm/loader.ts
- [x] T021 Implement checkWebGPU() function in packages/core/src/webgpu/check.ts
- [x] T022 Implement WebGPU device/adapter initialization in packages/core/src/webgpu/context.ts
- [x] T023 Implement error reporting with context per Constitution V in packages/core/src/errors.ts
- [x] T024 Implement event emitter system in packages/core/src/events/emitter.ts
- [x] T025 Define all event types from contracts in packages/core/src/events/types.ts

### GPU Buffer Infrastructure

- [x] T026 Implement GPUBuffer manager for position buffers in packages/core/src/renderer/buffers/positions.ts
- [x] T027 Implement ping-pong buffer swap logic in packages/core/src/renderer/buffers/pingpong.ts
- [x] T028 Implement uniform buffer for simulation params in packages/core/src/renderer/buffers/uniforms.ts
- [x] T029 Implement CSR edge buffer format in packages/core/src/renderer/buffers/edges.ts

### Viewport Foundation

- [x] T030 Implement Viewport class with pan/zoom state in packages/core/src/viewport/viewport.ts
- [x] T031 Implement screenToGraph and graphToScreen transforms in packages/core/src/viewport/transforms.ts
- [x] T032 Implement viewport uniform buffer for shaders in packages/core/src/viewport/uniforms.ts

**Checkpoint**: Foundation ready - WASM loads, WebGPU initializes, buffers work

---

## Phase 3: User Story 1 - Load and Visualize a Large Graph (Priority: P1) 🎯 MVP

**Goal**: Render 500K+ nodes at 30fps with running force simulation

**Independent Test**: Load a 500,000 node graph, verify it renders at 30fps, pan/zoom works

### WGSL Shaders for US1

- [x] T033 [P] [US1] Write node rendering vertex shader in packages/core/src/renderer/shaders/node.vert.wgsl
- [x] T034 [P] [US1] Write node rendering fragment shader in packages/core/src/renderer/shaders/node.frag.wgsl
- [x] T035 [P] [US1] Write edge rendering vertex shader in packages/core/src/renderer/shaders/edge.vert.wgsl
- [x] T036 [P] [US1] Write edge rendering fragment shader in packages/core/src/renderer/shaders/edge.frag.wgsl

### Force Simulation Compute Shaders

- [x] T037 [P] [US1] Write Morton code generation compute shader in packages/core/src/simulation/shaders/morton.comp.wgsl
- [x] T038 [P] [US1] Write parallel radix sort compute shader in packages/core/src/simulation/shaders/radix_sort.comp.wgsl
- [x] T039 [P] [US1] Write bottom-up quadtree construction shader in packages/core/src/simulation/shaders/quadtree_build.comp.wgsl
- [x] T040 [P] [US1] Write bounding box calculation shader in packages/core/src/simulation/shaders/quadtree_bounds.comp.wgsl
- [x] T041 [US1] Write Barnes-Hut force traversal compute shader in packages/core/src/simulation/shaders/barnes_hut.comp.wgsl
- [x] T042 [US1] Write attractive force (spring) compute shader in packages/core/src/simulation/shaders/springs.comp.wgsl
- [x] T043 [US1] Write integration step compute shader in packages/core/src/simulation/shaders/integrate.comp.wgsl

### Render Pipeline for US1

- [x] T044 [US1] Implement node render pipeline setup in packages/core/src/renderer/pipelines/nodes.ts
- [x] T045 [US1] Implement edge render pipeline setup in packages/core/src/renderer/pipelines/edges.ts
- [x] T046 [US1] Implement main render loop with frame timing in packages/core/src/renderer/render_loop.ts
- [x] T047 [US1] Implement GPU command encoder orchestration in packages/core/src/renderer/commands.ts

### Simulation Orchestration for US1

- [x] T048 [US1] Implement SimulationController with start/pause/stop in packages/core/src/simulation/controller.ts
- [x] T049 [US1] Implement compute pipeline orchestration (7-stage per frame) in packages/core/src/simulation/pipeline.ts
- [x] T050 [US1] Implement alpha decay and convergence logic in packages/core/src/simulation/alpha.ts
- [x] T051 [US1] Implement force configuration API in packages/core/src/simulation/config.ts

### Data Loading for US1

- [x] T052 [US1] Implement GraphInput parser (nodes/edges arrays) in packages/core/src/graph/parser.ts
- [x] T053 [US1] Implement GraphTypedInput parser (Float32Array) in packages/core/src/graph/typed_parser.ts
- [x] T054 [US1] Implement node ID mapping (string → u32) in packages/core/src/graph/id_map.ts
- [x] T055 [US1] Implement initial position randomization in packages/core/src/graph/initialize.ts

### Main API for US1

- [x] T056 [US1] Implement createHeroineGraph factory function in packages/core/src/api/factory.ts
- [x] T057 [US1] Implement HeroineGraph class with load() method in packages/core/src/api/graph.ts
- [x] T058 [US1] Implement resize() and dispose() methods in packages/core/src/api/graph.ts
- [x] T059 [US1] Implement simulation control methods in packages/core/src/api/graph.ts
- [x] T060 [US1] Implement viewport control methods (pan, zoom, fitToView) in packages/core/src/api/graph.ts
- [x] T061 [US1] Create public API exports in packages/core/mod.ts

### Benchmark for US1

- [x] T062 [US1] Create 500K node benchmark in tests/benchmarks/large_graph.ts
- [x] T063 [US1] Verify 30fps target with performance.now() measurements

**Checkpoint**: US1 complete - large graphs render at 30fps with force simulation ✓

---

## Phase 4: User Story 2 - Interactive Node Manipulation (Priority: P2)

**Goal**: Click/drag nodes, selection, real-time simulation response

**Independent Test**: Load graph, click node (selected), drag (follows cursor), release (rejoins simulation)

### Hit Testing for US2

- [x] T064 [P] [US2] Implement pointer event handling in packages/core/src/interaction/pointer.ts
- [x] T065 [US2] Implement hit test via WASM R-tree query in packages/core/src/interaction/hit_test.ts
- [x] T066 [US2] Implement getNodeAtPosition API method in packages/core/src/api/graph.ts
- [x] T067 [US2] Implement getEdgeAtPosition API method in packages/core/src/api/graph.ts

### Selection for US2

- [x] T068 [US2] Implement selection state management in packages/core/src/api/graph.ts (integrated into HeroineGraph class)
- [x] T069 [US2] Implement visual feedback for selected nodes (shader uniform) in packages/core/src/renderer/shaders/node.frag.wgsl
- [x] T070 [US2] Implement selectNodes/selectEdges API methods in packages/core/src/api/graph.ts
- [x] T071 [US2] Emit selection:change events in packages/core/src/api/graph.ts

### Dragging for US2

- [x] T072 [US2] Implement drag state machine in packages/core/src/api/graph.ts (integrated into HeroineGraph class)
- [x] T073 [US2] Implement node pinning in packages/core/src/api/graph.ts
- [x] T074 [US2] Implement real-time position update during drag in packages/core/src/api/graph.ts
- [x] T075 [US2] Emit node:dragstart, node:dragmove, node:dragend events in packages/core/src/api/graph.ts

### Hover for US2

- [x] T076 [US2] Implement hover detection in packages/core/src/api/graph.ts
- [x] T077 [US2] Implement visual feedback for hovered nodes in packages/core/src/renderer/shaders/node.frag.wgsl
- [x] T078 [US2] Emit node:hoverenter, node:hoverleave events in packages/core/src/api/graph.ts

### API Methods for US2

- [x] T079 [US2] Implement on/off/once event subscription in packages/core/src/api/graph.ts
- [x] T080 [US2] Implement pinNode/unpinNode API methods in packages/core/src/api/graph.ts
- [x] T081 [US2] Implement setNodePosition API method in packages/core/src/api/graph.ts

**Checkpoint**: US2 complete - nodes clickable, draggable, selection works

---

## Phase 5: User Story 3 - Visual Layers: Heatmaps and Density (Priority: P3)

**Goal**: Enable heatmap layer showing node density as color gradient

**Independent Test**: Load clustered graph, enable heatmap, verify density coloring updates in real-time

### Heatmap Infrastructure for US3

- [ ] T082 [P] [US3] Create density texture (RGBA32F) in packages/core/src/layers/heatmap/texture.ts
- [ ] T083 [P] [US3] Write Gaussian splat vertex shader in packages/core/src/layers/heatmap/shaders/splat.vert.wgsl
- [ ] T084 [P] [US3] Write Gaussian splat fragment shader in packages/core/src/layers/heatmap/shaders/splat.frag.wgsl
- [ ] T085 [US3] Write color mapping fragment shader in packages/core/src/layers/heatmap/shaders/colormap.frag.wgsl

### Heatmap Pipeline for US3

- [ ] T086 [US3] Implement heatmap render pipeline (additive blend) in packages/core/src/layers/heatmap/pipeline.ts
- [ ] T087 [US3] Implement color scale textures (viridis, plasma, etc.) in packages/core/src/layers/heatmap/colorscales.ts
- [ ] T088 [US3] Implement HeatmapLayer class in packages/core/src/layers/heatmap/layer.ts
- [ ] T089 [US3] Implement heatmap config (radius, intensity, opacity) in packages/core/src/layers/heatmap/config.ts

### Layer Management for US3

- [ ] T090 [US3] Implement LayerManager class in packages/core/src/layers/manager.ts
- [ ] T091 [US3] Implement layer visibility toggle in packages/core/src/layers/manager.ts
- [ ] T092 [US3] Implement layer compositing (multi-pass render) in packages/core/src/renderer/compositor.ts

### API Methods for US3

- [ ] T093 [US3] Implement showLayer/hideLayer/toggleLayer in packages/core/src/api/graph.ts
- [ ] T094 [US3] Implement setLayerConfig API method in packages/core/src/api/graph.ts
- [ ] T095 [US3] Implement getLayers/getLayer API methods in packages/core/src/api/graph.ts

**Checkpoint**: US3 complete - heatmap layer shows density, updates in real-time

---

## Phase 6: User Story 4 - Visual Layers: Contours and Metaballs (Priority: P4)

**Goal**: Enable contour lines and metaball rendering for cluster visualization

**Independent Test**: Load clustered graph, enable contours/metaballs, verify boundaries delineate clusters

### Contour Pipeline for US4

- [ ] T096 [P] [US4] Write active cell identification compute shader in packages/core/src/layers/contour/shaders/identify.comp.wgsl
- [ ] T097 [P] [US4] Write parallel prefix sum compute shader in packages/core/src/layers/contour/shaders/prefix_sum.comp.wgsl
- [ ] T098 [US4] Write vertex generation compute shader in packages/core/src/layers/contour/shaders/generate.comp.wgsl
- [ ] T099 [US4] Write contour line render shaders in packages/core/src/layers/contour/shaders/line.wgsl
- [ ] T100 [US4] Implement ContourLayer class in packages/core/src/layers/contour/layer.ts
- [ ] T101 [US4] Implement contour config (thresholds, stroke) in packages/core/src/layers/contour/config.ts

### Metaball Pipeline for US4

- [ ] T102 [P] [US4] Write metaball SDF fragment shader with smin in packages/core/src/layers/metaball/shaders/sdf.frag.wgsl
- [ ] T103 [US4] Implement MetaballLayer class in packages/core/src/layers/metaball/layer.ts
- [ ] T104 [US4] Implement metaball config (threshold, blendRadius) in packages/core/src/layers/metaball/config.ts

### Integration for US4

- [ ] T105 [US4] Register ContourLayer with LayerManager in packages/core/src/layers/manager.ts
- [ ] T106 [US4] Register MetaballLayer with LayerManager in packages/core/src/layers/manager.ts
- [ ] T107 [US4] Update layer compositing order in packages/core/src/renderer/compositor.ts

**Checkpoint**: US4 complete - contours and metaballs visualize cluster boundaries

---

## Phase 7: User Story 5 - Node Labels and Text (Priority: P5)

**Goal**: Display MSDF text labels with LOD culling, sharp at all zoom levels

**Independent Test**: Load labeled graph, zoom in/out, verify labels appear/disappear and remain sharp

### Font Atlas for US5

- [ ] T108 [P] [US5] Add MSDF font atlas (Inter) to packages/core/assets/fonts/
- [ ] T109 [P] [US5] Add font atlas JSON metadata to packages/core/assets/fonts/
- [ ] T110 [US5] Implement font atlas loader in packages/core/src/layers/labels/atlas.ts

### Label Shaders for US5

- [ ] T111 [P] [US5] Write label vertex shader in packages/core/src/layers/labels/shaders/label.vert.wgsl
- [ ] T112 [US5] Write MSDF fragment shader with median() in packages/core/src/layers/labels/shaders/label.frag.wgsl

### Label Culling for US5

- [ ] T113 [US5] Implement label priority sorting in packages/core/src/layers/labels/priority.ts
- [ ] T114 [US5] Implement screen-space collision grid in packages/core/src/layers/labels/collision.ts
- [ ] T115 [US5] Implement zoom-level thresholds in packages/core/src/layers/labels/lod.ts
- [ ] T116 [US5] Implement LabelManager orchestration in packages/core/src/layers/labels/manager.ts

### Label Layer for US5

- [ ] T117 [US5] Implement LabelsLayer class in packages/core/src/layers/labels/layer.ts
- [ ] T118 [US5] Implement label instance buffer updates in packages/core/src/layers/labels/instances.ts
- [ ] T119 [US5] Implement label config (font, size, maxLabels) in packages/core/src/layers/labels/config.ts

### Integration for US5

- [ ] T120 [US5] Register LabelsLayer with LayerManager in packages/core/src/layers/manager.ts
- [ ] T121 [US5] Update layer compositing (labels on top) in packages/core/src/renderer/compositor.ts

**Checkpoint**: US5 complete - labels render sharply, LOD culling works

---

## Phase 8: User Story 6 - Framework Integration (Priority: P6)

**Goal**: Provide React, Vue, Svelte wrapper components

**Independent Test**: Create app with each framework, verify graph renders, events fire

### React Wrapper for US6

- [ ] T122 [P] [US6] Implement HeroineGraph React component in packages/react/src/HeroineGraph.tsx
- [ ] T123 [P] [US6] Implement useGraph hook in packages/react/src/hooks/useGraph.ts
- [ ] T124 [P] [US6] Implement useSimulation hook in packages/react/src/hooks/useSimulation.ts
- [ ] T125 [US6] Create React package exports in packages/react/src/index.ts

### Vue Wrapper for US6

- [ ] T126 [P] [US6] Implement HeroineGraph Vue component in packages/vue/src/HeroineGraph.vue
- [ ] T127 [P] [US6] Implement composables (useGraph, useSimulation) in packages/vue/src/composables/
- [ ] T128 [US6] Create Vue package exports in packages/vue/src/index.ts

### Svelte Wrapper for US6

- [ ] T129 [P] [US6] Implement HeroineGraph Svelte component in packages/svelte/src/HeroineGraph.svelte
- [ ] T130 [P] [US6] Implement Svelte stores in packages/svelte/src/stores/
- [ ] T131 [US6] Create Svelte package exports in packages/svelte/src/index.ts

### Framework Examples for US6

- [ ] T132 [P] [US6] Create React example app in examples/framework-react/
- [ ] T133 [P] [US6] Create Vue example app in examples/framework-vue/
- [ ] T134 [P] [US6] Create Svelte example app in examples/framework-svelte/

**Checkpoint**: US6 complete - all three framework wrappers functional

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

### Examples & Documentation

- [ ] T135 [P] Create basic example in examples/basic/
- [ ] T136 [P] Create large-graph stress test example in examples/large-graph/
- [ ] T137 [P] Create interactive demo example in examples/interactive/
- [ ] T138 Validate quickstart.md instructions work end-to-end

### Performance & Quality

- [ ] T139 Run performance benchmark suite, verify SC-001 (30fps @ 500K nodes)
- [ ] T140 Profile memory usage, verify SC-006 (<500 bytes/node)
- [ ] T141 Test in Chrome, Firefox, Safari, Edge per SC-007
- [ ] T142 Verify framework wrapper bundle sizes <5KB per SC-009

### Edge Cases

- [ ] T143 Handle empty graph (0 nodes) gracefully
- [ ] T144 Handle disconnected components correctly
- [ ] T145 Handle WebGPU unavailable with clear error message
- [ ] T146 Handle window resize events
- [ ] T147 Handle tab visibility change (pause simulation when hidden)

### Final Polish

- [ ] T148 Review all error messages for actionability per Constitution V
- [ ] T149 Verify all defaults work beautifully per Constitution IV
- [ ] T150 Ensure no `todo!()` or placeholder code per Technical Conventions

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational - MVP
- **US2 (Phase 4)**: Depends on Foundational - can parallel with US1 after T061
- **US3 (Phase 5)**: Depends on US1 (needs render loop)
- **US4 (Phase 6)**: Depends on US3 (needs LayerManager)
- **US5 (Phase 7)**: Depends on US3 (needs LayerManager)
- **US6 (Phase 8)**: Depends on US1 (needs core API)
- **Polish (Phase 9)**: Depends on all user stories

### User Story Dependencies

```
Setup → Foundational → US1 (MVP)
                          ↓
                    ┌─────┴─────┐
                    ↓           ↓
                   US2         US3
                               ↓
                          ┌────┴────┐
                          ↓         ↓
                         US4       US5

US1 ─────────────────────────────→ US6

All → Polish
```

### Within Each User Story

1. Shaders (can parallelize)
2. TypeScript infrastructure
3. Pipeline setup
4. API methods
5. Integration/verification

---

## Parallel Opportunities

### Phase 1 (Setup)
```
Parallel: T002, T003, T004, T005, T006
```

### Phase 2 (Foundational)
```
Parallel: T014, T015
Parallel: T026, T027, T028, T029
```

### Phase 3 (US1)
```
Parallel: T033, T034, T035, T036 (render shaders)
Parallel: T037, T038, T039, T040 (compute shaders)
```

### Phase 4 (US2)
```
Can start after T061 while US1 benchmarks run
```

### Phase 5-7 (US3, US4, US5)
```
US4 and US5 can run in parallel after US3 LayerManager complete
```

### Phase 8 (US6)
```
Parallel: T122-125 (React), T126-128 (Vue), T129-131 (Svelte)
Parallel: T132, T133, T134 (examples)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Load 500K node graph, verify 30fps
5. Demo MVP if ready

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. US1 → Large graph rendering (MVP!)
3. US2 → Interactive nodes
4. US3 → Heatmap layer
5. US4 → Contours + metaballs
6. US5 → Labels
7. US6 → Framework wrappers
8. Polish → Production ready

---

## Notes

- [P] = parallelizable (different files, no dependencies)
- [USn] = belongs to User Story n
- All WGSL shaders go in respective `shaders/` directories
- Use `deno add` for TS deps, `cargo add` for Rust deps
- No `todo!()` or placeholder implementations
- Full implementations only per Constitution
