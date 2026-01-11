# Tasks: Graph Algorithms WASM Module

**Input**: Design documents from `/specs/003-graph-algorithms-wasm/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests are included as this is a Rust library with cargo test infrastructure already in place.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

All paths are relative to `packages/wasm/`:
- Source: `src/`
- Tests: `tests/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, dependencies, and module structure

- [ ] T001 Add rustworkx-core and geo dependencies to packages/wasm/Cargo.toml
- [ ] T002 [P] Create algorithms module structure with mod.rs in packages/wasm/src/algorithms/mod.rs
- [ ] T003 [P] Create geometry module structure with mod.rs in packages/wasm/src/geometry/mod.rs
- [ ] T004 [P] Create physics module structure with mod.rs in packages/wasm/src/physics/mod.rs
- [ ] T005 Register new modules in packages/wasm/src/lib.rs
- [ ] T006 Verify WASM build succeeds with `deno task build:wasm`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T007 Add graph accessor methods to GraphEngine for algorithm access in packages/wasm/src/graph/engine.rs (expose petgraph StableGraph reference, position buffers)
- [ ] T008 [P] Create shared types for algorithm results in packages/wasm/src/algorithms/types.rs (Community, CommunityAssignment, CentralityResult, ComponentResult)
- [ ] T009 [P] Create shared config types in packages/wasm/src/algorithms/config.rs (CommunityDetectionConfig, CentralityConfig, HullComputationConfig)
- [ ] T010 [P] Create geometry types in packages/wasm/src/geometry/types.rs (CommunityBoundary, HullType enum)
- [ ] T011 [P] Create physics types in packages/wasm/src/physics/types.rs (BoundaryPhysicsConfig, BoundaryPhysicsResult)
- [ ] T012 Create WASM-bindgen result wrapper utilities for JS interop in packages/wasm/src/algorithms/wasm_types.rs (convert Rust types to JS-compatible formats)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Community Detection (Priority: P1) 🎯 MVP

**Goal**: Detect communities in graphs using Louvain algorithm, returning community assignments for all nodes

**Independent Test**: Load a graph with known community structure, run `detectCommunities()`, verify nodes grouped correctly

### Tests for User Story 1

- [ ] T013 [P] [US1] Unit test for modularity calculation in packages/wasm/tests/algorithms_test.rs
- [ ] T014 [P] [US1] Unit test for Louvain local moving phase in packages/wasm/tests/algorithms_test.rs
- [ ] T015 [P] [US1] Integration test with known community graph (Zachary's karate club) in packages/wasm/tests/algorithms_test.rs

### Implementation for User Story 1

- [ ] T016 [US1] Implement modularity calculation function in packages/wasm/src/algorithms/community.rs
- [ ] T017 [US1] Implement Louvain local moving phase (node reassignment) in packages/wasm/src/algorithms/community.rs
- [ ] T018 [US1] Implement Louvain aggregation phase (community graph construction) in packages/wasm/src/algorithms/community.rs
- [ ] T019 [US1] Implement main `detect_communities()` function with iteration loop in packages/wasm/src/algorithms/community.rs
- [ ] T020 [US1] Add weighted edge support to community detection in packages/wasm/src/algorithms/community.rs
- [ ] T021 [US1] Add resolution parameter for community granularity in packages/wasm/src/algorithms/community.rs
- [ ] T022 [US1] Expose `detectCommunities()` via wasm-bindgen in packages/wasm/src/lib.rs
- [ ] T023 [US1] Expose `getNodeCommunity()` via wasm-bindgen in packages/wasm/src/lib.rs
- [ ] T024 [US1] Add edge case handling (empty graph, single node, disconnected components) in packages/wasm/src/algorithms/community.rs
- [ ] T025 [US1] Profile and optimize for 100K node performance target in packages/wasm/src/algorithms/community.rs

**Checkpoint**: Community detection functional - can detect and query communities independently

---

## Phase 4: User Story 2 - Hull Computation (Priority: P1)

**Goal**: Compute convex/concave hull boundaries around detected communities for visualization

**Independent Test**: Provide 2D points, verify hull correctly encloses all points with valid polygon

### Tests for User Story 2

- [ ] T026 [P] [US2] Unit test for convex hull computation in packages/wasm/tests/geometry_test.rs
- [ ] T027 [P] [US2] Unit test for concave hull with varying concavity in packages/wasm/tests/geometry_test.rs
- [ ] T028 [P] [US2] Unit test for degenerate cases (1-2 nodes, collinear points) in packages/wasm/tests/geometry_test.rs

### Implementation for User Story 2

- [ ] T029 [US2] Implement convex hull wrapper using geo crate in packages/wasm/src/geometry/hull.rs
- [ ] T030 [US2] Implement concave hull wrapper with concavity parameter in packages/wasm/src/geometry/hull.rs
- [ ] T031 [US2] Implement fallback geometry for 1-2 node communities (circle approximation) in packages/wasm/src/geometry/hull.rs
- [ ] T032 [US2] Implement centroid calculation for hulls in packages/wasm/src/geometry/hull.rs
- [ ] T033 [US2] Implement `compute_hulls()` function for batch hull computation in packages/wasm/src/geometry/hull.rs
- [ ] T034 [US2] Convert hull results to Float32Array format for JS in packages/wasm/src/geometry/hull.rs
- [ ] T035 [US2] Expose `computeHulls()` via wasm-bindgen in packages/wasm/src/lib.rs
- [ ] T036 [US2] Expose `computeHull()` for single community via wasm-bindgen in packages/wasm/src/lib.rs

**Checkpoint**: Hull computation functional - can generate visual boundaries for communities

---

## Phase 5: User Story 3 - Boundary Collision Physics (Priority: P2)

**Goal**: Soft-body repulsion between community boundaries to prevent visual overlap

**Independent Test**: Create two overlapping communities, run physics, verify boundaries separate

### Tests for User Story 3

- [ ] T037 [P] [US3] Unit test for hull intersection detection in packages/wasm/tests/physics_test.rs
- [ ] T038 [P] [US3] Unit test for repulsion vector calculation in packages/wasm/tests/physics_test.rs
- [ ] T039 [P] [US3] Integration test with overlapping communities in packages/wasm/tests/physics_test.rs

### Implementation for User Story 3

- [ ] T040 [US3] Implement hull-hull intersection detection using geo in packages/wasm/src/physics/boundary.rs
- [ ] T041 [US3] Implement centroid-based repulsion vector calculation in packages/wasm/src/physics/boundary.rs
- [ ] T042 [US3] Implement node displacement distribution to community members in packages/wasm/src/physics/boundary.rs
- [ ] T043 [US3] Implement BoundaryPhysicsState for persistent physics state in packages/wasm/src/physics/boundary.rs
- [ ] T044 [US3] Add damping and max displacement constraints in packages/wasm/src/physics/boundary.rs
- [ ] T045 [US3] Expose `initBoundaryPhysics()` via wasm-bindgen in packages/wasm/src/lib.rs
- [ ] T046 [US3] Expose `updateBoundaryPhysics()` via wasm-bindgen in packages/wasm/src/lib.rs
- [ ] T047 [US3] Expose `setBoundaryPhysicsConfig()` via wasm-bindgen in packages/wasm/src/lib.rs

**Checkpoint**: Boundary physics functional - communities repel each other to prevent overlap

---

## Phase 6: User Story 4 - Centrality Computation (Priority: P2)

**Goal**: Compute various centrality measures (PageRank, betweenness, closeness, eigenvector) for node importance

**Independent Test**: Compute centrality on known graph structure, verify results match expected values

### Tests for User Story 4

- [ ] T048 [P] [US4] Unit test for PageRank on simple graph in packages/wasm/tests/algorithms_test.rs
- [ ] T049 [P] [US4] Unit test for betweenness centrality on path graph in packages/wasm/tests/algorithms_test.rs
- [ ] T050 [P] [US4] Unit test for closeness centrality on star graph in packages/wasm/tests/algorithms_test.rs

### Implementation for User Story 4

- [ ] T051 [US4] Implement PageRank wrapper using petgraph in packages/wasm/src/algorithms/centrality.rs
- [ ] T052 [US4] Implement betweenness centrality wrapper using rustworkx-core in packages/wasm/src/algorithms/centrality.rs
- [ ] T053 [US4] Implement closeness centrality wrapper using rustworkx-core in packages/wasm/src/algorithms/centrality.rs
- [ ] T054 [US4] Implement eigenvector centrality wrapper using rustworkx-core in packages/wasm/src/algorithms/centrality.rs
- [ ] T055 [US4] Implement normalization and statistics (min, max, mean) calculation in packages/wasm/src/algorithms/centrality.rs
- [ ] T056 [US4] Implement bulk result format (CentralityResultBulk) for GPU upload in packages/wasm/src/algorithms/centrality.rs
- [ ] T057 [US4] Expose `computeCentrality()` via wasm-bindgen in packages/wasm/src/lib.rs
- [ ] T058 [US4] Expose `computeCentralityBulk()` via wasm-bindgen in packages/wasm/src/lib.rs

**Checkpoint**: Centrality computation functional - can identify important nodes

---

## Phase 7: User Story 5 - Connected Components (Priority: P3)

**Goal**: Identify connected components and strongly connected components in graphs

**Independent Test**: Create graph with known disconnected components, verify correct identification

### Tests for User Story 5

- [ ] T059 [P] [US5] Unit test for connected components on disconnected graph in packages/wasm/tests/algorithms_test.rs
- [ ] T060 [P] [US5] Unit test for SCC on directed graph with cycles in packages/wasm/tests/algorithms_test.rs

### Implementation for User Story 5

- [ ] T061 [US5] Implement connected components wrapper using petgraph in packages/wasm/src/algorithms/components.rs
- [ ] T062 [US5] Implement strongly connected components wrapper using petgraph in packages/wasm/src/algorithms/components.rs
- [ ] T063 [US5] Implement node-to-component mapping construction in packages/wasm/src/algorithms/components.rs
- [ ] T064 [US5] Expose `getConnectedComponents()` via wasm-bindgen in packages/wasm/src/lib.rs
- [ ] T065 [US5] Expose `getStronglyConnectedComponents()` via wasm-bindgen in packages/wasm/src/lib.rs
- [ ] T066 [US5] Expose `getNodeComponent()` via wasm-bindgen in packages/wasm/src/lib.rs

**Checkpoint**: Component analysis functional - can identify graph connectivity structure

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T067 [P] Add progress callback support for long-running operations in packages/wasm/src/algorithms/progress.rs
- [ ] T068 [P] Integrate progress callbacks into community detection in packages/wasm/src/algorithms/community.rs
- [ ] T069 [P] Integrate progress callbacks into centrality computation in packages/wasm/src/algorithms/centrality.rs
- [ ] T070 Add comprehensive error handling with context messages across all algorithms
- [ ] T071 [P] Run quickstart.md validation scenarios
- [ ] T072 Profile all algorithms against performance targets from spec
- [ ] T073 Verify bundle size increase is under 500KB gzipped
- [ ] T074 Update TypeScript type definitions to match WASM exports in packages/core/src/types.ts

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - US1 (Community Detection) and US2 (Hull Computation) can proceed in parallel
  - US3 (Boundary Physics) depends on US2 (needs hulls)
  - US4 (Centrality) can proceed in parallel with US1/US2
  - US5 (Components) can proceed in parallel with all others
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

```
                    ┌─────────────────┐
                    │  Foundational   │
                    │   (Phase 2)     │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┬─────────────────┐
         │                   │                   │                 │
         ▼                   ▼                   ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  US1: Community │ │  US2: Hull      │ │  US4: Centrality│ │  US5: Components│
│  Detection (P1) │ │  Computation(P1)│ │  (P2)           │ │  (P3)           │
└────────┬────────┘ └────────┬────────┘ └─────────────────┘ └─────────────────┘
         │                   │
         │                   ▼
         │          ┌─────────────────┐
         │          │  US3: Boundary  │
         │          │  Physics (P2)   │
         │          └─────────────────┘
         │
         └──────────────────────────────────────────┐
                                                    ▼
                                           ┌─────────────────┐
                                           │     Polish      │
                                           │   (Phase 8)     │
                                           └─────────────────┘
```

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Types/config before core algorithms
- Core algorithms before WASM bindings
- WASM bindings expose complete functionality

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel (T002, T003, T004)
- All Foundational type tasks marked [P] can run in parallel (T008, T009, T010, T011)
- US1, US2, US4, US5 can all start in parallel after Foundational completes
- All test tasks within a story marked [P] can run in parallel
- Polish tasks marked [P] can run in parallel

---

## Parallel Example: User Story 1 (Community Detection)

```bash
# Launch all tests together (they should fail initially):
Task: "Unit test for modularity calculation in packages/wasm/tests/algorithms_test.rs"
Task: "Unit test for Louvain local moving phase in packages/wasm/tests/algorithms_test.rs"
Task: "Integration test with known community graph in packages/wasm/tests/algorithms_test.rs"

# Then implement sequentially:
Task: "Implement modularity calculation function in packages/wasm/src/algorithms/community.rs"
Task: "Implement Louvain local moving phase in packages/wasm/src/algorithms/community.rs"
# ... and so on
```

---

## Parallel Example: Multiple User Stories

```bash
# After Foundational (Phase 2) completes, launch these in parallel:

# Agent 1: Community Detection (US1)
Task: "Implement modularity calculation function in packages/wasm/src/algorithms/community.rs"

# Agent 2: Hull Computation (US2)
Task: "Implement convex hull wrapper using geo crate in packages/wasm/src/geometry/hull.rs"

# Agent 3: Centrality (US4)
Task: "Implement PageRank wrapper using petgraph in packages/wasm/src/algorithms/centrality.rs"

# Agent 4: Components (US5)
Task: "Implement connected components wrapper in packages/wasm/src/algorithms/components.rs"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Community Detection)
4. Complete Phase 4: User Story 2 (Hull Computation)
5. **STOP and VALIDATE**: Test community detection + hull visualization
6. Deploy/demo if ready - users can see community boundaries!

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add US1 + US2 → Test independently → Deploy/Demo (MVP - visual communities!)
3. Add US3 (Boundary Physics) → Communities don't overlap
4. Add US4 (Centrality) → Node importance visualization
5. Add US5 (Components) → Connectivity analysis
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers/agents:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Agent A: User Story 1 (Community Detection)
   - Agent B: User Story 2 (Hull Computation)
   - Agent C: User Story 4 (Centrality)
   - Agent D: User Story 5 (Components)
3. After US2 completes:
   - Agent B moves to: User Story 3 (Boundary Physics)
4. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Custom Louvain implementation required (see research.md) - no external community detection crate
- rustworkx-core for centrality, geo for hulls, petgraph for components
