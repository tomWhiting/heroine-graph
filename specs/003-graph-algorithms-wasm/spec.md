# Feature Specification: Graph Algorithms WASM Module

**Feature Branch**: `003-graph-algorithms-wasm`
**Created**: 2026-01-12
**Status**: Draft
**Input**: Graph Algorithms WASM Module for heroine-graph: A Rust/WASM module providing CPU-side graph algorithms that complement GPU-based rendering. Core algorithms include: (1) Community detection using Louvain and Leiden algorithms for identifying node clusters, (2) Centrality measures including PageRank, betweenness centrality, closeness centrality, and eigenvector centrality, (3) Connected components and strongly connected components, (4) Hull computation for community boundaries using convex hull and concave hull algorithms, (5) Boundary collision physics for soft-body repulsion between community hulls so grouped nodes don't overlap. The module exposes a JavaScript API including detectCommunities(), computeCentrality(), computeHulls(), and updateBoundaryPhysics(). Results feed into the visualization layer for rendering community boundaries, coloring nodes by centrality, and animating boundary collisions. Performance target: community detection on 100K nodes in under 1 second.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Community Detection (Priority: P1)

A library user wants to automatically identify clusters of related nodes in their graph. They have a social network or code dependency graph and want to visualize which nodes naturally group together based on connection patterns. They call a single function and receive community assignments for all nodes.

**Why this priority**: Community detection is the primary use case driving this module. It enables the key visualization feature of drawing boundaries around node groups and is foundational for all boundary-related features.

**Independent Test**: Can be fully tested by loading a graph with known community structure (e.g., a multi-module codebase), running community detection, and verifying nodes are grouped correctly.

**Acceptance Scenarios**:

1. **Given** a graph with 1,000 nodes in 5 distinct clusters, **When** the user calls `detectCommunities()`, **Then** each node is assigned a community ID that correctly reflects its cluster membership
2. **Given** a graph with weighted edges, **When** the user calls `detectCommunities({ weighted: true })`, **Then** edge weights influence community assignments (strongly connected nodes grouped together)
3. **Given** a large graph with 100,000 nodes, **When** community detection runs, **Then** results are returned in under 1 second
4. **Given** a disconnected graph, **When** community detection runs, **Then** disconnected components are assigned to separate communities
5. **Given** community detection results, **When** the user queries a specific node, **Then** they can retrieve that node's community ID

---

### User Story 2 - Hull Computation for Community Boundaries (Priority: P1)

A library user wants to visualize community boundaries as shapes drawn around grouped nodes. After detecting communities, they want to compute geometric boundaries (convex or concave hulls) around each community so the visualization layer can render them.

**Why this priority**: Hull computation is essential for the visual representation of communities. Without boundaries, community detection has no visual output.

**Independent Test**: Can be tested by providing a set of 2D points (node positions) and verifying the computed hull correctly encloses all points.

**Acceptance Scenarios**:

1. **Given** community assignments and node positions, **When** the user calls `computeHulls()`, **Then** a boundary polygon is returned for each community
2. **Given** a community with 3 or more nodes, **When** convex hull is computed, **Then** the resulting polygon contains all community nodes
3. **Given** a community with scattered nodes, **When** concave hull is computed with concavity parameter, **Then** the hull follows the point distribution more closely than a convex hull
4. **Given** a community with only 1-2 nodes, **When** hull computation runs, **Then** a reasonable fallback is returned (circle or line segment)
5. **Given** hull computation results, **When** node positions change (simulation running), **Then** hulls can be efficiently recomputed

---

### User Story 3 - Boundary Collision Physics (Priority: P2)

A library user wants community boundaries to behave like soft bodies that repel each other, preventing overlap. When communities are positioned close together, the boundaries should push apart to maintain visual separation while nodes continue their force-directed simulation.

**Why this priority**: Collision physics enhances visual clarity by preventing community overlap, but the core value (seeing communities) is delivered by P1 features.

**Independent Test**: Can be tested by creating two overlapping communities, running boundary physics, and verifying the boundaries separate without losing internal coherence.

**Acceptance Scenarios**:

1. **Given** two community boundaries that overlap, **When** boundary physics is applied, **Then** the boundaries push apart to eliminate overlap
2. **Given** boundary collision enabled, **When** nodes are being simulated, **Then** boundary repulsion forces are applied to community member nodes
3. **Given** boundary physics configuration, **When** the user adjusts repulsion strength, **Then** boundaries separate more or less aggressively
4. **Given** multiple communities, **When** all boundaries are computed, **Then** no two boundaries overlap after physics stabilizes
5. **Given** boundary physics running, **When** the user disables it, **Then** boundaries stop repelling and communities can overlap

---

### User Story 4 - Centrality Computation (Priority: P2)

A library user wants to identify the most important or influential nodes in their graph. They want to compute various centrality measures (PageRank, betweenness, closeness, eigenvector) and use the results to size or color nodes in the visualization.

**Why this priority**: Centrality is a valuable analytical feature but doesn't enable the core community visualization. It complements the primary use case.

**Independent Test**: Can be tested by computing centrality on a known graph structure and verifying results match expected values.

**Acceptance Scenarios**:

1. **Given** a graph, **When** the user calls `computeCentrality('pagerank')`, **Then** each node receives a PageRank score between 0 and 1
2. **Given** a graph, **When** the user calls `computeCentrality('betweenness')`, **Then** nodes on critical paths receive higher scores
3. **Given** a graph, **When** the user calls `computeCentrality('closeness')`, **Then** central nodes receive higher scores than peripheral nodes
4. **Given** centrality results, **When** the user requests a specific node's score, **Then** they can retrieve it by node ID
5. **Given** a graph with 50,000 nodes, **When** PageRank is computed, **Then** results are returned in under 500ms

---

### User Story 5 - Connected Components (Priority: P3)

A library user wants to identify which parts of their graph are connected. They want to find all connected components (for undirected graphs) or strongly connected components (for directed graphs) to understand the graph's structure.

**Why this priority**: Connected components are a basic graph property that's already partially available. Lower priority as it's a supporting feature.

**Independent Test**: Can be tested by creating a graph with known disconnected components and verifying correct identification.

**Acceptance Scenarios**:

1. **Given** an undirected graph with 3 disconnected subgraphs, **When** the user calls `getConnectedComponents()`, **Then** 3 components are identified with correct node membership
2. **Given** a directed graph, **When** the user calls `getStronglyConnectedComponents()`, **Then** SCCs are correctly identified
3. **Given** component results, **When** the user queries component for a node, **Then** they receive the component ID

---

### Edge Cases

- What happens when community detection is called on an empty graph? (Return empty result, no error)
- How does hull computation handle a community with all nodes at the same position? (Return a point or small circle)
- What happens when boundary physics runs with only one community? (No-op, single boundary cannot collide with itself)
- How does centrality handle disconnected graphs? (Compute per-component, normalize appropriately)
- What happens when the user requests an unsupported centrality type? (Return error with list of supported types)
- How does the system handle graphs with self-loops for community detection? (Include in computation, treat as edge weight)
- What happens when node positions are updated mid-computation? (Use positions at computation start, ignore updates until complete)

## Requirements *(mandatory)*

### Functional Requirements

**Community Detection**
- **FR-001**: System MUST provide `detectCommunities()` method returning community ID for each node
- **FR-002**: System MUST support Leiden algorithm for community detection (preferred over Louvain)
- **FR-003**: System MUST support Louvain algorithm as fallback option
- **FR-004**: Community detection MUST accept optional edge weight consideration
- **FR-005**: System MUST support configurable resolution parameter for community granularity
- **FR-006**: System MUST return community membership as a mapping from node ID to community ID

**Hull Computation**
- **FR-007**: System MUST provide `computeHulls()` method accepting community assignments and node positions
- **FR-008**: System MUST support convex hull computation for each community
- **FR-009**: System MUST support concave hull computation with configurable concavity parameter
- **FR-010**: Hull results MUST be returned as arrays of polygon vertices suitable for rendering
- **FR-011**: System MUST handle degenerate cases (1-2 nodes) with reasonable fallback geometries

**Boundary Collision Physics**
- **FR-012**: System MUST provide `updateBoundaryPhysics()` method for boundary repulsion
- **FR-013**: Boundary physics MUST compute repulsion forces between overlapping community boundaries
- **FR-014**: System MUST support configurable repulsion strength parameter
- **FR-015**: Boundary physics MUST return displacement vectors for affected nodes
- **FR-016**: System MUST support enabling/disabling boundary collision independently

**Centrality Measures**
- **FR-017**: System MUST provide `computeCentrality(type)` method supporting multiple centrality types
- **FR-018**: System MUST support PageRank centrality
- **FR-019**: System MUST support betweenness centrality
- **FR-020**: System MUST support closeness centrality
- **FR-021**: System MUST support eigenvector centrality
- **FR-022**: Centrality results MUST be returned as a mapping from node ID to score

**Connected Components**
- **FR-023**: System MUST provide `getConnectedComponents()` for undirected component detection
- **FR-024**: System MUST provide `getStronglyConnectedComponents()` for directed graphs
- **FR-025**: Component results MUST include node membership for each component

**JavaScript API**
- **FR-026**: All algorithms MUST be callable from JavaScript via the module's public API
- **FR-027**: Results MUST be returned in JavaScript-native formats (arrays, objects, TypedArrays)
- **FR-028**: System MUST support both synchronous and asynchronous operation modes
- **FR-029**: System MUST provide progress callbacks for long-running operations on large graphs

### Key Entities

- **Community**: A group of nodes identified by community detection. Contains: community ID, member node IDs, optional metadata (modularity score, size).

- **CommunityBoundary**: A geometric boundary around a community. Contains: community ID, hull type (convex/concave), polygon vertices as coordinate pairs.

- **CentralityResult**: Centrality scores for all nodes. Contains: centrality type, mapping of node ID to score, optional statistics (min, max, mean).

- **ComponentResult**: Connected component identification. Contains: component ID, member node IDs, component type (weak/strong).

- **BoundaryPhysicsState**: State for boundary collision simulation. Contains: community boundaries, repulsion parameters, current displacement vectors.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Community detection completes on 100,000 nodes in under 1 second
- **SC-002**: Hull computation for 100 communities completes in under 100ms
- **SC-003**: Boundary physics update for 50 communities completes in under 16ms (60fps capable)
- **SC-004**: PageRank computation on 50,000 nodes completes in under 500ms
- **SC-005**: Betweenness centrality on 10,000 nodes completes in under 2 seconds
- **SC-006**: Module adds less than 500KB to the total bundle size (gzipped)
- **SC-007**: All algorithms produce deterministic results for identical inputs
- **SC-008**: Community detection correctly identifies 95%+ of nodes in graphs with known ground-truth communities
- **SC-009**: Boundary collision eliminates 100% of community overlaps when enabled

## Assumptions

- The existing WASM module infrastructure (wasm-bindgen, js-sys) is available and working
- Node positions are provided as Float32Arrays from the GPU readback or user input
- The graph structure is already loaded in the GraphEngine (petgraph-based)
- Community detection operates on the full graph; subgraph detection is out of scope
- Boundary physics operates independently from the GPU force simulation
- Users will call algorithms explicitly; no automatic re-computation on graph changes

## Out of Scope

- GPU-accelerated algorithm implementations (this module is CPU/WASM only)
- Real-time streaming community detection (batch processing only)
- Overlapping community detection (each node belongs to exactly one community)
- 3D hull computation (2D boundaries only)
- Automatic algorithm parameter tuning
- Visualization/rendering of results (handled by visualization layer)
- Graph modification based on algorithm results (read-only analysis)
