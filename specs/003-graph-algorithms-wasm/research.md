# Research: Graph Algorithms WASM Module

**Date**: 2026-01-12
**Feature**: 003-graph-algorithms-wasm

## Research Tasks

1. Community detection library selection for WASM
2. Centrality computation via rustworkx-core
3. Hull computation via geo crate
4. Boundary physics implementation approach

---

## 1. Community Detection

### Decision
**Implement Louvain in-house** using petgraph's graph structure. Do not use external crates for community detection.

### Rationale
- **single-clustering** (v0.6.1): Depends on `rayon`, `ndarray`, `hnsw_rs` - NOT WASM compatible without complex nightly builds
- **louvain-rs**: Targets C library output (`liblouvain.a`), not designed for WASM
- **fast-louvain**: Work in progress, incomplete
- Louvain algorithm is well-documented and straightforward to implement
- Our performance target (100K nodes < 1 second) is achievable with a clean single-threaded implementation
- We already have petgraph which provides the graph structure needed

### Implementation Approach
1. Implement modularity calculation per the [Louvain method](https://en.wikipedia.org/wiki/Louvain_method)
2. Use iterative local moving phase + aggregation phase
3. Single-threaded implementation for WASM compatibility
4. Optional: Add Leiden refinement phase later if needed

### Alternatives Considered
| Option | Why Rejected |
|--------|--------------|
| single-clustering | rayon dependency blocks WASM |
| wasm-bindgen-rayon | Requires nightly Rust, complex setup |
| louvain-rs | Designed for C FFI, not WASM |
| NetworkX via Pyodide | Python in browser, too heavy |

---

## 2. Centrality Computation

### Decision
Use **rustworkx-core** (v0.17.1) for betweenness, closeness, eigenvector centrality. Use **petgraph** (already in deps) for PageRank.

### Rationale
- rustworkx-core provides 7 centrality algorithms, all pure Rust
- Built on petgraph - compatible with our existing GraphEngine
- No rayon dependency in core centrality functions
- Well-tested (used by IBM's Qiskit quantum computing framework)

### Available Algorithms
From [rustworkx-core::centrality](https://docs.rs/rustworkx-core/latest/rustworkx_core/centrality/):
- `betweenness_centrality` - FR-019
- `closeness_centrality` - FR-020
- `eigenvector_centrality` - FR-021
- `degree_centrality` - bonus
- `katz_centrality` - bonus
- `edge_betweenness_centrality` - bonus

### Integration
```rust
// Convert GraphEngine's StableGraph to rustworkx-core compatible format
use rustworkx_core::centrality::betweenness_centrality;

let scores = betweenness_centrality(&graph, false, false, Some(100));
```

---

## 3. Hull Computation

### Decision
Use **geo** crate (v0.32.0) for convex and concave hull computation.

### Rationale
- Production-ready, widely used
- Pure Rust, no system dependencies
- Provides both `ConvexHull` and `ConcaveHull` traits
- Handles edge cases (collinear points, few points)

### API
From [geo::algorithm::concave_hull](https://docs.rs/geo/latest/geo/algorithm/concave_hull/index.html):
```rust
use geo::{ConvexHull, ConcaveHull, MultiPoint};

// Convex hull
let hull = points.convex_hull();

// Concave hull with concavity parameter
let concave = points.concave_hull(2.0);
```

### Concavity Parameter
- Higher values = tighter hull (more concave)
- Lower values = looser hull (approaches convex)
- Default recommendation: 2.0 for balanced results

---

## 4. Boundary Collision Physics

### Decision
**Implement custom** soft-body boundary repulsion physics.

### Rationale
- No existing Rust crate provides hull-to-hull collision with repulsion forces
- Requirements are specific: repel hulls, apply forces to member nodes
- Simple physics model sufficient for visualization purposes

### Implementation Approach
1. **Collision Detection**: Use geo's `intersects()` to detect hull overlaps
2. **Centroid Calculation**: Compute centroid of each community
3. **Repulsion Vector**: Direction from overlapping centroids
4. **Force Application**: Apply displacement to member nodes proportional to repulsion strength

### Algorithm Sketch
```rust
fn update_boundary_physics(
    communities: &[Community],
    hulls: &[Polygon],
    strength: f32,
) -> Vec<(NodeId, Vec2)> {
    let mut displacements = Vec::new();

    for i in 0..hulls.len() {
        for j in (i+1)..hulls.len() {
            if hulls[i].intersects(&hulls[j]) {
                let centroid_i = hulls[i].centroid();
                let centroid_j = hulls[j].centroid();
                let direction = (centroid_i - centroid_j).normalize();
                let force = direction * strength;

                // Apply force to all nodes in community i
                for node in &communities[i].members {
                    displacements.push((*node, force));
                }
                // Apply opposite force to community j
                for node in &communities[j].members {
                    displacements.push((*node, -force));
                }
            }
        }
    }

    displacements
}
```

---

## 5. Connected Components

### Decision
Use **petgraph** (already in deps) for connected components and SCC.

### Rationale
- Already implemented in petgraph's algo module
- `petgraph::algo::connected_components` for undirected
- `petgraph::algo::tarjan_scc` for strongly connected components
- No additional dependencies needed

---

## Dependency Summary

### Add to Cargo.toml
```toml
[dependencies]
# Existing
petgraph = { version = "0.8.3", default-features = false, features = ["stable_graph"] }
wasm-bindgen = "0.2"
js-sys = "0.3"

# New
rustworkx-core = "0.17"
geo = "0.32"
```

### NOT Using
- single-clustering (rayon dependency)
- louvain-rs (C FFI focused)
- fast-louvain (incomplete)

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Custom Louvain slower than expected | Profile early; algorithm is O(n log n), should meet 100K < 1s target |
| rustworkx-core WASM issues | Test early; pure Rust, should work |
| geo hull edge cases | Extensive test cases for degenerate inputs |
| Boundary physics instability | Add damping parameter; limit iterations per frame |

---

## Resolved Clarifications

None - all technical decisions made based on research.
