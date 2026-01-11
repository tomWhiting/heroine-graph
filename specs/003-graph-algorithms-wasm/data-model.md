# Data Model: Graph Algorithms WASM Module

**Date**: 2026-01-12
**Feature**: 003-graph-algorithms-wasm

## Overview

This document defines the data entities used by the graph algorithms module. All entities are designed for efficient WASM-JS interop using simple types and TypedArrays.

---

## Core Entities

### Community

Represents a group of nodes identified by community detection.

| Field | Type | Description |
|-------|------|-------------|
| id | u32 | Unique community identifier |
| members | Vec<u32> | Node IDs belonging to this community |
| size | u32 | Number of members (derived) |
| modularity | f32 | Community's contribution to total modularity |

**Validation Rules**:
- `id` must be unique across all communities
- `members` must contain valid node IDs from the graph
- A node can only belong to one community

**State Transitions**: N/A (immutable result)

---

### CommunityAssignment

Mapping from node to community, returned by `detectCommunities()`.

| Field | Type | Description |
|-------|------|-------------|
| node_to_community | Map<u32, u32> | Node ID → Community ID |
| communities | Vec<Community> | Full community details |
| total_modularity | f32 | Overall modularity score |
| algorithm | String | Algorithm used ("louvain" or "leiden") |

**JS Representation**:
```typescript
interface CommunityAssignment {
  nodeToCommunity: Map<number, number>;
  communities: Community[];
  totalModularity: number;
  algorithm: 'louvain' | 'leiden';
}
```

---

### CommunityBoundary

Geometric boundary polygon around a community.

| Field | Type | Description |
|-------|------|-------------|
| community_id | u32 | ID of the community this bounds |
| hull_type | HullType | Convex or Concave |
| vertices | Vec<[f32; 2]> | Polygon vertices as [x, y] pairs |
| centroid | [f32; 2] | Center point of the hull |

**Hull Types**:
- `Convex` - Smallest convex polygon containing all points
- `Concave` - Tighter boundary following point distribution

**JS Representation**:
```typescript
interface CommunityBoundary {
  communityId: number;
  hullType: 'convex' | 'concave';
  vertices: Float32Array; // [x0, y0, x1, y1, ...]
  centroid: [number, number];
}
```

**Validation Rules**:
- `vertices` must have at least 3 points (or be a fallback for 1-2 nodes)
- Vertices must be in counter-clockwise order
- Polygon must not self-intersect

---

### CentralityResult

Centrality scores for all nodes.

| Field | Type | Description |
|-------|------|-------------|
| centrality_type | CentralityType | Type of centrality computed |
| scores | Map<u32, f32> | Node ID → centrality score |
| min | f32 | Minimum score |
| max | f32 | Maximum score |
| mean | f32 | Average score |

**Centrality Types**:
- `PageRank` - Recursive influence based on incoming links
- `Betweenness` - Fraction of shortest paths passing through node
- `Closeness` - Inverse of average distance to all other nodes
- `Eigenvector` - Influence based on neighbor influence
- `Degree` - Number of connections (normalized)
- `Katz` - Weighted by path length

**JS Representation**:
```typescript
interface CentralityResult {
  centralityType: CentralityType;
  scores: Map<number, number>;
  min: number;
  max: number;
  mean: number;
}

// Efficient bulk access
interface CentralityResultBulk {
  centralityType: CentralityType;
  nodeIds: Uint32Array;
  scores: Float32Array;
  min: number;
  max: number;
  mean: number;
}
```

---

### ComponentResult

Connected component identification.

| Field | Type | Description |
|-------|------|-------------|
| component_type | ComponentType | Weak or Strong connectivity |
| components | Vec<Component> | List of components |
| node_to_component | Map<u32, u32> | Node ID → Component ID |

**Component Types**:
- `Weak` - Undirected connectivity (ignores edge direction)
- `Strong` - Directed connectivity (respects edge direction)

**Component**:
| Field | Type | Description |
|-------|------|-------------|
| id | u32 | Component identifier |
| members | Vec<u32> | Node IDs in this component |
| size | u32 | Number of members |

---

### BoundaryPhysicsConfig

Configuration for boundary collision physics.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| enabled | bool | true | Whether physics is active |
| repulsion_strength | f32 | 0.5 | Force magnitude (0.0-1.0) |
| damping | f32 | 0.9 | Velocity decay per frame |
| max_displacement | f32 | 10.0 | Maximum node movement per update |

**JS Representation**:
```typescript
interface BoundaryPhysicsConfig {
  enabled?: boolean;
  repulsionStrength?: number;
  damping?: number;
  maxDisplacement?: number;
}
```

---

### BoundaryPhysicsResult

Result of boundary physics update.

| Field | Type | Description |
|-------|------|-------------|
| displacements | Vec<(u32, [f32; 2])> | Node ID → displacement vector |
| has_overlaps | bool | Whether any boundaries still overlap |
| iteration | u32 | Current physics iteration count |

**JS Representation**:
```typescript
interface BoundaryPhysicsResult {
  // Efficient bulk format
  nodeIds: Uint32Array;
  displacementsX: Float32Array;
  displacementsY: Float32Array;
  hasOverlaps: boolean;
  iteration: number;
}
```

---

## Configuration Entities

### CommunityDetectionConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| algorithm | Algorithm | Louvain | Algorithm to use |
| resolution | f32 | 1.0 | Resolution parameter (higher = more communities) |
| weighted | bool | false | Whether to use edge weights |
| max_iterations | u32 | 100 | Maximum optimization iterations |
| min_modularity_gain | f32 | 0.0001 | Stop when gain below this |

### HullComputationConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| hull_type | HullType | Convex | Type of hull to compute |
| concavity | f32 | 2.0 | Concavity parameter (concave only) |
| min_points_for_hull | u32 | 3 | Minimum points for polygon hull |
| fallback_radius | f32 | 10.0 | Circle radius for 1-2 point communities |

### CentralityConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| centrality_type | CentralityType | PageRank | Type to compute |
| normalized | bool | true | Whether to normalize scores |
| max_iterations | u32 | 100 | For iterative algorithms |
| tolerance | f32 | 1e-6 | Convergence tolerance |
| damping | f32 | 0.85 | PageRank damping factor |

---

## Relationships

```
┌─────────────────┐       ┌──────────────────────┐
│   GraphEngine   │──────▶│  CommunityAssignment │
│   (existing)    │       │                      │
└────────┬────────┘       └──────────┬───────────┘
         │                           │
         │                           ▼
         │                ┌──────────────────────┐
         │                │      Community       │
         │                └──────────┬───────────┘
         │                           │
         │                           ▼
         │                ┌──────────────────────┐
         │                │  CommunityBoundary   │
         │                └──────────────────────┘
         │
         ▼
┌─────────────────┐       ┌──────────────────────┐
│ CentralityResult│       │   ComponentResult    │
└─────────────────┘       └──────────────────────┘
```

---

## TypedArray Conventions

For efficient GPU upload and bulk operations:

| Data Type | TypedArray | Layout |
|-----------|------------|--------|
| Node IDs | Uint32Array | [id0, id1, ...] |
| Positions | Float32Array | [x0, y0, x1, y1, ...] |
| Scores | Float32Array | [score0, score1, ...] |
| Polygon vertices | Float32Array | [x0, y0, x1, y1, ...] |
| Community membership | Uint32Array | Index = node index, value = community ID |

This enables zero-copy transfer between WASM and JS, and direct upload to GPU buffers.
