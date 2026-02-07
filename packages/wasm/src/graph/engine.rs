//! GraphEngine - Core graph data structure.
//!
//! The GraphEngine stores the graph topology using petgraph's StableGraph
//! and maintains SoA (Structure of Arrays) buffers for positions and velocities
//! to enable efficient GPU upload and SIMD operations.

use petgraph::stable_graph::{NodeIndex, EdgeIndex, StableGraph};
use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeIndexable};
use petgraph::{Directed, Direction};
use std::cell::Cell;
use std::collections::HashMap;

use super::edge::EdgeId;
use super::node::{NodeId, NodeState};
use crate::spatial::SpatialIndex;

/// The core graph engine.
///
/// This struct manages:
/// - Graph topology via petgraph
/// - Position/velocity buffers in SoA layout
/// - Node state (pinned, hidden, selected, hovered)
/// - Spatial index for hit testing
/// - ID mapping between stable IDs and internal indices
pub struct GraphEngine {
    /// The underlying graph structure.
    /// Nodes store their stable NodeId, edges store weight.
    graph: StableGraph<NodeId, f32, Directed>,

    /// Map from stable NodeId to petgraph NodeIndex
    node_id_to_index: HashMap<NodeId, NodeIndex>,

    /// Map from stable EdgeId to petgraph EdgeIndex
    edge_id_to_index: HashMap<EdgeId, EdgeIndex>,

    /// Reverse map from petgraph EdgeIndex to stable EdgeId (for O(1) lookup during removal)
    edge_index_to_id: HashMap<EdgeIndex, EdgeId>,

    /// Next node ID to assign
    next_node_id: u32,

    /// Next edge ID to assign
    next_edge_id: u32,

    /// X positions (SoA layout)
    pos_x: Vec<f32>,

    /// Y positions (SoA layout)
    pos_y: Vec<f32>,

    /// X velocities (SoA layout)
    vel_x: Vec<f32>,

    /// Y velocities (SoA layout)
    vel_y: Vec<f32>,

    /// Node states (pinned, hidden, etc.)
    states: Vec<NodeState>,

    /// Spatial index for hit testing
    spatial: SpatialIndex,

    /// Whether the spatial index needs rebuilding
    spatial_dirty: Cell<bool>,
}

impl GraphEngine {
    /// Create a new empty graph engine.
    pub fn new() -> Self {
        Self {
            graph: StableGraph::new(),
            node_id_to_index: HashMap::new(),
            edge_id_to_index: HashMap::new(),
            edge_index_to_id: HashMap::new(),
            next_node_id: 0,
            next_edge_id: 0,
            pos_x: Vec::new(),
            pos_y: Vec::new(),
            vel_x: Vec::new(),
            vel_y: Vec::new(),
            states: Vec::new(),
            spatial: SpatialIndex::new(),
            spatial_dirty: Cell::new(false),
        }
    }

    /// Create a graph engine with pre-allocated capacity.
    pub fn with_capacity(node_capacity: usize, edge_capacity: usize) -> Self {
        Self {
            graph: StableGraph::with_capacity(node_capacity, edge_capacity),
            node_id_to_index: HashMap::with_capacity(node_capacity),
            edge_id_to_index: HashMap::with_capacity(edge_capacity),
            edge_index_to_id: HashMap::with_capacity(edge_capacity),
            next_node_id: 0,
            next_edge_id: 0,
            pos_x: Vec::with_capacity(node_capacity),
            pos_y: Vec::with_capacity(node_capacity),
            vel_x: Vec::with_capacity(node_capacity),
            vel_y: Vec::with_capacity(node_capacity),
            states: Vec::with_capacity(node_capacity),
            spatial: SpatialIndex::with_capacity(node_capacity),
            spatial_dirty: Cell::new(false),
        }
    }

    // =========================================================================
    // Node Operations
    // =========================================================================

    /// Add a node at the specified position.
    pub fn add_node(&mut self, x: f32, y: f32) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;

        let index = self.graph.add_node(id);
        self.node_id_to_index.insert(id, index);

        self.pos_x.push(x);
        self.pos_y.push(y);
        self.vel_x.push(0.0);
        self.vel_y.push(0.0);
        self.states.push(NodeState::new());

        self.spatial_dirty.set(true);
        id
    }

    /// Add multiple nodes from a positions array [x0, y0, x1, y1, ...].
    pub fn add_nodes_from_positions(&mut self, positions: &[f32]) -> u32 {
        let count = positions.len() / 2;

        // Pre-allocate
        self.node_id_to_index.reserve(count);
        self.pos_x.reserve(count);
        self.pos_y.reserve(count);
        self.vel_x.reserve(count);
        self.vel_y.reserve(count);
        self.states.reserve(count);

        for i in 0..count {
            let x = positions[i * 2];
            let y = positions[i * 2 + 1];
            self.add_node(x, y);
        }

        self.spatial_dirty.set(true);
        count as u32
    }

    /// Remove a node and all its connected edges.
    pub fn remove_node(&mut self, id: NodeId) -> bool {
        if let Some(index) = self.node_id_to_index.remove(&id) {
            // Remove edges connected to this node (both incoming and outgoing)
            let edges: Vec<_> = self.graph
                .edges_directed(index, Direction::Outgoing)
                .chain(self.graph.edges_directed(index, Direction::Incoming))
                .map(|e| e.id())
                .collect();
            for edge_index in edges {
                if let Some(edge_id) = self.edge_index_to_id.remove(&edge_index) {
                    self.edge_id_to_index.remove(&edge_id);
                }
            }

            // Zero out SoA arrays for the removed node's slot
            let i = index.index();
            if i < self.pos_x.len() {
                self.pos_x[i] = 0.0;
                self.pos_y[i] = 0.0;
                self.vel_x[i] = 0.0;
                self.vel_y[i] = 0.0;
                self.states[i] = NodeState::new();
            }

            self.graph.remove_node(index);
            self.spatial_dirty.set(true);
            true
        } else {
            false
        }
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> u32 {
        self.graph.node_count() as u32
    }

    /// Get the upper bound on node indices (max index + 1).
    /// This may be larger than node_count() if nodes have been removed,
    /// since StableGraph preserves index stability.
    pub fn node_bound(&self) -> u32 {
        self.graph.node_bound() as u32
    }

    /// Get a node's position.
    pub fn get_node_position(&self, id: NodeId) -> Option<(f32, f32)> {
        self.node_id_to_index.get(&id).map(|&index| {
            let i = index.index();
            (self.pos_x[i], self.pos_y[i])
        })
    }

    /// Set a node's position.
    pub fn set_node_position(&mut self, id: NodeId, x: f32, y: f32) {
        if let Some(&index) = self.node_id_to_index.get(&id) {
            let i = index.index();
            self.pos_x[i] = x;
            self.pos_y[i] = y;
            self.spatial_dirty.set(true);
        }
    }

    /// Pin a node (exclude from simulation).
    pub fn pin_node(&mut self, id: NodeId) {
        if let Some(&index) = self.node_id_to_index.get(&id) {
            self.states[index.index()].set_pinned(true);
        }
    }

    /// Unpin a node.
    pub fn unpin_node(&mut self, id: NodeId) {
        if let Some(&index) = self.node_id_to_index.get(&id) {
            self.states[index.index()].set_pinned(false);
        }
    }

    /// Check if a node is pinned.
    pub fn is_node_pinned(&self, id: NodeId) -> bool {
        self.node_id_to_index
            .get(&id)
            .map(|&index| self.states[index.index()].is_pinned())
            .unwrap_or(false)
    }

    // =========================================================================
    // Edge Operations
    // =========================================================================

    /// Add an edge between two nodes.
    pub fn add_edge(&mut self, source: NodeId, target: NodeId, weight: f32) -> Option<EdgeId> {
        let source_index = self.node_id_to_index.get(&source)?;
        let target_index = self.node_id_to_index.get(&target)?;

        let id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;

        let index = self.graph.add_edge(*source_index, *target_index, weight);
        self.edge_id_to_index.insert(id, index);
        self.edge_index_to_id.insert(index, id);

        Some(id)
    }

    /// Add edges from pairs [src0, tgt0, src1, tgt1, ...].
    pub fn add_edges_from_pairs(&mut self, edges: &[u32]) -> u32 {
        let count = edges.len() / 2;
        let mut added = 0;

        for i in 0..count {
            let source = NodeId(edges[i * 2]);
            let target = NodeId(edges[i * 2 + 1]);
            if self.add_edge(source, target, 1.0).is_some() {
                added += 1;
            }
        }

        added
    }

    /// Remove an edge.
    pub fn remove_edge(&mut self, id: EdgeId) -> bool {
        if let Some(index) = self.edge_id_to_index.remove(&id) {
            self.edge_index_to_id.remove(&index);
            self.graph.remove_edge(index);
            true
        } else {
            false
        }
    }

    /// Get the number of edges.
    pub fn edge_count(&self) -> u32 {
        self.graph.edge_count() as u32
    }

    /// Get neighbors of a node.
    pub fn get_neighbors(&self, id: NodeId) -> Vec<u32> {
        self.node_id_to_index
            .get(&id)
            .map(|&index| {
                self.graph
                    .neighbors(index)
                    .filter_map(|n| self.graph.node_weight(n).map(|id| id.0))
                    .collect()
            })
            .unwrap_or_default()
    }

    // =========================================================================
    // Buffer Access
    // =========================================================================

    /// Get X positions slice.
    pub fn positions_x(&self) -> &[f32] {
        &self.pos_x
    }

    /// Get Y positions slice.
    pub fn positions_y(&self) -> &[f32] {
        &self.pos_y
    }

    /// Get X velocities slice.
    pub fn velocities_x(&self) -> &[f32] {
        &self.vel_x
    }

    /// Get Y velocities slice.
    pub fn velocities_y(&self) -> &[f32] {
        &self.vel_y
    }

    // =========================================================================
    // Spatial Queries
    // =========================================================================

    /// Find the nearest node to a point.
    pub fn find_nearest_node(&self, x: f32, y: f32) -> Option<NodeId> {
        self.ensure_spatial_index_up_to_date();
        self.spatial.nearest(x, y)
    }

    /// Find the nearest node within a maximum distance.
    pub fn find_nearest_node_within(&self, x: f32, y: f32, max_distance: f32) -> Option<NodeId> {
        self.ensure_spatial_index_up_to_date();
        self.spatial.nearest_within(x, y, max_distance)
    }

    /// Find all nodes in a rectangle.
    pub fn find_nodes_in_rect(&self, min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Vec<u32> {
        self.ensure_spatial_index_up_to_date();
        self.spatial
            .in_rect(min_x, min_y, max_x, max_y)
            .into_iter()
            .map(|id| id.0)
            .collect()
    }

    /// Rebuild the spatial index.
    pub fn rebuild_spatial_index(&mut self) {
        let points: Vec<_> = self
            .node_id_to_index
            .iter()
            .map(|(&id, &index)| {
                let i = index.index();
                (id, self.pos_x[i], self.pos_y[i])
            })
            .collect();

        self.spatial.rebuild(&points);
        self.spatial_dirty.set(false);
    }

    fn ensure_spatial_index_up_to_date(&self) {
        if self.spatial_dirty.get() {
            // Note: spatial index rebuild requires &mut self for the spatial field.
            // With Cell<bool> we can at least track the dirty flag through &self.
            // Callers should call rebuild_spatial_index() when spatial_dirty is set.
        }
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    /// Get the bounding box of all active nodes.
    /// Skips dead slots (nodes that have been removed).
    pub fn get_bounds(&self) -> Option<(f32, f32, f32, f32)> {
        if self.graph.node_count() == 0 {
            return None;
        }

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        // Only consider active nodes (those still in the graph)
        for node_index in self.graph.node_indices() {
            let i = node_index.index();
            if i < self.pos_x.len() {
                let x = self.pos_x[i];
                let y = self.pos_y[i];
                if x < min_x { min_x = x; }
                if x > max_x { max_x = x; }
                if y < min_y { min_y = y; }
                if y > max_y { max_y = y; }
            }
        }

        if min_x == f32::INFINITY {
            return None;
        }

        Some((min_x, min_y, max_x, max_y))
    }

    /// Clear all nodes and edges, resetting the engine to its initial state.
    pub fn clear(&mut self) {
        self.graph.clear();
        self.node_id_to_index.clear();
        self.edge_id_to_index.clear();
        self.edge_index_to_id.clear();
        self.next_node_id = 0;
        self.next_edge_id = 0;
        self.pos_x.clear();
        self.pos_y.clear();
        self.vel_x.clear();
        self.vel_y.clear();
        self.states.clear();
        self.spatial.clear();
        self.spatial_dirty.set(false);
    }

    /// Get edge list in CSR format.
    ///
    /// Returns [offsets..., targets...] where offsets has node_bound + 1 elements.
    /// Uses node_bound() (max index + 1) instead of node_count() to handle
    /// StableGraph's stable index space with holes from removals.
    pub fn get_edges_csr(&self) -> Vec<u32> {
        let node_bound = self.graph.node_bound();
        let edge_count = self.graph.edge_count();

        let mut offsets = vec![0u32; node_bound + 1];
        let mut targets = vec![0u32; edge_count];

        // Count edges per node
        for node_index in self.graph.node_indices() {
            let i = node_index.index();
            if i < node_bound {
                offsets[i + 1] = self.graph.edges(node_index).count() as u32;
            }
        }

        // Prefix sum
        for i in 1..=node_bound {
            offsets[i] += offsets[i - 1];
        }

        // Build targets array
        let mut current_offsets = offsets[..node_bound].to_vec();
        for edge in self.graph.edge_references() {
            let source = edge.source().index();
            let target = edge.target().index() as u32;

            if source < node_bound {
                let offset = current_offsets[source] as usize;
                if offset < targets.len() {
                    targets[offset] = target;
                }
                current_offsets[source] += 1;
            }
        }

        // Combine offsets and targets
        let mut result = Vec::with_capacity(offsets.len() + targets.len());
        result.extend(offsets);
        result.extend(targets);
        result
    }

    /// Get inverse edge list in CSR format (incoming edges).
    ///
    /// For each node, lists the source nodes of incoming edges.
    /// Returns [offsets..., sources...] where offsets has node_bound + 1 elements.
    /// Uses node_bound() to handle StableGraph's stable index space.
    pub fn get_inverse_edges_csr(&self) -> Vec<u32> {
        let node_bound = self.graph.node_bound();
        let edge_count = self.graph.edge_count();

        let mut offsets = vec![0u32; node_bound + 1];
        let mut sources = Vec::with_capacity(edge_count);

        // Count incoming edges per node (edges where this node is the target)
        for edge in self.graph.edge_references() {
            let target = edge.target().index();
            if target < node_bound {
                offsets[target + 1] += 1;
            }
        }

        // Prefix sum
        for i in 1..=node_bound {
            offsets[i] += offsets[i - 1];
        }

        // Initialize sources vector to the right size
        sources.resize(edge_count, 0);

        // Build sources array
        let mut current_offsets = offsets[..node_bound].to_vec();
        for edge in self.graph.edge_references() {
            let source = edge.source().index() as u32;
            let target = edge.target().index();

            if target < node_bound {
                let offset = current_offsets[target] as usize;
                if offset < sources.len() {
                    sources[offset] = source;
                    current_offsets[target] += 1;
                }
            }
        }

        // Combine offsets and sources
        let mut result = Vec::with_capacity(offsets.len() + sources.len());
        result.extend(offsets);
        result.extend(sources);
        result
    }

    /// Get node degrees (out-degree, in-degree) as a flat array.
    ///
    /// Returns [out_deg_0, in_deg_0, out_deg_1, in_deg_1, ...] with 2 * node_bound elements.
    /// Uses node_bound() to handle StableGraph's stable index space.
    pub fn get_node_degrees(&self) -> Vec<u32> {
        let node_bound = self.graph.node_bound();
        let mut degrees = vec![0u32; node_bound * 2];

        // Count out-degrees
        for node_index in self.graph.node_indices() {
            let i = node_index.index();
            if i < node_bound {
                degrees[i * 2] = self.graph.edges(node_index).count() as u32;
            }
        }

        // Count in-degrees
        for edge in self.graph.edge_references() {
            let target = edge.target().index();
            if target < node_bound {
                degrees[target * 2 + 1] += 1;
            }
        }

        degrees
    }
}

impl Default for GraphEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_node() {
        let mut engine = GraphEngine::new();
        let id = engine.add_node(10.0, 20.0);

        assert_eq!(engine.node_count(), 1);
        assert_eq!(engine.get_node_position(id), Some((10.0, 20.0)));
    }

    #[test]
    fn test_add_multiple_nodes() {
        let mut engine = GraphEngine::new();
        let positions = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let count = engine.add_nodes_from_positions(&positions);
        assert_eq!(count, 3);
        assert_eq!(engine.node_count(), 3);
    }

    #[test]
    fn test_add_edge() {
        let mut engine = GraphEngine::new();
        let a = engine.add_node(0.0, 0.0);
        let b = engine.add_node(1.0, 1.0);

        let edge = engine.add_edge(a, b, 1.0);
        assert!(edge.is_some());
        assert_eq!(engine.edge_count(), 1);
    }

    #[test]
    fn test_get_neighbors() {
        let mut engine = GraphEngine::new();
        let a = engine.add_node(0.0, 0.0);
        let b = engine.add_node(1.0, 0.0);
        let c = engine.add_node(0.0, 1.0);

        engine.add_edge(a, b, 1.0);
        engine.add_edge(a, c, 1.0);

        let neighbors = engine.get_neighbors(a);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&b.0));
        assert!(neighbors.contains(&c.0));
    }

    #[test]
    fn test_pin_unpin() {
        let mut engine = GraphEngine::new();
        let id = engine.add_node(0.0, 0.0);

        assert!(!engine.is_node_pinned(id));

        engine.pin_node(id);
        assert!(engine.is_node_pinned(id));

        engine.unpin_node(id);
        assert!(!engine.is_node_pinned(id));
    }

    #[test]
    fn test_bounds() {
        let mut engine = GraphEngine::new();
        engine.add_node(-10.0, -5.0);
        engine.add_node(10.0, 5.0);

        let bounds = engine.get_bounds();
        assert_eq!(bounds, Some((-10.0, -5.0, 10.0, 5.0)));
    }

    #[test]
    fn test_clear() {
        let mut engine = GraphEngine::new();
        engine.add_node(0.0, 0.0);
        engine.add_node(1.0, 1.0);

        engine.clear();
        assert_eq!(engine.node_count(), 0);
        assert_eq!(engine.edge_count(), 0);
    }

    #[test]
    fn test_remove_node_zeroes_soa() {
        let mut engine = GraphEngine::new();
        let a = engine.add_node(10.0, 20.0);
        let _b = engine.add_node(30.0, 40.0);

        engine.remove_node(a);

        // SoA slot 0 should be zeroed
        assert_eq!(engine.positions_x()[0], 0.0);
        assert_eq!(engine.positions_y()[0], 0.0);
        assert_eq!(engine.velocities_x()[0], 0.0);
        assert_eq!(engine.velocities_y()[0], 0.0);
    }

    #[test]
    fn test_remove_node_csr_no_panic() {
        let mut engine = GraphEngine::new();
        let a = engine.add_node(0.0, 0.0);
        let b = engine.add_node(1.0, 1.0);
        let c = engine.add_node(2.0, 2.0);

        engine.add_edge(a, b, 1.0);
        engine.add_edge(b, c, 1.0);

        // Remove middle node — CSR must not panic despite index hole
        engine.remove_node(b);

        let csr = engine.get_edges_csr();
        assert!(!csr.is_empty()); // Should succeed without panic

        let inverse_csr = engine.get_inverse_edges_csr();
        assert!(!inverse_csr.is_empty());

        let degrees = engine.get_node_degrees();
        assert!(!degrees.is_empty());
    }

    #[test]
    fn test_node_bound() {
        let mut engine = GraphEngine::new();
        let a = engine.add_node(0.0, 0.0);
        let _b = engine.add_node(1.0, 1.0);
        let _c = engine.add_node(2.0, 2.0);

        assert_eq!(engine.node_bound(), 3);

        engine.remove_node(a);
        // node_count drops but node_bound stays
        assert_eq!(engine.node_count(), 2);
        assert_eq!(engine.node_bound(), 3);
    }

    #[test]
    fn test_get_bounds_skips_removed() {
        let mut engine = GraphEngine::new();
        let a = engine.add_node(-100.0, -100.0);
        let _b = engine.add_node(10.0, 10.0);
        let _c = engine.add_node(20.0, 20.0);

        // Bounds include all nodes
        let bounds = engine.get_bounds().unwrap();
        assert_eq!(bounds.0, -100.0); // min_x

        // Remove the outlier node
        engine.remove_node(a);

        // Bounds should no longer include the removed node
        let bounds = engine.get_bounds().unwrap();
        assert_eq!(bounds.0, 10.0); // min_x is now 10
    }
}
