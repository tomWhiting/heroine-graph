//! Heroine Graph - WASM Module
//!
//! This module provides the core graph data structures and algorithms
//! for the Heroine Graph visualization library. It is compiled to WebAssembly
//! and exposes a JavaScript-friendly API via wasm-bindgen.
//!
//! # Architecture
//!
//! - `graph`: Graph data structure using petgraph's StableGraph
//! - `spatial`: R-tree spatial indexing for O(log n) hit testing
//! - `layout`: Force calculation utilities (CPU-side, for validation)
//! - `algorithms`: Graph algorithms (clustering, traversal, etc.)

use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

pub mod graph;
pub mod spatial;

use graph::{GraphEngine, NodeId};

/// Initialize the WASM module.
#[wasm_bindgen(start)]
pub fn init() {
    // Placeholder for panic hook initialization if needed
}

/// Main entry point for the graph engine.
///
/// This struct wraps the internal GraphEngine and provides the public API
/// exposed to JavaScript.
#[wasm_bindgen]
pub struct HeroineGraphWasm {
    engine: GraphEngine,
}

#[wasm_bindgen]
impl HeroineGraphWasm {
    /// Create a new empty graph engine.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            engine: GraphEngine::new(),
        }
    }

    /// Create a graph engine with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `node_capacity` - Expected number of nodes
    /// * `edge_capacity` - Expected number of edges
    #[wasm_bindgen(js_name = withCapacity)]
    pub fn with_capacity(node_capacity: usize, edge_capacity: usize) -> Self {
        Self {
            engine: GraphEngine::with_capacity(node_capacity, edge_capacity),
        }
    }

    // =========================================================================
    // Node Operations
    // =========================================================================

    /// Add a node at the specified position.
    ///
    /// Returns the stable node ID.
    #[wasm_bindgen(js_name = addNode)]
    pub fn add_node(&mut self, x: f32, y: f32) -> u32 {
        self.engine.add_node(x, y).0
    }

    /// Add multiple nodes from a Float32Array of positions.
    ///
    /// The positions array should be [x0, y0, x1, y1, ...].
    /// Returns the number of nodes added.
    #[wasm_bindgen(js_name = addNodesFromPositions)]
    pub fn add_nodes_from_positions(&mut self, positions: &[f32]) -> u32 {
        self.engine.add_nodes_from_positions(positions)
    }

    /// Remove a node by ID.
    ///
    /// Returns true if the node existed and was removed.
    #[wasm_bindgen(js_name = removeNode)]
    pub fn remove_node(&mut self, node_id: u32) -> bool {
        self.engine.remove_node(NodeId(node_id))
    }

    /// Get the number of nodes in the graph.
    #[wasm_bindgen(js_name = nodeCount)]
    pub fn node_count(&self) -> u32 {
        self.engine.node_count()
    }

    /// Get the upper bound on node indices (max index + 1).
    /// May be larger than nodeCount if nodes have been removed.
    #[wasm_bindgen(js_name = nodeBound)]
    pub fn node_bound(&self) -> u32 {
        self.engine.node_bound()
    }

    /// Get a node's X position.
    #[wasm_bindgen(js_name = getNodeX)]
    pub fn get_node_x(&self, node_id: u32) -> Option<f32> {
        self.engine.get_node_position(NodeId(node_id)).map(|(x, _)| x)
    }

    /// Get a node's Y position.
    #[wasm_bindgen(js_name = getNodeY)]
    pub fn get_node_y(&self, node_id: u32) -> Option<f32> {
        self.engine.get_node_position(NodeId(node_id)).map(|(_, y)| y)
    }

    /// Set a node's position.
    #[wasm_bindgen(js_name = setNodePosition)]
    pub fn set_node_position(&mut self, node_id: u32, x: f32, y: f32) {
        self.engine.set_node_position(NodeId(node_id), x, y);
    }

    /// Pin a node (exclude from simulation).
    #[wasm_bindgen(js_name = pinNode)]
    pub fn pin_node(&mut self, node_id: u32) {
        self.engine.pin_node(NodeId(node_id));
    }

    /// Unpin a node (include in simulation).
    #[wasm_bindgen(js_name = unpinNode)]
    pub fn unpin_node(&mut self, node_id: u32) {
        self.engine.unpin_node(NodeId(node_id));
    }

    /// Check if a node is pinned.
    #[wasm_bindgen(js_name = isNodePinned)]
    pub fn is_node_pinned(&self, node_id: u32) -> bool {
        self.engine.is_node_pinned(NodeId(node_id))
    }

    // =========================================================================
    // Edge Operations
    // =========================================================================

    /// Add an edge between two nodes.
    ///
    /// Returns the edge ID, or None if source/target don't exist.
    #[wasm_bindgen(js_name = addEdge)]
    pub fn add_edge(&mut self, source: u32, target: u32, weight: f32) -> Option<u32> {
        self.engine
            .add_edge(NodeId(source), NodeId(target), weight)
            .map(|id| id.0)
    }

    /// Add edges from a Uint32Array of pairs.
    ///
    /// The edges array should be [src0, tgt0, src1, tgt1, ...].
    /// All edges get weight 1.0.
    /// Returns the number of edges added.
    #[wasm_bindgen(js_name = addEdgesFromPairs)]
    pub fn add_edges_from_pairs(&mut self, edges: &[u32]) -> u32 {
        self.engine.add_edges_from_pairs(edges)
    }

    /// Remove an edge by ID.
    ///
    /// Returns true if the edge existed and was removed.
    #[wasm_bindgen(js_name = removeEdge)]
    pub fn remove_edge(&mut self, edge_id: u32) -> bool {
        self.engine.remove_edge(graph::EdgeId(edge_id))
    }

    /// Get the number of edges in the graph.
    #[wasm_bindgen(js_name = edgeCount)]
    pub fn edge_count(&self) -> u32 {
        self.engine.edge_count()
    }

    /// Get neighbors of a node.
    ///
    /// Returns a Uint32Array of neighbor node IDs.
    #[wasm_bindgen(js_name = getNeighbors)]
    pub fn get_neighbors(&self, node_id: u32) -> Vec<u32> {
        self.engine.get_neighbors(NodeId(node_id))
    }

    // =========================================================================
    // Position Buffer Access (Zero-Copy)
    // =========================================================================

    /// Get a zero-copy view of X positions.
    ///
    /// # Safety
    ///
    /// The returned view is invalidated if any Rust allocation occurs.
    /// Use immediately for GPU upload, do not store.
    #[wasm_bindgen(js_name = getPositionsXView)]
    pub fn get_positions_x_view(&self) -> Float32Array {
        unsafe { Float32Array::view(self.engine.positions_x()) }
    }

    /// Get a zero-copy view of Y positions.
    ///
    /// # Safety
    ///
    /// The returned view is invalidated if any Rust allocation occurs.
    /// Use immediately for GPU upload, do not store.
    #[wasm_bindgen(js_name = getPositionsYView)]
    pub fn get_positions_y_view(&self) -> Float32Array {
        unsafe { Float32Array::view(self.engine.positions_y()) }
    }

    /// Get a zero-copy view of X velocities.
    #[wasm_bindgen(js_name = getVelocitiesXView)]
    pub fn get_velocities_x_view(&self) -> Float32Array {
        unsafe { Float32Array::view(self.engine.velocities_x()) }
    }

    /// Get a zero-copy view of Y velocities.
    #[wasm_bindgen(js_name = getVelocitiesYView)]
    pub fn get_velocities_y_view(&self) -> Float32Array {
        unsafe { Float32Array::view(self.engine.velocities_y()) }
    }

    /// Get a pointer to the X positions buffer.
    ///
    /// Used for creating views after WASM memory growth.
    #[wasm_bindgen(js_name = positionsXPtr)]
    pub fn positions_x_ptr(&self) -> *const f32 {
        self.engine.positions_x().as_ptr()
    }

    /// Get the length of the positions buffer.
    #[wasm_bindgen(js_name = positionsLen)]
    pub fn positions_len(&self) -> usize {
        self.engine.positions_x().len()
    }

    // =========================================================================
    // Spatial Queries
    // =========================================================================

    /// Find the nearest node to a point.
    ///
    /// Returns the node ID, or None if the graph is empty.
    #[wasm_bindgen(js_name = findNearestNode)]
    pub fn find_nearest_node(&self, x: f32, y: f32) -> Option<u32> {
        self.engine.find_nearest_node(x, y).map(|id| id.0)
    }

    /// Find the nearest node within a maximum distance.
    ///
    /// Returns the node ID, or None if no node is within the distance.
    #[wasm_bindgen(js_name = findNearestNodeWithin)]
    pub fn find_nearest_node_within(&self, x: f32, y: f32, max_distance: f32) -> Option<u32> {
        self.engine
            .find_nearest_node_within(x, y, max_distance)
            .map(|id| id.0)
    }

    /// Find all nodes within a rectangular region.
    ///
    /// Returns a Uint32Array of node IDs.
    #[wasm_bindgen(js_name = findNodesInRect)]
    pub fn find_nodes_in_rect(&self, min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Vec<u32> {
        self.engine.find_nodes_in_rect(min_x, min_y, max_x, max_y)
    }

    /// Rebuild the spatial index after position changes.
    ///
    /// Call this after bulk position updates for accurate spatial queries.
    #[wasm_bindgen(js_name = rebuildSpatialIndex)]
    pub fn rebuild_spatial_index(&mut self) {
        self.engine.rebuild_spatial_index();
    }

    // =========================================================================
    // Graph Utilities
    // =========================================================================

    /// Get the bounding box of all nodes.
    ///
    /// Returns [min_x, min_y, max_x, max_y], or None if graph is empty.
    #[wasm_bindgen(js_name = getBounds)]
    pub fn get_bounds(&self) -> Option<Vec<f32>> {
        self.engine.get_bounds().map(|(min_x, min_y, max_x, max_y)| {
            vec![min_x, min_y, max_x, max_y]
        })
    }

    /// Clear all nodes and edges.
    pub fn clear(&mut self) {
        self.engine.clear();
    }

    /// Get the edge list in CSR format for GPU upload.
    ///
    /// Returns [offsets..., targets...] where offsets has node_count + 1 elements.
    #[wasm_bindgen(js_name = getEdgesCsr)]
    pub fn get_edges_csr(&self) -> Vec<u32> {
        self.engine.get_edges_csr()
    }

    /// Get the inverse edge list in CSR format (incoming edges).
    ///
    /// For each node, lists the source nodes of incoming edges (parents).
    /// Returns [offsets..., sources...] where offsets has node_count + 1 elements.
    /// Useful for hierarchical algorithms that need parent relationships.
    #[wasm_bindgen(js_name = getInverseEdgesCsr)]
    pub fn get_inverse_edges_csr(&self) -> Vec<u32> {
        self.engine.get_inverse_edges_csr()
    }

    /// Get node degrees as [out_deg_0, in_deg_0, out_deg_1, in_deg_1, ...].
    ///
    /// Returns a flat array with 2 * node_count elements.
    /// Useful for degree-weighted force calculations.
    #[wasm_bindgen(js_name = getNodeDegrees)]
    pub fn get_node_degrees(&self) -> Vec<u32> {
        self.engine.get_node_degrees()
    }
}

impl Default for HeroineGraphWasm {
    fn default() -> Self {
        Self::new()
    }
}
