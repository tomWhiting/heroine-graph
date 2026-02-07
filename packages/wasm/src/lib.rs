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
pub mod layout;
pub mod spatial;

use graph::{GraphEngine, NodeId};
use layout::tidy_tree::{CoordinateMode, TidyTreeConfig, TidyTreeLayout};

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

    // =========================================================================
    // Layout Algorithms
    // =========================================================================

    /// Compute a tidy tree layout using Buchheim's O(n) algorithm.
    ///
    /// Takes the tree edges as [parent0, child0, parent1, child1, ...] pairs.
    /// Returns a Float32Array of target positions [x0, y0, x1, y1, ...] with
    /// one (x, y) pair per node slot.
    ///
    /// # Arguments
    ///
    /// * `edges` - Flat array of directed parent→child edge pairs
    /// * `root_id` - The root node ID (u32::MAX means auto-detect)
    /// * `level_separation` - Spacing between tree levels (default: 80)
    /// * `sibling_separation` - Minimum separation between siblings (default: 1)
    /// * `subtree_separation` - Minimum separation between subtrees (default: 2)
    /// * `radial` - If true, use radial coordinates; if false, linear top-down
    #[wasm_bindgen(js_name = computeTreeLayout)]
    pub fn compute_tree_layout(
        &self,
        edges: &[u32],
        root_id: u32,
        level_separation: f32,
        sibling_separation: f32,
        subtree_separation: f32,
        radial: bool,
    ) -> Float32Array {
        let config = TidyTreeConfig {
            level_separation,
            sibling_separation,
            subtree_separation,
            coordinate_mode: if radial {
                CoordinateMode::Radial
            } else {
                CoordinateMode::Linear
            },
        };

        let layout = TidyTreeLayout::new(config);
        let node_count = self.engine.node_bound() as usize;
        let root = if root_id == u32::MAX {
            None
        } else {
            Some(root_id)
        };

        let result = layout.compute(node_count, edges, root);

        // Interleave x and y into [x0, y0, x1, y1, ...]
        let mut positions = Vec::with_capacity(node_count * 2);
        for i in 0..node_count {
            positions.push(result.positions_x[i]);
            positions.push(result.positions_y[i]);
        }

        Float32Array::from(&positions[..])
    }

    /// Compute a tidy tree layout using the graph's own edges.
    ///
    /// This uses the edges already stored in the graph engine rather than
    /// requiring external edge data. Returns a Float32Array of target
    /// positions [x0, y0, x1, y1, ...].
    ///
    /// # Arguments
    ///
    /// * `root_id` - The root node ID (u32::MAX means auto-detect)
    /// * `level_separation` - Spacing between tree levels
    /// * `sibling_separation` - Minimum separation between siblings
    /// * `subtree_separation` - Minimum separation between subtrees
    /// * `radial` - If true, use radial coordinates; if false, linear top-down
    #[wasm_bindgen(js_name = computeTreeLayoutFromGraph)]
    pub fn compute_tree_layout_from_graph(
        &self,
        root_id: u32,
        level_separation: f32,
        sibling_separation: f32,
        subtree_separation: f32,
        radial: bool,
    ) -> Float32Array {
        // Extract edges from the graph engine's CSR format
        let csr = self.engine.get_edges_csr();
        let node_bound = self.engine.node_bound() as usize;

        if csr.len() <= node_bound + 1 {
            // No edges — return sentinel-filled positions
            let sentinel = 3.402_823e+38_f32;
            let positions = vec![sentinel; node_bound * 2];
            return Float32Array::from(&positions[..]);
        }

        let offsets = &csr[..node_bound + 1];
        let targets = &csr[node_bound + 1..];

        // Convert CSR to flat edge pairs [src0, tgt0, src1, tgt1, ...]
        let mut edges = Vec::with_capacity(targets.len() * 2);
        for src in 0..node_bound {
            let start = offsets[src] as usize;
            let end = offsets[src + 1] as usize;
            for &tgt in &targets[start..end.min(targets.len())] {
                edges.push(src as u32);
                edges.push(tgt);
            }
        }

        self.compute_tree_layout(
            &edges,
            root_id,
            level_separation,
            sibling_separation,
            subtree_separation,
            radial,
        )
    }
}

impl Default for HeroineGraphWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test the full pipeline: engine → CSR → tidy tree layout
    /// This simulates exactly what computeTreeLayoutFromGraph does,
    /// but without wasm_bindgen JS types.
    #[test]
    fn test_engine_csr_to_tidy_tree() {
        let mut engine = GraphEngine::new();

        // Build a simple tree: 0→1, 0→2, 1→3, 1→4
        let n0 = engine.add_node(0.0, 0.0);
        let n1 = engine.add_node(100.0, 0.0);
        let n2 = engine.add_node(-100.0, 0.0);
        let n3 = engine.add_node(200.0, 0.0);
        let n4 = engine.add_node(300.0, 0.0);

        engine.add_edge(n0, n1, 1.0);
        engine.add_edge(n0, n2, 1.0);
        engine.add_edge(n1, n3, 1.0);
        engine.add_edge(n1, n4, 1.0);

        let node_bound = engine.node_bound() as usize;
        assert_eq!(node_bound, 5);

        // Extract CSR
        let csr = engine.get_edges_csr();
        println!("CSR length: {}, node_bound: {}", csr.len(), node_bound);
        println!("CSR offsets: {:?}", &csr[..node_bound + 1]);
        println!("CSR targets: {:?}", &csr[node_bound + 1..]);

        assert!(csr.len() > node_bound + 1, "CSR should have edges");

        let offsets = &csr[..node_bound + 1];
        let targets = &csr[node_bound + 1..];

        // Convert CSR to edge pairs (same logic as compute_tree_layout_from_graph)
        let mut edges = Vec::new();
        for src in 0..node_bound {
            let start = offsets[src] as usize;
            let end = offsets[src + 1] as usize;
            for &tgt in &targets[start..end.min(targets.len())] {
                edges.push(src as u32);
                edges.push(tgt);
            }
        }

        println!("Edge pairs: {:?}", edges);
        assert!(!edges.is_empty(), "Should have edge pairs");

        // Run tidy tree layout
        let config = TidyTreeConfig {
            level_separation: 80.0,
            sibling_separation: 1.0,
            subtree_separation: 2.0,
            coordinate_mode: CoordinateMode::Radial,
        };
        let layout = TidyTreeLayout::new(config);
        let result = layout.compute(node_bound, &edges, None);

        println!("Layout node_count: {}", result.node_count);
        println!("positions_x: {:?}", result.positions_x);
        println!("positions_y: {:?}", result.positions_y);

        assert_eq!(result.node_count, 5, "All 5 nodes should be laid out");

        // Check that non-sentinel positions exist
        let sentinel = 3.402_823e+38_f32;
        let non_sentinel: Vec<_> = result.positions_x.iter()
            .enumerate()
            .filter(|&(_, x)| *x < sentinel)
            .collect();
        assert_eq!(non_sentinel.len(), 5, "All 5 nodes should have non-sentinel x positions");

        // Check that positions span a reasonable range (not all zero)
        let min_x = result.positions_x.iter().copied().fold(f32::INFINITY, f32::min);
        let max_x = result.positions_x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_x - min_x;
        println!("X range: {min_x} to {max_x} (range={range})");
        assert!(range > 1.0, "Positions should span a non-trivial range, got {range}");
    }

    /// Test with a larger graph mimicking mission control's hierarchical generator.
    /// 100 nodes with branch factor ~3, verifying all nodes get laid out.
    #[test]
    fn test_large_hierarchical_tree() {
        let mut engine = GraphEngine::new();

        // Add 100 nodes
        for i in 0..100u32 {
            engine.add_node(i as f32 * 10.0, 0.0);
        }

        // Build tree: root=0, each node gets ~3 children
        let mut next_child = 1u32;
        let mut queue = vec![0u32];
        while next_child < 100 {
            let mut next_queue = Vec::new();
            for &parent in &queue {
                let children = 3.min(100 - next_child);
                for _ in 0..children {
                    if next_child >= 100 { break; }
                    engine.add_edge(NodeId(parent), NodeId(next_child), 1.0);
                    next_queue.push(next_child);
                    next_child += 1;
                }
            }
            queue = next_queue;
        }

        let node_bound = engine.node_bound() as usize;
        let edge_count = engine.edge_count();
        println!("Large tree: node_bound={}, edge_count={}", node_bound, edge_count);
        assert_eq!(node_bound, 100);
        assert_eq!(edge_count, 99); // tree has n-1 edges

        // Extract CSR
        let csr = engine.get_edges_csr();
        let offsets = &csr[..node_bound + 1];
        let targets = &csr[node_bound + 1..];
        println!("CSR: offsets.len={}, targets.len={}", offsets.len(), targets.len());
        println!("First 10 offsets: {:?}", &offsets[..10]);
        println!("First 10 targets: {:?}", &targets[..10.min(targets.len())]);

        // Convert to edge pairs
        let mut edges = Vec::new();
        for src in 0..node_bound {
            let start = offsets[src] as usize;
            let end = offsets[src + 1] as usize;
            for &tgt in &targets[start..end.min(targets.len())] {
                edges.push(src as u32);
                edges.push(tgt);
            }
        }
        println!("Edge pairs: {}", edges.len() / 2);
        assert_eq!(edges.len() / 2, 99, "Should extract 99 edge pairs from CSR");

        // Run tidy tree
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            level_separation: 80.0,
            sibling_separation: 1.0,
            subtree_separation: 2.0,
            coordinate_mode: CoordinateMode::Radial,
        });
        let result = layout.compute(node_bound, &edges, None);
        println!("Layout laid out {} of {} nodes", result.node_count, node_bound);

        assert_eq!(result.node_count, 100, "All 100 nodes should be laid out");
    }

    /// Test using bulk APIs (add_nodes_from_positions + add_edges_from_pairs)
    /// to exactly replicate the mission control flow.
    #[test]
    fn test_bulk_api_csr_pipeline() {
        let mut engine = GraphEngine::new();

        let node_count = 1000;

        // Bulk add nodes (like populateWasmEngine does)
        let mut positions = Vec::with_capacity(node_count * 2);
        for i in 0..node_count {
            positions.push(i as f32 * 10.0);
            positions.push(0.0);
        }
        let added = engine.add_nodes_from_positions(&positions);
        assert_eq!(added, node_count as u32);

        // Build tree edges (like the hierarchical generator)
        // Asymmetric branching: some nodes get 0 children, some get many
        let mut edge_pairs: Vec<u32> = Vec::new();
        let mut next_child = 1u32;
        let mut queue = vec![0u32];

        while next_child < node_count as u32 {
            let mut next_queue = Vec::new();
            for &parent in &queue {
                // Vary children count: 0-6
                let children_count = match parent % 5 {
                    0 => 5, // hubs
                    1 => 3,
                    2 => 2,
                    3 => 1,
                    _ => 0, // leaves
                };
                for _ in 0..children_count {
                    if next_child >= node_count as u32 { break; }
                    edge_pairs.push(parent);
                    edge_pairs.push(next_child);
                    next_queue.push(next_child);
                    next_child += 1;
                }
            }
            if next_queue.is_empty() { break; }
            queue = next_queue;
        }

        let expected_edges = edge_pairs.len() / 2;
        println!("Bulk test: {} nodes, {} edges to add", node_count, expected_edges);

        // Bulk add edges (like populateWasmEngine does)
        let added_edges = engine.add_edges_from_pairs(&edge_pairs);
        println!("Added {} edges (expected {})", added_edges, expected_edges);
        assert_eq!(added_edges as usize, expected_edges);

        // Verify engine state
        let node_bound = engine.node_bound() as usize;
        let edge_count = engine.edge_count() as usize;
        println!("Engine: node_bound={}, node_count={}, edge_count={}",
            node_bound, engine.node_count(), edge_count);

        // Extract CSR (same as compute_tree_layout_from_graph)
        let csr = engine.get_edges_csr();
        println!("CSR length: {}, expected offsets: {}, expected: offsets + targets = {}",
            csr.len(), node_bound + 1, node_bound + 1 + edge_count);

        if csr.len() <= node_bound + 1 {
            panic!("CSR has no edges! csr.len={}, node_bound+1={}", csr.len(), node_bound + 1);
        }

        let offsets = &csr[..node_bound + 1];
        let targets = &csr[node_bound + 1..];
        println!("CSR offsets[0..10]: {:?}", &offsets[..10.min(offsets.len())]);
        println!("CSR targets.len={}, first 10: {:?}", targets.len(), &targets[..10.min(targets.len())]);

        // Verify total edges in CSR matches
        let total_csr_edges = offsets[node_bound] as usize;
        println!("Total edges in CSR (from last offset): {}", total_csr_edges);
        assert_eq!(total_csr_edges, edge_count, "CSR total edges should match engine edge count");

        // Convert CSR to edge pairs
        let mut edges = Vec::new();
        for src in 0..node_bound {
            let start = offsets[src] as usize;
            let end = offsets[src + 1] as usize;
            for &tgt in &targets[start..end.min(targets.len())] {
                edges.push(src as u32);
                edges.push(tgt);
            }
        }
        println!("Extracted {} edge pairs from CSR", edges.len() / 2);
        assert_eq!(edges.len() / 2, edge_count, "CSR edge pairs should match edge count");

        // Run tidy tree layout
        let layout = TidyTreeLayout::new(TidyTreeConfig {
            level_separation: 80.0,
            sibling_separation: 1.0,
            subtree_separation: 2.0,
            coordinate_mode: CoordinateMode::Radial,
        });
        let result = layout.compute(node_bound, &edges, None);
        println!("Layout: {} nodes laid out of {} total", result.node_count, node_bound);

        // Check how many non-sentinel positions
        let sentinel = 3.402_823e+38_f32;
        let non_sentinel = result.positions_x.iter().filter(|&&x| x < sentinel).count();
        println!("Non-sentinel positions: {}", non_sentinel);

        // All connected nodes should be laid out
        assert!(result.node_count > node_count / 2,
            "Expected at least half the nodes laid out, got {}", result.node_count);
    }

    /// Test that clear() + reload works correctly.
    /// This replicates the mission control bug: first load 1000 nodes,
    /// then clear and reload 10000 nodes. Without the fix, next_node_id
    /// wouldn't reset, causing NodeId mismatch and edge addition failure.
    #[test]
    fn test_clear_and_reload_preserves_edges() {
        let mut engine = GraphEngine::new();

        // First load: 100 nodes with some edges
        let mut positions1 = Vec::with_capacity(200);
        for i in 0..100 {
            positions1.push(i as f32);
            positions1.push(0.0);
        }
        engine.add_nodes_from_positions(&positions1);

        // Add tree edges for first load
        let mut edges1 = Vec::new();
        for i in 1..100u32 {
            edges1.push((i - 1) / 3); // parent
            edges1.push(i);            // child
        }
        let added1 = engine.add_edges_from_pairs(&edges1);
        println!("First load: {} nodes, {} edges added", engine.node_count(), added1);
        assert_eq!(added1, 99);
        assert_eq!(engine.edge_count(), 99);

        // Clear and reload with new data
        engine.clear();
        assert_eq!(engine.node_count(), 0);
        assert_eq!(engine.edge_count(), 0);

        // Second load: 500 nodes
        let mut positions2 = Vec::with_capacity(1000);
        for i in 0..500 {
            positions2.push(i as f32);
            positions2.push(0.0);
        }
        engine.add_nodes_from_positions(&positions2);
        assert_eq!(engine.node_count(), 500);

        // Add tree edges for second load (same pattern)
        let mut edges2 = Vec::new();
        for i in 1..500u32 {
            edges2.push((i - 1) / 4); // parent
            edges2.push(i);            // child
        }
        let added2 = engine.add_edges_from_pairs(&edges2);
        println!("Second load: {} nodes, {} edges added (expected 499)", engine.node_count(), added2);

        // THE KEY ASSERTION: all edges should be added after clear+reload
        assert_eq!(added2, 499, "All edges should be added after clear(). Got {}. \
            This likely means clear() didn't reset next_node_id, causing NodeId mismatch.", added2);
        assert_eq!(engine.edge_count(), 499);

        // Verify CSR extraction works
        let csr = engine.get_edges_csr();
        let node_bound = engine.node_bound() as usize;
        assert_eq!(node_bound, 500);
        assert!(csr.len() > node_bound + 1, "CSR should have edges");

        let offsets = &csr[..node_bound + 1];
        let total = offsets[node_bound] as usize;
        assert_eq!(total, 499, "CSR should contain all 499 edges");

        // Verify tidy tree works with reloaded data
        let mut edges_flat = Vec::new();
        let targets = &csr[node_bound + 1..];
        for src in 0..node_bound {
            let start = offsets[src] as usize;
            let end = offsets[src + 1] as usize;
            for &tgt in &targets[start..end.min(targets.len())] {
                edges_flat.push(src as u32);
                edges_flat.push(tgt);
            }
        }

        let layout = TidyTreeLayout::new(TidyTreeConfig {
            level_separation: 80.0,
            sibling_separation: 1.0,
            subtree_separation: 2.0,
            coordinate_mode: CoordinateMode::Radial,
        });
        let result = layout.compute(node_bound, &edges_flat, None);
        println!("After reload: {} nodes laid out of {}", result.node_count, node_bound);
        assert_eq!(result.node_count, 500, "All 500 nodes should be laid out after clear+reload");
    }
}
