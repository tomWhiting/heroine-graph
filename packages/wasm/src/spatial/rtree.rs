//! R-tree based spatial index using the rstar crate.
//!
//! Provides O(log n) spatial queries for:
//! - Nearest neighbor
//! - Point-in-radius
//! - Rectangle intersection

use rstar::{RTree, RTreeObject, AABB, PointDistance};

use crate::graph::NodeId;

/// A point in the spatial index with associated node ID.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodePoint {
    /// The node identifier.
    pub id: NodeId,
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

impl NodePoint {
    /// Create a new NodePoint.
    pub fn new(id: NodeId, x: f32, y: f32) -> Self {
        Self { id, x, y }
    }
}

impl RTreeObject for NodePoint {
    type Envelope = AABB<[f32; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.x, self.y])
    }
}

impl PointDistance for NodePoint {
    fn distance_2(&self, point: &[f32; 2]) -> f32 {
        let dx = self.x - point[0];
        let dy = self.y - point[1];
        dx * dx + dy * dy
    }

    fn contains_point(&self, point: &[f32; 2]) -> bool {
        (self.x - point[0]).abs() < f32::EPSILON && (self.y - point[1]).abs() < f32::EPSILON
    }
}

/// Spatial index for graph nodes.
///
/// Uses an R*-tree for efficient spatial queries.
pub struct SpatialIndex {
    tree: RTree<NodePoint>,
}

impl SpatialIndex {
    /// Create a new empty spatial index.
    pub fn new() -> Self {
        Self {
            tree: RTree::new(),
        }
    }

    /// Create a spatial index with expected capacity.
    pub fn with_capacity(_capacity: usize) -> Self {
        // RTree doesn't have with_capacity, but we can bulk load
        Self::new()
    }

    /// Insert a node into the index.
    pub fn insert(&mut self, id: NodeId, x: f32, y: f32) {
        self.tree.insert(NodePoint::new(id, x, y));
    }

    /// Remove a node from the index.
    ///
    /// Returns true if the node was found and removed.
    pub fn remove(&mut self, id: NodeId, x: f32, y: f32) -> bool {
        let point = NodePoint::new(id, x, y);
        self.tree.remove(&point).is_some()
    }

    /// Find the nearest node to a point.
    pub fn nearest(&self, x: f32, y: f32) -> Option<NodeId> {
        self.tree
            .nearest_neighbor(&[x, y])
            .map(|point| point.id)
    }

    /// Find the nearest node within a maximum distance.
    pub fn nearest_within(&self, x: f32, y: f32, max_distance: f32) -> Option<NodeId> {
        let max_distance_sq = max_distance * max_distance;
        self.tree
            .nearest_neighbor(&[x, y])
            .filter(|point| point.distance_2(&[x, y]) <= max_distance_sq)
            .map(|point| point.id)
    }

    /// Find all nodes within a rectangle.
    pub fn in_rect(&self, min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> Vec<NodeId> {
        let envelope = AABB::from_corners([min_x, min_y], [max_x, max_y]);
        self.tree
            .locate_in_envelope(&envelope)
            .map(|point| point.id)
            .collect()
    }

    /// Find all nodes within a radius of a point.
    pub fn in_radius(&self, x: f32, y: f32, radius: f32) -> Vec<NodeId> {
        let radius_sq = radius * radius;
        self.tree
            .locate_within_distance([x, y], radius_sq)
            .map(|point| point.id)
            .collect()
    }

    /// Rebuild the index from a list of (id, x, y) tuples.
    ///
    /// This is more efficient than incremental inserts for bulk updates.
    pub fn rebuild(&mut self, points: &[(NodeId, f32, f32)]) {
        let node_points: Vec<_> = points
            .iter()
            .map(|&(id, x, y)| NodePoint::new(id, x, y))
            .collect();

        self.tree = RTree::bulk_load(node_points);
    }

    /// Clear all nodes from the index.
    pub fn clear(&mut self) {
        self.tree = RTree::new();
    }

    /// Get the number of nodes in the index.
    pub fn len(&self) -> usize {
        self.tree.size()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.tree.size() == 0
    }
}

impl Default for SpatialIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_nearest() {
        let mut index = SpatialIndex::new();
        index.insert(NodeId(0), 0.0, 0.0);
        index.insert(NodeId(1), 10.0, 10.0);
        index.insert(NodeId(2), 5.0, 5.0);

        // Nearest to origin should be node 0
        assert_eq!(index.nearest(0.0, 0.0), Some(NodeId(0)));

        // Nearest to (6, 6) should be node 2
        assert_eq!(index.nearest(6.0, 6.0), Some(NodeId(2)));

        // Nearest to (11, 11) should be node 1
        assert_eq!(index.nearest(11.0, 11.0), Some(NodeId(1)));
    }

    #[test]
    fn test_nearest_within() {
        let mut index = SpatialIndex::new();
        index.insert(NodeId(0), 0.0, 0.0);
        index.insert(NodeId(1), 10.0, 10.0);

        // Within 5 of origin
        assert_eq!(index.nearest_within(0.0, 0.0, 5.0), Some(NodeId(0)));

        // Nothing within 1 of (5, 5)
        assert_eq!(index.nearest_within(5.0, 5.0, 1.0), None);

        // Node 0 is ~7.07 from (5, 5), so within 8 should find it
        assert_eq!(index.nearest_within(5.0, 5.0, 8.0), Some(NodeId(0)));
    }

    #[test]
    fn test_in_rect() {
        let mut index = SpatialIndex::new();
        index.insert(NodeId(0), 0.0, 0.0);
        index.insert(NodeId(1), 5.0, 5.0);
        index.insert(NodeId(2), 10.0, 10.0);

        let in_rect = index.in_rect(-1.0, -1.0, 6.0, 6.0);
        assert_eq!(in_rect.len(), 2);
        assert!(in_rect.contains(&NodeId(0)));
        assert!(in_rect.contains(&NodeId(1)));
    }

    #[test]
    fn test_in_radius() {
        let mut index = SpatialIndex::new();
        index.insert(NodeId(0), 0.0, 0.0);
        index.insert(NodeId(1), 3.0, 0.0);
        index.insert(NodeId(2), 10.0, 0.0);

        let in_radius = index.in_radius(0.0, 0.0, 5.0);
        assert_eq!(in_radius.len(), 2);
        assert!(in_radius.contains(&NodeId(0)));
        assert!(in_radius.contains(&NodeId(1)));
    }

    #[test]
    fn test_rebuild() {
        let mut index = SpatialIndex::new();
        index.insert(NodeId(0), 0.0, 0.0);

        let points = vec![
            (NodeId(1), 1.0, 1.0),
            (NodeId(2), 2.0, 2.0),
            (NodeId(3), 3.0, 3.0),
        ];

        index.rebuild(&points);
        assert_eq!(index.len(), 3);
        assert_eq!(index.nearest(0.0, 0.0), Some(NodeId(1)));
    }

    #[test]
    fn test_clear() {
        let mut index = SpatialIndex::new();
        index.insert(NodeId(0), 0.0, 0.0);
        index.insert(NodeId(1), 1.0, 1.0);

        index.clear();
        assert!(index.is_empty());
        assert_eq!(index.nearest(0.0, 0.0), None);
    }
}
