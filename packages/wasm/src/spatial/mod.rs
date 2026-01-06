//! Spatial indexing for O(log n) hit testing.
//!
//! This module provides an R-tree based spatial index for efficient
//! nearest-neighbor and range queries on graph nodes.

mod rtree;

pub use rtree::SpatialIndex;
