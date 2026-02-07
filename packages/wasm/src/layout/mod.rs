//! Layout algorithms for graph visualization.
//!
//! This module provides CPU-side layout algorithms that compute target positions
//! for nodes. These positions can then be uploaded to GPU buffers and used with
//! spring-to-target force algorithms to animate the graph into the computed layout.

pub mod tidy_tree;

pub use tidy_tree::TidyTreeLayout;
