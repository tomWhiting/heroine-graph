//! Graph data structures and operations.
//!
//! This module provides the core graph structure using petgraph's StableGraph
//! for stable node/edge indices, with Structure of Arrays (SoA) layout for
//! positions and velocities to enable SIMD operations and cache-friendly access.

mod edge;
mod engine;
mod node;

pub use edge::EdgeId;
pub use engine::GraphEngine;
pub use node::NodeId;
