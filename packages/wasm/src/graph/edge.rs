//! Edge type and related structures.
//!
//! Edges are the connections between nodes. Each edge has:
//! - A stable unique identifier
//! - Source and target node IDs
//! - Weight for force simulation

use std::fmt;

/// Stable edge identifier.
///
/// This ID remains valid even after other edges are removed from the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub u32);

impl EdgeId {
    /// Create a new EdgeId from a raw u32.
    #[inline]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw u32 value.
    #[inline]
    pub fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Edge({})", self.0)
    }
}

impl From<u32> for EdgeId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<EdgeId> for u32 {
    #[inline]
    fn from(id: EdgeId) -> Self {
        id.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_id() {
        let id = EdgeId::new(42);
        assert_eq!(id.raw(), 42);
        assert_eq!(format!("{}", id), "Edge(42)");
    }
}
