//! Node type and related structures.
//!
//! Nodes are the vertices in the graph. Each node has:
//! - An identifier that is its slot index (stable while the node exists)
//! - Position (x, y) in graph space
//! - Velocity (vx, vy) for force simulation
//! - Pinned state (excluded from simulation when true)

use std::fmt;

/// Node identifier — the node's slot index.
///
/// This ID equals the node's petgraph/CSR/SoA/layout slot index and remains
/// valid while the node exists, even as other nodes are removed. Slots freed
/// by removal are reused by later additions (StableGraph semantics).
/// It wraps a u32 for efficient storage and WebAssembly interop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Create a new NodeId from a raw u32.
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

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

impl From<u32> for NodeId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<NodeId> for u32 {
    #[inline]
    fn from(id: NodeId) -> Self {
        id.0
    }
}

/// Node state flags packed into a single byte.
#[derive(Debug, Clone, Copy, Default)]
pub struct NodeState {
    flags: u8,
}

impl NodeState {
    const PINNED: u8 = 0b0000_0001;

    /// Create a new default node state.
    #[inline]
    pub fn new() -> Self {
        Self { flags: 0 }
    }

    /// Check if the node is pinned (excluded from simulation).
    #[inline]
    pub fn is_pinned(self) -> bool {
        self.flags & Self::PINNED != 0
    }

    /// Set the pinned state.
    #[inline]
    pub fn set_pinned(&mut self, pinned: bool) {
        if pinned {
            self.flags |= Self::PINNED;
        } else {
            self.flags &= !Self::PINNED;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id() {
        let id = NodeId::new(42);
        assert_eq!(id.raw(), 42);
        assert_eq!(id.0, 42);
        assert_eq!(format!("{}", id), "Node(42)");
    }

    #[test]
    fn test_node_id_conversion() {
        let id: NodeId = 123.into();
        let raw: u32 = id.into();
        assert_eq!(raw, 123);
    }

    #[test]
    fn test_node_state_default() {
        let state = NodeState::new();
        assert!(!state.is_pinned());
    }

    #[test]
    fn test_node_state_pinned() {
        let mut state = NodeState::new();
        state.set_pinned(true);
        assert!(state.is_pinned());

        state.set_pinned(false);
        assert!(!state.is_pinned());
    }
}
