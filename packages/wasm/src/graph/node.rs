//! Node type and related structures.
//!
//! Nodes are the vertices in the graph. Each node has:
//! - A stable unique identifier (survives graph mutations)
//! - Position (x, y) in graph space
//! - Velocity (vx, vy) for force simulation
//! - Pinned state (excluded from simulation when true)

use std::fmt;

/// Stable node identifier.
///
/// This ID remains valid even after other nodes are removed from the graph.
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
    const HIDDEN: u8 = 0b0000_0010;
    const SELECTED: u8 = 0b0000_0100;
    const HOVERED: u8 = 0b0000_1000;

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

    /// Check if the node is hidden.
    #[inline]
    pub fn is_hidden(self) -> bool {
        self.flags & Self::HIDDEN != 0
    }

    /// Set the hidden state.
    #[inline]
    pub fn set_hidden(&mut self, hidden: bool) {
        if hidden {
            self.flags |= Self::HIDDEN;
        } else {
            self.flags &= !Self::HIDDEN;
        }
    }

    /// Check if the node is selected.
    #[inline]
    pub fn is_selected(self) -> bool {
        self.flags & Self::SELECTED != 0
    }

    /// Set the selected state.
    #[inline]
    pub fn set_selected(&mut self, selected: bool) {
        if selected {
            self.flags |= Self::SELECTED;
        } else {
            self.flags &= !Self::SELECTED;
        }
    }

    /// Check if the node is hovered.
    #[inline]
    pub fn is_hovered(self) -> bool {
        self.flags & Self::HOVERED != 0
    }

    /// Set the hovered state.
    #[inline]
    pub fn set_hovered(&mut self, hovered: bool) {
        if hovered {
            self.flags |= Self::HOVERED;
        } else {
            self.flags &= !Self::HOVERED;
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
        assert!(!state.is_hidden());
        assert!(!state.is_selected());
        assert!(!state.is_hovered());
    }

    #[test]
    fn test_node_state_pinned() {
        let mut state = NodeState::new();
        state.set_pinned(true);
        assert!(state.is_pinned());
        assert!(!state.is_hidden());

        state.set_pinned(false);
        assert!(!state.is_pinned());
    }

    #[test]
    fn test_node_state_all_flags() {
        let mut state = NodeState::new();
        state.set_pinned(true);
        state.set_hidden(true);
        state.set_selected(true);
        state.set_hovered(true);

        assert!(state.is_pinned());
        assert!(state.is_hidden());
        assert!(state.is_selected());
        assert!(state.is_hovered());

        state.set_selected(false);
        assert!(state.is_pinned());
        assert!(state.is_hidden());
        assert!(!state.is_selected());
        assert!(state.is_hovered());
    }
}
