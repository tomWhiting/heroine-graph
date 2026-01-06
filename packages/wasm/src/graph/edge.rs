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

/// Edge state flags.
#[derive(Debug, Clone, Copy, Default)]
pub struct EdgeState {
    flags: u8,
}

impl EdgeState {
    const HIDDEN: u8 = 0b0000_0001;
    const SELECTED: u8 = 0b0000_0010;
    const HOVERED: u8 = 0b0000_0100;

    /// Create a new default edge state.
    #[inline]
    pub fn new() -> Self {
        Self { flags: 0 }
    }

    /// Check if the edge is hidden.
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

    /// Check if the edge is selected.
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

    /// Check if the edge is hovered.
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
    fn test_edge_id() {
        let id = EdgeId::new(42);
        assert_eq!(id.raw(), 42);
        assert_eq!(format!("{}", id), "Edge(42)");
    }

    #[test]
    fn test_edge_state() {
        let mut state = EdgeState::new();
        assert!(!state.is_hidden());
        assert!(!state.is_selected());

        state.set_selected(true);
        assert!(state.is_selected());
        assert!(!state.is_hidden());
    }
}
