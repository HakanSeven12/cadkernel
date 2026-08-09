//! Owned, mutable boundary representation.
//!
//! # Why this layer exists at all
//!
//! A file-format reader models topology as read-only views over a flat
//! record array, navigated by record index. That shape is right for parsing
//! and for lossless re-emission, and wrong for editing: the views borrow the
//! whole document, indices shift the moment anything is inserted, and there
//! is nowhere to put a face that a boolean has just split in two.
//!
//! So this layer owns its topology. Nodes live in arenas, keys stay stable
//! across edits, and adjacency is stored rather than rediscovered.
//!
//! # Provenance, and why it is not optional
//!
//! Every node carries the [`SourceRef`] it was lifted from, when it came
//! from a file. That single field is what keeps a save from degrading a
//! drawing:
//!
//! - a node that was never touched lowers back to its original record,
//!   byte for byte, carrying attributes and parameter-space curves that this
//!   kernel has no opinion about
//! - only a node the edit actually dirtied is re-emitted from scratch
//!
//! Without it, a boolean on one solid rewrites every solid in the file, and
//! anything the kernel does not model is lost on the way through. With it,
//! the blast radius of an edit is the edit.

pub mod arena;
pub mod geometry;
pub mod intersect;
pub mod make;
pub mod pcurve;
pub mod split;
pub mod topology;

pub use arena::{Arena, Key};
pub use geometry::{Circle3, Cone, Curve3, Cylinder, Ellipse3, Line3, Sphere, Surface, Torus};
pub use intersect::{surfaces as intersect_surfaces, Meeting};
pub use topology::{
    Body, Coedge, CoedgeKey, CurveKey, Edge, EdgeKey, Face, FaceKey, Flaw, Loop, LoopKey, Lump,
    LumpKey, Shell, ShellKey, SurfaceKey, Vertex, VertexKey,
};

/// A node's origin in the document it was lifted from.
///
/// Opaque here on purpose: this layer knows a node came from record *n* of
/// something, not what a record is. The format layer owns the mapping — see
/// the `acis` module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceRef(u32);

impl SourceRef {
    /// Wraps a format-layer record index.
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// The wrapped index, for the format layer to resolve.
    pub const fn index(&self) -> u32 {
        self.0
    }
}

/// Whether a node still matches the record it was lifted from.
///
/// Lowering consults this and nothing else: [`Provenance::Clean`] copies the
/// source record through untouched, anything else re-emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// Lifted from a source record and unchanged since.
    Clean(SourceRef),
    /// Lifted from a source record, then edited.
    Dirty(SourceRef),
    /// Built by this kernel; there is no source record to fall back on.
    Synthesized,
}

impl Provenance {
    /// The source record, if this node came from one.
    pub const fn source(&self) -> Option<SourceRef> {
        match self {
            Self::Clean(reference) | Self::Dirty(reference) => Some(*reference),
            Self::Synthesized => None,
        }
    }

    /// Whether lowering may reuse the source record verbatim.
    pub const fn is_reusable(&self) -> bool {
        matches!(self, Self::Clean(_))
    }

    /// Marks an edited node, keeping the source reference for diagnostics.
    pub fn soil(&mut self) {
        if let Self::Clean(reference) = *self {
            *self = Self::Dirty(reference);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_node_lowers_from_its_source() {
        let provenance = Provenance::Clean(SourceRef::new(7));
        assert!(provenance.is_reusable());
        assert_eq!(provenance.source().map(|r| r.index()), Some(7));
    }

    #[test]
    fn soiling_blocks_reuse_but_keeps_the_reference() {
        let mut provenance = Provenance::Clean(SourceRef::new(7));
        provenance.soil();
        assert!(!provenance.is_reusable());
        assert_eq!(provenance.source().map(|r| r.index()), Some(7));
    }

    #[test]
    fn soiling_is_idempotent() {
        let mut provenance = Provenance::Clean(SourceRef::new(7));
        provenance.soil();
        let once = provenance;
        provenance.soil();
        assert_eq!(provenance, once);
    }

    #[test]
    fn synthesized_nodes_have_nothing_to_fall_back_on() {
        let provenance = Provenance::Synthesized;
        assert!(!provenance.is_reusable());
        assert_eq!(provenance.source(), None);
    }
}
