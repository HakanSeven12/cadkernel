//! Bridge between an ACIS document and [`crate::brep`] topology.
//!
//! Everything that knows what a SAT record is lives behind this feature, so
//! the kernel core stays format-agnostic and a caller that only wants 2D
//! curve work pulls none of it in.
//!
//! # The round trip this exists to protect
//!
//! ```text
//! SAT/SAB ──lift──► brep ──edit──► brep ──lower──► SAT/SAB
//! ```
//!
//! Lift records a [`crate::brep::SourceRef`] on every node it creates.
//! Lower consults [`crate::brep::Provenance`]: a clean node is written back
//! as its original record, so attributes, parameter-space curves and the
//! analytic surface types this kernel has not yet learned survive an edit
//! that did not touch them. Only dirty and synthesized nodes are rebuilt.
//!
//! The alternative — rebuilding every body on every save — loses whatever
//! the kernel does not model, on every save, for bodies the user never
//! opened. That is the failure this module is shaped to avoid.
//!
//! # Status
//!
//! Not implemented. The intended entry points are:
//!
//! - `lift(&SatDocument) -> Body`, exact: analytic surfaces stay analytic,
//!   splines keep their control net rather than being tessellated
//! - `lower(&Body, &mut SatDocument)`, provenance-driven as described above
//!
//! Wiring this up requires a dependency on the codec crate, which is not
//! declared yet.
