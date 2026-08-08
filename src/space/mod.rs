//! Where plane geometry sits in space.
//!
//! [`geom2d`](crate::geom2d) is deliberately two-dimensional: an
//! intersection, an offset or a containment test is a plane problem, and
//! answering it in three coordinates would mean carrying a component that is
//! either zero or a source of error. This module is the other half of that
//! bargain — the frame that says *which* plane, and the map in and out of it.
//!
//! [`Plane`] is the whole of it, and [`PlanarCurve`] is the pairing every
//! consumer actually wants: a shape plus where it lives. A drawing's arcs,
//! circles, ellipses and polylines are stored exactly this way, and so is a
//! planar B-rep face.

pub mod curve;
pub mod plane;
pub mod polygon;
pub mod vec;

#[cfg(feature = "geom2d")]
pub mod planar;

pub use plane::Plane;
pub use vec::Vec3;

#[cfg(feature = "geom2d")]
pub use planar::PlanarCurve;
