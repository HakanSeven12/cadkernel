//! A plane curve together with the plane it lives on.
//!
//! Almost every curve a drawing stores is this: a shape defined in two
//! coordinates plus a frame saying where in space those two coordinates
//! point. Keeping the pair together is what lets the 2D layer stay genuinely
//! two-dimensional — intersection, offset, containment and trimming all run
//! in the plane's own coordinates, and only the results come back out into
//! space.
//!
//! It is also what a B-rep face needs. A face is a surface plus loops in its
//! `(u, v)` space; for a planar face that is precisely a [`Plane`] and a set
//! of [`Curve`]s on it.

use super::plane::Plane;
use crate::geom2d::{Curve, Extent};

/// A curve in space, expressed in the coordinates of the plane it lies on.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanarCurve {
    /// The frame the curve's coordinates are read in.
    pub plane: Plane,
    /// The shape, in that frame.
    pub curve: Curve,
}

impl PlanarCurve {
    /// A curve on a plane.
    pub const fn new(plane: Plane, curve: Curve) -> Self {
        Self { plane, curve }
    }

    /// A curve on the world XY plane, which is where most drawing geometry
    /// sits.
    pub const fn flat(curve: Curve) -> Self {
        Self::new(Plane::XY, curve)
    }

    /// The point at parameter `t`, in world coordinates.
    ///
    /// `t` runs `0..=1` across the curve, exactly as it does in the plane;
    /// the frame changes where the point is, never how it is parameterised.
    pub fn point_at(&self, t: f64) -> [f64; 3] {
        self.lower(self.curve.point_at(t))
    }

    /// The parameter at `point`, the inverse of
    /// [`point_at`](Self::point_at).
    ///
    /// A point off the plane is projected onto it first, so a cursor
    /// position or an intersection can be handed over as it is. `None` only
    /// when the plane is degenerate.
    pub fn parameter_at(&self, point: [f64; 3]) -> Option<f64> {
        Some(self.curve.parameter_at(self.lift(point)?))
    }

    /// Samples the curve into a polyline of world-space points.
    ///
    /// `segments_per_radian` sets how finely curved parts are cut, as in
    /// [`Curve::tessellate`].
    pub fn tessellate(&self, segments_per_radian: f64) -> Vec<[f64; 3]> {
        let flat = self.curve.tessellate(segments_per_radian);
        flat.into_iter().map(|uv| self.lower(uv)).collect()
    }

    /// The plane's unit normal, or `None` if its axes do not span one.
    pub fn normal(&self) -> Option<[f64; 3]> {
        self.plane.normal()
    }

    /// How far the curve's parameter runs. Delegates to the shape: a frame
    /// does not bound anything.
    pub fn extent(&self) -> Extent {
        self.curve.extent()
    }

    /// Whether the curve returns to where it started.
    pub fn is_closed(&self) -> bool {
        self.curve.is_closed()
    }

    /// Plane coordinates out into the world.
    ///
    /// The XY-aligned shortcut is the render path's inner loop, and covers
    /// every planar entity stored with the default +Z extrusion whatever its
    /// elevation. Correctness does not depend on the branch — the general arm
    /// produces the same points.
    fn lower(&self, uv: [f64; 2]) -> [f64; 3] {
        let origin = self.plane.origin;
        if self.plane.is_xy_aligned() {
            return [origin[0] + uv[0], origin[1] + uv[1], origin[2]];
        }
        self.plane.point_at(uv)
    }

    /// World coordinates into the plane's.
    fn lift(&self, point: [f64; 3]) -> Option<[f64; 2]> {
        let origin = self.plane.origin;
        if self.plane.is_xy_aligned() {
            return Some([point[0] - origin[0], point[1] - origin[1]]);
        }
        self.plane.project(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom2d::{Arc, Line};
    use std::f64::consts::{FRAC_PI_2, TAU};

    fn unit_arc() -> Curve {
        Curve::Arc(Arc {
            centre: [0.0, 0.0],
            radius: 1.0,
            start_angle: 0.0,
            end_angle: FRAC_PI_2,
        })
    }

    /// The XZ plane: X stays X, the curve's Y becomes world Z. An entity with
    /// an extrusion normal of +Y is stored this way.
    fn upright() -> Plane {
        Plane::orthonormal([0.0; 3], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]).unwrap()
    }

    #[test]
    fn a_flat_curve_keeps_its_coordinates_and_gains_a_zero() {
        let curve = PlanarCurve::flat(Curve::Line(Line {
            start: [1.0, 2.0],
            end: [4.0, 6.0],
        }));
        assert_eq!(curve.point_at(0.0), [1.0, 2.0, 0.0]);
        assert_eq!(curve.point_at(1.0), [4.0, 6.0, 0.0]);
        assert_eq!(curve.tessellate(20.0), vec![[1.0, 2.0, 0.0], [4.0, 6.0, 0.0]]);
    }

    #[test]
    fn a_frame_moves_the_curve_without_reparameterising_it() {
        let flat = PlanarCurve::flat(unit_arc());
        let raised = PlanarCurve::new(upright(), unit_arc());
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let (a, b) = (flat.point_at(t), raised.point_at(t));
            // Same X and same distance from the centre; only which axis the
            // second coordinate points along has changed.
            assert!((a[0] - b[0]).abs() < 1e-12);
            assert!((a[1] - b[2]).abs() < 1e-12, "t={t}: {a:?} vs {b:?}");
            assert!(b[1].abs() < 1e-12);
        }
    }

    #[test]
    fn the_general_arm_agrees_with_the_xy_shortcut() {
        // Guards the branch in `lower`: the fast path must not be a different
        // answer, only a cheaper one. Both planes here are the same plane —
        // one written in the basis the shortcut recognises, one rotated a
        // full turn so it is not.
        let fast_frame = Plane::from_axes([7.0, -3.0, 2.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let slow_frame = Plane::orthonormal([7.0, -3.0, 2.5], [1.0, 1e-17, 0.0], [0.0, 0.0, 1.0])
            .expect("still a plane");
        assert!(fast_frame.is_xy_aligned());
        assert!(!slow_frame.is_xy_aligned(), "this test needs the long way");

        let fast = PlanarCurve::new(fast_frame, unit_arc()).tessellate(20.0);
        let slow = PlanarCurve::new(slow_frame, unit_arc()).tessellate(20.0);
        assert_eq!(fast.len(), slow.len());
        for (a, b) in fast.iter().zip(slow.iter()) {
            assert!(
                (a[0] - b[0]).abs() < 1e-12
                    && (a[1] - b[1]).abs() < 1e-12
                    && (a[2] - b[2]).abs() < 1e-12,
                "{a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn an_elevated_xy_plane_still_takes_the_shortcut() {
        // The case the shortcut exists for: a drawing's planar entities are
        // stored +Z-up and differ only in elevation.
        let plane = Plane::from_axes([0.0, 0.0, 12.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(plane.is_xy_aligned() && !plane.is_xy());
        let curve = PlanarCurve::new(plane, unit_arc());
        assert_eq!(curve.point_at(0.0), [1.0, 0.0, 12.0]);
        assert!(curve.parameter_at([1.0, 0.0, 12.0]).unwrap().abs() < 1e-12);
    }

    #[test]
    fn parameter_at_inverts_point_at_through_the_frame() {
        let curve = PlanarCurve::new(upright(), unit_arc());
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            let back = curve.parameter_at(curve.point_at(t)).unwrap();
            assert!((back - t).abs() < 1e-9, "t={t} came back as {back}");
        }
    }

    #[test]
    fn a_point_off_the_plane_is_projected_onto_it_first() {
        let curve = PlanarCurve::new(upright(), unit_arc());
        // On the arc at t=0, but pushed a metre along the normal.
        let t = curve.parameter_at([1.0, -1.0, 0.0]).unwrap();
        assert!(t.abs() < 1e-9, "got {t}");
    }

    #[test]
    fn a_degenerate_plane_refuses_rather_than_guessing() {
        let flat = Plane::from_axes([0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        let curve = PlanarCurve::new(flat, unit_arc());
        assert!(curve.parameter_at([0.0; 3]).is_none());
        assert!(curve.normal().is_none());
    }

    #[test]
    fn the_shape_answers_for_closure_and_extent_not_the_frame() {
        let arc = PlanarCurve::new(upright(), unit_arc());
        assert!(!arc.is_closed());
        assert_eq!(arc.extent(), Extent::Bounded);

        let circle = PlanarCurve::new(
            upright(),
            Curve::Arc(Arc {
                centre: [0.0, 0.0],
                radius: 1.0,
                start_angle: 0.0,
                end_angle: TAU,
            }),
        );
        assert!(circle.is_closed());
    }

    #[test]
    fn the_normal_comes_from_the_plane() {
        assert_eq!(
            PlanarCurve::flat(unit_arc()).normal(),
            Some([0.0, 0.0, 1.0])
        );
        assert_eq!(
            PlanarCurve::new(upright(), unit_arc()).normal(),
            Some([0.0, -1.0, 0.0])
        );
    }

    #[test]
    fn tessellating_stays_on_the_plane() {
        let plane = Plane::orthonormal([3.0, 4.0, 5.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]).unwrap();
        let curve = PlanarCurve::new(plane, unit_arc());
        for point in curve.tessellate(20.0) {
            assert!(
                plane.contains(point, 1e-9),
                "{point:?} left the plane by {:?}",
                plane.distance_to(point)
            );
        }
    }
}
