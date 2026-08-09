//! The shapes a face sits on and an edge runs along.
//!
//! Topology says which face borders which; geometry says where they are. The
//! two are kept apart because they change independently — a boolean rewrites
//! adjacency while leaving every surface exactly as it was, and a face can be
//! moved without any coedge noticing.
//!
//! # Analytic first, spline last
//!
//! A cylinder is stored as a cylinder, not as a spline that happens to look
//! like one. That is not an optimisation: an ACIS file says `cone`, and
//! lowering it back as a B-spline surface loses the fact that it was ever
//! round — every downstream consumer, including the next modeller to open
//! the file, then sees an approximation. The analytic cases are also the ones
//! whose intersections have closed forms, which is the difference between a
//! boolean that lands on the exact circle two cylinders share and one that
//! lands near it.
//!
//! Surfaces this layer does not model yet are not lost either: a face lifted
//! from a file and left alone lowers back as its own record. See
//! [`Provenance`](super::Provenance).

use crate::geom2d::NurbsCurve;
use crate::space::{Plane, Vec3};

/// A surface a face lies on.
///
/// `(u, v)` is the surface's own parameter space, which is where a face's
/// loops are resolved and where a boolean does its cutting.
#[derive(Debug, Clone, PartialEq)]
pub enum Surface {
    /// Flat. `(u, v)` are the plane's own coordinates.
    Plane(Plane),
    /// A right circular cylinder about `axis`. `u` runs round it in radians,
    /// `v` along the axis.
    Cylinder(Cylinder),
    /// A right circular cone, which a cylinder is the zero-angle case of —
    /// kept separate because ACIS stores them as one record with a flag, and
    /// because their intersections behave differently.
    Cone(Cone),
    /// A sphere. `u` is longitude, `v` latitude, both in radians.
    Sphere(Sphere),
    /// A torus. `u` runs the major circle, `v` the minor.
    Torus(Torus),
}

/// A right circular cylinder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylinder {
    /// A point on the axis, and the frame `u` is measured from: `x_axis` is
    /// where `u = 0` points, and the axis itself is the frame's normal.
    pub base: Plane,
    /// Distance from the axis.
    pub radius: f64,
}

/// A right circular cone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cone {
    /// The frame at the reference circle, as for a cylinder.
    pub base: Plane,
    /// Radius of that circle. This is the radius *at the base*, not at the
    /// apex — reading it as the apex radius puts every generator on the wrong
    /// slope.
    pub radius: f64,
    /// Half-angle at the apex, positive when the cone narrows along the axis.
    pub half_angle: f64,
}

/// A sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    /// Centre, and the frame longitude and latitude are measured in.
    pub frame: Plane,
    /// Radius.
    pub radius: f64,
}

/// A torus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Torus {
    /// Centre, and the frame the major circle lies in.
    pub frame: Plane,
    /// Distance from the centre to the middle of the tube.
    pub major_radius: f64,
    /// Radius of the tube.
    pub minor_radius: f64,
}

impl Surface {
    /// The point at surface parameters `(u, v)`.
    pub fn point_at(&self, u: f64, v: f64) -> [f64; 3] {
        match self {
            Self::Plane(plane) => plane.point_at([u, v]),
            Self::Cylinder(cylinder) => {
                let around = cylinder.base.point_at([
                    cylinder.radius * u.cos(),
                    cylinder.radius * u.sin(),
                ]);
                offset_along_normal(&cylinder.base, around, v)
            }
            Self::Cone(cone) => {
                // The radius shrinks along the axis at the tangent of the
                // half-angle, from the *base* circle.
                let radius = cone.radius - v * cone.half_angle.tan();
                let around = cone
                    .base
                    .point_at([radius * u.cos(), radius * u.sin()]);
                offset_along_normal(&cone.base, around, v)
            }
            Self::Sphere(sphere) => {
                let ring = sphere.radius * v.cos();
                let around = sphere
                    .frame
                    .point_at([ring * u.cos(), ring * u.sin()]);
                offset_along_normal(&sphere.frame, around, sphere.radius * v.sin())
            }
            Self::Torus(torus) => {
                let ring = torus.major_radius + torus.minor_radius * v.cos();
                let around = torus.frame.point_at([ring * u.cos(), ring * u.sin()]);
                offset_along_normal(&torus.frame, around, torus.minor_radius * v.sin())
            }
        }
    }

    /// The frame a surface's parameters are measured in, for the kinds that
    /// have one. Every variant here does; the accessor exists so a caller can
    /// move a surface without matching on its kind.
    pub fn frame(&self) -> &Plane {
        match self {
            Self::Plane(plane) => plane,
            Self::Cylinder(cylinder) => &cylinder.base,
            Self::Cone(cone) => &cone.base,
            Self::Sphere(sphere) => &sphere.frame,
            Self::Torus(torus) => &torus.frame,
        }
    }

    /// Whether `point` lies on the surface, to within `tolerance`.
    ///
    /// The check a lift wants: a file's topology says a face sits on a
    /// surface, and a vertex that does not is a parse gone wrong rather than
    /// a shape.
    pub fn contains(&self, point: [f64; 3], tolerance: f64) -> bool {
        self.distance_to(point).abs() <= tolerance
    }

    /// Signed distance from `point` to the surface, positive outside.
    ///
    /// Exact for every analytic kind here — no search, no sampling.
    pub fn distance_to(&self, point: [f64; 3]) -> f64 {
        match self {
            Self::Plane(plane) => plane.distance_to(point).unwrap_or(f64::INFINITY),
            Self::Cylinder(cylinder) => {
                axial_distance(&cylinder.base, point).1 - cylinder.radius
            }
            Self::Cone(cone) => {
                let (along, across) = axial_distance(&cone.base, point);
                // Distance to the generator line in the (across, along) half
                // plane, which is the cone's own profile.
                let (sin, cos) = cone.half_angle.sin_cos();
                (across - cone.radius) * cos + along * sin
            }
            Self::Sphere(sphere) => {
                Vec3::from(point).distance(Vec3::from(sphere.frame.origin)) - sphere.radius
            }
            Self::Torus(torus) => {
                let (along, across) = axial_distance(&torus.frame, point);
                let from_tube = (across - torus.major_radius).hypot(along);
                from_tube - torus.minor_radius
            }
        }
    }
}

/// Moves `point` by `distance` along `frame`'s normal.
fn offset_along_normal(frame: &Plane, point: [f64; 3], distance: f64) -> [f64; 3] {
    match frame.normal() {
        Some(normal) => (Vec3::from(point) + Vec3::from(normal) * distance).to_array(),
        None => point,
    }
}

/// How far `point` is along a frame's axis, and how far from it.
fn axial_distance(frame: &Plane, point: [f64; 3]) -> (f64, f64) {
    let offset = Vec3::from(point) - Vec3::from(frame.origin);
    match frame.normal() {
        Some(normal) => {
            let along = offset.dot(Vec3::from(normal));
            let across = (offset - Vec3::from(normal) * along).length();
            (along, across)
        }
        None => (0.0, offset.length()),
    }
}

/// A curve an edge runs along, in space.
#[derive(Debug, Clone, PartialEq)]
pub enum Curve3 {
    /// A straight line. `t` advances by `direction` per unit, so a segment's
    /// two vertices sit at whatever parameters the edge records.
    Line(Line3),
    /// A circle or a circular arc, lying in `plane`. `t` is the angle in
    /// radians measured from the plane's x axis.
    Circle(Circle3),
    /// A spline curve lying in a plane. The planar case is what a drawing and
    /// a planar face's boundary produce; a spline that genuinely wanders in
    /// space is not modelled yet and stays with its source record.
    PlanarSpline {
        /// The plane it lies in.
        plane: Plane,
        /// The curve, in that plane's coordinates.
        curve: NurbsCurve,
    },
}

/// A straight line in space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line3 {
    /// A point on it, where `t` reads zero.
    pub origin: [f64; 3],
    /// One unit of `t`.
    pub direction: [f64; 3],
}

/// A circle in space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle3 {
    /// Centre, and the frame the angle is measured in.
    pub plane: Plane,
    /// Radius.
    pub radius: f64,
}

impl Curve3 {
    /// The point at parameter `t`.
    pub fn point_at(&self, t: f64) -> [f64; 3] {
        match self {
            Self::Line(line) => {
                (Vec3::from(line.origin) + Vec3::from(line.direction) * t).to_array()
            }
            Self::Circle(circle) => circle
                .plane
                .point_at([circle.radius * t.cos(), circle.radius * t.sin()]),
            Self::PlanarSpline { plane, curve } => plane.point_at(curve.point_at(t)),
        }
    }

    /// The parameter at `point`, the inverse of [`point_at`](Self::point_at).
    ///
    /// A point off the curve is projected onto it, so a caller matching a
    /// file's vertex against the curve its edge names does not have to be
    /// exact first.
    pub fn parameter_at(&self, point: [f64; 3]) -> f64 {
        match self {
            Self::Line(line) => {
                let along = Vec3::from(line.direction);
                let squared = along.length_squared();
                if squared <= 0.0 {
                    return 0.0;
                }
                (Vec3::from(point) - Vec3::from(line.origin)).dot(along) / squared
            }
            Self::Circle(circle) => match circle.plane.project(point) {
                Some(local) => local[1].atan2(local[0]),
                None => 0.0,
            },
            Self::PlanarSpline { plane, curve } => match plane.project(point) {
                Some(local) => curve.parameter_at(local),
                None => 0.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

    fn xy() -> Plane {
        Plane::XY
    }

    #[test]
    fn a_plane_surface_is_its_own_coordinates() {
        let surface = Surface::Plane(xy());
        assert_eq!(surface.point_at(3.0, -4.0), [3.0, -4.0, 0.0]);
        assert!(surface.contains([3.0, -4.0, 0.0], 1e-9));
        assert!(!surface.contains([3.0, -4.0, 1.0], 1e-9));
        assert!((surface.distance_to([0.0, 0.0, 2.5]) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn a_cylinder_wraps_round_its_axis_and_runs_along_it() {
        let surface = Surface::Cylinder(Cylinder {
            base: xy(),
            radius: 5.0,
        });
        let point = surface.point_at(0.0, 3.0);
        assert!((point[0] - 5.0).abs() < 1e-12 && point[1].abs() < 1e-12);
        assert!((point[2] - 3.0).abs() < 1e-12, "{point:?}");
        // Everything on it is exactly the radius from the axis, whatever the
        // height.
        for u in [0.0, 1.0, PI, 5.0] {
            for v in [-10.0, 0.0, 7.0] {
                assert!(surface.contains(surface.point_at(u, v), 1e-9));
            }
        }
        assert!((surface.distance_to([8.0, 0.0, 100.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn a_cone_narrows_from_its_base_radius() {
        // Forty-five degrees, so it loses a unit of radius per unit of
        // height. Reading the radius as the apex one instead would put this
        // point a long way off.
        let surface = Surface::Cone(Cone {
            base: xy(),
            radius: 10.0,
            half_angle: FRAC_PI_4,
        });
        let at_base = surface.point_at(0.0, 0.0);
        assert!((at_base[0] - 10.0).abs() < 1e-12);
        let higher = surface.point_at(0.0, 4.0);
        assert!((higher[0] - 6.0).abs() < 1e-12, "{higher:?}");
        for v in [0.0, 3.0, 9.0] {
            assert!(surface.contains(surface.point_at(1.2, v), 1e-9), "v={v}");
        }
    }

    #[test]
    fn a_sphere_holds_every_point_at_its_radius() {
        let surface = Surface::Sphere(Sphere {
            frame: Plane::from_axes([1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            radius: 4.0,
        });
        for u in [0.0, 1.0, PI] {
            for v in [-FRAC_PI_2 + 0.1, 0.0, FRAC_PI_2 - 0.1] {
                let point = surface.point_at(u, v);
                assert!(surface.contains(point, 1e-9), "u={u} v={v}");
                let radius = Vec3::from(point).distance(Vec3::new(1.0, 2.0, 3.0));
                assert!((radius - 4.0).abs() < 1e-9);
            }
        }
        // The pole sits a full radius up the frame's normal.
        let pole = surface.point_at(0.0, FRAC_PI_2);
        assert!((pole[2] - 7.0).abs() < 1e-9, "{pole:?}");
    }

    #[test]
    fn a_torus_holds_every_point_at_its_tube_radius() {
        let surface = Surface::Torus(Torus {
            frame: xy(),
            major_radius: 10.0,
            minor_radius: 2.0,
        });
        for u in [0.0, 1.0, PI, 5.0] {
            for v in [0.0, 1.0, PI, 5.0] {
                assert!(surface.contains(surface.point_at(u, v), 1e-9), "u={u} v={v}");
            }
        }
        // The outermost point of the ring, and the innermost.
        assert!((surface.point_at(0.0, 0.0)[0] - 12.0).abs() < 1e-12);
        assert!((surface.point_at(0.0, PI)[0] - 8.0).abs() < 1e-12);
        // The centre of the hole is a major radius from the tube.
        assert!((surface.distance_to([0.0, 0.0, 0.0]) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn a_tilted_frame_carries_the_whole_surface_with_it() {
        let upright = Plane::orthonormal([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]).unwrap();
        let surface = Surface::Cylinder(Cylinder {
            base: upright,
            radius: 3.0,
        });
        // The axis is now +Y, so running along it moves in Y.
        let point = surface.point_at(0.0, 5.0);
        assert!((point[1] - 5.0).abs() < 1e-9, "{point:?}");
        assert!(surface.contains(point, 1e-9));
    }

    #[test]
    fn a_straight_edge_runs_between_its_parameters() {
        let curve = Curve3::Line(Line3 {
            origin: [1.0, 2.0, 3.0],
            direction: [0.0, 0.0, 4.0],
        });
        assert_eq!(curve.point_at(0.0), [1.0, 2.0, 3.0]);
        assert_eq!(curve.point_at(2.0), [1.0, 2.0, 11.0]);
        assert!((curve.parameter_at([1.0, 2.0, 11.0]) - 2.0).abs() < 1e-12);
        // A point beside the line projects onto it rather than being refused.
        assert!((curve.parameter_at([9.0, 9.0, 7.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_circular_edge_reads_its_parameter_as_an_angle() {
        let curve = Curve3::Circle(Circle3 {
            plane: xy(),
            radius: 6.0,
        });
        assert_eq!(curve.point_at(0.0), [6.0, 0.0, 0.0]);
        let quarter = curve.point_at(FRAC_PI_2);
        assert!(quarter[0].abs() < 1e-12 && (quarter[1] - 6.0).abs() < 1e-12);
        assert!((curve.parameter_at([0.0, 6.0, 0.0]) - FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn a_circular_edge_on_a_tilted_plane_stays_on_it() {
        let plane = Plane::orthonormal([3.0, 4.0, 5.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]).unwrap();
        let curve = Curve3::Circle(Circle3 { plane, radius: 2.0 });
        for i in 0..8 {
            let point = curve.point_at(TAU * i as f64 / 8.0);
            assert!(plane.contains(point, 1e-9), "{point:?}");
            assert!(
                (Vec3::from(point).distance(Vec3::from(plane.origin)) - 2.0).abs() < 1e-9
            );
        }
    }

    #[test]
    fn a_planar_spline_edge_evaluates_through_its_plane() {
        let plane = Plane::orthonormal([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]).unwrap();
        let curve = Curve3::PlanarSpline {
            plane,
            curve: NurbsCurve::new(
                2,
                vec![[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]],
                Vec::new(),
                None,
            )
            .unwrap(),
        };
        assert_eq!(curve.point_at(0.0), [0.0, 0.0, 0.0]);
        let end = curve.point_at(1.0);
        assert!((end[0] - 10.0).abs() < 1e-9, "{end:?}");
        for i in 0..=8 {
            let t = i as f64 / 8.0;
            assert!(plane.contains(curve.point_at(t), 1e-9));
        }
    }

    #[test]
    fn survey_coordinates_keep_a_surface_where_it_is() {
        let far = Plane::from_axes(
            [512_345.678, 4_512_345.678, 91.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        let surface = Surface::Cylinder(Cylinder {
            base: far,
            radius: 0.5,
        });
        for u in [0.0, 1.0, 3.0] {
            assert!(surface.contains(surface.point_at(u, 2.0), 1e-9), "u={u}");
        }
    }
}
