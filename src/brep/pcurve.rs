//! A space curve seen in a surface's own coordinates.
//!
//! Splitting a face is a two-dimensional problem. The face is a region of its
//! surface's `(u, v)` space bounded by loops; the curve that cuts it is
//! another set of `(u, v)` points; and what has to happen — where they cross,
//! which pieces the cut leaves, which side of the cut a piece is on — is
//! exactly what [`geom2d`](crate::geom2d) answers. So the last step before a
//! boolean can do anything is bringing the curve down into that space.
//!
//! ACIS calls the result a pcurve and stores one on every coedge, for the
//! same reason.
//!
//! # Parameters do not carry across
//!
//! The result traces the same *points* as the space curve, not the same
//! parameter values: a circle in space is parameterised by angle here and by
//! a fraction of a turn in [`geom2d`]. A caller needing the other curve's
//! parameter goes through the point, which both kinds invert exactly.
//! Inventing a mapping type to carry the difference would put a conversion in
//! every signature to save one call at a handful of sites.
//!
//! # Where the projection does not exist
//!
//! On a cylinder, a slanted plane's section is a sine wave in `(u, v)` — no
//! [`Curve`] variant is one, and calling it a spline would be an
//! approximation dressed as a fact. Those cases answer `None`, on the same
//! principle as [`Meeting::Unknown`](super::Meeting): a caller told nothing
//! can refuse, a caller told something wrong cannot.

use super::geometry::{Curve3, Surface};
use crate::geom2d::{Circle, Curve, Ellipse, EllipseArc, Line, XLine};
use crate::space::Vec3;
use std::f64::consts::TAU;

/// The curve `curve` traces in `surface`'s parameter space.
///
/// `None` when the curve does not lie on the surface, and when the shape it
/// traces there is not one this kernel can write down.
pub fn project(surface: &Surface, curve: &Curve3, tolerance: f64) -> Option<Curve> {
    // A curve that is not on the surface has no image in its parameter
    // space, and projecting it anyway would produce a plausible shape in the
    // wrong place. Sampled rather than reasoned about per pair: the check is
    // the same for every combination and cheap next to what follows.
    if !lies_on(surface, curve, tolerance) {
        return None;
    }
    match surface {
        Surface::Plane(plane) => match curve {
            Curve3::Line(line) => Some(Curve::XLine(XLine {
                base: plane.project(line.origin)?,
                direction: plane.project_vector(line.direction)?,
            })),
            Curve3::Circle(circle) => Some(Curve::Circle(Circle {
                centre: plane.project(circle.plane.origin)?,
                radius: circle.radius,
            })),
            Curve3::Ellipse(ellipse) => {
                let major = plane.project_vector(ellipse.plane.x_axis)?;
                let length = major[0].hypot(major[1]);
                if length <= tolerance {
                    return None;
                }
                Some(Curve::Ellipse(EllipseArc {
                    ellipse: Ellipse {
                        centre: plane.project(ellipse.plane.origin)?,
                        major_radius: ellipse.major_radius,
                        minor_radius: ellipse.minor_radius,
                        major_axis: [major[0] / length, major[1] / length],
                    },
                    start_parameter: 0.0,
                    end_parameter: TAU,
                }))
            }
            // Already a plane curve; it only has to be re-expressed in this
            // plane's coordinates rather than its own.
            Curve3::PlanarSpline { curve, .. } => Some(Curve::Nurbs(curve.clone())),
        },

        // On a cylinder `u` runs round and `v` along the axis, so the two
        // curves that stay straight there are the ones aligned with it: a
        // circle at one height, and a generator. A slanted section is a sine
        // wave, which is why the general case is absent rather than
        // approximated.
        Surface::Cylinder(cylinder) => match curve {
            Curve3::Circle(circle) => {
                let axis = Vec3::from(cylinder.base.normal()?);
                let plane_normal = Vec3::from(circle.plane.normal()?);
                if !plane_normal.is_parallel_to(axis, tolerance) {
                    return None;
                }
                let height = (Vec3::from(circle.plane.origin)
                    - Vec3::from(cylinder.base.origin))
                .dot(axis);
                Some(band_at(height))
            }
            Curve3::Line(line) => {
                let axis = Vec3::from(cylinder.base.normal()?);
                if !Vec3::from(line.direction).is_parallel_to(axis, tolerance) {
                    return None;
                }
                Some(generator_at(angle_about(
                    &cylinder.base,
                    line.origin,
                )?))
            }
            _ => None,
        },

        // The same two shapes on a cone, with `v` measured along the axis as
        // it is for a cylinder.
        Surface::Cone(cone) => match curve {
            Curve3::Circle(circle) => {
                let axis = Vec3::from(cone.base.normal()?);
                let plane_normal = Vec3::from(circle.plane.normal()?);
                if !plane_normal.is_parallel_to(axis, tolerance) {
                    return None;
                }
                let height =
                    (Vec3::from(circle.plane.origin) - Vec3::from(cone.base.origin)).dot(axis);
                Some(band_at(height))
            }
            Curve3::Line(line) => {
                // A generator leans by the half-angle, so it is not parallel
                // to the axis; what identifies it is that it keeps one angle
                // all the way along.
                let start = angle_about(&cone.base, line.origin)?;
                let further = angle_about(
                    &cone.base,
                    (Vec3::from(line.origin) + Vec3::from(line.direction)).to_array(),
                )?;
                let turn = (further - start).abs();
                if turn.min(TAU - turn) > tolerance {
                    return None;
                }
                Some(generator_at(start))
            }
            _ => None,
        },

        Surface::Sphere(_) | Surface::Torus(_) => None,
    }
}

/// A face's loops as curves in its surface's parameter space, each trimmed
/// to the edge it came from and oriented the way its loop runs.
///
/// The trimming is the part that matters for containment. A straight edge
/// projects to an infinite line, and a boundary made of infinite lines
/// encloses nothing and reports every point as lying on it.
///
/// The orientation matters for anything walking the ring. An edge has one
/// direction and its two coedges disagree about it, so a boundary built from
/// the edges runs backwards wherever the loop does — half the time on any
/// closed solid. A caller chaining the pieces then gets a ring in no order at
/// all.
///
/// `None` when any edge's projection has no closed form: a boundary with a
/// piece missing is worse than none, since a caller would take the rest for
/// the whole.
pub fn face_boundary(
    body: &super::topology::Body,
    face: super::topology::FaceKey,
    tolerance: f64,
) -> Option<Vec<Curve>> {
    let node = body.faces.get(face)?;
    let surface = body.surfaces.get(node.surface)?;
    let mut out = Vec::new();
    for coedge in body.face_coedges(face) {
        let edge_key = body.coedges.get(coedge)?.edge;
        let edge = body.edges.get(edge_key)?;
        let curve = body.curves.get(edge.curve)?;
        let flat = project(surface, curve, tolerance)?;
        let (start, end) = body.edge_endpoints(edge_key)?;
        // The loop's own direction, not the edge's.
        let (start, end) = if body.coedges.get(coedge)?.forward {
            (start, end)
        } else {
            (end, start)
        };
        let (from, to) = (
            surface.parameters_at(start)?,
            surface.parameters_at(end)?,
        );
        out.push(trim_to(flat, [from.0, from.1], [to.0, to.1]));
    }
    Some(out)
}

/// The part of a projected boundary curve between two parameter-space points.
fn trim_to(curve: Curve, from: [f64; 2], to: [f64; 2]) -> Curve {
    match curve {
        // The kinds that run past their edge become the segment between the
        // two ends. A straight edge's projection is straight, so nothing is
        // lost by saying so.
        Curve::XLine(_) | Curve::Ray(_) => Curve::Line(Line { start: from, end: to }),
        other => other,
    }
}

/// A line of constant `v`, spanning one full turn of `u`.
///
/// Bounded rather than infinite because `u` on a closed surface is periodic:
/// past a turn it repeats, and a curve that ran on forever would report every
/// crossing an unbounded number of times.
fn band_at(height: f64) -> Curve {
    Curve::Line(Line {
        start: [0.0, height],
        end: [TAU, height],
    })
}

/// A line of constant `u`, unbounded in `v` — a generator, which the face's
/// own extent trims.
fn generator_at(angle: f64) -> Curve {
    Curve::XLine(XLine {
        base: [angle, 0.0],
        direction: [0.0, 1.0],
    })
}

/// Where `point` sits around a frame's axis, in radians from its x axis.
fn angle_about(frame: &crate::space::Plane, point: [f64; 3]) -> Option<f64> {
    let local = frame.project(point)?;
    Some(local[1].atan2(local[0]))
}

/// Whether every sampled point of the curve is on the surface.
fn lies_on(surface: &Surface, curve: &Curve3, tolerance: f64) -> bool {
    // Spread over a range wide enough to catch a curve that touches the
    // surface but does not follow it — a line crossing a sphere is on it at
    // two parameters and nowhere else.
    const SAMPLES: usize = 9;
    (0..SAMPLES).all(|index| {
        let t = match curve {
            Curve3::Line(_) => -2.0 + 4.0 * index as f64 / (SAMPLES - 1) as f64,
            Curve3::PlanarSpline { .. } => index as f64 / (SAMPLES - 1) as f64,
            _ => TAU * index as f64 / SAMPLES as f64,
        };
        surface.contains(curve.point_at(t), tolerance)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brep::geometry::{Circle3, Cone, Cylinder, Ellipse3, Line3, Sphere};
    use crate::geom2d::NurbsCurve;
    use crate::space::Plane;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    const TOL: f64 = 1e-9;

    fn plane_at(origin: [f64; 3], normal: [f64; 3]) -> Plane {
        let seed = if normal[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        Plane::orthonormal(origin, seed, normal).unwrap()
    }

    /// Every point of the projected curve, lifted back through the surface,
    /// must land on the space curve. That round trip is the whole contract.
    fn assert_round_trips(surface: &Surface, curve: &Curve3, flat: &Curve, span: (f64, f64)) {
        for index in 0..=10 {
            let t = span.0 + (span.1 - span.0) * index as f64 / 10.0;
            let uv = flat.point_at(t);
            let lifted = surface.point_at(uv[0], uv[1]);
            // The lifted point has to be *on* the space curve, which is
            // asked by inverting the curve and evaluating it back.
            let back = curve.point_at(curve.parameter_at(lifted));
            assert!(
                Vec3::from(lifted).distance(Vec3::from(back)) < 1e-6,
                "t={t}: {lifted:?} is not on the curve ({back:?})"
            );
        }
    }

    #[test]
    fn a_line_on_a_plane_projects_to_a_line() {
        let plane = plane_at([0.0, 0.0, 5.0], [0.0, 0.0, 1.0]);
        let surface = Surface::Plane(plane);
        let curve = Curve3::Line(Line3 {
            origin: [1.0, 2.0, 5.0],
            direction: [3.0, 4.0, 0.0],
        });
        let flat = project(&surface, &curve, TOL).expect("a line on its own plane");
        assert_round_trips(&surface, &curve, &flat, (-2.0, 2.0));
        // And the parameter happens to carry across for a straight curve,
        // since both are `base + t · direction`.
        assert_eq!(flat.point_at(1.0), [4.0, 6.0]);
    }

    #[test]
    fn a_circle_on_a_plane_projects_to_a_circle() {
        let plane = plane_at([0.0; 3], [0.0, 0.0, 1.0]);
        let surface = Surface::Plane(plane);
        let curve = Curve3::Circle(Circle3 {
            plane: plane_at([2.0, 3.0, 0.0], [0.0, 0.0, 1.0]),
            radius: 4.0,
        });
        let flat = project(&surface, &curve, TOL).expect("a circle on its own plane");
        let Curve::Circle(circle) = &flat else {
            panic!("expected a circle, got {flat:?}");
        };
        assert!((circle.radius - 4.0).abs() < 1e-12);
        assert_round_trips(&surface, &curve, &flat, (0.0, 1.0));
    }

    #[test]
    fn an_ellipse_on_a_plane_keeps_both_its_radii_and_its_direction() {
        let plane = plane_at([0.0; 3], [0.0, 0.0, 1.0]);
        let surface = Surface::Plane(plane);
        // Major axis along +Y, so a projection that assumed +X would be a
        // quarter turn out.
        let frame = Plane::from_axes([1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]);
        let curve = Curve3::Ellipse(Ellipse3 {
            plane: frame,
            major_radius: 6.0,
            minor_radius: 2.0,
        });
        let flat = project(&surface, &curve, TOL).expect("an ellipse on its own plane");
        let Curve::Ellipse(arc) = &flat else {
            panic!("expected an ellipse, got {flat:?}");
        };
        assert!((arc.ellipse.major_radius - 6.0).abs() < 1e-12);
        assert!((arc.ellipse.major_axis[1] - 1.0).abs() < 1e-12, "{:?}", arc.ellipse.major_axis);
        assert_round_trips(&surface, &curve, &flat, (0.0, 1.0));
    }

    #[test]
    fn a_curve_off_the_surface_has_no_projection() {
        // The failure this guards: projecting it anyway gives a shape of the
        // right kind sitting in the wrong place, which validates and is
        // wrong.
        let surface = Surface::Plane(plane_at([0.0; 3], [0.0, 0.0, 1.0]));
        let above = Curve3::Line(Line3 {
            origin: [0.0, 0.0, 4.0],
            direction: [1.0, 0.0, 0.0],
        });
        assert!(project(&surface, &above, TOL).is_none());
        // One that merely crosses the plane is not on it either.
        let through = Curve3::Line(Line3 {
            origin: [0.0, 0.0, -1.0],
            direction: [0.0, 0.0, 1.0],
        });
        assert!(project(&surface, &through, TOL).is_none());
    }

    #[test]
    fn a_circle_round_a_cylinder_becomes_a_straight_band() {
        // The point of parameter space: a circle is a straight line there.
        let cylinder = Cylinder {
            base: plane_at([0.0; 3], [0.0, 0.0, 1.0]),
            radius: 3.0,
        };
        let surface = Surface::Cylinder(cylinder);
        let curve = Curve3::Circle(Circle3 {
            plane: plane_at([0.0, 0.0, 7.0], [0.0, 0.0, 1.0]),
            radius: 3.0,
        });
        let flat = project(&surface, &curve, TOL).expect("a circle round its own cylinder");
        let Curve::Line(line) = &flat else {
            panic!("expected a line in (u, v), got {flat:?}");
        };
        assert!((line.start[1] - 7.0).abs() < 1e-12, "the height is v");
        assert!((line.end[0] - TAU).abs() < 1e-12, "one full turn of u");
        assert_round_trips(&surface, &curve, &flat, (0.0, 1.0));
    }

    #[test]
    fn a_generator_on_a_cylinder_becomes_a_line_of_constant_angle() {
        let cylinder = Cylinder {
            base: plane_at([0.0; 3], [0.0, 0.0, 1.0]),
            radius: 3.0,
        };
        let surface = Surface::Cylinder(cylinder);
        // The generator at ninety degrees round.
        let curve = Curve3::Line(Line3 {
            origin: [0.0, 3.0, 0.0],
            direction: [0.0, 0.0, 1.0],
        });
        let flat = project(&surface, &curve, TOL).expect("a generator on its own cylinder");
        let Curve::XLine(line) = &flat else {
            panic!("expected an infinite line, got {flat:?}");
        };
        assert!((line.base[0] - FRAC_PI_2).abs() < 1e-9, "{:?}", line.base);
        assert_round_trips(&surface, &curve, &flat, (-3.0, 3.0));
    }

    #[test]
    fn a_slanted_section_of_a_cylinder_has_no_written_form() {
        // It is a sine wave in (u, v). Calling it a spline would be an
        // approximation presented as a fact.
        let surface = Surface::Cylinder(Cylinder {
            base: plane_at([0.0; 3], [0.0, 0.0, 1.0]),
            radius: 3.0,
        });
        let slanted = Curve3::Ellipse(Ellipse3 {
            plane: plane_at([0.0; 3], [0.0, 1.0, 1.0]),
            major_radius: 3.0 * std::f64::consts::SQRT_2,
            minor_radius: 3.0,
        });
        assert!(project(&surface, &slanted, TOL).is_none());
    }

    #[test]
    fn a_circle_round_a_cone_becomes_a_band_at_its_height() {
        let cone = Cone {
            base: plane_at([0.0; 3], [0.0, 0.0, 1.0]),
            radius: 10.0,
            half_angle: FRAC_PI_4,
        };
        let surface = Surface::Cone(cone);
        let curve = Curve3::Circle(Circle3 {
            plane: plane_at([0.0, 0.0, 4.0], [0.0, 0.0, 1.0]),
            radius: 6.0,
        });
        let flat = project(&surface, &curve, TOL).expect("a circle round its own cone");
        let Curve::Line(line) = &flat else {
            panic!("expected a line, got {flat:?}");
        };
        assert!((line.start[1] - 4.0).abs() < 1e-12);
        assert_round_trips(&surface, &curve, &flat, (0.0, 1.0));
    }

    #[test]
    fn a_generator_on_a_cone_leans_and_is_still_one_angle() {
        // Unlike a cylinder's, a cone's generator is not parallel to the
        // axis, so a parallel test would reject it. What identifies it is
        // holding one angle the whole way.
        let cone = Cone {
            base: plane_at([0.0; 3], [0.0, 0.0, 1.0]),
            radius: 10.0,
            half_angle: FRAC_PI_4,
        };
        let surface = Surface::Cone(cone);
        let curve = Curve3::Line(Line3 {
            origin: [10.0, 0.0, 0.0],
            direction: [-1.0, 0.0, 1.0],
        });
        let flat = project(&surface, &curve, 1e-6).expect("a generator on its own cone");
        let Curve::XLine(line) = &flat else {
            panic!("expected an infinite line, got {flat:?}");
        };
        assert!(line.base[0].abs() < 1e-9, "at angle zero: {:?}", line.base);
    }

    #[test]
    fn a_spline_on_its_own_plane_carries_over_whole() {
        let frame = plane_at([0.0; 3], [0.0, 0.0, 1.0]);
        let surface = Surface::Plane(frame);
        let curve = Curve3::PlanarSpline {
            plane: frame,
            curve: NurbsCurve::new(
                3,
                vec![[0.0, 0.0], [1.0, 5.0], [4.0, 5.0], [5.0, 0.0]],
                Vec::new(),
                None,
            )
            .unwrap(),
        };
        let flat = project(&surface, &curve, TOL).expect("a spline on its own plane");
        assert!(matches!(flat, Curve::Nurbs(_)));
        assert_round_trips(&surface, &curve, &flat, (0.0, 1.0));
    }

    #[test]
    fn a_sphere_is_not_claimed_to_be_understood() {
        let surface = Surface::Sphere(Sphere {
            frame: plane_at([0.0; 3], [0.0, 0.0, 1.0]),
            radius: 5.0,
        });
        let equator = Curve3::Circle(Circle3 {
            plane: plane_at([0.0; 3], [0.0, 0.0, 1.0]),
            radius: 5.0,
        });
        assert!(project(&surface, &equator, TOL).is_none());
    }

    #[test]
    fn a_face_of_a_box_projects_its_own_boundary() {
        // The case the boolean will actually meet first: a planar face and
        // the straight edges that bound it.
        let body = crate::brep::make::cuboid([0.0; 3], [2.0, 3.0, 4.0]).unwrap();
        let mut projected = 0;
        for face in body.face_keys() {
            let surface = body
                .surfaces
                .get(body.faces.get(face).unwrap().surface)
                .unwrap();
            for coedge in body.face_coedges(face) {
                let edge = body.edges.get(body.coedges.get(coedge).unwrap().edge).unwrap();
                let curve = body.curves.get(edge.curve).unwrap();
                let flat = project(surface, curve, 1e-9)
                    .expect("a box's edge lies on the faces it bounds");
                assert_round_trips(surface, curve, &flat, (0.0, 1.0));
                projected += 1;
            }
        }
        assert_eq!(projected, 24, "four edges on each of six faces");
    }

    #[test]
    fn survey_coordinates_project_to_the_same_place() {
        let origin = [512_345.678, 4_512_345.678, 91.5];
        let plane = plane_at(origin, [0.0, 0.0, 1.0]);
        let surface = Surface::Plane(plane);
        let curve = Curve3::Circle(Circle3 {
            plane: plane_at([origin[0] + 1.0, origin[1] + 2.0, origin[2]], [0.0, 0.0, 1.0]),
            radius: 4.0,
        });
        let flat = project(&surface, &curve, 1e-6).expect("still on its plane");
        let Curve::Circle(circle) = &flat else {
            unreachable!()
        };
        assert!((circle.centre[0] - 1.0).abs() < 1e-6, "{:?}", circle.centre);
        assert!((circle.radius - 4.0).abs() < 1e-9);
    }
}
