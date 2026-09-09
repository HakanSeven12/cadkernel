//! Turning an open sheet body into a closed solid of constant thickness.
//!
//! Planar and rectangular analytic faces retain their exact geometry.
//! Other sheets return `UnsupportedSurface` until exact offsets are available.

use super::geometry::Surface;
use super::topology::{Body, FaceKey};
use crate::geom2d::{Arc, Curve, Line};
use crate::space::{Plane, Vec3};
use std::f64::consts::{FRAC_PI_2, TAU};

/// Why a surface could not be thickened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThickenError {
    /// The requested distance was zero or not finite.
    InvalidDistance,
    /// The input has no bounded sheet geometry that can become a solid.
    UnsupportedSurface,
    /// The offset collapses or crosses the source surface.
    SelfIntersection,
}

/// Thickens every bounded face of an open sheet by a signed distance.
///
/// Positive distance follows each face's oriented normal. The source body is
/// borrowed and never changed, including when an offset fails.
pub fn thicken(body: &Body, distance: f64) -> Result<Body, ThickenError> {
    if !distance.is_finite() || distance.abs() <= f64::EPSILON {
        return Err(ThickenError::InvalidDistance);
    }
    if !body.validate().is_empty() || body.faces.keys().next().is_none() {
        return Err(ThickenError::UnsupportedSurface);
    }

    let faces = body.face_keys().collect::<Vec<_>>();
    if faces.len() == 1 {
        let face_key = faces[0];
        let face = body
            .faces
            .get(face_key)
            .ok_or(ThickenError::UnsupportedSurface)?;
        let surface = body
            .surfaces
            .get(face.surface)
            .ok_or(ThickenError::UnsupportedSurface)?;
        let oriented = distance * if face.forward { 1.0 } else { -1.0 };

        if matches!(surface, Surface::Plane(_)) {
            let profile = super::planar_face_profile(body, face_key)
                .ok_or(ThickenError::UnsupportedSurface)?;
            return super::extrude_region(
                profile.plane,
                &profile.loops,
                (Vec3::from(profile.outward) * distance).to_array(),
            )
            .ok_or(ThickenError::SelfIntersection);
        }

        if let Some(patch) = rectangular_patch(body, face_key, surface) {
            return thicken_revolved_patch(surface, patch, oriented);
        }
    }

    Err(ThickenError::UnsupportedSurface)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Patch {
    pub(crate) u_start: f64,
    pub(crate) u_sweep: f64,
    pub(crate) v_start: f64,
    pub(crate) v_sweep: f64,
}

pub(crate) fn rectangular_patch(
    body: &Body,
    face_key: FaceKey,
    surface: &Surface,
) -> Option<Patch> {
    if body.faces.get(face_key)?.loops.len() != 1 {
        return None;
    }
    let boundary = super::pcurve::face_boundary(body, face_key, 1e-8)?;
    let [Curve::Line(a), Curve::Line(b), Curve::Line(c), Curve::Line(d)] = boundary.as_slice()
    else {
        return None;
    };
    let lines = [a, b, c, d];
    let near = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]) <= 1e-8;
    let mut low = [f64::INFINITY; 2];
    let mut high = [f64::NEG_INFINITY; 2];
    for (index, line) in lines.iter().enumerate() {
        if !near(line.end, lines[(index + 1) % 4].start) {
            return None;
        }
        let delta = [line.end[0] - line.start[0], line.end[1] - line.start[1]];
        if (delta[0].abs() <= 1e-8) == (delta[1].abs() <= 1e-8) {
            return None;
        }
        for axis in 0..2 {
            if !line.start[axis].is_finite() {
                return None;
            }
            low[axis] = low[axis].min(line.start[axis]);
            high[axis] = high[axis].max(line.start[axis]);
        }
    }
    let sweep = [high[0] - low[0], high[1] - low[1]];
    // Four distinct corners are required; a retraced boundary encloses nothing.
    for i in 0..4 {
        for j in i + 1..4 {
            if near(lines[i].start, lines[j].start) {
                return None;
            }
        }
    }
    if sweep[0] > TAU + 1e-8 || (matches!(surface, Surface::Torus(_)) && sweep[1] > TAU + 1e-8) {
        return None;
    }
    Some(Patch {
        u_start: low[0],
        u_sweep: sweep[0],
        v_start: low[1],
        v_sweep: sweep[1],
    })
}

fn thicken_revolved_patch(
    surface: &Surface,
    patch: Patch,
    distance: f64,
) -> Result<Body, ThickenError> {
    let (frame, profile) = match surface {
        Surface::Cylinder(cylinder) => {
            let second = cylinder.radius + distance;
            positive_radii(&[cylinder.radius, second])?;
            let v0 = patch.v_start;
            let v1 = patch.v_start + patch.v_sweep;
            (
                cylinder.base,
                line_ring(&[
                    [cylinder.radius, v0],
                    [cylinder.radius, v1],
                    [second, v1],
                    [second, v0],
                ]),
            )
        }
        Surface::Cone(cone) => {
            let sine = cone.half_angle.sin();
            let cosine = cone.half_angle.cos();
            let tangent = cone.half_angle.tan();
            let v0 = patch.v_start;
            let v1 = patch.v_start + patch.v_sweep;
            let source0 = cone.radius - v0 * tangent;
            let source1 = cone.radius - v1 * tangent;
            let offset0 = source0 + distance * cosine;
            let offset1 = source1 + distance * cosine;
            positive_radii(&[source0, source1, offset0, offset1])?;
            (
                cone.base,
                line_ring(&[
                    [source0, v0],
                    [source1, v1],
                    [offset1, v1 + distance * sine],
                    [offset0, v0 + distance * sine],
                ]),
            )
        }
        Surface::Sphere(sphere) => {
            let second = sphere.radius + distance;
            positive_radii(&[sphere.radius, second])?;
            let v0 = patch.v_start;
            let v1 = patch.v_start + patch.v_sweep;
            if v0 < -FRAC_PI_2 - 1e-7 || v1 > FRAC_PI_2 + 1e-7 {
                return Err(ThickenError::UnsupportedSurface);
            }
            (
                sphere.frame,
                arc_ring([0.0, 0.0], sphere.radius, second, v0, v1),
            )
        }
        Surface::Torus(torus) => {
            let second = torus.minor_radius + distance;
            positive_radii(&[torus.minor_radius, second])?;
            if torus.minor_radius >= torus.major_radius - 1e-9
                || second >= torus.major_radius - 1e-9
            {
                return Err(ThickenError::SelfIntersection);
            }
            let v0 = patch.v_start;
            let v1 = patch.v_start + patch.v_sweep;
            (
                torus.frame,
                arc_ring(
                    [torus.major_radius, 0.0],
                    torus.minor_radius,
                    second,
                    v0,
                    v1,
                ),
            )
        }
        Surface::Plane(_) | Surface::Nurbs(_) => return Err(ThickenError::UnsupportedSurface),
    };

    let radial = frame.vector_at([patch.u_start.cos(), patch.u_start.sin()]);
    let axis = frame.normal().ok_or(ThickenError::UnsupportedSurface)?;
    let section = Plane::from_axes(frame.origin, radial, axis);
    super::revolve(section, &profile, frame.origin, axis, patch.u_sweep)
        .ok_or(ThickenError::SelfIntersection)
}

fn line_ring(points: &[[f64; 2]; 4]) -> Vec<Curve> {
    (0..4)
        .map(|index| {
            Curve::Line(Line {
                start: points[index],
                end: points[(index + 1) % 4],
            })
        })
        .collect()
}

fn arc_ring(
    centre: [f64; 2],
    first_radius: f64,
    second_radius: f64,
    start: f64,
    end: f64,
) -> Vec<Curve> {
    let at = |radius: f64, angle: f64| {
        [
            centre[0] + radius * angle.cos(),
            centre[1] + radius * angle.sin(),
        ]
    };
    vec![
        Curve::Arc(Arc {
            centre,
            radius: first_radius,
            start_angle: start,
            end_angle: end,
        }),
        Curve::Line(Line {
            start: at(first_radius, end),
            end: at(second_radius, end),
        }),
        Curve::Arc(Arc {
            centre,
            radius: second_radius,
            start_angle: start,
            end_angle: end,
        }),
        Curve::Line(Line {
            start: at(second_radius, start),
            end: at(first_radius, start),
        }),
    ]
}

fn positive_radii(values: &[f64]) -> Result<(), ThickenError> {
    if values
        .iter()
        .all(|value| value.is_finite() && *value > 1e-9)
    {
        Ok(())
    } else {
        Err(ThickenError::SelfIntersection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brep::{
        analytic_mass_properties, body_bounds, mesh_body, planar_region, revolve_surface,
    };
    use crate::geom2d::Circle;
    use std::f64::consts::PI;

    fn cylinder_sheet(angle: f64) -> Body {
        revolve_surface(
            Plane::from_axes([0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            &[Curve::Line(Line {
                start: [4.0, 0.0],
                end: [4.0, 3.0],
            })],
            [0.0; 3],
            [0.0, 0.0, 1.0],
            angle,
        )
        .unwrap()
    }

    #[test]
    fn planar_thickening_preserves_holes_and_signed_direction() {
        let sheet = planar_region(
            Plane::XY,
            &[
                vec![Curve::Circle(Circle {
                    centre: [0.0; 2],
                    radius: 4.0,
                })],
                vec![Curve::Circle(Circle {
                    centre: [0.0; 2],
                    radius: 2.0,
                })],
            ],
        )
        .unwrap();
        let before = format!("{sheet:?}");
        for distance in [-2.0, 2.0] {
            let solid = thicken(&sheet, distance).unwrap();
            assert!(solid.validate().is_empty());
            assert!(solid.edges.iter().all(|(_, edge)| edge.coedges.len() == 2));
            let bounds = body_bounds(&solid).unwrap();
            assert!((bounds.max[2] - bounds.min[2] - 2.0).abs() < 1e-9);
            let volume = mesh_body(&solid, 0.03, 1e-7).mass_properties().unwrap().0;
            assert!((volume - 24.0 * PI).abs() < 0.1);
        }
        assert_eq!(format!("{sheet:?}"), before);
    }

    #[test]
    fn cylinder_sectors_keep_the_entire_angular_span_and_exact_mass() {
        for angle in [PI * 0.5, PI * 1.5, PI * 1.99, TAU, -PI * 1.5] {
            let sheet = cylinder_sheet(angle);
            for distance in [-0.5, 0.5] {
                let face = sheet.faces.get(sheet.face_keys().next().unwrap()).unwrap();
                let second = 4.0 + distance * if face.forward { 1.0 } else { -1.0 };
                let solid = thicken(&sheet, distance).unwrap();
                assert!(solid.validate().is_empty());
                assert!(solid.edges.iter().all(|(_, edge)| edge.coedges.len() == 2));
                let properties = analytic_mass_properties(&solid).unwrap();
                let expected = angle.abs() * 0.5 * (second * second - 16.0_f64).abs() * 3.0;
                assert!(
                    (properties.volume - expected).abs() < 1e-8,
                    "{angle}: {} != {expected}",
                    properties.volume
                );
                assert!((properties.centroid[2] - 1.5).abs() < 1e-8);
                let volume = mesh_body(&solid, 0.02, 1e-7).mass_properties().unwrap().0;
                assert!((volume - expected).abs() < 0.02 * expected);
            }
        }
    }

    #[test]
    fn annular_mass_does_not_replace_a_closed_cylinder_shell() {
        let body = crate::brep::make::cylinder([0.0; 3], 4.0, 6.0).unwrap();
        let shell = crate::brep::shell(&body, &[], 0.5).unwrap();
        let properties = analytic_mass_properties(&shell).unwrap();
        let expected = PI * (16.0 * 6.0 - 3.5_f64.powi(2) * 5.0);
        assert!((properties.volume - expected).abs() < 1e-8);
    }

    #[test]
    fn invalid_and_unsupported_offsets_do_not_create_approximate_solids() {
        let sheet = cylinder_sheet(TAU);
        for distance in [0.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                thicken(&sheet, distance),
                Err(ThickenError::InvalidDistance)
            ));
        }
        let face = sheet.faces.get(sheet.face_keys().next().unwrap()).unwrap();
        let collapsing = if face.forward { -5.0 } else { 5.0 };
        assert!(matches!(
            thicken(&sheet, collapsing),
            Err(ThickenError::SelfIntersection)
        ));
        let solid = crate::brep::make::cuboid([0.0; 3], [1.0; 3]).unwrap();
        assert!(matches!(
            thicken(&solid, 0.2),
            Err(ThickenError::UnsupportedSurface)
        ));
    }
    #[test]
    fn curved_analytic_patches_produce_closed_exact_bodies() {
        let profiles = [
            Curve::Line(Line {
                start: [4.0, 0.0],
                end: [2.0, 3.0],
            }),
            Curve::Arc(Arc {
                centre: [0.0; 2],
                radius: 4.0,
                start_angle: -PI / 4.0,
                end_angle: PI / 4.0,
            }),
            Curve::Arc(Arc {
                centre: [6.0, 0.0],
                radius: 2.0,
                start_angle: -PI / 4.0,
                end_angle: PI / 4.0,
            }),
        ];
        for profile in profiles {
            let sheet = revolve_surface(
                Plane::from_axes([0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
                &[profile],
                [0.0; 3],
                [0.0, 0.0, 1.0],
                PI * 1.5,
            )
            .unwrap();
            for distance in [-0.2, 0.2] {
                let solid = thicken(&sheet, distance).unwrap();
                assert!(solid.validate().is_empty());
                assert!(solid.edges.iter().all(|(_, edge)| edge.coedges.len() == 2));
                assert!(solid
                    .surfaces
                    .iter()
                    .all(|(_, surface)| !matches!(surface, Surface::Nurbs(_))));
                assert!(mesh_body(&solid, 0.05, 1e-7).mass_properties().unwrap().0 > 0.0);
            }
        }
    }

    #[cfg(feature = "acis")]
    #[test]
    fn thickened_sector_round_trips_through_acis() {
        let solid = thicken(&cylinder_sheet(PI * 1.99), 0.5).unwrap();
        let expected = analytic_mass_properties(&solid).unwrap();
        let mut sat = cadcodec::entities::acis::types::SatDocument::default();
        crate::acis::append(&solid, &mut sat).unwrap();
        let (bodies, loss) = crate::acis::lift(&sat);
        assert_eq!(bodies.len(), 1, "{loss:?}");
        let restored = analytic_mass_properties(&bodies[0]).unwrap();
        assert!((restored.volume - expected.volume).abs() < 1e-8);
        assert!(Vec3::from(restored.centroid).distance(Vec3::from(expected.centroid)) < 1e-8);
    }
}
