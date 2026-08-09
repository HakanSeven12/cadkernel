//! Making a solid by moving a profile.
//!
//! The two constructions every modeller is built on. An extrusion drags a
//! closed outline along a straight line; a revolution turns it about an axis.
//! Between them they account for most of what a drawing's solids are, and
//! every primitive here is one of the two applied to the right outline.
//!
//! # The profile is a plane curve, and stays one
//!
//! The outline is given in a plane's own coordinates, as the chain of curves
//! [`geom2d`](crate::geom2d) already speaks in. That is what lets a side wall
//! know what it is: a straight run sweeps into a plane, an arc into a
//! cylinder, and anything else has no analytic surface to sweep into — so it
//! is refused rather than approximated into a spline nobody asked for.
//!
//! # Why the caps are not just the profile twice
//!
//! They face opposite ways. A cap built by copying the other and moving it
//! points the same way as the original, which leaves a solid with two lids
//! and no bottom — it passes every local check and encloses nothing.

use super::geometry::{Circle3, Curve3, Cylinder, Line3, Surface};
use super::topology::{
    Body, Coedge, CoedgeKey, Edge, EdgeKey, Face, FaceKey, Loop, Lump, Shell, ShellKey, Vertex,
    VertexKey,
};
use super::Provenance;
use crate::geom2d::Curve as Curve2;
use crate::space::{Plane, Vec3};

/// Extrudes a closed profile along `direction`.
///
/// The profile is a chain of curves in `plane`'s coordinates, each starting
/// where the last ended and the last closing onto the first. Straight and
/// circular pieces are supported; a spline profile has no analytic side wall
/// and is refused.
///
/// `None` for a direction lying in the plane — an extrusion of no thickness
/// is not a solid — and for a profile that does not close.
pub fn extrude(plane: Plane, profile: &[Curve2], direction: [f64; 3]) -> Option<Body> {
    let along = Vec3::from(direction);
    let normal = Vec3::from(plane.normal()?);
    // A direction across the plane sweeps the profile through itself.
    if along.dot(normal).abs() <= 1e-12 * along.length().max(1.0) {
        return None;
    }
    let corners = profile_corners(profile)?;

    let mut body = Body::new();
    let lump = body.lumps.insert(Lump {
        shells: Vec::new(),
        provenance: Provenance::Synthesized,
    });
    let shell = body.shells.insert(Shell {
        faces: Vec::new(),
        owner: lump,
        provenance: Provenance::Synthesized,
    });

    // Both rings of vertices, and the rails between them.
    let base: Vec<VertexKey> = corners
        .iter()
        .map(|uv| add_vertex(&mut body, plane.point_at(*uv)))
        .collect();
    let top: Vec<VertexKey> = corners
        .iter()
        .map(|uv| add_vertex(&mut body, (Vec3::from(plane.point_at(*uv)) + along).to_array()))
        .collect();

    let base_edges = profile_edges(&mut body, &plane, profile, &base)?;
    let far = Plane::from_axes(
        (Vec3::from(plane.origin) + along).to_array(),
        plane.x_axis,
        plane.y_axis,
    );
    let top_edges = profile_edges(&mut body, &far, profile, &top)?;
    let rails: Vec<EdgeKey> = (0..corners.len())
        .map(|index| add_line_edge(&mut body, base[index], top[index]))
        .collect::<Option<Vec<_>>>()?;

    // The two caps. The near one looks back along the sweep, so its loop runs
    // the profile backwards; the far one runs it forwards.
    add_cap(&mut body, shell, plane, &base_edges, false)?;
    add_cap(&mut body, shell, far, &top_edges, true)?;

    // One wall per profile piece: up the rail, along the top, down the next
    // rail, back along the bottom.
    for index in 0..profile.len() {
        let next = (index + 1) % profile.len();
        let surface = wall_surface(&mut body, &plane, &profile[index], along)?;
        add_wall(
            &mut body,
            shell,
            surface,
            [
                (base_edges[index], true),
                (rails[next], true),
                (top_edges[index], false),
                (rails[index], false),
            ],
        )?;
    }

    body.lumps.get_mut(lump)?.shells = vec![shell];
    body.roots = vec![lump];
    body.validate().is_empty().then_some(body)
}

/// Where each piece of a closed profile begins.
///
/// `None` when the chain does not close, or has fewer than three pieces —
/// neither describes an outline with an inside.
fn profile_corners(profile: &[Curve2]) -> Option<Vec<[f64; 2]>> {
    if profile.len() < 3 {
        return None;
    }
    let corners: Vec<[f64; 2]> = profile.iter().map(|piece| piece.point_at(0.0)).collect();
    for index in 0..profile.len() {
        let ends = profile[index].point_at(1.0);
        let begins = corners[(index + 1) % profile.len()];
        if (ends[0] - begins[0]).hypot(ends[1] - begins[1]) > 1e-9 {
            return None;
        }
    }
    Some(corners)
}

/// The edges of one copy of the profile, on `plane`.
fn profile_edges(
    body: &mut Body,
    plane: &Plane,
    profile: &[Curve2],
    corners: &[VertexKey],
) -> Option<Vec<EdgeKey>> {
    let mut out = Vec::with_capacity(profile.len());
    for (index, piece) in profile.iter().enumerate() {
        let next = (index + 1) % profile.len();
        let (curve, start, end) = lift_piece(plane, piece)?;
        let curve = body.curves.insert(curve);
        out.push(body.edges.insert(Edge {
            curve,
            start_parameter: start,
            end_parameter: end,
            start: corners[index],
            end: corners[next],
            coedges: Vec::new(),
            provenance: Provenance::Synthesized,
        }));
    }
    Some(out)
}

/// A profile piece as a space curve on the plane, with the parameters its
/// ends sit at.
fn lift_piece(plane: &Plane, piece: &Curve2) -> Option<(Curve3, f64, f64)> {
    match piece {
        Curve2::Line(line) => {
            let start = plane.point_at(line.start);
            let end = plane.point_at(line.end);
            Some((
                Curve3::Line(Line3 {
                    origin: start,
                    direction: (Vec3::from(end) - Vec3::from(start)).to_array(),
                }),
                0.0,
                1.0,
            ))
        }
        Curve2::Arc(arc) => {
            let centre = plane.point_at(arc.centre);
            let frame = Plane::from_axes(centre, plane.x_axis, plane.y_axis);
            let start = super::super::geom2d::normalize_angle(arc.start_angle);
            Some((
                Curve3::Circle(Circle3 {
                    plane: frame,
                    radius: arc.radius,
                }),
                start,
                start + arc.sweep(),
            ))
        }
        // A spline or an ellipse sweeps into a surface with no analytic form,
        // so the wall could only be approximated. Refusing says so.
        _ => None,
    }
}

/// The surface a profile piece sweeps into.
fn wall_surface(
    body: &mut Body,
    plane: &Plane,
    piece: &Curve2,
    along: Vec3,
) -> Option<super::topology::SurfaceKey> {
    let surface = match piece {
        Curve2::Line(line) => {
            let start = Vec3::from(plane.point_at(line.start));
            let end = Vec3::from(plane.point_at(line.end));
            // The wall's own frame: along the run, and up the sweep.
            let normal = (end - start).cross(along).normalize()?;
            Surface::Plane(Plane::orthonormal(
                start.to_array(),
                (end - start).to_array(),
                normal.to_array(),
            )?)
        }
        Curve2::Arc(arc) => {
            let centre = plane.point_at(arc.centre);
            Surface::Cylinder(Cylinder {
                base: Plane::orthonormal(centre, plane.x_axis, along.to_array())?,
                radius: arc.radius,
            })
        }
        _ => return None,
    };
    Some(body.surfaces.insert(surface))
}

fn add_vertex(body: &mut Body, point: [f64; 3]) -> VertexKey {
    body.vertices.insert(Vertex {
        point,
        provenance: Provenance::Synthesized,
    })
}

fn add_line_edge(body: &mut Body, from: VertexKey, to: VertexKey) -> Option<EdgeKey> {
    let start = body.vertices.get(from)?.point;
    let end = body.vertices.get(to)?.point;
    let curve = body.curves.insert(Curve3::Line(Line3 {
        origin: start,
        direction: (Vec3::from(end) - Vec3::from(start)).to_array(),
    }));
    Some(body.edges.insert(Edge {
        curve,
        start_parameter: 0.0,
        end_parameter: 1.0,
        start: from,
        end: to,
        coedges: Vec::new(),
        provenance: Provenance::Synthesized,
    }))
}

/// One end of the sweep: a planar face bounded by that copy of the profile.
fn add_cap(
    body: &mut Body,
    shell: ShellKey,
    plane: Plane,
    edges: &[EdgeKey],
    forward: bool,
) -> Option<FaceKey> {
    let surface = body.surfaces.insert(Surface::Plane(plane));
    let face = body.faces.insert(Face {
        surface,
        forward,
        loops: Vec::new(),
        owner: shell,
        provenance: Provenance::Synthesized,
    });
    let ring = body.loops.insert(Loop {
        coedges: Vec::new(),
        owner: face,
        provenance: Provenance::Synthesized,
    });
    // The near cap looks back along the sweep, so it runs the profile the
    // other way — which is also what makes each profile edge shared by a cap
    // and a wall running it opposite ways.
    let order: Vec<EdgeKey> = if forward {
        edges.to_vec()
    } else {
        edges.iter().rev().copied().collect()
    };
    let mut coedges = Vec::with_capacity(order.len());
    for edge in order {
        coedges.push(add_coedge(body, ring, edge, forward)?);
    }
    body.loops.get_mut(ring)?.coedges = coedges;
    body.faces.get_mut(face)?.loops = vec![ring];
    body.shells.get_mut(shell)?.faces.push(face);
    Some(face)
}

/// One side wall, from the four edges bounding it.
fn add_wall(
    body: &mut Body,
    shell: ShellKey,
    surface: super::topology::SurfaceKey,
    edges: [(EdgeKey, bool); 4],
) -> Option<FaceKey> {
    let face = body.faces.insert(Face {
        surface,
        forward: true,
        loops: Vec::new(),
        owner: shell,
        provenance: Provenance::Synthesized,
    });
    let ring = body.loops.insert(Loop {
        coedges: Vec::new(),
        owner: face,
        provenance: Provenance::Synthesized,
    });
    let mut coedges = Vec::with_capacity(4);
    for (edge, forward) in edges {
        coedges.push(add_coedge(body, ring, edge, forward)?);
    }
    body.loops.get_mut(ring)?.coedges = coedges;
    body.faces.get_mut(face)?.loops = vec![ring];
    body.shells.get_mut(shell)?.faces.push(face);
    Some(face)
}

fn add_coedge(
    body: &mut Body,
    ring: super::topology::LoopKey,
    edge: EdgeKey,
    forward: bool,
) -> Option<CoedgeKey> {
    let coedge = body.coedges.insert(Coedge {
        edge,
        forward,
        owner: ring,
        provenance: Provenance::Synthesized,
    });
    body.edges.get_mut(edge)?.coedges.push(coedge);
    Some(coedge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom2d::{Arc, Line};

    fn square(size: f64) -> Vec<Curve2> {
        let corners = [[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]];
        (0..4)
            .map(|index| {
                Curve2::Line(Line {
                    start: corners[index],
                    end: corners[(index + 1) % 4],
                })
            })
            .collect()
    }

    #[test]
    fn a_square_extruded_is_a_box() {
        let solid = extrude(Plane::XY, &square(10.0), [0.0, 0.0, 4.0]).expect("a prism");
        assert_eq!(solid.faces.len(), 6, "two caps and four walls");
        assert_eq!(solid.edges.len(), 12);
        assert_eq!(solid.vertices.len(), 8);
        assert_eq!(solid.euler_characteristic(), 2);
        let flaws = solid.validate();
        assert!(flaws.is_empty(), "{flaws:?}");
    }

    #[test]
    fn every_edge_of_an_extrusion_is_shared_the_two_ways() {
        let solid = extrude(Plane::XY, &square(5.0), [0.0, 0.0, 3.0]).unwrap();
        for (key, edge) in solid.edges.iter() {
            assert_eq!(edge.coedges.len(), 2, "edge {key:?}");
            let senses: Vec<bool> = edge
                .coedges
                .iter()
                .map(|c| solid.coedges.get(*c).unwrap().forward)
                .collect();
            assert_ne!(senses[0], senses[1], "edge {key:?} runs one way twice");
        }
    }

    #[test]
    fn an_extrusion_reaches_from_its_profile_to_the_far_end() {
        let solid = extrude(Plane::XY, &square(10.0), [0.0, 0.0, 4.0]).unwrap();
        let bounds = crate::brep::body_bounds(&solid).unwrap();
        assert_eq!(bounds.min, [0.0, 0.0, 0.0]);
        assert_eq!(bounds.max, [10.0, 10.0, 4.0]);
    }

    #[test]
    fn every_wall_faces_out() {
        let solid = extrude(Plane::XY, &square(10.0), [0.0, 0.0, 4.0]).unwrap();
        let mesh = crate::brep::mesh::body(&solid, 0.01, 1e-9);
        let centre = Vec3::new(5.0, 5.0, 2.0);
        for triangle in &mesh.triangles {
            let corner = Vec3::from(mesh.positions[triangle[0]]);
            let normal = Vec3::from(mesh.normals[triangle[0]]);
            assert!(
                normal.dot(corner - centre) > 0.0,
                "a face pointed inwards at {corner:?}"
            );
        }
    }

    #[test]
    fn a_profile_with_an_arc_in_it_sweeps_into_a_cylinder_wall() {
        // A square with one side bowed out, which is the shape a slot or a
        // rounded end is.
        let profile = vec![
            Curve2::Line(Line {
                start: [0.0, 0.0],
                end: [10.0, 0.0],
            }),
            Curve2::Arc(Arc {
                centre: [10.0, 5.0],
                radius: 5.0,
                start_angle: -std::f64::consts::FRAC_PI_2,
                end_angle: std::f64::consts::FRAC_PI_2,
            }),
            Curve2::Line(Line {
                start: [10.0, 10.0],
                end: [0.0, 10.0],
            }),
            Curve2::Line(Line {
                start: [0.0, 10.0],
                end: [0.0, 0.0],
            }),
        ];
        let solid = extrude(Plane::XY, &profile, [0.0, 0.0, 3.0]).expect("a bowed prism");
        assert!(solid.validate().is_empty());
        let cylinders = solid
            .surfaces
            .iter()
            .filter(|(_, s)| matches!(s, Surface::Cylinder(_)))
            .count();
        assert_eq!(cylinders, 1, "the bowed side became a cylinder");
        // And it reaches past where the straight sides do.
        let bounds = crate::brep::body_bounds(&solid).unwrap();
        assert!((bounds.max[0] - 15.0).abs() < 1e-9, "{bounds:?}");
    }

    #[test]
    fn an_extrusion_on_a_tilted_plane_goes_where_the_plane_points() {
        let plane = Plane::orthonormal([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]).unwrap();
        let solid = extrude(plane, &square(2.0), [0.0, 5.0, 0.0]).expect("a tilted prism");
        assert!(solid.validate().is_empty());
        let bounds = crate::brep::body_bounds(&solid).unwrap();
        assert!((bounds.max[1] - 5.0).abs() < 1e-9, "{bounds:?}");
    }

    #[test]
    fn a_sweep_along_the_profiles_own_plane_is_refused() {
        // It would drag the outline through itself and enclose nothing.
        assert!(extrude(Plane::XY, &square(10.0), [1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn a_profile_that_does_not_close_is_refused() {
        let open = vec![
            Curve2::Line(Line {
                start: [0.0, 0.0],
                end: [10.0, 0.0],
            }),
            Curve2::Line(Line {
                start: [10.0, 0.0],
                end: [10.0, 10.0],
            }),
            Curve2::Line(Line {
                start: [10.0, 10.0],
                end: [5.0, 20.0],
            }),
        ];
        assert!(extrude(Plane::XY, &open, [0.0, 0.0, 1.0]).is_none());
        assert!(extrude(Plane::XY, &square(1.0)[..2], [0.0, 0.0, 1.0]).is_none());
    }

    #[test]
    fn a_spline_profile_is_refused_rather_than_approximated() {
        let profile = vec![
            Curve2::Nurbs(
                crate::geom2d::NurbsCurve::new(
                    2,
                    vec![[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]],
                    Vec::new(),
                    None,
                )
                .unwrap(),
            ),
            Curve2::Line(Line {
                start: [10.0, 0.0],
                end: [0.0, 0.0],
            }),
        ];
        assert!(extrude(Plane::XY, &profile, [0.0, 0.0, 1.0]).is_none());
    }

    #[test]
    fn an_extrusion_at_survey_coordinates_is_the_same_solid() {
        let origin = [512_345.678, 4_512_345.678, 91.5];
        let plane = Plane::from_axes(origin, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let solid = extrude(plane, &square(0.5), [0.0, 0.0, 0.5]).expect("a small prism");
        assert!(solid.validate().is_empty());
        assert_eq!(solid.euler_characteristic(), 2);
        assert!(solid.worst_vertex_gap() < 1e-6);
    }
}
