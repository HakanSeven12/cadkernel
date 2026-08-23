//! Constant edge chamfers and fillets on convex planar solids.

use super::geometry::{Circle3, Curve3, Cylinder, Line3, Surface};
use super::topology::{Body, Coedge, Edge, EdgeKey, Face, FaceKey, Loop, Lump, Shell, Vertex};
use super::Provenance;
use crate::space::{Plane, Vec3};
use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

#[derive(Clone, Copy)]
struct Halfspace {
    origin: Vec3,
    normal: Vec3,
    offset: f64,
    added: bool,
}

struct EdgeFrame {
    point: Vec3,
    axis: Vec3,
    first_normal: Vec3,
    second_normal: Vec3,
    first_inward: Vec3,
    second_inward: Vec3,
    halfspaces: Vec<Halfspace>,
}

#[derive(Clone)]
struct ExistingFillet {
    cut: Halfspace,
    cylinder: Cylinder,
    forward: bool,
}

/// Cuts a symmetric chamfer at one straight edge.
pub fn chamfer(body: &Body, edge: EdgeKey, distance: f64) -> Option<Body> {
    if !distance.is_finite() || distance <= 0.0 {
        return None;
    }
    let existing = existing_fillets(body)?;
    let frame = edge_frame(body, edge)?;
    let (_, mut result, _) = cut(&frame, distance)?;
    restore_fillets(&mut result, &existing)?;
    Some(result)
}

/// Rounds one straight edge with a constant-radius cylindrical face.
pub fn fillet(body: &Body, edge: EdgeKey, radius: f64) -> Option<Body> {
    if !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    let existing = existing_fillets(body)?;
    let frame = edge_frame(body, edge)?;
    let normals_angle = frame
        .first_normal
        .dot(frame.second_normal)
        .clamp(-1.0, 1.0)
        .acos();
    let interior = PI - normals_angle;
    let setback = radius / (interior * 0.5).tan();
    let (cut_plane, mut result, cut_face) = cut(&frame, setback)?;

    let tangent = frame.point + frame.first_inward * setback;
    let centre = tangent - frame.first_normal * radius;
    let cylinder_plane = Plane::orthonormal(
        centre.to_array(),
        frame.first_normal.to_array(),
        frame.axis.to_array(),
    )?;
    let cylinder = Cylinder {
        base: cylinder_plane,
        radius,
    };
    restore_fillets(&mut result, &existing)?;
    if round_face(&mut result, cut_face, &cylinder, true)? != 2
        || cut_plane.normal.dot(frame.first_normal + frame.second_normal) <= 0.0
    {
        return None;
    }
    if !result.validate().is_empty() || result.worst_vertex_gap() > tolerance_body(&result) {
        return None;
    }
    Some(result)
}

/// Moves one planar face of a convex planar solid along its outward normal.
pub fn presspull(body: &Body, face_key: FaceKey, distance: f64) -> Option<Body> {
    if !distance.is_finite() || distance.abs() <= f64::EPSILON {
        return None;
    }
    let mut selected = false;
    let mut halfspaces = Vec::new();
    for (key, face) in body.faces.iter() {
        let Surface::Plane(plane) = body.surfaces.get(face.surface)? else {
            return None;
        };
        let mut normal = Vec3::from(plane.normal()?);
        if !face.forward {
            normal = -normal;
        }
        let moved = key == face_key;
        selected |= moved;
        let origin = Vec3::from(plane.origin) + normal * if moved { distance } else { 0.0 };
        halfspaces.push(Halfspace {
            origin,
            normal,
            offset: normal.dot(origin),
            added: moved,
        });
    }
    if !selected {
        return None;
    }
    let (result, _) = convex_body(&halfspaces)?;
    Some(result)
}

fn circle_parameters(plane: &Plane, start: Vec3, end: Vec3) -> Option<(f64, f64)> {
    let parameter = |point: Vec3| {
        let local = plane.project(point.to_array())?;
        Some(local[1].atan2(local[0]))
    };
    let start = parameter(start)?;
    let span = (parameter(end)? - start).rem_euclid(TAU);
    (span > 1e-12).then_some((start, start + span))
}

fn cut(frame: &EdgeFrame, setback: f64) -> Option<(Halfspace, Body, FaceKey)> {
    if !setback.is_finite() || setback <= 0.0 {
        return None;
    }
    let first = frame.point + frame.first_inward * setback;
    let second = frame.point + frame.second_inward * setback;
    let normal = (frame.first_normal + frame.second_normal).normalize()?;
    let offset = 0.5 * (normal.dot(first) + normal.dot(second));
    let added = Halfspace {
        origin: first,
        normal,
        offset,
        added: true,
    };
    if normal.dot(frame.point) <= offset + tolerance(&[frame.point, first, second]) {
        return None;
    }
    let mut halfspaces = frame.halfspaces.clone();
    halfspaces.push(added);
    let (result, face) = convex_body(&halfspaces)?;
    Some((added, result, face))
}

fn edge_frame(body: &Body, edge_key: EdgeKey) -> Option<EdgeFrame> {
    let edge = body.edges.get(edge_key)?;
    if !matches!(body.curves.get(edge.curve)?, Curve3::Line(_)) || edge.coedges.len() != 2 {
        return None;
    }
    let start = Vec3::from(body.vertices.get(edge.start)?.point);
    let end = Vec3::from(body.vertices.get(edge.end)?.point);
    let axis = (end - start).normalize()?;
    let face_of = |coedge| {
        let loop_key = body.coedges.get(coedge)?.owner;
        Some(body.loops.get(loop_key)?.owner)
    };
    let faces = [face_of(edge.coedges[0])?, face_of(edge.coedges[1])?];
    let plane_of = |face_key| {
        let face = body.faces.get(face_key)?;
        let Surface::Plane(plane) = body.surfaces.get(face.surface)? else {
            return None;
        };
        let mut normal = Vec3::from(plane.normal()?);
        if !face.forward {
            normal = -normal;
        }
        Some(normal)
    };
    let first_normal = plane_of(faces[0])?;
    let second_normal = plane_of(faces[1])?;
    let dot = first_normal.dot(second_normal);
    if dot.abs() > 1.0 - 1e-9 {
        return None;
    }
    let first_inward = -(second_normal - first_normal * dot).normalize()?;
    let second_inward = -(first_normal - second_normal * dot).normalize()?;
    let mut halfspaces = Vec::new();
    for (_, face) in body.faces.iter() {
        let Some(Surface::Plane(plane)) = body.surfaces.get(face.surface) else {
            continue;
        };
        let mut normal = Vec3::from(plane.normal()?);
        if !face.forward {
            normal = -normal;
        }
        halfspaces.push(Halfspace {
            origin: Vec3::from(plane.origin),
            normal,
            offset: normal.dot(Vec3::from(plane.origin)),
            added: false,
        });
    }
    halfspaces.extend(existing_fillets(body)?.into_iter().map(|fillet| fillet.cut));
    Some(EdgeFrame {
        point: start,
        axis,
        first_normal,
        second_normal,
        first_inward,
        second_inward,
        halfspaces,
    })
}

fn existing_fillets(body: &Body) -> Option<Vec<ExistingFillet>> {
    let mut found = Vec::new();
    for (face_key, face) in body.faces.iter() {
        let Surface::Cylinder(cylinder) = body.surfaces.get(face.surface)? else {
            continue;
        };
        let axis = Vec3::from(cylinder.base.normal()?);
        let mut side_normals = Vec::new();
        let mut points = Vec::new();
        let mut line_edges = 0usize;
        for coedge_key in body.face_coedges(face_key) {
            let coedge = body.coedges.get(coedge_key)?;
            let edge = body.edges.get(coedge.edge)?;
            points.push(Vec3::from(body.vertices.get(edge.start)?.point));
            if !matches!(body.curves.get(edge.curve)?, Curve3::Line(_)) {
                continue;
            }
            line_edges += 1;
            let other_face = edge.coedges.iter().find_map(|candidate| {
                let candidate = body.coedges.get(*candidate)?;
                let owner = body.loops.get(candidate.owner)?.owner;
                (owner != face_key).then_some(owner)
            })?;
            let other = body.faces.get(other_face)?;
            let Surface::Plane(plane) = body.surfaces.get(other.surface)? else {
                return None;
            };
            let mut normal = Vec3::from(plane.normal()?);
            if !other.forward {
                normal = -normal;
            }
            if normal.dot(axis).abs() <= 1e-8 {
                side_normals.push(normal);
            }
        }
        if line_edges != 2 || side_normals.len() != 2 || points.is_empty() {
            continue;
        }
        let normal = (side_normals[0] + side_normals[1]).normalize()?;
        let offset = points.iter().map(|point| normal.dot(*point)).sum::<f64>()
            / points.len() as f64;
        found.push(ExistingFillet {
            cut: Halfspace {
                origin: points[0],
                normal,
                offset,
                added: true,
            },
            cylinder: cylinder.clone(),
            forward: face.forward,
        });
    }
    Some(found)
}

fn restore_fillets(body: &mut Body, fillets: &[ExistingFillet]) -> Option<()> {
    for fillet in fillets {
        let face = face_on(body, fillet.cut)?;
        if round_face(body, face, &fillet.cylinder, fillet.forward)? != 2 {
            return None;
        }
    }
    Some(())
}

fn face_on(body: &Body, halfspace: Halfspace) -> Option<FaceKey> {
    body.faces.iter().find_map(|(key, face)| {
        let Surface::Plane(plane) = body.surfaces.get(face.surface)? else {
            return None;
        };
        let mut normal = Vec3::from(plane.normal()?);
        if !face.forward {
            normal = -normal;
        }
        let scale = halfspace.offset.abs().max(1.0);
        (normal.dot(halfspace.normal) > 1.0 - 1e-8
            && (normal.dot(Vec3::from(plane.origin)) - halfspace.offset).abs() <= scale * 1e-8)
            .then_some(key)
    })
}

fn round_face(
    body: &mut Body,
    face: FaceKey,
    cylinder: &Cylinder,
    forward: bool,
) -> Option<usize> {
    let axis = Vec3::from(cylinder.base.normal()?);
    let centre = Vec3::from(cylinder.base.origin);
    let surface = body.faces.get(face)?.surface;
    *body.surfaces.get_mut(surface)? = Surface::Cylinder(cylinder.clone());
    body.faces.get_mut(face)?.forward = forward;
    let loop_key = *body.faces.get(face)?.loops.first()?;
    let edges = body
        .loops
        .get(loop_key)?
        .coedges
        .iter()
        .filter_map(|coedge| body.coedges.get(*coedge).map(|coedge| coedge.edge))
        .collect::<Vec<_>>();
    let mut rounded = 0usize;
    for edge_key in edges {
        let edge = body.edges.get(edge_key)?.clone();
        let start = Vec3::from(body.vertices.get(edge.start)?.point);
        let end = Vec3::from(body.vertices.get(edge.end)?.point);
        if (end - start).normalize()?.dot(axis).abs() > 1.0 - 1e-8 {
            continue;
        }
        let start_height = (start - centre).dot(axis);
        let end_height = (end - centre).dot(axis);
        if (start_height - end_height).abs() > tolerance(&[start, end, centre]) {
            return None;
        }
        let cross_centre = centre + axis * ((start_height + end_height) * 0.5);
        let mut circle_plane = Plane::orthonormal(
            cross_centre.to_array(),
            cylinder.base.x_axis,
            axis.to_array(),
        )?;
        let mut parameters = circle_parameters(&circle_plane, start, end)?;
        if parameters.1 - parameters.0 > PI + 1e-9 {
            circle_plane = Plane::orthonormal(
                cross_centre.to_array(),
                cylinder.base.x_axis,
                (-axis).to_array(),
            )?;
            parameters = circle_parameters(&circle_plane, start, end)?;
        }
        *body.curves.get_mut(edge.curve)? = Curve3::Circle(Circle3 {
            plane: circle_plane,
            radius: cylinder.radius,
        });
        let edge = body.edges.get_mut(edge_key)?;
        edge.start_parameter = parameters.0;
        edge.end_parameter = parameters.1;
        rounded += 1;
    }
    Some(rounded)
}

fn convex_body(halfspaces: &[Halfspace]) -> Option<(Body, FaceKey)> {
    let scale = halfspaces
        .iter()
        .map(|plane| plane.origin.length().max(plane.offset.abs()))
        .fold(1.0_f64, f64::max);
    let tol = scale * 1e-9;
    let mut points = Vec::<Vec3>::new();
    for first in 0..halfspaces.len() {
        for second in first + 1..halfspaces.len() {
            for third in second + 1..halfspaces.len() {
                let Some(point) = intersection(
                    halfspaces[first],
                    halfspaces[second],
                    halfspaces[third],
                ) else {
                    continue;
                };
                if halfspaces
                    .iter()
                    .any(|plane| plane.normal.dot(point) > plane.offset + tol)
                {
                    continue;
                }
                if points.iter().all(|other| other.distance(point) > tol) {
                    points.push(point);
                }
            }
        }
    }
    if points.len() < 4 {
        return None;
    }

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
    let vertices = points
        .iter()
        .map(|point| {
            body.vertices.insert(Vertex {
                point: point.to_array(),
                provenance: Provenance::Synthesized,
            })
        })
        .collect::<Vec<_>>();
    let mut edge_map = HashMap::<(usize, usize), EdgeKey>::new();
    let mut added_face = None;
    for halfspace in halfspaces {
        let mut indices = points
            .iter()
            .enumerate()
            .filter_map(|(index, point)| {
                ((halfspace.normal.dot(*point) - halfspace.offset).abs() <= tol)
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        if indices.len() < 3 {
            continue;
        }
        let centre = indices
            .iter()
            .fold(Vec3::ZERO, |sum, index| sum + points[*index])
            / indices.len() as f64;
        let seed = if halfspace.normal.x.abs() < 0.8 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let x = halfspace.normal.cross(seed).normalize()?;
        let y = halfspace.normal.cross(x);
        indices.sort_by(|a, b| {
            let angle = |index: usize| {
                let delta = points[index] - centre;
                delta.dot(y).atan2(delta.dot(x))
            };
            angle(*a).total_cmp(&angle(*b))
        });
        remove_collinear(&mut indices, &points, tol);
        if indices.len() < 3 {
            continue;
        }
        let plane = Plane::orthonormal(
            halfspace.origin.to_array(),
            x.to_array(),
            halfspace.normal.to_array(),
        )?;
        let surface = body.surfaces.insert(Surface::Plane(plane));
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
        let mut coedges = Vec::with_capacity(indices.len());
        for position in 0..indices.len() {
            let from = indices[position];
            let to = indices[(position + 1) % indices.len()];
            let key = (from.min(to), from.max(to));
            let edge = if let Some(edge) = edge_map.get(&key).copied() {
                edge
            } else {
                let direction = points[key.1] - points[key.0];
                let curve = body.curves.insert(Curve3::Line(Line3 {
                    origin: points[key.0].to_array(),
                    direction: direction.to_array(),
                }));
                let edge = body.edges.insert(Edge {
                    curve,
                    start_parameter: 0.0,
                    end_parameter: 1.0,
                    start: vertices[key.0],
                    end: vertices[key.1],
                    coedges: Vec::new(),
                    provenance: Provenance::Synthesized,
                });
                edge_map.insert(key, edge);
                edge
            };
            let coedge = body.coedges.insert(Coedge {
                edge,
                forward: from == key.0,
                pcurve: None,
                owner: ring,
                provenance: Provenance::Synthesized,
            });
            body.edges.get_mut(edge)?.coedges.push(coedge);
            coedges.push(coedge);
        }
        body.loops.get_mut(ring)?.coedges = coedges;
        body.faces.get_mut(face)?.loops = vec![ring];
        body.shells.get_mut(shell)?.faces.push(face);
        if halfspace.added {
            added_face = Some(face);
        }
    }
    body.lumps.get_mut(lump)?.shells = vec![shell];
    body.roots = vec![lump];
    let added_face = added_face?;
    body.validate().is_empty().then_some((body, added_face))
}

fn intersection(first: Halfspace, second: Halfspace, third: Halfspace) -> Option<Vec3> {
    let denominator = first.normal.dot(second.normal.cross(third.normal));
    if denominator.abs() <= 1e-12 {
        return None;
    }
    Some(
        (second.normal.cross(third.normal) * first.offset
            + third.normal.cross(first.normal) * second.offset
            + first.normal.cross(second.normal) * third.offset)
            / denominator,
    )
}

fn remove_collinear(indices: &mut Vec<usize>, points: &[Vec3], tolerance: f64) {
    loop {
        let count = indices.len();
        let remove = (0..count).find(|index| {
            let before = points[indices[(index + count - 1) % count]];
            let here = points[indices[*index]];
            let after = points[indices[(index + 1) % count]];
            let first = here - before;
            let second = after - here;
            first.cross(second).length() <= tolerance * first.length().max(second.length())
        });
        let Some(index) = remove else {
            break;
        };
        indices.remove(index);
        if indices.len() < 3 {
            break;
        }
    }
}

fn tolerance(points: &[Vec3]) -> f64 {
    points
        .iter()
        .map(|point| point.length())
        .fold(1.0_f64, f64::max)
        * 1e-8
}

fn tolerance_body(body: &Body) -> f64 {
    body.vertices
        .iter()
        .map(|(_, vertex)| Vec3::from(vertex.point).length())
        .fold(1.0_f64, f64::max)
        * 1e-8
}
