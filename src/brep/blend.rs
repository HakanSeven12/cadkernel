//! Constant edge chamfers and fillets on convex planar solids.

use super::bounds::operation_tolerance;
use super::geometry::{Circle3, Curve3, Cylinder, Line3, Surface};
use super::topology::{Body, Coedge, Edge, EdgeKey, Face, FaceKey, Loop, Lump, Shell, Vertex};
use super::Provenance;
use crate::space::{Plane, Vec3};
use std::collections::{HashMap, HashSet};
use std::f64::consts::{PI, TAU};
use std::fmt;

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

/// Why one or more selected edges could not be filleted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilletError {
    /// No edges were selected.
    EmptySelection,
    /// Radius was zero, negative, NaN or infinite.
    InvalidRadius,
    /// At least one selected key does not name an edge in this body.
    UnknownEdge,
    /// The input body's ownership or adjacency is inconsistent.
    InvalidBodyTopology,
    /// This operation currently accepts one lump, one shell and no face holes.
    UnsupportedBodyTopology,
    /// A selected edge is not straight.
    UnsupportedEdgeCurve(EdgeKey),
    /// A selected edge is not shared by exactly two faces.
    NonManifoldEdge(EdgeKey),
    /// A selected edge is not bounded by two planar faces.
    UnsupportedAdjacentSurface(EdgeKey),
    /// An edge or one of its adjacent surface frames is degenerate.
    DegenerateGeometry(EdgeKey),
    /// Adjacent selected edges need a corner-blend solver.
    AdjacentSelections(EdgeKey, EdgeKey),
    /// The body contains a surface this convex planar operation cannot preserve.
    UnsupportedBodySurface,
    /// The planar body is not the intersection of its face halfspaces.
    NonConvexBody,
    /// A previously created fillet does not have the supported cylindrical form.
    UnsupportedExistingFillet,
    /// Radius removes the body or makes selected blend regions meet.
    RadiusTooLargeOrInteracting,
    /// An end of the selected edge needs a non-circular cylinder section.
    UnsupportedEndCondition(EdgeKey),
    /// Constructed topology or edge geometry did not validate.
    InvalidResult,
}

impl fmt::Display for FilletError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySelection => formatter.write_str("no edges were selected"),
            Self::InvalidRadius => formatter.write_str("fillet radius must be finite and positive"),
            Self::UnknownEdge => formatter.write_str("a selected edge does not belong to the body"),
            Self::InvalidBodyTopology => formatter.write_str("the input body has invalid topology"),
            Self::UnsupportedBodyTopology => formatter.write_str(
                "filleting currently requires one lump, one shell and faces without holes",
            ),
            Self::UnsupportedEdgeCurve(edge) => {
                write!(formatter, "selected edge {edge:?} is not straight")
            }
            Self::NonManifoldEdge(edge) => write!(
                formatter,
                "selected edge {edge:?} is not shared by exactly two faces"
            ),
            Self::UnsupportedAdjacentSurface(edge) => write!(
                formatter,
                "selected edge {edge:?} is not bounded by two planar faces"
            ),
            Self::DegenerateGeometry(edge) => {
                write!(formatter, "selected edge {edge:?} has degenerate geometry")
            }
            Self::AdjacentSelections(first, second) => write!(
                formatter,
                "selected edges {first:?} and {second:?} meet; corner blends are unsupported"
            ),
            Self::UnsupportedBodySurface => formatter.write_str(
                "the body contains a surface this convex planar fillet cannot preserve",
            ),
            Self::NonConvexBody => {
                formatter.write_str("the body is not a convex intersection of planar halfspaces")
            }
            Self::UnsupportedExistingFillet => formatter.write_str(
                "an existing cylindrical face is not a supported constant edge fillet",
            ),
            Self::RadiusTooLargeOrInteracting => formatter.write_str(
                "the radius removes the body or makes selected fillet regions interact",
            ),
            Self::UnsupportedEndCondition(edge) => write!(
                formatter,
                "selected edge {edge:?} needs an unsupported fillet end condition"
            ),
            Self::InvalidResult => formatter.write_str("the filleted body did not validate"),
        }
    }
}

impl std::error::Error for FilletError {}

/// Cuts a symmetric chamfer at one straight edge.
pub fn chamfer(body: &Body, edge: EdgeKey, distance: f64) -> Option<Body> {
    if !distance.is_finite() || distance <= 0.0 {
        return None;
    }
    let existing = existing_fillets(body)?;
    let frame = edge_frame(body, edge)?;
    let tolerance = operation_tolerance(&[body]);
    let (_, mut result, _) = cut(&frame, distance, tolerance)?;
    restore_fillets(&mut result, &existing, tolerance)?;
    Some(result)
}

/// Rounds one straight edge with a constant-radius cylindrical face.
pub fn fillet(body: &Body, edge: EdgeKey, radius: f64) -> Option<Body> {
    fillet_edges(body, &[edge], radius).ok()
}

/// Rounds several independent straight edges in one atomic operation.
///
/// Selection order and duplicate keys do not affect the result. Edges sharing
/// a vertex are refused because joining their cylindrical faces needs a corner
/// blend surface; returning an error is safer than emitting intersecting faces.
pub fn fillet_edges(
    body: &Body,
    selected: &[EdgeKey],
    radius: f64,
) -> Result<Body, FilletError> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(FilletError::InvalidRadius);
    }
    if selected.is_empty() {
        return Err(FilletError::EmptySelection);
    }
    if !body.validate().is_empty() {
        return Err(FilletError::InvalidBodyTopology);
    }
    if selected.iter().any(|edge| !body.edges.contains(*edge)) {
        return Err(FilletError::UnknownEdge);
    }

    let mut selected = selected.to_vec();
    selected.sort_by_key(EdgeKey::slot);
    selected.dedup();
    reject_adjacent_selections(body, &selected)?;

    let tolerance = operation_tolerance(&[body]);
    let existing = existing_fillets(body).ok_or(FilletError::UnsupportedExistingFillet)?;
    validate_supported_body(body, &existing, tolerance)?;

    let mut frames = Vec::with_capacity(selected.len());
    for edge in &selected {
        validate_selected_edge(body, *edge)?;
        let frame = edge_frame(body, *edge).ok_or(FilletError::DegenerateGeometry(*edge))?;
        frames.push((*edge, frame));
    }
    reject_interacting_regions(body, &selected, radius, tolerance)?;

    let mut halfspaces = frames[0].1.halfspaces.clone();
    let mut blends = Vec::with_capacity(frames.len());
    for (edge, frame) in &frames {
        let (cut, cylinder) = fillet_geometry(frame, radius, tolerance)
            .ok_or(FilletError::DegenerateGeometry(*edge))?;
        halfspaces.push(cut);
        blends.push((*edge, cut, cylinder));
    }
    let (mut result, _) = convex_body(&halfspaces, tolerance)
        .ok_or(FilletError::RadiusTooLargeOrInteracting)?;

    let mut occupied = HashSet::new();
    let mut existing_faces = Vec::with_capacity(existing.len());
    for old in &existing {
        let face = face_on(&result, old.cut, tolerance)
            .ok_or(FilletError::RadiusTooLargeOrInteracting)?;
        if !occupied.insert(face)
            || existing_faces
                .iter()
                .any(|existing| faces_touch(&result, *existing, face))
        {
            return Err(FilletError::RadiusTooLargeOrInteracting);
        }
        existing_faces.push(face);
    }
    let mut targets = Vec::with_capacity(blends.len());
    for (edge, cut, cylinder) in blends {
        let face = face_on(&result, cut, tolerance)
            .ok_or(FilletError::RadiusTooLargeOrInteracting)?;
        if !occupied.insert(face) {
            return Err(FilletError::RadiusTooLargeOrInteracting);
        }
        if existing_faces
            .iter()
            .any(|existing| faces_touch(&result, *existing, face))
        {
            return Err(FilletError::RadiusTooLargeOrInteracting);
        }
        targets.push((edge, face, cylinder));
    }
    for first in 0..targets.len() {
        for second in first + 1..targets.len() {
            if faces_touch(&result, targets[first].1, targets[second].1) {
                return Err(FilletError::RadiusTooLargeOrInteracting);
            }
        }
    }

    restore_fillets(&mut result, &existing, tolerance)
        .ok_or(FilletError::UnsupportedExistingFillet)?;
    for (edge, face, cylinder) in targets {
        if round_face(&mut result, face, &cylinder, true, tolerance)
            .filter(|rounded| *rounded == 2)
            .is_none()
        {
            return Err(FilletError::UnsupportedEndCondition(edge));
        }
    }
    if !result.validate().is_empty() || result.worst_vertex_gap() > tolerance {
        return Err(FilletError::InvalidResult);
    }
    Ok(result)
}

/// Moves a planar face while extending or trimming its neighbouring surfaces.
pub fn presspull(body: &Body, face_key: FaceKey, distance: f64) -> Option<Body> {
    super::presspull_face(body, face_key, distance, super::PresspullMode::Offset)
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

fn cut(
    frame: &EdgeFrame,
    setback: f64,
    tolerance: f64,
) -> Option<(Halfspace, Body, FaceKey)> {
    let added = cut_halfspace(frame, setback, tolerance)?;
    let mut halfspaces = frame.halfspaces.clone();
    halfspaces.push(added);
    let (result, face) = convex_body(&halfspaces, tolerance)?;
    Some((added, result, face))
}

fn cut_halfspace(frame: &EdgeFrame, setback: f64, tolerance: f64) -> Option<Halfspace> {
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
    if normal.dot(frame.point) <= offset + tolerance {
        return None;
    }
    Some(added)
}

fn fillet_geometry(
    frame: &EdgeFrame,
    radius: f64,
    tolerance: f64,
) -> Option<(Halfspace, Cylinder)> {
    let normals_angle = frame
        .first_normal
        .dot(frame.second_normal)
        .clamp(-1.0, 1.0)
        .acos();
    let interior = PI - normals_angle;
    let setback = radius / (interior * 0.5).tan();
    let cut = cut_halfspace(frame, setback, tolerance)?;
    if cut.normal.dot(frame.first_normal + frame.second_normal) <= 0.0 {
        return None;
    }
    let tangent = frame.point + frame.first_inward * setback;
    let centre = tangent - frame.first_normal * radius;
    let base = Plane::orthonormal(
        centre.to_array(),
        frame.first_normal.to_array(),
        frame.axis.to_array(),
    )?;
    Some((cut, Cylinder { base, radius }))
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

fn validate_selected_edge(body: &Body, edge_key: EdgeKey) -> Result<(), FilletError> {
    let edge = body.edges.get(edge_key).ok_or(FilletError::UnknownEdge)?;
    if !matches!(body.curves.get(edge.curve), Some(Curve3::Line(_))) {
        return Err(FilletError::UnsupportedEdgeCurve(edge_key));
    }
    if edge.coedges.len() != 2 {
        return Err(FilletError::NonManifoldEdge(edge_key));
    }
    let start = body
        .vertices
        .get(edge.start)
        .ok_or(FilletError::InvalidBodyTopology)?;
    let end = body
        .vertices
        .get(edge.end)
        .ok_or(FilletError::InvalidBodyTopology)?;
    if (Vec3::from(end.point) - Vec3::from(start.point))
        .normalize()
        .is_none()
    {
        return Err(FilletError::DegenerateGeometry(edge_key));
    }
    for coedge in &edge.coedges {
        let coedge = body
            .coedges
            .get(*coedge)
            .ok_or(FilletError::InvalidBodyTopology)?;
        let ring = body
            .loops
            .get(coedge.owner)
            .ok_or(FilletError::InvalidBodyTopology)?;
        let face = body
            .faces
            .get(ring.owner)
            .ok_or(FilletError::InvalidBodyTopology)?;
        let Some(Surface::Plane(plane)) = body.surfaces.get(face.surface) else {
            return Err(FilletError::UnsupportedAdjacentSurface(edge_key));
        };
        if plane.normal().is_none() {
            return Err(FilletError::DegenerateGeometry(edge_key));
        }
    }
    Ok(())
}

fn reject_adjacent_selections(body: &Body, selected: &[EdgeKey]) -> Result<(), FilletError> {
    let mut owner = HashMap::new();
    for edge_key in selected {
        let edge = body.edges.get(*edge_key).ok_or(FilletError::UnknownEdge)?;
        for vertex in [edge.start, edge.end] {
            if let Some(first) = owner.insert(vertex, *edge_key) {
                return Err(FilletError::AdjacentSelections(first, *edge_key));
            }
        }
    }
    Ok(())
}

fn reject_interacting_regions(
    body: &Body,
    selected: &[EdgeKey],
    radius: f64,
    tolerance: f64,
) -> Result<(), FilletError> {
    let segments = selected
        .iter()
        .map(|edge_key| {
            let edge = body.edges.get(*edge_key).ok_or(FilletError::UnknownEdge)?;
            let start = body
                .vertices
                .get(edge.start)
                .ok_or(FilletError::InvalidBodyTopology)?;
            let end = body
                .vertices
                .get(edge.end)
                .ok_or(FilletError::InvalidBodyTopology)?;
            Ok((Vec3::from(start.point), Vec3::from(end.point)))
        })
        .collect::<Result<Vec<_>, FilletError>>()?;
    for first in 0..segments.len() {
        for second in first + 1..segments.len() {
            if segment_distance(segments[first], segments[second])
                <= radius * 2.0 + tolerance
            {
                return Err(FilletError::RadiusTooLargeOrInteracting);
            }
        }
    }
    Ok(())
}

fn segment_distance(first: (Vec3, Vec3), second: (Vec3, Vec3)) -> f64 {
    let first_direction = first.1 - first.0;
    let second_direction = second.1 - second.0;
    let offset = first.0 - second.0;
    let first_length = first_direction.dot(first_direction);
    let second_length = second_direction.dot(second_direction);
    let cross = first_direction.dot(second_direction);
    let first_offset = first_direction.dot(offset);
    let second_offset = second_direction.dot(offset);
    let denominator = first_length * second_length - cross * cross;
    let mut first_parameter;
    let mut second_parameter;
    if denominator > f64::EPSILON * first_length.max(second_length).max(1.0) {
        first_parameter = (cross * second_offset - second_length * first_offset) / denominator;
        first_parameter = first_parameter.clamp(0.0, 1.0);
    } else {
        first_parameter = 0.0;
    }
    second_parameter = (cross * first_parameter + second_offset) / second_length;
    if second_parameter < 0.0 {
        second_parameter = 0.0;
        first_parameter = (-first_offset / first_length).clamp(0.0, 1.0);
    } else if second_parameter > 1.0 {
        second_parameter = 1.0;
        first_parameter = ((cross - first_offset) / first_length).clamp(0.0, 1.0);
    }
    (offset + first_direction * first_parameter - second_direction * second_parameter).length()
}

fn validate_supported_body(
    body: &Body,
    existing: &[ExistingFillet],
    tolerance: f64,
) -> Result<(), FilletError> {
    if body.roots.len() != 1 {
        return Err(FilletError::UnsupportedBodyTopology);
    }
    let lump = body
        .lumps
        .get(body.roots[0])
        .ok_or(FilletError::InvalidBodyTopology)?;
    if lump.shells.len() != 1 {
        return Err(FilletError::UnsupportedBodyTopology);
    }
    let shell = body
        .shells
        .get(lump.shells[0])
        .ok_or(FilletError::InvalidBodyTopology)?;
    if shell.faces.len() != body.faces.len()
        || body.faces.iter().any(|(_, face)| face.loops.len() != 1)
    {
        return Err(FilletError::UnsupportedBodyTopology);
    }
    let cylinders = body
        .faces
        .iter()
        .filter(|(_, face)| {
            matches!(body.surfaces.get(face.surface), Some(Surface::Cylinder(_)))
        })
        .count();
    if cylinders != existing.len() {
        return Err(FilletError::UnsupportedExistingFillet);
    }
    if body.faces.iter().any(|(_, face)| {
        !matches!(
            body.surfaces.get(face.surface),
            Some(Surface::Plane(_) | Surface::Cylinder(_))
        )
    }) {
        return Err(FilletError::UnsupportedBodySurface);
    }

    for (_, face) in body.faces.iter() {
        let Some(Surface::Plane(plane)) = body.surfaces.get(face.surface) else {
            continue;
        };
        let Some(normal) = plane.normal().map(Vec3::from) else {
            return Err(FilletError::UnsupportedBodySurface);
        };
        let normal = if face.forward { normal } else { -normal };
        let offset = normal.dot(Vec3::from(plane.origin));
        if body.vertices.iter().any(|(_, vertex)| {
            normal.dot(Vec3::from(vertex.point)) > offset + tolerance
        }) {
            return Err(FilletError::NonConvexBody);
        }
    }
    Ok(())
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

fn restore_fillets(
    body: &mut Body,
    fillets: &[ExistingFillet],
    tolerance: f64,
) -> Option<()> {
    for fillet in fillets {
        let face = face_on(body, fillet.cut, tolerance)?;
        if round_face(body, face, &fillet.cylinder, fillet.forward, tolerance)? != 2 {
            return None;
        }
    }
    Some(())
}

fn face_on(body: &Body, halfspace: Halfspace, tolerance: f64) -> Option<FaceKey> {
    body.faces.iter().find_map(|(key, face)| {
        let Surface::Plane(plane) = body.surfaces.get(face.surface)? else {
            return None;
        };
        let mut normal = Vec3::from(plane.normal()?);
        if !face.forward {
            normal = -normal;
        }
        (normal.dot(halfspace.normal) > 1.0 - 1e-8
            && (normal.dot(Vec3::from(plane.origin)) - halfspace.offset).abs() <= tolerance)
            .then_some(key)
    })
}

fn faces_touch(body: &Body, first: FaceKey, second: FaceKey) -> bool {
    let vertices = |face| {
        body.face_coedges(face)
            .into_iter()
            .filter_map(|coedge| body.coedges.get(coedge))
            .filter_map(|coedge| body.edges.get(coedge.edge))
            .flat_map(|edge| [edge.start, edge.end])
            .collect::<HashSet<_>>()
    };
    let first = vertices(first);
    vertices(second).iter().any(|vertex| first.contains(vertex))
}

fn round_face(
    body: &mut Body,
    face: FaceKey,
    cylinder: &Cylinder,
    forward: bool,
    tolerance: f64,
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
        if (start_height - end_height).abs() > tolerance {
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

fn convex_body(halfspaces: &[Halfspace], tolerance: f64) -> Option<(Body, FaceKey)> {
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
                    .any(|plane| plane.normal.dot(point) > plane.offset + tolerance)
                {
                    continue;
                }
                if points
                    .iter()
                    .all(|other| other.distance(point) > tolerance)
                {
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
                ((halfspace.normal.dot(*point) - halfspace.offset).abs() <= tolerance)
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
        remove_collinear(&mut indices, &points, tolerance);
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
