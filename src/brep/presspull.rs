//! Planar face selection, extrusion, and topology-preserving face offsets.
//!
//! Extrusion adds or removes a prism. Offset instead keeps the neighbouring
//! surfaces and moves their intersections with the selected plane. Neither
//! operation reconstructs a solid from an intersection of half-spaces: doing
//! that changes concave solids, holes, and unrelated lumps into another shape.

use super::{Body, Curve3, EdgeKey, FaceKey, Meeting, Operation, Placement, Surface};
use crate::geom2d::{Arc, Curve, EllipseArc, Tolerance};
use crate::space::{Plane, Vec3};
use std::collections::{HashMap, HashSet};
use std::f64::consts::{FRAC_PI_2, TAU};

/// An exact planar boundary in the face's own coordinates.
#[derive(Debug, Clone)]
pub struct PlanarFaceProfile {
    pub plane: Plane,
    /// First loop encloses the face; subsequent loops cut holes.
    pub loops: Vec<Vec<Curve>>,
    /// Unit outward normal, independent of the parameter plane's handedness.
    pub outward: [f64; 3],
}

/// The two intentionally distinct face editing operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresspullMode {
    /// Add or remove a prism without extending neighbouring surfaces.
    Extrude,
    /// Extend or trim the neighbouring surfaces to the moved face plane.
    Offset,
}

/// Extracts every trimmed loop, retaining curved boundaries and holes.
pub fn planar_face_profile(body: &Body, key: FaceKey) -> Option<PlanarFaceProfile> {
    let face = body.faces.get(key)?;
    let Surface::Plane(plane) = body.surfaces.get(face.surface)? else {
        return None;
    };
    let tolerance = super::operation_tolerance(&[body]);
    let parts = super::pcurve::face_boundary_parts(body, key, tolerance)?;
    let mut loops = Vec::new();
    for ring in &face.loops {
        let coedges = &body.loops.get(*ring)?.coedges;
        let curves = coedges.iter().map(|key| {
            parts.iter().find(|(candidate, _)| candidate == key).map(|(_, curve)| curve.clone())
        }).collect::<Option<Vec<_>>>()?;
        if curves.is_empty() {
            return None;
        }
        loops.push(curves);
    }
    let normal = Vec3::from(plane.normal()?);
    Some(PlanarFaceProfile {
        plane: *plane,
        loops,
        outward: (normal * if face.forward { 1.0 } else { -1.0 }).to_array(),
    })
}

/// Finds a face actually containing the pick, never an infinite supporting plane.
/// Picks within `tolerance` of a trimmed boundary are accepted; holes are not.
pub fn planar_face_at_point(body: &Body, point: [f64; 3], tolerance: f64) -> Option<FaceKey> {
    if !tolerance.is_finite() || tolerance < 0.0 || point.iter().any(|v| !v.is_finite()) {
        return None;
    }
    body.face_keys().filter_map(|key| {
        let profile = planar_face_profile(body, key)?;
        let distance = profile.plane.distance_to(point)?.abs();
        if distance > tolerance {
            return None;
        }
        let local = profile.plane.project(point)?;
        let boundary = profile.loops.into_iter().flatten().collect::<Vec<_>>();
        crate::geom2d::contains(&boundary, local, Tolerance::new(tolerance.max(1e-12)))
            .then_some((key, distance))
    }).min_by(|a, b| a.1.total_cmp(&b.1)).map(|(key, _)| key)
}

/// Applies a signed edit along the selected face's outward normal.
/// The original is never changed, including on unsupported geometry or collapse.
pub fn presspull_face(body: &Body, key: FaceKey, distance: f64, mode: PresspullMode) -> Option<Body> {
    if !distance.is_finite() || distance.abs() <= f64::EPSILON {
        return None;
    }
    let profile = planar_face_profile(body, key)?;
    match mode {
        PresspullMode::Extrude => presspull_region(body, &profile, distance),
        PresspullMode::Offset => {
            // Working near the edited face avoids cancellation in intersections
            // of unit-sized solids located far from the world origin.
            let origin = Vec3::from(profile.plane.origin);
            let local = super::transform(body, &Placement::at((-origin).to_array()))?;
            let edited = offset_local(&local, key, distance)?;
            super::transform(&edited, &Placement::at(origin.to_array()))
        }
    }
}

/// Adds an outward bounded-region extrusion, or removes an inward extrusion.
/// `region.outward` must be a unit normal pointing out of the hosting face.
pub fn presspull_region(body: &Body, region: &PlanarFaceProfile, distance: f64) -> Option<Body> {
    if !distance.is_finite() || distance.abs() <= f64::EPSILON || region.loops.is_empty() {
        return None;
    }
    let normal = Vec3::from(region.outward).normalize()?;
    if normal.dot(Vec3::from(region.plane.normal()?)).abs() < 1.0 - 1e-9 {
        return None;
    }
    let origin = Vec3::from(region.plane.origin);
    let local = super::transform(body, &Placement::at((-origin).to_array()))?;
    let mut plane = region.plane;
    plane.origin = [0.0; 3];
    let loops = region.loops.iter().map(|ring| split_closed_curves(ring)).collect::<Vec<_>>();
    let tool = super::extrude_region(plane, &loops, (normal * distance).to_array())?;
    let tolerance = super::operation_tolerance(&[&local, &tool])
        .max(f64::EPSILON * origin.length().max(1.0) * 64.0);
    let edited = super::combine(local, tool, if distance > 0.0 {
        Operation::Union
    } else {
        Operation::Difference
    }, tolerance).ok()?;
    if edited.roots.is_empty() || !edited.validate().is_empty() {
        return None;
    }
    super::transform(&edited, &Placement::at(origin.to_array()))
}

/// Builds one bounded planar sheet face, including exact curved inner loops.
pub fn planar_region(plane: Plane, loops: &[Vec<Curve>]) -> Option<Body> {
    let loops = loops.iter().map(|ring| split_closed_curves(ring)).collect::<Vec<_>>();
    let solid = super::extrude_region(plane, &loops, plane.normal()?)?;
    let face = solid.face_keys().find(|key| {
        planar_face_profile(&solid, *key).is_some_and(|profile| {
            plane.distance_to(profile.plane.origin).is_some_and(|gap| gap.abs() < 1e-9)
                && Vec3::from(profile.outward).dot(Vec3::from(plane.normal().unwrap())) < 0.0
        })
    })?;
    let mut result = Body::new();
    let lump = result.lumps.insert(super::Lump { shells: Vec::new(), provenance: super::Provenance::Synthesized });
    let shell = result.shells.insert(super::Shell { faces: Vec::new(), owner: lump, provenance: super::Provenance::Synthesized });
    result.lumps.get_mut(lump)?.shells.push(shell);
    result.roots.push(lump);
    super::boolean::copy_face(&mut result, &solid, face, shell, true).ok()?;
    result.validate().is_empty().then_some(result)
}

/// Splits complete conics into exact bounded pieces for extrusion builders.
pub fn extrusion_profile_pieces(ring: &[Curve]) -> Vec<Curve> {
    ring.iter().flat_map(|curve| match curve {
        Curve::Circle(circle) => (0..4).map(|i| Curve::Arc(Arc {
            centre: circle.centre, radius: circle.radius,
            start_angle: i as f64 * FRAC_PI_2, end_angle: (i + 1) as f64 * FRAC_PI_2,
        })).collect(),
        Curve::Arc(arc) if arc.sweep() >= TAU - 1e-12 => (0..4).map(|i| Curve::Arc(Arc {
            centre: arc.centre, radius: arc.radius,
            start_angle: arc.start_angle + i as f64 * FRAC_PI_2,
            end_angle: arc.start_angle + (i + 1) as f64 * FRAC_PI_2,
        })).collect(),
        Curve::Ellipse(arc) if arc.sweep() >= TAU - 1e-12 => (0..4).map(|i| Curve::Ellipse(EllipseArc {
            ellipse: arc.ellipse, start_parameter: arc.start_parameter + i as f64 * FRAC_PI_2,
            end_parameter: arc.start_parameter + (i + 1) as f64 * FRAC_PI_2,
        })).collect(),
        Curve::Polyline(_) => curve.segments(),
        _ => vec![curve.clone()],
    }).collect()
}

fn split_closed_curves(ring: &[Curve]) -> Vec<Curve> {
    extrusion_profile_pieces(ring)
}

fn offset_local(body: &Body, key: FaceKey, distance: f64) -> Option<Body> {
    let profile = planar_face_profile(body, key)?;
    let normal = Vec3::from(profile.outward);
    let mut plane = profile.plane;
    plane.origin = (Vec3::from(plane.origin) + normal * distance).to_array();
    let tolerance = super::operation_tolerance(&[body]);
    if distance.abs() <= tolerance {
        return None;
    }
    let boundary: HashSet<EdgeKey> = body.face_coedges(key).into_iter()
        .map(|coedge| body.coedges.get(coedge).map(|coedge| coedge.edge))
        .collect::<Option<HashSet<_>>>()?;
    let mut vertices = HashSet::new();
    for edge in &boundary {
        let edge = body.edges.get(*edge)?;
        vertices.insert(edge.start);
        vertices.insert(edge.end);
    }
    let mut moved = HashMap::new();
    for vertex in &vertices {
        let original = Vec3::from(body.vertices.get(*vertex)?.point);
        let rails = body.edges.iter().filter(|(key, edge)| !boundary.contains(key)
            && (edge.start == *vertex || edge.end == *vertex)).collect::<Vec<_>>();
        let mut candidates = Vec::new();
        for (_, edge) in rails {
            let curve = body.curves.get(edge.curve)?;
            let parameters = plane_curve_parameters(&plane, curve)?;
            let start = edge.start == *vertex;
            let old = if start { edge.start_parameter } else { edge.end_parameter };
            let other = if start { edge.end_parameter } else { edge.start_parameter };
            let parameter = parameters.into_iter().filter(|t| {
                t.is_finite() && if start { *t < other - tolerance } else { *t > other + tolerance }
            }).min_by(|a, b| (a - old).abs().total_cmp(&(b - old).abs()))?;
            candidates.push(Vec3::from(curve.point_at(parameter)));
        }
        // A closed circular seam sometimes has no rail. Its radial parameter
        // on the neighbouring analytic surface identifies the same seam.
        let point = if let Some(first) = candidates.first().copied() {
            if candidates.iter().any(|other| first.distance(*other) > tolerance * 4.0) {
                return None;
            }
            first
        } else {
            let edge = boundary.iter().find_map(|edge| {
                let edge = body.edges.get(*edge)?;
                (edge.start == *vertex && edge.end == *vertex).then_some(edge)
            })?;
            let curve = body.curves.get(edge.curve)?;
            let other = adjacent_surface(body, edge, key)?;
            let next = intersect_curve(&plane, other, curve, tolerance)?;
            let estimated = original + normal * distance;
            Vec3::from(next.point_at(next.parameter_at(estimated.to_array())))
        };
        if plane.distance_to(point.to_array())?.abs() > tolerance * 4.0 {
            return None;
        }
        moved.insert(*vertex, point.to_array());
    }
    let mut result = body.clone();
    for (vertex, point) in &moved {
        result.vertices.get_mut(*vertex)?.point = *point;
        result.soil_vertex(*vertex);
    }
    let surface_key = result.surfaces.insert(Surface::Plane(plane));
    let selected = result.faces.get_mut(key)?;
    selected.surface = surface_key;
    selected.provenance.soil();
    let mut affected = HashSet::from([key]);
    for (edge_key, edge) in body.edges.iter() {
        if !vertices.contains(&edge.start) && !vertices.contains(&edge.end) {
            continue;
        }
        let original_curve = body.curves.get(edge.curve)?;
        let start = result.vertices.get(edge.start)?.point;
        let end = result.vertices.get(edge.end)?.point;
        let curve = if boundary.contains(&edge_key) {
            intersect_curve(&plane, adjacent_surface(body, edge, key)?, original_curve, tolerance)?
        } else {
            original_curve.clone()
        };
        let (start_parameter, end_parameter) = curve_span(&curve, start, end, edge, tolerance)?;
        let curve_key = result.curves.insert(curve);
        let edited = result.edges.get_mut(edge_key)?;
        edited.curve = curve_key;
        edited.start_parameter = start_parameter;
        edited.end_parameter = end_parameter;
        edited.provenance.soil();
        for coedge in &edge.coedges {
            let node = result.coedges.get_mut(*coedge)?;
            node.pcurve = None;
            node.provenance.soil();
            affected.insert(result.loops.get(node.owner)?.owner);
        }
    }
    for face in affected {
        let node = result.faces.get(face)?;
        let surface = result.surfaces.get(node.surface)?;
        if let Surface::Plane(_) = surface {
            let before = super::pcurve::face_boundary(body, face, tolerance)?;
            let after = super::pcurve::face_boundary(&result, face, tolerance)?;
            let old_area = boundary_area(&before);
            let new_area = boundary_area(&after);
            if old_area * new_area <= 0.0 || new_area.abs() <= tolerance * tolerance {
                return None;
            }
        }
        for coedge in result.face_coedges(face) {
            let edge = result.edges.get(result.coedges.get(coedge)?.edge)?;
            let curve = result.curves.get(edge.curve)?;
            for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let point = curve.point_at(edge.start_parameter
                    + fraction * (edge.end_parameter - edge.start_parameter));
                if !surface.distance_to(point).is_finite()
                    || surface.distance_to(point).abs() > tolerance * 8.0 {
                    return None;
                }
            }
        }
    }
    result.provenance.soil();
    (result.validate().is_empty() && result.worst_vertex_gap() <= tolerance * 4.0).then_some(result)
}

fn adjacent_surface<'a>(body: &'a Body, edge: &super::Edge, selected: FaceKey) -> Option<&'a Surface> {
    let other = edge.coedges.iter().find_map(|coedge| {
        let owner = body.loops.get(body.coedges.get(*coedge)?.owner)?.owner;
        (owner != selected).then_some(owner)
    })?;
    body.surfaces.get(body.faces.get(other)?.surface)
}

fn intersect_curve(plane: &Plane, other: &Surface, old: &Curve3, tolerance: f64) -> Option<Curve3> {
    let Meeting::Curves(curves) = super::intersect_surfaces(&Surface::Plane(*plane), other, tolerance) else {
        return None;
    };
    if curves.len() != 1 {
        return None;
    }
    let mut curve = curves.into_iter().next()?;
    match (&mut curve, old) {
        (Curve3::Line(new), Curve3::Line(old)) => {
            if Vec3::from(new.direction).dot(Vec3::from(old.direction)) < 0.0 {
                new.direction = (-Vec3::from(new.direction)).to_array();
            }
        }
        (Curve3::Circle(new), Curve3::Circle(old)) => {
            if Vec3::from(new.plane.normal()?).dot(Vec3::from(old.plane.normal()?)) < 0.0 {
                new.plane.y_axis = (-Vec3::from(new.plane.y_axis)).to_array();
            }
        }
        (Curve3::Ellipse(new), Curve3::Ellipse(old)) => {
            if Vec3::from(new.plane.normal()?).dot(Vec3::from(old.plane.normal()?)) < 0.0 {
                new.plane.y_axis = (-Vec3::from(new.plane.y_axis)).to_array();
            }
        }
        _ => return None,
    }
    Some(curve)
}

fn plane_curve_parameters(plane: &Plane, curve: &Curve3) -> Option<Vec<f64>> {
    let normal = Vec3::from(plane.normal()?);
    match curve {
        Curve3::Line(line) => {
            let along = normal.dot(Vec3::from(line.direction));
            if along.abs() <= 1e-12 * Vec3::from(line.direction).length() { return None; }
            Some(vec![normal.dot(Vec3::from(plane.origin) - Vec3::from(line.origin)) / along])
        }
        Curve3::Circle(circle) => trigonometric_parameters(plane, &circle.plane, circle.radius, circle.radius),
        Curve3::Ellipse(ellipse) => trigonometric_parameters(plane, &ellipse.plane, ellipse.major_radius, ellipse.minor_radius),
        _ => None,
    }
}

fn trigonometric_parameters(plane: &Plane, frame: &Plane, x: f64, y: f64) -> Option<Vec<f64>> {
    let normal = Vec3::from(plane.normal()?);
    let a = normal.dot(Vec3::from(frame.x_axis)) * x;
    let b = normal.dot(Vec3::from(frame.y_axis)) * y;
    let c = normal.dot(Vec3::from(plane.origin) - Vec3::from(frame.origin));
    let radius = a.hypot(b);
    if radius <= 1e-12 || c.abs() > radius { return None; }
    let phase = b.atan2(a);
    let angle = (c / radius).clamp(-1.0, 1.0).acos();
    Some((-2..=2).flat_map(|turn| [phase - angle + turn as f64 * TAU, phase + angle + turn as f64 * TAU]).collect())
}

fn curve_span(curve: &Curve3, start: [f64; 3], end: [f64; 3], original: &super::Edge, tolerance: f64) -> Option<(f64, f64)> {
    let from = curve.parameter_at(start);
    let mut to = curve.parameter_at(end);
    if matches!(curve, Curve3::Circle(_) | Curve3::Ellipse(_)) {
        to = from + (to - from).rem_euclid(TAU);
        if original.start == original.end { to = from + TAU; }
    }
    if !from.is_finite() || !to.is_finite() || to <= from
        || Vec3::from(curve.point_at(from)).distance(Vec3::from(start)) > tolerance * 4.0
        || Vec3::from(curve.point_at(to)).distance(Vec3::from(end)) > tolerance * 4.0 {
        return None;
    }
    Some((from, to))
}

fn boundary_area(curves: &[Curve]) -> f64 {
    curves.iter().map(Curve::enclosed_area).sum()
}
