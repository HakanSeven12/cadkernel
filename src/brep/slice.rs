//! Splitting a boundary-representation body with an infinite plane.
//!
//! The public operation returns both sides together.  This matters to hosts:
//! they can prepare every requested split before replacing any document
//! entity, so a failed cut never makes the source disappear.

use super::topology::{Body, FaceKey, Lump, Shell};
use super::{body_bounds, combine, imprint, operation_tolerance, Operation, Placement, Provenance, Snag};
use crate::space::{Plane, Vec3};

/// The two non-empty bodies produced by a plane crossing a body.
#[derive(Debug, Clone)]
pub struct PlaneSlice {
    /// Geometry on the side opposite the plane normal.
    pub negative: Body,
    /// Geometry on the side pointed to by the plane normal.
    pub positive: Body,
}

/// Splits a solid or open sheet body with an infinite plane.
///
/// `Ok(None)` means that the plane does not cross the body.  A tangent plane
/// is deliberately not reported as a split because it would create an empty
/// or zero-thickness result.  Closed bodies are capped by regular Boolean
/// intersections; open sheets retain their open boundaries.
pub fn slice_by_plane(body: &Body, plane: Plane) -> Result<Option<PlaneSlice>, Snag> {
    if body.faces.is_empty() {
        return Ok(None);
    }
    let normal = plane.normal().ok_or(Snag::CutRefused)?;
    let plane = Plane::orthonormal(plane.origin, plane.x_axis, normal)
        .ok_or(Snag::CutRefused)?;
    let bounds = body_bounds(body).ok_or(Snag::CutRefused)?;
    let frame = projected_bounds(bounds, plane).ok_or(Snag::CutRefused)?;
    let tolerance = operation_tolerance(&[body]);
    if frame.min[2] >= -tolerance || frame.max[2] <= tolerance {
        return Ok(None);
    }

    let closed = body
        .edges
        .iter()
        .filter(|(_, edge)| !edge.coedges.is_empty())
        .all(|(_, edge)| edge.coedges.len() == 2);
    if closed {
        split_solid(body, plane, frame, tolerance)
    } else {
        split_sheet(body, plane, frame, tolerance)
    }
}

#[derive(Clone, Copy)]
struct FrameBounds {
    min: [f64; 3],
    max: [f64; 3],
}

fn projected_bounds(bounds: super::Aabb, plane: Plane) -> Option<FrameBounds> {
    let origin = Vec3::from(plane.origin);
    let x = Vec3::from(plane.x_axis);
    let y = Vec3::from(plane.y_axis);
    let normal = Vec3::from(plane.normal()?);
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for bits in 0..8 {
        let point = Vec3::new(
            if bits & 1 == 0 { bounds.min[0] } else { bounds.max[0] },
            if bits & 2 == 0 { bounds.min[1] } else { bounds.max[1] },
            if bits & 4 == 0 { bounds.min[2] } else { bounds.max[2] },
        ) - origin;
        let local = [point.dot(x), point.dot(y), point.dot(normal)];
        for axis in 0..3 {
            min[axis] = min[axis].min(local[axis]);
            max[axis] = max[axis].max(local[axis]);
        }
    }
    min.iter()
        .chain(max.iter())
        .all(|value| value.is_finite())
        .then_some(FrameBounds { min, max })
}

fn split_solid(
    body: &Body,
    plane: Plane,
    frame: FrameBounds,
    tolerance: f64,
) -> Result<Option<PlaneSlice>, Snag> {
    let (negative_cutter, positive_cutter) = half_boxes(plane, frame, tolerance)?;
    let negative = combine(body.clone(), negative_cutter, Operation::Intersection, tolerance)?;
    let positive = combine(body.clone(), positive_cutter, Operation::Intersection, tolerance)?;
    if negative.faces.is_empty() || positive.faces.is_empty() {
        return Ok(None);
    }
    Ok(Some(PlaneSlice { negative, positive }))
}

fn split_sheet(
    body: &Body,
    plane: Plane,
    frame: FrameBounds,
    tolerance: f64,
) -> Result<Option<PlaneSlice>, Snag> {
    let (_, mut cutter) = half_boxes(plane, frame, tolerance)?;
    let mut divided = body.clone();
    let report = imprint(&mut divided, &mut cutter, tolerance)?;
    if report.cuts == 0 {
        return Ok(None);
    }

    let mut negative_faces = Vec::new();
    let mut positive_faces = Vec::new();
    for face in divided.face_keys() {
        match face_side(&divided, face, plane, tolerance)? {
            -1 => negative_faces.push(face),
            1 => positive_faces.push(face),
            _ => return Err(Snag::CutRefused),
        }
    }
    if negative_faces.is_empty() || positive_faces.is_empty() {
        return Ok(None);
    }
    Ok(Some(PlaneSlice {
        negative: copy_faces(&divided, &negative_faces)?,
        positive: copy_faces(&divided, &positive_faces)?,
    }))
}

fn face_side(body: &Body, face: FaceKey, plane: Plane, tolerance: f64) -> Result<i8, Snag> {
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for coedge in body.face_coedges(face) {
        let edge = body
            .edges
            .get(body.coedges.get(coedge).ok_or(Snag::CutRefused)?.edge)
            .ok_or(Snag::CutRefused)?;
        for point in [
            body.vertices.get(edge.start).ok_or(Snag::CutRefused)?.point,
            body.vertices.get(edge.end).ok_or(Snag::CutRefused)?.point,
            body.curves
                .get(edge.curve)
                .ok_or(Snag::CutRefused)?
                .point_at((edge.start_parameter + edge.end_parameter) * 0.5),
        ] {
            let distance = plane.distance_to(point).ok_or(Snag::CutRefused)?;
            low = low.min(distance);
            high = high.max(distance);
        }
    }
    if high <= tolerance {
        Ok(-1)
    } else if low >= -tolerance {
        Ok(1)
    } else {
        Err(Snag::CutRefused)
    }
}

fn copy_faces(source: &Body, faces: &[FaceKey]) -> Result<Body, Snag> {
    let mut result = Body::new();
    let lump = result.lumps.insert(Lump {
        shells: Vec::new(),
        provenance: Provenance::Synthesized,
    });
    let shell = result.shells.insert(Shell {
        faces: Vec::new(),
        owner: lump,
        provenance: Provenance::Synthesized,
    });
    result.lumps.get_mut(lump).ok_or(Snag::CutRefused)?.shells.push(shell);
    result.roots.push(lump);
    for face in faces {
        super::boolean::copy_face(&mut result, source, *face, shell, false)?;
    }
    if result.validate().is_empty() {
        Ok(result)
    } else {
        Err(Snag::CutRefused)
    }
}

fn half_boxes(
    plane: Plane,
    frame: FrameBounds,
    tolerance: f64,
) -> Result<(Body, Body), Snag> {
    let span = (0..3)
        .map(|axis| frame.max[axis] - frame.min[axis])
        .fold(1.0_f64, f64::max);
    let margin = span * 2.0 + tolerance * 64.0;
    let low_x = frame.min[0] - margin;
    let low_y = frame.min[1] - margin;
    let size_x = frame.max[0] - frame.min[0] + margin * 2.0;
    let size_y = frame.max[1] - frame.min[1] + margin * 2.0;
    let low_z = frame.min[2] - margin;
    let high_z = frame.max[2] + margin;
    let negative = super::make::cuboid([low_x, low_y, low_z], [size_x, size_y, -low_z])
        .ok_or(Snag::CutRefused)?;
    let positive = super::make::cuboid([low_x, low_y, 0.0], [size_x, size_y, high_z])
        .ok_or(Snag::CutRefused)?;
    let placement = Placement {
        x_axis: plane.x_axis,
        y_axis: plane.y_axis,
        z_axis: plane.normal().ok_or(Snag::CutRefused)?,
        origin: plane.origin,
    };
    Ok((
        super::transform(&negative, &placement).ok_or(Snag::CutRefused)?,
        super::transform(&positive, &placement).ok_or(Snag::CutRefused)?,
    ))
}
