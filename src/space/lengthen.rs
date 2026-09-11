//! Endpoint length changes which preserve a curve's supporting geometry.
use super::Vec3;

#[derive(Clone, Copy, Debug)]
pub enum LengthChange {
    Delta(f64),
    Total(f64),
    Percent(f64),
    DeltaAngle(f64),
    TotalAngle(f64),
    Dynamic([f64; 3]),
}

impl LengthChange {
    fn length(self, current: f64) -> Option<f64> {
        let value = match self {
            Self::Delta(value) => current + value,
            Self::Total(value) => value,
            Self::Percent(value) => current * value / 100.0,
            _ => return None,
        };
        (value.is_finite() && value > 1e-12).then_some(value)
    }
}

/// Move the endpoint closest to the pick, retaining the supporting 3D line.
/// Dynamic points are projected onto that line and may cross its fixed end.
pub fn lengthen_line(start: [f64; 3], end: [f64; 3], pick: [f64; 3], change: LengthChange) -> Option<[[f64; 3]; 2]> {
    if start.iter().chain(&end).chain(&pick).any(|v| !v.is_finite()) { return None; }
    let start = Vec3::from(start);
    let end = Vec3::from(end);
    let pick = Vec3::from(pick);
    let direction = end - start;
    let length = direction.length();
    if !length.is_finite() || length <= 1e-12 { return None; }
    let direction = direction / length;
    let change_end = pick.distance_squared(end) <= pick.distance_squared(start);
    let new_length = if let LengthChange::Dynamic(point) = change {
        let point = Vec3::from(point);
        if !point.is_finite() { return None; }
        if change_end { (point - start).dot(direction) } else { (end - point).dot(direction) }
    } else { change.length(length)? };
    if !new_length.is_finite() || new_length.abs() <= 1e-12 { return None; }
    let result = if change_end { [start.to_array(), (start + direction * new_length).to_array()] }
        else { [(end - direction * new_length).to_array(), end.to_array()] };
    result.iter().flatten().all(|v| v.is_finite()).then_some(result)
}

/// Change an arc endpoint in its own plane. Angle values are radians.
/// Return the replacement start and end angles; radius and plane stay intact.
#[cfg(feature = "geom2d")]
pub fn lengthen_arc(curve: &super::PlanarCurve, pick: [f64; 3], change: LengthChange) -> Option<(f64, f64)> {
    let crate::geom2d::Curve::Arc(arc) = &curve.curve else { return None; };
    if !arc.radius.is_finite() || arc.radius <= 1e-12 || pick.iter().any(|v| !v.is_finite()) { return None; }
    let pick = Vec3::from(pick);
    let change_end = pick.distance_squared(curve.point_at(1.0).into()) <= pick.distance_squared(curve.point_at(0.0).into());
    let current = arc.sweep();
    let span = match change {
        LengthChange::DeltaAngle(value) => current + value,
        LengthChange::TotalAngle(value) => value,
        LengthChange::Dynamic(point) => {
            if point.iter().any(|v| !v.is_finite()) { return None; }
            let point = curve.plane.project(point)?;
            let x = point[0] - arc.centre[0];
            let y = point[1] - arc.centre[1];
            if x.hypot(y) <= 1e-12 { return None; }
            let angle = y.atan2(x);
            if change_end { (angle - arc.start_angle).rem_euclid(std::f64::consts::TAU) }
            else { (arc.end_angle - angle).rem_euclid(std::f64::consts::TAU) }
        }
        _ => change.length(current * arc.radius)? / arc.radius,
    };
    if !span.is_finite() || span <= 1e-12 || span >= std::f64::consts::TAU - 1e-12 { return None; }
    Some(if change_end { (arc.start_angle, arc.start_angle + span) }
        else { (arc.end_angle - span, arc.end_angle) })
}

/// Change an elliptic arc endpoint in its supporting plane, retaining its axes.
#[cfg(feature = "geom2d")]
pub fn lengthen_ellipse(curve: &super::PlanarCurve, pick: [f64; 3], change: LengthChange) -> Option<(f64, f64)> {
    use crate::geom2d::{Curve, EllipseArc};
    let Curve::Ellipse(arc) = &curve.curve else { return None; };
    let shape = arc.ellipse;
    if pick.iter().any(|v| !v.is_finite()) || !shape.major_radius.is_finite()
        || !shape.minor_radius.is_finite() || shape.major_radius <= 1e-12 || shape.minor_radius <= 1e-12 { return None; }
    let tau = std::f64::consts::TAU;
    let start = arc.start_parameter;
    let end = if arc.end_parameter > start { arc.end_parameter } else { arc.end_parameter + tau };
    if !start.is_finite() || !end.is_finite() || end - start >= tau - 1e-12 { return None; }
    let change_end = Vec3::from(pick).distance_squared(curve.point_at(1.0).into())
        <= Vec3::from(pick).distance_squared(curve.point_at(0.0).into());
    let parameter = if let LengthChange::Dynamic(point) = change {
        if point.iter().any(|v| !v.is_finite()) { return None; }
        let point = curve.plane.project(point)?;
        let x = point[0] - shape.centre[0]; let y = point[1] - shape.centre[1];
        let u = (x * shape.major_axis[0] + y * shape.major_axis[1]) / shape.major_radius;
        let v = (-x * shape.major_axis[1] + y * shape.major_axis[0]) / shape.minor_radius;
        if u.hypot(v) <= 1e-12 { return None; }
        v.atan2(u)
    } else {
        let length = change.length(curve.curve.length())?;
        let from = if change_end { start } else { end - tau };
        let whole = Curve::Ellipse(EllipseArc { ellipse: shape, start_parameter: from, end_parameter: from + tau });
        if length >= whole.length() - 1e-12 { return None; }
        let distance = if change_end { length } else { whole.length() - length };
        from + whole.parameter_at_distance(distance) * tau
    };
    let span = if change_end { (parameter - start).rem_euclid(tau) } else { (end - parameter).rem_euclid(tau) };
    if !span.is_finite() || span <= 1e-12 || span >= tau - 1e-12 { return None; }
    Some(if change_end { (start, start + span) } else { (end - span, end) })
}

/// A retained polyline vertex and its original metadata index.
#[cfg(feature = "geom2d")]
#[derive(Clone, Debug)]
pub struct LengthenedVertex {
    pub position: [f64; 2], pub bulge: f64, pub source: usize,
    /// Fractions along the original segment for interpolating endpoint metadata.
    pub start_fraction: f64, pub end_fraction: f64,
}

/// Scalar whole-length changes for an open bulged polyline. Dynamic and angular
/// changes are unsupported. Trimming may remove complete terminal segments.
#[cfg(feature = "geom2d")]
pub fn lengthen_polyline(curve: &super::PlanarCurve, pick: [f64; 3], change: LengthChange) -> Option<Vec<LengthenedVertex>> {
    use crate::geom2d::{Curve, BulgeArc};
    let Curve::Polyline(poly) = &curve.curve else { return None; };
    if poly.closed || poly.vertices.len() < 2 || pick.iter().any(|v| !v.is_finite()) { return None; }
    let mut vertices = poly.vertices.iter().enumerate().map(|(source, v)| LengthenedVertex {
        position: v.position, bulge: v.bulge, source, start_fraction: 0.0, end_fraction: 1.0,
    }).collect::<Vec<_>>();
    if vertices.iter().any(|v| !v.bulge.is_finite() || v.position.iter().any(|c| !c.is_finite())) { return None; }
    let change_end = Vec3::from(pick).distance_squared(curve.point_at(1.0).into())
        <= Vec3::from(pick).distance_squared(curve.point_at(0.0).into());
    if !change_end {
        let original = vertices.clone(); vertices.reverse();
        for i in 0..vertices.len()-1 { vertices[i].bulge = -original[original.len()-2-i].bulge; }
        vertices.last_mut()?.bulge = 0.0;
    }
    let lengths = vertices.windows(2).map(|pair| {
        BulgeArc::from_bulge(pair[0].position, pair[1].position, pair[0].bulge)
            .map(|arc| arc.radius * arc.sweep.abs())
            .unwrap_or_else(|| (pair[1].position[0]-pair[0].position[0]).hypot(pair[1].position[1]-pair[0].position[1]))
    }).collect::<Vec<_>>();
    if lengths.iter().any(|v| !v.is_finite() || *v <= 1e-12) { return None; }
    let total: f64 = lengths.iter().sum();
    if !total.is_finite() { return None; }
    let target = change.length(total)?;
    let mut remaining = target;
    let mut index = 0;
    while index + 1 < lengths.len() && remaining > lengths[index] {
        remaining -= lengths[index]; index += 1;
    }
    let fraction = remaining / lengths[index];
    let first = vertices[index].clone(); let last = vertices[index+1].clone();
    let (position, bulge) = if let Some(arc) = BulgeArc::from_bulge(first.position, last.position, first.bulge) {
        let sweep = (remaining / arc.radius).rem_euclid(std::f64::consts::TAU);
        if sweep <= 1e-12 { return None; }
        (arc.sample(sweep / arc.sweep.abs()), (sweep * arc.sweep.signum() / 4.0).tan())
    } else {
        let fraction = remaining / lengths[index];
        ([first.position[0] + (last.position[0]-first.position[0])*fraction,
          first.position[1] + (last.position[1]-first.position[1])*fraction], 0.0)
    };
    if position.iter().any(|v| !v.is_finite()) || !bulge.is_finite() { return None; }
    vertices.truncate(index+2);
    vertices[index].bulge = bulge;
    vertices[index+1].position = position;
    // A cut point inherits the cut segment's metadata; an extension keeps its endpoint.
    if change_end {
        vertices[index].end_fraction = fraction;
        if target < total {
            vertices[index+1].source = first.source; vertices[index+1].bulge = bulge;
            vertices[index+1].end_fraction = fraction;
        }
    }
    if !change_end {
        let reversed = vertices.clone(); vertices.reverse();
        for i in 0..vertices.len()-1 { vertices[i].bulge = -reversed[reversed.len()-2-i].bulge; }
        vertices.last_mut()?.bulge = poly.vertices.last()?.bulge;
        vertices[0].source = last.source;
        vertices[0].start_fraction = 1.0 - fraction;
    }
    Some(vertices)
}
