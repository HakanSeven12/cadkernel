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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dynamic_endpoint_can_cross_the_fixed_endpoint() {
        assert_eq!(lengthen_line([0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 4.0],
            LengthChange::Dynamic([0.0, 2.0, -3.0])),
            Some([[0.0, 0.0, 0.0], [0.0, 0.0, -3.0]]));
    }
}
