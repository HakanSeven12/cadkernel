//! Source-directed joins which retain the source curve type and direction.

use super::Vec3;

/// Span two collinear finite lines, including the gap between them.
/// The output follows the first line's direction. Non-collinear and
/// degenerate inputs are rejected without changing either input.
pub fn join_collinear_lines(source: [[f64; 3]; 2], other: [[f64; 3]; 2], tolerance: f64) -> Option<[[f64; 3]; 2]> {
    if !tolerance.is_finite() || tolerance < 0.0 || source.iter().chain(other.iter()).flatten().any(|v| !v.is_finite()) { return None; }
    let a = Vec3::from(source[0]);
    let delta = Vec3::from(source[1]) - a;
    let length = delta.length();
    if length <= tolerance { return None; }
    let direction = delta.normalize()?;
    let c = Vec3::from(other[0]);
    let d = Vec3::from(other[1]);
    if c.distance(d) <= tolerance || (c-a).cross(direction).length() > tolerance || (d-a).cross(direction).length() > tolerance { return None; }
    let first = (c-a).dot(direction);
    let last = (d-a).dot(direction);
    Some([(a + direction * first.min(last).min(0.0)).to_array(), (a + direction * first.max(last).max(length)).to_array()])
}

/// Extend a counterclockwise angular interval to include another interval.
/// Angles share a circle and angular frame, established by the caller. The
/// source start is retained; a full revolution is returned as start + TAU.
pub fn join_counterclockwise_spans(source: [f64; 2], other: [f64; 2]) -> Option<[f64; 2]> {
    use std::f64::consts::TAU;
    if source.iter().chain(other.iter()).any(|v| !v.is_finite()) { return None; }
    let source_span = (source[1] - source[0]).rem_euclid(TAU);
    let other_span = (other[1] - other[0]).rem_euclid(TAU);
    if source_span == 0.0 || other_span == 0.0 { return None; }
    let offset = (other[0] - source[0]).rem_euclid(TAU);
    let end = source_span.max(offset + other_span).min(TAU);
    Some([source[0], source[0] + end])
}

/// Join arcs represented in the same object-coordinate angular frame.
/// Centers and normals must agree within tolerance, and radii must be positive.
/// Opposite normals are rejected because their angular frames differ.
pub fn join_cocircular_arcs(
    source: ([f64; 3], [f64; 3], f64, [f64; 2]),
    other: ([f64; 3], [f64; 3], f64, [f64; 2]),
    tolerance: f64,
) -> Option<[f64; 2]> {
    if !tolerance.is_finite() || tolerance < 0.0 || !source.2.is_finite() || !other.2.is_finite()
        || source.2 <= 0.0 || other.2 <= 0.0 || (source.2-other.2).abs() > tolerance
        || source.0.iter().chain(source.1.iter()).chain(other.0.iter()).chain(other.1.iter()).any(|v| !v.is_finite()) { return None; }
    if Vec3::from(source.0).distance(Vec3::from(other.0)) > tolerance
        || Vec3::from(source.1).normalize()?.distance(Vec3::from(other.1).normalize()?) > 1e-12 { return None; }
    join_counterclockwise_spans(source.3, other.3)
}
