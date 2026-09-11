//! Tolerant unions of collinear finite line segments in three dimensions.
use super::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineUnionKind {
    Duplicate,
    Overlap,
    EndToEnd,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineUnion {
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub kind: LineUnionKind,
}

/// Returns the union on the first segment's supporting line, preserving its
/// direction. Noncollinear, disjoint, degenerate and nonfinite inputs fail.
/// The caller chooses which relation kinds its operation permits.
pub fn line_union(
    a: [[f64; 3]; 2], b: [[f64; 3]; 2], tolerance: f64,
) -> Option<LineUnion> {
    if !tolerance.is_finite() || tolerance < 0.0
        || !a.iter().chain(b.iter()).flatten().all(|v| v.is_finite()) {
        return None;
    }
    let origin = Vec3::from(a[0]);
    let delta = Vec3::from(a[1]) - origin;
    let length = delta.length();
    let b_delta = Vec3::from(b[1]) - Vec3::from(b[0]);
    if length <= tolerance || b_delta.length() <= tolerance { return None; }
    let direction = delta / length;
    let first = Vec3::from(b[0]) - origin;
    let last = Vec3::from(b[1]) - origin;
    if first.cross(direction).length() > tolerance
        || last.cross(direction).length() > tolerance { return None; }
    let s = first.dot(direction);
    let e = last.dot(direction);
    let low = s.min(e);
    let high = s.max(e);
    if low > length + tolerance || high < -tolerance { return None; }
    let kind = if low.abs() <= tolerance && (high - length).abs() <= tolerance {
        LineUnionKind::Duplicate
    } else if high.min(length) - low.max(0.0) > tolerance {
        LineUnionKind::Overlap
    } else {
        LineUnionKind::EndToEnd
    };
    Some(LineUnion {
        start: (origin + direction * low.min(0.0)).to_array(),
        end: (origin + direction * high.max(length)).to_array(),
        kind,
    })
}
