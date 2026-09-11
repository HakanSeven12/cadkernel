//! Intersections used to extend two straight terminal segments to a joint.
use super::Vec3;

/// Find a common endpoint for two terminal segments. Each pair is ordered
/// from the fixed interior point to the movable endpoint. Both endpoint
/// displacements must fit `distance`; neither segment may reverse through
/// its fixed point. Parallel, skew, degenerate and nonfinite inputs fail.
/// This does not bridge parallel gaps or insert connector segments.
pub fn extend_line_ends(a: [[f64; 3]; 2], b: [[f64; 3]; 2], distance: f64) -> Option<[f64; 3]> {
    if !distance.is_finite() || distance < 0.0 ||
        a.iter().chain(b.iter()).flatten().any(|value| !value.is_finite()) { return None; }
    let p = Vec3::from(a[1]); let q = Vec3::from(b[1]);
    let u = p - Vec3::from(a[0]); let v = q - Vec3::from(b[0]);
    let lu = u.length(); let lv = v.length();
    if !lu.is_finite() || !lv.is_finite() || lu <= 0.0 || lv <= 0.0 { return None; }
    let u = u / lu; let v = v / lv; let n = u.cross(v);
    let denominator = n.length_squared();
    if !denominator.is_finite() || denominator <= 1e-24 { return None; }
    let delta = q - p;
    let ta = delta.cross(v).dot(n) / denominator;
    let tb = delta.cross(u).dot(n) / denominator;
    if !ta.is_finite() || !tb.is_finite() || ta.abs() > distance || tb.abs() > distance ||
        ta <= -lu || tb <= -lv { return None; }
    let pa = p + u * ta; let pb = q + v * tb;
    let tolerance: f64 = 1e-9;
    if !tolerance.is_finite() || pa.to_array().iter().chain(pb.to_array().iter()).any(|value| !value.is_finite()) ||
        pa.distance(pb) > tolerance { return None; }
    Some(pa.to_array())
}


/// How a gap between two open straight terminal spans is joined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinType { Extend, Add, Both }

/// A planar selection-relative connector threshold. The coordinate scale is
/// the largest absolute selected planar coordinate, not the extent span or
/// diagonal. Elevation and unselected geometry do not contribute. Consumers
/// supply all selected path coordinates before any paths are consumed.
pub fn planar_connector_distance(points: &[[f64; 2]], fuzz: f64) -> Option<f64> {
    if points.is_empty() || !fuzz.is_finite() || fuzz < 0.0 { return None; }
    let mut scale: f64 = 0.0;
    for coordinate in points.iter().flatten() {
        if !coordinate.is_finite() { return None; }
        scale = scale.max(coordinate.abs());
    }
    let distance = scale * fuzz;
    distance.is_finite().then_some(distance)
}

/// Choose an intersection or a connector for a pair of straight terminals.
/// `None` preserves both paths. `Some(None)` retains both endpoints and adds
/// a straight connector; `Some(Some(point))` moves both endpoints to a joint.
/// Zero fuzz never bridges a gap; callers may separately join exact endpoints.
pub fn join_line_ends(a: [[f64; 3]; 2], b: [[f64; 3]; 2], fuzz: f64,
    connector_distance: f64, kind: JoinType) -> Option<Option<[f64; 3]>> {
    if !fuzz.is_finite() || fuzz <= 0.0 || !connector_distance.is_finite() || connector_distance < 0.0 ||
        a.iter().chain(b.iter()).flatten().any(|value| !value.is_finite()) { return None; }
    if kind != JoinType::Add {
        if let Some(point) = extend_line_ends(a,b,fuzz) { return Some(Some(point)); }
        if kind == JoinType::Extend { return None; }
    }
    let a0 = Vec3::from(a[0]); let a1 = Vec3::from(a[1]);
    let b0 = Vec3::from(b[0]); let b1 = Vec3::from(b[1]);
    let lengths = [a0.distance(a1),b0.distance(b1),a1.distance(b1)];
    if lengths.iter().any(|length| !length.is_finite() || *length <= 0.0) || lengths[2] > connector_distance { return None; }
    Some(None)
}


/// Choose the closest eligible terminal pair. Both searches intersections
/// before connectors, so path direction does not select a farther connector.
pub fn closest_line_end_join(a: &[[[f64;3];2]], b: &[[[f64;3];2]], fuzz: f64,
    connector_distance: f64, kind: JoinType) -> Option<(usize,usize,Option<[f64;3]>)> {
    let phases: &[JoinType] = match kind { JoinType::Extend => &[JoinType::Extend], JoinType::Add => &[JoinType::Add], JoinType::Both => &[JoinType::Extend,JoinType::Add] };
    for phase in phases {
        let mut best = None; let mut best_distance = f64::INFINITY;
        for (i,first) in a.iter().enumerate() { for (j,second) in b.iter().enumerate() {
            let Some(joint) = join_line_ends(*first,*second,fuzz,connector_distance,*phase) else {continue;};
            let distance = Vec3::from(first[1]).distance(Vec3::from(second[1]));
            if distance.is_finite() && distance < best_distance {best_distance=distance;best=Some((i,j,joint));}
        }}
        if best.is_some() {return best;}
    }
    None
}
