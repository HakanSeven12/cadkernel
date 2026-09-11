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
    let tolerance = 1e-9;
    if !tolerance.is_finite() || pa.to_array().iter().chain(pb.to_array().iter()).any(|value| !value.is_finite()) ||
        pa.distance(pb) > tolerance { return None; }
    Some(pa.to_array())
}
