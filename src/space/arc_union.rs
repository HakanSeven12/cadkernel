//! Circular arc unions on an identical supporting circle and angular frame.

/// A counterclockwise arc in an object-coordinate angular frame.
#[derive(Clone, Copy, Debug)]
pub struct CircularArc {
    pub center: [f64; 3],
    pub normal: [f64; 3],
    pub radius: f64,
    pub start: f64,
    pub end: f64,
}
/// Relation between arcs eligible for union.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArcUnionKind { Duplicate, Overlap, EndToEnd }
/// Union angles in the input frame, with explicit full-circle coverage.
#[derive(Clone, Copy, Debug)]
pub struct ArcUnion {
    pub start: f64,
    pub end: f64,
    pub full_circle: bool,
    pub kind: ArcUnionKind,
}
/// Unite coincident, overlapping or touching arcs without bridging a real gap.
/// Supporting centers, radii and normals must match exactly. Tolerance applies
/// to endpoint distance along the circle, not to different supporting circles.
pub fn circular_arc_union(a: CircularArc, b: CircularArc, tolerance: f64) -> Option<ArcUnion> {
    use std::f64::consts::TAU;
    if !tolerance.is_finite() || tolerance<0.0 || a.center!=b.center || a.normal!=b.normal || a.radius!=b.radius {return None;}
    for arc in [a,b] {
        if !arc.center.iter().chain(arc.normal.iter()).all(|v|v.is_finite()) || !arc.radius.is_finite() || arc.radius<=0.0
            || !arc.start.is_finite() || !arc.end.is_finite() || super::Vec3::from(arc.normal).normalize().is_none() {return None;}
    }
    let span=(a.end-a.start).rem_euclid(TAU);let other_span=(b.end-b.start).rem_euclid(TAU);
    if !span.is_finite() || !other_span.is_finite() || span==0.0 || other_span==0.0 {return None;}
    let epsilon=(tolerance/a.radius).min(std::f64::consts::PI);
    let offset=(b.start-a.start).rem_euclid(TAU);
    if !offset.is_finite() {return None;}
    let distance=|x:f64,y:f64| {let d=(x-y).rem_euclid(TAU);d.min(TAU-d)};
    if distance(a.start,b.start)<=epsilon && distance(a.end,b.end)<=epsilon && (span-other_span).abs()<=epsilon {
        return Some(ArcUnion{start:a.start.rem_euclid(TAU),end:a.end.rem_euclid(TAU),full_circle:false,kind:ArcUnionKind::Duplicate});
    }
    let mut best: Option<(f64,f64,f64)>=None;
    for shift in [-TAU,0.0,TAU] {
        let low=offset+shift;let high=low+other_span;
        let overlap=span.min(high)-0.0_f64.max(low);
        if overlap < -epsilon {continue;}
        if best.is_none_or(|(_,_,previous)|overlap>previous) {best=Some((low.min(0.0),high.max(span),overlap));}
    }
    let (low,high,overlap)=best?;
    Some(ArcUnion{start:(a.start+low).rem_euclid(TAU),end:(a.start+high).rem_euclid(TAU),full_circle:high-low>=TAU-epsilon,
        kind:if overlap>epsilon {ArcUnionKind::Overlap}else{ArcUnionKind::EndToEnd}})
}

/// Whether a complete circle contains this finite arc on the same support.
/// Exact support matching follows cleanup semantics; no circle is displaced.
pub fn circle_contains_arc(center: [f64; 3], normal: [f64; 3], radius: f64, arc: CircularArc) -> bool {
    center == arc.center && normal == arc.normal && radius == arc.radius
        && center.iter().chain(normal.iter()).all(|v| v.is_finite())
        && radius.is_finite() && radius > 0.0 && super::Vec3::from(normal).normalize().is_some()
        && arc.start.is_finite() && arc.end.is_finite()
        && { let span = (arc.end - arc.start).rem_euclid(std::f64::consts::TAU); span.is_finite() && span > 0.0 }
}
