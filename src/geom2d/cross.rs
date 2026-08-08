//! Where any two curves meet.
//!
//! The per-shape functions in [`intersect`](super::intersect) answer one pair
//! each and speak in whatever parameters that pair naturally produces. This
//! module is the dispatch over [`Curve`], and it answers in the shared `0..=1`
//! parameter, so a caller holding two curves whose types it learns at runtime
//! does not need a match of its own.
//!
//! # How the pairs are solved
//!
//! Line against line, circle, arc or ellipse, and circle-or-arc against
//! circle-or-arc, are solved in closed form. Everything else — ellipse against
//! anything curved, and any pair involving a polyline — is solved by sampling
//! and intersecting the resulting segments, which is approximate and gets more
//! accurate with a finer `density`.
//!
//! # How a crossing is kept
//!
//! Each pair produces candidate *points*. Both parameters then come from
//! [`Curve::parameter_at`], and the candidate survives only if it genuinely
//! lies on both curves — the parameter is inside the curve's own extent, and
//! evaluating it lands back where the candidate was. That second check is what
//! rejects a point on an arc's circle but outside the arc, without every case
//! having to test containment for itself.

use super::curve::Curve;
use super::intersect::{circle_circle_points, line_circle, line_ellipse, line_line};
use super::tessellate::DEFAULT_SEGMENTS_PER_RADIAN;
use super::Tolerance;

/// A point where two curves meet, with where it falls on each.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crossing {
    /// Where they meet.
    pub point: [f64; 2],
    /// Parameter on the first curve, `0..=1`.
    pub t_a: f64,
    /// Parameter on the second.
    pub t_b: f64,
}

/// Every point where `a` and `b` cross, ordered along `a`.
///
/// Curved pairs with no closed form are sampled at
/// [`DEFAULT_SEGMENTS_PER_RADIAN`]; use [`intersect_with_density`] to choose.
pub fn intersect(a: &Curve, b: &Curve, tolerance: Tolerance) -> Vec<Crossing> {
    intersect_with_density(a, b, tolerance, DEFAULT_SEGMENTS_PER_RADIAN)
}

/// [`intersect`], with control over how finely sampled pairs are cut.
pub fn intersect_with_density(
    a: &Curve,
    b: &Curve,
    tolerance: Tolerance,
    density: f64,
) -> Vec<Crossing> {
    let mut crossings: Vec<Crossing> = candidates(a, b, density)
        .into_iter()
        .filter_map(|point| {
            let t_a = on_curve(a, point, tolerance)?;
            let t_b = on_curve(b, point, tolerance)?;
            Some(Crossing { point, t_a, t_b })
        })
        .collect();

    crossings.sort_by(|x, y| x.t_a.partial_cmp(&y.t_a).unwrap_or(std::cmp::Ordering::Equal));
    crossings.dedup_by(|x, y| {
        let dx = x.point[0] - y.point[0];
        let dy = x.point[1] - y.point[1];
        (dx * dx + dy * dy).sqrt() <= tolerance.linear()
    });
    crossings
}

/// The parameter at `point` if it really lies on `curve`, else `None`.
fn on_curve(curve: &Curve, point: [f64; 2], tolerance: Tolerance) -> Option<f64> {
    let t = curve.parameter_at(point);
    if !curve.is_closed() {
        // A parameter past either end means the crossing is on the curve's
        // extension, not on the curve.
        let slack = 1e-9;
        if t < -slack || t > 1.0 + slack {
            return None;
        }
    }
    // Evaluating the parameter has to land back where we started. For a point
    // on an arc's circle but outside the arc, `parameter_at` clamps and this
    // is what catches it.
    let back = curve.point_at(t);
    let dx = back[0] - point[0];
    let dy = back[1] - point[1];
    ((dx * dx + dy * dy).sqrt() <= tolerance.linear()).then(|| t.clamp(0.0, 1.0))
}

/// Candidate crossing points, exact where a closed form exists.
fn candidates(a: &Curve, b: &Curve, density: f64) -> Vec<[f64; 2]> {
    use Curve::{Arc, Circle, Ellipse, Line};

    match (a, b) {
        (Line(_), Line(_)) => {
            let (p, d) = ray(a);
            let (q, e) = ray(b);
            line_line(p, d, q, e)
                .map(|(t, _)| vec![[p[0] + t * d[0], p[1] + t * d[1]]])
                .unwrap_or_default()
        }

        (Line(_), Circle(_) | Arc(_)) => {
            let (p, d) = ray(a);
            let (centre, radius) = circle_of(b);
            line_circle(p, d, centre, radius)
                .into_iter()
                .map(|t| [p[0] + t * d[0], p[1] + t * d[1]])
                .collect()
        }
        (Circle(_) | Arc(_), Line(_)) => candidates(b, a, density),

        (Line(_), Ellipse(arc)) => {
            let (p, d) = ray(a);
            line_ellipse(p, d, &arc.ellipse)
                .into_iter()
                .map(|(_, parameter)| arc.ellipse.point_at(parameter))
                .collect()
        }
        (Ellipse(_), Line(_)) => candidates(b, a, density),

        (Circle(_) | Arc(_), Circle(_) | Arc(_)) => {
            let (c1, r1) = circle_of(a);
            let (c2, r2) = circle_of(b);
            circle_circle_points(c1, r1, c2, r2)
        }

        // No closed form worth the trouble, or a polyline on either side:
        // sample and cross the segments.
        _ => sampled(a, b, density),
    }
}

/// A point and direction spanning a straight curve.
fn ray(curve: &Curve) -> ([f64; 2], [f64; 2]) {
    match curve {
        Curve::Line(line) => (line.start, line.direction()),
        _ => unreachable!("ray is only called for lines"),
    }
}

/// The circle a circle or arc lies on.
fn circle_of(curve: &Curve) -> ([f64; 2], f64) {
    match curve {
        Curve::Circle(circle) => (circle.centre, circle.radius),
        Curve::Arc(arc) => (arc.centre, arc.radius),
        _ => unreachable!("circle_of is only called for circles and arcs"),
    }
}

/// Samples both curves and crosses the resulting segments.
fn sampled(a: &Curve, b: &Curve, density: f64) -> Vec<[f64; 2]> {
    let first = a.tessellate(density);
    let second = b.tessellate(density);
    let mut out = Vec::new();
    for pair_a in first.windows(2) {
        let p = pair_a[0];
        let d = [pair_a[1][0] - p[0], pair_a[1][1] - p[1]];
        for pair_b in second.windows(2) {
            let q = pair_b[0];
            let e = [pair_b[1][0] - q[0], pair_b[1][1] - q[1]];
            let Some((t, u)) = line_line(p, d, q, e) else {
                continue;
            };
            // Both hits have to be within the sampled segments, not on their
            // extensions, or the sampling would invent crossings.
            if !(-1e-9..=1.0 + 1e-9).contains(&t) || !(-1e-9..=1.0 + 1e-9).contains(&u) {
                continue;
            }
            out.push([p[0] + t * d[0], p[1] + t * d[1]]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom2d::curve::{Arc as ArcCurve, Circle as CircleShape, EllipseArc, Line as Seg};
    use crate::geom2d::polyline::{Polyline as Chain, PolylineVertex};
    use crate::geom2d::Ellipse;
    use std::f64::consts::FRAC_PI_2;

    fn tol() -> Tolerance {
        Tolerance::new(1e-6)
    }

    fn segment(start: [f64; 2], end: [f64; 2]) -> Curve {
        Curve::Line(Seg { start, end })
    }

    fn circle(centre: [f64; 2], radius: f64) -> Curve {
        Curve::Circle(CircleShape { centre, radius })
    }

    #[test]
    fn two_crossing_segments_meet_once() {
        let hits = intersect(
            &segment([0.0, 0.0], [10.0, 0.0]),
            &segment([5.0, -5.0], [5.0, 5.0]),
            tol(),
        );
        assert_eq!(hits.len(), 1);
        assert!((hits[0].point[0] - 5.0).abs() < 1e-9);
        assert!((hits[0].t_a - 0.5).abs() < 1e-9);
        assert!((hits[0].t_b - 0.5).abs() < 1e-9);
    }

    #[test]
    fn segments_that_would_only_meet_if_extended_do_not_count() {
        // They cross at x = 50, well past the first segment's end.
        let hits = intersect(
            &segment([0.0, 0.0], [10.0, 0.0]),
            &segment([50.0, -5.0], [50.0, 5.0]),
            tol(),
        );
        assert!(hits.is_empty(), "got {hits:?}");
    }

    #[test]
    fn a_segment_through_a_circle_meets_it_twice_in_order() {
        let hits = intersect(&segment([-10.0, 0.0], [10.0, 0.0]), &circle([0.0, 0.0], 5.0), tol());
        assert_eq!(hits.len(), 2);
        assert!(hits[0].t_a < hits[1].t_a, "should be ordered along the line");
        assert!((hits[0].point[0] + 5.0).abs() < 1e-6);
        assert!((hits[1].point[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn the_order_of_the_arguments_does_not_change_the_hits() {
        let line = segment([-10.0, 0.0], [10.0, 0.0]);
        let round = circle([0.0, 0.0], 5.0);
        let forward = intersect(&line, &round, tol());
        let backward = intersect(&round, &line, tol());
        assert_eq!(forward.len(), backward.len());
        for hit in &forward {
            assert!(
                backward.iter().any(|other| {
                    (other.point[0] - hit.point[0]).abs() < 1e-6
                        && (other.point[1] - hit.point[1]).abs() < 1e-6
                }),
                "missing {hit:?}"
            );
        }
    }

    #[test]
    fn a_point_on_an_arcs_circle_but_off_the_arc_is_rejected() {
        // Quarter arc in the first quadrant; the line crosses the circle on
        // the other side as well, and only the first quadrant hit counts.
        let arc = Curve::Arc(ArcCurve {
            centre: [0.0, 0.0],
            radius: 5.0,
            start_angle: 0.0,
            end_angle: FRAC_PI_2,
        });
        let hits = intersect(&segment([-10.0, 3.0], [10.0, 3.0]), &arc, tol());
        assert_eq!(hits.len(), 1, "got {hits:?}");
        assert!(hits[0].point[0] > 0.0, "should be the first-quadrant hit");
    }

    #[test]
    fn two_circles_cross_at_two_points() {
        let hits = intersect(&circle([0.0, 0.0], 1.0), &circle([1.0, 0.0], 1.0), tol());
        assert_eq!(hits.len(), 2);
        for hit in hits {
            assert!((hit.point[0] - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn a_segment_across_an_ellipse_meets_it_twice() {
        let ellipse = Curve::Ellipse(EllipseArc::full(Ellipse {
            centre: [0.0, 0.0],
            major_radius: 5.0,
            minor_radius: 2.0,
            major_axis: [1.0, 0.0],
        }));
        let hits = intersect(&segment([-10.0, 1.0], [10.0, 1.0]), &ellipse, tol());
        assert_eq!(hits.len(), 2, "got {hits:?}");
        for hit in hits {
            assert!((hit.point[1] - 1.0).abs() < 1e-6);
            // Back through the ellipse's own parameterisation.
            assert!((ellipse.point_at(hit.t_b)[1] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn a_polyline_crossing_falls_on_the_right_segment() {
        // Three unit segments along +X; a vertical line cuts the middle one.
        let chain = Curve::Polyline(Chain {
            vertices: vec![
                PolylineVertex::straight([0.0, 0.0]),
                PolylineVertex::straight([1.0, 0.0]),
                PolylineVertex::straight([2.0, 0.0]),
                PolylineVertex::straight([3.0, 0.0]),
            ],
            closed: false,
        });
        let hits = intersect(&chain, &segment([1.5, -1.0], [1.5, 1.0]), tol());
        assert_eq!(hits.len(), 1, "got {hits:?}");
        // Halfway along the second of three segments.
        assert!((hits[0].t_a - 0.5).abs() < 1e-6, "t_a = {}", hits[0].t_a);
    }

    #[test]
    fn parallel_segments_never_cross() {
        assert!(intersect(
            &segment([0.0, 0.0], [10.0, 0.0]),
            &segment([0.0, 3.0], [10.0, 3.0]),
            tol()
        )
        .is_empty());
    }

    #[test]
    fn a_circle_that_misses_reports_nothing() {
        assert!(intersect(
            &segment([-10.0, 9.0], [10.0, 9.0]),
            &circle([0.0, 0.0], 5.0),
            tol()
        )
        .is_empty());
    }

    #[test]
    fn the_reported_parameters_reproduce_the_point_on_both_curves() {
        let a = segment([-4.0, -1.0], [6.0, 4.0]);
        let b = circle([1.0, 1.0], 2.0);
        let hits = intersect(&a, &b, tol());
        assert!(!hits.is_empty());
        for hit in hits {
            let from_a = a.point_at(hit.t_a);
            let from_b = b.point_at(hit.t_b);
            assert!((from_a[0] - hit.point[0]).abs() < 1e-6);
            assert!((from_b[1] - hit.point[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn a_tangent_line_touches_once_rather_than_twice() {
        let hits = intersect(&segment([-10.0, 5.0], [10.0, 5.0]), &circle([0.0, 0.0], 5.0), tol());
        assert_eq!(hits.len(), 1, "got {hits:?}");
    }

    #[test]
    fn survey_coordinates_still_cross_where_they_should() {
        let origin = [512_345.678, 4_512_345.678];
        let hits = intersect(
            &segment(
                [origin[0] - 10.0, origin[1]],
                [origin[0] + 10.0, origin[1]],
            ),
            &circle(origin, 5.0),
            tol(),
        );
        assert_eq!(hits.len(), 2);
        assert!((hits[0].point[0] - (origin[0] - 5.0)).abs() < 1e-6);
        assert!((hits[1].point[0] - (origin[0] + 5.0)).abs() < 1e-6);
    }
}
