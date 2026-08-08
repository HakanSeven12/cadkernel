//! Curve–curve intersection in the plane.
//!
//! This is the workhorse the editing commands sit on: trimming, extending,
//! breaking, filleting, snapping to an intersection and resolving a hatch
//! boundary all reduce to asking where two curves meet.
//!
//! Every function returns *parameters*, not points. A trim needs to know
//! where along the curve the hit landed so it can cut there, and recovering a
//! parameter from a returned point means solving the same problem twice.
//! Callers that want coordinates evaluate the curve at the parameter.
//!
//! # On the epsilons here
//!
//! The thresholds below compare determinants and discriminants against zero.
//! They are not distances, so they deliberately do not come from
//! [`Tolerance`](crate::geom2d::Tolerance) — feeding a millimetre into a
//! determinant test would make the test scale with the drawing's units and
//! mean nothing.

use super::Ellipse;

/// Below this, two directions are parallel and there is no single crossing.
const DETERMINANT_EPSILON: f64 = 1e-10;

/// Below this, a quadratic has a double root: the line grazes the conic.
const DISCRIMINANT_EPSILON: f64 = 1e-14;

/// Below this, a direction is too short to define a line at all.
const DEGENERATE_DIRECTION: f64 = 1e-20;

/// Distance below which two centres coincide, or two radii touch exactly.
const PROXIMITY_EPSILON: f64 = 1e-9;

/// Where two infinite lines cross, as `(t, u)`.
///
/// The lines are `p + t·d` and `q + t·e`. Returns `None` when they are
/// parallel, including when they are the same line — a coincident pair has no
/// single crossing, and callers that care about overlap need to test for it
/// separately rather than read it out of an intersection result.
///
/// The parameters are unbounded, so a caller working with segments or arcs
/// has to range-check them itself. That is the point: extending a line to
/// meet another one needs the intersection that lies *past* the end.
pub fn line_line(p: [f64; 2], d: [f64; 2], q: [f64; 2], e: [f64; 2]) -> Option<(f64, f64)> {
    let determinant = d[0] * e[1] - d[1] * e[0];
    if determinant.abs() < DETERMINANT_EPSILON {
        return None;
    }
    let dx = q[0] - p[0];
    let dy = q[1] - p[1];
    Some((
        (dx * e[1] - dy * e[0]) / determinant,
        (dx * d[1] - dy * d[0]) / determinant,
    ))
}

/// Where the line `p + t·d` meets the circle at `centre` with `radius`.
///
/// Returns the line parameters: empty when they miss, one value where the
/// line is tangent, two otherwise, ordered by increasing `t`.
pub fn line_circle(p: [f64; 2], d: [f64; 2], centre: [f64; 2], radius: f64) -> Vec<f64> {
    let fx = p[0] - centre[0];
    let fy = p[1] - centre[1];
    let a = d[0] * d[0] + d[1] * d[1];
    if a < DEGENERATE_DIRECTION {
        return Vec::new();
    }
    let b = 2.0 * (fx * d[0] + fy * d[1]);
    let c = fx * fx + fy * fy - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return Vec::new();
    }
    if discriminant < DISCRIMINANT_EPSILON {
        return vec![-b / (2.0 * a)];
    }
    let root = discriminant.sqrt();
    vec![(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
}

/// Where two circles meet, as angles measured on the **first** circle.
///
/// Returns angles rather than points because the caller is almost always
/// trimming or splitting the first circle, and an angle is what an arc is cut
/// at. Empty when the circles miss, are nested, or are concentric; one angle
/// when they touch.
pub fn circle_circle_angles(
    centre1: [f64; 2],
    radius1: f64,
    centre2: [f64; 2],
    radius2: f64,
) -> Vec<f64> {
    let span_x = centre2[0] - centre1[0];
    let span_y = centre2[1] - centre1[1];
    let distance = (span_x * span_x + span_y * span_y).sqrt();

    // Concentric (no discrete solutions, even when the radii match and the
    // circles coincide), too far apart, or one swallowed by the other.
    if distance < PROXIMITY_EPSILON
        || distance > radius1 + radius2 + PROXIMITY_EPSILON
        || distance < (radius1 - radius2).abs() - PROXIMITY_EPSILON
    {
        return Vec::new();
    }

    // Distance from centre1 to the radical line, then the half-chord on it.
    let along = (radius1 * radius1 - radius2 * radius2 + distance * distance) / (2.0 * distance);
    let half_chord_squared = radius1 * radius1 - along * along;
    if half_chord_squared < 0.0 {
        return Vec::new();
    }
    let half_chord = half_chord_squared.sqrt();

    let mid_x = centre1[0] + along * span_x / distance;
    let mid_y = centre1[1] + along * span_y / distance;
    let offset_x = half_chord * span_y / distance;
    let offset_y = -half_chord * span_x / distance;

    let first = ((mid_y + offset_y) - centre1[1]).atan2((mid_x + offset_x) - centre1[0]);
    if half_chord < PROXIMITY_EPSILON {
        return vec![first];
    }
    let second = ((mid_y - offset_y) - centre1[1]).atan2((mid_x - offset_x) - centre1[0]);
    vec![first, second]
}

/// Where the line `p + s·d` meets an ellipse, as `(s, t)` pairs.
///
/// `t` is the ellipse's own parameter, so the point is
/// [`Ellipse::point_at(t)`](Ellipse::point_at).
///
/// Works by squashing the ellipse into a unit circle, solving there, and
/// reading the parameter off the squashed coordinates — on the unit circle
/// they *are* `(cos t, sin t)`.
///
/// The parameter has to come from the squashed pair, not the unscaled local
/// one. `atan2(y, x)` on unscaled coordinates is the geometric angle at the
/// centre, which only equals the parameter on a circle: for a 5:2 ellipse the
/// two differ by more than twenty degrees near the diagonals.
pub fn line_ellipse(p: [f64; 2], d: [f64; 2], ellipse: &Ellipse) -> Vec<(f64, f64)> {
    if ellipse.is_degenerate() {
        return Vec::new();
    }
    let Ellipse {
        centre,
        major_radius,
        minor_radius,
        major_axis,
    } = *ellipse;
    let (nx, ny) = (major_axis[0], major_axis[1]);

    // Into the ellipse's own frame.
    let rx = p[0] - centre[0];
    let ry = p[1] - centre[1];
    let local_x = rx * nx + ry * ny;
    let local_y = -rx * ny + ry * nx;
    let dir_x = d[0] * nx + d[1] * ny;
    let dir_y = -d[0] * ny + d[1] * nx;

    // Then scaled so the ellipse becomes the unit circle.
    let unit_x = local_x / major_radius;
    let unit_dx = dir_x / major_radius;
    let unit_y = local_y / minor_radius;
    let unit_dy = dir_y / minor_radius;

    let a = unit_dx * unit_dx + unit_dy * unit_dy;
    if a < DEGENERATE_DIRECTION {
        return Vec::new();
    }
    let b = 2.0 * (unit_x * unit_dx + unit_y * unit_dy);
    let c = unit_x * unit_x + unit_y * unit_y - 1.0;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return Vec::new();
    }

    let line_params = if discriminant < DISCRIMINANT_EPSILON {
        vec![-b / (2.0 * a)]
    } else {
        let root = discriminant.sqrt();
        vec![(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    };

    line_params
        .into_iter()
        .map(|s| {
            let cos_t = unit_x + s * unit_dx;
            let sin_t = unit_y + s * unit_dy;
            (s, sin_t.atan2(cos_t))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn crossing_lines_report_both_parameters() {
        let hit = line_line([0.0, 0.0], [1.0, 0.0], [5.0, -5.0], [0.0, 1.0]).unwrap();
        assert!((hit.0 - 5.0).abs() < 1e-12);
        assert!((hit.1 - 5.0).abs() < 1e-12);
    }

    #[test]
    fn parallel_lines_have_no_crossing() {
        assert!(line_line([0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [1.0, 0.0]).is_none());
    }

    #[test]
    fn coincident_lines_report_no_crossing_rather_than_one() {
        assert!(line_line([0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 0.0]).is_none());
    }

    #[test]
    fn parameters_run_past_the_ends_so_extend_can_use_them() {
        // The crossing sits well beyond p + d, which is what EXTEND needs.
        let (t, _) = line_line([0.0, 0.0], [1.0, 0.0], [100.0, -1.0], [0.0, 1.0]).unwrap();
        assert!(t > 1.0, "expected a parameter past the end, got {t}");
    }

    #[test]
    fn a_line_through_a_circle_gives_two_ordered_parameters() {
        let hits = line_circle([-10.0, 0.0], [1.0, 0.0], [0.0, 0.0], 5.0);
        assert_eq!(hits.len(), 2);
        assert!(hits[0] < hits[1]);
        assert!((hits[0] - 5.0).abs() < 1e-9);
        assert!((hits[1] - 15.0).abs() < 1e-9);
    }

    #[test]
    fn a_tangent_line_gives_one_parameter() {
        let hits = line_circle([-10.0, 5.0], [1.0, 0.0], [0.0, 0.0], 5.0);
        assert_eq!(hits.len(), 1);
        assert!((hits[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn a_missing_line_gives_nothing() {
        assert!(line_circle([-10.0, 9.0], [1.0, 0.0], [0.0, 0.0], 5.0).is_empty());
    }

    #[test]
    fn a_zero_length_direction_is_refused_rather_than_dividing_by_zero() {
        assert!(line_circle([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 5.0).is_empty());
    }

    #[test]
    fn two_circles_meet_at_mirrored_angles() {
        // Unit circles at 0 and 1 cross at ±60°.
        let angles = circle_circle_angles([0.0, 0.0], 1.0, [1.0, 0.0], 1.0);
        assert_eq!(angles.len(), 2);
        let mut cosines: Vec<f64> = angles.iter().map(|a| a.cos()).collect();
        cosines.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((cosines[0] - 0.5).abs() < 1e-9);
        assert!((cosines[1] - 0.5).abs() < 1e-9);
        let sines: Vec<f64> = angles.iter().map(|a| a.sin()).collect();
        assert!((sines[0] + sines[1]).abs() < 1e-9, "should be mirrored");
    }

    #[test]
    fn touching_circles_meet_once() {
        let angles = circle_circle_angles([0.0, 0.0], 1.0, [2.0, 0.0], 1.0);
        assert_eq!(angles.len(), 1);
        assert!(angles[0].abs() < 1e-4);
    }

    #[test]
    fn separated_nested_and_concentric_circles_all_report_nothing() {
        assert!(circle_circle_angles([0.0, 0.0], 1.0, [10.0, 0.0], 1.0).is_empty());
        assert!(circle_circle_angles([0.0, 0.0], 5.0, [0.5, 0.0], 1.0).is_empty());
        assert!(circle_circle_angles([0.0, 0.0], 1.0, [0.0, 0.0], 1.0).is_empty());
    }

    #[test]
    fn a_line_through_an_ellipse_returns_both_parameterisations() {
        // Axis-aligned 3x2 ellipse, horizontal line through the centre.
        let ellipse = Ellipse {
            centre: [0.0, 0.0],
            major_radius: 3.0,
            minor_radius: 2.0,
            major_axis: [1.0, 0.0],
        };
        let hits = line_ellipse([-10.0, 0.0], [1.0, 0.0], &ellipse);
        assert_eq!(hits.len(), 2);
        // Crossings at x = ±3, so t = pi then 0.
        let mut xs: Vec<f64> = hits.iter().map(|(s, _)| -10.0 + s).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((xs[0] + 3.0).abs() < 1e-9);
        assert!((xs[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn the_ellipse_parameter_reproduces_the_crossing_point() {
        let ellipse = Ellipse {
            centre: [4.0, -1.0],
            major_radius: 5.0,
            minor_radius: 2.0,
            major_axis: [
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
        };
        // Skew, and through the centre, so it crosses twice well away from the
        // axes — where a parameter mistaken for a centre angle goes furthest
        // wrong.
        let p = [-6.0, -4.0];
        let d = [1.0, 0.3];

        let hits = line_ellipse(p, d, &ellipse);
        assert_eq!(hits.len(), 2, "expected two crossings, got {hits:?}");

        for (s, t) in hits {
            let on_line = [p[0] + s * d[0], p[1] + s * d[1]];
            let on_ellipse = ellipse.point_at(t);
            assert!(
                (on_line[0] - on_ellipse[0]).abs() < 1e-9
                    && (on_line[1] - on_ellipse[1]).abs() < 1e-9,
                "line gave {on_line:?} but ellipse parameter gave {on_ellipse:?}"
            );
        }
    }

    #[test]
    fn the_returned_parameter_is_not_the_centre_angle() {
        // Guards the mistake this function used to make: reading `atan2` off
        // the unscaled local coordinates yields the geometric angle at the
        // centre, which coincides with the parameter only on a circle. On a
        // 5:2 ellipse the gap is tens of degrees, and every consumer that cuts
        // an ellipse at a parameter inherits it.
        let ellipse = Ellipse {
            centre: [0.0, 0.0],
            major_radius: 5.0,
            minor_radius: 2.0,
            major_axis: [1.0, 0.0],
        };
        // Horizontal line at y = 1, crossing the upper half either side of the
        // minor axis.
        let hits = line_ellipse([-10.0, 1.0], [1.0, 0.0], &ellipse);
        assert_eq!(hits.len(), 2);

        for (s, t) in hits {
            let on_line = [-10.0 + s, 1.0];
            let centre_angle = on_line[1].atan2(on_line[0]);
            assert!(
                (t - centre_angle).abs() > 0.1,
                "t {t} collapsed onto the centre angle {centre_angle}"
            );
            // sin(t) = y / minor_radius is the defining relation.
            assert!(
                (t.sin() - 0.5).abs() < 1e-9,
                "sin(t) should be y/minor_radius = 0.5, got {}",
                t.sin()
            );
        }
    }

    #[test]
    fn a_tangent_line_touches_an_ellipse_once() {
        let ellipse = Ellipse {
            centre: [0.0, 0.0],
            major_radius: 3.0,
            minor_radius: 2.0,
            major_axis: [1.0, 0.0],
        };
        let hits = line_ellipse([-10.0, 2.0], [1.0, 0.0], &ellipse);
        assert_eq!(hits.len(), 1);
        assert!((hits[0].1 - FRAC_PI_2).abs() < 1e-4);
    }

    #[test]
    fn a_degenerate_ellipse_is_refused() {
        let flat = Ellipse {
            centre: [0.0, 0.0],
            major_radius: 0.0,
            minor_radius: 2.0,
            major_axis: [1.0, 0.0],
        };
        assert!(line_ellipse([0.0, 0.0], [1.0, 0.0], &flat).is_empty());
    }

    #[test]
    fn intersections_survive_survey_coordinates() {
        // The same crossing, moved out to a UTM-scale origin. This is the case
        // that degrades once anything downstream drops to f32.
        let offset = [512_345.678, 4_512_345.678];
        let hit = line_line(
            offset,
            [1.0, 0.0],
            [offset[0] + 5.0, offset[1] - 5.0],
            [0.0, 1.0],
        )
        .unwrap();
        assert!((hit.0 - 5.0).abs() < 1e-6, "t drifted to {}", hit.0);

        let hits = line_circle([offset[0] - 10.0, offset[1]], [1.0, 0.0], offset, 5.0);
        assert_eq!(hits.len(), 2);
        assert!(
            (hits[0] - 5.0).abs() < 1e-6,
            "near hit drifted to {}",
            hits[0]
        );
        assert!(
            (hits[1] - 15.0).abs() < 1e-6,
            "far hit drifted to {}",
            hits[1]
        );
    }

    #[test]
    fn a_half_turn_apart_pair_of_circles_is_symmetric() {
        let angles = circle_circle_angles([0.0, 0.0], 2.0, [0.0, 3.0], 2.0);
        assert_eq!(angles.len(), 2);
        let sum: f64 = angles.iter().map(|a| a.cos()).sum();
        assert!(sum.abs() < 1e-9, "x components should cancel, got {sum}");
        for angle in angles {
            assert!(angle.sin() > 0.0, "both hits sit above the centre");
        }
    }
}
