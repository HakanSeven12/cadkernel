//! de Boor's algorithm, over whatever a control point happens to be.
//!
//! A B-spline is evaluated by repeatedly taking weighted averages of nearby
//! control points, and nothing in that depends on how many numbers a control
//! point holds. A plane curve averages `(x, y, w)`, a space curve
//! `(x, y, z, w)`, and a surface averages rows of either — so the algorithm is
//! written once here rather than once per dimension.
//!
//! Homogeneous throughout: a rational curve is a polynomial one in a space
//! with an extra coordinate, and dividing by that coordinate at the end is
//! what makes it rational. Keeping the division out of the recursion is what
//! makes a weighted curve come out exactly right rather than nearly.

/// The value of a B-spline at knot parameter `u`.
///
/// `points` are the control points in whatever space they live in, already
/// homogeneous if the curve is rational. `knots` must hold
/// `points.len() + degree + 1` entries; a caller that cannot promise that
/// should build the curve through a type that checks.
pub fn de_boor<const N: usize>(
    degree: usize,
    knots: &[f64],
    points: &[[f64; N]],
    u: f64,
) -> [f64; N] {
    de_boor_by(degree, knots, points.len(), u, |index| points[index])
}

pub(crate) fn de_boor_by<const N: usize, F>(
    degree: usize,
    knots: &[f64],
    point_count: usize,
    u: f64,
    mut point_at: F,
) -> [f64; N]
where
    F: FnMut(usize) -> [f64; N],
{
    if point_count == 0 {
        return [0.0; N];
    }
    if degree == 0 {
        let index = knots
            .iter()
            .rposition(|knot| *knot <= u)
            .unwrap_or(0)
            .min(point_count - 1);
        return point_at(index);
    }
    let last = point_count - 1;
    let span = span_of(degree, knots, last, u);

    let mut working: Vec<[f64; N]> = (0..=degree)
        .map(|step| point_at(span + step - degree))
        .collect();
    for round in 1..=degree {
        for step in (round..=degree).rev() {
            let index = span + step - degree;
            let lower = knots[index];
            let upper = knots[index + degree + 1 - round];
            let alpha = if (upper - lower).abs() < 1e-15 {
                0.0
            } else {
                (u - lower) / (upper - lower)
            };
            let (previous, current) = (working[step - 1], working[step]);
            for (slot, (before, after)) in working[step]
                .iter_mut()
                .zip(previous.iter().zip(current.iter()))
            {
                *slot = (1.0 - alpha) * before + alpha * after;
            }
        }
    }
    working[degree]
}

/// Which knot span `u` falls in, found by bisection.
///
/// Past the end it is the last span rather than none: a parameter at exactly
/// the domain's top is on the curve, and reporting it as outside would leave
/// every curve one point short.
pub fn span_of(degree: usize, knots: &[f64], last: usize, u: f64) -> usize {
    if u >= knots[last + 1] {
        return last;
    }
    let mut low = degree;
    let mut high = last + 1;
    while high - low > 1 {
        let middle = (low + high) / 2;
        if u < knots[middle] {
            high = middle;
        } else {
            low = middle;
        }
    }
    low
}

/// A clamped uniform knot vector: `degree + 1` zeros, evenly spaced interior
/// values, then `degree + 1` ones.
///
/// Clamped so the curve starts at its first control point and ends at its
/// last, which is what every drawing format assumes and what makes a spline
/// joinable to what comes before and after it.
pub fn clamped_uniform_knots(degree: usize, control_point_count: usize) -> Vec<f64> {
    let spans = control_point_count.saturating_sub(degree);
    let mut knots = Vec::with_capacity(control_point_count + degree + 1);
    knots.extend(std::iter::repeat_n(0.0, degree + 1));
    for step in 1..spans {
        knots.push(step as f64 / spans.max(1) as f64);
    }
    knots.extend(std::iter::repeat_n(1.0, degree + 1));
    knots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clamped_curve_starts_and_ends_on_its_own_control_points() {
        let points = [[0.0, 0.0, 1.0], [1.0, 4.0, 1.0], [5.0, 4.0, 1.0], [6.0, 0.0, 1.0]];
        let knots = clamped_uniform_knots(3, 4);
        let start = de_boor(3, &knots, &points, 0.0);
        let end = de_boor(3, &knots, &points, 1.0);
        assert!((start[0]).abs() < 1e-12 && (start[1]).abs() < 1e-12, "{start:?}");
        assert!((end[0] - 6.0).abs() < 1e-12 && (end[1]).abs() < 1e-12, "{end:?}");
    }

    #[test]
    fn the_same_curve_evaluates_alike_however_many_coordinates_it_carries() {
        // The reason this is generic: a plane curve and the same curve with a
        // zero z must agree, and they only do if one algorithm evaluates both.
        let flat = [[0.0, 0.0, 1.0], [2.0, 6.0, 1.0], [8.0, 6.0, 1.0], [10.0, 0.0, 1.0]];
        let spatial: Vec<[f64; 4]> = flat
            .iter()
            .map(|point| [point[0], point[1], 0.0, point[2]])
            .collect();
        let knots = clamped_uniform_knots(3, 4);
        for step in 0..=10 {
            let t = step as f64 / 10.0;
            let here = de_boor(3, &knots, &flat, t);
            let there = de_boor(3, &knots, &spatial, t);
            assert!((here[0] - there[0]).abs() < 1e-12, "t={t}");
            assert!((here[1] - there[1]).abs() < 1e-12, "t={t}");
            assert!(there[2].abs() < 1e-12, "t={t}");
        }
    }

    #[test]
    fn a_weighted_curve_bends_towards_the_heavier_point() {
        // Degree two, three points, the middle one weighted. In homogeneous
        // coordinates that is one control point further out, and dividing at
        // the end is what makes it rational.
        let knots = clamped_uniform_knots(2, 3);
        let plain = [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]];
        let heavy = [[0.0, 0.0, 1.0], [5.0, 5.0, 5.0], [2.0, 0.0, 1.0]];
        let middle = |points: &[[f64; 3]; 3]| {
            let raw = de_boor(2, &knots, points, 0.5);
            [raw[0] / raw[2], raw[1] / raw[2]]
        };
        assert!(middle(&heavy)[1] > middle(&plain)[1]);
    }

    #[test]
    fn a_degree_one_curve_is_the_polyline_through_its_points() {
        let points = [[0.0, 0.0, 1.0], [10.0, 0.0, 1.0], [10.0, 10.0, 1.0]];
        let knots = clamped_uniform_knots(1, 3);
        let half = de_boor(1, &knots, &points, 0.25);
        assert!((half[0] - 5.0).abs() < 1e-12, "{half:?}");
        assert!(half[1].abs() < 1e-12, "{half:?}");
    }

    #[test]
    fn a_span_past_the_end_is_the_last_one_rather_than_none() {
        let knots = clamped_uniform_knots(3, 6);
        assert_eq!(span_of(3, &knots, 5, 1.0), 5);
        assert_eq!(span_of(3, &knots, 5, 2.0), 5);
        assert_eq!(span_of(3, &knots, 5, 0.0), 3);
    }
}
