//! NURBS curves.
//!
//! A spline arrives from a drawing as a degree, a knot vector, control points
//! and — if it is rational — a weight per control point. Evaluating it is de
//! Boor's algorithm, which is short and exact, so there is no reason for this
//! to be the one curve the kernel can only approximate.
//!
//! # Weights are not optional
//!
//! Dropping the weights and evaluating a rational curve as though it were
//! polynomial is a quiet, plausible-looking wrong answer: the curve still
//! passes near its control points and still looks like a spline, but conic
//! sections written as NURBS — which is how a great many CAD circles and
//! ellipses arrive once they have been through a modeller — come out visibly
//! off. Weights are carried here and applied in homogeneous coordinates.

/// A non-uniform rational B-spline curve in the plane.
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsCurve {
    degree: usize,
    knots: Vec<f64>,
    control_points: Vec<[f64; 2]>,
    weights: Vec<f64>,
}

impl NurbsCurve {
    /// Builds a curve, filling in what the caller left out.
    ///
    /// A knot vector of the wrong length is replaced with a clamped uniform
    /// one, which is what a drawing with a malformed spline needs in order to
    /// still draw something sensible. Weights are optional; absent means all
    /// ones, and the curve is polynomial.
    ///
    /// `None` when there are too few control points for the degree, since
    /// there is no curve to evaluate.
    pub fn new(
        degree: usize,
        control_points: Vec<[f64; 2]>,
        knots: Vec<f64>,
        weights: Option<Vec<f64>>,
    ) -> Option<Self> {
        if degree == 0 || control_points.len() <= degree {
            return None;
        }
        let expected = control_points.len() + degree + 1;
        let knots = if knots.len() == expected {
            knots
        } else {
            clamped_uniform_knots(degree, control_points.len())
        };
        let weights = match weights {
            Some(w) if w.len() == control_points.len() && w.iter().all(|v| *v > 0.0) => w,
            _ => vec![1.0; control_points.len()],
        };
        Some(Self {
            degree,
            knots,
            control_points,
            weights,
        })
    }

    /// The degree.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The control points, in order.
    pub fn control_points(&self) -> &[[f64; 2]] {
        &self.control_points
    }

    /// The weights, one per control point. All ones for a polynomial curve.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Whether any weight differs from the others, making the curve rational.
    pub fn is_rational(&self) -> bool {
        let first = self.weights[0];
        self.weights.iter().any(|w| (w - first).abs() > 1e-12)
    }

    /// The knot values the curve is actually defined over, `(start, end)`.
    ///
    /// The knot vector runs wider than this; the first and last `degree` knots
    /// exist to define the basis, not to be evaluated at.
    pub fn domain(&self) -> (f64, f64) {
        (
            self.knots[self.degree],
            self.knots[self.control_points.len()],
        )
    }

    /// The point at knot value `u`, clamped to the curve's [`domain`].
    ///
    /// This is de Boor's algorithm run in homogeneous coordinates, so the
    /// weights are honoured rather than ignored.
    pub fn point_at_knot(&self, u: f64) -> [f64; 2] {
        let (start, end) = self.domain();
        let u = u.clamp(start, end);
        let span = self.span_containing(u);

        // Lift into (w·x, w·y, w), interpolate there, project back.
        let mut points: Vec<[f64; 3]> = (0..=self.degree)
            .map(|j| {
                let index = span + j - self.degree;
                let w = self.weights[index];
                let p = self.control_points[index];
                [p[0] * w, p[1] * w, w]
            })
            .collect();

        for round in 1..=self.degree {
            for j in (round..=self.degree).rev() {
                let index = span + j - self.degree;
                let lower = self.knots[index];
                let upper = self.knots[index + self.degree + 1 - round];
                let alpha = if (upper - lower).abs() < 1e-15 {
                    0.0
                } else {
                    (u - lower) / (upper - lower)
                };
                let (previous, current) = (points[j - 1], points[j]);
                for (slot, (before, after)) in points[j]
                    .iter_mut()
                    .zip(previous.iter().zip(current.iter()))
                {
                    *slot = (1.0 - alpha) * before + alpha * after;
                }
            }
        }

        let result = points[self.degree];
        if result[2].abs() < 1e-15 {
            [result[0], result[1]]
        } else {
            [result[0] / result[2], result[1] / result[2]]
        }
    }

    /// The point at `t`, which runs `0..=1` across the [`domain`].
    pub fn point_at(&self, t: f64) -> [f64; 2] {
        let (start, end) = self.domain();
        self.point_at_knot(start + t.clamp(0.0, 1.0) * (end - start))
    }

    /// Where `point` falls on the curve, as `0..=1`.
    ///
    /// Found by sampling and then narrowing, because a NURBS curve has no
    /// closed-form inverse. The result is the nearest point on the curve, so a
    /// point that is not on it is projected rather than rejected.
    pub fn parameter_at(&self, point: [f64; 2]) -> f64 {
        let coarse = (self.control_points.len() * 8).max(64);
        let distance_at = |t: f64| {
            let on_curve = self.point_at(t);
            let dx = on_curve[0] - point[0];
            let dy = on_curve[1] - point[1];
            dx * dx + dy * dy
        };

        let mut best = 0.0;
        let mut best_distance = f64::INFINITY;
        for i in 0..=coarse {
            let t = i as f64 / coarse as f64;
            let d = distance_at(t);
            if d < best_distance {
                best_distance = d;
                best = t;
            }
        }

        // Narrow within the neighbouring samples. Bisection on the bracket
        // rather than Newton: it needs no derivative and cannot diverge where
        // the curve doubles back on itself.
        let step = 1.0 / coarse as f64;
        let (mut low, mut high) = ((best - step).max(0.0), (best + step).min(1.0));
        for _ in 0..60 {
            if high - low < 1e-15 {
                break;
            }
            let third = (high - low) / 3.0;
            let left = low + third;
            let right = high - third;
            if distance_at(left) < distance_at(right) {
                high = right;
            } else {
                low = left;
            }
            best = (low + high) * 0.5;
        }
        best.clamp(0.0, 1.0)
    }

    /// Samples the curve, giving every knot span at least `per_span` pieces.
    ///
    /// Sampling per span rather than uniformly means a curve whose knots
    /// bunch up — which is where its shape changes fastest — is cut finely
    /// exactly there.
    pub fn tessellate(&self, per_span: usize) -> Vec<[f64; 2]> {
        let per_span = per_span.max(1);
        let (start, end) = self.domain();
        let mut values: Vec<f64> = self.knots[self.degree..=self.control_points.len()]
            .iter()
            .copied()
            .filter(|u| *u >= start && *u <= end)
            .collect();
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        if values.len() < 2 {
            values = vec![start, end];
        }

        let mut out = Vec::new();
        for pair in values.windows(2) {
            for step in 0..per_span {
                let f = step as f64 / per_span as f64;
                out.push(self.point_at_knot(pair[0] + f * (pair[1] - pair[0])));
            }
        }
        out.push(self.point_at_knot(end));
        out
    }

    /// Whether the curve comes back to where it started.
    pub fn is_closed(&self) -> bool {
        let start = self.point_at(0.0);
        let end = self.point_at(1.0);
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        (dx * dx + dy * dy).sqrt() < 1e-9
    }

    /// Index of the knot span holding `u`.
    fn span_containing(&self, u: f64) -> usize {
        let last = self.control_points.len() - 1;
        if u >= self.knots[last + 1] {
            return last;
        }
        let mut low = self.degree;
        let mut high = last + 1;
        while high - low > 1 {
            let middle = (low + high) / 2;
            if u < self.knots[middle] {
                high = middle;
            } else {
                low = middle;
            }
        }
        low
    }
}

/// A clamped uniform knot vector: `degree + 1` zeros, evenly spaced interior
/// values, then `degree + 1` ones.
///
/// Clamped so the curve starts at its first control point and ends at its
/// last, which is what a drawing expects to see.
fn clamped_uniform_knots(degree: usize, control_point_count: usize) -> Vec<f64> {
    let interior = control_point_count.saturating_sub(degree + 1);
    let mut knots = vec![0.0; degree + 1];
    for i in 1..=interior {
        knots.push(i as f64 / (interior + 1) as f64);
    }
    knots.extend(std::iter::repeat_n(1.0, degree + 1));
    knots
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 2], b: [f64; 2], slack: f64) -> bool {
        (a[0] - b[0]).abs() < slack && (a[1] - b[1]).abs() < slack
    }

    /// Degree 1 is a polyline, which makes the expected answers obvious.
    fn linear() -> NurbsCurve {
        NurbsCurve::new(
            1,
            vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
            Vec::new(),
            None,
        )
        .unwrap()
    }

    #[test]
    fn a_clamped_curve_starts_and_ends_on_its_outer_control_points() {
        let curve = NurbsCurve::new(
            3,
            vec![[0.0, 0.0], [1.0, 5.0], [4.0, 5.0], [5.0, 0.0]],
            Vec::new(),
            None,
        )
        .unwrap();
        assert!(close(curve.point_at(0.0), [0.0, 0.0], 1e-9));
        assert!(close(curve.point_at(1.0), [5.0, 0.0], 1e-9));
    }

    #[test]
    fn degree_one_reproduces_the_control_polygon() {
        let curve = linear();
        assert!(close(curve.point_at(0.0), [0.0, 0.0], 1e-9));
        assert!(close(curve.point_at(0.5), [10.0, 0.0], 1e-9));
        assert!(close(curve.point_at(1.0), [10.0, 10.0], 1e-9));
        assert!(close(curve.point_at(0.25), [5.0, 0.0], 1e-9));
    }

    #[test]
    fn a_curve_stays_inside_its_control_polygon_bounds() {
        let curve = NurbsCurve::new(
            3,
            vec![[0.0, 0.0], [1.0, 5.0], [4.0, 5.0], [5.0, 0.0]],
            Vec::new(),
            None,
        )
        .unwrap();
        for i in 0..=50 {
            let p = curve.point_at(i as f64 / 50.0);
            assert!((0.0..=5.0).contains(&p[0]), "x escaped: {p:?}");
            assert!((0.0..=5.0).contains(&p[1]), "y escaped: {p:?}");
        }
    }

    #[test]
    fn weights_are_applied_rather_than_ignored() {
        // The standard rational quarter circle: three control points, the
        // middle one weighted 1/sqrt(2). Evaluated as a polynomial it misses
        // the unit circle badly; evaluated rationally it lies on it.
        let weight = std::f64::consts::FRAC_1_SQRT_2;
        let control = vec![[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let rational =
            NurbsCurve::new(2, control.clone(), knots.clone(), Some(vec![1.0, weight, 1.0]))
                .unwrap();
        assert!(rational.is_rational());
        for i in 0..=20 {
            let p = rational.point_at(i as f64 / 20.0);
            let radius = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                (radius - 1.0).abs() < 1e-9,
                "rational curve left the unit circle: {radius}"
            );
        }

        // Same control net without weights is a parabola, not a circle — the
        // difference this test exists to keep.
        let polynomial = NurbsCurve::new(2, control, knots, None).unwrap();
        assert!(!polynomial.is_rational());
        let middle = polynomial.point_at(0.5);
        let radius = (middle[0] * middle[0] + middle[1] * middle[1]).sqrt();
        assert!(
            (radius - 1.0).abs() > 0.05,
            "expected the unweighted curve to miss the circle, got {radius}"
        );
    }

    #[test]
    fn a_malformed_knot_vector_falls_back_to_a_clamped_uniform_one() {
        let curve = NurbsCurve::new(
            2,
            vec![[0.0, 0.0], [1.0, 2.0], [2.0, 0.0], [3.0, 2.0]],
            vec![0.0, 1.0], // nowhere near the right length
            None,
        )
        .unwrap();
        assert!(close(curve.point_at(0.0), [0.0, 0.0], 1e-9));
        assert!(close(curve.point_at(1.0), [3.0, 2.0], 1e-9));
    }

    #[test]
    fn too_few_control_points_is_not_a_curve() {
        assert!(NurbsCurve::new(3, vec![[0.0, 0.0], [1.0, 1.0]], Vec::new(), None).is_none());
        assert!(NurbsCurve::new(0, vec![[0.0, 0.0], [1.0, 1.0]], Vec::new(), None).is_none());
    }

    #[test]
    fn nonsense_weights_are_discarded_rather_than_dividing_by_zero() {
        let curve = NurbsCurve::new(
            2,
            vec![[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]],
            Vec::new(),
            Some(vec![1.0, 0.0, 1.0]),
        )
        .unwrap();
        assert!(!curve.is_rational(), "a zero weight should be refused");
        assert!(curve.point_at(0.5)[1].is_finite());
    }

    #[test]
    fn the_parameter_inverts_the_evaluation() {
        let curve = NurbsCurve::new(
            3,
            vec![[0.0, 0.0], [1.0, 5.0], [4.0, 5.0], [5.0, 0.0], [8.0, -3.0]],
            Vec::new(),
            None,
        )
        .unwrap();
        for i in 1..10 {
            let t = i as f64 / 10.0;
            let point = curve.point_at(t);
            let recovered = curve.parameter_at(point);
            assert!(
                (recovered - t).abs() < 1e-4,
                "asked for {t}, got back {recovered}"
            );
        }
    }

    #[test]
    fn a_point_off_the_curve_projects_onto_it() {
        let curve = linear();
        // Above the middle of the first, horizontal, segment and nearer to it
        // than to the vertical one.
        let t = curve.parameter_at([5.0, 3.0]);
        assert!(
            close(curve.point_at(t), [5.0, 0.0], 1e-3),
            "expected the foot of the perpendicular, got {:?}",
            curve.point_at(t)
        );
    }

    #[test]
    fn tessellation_keeps_both_ends_and_stays_on_the_curve() {
        let curve = NurbsCurve::new(
            3,
            vec![[0.0, 0.0], [1.0, 5.0], [4.0, 5.0], [5.0, 0.0], [8.0, -3.0]],
            Vec::new(),
            None,
        )
        .unwrap();
        let points = curve.tessellate(8);
        assert!(close(points[0], curve.point_at(0.0), 1e-9));
        assert!(close(*points.last().unwrap(), curve.point_at(1.0), 1e-9));
        assert!(points.len() > 8, "expected a span per knot interval");
    }

    #[test]
    fn a_closed_curve_is_recognised() {
        let open = linear();
        assert!(!open.is_closed());

        let closed = NurbsCurve::new(
            1,
            vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 0.0]],
            Vec::new(),
            None,
        )
        .unwrap();
        assert!(closed.is_closed());
    }

    #[test]
    fn survey_coordinates_evaluate_without_drift() {
        let origin = [512_345.678, 4_512_345.678];
        let curve = NurbsCurve::new(
            3,
            vec![
                origin,
                [origin[0] + 1.0, origin[1] + 5.0],
                [origin[0] + 4.0, origin[1] + 5.0],
                [origin[0] + 5.0, origin[1]],
            ],
            Vec::new(),
            None,
        )
        .unwrap();
        assert!(close(curve.point_at(0.0), origin, 1e-6));
        assert!(
            close(curve.point_at(1.0), [origin[0] + 5.0, origin[1]], 1e-6),
            "end drifted"
        );
    }
}
