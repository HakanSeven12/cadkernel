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

use super::vec::Vec2;

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

    /// The knot vector.
    pub fn knots(&self) -> &[f64] {
        &self.knots
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
        let point = Vec2::from(point);
        let distance_at = |t: f64| Vec2::from(self.point_at(t)).distance_squared(point);

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
        Vec2::from(self.point_at(0.0)).distance(Vec2::from(self.point_at(1.0))) < 1e-9
    }

    /// Inserts the knot value `u` once, leaving the curve's shape unchanged.
    ///
    /// Boehm's algorithm: the control points either side of the affected span
    /// are replaced by a blend of their neighbours, in homogeneous coordinates
    /// so a rational curve stays exact. Adding knots is how a curve gets split
    /// without approximating it — see [`split_at`](Self::split_at).
    pub fn insert_knot(&mut self, u: f64) {
        let (start, end) = self.domain();
        if u <= start || u >= end {
            return;
        }
        let span = self.span_containing(u);
        let degree = self.degree;

        let homogeneous: Vec<[f64; 3]> = self
            .control_points
            .iter()
            .zip(&self.weights)
            .map(|(p, w)| [p[0] * w, p[1] * w, *w])
            .collect();

        let mut updated: Vec<[f64; 3]> = Vec::with_capacity(homogeneous.len() + 1);
        updated.extend_from_slice(&homogeneous[..=span - degree]);
        for i in (span - degree + 1)..=span {
            let lower = self.knots[i];
            let upper = self.knots[i + degree];
            let alpha = if (upper - lower).abs() < 1e-15 {
                0.0
            } else {
                (u - lower) / (upper - lower)
            };
            updated.push([
                (1.0 - alpha) * homogeneous[i - 1][0] + alpha * homogeneous[i][0],
                (1.0 - alpha) * homogeneous[i - 1][1] + alpha * homogeneous[i][1],
                (1.0 - alpha) * homogeneous[i - 1][2] + alpha * homogeneous[i][2],
            ]);
        }
        updated.extend_from_slice(&homogeneous[span..]);

        self.knots.insert(span + 1, u);
        self.control_points = updated
            .iter()
            .map(|h| {
                if h[2].abs() < 1e-15 {
                    [h[0], h[1]]
                } else {
                    [h[0] / h[2], h[1] / h[2]]
                }
            })
            .collect();
        self.weights = updated.iter().map(|h| h[2]).collect();
    }

    /// Splits the curve at `t`, which runs `0..=1` across the domain.
    ///
    /// Both halves are exact: the knot is inserted until it has full
    /// multiplicity, at which point the control points divide cleanly and each
    /// half is a clamped curve in its own right. Weights survive, so splitting
    /// a rational curve does not quietly turn it polynomial.
    ///
    /// `None` when the cut falls at or outside an end, where there is nothing
    /// to divide.
    pub fn split_at(&self, t: f64) -> Option<(Self, Self)> {
        let (start, end) = self.domain();
        let u = start + t.clamp(0.0, 1.0) * (end - start);
        if u <= start + 1e-12 || u >= end - 1e-12 {
            return None;
        }

        let mut work = self.clone();
        let existing = work.knots.iter().filter(|k| (**k - u).abs() < 1e-12).count();
        for _ in existing..self.degree {
            work.insert_knot(u);
        }

        // With the knot at full multiplicity the control points divide at the
        // last one that carries it.
        let cut = work
            .knots
            .iter()
            .rposition(|k| (*k - u).abs() < 1e-12)?;
        let degree = self.degree;
        let left_count = cut - degree + 1;

        let left = Self {
            degree,
            knots: work.knots[..=cut]
                .iter()
                .copied()
                .chain(std::iter::once(u))
                .collect(),
            control_points: work.control_points[..left_count].to_vec(),
            weights: work.weights[..left_count].to_vec(),
        };
        let right = Self {
            degree,
            knots: std::iter::once(u)
                .chain(work.knots[cut - degree + 1..].iter().copied())
                .collect(),
            control_points: work.control_points[left_count - 1..].to_vec(),
            weights: work.weights[left_count - 1..].to_vec(),
        };
        (left.control_points.len() > degree && right.control_points.len() > degree)
            .then_some((left, right))
    }

    /// The C² cubic through every one of `points`.
    ///
    /// This is the other way a drawing describes a spline: fit points it must
    /// pass through, with optional tangents at the ends, rather than a control
    /// polygon it merely approaches. Both arrive as SPLINE entities and both
    /// have to be trimmable, so a fit-point spline is turned into an exact
    /// curve here rather than sampled into a polyline.
    ///
    /// A tangent that is absent — or zero, which is how "unset" is stored —
    /// gets the natural end condition instead, leaving the second derivative
    /// zero there.
    ///
    /// `None` for fewer than two points, where there is no curve.
    pub fn interpolate(
        points: &[[f64; 2]],
        start_tangent: Option<[f64; 2]>,
        end_tangent: Option<[f64; 2]>,
        parameterization: Parameterization,
    ) -> Option<Self> {
        let count = points.len();
        if count < 2 {
            return None;
        }

        // Parameter values, left unnormalised so a unit end tangent — which is
        // how tangents are stored — stays a consistent dP/dt.
        let mut knots_at = vec![0.0f64; count];
        for i in 1..count {
            let chord = Vec2::from(points[i - 1])
                .distance(Vec2::from(points[i]))
                .max(1e-9);
            let step = match parameterization {
                Parameterization::Uniform => 1.0,
                Parameterization::Centripetal => chord.sqrt(),
                Parameterization::Chord => chord,
            };
            knots_at[i] = knots_at[i - 1] + step;
        }
        let spans: Vec<f64> = (0..count - 1)
            .map(|i| (knots_at[i + 1] - knots_at[i]).max(1e-9))
            .collect();

        let usable = |t: Option<[f64; 2]>| {
            t.filter(|v| v[0] * v[0] + v[1] * v[1] > 1e-18)
        };
        let start_tangent = usable(start_tangent);
        let end_tangent = usable(end_tangent);

        // Solve for dP/dt at each fit point, one axis at a time. The interior
        // rows are C² continuity; the end rows are either the clamped tangent
        // or the natural condition.
        let mut slopes = [vec![0.0f64; count], vec![0.0f64; count]];
        for (axis, slope) in slopes.iter_mut().enumerate() {
            let segment_slope =
                |i: usize| (points[i + 1][axis] - points[i][axis]) / spans[i];
            let mut sub = vec![0.0; count];
            let mut main = vec![0.0; count];
            let mut sup = vec![0.0; count];
            let mut rhs = vec![0.0; count];

            match start_tangent {
                Some(t) => {
                    main[0] = 1.0;
                    rhs[0] = t[axis];
                }
                None => {
                    main[0] = 2.0;
                    sup[0] = 1.0;
                    rhs[0] = 3.0 * segment_slope(0);
                }
            }
            for i in 1..count - 1 {
                sub[i] = spans[i];
                main[i] = 2.0 * (spans[i - 1] + spans[i]);
                sup[i] = spans[i - 1];
                rhs[i] = 3.0
                    * (spans[i] * segment_slope(i - 1) + spans[i - 1] * segment_slope(i));
            }
            match end_tangent {
                Some(t) => {
                    main[count - 1] = 1.0;
                    rhs[count - 1] = t[axis];
                }
                None => {
                    sub[count - 1] = 1.0;
                    main[count - 1] = 2.0;
                    rhs[count - 1] = 3.0 * segment_slope(count - 2);
                }
            }
            solve_tridiagonal(&sub, &main, &sup, &mut rhs);
            *slope = rhs;
        }

        // Each Hermite span becomes a cubic Bezier: the inner control points
        // sit a third of the span's derivative from each end. Chaining them is
        // a degree-3 B-spline whose interior knots carry multiplicity 3.
        let mut control_points = Vec::with_capacity(3 * count - 2);
        control_points.push(points[0]);
        for i in 0..count - 1 {
            let h = spans[i];
            let from = points[i];
            let to = points[i + 1];
            let m0 = [slopes[0][i] * h, slopes[1][i] * h];
            let m1 = [slopes[0][i + 1] * h, slopes[1][i + 1] * h];
            control_points.push([from[0] + m0[0] / 3.0, from[1] + m0[1] / 3.0]);
            control_points.push([to[0] - m1[0] / 3.0, to[1] - m1[1] / 3.0]);
            control_points.push(to);
        }

        let mut knots = vec![knots_at[0]; 4];
        for value in knots_at.iter().take(count - 1).skip(1) {
            knots.extend([*value; 3]);
        }
        knots.extend([knots_at[count - 1]; 4]);

        let weights = vec![1.0; control_points.len()];
        Some(Self {
            degree: 3,
            knots,
            control_points,
            weights,
        })
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

/// How fit points are spaced along the parameter when interpolating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parameterization {
    /// Equal parameter per span, ignoring distance. Bunches the curve where
    /// the points are far apart.
    Uniform,
    /// Square root of chord length. The usual compromise, and what tends to
    /// avoid the loops chord-length parameterisation produces at sharp turns.
    Centripetal,
    /// Chord length.
    Chord,
}

/// In-place Thomas solve for a tridiagonal system: `a` sub-diagonal, `b` main,
/// `c` super, `d` right-hand side, overwritten with the solution.
fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], d: &mut [f64]) {
    let n = d.len();
    let mut scratch = vec![0.0f64; n];
    scratch[0] = c[0] / b[0];
    d[0] /= b[0];
    for i in 1..n {
        let m = b[i] - a[i] * scratch[i - 1];
        scratch[i] = c[i] / m;
        d[i] = (d[i] - a[i] * d[i - 1]) / m;
    }
    for i in (0..n - 1).rev() {
        d[i] -= scratch[i] * d[i + 1];
    }
}

/// A clamped uniform knot vector: `degree + 1` zeros, evenly spaced interior
/// values, then `degree + 1` ones.
///
/// Clamped so the curve starts at its first control point and ends at its
/// last, which is what a drawing expects to see. Public because rebuilding a
/// spline after its control polygon changed needs the same vector, and it is
/// better shared than guessed at twice.
pub fn clamped_uniform_knots(degree: usize, control_point_count: usize) -> Vec<f64> {
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
    fn inserting_a_knot_leaves_the_shape_alone() {
        let curve = NurbsCurve::new(
            3,
            vec![[0.0, 0.0], [1.0, 5.0], [4.0, 5.0], [5.0, 0.0], [8.0, -3.0]],
            Vec::new(),
            None,
        )
        .unwrap();
        let before: Vec<[f64; 2]> = (0..=40).map(|i| curve.point_at(i as f64 / 40.0)).collect();

        let mut inserted = curve.clone();
        inserted.insert_knot(0.37);
        assert_eq!(inserted.control_points().len(), curve.control_points().len() + 1);

        for (i, expected) in before.iter().enumerate() {
            let got = inserted.point_at(i as f64 / 40.0);
            assert!(close(got, *expected, 1e-9), "moved at {i}: {got:?} vs {expected:?}");
        }
    }

    #[test]
    fn both_halves_of_a_split_lie_on_the_original() {
        let curve = NurbsCurve::new(
            3,
            vec![[0.0, 0.0], [1.0, 5.0], [4.0, 5.0], [5.0, 0.0], [8.0, -3.0]],
            Vec::new(),
            None,
        )
        .unwrap();
        let (left, right) = curve.split_at(0.4).expect("should split");

        // The halves meet where the cut was, and each covers its own stretch.
        assert!(close(left.point_at(0.0), curve.point_at(0.0), 1e-9));
        assert!(close(left.point_at(1.0), curve.point_at(0.4), 1e-9));
        assert!(close(right.point_at(0.0), curve.point_at(0.4), 1e-9));
        assert!(close(right.point_at(1.0), curve.point_at(1.0), 1e-9));

        for i in 0..=20 {
            let f = i as f64 / 20.0;
            assert!(
                close(left.point_at(f), curve.point_at(f * 0.4), 1e-9),
                "left half drifted at {f}"
            );
            assert!(
                close(right.point_at(f), curve.point_at(0.4 + f * 0.6), 1e-9),
                "right half drifted at {f}"
            );
        }
    }

    #[test]
    fn splitting_a_rational_curve_keeps_it_rational() {
        let weight = std::f64::consts::FRAC_1_SQRT_2;
        let curve = NurbsCurve::new(
            2,
            vec![[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            Some(vec![1.0, weight, 1.0]),
        )
        .unwrap();
        let (left, right) = curve.split_at(0.5).expect("should split");
        // Both halves must still trace the unit circle.
        for half in [&left, &right] {
            for i in 0..=10 {
                let p = half.point_at(i as f64 / 10.0);
                let radius = (p[0] * p[0] + p[1] * p[1]).sqrt();
                assert!(
                    (radius - 1.0).abs() < 1e-9,
                    "half left the unit circle: {radius}"
                );
            }
        }
    }

    #[test]
    fn splitting_at_an_end_has_nothing_to_divide() {
        let curve = linear();
        assert!(curve.split_at(0.0).is_none());
        assert!(curve.split_at(1.0).is_none());
    }

    #[test]
    fn an_interpolated_curve_passes_through_every_fit_point() {
        let fit = [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [5.0, 4.0]];
        let curve =
            NurbsCurve::interpolate(&fit, None, None, Parameterization::Chord).unwrap();
        // Each fit point sits at the knot value its parameterisation gave it,
        // and the joins carry multiplicity 3, so they are knots in the vector.
        let (start, end) = curve.domain();
        assert!(close(curve.point_at_knot(start), fit[0], 1e-9));
        assert!(close(curve.point_at_knot(end), fit[3], 1e-9));
        for target in fit {
            let t = curve.parameter_at(target);
            assert!(
                close(curve.point_at(t), target, 1e-6),
                "missed {target:?}, nearest was {:?}",
                curve.point_at(t)
            );
        }
    }

    #[test]
    fn two_fit_points_with_tangents_make_one_cubic() {
        // The case a drawing produces most often for a simple curve, and the
        // one that has no interior fit point to lean on.
        let fit = [[0.0, 0.0], [10.0, 0.0]];
        let curve = NurbsCurve::interpolate(
            &fit,
            Some([0.0, 1.0]),
            Some([0.0, -1.0]),
            Parameterization::Chord,
        )
        .unwrap();
        assert_eq!(curve.degree(), 3);
        assert_eq!(curve.control_points().len(), 4, "a single Bezier span");
        assert!(close(curve.point_at(0.0), fit[0], 1e-9));
        assert!(close(curve.point_at(1.0), fit[1], 1e-9));
        // Leaves upward and arrives heading downward, so it arches over.
        assert!(curve.point_at(0.5)[1] > 0.0, "should bow upward");
    }

    #[test]
    fn the_stored_tangents_are_followed_at_the_ends() {
        let fit = [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]];
        let curve = NurbsCurve::interpolate(
            &fit,
            Some([1.0, 2.0]),
            Some([1.0, -2.0]),
            Parameterization::Chord,
        )
        .unwrap();
        // Direction leaving the start, by finite difference.
        let a = curve.point_at(0.0);
        let b = curve.point_at(0.001);
        let leaving = (b[1] - a[1]) / (b[0] - a[0]);
        assert!((leaving - 2.0).abs() < 0.05, "left along {leaving}, wanted 2");

        let c = curve.point_at(0.999);
        let d = curve.point_at(1.0);
        let arriving = (d[1] - c[1]) / (d[0] - c[0]);
        assert!((arriving + 2.0).abs() < 0.05, "arrived along {arriving}");
    }

    #[test]
    fn interpolation_without_tangents_still_passes_through() {
        let fit = [[0.0, 0.0], [2.0, 3.0], [4.0, -1.0]];
        for mode in [
            Parameterization::Uniform,
            Parameterization::Centripetal,
            Parameterization::Chord,
        ] {
            let curve = NurbsCurve::interpolate(&fit, None, None, mode).unwrap();
            for target in fit {
                let t = curve.parameter_at(target);
                assert!(close(curve.point_at(t), target, 1e-6), "{mode:?} missed {target:?}");
            }
        }
    }

    #[test]
    fn an_interpolated_curve_can_be_split_like_any_other() {
        let fit = [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [5.0, 4.0]];
        let curve =
            NurbsCurve::interpolate(&fit, None, None, Parameterization::Chord).unwrap();
        let (left, right) = curve.split_at(0.45).expect("should split");
        assert!(close(left.point_at(1.0), right.point_at(0.0), 1e-9));
        for i in 0..=10 {
            let f = i as f64 / 10.0;
            assert!(close(left.point_at(f), curve.point_at(f * 0.45), 1e-9));
        }
    }

    #[test]
    fn fewer_than_two_fit_points_is_not_a_curve() {
        assert!(NurbsCurve::interpolate(&[], None, None, Parameterization::Chord).is_none());
        assert!(
            NurbsCurve::interpolate(&[[1.0, 1.0]], None, None, Parameterization::Chord)
                .is_none()
        );
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
