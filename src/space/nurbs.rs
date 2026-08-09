//! NURBS in space: a curve that genuinely wanders, and a surface.
//!
//! [`geom2d`](crate::geom2d) has the plane's own NURBS, which is what a
//! drawing's SPLINE entity almost always is. This is the rest: a spline whose
//! points do not share a plane — a helix, a 3-D fit curve, an edge lifted out
//! of a file — and the tensor-product surface an ACIS `spline-surface` or a
//! DWG surface patch carries.
//!
//! # The same algorithm, one dimension up
//!
//! Both evaluate through [`space::spline::de_boor`](super::spline::de_boor),
//! which is written over however many coordinates a control point holds. A
//! surface is that applied twice: once along `u` for each row of the control
//! net, then once along `v` through the results. Writing the recursion again
//! per dimension is how the plane case and the space case drift apart.
//!
//! # Rational is polynomial, one dimension further up
//!
//! Weights are carried by multiplying each control point into homogeneous
//! coordinates and dividing at the very end. Dividing earlier — averaging
//! already-divided points — gives a curve that looks close and is not, and
//! the difference is exactly where the weights matter.

use super::spline::{clamped_uniform_knots, de_boor};
use super::Vec3;

/// A NURBS curve in space.
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsCurve3 {
    degree: usize,
    knots: Vec<f64>,
    control_points: Vec<[f64; 3]>,
    weights: Vec<f64>,
}

impl NurbsCurve3 {
    /// Builds a curve, filling in what the caller left out.
    ///
    /// A knot vector of the wrong length is replaced with a clamped uniform
    /// one — a drawing with a malformed spline should still draw something
    /// rather than nothing. Weights are optional; absent means all ones and
    /// the curve is polynomial.
    ///
    /// `None` when there are fewer control points than the degree needs,
    /// since there is then no curve to evaluate at all.
    pub fn new(
        degree: usize,
        control_points: Vec<[f64; 3]>,
        knots: Vec<f64>,
        weights: Option<Vec<f64>>,
    ) -> Option<Self> {
        if control_points.len() <= degree || degree == 0 {
            return None;
        }
        let wanted = control_points.len() + degree + 1;
        let knots = if knots.len() == wanted {
            knots
        } else {
            clamped_uniform_knots(degree, control_points.len())
        };
        let weights = match weights {
            Some(weights) if weights.len() == control_points.len() => weights,
            _ => vec![1.0; control_points.len()],
        };
        Some(Self {
            degree,
            knots,
            control_points,
            weights,
        })
    }

    /// The degree of the curve.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Its control points, in order.
    pub fn control_points(&self) -> &[[f64; 3]] {
        &self.control_points
    }

    /// Its knot vector.
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Its weights, all ones when the curve is polynomial.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Whether any weight differs from the others. A curve whose weights are
    /// all equal is polynomial however large they are.
    pub fn is_rational(&self) -> bool {
        let first = self.weights.first().copied().unwrap_or(1.0);
        self.weights
            .iter()
            .any(|weight| (weight - first).abs() > 1e-12)
    }

    /// The knot range the curve is defined over.
    pub fn domain(&self) -> (f64, f64) {
        let last = self.control_points.len();
        (self.knots[self.degree], self.knots[last])
    }

    /// The point at a knot parameter.
    pub fn point_at_knot(&self, u: f64) -> [f64; 3] {
        let homogeneous: Vec<[f64; 4]> = self
            .control_points
            .iter()
            .zip(&self.weights)
            .map(|(point, weight)| {
                [
                    point[0] * weight,
                    point[1] * weight,
                    point[2] * weight,
                    *weight,
                ]
            })
            .collect();
        let raw = de_boor(self.degree, &self.knots, &homogeneous, u);
        if raw[3].abs() < 1e-15 {
            return [raw[0], raw[1], raw[2]];
        }
        [raw[0] / raw[3], raw[1] / raw[3], raw[2] / raw[3]]
    }

    /// The point at `t` in `0..=1`, which is the domain rescaled.
    ///
    /// Every other curve in this kernel runs zero to one, and a caller
    /// stepping along one should not have to ask which convention it is.
    pub fn point_at(&self, t: f64) -> [f64; 3] {
        let (from, to) = self.domain();
        self.point_at_knot(from + (to - from) * t.clamp(0.0, 1.0))
    }

    /// The tangent at `t`, by a central difference on the knot parameter.
    ///
    /// Differenced rather than differentiated: the derivative of a rational
    /// curve is a quotient rule over two B-splines, and for the one thing
    /// this is wanted for — which way the curve is heading — a difference at
    /// a thousandth of the domain is indistinguishable and cannot be
    /// subtly wrong.
    pub fn tangent_at(&self, t: f64) -> [f64; 3] {
        let (from, to) = self.domain();
        let span = to - from;
        if span <= 0.0 {
            return [0.0; 3];
        }
        let step = span * 1e-4;
        let here = from + span * t.clamp(0.0, 1.0);
        let behind = self.point_at_knot((here - step).max(from));
        let ahead = self.point_at_knot((here + step).min(to));
        let scale = 1.0 / (2.0 * step);
        [
            (ahead[0] - behind[0]) * scale,
            (ahead[1] - behind[1]) * scale,
            (ahead[2] - behind[2]) * scale,
        ]
    }

    /// Whether the curve returns to where it started.
    pub fn is_closed(&self) -> bool {
        let (from, to) = self.domain();
        Vec3::from(self.point_at_knot(from)).distance(Vec3::from(self.point_at_knot(to))) < 1e-9
    }

    /// Points along the curve, no farther from it than `tolerance`.
    ///
    /// Refined where it bends rather than sampled evenly: a curve that is
    /// nearly straight for most of its length and turns sharply once needs
    /// its points where the turn is, and a uniform sampling either misses the
    /// turn or carries thousands of points through the straight part.
    pub fn tessellate_within(&self, tolerance: f64) -> Vec<[f64; 3]> {
        let (from, to) = self.domain();
        let mut points = vec![self.point_at_knot(from)];
        self.refine(from, to, tolerance.max(1e-12), 0, &mut points);
        points.push(self.point_at_knot(to));
        points
    }

    /// Splits a span until its middle sits close enough to the chord.
    fn refine(&self, from: f64, to: f64, tolerance: f64, depth: u32, out: &mut Vec<[f64; 3]>) {
        // Sixteen levels is sixty-five thousand points on one span, which no
        // tolerance a drawing uses reaches; it is a backstop against a curve
        // that never converges, not a limit.
        const MAX_DEPTH: u32 = 16;
        let middle = 0.5 * (from + to);
        let curved = Vec3::from(self.point_at_knot(middle));
        if depth < MAX_DEPTH {
            let start = Vec3::from(self.point_at_knot(from));
            let end = Vec3::from(self.point_at_knot(to));
            if start.lerp(end, 0.5).distance(curved) > tolerance {
                self.refine(from, middle, tolerance, depth + 1, out);
                out.push(curved.to_array());
                self.refine(middle, to, tolerance, depth + 1, out);
                return;
            }
        }
    }
}

/// A NURBS surface in space.
///
/// The control net is stored row by row: `control_points[i][j]` is the point
/// at `u` index `i` and `v` index `j`, which is the order every file format
/// writes and the order the evaluation below reads.
#[derive(Debug, Clone, PartialEq)]
pub struct NurbsSurface3 {
    u_degree: usize,
    v_degree: usize,
    u_knots: Vec<f64>,
    v_knots: Vec<f64>,
    control_points: Vec<Vec<[f64; 3]>>,
    weights: Vec<Vec<f64>>,
}

impl NurbsSurface3 {
    /// Builds a surface, filling in what the caller left out.
    ///
    /// `None` for a ragged control net, or one too small for the degrees:
    /// a surface with rows of different lengths is not a tensor product and
    /// there is nothing sensible to evaluate.
    pub fn new(
        u_degree: usize,
        v_degree: usize,
        control_points: Vec<Vec<[f64; 3]>>,
        u_knots: Vec<f64>,
        v_knots: Vec<f64>,
        weights: Option<Vec<Vec<f64>>>,
    ) -> Option<Self> {
        let rows = control_points.len();
        let columns = control_points.first()?.len();
        if rows <= u_degree || columns <= v_degree || u_degree == 0 || v_degree == 0 {
            return None;
        }
        if control_points.iter().any(|row| row.len() != columns) {
            return None;
        }
        let u_knots = if u_knots.len() == rows + u_degree + 1 {
            u_knots
        } else {
            clamped_uniform_knots(u_degree, rows)
        };
        let v_knots = if v_knots.len() == columns + v_degree + 1 {
            v_knots
        } else {
            clamped_uniform_knots(v_degree, columns)
        };
        let weights = match weights {
            Some(weights)
                if weights.len() == rows && weights.iter().all(|row| row.len() == columns) =>
            {
                weights
            }
            _ => vec![vec![1.0; columns]; rows],
        };
        Some(Self {
            u_degree,
            v_degree,
            u_knots,
            v_knots,
            control_points,
            weights,
        })
    }

    /// The knot ranges the surface is defined over, `u` then `v`.
    pub fn domain(&self) -> ((f64, f64), (f64, f64)) {
        (
            (self.u_knots[self.u_degree], self.u_knots[self.control_points.len()]),
            (
                self.v_knots[self.v_degree],
                self.v_knots[self.control_points[0].len()],
            ),
        )
    }

    /// The point at knot parameters `(u, v)`.
    ///
    /// Two passes of the same algorithm: along `v` through each row of the
    /// net, then along `u` through what those gave. Homogeneous the whole
    /// way, so the weights come out right.
    pub fn point_at_knot(&self, u: f64, v: f64) -> [f64; 3] {
        let along_rows: Vec<[f64; 4]> = self
            .control_points
            .iter()
            .zip(&self.weights)
            .map(|(row, row_weights)| {
                let homogeneous: Vec<[f64; 4]> = row
                    .iter()
                    .zip(row_weights)
                    .map(|(point, weight)| {
                        [
                            point[0] * weight,
                            point[1] * weight,
                            point[2] * weight,
                            *weight,
                        ]
                    })
                    .collect();
                de_boor(self.v_degree, &self.v_knots, &homogeneous, v)
            })
            .collect();
        let raw = de_boor(self.u_degree, &self.u_knots, &along_rows, u);
        if raw[3].abs() < 1e-15 {
            return [raw[0], raw[1], raw[2]];
        }
        [raw[0] / raw[3], raw[1] / raw[3], raw[2] / raw[3]]
    }

    /// The point at `(s, t)` in `0..=1` each, the domains rescaled.
    pub fn point_at(&self, s: f64, t: f64) -> [f64; 3] {
        let ((u0, u1), (v0, v1)) = self.domain();
        self.point_at_knot(
            u0 + (u1 - u0) * s.clamp(0.0, 1.0),
            v0 + (v1 - v0) * t.clamp(0.0, 1.0),
        )
    }

    /// The unit normal at knot parameters `(u, v)`.
    ///
    /// `None` at a point where the net is degenerate — a pole, or a row of
    /// coincident control points — since there is no plane there to be
    /// perpendicular to and any answer would be invented.
    pub fn normal_at_knot(&self, u: f64, v: f64) -> Option<[f64; 3]> {
        let ((u0, u1), (v0, v1)) = self.domain();
        let (u_span, v_span) = (u1 - u0, v1 - v0);
        if u_span <= 0.0 || v_span <= 0.0 {
            return None;
        }
        let (du, dv) = (u_span * 1e-4, v_span * 1e-4);
        let along_u = Vec3::from(self.point_at_knot((u + du).min(u1), v))
            - Vec3::from(self.point_at_knot((u - du).max(u0), v));
        let along_v = Vec3::from(self.point_at_knot(u, (v + dv).min(v1)))
            - Vec3::from(self.point_at_knot(u, (v - dv).max(v0)));
        along_u.cross(along_v).normalize().map(Vec3::to_array)
    }

    /// The same at `(s, t)` in `0..=1` each.
    pub fn normal_at(&self, s: f64, t: f64) -> Option<[f64; 3]> {
        let ((u0, u1), (v0, v1)) = self.domain();
        self.normal_at_knot(
            u0 + (u1 - u0) * s.clamp(0.0, 1.0),
            v0 + (v1 - v0) * t.clamp(0.0, 1.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn helix_points() -> Vec<[f64; 3]> {
        vec![
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 2.0],
            [-1.0, -1.0, 3.0],
            [1.0, -1.0, 4.0],
        ]
    }

    #[test]
    fn a_clamped_curve_ends_on_its_own_control_points() {
        let curve = NurbsCurve3::new(3, helix_points(), Vec::new(), None).unwrap();
        let start = curve.point_at(0.0);
        let end = curve.point_at(1.0);
        assert!(Vec3::from(start).distance(Vec3::new(1.0, 0.0, 0.0)) < 1e-9, "{start:?}");
        assert!(Vec3::from(end).distance(Vec3::new(1.0, -1.0, 4.0)) < 1e-9, "{end:?}");
    }

    #[test]
    fn a_curve_that_wanders_is_not_flat() {
        // The reason this exists rather than the plane one: these points
        // share no plane, and projecting them onto one would lose the climb.
        let curve = NurbsCurve3::new(3, helix_points(), Vec::new(), None).unwrap();
        let heights: Vec<f64> = (0..=8).map(|step| curve.point_at(step as f64 / 8.0)[2]).collect();
        assert!(heights.windows(2).all(|pair| pair[1] > pair[0]), "{heights:?}");
    }

    #[test]
    fn a_weighted_curve_is_pulled_towards_the_heavy_point() {
        let points = vec![[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]];
        let plain = NurbsCurve3::new(2, points.clone(), Vec::new(), None).unwrap();
        let heavy =
            NurbsCurve3::new(2, points, Vec::new(), Some(vec![1.0, 8.0, 1.0])).unwrap();
        assert!(heavy.is_rational());
        assert!(!plain.is_rational());
        assert!(heavy.point_at(0.5)[1] > plain.point_at(0.5)[1]);
    }

    #[test]
    fn a_finer_tolerance_gets_closer_to_the_curve() {
        // Measured by length rather than by distance to a sampled reference.
        // A reference fine enough to check a thousandth would need to be
        // finer than that itself, and comparing against a coarse one measures
        // the reference's own spacing instead of the tessellation's error.
        //
        // Length works because a chord always cuts the corner: a polyline
        // inscribed in a curve is short, and less short the closer it fits.
        let curve = NurbsCurve3::new(3, helix_points(), Vec::new(), None).unwrap();
        let length = |points: &[[f64; 3]]| -> f64 {
            points
                .windows(2)
                .map(|pair| Vec3::from(pair[0]).distance(Vec3::from(pair[1])))
                .sum()
        };
        let truth = length(
            &(0..=20_000)
                .map(|step| curve.point_at(step as f64 / 20_000.0))
                .collect::<Vec<_>>(),
        );
        let coarse = curve.tessellate_within(0.1);
        let fine = curve.tessellate_within(0.001);
        assert!(fine.len() > coarse.len(), "a finer tolerance asks for more");
        for points in [&coarse, &fine] {
            assert!(length(points) <= truth + 1e-9, "a chord cannot read long");
        }
        assert!(length(&fine) > length(&coarse));
        // A sag bound is a distance, not a length, so what it buys in length
        // is a consequence rather than the promise — a tenth of a per cent
        // here on a curve six units long.
        assert!(length(&fine) > truth * 0.999, "{} vs {truth}", length(&fine));
    }

    #[test]
    fn a_tangent_points_the_way_the_curve_runs() {
        let curve = NurbsCurve3::new(3, helix_points(), Vec::new(), None).unwrap();
        for step in 1..8 {
            let t = step as f64 / 8.0;
            let along = Vec3::from(curve.tangent_at(t));
            let ahead = Vec3::from(curve.point_at(t + 0.05)) - Vec3::from(curve.point_at(t));
            assert!(along.dot(ahead) > 0.0, "t={t}");
        }
    }

    #[test]
    fn too_few_points_for_the_degree_is_refused() {
        assert!(NurbsCurve3::new(3, vec![[0.0; 3], [1.0; 3]], Vec::new(), None).is_none());
        assert!(NurbsCurve3::new(0, vec![[0.0; 3], [1.0; 3]], Vec::new(), None).is_none());
    }

    /// A flat 3 × 3 net, raised in the middle.
    fn hill() -> NurbsSurface3 {
        let mut net = Vec::new();
        for row in 0..3 {
            let mut across = Vec::new();
            for column in 0..3 {
                let height = if row == 1 && column == 1 { 4.0 } else { 0.0 };
                across.push([row as f64 * 5.0, column as f64 * 5.0, height]);
            }
            net.push(across);
        }
        NurbsSurface3::new(2, 2, net, Vec::new(), Vec::new(), None).unwrap()
    }

    #[test]
    fn a_surface_holds_its_own_corners() {
        let surface = hill();
        for (s, t, corner) in [
            (0.0, 0.0, [0.0, 0.0, 0.0]),
            (1.0, 0.0, [10.0, 0.0, 0.0]),
            (0.0, 1.0, [0.0, 10.0, 0.0]),
            (1.0, 1.0, [10.0, 10.0, 0.0]),
        ] {
            let point = surface.point_at(s, t);
            assert!(
                Vec3::from(point).distance(Vec3::from(corner)) < 1e-9,
                "({s}, {t}): {point:?}"
            );
        }
    }

    #[test]
    fn a_raised_control_point_lifts_the_middle_but_not_to_its_own_height() {
        // A B-spline approximates rather than interpolates its interior
        // points, which is the property that distinguishes it from a mesh of
        // the same net.
        let middle = hill().point_at(0.5, 0.5);
        assert!(middle[2] > 0.5, "{middle:?}");
        assert!(middle[2] < 4.0, "{middle:?}");
        assert!((middle[0] - 5.0).abs() < 1e-9 && (middle[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn a_flat_surfaces_normal_is_perpendicular_to_it() {
        let net = vec![
            vec![[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 10.0, 0.0]],
            vec![[5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [5.0, 10.0, 0.0]],
            vec![[10.0, 0.0, 0.0], [10.0, 5.0, 0.0], [10.0, 10.0, 0.0]],
        ];
        let surface = NurbsSurface3::new(2, 2, net, Vec::new(), Vec::new(), None).unwrap();
        let normal = Vec3::from(surface.normal_at(0.5, 0.5).unwrap());
        assert!(normal.cross(Vec3::Z).length() < 1e-9, "{normal:?}");
    }

    #[test]
    fn a_ragged_net_is_refused_rather_than_padded() {
        let net = vec![
            vec![[0.0; 3], [1.0; 3], [2.0; 3]],
            vec![[0.0; 3], [1.0; 3]],
            vec![[0.0; 3], [1.0; 3], [2.0; 3]],
        ];
        assert!(NurbsSurface3::new(2, 2, net, Vec::new(), Vec::new(), None).is_none());
    }
}
