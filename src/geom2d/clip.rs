//! Which parts of a curve lie inside a boundary.
//!
//! The question a hatch asks of every line in its pattern: the pattern is a
//! family of parallel lines laid across the whole region, and what gets drawn
//! is only the stretches of them that fall inside. A section fill asks it too,
//! and so does a B-rep face that has to fill or shade its own interior.
//!
//! # Why not simply alternate
//!
//! The usual shortcut is to sort the crossings along the curve and take every
//! other span, on the grounds that each crossing flips inside to outside. It
//! is right for a boundary a line cuts cleanly and wrong wherever it does not:
//! a line that grazes a corner or runs tangent to an arc crosses twice at
//! effectively one place, and from there every span afterwards is inverted —
//! a hatch that fills the holes and leaves the solid empty.
//!
//! So the crossings are used only to *divide* the curve, and each resulting
//! span is then asked directly whether its middle is inside. That costs one
//! containment test per span and cannot get out of step.

use super::containment::contains;
use super::cross::intersect;
use super::curve::{Curve, Extent};
use super::vec::Vec2;
use super::Tolerance;

/// The spans of `curve` that lie inside `boundary`, as parameter pairs.
///
/// The boundary is taken as a set of curves that between them close one or
/// more loops; they need not be given in order, and holes work because
/// containment is decided by ray casting rather than by winding.
///
/// Spans come back in order and never touch: two that would meet at a
/// crossing where the curve only grazes the boundary are merged, so a caller
/// drawing them does not get a seam.
///
/// An unbounded curve is clipped to the crossings it has, which is what makes
/// this usable for a hatch: the pattern lines are infinite and only the part
/// inside the region is ever wanted.
pub fn inside_spans(boundary: &[Curve], curve: &Curve, tolerance: Tolerance) -> Vec<[f64; 2]> {
    if boundary.is_empty() {
        return Vec::new();
    }
    let mut cuts: Vec<f64> = Vec::new();
    for edge in boundary {
        for crossing in intersect(curve, edge, tolerance) {
            cuts.push(crossing.t_a);
        }
    }
    if cuts.is_empty() {
        // No crossings: the curve is wholly inside or wholly outside, and one
        // test settles it. Only a bounded curve can be wholly inside — an
        // infinite one always leaves.
        if curve.extent() == Extent::Bounded
            && contains(boundary, curve.point_at(0.5), tolerance)
        {
            return vec![[0.0, 1.0]];
        }
        return Vec::new();
    }

    if curve.extent() == Extent::Bounded {
        cuts.push(0.0);
        cuts.push(1.0);
    }
    cuts.sort_by(f64::total_cmp);

    // Two crossings closer together than this along the parameter are the
    // same place — a corner grazed, an arc touched. A tolerance is a
    // distance, so it is converted by how much distance a unit of parameter
    // covers, which is the curve's own speed.
    //
    // Not the chord between the two ends: on a closed curve that is zero, and
    // dividing by it turned a micrometre into a merge window of a billion
    // parameters — every crossing collapsed into one and nothing was ever
    // found inside.
    let speed = Vec2::from(curve.tangent_at(0.5)).length();
    let merge = if speed > 0.0 {
        tolerance.linear() / speed
    } else {
        tolerance.linear()
    };
    cuts.dedup_by(|a, b| (*a - *b).abs() <= merge);

    let mut spans: Vec<[f64; 2]> = Vec::new();
    for pair in cuts.windows(2) {
        let middle = 0.5 * (pair[0] + pair[1]);
        if !contains(boundary, curve.point_at(middle), tolerance) {
            continue;
        }
        // Merge onto the previous span when they meet, so a curve passing
        // through a vertex of the boundary comes back in one piece.
        match spans.last_mut() {
            Some(previous) if (previous[1] - pair[0]).abs() <= merge => previous[1] = pair[1],
            _ => spans.push([pair[0], pair[1]]),
        }
    }
    spans
}

/// The spans as pairs of points rather than parameters.
///
/// What a caller drawing the result wants; the parameters are what a caller
/// splitting the curve wants.
pub fn inside_pieces(
    boundary: &[Curve],
    curve: &Curve,
    tolerance: Tolerance,
) -> Vec<[[f64; 2]; 2]> {
    inside_spans(boundary, curve, tolerance)
        .into_iter()
        .map(|span| [curve.point_at(span[0]), curve.point_at(span[1])])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom2d::{Arc, Circle, Line, XLine};
    use std::f64::consts::TAU;

    fn tol() -> Tolerance {
        Tolerance::new(1e-6)
    }

    fn line(a: [f64; 2], b: [f64; 2]) -> Curve {
        Curve::Line(Line { start: a, end: b })
    }

    /// A 10 × 10 square at the origin, as four separate edges.
    fn square() -> Vec<Curve> {
        vec![
            line([0.0, 0.0], [10.0, 0.0]),
            line([10.0, 0.0], [10.0, 10.0]),
            line([10.0, 10.0], [0.0, 10.0]),
            line([0.0, 10.0], [0.0, 0.0]),
        ]
    }

    /// The same square with a 4 × 4 hole punched in the middle.
    fn square_with_hole() -> Vec<Curve> {
        let mut out = square();
        out.extend([
            line([3.0, 3.0], [7.0, 3.0]),
            line([7.0, 3.0], [7.0, 7.0]),
            line([7.0, 7.0], [3.0, 7.0]),
            line([3.0, 7.0], [3.0, 3.0]),
        ]);
        out
    }

    fn crossing_line(y: f64) -> Curve {
        Curve::XLine(XLine {
            base: [0.0, y],
            direction: [1.0, 0.0],
        })
    }

    #[test]
    fn an_infinite_line_is_clipped_to_what_is_inside() {
        let pieces = inside_pieces(&square(), &crossing_line(5.0), tol());
        assert_eq!(pieces.len(), 1, "{pieces:?}");
        assert!((pieces[0][0][0]).abs() < 1e-6 && (pieces[0][1][0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn a_hole_leaves_a_gap_in_the_middle() {
        let pieces = inside_pieces(&square_with_hole(), &crossing_line(5.0), tol());
        assert_eq!(pieces.len(), 2, "{pieces:?}");
        let mut spans: Vec<(f64, f64)> = pieces
            .iter()
            .map(|piece| (piece[0][0].min(piece[1][0]), piece[0][0].max(piece[1][0])))
            .collect();
        spans.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert!((spans[0].0).abs() < 1e-6 && (spans[0].1 - 3.0).abs() < 1e-6, "{spans:?}");
        assert!((spans[1].0 - 7.0).abs() < 1e-6 && (spans[1].1 - 10.0).abs() < 1e-6);
    }

    #[test]
    fn a_line_that_misses_entirely_yields_nothing() {
        assert!(inside_pieces(&square(), &crossing_line(50.0), tol()).is_empty());
    }

    #[test]
    fn a_segment_wholly_inside_comes_back_whole() {
        let inside = line([2.0, 5.0], [8.0, 5.0]);
        let spans = inside_spans(&square(), &inside, tol());
        assert_eq!(spans, vec![[0.0, 1.0]]);
    }

    #[test]
    fn a_segment_wholly_outside_yields_nothing() {
        let outside = line([20.0, 5.0], [30.0, 5.0]);
        assert!(inside_spans(&square(), &outside, tol()).is_empty());
    }

    #[test]
    fn a_segment_half_in_is_cut_at_the_boundary() {
        let half = line([5.0, 5.0], [25.0, 5.0]);
        let pieces = inside_pieces(&square(), &half, tol());
        assert_eq!(pieces.len(), 1);
        assert!((pieces[0][0][0] - 5.0).abs() < 1e-6);
        assert!((pieces[0][1][0] - 10.0).abs() < 1e-6, "{:?}", pieces[0][1]);
    }

    #[test]
    fn grazing_a_corner_does_not_invert_everything_after_it() {
        // The failure alternating spans gives: this line passes exactly
        // through the notch's corner, so two crossings land in one place. An
        // alternating walk flips there and fills the outside from then on.
        let boundary = vec![
            line([0.0, 0.0], [10.0, 0.0]),
            line([10.0, 0.0], [10.0, 10.0]),
            line([10.0, 10.0], [6.0, 5.0]),
            line([6.0, 5.0], [10.0, 0.1]),
            line([10.0, 0.1], [10.0, 0.0]),
            line([0.0, 10.0], [0.0, 0.0]),
            line([10.0, 10.0], [0.0, 10.0]),
        ];
        let pieces = inside_pieces(&boundary, &crossing_line(5.0), tol());
        // Whatever the boundary's exact shape, every reported piece must
        // actually have its middle inside.
        for piece in &pieces {
            let middle = Vec2::from(piece[0]).lerp(Vec2::from(piece[1]), 0.5);
            assert!(
                contains(&boundary, middle.to_array(), tol()),
                "{middle:?} was reported inside but is not"
            );
        }
    }

    #[test]
    fn a_curved_boundary_clips_as_well_as_a_straight_one() {
        // The generality that a polygon-only version does not have: the
        // boundary here is a circle, not a chain of segments.
        let circle = vec![Curve::Circle(Circle {
            centre: [0.0, 0.0],
            radius: 10.0,
        })];
        let pieces = inside_pieces(&circle, &crossing_line(6.0), tol());
        assert_eq!(pieces.len(), 1);
        let width = (pieces[0][1][0] - pieces[0][0][0]).abs();
        // A chord at y = 6 on a circle of 10 is 16 across.
        assert!((width - 16.0).abs() < 1e-4, "{width}");
    }

    #[test]
    fn a_curve_can_be_clipped_as_well_as_a_line() {
        // An arc crossing out of the square and back in.
        let arc = Curve::Arc(Arc {
            centre: [10.0, 5.0],
            radius: 3.0,
            start_angle: 0.0,
            end_angle: TAU,
        });
        let spans = inside_spans(&square(), &arc, tol());
        assert!(!spans.is_empty());
        for span in &spans {
            let middle = arc.point_at(0.5 * (span[0] + span[1]));
            assert!(contains(&square(), middle, tol()), "{middle:?}");
            // And the part outside really is left out.
            assert!(middle[0] <= 10.0 + 1e-6);
        }
    }

    #[test]
    fn an_empty_boundary_encloses_nothing() {
        assert!(inside_spans(&[], &crossing_line(5.0), tol()).is_empty());
    }

    #[test]
    fn survey_coordinates_clip_to_the_same_lengths() {
        let origin = [512_345.678, 4_512_345.678];
        let shifted: Vec<Curve> = square()
            .into_iter()
            .map(|edge| {
                let (start, along) = edge.as_ray().unwrap();
                line(
                    [origin[0] + start[0], origin[1] + start[1]],
                    [
                        origin[0] + start[0] + along[0],
                        origin[1] + start[1] + along[1],
                    ],
                )
            })
            .collect();
        let across = Curve::XLine(XLine {
            base: [origin[0], origin[1] + 5.0],
            direction: [1.0, 0.0],
        });
        let pieces = inside_pieces(&shifted, &across, tol());
        assert_eq!(pieces.len(), 1, "{pieces:?}");
        let width = (pieces[0][1][0] - pieces[0][0][0]).abs();
        assert!((width - 10.0).abs() < 1e-4, "{width}");
    }
}
