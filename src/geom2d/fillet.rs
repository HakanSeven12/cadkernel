//! Rounding the corner where two lines meet.
//!
//! A fillet replaces a corner with an arc of a given radius tangent to both
//! sides. The geometry is small but has two things that are easy to get wrong:
//! which of the four corners around an intersection is being rounded, and
//! which way the arc sweeps once it is placed.
//!
//! Both are settled by the *keep directions* — unit vectors pointing from the
//! corner along the parts of each line that survive. They carry the choice of
//! corner, so this module never has to guess it, and the caller resolves them
//! from whatever it has: a pick point, a segment's own extent, a rule.

use super::angle::normalize_angle;

/// The arc that rounds a corner, and where it meets the two sides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fillet {
    /// Where the arc meets the first line.
    pub tangent1: [f64; 2],
    /// Where it meets the second.
    pub tangent2: [f64; 2],
    /// Centre of the arc.
    pub centre: [f64; 2],
    /// Start angle, normalised to `0..TAU`, sweeping counter-clockwise to
    /// [`end_angle`](Self::end_angle).
    pub start_angle: f64,
    /// End angle, normalised the same way.
    pub end_angle: f64,
}

/// The fillet of `radius` at the corner where two rays leave `apex`.
///
/// `direction1` and `direction2` must be unit vectors pointing along the parts
/// that survive; they select which of the four corners around the intersection
/// gets rounded.
///
/// `None` when there is no corner to round: the directions are parallel or
/// exactly opposed, so no arc of finite radius is tangent to both.
///
/// A radius of zero is a corner, not a fillet, and is the caller's business —
/// it wants the apex itself, and there is no arc to report.
pub fn fillet_between_rays(
    apex: [f64; 2],
    direction1: [f64; 2],
    direction2: [f64; 2],
    radius: f64,
) -> Option<Fillet> {
    let cosine = (direction1[0] * direction2[0] + direction1[1] * direction2[1]).clamp(-1.0, 1.0);
    let corner = cosine.acos();
    if corner < 1e-6 || (corner - std::f64::consts::PI).abs() < 1e-6 {
        return None;
    }
    if radius < 1e-9 {
        return None;
    }

    let half = corner / 2.0;
    // Along each ray, the tangent point sits r / tan(half) from the apex, and
    // the centre sits r / sin(half) along the bisector.
    let along = radius / half.tan();
    let tangent1 = [
        apex[0] + along * direction1[0],
        apex[1] + along * direction1[1],
    ];
    let tangent2 = [
        apex[0] + along * direction2[0],
        apex[1] + along * direction2[1],
    ];

    let bisector = [
        direction1[0] + direction2[0],
        direction1[1] + direction2[1],
    ];
    let length = (bisector[0] * bisector[0] + bisector[1] * bisector[1]).sqrt();
    if length < 1e-10 {
        return None;
    }
    let to_centre = radius / half.sin();
    let centre = [
        apex[0] + to_centre * bisector[0] / length,
        apex[1] + to_centre * bisector[1] / length,
    ];

    let angle1 = (tangent1[1] - centre[1]).atan2(tangent1[0] - centre[0]);
    let angle2 = (tangent2[1] - centre[1]).atan2(tangent2[0] - centre[0]);

    // Arcs are stored sweeping counter-clockwise, so the endpoints go in the
    // order that fills the corner rather than the one that goes the long way
    // round the outside. Which order that is follows the turn from the first
    // direction to the second.
    let turn = direction1[0] * direction2[1] - direction1[1] * direction2[0];
    let (start_angle, end_angle) = if turn <= 0.0 {
        (angle1, angle2)
    } else {
        (angle2, angle1)
    };

    Some(Fillet {
        tangent1,
        tangent2,
        centre,
        start_angle: normalize_angle(start_angle),
        end_angle: normalize_angle(end_angle),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::angle::arc_span;
    use std::f64::consts::{FRAC_PI_2, PI};

    /// A right-angle corner at the origin, opening into the first quadrant.
    fn right_angle(radius: f64) -> Fillet {
        fillet_between_rays([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], radius).unwrap()
    }

    #[test]
    fn the_tangent_points_sit_on_their_rays() {
        let fillet = right_angle(1.0);
        // Along +X and +Y respectively, at r / tan(45°) = r.
        assert!((fillet.tangent1[0] - 1.0).abs() < 1e-12 && fillet.tangent1[1].abs() < 1e-12);
        assert!(fillet.tangent2[0].abs() < 1e-12 && (fillet.tangent2[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_centre_is_the_radius_from_both_tangent_points() {
        let fillet = fillet_between_rays([3.0, -2.0], [1.0, 0.0], [0.0, 1.0], 2.5).unwrap();
        for tangent in [fillet.tangent1, fillet.tangent2] {
            let dx = tangent[0] - fillet.centre[0];
            let dy = tangent[1] - fillet.centre[1];
            assert!(((dx * dx + dy * dy).sqrt() - 2.5).abs() < 1e-9);
        }
    }

    #[test]
    fn the_arc_is_tangent_to_both_rays() {
        // Tangency means the centre-to-tangent vector is perpendicular to the
        // ray it touches.
        let d1 = [1.0, 0.0];
        let d2 = [0.6, 0.8];
        let fillet = fillet_between_rays([0.0, 0.0], d1, d2, 1.5).unwrap();
        for (tangent, direction) in [(fillet.tangent1, d1), (fillet.tangent2, d2)] {
            let radial = [tangent[0] - fillet.centre[0], tangent[1] - fillet.centre[1]];
            let dot = radial[0] * direction[0] + radial[1] * direction[1];
            assert!(dot.abs() < 1e-9, "not perpendicular, dot = {dot}");
        }
    }

    #[test]
    fn the_arc_fills_the_corner_rather_than_going_the_long_way() {
        // A right angle should be rounded by a quarter turn, not three.
        let fillet = right_angle(1.0);
        let span = arc_span(fillet.start_angle, fillet.end_angle);
        assert!(
            (span - FRAC_PI_2).abs() < 1e-9,
            "expected a quarter turn, swept {span}"
        );
    }

    #[test]
    fn the_sweep_stays_short_whichever_way_the_corner_turns() {
        // Same corner, rays given in the other order: still a quarter turn.
        let fillet = fillet_between_rays([0.0, 0.0], [0.0, 1.0], [1.0, 0.0], 1.0).unwrap();
        let span = arc_span(fillet.start_angle, fillet.end_angle);
        assert!((span - FRAC_PI_2).abs() < 1e-9, "swept {span}");
    }

    #[test]
    fn a_sharper_corner_needs_a_longer_sweep() {
        // 60° between the rays leaves 120° of arc to fill it.
        let d2 = [0.5, 3f64.sqrt() / 2.0];
        let fillet = fillet_between_rays([0.0, 0.0], [1.0, 0.0], d2, 1.0).unwrap();
        let span = arc_span(fillet.start_angle, fillet.end_angle);
        assert!((span - 2.0 * PI / 3.0).abs() < 1e-9, "swept {span}");
    }

    #[test]
    fn a_bigger_radius_pushes_the_tangent_points_further_out() {
        let small = right_angle(1.0);
        let large = right_angle(4.0);
        assert!(large.tangent1[0] > small.tangent1[0]);
        assert!(large.centre[1] > small.centre[1]);
    }

    #[test]
    fn parallel_and_opposed_rays_have_no_fillet() {
        assert!(fillet_between_rays([0.0, 0.0], [1.0, 0.0], [1.0, 0.0], 1.0).is_none());
        assert!(fillet_between_rays([0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], 1.0).is_none());
    }

    #[test]
    fn a_zero_radius_is_a_corner_not_a_fillet() {
        assert!(fillet_between_rays([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], 0.0).is_none());
    }

    #[test]
    fn survey_coordinates_keep_the_radius() {
        let apex = [512_345.678, 4_512_345.678];
        let fillet = fillet_between_rays(apex, [1.0, 0.0], [0.0, 1.0], 0.75).unwrap();
        let dx = fillet.tangent1[0] - fillet.centre[0];
        let dy = fillet.tangent1[1] - fillet.centre[1];
        assert!(((dx * dx + dy * dy).sqrt() - 0.75).abs() < 1e-9);
    }
}
