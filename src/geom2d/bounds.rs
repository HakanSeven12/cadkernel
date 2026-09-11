//! Analytic planar curve extents for placement and measurement.

use super::{Curve, Ellipse};
use super::angle::angle_within_arc;
use std::f64::consts::{PI, TAU};

type Bounds = ([f64; 2], [f64; 2]);

fn absorb(bounds: &mut Option<Bounds>, point: [f64; 2]) -> Option<()> {
    if !point.iter().all(|coordinate| coordinate.is_finite()) { return None; }
    if let Some((min, max)) = bounds {
        for axis in 0..2 { min[axis] = min[axis].min(point[axis]); max[axis] = max[axis].max(point[axis]); }
    } else { *bounds = Some((point, point)); }
    Some(())
}

fn conic(bounds: &mut Option<Bounds>, ellipse: Ellipse, start: f64, end: f64) -> Option<()> {
    if !start.is_finite() || !end.is_finite() || !ellipse.major_radius.is_finite()
        || !ellipse.minor_radius.is_finite() || ellipse.major_radius < 0.0 || ellipse.minor_radius < 0.0 { return None; }
    absorb(bounds, ellipse.point_at(start))?;
    absorb(bounds, ellipse.point_at(end))?;
    let minor = ellipse.minor_axis();
    for axis in 0..2 {
        // The coordinate derivative is -a*sin(t) + b*cos(t).
        let maximum = (ellipse.minor_radius * minor[axis]).atan2(ellipse.major_radius * ellipse.major_axis[axis]);
        for parameter in [maximum, maximum + PI] {
            if (end-start).abs() >= TAU || angle_within_arc(parameter, start, end) {
                absorb(bounds, ellipse.point_at(parameter))?;
            }
        }
    }
    Some(())
}

fn circular(bounds: &mut Option<Bounds>, centre: [f64;2], radius:f64, start:f64, end:f64) -> Option<()> {
    conic(bounds, Ellipse { centre, major_radius:radius, minor_radius:radius, major_axis:[1.0,0.0] }, start, end)
}

/// Tight axis-aligned extents of bounded analytic curves, including partial conics
/// and signed polyline bulges. Returns `None` for invalid/unbounded inputs, empty
/// collections or NURBS, whose extrema require a separate root-isolation algorithm.
/// This deliberately never substitutes a tessellation or control hull.
pub fn analytic_curve_bounds(curves: &[Curve]) -> Option<Bounds> {
    let mut bounds = None;
    for curve in curves {
        match curve {
            Curve::Line(line) => { absorb(&mut bounds, line.start)?; absorb(&mut bounds, line.end)?; }
            Curve::Circle(circle) => { circular(&mut bounds, circle.centre, circle.radius, 0.0, TAU)?; }
            Curve::Arc(arc) => { circular(&mut bounds, arc.centre, arc.radius, arc.start_angle, arc.end_angle)?; }
            Curve::Ellipse(arc) => { conic(&mut bounds, arc.ellipse, arc.start_parameter, arc.end_parameter)?; }
            Curve::Polyline(polyline) => {
                for vertex in &polyline.vertices { absorb(&mut bounds, vertex.position)?; if !vertex.bulge.is_finite() { return None; } }
                let count = if polyline.closed { polyline.vertices.len() } else { polyline.vertices.len().saturating_sub(1) };
                for index in 0..count {
                    if let Some(arc) = polyline.segment_arc(index) {
                        let (start,end) = if arc.sweep >= 0.0 { (arc.start_angle,arc.start_angle+arc.sweep) } else { (arc.start_angle+arc.sweep,arc.start_angle) };
                        circular(&mut bounds,arc.center,arc.radius,start,end)?;
                    }
                }
            }
            Curve::Ray(_) | Curve::XLine(_) | Curve::Nurbs(_) => return None,
        }
    }
    bounds
}
