//! Geometric relations that are already present in a set of sketch curves.

use super::{Arc, Circle, Line, Tolerance, Vec2};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SketchPrimitive {
    Line(Line),
    Circle(Circle),
    Arc(Arc),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    Start,
    End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferredConstraint {
    Coincident {
        first: usize,
        first_endpoint: Endpoint,
        second: usize,
        second_endpoint: Endpoint,
    },
    Collinear {
        first: usize,
        second: usize,
    },
    Concentric {
        first: usize,
        second: usize,
    },
    Parallel {
        first: usize,
        second: usize,
    },
    Perpendicular {
        first: usize,
        second: usize,
    },
    Horizontal {
        entity: usize,
    },
    Vertical {
        entity: usize,
    },
    Tangent {
        first: usize,
        second: usize,
    },
}

fn endpoints(primitive: SketchPrimitive) -> Option<[[f64; 2]; 2]> {
    match primitive {
        SketchPrimitive::Line(line) => Some([line.start, line.end]),
        SketchPrimitive::Arc(arc) => Some([
            [
                arc.centre[0] + arc.radius * arc.start_angle.cos(),
                arc.centre[1] + arc.radius * arc.start_angle.sin(),
            ],
            [
                arc.centre[0] + arc.radius * arc.end_angle.cos(),
                arc.centre[1] + arc.radius * arc.end_angle.sin(),
            ],
        ]),
        SketchPrimitive::Circle(_) => None,
    }
}

fn circle(primitive: SketchPrimitive) -> Option<Circle> {
    match primitive {
        SketchPrimitive::Circle(circle) => Some(circle),
        SketchPrimitive::Arc(arc) => Some(Circle {
            centre: arc.centre,
            radius: arc.radius,
        }),
        SketchPrimitive::Line(_) => None,
    }
}

fn unit_line(line: Line) -> Option<Vec2> {
    Vec2::from(line.direction()).normalize()
}

fn line_distance(line: Line, point: [f64; 2]) -> Option<f64> {
    let direction = unit_line(line)?;
    Some(
        (Vec2::from(point) - Vec2::from(line.start))
            .cross(direction)
            .abs(),
    )
}

/// Returns the constraints already implied by `primitives`, in application
/// priority order. This never changes geometry; the caller decides which
/// returned relations to persist.
pub fn infer_constraints(
    primitives: &[SketchPrimitive],
    distance_tolerance: Tolerance,
    angle_tolerance_radians: f64,
) -> Vec<InferredConstraint> {
    if !angle_tolerance_radians.is_finite() || angle_tolerance_radians < 0.0 {
        return Vec::new();
    }
    let distance = distance_tolerance.linear();
    let sin_angle = angle_tolerance_radians.sin().abs();
    let mut coincident = Vec::new();
    let mut collinear = Vec::new();
    let mut concentric = Vec::new();
    let mut parallel = Vec::new();
    let mut perpendicular = Vec::new();
    let mut horizontal = Vec::new();
    let mut vertical = Vec::new();
    let mut tangent = Vec::new();

    for (index, primitive) in primitives.iter().copied().enumerate() {
        if let SketchPrimitive::Line(line) = primitive {
            if let Some(direction) = unit_line(line) {
                if direction.y.abs() <= sin_angle {
                    horizontal.push(InferredConstraint::Horizontal { entity: index });
                }
                if direction.x.abs() <= sin_angle {
                    vertical.push(InferredConstraint::Vertical { entity: index });
                }
            }
        }
    }

    for first in 0..primitives.len() {
        for second in first + 1..primitives.len() {
            if let (Some(a), Some(b)) =
                (endpoints(primitives[first]), endpoints(primitives[second]))
            {
                for (ai, ap) in a.into_iter().enumerate() {
                    for (bi, bp) in b.into_iter().enumerate() {
                        if Vec2::from(ap).distance(Vec2::from(bp)) <= distance {
                            coincident.push(InferredConstraint::Coincident {
                                first,
                                first_endpoint: if ai == 0 {
                                    Endpoint::Start
                                } else {
                                    Endpoint::End
                                },
                                second,
                                second_endpoint: if bi == 0 {
                                    Endpoint::Start
                                } else {
                                    Endpoint::End
                                },
                            });
                        }
                    }
                }
            }

            if let (SketchPrimitive::Line(a), SketchPrimitive::Line(b)) =
                (primitives[first], primitives[second])
            {
                if let (Some(ua), Some(ub)) = (unit_line(a), unit_line(b)) {
                    let cross = ua.cross(ub).abs();
                    if cross <= sin_angle {
                        if line_distance(a, b.start).is_some_and(|value| value <= distance) {
                            collinear.push(InferredConstraint::Collinear { first, second });
                        } else {
                            parallel.push(InferredConstraint::Parallel { first, second });
                        }
                    } else if ua.dot(ub).abs() <= sin_angle {
                        perpendicular.push(InferredConstraint::Perpendicular { first, second });
                    }
                }
            }

            match (circle(primitives[first]), circle(primitives[second])) {
                (Some(a), Some(b)) => {
                    let centres = Vec2::from(a.centre).distance(Vec2::from(b.centre));
                    if centres <= distance {
                        concentric.push(InferredConstraint::Concentric { first, second });
                    } else if (centres - (a.radius + b.radius)).abs() <= distance
                        || (centres - (a.radius - b.radius).abs()).abs() <= distance
                    {
                        tangent.push(InferredConstraint::Tangent { first, second });
                    }
                }
                (Some(circle), None) if matches!(primitives[second], SketchPrimitive::Line(_)) => {
                    let SketchPrimitive::Line(line) = primitives[second] else {
                        unreachable!()
                    };
                    if line_distance(line, circle.centre)
                        .is_some_and(|value| (value - circle.radius).abs() <= distance)
                    {
                        tangent.push(InferredConstraint::Tangent { first, second });
                    }
                }
                (None, Some(circle)) if matches!(primitives[first], SketchPrimitive::Line(_)) => {
                    let SketchPrimitive::Line(line) = primitives[first] else {
                        unreachable!()
                    };
                    if line_distance(line, circle.centre)
                        .is_some_and(|value| (value - circle.radius).abs() <= distance)
                    {
                        tangent.push(InferredConstraint::Tangent { first, second });
                    }
                }
                _ => {}
            }
        }
    }

    coincident
        .into_iter()
        .chain(collinear)
        .chain(concentric)
        .chain(parallel)
        .chain(perpendicular)
        .chain(horizontal)
        .chain(vertical)
        .chain(tangent)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relations_are_in_priority_order() {
        let curves = [
            SketchPrimitive::Line(Line {
                start: [0.0, 0.0],
                end: [2.0, 0.0],
            }),
            SketchPrimitive::Line(Line {
                start: [2.0, 0.0],
                end: [4.0, 0.0],
            }),
        ];
        let found = infer_constraints(&curves, Tolerance::new(1e-6), 1e-6);
        assert!(matches!(found[0], InferredConstraint::Coincident { .. }));
        assert!(matches!(found[1], InferredConstraint::Collinear { .. }));
        assert!(found
            .iter()
            .any(|item| matches!(item, InferredConstraint::Horizontal { .. })));
    }

    #[test]
    fn infers_circle_line_tangency() {
        let curves = [
            SketchPrimitive::Circle(Circle {
                centre: [0.0, 0.0],
                radius: 2.0,
            }),
            SketchPrimitive::Line(Line {
                start: [-3.0, 2.0],
                end: [3.0, 2.0],
            }),
        ];
        let found = infer_constraints(&curves, Tolerance::new(1e-6), 1e-6);
        assert!(found
            .iter()
            .any(|item| matches!(item, InferredConstraint::Tangent { .. })));
    }
}
