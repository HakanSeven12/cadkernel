use cadcodec::entities::EmbeddedEntity;
use cadcodec::objects::{SolidHistoryLoft, SolidHistoryOperation, SolidHistorySweep};
use cadcodec::types::{Matrix3, Vector3};

use crate::brep::{self, Body, Placement};
use crate::geom2d::{
    Arc, Circle, Curve, Ellipse, EllipseArc, Line, NurbsCurve, Parameterization, Polyline,
    PolylineVertex,
};
use crate::space::{coplanarity_tolerance, PlanarCurve, Plane, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryRebuildError {
    Unsupported,
    InvalidParameters,
    InvalidTransform,
    InvalidBrep,
}

impl std::fmt::Display for HistoryRebuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Unsupported => "unsupported solid history operation",
            Self::InvalidParameters => "invalid solid history parameters",
            Self::InvalidTransform => "invalid solid history transform",
            Self::InvalidBrep => "invalid solid history B-rep",
        })
    }
}

impl std::error::Error for HistoryRebuildError {}

fn placement(matrix: [f64; 16]) -> Result<Placement, HistoryRebuildError> {
    if matrix.iter().any(|value| !value.is_finite())
        || matrix[3].abs() > 1e-9
        || matrix[7].abs() > 1e-9
        || matrix[11].abs() > 1e-9
        || (matrix[15] - 1.0).abs() > 1e-9
    {
        return Err(HistoryRebuildError::InvalidTransform);
    }
    Ok(Placement {
        x_axis: [matrix[0], matrix[1], matrix[2]],
        y_axis: [matrix[4], matrix[5], matrix[6]],
        z_axis: [matrix[8], matrix[9], matrix[10]],
        origin: [matrix[12], matrix[13], matrix[14]],
    })
}

fn finish(
    body: Option<Body>,
    transform: [f64; 16],
) -> Result<Body, HistoryRebuildError> {
    let body = body.ok_or(HistoryRebuildError::InvalidParameters)?;
    brep::transform(&body, &placement(transform)?)
        .ok_or(HistoryRebuildError::InvalidTransform)
}

fn circular_radius(
    major: f64,
    minor: f64,
    x_radius: f64,
) -> Result<f64, HistoryRebuildError> {
    if [major, minor, x_radius]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(HistoryRebuildError::InvalidParameters);
    }
    let tolerance = 1e-9 * major.max(minor).max(x_radius);
    if (major - minor).abs() > tolerance || (major - x_radius).abs() > tolerance {
        return Err(HistoryRebuildError::Unsupported);
    }
    Ok(major)
}

fn ocs_plane(normal: Vector3, elevation: f64) -> Result<Plane, HistoryRebuildError> {
    let normal = Vec3::from([normal.x, normal.y, normal.z])
        .normalize()
        .ok_or(HistoryRebuildError::InvalidParameters)?;
    let normal = Vector3::new(normal.x, normal.y, normal.z);
    let axes = Matrix3::arbitrary_axis(normal);
    Ok(Plane::from_axes(
        [
            normal.x * elevation,
            normal.y * elevation,
            normal.z * elevation,
        ],
        [axes.m[0][0], axes.m[1][0], axes.m[2][0]],
        [axes.m[0][1], axes.m[1][1], axes.m[2][1]],
    ))
}

fn straight_curve(
    start: Vector3,
    end: Vector3,
) -> Result<PlanarCurve, HistoryRebuildError> {
    let start = [start.x, start.y, start.z];
    let end = [end.x, end.y, end.z];
    let direction = Vec3::from(end) - Vec3::from(start);
    if direction.length_squared() <= 1e-24 {
        return Err(HistoryRebuildError::InvalidParameters);
    }
    let plane = if (end[2] - start[2]).abs() <= 1e-9 * direction.length().max(1.0) {
        Plane::from_axes([0.0, 0.0, start[2]], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    } else {
        let normal = direction.cross(Vec3::Z).normalize().unwrap_or(Vec3::Y);
        Plane::orthonormal(start, direction.to_array(), normal.to_array())
            .ok_or(HistoryRebuildError::InvalidParameters)?
    };
    let start = plane
        .project(start)
        .ok_or(HistoryRebuildError::InvalidParameters)?;
    let end = plane
        .project(end)
        .ok_or(HistoryRebuildError::InvalidParameters)?;
    Ok(PlanarCurve::new(plane, Curve::Line(Line { start, end })))
}

fn spline_curve(
    value: &cadcodec::entities::Spline,
) -> Result<PlanarCurve, HistoryRebuildError> {
    let degree = value.degree.max(1) as usize;
    let fit_method = !value.fit_points.is_empty() && value.control_points.len() <= degree;
    let source = if fit_method {
        &value.fit_points
    } else {
        &value.control_points
    };
    let first = source
        .first()
        .ok_or(HistoryRebuildError::InvalidParameters)?;
    let normal = Vec3::from([value.normal.x, value.normal.y, value.normal.z])
        .normalize()
        .ok_or(HistoryRebuildError::InvalidParameters)?;
    let elevation = Vec3::from([first.x, first.y, first.z]).dot(normal);
    let plane = ocs_plane(
        Vector3::new(normal.x, normal.y, normal.z),
        elevation,
    )?;
    let source_points = source
        .iter()
        .map(|point| [point.x, point.y, point.z])
        .collect::<Vec<_>>();
    let tolerance = coplanarity_tolerance(&source_points);
    if !tolerance.is_finite()
        || !source_points
            .iter()
            .all(|point| plane.contains(*point, tolerance))
    {
        return Err(HistoryRebuildError::InvalidParameters);
    }
    let point = |value: &Vector3| {
        plane
            .project([value.x, value.y, value.z])
            .ok_or(HistoryRebuildError::InvalidParameters)
    };
    let nurbs = if fit_method {
        if !value.flags.periodic {
            for tangent in [value.begin_tangent, value.end_tangent] {
                let tangent = Vec3::from([tangent.x, tangent.y, tangent.z]);
                if tangent.length_squared() > 1e-18
                    && tangent.dot(normal).abs() > 1e-9 * tangent.length().max(1.0)
                {
                    return Err(HistoryRebuildError::InvalidParameters);
                }
            }
        }
        let mut points = value
            .fit_points
            .iter()
            .map(point)
            .collect::<Result<Vec<_>, _>>()?;
        let parameterization = match value.knot_parameterization {
            2 => Parameterization::Uniform,
            1 => Parameterization::Centripetal,
            _ => Parameterization::Chord,
        };
        if value.flags.periodic {
            NurbsCurve::interpolate_periodic(&points, parameterization)
        } else {
            if value.flags.closed && points.first() != points.last() {
                points.push(points[0]);
            }
            let tangent = |value: Vector3| {
                let projected = plane.project_vector([value.x, value.y, value.z])?;
                (projected[0].hypot(projected[1]) > 1e-9).then_some(projected)
            };
            NurbsCurve::interpolate(
                &points,
                tangent(value.begin_tangent),
                tangent(value.end_tangent),
                parameterization,
            )
        }
    } else {
        NurbsCurve::new(
            degree,
            value
                .control_points
                .iter()
                .map(point)
                .collect::<Result<Vec<_>, _>>()?,
            value.knots.clone(),
            (!value.weights.is_empty()).then(|| value.weights.clone()),
        )
    }
    .ok_or(HistoryRebuildError::InvalidParameters)?;
    Ok(PlanarCurve::new(plane, Curve::Nurbs(nurbs)))
}

fn embedded_curve(entity: &EmbeddedEntity) -> Result<PlanarCurve, HistoryRebuildError> {
    match entity {
        EmbeddedEntity::Line(value) => straight_curve(value.start, value.end),
        EmbeddedEntity::Circle(value) => Ok(PlanarCurve::new(
            ocs_plane(value.normal, value.center.z)?,
            Curve::Circle(Circle {
                centre: [value.center.x, value.center.y],
                radius: value.radius,
            }),
        )),
        EmbeddedEntity::Arc(value) => Ok(PlanarCurve::new(
            ocs_plane(value.normal, value.center.z)?,
            Curve::Arc(Arc {
                centre: [value.center.x, value.center.y],
                radius: value.radius,
                start_angle: value.start_angle,
                end_angle: value.end_angle,
            }),
        )),
        EmbeddedEntity::Ellipse(value) => {
            let normal = Vec3::from([value.normal.x, value.normal.y, value.normal.z])
                .normalize()
                .ok_or(HistoryRebuildError::InvalidParameters)?;
            let center = [value.center.x, value.center.y, value.center.z];
            let elevation = Vec3::from(center).dot(normal);
            let plane = ocs_plane(
                Vector3::new(normal.x, normal.y, normal.z),
                elevation,
            )?;
            let centre = plane
                .project(center)
                .ok_or(HistoryRebuildError::InvalidParameters)?;
            let major = plane
                .project_vector([
                    value.major_axis.x,
                    value.major_axis.y,
                    value.major_axis.z,
                ])
                .ok_or(HistoryRebuildError::InvalidParameters)?;
            let major_radius = major[0].hypot(major[1]);
            if !major_radius.is_finite()
                || major_radius <= 0.0
                || !value.minor_axis_ratio.is_finite()
                || value.minor_axis_ratio <= 0.0
            {
                return Err(HistoryRebuildError::InvalidParameters);
            }
            Ok(PlanarCurve::new(
                plane,
                Curve::Ellipse(EllipseArc {
                    ellipse: Ellipse {
                        centre,
                        major_radius,
                        minor_radius: major_radius * value.minor_axis_ratio,
                        major_axis: [major[0] / major_radius, major[1] / major_radius],
                    },
                    start_parameter: value.start_parameter,
                    end_parameter: value.end_parameter,
                }),
            ))
        }
        EmbeddedEntity::Spline(value) => spline_curve(value),
        EmbeddedEntity::LwPolyline(value) => {
            if value.vertices.len() < 2 {
                return Err(HistoryRebuildError::InvalidParameters);
            }
            Ok(PlanarCurve::new(
                ocs_plane(value.normal, value.elevation)?,
                Curve::Polyline(Polyline {
                    vertices: value
                        .vertices
                        .iter()
                        .map(|vertex| PolylineVertex {
                            position: [vertex.location.x, vertex.location.y],
                            bulge: vertex.bulge,
                        })
                        .collect(),
                    closed: value.is_closed,
                }),
            ))
        }
        _ => Err(HistoryRebuildError::Unsupported),
    }
}

fn placed_curve(
    mut curve: PlanarCurve,
    transform: [f64; 16],
) -> Result<PlanarCurve, HistoryRebuildError> {
    let placement = placement(transform)?;
    if placement.scale().is_none() {
        return Err(HistoryRebuildError::InvalidTransform);
    }
    curve.plane = Plane::from_axes(
        placement.point(curve.plane.origin),
        placement.vector(curve.plane.x_axis),
        placement.vector(curve.plane.y_axis),
    );
    Ok(curve)
}

fn profile_pieces(curve: &Curve) -> Result<Vec<Curve>, HistoryRebuildError> {
    let pieces = match curve {
        Curve::Polyline(value) if value.closed => curve.segments(),
        Curve::Circle(value) => {
            (0..4)
                .map(|part| {
                    let start = std::f64::consts::FRAC_PI_2 * part as f64;
                    Curve::Arc(Arc {
                        centre: value.centre,
                        radius: value.radius,
                        start_angle: start,
                        end_angle: start + std::f64::consts::FRAC_PI_2,
                    })
                })
                .collect()
        }
        Curve::Arc(value) if curve.is_closed() => {
            let start = value.start_angle;
            let step = value.sweep() / 4.0;
            (0..4)
                .map(|part| {
                    Curve::Arc(Arc {
                        centre: value.centre,
                        radius: value.radius,
                        start_angle: start + step * part as f64,
                        end_angle: start + step * (part + 1) as f64,
                    })
                })
                .collect()
        }
        Curve::Ellipse(value) if curve.is_closed() => {
            let start = value.start_parameter;
            let step = value.sweep() / 4.0;
            (0..4)
                .map(|part| {
                    Curve::Ellipse(EllipseArc {
                        ellipse: value.ellipse,
                        start_parameter: start + step * part as f64,
                        end_parameter: start + step * (part + 1) as f64,
                    })
                })
                .collect()
        }
        _ => return Err(HistoryRebuildError::InvalidParameters),
    };
    (pieces.len() >= 3)
        .then_some(pieces)
        .ok_or(HistoryRebuildError::InvalidParameters)
}

fn path_pieces(curve: &Curve) -> Result<Vec<Curve>, HistoryRebuildError> {
    let pieces = match curve {
        Curve::Line(value) => vec![Curve::Line(*value)],
        Curve::Arc(value) => vec![Curve::Arc(*value)],
        Curve::Circle(value) => (0..4)
            .map(|part| {
                let start = std::f64::consts::FRAC_PI_2 * part as f64;
                Curve::Arc(Arc {
                    centre: value.centre,
                    radius: value.radius,
                    start_angle: start,
                    end_angle: start + std::f64::consts::FRAC_PI_2,
                })
            })
            .collect(),
        Curve::Polyline(_) => curve.segments(),
        Curve::Ellipse(value) => {
            let scale = value
                .ellipse
                .major_radius
                .max(value.ellipse.minor_radius)
                .max(1.0);
            let points = curve.tessellate_within(scale * 1e-4);
            points
                .windows(2)
                .filter(|pair| pair[0] != pair[1])
                .map(|pair| {
                    Curve::Line(Line {
                        start: pair[0],
                        end: pair[1],
                    })
                })
                .collect()
        }
        Curve::Nurbs(_) => {
            let points = curve.tessellate_within(1e-4);
            points
                .windows(2)
                .filter(|pair| pair[0] != pair[1])
                .map(|pair| {
                    Curve::Line(Line {
                        start: pair[0],
                        end: pair[1],
                    })
                })
                .collect()
        }
        _ => return Err(HistoryRebuildError::Unsupported),
    };
    (!pieces.is_empty())
        .then_some(pieces)
        .ok_or(HistoryRebuildError::InvalidParameters)
}

fn rebuild_sweep(value: &SolidHistorySweep) -> Result<Body, HistoryRebuildError> {
    if !value.scale_factor.is_finite()
        || value.scale_factor <= 1e-9
        || value.draft_angle.abs() > 1e-9
        || !value.twist_angle.is_finite()
        || !value.align_angle.is_finite()
    {
        return Err(HistoryRebuildError::Unsupported);
    }
    let profile = placed_curve(
        embedded_curve(
            value
                .sweep_entity
                .as_ref()
                .ok_or(HistoryRebuildError::InvalidParameters)?,
        )?,
        value.sweep_entity_transform,
    )?;
    let path = placed_curve(
        embedded_curve(
            value
                .path_entity
                .as_ref()
                .ok_or(HistoryRebuildError::InvalidParameters)?,
        )?,
        value.path_entity_transform,
    )?;
    finish(
        brep::sweep_along_deformed(
            profile.plane,
            &profile_pieces(&profile.curve)?,
            path.plane,
            &path_pieces(&path.curve)?,
            value.align_angle,
            value.twist_angle,
            value.scale_factor,
        ),
        value.base.transform,
    )
}

fn rebuild_loft(value: &SolidHistoryLoft) -> Result<Body, HistoryRebuildError> {
    if !value.guides.is_empty() {
        return Err(HistoryRebuildError::Unsupported);
    }
    let sections = value
        .cross_sections
        .iter()
        .map(|entity| {
            let curve = embedded_curve(entity)?;
            Ok((curve.plane, profile_pieces(&curve.curve)?))
        })
        .collect::<Result<Vec<_>, HistoryRebuildError>>()?;
    if sections.len() < 2 {
        return Err(HistoryRebuildError::InvalidParameters);
    }
    finish(brep::loft(&sections), value.base.transform)
}

pub fn rebuild_body(
    operation: &SolidHistoryOperation,
) -> Result<Body, HistoryRebuildError> {
    match operation {
        SolidHistoryOperation::Box(value) => finish(
            brep::make::cuboid(
                [0.0; 3],
                [value.length, value.width, value.height],
            ),
            value.base.transform,
        ),
        SolidHistoryOperation::Wedge(value) => finish(
            brep::make::wedge(
                [0.0; 3],
                value.length,
                value.width,
                value.height,
            ),
            value.base.transform,
        ),
        SolidHistoryOperation::Sphere(value) => finish(
            brep::make::sphere([0.0; 3], value.radius),
            value.base.transform,
        ),
        SolidHistoryOperation::Cylinder(value) => {
            if [value.major_radius, value.minor_radius, value.x_radius]
                .iter()
                .any(|radius| !radius.is_finite() || *radius <= 0.0)
            {
                return Err(HistoryRebuildError::InvalidParameters);
            }
            finish(
                brep::make::elliptical_cylinder(
                    [0.0; 3],
                    value.major_radius,
                    value.minor_radius,
                    value.height,
                ),
                value.base.transform,
            )
        }
        SolidHistoryOperation::Cone(value) => {
            finish(
                brep::make::frustum(
                    [0.0; 3],
                    value.base_x_radius,
                    value.base_y_radius,
                    value.top_radius,
                    value.height,
                ),
                value.base.transform,
            )
        }
        SolidHistoryOperation::Pyramid(value) => {
            if value.sides < 3 {
                return Err(HistoryRebuildError::InvalidParameters);
            }
            finish(
                brep::make::pyramid_frustum(
                    [0.0; 3],
                    value.radius,
                    value.top_radius,
                    value.height,
                    value.sides as usize,
                ),
                value.base.transform,
            )
        }
        SolidHistoryOperation::Torus(value) => finish(
            brep::make::torus(
                [0.0; 3],
                value.major_radius,
                value.minor_radius,
            ),
            value.base.transform,
        ),
        SolidHistoryOperation::Brep(value) => {
            let document = value
                .acis_data
                .parse()
                .ok_or(HistoryRebuildError::InvalidBrep)?;
            let (bodies, _) = super::lift(&document);
            finish(bodies.into_iter().next(), value.base.transform)
                .map_err(|error| match error {
                    HistoryRebuildError::InvalidParameters => {
                        HistoryRebuildError::InvalidBrep
                    }
                    other => other,
                })
        }
        SolidHistoryOperation::Sweep(value) => rebuild_sweep(value),
        SolidHistoryOperation::Loft(value) => rebuild_loft(value),
        _ => Err(HistoryRebuildError::Unsupported),
    }
}
