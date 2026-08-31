use cadcodec::objects::SolidHistoryOperation;

use crate::brep::{self, Body, Placement};

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
            let radius = circular_radius(
                value.major_radius,
                value.minor_radius,
                value.x_radius,
            )?;
            finish(
                brep::make::cylinder([0.0; 3], radius, value.height),
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
            if value.top_radius.abs() > 1e-9 || value.sides < 3 {
                return Err(HistoryRebuildError::Unsupported);
            }
            finish(
                brep::make::pyramid(
                    [0.0; 3],
                    value.radius,
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
        _ => Err(HistoryRebuildError::Unsupported),
    }
}
