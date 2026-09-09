//! Turning an open sheet body into a closed solid of constant thickness.
//!
//! Planar and rectangular analytic faces stay analytic. Other trimmed or
//! multi-face sheets use a welded faceted fallback so the operation remains
//! useful without claiming an inexact offset is an exact spline surface.

use super::geometry::Surface;
use super::mesh::TessellationTolerance;
use super::topology::{Body, FaceKey};
use crate::geom2d::{Arc, Curve, Line};
use crate::space::{Plane, Vec3};
use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, TAU};

/// Why a surface could not be thickened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThickenError {
    /// The requested distance was zero or not finite.
    InvalidDistance,
    /// The input has no bounded sheet geometry that can become a solid.
    UnsupportedSurface,
    /// The offset collapses or crosses the source surface.
    SelfIntersection,
}

/// Thickens every bounded face of an open sheet by a signed distance.
///
/// Positive distance follows each face's oriented normal. The source body is
/// borrowed and never changed, including when an offset fails.
pub fn thicken(body: &Body, distance: f64) -> Result<Body, ThickenError> {
    if !distance.is_finite() || distance.abs() <= f64::EPSILON {
        return Err(ThickenError::InvalidDistance);
    }
    if !body.validate().is_empty() || body.faces.keys().next().is_none() {
        return Err(ThickenError::UnsupportedSurface);
    }

    let faces = body.face_keys().collect::<Vec<_>>();
    if faces.len() == 1 {
        let face_key = faces[0];
        let face = body
            .faces
            .get(face_key)
            .ok_or(ThickenError::UnsupportedSurface)?;
        let surface = body
            .surfaces
            .get(face.surface)
            .ok_or(ThickenError::UnsupportedSurface)?;
        let oriented = distance * if face.forward { 1.0 } else { -1.0 };

        if matches!(surface, Surface::Plane(_)) {
            let profile = super::planar_face_profile(body, face_key)
                .ok_or(ThickenError::UnsupportedSurface)?;
            return super::extrude_region(
                profile.plane,
                &profile.loops,
                (Vec3::from(profile.outward) * distance).to_array(),
            )
            .ok_or(ThickenError::SelfIntersection);
        }

        if let Some(patch) = rectangular_patch(body, face_key, surface) {
            return thicken_revolved_patch(surface, patch, oriented);
        }
    }

    thicken_faceted(body, distance)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Patch {
    pub(crate) u_start: f64,
    pub(crate) u_sweep: f64,
    pub(crate) v_start: f64,
    pub(crate) v_sweep: f64,
}

pub(crate) fn rectangular_patch(
    body: &Body,
    face_key: FaceKey,
    surface: &Surface,
) -> Option<Patch> {
    let face = body.faces.get(face_key)?;
    if face.loops.len() != 1 {
        return None;
    }
    let ring = body.loops.get(face.loops[0])?;
    if ring.coedges.len() < 2 {
        return None;
    }

    let mut all_u = Vec::new();
    let mut all_v = Vec::new();
    let mut full_u = false;
    let mut full_v = false;
    for coedge_key in &ring.coedges {
        let coedge = body.coedges.get(*coedge_key)?;
        let edge = body.edges.get(coedge.edge)?;
        let curve = body.curves.get(edge.curve)?;
        let closed = Vec3::from(curve.point_at(edge.start_parameter))
            .distance(Vec3::from(curve.point_at(edge.end_parameter)))
            <= 1e-8;
        let mut edge_u = Vec::new();
        let mut edge_v = Vec::new();
        for sample in 0..=16 {
            let fraction = sample as f64 / 16.0;
            let parameter = edge.start_parameter
                + (edge.end_parameter - edge.start_parameter) * fraction;
            let (u, v) = surface.parameters_at(curve.point_at(parameter))?;
            edge_u.push(u);
            edge_v.push(v);
            all_u.push(u);
            all_v.push(v);
        }
        let (_, u_sweep) = angular_cover(&edge_u)?;
        let v_span = span(&edge_v)?;
        let u_constant = u_sweep <= 1e-7;
        let v_constant = match surface {
            Surface::Torus(_) => angular_cover(&edge_v)?.1 <= 1e-7,
            _ => v_span <= 1e-7,
        };
        if !u_constant && !v_constant {
            return None;
        }
        if closed && v_constant && u_sweep > 1e-4 {
            full_u = true;
        }
        if closed && u_constant && matches!(surface, Surface::Torus(_)) && v_span > 1e-4 {
            full_v = true;
        }
    }

    let (u_start, mut u_sweep) = angular_cover(&all_u)?;
    if full_u {
        u_sweep = TAU;
    }
    let (v_start, v_sweep) = match surface {
        Surface::Torus(_) => {
            let (start, mut sweep) = angular_cover(&all_v)?;
            if full_v {
                sweep = TAU;
            }
            (start, sweep)
        }
        _ => {
            let (low, high) = range(&all_v)?;
            (low, high - low)
        }
    };
    (u_sweep > 1e-8 && v_sweep > 1e-8).then_some(Patch {
        u_start,
        u_sweep,
        v_start,
        v_sweep,
    })
}

fn thicken_revolved_patch(
    surface: &Surface,
    patch: Patch,
    distance: f64,
) -> Result<Body, ThickenError> {
    let (frame, profile) = match surface {
        Surface::Cylinder(cylinder) => {
            let second = cylinder.radius + distance;
            positive_radii(&[cylinder.radius, second])?;
            let v0 = patch.v_start;
            let v1 = patch.v_start + patch.v_sweep;
            (
                cylinder.base,
                line_ring(&[
                    [cylinder.radius, v0],
                    [cylinder.radius, v1],
                    [second, v1],
                    [second, v0],
                ]),
            )
        }
        Surface::Cone(cone) => {
            let sine = cone.half_angle.sin();
            let cosine = cone.half_angle.cos();
            let tangent = cone.half_angle.tan();
            let v0 = patch.v_start;
            let v1 = patch.v_start + patch.v_sweep;
            let source0 = cone.radius - v0 * tangent;
            let source1 = cone.radius - v1 * tangent;
            let offset0 = source0 + distance * cosine;
            let offset1 = source1 + distance * cosine;
            positive_radii(&[source0, source1, offset0, offset1])?;
            (
                cone.base,
                line_ring(&[
                    [source0, v0],
                    [source1, v1],
                    [offset1, v1 + distance * sine],
                    [offset0, v0 + distance * sine],
                ]),
            )
        }
        Surface::Sphere(sphere) => {
            let second = sphere.radius + distance;
            positive_radii(&[sphere.radius, second])?;
            let v0 = patch.v_start;
            let v1 = patch.v_start + patch.v_sweep;
            if v0 < -FRAC_PI_2 - 1e-7 || v1 > FRAC_PI_2 + 1e-7 {
                return Err(ThickenError::UnsupportedSurface);
            }
            (
                sphere.frame,
                arc_ring([0.0, 0.0], sphere.radius, second, v0, v1),
            )
        }
        Surface::Torus(torus) => {
            let second = torus.minor_radius + distance;
            positive_radii(&[torus.minor_radius, second])?;
            if second >= torus.major_radius - 1e-9 {
                return Err(ThickenError::SelfIntersection);
            }
            let v0 = patch.v_start;
            let v1 = patch.v_start + patch.v_sweep;
            (
                torus.frame,
                arc_ring(
                    [torus.major_radius, 0.0],
                    torus.minor_radius,
                    second,
                    v0,
                    v1,
                ),
            )
        }
        Surface::Plane(_) | Surface::Nurbs(_) => {
            return Err(ThickenError::UnsupportedSurface)
        }
    };

    let radial = frame.vector_at([patch.u_start.cos(), patch.u_start.sin()]);
    let axis = frame.normal().ok_or(ThickenError::UnsupportedSurface)?;
    let section = Plane::from_axes(frame.origin, radial, axis);
    super::revolve(section, &profile, frame.origin, axis, patch.u_sweep)
        .ok_or(ThickenError::SelfIntersection)
}

fn line_ring(points: &[[f64; 2]; 4]) -> Vec<Curve> {
    (0..4)
        .map(|index| {
            Curve::Line(Line {
                start: points[index],
                end: points[(index + 1) % 4],
            })
        })
        .collect()
}

fn arc_ring(
    centre: [f64; 2],
    first_radius: f64,
    second_radius: f64,
    start: f64,
    end: f64,
) -> Vec<Curve> {
    let at = |radius: f64, angle: f64| {
        [
            centre[0] + radius * angle.cos(),
            centre[1] + radius * angle.sin(),
        ]
    };
    vec![
        Curve::Arc(Arc {
            centre,
            radius: first_radius,
            start_angle: start,
            end_angle: end,
        }),
        Curve::Line(Line {
            start: at(first_radius, end),
            end: at(second_radius, end),
        }),
        Curve::Arc(Arc {
            centre,
            radius: second_radius,
            start_angle: start,
            end_angle: end,
        }),
        Curve::Line(Line {
            start: at(second_radius, start),
            end: at(first_radius, start),
        }),
    ]
}

fn positive_radii(values: &[f64]) -> Result<(), ThickenError> {
    if values.iter().all(|value| value.is_finite() && *value > 1e-9) {
        Ok(())
    } else {
        Err(ThickenError::SelfIntersection)
    }
}

fn thicken_faceted(body: &Body, distance: f64) -> Result<Body, ThickenError> {
    let bounds = super::body_bounds(body).ok_or(ThickenError::UnsupportedSurface)?;
    let scale = Vec3::from(bounds.max)
        .distance(Vec3::from(bounds.min))
        .max(distance.abs())
        .max(1.0);
    let tessellation = super::mesh::tessellate(
        body,
        TessellationTolerance::new(0.04, scale * 1e-8)
            .with_chordal_deflection(scale * 2e-5),
    );
    if !tessellation.missing_faces.is_empty() || tessellation.mesh.triangles.is_empty() {
        return Err(ThickenError::UnsupportedSurface);
    }

    let mesh = tessellation.mesh;
    let mut welded = HashMap::<[u64; 3], usize>::new();
    let mut vertices = Vec::<[f64; 3]>::new();
    let mut normals = Vec::<Vec3>::new();
    let mut counts = Vec::<usize>::new();
    let mut remap = Vec::with_capacity(mesh.positions.len());
    for (position, normal) in mesh.positions.iter().zip(&mesh.normals) {
        let key = position.map(|value| if value == 0.0 { 0 } else { value.to_bits() });
        let index = *welded.entry(key).or_insert_with(|| {
            let index = vertices.len();
            vertices.push(*position);
            normals.push(Vec3::new(0.0, 0.0, 0.0));
            counts.push(0);
            index
        });
        normals[index] = normals[index] + Vec3::from(*normal);
        counts[index] += 1;
        remap.push(index);
    }
    let normals = normals
        .into_iter()
        .zip(counts)
        .map(|(normal, count)| (normal / count as f64).normalize())
        .collect::<Option<Vec<_>>>()
        .ok_or(ThickenError::UnsupportedSurface)?;

    let triangles = mesh
        .triangles
        .iter()
        .map(|triangle| triangle.map(|index| remap[index]))
        .filter(|triangle| {
            triangle[0] != triangle[1]
                && triangle[1] != triangle[2]
                && triangle[2] != triangle[0]
        })
        .collect::<Vec<_>>();
    if triangles.is_empty() {
        return Err(ThickenError::UnsupportedSurface);
    }

    let source_count = vertices.len();
    vertices.extend(
        (0..source_count)
            .map(|index| (Vec3::from(vertices[index]) + normals[index] * distance).to_array())
            .collect::<Vec<_>>(),
    );
    let minimum_area = scale * scale * 1e-16;
    for triangle in &triangles {
        let source = triangle.map(|index| Vec3::from(vertices[index]));
        let offset = triangle.map(|index| Vec3::from(vertices[index + source_count]));
        let first = (source[1] - source[0]).cross(source[2] - source[0]);
        let second = (offset[1] - offset[0]).cross(offset[2] - offset[0]);
        if first.length() <= minimum_area
            || second.length() <= minimum_area
            || first.dot(second) <= 0.0
        {
            return Err(ThickenError::SelfIntersection);
        }
    }

    let mut uses = HashMap::<(usize, usize), (usize, usize, usize)>::new();
    for triangle in &triangles {
        for corner in 0..3 {
            let from = triangle[corner];
            let to = triangle[(corner + 1) % 3];
            let key = if from < to { (from, to) } else { (to, from) };
            uses.entry(key)
                .and_modify(|entry| entry.2 += 1)
                .or_insert((from, to, 1));
        }
    }
    if uses.values().any(|entry| entry.2 > 2) {
        return Err(ThickenError::UnsupportedSurface);
    }
    let boundary = uses
        .values()
        .filter(|entry| entry.2 == 1)
        .map(|entry| (entry.0, entry.1))
        .collect::<Vec<_>>();
    if boundary.is_empty() {
        return Err(ThickenError::UnsupportedSurface);
    }

    let mut faces = Vec::<Vec<usize>>::with_capacity(triangles.len() * 2 + boundary.len());
    faces.extend(triangles.iter().map(|triangle| triangle.to_vec()));
    faces.extend(triangles.iter().map(|triangle| {
        vec![
            triangle[2] + source_count,
            triangle[1] + source_count,
            triangle[0] + source_count,
        ]
    }));
    faces.extend(boundary.into_iter().map(|(from, to)| {
        vec![from, to, to + source_count, from + source_count]
    }));
    super::make::faceted_solid(&vertices, &faces).ok_or(ThickenError::SelfIntersection)
}

fn range(values: &[f64]) -> Option<(f64, f64)> {
    let low = values.iter().copied().min_by(f64::total_cmp)?;
    let high = values.iter().copied().max_by(f64::total_cmp)?;
    Some((low, high))
}

fn span(values: &[f64]) -> Option<f64> {
    let (low, high) = range(values)?;
    Some(high - low)
}

fn angular_cover(values: &[f64]) -> Option<(f64, f64)> {
    let mut angles = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(|value| value.rem_euclid(TAU))
        .collect::<Vec<_>>();
    if angles.is_empty() {
        return None;
    }
    angles.sort_by(f64::total_cmp);
    angles.dedup_by(|left, right| (*left - *right).abs() <= 1e-10);
    if angles.len() == 1 {
        return Some((angles[0], 0.0));
    }
    let mut largest = (-1.0, 0usize);
    for index in 0..angles.len() {
        let next = if index + 1 < angles.len() {
            angles[index + 1]
        } else {
            angles[0] + TAU
        };
        let gap = next - angles[index];
        if gap > largest.0 {
            largest = (gap, index);
        }
    }
    let start = angles[(largest.1 + 1) % angles.len()];
    Some((start, TAU - largest.0))
}
