//! Exact mass properties for analytic closed solids.

use super::{Body, Surface};
use crate::space::Vec3;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassProperties {
    pub volume: f64,
    pub centroid: [f64; 3],
    pub moment_of_inertia: [f64; 3],
    pub principal_directions: [f64; 9],
    pub principal_moments: [f64; 3],
    pub product_of_inertia: [f64; 3],
    pub radii_of_gyration: [f64; 3],
}

impl MassProperties {
    /// Return the same body translated in world space.
    pub fn translated(mut self, delta: [f64; 3]) -> Self {
        let old = self.centroid;
        let new = [old[0] + delta[0], old[1] + delta[1], old[2] + delta[2]];
        let square_change = |axis| delta[axis] * (2.0 * old[axis] + delta[axis]);
        self.moment_of_inertia[0] += self.volume * (square_change(1) + square_change(2));
        self.moment_of_inertia[1] += self.volume * (square_change(0) + square_change(2));
        self.moment_of_inertia[2] += self.volume * (square_change(0) + square_change(1));
        let product_change = |first, second| {
            old[first] * delta[second]
                + old[second] * delta[first]
                + delta[first] * delta[second]
        };
        self.product_of_inertia[0] -= self.volume * product_change(0, 1);
        self.product_of_inertia[1] -= self.volume * product_change(1, 2);
        self.product_of_inertia[2] -= self.volume * product_change(2, 0);
        self.radii_of_gyration = self
            .moment_of_inertia
            .map(|moment| (moment.max(0.0) / self.volume).sqrt());
        self.centroid = new;
        self
    }
}

/// Exact properties for complete analytic spheres and circular cylinders,
/// including the concentric closed shells produced by [`super::shell`].
pub fn analytic_mass_properties(body: &Body) -> Option<MassProperties> {
    sphere_properties(body).or_else(|| cylinder_properties(body))
}

fn sphere_properties(body: &Body) -> Option<MassProperties> {
    let mut spheres = Vec::new();
    for face_key in body.face_keys() {
        let face = body.faces.get(face_key)?;
        let Surface::Sphere(sphere) = body.surfaces.get(face.surface)? else {
            return None;
        };
        spheres.push(*sphere);
    }
    if !(1..=2).contains(&spheres.len()) {
        return None;
    }
    spheres.sort_by(|left, right| right.radius.total_cmp(&left.radius));
    let center = Vec3::from(spheres[0].frame.origin);
    let scale = spheres[0].radius.abs().max(1.0);
    if spheres.iter().any(|sphere| {
        !sphere.radius.is_finite()
            || sphere.radius <= 0.0
            || center.distance(Vec3::from(sphere.frame.origin)) > scale * 1e-8
    }) {
        return None;
    }
    let outer = spheres[0].radius;
    let inner = spheres.get(1).map_or(0.0, |sphere| sphere.radius);
    if inner >= outer {
        return None;
    }
    let volume = 4.0 * PI * (outer.powi(3) - inner.powi(3)) / 3.0;
    let principal = 8.0 * PI * (outer.powi(5) - inner.powi(5)) / 15.0;
    assemble(
        volume,
        center.to_array(),
        [principal; 3],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
}

fn cylinder_properties(body: &Body) -> Option<MassProperties> {
    let mut cylinders = Vec::new();
    let mut planes = Vec::new();
    for face_key in body.face_keys() {
        let face = body.faces.get(face_key)?;
        match body.surfaces.get(face.surface)? {
            Surface::Cylinder(cylinder) => cylinders.push(*cylinder),
            Surface::Plane(plane) => planes.push(*plane),
            _ => return None,
        }
    }
    if cylinders.is_empty() || planes.is_empty() {
        return None;
    }
    cylinders.sort_by(|left, right| right.radius.total_cmp(&left.radius));
    let outer = cylinders[0];
    let axis = Vec3::from(outer.base.normal()?).normalize()?;
    let x = Vec3::from(outer.base.x_axis).normalize()?;
    let x = (x - axis * x.dot(axis)).normalize()?;
    let y = axis.cross(x).normalize()?;
    let reference = Vec3::from(outer.base.origin);
    let reference_lateral = reference - axis * reference.dot(axis);
    let scale = outer.radius.abs().max(1.0);
    if cylinders.iter().any(|cylinder| {
        if !cylinder.radius.is_finite() || cylinder.radius <= 0.0 {
            return true;
        }
        let Some(candidate_normal) = cylinder.base.normal() else {
            return true;
        };
        let Some(candidate_axis) = Vec3::from(candidate_normal).normalize() else {
            return true;
        };
        let origin = Vec3::from(cylinder.base.origin);
        let lateral = origin - axis * origin.dot(axis);
        candidate_axis.dot(axis).abs() < 1.0 - 1e-8
            || lateral.distance(reference_lateral) > scale * 1e-8
    }) {
        return None;
    }
    cylinders.dedup_by(|left, right| (left.radius - right.radius).abs() <= scale * 1e-8);
    if !(1..=2).contains(&cylinders.len()) {
        return None;
    }
    let mut positions = planes
        .iter()
        .map(|plane| {
            let normal = Vec3::from(plane.normal()?);
            (normal.dot(axis).abs() >= 1.0 - 1e-8)
                .then(|| (Vec3::from(plane.origin) - reference).dot(axis))
        })
        .collect::<Option<Vec<_>>>()?;
    positions.sort_by(f64::total_cmp);
    let tolerance = positions
        .iter()
        .map(|value| value.abs())
        .fold(scale, f64::max)
        * 1e-8;
    positions.dedup_by(|left, right| (*left - *right).abs() <= tolerance);
    if positions.len() != cylinders.len() * 2 {
        return None;
    }

    let outer_min = positions[0];
    let outer_max = *positions.last()?;
    let outer_height = outer_max - outer_min;
    if outer_height <= 0.0 {
        return None;
    }
    let outer_volume = PI * outer.radius.powi(2) * outer_height;
    let outer_center = (outer_min + outer_max) * 0.5;

    let (inner_radius, inner_height, inner_volume, inner_center) = if cylinders.len() == 2 {
        let inner = cylinders[1];
        if inner.radius >= outer.radius {
            return None;
        }
        let inner_min = positions[1];
        let inner_max = positions[2];
        let height = inner_max - inner_min;
        if height <= 0.0 {
            return None;
        }
        (
            inner.radius,
            height,
            PI * inner.radius.powi(2) * height,
            (inner_min + inner_max) * 0.5,
        )
    } else {
        (0.0, 0.0, 0.0, outer_center)
    };
    let volume = outer_volume - inner_volume;
    if !volume.is_finite() || volume <= 0.0 {
        return None;
    }
    let center_at = (outer_volume * outer_center - inner_volume * inner_center) / volume;
    let centroid = (reference + axis * center_at).to_array();
    let axial = 0.5
        * (outer_volume * outer.radius.powi(2)
            - inner_volume * inner_radius.powi(2));
    let transverse = outer_volume
        * (3.0 * outer.radius.powi(2) + outer_height.powi(2))
        / 12.0
        + outer_volume * (outer_center - center_at).powi(2)
        - inner_volume
            * (3.0 * inner_radius.powi(2) + inner_height.powi(2))
            / 12.0
        - inner_volume * (inner_center - center_at).powi(2);
    assemble(
        volume,
        centroid,
        [transverse, transverse, axial],
        [x.to_array(), y.to_array(), axis.to_array()],
    )
}

fn assemble(
    volume: f64,
    centroid: [f64; 3],
    principal_moments: [f64; 3],
    axes: [[f64; 3]; 3],
) -> Option<MassProperties> {
    if !volume.is_finite()
        || volume <= 0.0
        || centroid.iter().chain(principal_moments.iter()).any(|value| !value.is_finite())
    {
        return None;
    }
    let mut central = [[0.0; 3]; 3];
    for principal in 0..3 {
        for row in 0..3 {
            for column in 0..3 {
                central[row][column] +=
                    principal_moments[principal] * axes[principal][row] * axes[principal][column];
            }
        }
    }
    let c = Vec3::from(centroid);
    let c2 = c.dot(c);
    let mut origin = central;
    for row in 0..3 {
        for column in 0..3 {
            origin[row][column] += volume
                * (if row == column { c2 } else { 0.0 } - centroid[row] * centroid[column]);
        }
    }
    let moment_of_inertia = [origin[0][0], origin[1][1], origin[2][2]];
    let product_of_inertia = [origin[0][1], origin[1][2], origin[2][0]];
    let radii_of_gyration = moment_of_inertia.map(|moment| (moment.max(0.0) / volume).sqrt());
    Some(MassProperties {
        volume,
        centroid,
        moment_of_inertia,
        principal_directions: [
            axes[0][0], axes[0][1], axes[0][2], axes[1][0], axes[1][1], axes[1][2],
            axes[2][0], axes[2][1], axes[2][2],
        ],
        principal_moments,
        product_of_inertia,
        radii_of_gyration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brep::make;

    #[test]
    fn a_translated_sphere_has_exact_mass_properties() {
        let body = make::sphere([2.0, -3.0, 5.0], 2.0).unwrap();
        let properties = analytic_mass_properties(&body).unwrap();

        assert!((properties.volume - 32.0 * PI / 3.0).abs() < 1e-10);
        assert_eq!(properties.centroid, [2.0, -3.0, 5.0]);
        let central = 2.0 * properties.volume * 4.0 / 5.0;
        assert!(properties
            .principal_moments
            .iter()
            .all(|moment| (*moment - central).abs() < 1e-10));
    }

    #[test]
    fn a_cylinder_has_exact_volume_centroid_and_principal_moments() {
        let body = make::cylinder([1.0, 2.0, 3.0], 2.0, 6.0).unwrap();
        let properties = analytic_mass_properties(&body).unwrap();

        let volume = 24.0 * PI;
        assert!((properties.volume - volume).abs() < 1e-10);
        assert_eq!(properties.centroid, [1.0, 2.0, 6.0]);
        assert!((properties.principal_moments[0] - 4.0 * volume).abs() < 1e-10);
        assert!((properties.principal_moments[1] - 4.0 * volume).abs() < 1e-10);
        assert!((properties.principal_moments[2] - 2.0 * volume).abs() < 1e-10);
    }

    #[test]
    fn translating_properties_matches_translating_the_body() {
        let properties = analytic_mass_properties(&make::sphere([2.0, -3.0, 5.0], 2.0).unwrap())
            .unwrap()
            .translated([7.0, 11.0, -13.0]);
        let expected = analytic_mass_properties(&make::sphere([9.0, 8.0, -8.0], 2.0).unwrap())
            .unwrap();

        assert_eq!(properties.centroid, expected.centroid);
        for (actual, expected) in properties
            .moment_of_inertia
            .into_iter()
            .chain(properties.product_of_inertia)
            .zip(
                expected
                    .moment_of_inertia
                    .into_iter()
                    .chain(expected.product_of_inertia),
            )
        {
            assert!((actual - expected).abs() < 1e-9);
        }
    }
}
