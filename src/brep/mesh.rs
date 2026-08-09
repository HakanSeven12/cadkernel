//! Turning a solid into triangles.
//!
//! Nothing draws a B-rep directly. A renderer wants positions, normals and
//! indices, and so does anything measuring volume or exporting to a mesh
//! format — so this is the last step out of the kernel for most callers.
//!
//! # In parameter space, then lifted
//!
//! Each face is triangulated where it is flat: its own `(u, v)`. The boundary
//! becomes a ring of parameter points, ear clipping fills it, and every
//! vertex is then mapped through the surface. On a plane that is exact. On a
//! cylinder it is not — the surface bulges between two parameter points — so
//! curved faces have their triangles subdivided until the middle of each sits
//! within tolerance of the surface.
//!
//! Doing it the other way round, triangulating in space, would mean deciding
//! what "inside the boundary" means on a curved patch, which is the question
//! parameter space already answers.
//!
//! # Orientation
//!
//! Every triangle comes out wound so its normal points out of the solid. A
//! face whose sense disagrees with its surface has its triangles reversed;
//! getting that wrong lights a solid inside out, and no amount of shading
//! afterwards recovers it.

use super::pcurve;
use super::topology::{Body, FaceKey};
use crate::geom2d::triangulate;
use crate::space::Vec3;
use std::f64::consts::{FRAC_PI_2, TAU};

/// A triangulated solid.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mesh {
    /// Positions, in world coordinates.
    pub positions: Vec<[f64; 3]>,
    /// One outward unit normal per position.
    pub normals: Vec<[f64; 3]>,
    /// Three indices per triangle.
    pub triangles: Vec<[usize; 3]>,
}

impl Mesh {
    /// How many triangles it holds.
    pub fn len(&self) -> usize {
        self.triangles.len()
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Adds another mesh's triangles, keeping both.
    pub fn absorb(&mut self, other: Mesh) {
        let offset = self.positions.len();
        self.positions.extend(other.positions);
        self.normals.extend(other.normals);
        self.triangles.extend(
            other
                .triangles
                .into_iter()
                .map(|t| [t[0] + offset, t[1] + offset, t[2] + offset]),
        );
    }
}

/// How far a triangle's middle may sit from the surface before it is split.
///
/// Only curved faces are ever split; a plane's triangles are exact whatever
/// this is.
const DEFAULT_SAG: f64 = 0.01;

/// How deep the subdivision will go.
///
/// Each level quadruples the triangle count, so this is a ceiling of 4^5 per
/// original triangle — far past what any tolerance a drawing uses reaches,
/// and a backstop against a surface that never converges.
const MAX_DEPTH: u32 = 5;

/// Triangulates a whole body.
///
/// `sag` is how far a triangle may depart from the surface it lies on. A
/// caller rendering at a known zoom passes its own; the default is fine for
/// a drawing measured in millimetres.
pub fn body(body: &Body, sag: f64, tolerance: f64) -> Mesh {
    let mut out = Mesh::default();
    for face in body.face_keys() {
        if let Some(mesh) = self::face(body, face, sag, tolerance) {
            out.absorb(mesh);
        }
    }
    out
}

/// Triangulates one face.
///
/// `None` when the face's boundary cannot be expressed in its surface's
/// parameter space — the same limit [`pcurve::project`] has. A face left out
/// is a hole in the mesh, which is why it is reported rather than skipped
/// silently by the caller's own loop.
pub fn face(body: &Body, face: FaceKey, sag: f64, tolerance: f64) -> Option<Mesh> {
    let node = body.faces.get(face)?;
    let surface = body.surfaces.get(node.surface)?;
    // A face bounded by nothing but its own seams is the whole of its
    // surface, and its region in (u, v) is the whole parameter rectangle.
    // Worth asking first, because a sphere's seam ends at the poles where
    // longitude has no value at all — the boundary cannot be projected, and
    // taking that for "cannot be drawn" left every sphere missing.
    if let Some(domain) = whole_surface(body, face, surface) {
        return fill(body, face, surface, &[domain], sag);
    }
    let boundary = pcurve::face_boundary(body, face, tolerance)?;
    if boundary.is_empty() {
        return None;
    }

    // The outer loop bounds the face; the rest cut holes in it. They arrive
    // in that order, one curve per coedge, so the rings are rebuilt by
    // walking each loop's own coedges.
    let mut rings: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut taken = 0;
    for ring in &node.loops {
        let count = body.loops.get(*ring)?.coedges.len();
        let mut points: Vec<[f64; 2]> = Vec::new();
        for (order, curve) in boundary.get(taken..taken + count)?.iter().enumerate() {
            let mut sampled = curve.tessellate_within(sag);
            if sampled.len() < 2 {
                continue;
            }
            // The straight kinds come out of the projection already running
            // the loop's way; a circle or a spline carries its direction in
            // its own parameter, so the run is turned round when it starts at
            // the wrong end.
            if let Some(last) = points.last() {
                let head = distance(*last, sampled[0]);
                let tail = distance(*last, sampled[sampled.len() - 1]);
                if tail < head {
                    sampled.reverse();
                }
            } else if count > 1 {
                // Nothing to chain onto yet, so the first piece is oriented
                // against the next one instead.
                let next = &boundary[taken + 1];
                let following = next.point_at(0.0);
                if distance(sampled[0], following) < distance(sampled[sampled.len() - 1], following)
                {
                    sampled.reverse();
                }
            }
            let skip = usize::from(
                points
                    .last()
                    .is_some_and(|last| distance(*last, sampled[0]) <= tolerance),
            );
            points.extend_from_slice(&sampled[skip..]);
            let _ = order;
        }
        taken += count;
        if points.len() >= 3 {
            rings.push(points);
        }
    }
    fill(body, face, surface, &rings, sag)
}

/// Triangulates a face over the rings its boundary makes in `(u, v)`.
///
/// The first bounds it and the rest cut holes out of it.
fn fill(
    body: &Body,
    face: FaceKey,
    surface: &super::geometry::Surface,
    rings: &[Vec<[f64; 2]>],
    sag: f64,
) -> Option<Mesh> {
    let (outer, holes) = rings.split_first()?;
    let (parameters, triangles) = triangulate(outer, holes);
    if triangles.is_empty() {
        return None;
    }

    let mut mesh = Mesh::default();
    let flat = matches!(surface, super::geometry::Surface::Plane(_));
    for triangle in triangles {
        let corners = [
            parameters[triangle[0]],
            parameters[triangle[1]],
            parameters[triangle[2]],
        ];
        if flat {
            emit(&mut mesh, body, face, corners);
        } else {
            refine(&mut mesh, body, face, corners, sag, 0);
        }
    }
    Some(mesh)
}

/// The whole of a closed surface, as a ring in `(u, v)`, when that is what a
/// face covers.
///
/// A face covers all of its surface when every edge bounding it is used
/// twice by that same face and by nothing else — which is to say the only
/// things bounding it are its own seams, and a seam is where a surface was
/// cut open to have a boundary rather than a border with anything.
///
/// `None` for a surface with no closed extent to fill: a plane and the
/// unbounded run of a cylinder or a cone go on forever, so a face on one is
/// always bounded by real edges.
fn whole_surface(
    body: &Body,
    face: FaceKey,
    surface: &super::geometry::Surface,
) -> Option<Vec<[f64; 2]>> {
    use super::geometry::Surface;
    let span = match surface {
        Surface::Sphere(_) => [[0.0, TAU], [-FRAC_PI_2, FRAC_PI_2]],
        Surface::Torus(_) => [[0.0, TAU], [0.0, TAU]],
        _ => return None,
    };
    let coedges = body.face_coedges(face);
    if coedges.is_empty() {
        return None;
    }
    let edges: Vec<_> = coedges
        .iter()
        .filter_map(|coedge| Some(body.coedges.get(*coedge)?.edge))
        .collect();
    if edges.len() != coedges.len() {
        return None;
    }
    // Each edge used twice, and by nothing but this face. Counted by walking
    // rather than sorting, since a generational key has no order of its own —
    // and giving it one would invite comparing keys from two arenas.
    let seams = edges.iter().filter(|edge| {
        let here = edges.iter().filter(|other| *other == *edge).count();
        let anywhere = body
            .edges
            .get(**edge)
            .map_or(0, |node| node.coedges.len());
        here == 2 && anywhere == 2
    });
    if seams.count() != coedges.len() {
        return None;
    }
    Some(vec![
        [span[0][0], span[1][0]],
        [span[0][1], span[1][0]],
        [span[0][1], span[1][1]],
        [span[0][0], span[1][1]],
    ])
}

/// Splits a triangle until its middle lies close enough to the surface.
fn refine(
    mesh: &mut Mesh,
    body: &Body,
    face: FaceKey,
    corners: [[f64; 2]; 3],
    sag: f64,
    depth: u32,
) {
    let Some(node) = body.faces.get(face) else {
        return;
    };
    let Some(surface) = body.surfaces.get(node.surface) else {
        return;
    };
    if depth < MAX_DEPTH {
        let middle = [
            (corners[0][0] + corners[1][0] + corners[2][0]) / 3.0,
            (corners[0][1] + corners[1][1] + corners[2][1]) / 3.0,
        ];
        // Where the flat triangle puts that point against where the surface
        // does. On a plane the two agree exactly and nothing ever splits.
        let flat = Vec3::from(surface.point_at(corners[0][0], corners[0][1]))
            .lerp(
                Vec3::from(surface.point_at(corners[1][0], corners[1][1])),
                1.0 / 3.0,
            );
        let flat = flat.lerp(
            Vec3::from(surface.point_at(corners[2][0], corners[2][1])),
            1.0 / 3.0,
        );
        let curved = Vec3::from(surface.point_at(middle[0], middle[1]));
        if flat.distance(curved) > sag {
            let midpoints = [
                midpoint(corners[0], corners[1]),
                midpoint(corners[1], corners[2]),
                midpoint(corners[2], corners[0]),
            ];
            for part in [
                [corners[0], midpoints[0], midpoints[2]],
                [midpoints[0], corners[1], midpoints[1]],
                [midpoints[2], midpoints[1], corners[2]],
                [midpoints[0], midpoints[1], midpoints[2]],
            ] {
                refine(mesh, body, face, part, sag, depth + 1);
            }
            return;
        }
    }
    emit(mesh, body, face, corners);
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (b[0] - a[0]).hypot(b[1] - a[1])
}

fn midpoint(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

/// Adds one triangle, wound so its normal points out of the solid.
fn emit(mesh: &mut Mesh, body: &Body, face: FaceKey, corners: [[f64; 2]; 3]) {
    let Some(node) = body.faces.get(face) else {
        return;
    };
    let Some(surface) = body.surfaces.get(node.surface) else {
        return;
    };
    let points: Vec<Vec3> = corners
        .iter()
        .map(|uv| Vec3::from(surface.point_at(uv[0], uv[1])))
        .collect();
    let Some(normal) = (points[1] - points[0])
        .cross(points[2] - points[0])
        .normalize()
    else {
        // A collapsed triangle has no normal and nothing to draw.
        return;
    };
    // A face whose sense disagrees with its surface faces the other way, and
    // so does everything on it.
    let normal = if node.forward { normal } else { -normal };
    let base = mesh.positions.len();
    let order = if node.forward { [0, 1, 2] } else { [0, 2, 1] };
    for step in order {
        mesh.positions.push(points[step].to_array());
        mesh.normals.push(normal.to_array());
    }
    mesh.triangles.push([base, base + 1, base + 2]);
}

/// The default sag, for a caller with no opinion.
pub fn default_sag() -> f64 {
    DEFAULT_SAG
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brep::make::cuboid;

    const TOL: f64 = 1e-9;

    #[test]
    fn a_box_meshes_into_two_triangles_a_side() {
        let solid = cuboid([0.0; 3], [2.0, 3.0, 4.0]).unwrap();
        let mesh = self::body(&solid, 0.01, TOL);
        assert_eq!(mesh.len(), 12, "six faces, two triangles each");
        assert_eq!(mesh.positions.len(), 36);
    }

    #[test]
    fn the_triangles_cover_the_boxs_own_area() {
        let solid = cuboid([0.0; 3], [2.0, 3.0, 4.0]).unwrap();
        let mesh = self::body(&solid, 0.01, TOL);
        let area: f64 = mesh
            .triangles
            .iter()
            .map(|t| {
                let a = Vec3::from(mesh.positions[t[0]]);
                let b = Vec3::from(mesh.positions[t[1]]);
                let c = Vec3::from(mesh.positions[t[2]]);
                (b - a).cross(c - a).length() * 0.5
            })
            .sum();
        // 2·(2·3 + 3·4 + 2·4)
        assert!((area - 52.0).abs() < 1e-9, "{area}");
    }

    #[test]
    fn every_normal_points_out_of_the_solid() {
        // The one that matters for anything drawn: a face wound the wrong way
        // lights the solid inside out and no shading afterwards recovers it.
        let solid = cuboid([0.0; 3], [4.0, 6.0, 8.0]).unwrap();
        let mesh = self::body(&solid, 0.01, TOL);
        let centre = Vec3::new(2.0, 3.0, 4.0);
        for triangle in &mesh.triangles {
            let corner = Vec3::from(mesh.positions[triangle[0]]);
            let normal = Vec3::from(mesh.normals[triangle[0]]);
            assert!(
                normal.dot(corner - centre) > 0.0,
                "a triangle faced inwards at {corner:?}"
            );
        }
    }

    #[test]
    fn the_winding_agrees_with_the_normal() {
        let solid = cuboid([0.0; 3], [4.0, 6.0, 8.0]).unwrap();
        let mesh = self::body(&solid, 0.01, TOL);
        for triangle in &mesh.triangles {
            let a = Vec3::from(mesh.positions[triangle[0]]);
            let b = Vec3::from(mesh.positions[triangle[1]]);
            let c = Vec3::from(mesh.positions[triangle[2]]);
            let wound = (b - a).cross(c - a).normalize().unwrap();
            let stored = Vec3::from(mesh.normals[triangle[0]]);
            assert!(
                wound.dot(stored) > 0.9,
                "the winding and the normal disagree"
            );
        }
    }

    #[test]
    fn every_vertex_is_on_the_solid_it_came_from() {
        let solid = cuboid([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).unwrap();
        let mesh = self::body(&solid, 0.01, TOL);
        for position in &mesh.positions {
            let on_a_face = solid.face_keys().any(|face| {
                solid
                    .faces
                    .get(face)
                    .and_then(|node| solid.surfaces.get(node.surface))
                    .is_some_and(|surface| surface.contains(*position, 1e-9))
            });
            assert!(on_a_face, "{position:?} is not on the solid");
        }
    }

    #[test]
    fn a_flat_face_is_never_split_however_fine_the_sag() {
        // A plane's triangles are exact, so refining them would only cost
        // vertices.
        let solid = cuboid([0.0; 3], [10.0; 3]).unwrap();
        let coarse = self::body(&solid, 1.0, TOL).len();
        let fine = self::body(&solid, 1e-9, TOL).len();
        assert_eq!(coarse, fine, "a plane does not curve");
        assert_eq!(fine, 12);
    }

    #[test]
    fn a_cylinder_wall_is_split_until_it_follows_its_surface() {
        // Triangulating the boundary alone would give the wall two triangles
        // whatever the tolerance, and a cylinder would render as a flat
        // ribbon.
        let solid = crate::brep::make::cylinder([0.0; 3], 5.0, 10.0).unwrap();
        let coarse = self::body(&solid, 2.0, TOL).len();
        let fine = self::body(&solid, 0.01, TOL).len();
        assert!(fine > coarse * 4, "{coarse} then {fine}");
    }

    #[test]
    fn a_cylinders_mesh_stays_on_the_cylinder() {
        let solid = crate::brep::make::cylinder([0.0; 3], 5.0, 10.0).unwrap();
        let mesh = self::body(&solid, 0.02, TOL);
        assert!(!mesh.is_empty());
        for position in &mesh.positions {
            let radius = (position[0] * position[0] + position[1] * position[1]).sqrt();
            let on_wall = (radius - 5.0).abs() < 0.05;
            let on_cap = radius <= 5.0 + 1e-6
                && (position[2].abs() < 1e-9 || (position[2] - 10.0).abs() < 1e-9);
            assert!(on_wall || on_cap, "{position:?} is off the cylinder");
        }
    }

    #[test]
    fn a_body_with_nothing_in_it_meshes_to_nothing() {
        let mesh = self::body(&Body::new(), 0.01, TOL);
        assert!(mesh.is_empty());
    }

    #[test]
    fn two_meshes_join_without_their_indices_colliding() {
        let solid = cuboid([0.0; 3], [1.0; 3]).unwrap();
        let mut one = self::body(&solid, 0.01, TOL);
        let other = self::body(&solid, 0.01, TOL);
        let counts = (one.len(), other.len());
        one.absorb(other);
        assert_eq!(one.len(), counts.0 + counts.1);
        for triangle in &one.triangles {
            assert!(triangle.iter().all(|index| *index < one.positions.len()));
        }
    }

    #[test]
    fn a_solid_at_survey_coordinates_meshes_where_it_is() {
        let origin = [512_345.678, 4_512_345.678, 91.5];
        let solid = cuboid(origin, [0.5, 0.5, 0.5]).unwrap();
        let mesh = self::body(&solid, 0.01, 1e-6);
        assert_eq!(mesh.len(), 12);
        for position in &mesh.positions {
            assert!((position[0] - origin[0]).abs() <= 0.5 + 1e-6, "{position:?}");
        }
    }

    #[test]
    fn a_boolean_result_meshes_too() {
        let a = cuboid([0.0; 3], [10.0; 3]).unwrap();
        let b = cuboid([5.0; 3], [10.0; 3]).unwrap();
        let joined =
            crate::brep::boolean::combine(a, b, crate::brep::boolean::Operation::Union, TOL)
                .unwrap();
        let mesh = self::body(&joined, 0.01, TOL);
        assert!(!mesh.is_empty());
        // An imprinted face is no longer a rectangle, so it takes more than
        // two triangles — but never fewer.
        assert!(
            mesh.len() >= joined.faces.len() * 2,
            "{} faces gave only {} triangles",
            joined.faces.len(),
            mesh.len()
        );
        for triangle in &mesh.triangles {
            assert!(triangle.iter().all(|i| *i < mesh.positions.len()));
        }
    }
}
