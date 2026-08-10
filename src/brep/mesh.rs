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
/// A backstop against a surface that never converges, not a quality setting:
/// refinement stops as soon as the sag is met, so the depth only costs where
/// it is actually reached.
///
/// It has to be deep enough for the widest triangle any face starts from, and
/// that is a whole turn — a tube bounded by two rims begins as one rectangle
/// spanning all of `u`. Each level halves the step, so five got 32 segments to
/// a full circle whatever tolerance was asked for, and a cylinder wall stayed
/// visibly coarser than the rim drawn around it. Seven reaches 128, past what
/// the edge sampling asks for.
const MAX_DEPTH: u32 = 7;

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
    // A tube: closed the whole way round, bounded by a rim at each end and no
    // seam between them. Its boundary is two closed circles, which trace no
    // ring in (u, v) at all — but the region is not in doubt, being the whole
    // turn between the two heights the rims sit at.
    if let Some(domain) = banded(body, face, surface) {
        return fill(body, face, surface, &[domain], sag);
    }
    let boundary = pcurve::face_boundary(body, face, tolerance)?;
    if boundary.is_empty() {
        return None;
    }

    // One curve per coedge, so the rings are rebuilt by walking each loop's
    // own coedges. Which of them bounds the face is settled in `fill`, by
    // area rather than by the order they were listed in.
    let mut rings: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut taken = 0;
    for ring in &node.loops {
        let count = body.loops.get(*ring)?.coedges.len();
        let mut pending: Vec<Vec<[f64; 2]>> = boundary
            .get(taken..taken + count)?
            .iter()
            .map(|curve| curve.tessellate_within(sag))
            .filter(|sampled| sampled.len() >= 2)
            .collect();
        taken += count;
        let Some(mut points) = pending.pop() else {
            continue;
        };
        // Walk the ring by whichever piece continues it, rather than by the
        // order the pieces arrived in. A loop's coedges are supposed to be
        // stored in order, and in a file they are not always: two sides of a
        // rectangle came back both starting from the same corner. Reversing
        // the next piece in the list cannot fix that — the ring joins itself
        // in the wrong place, crosses, and the face is dropped.
        while !pending.is_empty() {
            let head = *points.last()?;
            let mut best = (f64::INFINITY, 0usize, false);
            for (index, piece) in pending.iter().enumerate() {
                let front = distance(head, piece[0]);
                let back = distance(head, piece[piece.len() - 1]);
                if front < best.0 {
                    best = (front, index, false);
                }
                if back < best.0 {
                    best = (back, index, true);
                }
            }
            let mut next = pending.remove(best.1);
            if best.2 {
                next.reverse();
            }
            let skip = usize::from(distance(head, next[0]) <= tolerance);
            points.extend_from_slice(&next[skip..]);
        }
        if points.len() >= 3 {
            rings.push(points);
        }
    }
    fill(body, face, surface, &rings, sag)
}

/// Triangulates a face over the rings its boundary makes in `(u, v)`.
///
/// Which ring bounds the face and which cut holes in it is decided by area,
/// not by the order they arrive in. Nothing guarantees that order: a face
/// lifted from a file lists its loops however the file did, and taking the
/// first one on trust draws a plate with its bolt hole filled in and the
/// metal around it missing — a picture that looks deliberate.
fn fill(
    body: &Body,
    face: FaceKey,
    surface: &super::geometry::Surface,
    rings: &[Vec<[f64; 2]>],
    sag: f64,
) -> Option<Mesh> {
    let widest = rings
        .iter()
        .enumerate()
        .max_by(|a, b| {
            crate::geom2d::signed_area(a.1)
                .abs()
                .total_cmp(&crate::geom2d::signed_area(b.1).abs())
        })
        .map(|(index, _)| index)?;
    let outer = &rings[widest];
    let holes: Vec<Vec<[f64; 2]>> = rings
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != widest)
        .map(|(_, ring)| ring.clone())
        .collect();
    let (parameters, triangles) = triangulate(outer, &holes);
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

/// The band a tube covers, as a ring in `(u, v)`.
///
/// A cylinder or cone face can be bounded by two rims and nothing else: it
/// wraps the whole way round, so there is no seam cutting it open and no ring
/// for its boundary to trace. Each rim is a closed circle, shared with the
/// disc that caps it, and projects to a line spanning a full turn — two of
/// those do not join up.
///
/// What they do say is where the band starts and stops, which with a full
/// turn of `u` is the whole region. `None` for anything else: a face bounded
/// by arcs and generators traces a proper ring and goes the ordinary way.
fn banded(
    body: &Body,
    face: FaceKey,
    surface: &super::geometry::Surface,
) -> Option<Vec<[f64; 2]>> {
    use super::geometry::Surface;
    if !matches!(surface, Surface::Cylinder(_) | Surface::Cone(_)) {
        return None;
    }
    let node = body.faces.get(face)?;
    if node.loops.len() != 2 {
        return None;
    }
    let mut heights = Vec::with_capacity(2);
    for ring in &node.loops {
        let coedges = &body.loops.get(*ring)?.coedges;
        // One coedge, closing on itself: a rim rather than a chain of pieces.
        let [only] = coedges[..] else { return None };
        let edge = body.coedges.get(only)?.edge;
        let node = body.edges.get(edge)?;
        if node.start != node.end {
            return None;
        }
        let point = body.vertices.get(node.start)?.point;
        heights.push(surface.parameters_at(point)?.1);
    }
    let (low, high) = (
        heights[0].min(heights[1]),
        heights[0].max(heights[1]),
    );
    (high - low > 0.0).then(|| {
        vec![[0.0, low], [TAU, low], [TAU, high], [0.0, high]]
    })
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
    fn a_wall_is_sampled_as_finely_as_the_tolerance_asks() {
        // The subdivision cap used to decide this instead of the tolerance. A
        // tube starts as one rectangle spanning a whole turn, so five levels
        // of halving reached 32 segments and stopped — however fine a sag was
        // asked for. The wall then stayed visibly coarser than the rim drawn
        // around it, which is a mismatch no shading hides.
        let solid = crate::brep::make::cylinder([0.0; 3], 5.0, 10.0).unwrap();
        let wall = solid
            .faces
            .iter()
            .find(|(_, face)| {
                matches!(
                    solid.surfaces.get(face.surface),
                    Some(crate::brep::geometry::Surface::Cylinder(_))
                )
            })
            .map(|(key, _)| key)
            .unwrap();

        // A fiftieth of a per cent of the radius: about fifty sides, which is
        // what the edge sampling in a drawing asks for.
        let mesh = face(&solid, wall, 5.0 * 0.002, 1e-9).expect("a drawn wall");
        // Read the count back off the mesh: how many distinct angles the
        // triangle corners land on around the axis.
        let mut angles: Vec<i64> = mesh
            .positions
            .iter()
            .map(|p| (p[1].atan2(p[0]) * 1e6) as i64)
            .collect();
        angles.sort_unstable();
        angles.dedup();
        assert!(angles.len() >= 48, "only {} sides", angles.len());
    }

    #[test]
    fn a_tube_with_no_seam_still_knows_the_band_it_covers() {
        // A cylinder wall bounded by a rim at each end and nothing else: it
        // wraps the whole way round, so no seam cuts it open and its boundary
        // traces no ring in (u, v). Files carry solids shaped that way, and
        // the face was dropped for want of a ring to fill.
        let mut solid = crate::brep::make::cylinder([0.0; 3], 3.0, 6.0).unwrap();
        let wall = solid
            .faces
            .iter()
            .find(|(_, face)| {
                matches!(
                    solid.surfaces.get(face.surface),
                    Some(crate::brep::Surface::Cylinder(_))
                )
            })
            .map(|(key, _)| key)
            .unwrap();

        // Take the seam away, leaving the wall on its two rims alone.
        let ring = solid.faces.get(wall).unwrap().loops[0];
        let kept: Vec<_> = solid
            .loops
            .get(ring)
            .unwrap()
            .coedges
            .iter()
            .copied()
            .filter(|coedge| {
                let edge = solid.coedges.get(*coedge).unwrap().edge;
                let node = solid.edges.get(edge).unwrap();
                node.start == node.end
            })
            .collect();
        assert_eq!(kept.len(), 2, "two rims");
        let face = solid.faces.get_mut(wall).unwrap();
        face.loops = Vec::new();
        for coedge in kept {
            let owner = solid.loops.insert(crate::brep::topology::Loop {
                coedges: vec![coedge],
                owner: wall,
                provenance: crate::brep::Provenance::Synthesized,
            });
            solid.coedges.get_mut(coedge).unwrap().owner = owner;
            solid.faces.get_mut(wall).unwrap().loops.push(owner);
        }

        let mesh = crate::brep::mesh::face(&solid, wall, 0.01, 1e-9).expect("a drawn wall");
        let area: f64 = mesh
            .triangles
            .iter()
            .map(|t| {
                let at = |i: usize| Vec3::from(mesh.positions[t[i]]);
                (at(1) - at(0)).cross(at(2) - at(0)).length() * 0.5
            })
            .sum();
        let expected = TAU * 3.0 * 6.0;
        assert!((area - expected).abs() < 0.02 * expected, "{area} vs {expected}");
    }

    #[test]
    fn a_hole_stays_a_hole_however_its_loop_was_listed() {
        // A plate with a hole through it. Nothing says the outer loop comes
        // first — a face lifted from a file lists its loops however the file
        // did — so the ring that bounds the face is chosen by area. Taking
        // the first on trust fills the hole and empties the metal, which is a
        // picture that looks deliberate.
        // A ring of square section: its two flat faces are annuli, each
        // bounded by an outer rim with an inner one cut out of it.
        use crate::geom2d::{Curve as Curve2, Line};
        let corners = [[4.0, 0.0], [7.0, 0.0], [7.0, 2.0], [4.0, 2.0]];
        let profile: Vec<Curve2> = (0..4)
            .map(|index| {
                Curve2::Line(Line {
                    start: corners[index],
                    end: corners[(index + 1) % 4],
                })
            })
            .collect();
        let plane =
            crate::space::Plane::orthonormal([0.0; 3], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]).unwrap();
        let drilled = crate::brep::revolve(plane, &profile, [0.0; 3], [0.0, 0.0, 1.0], TAU)
            .expect("a ring");

        let holed = drilled
            .faces
            .iter()
            .find(|(_, face)| face.loops.len() == 2)
            .map(|(key, _)| key)
            .expect("a face with a hole in it");

        let area = |body: &Body, face| {
            crate::brep::mesh::face(body, face, 0.01, 1e-9)
                .map(|mesh| {
                    mesh.triangles
                        .iter()
                        .map(|t| {
                            let at = |i: usize| Vec3::from(mesh.positions[t[i]]);
                            (at(1) - at(0)).cross(at(2) - at(0)).length() * 0.5
                        })
                        .sum::<f64>()
                })
                .unwrap_or(0.0)
        };
        let expected = std::f64::consts::PI * (49.0 - 16.0);
        let drawn = area(&drilled, holed);
        assert!(
            (drawn - expected).abs() < 0.02 * expected,
            "{drawn} vs {expected}"
        );

        // And the same face with its loops listed the other way round has to
        // come out identical.
        let mut swapped = drilled.clone();
        swapped.faces.get_mut(holed).unwrap().loops.swap(0, 1);
        let other = area(&swapped, holed);
        assert!((drawn - other).abs() < 1e-9, "{drawn} vs {other}");
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
