//! Combining two solids.
//!
//! Once both have been imprinted with the curves they share, every face of
//! each is wholly inside the other or wholly outside it. What remains is
//! choosing which of those faces the result keeps, and turning the chosen set
//! back into a solid.
//!
//! | operation    | keeps of A | keeps of B          |
//! |--------------|------------|---------------------|
//! | union        | outside B  | outside A           |
//! | intersection | inside B   | inside A            |
//! | difference   | outside B  | inside A, flipped   |
//!
//! The flip in the last row is the whole of what makes a difference a
//! difference: B's faces become the walls of the cavity it leaves, so they
//! have to face the other way.
//!
//! # Why it refuses so readily
//!
//! A boolean that half-works produces a solid — one with a wall missing,
//! which looks finished, saves, and fails much later in something else's
//! hands. Every step that cannot be completed exactly is returned as a
//! [`Snag`] instead: a face pair with no closed form, a face that cannot be
//! classified, a coincident pair. Refusing is cheap; a leaking solid is not.

use super::classify::{contains_point, Containment};
use super::imprint::{imprint, Snag};
use super::pcurve;
use super::topology::{Body, Face, FaceKey, Lump, Shell};
use super::Provenance;
use crate::geom2d::Curve;
use super::geometry::Surface;
use crate::space::Vec3;

/// Which combination to take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    /// Everything in either.
    Union,
    /// Only what is in both.
    Intersection,
    /// What is in the first and not the second.
    Difference,
}

/// Combines two solids.
///
/// Neither input is consumed; the result is a new body. Both are imprinted
/// along the way, which is why they are taken by value — an imprinted copy is
/// not the body the caller handed over, and returning it silently would be
/// worse than asking for ownership.
pub fn combine(mut a: Body, mut b: Body, how: Operation, tolerance: f64) -> Result<Body, Snag> {
    imprint(&mut a, &mut b, tolerance)?;

    let (keep_a, keep_b, flip_b) = match how {
        Operation::Union => (Containment::Outside, Containment::Outside, false),
        Operation::Intersection => (Containment::Inside, Containment::Inside, false),
        Operation::Difference => (Containment::Outside, Containment::Inside, true),
    };

    let mut result = Body::new();
    let lump = result.lumps.insert(Lump {
        shells: Vec::new(),
        provenance: Provenance::Synthesized,
    });
    let shell = result.shells.insert(Shell {
        faces: Vec::new(),
        owner: lump,
        provenance: Provenance::Synthesized,
    });
    result.lumps.get_mut(lump).expect("just inserted").shells = vec![shell];
    result.roots = vec![lump];

    let mut kept = 0;
    for (body, other, wanted, flip, first) in [
        (&a, &b, keep_a, false, true),
        (&b, &a, keep_b, flip_b, false),
    ] {
        for face in body.face_keys() {
            // A shared wall is settled by the two normals rather than by
            // which side it is on: it is on both.
            if let Some(twin) = coincident_twin(body, other, face, tolerance) {
                if keeps_shared_wall(body, face, other, twin, flip_b, first) {
                    copy_face(&mut result, body, face, shell, flip)?;
                    kept += 1;
                }
                continue;
            }
            match face_side(body, other, face, tolerance) {
                Containment::OnBoundary => return Err(Snag::Coincident),
                Containment::Unknown => return Err(Snag::CutRefused),
                side if side == wanted => {
                    copy_face(&mut result, body, face, shell, flip)?;
                    kept += 1;
                }
                _ => {}
            }
        }
    }
    if kept == 0 {
        // Nothing survived: the operation's answer is genuinely empty — two
        // solids that do not touch, intersected.
        return Ok(Body::new());
    }
    Ok(result)
}

/// The other body's face covering the same ground as this one, if there is
/// one.
fn coincident_twin(
    body: &Body,
    other: &Body,
    face: FaceKey,
    tolerance: f64,
) -> Option<FaceKey> {
    let surface = body.surfaces.get(body.faces.get(face)?.surface)?;
    other.face_keys().find(|candidate| {
        other
            .faces
            .get(*candidate)
            .and_then(|node| other.surfaces.get(node.surface))
            .is_some_and(|theirs| same_surface(surface, theirs, tolerance))
            && super::imprint::same_ground(body, face, other, *candidate, tolerance)
    })
}

/// Whether two surfaces are the same one, ignoring how each is parameterised.
fn same_surface(one: &Surface, other: &Surface, tolerance: f64) -> bool {
    // Compared by what each says about the other's frame origin and by their
    // normals, which is enough for the planar case a shared wall is and does
    // not depend on either having picked the same u direction.
    let (Some(one_frame), Some(other_frame)) = (one.frame(), other.frame()) else {
        return false;
    };
    let (Some(first), Some(second)) = (one_frame.normal(), other_frame.normal()) else {
        return false;
    };
    let parallel = Vec3::from(first).is_parallel_to(Vec3::from(second), tolerance);
    parallel
        && one.distance_to(other_frame.origin).abs() <= tolerance
        && other.distance_to(one_frame.origin).abs() <= tolerance
}

/// Whether this side's copy of a shared wall survives.
///
/// Two faces covering the same ground are both on the boundary of the result
/// or neither is, and which it is comes from their normals once the
/// operation's own flip has been applied:
///
/// - facing the **same** way, the wall is real and exactly one copy is kept —
///   the first body's, arbitrarily but consistently
/// - facing **opposite** ways, the two solids are on either side of it, so it
///   is interior and both copies go
///
/// A difference flips the second solid, which turns two boxes stacked face to
/// face from opposite into same — and that is what leaves the first one whole
/// instead of open at the top.
fn keeps_shared_wall(
    body: &Body,
    face: FaceKey,
    other: &Body,
    twin: FaceKey,
    flip_second: bool,
    is_first: bool,
) -> bool {
    let outward = |body: &Body, face: FaceKey, flipped: bool| -> Option<Vec3> {
        let node = body.faces.get(face)?;
        let normal = Vec3::from(body.surfaces.get(node.surface)?.frame()?.normal()?);
        Some(if node.forward != flipped { normal } else { -normal })
    };
    // Each face is flipped only if it belongs to the second solid and the
    // operation turns that solid around.
    let (Some(mine), Some(theirs)) = (
        outward(body, face, flip_second && !is_first),
        outward(other, twin, flip_second && is_first),
    ) else {
        return false;
    };
    mine.dot(theirs) > 0.0 && is_first
}

/// Whether a face is inside the other solid, outside it, or on its surface.
///
/// Asked of a point in the face's interior rather than of a vertex: every
/// vertex of an imprinted face sits on the seam, where the answer is
/// "boundary" for reasons that have nothing to do with which side the face is
/// on.
fn face_side(body: &Body, other: &Body, face: FaceKey, tolerance: f64) -> Containment {
    match interior_point(body, face, tolerance) {
        Some(point) => contains_point(other, point, tolerance),
        None => Containment::Unknown,
    }
}

/// A point strictly inside a face, in space.
///
/// Found in the surface's parameter space, where "inside the boundary" is a
/// question [`geom2d`](crate::geom2d) already answers. The average of the
/// boundary's own points is inside a convex face and can fall outside a
/// concave one, so it is checked rather than assumed, and the midpoints
/// between it and each boundary point are tried after it.
fn interior_point(body: &Body, face: FaceKey, tolerance: f64) -> Option<[f64; 3]> {
    let node = body.faces.get(face)?;
    let surface = body.surfaces.get(node.surface)?;
    let boundary = pcurve::face_boundary(body, face, tolerance)?;
    let samples: Vec<[f64; 2]> = boundary
        .iter()
        .flat_map(|curve| {
            (0..4).map(move |step| curve.point_at(step as f64 / 4.0))
        })
        .collect();
    if samples.is_empty() {
        return None;
    }
    let middle = samples.iter().fold([0.0f64; 2], |sum, point| {
        [sum[0] + point[0], sum[1] + point[1]]
    });
    let count = samples.len() as f64;
    let centre = [middle[0] / count, middle[1] / count];

    let usable = |candidate: [f64; 2]| {
        let on_edge = boundary
            .iter()
            .any(|edge| crate::geom2d::distance_to(edge, candidate) <= tolerance);
        (!on_edge
            && crate::geom2d::contains(
                &boundary,
                candidate,
                crate::geom2d::Tolerance::new(tolerance),
            ))
        .then_some(candidate)
    };
    let chosen = usable(centre).or_else(|| {
        samples
            .iter()
            .find_map(|point| usable([(centre[0] + point[0]) * 0.5, (centre[1] + point[1]) * 0.5]))
    })?;
    Some(surface.point_at(chosen[0], chosen[1]))
}

/// Copies a face and everything bounding it into the result.
///
/// Geometry is copied rather than shared: the two inputs have their own
/// arenas, and a key from one means nothing in the other.
fn copy_face(
    result: &mut Body,
    source: &Body,
    face: FaceKey,
    shell: super::topology::ShellKey,
    flip: bool,
) -> Result<(), Snag> {
    let node = source.faces.get(face).ok_or(Snag::CutRefused)?;
    let surface = source
        .surfaces
        .get(node.surface)
        .ok_or(Snag::CutRefused)?
        .clone();
    let surface = result.surfaces.insert(surface);
    let new_face = result.faces.insert(Face {
        surface,
        // A difference turns the second solid's faces into the walls of the
        // cavity it leaves, which face the other way.
        forward: node.forward != flip,
        loops: Vec::new(),
        owner: shell,
        provenance: Provenance::Synthesized,
    });
    let mut rings = Vec::new();
    for ring in &node.loops {
        let source_ring = source.loops.get(*ring).ok_or(Snag::CutRefused)?;
        let new_ring = result.loops.insert(super::topology::Loop {
            coedges: Vec::new(),
            owner: new_face,
            provenance: Provenance::Synthesized,
        });
        let mut coedges = Vec::new();
        for coedge in &source_ring.coedges {
            let source_coedge = source.coedges.get(*coedge).ok_or(Snag::CutRefused)?;
            let source_edge = source
                .edges
                .get(source_coedge.edge)
                .ok_or(Snag::CutRefused)?;
            let curve = source
                .curves
                .get(source_edge.curve)
                .ok_or(Snag::CutRefused)?
                .clone();
            let start = copy_vertex(result, source, source_edge.start)?;
            let end = copy_vertex(result, source, source_edge.end)?;
            let middle = curve.point_at(
                0.5 * (source_edge.start_parameter + source_edge.end_parameter),
            );
            // Reuse rather than copy where the same edge is already there.
            // This is the stitching: two kept faces that met along a seam in
            // their own solid meet along it here only if their copies share
            // the edge, and without that the result is a shell of loose
            // faces that still passes every local check.
            let edge = match find_edge(result, start, end, middle) {
                Some(existing) => existing,
                None => {
                    let curve = result.curves.insert(curve);
                    result.edges.insert(super::topology::Edge {
                        curve,
                        start_parameter: source_edge.start_parameter,
                        end_parameter: source_edge.end_parameter,
                        start,
                        end,
                        coedges: Vec::new(),
                        provenance: Provenance::Synthesized,
                    })
                }
            };
            let reversed = result
                .edges
                .get(edge)
                .is_some_and(|node| node.start != start);
            let reverse_pcurve = flip != reversed;
            let pcurve = match (&source_coedge.pcurve, reverse_pcurve) {
                (Some(curve), true) => Some(reversed_pcurve(curve).ok_or(Snag::CutRefused)?),
                (Some(curve), false) => Some(curve.clone()),
                (None, _) => None,
            };
            let new_coedge = result.coedges.insert(super::topology::Coedge {
                edge,
                // Flipping a face reverses the way its boundary runs, or the
                // loop would wind the wrong way round the new outward normal.
                // A reused edge may also run the other way than the one this
                // coedge came from, which the sense has to absorb.
                forward: (source_coedge.forward != flip) != reversed,
                pcurve,
                owner: new_ring,
                provenance: Provenance::Synthesized,
            });
            result
                .edges
                .get_mut(edge)
                .ok_or(Snag::CutRefused)?
                .coedges
                .push(new_coedge);
            coedges.push(new_coedge);
        }
        if flip {
            // The loop is traversed the other way as well as each coedge.
            coedges.reverse();
        }
        result
            .loops
            .get_mut(new_ring)
            .ok_or(Snag::CutRefused)?
            .coedges = coedges;
        rings.push(new_ring);
    }
    result
        .faces
        .get_mut(new_face)
        .ok_or(Snag::CutRefused)?
        .loops = rings;
    result
        .shells
        .get_mut(shell)
        .ok_or(Snag::CutRefused)?
        .faces
        .push(new_face);
    Ok(())
}

fn reversed_pcurve(curve: &crate::geom2d::Curve) -> Option<crate::geom2d::Curve> {
    use crate::geom2d::{Curve, Line};
    Some(match curve {
        Curve::Line(line) => Curve::Line(Line {
            start: line.end,
            end: line.start,
        }),
        Curve::Nurbs(curve) => Curve::Nurbs(curve.reversed()),
        _ => return None,
    })
}

/// An edge already joining the two vertices along the same path, if there is
/// one.
///
/// Endpoints alone are not enough — two arcs can join the same pair of
/// vertices going opposite ways round — so the middle is compared too.
fn find_edge(
    result: &Body,
    start: super::topology::VertexKey,
    end: super::topology::VertexKey,
    middle: [f64; 3],
) -> Option<super::topology::EdgeKey> {
    result
        .edges
        .iter()
        .find(|(key, edge)| {
            let ends_match = (edge.start == start && edge.end == end)
                || (edge.start == end && edge.end == start);
            ends_match
                && result
                    .edge_endpoints(*key)
                    .is_some()
                    .then(|| {
                        result.curves.get(edge.curve).map(|curve| {
                            curve.point_at(
                                0.5 * (edge.start_parameter + edge.end_parameter),
                            )
                        })
                    })
                    .flatten()
                    .is_some_and(|point| {
                        Vec3::from(point).distance(Vec3::from(middle)) <= 1e-9
                    })
        })
        .map(|(key, _)| key)
}

/// Copies a vertex, reusing one already at the same place.
///
/// Reuse is what stitches the result: two faces that met along a seam in
/// their own solids meet along it here only if their copies share vertices.
fn copy_vertex(
    result: &mut Body,
    source: &Body,
    vertex: super::topology::VertexKey,
) -> Result<super::topology::VertexKey, Snag> {
    let point = source.vertices.get(vertex).ok_or(Snag::CutRefused)?.point;
    let existing = result
        .vertices
        .iter()
        .find(|(_, node)| Vec3::from(node.point).distance(Vec3::from(point)) <= 1e-9)
        .map(|(key, _)| key);
    Ok(match existing {
        Some(key) => key,
        None => result.vertices.insert(super::topology::Vertex {
            point,
            provenance: Provenance::Synthesized,
        }),
    })
}

/// The boundary curves of a face, for a caller inspecting a result.
pub fn boundary_of(body: &Body, face: FaceKey, tolerance: f64) -> Option<Vec<Curve>> {
    pcurve::face_boundary(body, face, tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brep::bounds::body_bounds;
    use crate::brep::make::cuboid;

    const TOL: f64 = 1e-9;

    /// Two ten-unit boxes overlapping in a five-unit corner.
    fn pair() -> (Body, Body) {
        (
            cuboid([0.0; 3], [10.0, 10.0, 10.0]).unwrap(),
            cuboid([5.0; 3], [10.0, 10.0, 10.0]).unwrap(),
        )
    }

    #[test]
    fn a_union_reaches_across_both() {
        let (a, b) = pair();
        let result = combine(a, b, Operation::Union, TOL).expect("two boxes union");
        let bounds = body_bounds(&result).expect("a planar result is bounded");
        assert_eq!(bounds.min, [0.0; 3]);
        assert_eq!(bounds.max, [15.0; 3]);
    }

    #[test]
    fn an_intersection_is_only_the_overlap() {
        let (a, b) = pair();
        let result = combine(a, b, Operation::Intersection, TOL).expect("two boxes intersect");
        let bounds = body_bounds(&result).unwrap();
        assert_eq!(bounds.min, [5.0; 3]);
        assert_eq!(bounds.max, [10.0; 3]);
    }

    #[test]
    fn a_difference_keeps_the_first_and_hollows_the_second_out_of_it() {
        let (a, b) = pair();
        let result = combine(a, b, Operation::Difference, TOL).expect("two boxes differ");
        let bounds = body_bounds(&result).unwrap();
        // The outer extent is the first box's; the second only removes.
        assert_eq!(bounds.min, [0.0; 3]);
        assert_eq!(bounds.max, [10.0; 3]);
    }

    #[test]
    fn a_difference_keeps_more_faces_than_the_original_had() {
        // The corner bitten out adds walls.
        let (a, b) = pair();
        let result = combine(a, b, Operation::Difference, TOL).unwrap();
        assert!(result.faces.len() > 6, "{}", result.faces.len());
    }

    #[test]
    fn solids_that_do_not_touch_union_to_both_and_intersect_to_nothing() {
        let a = cuboid([0.0; 3], [1.0; 3]).unwrap();
        let b = cuboid([50.0; 3], [1.0; 3]).unwrap();
        let joined = combine(a.clone(), b.clone(), Operation::Union, TOL).unwrap();
        assert_eq!(joined.faces.len(), 12, "both boxes, whole");
        let shared = combine(a, b, Operation::Intersection, TOL).unwrap();
        assert_eq!(shared.faces.len(), 0, "nothing is in both");
    }

    #[test]
    fn a_difference_by_something_far_away_leaves_the_original_alone() {
        let a = cuboid([0.0; 3], [4.0; 3]).unwrap();
        let b = cuboid([50.0; 3], [1.0; 3]).unwrap();
        let result = combine(a, b, Operation::Difference, TOL).unwrap();
        assert_eq!(result.faces.len(), 6);
        let bounds = body_bounds(&result).unwrap();
        assert_eq!(bounds.max, [4.0; 3]);
    }

    /// Two boxes stacked face to face, the shared wall covering exactly the
    /// same ground on both.
    fn stacked() -> (Body, Body) {
        (
            cuboid([0.0; 3], [10.0; 3]).unwrap(),
            cuboid([0.0, 0.0, 10.0], [10.0; 3]).unwrap(),
        )
    }

    #[test]
    fn stacked_boxes_union_into_one_solid_with_no_wall_between() {
        let (a, b) = stacked();
        let result = combine(a, b, Operation::Union, TOL).expect("a stack unions");
        let bounds = body_bounds(&result).unwrap();
        assert_eq!(bounds.min, [0.0; 3]);
        assert_eq!(bounds.max, [10.0, 10.0, 20.0]);
        // Five sides each, and the two that met are gone.
        assert_eq!(result.faces.len(), 10);
        assert!(result.validate().is_empty());
        assert_eq!(result.euler_characteristic(), 2);
    }

    #[test]
    fn a_stack_differenced_leaves_the_first_box_whole() {
        // The flip is what does it: the second solid's bottom turns to face
        // up, matching the first's top, so the wall is real and one copy
        // stays. Without it the result would be open at the top.
        let (a, b) = stacked();
        let result = combine(a, b, Operation::Difference, TOL).expect("a stack differs");
        assert_eq!(result.faces.len(), 6, "the first box, untouched");
        let bounds = body_bounds(&result).unwrap();
        assert_eq!(bounds.max, [10.0; 3]);
        assert!(result.validate().is_empty());
    }

    #[test]
    fn a_stack_intersects_to_nothing() {
        // They share a face and no volume.
        let (a, b) = stacked();
        let result = combine(a, b, Operation::Intersection, TOL).expect("a stack intersects");
        assert_eq!(result.faces.len(), 0);
    }

    #[test]
    fn two_identical_solids_union_to_one_of_them() {
        let a = cuboid([0.0; 3], [4.0; 3]).unwrap();
        let b = cuboid([0.0; 3], [4.0; 3]).unwrap();
        let result = combine(a, b, Operation::Union, TOL).expect("identical boxes union");
        assert_eq!(result.faces.len(), 6, "one box, not two");
        assert!(result.validate().is_empty());
        assert_eq!(result.euler_characteristic(), 2);
    }

    #[test]
    fn a_solid_differenced_by_itself_leaves_nothing() {
        let a = cuboid([0.0; 3], [4.0; 3]).unwrap();
        let b = cuboid([0.0; 3], [4.0; 3]).unwrap();
        let result = combine(a, b, Operation::Difference, TOL).expect("a box minus itself");
        assert_eq!(result.faces.len(), 0);
    }

    /// How much a body encloses once meshed. Positive when it is the right
    /// way out; a shared wall left in twice, or taken out twice, changes it.
    fn volume(solid: &Body) -> f64 {
        let mesh = super::super::mesh::body(
            solid,
            crate::tessellation::DEFAULT_ANGLE,
            1e-9,
        );
        mesh.triangles
            .iter()
            .map(|triangle| {
                let at = |index: usize| {
                    crate::space::Vec3::from(mesh.positions[triangle[index]])
                };
                at(0).cross(at(1)).dot(at(2)) / 6.0
            })
            .sum()
    }

    #[test]
    fn a_wall_shared_in_part_is_resolved_rather_than_refused() {
        // A small box standing on a big one. They meet over a corner of the
        // big one's top face — the shared plane holds all of one region and
        // part of the other — and the imprint now cuts the larger face where
        // the sharing stops, so each piece is wholly shared or wholly not.
        //
        // The volume is the check: a doubled wall or a missing one both leave
        // something that looks finished.
        let a = cuboid([0.0; 3], [10.0; 3]).unwrap();
        let b = cuboid([0.0, 0.0, 10.0], [4.0, 4.0, 4.0]).unwrap();
        let result = combine(a, b, Operation::Union, TOL).expect("a stacked union");
        assert!(result.validate().is_empty());
        assert!(
            (volume(&result) - (1000.0 + 64.0)).abs() < 1e-6,
            "{}",
            volume(&result)
        );
    }

    #[test]
    fn a_box_standing_on_another_can_be_cut_back_out_of_it() {
        // The same pair the other way about. What the difference leaves is
        // the big box untouched — the small one sits on top of it rather
        // than in it — which only comes out right if the shared wall is
        // resolved rather than doubled.
        let a = cuboid([0.0; 3], [10.0; 3]).unwrap();
        let b = cuboid([0.0, 0.0, 10.0], [4.0, 4.0, 4.0]).unwrap();
        let result = combine(a, b, Operation::Difference, TOL).expect("a stacked difference");
        assert!(result.validate().is_empty());
        assert!((volume(&result) - 1000.0).abs() < 1e-6, "{}", volume(&result));
    }

    #[test]
    fn every_face_kept_is_on_the_side_the_operation_asked_for() {
        // Read back from the result rather than trusted: each face's own
        // interior point, tested against the solid it was cut against.
        let (a, b) = pair();
        let original_b = b.clone();
        let result = combine(a, b, Operation::Union, TOL).unwrap();
        for face in result.face_keys() {
            let Some(point) = interior_point(&result, face, 1e-9) else {
                continue;
            };
            assert_ne!(
                contains_point(&original_b, point, 1e-6),
                Containment::Inside,
                "a union kept a face inside the other solid"
            );
        }
    }

    #[test]
    fn a_union_at_survey_coordinates_reaches_the_same_way() {
        let origin = [512_345.678, 4_512_345.678, 91.5];
        let a = cuboid(origin, [10.0; 3]).unwrap();
        let b = cuboid(
            [origin[0] + 5.0, origin[1] + 5.0, origin[2] + 5.0],
            [10.0; 3],
        )
        .unwrap();
        let result = combine(a, b, Operation::Union, 1e-6).expect("the same union, further out");
        let bounds = body_bounds(&result).unwrap();
        assert!((bounds.max[0] - (origin[0] + 15.0)).abs() < 1e-6, "{bounds:?}");
    }

    #[test]
    fn the_result_carries_its_own_geometry() {
        // Keys from one body mean nothing in another, so every surface,
        // curve and vertex is copied rather than referred to.
        let (a, b) = pair();
        let result = combine(a, b, Operation::Union, TOL).unwrap();
        assert!(result.surfaces.len() >= result.faces.len());
        assert!(!result.curves.is_empty());
        for (_, face) in result.faces.iter() {
            assert!(result.surfaces.contains(face.surface));
        }
        for (_, edge) in result.edges.iter() {
            assert!(result.curves.contains(edge.curve));
            assert!(result.vertices.contains(edge.start));
            assert!(result.vertices.contains(edge.end));
        }
    }

    #[test]
    fn a_union_comes_out_a_closed_solid() {
        // The check that the faces were stitched rather than merely
        // gathered. Copied face by face, an edge shared by two of them
        // becomes two edges with one coedge apiece: every local check still
        // passes and the result is a bag of loose walls.
        let (a, b) = pair();
        let result = combine(a, b, Operation::Union, TOL).unwrap();
        let flaws = result.validate();
        assert!(flaws.is_empty(), "{flaws:?}");
        for (key, edge) in result.edges.iter() {
            assert_eq!(edge.coedges.len(), 2, "edge {key:?} is not shared");
        }
        assert_eq!(result.euler_characteristic(), 2, "a closed surface");
    }

    #[test]
    fn an_intersection_comes_out_a_closed_solid_too() {
        let (a, b) = pair();
        let result = combine(a, b, Operation::Intersection, TOL).unwrap();
        assert!(result.validate().is_empty());
        assert_eq!(result.euler_characteristic(), 2);
        // The overlap of two boxes is a box.
        assert_eq!(result.faces.len(), 6);
    }

    #[test]
    fn a_difference_turns_the_cutting_solid_s_faces_around() {
        // The whole of what makes a difference a difference: B's walls become
        // the inside of the cavity, so they face the other way than they did.
        let (a, b) = pair();
        let before: Vec<bool> = b.faces.iter().map(|(_, f)| f.forward).collect();
        let result = combine(a, b, Operation::Difference, TOL).unwrap();
        let flipped = result.faces.iter().filter(|(_, f)| !f.forward).count();
        assert!(before.iter().all(|forward| *forward), "the input was all forward");
        assert!(flipped > 0, "nothing was turned around");
    }
}
