//! Cutting topology apart without breaking it.
//!
//! Every modelling operation that adds detail ends here: a boolean splits the
//! faces its intersection curves cross, a fillet splits the edges it runs
//! along, and an imprint splits both. What they have in common is that the
//! result has to be as consistent as the input — a face split into two halves
//! that do not quite share their new edge is worse than one not split at all,
//! because it looks finished.
//!
//! So each operation here leaves [`Body::validate`] finding nothing, and the
//! tests check that rather than checking the pieces individually.
//!
//! # The part that is easy to forget
//!
//! An edge belongs to two faces. Splitting it changes the loop of the face
//! being worked on *and* the loop of the one on the other side, which is not
//! mentioned anywhere in the operation's own description and is exactly what
//! a first implementation misses. The far face's loop is then one coedge
//! short, its ring no longer closes, and the failure surfaces later as a
//! boolean that loses a wall.

use super::geometry::Curve3;
use super::pcurve;
use super::topology::{
    Body, Coedge, CoedgeKey, Edge, EdgeKey, Face, FaceKey, Loop, Vertex, VertexKey,
};
use super::Provenance;
use crate::geom2d::{distance_to, intersect as cross, Tolerance};
use crate::space::Vec3;

/// Splits an edge at a parameter along its curve, returning the two halves.
///
/// The first keeps the original key, so anything already pointing at the edge
/// still names its first half. The second is new.
///
/// Every coedge that used the edge is split in step, in whichever face it
/// belongs to — including the face on the far side, whose loop is just as
/// much a part of this operation as the near one's.
///
/// `None` when the parameter is at or outside the edge's own span: there is
/// nothing to divide, and creating a zero-length piece would leave topology
/// no later operation can make sense of.
pub fn split_edge(body: &mut Body, edge: EdgeKey, parameter: f64) -> Option<(EdgeKey, EdgeKey)> {
    let original = body.edges.get(edge)?.clone();
    let span = original.end_parameter - original.start_parameter;
    if span == 0.0 {
        return None;
    }
    // Measured as a fraction of the span so the guard means the same thing on
    // an edge parameterised over a unit and one parameterised over a turn.
    let across = (parameter - original.start_parameter) / span;
    if !(1e-9..=1.0 - 1e-9).contains(&across) {
        return None;
    }

    let point = body.curves.get(original.curve)?.point_at(parameter);
    let middle = body.vertices.insert(Vertex {
        point,
        provenance: Provenance::Synthesized,
    });

    let far = body.edges.insert(Edge {
        curve: original.curve,
        start_parameter: parameter,
        end_parameter: original.end_parameter,
        start: middle,
        end: original.end,
        coedges: Vec::new(),
        provenance: Provenance::Synthesized,
    });
    {
        let near = body.edges.get_mut(edge)?;
        near.end_parameter = parameter;
        near.end = middle;
        near.provenance.soil();
    }

    for coedge in &original.coedges {
        let existing = body.coedges.get(*coedge)?.clone();
        let twin = body.coedges.insert(Coedge {
            edge: far,
            forward: existing.forward,
            owner: existing.owner,
            provenance: Provenance::Synthesized,
        });
        body.edges.get_mut(far)?.coedges.push(twin);

        let ring = body.loops.get_mut(existing.owner)?;
        let at = ring.coedges.iter().position(|key| *key == *coedge)?;
        // A coedge running with the curve meets the near half first, so the
        // far half follows it. One running against the curve meets the far
        // half first, so the new coedge goes in ahead of it.
        ring.coedges.insert(if existing.forward { at + 1 } else { at }, twin);
        ring.provenance.soil();

        let face = ring.owner;
        if let Some(face) = body.faces.get_mut(face) {
            face.provenance.soil();
        }
        if let Some(coedge) = body.coedges.get_mut(*coedge) {
            coedge.provenance.soil();
        }
    }

    Some((edge, far))
}

/// Splits an edge at the point on it nearest `point`.
///
/// The form a caller with an intersection result has: it knows where the
/// curves met, not what parameter that was.
pub fn split_edge_at(body: &mut Body, edge: EdgeKey, point: [f64; 3]) -> Option<(EdgeKey, EdgeKey)> {
    let parameter = {
        let node = body.edges.get(edge)?;
        body.curves.get(node.curve)?.parameter_at(point)
    };
    split_edge(body, edge, parameter)
}


/// Cuts a face in two along `cutter`, returning both halves.
///
/// The first keeps the original key. Both lie on the same surface — a cut
/// divides a face without moving it — and both belong to the same shell.
///
/// # What is handled
///
/// A cut that enters the face's boundary once and leaves once, which is the
/// case a boolean produces at every face an intersection curve passes
/// through. A cut that crosses more often would leave more than two pieces,
/// and one that closes inside the face makes a hole rather than a division;
/// both answer `None` rather than returning one of the pieces and dropping
/// the others.
///
/// `None` also when the cutter does not lie on the face's surface, when the
/// surface's parameter space cannot be written down — see
/// [`pcurve::project`] — and when the face already has holes, which needs the
/// inner loops assigned to the right half and is a case of its own.
///
/// # Where the crossings come from
///
/// In the surface's parameter space, where the cut and the boundary are both
/// plane curves and the crossing is a plane intersection. Doing it in space
/// would mean intersecting two curves that share no natural parameter and
/// meet at a tangent as often as not.
pub fn split_face(
    body: &mut Body,
    face: FaceKey,
    cutter: &Curve3,
    tolerance: f64,
) -> Option<[FaceKey; 2]> {
    let node = body.faces.get(face)?.clone();
    if node.loops.len() != 1 {
        return None;
    }
    let ring_key = node.loops[0];
    let surface = body.surfaces.get(node.surface)?.clone();
    let flat_cutter = pcurve::project(&surface, cutter, tolerance)?;

    // Where the cut meets the boundary, as points in space.
    let mut landings: Vec<Landing> = Vec::new();
    for coedge in body.loops.get(ring_key)?.coedges.clone() {
        let edge_key = body.coedges.get(coedge)?.edge;
        let edge = body.edges.get(edge_key)?.clone();
        let curve = body.curves.get(edge.curve)?.clone();
        let Some(flat_edge) = pcurve::project(&surface, &curve, tolerance) else {
            continue;
        };
        for crossing in cross(&flat_cutter, &flat_edge, Tolerance::new(tolerance)) {
            let point = surface.point_at(crossing.point[0], crossing.point[1]);
            let along = curve.parameter_at(point);
            // The boundary pcurve may run past the edge it came from — a
            // straight edge projects to an infinite line — so a crossing off
            // the edge's own span is not on the boundary at all.
            let (low, high) = (
                edge.start_parameter.min(edge.end_parameter),
                edge.start_parameter.max(edge.end_parameter),
            );
            let slack = (high - low).abs() * 1e-9;
            if along < low - slack || along > high + slack {
                continue;
            }
            if landings
                .iter()
                .any(|seen| Vec3::from(seen.point).distance(Vec3::from(point)) <= tolerance)
            {
                continue;
            }
            landings.push(Landing {
                edge: edge_key,
                point,
                parameter: along,
            });
        }
    }
    if landings.len() != 2 {
        return None;
    }

    // Each landing becomes a vertex: an existing one where the cut runs into
    // a corner, a new one where it crosses an edge partway.
    let mut ends: Vec<VertexKey> = Vec::with_capacity(2);
    for landing in &landings {
        ends.push(vertex_at(body, landing, tolerance)?);
    }
    let [first, second] = [ends[0], ends[1]];
    if first == second {
        return None;
    }

    // A cut that runs along the boundary divides nothing. It happens
    // naturally the moment a face has already been cut: the new edge lies on
    // the cutter, its two ends are vertices, and the next pass finds the same
    // two landings and cuts again — the same face for ever. Asked of the
    // midpoint rather than the ends, since a genuine cut also touches the
    // boundary at both of those.
    let midway = cutter.point_at(
        0.5 * (cutter.parameter_at(body.vertices.get(first)?.point)
            + cutter.parameter_at(body.vertices.get(second)?.point)),
    );
    let boundary = pcurve::face_boundary(body, face, tolerance)?;
    let (u, v) = surface.parameters_at(midway)?;
    if boundary
        .iter()
        .any(|edge| distance_to(edge, [u, v]) <= tolerance)
    {
        return None;
    }

    // The new edge, running along the cut between them.
    let point_of = |body: &Body, key: VertexKey| Some(body.vertices.get(key)?.point);
    let first_parameter = cutter.parameter_at(point_of(body, first)?);
    let second_parameter = cutter.parameter_at(point_of(body, second)?);
    let (start, end, start_parameter, end_parameter) = if first_parameter <= second_parameter {
        (first, second, first_parameter, second_parameter)
    } else {
        (second, first, second_parameter, first_parameter)
    };
    let curve = body.curves.insert(cutter.clone());
    let cut = body.edges.insert(Edge {
        curve,
        start_parameter,
        end_parameter,
        start,
        end,
        coedges: Vec::new(),
        provenance: Provenance::Synthesized,
    });

    // The boundary, now split at both ends of the cut, divides into the two
    // arcs between them.
    let ring = body.loops.get(ring_key)?.coedges.clone();
    let begins_at = |body: &Body, coedge: CoedgeKey, vertex: VertexKey| {
        body.coedge_vertices(coedge)
            .is_some_and(|(from, _)| from == vertex)
    };
    let at_first = ring
        .iter()
        .position(|c| begins_at(body, *c, first))?;
    let at_second = ring
        .iter()
        .position(|c| begins_at(body, *c, second))?;
    // The stretch of the ring from one index round to the other, not
    // including where it stops.
    let arc = |from: usize, to: usize| -> Vec<CoedgeKey> {
        let count = ring.len();
        let mut out = Vec::new();
        let mut index = from;
        while index != to {
            out.push(ring[index]);
            index = (index + 1) % count;
        }
        out
    };
    let near = arc(at_first, at_second);
    let far = arc(at_second, at_first);
    if near.is_empty() || far.is_empty() {
        return None;
    }

    // Each arc is closed by the cut, traversed whichever way takes it back to
    // where the arc began. The two therefore run it opposite ways, which is
    // what makes the new edge a shared one rather than two coincident walls.
    let sense = |from: VertexKey| start == from;
    let near_closer = body.coedges.insert(Coedge {
        edge: cut,
        forward: sense(second),
        owner: ring_key,
        provenance: Provenance::Synthesized,
    });
    let mut kept = near;
    kept.push(near_closer);
    {
        let ring = body.loops.get_mut(ring_key)?;
        ring.coedges = kept;
        ring.provenance.soil();
    }
    body.edges.get_mut(cut)?.coedges.push(near_closer);
    if let Some(face) = body.faces.get_mut(face) {
        face.provenance.soil();
    }

    // The far arc moves onto a new face on the same surface, in the same
    // shell.
    let other = body.faces.insert(Face {
        surface: node.surface,
        forward: node.forward,
        loops: Vec::new(),
        owner: node.owner,
        provenance: Provenance::Synthesized,
    });
    let other_ring = body.loops.insert(Loop {
        coedges: Vec::new(),
        owner: other,
        provenance: Provenance::Synthesized,
    });
    let far_closer = body.coedges.insert(Coedge {
        edge: cut,
        forward: sense(first),
        owner: other_ring,
        provenance: Provenance::Synthesized,
    });
    body.edges.get_mut(cut)?.coedges.push(far_closer);
    for coedge in &far {
        body.coedges.get_mut(*coedge)?.owner = other_ring;
    }
    let mut moved = far;
    moved.push(far_closer);
    body.loops.get_mut(other_ring)?.coedges = moved;
    body.faces.get_mut(other)?.loops = vec![other_ring];
    body.shells.get_mut(node.owner)?.faces.push(other);

    Some([face, other])
}

/// Where the cut met the boundary.
struct Landing {
    edge: EdgeKey,
    point: [f64; 3],
    parameter: f64,
}

/// The vertex at a landing: an existing end of the edge when the cut runs
/// into a corner, otherwise a new one from splitting the edge there.
fn vertex_at(body: &mut Body, landing: &Landing, tolerance: f64) -> Option<VertexKey> {
    let edge = body.edges.get(landing.edge)?.clone();
    for end in [edge.start, edge.end] {
        let point = body.vertices.get(end)?.point;
        if Vec3::from(point).distance(Vec3::from(landing.point)) <= tolerance {
            return Some(end);
        }
    }
    let (near, far) = split_edge(body, landing.edge, landing.parameter)?;
    shared_vertex(body, near, far)
}

/// The vertex an edge split introduced, given the two halves.
pub fn shared_vertex(body: &Body, near: EdgeKey, far: EdgeKey) -> Option<VertexKey> {
    let near = body.edges.get(near)?;
    let far = body.edges.get(far)?;
    [near.start, near.end]
        .into_iter()
        .find(|key| *key == far.start || *key == far.end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brep::make::cuboid;
    use crate::space::Vec3;

    fn box_body() -> Body {
        cuboid([0.0, 0.0, 0.0], [2.0, 4.0, 6.0]).expect("a box")
    }

    #[test]
    fn splitting_an_edge_leaves_the_body_consistent() {
        let mut body = box_body();
        let edge = body.edges.keys().next().unwrap();
        split_edge(&mut body, edge, 0.5).expect("a split at the middle");
        let flaws = body.validate();
        assert!(flaws.is_empty(), "{flaws:?}");
    }

    #[test]
    fn splitting_an_edge_adds_one_vertex_and_one_edge() {
        // Which leaves V − E + F where it was: the shape has not changed,
        // only how it is written down.
        let mut body = box_body();
        let before = (body.vertices.len(), body.edges.len(), body.faces.len());
        let characteristic = body.euler_characteristic();
        let edge = body.edges.keys().next().unwrap();
        split_edge(&mut body, edge, 0.5).unwrap();
        assert_eq!(body.vertices.len(), before.0 + 1);
        assert_eq!(body.edges.len(), before.1 + 1);
        assert_eq!(body.faces.len(), before.2);
        assert_eq!(body.euler_characteristic(), characteristic);
    }

    #[test]
    fn both_faces_that_used_the_edge_gain_a_coedge() {
        // The one that is easy to forget. Splitting only the near face's loop
        // leaves the far one a coedge short, its ring open, and the failure
        // turns up much later as a boolean losing a wall.
        let mut body = box_body();
        let edge = body.edges.keys().next().unwrap();
        let faces: Vec<_> = body
            .edges
            .get(edge)
            .unwrap()
            .coedges
            .iter()
            .map(|c| {
                let owner = body.coedges.get(*c).unwrap().owner;
                body.loops.get(owner).unwrap().owner
            })
            .collect();
        assert_eq!(faces.len(), 2);
        let before: Vec<usize> = faces.iter().map(|f| body.face_coedges(*f).len()).collect();
        split_edge(&mut body, edge, 0.5).unwrap();
        for (face, was) in faces.iter().zip(before) {
            assert_eq!(body.face_coedges(*face).len(), was + 1, "face {face:?}");
        }
    }

    #[test]
    fn the_new_vertex_sits_on_the_curve_where_it_was_asked_for() {
        let mut body = box_body();
        let edge = body.edges.keys().next().unwrap();
        let expected = {
            let node = body.edges.get(edge).unwrap();
            body.curves.get(node.curve).unwrap().point_at(0.25)
        };
        let (near, far) = split_edge(&mut body, edge, 0.25).unwrap();
        let vertex = shared_vertex(&body, near, far).unwrap();
        let point = body.vertices.get(vertex).unwrap().point;
        assert!(Vec3::from(point).distance(Vec3::from(expected)) < 1e-12);
        assert!(body.worst_vertex_gap() < 1e-12);
    }

    #[test]
    fn the_halves_run_the_same_way_the_whole_did() {
        let mut body = box_body();
        let edge = body.edges.keys().next().unwrap();
        let (start, end) = {
            let node = body.edges.get(edge).unwrap();
            (node.start, node.end)
        };
        let (near, far) = split_edge(&mut body, edge, 0.5).unwrap();
        assert_eq!(body.edges.get(near).unwrap().start, start);
        assert_eq!(body.edges.get(far).unwrap().end, end);
        let middle = shared_vertex(&body, near, far).unwrap();
        assert_eq!(body.edges.get(near).unwrap().end, middle);
        assert_eq!(body.edges.get(far).unwrap().start, middle);
    }

    #[test]
    fn every_loop_still_closes_after_a_split() {
        let mut body = box_body();
        let edge = body.edges.keys().next().unwrap();
        split_edge(&mut body, edge, 0.5).unwrap();
        for (key, ring) in body.loops.iter() {
            let count = ring.coedges.len();
            for index in 0..count {
                let (_, ends) = body.coedge_vertices(ring.coedges[index]).unwrap();
                let (begins, _) = body
                    .coedge_vertices(ring.coedges[(index + 1) % count])
                    .unwrap();
                assert_eq!(ends, begins, "loop {key:?} breaks after {index}");
            }
        }
    }

    #[test]
    fn a_backward_coedge_gets_its_new_piece_on_the_right_side() {
        // Every edge of a box is traversed forwards by one face and backwards
        // by the other, so this case is covered by any split — but only if
        // the insertion point depends on the sense. Putting the new coedge
        // after the old one in both cases breaks exactly the backward loop,
        // which is what the closure check above would catch.
        let mut body = box_body();
        let edge = body.edges.keys().next().unwrap();
        let backward = body
            .edges
            .get(edge)
            .unwrap()
            .coedges
            .iter()
            .copied()
            .find(|c| !body.coedges.get(*c).unwrap().forward)
            .expect("one side runs against the curve");
        let ring = body.coedges.get(backward).unwrap().owner;
        let before = body.loops.get(ring).unwrap().coedges.clone();
        let at = before.iter().position(|c| *c == backward).unwrap();
        split_edge(&mut body, edge, 0.5).unwrap();
        let after = body.loops.get(ring).unwrap().coedges.clone();
        assert_eq!(after.len(), before.len() + 1);
        assert_eq!(after[at + 1], backward, "the new piece comes first");
    }

    #[test]
    fn splitting_twice_divides_an_edge_into_three() {
        let mut body = box_body();
        let edge = body.edges.keys().next().unwrap();
        let (near, far) = split_edge(&mut body, edge, 0.5).unwrap();
        split_edge(&mut body, far, 0.75).unwrap();
        let flaws = body.validate();
        assert!(flaws.is_empty(), "{flaws:?}");
        assert_eq!(body.edges.len(), 14);
        assert_eq!(body.vertices.len(), 10);
        assert_eq!(body.euler_characteristic(), 2);
        assert!(body.edges.contains(near));
    }

    #[test]
    fn a_split_at_an_end_is_refused() {
        let mut body = box_body();
        let edge = body.edges.keys().next().unwrap();
        assert!(split_edge(&mut body, edge, 0.0).is_none());
        assert!(split_edge(&mut body, edge, 1.0).is_none());
        assert!(split_edge(&mut body, edge, 1.5).is_none());
        assert!(split_edge(&mut body, edge, -0.5).is_none());
        assert!(body.validate().is_empty(), "nothing was changed");
        assert_eq!(body.edges.len(), 12);
    }

    #[test]
    fn splitting_by_a_point_lands_where_the_point_is() {
        let mut body = box_body();
        let edge = body.edges.keys().next().unwrap();
        let (start, end) = body.edge_endpoints(edge).unwrap();
        let middle = Vec3::from(start).lerp(Vec3::from(end), 0.3).to_array();
        let (near, far) = split_edge_at(&mut body, edge, middle).unwrap();
        let vertex = shared_vertex(&body, near, far).unwrap();
        let point = body.vertices.get(vertex).unwrap().point;
        assert!(Vec3::from(point).distance(Vec3::from(middle)) < 1e-12);
        assert!(body.validate().is_empty());
    }


    /// The face of the box lying on z = 0, and its surface.
    fn bottom_face(body: &Body) -> FaceKey {
        body.face_keys()
            .find(|key| {
                let face = body.faces.get(*key).unwrap();
                let crate::brep::Surface::Plane(plane) =
                    body.surfaces.get(face.surface).unwrap()
                else {
                    unreachable!()
                };
                plane.origin[2].abs() < 1e-9 && plane.normal().unwrap()[2].abs() > 0.9
            })
            .expect("a box has a bottom")
    }

    /// A line across the box's bottom face at constant y, running in x.
    fn across_bottom(y: f64) -> Curve3 {
        Curve3::Line(crate::brep::Line3 {
            origin: [-1.0, y, 0.0],
            direction: [1.0, 0.0, 0.0],
        })
    }

    #[test]
    fn cutting_a_face_leaves_two_and_the_body_consistent() {
        let mut body = box_body();
        let face = bottom_face(&body);
        let [kept, made] = split_face(&mut body, face, &across_bottom(2.0), 1e-9)
            .expect("a cut straight across");
        assert_eq!(kept, face);
        assert_ne!(made, face);
        assert_eq!(body.faces.len(), 7);
        let flaws = body.validate();
        assert!(flaws.is_empty(), "{flaws:?}");
    }

    #[test]
    fn a_cut_face_is_still_a_closed_solid() {
        // Two vertices, three edges and one face are added, which leaves
        // V − E + F alone: the shape has not changed, only its description.
        let mut body = box_body();
        let before = body.euler_characteristic();
        let face = bottom_face(&body);
        split_face(&mut body, face, &across_bottom(2.0), 1e-9).unwrap();
        assert_eq!(body.euler_characteristic(), before);
        assert_eq!(body.vertices.len(), 10);
        assert_eq!(body.edges.len(), 15);
        assert_eq!(body.faces.len(), 7);
    }

    #[test]
    fn both_halves_lie_on_the_surface_the_face_did() {
        let mut body = box_body();
        let face = bottom_face(&body);
        let surface = body.faces.get(face).unwrap().surface;
        let [kept, made] = split_face(&mut body, face, &across_bottom(2.0), 1e-9).unwrap();
        assert_eq!(body.faces.get(kept).unwrap().surface, surface);
        assert_eq!(body.faces.get(made).unwrap().surface, surface);
        // And in the same shell, so the solid is still one piece.
        assert_eq!(
            body.faces.get(kept).unwrap().owner,
            body.faces.get(made).unwrap().owner
        );
    }

    #[test]
    fn the_two_halves_share_the_new_edge_running_it_opposite_ways() {
        // What makes it a division rather than two coincident walls.
        let mut body = box_body();
        let face = bottom_face(&body);
        let before: Vec<_> = body.edges.keys().collect();
        split_face(&mut body, face, &across_bottom(2.0), 1e-9).unwrap();
        let cut = body
            .edges
            .keys()
            .find(|key| !before.contains(key) && body.edges.get(*key).unwrap().coedges.len() == 2)
            .expect("a new shared edge");
        let senses: Vec<bool> = body
            .edges
            .get(cut)
            .unwrap()
            .coedges
            .iter()
            .map(|c| body.coedges.get(*c).unwrap().forward)
            .collect();
        assert_ne!(senses[0], senses[1]);
    }

    #[test]
    fn the_halves_add_up_to_the_whole() {
        // Four coedges before; after the cut each half has three of the
        // original's pieces plus the cut, and the two edges it crossed have
        // each become two.
        let mut body = box_body();
        let face = bottom_face(&body);
        let [kept, made] = split_face(&mut body, face, &across_bottom(2.0), 1e-9).unwrap();
        assert_eq!(body.face_coedges(kept).len(), 4);
        assert_eq!(body.face_coedges(made).len(), 4);
    }

    #[test]
    fn a_cut_from_corner_to_corner_uses_the_corners_it_finds() {
        // The diagonal of the bottom face. Both ends land exactly on
        // existing vertices, so nothing is split and no vertex is added —
        // which a version that always split would get wrong by leaving two
        // zero-length edges behind.
        let mut body = cuboid([0.0; 3], [4.0, 4.0, 4.0]).unwrap();
        let face = bottom_face(&body);
        let diagonal = Curve3::Line(crate::brep::Line3 {
            origin: [0.0, 0.0, 0.0],
            direction: [1.0, 1.0, 0.0],
        });
        let vertices = body.vertices.len();
        split_face(&mut body, face, &diagonal, 1e-9).expect("a diagonal cut");
        assert_eq!(body.vertices.len(), vertices, "no new corners were needed");
        assert_eq!(body.edges.len(), 13, "only the cut itself");
        let flaws = body.validate();
        assert!(flaws.is_empty(), "{flaws:?}");
    }

    #[test]
    fn a_cut_that_misses_the_face_does_nothing() {
        let mut body = box_body();
        let face = bottom_face(&body);
        let before = body.faces.len();
        // Well outside the box's footprint.
        assert!(split_face(&mut body, face, &across_bottom(99.0), 1e-9).is_none());
        assert_eq!(body.faces.len(), before);
        assert!(body.validate().is_empty());
    }

    #[test]
    fn a_cutter_off_the_surface_is_refused() {
        let mut body = box_body();
        let face = bottom_face(&body);
        let above = Curve3::Line(crate::brep::Line3 {
            origin: [-1.0, 2.0, 1.0],
            direction: [1.0, 0.0, 0.0],
        });
        assert!(split_face(&mut body, face, &above, 1e-9).is_none());
        assert!(body.validate().is_empty());
    }

    #[test]
    fn cutting_twice_gives_three_pieces() {
        let mut body = cuboid([0.0; 3], [9.0, 9.0, 9.0]).unwrap();
        let face = bottom_face(&body);
        let [kept, _] = split_face(&mut body, face, &across_bottom(3.0), 1e-9).unwrap();
        // The half that still reaches y = 6 is cut again.
        let second = [kept]
            .into_iter()
            .chain(body.face_keys())
            .find(|key| split_face(&mut body.clone(), *key, &across_bottom(6.0), 1e-9).is_some())
            .expect("one half still spans y = 6");
        split_face(&mut body, second, &across_bottom(6.0), 1e-9).unwrap();
        assert_eq!(body.faces.len(), 8, "six sides, cut twice");
        let flaws = body.validate();
        assert!(flaws.is_empty(), "{flaws:?}");
        assert_eq!(body.euler_characteristic(), 2);
    }

    #[test]
    fn a_cut_at_survey_coordinates_works_the_same() {
        let origin = [512_345.678, 4_512_345.678, 91.5];
        let mut body = cuboid(origin, [2.0, 4.0, 6.0]).unwrap();
        let face = body
            .face_keys()
            .find(|key| {
                let face = body.faces.get(*key).unwrap();
                let crate::brep::Surface::Plane(plane) =
                    body.surfaces.get(face.surface).unwrap()
                else {
                    unreachable!()
                };
                (plane.origin[2] - origin[2]).abs() < 1e-6
                    && plane.normal().unwrap()[2].abs() > 0.9
            })
            .unwrap();
        let cutter = Curve3::Line(crate::brep::Line3 {
            origin: [origin[0] - 1.0, origin[1] + 2.0, origin[2]],
            direction: [1.0, 0.0, 0.0],
        });
        split_face(&mut body, face, &cutter, 1e-6).expect("a cut at survey coordinates");
        let flaws = body.validate();
        assert!(flaws.is_empty(), "{flaws:?}");
        assert!(body.worst_vertex_gap() < 1e-6);
    }

    #[test]
    fn a_face_with_holes_is_not_guessed_at() {
        let mut body = box_body();
        let face = bottom_face(&body);
        // Give it a second loop, standing in for a hole.
        let extra = body.loops.insert(Loop {
            coedges: Vec::new(),
            owner: face,
            provenance: Provenance::Synthesized,
        });
        body.faces.get_mut(face).unwrap().loops.push(extra);
        assert!(split_face(&mut body, face, &across_bottom(2.0), 1e-9).is_none());
    }

    #[test]
    fn a_split_dirties_what_it_touched_and_leaves_the_rest_clean() {
        let mut body = box_body();
        for node in body.edges.values_mut() {
            node.provenance = Provenance::Clean(crate::brep::SourceRef::new(0));
        }
        for node in body.faces.values_mut() {
            node.provenance = Provenance::Clean(crate::brep::SourceRef::new(0));
        }
        let edge = body.edges.keys().next().unwrap();
        split_edge(&mut body, edge, 0.5).unwrap();
        let dirty_faces = body
            .faces
            .iter()
            .filter(|(_, f)| !f.provenance.is_reusable())
            .count();
        assert_eq!(dirty_faces, 2, "only the two the edge bounded");
        let clean_edges = body
            .edges
            .iter()
            .filter(|(_, e)| e.provenance.is_reusable())
            .count();
        assert_eq!(clean_edges, 11, "the other eleven are untouched");
    }
}
