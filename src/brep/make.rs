//! Building a body from nothing.
//!
//! Two purposes. A modeller needs primitives to start from, and a kernel
//! needs a solid it built itself to test against — one lifted from a file
//! tests the reader as much as the topology, and when it fails there is no
//! saying which.
//!
//! Everything here comes out passing [`Body::validate`] with no flaws, which
//! is the property that makes it worth having: a builder that can produce an
//! inconsistent body is a builder that will.

use super::arena::Key;
use super::geometry::{Curve3, Line3, Surface};
use super::topology::{
    Body, Coedge, CoedgeKey, Edge, EdgeKey, Face, Loop, Lump, Shell, Vertex, VertexKey,
};
use super::Provenance;
use crate::space::{Plane, Vec3};

/// A rectangular box with one corner at `origin` and the opposite at
/// `origin + size`.
///
/// Six planar faces, twelve edges, eight vertices — and every edge shared by
/// exactly two faces that run it opposite ways, which is what makes the
/// result a solid rather than six unrelated rectangles.
///
/// `None` for a size with a zero or negative component: a box of no thickness
/// has faces on top of each other, and no amount of care downstream recovers
/// from that.
pub fn cuboid(origin: [f64; 3], size: [f64; 3]) -> Option<Body> {
    // NaN needs naming: it compares false against everything, so a size test
    // alone would let it through and put every corner at nowhere.
    if size
        .iter()
        .any(|extent| extent.is_nan() || *extent <= 0.0 || !extent.is_finite())
    {
        return None;
    }
    let mut body = Body::new();
    let base = Vec3::from(origin);
    let (dx, dy, dz) = (size[0], size[1], size[2]);

    // The eight corners, indexed by bit: 1 = +x, 2 = +y, 4 = +z. That
    // numbering is what makes the face and edge tables below readable.
    let corners: Vec<VertexKey> = (0..8)
        .map(|bits| {
            let point = base
                + Vec3::new(
                    if bits & 1 != 0 { dx } else { 0.0 },
                    if bits & 2 != 0 { dy } else { 0.0 },
                    if bits & 4 != 0 { dz } else { 0.0 },
                );
            body.vertices.insert(Vertex {
                point: point.to_array(),
                provenance: Provenance::Synthesized,
            })
        })
        .collect();

    // The twelve edges as corner pairs, each running from the lower index to
    // the higher so a shared edge is found rather than duplicated.
    const EDGES: [(usize, usize); 12] = [
        (0, 1), (2, 3), (4, 5), (6, 7), // along x
        (0, 2), (1, 3), (4, 6), (5, 7), // along y
        (0, 4), (1, 5), (2, 6), (3, 7), // along z
    ];
    let edges: Vec<EdgeKey> = EDGES
        .iter()
        .map(|(from, to)| {
            let start = body.vertices.get(corners[*from])?.point;
            let end = body.vertices.get(corners[*to])?.point;
            let direction = Vec3::from(end) - Vec3::from(start);
            let curve = body.curves.insert(Curve3::Line(Line3 {
                origin: start,
                direction: direction.to_array(),
            }));
            Some(body.edges.insert(Edge {
                curve,
                // The curve is parameterised by the edge's own span, so the
                // edge runs 0 to 1 along it.
                start_parameter: 0.0,
                end_parameter: 1.0,
                start: corners[*from],
                end: corners[*to],
                coedges: Vec::new(),
                provenance: Provenance::Synthesized,
            }))
        })
        .collect::<Option<Vec<_>>>()?;

    let lump = body.lumps.insert(Lump {
        shells: Vec::new(),
        provenance: Provenance::Synthesized,
    });
    let shell = body.shells.insert(Shell {
        faces: Vec::new(),
        owner: lump,
        provenance: Provenance::Synthesized,
    });

    // Each face as the four corners of its outer loop, listed
    // counter-clockwise seen from outside the box. That ordering is what
    // makes every edge come out traversed once each way.
    const FACES: [[usize; 4]; 6] = [
        [0, 2, 6, 4], // −x
        [1, 5, 7, 3], // +x
        [0, 4, 5, 1], // −y
        [2, 3, 7, 6], // +y
        [0, 1, 3, 2], // −z
        [4, 6, 7, 5], // +z
    ];
    for ring in FACES {
        let plane = face_plane(&body, &corners, ring)?;
        let surface = body.surfaces.insert(Surface::Plane(plane));
        let face = body.faces.insert(Face {
            surface,
            forward: true,
            loops: Vec::new(),
            owner: shell,
            provenance: Provenance::Synthesized,
        });
        let boundary = body.loops.insert(Loop {
            coedges: Vec::new(),
            owner: face,
            provenance: Provenance::Synthesized,
        });

        let mut coedges: Vec<CoedgeKey> = Vec::with_capacity(4);
        for index in 0..4 {
            let from = ring[index];
            let to = ring[(index + 1) % 4];
            let (position, forward) = find_edge(from, to)?;
            let edge = edges[position];
            let coedge = body.coedges.insert(Coedge {
                edge,
                forward,
                owner: boundary,
                provenance: Provenance::Synthesized,
            });
            body.edges.get_mut(edge)?.coedges.push(coedge);
            coedges.push(coedge);
        }
        body.loops.get_mut(boundary)?.coedges = coedges;
        body.faces.get_mut(face)?.loops = vec![boundary];
        body.shells.get_mut(shell)?.faces.push(face);
    }

    body.lumps.get_mut(lump)?.shells = vec![shell];
    body.roots = vec![lump];
    Some(body)
}

/// Which of the twelve edges joins two corners, and whether the given order
/// runs along it or against it.
fn find_edge(from: usize, to: usize) -> Option<(usize, bool)> {
    const EDGES: [(usize, usize); 12] = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    EDGES.iter().enumerate().find_map(|(index, (a, b))| {
        if (*a, *b) == (from, to) {
            Some((index, true))
        } else if (*a, *b) == (to, from) {
            Some((index, false))
        } else {
            None
        }
    })
}

/// The plane a face's four corners lie in, with its normal pointing out of
/// the box.
///
/// Built from the ring rather than from a table of axis directions, so the
/// two cannot disagree: the loop's own winding is what decides which way is
/// out.
fn face_plane(body: &Body, corners: &[Key<Vertex>], ring: [usize; 4]) -> Option<Plane> {
    let at = |index: usize| -> Option<Vec3> {
        Some(Vec3::from(body.vertices.get(corners[ring[index]])?.point))
    };
    let origin = at(0)?;
    let along = at(1)? - origin;
    let across = at(3)? - origin;
    // Counter-clockwise seen from outside, so `along × across` points in.
    let normal = across.cross(along).normalize()?;
    Plane::orthonormal(origin.to_array(), along.to_array(), normal.to_array())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brep::topology::Flaw;
    use std::collections::HashSet;

    fn unit_box() -> Body {
        cuboid([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]).expect("a unit box")
    }

    #[test]
    fn a_box_has_the_parts_a_box_has() {
        let body = unit_box();
        assert_eq!(body.vertices.len(), 8);
        assert_eq!(body.edges.len(), 12);
        assert_eq!(body.faces.len(), 6);
        assert_eq!(body.coedges.len(), 24, "four per face");
        assert_eq!(body.loops.len(), 6);
        assert_eq!(body.shells.len(), 1);
        assert_eq!(body.lumps.len(), 1);
    }

    #[test]
    fn a_box_is_a_closed_surface() {
        // V − E + F = 8 − 12 + 6 = 2, which is what a shell of genus zero
        // must give. A face left out or an edge duplicated changes it.
        assert_eq!(unit_box().euler_characteristic(), 2);
    }

    #[test]
    fn a_box_has_nothing_wrong_with_it() {
        let flaws = unit_box().validate();
        assert!(flaws.is_empty(), "{flaws:?}");
    }

    #[test]
    fn every_edge_is_shared_by_two_faces_running_it_opposite_ways() {
        // The property that makes it a solid. Getting one face's winding
        // backwards leaves that face's edges traversed the same way on both
        // sides, and the box is inside out along that seam.
        let body = unit_box();
        for (key, edge) in body.edges.iter() {
            assert_eq!(edge.coedges.len(), 2, "edge {key:?}");
            let senses: Vec<bool> = edge
                .coedges
                .iter()
                .map(|c| body.coedges.get(*c).unwrap().forward)
                .collect();
            assert_ne!(senses[0], senses[1], "edge {key:?} is used twice one way");
        }
    }

    #[test]
    fn every_loop_closes() {
        let body = unit_box();
        for (key, ring) in body.loops.iter() {
            let count = ring.coedges.len();
            assert_eq!(count, 4, "loop {key:?}");
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
    fn every_face_lies_on_its_own_surface() {
        let body = unit_box();
        for (key, face) in body.faces.iter() {
            let surface = body.surfaces.get(face.surface).unwrap();
            for coedge in body.face_coedges(key) {
                let (start, _) = body.coedge_vertices(coedge).unwrap();
                let point = body.vertices.get(start).unwrap().point;
                assert!(surface.contains(point, 1e-9), "{point:?} off face {key:?}");
            }
        }
    }

    #[test]
    fn every_face_normal_points_out_of_the_box() {
        // The centre is inside; a face's normal must lead away from it.
        let body = cuboid([0.0, 0.0, 0.0], [2.0, 4.0, 6.0]).unwrap();
        let centre = Vec3::new(1.0, 2.0, 3.0);
        for (_, face) in body.faces.iter() {
            let Surface::Plane(plane) = body.surfaces.get(face.surface).unwrap() else {
                panic!("a box is planar");
            };
            let normal = Vec3::from(plane.normal().unwrap());
            let outward = Vec3::from(plane.origin) - centre;
            assert!(normal.dot(outward) > 0.0, "a face pointed inwards");
        }
    }

    #[test]
    fn every_vertex_sits_where_its_edges_end() {
        assert!(unit_box().worst_vertex_gap() < 1e-12);
    }

    #[test]
    fn the_corners_are_where_the_size_puts_them() {
        let body = cuboid([10.0, 20.0, 30.0], [1.0, 2.0, 3.0]).unwrap();
        let points: HashSet<[u64; 3]> = body
            .vertices
            .iter()
            .map(|(_, v)| [v.point[0].to_bits(), v.point[1].to_bits(), v.point[2].to_bits()])
            .collect();
        assert_eq!(points.len(), 8, "no two corners coincide");
        for corner in [[10.0_f64, 20.0, 30.0], [11.0, 22.0, 33.0]] {
            let bits = [corner[0].to_bits(), corner[1].to_bits(), corner[2].to_bits()];
            assert!(points.contains(&bits), "{corner:?} missing");
        }
    }

    #[test]
    fn each_coedge_has_the_one_on_the_far_side() {
        let body = unit_box();
        for key in body.coedges.keys() {
            let partner = body.partner(key).expect("a closed box shares every edge");
            assert_ne!(partner, key);
            assert_eq!(body.partner(partner), Some(key));
        }
    }

    #[test]
    fn a_box_with_no_thickness_is_refused() {
        assert!(cuboid([0.0; 3], [1.0, 0.0, 1.0]).is_none());
        assert!(cuboid([0.0; 3], [1.0, -2.0, 1.0]).is_none());
        assert!(cuboid([0.0; 3], [1.0, f64::NAN, 1.0]).is_none());
    }

    #[test]
    fn a_box_at_survey_coordinates_is_still_a_box() {
        let body = cuboid([512_345.678, 4_512_345.678, 91.5], [0.5, 0.5, 0.5]).unwrap();
        assert!(body.validate().is_empty());
        assert_eq!(body.euler_characteristic(), 2);
        assert!(body.worst_vertex_gap() < 1e-9);
    }

    #[test]
    fn validation_notices_a_face_taken_away() {
        let mut body = unit_box();
        let victim = body.faces.keys().next().unwrap();
        body.faces.remove(victim);
        let flaws = body.validate();
        assert!(
            flaws.iter().any(|flaw| matches!(flaw, Flaw::DanglingKey(_))),
            "{flaws:?}"
        );
        assert_ne!(body.euler_characteristic(), 2);
    }

    #[test]
    fn validation_notices_a_loop_that_no_longer_closes() {
        let mut body = unit_box();
        let ring = body.loops.keys().next().unwrap();
        body.loops.get_mut(ring).unwrap().coedges.swap(0, 1);
        let flaws = body.validate();
        assert!(
            flaws.iter().any(|flaw| matches!(flaw, Flaw::OpenLoop(_))),
            "{flaws:?}"
        );
    }

    #[test]
    fn validation_notices_a_face_wound_the_wrong_way() {
        let mut body = unit_box();
        let ring = body.loops.keys().next().unwrap();
        let coedges = body.loops.get(ring).unwrap().coedges.clone();
        for key in coedges {
            let coedge = body.coedges.get_mut(key).unwrap();
            coedge.forward = !coedge.forward;
        }
        let flaws = body.validate();
        assert!(
            flaws.iter().any(|flaw| matches!(flaw, Flaw::SameSidedEdge(_))),
            "{flaws:?}"
        );
    }

    #[test]
    fn moving_a_vertex_dirties_what_names_it_and_nothing_else() {
        let mut body = unit_box();
        // Pretend it came from a file, so there is something to dirty.
        for node in body.vertices.values_mut() {
            node.provenance = Provenance::Clean(crate::brep::SourceRef::new(0));
        }
        for node in body.edges.values_mut() {
            node.provenance = Provenance::Clean(crate::brep::SourceRef::new(0));
        }
        for node in body.faces.values_mut() {
            node.provenance = Provenance::Clean(crate::brep::SourceRef::new(0));
        }
        let corner = body.vertices.keys().next().unwrap();
        body.soil_vertex(corner);

        // Three edges meet at a corner of a box, and three faces.
        let dirty_edges = body
            .edges
            .iter()
            .filter(|(_, e)| !e.provenance.is_reusable())
            .count();
        assert_eq!(dirty_edges, 3, "only the edges that end on it");
        let dirty_faces = body
            .faces
            .iter()
            .filter(|(_, f)| !f.provenance.is_reusable())
            .count();
        assert_eq!(dirty_faces, 3, "only the faces those bound");
    }
}
