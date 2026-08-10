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
use super::geometry::{Circle3, Cone, Curve3, Cylinder, Line3, Sphere, Surface, Torus};
use super::topology::{
    Body, Coedge, CoedgeKey, Edge, EdgeKey, Face, Loop, Lump, Shell, Vertex, VertexKey,
};
use std::f64::consts::{FRAC_PI_2, TAU};
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
                pcurve: None,
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

/// A right circular cylinder standing on `base`, `height` tall.
///
/// Three faces — the two discs and the wall — and the smallest topology that
/// describes them: two circular edges, one seam running between them, and two
/// vertices where the seam meets each rim.
///
/// The discs are bounded by a single coedge each, which is what a loop looks
/// like when its edge closes on itself. Requiring two would mean inventing a
/// split in a rim that has none.
///
/// `None` for a radius or height that is not positive.
pub fn cylinder(base: [f64; 3], radius: f64, height: f64) -> Option<Body> {
    if radius.is_nan() || height.is_nan() || radius <= 0.0 || height <= 0.0 {
        return None;
    }
    let mut body = Body::new();
    let bottom_plane = Plane::orthonormal(base, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])?;
    let top_origin = (Vec3::from(base) + Vec3::Z * height).to_array();
    let top_plane = Plane::orthonormal(top_origin, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])?;

    // The seam sits where the frame's own x axis meets each rim.
    let seam_low = bottom_plane.point_at([radius, 0.0]);
    let seam_high = top_plane.point_at([radius, 0.0]);
    let low = body.vertices.insert(Vertex {
        point: seam_low,
        provenance: Provenance::Synthesized,
    });
    let high = body.vertices.insert(Vertex {
        point: seam_high,
        provenance: Provenance::Synthesized,
    });

    let bottom_circle = body.curves.insert(Curve3::Circle(Circle3 {
        plane: bottom_plane,
        radius,
    }));
    let top_circle = body.curves.insert(Curve3::Circle(Circle3 {
        plane: top_plane,
        radius,
    }));
    let seam_line = body.curves.insert(Curve3::Line(Line3 {
        origin: seam_low,
        direction: (Vec3::from(seam_high) - Vec3::from(seam_low)).to_array(),
    }));

    let rim_low = body.edges.insert(Edge {
        curve: bottom_circle,
        start_parameter: 0.0,
        end_parameter: TAU,
        start: low,
        end: low,
        coedges: Vec::new(),
        provenance: Provenance::Synthesized,
    });
    let rim_high = body.edges.insert(Edge {
        curve: top_circle,
        start_parameter: 0.0,
        end_parameter: TAU,
        start: high,
        end: high,
        coedges: Vec::new(),
        provenance: Provenance::Synthesized,
    });
    let seam = body.edges.insert(Edge {
        curve: seam_line,
        start_parameter: 0.0,
        end_parameter: 1.0,
        start: low,
        end: high,
        coedges: Vec::new(),
        provenance: Provenance::Synthesized,
    });

    let lump = body.lumps.insert(Lump {
        shells: Vec::new(),
        provenance: Provenance::Synthesized,
    });
    let shell = body.shells.insert(Shell {
        faces: Vec::new(),
        owner: lump,
        provenance: Provenance::Synthesized,
    });

    // The bottom disc faces down, so its own plane — which faces up — is the
    // wrong way round and the face says so rather than a second plane being
    // made for it.
    let bottom = disc(&mut body, shell, bottom_plane, rim_low, false)?;
    let top = disc(&mut body, shell, top_plane, rim_high, true)?;
    let wall = body.surfaces.insert(Surface::Cylinder(Cylinder {
        base: bottom_plane,
        radius,
    }));
    let side = body.faces.insert(Face {
        surface: wall,
        forward: true,
        loops: Vec::new(),
        owner: shell,
        provenance: Provenance::Synthesized,
    });
    let ring = body.loops.insert(Loop {
        coedges: Vec::new(),
        owner: side,
        provenance: Provenance::Synthesized,
    });
    // Round the bottom rim, up the seam, back round the top rim, down again.
    let mut coedges = Vec::with_capacity(4);
    for (edge, forward) in [
        (rim_low, true),
        (seam, true),
        (rim_high, false),
        (seam, false),
    ] {
        let coedge = body.coedges.insert(Coedge {
            edge,
            forward,
            pcurve: None,
            owner: ring,
            provenance: Provenance::Synthesized,
        });
        body.edges.get_mut(edge)?.coedges.push(coedge);
        coedges.push(coedge);
    }
    body.loops.get_mut(ring)?.coedges = coedges;
    body.faces.get_mut(side)?.loops = vec![ring];
    body.shells.get_mut(shell)?.faces.push(side);

    let _ = (bottom, top);
    body.lumps.get_mut(lump)?.shells = vec![shell];
    body.roots = vec![lump];
    Some(body)
}

/// One end cap: a planar face bounded by the rim alone.
fn disc(
    body: &mut Body,
    shell: super::topology::ShellKey,
    plane: Plane,
    rim: EdgeKey,
    forward: bool,
) -> Option<super::topology::FaceKey> {
    let surface = body.surfaces.insert(Surface::Plane(plane));
    let face = body.faces.insert(Face {
        surface,
        forward,
        loops: Vec::new(),
        owner: shell,
        provenance: Provenance::Synthesized,
    });
    let ring = body.loops.insert(Loop {
        coedges: Vec::new(),
        owner: face,
        provenance: Provenance::Synthesized,
    });
    let coedge = body.coedges.insert(Coedge {
        edge: rim,
        // A cap's loop runs counter-clockwise seen from outside it, which for
        // the one facing down is clockwise seen from above — the rim
        // traversed backwards. So the sense follows the face's own, and the
        // wall takes the opposite of each: that is what makes a rim one
        // shared edge instead of two coincident ones.
        forward,
        pcurve: None,
        owner: ring,
        provenance: Provenance::Synthesized,
    });
    body.edges.get_mut(rim)?.coedges.push(coedge);
    body.loops.get_mut(ring)?.coedges = vec![coedge];
    body.faces.get_mut(face)?.loops = vec![ring];
    body.shells.get_mut(shell)?.faces.push(face);
    Some(face)
}

/// A sphere of `radius` about `centre`.
///
/// One face, as ACIS models it. A sphere has no edges of its own — it is
/// closed in both directions — so what bounds the face is a seam: a single
/// meridian, traversed once each way. The two poles are its vertices, and the
/// surface is singular there, which is a thing a B-rep says rather than
/// avoids.
///
/// `V − E + F = 2 − 1 + 1 = 2`, the same as any other closed shell.
pub fn sphere(centre: [f64; 3], radius: f64) -> Option<Body> {
    if radius.is_nan() || radius <= 0.0 {
        return None;
    }
    let mut body = Body::new();
    let frame = Plane::orthonormal(centre, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])?;
    // The meridian the seam runs along: a great circle in the plane holding
    // the pole axis and the frame's own x direction, so the seam sits where
    // the surface's `u` reads zero.
    let meridian = Plane::orthonormal(centre, [1.0, 0.0, 0.0], [0.0, -1.0, 0.0])?;
    let south = add_vertex(&mut body, meridian.point_at([0.0, -radius]));
    let north = add_vertex(&mut body, meridian.point_at([0.0, radius]));

    let curve = body.curves.insert(Curve3::Circle(Circle3 {
        plane: meridian,
        radius,
    }));
    let seam = body.edges.insert(Edge {
        curve,
        start_parameter: -FRAC_PI_2,
        end_parameter: FRAC_PI_2,
        start: south,
        end: north,
        coedges: Vec::new(),
        provenance: Provenance::Synthesized,
    });

    let (lump, shell) = add_shell(&mut body);
    let surface = body.surfaces.insert(Surface::Sphere(Sphere { frame, radius }));
    // Up the seam and back down it: the loop closes on itself because the
    // face wraps the whole way round in between.
    close_shell(&mut body, lump, shell, surface, true, &[(seam, true), (seam, false)])?;
    body.validate().is_empty().then_some(body)
}

/// A cone standing on `base`, `height` tall, with a base circle of `radius`
/// and a point at the top.
///
/// Two faces: the cone-surface wall and the disc it stands on. The wall's
/// loop runs round the rim, up the seam and back down it — the apex is where
/// the seam meets itself, a vertex the surface is singular at.
///
/// `None` for a size that is not positive. A truncated cone is a different
/// shape and is not this: pass a cylinder's radius twice, or build it from
/// the surface directly.
pub fn cone(base: [f64; 3], radius: f64, height: f64) -> Option<Body> {
    if radius.is_nan() || height.is_nan() || radius <= 0.0 || height <= 0.0 {
        return None;
    }
    let mut body = Body::new();
    let base_plane = Plane::orthonormal(base, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])?;
    let rim_point = base_plane.point_at([radius, 0.0]);
    let apex_point = (Vec3::from(base) + Vec3::Z * height).to_array();
    let rim_vertex = add_vertex(&mut body, rim_point);
    let apex = add_vertex(&mut body, apex_point);

    let rim_curve = body.curves.insert(Curve3::Circle(Circle3 {
        plane: base_plane,
        radius,
    }));
    let rim = body.edges.insert(Edge {
        curve: rim_curve,
        start_parameter: 0.0,
        end_parameter: TAU,
        start: rim_vertex,
        end: rim_vertex,
        coedges: Vec::new(),
        provenance: Provenance::Synthesized,
    });
    let seam = add_line_edge(&mut body, rim_vertex, apex)?;

    let (lump, shell) = add_shell(&mut body);
    // The base looks down, so its own plane — which looks up — is the wrong
    // way round and the face says so.
    disc(&mut body, shell, base_plane, rim, false)?;
    let wall = body.surfaces.insert(Surface::Cone(Cone {
        base: base_plane,
        radius,
        // The radius falls to nothing over the height, which is what sets
        // the slope.
        half_angle: (radius / height).atan(),
    }));
    close_shell(
        &mut body,
        lump,
        shell,
        wall,
        true,
        &[(rim, true), (seam, true), (seam, false)],
    )?;
    body.validate().is_empty().then_some(body)
}

/// A torus about `centre`, its tube `minor_radius` thick at
/// `major_radius` out.
///
/// One face and two seams — one round the ring, one round the tube — which is
/// what a surface closed in both directions needs. Its Euler characteristic
/// is zero rather than two, because a torus has a hole through it and no
/// closed shell of genus one can have any other.
///
/// `None` unless the tube fits: a minor radius at or past the major one
/// closes the hole and the surface crosses itself.
pub fn torus(centre: [f64; 3], major_radius: f64, minor_radius: f64) -> Option<Body> {
    if major_radius.is_nan()
        || minor_radius.is_nan()
        || minor_radius <= 0.0
        || major_radius <= minor_radius
    {
        return None;
    }
    let mut body = Body::new();
    let frame = Plane::orthonormal(centre, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])?;
    // Both seams pass through the outermost point of the ring, so the whole
    // face has one vertex.
    let outer = frame.point_at([major_radius + minor_radius, 0.0]);
    let corner = add_vertex(&mut body, outer);

    // Round the ring, at the tube's outer edge.
    let around = body.curves.insert(Curve3::Circle(Circle3 {
        plane: frame,
        radius: major_radius + minor_radius,
    }));
    let ring = body.edges.insert(Edge {
        curve: around,
        start_parameter: 0.0,
        end_parameter: TAU,
        start: corner,
        end: corner,
        coedges: Vec::new(),
        provenance: Provenance::Synthesized,
    });
    // Round the tube, in the plane holding the axis and the frame's x.
    let tube_plane = Plane::orthonormal(
        frame.point_at([major_radius, 0.0]),
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    )?;
    let across = body.curves.insert(Curve3::Circle(Circle3 {
        plane: tube_plane,
        radius: minor_radius,
    }));
    let tube = body.edges.insert(Edge {
        curve: across,
        start_parameter: 0.0,
        end_parameter: TAU,
        start: corner,
        end: corner,
        coedges: Vec::new(),
        provenance: Provenance::Synthesized,
    });

    let (lump, shell) = add_shell(&mut body);
    let surface = body.surfaces.insert(Surface::Torus(Torus {
        frame,
        major_radius,
        minor_radius,
    }));
    close_shell(
        &mut body,
        lump,
        shell,
        surface,
        true,
        &[(ring, true), (tube, true), (ring, false), (tube, false)],
    )?;
    body.validate().is_empty().then_some(body)
}

/// A right triangular prism: a right triangle in the XZ plane at `origin`,
/// `width` deep along Y.
pub fn wedge(origin: [f64; 3], length: f64, width: f64, height: f64) -> Option<Body> {
    if [length, width, height]
        .iter()
        .any(|extent| extent.is_nan() || *extent <= 0.0)
    {
        return None;
    }
    // The profile lies in the XZ plane, so the sweep runs along Y.
    let plane = Plane::orthonormal(origin, [1.0, 0.0, 0.0], [0.0, -1.0, 0.0])?;
    let triangle = [[0.0, 0.0], [length, 0.0], [0.0, height]];
    let profile: Vec<crate::geom2d::Curve> = (0..3)
        .map(|index| {
            crate::geom2d::Curve::Line(crate::geom2d::Line {
                start: triangle[index],
                end: triangle[(index + 1) % 3],
            })
        })
        .collect();
    super::sweep::extrude(plane, &profile, [0.0, width, 0.0])
}

/// A pyramid on a regular polygon of `sides` corners, `radius` out from the
/// middle of `base` and `height` tall.
///
/// Not a revolution — a polygon is not round — so it is built face by face:
/// the base, and one triangle per side meeting at the apex. `V − E + F` is
/// `(n + 1) − 2n + (n + 1) = 2` for any `n`, which is the check that the
/// rails and the base edges are each counted once.
///
/// `None` for fewer than three sides, or a size that is not positive.
pub fn pyramid(base: [f64; 3], radius: f64, height: f64, sides: usize) -> Option<Body> {
    if sides < 3 || radius.is_nan() || height.is_nan() || radius <= 0.0 || height <= 0.0 {
        return None;
    }
    let mut body = Body::new();
    let ground = Plane::orthonormal(base, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])?;
    let corners: Vec<VertexKey> = (0..sides)
        .map(|index| {
            let angle = TAU * index as f64 / sides as f64;
            add_vertex(
                &mut body,
                ground.point_at([radius * angle.cos(), radius * angle.sin()]),
            )
        })
        .collect();
    let apex = add_vertex(&mut body, (Vec3::from(base) + Vec3::Z * height).to_array());

    let rim: Vec<EdgeKey> = (0..sides)
        .map(|index| add_line_edge(&mut body, corners[index], corners[(index + 1) % sides]))
        .collect::<Option<Vec<_>>>()?;
    let rails: Vec<EdgeKey> = corners
        .iter()
        .map(|corner| add_line_edge(&mut body, *corner, apex))
        .collect::<Option<Vec<_>>>()?;

    let (lump, shell) = add_shell(&mut body);

    // The base looks down, so its loop runs the rim backwards — which is also
    // what leaves each rim edge traversed one way by the base and the other
    // by the side standing on it.
    let ground_down = Plane::orthonormal(base, [1.0, 0.0, 0.0], [0.0, 0.0, -1.0])?;
    let surface = body.surfaces.insert(Surface::Plane(ground_down));
    let floor: Vec<(EdgeKey, bool)> = rim.iter().rev().map(|edge| (*edge, false)).collect();
    close_shell(&mut body, lump, shell, surface, true, &floor)?;

    for index in 0..sides {
        let next = (index + 1) % sides;
        let here = Vec3::from(body.vertices.get(corners[index])?.point);
        let along = Vec3::from(body.vertices.get(corners[next])?.point) - here;
        let up = Vec3::from(body.vertices.get(apex)?.point) - here;
        let surface = body.surfaces.insert(Surface::Plane(Plane::orthonormal(
            here.to_array(),
            along.to_array(),
            along.cross(up).normalize()?.to_array(),
        )?));
        close_shell(
            &mut body,
            lump,
            shell,
            surface,
            true,
            &[(rim[index], true), (rails[next], true), (rails[index], false)],
        )?;
    }
    body.validate().is_empty().then_some(body)
}

/// A lump with one shell in it, ready for faces.
fn add_shell(body: &mut Body) -> (super::topology::LumpKey, super::topology::ShellKey) {
    let lump = body.lumps.insert(Lump {
        shells: Vec::new(),
        provenance: Provenance::Synthesized,
    });
    let shell = body.shells.insert(Shell {
        faces: Vec::new(),
        owner: lump,
        provenance: Provenance::Synthesized,
    });
    (lump, shell)
}

/// Adds one face from a list of edges and senses, then closes the body around
/// it.
fn close_shell(
    body: &mut Body,
    lump: super::topology::LumpKey,
    shell: super::topology::ShellKey,
    surface: super::topology::SurfaceKey,
    forward: bool,
    edges: &[(EdgeKey, bool)],
) -> Option<()> {
    let face = body.faces.insert(Face {
        surface,
        forward,
        loops: Vec::new(),
        owner: shell,
        provenance: Provenance::Synthesized,
    });
    let ring = body.loops.insert(Loop {
        coedges: Vec::new(),
        owner: face,
        provenance: Provenance::Synthesized,
    });
    let mut coedges = Vec::with_capacity(edges.len());
    for (edge, sense) in edges {
        let coedge = body.coedges.insert(Coedge {
            edge: *edge,
            forward: *sense,
            pcurve: None,
            owner: ring,
            provenance: Provenance::Synthesized,
        });
        body.edges.get_mut(*edge)?.coedges.push(coedge);
        coedges.push(coedge);
    }
    body.loops.get_mut(ring)?.coedges = coedges;
    body.faces.get_mut(face)?.loops = vec![ring];
    body.shells.get_mut(shell)?.faces.push(face);
    if body.lumps.get(lump)?.shells.is_empty() {
        body.lumps.get_mut(lump)?.shells = vec![shell];
    }
    if body.roots.is_empty() {
        body.roots = vec![lump];
    }
    Some(())
}

fn add_vertex(body: &mut Body, point: [f64; 3]) -> VertexKey {
    body.vertices.insert(Vertex {
        point,
        provenance: Provenance::Synthesized,
    })
}

fn add_line_edge(body: &mut Body, from: VertexKey, to: VertexKey) -> Option<EdgeKey> {
    let start = body.vertices.get(from)?.point;
    let end = body.vertices.get(to)?.point;
    let curve = body.curves.insert(Curve3::Line(Line3 {
        origin: start,
        direction: (Vec3::from(end) - Vec3::from(start)).to_array(),
    }));
    Some(body.edges.insert(Edge {
        curve,
        start_parameter: 0.0,
        end_parameter: 1.0,
        start: from,
        end: to,
        coedges: Vec::new(),
        provenance: Provenance::Synthesized,
    }))
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
    fn a_cylinder_has_the_parts_a_cylinder_has() {
        let solid = cylinder([0.0; 3], 5.0, 10.0).expect("a cylinder");
        assert_eq!(solid.faces.len(), 3, "two caps and a wall");
        assert_eq!(solid.edges.len(), 3, "two rims and a seam");
        assert_eq!(solid.vertices.len(), 2, "where the seam meets each rim");
        let flaws = solid.validate();
        assert!(flaws.is_empty(), "{flaws:?}");
        assert_eq!(solid.euler_characteristic(), 2);
    }

    #[test]
    fn a_cylinders_rims_are_each_shared_by_two_faces() {
        let solid = cylinder([0.0; 3], 3.0, 4.0).unwrap();
        for (key, edge) in solid.edges.iter() {
            assert_eq!(edge.coedges.len(), 2, "edge {key:?}");
            let senses: Vec<bool> = edge
                .coedges
                .iter()
                .map(|c| solid.coedges.get(*c).unwrap().forward)
                .collect();
            assert_ne!(senses[0], senses[1], "edge {key:?} runs one way twice");
        }
    }

    #[test]
    fn a_sphere_is_one_face_bounded_by_a_seam() {
        // ACIS's own shape for it: no edges of its own, so a meridian
        // traversed once each way is what bounds the face, and the poles are
        // vertices the surface is singular at.
        let solid = sphere([0.0; 3], 5.0).expect("a sphere");
        assert_eq!(solid.faces.len(), 1);
        assert_eq!(solid.edges.len(), 1, "one seam");
        assert_eq!(solid.vertices.len(), 2, "two poles");
        assert_eq!(solid.euler_characteristic(), 2);
        let flaws = solid.validate();
        assert!(flaws.is_empty(), "{flaws:?}");
    }

    #[test]
    fn a_spheres_poles_are_a_radius_apart_along_its_axis() {
        let solid = sphere([1.0, 2.0, 3.0], 4.0).unwrap();
        let mut heights: Vec<f64> = solid.vertices.iter().map(|(_, v)| v.point[2]).collect();
        heights.sort_by(f64::total_cmp);
        assert!((heights[0] - (-1.0)).abs() < 1e-9, "{heights:?}");
        assert!((heights[1] - 7.0).abs() < 1e-9, "{heights:?}");
    }

    #[test]
    fn a_spheres_seam_lies_on_the_sphere() {
        let solid = sphere([0.0; 3], 5.0).unwrap();
        let (_, face) = solid.faces.iter().next().unwrap();
        let surface = solid.surfaces.get(face.surface).unwrap();
        let (key, edge) = solid.edges.iter().next().unwrap();
        let curve = solid.curves.get(edge.curve).unwrap();
        for step in 0..=8 {
            let t = edge.start_parameter
                + (edge.end_parameter - edge.start_parameter) * step as f64 / 8.0;
            let point = curve.point_at(t);
            assert!(surface.contains(point, 1e-9), "{point:?} off the sphere");
        }
        let _ = key;
    }

    #[test]
    fn a_cone_is_a_wall_and_a_base_meeting_at_a_point() {
        let solid = cone([0.0; 3], 5.0, 12.0).expect("a cone");
        assert_eq!(solid.faces.len(), 2, "the wall and the disc");
        assert_eq!(solid.edges.len(), 2, "the rim and the seam");
        assert_eq!(solid.vertices.len(), 2, "the seam's foot and the apex");
        assert_eq!(solid.euler_characteristic(), 2);
        assert!(solid.validate().is_empty());
    }

    #[test]
    fn a_cones_surface_reaches_its_own_apex() {
        let solid = cone([0.0; 3], 5.0, 12.0).unwrap();
        let apex = solid
            .vertices
            .iter()
            .map(|(_, v)| v.point)
            .find(|p| p[2] > 1.0)
            .expect("an apex above the base");
        assert!((apex[2] - 12.0).abs() < 1e-9, "{apex:?}");
        // The wall's surface holds it: the radius really does fall to nothing
        // over the height, which the half-angle is what sets.
        let wall = solid
            .surfaces
            .iter()
            .find_map(|(_, s)| matches!(s, Surface::Cone(_)).then_some(s))
            .unwrap();
        assert!(wall.contains(apex, 1e-9), "the apex is off its own cone");
    }

    #[test]
    fn a_torus_has_a_hole_through_it() {
        // Euler zero rather than two: no closed shell of genus one can have
        // any other, so this is the check that the two seams are right.
        let solid = torus([0.0; 3], 10.0, 2.0).expect("a torus");
        assert_eq!(solid.faces.len(), 1);
        assert_eq!(solid.edges.len(), 2, "one seam each way");
        assert_eq!(solid.vertices.len(), 1, "where they cross");
        assert_eq!(solid.euler_characteristic(), 0);
        let flaws = solid.validate();
        assert!(flaws.is_empty(), "{flaws:?}");
    }

    #[test]
    fn a_torus_that_would_close_its_own_hole_is_refused() {
        assert!(torus([0.0; 3], 2.0, 2.0).is_none());
        assert!(torus([0.0; 3], 1.0, 2.0).is_none());
        assert!(torus([0.0; 3], 10.0, 0.0).is_none());
    }

    #[test]
    fn both_of_a_toruss_seams_lie_on_it() {
        let solid = torus([0.0; 3], 10.0, 2.0).unwrap();
        let (_, face) = solid.faces.iter().next().unwrap();
        let surface = solid.surfaces.get(face.surface).unwrap();
        for (_, edge) in solid.edges.iter() {
            let curve = solid.curves.get(edge.curve).unwrap();
            for step in 0..8 {
                let point = curve.point_at(TAU * step as f64 / 8.0);
                assert!(surface.contains(point, 1e-9), "{point:?} off the torus");
            }
        }
    }

    #[test]
    fn a_wedge_is_a_prism_on_a_right_triangle() {
        let solid = wedge([0.0; 3], 4.0, 3.0, 5.0).expect("a wedge");
        assert_eq!(solid.faces.len(), 5, "two triangles and three walls");
        assert_eq!(solid.edges.len(), 9);
        assert_eq!(solid.vertices.len(), 6);
        assert_eq!(solid.euler_characteristic(), 2);
        assert!(solid.validate().is_empty());
        let bounds = crate::brep::body_bounds(&solid).unwrap();
        assert_eq!(bounds.max, [4.0, 3.0, 5.0]);
    }

    /// How much a meshed solid encloses, by the divergence theorem.
    ///
    /// Negative when the faces are wound inwards, and short of the true
    /// figure by however much the flat facets cut off — never over, since
    /// every chord lies inside the surface it approximates.
    fn meshed_volume(solid: &Body) -> f64 {
        let mesh = crate::brep::mesh::body(solid, crate::tessellation::DEFAULT_ANGLE, 1e-9);
        assert!(!mesh.is_empty(), "nothing meshed");
        mesh.triangles
            .iter()
            .map(|triangle| {
                let at = |index: usize| Vec3::from(mesh.positions[triangle[index]]);
                at(0).cross(at(1)).dot(at(2)) / 6.0
            })
            .sum()
    }

    #[test]
    fn every_primitive_that_meshes_encloses_what_it_should() {
        // Measuring the whole mesh rather than looking at each triangle. A
        // face left out entirely passes any per-triangle check — the ones
        // that are there are still wound correctly — and for a long while a
        // cylinder's wall was missing from every mesh this way, with only
        // its two discs drawn.
        //
        // A volume notices both faults at once: a missing face reads far too
        // small, and an inside-out one reads negative.
        let cases: [(Body, f64); 7] = [
            (cuboid([0.0; 3], [4.0; 3]).unwrap(), 64.0),
            (wedge([0.0; 3], 4.0, 3.0, 5.0).unwrap(), 0.5 * 4.0 * 5.0 * 3.0),
            (
                cylinder([0.0; 3], 3.0, 6.0).unwrap(),
                std::f64::consts::PI * 9.0 * 6.0,
            ),
            (
                cone([0.0; 3], 5.0, 12.0).unwrap(),
                std::f64::consts::PI * 25.0 * 12.0 / 3.0,
            ),
            (
                torus([0.0; 3], 10.0, 2.0).unwrap(),
                2.0 * std::f64::consts::PI.powi(2) * 10.0 * 4.0,
            ),
            (
                sphere([0.0; 3], 5.0).unwrap(),
                4.0 / 3.0 * std::f64::consts::PI * 125.0,
            ),
            (
                pyramid([0.0; 3], 4.0, 9.0, 6).unwrap(),
                // A regular hexagon of circumradius four: six equilateral
                // triangles, each with area r²√3/4 at side r.
                6.0 * (16.0 * 3.0_f64.sqrt() / 4.0) * 9.0 / 3.0,
            ),
        ];
        for (solid, expected) in cases {
            let volume = meshed_volume(&solid);
            assert!(volume > 0.0, "wound inwards: {volume}");
            assert!(volume <= expected * 1.000_001, "reads over: {volume}");
            assert!(volume > 0.98 * expected, "{volume} vs {expected}");
        }
    }

    #[test]
    fn the_primitives_that_mesh_do_so_with_their_normals_out() {
        // The check that each one's senses hang together, for the shapes
        // convex enough that every normal must lead away from one point
        // inside: a face wound the wrong way lights the solid inside out.
        let solids = [
            (cuboid([0.0; 3], [4.0; 3]).unwrap(), Vec3::new(2.0, 2.0, 2.0)),
            (cylinder([0.0; 3], 3.0, 6.0).unwrap(), Vec3::new(0.0, 0.0, 3.0)),
            (cone([0.0; 3], 5.0, 12.0).unwrap(), Vec3::new(0.0, 0.0, 1.0)),
            (wedge([0.0; 3], 4.0, 3.0, 5.0).unwrap(), Vec3::new(1.0, 1.5, 1.0)),
        ];
        for (solid, inside) in solids {
            let mesh = crate::brep::mesh::body(
                &solid,
                crate::tessellation::DEFAULT_ANGLE,
                1e-9,
            );
            assert!(!mesh.is_empty(), "nothing meshed");
            for triangle in &mesh.triangles {
                let corner = Vec3::from(mesh.positions[triangle[0]]);
                let normal = Vec3::from(mesh.normals[triangle[0]]);
                assert!(
                    normal.dot(corner - inside) > 0.0,
                    "a face pointed inwards at {corner:?}"
                );
            }
        }
    }

    #[test]
    fn a_pyramid_is_a_base_and_a_triangle_per_side() {
        for sides in [3, 4, 5, 8] {
            let solid = pyramid([0.0; 3], 4.0, 9.0, sides).expect("a pyramid");
            assert_eq!(solid.faces.len(), sides + 1, "{sides}");
            assert_eq!(solid.edges.len(), 2 * sides, "the rim and one rail each");
            assert_eq!(solid.vertices.len(), sides + 1);
            assert_eq!(solid.euler_characteristic(), 2, "{sides}");
            assert!(solid.validate().is_empty(), "{sides}");
        }
        assert!(pyramid([0.0; 3], 4.0, 9.0, 2).is_none(), "a sliver is not a solid");
        assert!(pyramid([0.0; 3], 0.0, 9.0, 4).is_none());
        assert!(pyramid([0.0; 3], 4.0, -1.0, 4).is_none());
    }

    #[test]
    fn a_pyramid_encloses_a_third_of_the_prism_around_it() {
        // Base area times height over three, whatever the base is. A square
        // one is the case worth naming, since its base area is exact.
        let solid = pyramid([0.0; 3], 4.0, 9.0, 4).unwrap();
        // A square inscribed in a circle of radius four has diagonal eight.
        let expected = (0.5 * 8.0 * 8.0) * 9.0 / 3.0;
        let volume = meshed_volume(&solid);
        assert!((volume - expected).abs() < 1e-9 * expected, "{volume} vs {expected}");
    }

    #[test]
    fn a_sphere_meshes_from_what_bounds_it_rather_than_where() {
        // Its seam ends at the poles, where longitude has no single value —
        // every meridian passes through them — so where that seam sits in
        // (u, v) cannot be worked out from the geometry at all. What can be
        // worked out is that the seam is the *only* thing bounding the face,
        // and a face bounded by nothing but its own seams covers the whole
        // of its surface. That is enough, and it needs no pcurve.
        let solid = sphere([0.0; 3], 5.0).unwrap();
        let volume = meshed_volume(&solid);
        let expected = 4.0 / 3.0 * std::f64::consts::PI * 125.0;
        assert!(volume > 0.0, "wound inwards: {volume}");
        assert!(volume < expected, "a tessellation cannot read over: {volume}");
        assert!(volume > 0.98 * expected, "{volume} vs {expected}");
    }

    #[test]
    fn a_sphere_with_no_size_is_refused() {
        assert!(sphere([0.0; 3], 0.0).is_none());
        assert!(sphere([0.0; 3], -1.0).is_none());
        assert!(cone([0.0; 3], 1.0, 0.0).is_none());
        assert!(wedge([0.0; 3], 1.0, 0.0, 1.0).is_none());
    }

    #[test]
    fn a_cylinder_with_no_size_is_refused() {
        assert!(cylinder([0.0; 3], 0.0, 1.0).is_none());
        assert!(cylinder([0.0; 3], 1.0, -1.0).is_none());
        assert!(cylinder([0.0; 3], f64::NAN, 1.0).is_none());
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
