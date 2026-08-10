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

use super::topology::{Body, EdgeKey, FaceKey};
use crate::geom2d::triangulate;
use crate::space::Vec3;
use std::collections::HashMap;
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

/// One tolerance policy for a body's faces and edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TessellationTolerance {
    pub chord: f64,
    pub linear: f64,
    relative: bool,
    isolines: usize,
}

impl TessellationTolerance {
    pub fn new(chord: f64, linear: f64) -> Self {
        Self {
            chord: finite_positive(chord, DEFAULT_SAG),
            linear: finite_positive(linear, 1e-9),
            relative: false,
            isolines: 0,
        }
    }

    pub fn relative(chord_fraction: f64, linear: f64) -> Self {
        Self {
            chord: finite_positive(chord_fraction, 0.002),
            linear: finite_positive(linear, 1e-9),
            relative: true,
            isolines: 0,
        }
    }

    pub fn with_isolines(mut self, count: usize) -> Self {
        self.isolines = count;
        self
    }

    fn resolve(self, scale: f64) -> f64 {
        if self.relative {
            finite_positive(scale, 1.0) * self.chord
        } else {
            self.chord
        }
    }
}

/// Tessellation of one topological edge.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeMesh {
    pub edge: EdgeKey,
    pub parameters: Vec<f64>,
    pub positions: Vec<[f64; 3]>,
}

/// All display geometry derived from one body under one tolerance policy.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BodyMesh {
    pub mesh: Mesh,
    pub triangle_faces: Vec<FaceKey>,
    pub edges: Vec<EdgeMesh>,
    pub isolines: Vec<FacePolyline>,
    pub chord: f64,
    pub missing_faces: Vec<FaceKey>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FacePolyline {
    pub face: FaceKey,
    pub positions: Vec<[f64; 3]>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SurfaceMesh {
    pub mesh: Mesh,
    pub edges: Vec<Vec<[f64; 3]>>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SilhouetteSource {
    sides: Vec<SilhouetteSide>,
    chord: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct SilhouetteSide {
    positions: [[f64; 3]; 2],
    normals: [[f64; 3]; 2],
}

impl BodyMesh {
    pub fn silhouette_source(&self) -> SilhouetteSource {
        let mut groups = HashMap::new();
        let mut next = 0_u64;
        let triangle_groups = self
            .triangle_faces
            .iter()
            .map(|face| {
                *groups.entry(*face).or_insert_with(|| {
                    let group = next;
                    next += 1;
                    group
                })
            })
            .collect::<Vec<_>>();
        silhouette_source(&self.mesh, &triangle_groups, self.chord)
    }
}

impl SurfaceMesh {
    pub fn silhouette_source(&self, chord: f64) -> SilhouetteSource {
        silhouette_source(&self.mesh, &vec![0; self.mesh.triangles.len()], chord)
    }
}

/// View-dependent smooth-face lines from the same triangles as the surface.
pub fn silhouette(source: &SilhouetteSource, view_direction: [f64; 3]) -> Vec<[f64; 3]> {
    let Some(view) = Vec3::from(view_direction).normalize() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for side in &source.sides {
        let signs = side.normals.map(|normal| Vec3::from(normal).dot(view).signum());
        if signs[0] != signs[1] {
            out.extend(side.positions);
        }
    }
    out
}

fn silhouette_source(mesh: &Mesh, triangle_groups: &[u64], chord: f64) -> SilhouetteSource {
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    struct Point([i64; 3]);
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    struct Side {
        group: u64,
        a: Point,
        b: Point,
    }
    let precision = chord.max(1e-9) * 1e-6;
    let origin = mesh.positions.first().copied().unwrap_or([0.0; 3]);
    let point_key = |point: [f64; 3]| {
        Point([0, 1, 2].map(|axis| ((point[axis] - origin[axis]) / precision).round() as i64))
    };
    let mut open: HashMap<Side, ([f64; 3], [[f64; 3]; 2])> = HashMap::new();
    let mut sides = Vec::new();
    for (index, triangle) in mesh.triangles.iter().enumerate() {
        let Some(group) = triangle_groups.get(index).copied() else {
            continue;
        };
        let positions = triangle.map(|vertex| mesh.positions[vertex]);
        let Some(normal) = (Vec3::from(positions[1]) - Vec3::from(positions[0]))
            .cross(Vec3::from(positions[2]) - Vec3::from(positions[0]))
            .normalize()
        else {
            continue;
        };
        for [from, to] in [[0, 1], [1, 2], [2, 0]] {
            let mut a = point_key(positions[from]);
            let mut b = point_key(positions[to]);
            let mut segment = [positions[from], positions[to]];
            if a.0 > b.0 {
                std::mem::swap(&mut a, &mut b);
                segment.swap(0, 1);
            }
            let key = Side { group, a, b };
            if let Some((other, previous)) = open.remove(&key) {
                sides.push(SilhouetteSide {
                    positions: previous,
                    normals: [other, normal.to_array()],
                });
            } else {
                open.insert(key, (normal.to_array(), segment));
            }
        }
    }
    SilhouetteSource { sides, chord }
}

pub fn transform_silhouette(
    source: &SilhouetteSource,
    placement: &super::place::Placement,
) -> Option<SilhouetteSource> {
    transform_silhouette_affine(
        source,
        [placement.x_axis, placement.y_axis, placement.z_axis],
        placement.origin,
    )
}

/// Moves a silhouette source by an affine map whose vectors are columns.
pub fn transform_silhouette_affine(
    source: &SilhouetteSource,
    vectors: [[f64; 3]; 3],
    origin: [f64; 3],
) -> Option<SilhouetteSource> {
    let vectors = vectors.map(Vec3::from);
    if origin.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let lengths = vectors.map(Vec3::length);
    if lengths.iter().any(|length| !length.is_finite() || *length <= 0.0) {
        return None;
    }
    let unit = [0, 1, 2].map(|axis| vectors[axis] / lengths[axis]);
    let determinant = unit[0].dot(unit[1].cross(unit[2]));
    if !determinant.is_finite() || determinant.abs() <= 1e-12 {
        return None;
    }
    let normal_vectors = [
        unit[1].cross(unit[2]) / (determinant * lengths[0]),
        unit[2].cross(unit[0]) / (determinant * lengths[1]),
        unit[0].cross(unit[1]) / (determinant * lengths[2]),
    ];
    let stretch = vectors
        .iter()
        .map(|vector| vector.dot(*vector))
        .sum::<f64>()
        .sqrt();
    let mut out = source.clone();
    for side in &mut out.sides {
        for point in &mut side.positions {
            let moved = Vec3::from(origin)
                + vectors[0] * point[0]
                + vectors[1] * point[1]
                + vectors[2] * point[2];
            *point = moved.to_array();
        }
        for normal in &mut side.normals {
            let moved = normal_vectors[0] * normal[0]
                + normal_vectors[1] * normal[1]
                + normal_vectors[2] * normal[2];
            *normal = moved.normalize()?.to_array();
        }
    }
    out.chord *= stretch;
    Some(out)
}

impl BodyMesh {
    pub fn is_complete(&self) -> bool {
        self.missing_faces.is_empty() && !self.mesh.is_empty()
    }
}

pub fn sweep_surface(
    profile: &crate::space::PlanarCurve,
    path: &crate::space::PlanarCurve,
    tolerance: f64,
) -> Option<SurfaceMesh> {
    if path.curve.is_closed() {
        return None;
    }
    let closed = profile.curve.is_closed();
    let outline = open_samples(curve_samples(profile, tolerance)?, closed);
    let track = open_samples(curve_samples(path, tolerance)?, false);
    if outline.len() < 2 || track.len() < 2 {
        return None;
    }
    let origin = Vec3::from(track[0]);
    let sections: Vec<Vec<[f64; 3]>> = track
        .iter()
        .map(|point| {
            let shift = Vec3::from(*point) - origin;
            outline
                .iter()
                .map(|position| (Vec3::from(*position) + shift).to_array())
                .collect()
        })
        .collect();
    surface_from_sections(&sections, closed, Some(&profile.plane))
}

pub fn loft_surface(
    profiles: &[crate::space::PlanarCurve],
    tolerance: f64,
) -> Option<SurfaceMesh> {
    if profiles.len() < 2 {
        return None;
    }
    let closed = profiles.iter().all(|profile| profile.curve.is_closed());
    let sampled: Vec<Vec<[f64; 3]>> = profiles
        .iter()
        .map(|profile| {
            Some(open_samples(
                curve_samples(profile, tolerance)?,
                profile.curve.is_closed(),
            ))
        })
        .collect::<Option<_>>()?;
    let count = sampled.iter().map(Vec::len).max()?;
    if count < 2 || sampled.iter().any(|ring| ring.len() < 2) {
        return None;
    }
    let sections: Vec<Vec<[f64; 3]>> = sampled
        .iter()
        .map(|ring| resample_ring(ring, count, closed))
        .collect();
    let mut surface = surface_from_sections(&sections, closed, None)?;
    if closed {
        cap_ring(
            &mut surface.mesh,
            &profiles.first()?.plane,
            sections.first()?,
            true,
        );
        cap_ring(
            &mut surface.mesh,
            &profiles.last()?.plane,
            sections.last()?,
            false,
        );
    }
    Some(surface)
}

fn surface_from_sections(
    sections: &[Vec<[f64; 3]>],
    closed: bool,
    cap_plane: Option<&crate::space::Plane>,
) -> Option<SurfaceMesh> {
    let mut out = SurfaceMesh::default();
    for pair in sections.windows(2) {
        band(&mut out.mesh, &pair[0], &pair[1], closed);
    }
    let mut first_section = sections.first()?.clone();
    let mut last_section = sections.last()?.clone();
    if closed {
        first_section.push(first_section[0]);
        last_section.push(last_section[0]);
    }
    out.edges.extend([first_section, last_section]);
    if closed {
        if let Some(plane) = cap_plane {
            cap_ring(&mut out.mesh, plane, sections.first()?, true);
            let shift = Vec3::from(sections.last()?.first().copied()?)
                - Vec3::from(sections.first()?.first().copied()?);
            let far = crate::space::Plane::from_axes(
                (Vec3::from(plane.origin) + shift).to_array(),
                plane.x_axis,
                plane.y_axis,
            );
            cap_ring(&mut out.mesh, &far, sections.last()?, false);
        }
    } else {
        let first: Vec<[f64; 3]> = sections.iter().filter_map(|ring| ring.first().copied()).collect();
        let last: Vec<[f64; 3]> = sections.iter().filter_map(|ring| ring.last().copied()).collect();
        out.edges.extend([first, last]);
    }
    out.edges.retain(|edge| edge.len() >= 2);
    Some(out)
}

fn open_samples(mut points: Vec<[f64; 3]>, closed: bool) -> Vec<[f64; 3]> {
    if closed
        && points.len() > 1
        && distance3(points[0], points[points.len() - 1]) <= 1e-9
    {
        points.pop();
    }
    points
}

fn curve_samples(
    curve: &crate::space::PlanarCurve,
    tolerance: f64,
) -> Option<Vec<[f64; 3]>> {
    let points = curve.tessellate_within(tolerance);
    for pair in points.windows(2) {
        let from = curve.parameter_at(pair[0])?;
        let mut to = curve.parameter_at(pair[1])?;
        if curve.curve.is_closed() && to < from {
            to += 1.0;
        }
        let start = Vec3::from(pair[0]);
        let end = Vec3::from(pair[1]);
        let chord = end - start;
        let square = chord.length_squared();
        for step in 1..8 {
            let parameter = from + (to - from) * step as f64 / 8.0;
            let parameter = if parameter > 1.0 { parameter - 1.0 } else { parameter };
            let point = Vec3::from(curve.point_at(parameter));
            let unit = if square > 0.0 {
                (point - start).dot(chord) / square
            } else {
                0.0
            }
            .clamp(0.0, 1.0);
            if point.distance(start.lerp(end, unit)) > tolerance {
                return None;
            }
        }
    }
    Some(points)
}

fn resample_ring(points: &[[f64; 3]], count: usize, closed: bool) -> Vec<[f64; 3]> {
    let mut chain: Vec<Vec3> = points.iter().copied().map(Vec3::from).collect();
    if closed {
        chain.push(chain[0]);
    }
    let mut walked = vec![0.0];
    for pair in chain.windows(2) {
        walked.push(walked[walked.len() - 1] + pair[0].distance(pair[1]));
    }
    let total = *walked.last().unwrap_or(&0.0);
    if total <= 0.0 {
        return points.to_vec();
    }
    let divisor = if closed { count } else { count.saturating_sub(1).max(1) };
    (0..count)
        .map(|step| {
            let wanted = total * step as f64 / divisor as f64;
            let at = walked
                .iter()
                .rposition(|reached| *reached <= wanted)
                .unwrap_or(0)
                .min(chain.len() - 2);
            let span = walked[at + 1] - walked[at];
            let unit = if span > 0.0 {
                (wanted - walked[at]) / span
            } else {
                0.0
            };
            chain[at].lerp(chain[at + 1], unit).to_array()
        })
        .collect()
}

fn band(mesh: &mut Mesh, lower: &[[f64; 3]], upper: &[[f64; 3]], closed: bool) {
    let count = lower.len().min(upper.len());
    let spans = if closed { count } else { count.saturating_sub(1) };
    for index in 0..spans {
        let next = (index + 1) % count;
        emit_points(mesh, lower[index], lower[next], upper[next]);
        emit_points(mesh, lower[index], upper[next], upper[index]);
    }
}

fn cap_ring(mesh: &mut Mesh, plane: &crate::space::Plane, ring: &[[f64; 3]], reverse: bool) {
    let parameters: Vec<[f64; 2]> = ring.iter().filter_map(|point| plane.project(*point)).collect();
    if parameters.len() != ring.len() {
        return;
    }
    let (points, triangles) = triangulate(&parameters, &[]);
    for triangle in triangles {
        let mut positions = [
            plane.point_at(points[triangle[0]]),
            plane.point_at(points[triangle[1]]),
            plane.point_at(points[triangle[2]]),
        ];
        if reverse {
            positions.swap(1, 2);
        }
        emit_points(mesh, positions[0], positions[1], positions[2]);
    }
}

fn emit_points(mesh: &mut Mesh, a: [f64; 3], b: [f64; 3], c: [f64; 3]) {
    let Some(normal) = (Vec3::from(b) - Vec3::from(a))
        .cross(Vec3::from(c) - Vec3::from(a))
        .normalize()
    else {
        return;
    };
    let base = mesh.positions.len();
    mesh.positions.extend([a, b, c]);
    mesh.normals.extend([normal.to_array(); 3]);
    mesh.triangles.push([base, base + 1, base + 2]);
}

/// Tessellates faces and edges from one shared sample schedule.
pub fn tessellate(body: &Body, tolerance: TessellationTolerance) -> BodyMesh {
    let sag = tolerance.resolve(body_span(body));
    let schedules: HashMap<EdgeKey, Vec<super::place::EdgeSample>> = body
        .edge_keys()
        .filter_map(|edge| Some((edge, shared_edge_samples(body, edge, sag, tolerance.linear)?)))
        .collect();
    let mut out = BodyMesh::default();
    out.chord = sag;
    for face_key in body.face_keys() {
        match scheduled_face(body, face_key, sag, tolerance.linear, &schedules) {
            Some(mesh) => {
                out.triangle_faces
                    .extend(std::iter::repeat(face_key).take(mesh.triangles.len()));
                out.mesh.absorb(mesh);
                if tolerance.isolines > 0 {
                    match face_isolines(
                        body,
                        face_key,
                        sag,
                        tolerance.linear,
                        tolerance.isolines,
                        &schedules,
                    ) {
                        Some(lines) => out.isolines.extend(lines),
                        None => out.missing_faces.push(face_key),
                    }
                }
            }
            None => out.missing_faces.push(face_key),
        }
    }
    for edge_key in body.edge_keys() {
        if let Some(schedule) = schedules.get(&edge_key) {
            if schedule.len() >= 2 {
                out.edges.push(EdgeMesh {
                    edge: edge_key,
                    parameters: schedule.iter().map(|sample| sample.parameter).collect(),
                    positions: schedule.iter().map(|sample| sample.position).collect(),
                });
            }
        }
    }
    out.missing_faces.dedup();
    out
}

fn face_isolines(
    body: &Body,
    face: FaceKey,
    sag: f64,
    tolerance: f64,
    count: usize,
    schedules: &HashMap<EdgeKey, Vec<super::place::EdgeSample>>,
) -> Option<Vec<FacePolyline>> {
    let Some(node) = body.faces.get(face) else {
        return None;
    };
    let Some(surface) = body.surfaces.get(node.surface) else {
        return None;
    };
    if matches!(surface, super::geometry::Surface::Plane(_)) {
        return Some(Vec::new());
    }
    let rings = face_rings(body, face, surface, schedules, tolerance)?;
    let parameters: Vec<Vec<[f64; 2]>> = rings
        .iter()
        .map(|ring| ring.iter().map(|point| point.parameters).collect())
        .collect();
    let Some(bounds) = parameter_bounds(&parameters) else {
        return None;
    };
    let mut out = Vec::new();
    for fixed_axis in 0..2 {
        let varying_axis = 1 - fixed_axis;
        let fixed_span = bounds[fixed_axis][1] - bounds[fixed_axis][0];
        if fixed_span <= 0.0 {
            continue;
        }
        for index in 0..count {
            let fixed = bounds[fixed_axis][0]
                + fixed_span * (index as f64 + 1.0) / (count as f64 + 1.0);
            for interval in line_intervals(&parameters, fixed_axis, fixed) {
                let mut points = Vec::new();
                if !sample_isoline(
                    surface,
                    fixed_axis,
                    fixed,
                    interval[0],
                    interval[1],
                    sag,
                    0,
                    &mut points,
                ) {
                    return None;
                }
                let mut end = [0.0; 2];
                end[fixed_axis] = fixed;
                end[varying_axis] = interval[1];
                points.push(surface.point_at(end[0], end[1]));
                if points.len() >= 2 {
                    out.push(FacePolyline {
                        face,
                        positions: points,
                    });
                }
            }
        }
    }
    Some(out)
}

fn parameter_bounds(rings: &[Vec<[f64; 2]>]) -> Option<[[f64; 2]; 2]> {
    let mut bounds = [[f64::INFINITY, f64::NEG_INFINITY]; 2];
    for point in rings.iter().flatten() {
        for axis in 0..2 {
            bounds[axis][0] = bounds[axis][0].min(point[axis]);
            bounds[axis][1] = bounds[axis][1].max(point[axis]);
        }
    }
    bounds
        .iter()
        .all(|range| range[0].is_finite() && range[1].is_finite())
        .then_some(bounds)
}

fn line_intervals(
    rings: &[Vec<[f64; 2]>],
    fixed_axis: usize,
    fixed: f64,
) -> Vec<[f64; 2]> {
    let varying_axis = 1 - fixed_axis;
    let mut crossings = Vec::new();
    for ring in rings {
        for index in 0..ring.len() {
            let a = ring[index];
            let b = ring[(index + 1) % ring.len()];
            let crosses = (a[fixed_axis] <= fixed && b[fixed_axis] > fixed)
                || (b[fixed_axis] <= fixed && a[fixed_axis] > fixed);
            if !crosses {
                continue;
            }
            let unit = (fixed - a[fixed_axis]) / (b[fixed_axis] - a[fixed_axis]);
            crossings.push(a[varying_axis] + (b[varying_axis] - a[varying_axis]) * unit);
        }
    }
    crossings.sort_by(f64::total_cmp);
    crossings.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);
    crossings
        .chunks_exact(2)
        .filter_map(|pair| (pair[1] > pair[0]).then_some([pair[0], pair[1]]))
        .collect()
}

fn sample_isoline(
    surface: &super::geometry::Surface,
    fixed_axis: usize,
    fixed: f64,
    from: f64,
    to: f64,
    sag: f64,
    depth: u32,
    points: &mut Vec<[f64; 3]>,
) -> bool {
    let at = |value: f64| {
        let mut parameters = [0.0; 2];
        parameters[fixed_axis] = fixed;
        parameters[1 - fixed_axis] = value;
        surface.point_at(parameters[0], parameters[1])
    };
    let middle = 0.5 * (from + to);
    let start = Vec3::from(at(from));
    let end = Vec3::from(at(to));
    let split = [0.25, 0.5, 0.75].into_iter().any(|unit| {
        start
            .lerp(end, unit)
            .distance(Vec3::from(at(from + (to - from) * unit)))
            > sag
    });
    if split {
        if depth >= MAX_DEPTH {
            return false;
        }
        if !sample_isoline(surface, fixed_axis, fixed, from, middle, sag, depth + 1, points)
            || !sample_isoline(surface, fixed_axis, fixed, middle, to, sag, depth + 1, points)
        {
            return false;
        }
    } else {
        points.push(start.to_array());
    }
    true
}

fn finite_positive(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn body_span(body: &Body) -> f64 {
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    let mut include = |point: [f64; 3]| {
        for axis in 0..3 {
            low[axis] = low[axis].min(point[axis]);
            high[axis] = high[axis].max(point[axis]);
        }
    };
    for (_, vertex) in body.vertices.iter() {
        include(vertex.point);
    }
    for (_, edge) in body.edges.iter() {
        let Some(curve) = body.curves.get(edge.curve) else {
            continue;
        };
        for step in 0..=32 {
            let parameter = edge.start_parameter
                + (edge.end_parameter - edge.start_parameter) * step as f64 / 32.0;
            include(curve.point_at(parameter));
        }
    }
    (0..3)
        .map(|axis| high[axis] - low[axis])
        .filter(|span| span.is_finite())
        .fold(0.0_f64, f64::max)
        .max(1e-9)
}

fn shared_edge_samples(
    body: &Body,
    edge_key: EdgeKey,
    sag: f64,
    tolerance: f64,
) -> Option<Vec<super::place::EdgeSample>> {
    let edge = body.edges.get(edge_key)?;
    let curve = body.curves.get(edge.curve)?;
    let mut samples = vec![super::place::EdgeSample {
        parameter: edge.start_parameter,
        position: curve.point_at(edge.start_parameter),
    }];
    refine_edge(
        body,
        edge_key,
        edge.start_parameter,
        edge.end_parameter,
        sag.max(1e-12),
        tolerance.max(1e-12),
        0,
        &mut samples,
    )?;
    samples.push(super::place::EdgeSample {
        parameter: edge.end_parameter,
        position: curve.point_at(edge.end_parameter),
    });
    Some(samples)
}

fn refine_edge(
    body: &Body,
    edge_key: EdgeKey,
    from: f64,
    to: f64,
    sag: f64,
    tolerance: f64,
    depth: u32,
    samples: &mut Vec<super::place::EdgeSample>,
) -> Option<()> {
    let edge = body.edges.get(edge_key)?;
    let curve = body.curves.get(edge.curve)?;
    let middle = 0.5 * (from + to);
    let start = Vec3::from(curve.point_at(from));
    let end = Vec3::from(curve.point_at(to));
    let curved = Vec3::from(curve.point_at(middle));
    let mut split = [0.25, 0.5, 0.75].into_iter().any(|unit| {
        start
            .lerp(end, unit)
            .distance(Vec3::from(curve.point_at(from + (to - from) * unit)))
            > sag
    });
    for coedge_key in &edge.coedges {
        let Some((surface, pcurve, reversed)) = coedge_geometry(body, *coedge_key)
        else {
            continue;
        };
        let at = |parameter: f64| {
            let mut unit = (parameter - edge.start_parameter)
                / (edge.end_parameter - edge.start_parameter);
            if reversed {
                unit = 1.0 - unit;
            }
            let uv = pcurve.point_at(unit);
            surface.point_at(uv[0], uv[1])
        };
        let flat_start = Vec3::from(at(from));
        let flat_end = Vec3::from(at(to));
        for unit in [0.25, 0.5, 0.75] {
            let parameter = from + (to - from) * unit;
            let on_surface = Vec3::from(at(parameter));
            if on_surface.distance(Vec3::from(curve.point_at(parameter))) > tolerance {
                return None;
            }
            split |= flat_start.lerp(flat_end, unit).distance(on_surface) > sag;
        }
    }
    if split {
        if depth >= MAX_DEPTH {
            return None;
        }
        refine_edge(body, edge_key, from, middle, sag, tolerance, depth + 1, samples)?;
        samples.push(super::place::EdgeSample {
            parameter: middle,
            position: curved.to_array(),
        });
        refine_edge(body, edge_key, middle, to, sag, tolerance, depth + 1, samples)?;
    }
    Some(())
}

fn coedge_geometry<'a>(
    body: &'a Body,
    coedge_key: super::topology::CoedgeKey,
) -> Option<(
    &'a super::geometry::Surface,
    &'a crate::geom2d::Curve,
    bool,
)> {
    let coedge = body.coedges.get(coedge_key)?;
    let pcurve = coedge.pcurve.as_ref()?;
    let face = body.faces.get(body.loops.get(coedge.owner)?.owner)?;
    let surface = body.surfaces.get(face.surface)?;
    Some((surface, pcurve, !coedge.forward))
}

#[derive(Clone)]
struct BoundaryPoint {
    parameters: [f64; 2],
    position: [f64; 3],
}

fn scheduled_face(
    body: &Body,
    face: FaceKey,
    sag: f64,
    tolerance: f64,
    schedules: &HashMap<EdgeKey, Vec<super::place::EdgeSample>>,
) -> Option<Mesh> {
    let node = body.faces.get(face)?;
    let surface = body.surfaces.get(node.surface)?;
    let rings = face_rings(body, face, surface, schedules, tolerance)?;
    fill_scheduled(body, face, surface, &rings, sag, tolerance)
}

fn face_rings(
    body: &Body,
    face: FaceKey,
    surface: &super::geometry::Surface,
    schedules: &HashMap<EdgeKey, Vec<super::place::EdgeSample>>,
    tolerance: f64,
) -> Option<Vec<Vec<BoundaryPoint>>> {
    if let Some(rings) = scheduled_rings(body, face, surface, schedules, tolerance) {
        if rings.iter().any(|ring| boundary_area(ring)) {
            return Some(align_rings(rings, periods(surface)));
        }
    }
    scheduled_band(body, face, surface, schedules, tolerance).map(|ring| vec![ring])
}

fn scheduled_rings(
    body: &Body,
    face: FaceKey,
    surface: &super::geometry::Surface,
    schedules: &HashMap<EdgeKey, Vec<super::place::EdgeSample>>,
    tolerance: f64,
) -> Option<Vec<Vec<BoundaryPoint>>> {
    let node = body.faces.get(face)?;
    let mut rings = Vec::with_capacity(node.loops.len());
    for loop_key in &node.loops {
        let ring = body.loops.get(*loop_key)?;
        let mut pieces = Vec::with_capacity(ring.coedges.len());
        for coedge_key in &ring.coedges {
            let coedge = body.coedges.get(*coedge_key)?;
            let mut samples = schedules.get(&coedge.edge)?.clone();
            if !coedge.forward {
                samples.reverse();
            }
            if samples.len() >= 2 {
                pieces.push((samples, coedge.pcurve.as_ref()));
            }
        }
        let points = chain_samples(surface, pieces, tolerance)?;
        if points.len() >= 3 {
            rings.push(points);
        }
    }
    (!rings.is_empty()).then_some(rings)
}

fn chain_samples(
    surface: &super::geometry::Surface,
    pieces: Vec<(Vec<super::place::EdgeSample>, Option<&crate::geom2d::Curve>)>,
    tolerance: f64,
) -> Option<Vec<BoundaryPoint>> {
    let mut pieces = pieces.into_iter();
    let (first, pcurve) = pieces.next()?;
    let mut points = parameterize_samples(surface, &first, pcurve)?;
    unwrap_boundary(&mut points, periods(surface));
    for (samples, pcurve) in pieces {
        let head = points.last()?.position;
        let mut next = parameterize_samples(surface, &samples, pcurve)?;
        unwrap_boundary(&mut next, periods(surface));
        align_parameters(&mut next, &points, periods(surface));
        if distance3(head, next[0].position) > tolerance {
            return None;
        }
        let skip = 1;
        points.extend_from_slice(&next[skip..]);
    }
    if distance3(points.first()?.position, points.last()?.position) > tolerance {
        return None;
    }
    if parameter_near(points.first()?.parameters, points.last()?.parameters) {
        points.pop();
    }
    Some(points)
}

fn boundary_area(points: &[BoundaryPoint]) -> bool {
    let parameters: Vec<[f64; 2]> = points.iter().map(|point| point.parameters).collect();
    let area = boundary_area_value(points);
    let spans = [0, 1].map(|axis| {
        let low = parameters.iter().map(|point| point[axis]).fold(f64::INFINITY, f64::min);
        let high = parameters.iter().map(|point| point[axis]).fold(f64::NEG_INFINITY, f64::max);
        high - low
    });
    area.is_finite() && area > f64::EPSILON * 64.0 * spans[0] * spans[1]
}

fn boundary_area_value(points: &[BoundaryPoint]) -> f64 {
    crate::geom2d::signed_area(
        &points.iter().map(|point| point.parameters).collect::<Vec<_>>(),
    )
    .abs()
}

fn align_rings(mut rings: Vec<Vec<BoundaryPoint>>, periods: [Option<f64>; 2]) -> Vec<Vec<BoundaryPoint>> {
    let Some(outer) = rings
        .iter()
        .max_by(|a, b| boundary_area_value(a).total_cmp(&boundary_area_value(b)))
        .map(|ring| boundary_centroid(ring))
    else {
        return rings;
    };
    for ring in &mut rings {
        let centre = boundary_centroid(ring);
        for axis in 0..2 {
            if let Some(period) = periods[axis] {
                let shift = period * ((outer[axis] - centre[axis]) / period).round();
                for point in ring.iter_mut() {
                    point.parameters[axis] += shift;
                }
            }
        }
    }
    rings
}

fn boundary_centroid(points: &[BoundaryPoint]) -> [f64; 2] {
    let count = points.len().max(1) as f64;
    points.iter().fold([0.0; 2], |sum, point| {
        [sum[0] + point.parameters[0] / count, sum[1] + point.parameters[1] / count]
    })
}

fn parameterize(
    surface: &super::geometry::Surface,
    positions: &[[f64; 3]],
) -> Option<Vec<BoundaryPoint>> {
    let periods = periods(surface);
    let mut parameters: Vec<[f64; 2]> = positions
        .iter()
        .map(|position| match surface.parameters_at(*position) {
            Some((u, v)) => Some([u, v]),
            None => singular_parameters(surface, *position),
        })
        .collect::<Option<_>>()?;
    for axis in 0..2 {
        if periods[axis].is_some() {
            for index in 0..parameters.len() {
                if parameters[index][axis].is_finite() {
                    continue;
                }
                parameters[index][axis] = (1..parameters.len())
                    .flat_map(|offset| {
                        [index.checked_sub(offset), (index + offset < parameters.len()).then_some(index + offset)]
                    })
                    .flatten()
                    .find_map(|near| parameters[near][axis].is_finite().then_some(parameters[near][axis]))?;
            }
        }
    }
    let mut previous: Option<[f64; 2]> = None;
    positions
        .iter()
        .zip(parameters)
        .map(|(position, mut parameters)| {
            if let Some(last) = previous {
                for axis in 0..2 {
                    if let Some(period) = periods[axis] {
                        parameters[axis] = unwound(parameters[axis], last[axis], period);
                    }
                }
            }
            previous = Some(parameters);
            Some(BoundaryPoint {
                parameters,
                position: *position,
            })
        })
        .collect()
}

fn singular_parameters(
    surface: &super::geometry::Surface,
    position: [f64; 3],
) -> Option<[f64; 2]> {
    let super::geometry::Surface::Sphere(sphere) = surface else {
        return None;
    };
    let normal = Vec3::from(sphere.frame.normal()?);
    let height = (Vec3::from(position) - Vec3::from(sphere.frame.origin)).dot(normal);
    Some([f64::NAN, if height >= 0.0 { FRAC_PI_2 } else { -FRAC_PI_2 }])
}

fn parameterize_samples(
    surface: &super::geometry::Surface,
    samples: &[super::place::EdgeSample],
    pcurve: Option<&crate::geom2d::Curve>,
) -> Option<Vec<BoundaryPoint>> {
    if let Some(pcurve) = pcurve {
        let first = samples.first()?.parameter;
        let last = samples.last()?.parameter;
        let span = last - first;
        return samples
            .iter()
            .map(|sample| {
                let parameter = if span.abs() > f64::EPSILON {
                    (sample.parameter - first) / span
                } else {
                    0.0
                };
                Some(BoundaryPoint {
                    parameters: pcurve.point_at(parameter),
                    position: sample.position,
                })
            })
            .collect();
    }
    let positions: Vec<[f64; 3]> = samples.iter().map(|sample| sample.position).collect();
    parameterize(surface, &positions)
}

fn align_parameters(points: &mut [BoundaryPoint], chain: &[BoundaryPoint], periods: [Option<f64>; 2]) {
    let Some(first) = points.first() else {
        return;
    };
    let previous = chain.last().map(|point| point.parameters).unwrap_or(first.parameters);
    let behind = (chain.len() >= 2).then(|| [chain[chain.len() - 2].parameters, previous]);
    let last = points.last().map(|point| point.parameters).unwrap_or(first.parameters);
    let mut best = (f64::INFINITY, [0.0; 2]);
    for across in period_shifts(periods[0]) {
        for along in period_shifts(periods[1]) {
            let shift = [across, along];
            let moved_first = [first.parameters[0] + across, first.parameters[1] + along];
            let moved_last = [last[0] + across, last[1] + along];
            if behind.is_some_and(|pair| {
                parameter_near(moved_first, pair[1]) && parameter_near(moved_last, pair[0])
            }) || (parameter_near(moved_first, previous)
                && chain
                    .first()
                    .is_some_and(|first| parameter_near(moved_last, first.parameters)))
            {
                continue;
            }
            let gap = (moved_first[0] - previous[0]).hypot(moved_first[1] - previous[1]);
            if gap < best.0 {
                best = (gap, shift);
            }
        }
    }
    for point in points {
        point.parameters[0] += best.1[0];
        point.parameters[1] += best.1[1];
    }
}

fn period_shifts(period: Option<f64>) -> impl Iterator<Item = f64> {
    let period = period.filter(|value| value.is_finite() && *value > 0.0);
    let turns = if period.is_some() { 2 } else { 0 };
    (-turns..=turns).map(move |turn| f64::from(turn) * period.unwrap_or(0.0))
}

fn parameter_near(a: [f64; 2], b: [f64; 2]) -> bool {
    let scale = a
        .into_iter()
        .chain(b)
        .map(f64::abs)
        .fold(1.0, f64::max);
    (a[0] - b[0]).hypot(a[1] - b[1]) <= f64::EPSILON * 64.0 * scale
}

fn scheduled_band(
    body: &Body,
    face: FaceKey,
    surface: &super::geometry::Surface,
    schedules: &HashMap<EdgeKey, Vec<super::place::EdgeSample>>,
    tolerance: f64,
) -> Option<Vec<BoundaryPoint>> {
    let node = body.faces.get(face)?;
    if !(1..=2).contains(&node.loops.len()) {
        return None;
    }
    let surface_periods = periods(surface);
    let mut rims = Vec::with_capacity(node.loops.len());
    for loop_key in &node.loops {
        let ring = body.loops.get(*loop_key)?;
        let [coedge_key] = ring.coedges[..] else {
            return None;
        };
        let coedge = body.coedges.get(coedge_key)?;
        let mut samples = schedules.get(&coedge.edge)?.clone();
        if !coedge.forward {
            samples.reverse();
        }
        let mut points = parameterize_samples(surface, &samples, coedge.pcurve.as_ref())?;
        unwrap_boundary(&mut points, surface_periods);
        rims.push(points);
    }
    let varying = (0..2)
        .filter(|axis| surface_periods[*axis].is_some())
        .max_by(|a, b| {
            parameter_range(&rims[0], *a)
                .total_cmp(&parameter_range(&rims[0], *b))
        })?;
    let period = surface_periods[varying]?;
    if rims
        .iter()
        .any(|rim| parameter_range(rim, varying) < period * 0.5)
    {
        return None;
    }
    let fixed = 1 - varying;
    let traversal: Vec<f64> = rims
        .iter()
        .map(|rim| rim.last().unwrap().parameters[varying] - rim[0].parameters[varying])
        .collect();
    for rim in &mut rims {
        rim.sort_by(|a, b| a.parameters[varying].total_cmp(&b.parameters[varying]));
    }
    let base = rims[0][0].parameters[varying];
    for rim in rims.iter_mut().skip(1) {
        let shift = period * ((base - rim[0].parameters[varying]) / period).round();
        for point in rim {
            point.parameters[varying] += shift;
        }
    }
    let mut bounds: Vec<f64> = rims
        .iter()
        .map(|rim| average_parameter(rim, fixed))
        .collect();
    for (rim, value) in rims.iter().zip(&bounds) {
        if rim.iter().any(|point| {
            let mut parameters = point.parameters;
            parameters[fixed] = *value;
            distance3(
                point.position,
                surface.point_at(parameters[0], parameters[1]),
            ) > tolerance
        }) {
            return None;
        }
    }
    if rims.len() == 2 {
        if let Some(fixed_period) = surface_periods[fixed] {
            let outward = if node.forward { traversal[0] } else { -traversal[0] };
            let positive = (varying == 0 && outward > 0.0)
                || (varying == 1 && outward < 0.0);
            let delta = if positive {
                (bounds[1] - bounds[0]).rem_euclid(fixed_period)
            } else {
                -(bounds[0] - bounds[1]).rem_euclid(fixed_period)
            };
            let target = bounds[0] + delta;
            let shift = target - bounds[1];
            bounds[1] = target;
            for point in &mut rims[1] {
                point.parameters[fixed] += shift;
            }
        }
    }
    if rims.len() == 1 {
        let domain = surface_domain(surface)?;
        let candidates = [domain[fixed][0], domain[fixed][1]];
        let collapsed = candidates.map(|value| {
            boundary_collapsed(surface, varying, base, period, fixed, value, tolerance)
        });
        let target = match collapsed {
            [true, false] => candidates[0],
            [false, true] => candidates[1],
            [true, true] => {
                let outward = if node.forward { traversal[0] } else { -traversal[0] };
                if (varying == 0 && outward > 0.0) || (varying == 1 && outward < 0.0) {
                    candidates[1]
                } else {
                    candidates[0]
                }
            }
            _ => return None,
        };
        let position = surface.point_at(
            if varying == 0 { base } else { target },
            if varying == 0 { target } else { base },
        );
        let mut singular = vec![
            BoundaryPoint {
                parameters: [0.0; 2],
                position,
            },
            BoundaryPoint {
                parameters: [0.0; 2],
                position,
            },
        ];
        singular[0].parameters[varying] = base;
        singular[1].parameters[varying] = base + period;
        singular[0].parameters[fixed] = target;
        singular[1].parameters[fixed] = target;
        rims.push(singular);
        bounds.push(target);
    }
    let low_index = usize::from(bounds[1] < bounds[0]);
    let high_index = 1 - low_index;
    let mut low = rims.remove(low_index);
    let mut high = rims.remove(if high_index > low_index { high_index - 1 } else { high_index });
    let low_fixed = bounds[low_index];
    let high_fixed = bounds[high_index];
    for point in &mut low {
        point.parameters[fixed] = low_fixed;
    }
    for point in &mut high {
        point.parameters[fixed] = high_fixed;
    }
    high.reverse();
    low.extend(high);
    Some(low)
}

fn unwrap_boundary(points: &mut [BoundaryPoint], periods: [Option<f64>; 2]) {
    for index in 1..points.len() {
        for axis in 0..2 {
            if let Some(period) = periods[axis] {
                points[index].parameters[axis] = unwound(
                    points[index].parameters[axis],
                    points[index - 1].parameters[axis],
                    period,
                );
            }
        }
    }
}

fn parameter_range(points: &[BoundaryPoint], axis: usize) -> f64 {
    let low = points
        .iter()
        .map(|point| point.parameters[axis])
        .fold(f64::INFINITY, f64::min);
    let high = points
        .iter()
        .map(|point| point.parameters[axis])
        .fold(f64::NEG_INFINITY, f64::max);
    high - low
}

fn average_parameter(points: &[BoundaryPoint], axis: usize) -> f64 {
    points.iter().map(|point| point.parameters[axis]).sum::<f64>() / points.len() as f64
}

fn surface_domain(surface: &super::geometry::Surface) -> Option<[[f64; 2]; 2]> {
    use super::geometry::Surface;
    match surface {
        Surface::Sphere(_) => Some([[0.0, TAU], [-FRAC_PI_2, FRAC_PI_2]]),
        Surface::Cone(surface) if surface.half_angle.tan().abs() > 1e-15 => {
            let apex = surface.radius / surface.half_angle.tan();
            Some([[0.0, TAU], [0.0_f64.min(apex), 0.0_f64.max(apex)]])
        }
        Surface::Torus(_) => Some([[0.0, TAU], [0.0, TAU]]),
        Surface::Nurbs(surface) => {
            let ((u0, u1), (v0, v1)) = surface.domain();
            Some([[u0, u1], [v0, v1]])
        }
        _ => None,
    }
}

fn boundary_collapsed(
    surface: &super::geometry::Surface,
    varying: usize,
    from: f64,
    period: f64,
    fixed: usize,
    value: f64,
    tolerance: f64,
) -> bool {
    let mut first = [0.0; 2];
    first[varying] = from;
    first[fixed] = value;
    let origin = surface.point_at(first[0], first[1]);
    (1..=4).all(|step| {
        let mut parameters = first;
        parameters[varying] = from + period * step as f64 / 4.0;
        distance3(origin, surface.point_at(parameters[0], parameters[1])) <= tolerance
    })
}

fn periods(surface: &super::geometry::Surface) -> [Option<f64>; 2] {
    use super::geometry::Surface;
    match surface {
        Surface::Plane(_) => [None, None],
        Surface::Cylinder(_) | Surface::Cone(_) | Surface::Sphere(_) => [Some(TAU), None],
        Surface::Torus(_) => [Some(TAU), Some(TAU)],
        Surface::Nurbs(surface) => {
            let ((u0, u1), (v0, v1)) = surface.domain();
            let periodic = surface.periodicity();
            [
                periodic[0].then_some(u1 - u0),
                periodic[1].then_some(v1 - v0),
            ]
        }
    }
}

fn unwound(value: f64, previous: f64, period: f64) -> f64 {
    value + period * ((previous - value) / period).round()
}

fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    Vec3::from(a).distance(Vec3::from(b))
}

fn parameter_in_span(
    curve: &super::geometry::Curve3,
    parameter: f64,
    edge: &super::topology::Edge,
) -> Option<f64> {
    use super::geometry::Curve3;
    if matches!(curve, Curve3::Circle(_) | Curve3::Ellipse(_)) {
        for turn in -3..=3 {
            let candidate = parameter + TAU * f64::from(turn);
            if candidate >= edge.start_parameter - 1e-9
                && candidate <= edge.end_parameter + 1e-9
            {
                return Some(candidate);
            }
        }
        return None;
    }
    if let Curve3::Nurbs(curve) = curve {
        if curve.periodicity() {
            let (from, to) = curve.domain();
            let period = to - from;
            for turn in -3..=3 {
                let candidate = parameter + period * f64::from(turn);
                if candidate >= edge.start_parameter - 1e-9
                    && candidate <= edge.end_parameter + 1e-9
                {
                    return Some(candidate);
                }
            }
            return None;
        }
    }
    (parameter >= edge.start_parameter - 1e-9 && parameter <= edge.end_parameter + 1e-9)
        .then_some(parameter)
}

/// How far a triangle's middle may sit from the surface before it is split.
///
/// Only curved faces are ever split; a plane's triangles are exact whatever
/// this is.
const DEFAULT_SAG: f64 = 0.01;

/// Recursion guard; tolerance normally stops first.
const MAX_DEPTH: u32 = 16;

/// Triangulates a whole body.
///
/// `sag` is how far a triangle may depart from the surface it lies on. A
/// caller rendering at a known zoom passes its own; the default is fine for
/// a drawing measured in millimetres.
pub fn body(body: &Body, sag: f64, tolerance: f64) -> Mesh {
    tessellate(body, TessellationTolerance::new(sag, tolerance)).mesh
}

/// Triangulates one face.
///
/// `None` when its canonical edge schedule cannot be mapped to the surface.
pub fn face(body: &Body, face: FaceKey, sag: f64, tolerance: f64) -> Option<Mesh> {
    let schedules: HashMap<EdgeKey, Vec<super::place::EdgeSample>> = body
        .edge_keys()
        .filter_map(|edge| Some((edge, shared_edge_samples(body, edge, sag, tolerance)?)))
        .collect();
    scheduled_face(body, face, sag, tolerance, &schedules)
}

/// Triangulates a face over the rings its boundary makes in `(u, v)`.
///
/// Which ring bounds the face and which cut holes in it is decided by area,
/// not by the order they arrive in. Nothing guarantees that order: a face
/// lifted from a file lists its loops however the file did, and taking the
/// first one on trust draws a plate with its bolt hole filled in and the
/// metal around it missing — a picture that looks deliberate.
fn fill_scheduled(
    body: &Body,
    face: FaceKey,
    surface: &super::geometry::Surface,
    rings: &[Vec<BoundaryPoint>],
    sag: f64,
    tolerance: f64,
) -> Option<Mesh> {
    let widest = rings
        .iter()
        .enumerate()
        .max_by(|a, b| {
            crate::geom2d::signed_area(
                &a.1.iter().map(|point| point.parameters).collect::<Vec<_>>(),
            )
            .abs()
            .total_cmp(
                &crate::geom2d::signed_area(
                    &b.1.iter().map(|point| point.parameters).collect::<Vec<_>>(),
                )
                .abs(),
            )
        })
        .map(|(index, _)| index)?;
    let outer: Vec<[f64; 2]> = rings[widest]
        .iter()
        .map(|point| point.parameters)
        .collect();
    let holes: Vec<Vec<[f64; 2]>> = rings
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != widest)
        .map(|(_, ring)| ring.iter().map(|point| point.parameters).collect())
        .collect();
    let pins: Vec<BoundaryPoint> = rings.iter().flatten().cloned().collect();
    let edges: Vec<EdgeKey> = body
        .face_coedges(face)
        .into_iter()
        .filter_map(|coedge| Some(body.coedges.get(coedge)?.edge))
        .collect();
    let (parameters, triangles) = triangulate(&outer, &holes);
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
            emit_scheduled(&mut mesh, body, face, corners, &pins, &edges, tolerance);
        } else {
            if !refine_scheduled(
                &mut mesh,
                body,
                face,
                corners,
                sag,
                0,
                &pins,
                &edges,
                tolerance,
            ) {
                return None;
            }
        }
    }
    Some(mesh)
}

fn refine_scheduled(
    mesh: &mut Mesh,
    body: &Body,
    face: FaceKey,
    corners: [[f64; 2]; 3],
    sag: f64,
    depth: u32,
    pins: &[BoundaryPoint],
    edges: &[EdgeKey],
    tolerance: f64,
) -> bool {
    let Some(node) = body.faces.get(face) else {
        return false;
    };
    let Some(surface) = body.surfaces.get(node.surface) else {
        return false;
    };
    let split = surface_triangle_error(surface, corners) > sag;
    if split {
        if depth >= MAX_DEPTH {
            return false;
        }
        let middle = [
            (corners[0][0] + corners[1][0] + corners[2][0]) / 3.0,
            (corners[0][1] + corners[1][1] + corners[2][1]) / 3.0,
        ];
        for part in [
            [corners[0], corners[1], middle],
            [corners[1], corners[2], middle],
            [corners[2], corners[0], middle],
        ] {
            if !refine_scheduled(
                mesh,
                body,
                face,
                part,
                sag,
                depth + 1,
                pins,
                edges,
                tolerance,
            ) {
                return false;
            }
        }
        return true;
    }
    emit_scheduled(mesh, body, face, corners, pins, edges, tolerance);
    true
}

fn surface_triangle_error(
    surface: &super::geometry::Surface,
    corners: [[f64; 2]; 3],
) -> f64 {
    let positions = corners.map(|uv| Vec3::from(surface.point_at(uv[0], uv[1])));
    [
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [1.0 / 3.0; 3],
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
    ]
    .into_iter()
    .map(|weights| {
        let uv = [
            corners[0][0] * weights[0]
                + corners[1][0] * weights[1]
                + corners[2][0] * weights[2],
            corners[0][1] * weights[0]
                + corners[1][1] * weights[1]
                + corners[2][1] * weights[2],
        ];
        let flat = positions[0] * weights[0]
            + positions[1] * weights[1]
            + positions[2] * weights[2];
        flat.distance(Vec3::from(surface.point_at(uv[0], uv[1])))
    })
    .fold(0.0, f64::max)
}

fn emit_scheduled(
    mesh: &mut Mesh,
    body: &Body,
    face: FaceKey,
    corners: [[f64; 2]; 3],
    pins: &[BoundaryPoint],
    edges: &[EdgeKey],
    tolerance: f64,
) {
    let Some(node) = body.faces.get(face) else {
        return;
    };
    let Some(surface) = body.surfaces.get(node.surface) else {
        return;
    };
    let points: Vec<Vec3> = corners
        .iter()
        .map(|parameters| {
            Vec3::from(canonical_point(
                body,
                surface,
                *parameters,
                pins,
                edges,
                tolerance,
            ))
        })
        .collect();
    let Some(normal) = (points[1] - points[0])
        .cross(points[2] - points[0])
        .normalize()
    else {
        return;
    };
    let normal = if node.forward { normal } else { -normal };
    let base = mesh.positions.len();
    let order = if node.forward { [0, 1, 2] } else { [0, 2, 1] };
    for step in order {
        mesh.positions.push(points[step].to_array());
        mesh.normals.push(normal.to_array());
    }
    mesh.triangles.push([base, base + 1, base + 2]);
}

fn canonical_point(
    body: &Body,
    surface: &super::geometry::Surface,
    parameters: [f64; 2],
    pins: &[BoundaryPoint],
    edges: &[EdgeKey],
    tolerance: f64,
) -> [f64; 3] {
    if let Some(pin) = pins.iter().find(|pin| pin.parameters == parameters) {
        return pin.position;
    }
    let point = surface.point_at(parameters[0], parameters[1]);
    for edge_key in edges {
        let Some(edge) = body.edges.get(*edge_key) else {
            continue;
        };
        let Some(curve) = body.curves.get(edge.curve) else {
            continue;
        };
        let Some(parameter) = parameter_in_span(curve, curve.parameter_at(point), edge) else {
            continue;
        };
        let on_curve = curve.point_at(parameter);
        if distance3(point, on_curve) <= tolerance {
            return on_curve;
        }
    }
    point
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
