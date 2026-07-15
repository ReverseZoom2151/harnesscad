//! harnesscad truck driver
//!
//! A tiny CLI over the `truck` B-rep NURBS kernel (github.com/ricosjp/truck,
//! Apache-2.0). It reads a normalised geometry job (JSON, produced by the Python
//! `TruckBackend` from the kernel-neutral F-rep tree) and emits, into an output
//! directory:
//!
//!   * `model.stl`  -- the tessellated solid (binary STL), the mesh the harness
//!     reads back for volume / bbox / manifold checks;
//!   * `model.json` -- {ok, volume, bbox, n_faces, n_edges, step, reason}, where
//!     `volume`/`n_faces`/`n_edges` come from truck's OWN B-rep + mesh (an
//!     independent voice for the differential oracle);
//!   * `model.step` -- ISO-10303-21 STEP, ONLY when the model is pure modeling
//!     (extrude / revolve). truck-stepio 0.3 cannot yet serialise the output of a
//!     boolean, so a model containing any boolean reports `step:false` rather than
//!     writing a lie.
//!
//! Every op truck cannot do HONESTLY fails with a reason in `model.json` and a
//! non-zero exit -- nothing is faked.
//!
//! truck crate versions (pinned by Cargo.lock): truck-modeling 0.6.0,
//! truck-topology 0.6.0, truck-polymesh 0.6.0, truck-meshalgo 0.4.0,
//! truck-shapeops 0.4.0, truck-stepio 0.3.0.

use std::collections::HashSet;
use std::path::Path;

use serde::Deserialize;
use truck_meshalgo::prelude::{CalcVolume, MeshableShape, MeshedShape};
use truck_modeling::*;
// `use truck_modeling::*` pulls in truck's `pub type Result<T>` alias, which
// shadows the std two-parameter Result. Re-import the std one explicitly (an
// explicit `use` wins over a glob) so `Result<T, String>` means what we expect.
use std::result::Result;

// ---------------------------------------------------------------------------
// input schema (mirrors TruckBackend._lower in truck.py)
// ---------------------------------------------------------------------------
#[derive(Deserialize)]
struct Job {
    #[serde(default = "default_tol")]
    tol: f64,
    #[serde(default = "default_bool_tol")]
    bool_tol: f64,
    node: NodeJson,
}

fn default_tol() -> f64 {
    0.02
}
fn default_bool_tol() -> f64 {
    0.05
}

#[derive(Deserialize)]
struct FaceJson {
    outer: Vec<[f64; 3]>,
    #[serde(default)]
    holes: Vec<Vec<[f64; 3]>>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum NodeJson {
    Extrude {
        faces: Vec<FaceJson>,
        vector: [f64; 3],
    },
    Revolve {
        faces: Vec<FaceJson>,
        origin: [f64; 3],
        axis: [f64; 3],
        angle_deg: f64,
    },
    Boolean {
        op: String,
        a: Box<NodeJson>,
        b: Box<NodeJson>,
    },
    /// A body replicated by a list of rigid (and, for mirror, reflecting) 3x4
    /// row-major matrices, then combined. Overlapping copies are merged with a
    /// real `truck-shapeops` boolean; provably-disjoint copies become distinct
    /// boundary shells of ONE multi-shell `Solid` -- which is how truck honours a
    /// pattern or mirror of disjoint bodies WITHOUT a boolean (`truck-shapeops`
    /// cannot union disjoint solids; a multi-shell solid needs no union). This one
    /// node serves `linear_pattern`, `circular_pattern` and `mirror`.
    TransformUnion {
        child: Box<NodeJson>,
        /// each entry is a 3x4 row-major affine: [r00 r01 r02 tx  r10 r11 r12 ty
        /// r20 r21 r22 tz]; a negative linear determinant is a reflection.
        matrices: Vec<[f64; 12]>,
    },
}

// ---------------------------------------------------------------------------
// B-rep construction
// ---------------------------------------------------------------------------
fn build_wire(pts: &[[f64; 3]]) -> Result<Wire, String> {
    if pts.len() < 3 {
        return Err(format!("loop has {} points (< 3)", pts.len()));
    }
    let verts: Vec<Vertex> = pts
        .iter()
        .map(|p| builder::vertex(Point3::new(p[0], p[1], p[2])))
        .collect();
    let n = verts.len();
    let mut wire = Wire::new();
    for i in 0..n {
        wire.push_back(builder::line(&verts[i], &verts[(i + 1) % n]));
    }
    Ok(wire)
}

fn build_face(f: &FaceJson) -> Result<Face, String> {
    let mut wires = vec![build_wire(&f.outer)?];
    // Inner loops arrive with the same winding as the outer boundary; a hole is
    // the reverse orientation, so it is inverted before attaching.
    for h in &f.holes {
        wires.push(build_wire(h)?.inverse());
    }
    builder::try_attach_plane(&wires).map_err(|e| format!("try_attach_plane failed: {e:?}"))
}

fn normalize(v: [f64; 3]) -> Vector3 {
    let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if m == 0.0 {
        Vector3::new(0.0, 0.0, 1.0)
    } else {
        Vector3::new(v[0] / m, v[1] / m, v[2] / m)
    }
}

/// The axis-aligned bound of a solid, from its topological vertices.
fn solid_aabb(s: &Solid) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for v in s.vertex_iter() {
        let p = v.point();
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

/// Do two solids' bounds overlap (share interior) within `tol`? A gap wider than
/// `tol` on any axis proves they are disjoint.
fn aabb_overlap(a: &Solid, b: &Solid, tol: f64) -> bool {
    let (alo, ahi) = solid_aabb(a);
    let (blo, bhi) = solid_aabb(b);
    for k in 0..3 {
        if ahi[k] < blo[k] - tol || bhi[k] < alo[k] - tol {
            return false;
        }
    }
    true
}

/// Combine solids into one, HONESTLY: overlapping bodies are merged with a real
/// `truck-shapeops` union; provably-disjoint bodies become separate connected
/// boundary shells of a single multi-shell `Solid` (no boolean, exact). A pair
/// whose bounds overlap but whose boolean union `truck-shapeops` cannot resolve
/// is a genuine kernel failure and is surfaced as an error -- never faked.
fn combine_solids(solids: Vec<Solid>, tol: f64) -> Result<Solid, String> {
    if solids.is_empty() {
        return Err("no solids produced".into());
    }
    let mut comps: Vec<Solid> = Vec::new();
    for s in solids {
        let mut merged = false;
        // Try a real boolean union with an existing component first: if the
        // bodies actually intersect, shapeops resolves it and we keep one shell.
        for i in 0..comps.len() {
            if let Some(u) = truck_shapeops::or(&comps[i], &s, tol) {
                comps[i] = u;
                merged = true;
                break;
            }
        }
        if merged {
            continue;
        }
        // No union resolved. Either s is disjoint from every component (correct,
        // keep it as its own shell) or it overlaps one and shapeops failed
        // (a real failure -- do not fake it).
        for c in &comps {
            if aabb_overlap(c, &s, tol) {
                return Err("truck-shapeops could not resolve a union of two \
                            overlapping bodies (the boolean returned None)"
                    .to_string());
            }
        }
        comps.push(s);
    }
    if comps.len() == 1 {
        return Ok(comps.pop().unwrap());
    }
    // A multi-shell solid: every disjoint body contributes its connected,
    // closed boundary shells. truck's own tessellation and divergence-theorem
    // volume sum correctly across them.
    let shells: Vec<Shell> = comps
        .into_iter()
        .flat_map(|c| c.into_boundaries())
        .collect();
    Solid::try_new(shells).map_err(|e| format!("combined multi-shell solid invalid: {e:?}"))
}

/// Backwards-compatible name: the extrude/revolve paths union the face solids.
fn union_all(solids: Vec<Solid>, tol: f64) -> Result<Solid, String> {
    combine_solids(solids, tol)
}

/// A cgmath `Matrix4` (column-major) from a row-major 3x4 affine.
fn matrix4_from_rowmajor(m: &[f64; 12]) -> Matrix4 {
    Matrix4::new(
        m[0], m[4], m[8], 0.0, // column 0
        m[1], m[5], m[9], 0.0, // column 1
        m[2], m[6], m[10], 0.0, // column 2
        m[3], m[7], m[11], 1.0, // column 3
    )
}

/// Determinant of the 3x3 linear part of a row-major 3x4 affine. Negative => a
/// reflection (orientation-reversing), which must be un-inverted after mapping.
fn linear_det(m: &[f64; 12]) -> f64 {
    m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8])
        + m[2] * (m[4] * m[9] - m[5] * m[8])
}

fn build_solid(node: &NodeJson, bool_tol: f64) -> Result<Solid, String> {
    match node {
        NodeJson::Extrude { faces, vector } => {
            let vec = Vector3::new(vector[0], vector[1], vector[2]);
            let mut solids = Vec::new();
            for f in faces {
                let face = build_face(f)?;
                solids.push(builder::tsweep(&face, vec));
            }
            union_all(solids, bool_tol)
        }
        NodeJson::Revolve {
            faces,
            origin,
            axis,
            angle_deg,
        } => {
            let o = Point3::new(origin[0], origin[1], origin[2]);
            let ax = normalize(*axis);
            let ang = Rad(angle_deg.to_radians());
            let mut solids = Vec::new();
            for f in faces {
                let face = build_face(f)?;
                solids.push(builder::rsweep(&face, o, ax, ang));
            }
            union_all(solids, bool_tol)
        }
        NodeJson::Boolean { op, a, b } => {
            let sa = build_solid(a, bool_tol)?;
            let mut sb = build_solid(b, bool_tol)?;
            match op.as_str() {
                "union" => truck_shapeops::or(&sa, &sb, bool_tol)
                    .ok_or_else(|| "truck-shapeops union (or) returned None".to_string()),
                "intersect" => truck_shapeops::and(&sa, &sb, bool_tol)
                    .ok_or_else(|| "truck-shapeops intersect (and) returned None".to_string()),
                "cut" => {
                    // A - B = A AND (complement of B); Solid::not() inverts every
                    // face, turning B into its complement.
                    sb.not();
                    truck_shapeops::and(&sa, &sb, bool_tol).ok_or_else(|| {
                        "truck-shapeops difference (and/not) returned None".to_string()
                    })
                }
                other => Err(format!("unknown boolean op {other:?}")),
            }
        }
        NodeJson::TransformUnion { child, matrices } => {
            if matrices.is_empty() {
                return Err("transform_union has no instances".into());
            }
            let base = build_solid(child, bool_tol)?;
            let mut instances = Vec::with_capacity(matrices.len());
            for m in matrices {
                let mat = matrix4_from_rowmajor(m);
                let mut inst = builder::transformed(&base, mat);
                // A reflection (det < 0) reverses face orientation; flip it back
                // so the copy's normals point outward like the original's.
                if linear_det(m) < 0.0 {
                    inst.not();
                }
                instances.push(inst);
            }
            combine_solids(instances, bool_tol)
        }
    }
}

/// A model contains geometry `truck-stepio` 0.3 cannot serialise: the output of a
/// boolean, OR a transform-union (which may itself have merged via a boolean, and
/// whose multi-shell result stepio does not round-trip). STEP is offered only for
/// pure extrude/revolve models.
fn blocks_step(node: &NodeJson) -> bool {
    match node {
        NodeJson::Boolean { .. } => true,
        NodeJson::TransformUnion { .. } => true,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// meshing + metrics
// ---------------------------------------------------------------------------
struct Metrics {
    volume: f64,
    bbox: [f64; 3],
    n_faces: usize,
    n_edges: usize,
    stl: Vec<u8>,
}

fn tessellate(solid: &Solid, tol: f64) -> Result<Metrics, String> {
    let poly = solid.triangulation(tol).to_polygon();
    let positions = poly.positions();
    if positions.is_empty() {
        return Err("triangulation produced an empty mesh".into());
    }

    // bounding box
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let bbox = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];

    // volume: truck's own closed-mesh integral (divergence theorem)
    let volume = poly.volume().abs();

    // B-rep counts straight off the solid's topology
    let n_faces = solid.face_iter().count();
    let mut edge_ids = HashSet::new();
    for e in solid.edge_iter() {
        edge_ids.insert(e.id());
    }
    let n_edges = edge_ids.len();

    // binary STL from the triangulated polygon mesh
    let mut tris: Vec<[[f32; 3]; 3]> = Vec::new();
    for tri in poly.faces().triangle_iter() {
        let a = positions[tri[0].pos];
        let b = positions[tri[1].pos];
        let c = positions[tri[2].pos];
        tris.push([
            [a[0] as f32, a[1] as f32, a[2] as f32],
            [b[0] as f32, b[1] as f32, b[2] as f32],
            [c[0] as f32, c[1] as f32, c[2] as f32],
        ]);
    }
    if tris.is_empty() {
        return Err("triangulation produced no triangles".into());
    }
    let stl = write_binary_stl(&tris);

    Ok(Metrics {
        volume,
        bbox,
        n_faces,
        n_edges,
        stl,
    })
}

fn write_binary_stl(tris: &[[[f32; 3]; 3]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(84 + tris.len() * 50);
    let header =
        b"harnesscad-truck-driver binary STL output payload_______________________________";
    out.extend_from_slice(&header[..80]);
    out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for t in tris {
        // face normal (right-hand rule)
        let (ax, ay, az) = (t[0][0], t[0][1], t[0][2]);
        let (bx, by, bz) = (t[1][0], t[1][1], t[1][2]);
        let (cx, cy, cz) = (t[2][0], t[2][1], t[2][2]);
        let (ux, uy, uz) = (bx - ax, by - ay, bz - az);
        let (vx, vy, vz) = (cx - ax, cy - ay, cz - az);
        let mut nx = uy * vz - uz * vy;
        let mut ny = uz * vx - ux * vz;
        let mut nz = ux * vy - uy * vx;
        let m = (nx * nx + ny * ny + nz * nz).sqrt();
        if m > 0.0 {
            nx /= m;
            ny /= m;
            nz /= m;
        }
        let _ = (az, bz, cz);
        for comp in [nx, ny, nz] {
            out.extend_from_slice(&comp.to_le_bytes());
        }
        for v in t {
            for comp in v {
                out.extend_from_slice(&comp.to_le_bytes());
            }
        }
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// STEP (pure-modeling solids only)
// ---------------------------------------------------------------------------
fn step_string(solid: &Solid) -> String {
    use truck_stepio::out;
    let compressed = solid.compress();
    let model = out::CompleteStepDisplay::new(
        out::StepModel::from(&compressed),
        out::StepHeaderDescriptor {
            organization_system: "harnesscad-truck-driver".to_string(),
            authorization: "harnesscad".to_string(),
            ..Default::default()
        },
    );
    model.to_string()
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------
fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ")
}

fn write_sidecar(dir: &Path, body: &str) {
    let _ = std::fs::write(dir.join("model.json"), body);
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err(format!("usage: {} <job.json> <out_dir>", args[0]));
    }
    let job_path = &args[1];
    let out_dir = Path::new(&args[2]);
    std::fs::create_dir_all(out_dir).map_err(|e| format!("mkdir out_dir: {e}"))?;

    let raw = std::fs::read_to_string(job_path).map_err(|e| format!("read job: {e}"))?;
    let job: Job = serde_json::from_str(&raw).map_err(|e| format!("parse job json: {e}"))?;

    let solid = build_solid(&job.node, job.bool_tol).map_err(|e| {
        write_sidecar(
            out_dir,
            &format!(
                "{{\"ok\":false,\"unsupported\":true,\"reason\":\"{}\"}}",
                json_escape(&e)
            ),
        );
        e
    })?;

    let metrics = tessellate(&solid, job.tol).map_err(|e| {
        write_sidecar(
            out_dir,
            &format!(
                "{{\"ok\":false,\"unsupported\":false,\"reason\":\"{}\"}}",
                json_escape(&e)
            ),
        );
        e
    })?;

    std::fs::write(out_dir.join("model.stl"), &metrics.stl).map_err(|e| format!("write stl: {e}"))?;

    // STEP only for pure-modeling solids (truck-stepio can't serialise boolean output)
    let step_ok = if blocks_step(&job.node) {
        false
    } else {
        let step = step_string(&solid);
        std::fs::write(out_dir.join("model.step"), step.as_bytes()).is_ok()
    };

    let sidecar = format!(
        "{{\"ok\":true,\"unsupported\":false,\"volume\":{},\"bbox\":[{},{},{}],\"n_faces\":{},\"n_edges\":{},\"step\":{}}}",
        metrics.volume,
        metrics.bbox[0],
        metrics.bbox[1],
        metrics.bbox[2],
        metrics.n_faces,
        metrics.n_edges,
        step_ok
    );
    write_sidecar(out_dir, &sidecar);
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("harnesscad-truck-driver: {e}");
        std::process::exit(1);
    }
}
