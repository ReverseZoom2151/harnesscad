# CISP Op-Vocabulary Completeness Audit

**Question.** Does the CISP op protocol itself
(`src/harnesscad/core/cisp/ops.py`) contain every operation the CAD world we
integrated actually needs? The doc-honour census asks whether each backend
*honours* its op fields; this audit asks the prior question: is the op set
*complete* against everything the integrations and mined repos express?

**Method.** Read `ops.py` in full (the current op set), then enumerated the op
vocabulary of every backend (`io/backends/*.py`) and every mined
representation module, and cross-tabulated. Read-only except this file.

**Verdict (one line).** CISP's op set is *complete for the sketch-extrude-
feature-history B-rep core* every integration shares, but it has **no primitive-
solid op, no split/section, no offset/thicken, no hull/minkowski, no arc/spline
sketch entity, no scale transform, no helix path, and no non-B-rep modality
(voxel assembly, span-infill edits)** — all of which one or more integrated
tools express natively.

---

## 1. Current CISP op set (from `core/cisp/ops.py`)

Mutating ops (22): `NewSketch`, `AddPoint`, `AddLine`, `AddCircle`,
`AddRectangle`, `Constrain`, `Extrude`, `Fillet`, `Boolean`, `Revolve`,
`Chamfer`, `Hole` (+cbore/csk fields), `Shell`, `Draft`, `Loft`, `Sweep`,
`LinearPattern`, `CircularPattern`, `Mirror`, `AddInstance`, `Mate`, `SetParam`.

Supporting tables: `CONSTRAINT_DOF` (8 kinds: coincident, horizontal, vertical,
parallel, perpendicular, distance, radius, equal); `PRIMITIVE_DOF` (point, line,
circle, rectangle). `Mate.kind` is validated against
`eval/verifiers/assembly.py::MATE_DOF` (rigid/fixed, revolute/hinge,
slider/prismatic, cylindrical, planar).

Query/export are not ops (handled by `backend.query()/export()`).

---

## 2. The definitive table

Legend — **HAS**: a CISP op expresses it directly. **PARTIAL**: expressible but
lossily / only a subset of what the tool exposes. **MISSING**: no CISP op.

| # | Operation | Expressed by (tools / mined modules) | CISP status | Why it matters if missing/partial |
|---|-----------|--------------------------------------|-------------|-----------------------------------|
| 1 | new sketch / plane | all backends; KCL `start_sketch_on`; CADmium `sketch`; FeatureScript `sketch` | **HAS** (`NewSketch`) | — |
| 2 | point / line / circle / rectangle | all; KCL sketch; CADmium; csg 2D prims | **HAS** | — |
| 3 | **arc** | KCL `arc`/`tangential_arc`/`circle_three_point`; CADmium `arc`; FeatureScript sketch | **MISSING** | No arc entity ⇒ cannot draw a fillet-free rounded profile, cannot revolve a semicircle into a sphere, cannot round a slot. Forces every curved profile through `AddCircle` (full circles only). |
| 4 | **spline / bezier** | KCL `bezier_curve`/`control_point_spline`; csg `polygon` | **MISSING** | No freeform profile curves; airfoils/cams inexpressible. |
| 5 | **ellipse** | KCL `ellipse`/`elliptic`/`conic`; csg none | **MISSING** | Elliptical bosses/holes must be faked. |
| 6 | **polygon (n-gon / free)** | KCL `polygon`; csg `polygon`/`polyhedron`; OpenSCAD `polygon` | **MISSING** | Only axis-aligned `AddRectangle`; hex/triangle profiles inexpressible without N `AddLine`s (and no way to close/region them). |
| 7 | constrain (coincident/h/v/parallel/perp/distance/radius/equal) | KCL `constraints`/`solver`; freecad; cadquery | **HAS** (`Constrain`, 8 kinds) | — |
| 8 | **constrain (tangent/angle/symmetric/midpoint/concentric/diameter/equal_radius)** | KCL `solver` (all of these); CADmium | **PARTIAL** | `CONSTRAINT_DOF` lacks 7 of KCL's solver constraints; a sketch that needs tangency or symmetry can't have its DOF counted, so the constraint verifier under-reports. |
| 9 | extrude | all backends; KCL; CADmium; FeatureScript | **HAS** (`Extrude`) | — |
| 10 | revolve | cadquery/build123d/rhino3dm/frep/…; KCL `revolve`; FeatureScript | **HAS** (`Revolve`) | — |
| 11 | loft | cadquery/build123d; KCL `loft`; FeatureScript | **HAS** (`Loft`); refused by CSG/F-rep backends | — (refusal is honest) |
| 12 | sweep (along path sketch) | cadquery/build123d; KCL `sweep`; FeatureScript | **HAS** (`Sweep`) | — |
| 13 | **sweep along helix / twist-extrude** | KCL `helix`, engine `entity_make_helix*`, `twist_extrude` | **MISSING** | Threads, springs, augers inexpressible — `Sweep` needs a *sketch* path and CISP has no helical path entity. |
| 14 | fillet (constant radius, edge-selected) | cadquery/build123d/freecad/blender/rhino3dm; KCL `fillet` | **HAS** (`Fillet`) | — |
| 15 | **fillet (variable radius / blend)** | KCL `blend`; OCCT variable fillet | **PARTIAL** | `Fillet.radius` is a single scalar; no per-edge or start/end radius. |
| 16 | chamfer (sym + asym) | cadquery/build123d/freecad/rhino3dm; KCL `chamfer` | **HAS** (`Chamfer`, `distance2`) | — |
| 17 | hole (simple/cbore/csk) | cadquery/build123d/frep/rhino3dm; KCL (via subtract); FeatureScript `hole` | **HAS** (`Hole`) | — |
| 18 | shell / hollow | cadquery/build123d/freecad/manifold/openscad/truck/microcad; KCL `shell`/`hollow` | **HAS** (`Shell`) | — |
| 19 | draft (+ neutral plane) | cadquery(refused)/build123d; KCL n/a; FeatureScript `draft` | **HAS** (`Draft`, `neutral_plane`) | — (backends may refuse, but the op names it) |
| 20 | **thicken / offset-solid** | FeatureScript `thicken`; KCL `offset_surface`/`make_offset_path`; csg `offset`; blender solidify (`OFFSET`); frep native SDF offset | **MISSING** | Sheet-to-solid and grow/shrink-a-solid are core; every family has it; frep and OCCT can do it exactly. |
| 21 | **split / section by plane** | KCL `solid::split`, `csg::split`; engine `plane_intersect_and_project` | **MISSING** | Cut a body in two / keep one side. Trivially a boolean against a half-space in every solid backend. |
| 22 | **project (entity/points → plane)** | KCL engine `project_entity_to_plane`, `project_points_to_plane` | **MISSING** | Deriving a sketch from existing geometry (project an edge to sketch on) — the standard way references chain. |
| 23 | boolean (union/cut/intersect) | all; KCL `csg`; csg all families; FeatureScript | **HAS** (`Boolean`) | — |
| 24 | **hull (convex)** | csg all 5 families (`hull`/`hullChain`); real 2D impl in `csg_vocabulary.convex_hull_2d` | **MISSING** | A non-boolean set operator every CSG family exposes; a boolean-only kernel can't express it. |
| 25 | **minkowski sum** | OpenSCAD/RapCAD `minkowski`, JSCAD `expand`; real 2D impl in `csg_vocabulary.minkowski_sum_2d` | **MISSING** | The other non-boolean operator; used for offsets/clearances; already implemented (2D) in the harness but unreachable as an op. |
| 26 | linear pattern | cadquery/build123d/frep/rhino3dm; KCL `pattern_linear_3d`; FeatureScript `pattern` | **HAS** (`LinearPattern`) | — |
| 27 | circular pattern | cadquery/build123d/frep/rhino3dm; KCL `pattern_circular_3d` | **HAS** (`CircularPattern`) | — |
| 28 | **pattern-on-path / transform pattern** | KCL `pattern_transform`/`pattern_transform_2d` | **MISSING** | Arbitrary per-instance transform patterns (KCL's most general pattern). |
| 29 | mirror | all; KCL `mirror_2d`/`mirror_3d`; csg `mirror` | **HAS** (`Mirror`) | — |
| 30 | **scale / resize** | csg `scale`+`resize` (all families); KCL `scale` | **MISSING** | No non-uniform scale / resize-to-bbox op. csg canon lists both. |
| 31 | translate / rotate (standalone) | csg all families; KCL `translate`/`rotate` | **PARTIAL** | Only via `AddInstance` (assembly placement); no in-place body transform op. |
| 32 | add instance (place part) | build123d/cadquery/rhino3dm/frep | **HAS** (`AddInstance`) | — |
| 33 | mate (rigid/revolute/slider/cylindrical/planar) | cadquery/build123d/frep; `mates.py`; `mobility.py` | **HAS** (`Mate`) | — |
| 34 | **mate/joint (ball/spherical, screw/helical, gear, press-fit, thread, snap)** | `mates.py` `MATE_TYPES` (gear_mesh/press_fit/thread_engage/snap_to_face); `mobility.py::JOINT_FREEDOM` (ball/spherical); ASSEMCAD/ArtiCAD | **PARTIAL** | `Mate.kind` is only validated for 5 kinds in `assembly.py::MATE_DOF`; ball, spherical, gear, press-fit, thread, snap are known to the mined mobility/mates modules but a `Mate(kind="ball")` is *unknown* to the DOF verifier. |
| 35 | **port-typed mate + compatibility gate** | `mates.py` (12 port types, type-compat relation) | **MISSING** | CISP `Mate` couples two ids by name; it has no *port* concept, so the ASSEMCAD port-type admissibility check has nothing in the op stream to run against. |
| 36 | set param (edit a prior op) | onshape (POST feature); everywhere via replay | **HAS** (`SetParam`) | — |
| 37 | **insert / delete op (structural history edit)** | CAD-Editor locate-then-infill; onshape feature delete; KCL `delete`/`hide` | **MISSING** | `SetParam` mutates a field; it cannot *insert* or *remove* an op from the history. CAD-Editor's whole paradigm is span replace (insert+delete of a token run). |
| 38 | **span-mask-infill edit modality** | `locate_then_infill.py` (CAD-Editor) | **MISSING (paradigm)** | A fundamentally different edit model than op-DAG deltas: locate a contiguous span, mask it, infill. Not reducible to `SetParam`. |
| 39 | **discrete voxel / stud brick assembly** | `brick_assembly.py` (BrickGPT) | **MISSING (modality)** | A whole modality CISP lacks: an integer-lattice, stud-connected assembly with buildability predicates (bounds/collision/support/connectivity). No B-rep op reaches it. |
| 40 | **primitive solid: box** | rhino3dm `box`; csg `cube`; KCL n/a (via sketch); truck/manifold internal | **MISSING** | See §3 gap #1. |
| 41 | **primitive solid: sphere** | csg `sphere` (all families); rhino3dm; frep SDF | **MISSING** | Cannot even *approximate* one: needs revolve of a **semicircle**, and CISP has no arc (#3). |
| 42 | **primitive solid: cylinder** | rhino3dm `cylinder`; csg `cylinder`; frep `cyl` node; microcad `cylinder` | **MISSING** | Every geometry backend has an internal cylinder builder (holes use it); no op names a standalone cylinder. |
| 43 | **primitive solid: cone** | csg `cone`; frep/manifold/truck/microcad/openscad/blender internal `cone` node (countersinks use it) | **MISSING** | Same as cylinder — the builder exists in every kernel, unnamed by CISP. |
| 44 | **primitive solid: torus / wedge / ellipsoid / polyhedron** | csg `polyhedron`; KCL; OCCT torus/wedge | **MISSING** | Standard primitive library absent. |
| 45 | **helix as a first-class curve** | KCL `helix` / `involute_circular` | **MISSING** | (See #13.) |
| 46 | GD&T / datum / tolerance annotation | KCL `gdt` module (flatness, position, runout, …) | **MISSING (out of geometry scope)** | Not a mutating geometry op; noted for completeness — a manufacturing-intent layer CISP has no vocabulary for. |
| 47 | appearance / material / color | KCL `appearance`; blender; rhino3dm | **MISSING (non-geometric)** | Cosmetic; low priority. |

---

## 3. Ranked gap list — (how many tools need it) × (how buildable it is)

Ranking weighs **breadth** (distinct integrated tools/modules expressing it) and
**buildability** (can existing backends already do it, or is it new kernel work).

| Rank | Gap | Breadth | Buildability | Score |
|------|-----|---------|--------------|-------|
| **1** | **Primitive solids** (box/sphere/cylinder/cone/torus/wedge) | Very high — all 5 csg families, KCL, rhino3dm, and the *internal* builders already in frep (`cyl`/`cone`), manifold, truck, microcad, openscad, blender | Very high — the builders already exist inside nearly every backend | **Top** |
| **2** | **Split / section by plane** | High — KCL `split`, csg `split`, Onshape engine `plane_intersect` | Very high — a boolean cut against a half-space; every solid backend already has booleans | **High** |
| **3** | **Thicken / offset-solid** | High — FeatureScript `thicken`, KCL `offset_surface`, csg `offset`, blender solidify | Medium-high — exact in frep (native SDF offset) and OCCT (`MakeThickSolid`); Manifold offset is 2D-only (would refuse) | **High** |
| 4 | **Arc sketch entity** | High — KCL, CADmium, FeatureScript | High — every B-rep sketcher has arcs; unblocks sphere-via-revolve | High |
| 5 | **Hull + Minkowski** | Medium-high — all 5 csg families; 2D already implemented in `csg_vocabulary` | Medium — native in CSG/F-rep/mesh backends, hard in pure B-rep | Medium |
| 6 | **Scale / resize transform** | Medium — csg all families, KCL | High — trivial affine in every backend | Medium |
| 7 | **Extended mate kinds** (ball/spherical/gear/press-fit/thread/snap) + **port-typed mates** | Medium — `mates.py`, `mobility.py`, ASSEMCAD/ArtiCAD | High (DOF table only) / Medium (port geometry) | Medium |
| 8 | **Extended sketch constraints** (tangent/angle/symmetric/midpoint/concentric/diameter) | Medium — KCL solver, CADmium | High — just `CONSTRAINT_DOF` entries + solver support | Medium |
| 9 | **Insert/delete history op + span-infill edit modality** | Medium — CAD-Editor, Onshape delete, KCL delete | Medium — new op-log surgery + a second edit paradigm | Medium |
| 10 | **Helix/twist path** | Low-medium — KCL only | Medium — needs a helical curve entity | Low-medium |
| 11 | **Spline/bezier, ellipse, polygon profiles** | Medium — KCL, csg | Medium — sketcher curve work | Low-medium |
| 12 | **Voxel/brick assembly modality** | Low — BrickGPT only | Low — a wholly separate representation; does not lower to B-rep | Low |
| 13 | **GD&T / appearance** | Low-medium (KCL) | N/A — non-geometric annotation layer | Low |

---

## 4. Design for the top gaps (PROPOSED — not added)

### 4.0 Why nothing was added to `ops.py` in this pass

The hard rules permit adding *one* proven-trivial-safe primitive to `ops.py`
**only** with a differential-oracle proof that **every backend** handles it — but
they also forbid editing any backend. A new op is dispatched inside each backend
(`isinstance(op, …)` in cadquery/build123d/rhino3dm/frep; OP-tag lowering in
frep_ir/openscad/manifold/truck/microcad; the SUPPORTED/REFUSED tables in
onshape/freecad). An op added to `ops.py` alone would fall through every
backend's dispatch and could not pass a differential oracle **without** backend
edits. Those two constraints are mutually exclusive, so the correct move is to
**propose designs** and leave the coordinated add to a follow-up that may touch
backends. Stated loudly: **no op was added.**

### 4.1 Gap #1 — `Primitive` (box/sphere/cylinder/cone/torus/wedge)

The single highest-value, lowest-risk gap: the builders already exist inside
nearly every backend, they are just unreachable as ops.

**Proposed dataclass** (single tagged op, shape-discriminated — mirrors the
`Hole.kind` and `Boolean.kind` pattern already in CISP):

```python
@dataclass(frozen=True)
class Primitive(Op):
    """A parametric solid primitive placed at the origin (before any transform).

    ``shape`` selects the family; only the fields that shape uses are read:
      box       -> dx, dy, dz
      sphere    -> r
      cylinder  -> r, h
      cone      -> r, r2 (top radius; 0 = point), h
      torus     -> r (major), r2 (minor)
      wedge     -> dx, dy, dz  (right-triangular prism)
    Unused fields are ignored; a backend that cannot build ``shape`` refuses with
    a typed ``unsupported-op`` (the same discipline Shell/Draft already use).
    """
    OP: ClassVar[str] = "primitive"
    shape: str = "box"        # box|sphere|cylinder|cone|torus|wedge
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    r: float = 1.0
    r2: float = 0.0
    h: float = 1.0
```

**Per-backend mapping (all already have the builder):**

| Backend | Maps to |
|---------|---------|
| cadquery / build123d | `Workplane.box/sphere/cylinder`; OCCT `BRepPrimAPI_MakeBox/Sphere/Cylinder/Cone/Torus/Wedge` |
| rhino3dm | existing internal `box`/`cylinder` helpers (already present); add sphere via `Brep.CreateFromSphere` |
| frep / manifold / truck / microcad / openscad / blender | F-rep IR already has `cyl` and `cone` nodes; box = `rect_exact`-extrude; sphere/torus = SDF prims frep already carries; **CSG backends map box/sphere/cylinder/cone directly** and refuse `wedge`/`torus` only if their alpha lacks it |
| onshape | primitive is a sketch+extrude behind the scenes; either lower to that or add to `REFUSED_OPS` with a precise reason |

**DOF/verifier impact:** none — a primitive is a body, not a sketch, so
`PRIMITIVE_DOF`/`CONSTRAINT_DOF` are untouched. `_REGISTRY` gains one entry;
`parse_op` needs no tuple-field handling (all scalar fields).

**Risk:** low for the pure geometry backends, but **not zero** — onshape and any
query-first backend need an explicit map-or-refuse, so this is a *coordinated*
add, not a solo edit. That is exactly why it is proposed, not committed.

### 4.2 Gap #2 — `Split` (section a body by a plane)

```python
@dataclass(frozen=True)
class Split(Op):
    """Cut the current solid by an infinite plane; keep one or both halves.

    ``plane`` is a named datum (XY|XZ|YZ) or a point+normal 6-tuple
    (px,py,pz,nx,ny,nz). ``keep`` is 'positive' (normal side), 'negative', or
    'both' (yields two bodies). A pure boolean against a half-space.
    """
    OP: ClassVar[str] = "split"
    plane: str = "XY"
    offset: float = 0.0
    keep: str = "positive"   # positive|negative|both
```

**Per-backend mapping:** every solid backend already has booleans and a
half-space: cadquery `Workplane.split(keepTop/keepBottom)`; OCCT
`BRepAlgoAPI_Section`/`Splitter`; frep/manifold/truck/openscad/microcad =
`Boolean(cut)` of the body with a large half-space box on the discard side (the
half-space is a `box`/`rect_exact` node they already build); KCL `solid::split`;
Onshape = `REFUSED_OPS["split"]` ("split needs a face/plane query"). **Risk:
low** — reuses the boolean path that is already differential-tested.

### 4.3 Gap #3 — `Thicken` (sheet→solid / offset-solid)

```python
@dataclass(frozen=True)
class Thicken(Op):
    """Give a surface/face-set a wall thickness, or grow/shrink a solid.

    ``faces`` selects the surfaces to thicken (empty = the whole last body's
    outer surface -> an offset-solid). ``thickness`` may be negative (inward).
    ``both`` thickens symmetrically about the surface.
    """
    OP: ClassVar[str] = "thicken"
    faces: tuple = ()
    thickness: float = 1.0
    both: bool = False
```

**Per-backend mapping:** frep — native SDF offset (exact, cheap); OCCT
(cadquery/build123d/freecad) — `BRepOffsetAPI_MakeThickSolid` /
`makeOffsetShape`; blender — Solidify modifier (`OFFSET` already referenced);
KCL — `offset_surface`; **Manifold refuses** (its offset is 2D-only — an honest
`unsupported-op`, exactly like it already refuses non-prism shells); microcad
refuses (alpha). **Risk: medium** — several backends refuse, so it needs the
same map-or-refuse coordination as Shell.

---

## 5. Cross-cutting observations

* **The F-rep backends already own a primitive library CISP never surfaces.**
  `frep_ir` node kinds are `extrude, cyl, cone, revolve, bool, shell, blendcut,
  blend, mirror, pattern`; `cyl`/`cone` exist only because holes and countersinks
  need them. Exposing a `Primitive` op is pure *plumbing*, not new geometry —
  the strongest argument that gap #1 is the right first move.

* **`csg_vocabulary` already ships real `hull` and `minkowski` (2D).** Two
  non-boolean operators every CSG family exposes are implemented and unit-clean
  in the harness but have no CISP op to invoke them (gap #5).

* **Two mined modules are whole modalities, not missing ops.** BrickGPT
  (`brick_assembly.py`, discrete voxel/stud lattice) and CAD-Editor
  (`locate_then_infill.py`, span-mask-infill) do not lower to the B-rep op-DAG at
  all. They are correctly kept as *sidecar representations*; "adding an op" would
  not reconcile them. If the harness wants them first-class, they need a second
  op family (a `BrickOp` stream) and a second edit paradigm (`InfillEdit`),
  respectively — a larger design than a primitive.

* **The assembly stack is the most internally inconsistent surface.**
  `Mate.kind` is validated by `assembly.py::MATE_DOF` (5 kinds), while the mined
  `mobility.py::JOINT_FREEDOM` (9 kinds incl. ball/spherical/free) and
  `mates.py::MATE_TYPES` (7 typed mates incl. gear/press-fit/thread/snap) know
  strictly more. A `Mate(kind="ball")` is a legal op that the DOF verifier treats
  as unknown. Reconciling these three tables (gap #7) is low-risk table work and
  should ride alongside any assembly change.

* **No arc = no sphere.** CISP cannot revolve a semicircle (no arc entity), so it
  cannot even *approximate* a sphere from its existing ops — which is why the
  `Primitive` sphere (gap #1) and the arc entity (gap #4) reinforce each other.

---

## 6. Bottom line

CISP is **complete for the common denominator** — the sketch → constrain →
extrude/revolve/loft/sweep → fillet/chamfer/hole/shell/draft → pattern/mirror →
assemble/mate/set-param history that every integrated B-rep tool shares. It is
**incomplete** wherever an integration reaches past that denominator: primitive
solids, split, offset/thicken, hull/minkowski, richer sketch curves and
constraints, scale, helix, extended joints, structural history edits, and the two
non-B-rep modalities (bricks, span-infill).

The three lowest-risk, highest-value closures are **`Primitive`**, **`Split`**,
and **`Thicken`** — designed above. All three are backed by builders that already
exist inside the backends; the only reason they are proposed rather than added is
that naming a new op is a schema change every backend's dispatch and the
field-liveness census depend on, which the hard rules (correctly) require be done
as a coordinated change, not a solo edit. **No op was added to `ops.py`.**
