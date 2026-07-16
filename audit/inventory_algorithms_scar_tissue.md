# Deep inventory: buried geometry algorithms, kernel scar tissue, format corner cases

Read-only sweep (2026-07-16) of resources/cad_repos (~125 repos) + OpenCAD-main.
Keyword greps (workaround/OCCT/epsilon/fuzzy/malformed/SI_UNIT/ShapeFix/Sewing) +
directory scans (geom/math/utils/repair/offset/nurbs/boolean/watertight), then full
reads of the highest-signal files, each checked against domain/geometry/*,
domain/numeric/*, io/backends/*, io/formats/*, eval/reliability/*.

## Top-10 highest-value finds (ranked)

1. **Brepler ShapeFix/tolerance-escalation ladder**
   (`Brepler-main/network/post/utils.py`, 1200+ ln) -- the densest OCCT scar tissue
   found: escalating tolerance ladders (EDGE_FITTING 1e-3..5e-2, CONNECT 2e-3..8e-2,
   SEWING 1e-1), full ShapeFix_Wire/Face flag recipe (FixSmall, FixGaps3d,
   FixMissingSeam, FixShifted, DISABLE FixIntersectingWires), periodic-surface
   cylinder-seam 2-edge special case. Upgrades eval/reliability/brep_repair.py
   (currently one generic ShapeFix_Shape at one tolerance). AutoBrep corroborates:
   sew tolerance for GENERATED geometry is 1e-2, not 1e-6.
2. **kerf `geom/mesh_repair.py`** (1908 ln, pure Python) -- weld_vertices,
   unify_normals (BFS dual graph), fill_holes (boundary-loop fan), remove_degenerate,
   QEM decimate, mesh_offset, full mesh_boolean, is_closed/is_manifold, never-raise
   dict contract. Fills the largest hole in domain/geometry/mesh/ (we have smoothing/
   quality/intersection_repair/winding -- no weld/hole-fill/decimate/boolean).
3. **STEP unit-scale extraction gap** -- the harness has ZERO SI_UNIT handling
   (grep: no hits). cadquery importer sets `xstep.cascade.unit=MM` to rescale;
   kerf's step reader resolves `UNCERTAINTY_MEASURE_WITH_UNIT` + SI_UNIT prefix from
   the DATA section. Without this, m<->mm silent 1000x errors -- the classic
   text-to-CAD failure. Build: `io/formats/step_units.py` (small, high payoff) +
   a unit-sanity verifier.
4. **kerf `sew.py` + `body_heal.py`** -- tolerant face->shell sewing with tolerance
   monotonicity (vertex.tol >= edge.tol >= face.tol; merges never narrow tolerance),
   heal pass (sub-tolerance removal, sliver snap within 10x tol). The OCCT-free
   counterpart to brep_repair. Build: domain/geometry/topology/sew.py.
5. **Roshera `lifecycle.rs` transactional ops** -- validate_can_apply pre-flight +
   with_rollback snapshot restore for every mutating kernel op (documented failure
   modes: half-done fillet, boolean-degeneracy orphaned faces). Pattern for
   io/backends/base.py: byte-equivalent state on failure.
6. **cadgenbench "no OCCT booleans for metrics" policy** -- OCCT booleans HANG on
   interface-overlay geometry; policy = manifold3d-only for metric booleans,
   ENFORCED BY A TEST that greps for OCCT imports; sub-epsilon overlap classified as
   numerical noise. Adopt in eval/verifiers.
7. **Roshera cyl-cyl saddle-boolean refusal guard** (`modify.ts`) -- adjacent holes
   with chord spacing <= 2*hole_r produce a known-open kernel bug; refuse loudly with
   the exact formula 2*ring_r*sin(pi/count). Cheap feasibility predicate for
   pre-refusal.
8. **kerf `offset.py`** -- curve/surface/loop offsets with corner cases (convex ->
   arc fillet, concave -> extend/trim), analytic exact cases, refit with
   actual_max_deviation reporting. Our path_offset.py is 2D polyline miter only.
9. **cadquery kernel-quirk set** -- SetFuzzyValue on every boolean builder;
   0-degree revolve must be rewritten as 360; infinite faces have center (1e99,1e99).
   Three concrete rows for an OCCT quirk catalog in error_patterns/backends.
10. **kerf `isotropic_remesh.py`** (Botsch-Kobbelt, boundary-preserving) + **muse
    subprocess-isolated STEP checking** (timeout so malformed STEP cannot crash the
    parent; bbox pre-filter interpenetration metric) -- remesher fills a real gap;
    subprocess isolation is a one-day ingest hardening.

Also notable: kerf fillet_solid/chamfer with radius-feasibility predicate (expose to
codegen as pre-flight); kerf surface_boolean_robust (health-check + bounded retry
ladder + structured via/attempts/reason result); kerf SSI/trim/mesh->NURBS; Roshera
polygon-clip degeneracy taxonomy + coplanar-face boolean imprint + math_oracle
harness idea; OpenCAD BBOX_NEAR_TANGENT pre-flight (<1e-6 overlap = unstable,
refuse); forgent3d outer_edges_at_z length-ratio filter (drop tiny edges before
fillet); Zoo/KCL revolve-touching-axis bug workaround (revolve one + pattern);
kerf DXF $INSUNITS code table; muse/copilot malformed-STEP status-vs-empty-roots
distinction.

## Skips (verified)

SKIP-exists: instant-mesh-intersection-repair (intersection_repair.py is an explicit
port), SolidPython 2D offsets (path_offset.py port), sdfx DC/MC (volumes/*), ruststep
(step_header.py cites it), CADmium/DeepCAD EPSILON=1e-6 (subsumed by cadquery find),
cadsmith fillet-clamp prompts. SKIP-irrelevant: ComplexGen vendored GTE (C++),
solvespace SSI (C++/GPL -- reference only), manifold boolean3 (spec reference only),
Text-to-CAD-dean (venv noise).

## Honest coverage note

Fully read ~25 files, header-read ~20 more, across ~30 of 125 repos; the rest covered
by greps/dir scans. Large non-Python kernels (libfive, curv, ImplicitCAD, OpenJSCAD
internals, RapCAD, angelcad/xcsg, scad-hs/clj, replicad, CADmium Rust, truck) sampled
only. **oce-oce-patches is effectively UNSWEPT** -- OCCT devs annotate with reviewer
tags (//:abv, //:pdn, skl) not the word "workaround"; the most likely remaining
scar-tissue trove. **kerf-main (~69k lines geom alone, 50+ sibling packages -- tess/
topo/imports only skimmed) and Roshera-CAD dwarf everything else and read like sibling
verifier-first projects**; verdicts based on headers/contracts, not line-level
verification. Harness "NONE" claims based on targeted greps + dir listings; a module
hidden under an unrelated name could have been missed.
