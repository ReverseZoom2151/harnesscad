# Deep read: geometry kernels

Genuine deep read, not a grep sweep. Every claim below was re-derived against the
actual sources in `resources/cad_repos/` and checked against the harness with
`registry.index()` (1579 modules) plus targeted greps of `src/harnesscad/`.

Prior inventories described this set as "sampled only". They were wrong in both
directions, so nothing here is inherited — where a prior claim is contradicted it
is called out explicitly with evidence.

Volume is reported honestly per repo: how many files were read in full, how many
were header- or grep-indexed, out of what total. The word "sampled" does not
appear. **"Nothing here" is a result**, and three repos below get it.

---

## manifold-master (394 files; 103 .cpp/.h; 17 .obj models; 4 polygon corpora)

LICENSE: **Apache-2.0** (`LICENSE:1-3`). Permissive -> vendorable with attribution.
The harness already does this correctly in `eval/corpus/fixtures/manifold_meshes.py`
(MANIFEST.json + LICENSE-NOTICE.txt).

WHAT IT IS: the guaranteed-manifold mesh boolean kernel. **This checkout is newer
than the one the prior audit saw** — it contains a full `boolean2` 2D-arrangement
rewrite (`boolean2.cpp/h`, `boolean2_predicates.cpp`, `boolean2_winding.cpp`,
`boolean2_offset.cpp`, `docs/Boolean2.md`), Clipper2 is gone, and there is a new
`ExecutionContext` cancellation API. Copyright headers say 2026.

The prior audit filed this as "spec reference only". That call is now stale and
wrong: the harness adopted a manifold3d-only policy for metric booleans in
`cca2e3e`, so Manifold's actual semantics are load-bearing.

READ (in full): `include/manifold/manifold.h`, `include/manifold/common.h`,
`src/boolean3.cpp` (656L), `src/boolean3.h`, `src/boolean2.h` (200 of 332L),
`src/boolean2_diagnostics.h`, `docs/Boolean2.md` (144L), `src/impl.h` L285-500,
`src/properties.cpp` L100-260, `src/impl.cpp` L655-700, `src/utils.h` L120-162,
`test/manifold_fuzz.cpp`, `test/polygon_fuzz.cpp` L1-60, plus targeted full reads
of `test/boolean_complex_test.cpp` L1455-1560, `test/boolean_test.cpp` L415-445 /
L780-830, `test/cross_section_test.cpp` L520-620 / L1000-1030 / L1125-1150 /
L1560-1620, `test/properties_test.cpp` L25-60, `src/csg_tree.cpp` L225-275,
`src/boolean2_predicates.cpp` L88-128.

SKIMMED-NOT-READ: structurally grepped, not read line by line —
`boolean_result.cpp` (979L, read its 100-comment index + all `Error::` sites),
`edge_op.cpp` (947L, all 80 doc-comments, not bodies), `manifold.cpp` (1158L,
Boolean/tolerance/Simplify sections + all `Error::` sites), `sort.cpp`,
`smoothing.cpp`, `quickhull.cpp`, `subdivision.cpp`, `sdf.cpp`, `polygon.cpp`,
`minkowski.cpp`, `collider.h`, `parallel.h` (1221L), `linalg.h` (2454L), `math.h`
(527L — confirmed no kernel epsilons, it is a musl libm port). Bindings and CMake
ignored by policy. **Honest ratio: ~14 of 103 source files read in full or
near-full; the rest header-scanned or grep-indexed.**

FINDINGS

1. **Manifold's boolean has NO geometric epsilon — it uses symbolic perturbation
   keyed on the op type.** `src/boolean3.cpp:473-481`, constructor comment:
   *"Symbolic perturbation: Union -> expand inP, expand inQ; Difference,
   Intersection -> contract inP, expand inQ. Technically Intersection should
   contract inQ, but doing it this way makes Split faster and any suboptimal cases
   seem pretty rare."* `expandP_(op == OpType::Add)` at L475, threaded as a
   template parameter through `Shadow01`/`Kernel02`/`Kernel11`/`Kernel12`.
   Why: this is the semantic the new manifold3d-only policy depends on, and it is
   counter-intuitive. Coincident/coplanar faces are resolved by a **sign
   convention keyed on the op**, not a tolerance. `A + B` and `A ^ B` resolve a
   shared coplanar face differently and deliberately, and intersection is
   *knowingly* asymmetric (P contracted, Q expanded), so `A ^ B` and `B ^ A` are
   not guaranteed bit-identical. Any harness invariant asserting boolean
   commutativity on interface-overlay geometry asserts something Manifold does not
   promise.
   Harness equivalent: **NONE**, and worse — `io/backends/manifold.py` L1-70
   describes the algorithm as *"exact-predicate mesh booleans over a collider"*,
   which is **factually wrong**: `boolean3.cpp` uses no exact predicates, it uses
   float `Shadows()` tests disambiguated by symbolic perturbation. Verified by
   grepping `src/harnesscad/` for "symbolic perturbation" (0 hits) and reading the
   backend docstring in full. Stdlib-portable: yes (a documented convention plus a
   docstring fix, not code).

2. **`kPrecision = 1e-12` and the epsilon/tolerance two-tier model, with
   propagation rules.** `src/utils.h:39`; `src/shared.h:28-29`
   (`MaxEpsilon(minEpsilon, bBox) = max(minEpsilon, kPrecision * bBox.Scale())`);
   `src/impl.cpp:684-691` (`SetEpsilon`: `tolerance_ = max(tolerance_, epsilon_)`,
   and for float32 input `minTol = max(minTol, FLT_EPSILON * bBox.Scale())`);
   `src/impl.cpp:660-663` transform rule (`epsilon_ *= SpectralNorm(mat3)` then
   re-floored, *"Maximum of inherited epsilon loss and translational epsilon
   loss"*); `src/csg_tree.cpp:243-249` CSG-combine rule
   (`nodeEpsilon *= max(1, newScale/oldScale)`, floored at `kPrecision*newScale`,
   combined node takes the **max** over children; non-finite -> -1).
   Why: two distinct quantities. `epsilon_` = how far a vertex may have moved
   (validity); `tolerance_` = what counts as coplanar/short for simplification,
   always >= epsilon (`manifold.cpp:380-382`). Both are **bbox-relative, not
   absolute**, so a part modelled 1 m from origin has 100x the epsilon of the same
   part at origin — confirmed by the committed test `test/properties_test.cpp:33-41`
   (`cube.Scale({0.1,1,10})` -> `10*kPrecision`; then `.Translate({-100,-10,-1})`
   -> `100*kPrecision`). Directly actionable: **model at the origin**, and expect
   boolean fidelity to degrade linearly with distance from it.
   Harness equivalent: **NONE for the propagation rules.** Verified: greps for
   `kPrecision`, `symbolic perturbation`, `epsilon.valid` return zero; the 40+
   `1e-12` hits in the harness are unrelated local guards (`lm_solver`,
   `offset_nurbs`, `mates`), none bbox-relative. `io/backends/manifold.py` never
   reads `GetTolerance`/`GetEpsilon`. Stdlib-portable: yes.

3. **The `Manifold::Error` taxonomy — 15 named refusal codes — collapsed by the
   harness to one string.** `include/manifold/manifold.h:122-138`: `NoError,
   NonFiniteVertex, NotManifold, VertexOutOfBounds, PropertiesWrongLength,
   MissingPositionProperties, MergeVectorsDifferentLengths,
   MergeIndexOutOfBounds, TransformWrongLength, RunIndexWrongLength,
   FaceIDWrongLength, InvalidConstruction, ResultTooLarge, InvalidTangents,
   Cancelled`. Exposed in Python at `bindings/python/manifold3d.cpp:826-834`.
   The raising sites are the real content: `src/impl.h:304` (`numVert < 4 ||
   numTri < 4` -> `NotManifold`, so a 3-triangle mesh is refused outright);
   `impl.h:454` (`vert >= numVert` -> `VertexOutOfBounds`); `impl.h:477`
   (post-`CreateHalfedges` `!IsManifold()` -> `NotManifold`); `impl.h:337/348/355`
   (length mismatches); `impl.h:~340` (`!all_of(isfinite)` -> `NonFiniteVertex`,
   separately for `vertProperties`, `runTransform`, `halfedgeTangent`, the latter
   two mapping to `InvalidConstruction`); `src/boolean_result.cpp:780`
   (`ResultTooLarge` when an intersection list exceeds `INT_MAX`, set at
   `boolean3.cpp:519-522`); `src/smoothing.cpp:1099` (`InvalidTangents`). Status
   **propagates**: `boolean_result.cpp:735-741` short-circuits if either operand is
   already in error; `csg_tree.cpp:237-240` propagates through the whole tree.
   Harness equivalent: **NONE.** `io/backends/manifold.py` defines one
   `ManifoldError(RuntimeError)` (L135) raised at 6 sites with free text and
   **never calls `.status()`**. Registry grep for "manifold" returns 8 modules,
   none an error taxonomy. Stdlib-portable: yes — an enum plus a dispatch table.

4. **"This hangs on X" is real: Manifold's own fuzzers wrap every boolean in a
   10-second watchdog.** `test/manifold_fuzz.cpp:71-87` — each boolean runs on a
   detached thread with `asyncFuture.wait_for(10000ms)`, and on timeout
   `pthread_cancel(tid)` + `printf("timeout after %dms...")`. Same pattern in
   `test/polygon_fuzz.cpp:52-59` for `Triangulate` (timeout `max(size, 10000)` ms).
   Why: this **qualifies** the harness's just-adopted "manifold3d because OCCT
   hangs" policy. Manifold's own authors treat unbounded boolean runtime as a live
   failure mode requiring a watchdog. The swap reduces hang probability; it does
   not eliminate hangs. `docs/Boolean2.md` names the mechanism for the 2D side:
   *"Splitting first can make the sub-edge broad phase super-quadratic on dense
   near-collinear input, which has not mattered in practice."* Near-collinear dense
   input is precisely interface-overlay geometry.
   Harness equivalent: **NONE.** Grepping `src/harnesscad/io/` for
   `timeout|watchdog|hang` yields only a docstring in `backends/manifold.py:99`
   claiming *"it never hangs"* — which refers to the import stub, not to boolean
   execution. There is no time budget around any manifold3d call.
   Stdlib-portable: yes (`concurrent.futures` / `signal` / subprocess timeout).

5. **What Manifold silently repairs vs refuses — the ingest pipeline, in order.**
   `src/impl.h:466-492`, run unconditionally on every `Manifold(MeshGL)`.
   Silently repaired: degenerate index triangles *dropped* before halfedge
   construction (`impl.h:~448`); then `CleanupTopology()` (`edge_op.cpp:105-107`:
   *"Duplicates just enough verts to convert an even-manifold to a proper
   2-manifold, splitting non-manifold verts and edges with too many triangles"*;
   L113-114: *"In the case of a very bad triangulation, it is possible to create
   pinched verts. They must be removed before edge collapse"*), `DedupePropVerts()`,
   `RemoveDegenerates()`, `RemoveUnreferencedVerts()`. Refused: only
   `!IsManifold()` after halfedge construction. Note the ordering — refusal happens
   *before* the repair passes, so a mesh repair could have fixed is still rejected
   if its halfedge structure is inconsistent.
   Why: a precise contract for what to pre-clean before handing a mesh to
   manifold3d and, inverted, the list of defects manifold3d **silently absorbs
   without telling you** — a verifier blind spot.
   Harness equivalent: **PARTIAL** — `domain/geometry/mesh/repair_toolkit.py`,
   `sew.py`, `halfedge.py` (summary: *"Half-edge triangle mesh with Manifold's
   manifoldness invariants"*) cover the invariants; no module encodes the ordering
   or the silent-drop list. Stdlib-portable: yes.

6. **The epsilon-stacking rule in edge collapse.** `src/edge_op.cpp:172-193`:
   *"Short edges get to skip several checks and hence remove more classes of
   degenerate triangles than flagged edges do, but this could in theory lead to
   error stacking where a vertex moves too far. For this reason this is restricted
   to epsilon, rather than tolerance. However, in the case of a Boolean operation,
   we set `firstNewVert` in order to only operate on newly-created verts, which
   means error stacking is not a concern, so we allow collapsing up to tolerance in
   that case."* Plus L219-224: *"Colinear is defined not by a local check, but by
   the global `MarkCoplanar` function, which keeps this from being vulnerable to
   error stacking."* And L130-133: an edge collapse that would be non-manifold
   duplicates vertices so as to remove handles, **decreasing the `Genus()`** — i.e.
   simplification can silently change topology.
   Two transferable rules: (a) old verts may move by <= epsilon, new verts by <=
   tolerance, and you must track which is which; (b) colinearity/coplanarity must
   be decided **globally, once**, never by a local per-edge test, or errors compound.
   Harness equivalent: **NONE explicit** — `isotropic_remesh.py` and `smoothing.py`
   exist; greps for `firstNewVert`, `error stacking` return 0. Stdlib-portable: yes.

7. **`boolean2`'s crossing predicate is sign-based, not epsilon-based.**
   `docs/Boolean2.md`, "Regularization And Epsilon": *"Whether two segments cross is
   a sign decision: a crossing exists where each strictly straddles the other over a
   positive-width shared projection interval, with no epsilon band on nearness to an
   endpoint. A crossing that lands within epsilon of an endpoint is kept and snapped
   to that endpoint at insertion, not rejected. Orthogonal-coordinate ties within
   epsilon are treated as symbolic ties, not raw CCW fallbacks: the tie policy first
   uses canonical segment geometry, then falls back to stable edge ID."* Backed by
   `GraphSegment2D::stableEdgeId` at `src/boolean2.h:~50`, commented *"Must come
   from a deterministic source, not BVH pair order."* Also: *"Repeated `Simplify()`
   calls are not part of the public contract"* — idempotence explicitly disclaimed.
   Why: the correct pattern for any arrangement/planar-boolean code — decide
   *existence* by sign, *identity* by epsilon, break ties by a stable ID rather than
   traversal order. The non-idempotence disclaimer is a fact the harness's
   differential oracle must not assume away.
   Harness equivalent: **NONE** — registry greps for `arrangement`, `planar
   arrangement`, `bentley` return 0. Stdlib-portable: yes.

8. **`EpsilonFromScale` — a derived, non-magic 2D epsilon.**
   `src/boolean2_predicates.cpp:111-118`:
   `EpsilonFromScale(L, k_budget=1000) = ldexp((k_budget+1) * kAlphaCoeff * kU, frexp(L).exp)`
   with `kU = 1.110223024625156540423631668e-16` (2^-53, double unit roundoff) and
   `kAlphaCoeff = 12.37` at `src/boolean2.h:44-45`. Comment: *"Choose epsilon from
   the operation scale using Smith's rounded power-of-two length bound and the
   caller's adjustment budget."*
   Why: an epsilon *derived* from unit roundoff x error-growth constant x operation
   budget x power-of-two scale, rather than picked. This is the honest way to
   justify a tolerance; the harness's constants are currently hand-picked.
   Harness equivalent: **NONE.** Stdlib-portable: yes — `math.frexp`/`math.ldexp`.

9. **`IsSelfIntersecting` is a *tolerant* predicate with a two-sided
   normal-displacement escape.** `src/properties.cpp:136-190`: `ep = 2*epsilon_`;
   any two triangles with **any** vertex pair within `ep` are skipped (adjacency
   relaxed to a distance test, L155-158); then when
   `DistanceTriangleTriangleSquared == 0` it tries **four** displacements (+/-`ep`
   along each triangle's normal, applied to the *other* triangle) and declares "not
   intersecting" if any of the four separates them. Only if all four fail is it
   flagged. Docstring: *"Note that this is not checking for epsilon-validity."*
   Also: `ExecutionParams::selfIntersectionChecks` is documented in `common.h` as
   *"For debug purposes only"* and is **off by default** — manifold3d does not
   validate its own boolean output for self-intersection in production.
   Harness equivalent: **PARTIAL** — `mesh/triangle_intersect.py` (exact predicate)
   and `intersection_repair.find_self_intersections` do a hard binary test with
   `ignore_shared_vertices` on *index* sharing, not distance. Stdlib-portable: yes.

10. **Committed known-bad and known-good fixtures with named provenance.**
    `test/models/` (17 .obj): `self_intersectA/B.obj`,
    `openscad-nonmanifold-crash.obj`, `Cray_left/right.obj`,
    `Generic_Twin_7081/7863_left/right.obj`, `Havocglass8_left/right.obj`,
    `hull-body/mask.obj`, `Offset1-4.obj`; `test/polygons/` (`polygon_corpus.txt`,
    `sponge.txt`, `zebra.txt`, `zebra3.txt`). Machine-checkable expectations at
    `test/boolean_complex_test.cpp:1455-1560` — `CraycloudBool` asserts `res` is
    non-empty but `res.AsOriginal().Simplify()` **is** empty (a zero-volume shell);
    several assert only "does not crash" (`res.GetMeshGL(); // test crash`), itself
    an honest ground truth. `ManifoldParams().processOverlaps = true` is required
    for `Ring`, `SelfIntersect`, `GenericTwin7863`, `Havocglass8` — those four fail
    the CCW assertion otherwise.
    Harness equivalent: **ALREADY COVERED** — `eval/corpus/fixtures/manifold_meshes.py`
    (346L) vendors these with SHA-256 provenance and *measured* rather than assumed
    labels (it documents that `openscad-nonmanifold-crash.obj` has clean index
    topology but 717 verts on 658 positions -> 18 incidence-4 edges after welding).
    **Not ported:** `test/polygons/sponge.txt`, `zebra.txt`, `zebra3.txt` (the
    fixture module parses only `polygon_corpus.txt`), and the per-fixture
    `processOverlaps` requirement.

11. **Two new adversarial 2D corpora the fixture module predates.**
    `test/cross_section_test.cpp` now carries fuzz-seed regression tables with
    per-row prose diagnoses: `kPrismSeeds` (L1128-1147 — two near-identical
    equilateral triangles, radii differing by 6.88e-13, whose prism union volume is
    off by 0.26 absolute); the distributivity table (L1560-1620, named seeds
    `ZerosInANonzeroUnion`, `RightOverMerges`, with exact failure magnitudes —
    *"left has 1 contour, right has 3, right strictly contained in left, missing
    ~4001 area units out of ~16255 (~25%)"*); the commutativity table (L1000-1030,
    `MixedScaleStars`, `VeryMixedStars` — *"A+B = 11.12 but B+A = 72.75 - off by
    ~62"*). Plus `TEST(CrossSection, DISABLED_CenteredSubEpsNonClosingWalk)` (L529)
    and `NearCoincidentCornersNonClosingWalk` (L555) with full root-cause
    write-ups, and `TinyEdgeFeatureKeepsSquare` (L~600: a 2^22 x 2^22 square that a
    crossing-merge bug **deleted entirely**). `test/cross_section_offset_corpus_test.cpp`
    (95L) is a separate offset corpus.
    Why: algebraic-law property tests (commutativity, distributivity, monotonicity,
    inclusion-exclusion) over adversarial numeric inputs with stated tolerances and
    **stated known-failure magnitudes** — a ready-made property battery, and more
    valuably a list of laws a real kernel *violates*.
    Harness equivalent: **NONE** — `manifold_meshes.py` covers 3D meshes plus
    `polygon_corpus.txt` only. Stdlib-portable: yes (literal coordinate tables).

12. **`ExecutionContext` — cancellation with documented granularity and an
    irreversibility trap.** `include/manifold/common.h:~195-250`. Granularity is
    spelled out: *"Boolean trees check per sub-boolean (so a single very large
    boolean may run to completion before the next check); Hull checks at the
    boundaries of its main phases; Minkowski checks per face of the first input and
    per internal BatchBoolean batch."* The trap: *"once `Cancel()` has been called
    on a context, every subsequent evaluation with that context (or any copy of it)
    will short-circuit to `Error::Cancelled`."* Only eager ops observe ctx
    (`Status`, `Refine*`, `Hull`, `Minkowski*`); deferred ops (`+ - ^`, transforms,
    `BatchBoolean`) ignore it and drop the attachment.
    Why: the supported way to bound a boolean — but the granularity caveat means it
    **cannot interrupt a single large boolean**, reconfirming finding 4 that an
    external watchdog is still required. Harness equivalent: **NONE.**
    Stdlib-portable: n/a (an API-usage fact). Worth checking whether the pinned
    `manifold3d` wheel exposes it at all.

ALREADY COVERED: `eval/corpus/fixtures/manifold_meshes.py` (models + polygon corpus,
with better-than-source labelling); `domain/geometry/mesh/bvh.py` (Collider);
`mesh/halfedge.py`; `mesh/winding_number.py` (the `Kernel02` winding cascade);
`domain/numeric/exact_predicates.py`; `domain/geometry/volumes/marching_tetrahedra.py`
(LevelSet); `io/backends/manifold.py` (op mapping + refusals).

VERDICT: **mine-further** — the highest-value repo in this set. Findings 1-4 are
load-bearing for the just-adopted manifold3d-only policy, and finding 1 exposes a
factual error in the harness's own backend docstring.

---

## sdf-csg-master (40 files; 13 .ts of which 5 are source and 8 generated .d.ts)

LICENSE: **Unlicense** (public domain, `LICENSE:1`). Vendorable; attribution
optional, and the harness already gives it.

WHAT IT IS: a small TypeScript SDF-CSG library — an `SDF` base class with operator
chaining, ~14 analytic IQ primitives, hard and smooth (polynomial `smin`) booleans,
transform/round modifiers, per-primitive user data interpolated across blends, and
a dual-contouring-style isosurface extractor.

READ (in full): `src/sdf.ts` (457L — the entire CSG algebra, all six combinator
classes, `Transform`, `Round`, `generateGrid`, `generateMesh`), `README.md` (167L),
`src/isosurface.ts` L1-40, `src/primitives.ts` via full class index (all 14 classes
enumerated), `LICENSE`.

SKIMMED-NOT-READ: `src/isosurface.ts` L40-134 (vertex-placement body),
`src/primitives.ts` bodies (261L — class list read, each checked against the
harness port), `src/util.ts` (22L, trivial `clamp`/`mix`), `example/*.ts`, `lib/*`
(generated), `docs/assets/*.js` (bundles). Build config excluded per policy.

FINDINGS

1. **Nothing of substance remains.** The three things this repo has — the IQ
   smooth-min combinator family, the "exotic" analytic primitives, and user-data
   interpolation across a blend — are **all already ported with source attribution
   naming this exact repo**. Verified concretely: `domain/geometry/sdf/extra_shapes.py`
   registry summary reads *"Additional analytic signed-distance primitives from
   `sdf-csg`"* and its function list (`box_frame`, `capped_torus`, `link`,
   `hexagonal_prism`, `triangular_prism`, `solid_angle`) matches the non-trivial
   half of `primitives.ts`, with `Box`/`Sphere`/`Torus`/`Cone`/`Capsule`/
   `CappedCylinder`/`CappedCone` covered by `sdf/primitives.py` from Curv.
   `domain/geometry/volumes/surface_nets.py:232-247` `interpolate_attribute`
   docstring reads *"Reproduces sdf-csg's user-data interpolation"* and implements
   exactly `frac = |d1|/(|d1|+|d2|)` from `sdf.ts:176-186`. `sdf/combinators.py:100`
   cites `iquilezles.org/articles/smin` and implements the same polynomial smin as
   `SmoothUnion.density`.

2. (Non-finding, recorded for honesty) **There is a real bug in the source.**
   `src/sdf.ts:75-84`: `generateGrid` allocates `(r0+1)(r1+1)(r2+1)` floats but
   indexes with `r0*r2*j + r0*k + i` — strides use `r0`/`r2`, not `r0+1`/`r2+1`, so
   the grid is written out of stride and the top slab is never addressed. Not scar
   tissue, just a defect, but it means anything derived from this repo's *sampling*
   code (as opposed to its *field* code) should not be trusted. The harness's
   `surface_nets.py`/`marching_cubes.py` do their own sampling, so there is no
   exposure.

ALREADY COVERED: `domain/geometry/sdf/extra_shapes.py`, `sdf/combinators.py`,
`sdf/field_transforms.py`, `sdf/symmetry.py`, `domain/geometry/volumes/surface_nets.py`,
`volumes/dual_contouring.py`, `dual_contouring_3d.py`, `marching_cubes.py`.

VERDICT: **already-covered / nothing-here.** No epsilons with reasons, no refusal
predicates, no fixtures, **no test suite at all** (zero test files in the repo), no
error taxonomy. A clean small demo library, mined thoroughly on a prior pass.

---

## instant-mesh-intersection-repair-master (39 files; 22 .py, 10 .obj, 2 configs)

LICENSE: **NONE.** Verified — no `LICENSE`/`LICENCE`/`COPYING` at root, no license
header in any `.py`. README gives only a BibTeX citation (Jang, Jung, Lee & Lee,
*Instant Self-Intersection Repair for 3D Meshes*, ACM TOG 44(4), 2025) and a
"contact the author for benchmark data" line. -> **facts-with-citation only; do not
vendor code or the `data/misc/*.obj` meshes.** The harness's existing port is
compliant: `intersection_repair.py` cites the paper, re-derives rather than copies,
and vendors none of the `.obj` files.

WHAT IT IS: the official CUDA/PyTorch implementation of a penetration-energy
gradient-descent self-intersection repair — BVH collision detection -> penetration
energy -> Laplacian-preconditioned optimizer step -> re-detect -> stop at zero.

READ (in full): `README.md` (167L), `constraints/volume.py` (66L),
`constraints/area.py` (60L), `constraints/curvature.py` (29L),
`optimizers/MomentumBrake.py` (126L), `energies/conical.py` (51L),
`energies/distance.py` (41L), `energies/TPE.py` L1-120 of 401 (the `TPE` and
`signed_TPE` formulations in full), `repair_factory.py` via full def/control-flow
index (449L), `mutils/meshes.py` via full def index (210L). Also read the harness's
`domain/geometry/mesh/intersection_repair.py` in full (281L) to establish exactly
what was ported.

SKIMMED-NOT-READ: `energies/TPE.py` L120-401; `largesteps/*.py` (535L across 5 files
— vendored third-party code from Nicolet et al., excluded per policy);
`mutils/matrices.py` (140L); `optimizers/GD.py` (52L, plain SGD);
`repair_factory.py` bodies of `main_vis` (Polyscope UI, excluded).
**Honest ratio: ~9 of 22 .py read in full or near-full.**

WHAT WAS PORTED (verified): `intersection_repair.py` (281L) ports, and honestly
declares as a "deterministic, stdlib-only reduction", the detect->step->re-detect->
stop-at-zero loop skeleton and its stopping rule; BVH broad phase plus exact
triangle-triangle narrow phase; a centroid-separation displacement as an explicit
stand-in for the energy gradient; uniform (umbrella) Laplacian smoothing as an
explicit stand-in for the curvature constraint. It correctly disclaims being a
boolean resolver.

FINDINGS (what was NOT ported)

1. **The TPE energy is a closed-form, differentiation-free scalar penetration
   metric — and the harness has no penetration magnitude at all, only a count.**
   `energies/TPE.py:56-67`: for each colliding face pair, with centroids `X`,`Y` and
   unit normals `n_x`,`n_y`: `dist = ||Y-X||`, `Pdist_x = |(X-Y).n_x|`,
   `Pdist_y = |(Y-X).n_y|`, `r = (Pdist/dist)^3`, and
   `TPE = sum sqrt(area_x*area_y)*(r_x + r_y)`. Every term is elementary arithmetic
   over quantities `find_self_intersections` already computes. The `signed_TPE`
   variant (L69-115) drops the `abs()` and uses `area_x*area_y` (not the geometric
   mean) so the sign encodes penetration *direction*.
   Why: `RepairResult` reports `initial_intersections`/`final_intersections`/
   `history` — pure **counts**. A count is a bad convergence signal: integer-valued,
   non-monotone, and unable to distinguish one deep penetration from one grazing
   touch. TPE gives a continuous area-weighted severity that makes `history` a real
   convergence curve, supplies a principled `best` criterion, and gives the output
   gate a *severity* rather than a boolean verdict.
   Harness equivalent: **NONE** — verified by reading all 281L of
   `intersection_repair.py`; there is no scalar energy anywhere, and the repair step
   uses `mag = step * mean_edge_length` without ever measuring penetration depth.
   Stdlib-portable: yes — dot/cross products and `sqrt`, no autodiff needed to
   evaluate it.

2. **Best-iterate tracking — the harness's loop can return a result strictly worse
   than one it already found.** `repair_factory.py:116-118, 156-166, 180` maintains
   `best_col`/`best_vertices`/`best_iter`, updates whenever `num_col < best_col`,
   writes **both** `<expname>_best.obj` and `<expname>_final.obj` (L182-187), and
   prints which iteration won.
   Why: the penetration objective is non-convex and the collision count genuinely
   oscillates. Harness equivalent: **NONE, and this is a live correctness gap, not
   a nicety.** Verified by reading `repair_self_intersections` L203-256: it returns
   `RepairResult(list(verts), it, len(initial), len(pairs), history)` — the **final**
   `verts`, unconditionally. Because it also records `history`, the failure is
   directly observable: a run whose `history` is `[12, 3, 0]` is fine, but
   `[12, 1, 7]` returns the 7-collision mesh while having held a 1-collision mesh in
   hand. One `if len(pairs) < best: best = ...` fixes it. Stdlib-portable: yes.

3. **Volume and area preservation as cheap checkable invariants of the repair.**
   `constraints/volume.py:10-33` (`total_volume` = signed tetrahedral sum
   `sum v0.((v1-v0)x(v2-v0))/6`) with an analytic per-vertex Jacobian at L35-66
   (`dV/dv_i = (v_j x v_k)/6`, accumulated by `index_add_`, unit-normalised with a
   `1e-6` floor); `constraints/area.py:9-35` with a Laplacian-based Jacobian L38-60;
   `constraints/curvature.py` (penalise `||Lx - u||^2` against the *initial*
   Laplacian coordinates, Jacobian `2L^T(Lx-u)`). Config-selectable per run.
   Why: the volume Jacobian is closed-form, no autodiff — but the more immediately
   valuable use is as **post-conditions**, not gradients. A repair that resolves
   self-intersections while changing the part's volume by 8% has silently built a
   different part, which is exactly the failure class this codebase exists to
   eradicate. `RepairResult` currently reports nothing about whether the part
   survived. Harness equivalent: **PARTIAL for the measurements**
   (`mesh/quality_metrics.py`, `eval/bench/geometry/mesh_topology.py`,
   `winding_number.py` Kahan mass properties); **NONE as a repair post-condition**.
   Stdlib-portable: yes.

4. **`MomentumBrake` — an optimizer whose idea transfers though its implementation
   does not.** `optimizers/MomentumBrake.py:33-72, 96-115`: Adam-style moments
   `(b1,b2)=(0.99,0.999)`, but it computes per-component confidence bands
   `mean +/- 3*std` from the bias-corrected moments, reduces them per-vertex with
   `torch.all(..., dim=1)`, and after a 10-step warmup gates the *first-moment
   update itself* by that mask, so an outlier gradient **resets** that vertex's
   momentum to the raw gradient rather than being smoothed into it. Update is
   `2*m1_hat`, no adaptive denominator.
   Why: the transferable principle is *"when a per-element gradient is a 3-sigma
   outlier relative to its own history, discard the accumulated momentum for that
   element rather than blending the outlier in."* In the harness's discrete repair
   loop the analogue is: a vertex whose separation direction flips violently
   between iterations is oscillating between two colliding pairs and its
   accumulated displacement should be **reset**, not averaged. The harness averages
   unconditionally (`acc/c` at L245-249) — precisely the oscillation-prone behaviour
   this optimizer exists to prevent. Harness equivalent: **NONE** (no per-vertex
   state carried across iterations at all). Stdlib-portable: yes as a principle
   (running per-vertex mean/variance of displacement direction, a few floats per
   vertex); the torch optimizer itself must not be copied.

5. **A named topological stress corpus — cite, do not vendor.** `data/misc/`
   (10 .obj): `celtic_knot`, `cinquefoil_knot`, `disc_kleinbottle`,
   `figure_of_eight_knot`, `knot_8_18`, `mobius_strip`, `pretzel`, `septoil_knot`,
   `three_twist_knot`, `trefoil_knot`.
   Why: self-intersecting **by construction** and spanning nontrivial topology —
   knots (genus/linking), a Mobius strip (**non-orientable**), a disconnected Klein
   bottle (non-orientable *and* disconnected). A non-orientable surface is a fixture
   class the harness's mesh checks have probably never seen, and a sharp test of
   anything assuming a consistent outward normal — including manifold3d's ingest,
   which would reject them with `Error::NotManifold`.
   Harness equivalent: **NONE** (registry greps for `knot`, `mobius`, `klein`
   return 0). **Licence-blocked: do not vendor these files.** They are standard
   mathematical surfaces, so the correct move is to **generate** them — torus knots
   are a one-line parametric tube mesh, a Mobius strip a two-line parametrisation —
   and cite the paper for the *idea* of using them as a self-intersection corpus.

6. **Honest negative: nothing here is a judge rubric or a ground-truth task
   corpus.** The paper's benchmark data is explicitly withheld (README: *"Please
   contact wonjong@postech.ac.kr to request our benchmark data"*), so this checkout
   has only the 10 toy meshes and a single config. No test suite, no epsilons with
   stated reasons, no error taxonomy, no refusal predicates.

ALREADY COVERED: `mesh/intersection_repair.py` (loop skeleton, stopping rule,
centroid-separation step, umbrella smoothing — correctly attributed), `mesh/bvh.py`,
`mesh/triangle_intersect.py`, `mesh/smoothing.py`, `mesh/isotropic_remesh.py`.

VERDICT: **mine-further**, narrowly and licence-carefully. Finding 2 is a genuine
bug in shipped harness code and a ~4-line fix. Findings 1 and 3 turn a count-based
loop into a measurable one and are pure stdlib. Findings 4 and 5 are ideas to
re-derive, not code or data to take.

---

## solvespace-python (648 files, all tracked; no .git, no .pyc)

LICENSE: **GPL-3** — `COPYING.txt`, "GNU GENERAL PUBLIC LICENSE Version 3,
29 June 2007". **The prior run's claim is CONFIRMED CORRECT.** Verdict:
**facts-with-citation ONLY, vendor nothing.** The Cython wrapper `cython/` is the
`python_solvespace` binding to the same GPL-3 core; `THIRD_PARTIES.txt` covers
`extlib/` only and does not relicense `src/`.

WHAT IT IS: the full SolveSpace parametric CAD application (C++) with a Cython
binding exposing the sketch constraint solver. The interesting third is `src/`: a
symbolic-algebra expression tree (`expr.cpp`), constraint->equation generators
(`constrainteq.cpp`), and a modified Newton solver over a sparse Jacobian with
Eigen `SparseQR` (`system.cpp`).

READ (in full): `src/system.cpp` (593L, entire); `src/solvespace.h` L95-180
(epsilons, `IsReasonable`, `SolveResult`); `src/sketch.h` L180-205 (`Group::solved`);
`src/textscreens.cpp` L590-645 (the user-facing failure taxonomy);
`src/constrainteq.cpp` L85-215; `COPYING.txt`; `cython/test/test_slvs.py` (250L).

SKIMMED-NOT-READ: the other ~44 files of `src/` (draw/mouse/ui/export/ttf/render/
srf/polygon — ~50k lines), all of `extlib/` (10 vendored libraries), `res/`, `pkg/`,
`cmake/`. `src/expr.cpp` and `src/constrainteq.cpp` were grepped exhaustively for
epsilons and degeneracy rather than read line by line.

FINDINGS

1. **`CONVERGE_TOLERANCE = LENGTH_EPS/1e2 = 1e-8`, with the reason stated.**
   `src/system.cpp:14-16`, comment: *"The solver will converge all unknowns to
   within this tolerance. This must always be much less than LENGTH_EPS, and in
   practice should be much less."* `LENGTH_EPS = 1e-6` at `src/solvespace.h:129`.
   The exact scar-tissue pattern: the solver tolerance is deliberately **two orders
   below the geometric equality tolerance**, so a converged sketch is never near the
   kernel's own merge threshold.
   Harness equivalent: **PARTIAL** — `core/lm_solver.py:156-157` uses
   `residual_tolerance=1e-8`, `step_tolerance=1e-12`, but the values are
   free-standing; there is no documented *derivation* from a geometric `LENGTH_EPS`.
   Stdlib-portable: yes (a documented ratio, not code).

2. **Divergence guard `IsReasonable(x) := isnan(x) || x > 1e11 || x < -1e11`,
   applied per-parameter inside the Newton step.** `src/solvespace.h:103-105`, used
   at `src/system.cpp:310-313` (*"Very bad, and clearly not convergent"*) and
   L322-325. The solver bails the instant any parameter or residual blows past
   1e11, rather than iterating to the cap.
   Harness equivalent: **NONE** — `core/lm_solver.py` has
   `CONVERGED/MAX_ITERATIONS/STALLED` and a step-norm test but no NaN/blowup abort;
   grepping it for `isnan|inf|1e1` returns no hits. Stdlib-portable: yes, trivially.

3. **The 5-value failure taxonomy — the refusal-predicate source.**
   `src/solvespace.h:167-173`: `OKAY=0`, `DIDNT_CONVERGE=10`, `REDUNDANT_OKAY=11`,
   `REDUNDANT_DIDNT_CONVERGE=12`, `TOO_MANY_UNKNOWNS=20`. The user-facing gloss at
   `src/textscreens.cpp:596-615` is the valuable part because it distinguishes two
   failures the harness conflates:
   - `DIDNT_CONVERGE` -> *"unsolvable constraints / the following constraints are
     **incompatible**"* (rank was fine, so the constraints genuinely contradict);
   - `REDUNDANT_DIDNT_CONVERGE` -> *"unsolvable constraints / the following
     constraints are **unsatisfied**"* (rank-deficient *and* non-convergent — you
     cannot say which is at fault);
   - `REDUNDANT_OKAY` -> *"redundant constraints / **remove any one of these to fix
     it**"* — solved fine but rank-deficient, and the remedy is stated as a set, not
     a single culprit;
   - `TOO_MANY_UNKNOWNS` -> a hard **refusal**, emitted before any solving.
   The structural insight: **rank status and convergence status are orthogonal
   axes**, and it is their *cross product* that names the user's problem.
   Harness equivalent: **PARTIAL/MISSING** — `core/constraints.py:48` `SketchStatus`
   = {WELL, OVER, UNDER, EMPTY} (rank axis) and `core/lm_solver.py:65` `SolveStatus`
   = {CONVERGED, MAX_ITERATIONS, STALLED} (convergence axis) exist **independently
   and are never crossed**; there is no "redundant AND didn't converge -> I cannot
   tell you which constraint is wrong" state, and no `TOO_MANY_UNKNOWNS` refusal.
   Verified by reading both enums and grepping `src/harnesscad/core/` for
   `DIDNT_CONVERGE|REDUNDANT|TOO_MANY` — zero hits. Stdlib-portable: yes.

4. **`MAX_UNKNOWNS = 2048` as a pre-solve refusal.** `src/solvespace.h:218`,
   enforced at `src/system.cpp:47-49` (`WriteJacobian` returns false) -> L476-478
   returning `TOO_MANY_UNKNOWNS`. A separate, much tighter `MAX_UNKNOWNS = 16`
   bounds the dense fixed-size solver at `src/dsc.h:626`. The system declines to
   attempt a solve it knows will not be tractable — a size-based refusal checked
   *before* any numerics. Harness equivalent: **NONE** (greps of
   `core/constraints.py`, `core/lm_solver.py`, `domain/numeric/constraint_solver.py`
   for `MAX_UNKNOWN|too_many|size limit` return nothing). Stdlib-portable: yes.

5. **`FindWhichToRemoveToFixJacobian` — leave-one-out re-rank with a wall-clock
   timeout and a two-pass blame ordering.** `src/system.cpp:371-415`. For each
   constraint: drop it, rewrite the Jacobian, recompute rank; if `rank == mat.m` the
   sketch is fixed, so that constraint is a candidate culprit. Three pieces of scar
   tissue: (a) **two passes** — everything *except* `POINTS_COINCIDENT` first, then
   only those, *"so they appear last in the list"* (L385-392) — i.e. coincidence
   constraints are deprioritised as blame targets because they are almost never what
   the user got wrong; (b) a **timeout** (`g->solved.findToFixTimeout`,
   `src/sketch.h:193`, set from `SS.timeoutRedundantConstr` at `src/generate.cpp:538`)
   that sets `solved.timeout`, truncates the culprit list, and is surfaced honestly
   as *"Some items in list have been omitted because the operation timed out"*
   (`textscreens.cpp:632-634`); (c) `SolveBySubstitution()` runs first as *"a major
   speedup ... and that doesn't break anything"* (L399-403), skipped under
   `forceDofCheck`.
   Harness equivalent: **PARTIAL** — `core/constraints.py:352` `_drop_set` does
   iterative drop-and-re-rank, but it is a union-find heuristic with **no timeout**,
   no partial-result honesty flag, and no blame-ordering policy. Stdlib-portable:
   yes (concept only).

6. **DOF is computed from actual Jacobian rank, not equation count.**
   `src/system.cpp:232-239`: `TestRank` sets `*dof = mat.n - jacobianRank`, commented
   *"We are calculating dof based on real rank, not mat.m. Using this approach we can
   calculate real dof even when redundant is allowed."* Rank via Eigen `SparseQR`
   with COLAMD ordering (L223-230). Also `dof` is initialised to `-1` at L480 *"in
   order to have indication when dof is actually not calculated"* — a sentinel for
   "I did not measure this", distinct from zero.
   Harness equivalent: **PARTIAL** — `core/constraints.py` computes DOF from a
   union-find pooling heuristic and self-documents (L80-85, L23) as *"a documented
   heuristic, not a physical solver"*; `core/lm_solver.py:107` has a real
   `matrix_rank` with `tol=1e-9`. The `-1` "not computed" sentinel and the
   "rank-based DOF stays valid even when redundancy is tolerated" property are
   absent. Stdlib-portable: yes.

7. **Per-parameter free-variable detection is deliberately opt-in because it is
   slow.** `src/system.cpp:572-592` `MarkParamsFree` tags one parameter at a time and
   rebuilds/re-ranks the whole Jacobian per parameter — O(n) full rank computations.
   Comment: *"This might be more than the number of degrees of freedom. Don't always
   do this, because the display would get annoying and it's slow."* The stated caveat
   — **the count of free parameters can exceed the DOF** — is a correctness note, not
   just a perf note. Harness equivalent: `core/lm_solver.py:164` `FreedomReport`
   exists; whether it carries the "free params >= DOF" caveat was not verified in
   full. Stdlib-portable: yes.

8. **Degeneracy that is NOT guarded — division by a magnitude that can be zero.**
   `src/constrainteq.cpp:103` (`PointLineDistance`, 3-D branch: `.Div(m)` where
   `m = |a-b|`, zero for a zero-length line), L115 (workplane branch, `Div` by
   `sqrt(du^2+dv^2)`), L167 and L176 (`DirectionCosine`, `Div` by `|a|*|b|`), L205
   (`ModifyToSatisfy` for ANGLE: `acos(dot/(|a||b|))` in raw doubles). Nothing clamps
   these. The **only** thing catching a degenerate sketch is the downstream
   `IsReasonable` NaN/blowup test in `NewtonSolve` — i.e. SolveSpace's design choice
   is *"let the equation produce NaN, catch it at the Newton step, report
   DIDNT_CONVERGE"* rather than validating inputs. That is itself a design fact worth
   recording. Harness equivalent: n/a — an anti-pattern to note, not to copy.

9. **Solve ordering: singly-referenced equations are solved alone first.**
   `src/system.cpp:449-472`. Before the big system, any equation referencing exactly
   one parameter is solved by itself (*"This can be a huge speedup. We don't know
   whether the system is consistent yet, but if it isn't then we'll catch that
   later"*), and if that mini-solve fails it short-circuits straight to
   `DIDNT_CONVERGE` with `rankOk = true` — an arbitrary attribution the code admits
   to: *"We don't do the rank test, so let's arbitrarily return the DIDNT_CONVERGE
   result here."* Harness equivalent: **NONE** (no pre-pass decomposition).
   Stdlib-portable: yes.

10. **Dragged-parameter column scaling `1/20.0`.** `src/system.cpp:253-265`:
    least-squares columns for parameters the user is dragging are weighted `1/20`,
    *"so that we can encourage the solver to make bigger changes in some parameters,
    and smaller in others... It's least squares, so this parameter doesn't need to be
    all that big to get a large effect."* Related: substitution chains are re-rooted
    so the **dragged** parameter survives as representative
    (`SortSubstitutionByDragged`, L118-137), and substitution-chain **cycles are
    explicitly broken** (`GetLastParamSubstitution`, L105-116, L190-198) — a
    self-referential-constraint hazard. Harness equivalent: **NONE.**
    Stdlib-portable: yes.

11. **Iteration cap = 50 Newton iterations.** `src/system.cpp:331`
    (`while(iter++ < 50 && !converged)`). Compare harness `max_iterations: int = 35`
    (`core/lm_solver.py:155`). Different algorithm (Newton vs LM), so not directly
    comparable, but a useful reference point.

12. **Committed ground-truth solver fixtures with expected values to 4 decimals.**
    `cython/test/test_slvs.py` (250L): `test_crank_rocker` (expect 39.54852,
    61.91009), `test_involute` (12.62467, 1.51746), `test_jansen_linkage` (18.93036,
    13.63778), `test_nut_cracker` (min 1.01576, max 2.625), `test_pydemo` (7 point
    assertions **plus `assertEqual(6, sys.dof())`** at L250). Real linkage mechanisms
    with published-precision expected coordinates and an expected DOF — an excellent
    independent cross-check corpus for `core/lm_solver.py` / `core/constraints.py`.
    **GPL-3: cite the constructions and expected values as facts; do not copy the
    file.** Stdlib-portable: the *problem statements* are; the code is not vendorable.

ALREADY COVERED: rank/redundancy analysis and under/over/well classification
(`core/constraints.py`, incl. a `SolveSpaceSketch` wrapper at L446 and its own
`SolveResult` dataclass at L436 — note this shadows the name **without** the
taxonomy); a real Gauss-Newton/LM solver with convergence and stall statuses and a
`matrix_rank` (`core/lm_solver.py`, `domain/numeric/constraint_solver.py`); iterative
drop-and-re-rank conflict sets; assembly 6-DOF analysis
(`domain/numeric/assembly_dof`); SketchGraphs DOF tables
(`domain/reconstruction/sketch/dof_mask`).

VERDICT: **mine-further** (facts-with-citation only). Highest value: finding 3 (the
orthogonal rank x convergence taxonomy and its exact user-facing wording), 2
(`IsReasonable` divergence abort), 4 (`TOO_MANY_UNKNOWNS` size refusal), 1 (tolerance
derived as `LENGTH_EPS/100` with the reason), and 5's timeout-and-partial-list
honesty. Fixtures in 12 are usable as facts.

---

## CADmium-main (123 files; 72 excluding `__pycache__`; of 44 non-cached .py, 14 are in `OCCUtils/`)

LICENSE: `LICENSE.txt` — **MIT**, but with an unfilled placeholder
(`Copyright 2025 <COPYRIGHT HOLDER>`). Vendorable with attribution — **except** the
`OCCUtils/` subtree, which carries its own **LGPL-3-or-later** header
(`cadmium/src/utils/CadSeqProc/OCCUtils/Construct.py:6-18`: *"This file is part of
pythonOCC ... GNU Lesser General Public License ... version 3 ... or (at your option)
any later version"*, (c) 2011-2015 Jelle Feringa). **Vendor nothing under `OCCUtils/`.**

### The specific claim: the prior audit is a MISATTRIBUTION. Confirmed.

- The literal line `EPSILON = TOLERANCE = 1e-6` is at
  **`cadmium/src/utils/CadSeqProc/OCCUtils/Construct.py:96`**.
- A second, independent `TOLERANCE = 1e-6` is at
  **`cadmium/src/utils/CadSeqProc/OCCUtils/Common.py:93`**.
- Both files are **pythonocc-utils** verbatim, proven by the header at
  `Construct.py:2-18` naming pythonOCC and Jelle Feringa and declaring LGPL — a
  license *incompatible with the repo's own MIT `LICENSE.txt`*, which is itself the
  tell that it is a vendored drop-in.
- All 14 `OCCUtils/*.py` files are vendored; every `TOLERANCE`/`EPSILON` reference
  elsewhere in the repo (`base.py:58`, `face.py:36`, ...) is an *import from* that
  vendored module, not CADmium code.
- **CADmium's own constants live elsewhere and are different values:**
  `cadmium/src/utils/CadSeqProc/utility/macro.py:32-33` defines `PRECISION = 1e-5`
  and `eps = 1e-7`. Neither is `1e-6`.

Conclusion: attributing `1e-6` to CADmium is wrong on **three** counts — wrong file
(vendored), wrong project (pythonOCC), wrong license (LGPL, not MIT). Any harness
note citing "CADmium: EPSILON = TOLERANCE = 1e-6" should be corrected to
"pythonocc-utils (LGPL-3+), vendored into CADmium at `OCCUtils/Construct.py:96`",
or dropped.

WHAT IT IS: a text-to-CAD VLM fine-tuning repo (train/predict/annotate/tokenize +
a Qwen-VL-ish pipeline) whose entire geometry layer `utils/CadSeqProc/` is a **fork
of Text2CAD's `CadSeqProc`** — proven by `utility/macro.py` reproducing Text2CAD's
exact token vocabulary (`N_BIT=8`, `END_PAD=7`, `BOOLEAN_PAD=4`,
`MAX_CAD_SEQUENCE_LENGTH=272`, `NORM_FACTOR=0.75`, `MAX_EXTRUSION=10`,
`ONE_EXT_SEQ_LENGTH=10`).

READ (in full): `LICENSE.txt`; `utility/macro.py` (45L); `utils/prompts.py` (94L);
`geometry/arc.py` L232-280; `sequence/sketch/loop.py` L265-330;
`OCCUtils/Construct.py` L1-40 (header, for the license verdict).

SKIMMED-NOT-READ: the two large files `cad_sequence.py` (1777L) and
`utility/utils.py` (2142L) — function index plus targeted greps only, not read;
`train.py`, `predict.py`, `annotate.py`, `tokenize_dataset.py`, `customtrainer.py`,
`warmstabledecay.py`, `rendering/render.py`, `Demo/app.py`, all of `config/`, all 14
`OCCUtils/` files beyond headers.

FINDINGS (thin, and mostly already covered)

1. **Collinearity is a *type-conversion repair*, not a rejection.**
   `geometry/arc.py:245-259`. On decoding an arc from tokens, if
   `get_orientation(start, mid, end) == "collinear"`, then under
   `post_processing=True` the arc is **silently rebuilt as a `Line`**; otherwise the
   raise is **commented out** (`# raise Exception(f"Collinear points {metadata}")`)
   and it falls through to construct a degenerate `Arc` anyway. A real decoder
   robustness decision: 8-bit quantisation routinely collapses a shallow arc's three
   points onto a line, and the pipeline chooses silent degradation over refusal.
   Harness equivalent: **PARTIAL** — `domain/geometry/sketch/normalization.py:52`
   documents rejecting a collinear arc (returns `None`, threshold `1e-10`), the
   *opposite* policy; `domain/reconstruction/tokens/text2cad_vector_codec` covers the
   codec. The arc->line demotion variant appears absent. Stdlib-portable: yes.
2. **`eps = 1e-8` added to a norm denominator to avoid divide-by-zero.**
   `arc.py:265-274` (`get_angles_counterclockwise(self, eps=1e-8)`): every
   `(p - center)` is divided by `norm(...) + eps`. Additive-epsilon-in-denominator —
   it biases the result rather than branching. Stdlib-portable: yes.
3. **Loop canonicalisation: leftmost-then-lowest start, forced CCW, collinear-line
   merge.** `sequence/sketch/loop.py:270-330` (`reorder`): rotate the curve list to
   start at the lexicographically leftmost-lowest start point (compared at
   `round(..., 6)`), reverse the list *and* every curve if the loop is clockwise,
   then merge runs of mutually collinear line segments. A deterministic canonical
   form for a sketch loop. Harness equivalent: **PARTIAL** —
   `domain/reconstruction/fitting/wire_assembly` assembles *oriented* closed loops
   with per-edge +/-1 direction, but its docstring shows no leftmost-start
   canonicalisation or collinear-run merging. Stdlib-portable: yes.
4. **`PRECISION = 1e-5`, `eps = 1e-7`** (`utility/macro.py:32-33`) — CADmium's
   *actual* own constants, the correct replacement for the misattributed `1e-6`.
   Both bare, no stated reason.
5. **Invalidity ratio computed as "fraction of UIDs with no valid chamfer
   distance"** — `utils/Evaluation/eval_seq.py:155`, with an in-code
   `## Change this to len(test_data)` FIXME admitting the denominator is wrong.
   Harness equivalent: **ALREADY COVERED** — `eval.bench.sequence.invalidity_ratio`.
6. **Schema-first system prompt with explicit "STRICT RULES"** —
   `utils/prompts.py:56-74`. Harness equivalent: **ALREADY COVERED** —
   `data.dataengine.schemas.minimal_json`, `annotation.prompt_templates`.

ALREADY COVERED: essentially the whole geometry layer —
`domain/reconstruction/tokens/text2cad_vector_codec` explicitly cites
`CadSeqProc/cad_sequence.py`; plus `text2cad_tokens`, `deepcad_quantize`,
`programs/quantization_ranges`, `schemas/minimal_json`, `annotation/prompt_levels`,
`annotation/prompt_templates`, `bench/protocols/primitive_f1_null_class`,
`bench/sequence/invalidity_ratio`, `bench/sequence/sequence_f1`.

VERDICT: **nothing-here** (beyond the correction). The geometry is a Text2CAD fork
already mined at the source, the numeric constants layer is vendored LGPL pythonOCC,
and the rest is VLM training/inference glue and rendering. The single genuinely new
item is finding 1, with 3 a marginal second. **The primary deliverable here is the
misattribution correction.**

---

## arcs-master (51 files; 37 .rs, 3,773 lines of Rust)

LICENSE: dual **MIT OR Apache-2.0** (`LICENSE_MIT.md` (c) 2019 Michael Bryan,
`LICENSE_APACHE.md`, confirmed by `README.md:55-63`). Vendorable with attribution.

WHAT IT IS: a 2D CAD framework in Rust on an ECS architecture (`specs`), split into
`arcs-core` (pure geometry) and `arcs` (ECS app layer, r-tree spatial index, `piet`
render window). Explicitly a work in progress — the README marks B-splines, Bezier
curves, elliptical sections, and robust undo/redo as unimplemented.

READ (in full): `README.md`; both licences;
`core/src/algorithms/line_simplification.rs` (181L incl. its 6 tests);
`core/src/algorithms/approximate.rs` (140L incl. tests); `core/src/orientation.rs`
(96L); `core/src/algorithms/closest_point.rs` L1-120.

SKIMMED-NOT-READ: the remaining ~2,500 lines — `bounding_box.rs`, `scale.rs`,
`scale_non_uniform.rs`, `translate.rs`, `length.rs`, `affine_transform.rs`,
`primitives/{arc,line}.rs`, and the entire `arcs/` app crate (ECS and rendering glue,
out of scope).

FINDINGS

The three algorithms of substance **have already been mined, by name, with the
source file cited in the harness docstring**:

1. **Ramer-Douglas-Peucker simplification** | `core/src/algorithms/line_simplification.rs:23`
   | Harness: **`domain.geometry.parametric.simplify`**, registry summary
   *"Ramer-Douglas-Peucker polyline decimation (from the `arcs` Rust CAD core)"*.
2. **Sagitta / chord-tolerance arc tessellation** |
   `core/src/algorithms/approximate.rs:48-80` — derives
   `theta = 2*acos(1 - tol/R)`, `N = ceil(sweep/theta)`, floors at 2 segments, and
   collapses to a single chord when `tolerance <= 0 || radius <= tolerance` |
   Harness: **`domain.geometry.parametric.chord_tolerance`**, whose docstring
   reproduces the derivation, cites the Rust file by path, and records **both**
   degenerate guards.
3. **Closest-point with multiplicity (`One`/`Many`/`Infinite`)** |
   `core/src/algorithms/closest_point.rs:70-120` — zero-length line -> `One(start)`;
   target at arc centre -> `Infinite` | Harness:
   **`domain.geometry.parametric.closest_point`**, whose docstring cites the Rust
   file and states it **fixed a bug in the original**: `contains_angle` is
   reimplemented on the normalised sweep parameter *"rather than the raw comparison
   used by the Rust source, so arcs which cross the +/-pi branch cut behave
   correctly."*

Remaining unmined candidates, both **low value**:

4. **`centre_of_three_points` refuses on collinear input via an exact determinant
   test** | `core/src/orientation.rs:50-77` — returns `Option`, `None` when
   `determinant == 0.0`, doc-comment reason: *"If the points are collinear then the
   problem is ambiguous, the radius effectively becomes infinite and our centre could
   be literally anywhere."* Good framing of a refusal; weak implementation (**exact
   `== 0.0` float comparison**, no tolerance — near-collinear input yields an
   astronomically distant centre rather than `None`). Harness: **ALREADY COVERED,
   and better** — four independent implementations exist
   (`fitting/wire_assembly.py:69`, `drawings/rasterizer.py:108`,
   `reconstruction/sketch/primitives.py:253`, `tokens/skexgen_decode.py:46`).
5. **`Orientation::of` (CW/CCW/Collinear) via a cross-product sign** |
   `core/src/orientation.rs:16-31` — naive `> 0.0 / < 0.0 / else` on a raw float
   determinant, **not numerically robust**. Harness: **ALREADY COVERED, and strictly
   better** — `domain.numeric.exact_predicates`, *"Robust adaptive geometric
   orientation predicates (Manifold kernel substrate)"*. Adopting the arcs version
   would be a regression.

Nothing else qualifies: `bounding_box`/`scale`/`translate`/`length`/`affine_transform`
are one-screen euclid wrappers; `arcs/src/{window,systems,components}/` is ECS/piet/
r-tree glue. **There is no fixture corpus, no fuzz/property corpus, no error
taxonomy, and no epsilon table** — the crate uses `euclid`'s `ApproxEq` default
rather than defining its own tolerances.

ALREADY COVERED: `parametric/simplify`, `parametric/chord_tolerance`,
`parametric/closest_point` (all three cite `arcs` in-docstring),
`numeric/exact_predicates`, `transforms/orientation`, plus four circumcircle
implementations.

VERDICT: **already-covered.** Mined thoroughly in a previous pass — three modules
name it as their source and one documents a correctness improvement over the
original. The only unmined items are numerically inferior to what the harness has.

---

## ruststep-master (842 files, of which 711 are EXPRESS schemas in `schemas/`)

LICENSE: **Apache-2.0** (`LICENSE`, verified verbatim, RICOS Co. Ltd.) ->
vendorable with attribution. Two carve-outs the repo states itself:
- `ruststep/tests/steps/README.md:3` — *"This directory is not a part of ruststep
  project. See the original licenses for each files"* (ABC Dataset part).
- `schemas/README.md:3-9` — *"This is not a part of ruststep... Copied from
  steptools.com/stds/archive"*, with the ISO note quoted verbatim: **"ISO standards
  are copyrighted, but the EXPRESS schemas can be distributed without
  restrictions."** That is an explicit redistribution grant for the 711 `.exp` files.

WHAT IT IS: two crates. `espr` = an EXPRESS (ISO-10303-11) compiler (parser -> AST ->
semantically-legalized IR -> Rust codegen). `ruststep` = the part-21 exchange-file
parser plus serde de/serializer and generated AP201/AP203 bindings.

READ (in full): `ruststep/src/error.rs`; `espr/src/ir/mod.rs` (doc header +
`SemanticError`); `espr/src/ir/constraints.rs` (all 240L incl. its 11 tests);
`espr/src/ir/complex_entity.rs`; `espr/src/parser/reserved.rs`;
`ruststep/tests/abc_dataset.rs`; `ruststep/tests/steps/README.md`;
`espr/tests/README.md`; `schemas/README.md`. Directory listings of
`espr/src/parser/**`, `espr/tests/`, `ruststep/tests/`, `schemas/**`.

SKIMMED-NOT-READ: all 711 `.exp` schema files (names/dirs only);
`espr/src/codegen/rust/**` (7 files, token-stream emission — bindings boilerplate);
`ruststep-derive/**` (proc-macro); the generated `ap201.rs`/`ap203.rs`;
`ruststep/src/parser/exchange/**`.

FINDINGS

1. **711-file official ISO EXPRESS schema corpus, explicitly redistributable.**
   `schemas/` — `IRs/` (43: 10303-041/042/043 e2/e3/TC1/TC2 ...), `APs/` (49),
   `AICs/` (24), `modules/` (517), `PLIB/`, `oil-and-gas/`. A parser-conformance
   corpus for the harness's own EXPRESS parser, with the ISO redistribution grant
   quoted in-repo. The harness has `domain/spec/express_schema_parser.py` but —
   verified by `grep -rn "schemas/IRs\|10303-04" src/` -> **zero hits** — it has
   never been exercised against a single real ISO schema. Multiple editions of the
   same schema (`10303-041.exp`, `-041e2`, `-041e3`, `-041e3TC1`, `-041e3TC2`) also
   give free **schema-version-drift differential tests**.
   Harness equivalent: parser exists, **corpus does NOT**. Stdlib-portable: yes.
2. **`SUBTYPE_CONSTRAINT` / complex-entity instantiables algorithm (ISO-10303-11
   Annex B), fully implemented with 11 unit tests carrying ISO-sourced examples.**
   `espr/src/ir/constraints.rs:104-215` (`gather_constraint_expr`, steps b/d/c with
   the ordering comment *"step d) is done before c) because c) have to look up all
   SUBTYPE_CONSTRAINT"*) and `espr/src/ir/complex_entity.rs:1-90`
   (`PartialComplexEntity`, idempotent/commutative/associative `&` with doctests).
   This computes which *combinations* of subtypes an entity may legally instantiate
   — what a part-21 validator needs to accept
   `#5 = (PERSON('X') EMPLOYEE(15) STUDENT('Y'))` external mappings.
   **The harness's parser explicitly skips these blocks:**
   `express_schema_parser.py:734-735`
   `elif cur.is_kw("SUBTYPE_CONSTRAINT"): _skip_to(cur, "END_SUBTYPE_CONSTRAINT")`,
   and `express_p21_validator.py:54` records complex instances under `skipped`.
   **This is a real gap.** The tests at `constraints.rs:263-680` are ready-made
   ground truth (PET/ONEOF, PERSON_ANDOR, PERSON_AND -> exact expected instantiable
   sets; e.g. `AND` of two `ONEOF`s -> 4 pairs; `ANDOR` -> 3 sets including the
   both-at-once one).
   Harness equivalent: **NONE** — verified by grepping for `instantiab` across
   `domain/spec/*.py` (0 hits) and by reading `express_inheritance.py` (only
   `build_inheritance`/`flatten_attributes`/`expand_select`). Stdlib-portable: yes
   (pure set algebra).
3. **STEP parse/link error taxonomy.** `ruststep/src/error.rs:7-26` —
   `TokenizeFailed`, `ExtraInputRemaining(String)`, `DeserializeFailed`,
   **`UnknownEntity(u64)`** ("Lookup failed for #N", dangling forward reference),
   **`DuplicatedEntity(u64)`** ("Entity ID #N is duplicated"),
   **`UnknownEntityName{entity_name, schema}`** ("not a member of the schema").
   Five refusal predicates a STEP reader owes callers, three of which (dangling `#N`,
   duplicate `#N`, extra trailing input) are file-level integrity checks independent
   of any schema. Harness `express_p21_validator.py` emits only "unknown entity type"
   / "unknown entity type in complex instance" (L162, L168) — it does **not** check
   ID duplication, dangling references, or trailing input.
   Harness equivalent: partial; **missing 3 of 5**. Stdlib-portable: yes.
4. **Semantic-legalization error taxonomy.** `espr/src/ir/mod.rs:189-200` —
   `TypeNotFound{name, scope}`, `InvalidPath(Path)`, `DuplicatedDeclaration(Path)`:
   the three ways a *syntactically valid* EXPRESS schema is still illegal.
   `DuplicatedDeclaration` is raised at `constraints.rs:135-141` specifically for a
   second `SUPERTYPE OF` on the same entity. Harness has `InheritanceError` and
   `_check_acyclic` only. Stdlib-portable: yes.
5. **The 262-keyword EXPRESS reserved-word list.** `espr/src/parser/reserved.rs:1+`
   (`KEYWORDS`, lowercase, alphabetical). Identifier-vs-keyword disambiguation is
   exactly where hand-rolled EXPRESS parsers silently mis-parse. Cheap, mechanical,
   high-confidence import. Stdlib-portable: yes.
6. **4 committed part-21 data files beyond the already-mined `.step`.**
   `schemas/PLIB/Part24/Part24-DIS/liim_24_1_file_test1.p21`, `...test2.p21`,
   `schemas/PLIB/Part25/Part25-IS/Part25-IS-initial/FM_model_P25_IS.p21`,
   `GM_Model_P25_IS.p21`. Real part-21 instances paired with their own schemas in the
   same tree -> end-to-end schema+data validation fixtures for `validate_part21`
   (`domain/spec/registry.py:571`). Verified not referenced:
   `grep -rn "liim_24\|Part24" src/` -> 0 hits. Harness equivalent: **NONE.**
7. **The internal-vs-external mapping explainer.** `espr/src/ir/mod.rs:47-80` (the
   `#1..#5` worked example and the constraint that internal mapping *cannot* express
   `#5`). Doc value only.

ALREADY COVERED: `ruststep/tests/steps/*.step` (the ABC part) —
`eval/corpus/fixtures/step_canaries.py` already manifests it, correctly flags the
README license carve-out, and even records that it is **byte-identical (same
SHA-256)** to pythonocc-core's `stp_multiple_shp_at_root.stp`. EXPRESS grammar core
(ENTITY/TYPE/SELECT/ENUMERATION/aggregates/bounds/both comment forms) ->
`express_schema_parser.py`. Inheritance flattening and SELECT expansion ->
`express_inheritance.py`. Part-21 data-vs-schema checking -> `express_p21_validator.py`.

**NO known-bad STEP files are committed.** Verified:
`find . -name "*.step" -o -name "*.stp" -o -name "*.p21"` returns exactly 5 files,
all well-formed. `ruststep` has **no negative-parse tests at all** —
`abc_dataset.rs` only asserts the good file parses. The refusal predicates exist as
`Error` variants but nothing in-repo exercises them. Do not expect a malformed
corpus here.

VERDICT: **mine-further** — specifically (a) the 711-schema `.exp` conformance
corpus, (b) the Annex-B `SUBTYPE_CONSTRAINT` instantiables algorithm plus its 11
ISO-derived expected outputs, (c) the 5-variant parse error taxonomy, (d) the 4
`.p21` files.

---

## ezpz-main (93 files)

LICENSE: **MIT** (`LICENSE:1`, unfilled template `Copyright (c) [year] [fullname]`)
-> vendorable with attribution.

WHAT IT IS: a 2D sketch constraint solver in Rust — Levenberg-Marquardt /
Gauss-Newton over a sparse Jacobian (faer), plus DOF/nullspace freedom analysis, a
textual problem format, a linter, and a residual visualiser.

READ (in full): `ezpz/proptest-regressions/tests/proptests.txt` (all 11 lines);
`ezpz/src/error.rs`; `ezpz/src/warnings.rs`; `ezpz/src/solve_outcome.rs`;
`ezpz/src/solver/find_dof.rs`; `ezpz/src/solver.rs:1-120`; all doc-comments and
property names in `ezpz/src/tests/proptests.rs` (1282L, structural read). Full file
listing plus every `EPSILON`/tolerance grep hit in `ezpz/src`, `ezpz-cli/src`, `fuzz/`.

SKIMMED-NOT-READ: `ezpz/src/constraints.rs` (~2870L — read only its ~25 degeneracy
guard sites via grep, not the residual/Jacobian bodies); `ezpz/src/textual/**`;
`residual_viz.rs`; `ezpz-wasm/`; the 28 `test_cases/*/problem.md`.

FINDINGS

1. **THE FUZZ CORPUS — 5 committed proptest regression seeds, each a shrunken
   failing input, with the exact numbers.**
   `ezpz/proptest-regressions/tests/proptests.txt:7-11`. Five real solver-breaking
   configurations, already minimised, each traceable to the property it broke:
   - L7 `x = 0.0, y = 0.0, guess_x = 0.0, guess_y = 0.846792320291437` -> broke
     `scalar_eq` (`proptests.rs:332`): two-variable equality solve failing from an
     asymmetric zero guess.
   - L8 `arc_center = (0,0), arc_radius = 1.0, arc_start = 326.0065646718824 deg,
     arc_degrees = 5.0, point_guess = (0,0)` and L9 the same shape with
     `arc_radius = 22.73229937272911, arc_start = 294.58471976001573 deg` -> broke
     `point_arc_coincident` (`proptests.rs:515`). Both are **5-degree-span arcs with
     the guess point exactly at the arc centre** — the two documented failure modes
     in one seed: *"Very narrow arcs make the angle inequalities stiff and Newton may
     not converge"* (L520-521) and *"the point is exactly at the arc center; that
     makes the distance Jacobian singular and the solver refuses to proceed"*
     (L526-527).
   - L10 `guess_line_p0 = (39.74751056036584, -95.46159322882576),
     p1 = (0.0, -95.45694757549501), guess_point = (0,0), desired_distance = 0.0` ->
     broke `horizontal_point_line_dist`/`vertical_point_line_dist`
     (`proptests.rs:442`, L472): a line **near-horizontal but not exactly**
     (dy = 0.0046 over dx = 39.7) with a **zero** target distance.
   - L11 `arc_center = (0, 6.850539916263869), arc_radius = 19.460231588106844,
     arc_start = 0.0, arc_degrees = 179.95268332677125, point_guess = (0,0)` -> arc
     span **within 0.05 degrees of a half-circle**, i.e. the angle-wrap branch cut.
   Directly replayable against `harnesscad.core.lm_solver` as regression canaries.
   Harness equivalent: **NONE** — `grep -n "proptest\|regression\|seed"
   src/harnesscad/core/lm_solver.py` -> 0 hits; the module is 407 lines and ports the
   algorithm, not the corpus. Stdlib-portable: yes (they are float tuples).
2. **The five degeneracy/stiffness invariants stated as prose, next to the seeds
   that violate them.** `proptests.rs:520-521, 526-527, 554-555, 580-581, 636, 451,
   481`. Verbatim scar tissue: narrow arcs -> stiff angle inequalities -> Newton
   non-convergence; point at arc centre -> singular distance Jacobian -> **solver
   refuses**; internal circle-circle tangency has centre distance `|ra-rb|` and must
   be kept off zero or the centre-centre distance is singular; vertical/horizontal
   point-line distance is undefined for a vertical/horizontal line. These are
   *refusal predicates*, not just test hygiene. Harness equivalent: **NONE** in
   `lm_solver`. Stdlib-portable: yes.
3. **Warning/lint taxonomy — `Degenerate`, `ShouldBeParallel(theta)`,
   `ShouldBePerpendicular(theta)`.** `ezpz/src/warnings.rs:24-58` (`lint()`), with
   user-facing text at L66-86. The linter rewrites `LinesAtAngle(0/180/360 deg)` ->
   "use Parallel" and `+/-90 deg` -> "use Perpendicular" within `EPSILON`, **because
   the angle constraint is numerically worse than the dedicated one**. The
   `Degenerate` message (*"two points are so close together that they practically
   overlap... place your initial guesses further apart"*) is a ready-made
   agent-facing diagnostic. The test at L117 asserts `360.00005 deg` still lints as
   parallel. Harness equivalent: **NONE** (0 grep hits in `lm_solver`).
   Stdlib-portable: yes.
4. **`NonLinearSystemError` taxonomy incl. `EmptySystemNotAllowed` and
   `WrongNumberGuesses`.** `ezpz/src/error.rs:34-88`, plus `TextualError` at L8-32
   (`MissingGuess`, `UnusedGuesses`, `UndefinedPoint`). A two-layer refusal set:
   problem-statement errors vs numeric errors. Note `EmptySystemNotAllowed` is reused
   at `find_dof.rs:44` and L65 as the *rank-deficiency* error path, which is arguably
   a mislabel worth **not** copying. Harness equivalent: partial (`FreedomReport`,
   `is_underconstrained`, but no named error enum). Stdlib-portable: yes.
5. **Two epsilon regimes, and the reason for the split.** `ezpz/src/lib.rs:43`
   `const EPSILON: f64 = 1e-4` (satisfaction/degeneracy/lint threshold, used at
   `lib.rs:359-361` and ~25 sites in `constraints.rs`) vs `solver.rs:70-77`
   `Config::default{max_iterations: 35, residual_tolerance: 1e-8,
   step_tolerance: 1e-12}`. **1e-4 is four orders looser than the 1e-8 convergence
   tolerance** — deliberate: "is this constraint satisfied / is this geometry
   degenerate?" is a user-facing question and must not be as tight as the numerical
   stopping rule. Harness `lm_solver.py:155-156` ports `35 / 1e-8 / 1e-12` but **not**
   the separate 1e-4 satisfaction epsilon. Stdlib-portable: yes.
6. **`DEFAULT_INITIAL_LAMBDA = 1e-9` with a citation and an unusually honest
   comment.** `ezpz/src/solver.rs:18-23` — adapted during the solve, *"Some texts use
   lambda^2 as their scaling parameter, but it's a magic constant we have to tune
   either way so who cares"*, ref. Solomon's *Numerical Algorithms* 4.1.3.
7. **Two-stage rank tolerance in the freedom analysis.** `find_dof.rs:12`
   `TOLERANCE_BASE = 1E-8` -> L44 `tolerance = TOLERANCE_BASE * largest_diagonal`
   (**relative** to the largest R diagonal, not absolute) and L97
   `var_tol = 1e-3 * max_participation` (a *second, much looser* relative threshold
   classifying a variable as underconstrained from its nullspace participation norm).
   Two different relative scales for two different questions — rank vs per-variable
   freedom. Harness: `lm_solver.py:307-312` computes rank and `under`; worth diffing
   against these exact thresholds. Stdlib-portable: yes.
8. **Two structural solver properties worth stealing as harness self-tests.**
   `proptests.rs:236-243` `residual_jacobian_is_scale_invariant` — every residual
   must be degree-1 homogeneous in length units so the assembled system is
   well-conditioned *regardless of model size*, asserted via the equivalent degree-0
   invariant on the Jacobian; and L182-186
   `analytic_jacobian_matches_finite_difference` with a non-smoothness detector
   (`finite_difference_derivative`, L738-742) that **skips `abs()` kinks and
   angle-wrap branch cuts rather than papering over them**. These are *the* two
   bug-classes in any hand-derived Jacobian. Harness equivalent: **NONE.**
   Stdlib-portable: yes.

ALREADY COVERED: the LM algorithm itself, `Config` defaults, and the Jacobian-rank
freedom report -> `core/lm_solver.py` (docstring L31 cites "35 iterations, 1e-8
residual tol, 1e-12 step tol"; the module header explicitly says "ported from ezpz
(Rust)"). Also `core/constraints`, `core/state/constraint_model`.

ALSO PRESENT, NOT CHASED (flagged honestly): `test_cases/` = 28 human-written
`problem.md` files in the textual format, several self-describing —
`arc_line_coincident_bug/` ships **both** `problem.md` and
`problem_without_arc_constraint.md` (a bug repro plus its isolating control), plus
`inconsistent/`, `underconstrained/`, `underdetermined_lines/`, `nonsquare/`,
`tiny/`, `massive_parallel_system/` (with a Python generator). That is a small
**labelled solvability corpus** (solvable / inconsistent / underconstrained).
`fuzz/fuzz_targets/fuzz_target_1.rs` exists but has **no committed crash corpus**.

VERDICT: **mine-further** — the 5 seeds, the degeneracy prose, the warning taxonomy,
the 1e-4/1e-8 split, and the `test_cases/` solvability corpus.

---

## angelcad-master (393 files)

LICENSE: **GPL-2 or GPL-3** (`LICENSE.txt:1-3`, Carsten Arnholm 2017; plus
`LICENSE.GPL2`, `LICENSE.GPL3`) -> **facts-with-citation only. Nothing vendorable.**

WHAT IT IS: **not a kernel.** It is the AngelCAD IDE (`AngelCAD/`, wxWidgets),
viewer (`AngelView/`, OpenGL), the `as_csg` AngelScript language bindings that emit
an XCSG XML file, and thin CLI wrappers (`polyfix/`, `csgfix/`, `csgtext/`,
`dxfread/`). Confirmed by its own docs (`setup/doxygen/mainpage.h:47`): all 3D
computation is done by **xcsg** (-> carve, Clipper, qhull, libtess2) and all mesh
healing by **polyhealer** — *neither is in this repo* (`polyfix/main.cpp:20-21`
includes `"polyhealer/polyhealer.h"`, a missing external dependency).

READ (in full): `LICENSE.txt` header; `polyfix/main.cpp` (372L); `csgfix/main.cpp`
constants; `setup/doxygen/mainpage.h`; every `throw` site with surrounding predicate
in `as_csg/polygon.cpp`, `polyhedron.cpp`, `polyhedron_face.cpp`, `circle.cpp`,
`spline2.cpp`, `spline3.cpp`, `spline_path.cpp`, `minkowski2d/3d.cpp`,
`offset2d.cpp`, `pointcloud.cpp`, `shape.cpp`; full tolerance grep across all
`.cpp`/`.h`.

SKIMMED-NOT-READ: all of `AngelCAD/` (~40 files) and `AngelView/` (~28) — wxWidgets
GUI, out of scope; the ~55 remaining `as_csg/*.cpp` transform/primitive wrappers
(each a ctor plus an XML-node emitter); `setup/debian`, `setup/windows`,
`setup/doxygen/angelcad.h` (generated API dump); `.cbp` project files.

FINDINGS

1. **A complete CSG-primitive refusal-predicate table with user-facing messages.**
   `as_csg/polygon.cpp:49-143`, `polyhedron_face.cpp:39-79`, `polyhedron.cpp:75-109`,
   `circle.cpp:56-58`, `spline2.cpp:48`, `spline3.cpp:45`, `spline_path.cpp:44-62`,
   `minkowski2d.cpp:44-45`, `offset2d.cpp:82-86`. Every construction-time rejection,
   verbatim: polygon needs **>=3 points**; polygon must have **area > 0**
   (`!(area > 0)` — note this also rejects negative/reversed-winding area, not just
   zero); polygon must **not self-intersect**; polygon-from-spline needs **>=2
   segments**; polygon-from-radius needs **r > 0 and >=3 points**; a
   `polyhedron_face` needs **>=3 indices != -1** and **no repeated vertex index
   within a face**; circle radius must be **> 0**; splines need **>=2 points**;
   `spline_path` requires **len(points) == len(vectors)**; minkowski operands must be
   **non-null**; `offset2d` requires **exactly one of `r` or `delta`** — never both,
   never neither. Together: a ready-made input-validation contract for a CSG DSL
   front-end, and exactly the refusals a text-to-CAD agent must produce *before*
   invoking any kernel.
   Harness equivalent: partial and scattered (`mesh/intersection_repair`,
   `domain/editing/curve_degeneracy`); **no single primitive-argument validation
   table** — verified by grepping the registry index for
   `refus`/`reject`/`self-intersect`/`degener`/`primitive` (hits are all ML/data
   modules). Stdlib-portable: yes (facts only — reimplement, do not copy GPL code).
2. **Self-intersection test uses a two-epsilon scheme, point vs parameter.**
   `as_csg/polygon.cpp:22` — `static const double epspnt=1.0E-4, epspar=1.0E-3;`
   passed to `poly2d.is_self_interesecting(epspnt, epspar)` at L69, L97, L141. A
   **point-space** tolerance (1e-4 mm) and a **curve-parameter-space** tolerance
   (1e-3, 10x looser) are deliberately different quantities. Most reimplementations
   use one epsilon for both and get spurious self-intersections at segment joins.
   No inline justification comment, so this is an observed constant, not a documented
   one — cite it as such. Harness equivalent: **NONE** that separates the two spaces.
   Stdlib-portable: yes.
3. **Mesh-healing tolerance defaults, with units, and the remesh interaction.**
   `polyfix/main.cpp:43-45` and identically `csgfix/main.cpp:36-38`:
   `dist_tol = 1.0E-2` *"coordinate tolerance in mm"*, `area_tol = 1.0E-6` *"area
   tolerance in mm^2"*, `maxiter = 10`. Note **`area_tol == dist_tol^2`** —
   dimensionally consistent by construction. Critically, `polyfix/main.cpp:59`:
   *"[remesh] Heal & remesh surfaces to given edge length (set dtol to small value
   ~1.E-6)"* — **the healing tolerance must be dropped 4 orders of magnitude before
   remeshing**, or healing eats features the remesher needs. That interaction warning
   is the real find. Harness equivalent: `isotropic_remesh` and `repair_toolkit`
   exist; the dtol<->remesh coupling constraint is not obviously encoded.
   Stdlib-portable: yes.
4. **The heal pipeline's stage order and its "lumps" concept.**
   `polyfix/main.cpp:277-311` — per polyhedron:
   `polyhealer(poly, dist_tol, area_tol)` -> `run_healing(maxiter)` (**bounded,
   iterative, returns a warning string per polyhedron**) -> optional
   `find_lumps(flip_faces)` (split into connected components) -> optional
   `polyremesh(poly, dist_tol, edge_len).flip_split()` then `aspect_ratio_flip()`.
   Healing is iterative-to-fixpoint with a **hard iteration cap of 10** and produces
   a *per-object warning summary* rather than pass/fail — the honest shape for a
   repair API. `-nflip` ships marked **"(experimental)"** (`main.cpp:57`) and
   `--lumps` is documented as **"not supported for stl output"** (L56) — a real
   format/feature incompatibility.
5. **`secant_tolerance` — the single knob converting exact CSG to mesh.**
   `as_csg/shape.cpp:91-113` (only written into the XCSG file when `> 0.0`; sentinel
   `-1.0` = "kernel default"), `setup/doxygen/mainpage.h:55` *"The secant tolerance
   controls the density of the generated mesh"*, example value `0.01`. A named
   single-scalar tessellation contract with an explicit "unset" sentinel.

ALREADY COVERED: the AngelCAD CSG operator/primitive vocabulary ->
`domain/programs/ast/csg_vocabulary` (*"Cross-family CSG vocabulary superset —
RapCAD / **AngelCAD** / OpenJSCAD / replicad"*).

**NO committed test corpus with expected geometry.** Verified:
`find . -name "*.as"` returns exactly one file, `setup/doxygen/doc.as` (a
documentation stub). No `test/`, no `examples/`, no reference meshes, no golden
outputs anywhere in the 393 files. The only model source in the repo is the ~12-line
cube-intersect-sphere-minus-3-cylinders snippet inlined at
`setup/doxygen/mainpage.h:20-42`.

VERDICT: **mine-further, narrowly** — items 1-3 only (refusal table, two-epsilon
self-intersection, dtol/atol/remesh coupling), all as **cited facts, never code**
(GPL). Everything else is GUI, bindings, or lives in the absent xcsg/polyhealer
repos. If only one thing is taken, take the refusal-predicate table.

---

## replicad-main (393 files)

LICENSE: **MIT** (`LICENSE:1`, QuaroTech Sarl 2023) -> vendorable with attribution.

WHAT IT IS: a TypeScript CAD library wrapping OpenCascade.js. **It is not mostly
glue — I expected it to be and it is not.** `packages/replicad/src/lib2d/` and
`src/blueprints/` are ~20 files of genuine hand-written 2D geometry (curve
intersection, planar booleans, offsetting, stitching, corner filleting) built *on
top of* OCCT's 2D primitives, with real workaround scar tissue. The other 7 packages
(`studio/`, `replicad-app-example/`, `replicad-threejs-helper/`, `replicad-cli/`,
`replicad-docs/`, `replicad-opencascadejs/`, `replicad-evaluator/`) **are** UI/build
glue and nothing is claimed about them.

READ (in full): `packages/replicad/src/constants.ts`; `lib2d/stitching.ts`;
`lib2d/offset.ts` (offset + collapse logic); `lib2d/intersections.ts:20-100`;
`blueprints/booleanOperations.ts:180-200, 400-425`; `blueprints/offset.ts:15-60`;
`lib2d/customCorners.ts` guard sites; `shapeHelpers.ts:95-112`; `Sketcher.ts:270-295`;
`shapes.ts:95-115`. Exhaustive greps for every tolerance literal and every
`throw new Error` across all `.ts`.

SKIMMED-NOT-READ: the ~2000-line `shapes.ts`, plus `draw.ts`, `geom.ts`, `finders/**`,
`projection/**`, `meshShapes.ts`, `Sketcher2d.ts`; all 7 non-core packages.

FINDINGS

1. **A four-tier tolerance ladder, each tier answering a different geometric
   question.** `blueprints/booleanOperations.ts:15` `PRECISION = 1e-9` (curve-curve
   intersection in planar booleans) · `blueprints/offset.ts:24` `PRECISION = 1e-8`
   with `samePoint(..., PRECISION * 10)` (**point-merge deliberately 10x looser than
   the algorithmic precision**) · `lib2d/stitching.ts:6` `precision = 1e-7` (curve
   **endpoint** matching for wire stitching — the loosest, because chained
   approximation error accumulates at joints) · `lib2d/approximations.ts:10,77`
   `tolerance = 1e-4` and `makeCurves.ts:285` `1e-3` (B-spline **approximation**
   tolerance, 5-6 orders looser than intersection).
   The ladder itself — 1e-9 intersect / 1e-8 offset / 1e-7 stitch / 1e-4..1e-3
   approximate — is the reusable fact. Getting these in the wrong order is *the*
   classic 2D-kernel bug. Harness equivalent: `parametric/offset_nurbs` has its own;
   no unified ladder. Stdlib-portable: yes.
2. **`PRECISION / 100` with a stated reason — an OCCT robustness bug worked around
   by going *tighter*.** `blueprints/booleanOperations.ts:189-193`:
   `// The algorithm used here seems to fail for smaller precisions (it detects
   overlaps in circle that do not exist` then
   `intersectCurves(thisCurve, otherCurve, PRECISION / 100)`. Counterintuitive and
   exactly the kind of thing only learned by shipping: at 1e-9 the intersector
   reports **phantom common segments between circles**; at 1e-11 it does not. Note
   the comment says "fail for smaller precisions" while the code passes a *smaller*
   number — the comment means smaller *tightness*, i.e. larger epsilon. Record with
   that ambiguity flagged. Harness equivalent: **NONE.**
3. **Offset collapse handling — the honest degenerate return, twice.**
   `lib2d/offset.ts:42-58`: if `newRadius = radius + orientedOffset < 1e-10` the arc
   has collapsed, and instead of failing it returns
   `{collapsed: true, firstPoint, lastPoint}` with endpoints projected through the
   centre — *"We replace collapsed arcs by a segment of line"*. And L99-108: after
   approximating a general offset curve as a B-spline it runs `selfIntersections()`
   on the result and, if non-empty, **also** returns `{collapsed: true, ...}` —
   *"We need a better way to handle curves that self intersect, for now we replace
   them with a line"* (an admitted-incomplete workaround). Plus L93-96: *"While
   return the offset curve itself would be the more correct thing to do, opencascade
   does some weird stuff with it (for instance after mirroring it). This approximates
   it with a continuous bspline"* — the offset is B-spline-approximated **not for
   accuracy but to dodge an OCCT representation bug**.
   Three separate cited degeneracy decisions in 70 lines, including a self-admitted
   TODO. Harness equivalent: `parametric/offset_nurbs` **refuses** collapse (L700,
   L720, L852: "offset distance %g collapses circle of radius %g") where replicad
   **degrades to a line segment**. That is a genuine **design fork worth recording,
   not a gap**. Stdlib-portable: yes.
4. **Wire stitching via a spatial index plus an explicit infinite-loop guard.**
   `lib2d/stitching.ts:1-80`: builds a Flatbush R-tree over curve **start points**
   inflated by +/-`precision`, then walks endpoint->neighbour, tie-breaking by
   **cyclic index distance** `Math.abs((currentIndex - otherIndex) % curves.length)`
   — i.e. prefer the curve that was *authored* next, a nice heuristic for ambiguous
   joins. Guarded by `let maxLoops = curves.length; if (maxLoops-- < 0) throw new
   Error("Infinite loop detected")` — a hard bound admitting the walk can cycle.
   Harness equivalent: `domain/geometry/topology/sew`, `domain/geometry/views/patch_stitch`
   (verified present); the cyclic-index tie-break and the loop guard are the deltas.
   Stdlib-portable: yes (Flatbush replaceable with a grid).
5. **OCCT bug swallowed in a `try/catch` inside a generator.**
   `lib2d/intersections.ts:29-36` — *"There seem to be a bug in occt where it returns
   segments but fails to fetch them"*: `intersector.Segment(i, h1, h2)` is wrapped in
   try/catch and `continue`s on throw, i.e. `NbSegments()` over-reports. Pure kernel
   scar tissue about the most-used OCCT 2D intersector.
6. **Two more admitted-broken spots.** `shapeHelpers.ts:103` *"We do not GC this
   surface (or it can break for some reason)"* — a cylindrical surface underlying a
   helix edge must outlive its builder (a lifetime bug); `Sketcher.ts:281` *"This does
   not work, we may need to hack a bit more within makeEllipseArc"* immediately above
   `arc.wrapped.Reverse()` — **counter-clockwise elliptical arcs are known-wrong in
   the sketcher**; `blueprints/booleanOperations.ts:412` `console.error("weird
   situation")` — an unreachable-in-theory branch in the planar boolean segment
   classifier that ships as a log line.
7. **Degenerate-angle guards in corner filleting.** `lib2d/customCorners.ts:30` and
   L130: `if (Math.abs(sinAngle) < 1e-10) return null` / `return [firstCurve,
   secondCurve]` — **collinear adjacent curves cannot be filleted/chamfered, and the
   operation is a silent no-op** rather than an error. Harness equivalent:
   `domain/geometry/features/fillet_feasibility` exists; worth diffing whether it
   refuses or no-ops. Stdlib-portable: yes.
8. **Cheap refusal predicates worth harvesting wholesale.** `shapeHelpers.ts:63` and
   L151 *"The minor radius must be smaller than the major one"* (torus/ellipse); L641
   *"You need at least 3 points to make a polygon"*; L315 *"Failed to build the face.
   Your wire might be non planar."* (a diagnostic that **names the likely cause**);
   `importers.ts:181` *"STL file contains no triangles"*;
   `finders/definitions.ts:117` *"Finder has not found a unique solution"* (selector
   ambiguity as a first-class error — relevant to
   `domain/programs/expressions/cad_ref_selectors`). Note also that ~12 `throw`s are
   self-labelled *"Bug in the ... algorithm"* (`boolean2D.ts:42,71,143,381`,
   `booleanOperations.ts:148`, `customCorners.ts:41,46,54`, `offset.ts:255,307`) — an
   explicit internal-invariant-violation class, distinct from user error. **That
   two-class split (user-error vs "our algorithm is wrong") is itself the finding.**

ALREADY COVERED: replicad's CSG operator vocabulary ->
`domain/programs/ast/csg_vocabulary`; also referenced from
`io/adapters/ecosystem_catalog.py` and `domain/geometry/sdf/sweep.py`. So the repo is
*catalogued* but its 2D geometry has clearly not been mined.

VERDICT: **mine-further** — explicitly **not** nothing-here. Take the tolerance
ladder (1), the `PRECISION/100` OCCT phantom-overlap fact (2), the offset-collapse
fork vs `offset_nurbs` (3), and the user-error / "bug in our algorithm" error-class
split (8). Ignore all 7 non-`replicad` packages.

---

## sdfx-master (373 files; 177 .go)

LICENSE: **MIT** (`LICENSE`, Copyright (c) 2017-2019 Jason T. Harris) -> vendorable
with attribution. Tables and math port directly with a citation line.

WHAT IT IS: a Go code-CAD library. Everything is an `SDF2`/`SDF3` interface
(`Evaluate(p) float64` plus a bounding box precomputed at construction). `sdf/` =
primitives, CSG, transforms, 2D->3D lifts; `obj/` = parametric real-world parts;
`render/` = marching cubes (uniform / octree / parallel octree), marching squares,
dual contouring, Delaunay, and STL/3MF/DXF/SVG/PNG writers; `examples/` = 75 model
programs, each a regression fixture.

Already mined on a prior pass and **excluded from this report**: `sdf/screw.go`,
`obj/servo.go`, `obj/gridfinity.go`, `render/dc/*`, marching cubes.

READ (in full): `LICENSE`, `CLAUDE.md`, `mk/example.mk`, `tools/stldiff/main.go`,
`tools/stldiff/run.sh`, `sdf/utils.go`, `sdf/poly.go`, `sdf/rack.go`,
`sdf/quadratic.go`, `sdf/flange.go`, `render/render.go`, `obj/nut.go`, `obj/bolt.go`,
`obj/washer.go`, `obj/hex.go`, `obj/gear.go`, `obj/pipe.go`, `obj/knurl.go`,
`obj/standoff.go`, `obj/keyway.go`, `obj/spring.go`, `obj/hole.go`, `obj/panel.go`,
`obj/tab.go`, `obj/geneva.go`, `obj/trp.go`, `obj/angle.go`, `obj/chamfer.go`,
`obj/finger.go`, `obj/shape.go`, `obj/display.go`, `examples/test/SHA1SUM`.

Partially read (substantial sections): `obj/panelbox.go` (1-200 of 400),
`sdf/cams.go` (155-200), `obj/draincover.go` (1-60), `render/march3.go` (1-120),
`render/march3x.go` (1-90), `sdf/text.go` (1-80), `render/delaunay.go` (1-70),
`examples/challenge/cc16.go` (1-40), `sdf/sdf3.go` (doc blocks + `sdfBox3d`).

SKIMMED-NOT-READ: grep-swept all 177 `.go` files for epsilon/tolerance/degenerate/
panic/TODO/NaN and read every hit in context; line-counted every file in `sdf/`,
`obj/`, `render/`, `vec/`. Not opened: ~40 of the 75 `examples/*/main.go`,
`sdf/matrix.go`, `bezier.go`, `spline.go`, `mesh2.go`, `mesh3.go`, `box2.go`,
`box3.go`, `line.go`, `triangle*.go`, `voxel.go`, `cache2.go`, `spiral.go`,
`gyroid.go`, `render/dc/*`, `render/{stl,dxf,svg,png,3mf,mesh,march2,march2x,dc2}.go`,
`obj/{drone,arrow,stl,gridfinity,servo}.go`, `vec/*`.
**Honest ratio: full-read 32 of 177 Go files, section-read ~14 more, grep-swept 100%.**

FINDINGS

1. **Golden-hash regression harness over rendered geometry — 75 committed `SHA1SUM`
   fixtures.** `mk/example.mk:8-15` plus `examples/*/SHA1SUM` (75 files;
   `examples/test/SHA1SUM` alone has 42 entries: `48fb28...  test15.stl`,
   `a9e78227...  flange.stl`, `e84b6cd5...  standard_pipe.stl`, ...). `make test`
   runs each example binary then `shasum -c SHA1SUM`.
   Why: exactly the pattern for "did a kernel change silently move geometry" —
   byte-exact hash of the rendered artifact, one fixture per model, committed. And
   the determinism prerequisite is solved explicitly (finding 2).
   Harness equivalent: **NONE** — the eval tree has metric protocols
   (`eval.bench.protocols.*`, `eval.bench.harness.metric_aggregation`) but no
   artifact-hash regression fixture layer. Stdlib-portable: yes (`hashlib`).
2. **Deterministic-RNG scar tissue, with the reason stated.** `sdf/utils.go:42-47`:
   *"From go 1.20 the rand.* are initialized to a random seed. Different results are
   generated every run. **We want consistent results from run to run for binary
   verification**, so we have our own local random source."* ->
   `var sdfRand = rand.New(rand.NewSource(1))`. The load-bearing invariant behind
   finding 1, and the exact failure mode (library-default seeding silently breaking
   artifact reproducibility) is one the harness will hit.
   Harness equivalent: **NONE** as a documented invariant. Stdlib-portable: yes
   (`random.Random(1)` instance, never module-level `random`).
3. **STL-diff error taxonomy: IDENTICAL / MINOR / MATERIAL, with numeric
   thresholds.** `tools/stldiff/main.go:163-176`. Canonical hash = sort the 3 verts
   of each triangle, sort all triangles, SHA1 over float32 bits (`canonicalHash`,
   L66-92) — winding- and order-independent. If hashes differ:
   `relBbox = bboxDelta/bboxSize`, `triRel = |dtris|/(trisA+1)`; **`relBbox < 1e-4 &&
   triRel < 0.01` -> MINOR (float drift), else MATERIAL (real geometry change)**.
   A machine-checkable three-valued verdict distinguishing "float noise" from "the
   part changed", with concrete thresholds. Harness equivalent: **NONE** —
   `eval.bench.protocols.chamfer_bbox_judged` does CD/F1/IoU against a reference,
   which is a similarity score, not a drift-vs-material classifier.
   Stdlib-portable: yes.
4. **Architecture-dependence of the golden hashes, documented.** `CLAUDE.md`,
   Regression Testing Model 1: *"The committed hashes are synced to amd64
   floating-point results, so other architectures may diverge."* Plus 2 documents
   `make stldiff` as the pre-landing check for core `sdf/`/`render/` changes. The
   honest limitation of finding 1 — any harness golden-artifact gate needs this
   caveat baked in or it will produce false MATERIAL verdicts on ARM CI.
5. **Schedule-40 pipe dimension table — 23 named sizes, exact OD/ID.**
   `obj/pipe.go:49-78`. `Sch40Add(nominal, OD_inch, ID_inch)`: `"1/8" 0.405/0.249`,
   `"1/4" 0.540/0.344`, `"3/8" 0.675/0.473`, `"1/2" 0.840/0.602`,
   `"3/4" 1.050/0.804`, `"1" 1.315/1.029`, `"1-1/4" 1.660/1.360`,
   `"1-1/2" 1.900/1.590`, `"2" 2.375/2.047`, `"2-1/2" 2.875/2.445`,
   `"3" 3.500/3.042`, `"3-1/2" 4.000/3.521`, `"4" 4.500/3.998`, `"5" 5.563/5.016`,
   `"6" 6.625/6.031`, `"8" 8.625/7.942`, `"10" 10.750/9.976`, `"12" 12.750/11.889`,
   `"14" 14.000/13.073`, `"16" 16.000/14.940`, `"18" 18.000/16.809`,
   `"20" 20.000/18.743`, `"24" 24.000/22.544`. Lookup with inch<->mm scaling at
   L81-106; refuses unknown units and unknown names. Load-time invariant
   `inner >= outer -> panic` (L36).
   Why: **named part -> exact dimensions**, machine-checkable ground truth. "3/4 inch
   schedule 40 pipe" is a brief a text-to-CAD system must resolve to 1.050/0.804 in.
   Harness equivalent: **NONE** — the 17 `pipe` hits in the registry are all
   `pipeline`; grep for `sch40|schedule40` returns 0. Nearest sibling is
   `domain.standards.thread_database` (also from sdfx), so the slot is
   `domain/standards/pipe_schedule`. Stdlib-portable: yes (a dict of floats).
6. **EuroRack (Doepfer 3U) panel standard — real spec constants with derived gaps.**
   `obj/panel.go:122-188`. `erU = 1.75 * 25.4`, `erHP = 0.2 * 25.4`,
   `erHoleDiameter = 3.2` mm, and the gap derivations
   `erUGap = ((3*erU) - 128.5) * 0.5`, `erHPGap = ((3*erHP) - 15) * 0.5` — the spec
   pins a 3U panel at exactly 128.5 mm tall and 3HP at 15 mm wide, and the per-unit
   gap falls out. Mount-hole margin `vMargin = 3.0`,
   `hMargin = (3*erHP*0.5) - erHPGap`. Hole-count rule: `HP < 8` -> 2 holes, else 4
   (L180-186). Another named-part -> exact-dimension table, and the gap math is the
   kind of derived constraint a generator gets wrong. "6HP eurorack blank panel" has
   one right answer. Harness equivalent: **NONE** (`panel*` -> 0 hits).
7. **Generic 2D panel generator with a string hole-pattern DSL.**
   `obj/panel.go:18-119`. `HolePattern [4]string` per edge, `"x"`=hole, `"."`=skip:
   `"xx.x.xx"` = five holes with spacing; consumed by
   `sdf.LineOf2D(hole, from, to, pattern)`. Plus per-edge margins and optional
   reinforcing ridges. A compact LLM-friendly textual encoding of a hole array along
   an edge — good CISP op-parameter shape, and trivially machine-checkable (count the
   `x`). Harness equivalent: **NONE** (`hole_pattern` -> 0).
8. **Four-part panel-box enclosure generator with tab/clearance engineering.**
   `obj/panelbox.go` (400L). `PanelBoxParms` (L144-155): size, wall, panel thickness,
   rounding, front/back inset, clearance, screw-hole dia, `SideTabs string` pattern in
   `b/B/t/T/.` (capital = with screw hole). Key rules: default fit clearance **0.05**
   when unset (L181-184), validated to `[0,1]`; panel slot gap
   `panelGap = (1.0 + 4.0*Clearance) * Panel` (L195); tab length shrunk by
   `(1 - 2*Clearance)` (L35); tab height `6*Wall` with a screw hole else `4*Wall`
   (L37-42); **overhang avoidance** — the tab root is cut at 45 degrees so it prints
   without support, `Cut3D(tab, {0, h/2, w/2}, {0,-1,1})`, commented *"add a slope
   where the tab attaches to the box, avoiding overhangs"* (L46). Refusals:
   `Hole > 0` but no `T`/`B` tabs -> error (L188-192); `midZ <= 0` -> *"the front and
   back panel depths exceed the total box length"* (L197-200).
   A complete multi-part enclosure generator with fit clearance, printability-driven
   geometry, and named refusals. `domain.fabrication.flatpack_panels` does box->2D
   panels for flat-pack; this is the complementary 3D-printed-enclosure-with-tabs
   case. Stdlib-portable: yes.
9. **Involute spur-gear tooth construction, flank-sampled.** `obj/gear.go:20-173`.
   `involuteXY(r,t) = r*(cos t + t sin t), r*(sin t - t cos t)`;
   `involuteTheta(r,d) = sqrt((d/r)^2 - 1)`. Tooth layout:
   `centerAngle = pi/(2N) + faceAngle - backlashAngle` where `faceAngle` is the
   `atan2` of the involute point at pitch radius and
   `backlashAngle = backlash/(2*pitchRadius)` (L58-61). Flank sampled from
   `involuteTheta(base, max(base, root))` to `involuteTheta(base, outer)` over
   `Facets` steps; upper flank is the mirror; origin appended to close the wedge.
   `addendum = 1.0*Module`, `dedendum = addendum + Clearance`. Seven explicit refusals
   (L106-126).
   `sdf/rack.go` is the matching linear rack: `addendum = 1.0*M`, **`dedendum =
   1.25*M`**, `pitch = M*pi`, flank `dx = (add+ded)*tan(pressureAngle)`, tooth-top
   half-width `dxt = (pitch/2 - dx)/2`, backlash `bl = Backlash/2`; `Evaluate` folds x
   through `SawTooth(p.X, pitch)` and intersects with the finite rack length (L96-105).
   Harness equivalent: `domain.geometry.kinematics.involute_gear` + `gear_train` +
   `gear_modules` + `library.gear_train` already exist — **gear generation is
   covered**. The **linear rack** is the gap: grep `rack` returns only substring hits
   (`greedy_refine`, `utility_retrieval`). Slot:
   `domain/geometry/kinematics/gear_rack`.
10. **Polygon builder with per-vertex fillet/chamfer/arc and an honest "can't smooth"
    refusal.** `sdf/poly.go`. Fluent per-vertex ops: `.Smooth(radius, facets)`,
    `.Chamfer(size)`, `.Arc(radius, facets)`, `.Rel()`, `.Polar()`. Fillet math
    (`smoothVertex`, L189-235): `theta = acos(v0.v1)`, tangent offset
    `d1 = r/tan(theta/2)`, center offset `d2 = r/sin(theta/2)`, rotation step
    `dtheta = sign(v1 x v0)*(pi-theta)/facets`. Two pieces of scar tissue:
    - **L209-212** — `if d1 > |vp-v| || d1 > |vn-v| { return false }` *"unable to
      smooth - radius is too large"*. A geometric feasibility predicate: the fillet
      silently does not apply rather than producing self-intersecting garbage.
    - **L74-81** — `Chamfer` is admittedly fake, implemented as a 1-facet smooth with
      `radius = size*sqrtHalf`, commented *"The size will be inaccurate for anything
      other than 90 degree segments, but this is easy, and I'm lazy"*. A documented
      accuracy limitation.
    Also `arcVertex` (L123-170): **the sign of the radius selects which side of the
    chord the arc lies on**; `dCenter = sqrt(r^2 - dMid^2)` will NaN if `r <
    half-chord` and is undefended. `relToAbs` (L253-266) refuses two consecutive
    relative vertices. `fixups()` order is fixed and load-bearing: relToAbs ->
    createArcs -> smoothVertices (L270-274); both passes loop to fixpoint because they
    mutate the vertex list mid-iteration.
    Harness equivalent: `domain/geometry/fillet_feasibility` exists (3D); the 2D
    per-vertex builder does not. The `r/tan(theta/2)` must-fit-inside-both-edges test
    is a second, independent, cheap 2D formulation of fillet feasibility.
11. **Octree emptiness test — the exact conservative pruning rule.**
    `render/march3x.go:46-60` and `render/march3p.go:127-158`. Precomputed LUT of cube
    half-diagonals: `hdiag[i] = 0.5*sqrt(3*s*s)` for `s = (1<<i)*resolution`.
    `isEmpty(c)`: evaluate the SDF at the cube centre; **if `|d| >= hdiag[level]` the
    surface cannot intersect the cube** -> skip entirely (`march3p.go:149-152`). Two
    more rules: `resolution = 0.5 * requested` because *"we want to test the smallest
    cube (side == resolution) for emptiness so the level 0 cube is at half
    resolution"* (`march3x.go:146-148`), and
    `levels = ceil(log2(longAxis/resolution)) + 1` (L150). Cache hit rate stated
    empirically: *"about 2/3 of lookups get a hit, overall speedup about 2x"*
    (`march3x.go:33-35`).
    Why: the correctness condition for hierarchical SDF evaluation — it **requires a
    true lower-bound distance and silently breaks on non-metric SDFs** — plus the
    off-by-one half-resolution rule that is easy to get wrong.
    Harness equivalent: `domain.geometry.volumes.marching_cubes` and
    `dual_contouring*` exist but are uniform-grid; no octree/hierarchical pruning
    module. Stdlib-portable: the pruning rule and level math yes; the
    `runtime.NumCPU()` worker pool and channel batching (`march3.go:26-50`, batch size
    100, *"performance doesn't seem to improve past 100"*) does **not** port.
12. **Knurl generated as intersecting opposite-hand multi-start threads, with a
    start-count formula.** `obj/knurl.go`. Diamond knurl =
    `Intersect3D(Screw3D(profile, L, 0, pitch, +n), Screw3D(profile, L, 0, pitch, -n))`.
    **Starts derived from the desired helix angle:
    `n = int(tau * radius * tan(theta) / pitch)`** (L66). Refuses `theta >= 90 deg`
    (L62-63). `KnurledHead3D` snaps the knurl length down to a whole number of
    pitches: `pitch * floor((h - r*0.05)/pitch)` (L91); default `theta=45 deg`, knurl
    depth `= 0.3*pitch`. A non-obvious constructive trick plus the helix-angle->starts
    formula and whole-pitch snapping. Harness equivalent: **NONE** (`knurl` -> 0);
    `domain.geometry.features.screw_thread` provides the `Screw3D` half.
13. **PCB standoff / mounting pillar with gussets, and the signed-depth convention.**
    `obj/standoff.go`. `HoleDepth > 0` = a hole (subtracted); **`HoleDepth < 0` = a
    support stub (unioned)** — one signed parameter switches the boolean (L24,
    L103-108). `NumberWebs` triangular gussets rotate-copied around the base (L33-58),
    then **intersected with a cylinder of radius `WebDiameter` to trim any web
    protruding above the pillar top** (L90-97, *"Cut off any part of the webs that
    protrude from the top"*). Board-mount pillars are a stock text-to-CAD ask and the
    web-trim intersection is the cleanup step a naive generator omits.
    Harness equivalent: **NONE** (`standoff` -> 0); `domain.standards.heatsert_bores`
    covers the insert bore but not the pillar.
14. **Hole-feature vocabulary: counterbore / chamfer / countersink / bolt circle /
    keyed hole / circular grille.** `obj/hole.go`.
    `CounterBoredHole3D(l, r, cbR, cbDepth)` (L76); `ChamferedHole3D` builds the
    chamfer as a **cone from `r` to `r+chRadius`** (L104);
    `CounterSunkHole3D(l,r) = ChamferedHole3D(l, r, r)` — i.e. **countersink is
    defined as the 45-degree case of chamfer** (L113-118);
    `BoltCircle2D/3D(holeR, circleR, n)` (L123-149); `KeyedHole2D` — round hole with 1
    or 2 flats, `KeySize` normalised to `[0,1]` of diameter, refuses
    `NumKeys not in {1,2}` (L162-176); `CircleGrille2D` (L31-62) — concentric rings of
    holes, ring count `steps = floor(grilleR/(rSpacing*holeD) - 0.5)`, per-ring hole
    count `k = floor(tau*r/(holeD*tSpacing))`, each ring **rotated by half its own
    angular step to stagger against the previous ring** (L59). The
    countersink-as-45-degree-chamfer identity and the grille stagger rule are both
    concrete checkable conventions. Harness: `chamfer` has 12 registry hits so
    chamfering exists; the hole *taxonomy* as named features does not.
15. **Keyway (shaft vs bore) — one parameter flips male and female.**
    `obj/keyway.go:42-53`. **If `KeyRadius < ShaftRadius` the key is cut *into* the
    shaft (Difference); otherwise the key stands *proud* (Union)** — the same
    parameter set generates the mating male and female profiles, which is exactly what
    a fit check needs. Harness equivalent: **NONE** (`keyway` -> 0).
16. **Geneva drive — the pin-offset law and the too-large-centre-distance refusal.**
    `obj/geneva.go:51-76`. `theta = tau/(2N)`; **`pinOffset = sqrt(d^2 + r^2 -
    2dr cos theta)`** (law of cosines on centre distance and driven radius). Slot
    length `= pinOffset + drivenR - centerDistance`. Clearance applied
    *asymmetrically and correctly*: the driven wheel shrinks by clearance, its cutouts
    grow by it. Refusals: `NumSectors < 2`, any dimension `<= 0`, `Clearance < 0`, and
    **`CenterDistance > DrivenRadius + DriverRadius` -> "center distance is too
    large"** (L47-49). A complete intermittent-motion mechanism with a closed-form
    driving relation and a real assembly-feasibility predicate.
    Harness equivalent: **NONE** (`geneva` -> 0).
17. **Tab interface — Body/Envelope, the general mating-part pattern.**
    `obj/tab.go:22-38`. A `Tab` exposes `Body(upper, M44)` (**+** to the connected
    body) and `Envelope(upper, M44)` (**-** from the connected body);
    `AddTabs = Union3D(Difference3D(s, union of envelopes), union of bodies)`. Three
    implementations: `StraightTab`, `AngleTab` (45-degree cut both ends,
    `xCut = size.X/2 - size.Z/2`, L100-102), `ScrewTab` (pillar plus through hole,
    where the *upper* part's envelope is just the hole and the *lower* part's envelope
    is the whole pillar, L154-167). Clearance applied as `size + {2c, 2c, c}` — **2x
    in-plane, 1x in depth** (L69, L109).
    A clean reusable abstraction for "male feature plus the negative it must be given
    in the mating part" — precisely the keep-in/keep-out shape
    `eval.bench.geometry.interface_match` scores. The anisotropic clearance is a real
    fit convention. Harness: `interface_match` scores mating features but there is no
    generator-side Body/Envelope contract.
18. **Sand-casting / draft-angle features.** `obj/trp.go` (truncated rectangular
    pyramid) states the intent in its header: *"particularly useful for sand-casting
    patterns because the slope implements a pattern draft and the rounded edges
    minimise sand crumbling"*. `dr = h/tan(baseAngle)`, `rb = baseR + dr`,
    `rt = max(baseR - dr, 0)`, `round = min(0.5*rt, RoundRadius)` — both clamps
    prevent inverted geometry; built as an elongated cone then cut. Refuses
    `BaseAngle not in (0, 90] deg`. `obj/draincover.go:41-60` applies the same idea:
    draft offsets `dx = 0.5 * thickness * tan(draft)` applied per level to a revolved
    polygon, with separate `WallDraft` and `GrateDraft`.
    Draft angle is a manufacturing constraint (casting, injection moulding) with an
    explicit geometric consequence, and the two clamps are degeneracy guards worth
    copying. Harness: `eval.bench.protocols.dfm_scoring` *scores* DFM but nothing
    *generates* draft.
19. **Structural-angle (L-section) profile with root fillet, fully guarded.**
    `obj/angle.go:31-69`. Independent `X`/`Y` leg length and thickness, inside root
    fillet via `.Smooth(RootRadius, 6)`. Six refusals including the two non-obvious
    cross-constraints: **`Y.Thickness >= X.Length`**, **`X.Thickness >= Y.Length`**,
    and `RootRadius > (X.Length - Y.Thickness)` / `> (Y.Length - X.Thickness)`
    (L44-58) — the fillet must fit in the remaining web on both legs. Extrusion-profile
    generation with a complete correct feasibility predicate set; the
    fillet-fits-in-the-web check is the 2D companion to finding 10.
    Harness equivalent: **NONE**.
20. **Exact analytic SDF for a three-arc flange/cam profile (arc-region dispatch).**
    `sdf/flange.go:31-79` and `sdf/cams.go:155-198`. Both precompute the tangent/flank
    geometry once in the constructor, then `Evaluate` **dispatches on which arc region
    the query point falls into** and returns the exact distance for that region — the
    flange projects onto the flank line via `t = v.u` and branches on `t<0` / `t<=l` /
    `t>l`; the three-arc cam branches on `atan2` angle vs `thetaBase`/`thetaNose`.
    Flank-centre solution (`cams.go:159-163`): `y = (r0^2 - r1^2 + d^2)/(2d)`,
    `x = -sqrt(r0^2 - y^2)` with `r0 = flankR - baseR`, `r1 = flankR - noseR` —
    circle-circle intersection.
    **Documented failure mode:** `sdf/cams.go:172` — *"TODO fix this - it's wrong if
    the flank radius is small"* on the bounding box. A known-wrong precomputed bound,
    which under the octree pruning rule of finding 11 means **silently missing
    geometry**. Harness: `domain.geometry.sdf.cam_profile` **already covers the cam**
    (mined from this repo). The **flange** SDF and, more importantly, **the recorded
    bounding-box failure mode**, are not covered.
21. **Blend/min-function library, with two honestly-flagged broken entries.**
    `sdf/utils.go:124-182`. `RoundMin(k)` (quarter-circle join), `ChamferMin(k)`
    (45-degree chamfer, `min(min(a,b), (a-k+b)*sqrtHalf)`), `ExpMin(k)` (k~32),
    `PowMin(k)` (k~8), `PolyMin`/`PolyMax(k)` via
    `poly(a,b,k) = mix(b,a,h) - k*h*(1-h)`, `h = clamp(0.5 + 0.5(b-a)/k, 0, 1)`.
    Two carry admissions of failure: `ChamferMin` — *"TODO: why the holes in the
    rendering?"* (L136); `PowMin` — *"TODO - weird results, is this correct?"* (L151).
    A smooth-blend catalogue plus a documented note that two of five are unreliable —
    the "which blend do I actually trust" knowledge otherwise learned by wasting a day.
    Harness equivalent: **NONE** found for the blend catalogue.
22. **Two-tier epsilon constants and float-comparison helpers.**
    `sdf/utils.go:36-38`: `sqrtHalf = 0.7071067811865476`, **`tolerance = 1e-9`**
    (geometric), **`epsilon = 1e-12`** (numeric). `EqualFloat64` at L349-355 is a plain
    absolute-difference test — and immediately above it (L326-347) sits a
    **commented-out relative-error implementation** with
    `minNormal = 2.2250738585072014e-308`, cited to
    `floating-point-gui.de/errors/NearlyEqualsTest.java`. `SnapFloat64(a,b,eps)` snaps
    a to b within eps; `ZeroSmall(x, y, eps)` zeroes x when `|x|/y < eps`
    (relative-to-a-reference-magnitude zeroing, used in `bezier.go:155-159` on curve
    coefficients relative to their sum).
    The abs-vs-relative comparison decision is preserved **with the rejected
    alternative left in the file**. `ZeroSmall`'s scale-relative zeroing is a good
    pattern for cleaning near-zero polynomial coefficients.
23. **Degeneracy handling in the quadratic solver — a 4-valued result including
    `infSoln`.** `sdf/quadratic.go:15-49`. Returns `(roots, qSoln)` with
    `qSoln in {zeroSoln, oneSoln, twoSoln, infSoln}`, fully branching on `a==0` ->
    `b==0` -> `c==0` and giving **`infSoln` (every x is a root)** as a distinct outcome
    from "no roots". Carries its own honest defect marker: **L26
    `// TODO Fix all comparisons to 0`** — the author knows exact `== 0` tests on
    floats are wrong here. The canonical worked example of "the degenerate case is a
    distinct return value, not an exception", including the acknowledged remaining bug.
    Harness equivalent: **NONE** (`quadratic` -> 0).
24. **Error convention: caller-located messages, and panic reserved for invariants.**
    `sdf/utils.go:377-385` — `ErrMsg(msg)` uses `runtime.Caller(1)` to prefix every
    error with the **function name and line number** that raised it, so
    `"obj.Nut line 76: Tolerance < 0"`. Applied with total consistency: every one of
    the ~30 `obj/` constructors validates parameters and returns `(SDF, error)`. The
    rule is written down in `CLAUDE.md` Conventions (from `docs/ROADMAP.md`): *"Every
    SDF-generating function should return `(SDFx, error)`. Use the error for bad
    parameters and propagate it to the ultimate caller. **Reserve `panic` for
    fundamental code problems, not parameter validation.**"* And it is honoured: the
    only `panic` in `obj/` is `pipe.go:36`, a load-time table-integrity check, which is
    genuinely an invariant not user input.
    A stated, enforced, verifiable refusal discipline plus a mechanically-located error
    format. The refusal predicates themselves, collected across `obj/`, are a
    ready-made validation corpus. Stdlib-portable: yes —
    `inspect.currentframe()` or a decorator gives the same locator.
25. **Miscellaneous smaller items.**
    - `sdf/utils.go:115-120` `SawTooth(x, period)` returning `[-period/2, period/2)` —
      the domain-folding primitive that makes periodic SDFs (rack, thread) O(1).
    - `sdf/poly.go:353-365` `Nagon(n, radius)`; refuses `n < 3`.
    - `obj/hex.go:22-27` — rounded hexagon as
      `Offset2D(Nagon(6, radius - 2*round/sqrt(3)), round)`; the `2r/sqrt(3)` inradius
      correction is the non-obvious bit. `HexHead3D` (L42-68) rounds the head by
      intersecting with a sphere of radius `1.6*r` offset by
      `sqrt(R^2 - d^2) - h/2` where `d = r*cos(30 deg)`.
    - `obj/chamfer.go:16-33` `ChamferedCylinder(s, kb, kt)` — **derives length and
      radius from the SDF's own bounding box** (`s.BoundingBox().Max.Z/.X`) and
      intersects with a revolved chamfered profile, retrofitting chamfers onto an
      arbitrary solid.
    - `obj/washer.go:77-91` — the partial washer (`Remove` fraction) is built by
      revolving a translated box through `theta = tau(1-Remove)` then rotating by
      `(tau-theta)/2` **to centre the removed sector on the +x axis** (a
      determinism/orientation convention). Refuses `Remove not in [0,1)`; `Washer2D`
      honestly refuses `Remove != 0` with *"TODO support Remove != 0"* rather than
      silently ignoring it.
    - `obj/spring.go` — 3D-printable flat serpentine spring; `SpringLength()` gives the
      closed form `Boss[0] + Boss[1] + WallThickness*(N-1) + (Diameter -
      2*WallThickness)*N`. Note it **mutates the input params** to clamp
      `Boss[i] < WallThickness` up to `WallThickness` (L58-63).
      `domain.geometry.features.serpentine` already covers the generator; the
      closed-form length and the boss clamp are the increment.
    - `obj/display.go` — LCD/OLED mount: `Display(k, negative bool)` returns **either
      the positive supports or the negative window+holes from the same parameters**,
      the same dual-polarity idea as finding 17.
    - `render/delaunay.go:57-70` `TriangleI.Canonical()` — rotate a triangle's indices
      to lowest-first **while preserving winding order**, the basis for
      order-independent mesh comparison (cf. finding 3). Cites O'Rourke
      *Computational Geometry* 2e Code 5.1 and Paul Bourke.
    - `sdf/text.go:47-80` — TrueType glyph -> SDF2. Handles the quadratic-B-spline
      **implicit on-curve point at the midpoint of two consecutive off-curve points**
      rule (L66-70), and accumulates
      `sum += (v.X - vPrev.X)*(v.Y + vPrev.Y)` to determine **contour winding
      direction** (shoelace) so holes are subtracted rather than added.
      `domain/geometry/winding` exists; the TrueType implicit-midpoint rule
      specifically does not.
    - `sdf/sdf3.go:33-64` — the branchy `sdfBox3d` is kept over the elegant 3-line
      `d.Max(0).Length() + min(d.MaxComponent(), 0)` version, which sits commented out
      directly above it. Undocumented reason, but the choice was deliberate.
    - `examples/challenge/` — CAD Challenge #16/#18 parts transcribed from the r/cad
      challenge threads (`cc16.go:15-16` cites the reddit URL) with named dimension
      variables (`base_w = 4.5`, `base_d = 2.0`, `base_h = 0.62`, `slot_r = 0.38/2`).
      Two more graded text-to-CAD briefs with reference geometry **and** committed
      output hashes, alongside the existing `eval.bench.imports.*` brief sets.

ALREADY COVERED: `sdf/screw.go` thread tables, `obj/servo.go`, `obj/gridfinity.go`,
`render/dc/*`, marching cubes (excluded per brief); involute spur gears ->
`kinematics.involute_gear`/`gear_train`/`gear_modules`/`bevel_gear`/`library.gear_train`
(only the linear rack is new); cam profiles -> `sdf.cam_profile` (only the
bounding-box TODO is new); Archimedean spiral -> `sdf.spiral`; twist/scale extrusion ->
`sdf.sweep`; sphere tracing/raycast -> `numeric.sphere_tracing`; Bezier/spline ->
`parametric.bezier`, `catmull_rom`, `nurbs_curve`, `numeric.nurbs_basis`,
`io.ingest.bezier`; 3MF/DXF/SVG/STL writers -> `io.formats.threemf`,
`threemf_extensions`, `io.formats.dxf`, `io.formats.svg`, `io.formats.obj`; serpentine
spring -> `features.serpentine`; winding-order determination ->
`domain/geometry/winding`; flat-pack box panels -> `fabrication.flatpack_panels`;
fillet feasibility -> `domain/geometry/fillet_feasibility`.

VERDICT: **mine-further.** Sharpest, in order: the **SHA1SUM + stldiff regression
apparatus** (1-4) — a working, honest, reproducible geometry-regression discipline
with numeric drift-vs-material thresholds, a stated determinism invariant, and a
stated architecture caveat; the **Sch40 pipe** and **EuroRack** tables (5, 6) as
machine-checkable named-part -> dimension ground truth; **panelbox** (8) as a
complete enclosure generator with clearance and printability rules; the **`obj/`
refusal-predicate corpus and `ErrMsg` convention** (24) as a ready-made validation
dataset; and the **polygon fillet/chamfer builder** (10) with its "radius too large"
predicate and admitted chamfer inaccuracy. Second tier: rack, knurl, standoff,
keyway, geneva, tabs, hole taxonomy, draft-angle features, angle profile, octree
pruning rule.

Everything of value is a table, a closed-form formula, or a validation predicate —
all pure-Python-stdlib portable. The only non-portable material is the
`runtime.NumCPU()` worker pools and channel-batched evaluation in `render/march3.go`
/ `march3p.go`, plus the `freetype`/`golang.org/x/image` dependency in `sdf/text.go`.

---

## libfive-master (321 files; ~157 is the C++ source-only subset)

LICENSE: **no root LICENSE file.** `README.md:54-63` plus every source header: the
`libfive` library/stdlib is **MPL-2.0**; the `Studio` GUI is **GPL-2.0-or-later**.
MPL is file-level copyleft -> **facts-with-citation only** for the kernel (constants,
algorithms and rules are facts; do not paste code). The only `LICENSE`-named file in
the tree is `studio/font/SIL Open Font License.txt` (a font).

WHAT IT IS: Matt Keeter's f-rep (implicit/SDF) CAD kernel. An interval-arithmetic
evaluator over a tape IR, an octree/quadtree worker pool, three meshers (Dual
Contouring, ISO-Simplex, Hybrid), a `VolTree` acceleration structure, and an oracle
interface for opaque geometry.

READ (in full): `include/libfive/render/brep/settings.hpp` (76L);
`include/libfive/render/brep/simplex/qef.hpp` (685L — the entire QEF class);
`include/libfive/render/brep/dc/intersection.hpp` (107L);
`include/libfive/render/brep/dc/dc_flags.hpp`; `src/render/brep/vol/vol_tree.cpp`
(213L); `src/render/brep/dc/dc_tree3.cpp` (the `leafsAreManifold` /
`cornersAreManifold` specializations); `src/render/brep/manifold_tables.cpp`;
`include/libfive/eval/interval.hpp`; `test/util/mesh_checks.cpp`;
`test/util/shapes.hpp`; `src/render/brep/dc/dc_tree.inl` L1-120, L250-470, L600-830;
`src/render/brep/hybrid/hybrid_tree.inl` L430-530.

SKIMMED-NOT-READ: ~45 remaining `.cpp/.inl` under `src/render/brep/`
(simplex_mesher, hybrid_mesher, neighbor/edge/marching table generators, object
pools, worker pools, `dual.hpp` walk); all of `src/eval/*` except `feature.cpp` /
`eval_feature.cpp` grep hits; all of `libfive/stdlib/`, `libfive/bind/`; all of
`studio/` (GPL GUI, deliberately untouched); ~30 of 37 test files beyond the
TEST_CASE-name inventory plus the 5 read.

FINDINGS

1. **QEF `solveBounded` — descent through subspaces instead of clamping.**
   `qef.hpp:325-377` plus `UnrollDimension` at L439-482. Solve unconstrained; if the
   vertex escapes the cell, re-solve constrained to every face, then every edge, then
   every corner, in *descending dimension*, taking the lowest-error solution that
   stays in bounds; terminate at -1D. Tie-break at L507-509 prefers an in-region
   solution at equal error. The region is pre-shrunk by `shrink = 1 - 1e-9` (L325) so
   a vertex exactly on the boundary is not accepted.
   Why: the correct answer to "the QEF wandered out of its cell", and it preserves
   sharp features that naive clamping destroys.
   Harness equivalent: **NONE** —
   `domain/geometry/volumes/dual_contouring.py:233-237` explicitly does the naive
   thing: `# clamp the vertex into the cell (QEF can wander for near-degenerate data)`
   then `min(max(pos[0], x0), x1)`. Verified by reading both. Stdlib-portable: yes
   (needs a symmetric eigensolver; the harness already ships `_jacobi_eigen`/`_jacobi3`).

2. **Two different eigenvalue cutoffs, with the reason written down.**
   `qef.hpp:551-552` — the general solver uses
   `eigenvalue_cutoff_relative = 1e-12`, `absolute = 0`. But `solveDC`
   (`qef.hpp:643-644`) calls it with `(0, 0.1)` — relative off, **absolute 0.1** —
   and `rankDC()` (L281-297) hard-codes `> 0.1`, commented *"This assumes that the
   normals have been normalized, because it uses an absolute threshold"*.
   `dc_flags.hpp:12` defines `EIGENVALUE_CUTOFF 0.1` as *"Eigenvalue threshold for
   determining feature rank"*. The long-form reason is at `dc_tree.inl:764-777`: when
   derivatives are **not** normalized, 0.1 *"can cause one feature to be entirely
   ignored if its derivative is fairly small in comparison to another"*, so the cutoff
   becomes `highest_val * EIGENVALUE_CUTOFF^2`; and if `highest_val <= 1e-20` the
   cutoff is set to **+infinity** deliberately, forcing `D = 0` so the mass point is
   used — *"the best we can do without good gradients"*.
   Why: rank determination *is* the sharp-feature decision, and the harness uses one
   arbitrary relative tolerance for it.
   Harness equivalent: `volumes.dual_contouring` (`QEF.solve`, `svd_tol=1e-6`
   relative) and `dual_contouring_3d` (`QEF_EIGEN_CUTOFF`, relative) — **neither has
   the normalized/unnormalized split, neither has the infinite-cutoff degenerate
   fallback.** Stdlib-portable: yes.

3. **Topology-safety collapse rules (Ju et al. 2002).** A cell may only be collapsed
   if **three** conditions hold (`dc_tree.inl:645-655`): `cornersAreManifold(corner_mask)`
   (a precomputed table), *all children are themselves manifold*, and
   `leafsAreManifold(children, corners)`. The third is spelled out at
   `dc_tree3.cpp:20-27`: the sign at the middle of a coarse edge must agree with >=1
   of its 2 endpoints; at the middle of a coarse face with >=1 of its 4 corners; at
   the cube centre with >=1 of its 8 corners — then enumerated exhaustively as
   `edges_safe && faces_safe && center_safe`. Separately,
   `manifold_tables.cpp:19-54` builds the table by flood-filling the inside set over
   the subspace boundary graph and requiring `connected_inside == b`, i.e.
   *"topologically equivalent to a disk"*, with all-empty and all-filled explicitly
   rejected.
   Why: this is the only thing standing between adaptive DC and a non-manifold mesh,
   and it is a **refusal predicate** — "do not simplify here".
   Harness equivalent: **NONE.** Grepping `src/harnesscad/` for
   `corners_are_manifold|cornersAreManifold|manifold_table|topology.safety|Ju et al`
   yields one hit: a prose mention in `volumes/dual_contouring_3d.py` acknowledging
   the output *"can make the output non-manifold"*. The harness DC has no collapse
   step at all. Stdlib-portable: yes (pure bitfield tables plus flood fill).

4. **The collapse gate is triple, not single.** `dc_tree.inl:701-724`: collapse
   requires (a) QEF error `< max_err` — squared if `LIBFIVE_LINEAR_ERROR`;
   (b) `region.contains(vert(0), 1e-6)` — the collapsed vertex must lie in the cell
   with `1e-6` slack; and (c) `fabs(eval->value(v)) < max_err` — **an independent
   re-evaluation of the field at the proposed vertex**. Fail any one and the leaf is
   returned to the pool and the branch survives. `settings.hpp:50-53`: `max_err`
   default `1e-8`, and `-1` is documented as *"completely disable cell merging"*.
   Why: condition (c) is the part everyone omits — error-metric agreement is not the
   same as being on the surface. Harness equivalent: **NONE.** Stdlib-portable: yes.

5. **Interval `maybe_nan` as a refusal predicate — with a subtlety the harness got
   right and one it may not have.** `interval.hpp:59` `isSafe() { return !maybe_nan; }`;
   `state()` at L66-77 returns `AMBIGUOUS` whenever `maybe_nan`, **regardless of
   bounds**. Both `DCTree::evalInterval` (`dc_tree.inl:100-104`) and
   `VolTree::evalInterval` (`vol_tree.cpp:49-54`) `assert(type == AMBIGUOUS)` and
   return the **un-pushed** tape — i.e. an unsafe interval forbids tape
   specialization, not just pruning. The min/max NaN-propagation asymmetry is
   documented at `interval.hpp:84-90`: Eigen's `std::min` returns NaN iff the *first*
   input is NaN, so `maybe_nan` is copied from the first input only.
   Harness equivalent: `domain/numeric/interval_arithmetic` — **COVERED for the flag
   itself** (`maybe_nan` slot, `sqrt`/`log` domain guards, `classify` -> AMBIGUOUS).
   **NOT covered:** the asymmetric min/max NaN rule (`Interval.min` at L155-157 ORs
   both flags — safe but conservative, and divergent from libfive's documented
   Eigen-compat behaviour), and the "unsafe interval implies do not specialize the
   tape" consequence, which has no harness analogue.

6. **Edge-crossing search is 4x16 sampling, not bisection, and the
   numerical-disagreement workarounds are load-bearing.** `dc_tree.inl:339-341`:
   `SEARCH_COUNT = 4`, `POINTS_PER_SEARCH = 16` (an effective 16^4 = 65536-way
   subdivision). Three hard-won guards: (a) L374-377 — the loop starts at `j=1`
   because *"the very first point is already known to be inside the shape (but
   sometimes, due to numerical issues, it registers as outside!)"*; (b) L386-392 —
   when a sample reads exactly `0` it re-tests with the *feature evaluator*
   `isInside<N>` rather than trusting the sign; (c) L396-403 — if no crossing is
   found by the last sample the last interval is taken anyway, *"working around
   numerical issues where different evaluators disagree with whether points are inside
   or outside"*.
   Why: exactly the failure modes a bisection-based crosser hits silently.
   Harness equivalent: `dual_contouring._bisect_edge` / `dual_contouring_3d._crossing`
   — plain bisection, **none of the three guards**. Stdlib-portable: yes.

7. **Ambiguous-point handling: multi-derivative feature expansion.**
   `dc_tree.inl:426-448`: for each intersection sample `eval->getAmbiguous()` is
   consulted; if the point is ambiguous (a min/max seam, i.e. a crease) the single
   derivative is discarded and **every** feature derivative from `eval->features()` is
   pushed into the QEF as a separate sample. The feature algebra lives in
   `eval/feature.cpp`: each feature carries a set of unit "epsilons"
   (`feature.hpp:18-24`) defining the cone of directions selecting it;
   `feature.cpp:105` dedupes epsilons at `e.dot(i) > 1 - 1e-8`; `feature.cpp:176`
   rejects a corner solve when `fabs(det) < 1e-6`; `eval_feature.cpp:346,354` collapse
   features whose derivative difference satisfies `derivDiff.dot(derivDiff) <= 1e-10`.
   `eval_feature.cpp:336` notes the residual known-imperfect case (*"followed by 0,
   1e-8, 0), but that should be rare enough to not be [handled]"*).
   Why: this is *the* mechanism keeping a CSG intersection edge sharp instead of
   rounded. Harness equivalent: **NONE** — greps for
   `epsilons|sharp-feature|feature epsilon` in `src/harnesscad/` return nothing, and
   `domain/numeric` has no feature-cone type. Stdlib-portable: yes, though it needs
   the f-rep IR to expose min/max ambiguity — which `domain/geometry/sdf/frep`
   plausibly can.

8. **Invalid normals are dropped, not zeroed — in two different places, differently.**
   `intersection.hpp:47-50`: `if (norm <= 1e-12 || !deriv.isFinite().all()) return;`
   — the sample is discarded entirely **after** the mass point has already been
   accumulated (the `mass_point += mp` at L36 happens first). `qef.hpp:81-83` does the
   opposite: a non-finite normal is replaced with **all-zeros** before insertion,
   contributing nothing to `AtA` but still contributing a row to `BptBp`.
   Why: the divergence is deliberate and worth copying exactly — position evidence
   survives even when gradient evidence is garbage. Harness equivalent: **NONE**
   (`QEF.insert` in `dual_contouring.py:106` has no finite/degenerate check).
   Stdlib-portable: yes.

9. **`VolTree` — interval pruning as a reusable acceleration structure, decoupled
   from meshing.** `vol_tree.cpp` plus `settings.hpp:70-71` (`const VolTree* vol`). An
   octree of EMPTY/FILLED/AMBIGUOUS built once by interval evaluation, queried by
   later mesher passes via `check(Region)` which returns `UNKNOWN` for any region not
   fully contained (L149-169) — **a conservative "I refuse to answer" rather than a
   guess**. `push()` returns `nullptr` on an ambiguous leaf (L191-193). `evalLeaf`
   (L79-83) forces AMBIGUOUS on `!isSafe()`.
   Harness equivalent: `interval_arithmetic.classify` gives the three-valued
   primitive; **the cached tree and the UNKNOWN-on-partial-containment contract are
   NONE.** Stdlib-portable: yes.

10. **Committed known-bad shapes and a watertightness predicate.**
    `test/util/mesh_checks.cpp` — `CHECK_EDGE_PAIRS`: every undirected edge must be
    traversed exactly once forward and once reverse (the bitfield must equal 3), which
    catches non-manifold edges and flipped triangles in one pass. Committed
    adversarial shapes, all procedural with no binary assets:
    - `test/util/shapes.cpp:149-168` `sphereGyroid()` — a gyroid `shell`ed, `max`'d
      with a sphere, then folded through `sqrt(abs(...))`; the standard stress case.
    - `test/simplex.cpp:220` *"box with problematic edges"* —
      `box({-1,-1,-1},{1.1,1.1,1.1})` on a `{-2,2}^3` region with `min_feature=1,
      max_err=-1` (collapsing disabled); asserts the field value ~0 at every vertex
      **and at every edge midpoint**, margin `1e-4`.
    - `test/simplex.cpp:260` *"tricky shape"* — an L-shaped `max(box, -box)` sharing a
      face, checked with `CHECK_EDGE_PAIRS`.
    - `test/mesh.cpp:137` *"triangles that are lines"* and L155 *"flipped triangles"* —
      `min(sphere(0.7,{0,0,0.1}), box(...,z=0.1))`, a sphere tangent to a box face;
      the flipped-triangle check isolates the coplanar top face and requires the
      normal to be `+Z` within 0.01.
    - `test/xtree.cpp:240` *"DCTree<3> cancellation"*, L274 *"checkConsistency"*.
    **DOCUMENTED FAILURE MODE:** `test/hybrid_meshing.cpp:255` is tagged
    **`"[!mayfail]"`** — `HybridMesher<3>: cylinder meshing`, `cylinder(1.3, 1.3)`
    restricted to the single cell `{0.375,-1.5,-0.375}..{0.75,-1.125,0}`. The kernel
    *knowingly* fails to co-locate the body vertex and the top surface vertex
    (`norm < 1e-6`) at the cylinder's bottom circular edge, producing cracks. A
    committed, named, still-open bug.
    Harness equivalent: `eval/corpus/fixtures/manifold_meshes` holds manifold's crash
    reproducers; **the libfive set is NOT present.** `repair_toolkit` and
    `eval/bench/geometry/mesh_topology` implement edge-pair/manifold-edge checks — that
    predicate is **already covered**; the *shapes* are not. Stdlib-portable: yes (all
    are ~10-line SDF expressions).

11. **Nothing hangs; cancellation is a first-class kernel concern.**
    `settings.hpp:73` `mutable std::atomic_bool cancel`, plus `FreeThreadHandler`
    (L62-65) for embedding in a host thread pool, and a `ProgressHandler`.
    `settings.hpp:55-57` documents `workers` *"must be > 0; otherwise the renderer
    will segfault"* — an unchecked precondition, i.e. a real if trivial known crash.
    Harness equivalent: **NONE** in the meshing modules. Low value for a pure-Python
    harness; noted for completeness.

ALREADY COVERED: interval arithmetic with `maybe_nan` and EMPTY/FILLED/AMBIGUOUS
classification (`domain/numeric/interval_arithmetic`, explicitly "after libfive");
interval-pruned adaptive DC in 2D and uniform DC in 3D with rank-deficient QEF
pseudo-inverse (`volumes/dual_contouring`, `dual_contouring_3d`, both citing
libfive); Jacobi eigensolvers; marching cubes / marching tetrahedra; edge-pair
manifoldness checking (`mesh/repair_toolkit`, `eval/bench/geometry/mesh_topology`);
sphere tracing; SDF combinators.

VERDICT: **mine-further.** Seven distinct load-bearing gaps (findings 1, 3, 4, 6, 7,
8, 10) sit directly on top of modules the harness already advertises as "after
libfive". The highest-value single item is **finding 3** (manifold tables plus the
three Ju et al. topology-safety checks) — a pure-table refusal predicate with no
numerical baggage — closely followed by **finding 1** (bounded QEF descent, which
replaces a clamp the harness's own comment apologizes for).

---

## ComplexGen-main (1053 files; ~817 is the source-only subset)

LICENSE: **MIT** (`LICENSE`, "Copyright (c) 2022 Haoxiang Guo") -> vendorable with
attribution. Vendored subtrees carry their own: yaml-cpp is MIT, GTE is the Boost
Software License.

WHAT IT IS: SIGGRAPH 2022. Reconstructs a CAD B-rep from a point cloud by predicting
a *chain complex* (corners -> curves -> patches, plus incidence matrices) with a
sparse-CNN + tri-path transformer, then recovering a **definite** complex from the
**probabilistic** one via a global ILP maximizing likelihood subject to structural
validity constraints, then geometrically refining with primitive/NURBS fitting.

### Verifying the "vendored GTE" excuse — measured, with counts

| Subtree | Files | Verdict |
|---|---:|---|
| `GeometricRefine/GeometricRefine/src/Mathematics/` (GTE) | **453** | vendored (Boost) |
| `GeometricRefine/yaml-cpp/` (295 of which are `test/`) | **400** | vendored |
| `GeometricRefine/allquadricsdirect/` | **51** | vendored |
| `GeometricRefine/GeometricRefine/src/HOptimizer/` (HLBFGS) | 18 | vendored |
| single-header libs in `src/` (`json.hpp`, `cxxopts.hpp`, `happly.h`, `nanoflann.hpp`, `algebra3.h`) | 5 | vendored |
| `chamferdist/` (CUDA ext) | 18 | vendored |
| **ComplexGen's own Python** (root + `src/` + `PostProcess/` + `vis/` + `scripts/`) | **76 .py, 24,846 LOC** | own |
| **ComplexGen's own C++** (`CurveFitter`/`SurfFitter`/`NURBSFitting`/`Helper`/`Mesh3D`/`Primal_Dual_graph`/`main`) | ~33 files, ~7,000 LOC | own |

**The prior skip was NOT justified — the stated reason is factually wrong.** GTE is
453/1053 = 43% of the file count, and vendored code overall is ~86% *by file count* —
but file count is the wrong denominator. ComplexGen ships **~32,000 lines of its own
code**, including a 2,430-line ILP post-process that is the single most transferable
artifact in the repo. "Vendored GTE (C++)" describes a directory the prior auditor
apparently listed and never entered.

**However** — and this matters more than the excuse — **the harness has in fact
already mined it.** `domain/reconstruction/brep/chain_complex`, `chain_complex_io`,
`chain_complex_nms`, and `eval/bench/geometry/complex_matching` all exist and all cite
ComplexGen (SIGGRAPH 2022) in their docstrings. So the recorded skip is wrong twice
over: wrong about the repo's composition, and contradicted by the harness's own
registry.

READ (in full): `LICENSE`; `README.md` (all 3 phases);
`docs/complex_extraction_complex_description.md`; `PostProcess/complex_extraction.py`
L1-60, L140-260, L315-400, L400-650, L855-1000, L1406-1480, L2331-2430 (the objective
vector, all inequality constraints, both equality-constraint families,
`check_feasibility`, `check_geom_topo_cons`, `process_one_file`); `src/guard.py` (14L);
`src/primitives.py` head; `GeometricRefine/.../Primal_Dual_graph.h` (119L);
`scripts/eval_default.sh`.

SKIMMED-NOT-READ: `Minkowski_backbone.py` (4,584L — read only its `def`/metric grep
index); `mink_resnet*.py`, `transformer3d.py`, `attention.py`, `data_loader_abc.py`,
`matcher_{corner,curve,patch}.py` (551L combined, Hungarian matching cost, grep-level
only); `SurfFitter.cpp` (1,923L — read only its constant sites), `NURBSFitting.cpp`
(1,113L), `main.cpp` (2,432L), `Helper.cpp` (999L); all of `vis/`; every vendored
subtree.

FINDINGS

1. **A complete, machine-checkable B-rep chain-complex validity constraint system.**
   `PostProcess/complex_extraction.py`. Binary variables per corner `V(i)`, curve
   `E(i)`, patch `F(i)`, plus incidence variables `EV`, `FV`, `FE` and auxiliary
   `Z(i,k,j)`. Constraints, verbatim from the code:
   - **Downward closure** — `EV(i,j) <= E(i)` and `EV(i,j) <= V(j)` (L436-473);
     `FE(i,j) <= F(i)` (L475-490); `FV(i,j) <= F(i)`, `FV(i,j) <= V(j)` (L510-548).
   - **Non-orphaning** — `V(i) <= sum_j EV(j,i)` (L550-577): no corner without an
     incident curve. `F(i) <= sum_j FE(i,j)` (L580-601, guarded `if nc > 0`): no patch
     without a boundary curve. `V(i) <= sum_j FV(j,i)` (L605-628).
   - **Equality: every curve bounds exactly two patches** — `2*E(i) = sum_j FE(j,i)`
     (L864-885). The watertightness / 2-manifold condition, as an equality.
   - **Equality: every open curve has exactly two endpoints** —
     `2*Y(i) = sum_j EV(i,j)` (L887-917), where `Y` is the open-curve indicator (the
     comment at L901 records the `O -> Y` substitution dated 0114).
   - **Equality: chain-complex consistency (boundary of boundary = 0)** —
     `2*FV(i,j) = sum_k Z(i,k,j)` with `Z` linearizing `FE(i,k)*EV(k,j)` (L950-975),
     i.e. `FV = FE*EV/2` must hold exactly. L190 prints the pre-solve violation as
     *"mean topo error"*.
   Why: a complete, solver-independent axiomatization of B-rep topological validity
   that can be evaluated as a **checker** (violated / satisfied) with zero LP
   machinery.
   Harness equivalent: `domain/reconstruction/brep/chain_complex` — **LARGELY
   COVERED.** Verified by reading it: it has `is_watertight` (L227-229, exactly the
   "every curve has 2 patches" rule), `patch_corner_product`/`patch_corner_incidence`
   (L145-177, the `FE*EV/2` product), `corner_degree`, `patch_loops`,
   `euler_characteristic`, `is_connected`, and a `check()`/`is_valid()` pair returning
   a `Diagnostic`. **Not obviously covered:** the *open-curve* two-endpoint equality
   conditioned on a closedness indicator (ComplexGen's `Y`, L901); and the harness's
   `min_corner_degree` default of 3 is a **stricter** rule than ComplexGen's
   `V <= sum EV` (degree >= 1). Stdlib-portable: yes as a checker; the ILP itself needs
   Mosek/Gurobi and is not portable.

2. **Geometry-topology consistency checker with a typed error taxonomy.**
   `PostProcess/complex_extraction.py:2331-2385` `check_geom_topo_cons(data)` returns
   a **small integer error code**, not a bool:
   - `1` = a curve is incident to a corner (similarity > 0.5) but **neither of its two
     endpoints** is within `th_dist` of that corner;
   - `2` = a patch is incident to a curve but the mean of per-point nearest distances
     from curve to patch exceeds `th_dist`;
   - `3` = a patch is incident to a corner but the corner's nearest distance to the
     patch cloud exceeds `th_dist`;
   - `0` = consistent.
   Note the asymmetry: curve<->corner uses **endpoint-exact** distance, patch<->curve
   uses **mean-of-min**, patch<->corner uses **min**. Closed curves
   (`curve_close[i] >= 0.5`) are skipped from the corner check entirely.
   Why: a numbered failure taxonomy where declared topology must be *witnessed by
   geometry* — precisely the shape of a refusal predicate.
   Harness equivalent: `chain_complex.check_geometry(cx, tol=0.1)` (L330) — **COVERED
   in substance** (it has `_point_to_cloud` and `_mean_cloud_to_cloud`, matching the
   two distinct metrics). Whether it returns codes or a Diagnostic, and whether it
   skips closed curves, was not verified line by line.

3. **Thresholds and weights, with the tuning history left in the file.** L29-30
   `geom_th = 0.3` (geometric constraint cutoff), `th_valid = 0.3` (primitive-validity
   cutoff, also the documented cutoff for `.complex` export in the README); L147
   `w_curvecorner_topo = 0.5`; L165-166 `weight_topo = 10.0` *"only used when
   normalized"*, with `# weight_topo = 1` left commented beside it; L169
   `flag_remove_extra_constraints = True` *"set to true if you want to remove extra
   constraints"* — meaning the shipped configuration **disables** three of the
   constraint families above (`EV <= E`, both `FV <= .`, and `V <= sum FV`) as
   redundant-or-too-slow. L346-353: the incidence coefficient is a **50/50 blend** of
   network similarity and geometric similarity. Commented-out `linprog` calls at
   L988-991 record abandoned tolerances (`dual_feasibility_tolerance: 1e-16`, then
   `1.25e-8` *"for model 06"*).
   Why: the 0.3/0.5 split (0.3 to admit a primitive, 0.5 to admit an incidence) and
   the "these constraints are redundant in practice" finding are both non-obvious.
   Harness equivalent: `chain_complex.threshold_incidence(similarity, threshold=0.5)`
   and `geometric_similarity(distance, sigma)` — **COVERED for the 0.5 incidence
   cutoff.** The 0.3 validity cutoff and the redundant-constraint finding are not
   encoded anywhere found.

4. **Evaluation metric suite: matched-chamfer plus structure P/R/F plus topology
   P/R/F.** `Minkowski_backbone.py:1336` `Curve_Corner_Matching_tripath`, L1412
   `Patch_Curve_Matching_tripath`, L1488 `Patch_Corner_Matching_tripath` (each taking
   a `flag_round` for eval-time rounding); precision/recall/fscore at L1685-1687,
   L1957-1971, L2459-2461, all with the same `+1e-6` denominator guard.
   `scripts/eval_default.sh` names the four eval axes:
   `--evalfinal --eval_res_cov --eval_param --eval_matched`. `compute_overall_singlecd`
   (L355, and a second same-named overload at L391) is the residual/coverage metric.
   Harness equivalent: `eval/bench/geometry/complex_matching` — **COVERED.** Verified
   by reading its API: `corner/curve/patch_cost_matrix`, `match(cost, threshold)`,
   `PRF`, `structure_prf`, `topology_prf`, `matched_chamfer`, `complex_chamfer`,
   `evaluate_complex`, plus `closed_curve_distance` for the cyclic-shift-invariant
   case. A faithful port.

5. **`Primal_Dual_graph` — non-manifold edge repair as a graph problem.**
   `GeometricRefine/GeometricRefine/src/Primal_Dual_graph.h:1-119`. Nodes = connected
   components of the mesh; edges = **non-manifold edges**, each recording
   `incident_components` (a `std::set<int>`) and `original_edge_id`.
   `build_connection_graph` calls `mark_component_with_coherence()` then
   `update_edge_manifold_flag()` and inserts an edge for every
   `is_nonmanifold_edge == 1`. `flag_node_valid` / `flag_edge_valid` allow nodes to be
   excluded (*"not considering the connectivity if set to false"*), and
   `Node::sub_nodes` supports **merged** nodes — the structure is designed for
   iterative component merging to resolve non-manifoldness.
   Why: a component-level dual graph is the right abstraction for "which shells do I
   merge to make this manifold", which is strictly above per-edge repair.
   Harness equivalent: **NONE found.** `mesh/repair_toolkit` and
   `eval/bench/geometry/mesh_topology` operate per-edge/per-segment; grepping the
   registry for `incidence` returned nothing, and no module describes a component-merge
   dual graph. Stdlib-portable: yes (pure combinatorics over an existing half-edge
   structure — `mesh/halfedge` already exists).

6. **Iterative primitive-fitting tolerances, uniform across all fitters.**
   `SurfFitter.cpp:925-926, 945-946, 1496-1497, 1511-1512`: every Levenberg-Marquardt
   and Gauss-Newton primitive fit (cone, sphere, cylinder, torus) uses
   `maxIterations = 32`, `updateLengthTolerance = 1e-04`,
   `errorDifferenceTolerance = 1e-08`; LM adds `lambdaFactor = 0.001`,
   `lambdaAdjust = 10.0`, `maxAdjustments = 8`. `SurfFitter.cpp:15`
   `#define TH_POS 1e-5`, used at L624 and L1058 as `if (abs(D(i,i)) > TH_POS)` — a
   **pseudo-inverse rank cutoff on the fitting normal matrix**, the same idea as
   libfive's eigenvalue cutoff at a different scale.
   Degeneracy handling: `SurfFitter.cpp:1571-1575`, torus UV inversion clamps `sinv`
   to `1.0 - 1e-6` / `-1.0 + 1e-6` before `asin` — a deliberate **strict** inequality
   so the derivative stays finite at the poles, not a plain clamp to +/-1.
   `src/guard.py:8-13` is the same instinct in torch: `guard_exp` clamps to +/-75
   before `exp`, `guard_sqrt` clamps to `min=1e-5` before `sqrt`.
   Harness equivalent: partial — `parametric/offset_nurbs` exists with an "honesty
   contract"; whether it carries these specific fit tolerances was not verified. The
   torus-pole `+/-(1-1e-6)` guard and `TH_POS = 1e-5` are worth transplanting
   regardless. Stdlib-portable: yes.

7. **`.complex` file format — a committed, fully specified text serialization of a
   B-rep chain complex.** `docs/complex_extraction_complex_description.md` gives the
   exact line layout: counts, corner positions, then per-curve
   `(type, closedness probability, 34 sampled points)`, per-patch
   `(type, 20x20 sampled points)`, then the curve->corner and patch->curve **adjacency
   matrices**, then per-patch u-direction closedness.
   Harness equivalent: `domain/reconstruction/brep/chain_complex_io` — **COVERED**
   (219L, *"Reader / writer for the ComplexGen `.complex` chain-complex file format"*).

GROUND-TRUTH CORPUS — the honest answer: **there is none committed.** Every `.ply`,
`.pkl`, `.complex` and `.json` in the tree was enumerated; the only JSON files are
`.vscode/settings.json`, a GTE `cmake-variants.json`, and a yaml-cpp gtest
`library.json`. The preprocessed ABC dataset, the pretrained weights, and every phase
output are BaiduYun/OneDrive/GoogleDrive downloads (README, "Data downloading"). The
`docs/` files specify the *formats* precisely enough to synthesize fixtures, but
**zero ground-truth instances ship with the repo.** Anyone claiming a "ground-truth
evaluation corpus with machine-checkable outputs" from this repo is describing a
download link.

ALREADY COVERED: the chain complex as a checkable structure, its `.complex` I/O,
probabilistic->definite NMS, and the full matching/PRF/chamfer evaluation suite —
`domain/reconstruction/brep/{chain_complex, chain_complex_io, chain_complex_nms}`
(926L combined) and `eval/bench/geometry/complex_matching` (267L). Together a
competent port of everything in findings 1-4 and 7.

VERDICT: **mine-further, narrowly.** The prior skip's *stated reason* is false and
should be corrected in the audit record — but the *practical* loss was small, because
the harness independently mined the repo's core. What remains genuinely unmined:
**finding 5** (the `Primal_Dual_graph` component-merge dual graph for non-manifold
repair — the single most valuable unmined item here), **finding 6** (fit tolerances
and the torus-pole guard), and two small deltas in findings 1/3 (the open-curve
two-endpoint equality; the `flag_remove_extra_constraints` redundancy finding).
Everything else is done.

---

## SolidPython-master (64 files; ~40 is the .py-only subset — 29 .py, 8,399 LOC)

LICENSE: **LGPL-2.1** (`pyproject.toml:6`; `README.rst:460-479`), with documentation
under CC-BY-SA 3.0. -> **facts-with-citation only.**

WHAT IT IS: a Python DSL that emits OpenSCAD source. Everything downstream of code
generation — booleans, meshing, rendering — is OpenSCAD's job. It contains essentially
no kernel.

READ (in full): `solid/extrude_along_path.py` (213L); `solid/utils.py:971-1016`
(`transform_to_point` plus the inlined pre-2015 PyEuclid `look_at`); `solid/splines.py`
(def-level, 522L); `README.rst` license section; `pyproject.toml`.

SKIMMED-NOT-READ: `solid/utils.py` (1,435L — read only its `def`/`EPSILON` index),
`solid/objects.py` (1,044L), `solid/solidpython.py` (827L — the OpenSCAD emitter),
`solid/screw_thread.py` (253L), `solid/py_scadparser/` (a PLY grammar for OpenSCAD),
`solid/test/` (1,455L across 5 files — these test *emitted SCAD strings*, not
geometry), all 12 `examples/`.

FINDINGS

1. **Degenerate-frame fallback in `transform_to_point`.** `solid/utils.py:993-999`:
   when `dest_normal` is parallel to `src_up`, `dest_normal.cross(src_up) == ORIGIN`
   and *"the transform collapses all points to dest_point"*. The fix is a two-level
   fallback — try `EUC_UP`, and if `src_up` is itself parallel to `EUC_UP`, use
   `EUC_FORWARD`. A real sweep-along-path failure mode (the section vanishes wherever
   the path is vertical).
   Harness equivalent: `domain/geometry/features/sweep` — **ALREADY COVERED, and
   structurally identical.** Verified by reading `sweep.py:97-108`: same comment
   (*"normal is parallel to `up` the frame would collapse, so a fallback up-vector is
   [used]"*), same two-level test via `_norm(_cross(...)) < EPSILON`, same fallback
   pair. The harness uses `EPSILON = 1e-9` (`sweep.py:50`) versus SolidPython's
   `EPSILON = 0.01` (`utils.py:38`) — **the harness value is the better one**;
   SolidPython's 0.01 is a *modelling* epsilon (used to oversize cut solids at
   `utils.py:456`, L772) reused here for a numerical test, which is a latent bug in
   SolidPython, not a finding for us.

2. **Closed-path detection and tangent wrap.** `extrude_along_path.py:71-85`: if
   `(path[0] - path[-1]).magnitude_squared() < EPSILON`, drop the duplicate point and
   force `connect_ends`; tangents are central differences over a padded array —
   wrapped (`[path[-1]] + path + [path[0]]`) when closed, and **reflected** when open
   (`first = path[0] - (path[1] - path[0])`), so the end tangents are extrapolated
   rather than one-sided.
   Harness equivalent: `features/sweep` — **COVERED.** Its module docstring
   (`sweep.py:8-9`) states *"central difference of the neighbouring points, wrapped for
   closed paths"*. Reflected-vs-wrapped for the open case was not verified line by
   line; it is a one-line check if anyone cares.

3. **Everything else is OpenSCAD glue or already mined.** `splines.py` (Catmull-Rom
   plus cubic Bezier, `catmull_rom_prism_smooth_edges`) ->
   `domain/geometry/parametric/catmull_rom`, whose docstring cites *"SolidPython
   `splines.py`"*. `screw_thread.py` -> `features/screw_thread`. The 2D offsets ->
   `parametric/path_offset` (as stated in the brief). What is left is
   `objects.py`/`solidpython.py` (SCAD emitters, excluded), `py_scadparser` (a PLY
   lexer/parser for OpenSCAD source — the harness has its own evaluator at
   `sdf/csg_eval`), `utils.py`'s `bill_of_materials`/`bom_part` decorator machinery
   (project bookkeeping, not geometry), and directional helpers (`up`/`down`/
   `rot_z_to_x`). **No epsilon carries a stated reason beyond finding 1. No fixture
   corpus** — `test/` asserts on generated OpenSCAD *text*.

ALREADY COVERED: 2D path offsets (`path_offset`), Catmull-Rom splines and lofted
prisms (`catmull_rom`), screw threads (`features/screw_thread`), sweep-along-path with
frame transport and the parallel-tangent degeneracy guard (`features/sweep`), Bezier
evaluation (`parametric/bezier`).

VERDICT: **nothing-here.** The only two pieces of real scar tissue in this repo
(findings 1 and 2) are already in `domain/geometry/features/sweep`, ported with the
same structure and a strictly better epsilon. Nothing further to extract, and the LGPL
licence means we could only cite it anyway.

---

# Cross-cutting summary — geometry kernels

## Verdicts at a glance

| Repo | Files | License | Verdict |
|---|---:|---|---|
| manifold-master | 394 | Apache-2.0 | **mine-further** (highest value) |
| libfive-master | 321 | MPL-2.0 (Studio GPL-2) | **mine-further** |
| sdfx-master | 373 | MIT | **mine-further** |
| ruststep-master | 842 | Apache-2.0 (+ ISO schema grant) | **mine-further** |
| ezpz-main | 93 | MIT | **mine-further** |
| replicad-main | 393 | MIT | **mine-further** |
| solvespace-python | 648 | **GPL-3** | mine-further (facts only) |
| instant-mesh-intersection-repair | 39 | **NONE** | mine-further (facts only) |
| ComplexGen-main | 1053 | MIT | mine-further, narrowly |
| angelcad-master | 393 | **GPL-2/3** | mine-further, narrowly (facts only) |
| arcs-master | 51 | MIT/Apache-2.0 | already-covered |
| sdf-csg-master | 40 | Unlicense | already-covered / nothing-here |
| CADmium-main | 123 | MIT (+ LGPL subtree) | **nothing-here** (plus a correction) |
| SolidPython-master | 64 | LGPL-2.1 | **nothing-here** |

## Prior-audit claims tested

- **solvespace is GPL-3** — **CONFIRMED** from `COPYING.txt`. Facts-with-citation only.
- **The "CADmium EPSILON = TOLERANCE = 1e-6" claim** — **CONFIRMED MISATTRIBUTION.**
  The constant lives in vendored **pythonocc-utils (LGPL-3+)** at
  `OCCUtils/Construct.py:96`, not in CADmium's own code. CADmium's actual constants are
  `PRECISION = 1e-5` and `eps = 1e-7` (`utility/macro.py:32-33`). Wrong file, wrong
  project, wrong licence.
- **"manifold's boolean3 is spec reference only"** — **STALE AND WRONG.** The harness
  made manifold3d the sole engine for metric booleans in `cca2e3e`, so its semantics
  are now load-bearing; and this checkout is newer than the one previously seen (a full
  `boolean2` rewrite, Clipper2 removed, a new `ExecutionContext`).
- **"ComplexGen skipped: vendored GTE (C++)"** — **MEASURABLY FALSE.** GTE is 453 of
  1053 files; ComplexGen ships ~32,000 lines of its own code including a 2,430-line ILP
  validity system. The claim is also self-contradicting, since
  `domain/reconstruction/brep/chain_complex*` and `eval/bench/geometry/complex_matching`
  are faithful ports of exactly that code.
- **"replicad is bindings glue"** — **FALSE.** `packages/replicad/src/lib2d/` and
  `src/blueprints/` are ~20 files of genuine hand-written 2D geometry with a
  documented four-tier tolerance ladder and several cited OCCT workarounds.

## Live defects found in shipped harness code

1. **`io/backends/manifold.py` docstring is factually wrong.** It describes Manifold
   as *"exact-predicate mesh booleans over a collider"*. `src/boolean3.cpp` uses **no
   exact predicates** — it uses float `Shadows()` tests disambiguated by op-dependent
   **symbolic perturbation**, with intersection knowingly asymmetric between operands
   (`boolean3.cpp:479-481`). Since the harness has just made manifold3d the sole engine
   for metric booleans, and the differential oracle's value rests on correctly
   characterising each engine's failure surface, that docstring overstates the
   guarantee being relied on.
2. **`mesh/intersection_repair.py` can return a result strictly worse than one it
   already held.** `repair_self_intersections` (L203-256) returns the **final**
   vertices unconditionally, while the upstream implementation tracks a best iterate
   (`repair_factory.py:116-118, 156-166`). Because the harness also records `history`,
   the bug is directly observable: `[12, 1, 7]` returns the 7-collision mesh. ~4-line
   fix.
3. **No time budget around any manifold3d call.** Manifold's own fuzzers wrap every
   boolean in a 10-second watchdog (`test/manifold_fuzz.cpp:71-87`), and
   `ExecutionContext` cancellation explicitly **cannot** interrupt a single large
   boolean. The OCCT->manifold3d swap reduced hang probability; it did not eliminate
   hangs, and the docstring at `backends/manifold.py:99` claiming *"it never hangs"*
   refers only to the import stub.

## Highest-value unmined items, ranked

1. **libfive manifold tables + the three Ju et al. topology-safety checks**
   (`dc_tree3.cpp:20-27`, `manifold_tables.cpp:19-54`) — a pure-table refusal
   predicate gating cell collapse, no numerical baggage, entirely absent.
2. **Manifold's epsilon/tolerance propagation rules** (`impl.cpp:660-691`,
   `csg_tree.cpp:243-249`) — bbox-relative, not absolute, with the actionable
   consequence: model at the origin or lose fidelity linearly with distance.
3. **Manifold's 15-code `Error` taxonomy** — reachable from Python via `.status()`,
   which the harness never calls.
4. **ruststep's 711-file ISO EXPRESS schema corpus** — explicitly redistributable, and
   the harness's EXPRESS parser has never seen a single real ISO schema.
5. **libfive bounded QEF descent** (`qef.hpp:325-377`) — replaces a clamp the harness's
   own comment apologizes for.
6. **sdfx's SHA1SUM + stldiff regression apparatus** — a working IDENTICAL/MINOR/
   MATERIAL drift classifier with a stated determinism invariant and a stated
   architecture caveat.
7. **ezpz's 5 proptest seeds** — pre-minimised hostile inputs for `core/lm_solver`,
   which was ported from that exact codebase *without* its regression corpus.
8. **solvespace's orthogonal rank x convergence failure taxonomy** — the harness has
   both axes and never crosses them.
9. **ruststep's Annex-B `SUBTYPE_CONSTRAINT` instantiables algorithm** — the harness's
   parser explicitly `_skip_to`s these blocks.
10. **ComplexGen's `Primal_Dual_graph`** — component-level dual graph for non-manifold
    repair, strictly above the harness's per-edge repair.

## Confirmed absences (stated plainly, because they save future work)

- **ruststep commits no malformed STEP.** 5 STEP/p21 files, all valid; no negative-parse
  tests exist at all.
- **angelcad commits no test corpus and no kernel.** One `.as` file (a doc stub); xcsg
  and polyhealer are external and absent from the repo.
- **ComplexGen commits no ground-truth instances.** All data is behind download links.
- **sdf-csg has no test suite whatsoever.** Zero test files.
- **arcs has no fixture corpus, no fuzz corpus, no error taxonomy, no epsilon table.**
- **instant-mesh-intersection-repair's benchmark data is explicitly withheld** by the
  authors.
- **SolidPython's tests assert on emitted OpenSCAD text**, not geometry.
