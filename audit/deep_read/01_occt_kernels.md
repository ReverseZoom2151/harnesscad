# Deep read — OCCT / kernel scar tissue

Repo set A. Sources under `resources/cad_repos/` (gitignored, double-nested `X-main/X-main/`).

This is a genuine read pass, not a grep sweep. File counts below are **actual**
`find -type f` counts and disagree with the counts in the prior inventories —
those appear to have counted only some subset. Where a prior audit claim is
wrong, it is called out inline with evidence.

Method note: "read fully" means the whole file. "header-read" means the top of
the file / its declarations only. Nothing here is described as "sampled".

---

## oce-oce-patches (35,907 files)

### LICENSE

`LICENSE_LGPL_21.txt` + `OCCT_LGPL_EXCEPTION.txt` — **LGPL-2.1 with the OCCT
linking exception**, dual-licensable commercially. Behavioural facts (status
enum members, message-catalogue causes, resource-file defaults, bug ids) are
recordable with citation. Source text is not; nothing is vendored.

### WHAT IT IS — and a correction to the repo's own name

**This is not OCE.** The directory name says `oce-oce-patches`, and the harness
module mined from it is called `agents/generation/occt_quirks_oce.py`, but
`adm/cmake/version.cmake:19-21` reads `OCC_VERSION_MAJOR 8 / MINOR 0 /
MAINTENANCE 0`, and the tree uses the modern
`src/<Module>/TK<Toolkit>/<Package>/` layout introduced well after OCE was
abandoned. This is **upstream Open CASCADE Technology 8.0.0**. OCE (the
community edition) forked at OCCT 6.x and died in 2017.

This matters twice over. First, provenance: `occt_quirks_oce.py` cites a
kernel it is not actually quoting. Second, currency: any reasoning that treated
this trove as "stale 6.x community fork" was wrong — it is the *newest* kernel
in the corpus, and newer than OCP's target (7.9.3, see below) and than the
OCCT that pythonocc's tests were written against (7.7.0).

### READ

- `resources/SHMessage/SHAPE.us` — **read fully** (5,780 bytes, the entire shape-healing message catalogue)
- `resources/XSMessage/XSTEP.us` — **read fully in three passes** (30,408 bytes, 219 keys)
- `resources/XSTEPResource/STEP` — read fully; `resources/XSTEPResource/IGES` — read fully and diffed against STEP
- `resources/BOPAlgo/BOPAlgo.msg` — read fully
- `resources/UnitsAPI/CurrentUnits`, `MDTVBaseUnits` — read fully
- `src/ModelingAlgorithms/TKShHealing/ShapeExtend/ShapeExtend_Status.hxx` — read fully
- `src/ModelingAlgorithms/TKBO/BOPAlgo/BOPAlgo_Alerts.hxx` — read fully
- `src/ModelingAlgorithms/TKTopAlgo/BRepCheck/BRepCheck_Status.hxx` — read fully
- The 16 `TKShHealing` headers that document `DONE1..8`/`FAIL1..8` semantics — **status doc-blocks read in full for all 16** (137 documented status lines)
- `adm/cmake/version.cmake`, `tests/parse.rules`, `tests/bugs/parse.rules`, `tests/heal/parse.rules` — read fully
- `src/harnesscad/agents/generation/occt_quirks_oce.py` (the harness side) — read for coverage, not edited

### SKIMMED-NOT-READ

Structure-read only: `tests/` (19,090 files; directory tree, group names and
`grids.list` enumerated, individual `.tcl` cases not read beyond two samples),
`data/` (77 files, filenames listed), `src/` beyond TKShHealing/TKBO/TKTopAlgo
(the other ~34,000 C++ files, of which `dox/` and `samples/` were not opened at
all). Realistically: **~30 files read fully, ~25 header-read, of 35,907.** The
prior 30-quirk pass covered `ShapeFix_*`/`ShapeAnalysis_*` `.cxx` reviewer
comments; this pass deliberately went elsewhere.

### FINDINGS

**1. The complete healing-outcome vocabulary — machine-readable, per-fixer.
NONE-verified.**

Two halves, both committed:

*(a) The status bit semantics.* `ShapeExtend_Status.hxx:60-79` defines 8 DONE
and 8 FAIL flags plus `OK`/`DONE`/`FAIL` aggregates, and states the contract:
statuses are **a bitset, not a scalar** — "status can have several flags set
simultaneously", queried via `Standard_Boolean Status(ShapeExtend_Status)`.
Each fixer then documents what *its* DONE*i*/FAIL*i* mean. 137 documented lines
across 16 headers. The load-bearing ones:

| Header | Status | Meaning |
|---|---|---|
| `ShapeFix/ShapeFix_Face.hxx:157-165, 234-243` | DONE1..5, DONE8, FAIL1..4 | wires fixed / orientation fixed / **missing seam added** / **small area wire removed** / **natural bounds added** / face may be split; and the four matching "cannot" failures |
| `ShapeFix/ShapeFix_Edge.hxx:112-115` | FAIL1/2, DONE1/2 | no 3d curve / projection failed; pcurve added; pcurve through degenerate point |
| `ShapeFix/ShapeFix_Edge.hxx:144-145` | DONE1/2 | **tolerance of first / last vertex has been increased** |
| `ShapeFix/ShapeFix_Edge.hxx:189-196` | FAIL1/2, DONE1/2/3/5 | SameParameter: deviation computation failed / `BRepLib::SameParameter` failed; **edge tolerance increased**; flag forced True; edge modified by BRepLib; BRepLib's edge chosen. DONE4 explicitly "not used anymore" |
| `ShapeFix/ShapeFix_ComposeShell.hxx:128-133` | DONE1/2, FAIL1..4 | split produced ≥1 / several faces; misoriented wire (handled); **recoverable** parity error; edge with no pcurve; **FAIL4 = unrecoverable algorithm error (parity check)** |
| `ShapeFix/ShapeFix_Wireframe.hxx:82-91` | DONE1/2, FAIL1/2 | gaps fixed in 3D / in 2D; failed to fix either |
| `ShapeFix/ShapeFix_Shape.hxx:83-88` | DONE1..6 | which *level* was fixed (free edges / wires / faces / shells / solids / compounds) |
| `ShapeBuild/ShapeBuild_ReShape.hxx:104-108` | DONE1..4, FAIL1 | source shape **replaced** vs **removed**; subshapes replaced vs removed |
| `ShapeUpgrade/ShapeUpgrade_SplitSurface.hxx:92-94` | DONE1/2/3 | split gave >1 patch / gave only the initial patch / **geometric form or parametrisation modified** |
| `ShapeUpgrade/ShapeUpgrade_WireDivide.hxx:99-101` | DONE1, FAIL1/2 | edges split; edges **skipped** for want of a 3d curve / a pcurve |
| `ShapeAnalysis/ShapeAnalysis_Wire.hxx:273-277` | DONE1/2/3, FAIL1/2 | confused at `gp::Resolution` / at `preci` / at `prec` but not `preci`; not confused; not confused unless edge *n* reversed |

*(b) The human-readable catalogue.* `resources/SHMessage/SHAPE.us` maps
fixer+method keys to the sentence OCCT prints. Read as a taxonomy of **silent
mutations the healer may perform on a user's geometry**:

- destructive: `Small edge removed`, `Small wire removed`, `Small face removed`,
  `Spot face removed`, `Strip face removed`, `Small solid removed`,
  `Small solid merged with other`, `Null area wire detected, wire skipped`,
  `Incomplete edge (with no pcurves or 3d curve) removed`
- additive: `Missing seam-edge added`, `Lacking edge(s) inserted`,
  `Face created with natural bounds`
- topology-changing: `Improperly connected shell split into parts`,
  `Improperly connected solid split into several parts`,
  `Impossible to orient faces in shell, several shells created`,
  `Wire was split on several wires`, `Wires with common vertex fixed`
- geometry-changing: `Face converted to BSpline`, `BSpline Face re-approximated`,
  `Face converted to surface of revolution`, `Swept Face converted to elementary`,
  `Direction of Face of revolution corrected`, `Not same parameter edge fixed`
- refusal: `Solid cannot be created from an open shell`, `Cannot orient wire`
- process: `Error: Shape Processing: Operator %s failed with exception %s`,
  `Warning: Shape Processing: Sequence not defined for %s, nothing to do`

**Why we want it.** The harness *already heals* — `eval/reliability/brep_repair.py`
runs an 11-rung escalating ShapeFix ladder (`default_ladder()`, lines 179-190)
with an explicit per-face flag recipe (`_apply_face_recipe`, lines 219-303). But
it reads the outcome back as a **single boolean**: `_shape_is_valid` (line 196)
is `bool(BRepCheck_Analyzer(wrapped).IsValid())`, and `RepairResult.actions`
(line 97) holds the harness's own rung names, not OCCT's report. So the harness
can currently say *"rung 7 made it valid"* but cannot say *"we made it valid by
deleting a face and inflating two vertex tolerances."* For a project whose bar
is "what can be measured or refused honestly", healing that silently removes
user geometry and reports only `healed: true` is precisely the failure mode this
vocabulary closes. The DONE/FAIL split also separates *repaired* from
*unrepairable* (`ShapeFix_ComposeShell` FAIL4 "unrecoverable") — a refusal
predicate the ladder cannot presently express and instead burns a rung on.

**HOW verified NONE.** `grep -rniE 'ShapeExtend|StatusFace|StatusWire|\.Status\(|DONE1|FAIL1'`
over `src/harnesscad/` returns zero relevant hits (only unrelated
`UIEvent.status` / `Gui.SendMsgToActiveView`). `grep -rn 'ShapeExtend_DONE|ShapeExtend_FAIL' src/`
returns nothing. Registry search for `heal` returns three modules
(`agents.agent.build123d_lints`, `domain.geometry.topology.sew`,
`eval.reliability.brep_repair`); none carries a status vocabulary. `shapefix`,
`spot face`, `natural bound` return nothing.

**2. `BRepCheck_Status` — a 37-member invalidity taxonomy the harness collapses
to one bit. NONE-verified.**

`BRepCheck/BRepCheck_Status.hxx` (whole enum, 37 members):
`NoError, InvalidPointOnCurve, InvalidPointOnCurveOnSurface,
InvalidPointOnSurface, No3DCurve, Multiple3DCurve, Invalid3DCurve,
NoCurveOnSurface, InvalidCurveOnSurface, InvalidCurveOnClosedSurface,
InvalidSameRangeFlag, InvalidSameParameterFlag, InvalidDegeneratedFlag,
FreeEdge, InvalidMultiConnexity, InvalidRange, EmptyWire, RedundantEdge,
SelfIntersectingWire, NoSurface, InvalidWire, RedundantWire, IntersectingWires,
InvalidImbricationOfWires, EmptyShell, RedundantFace,
InvalidImbricationOfShells, UnorientableShape, NotClosed, NotConnected,
SubshapeNotInShape, BadOrientation, BadOrientationOfSubshape,
InvalidPolygonOnTriangulation, InvalidToleranceValue, EnclosedRegion, CheckFail`.

**Why we want it.** These are per-subshape diagnoses, retrievable from
`BRepCheck_Analyzer`'s result tree, and they are *actionable in different
directions*: `NotClosed` on a shell means "sew or refuse to call this a solid";
`SelfIntersectingWire` means "the sketch is bad, regenerate"; `FreeEdge` means
"a face is missing"; `InvalidToleranceValue` means "the healer's own tolerance
ladder overshot"; `CheckFail` means "we do not know" and is the only honest
refusal. The harness has exactly two call sites — `brep_repair.py:200` and
`io/backends/cadquery.py:1405` — and both take `.IsValid()` and throw the
diagnosis away. A verifier-first harness that reports "invalid" without saying
*how* is losing the whole signal at the last inch.

**HOW verified NONE.** `grep -rn 'BRepCheck' src/harnesscad/ --include=*.py`
returns 7 hits, all `BRepCheck_Analyzer(...).IsValid()` or prose mentions.
`grep -rn 'BRepCheck_Status' src/` returns zero.

**3. The BOPAlgo alert taxonomy — 37 boolean-operation failure causes, 26 of
which carry the offending subshape. NONE-verified.**

`BOPAlgo/BOPAlgo_Alerts.hxx` — **11 `DEFINE_SIMPLE_ALERT` + 26
`DEFINE_ALERT_WITH_SHAPE`**, each with a one-line documented cause. Text
duplicated for the message layer in `resources/BOPAlgo/BOPAlgo.msg`. Retrieved
via `BOPAlgo_Options::HasErrors()` / `HasWarnings()`
(`BOPAlgo_Options.hxx:73, 82`), which filter the report by `Message_Fail` /
`Message_Warning` — so severity is already separated for us.

The simple alerts (`:21-51`) are configuration/precondition failures:
`AlertUserBreak, AlertBOPNotAllowed, AlertBOPNotSet, AlertBuilderFailed,
AlertIntersectionFailed, AlertMultipleArguments, AlertNoFiller,
AlertNullInputShapes, AlertPostTreatFF` ("cannot connect face intersection
curves"), `AlertSolidBuilderFailed, AlertTooFewArguments`.

The with-shape alerts (`:54-145`) are the valuable half — geometry-localised:
`AlertBadPositioning` ("positioning leads to creation of small edges without
valid range"), `AlertEmptyShape`, `AlertNotSplittableEdge`,
`AlertRemovalOfIBForEdgesFailed / ForFacesFailed / ForSolidsFailed /
ForMDimShapes` ("not supported yet"), `AlertSelfInterferingShape`,
`AlertShellSplitterFailed`, `AlertTooSmallEdge`,
`AlertIntersectionOfPairOfShapesFailed`, `AlertBuildingPCurveFailed`,
`AlertAcquiredSelfIntersection` ("sub-shapes became connected through other
shapes and the argument became self-interfered"), `AlertUnsupportedType`,
`AlertNoFacesToRemove`, `AlertUnableToRemoveTheFeature`,
`AlertRemoveFeaturesFailed`, `AlertSolidBuilderUnusedFaces` ("some faces … not
classified and not used for solids creation"), `AlertFaceBuilderUnusedEdges`,
`AlertUnableToOrientTheShape`, `AlertUnknownShape`, `AlertNoPeriodicityRequired`,
`AlertUnableToTrim`, `AlertUnableToMakeIdentical`, `AlertUnableToRepeat`,
`AlertMultiDimensionalArguments`, `AlertUnableToMakePeriodic`,
`AlertUnableToGlue`, `AlertShapeIsNotPeriodic`,
`AlertUnableToMakeClosedEdgeOnFace`.

**Why we want it.** `DEFINE_ALERT_WITH_SHAPE` means the alert **carries the
`TopoDS_Shape` that caused it**. That converts "your cut failed" into "your cut
failed on *this* edge, which is too small to split" — a localised, renderable,
repairable refusal. Three of these are near-verbatim the failure modes the
harness's existing quirk `SetFuzzyValue-on-every-boolean`
(`agents/generation/occt_quirks.py`) is a blind workaround *for*:
`AlertTooSmallEdge`, `AlertNotSplittableEdge`, `AlertBadPositioning`. Reading
the alert lets the harness stop guessing at a fuzzy value and instead say which
input feature is under-scaled. `AlertSolidBuilderUnusedFaces` and
`AlertFaceBuilderUnusedEdges` are worse than errors — they are **warnings that
accompany a returned result**, i.e. a boolean that "succeeded" while dropping
geometry.

**HOW verified NONE.** `grep -rn 'BOPAlgo' src/` returns one hit:
`eval/verifiers/metric_booleans.py:98`, where `"BOPAlgo"` is a string in
`OCCT_BOOLEAN_MARKERS` — an *import-detection* marker used to spot modules an
OCCT boolean can reach. It is not the taxonomy. Nothing reads
`HasErrors()`/`GetReport()`.

**4. `resources/XSTEPResource/STEP` vs `IGES` — 50 healing knobs, and two real
defects in the shipped IGES defaults. NONE-verified.**

`XSTEPResource/STEP` and `XSTEPResource/IGES` are the **shipped default
post-import healing configuration** for the two translators: ~50
`From<FMT>.FixShape.*Mode` knobs plus tolerance bindings
(`Tolerance3d : &Runtime.Tolerance`, `MaxTolerance3d : &Runtime.MaxTolerance`,
`MinTolerance3d : 1.e-7`) and the operator sequences
(`FromSTEP.exec.op : FixShape`; `ToSTEP.exec.op : SplitCommonVertex,DirectFaces`;
`ToIGES.exec.op : DirectFaces`). Convention: `-1` = leave to the fixer's own
default, `0`/`1` = force off/on.

Diffing the two files line-for-line surfaces two things a grep would never find:

- **`CreateOpenSolidMode` differs by format**: `STEP:23` sets `0`, `IGES:23`
  sets `1`. Importing the *same* nominal geometry via IGES will therefore
  produce a "solid" built from an **open** shell, where STEP will not. That is a
  format corner case with a direct verifier consequence — a volume or mass
  property computed on an IGES-imported open solid is meaningless, and the
  harness would have no signal that it happened.
- **Two knobs are dead in the IGES path.** `IGES:22` and `IGES:26` read
  `FromSTEP.FixShape.FixShellOrientationMode` and
  `FromSTEP.FixShape.FixFaceOrientationMode` — a copy-paste of the STEP prefix
  into the IGES resource file. They can never match the `FromIGES.` lookup, so
  **shell- and face-orientation fixing are silently unconfigured on IGES import**
  and fall through to the fixer default. Verified: `grep -n 'FromSTEP' IGES`
  returns exactly those two lines and nothing else.

STEP additionally carries three knobs IGES lacks entirely (`FixTailMode : 0`,
`MaxTailAngle : 0`, `MaxTailWidth : -1`).

**Why we want it.** These are the defaults that actually run against every
imported file in an OCCT-backed stack, including the harness's own
`io/ingest/import_brep.py` path — so the harness inherits `CreateOpenSolidMode=1`
on IGES today without recording that it does. And the knob list is a domain
table: it is the authoritative enumeration of what ShapeFix can be asked to do,
which is the configuration side of finding 1's outcome side.

**HOW verified NONE.** Registry search for `iges` returns three modules, none
about healing configuration (`io.ingest.import_brep` mentions IGES only in its
summary as a loadable format). `grep -rn 'CreateOpenSolid\|FixShellOrientation\|MinTolerance3d' src/`
returns zero.

**5. The IGES silent-default substitution list — three of them change scale.
NONE-verified.**

`resources/XSMessage/XSTEP.us` is, despite the name, **entirely the IGES
translator catalogue** (219 keys, ~30KB, organised into a LOADING PHASE and an
ANALYSIS PHASE, then per-entity-type parameter checks tagged `!Type NNN`).

The prior run flagged this as "the STEP/IGES translator message catalog". Read
in full, the great majority of it is low-density per-parameter type assertions
("parameter %d … Real was expected") across 31 IGES entity types — **not** worth
transcribing. Two distillates are worth having:

*(a) Seventeen "Default value X taken" sites* — places where a malformed file is
silently repaired rather than rejected. Three of them change the **size of the
resulting part**:

| Line | Field | Silent default |
|---|---|---|
| `XSTEP.us:142` | Global §, param 13, **Model Space Scale** | `1.0` |
| `XSTEP.us:145` | Global §, param 14, **Unit Flag** | `2` = **millimetres** |
| `XSTEP.us:148` | Global §, param 15, Unit Name unrecognised | `2` = **millimetres** |
| `XSTEP.us:151` (key `_51`) | params 14 vs 15 disagree | **Unit Name ignored**, flag wins |
| `XSTEP.us:643` | Type 408 subfigure, param 5, **Scale Factor** | `1.0` |
| `XSTEP.us:295` | Type 126/128, Degree of Basis Functions | `0` |
| `XSTEP.us:130-139` | precision magnitude/significance ×4 | `38 / 6 / 308 / 15` |
| `XSTEP.us:157` | Global §, param 19 (granularity) | `0.0` |
| `XSTEP.us:160,163` | Version Flag / Drafting Standard | `3` / `3` |
| `XSTEP.us:181-193` | Directory Entry fields 4,5,6,7,8 | `0` ×5 |

*(b) The entity-type inventory the catalogue validates* — `100` arc, `102`
composite curve, `104` conic, `106` copious data, `108` plane, `110` line, `112`
parametric spline curve, `114` parametric spline surface, `116` point, `118`
ruled surface, `120` surface of revolution, `122` tabulated cylinder, `124`
transformation matrix, `126` rational B-spline curve, `128` rational B-spline
surface, `130` offset curve, `140` offset surface, `141` boundary, `142` curve
on parametric surface, `143` bounded surface, `144` trimmed surface, `186`
manifold solid B-rep object, `190` plane surface, `308` subfigure definition,
`402` associativity instance, `408` subfigure instance, `502` vertex list, `504`
edge list, `508` loop, `510` face, `514` closed shell.

**Why we want it.** The scale trio is the IGES twin of a trap the harness has
already decided is worth a module on the DXF side — commit `fbde92e
feat(formats): DXF $INSUNITS code table — the 2D half of the silent-scale trap`.
IGES is the other half and is unhandled: a file with a corrupt Unit Flag imports
**as millimetres with no error**, and if the author meant inches the part is
25.4× wrong while every downstream verifier reports success. This is exactly a
"refuse honestly" case — the correct behaviour is to detect the substitution and
decline to certify dimensions, not to measure a silently-rescaled solid.

**HOW verified NONE.** Registry search for `insunits` returns the DXF module
only; `iges` returns nothing about units. `grep -rn 'Model Space Scale\|Unit Flag'
src/` returns zero.

**6. `tests/` — 19,090 test scripts, but the fixtures are NOT here.
Deliberately reported as a near-miss so it is not re-chased.**

`tests/` holds OCCT's regression suite across 41 groups (`boolean`, `bugs`,
`heal`, `blend`, `chamfer`, `offset`, `sewing`, `thrusection`, `pipe`, `mesh`,
`de`, `lowalgos`, `perf`, …), 19,090 files, of which `tests/bugs` alone is 4,399
across `modalg_1..8`, `moddata_1..3`, `heal`, `step`, `iges`, `xde`,
`splitshape` — i.e. thousands of *named, reproduced kernel bugs* organised by
subsystem. `tests/heal` alone is 1,279 files across 31 scenario directories
(`drop_small_edges`, `drop_small_solids`, `fix_gaps`, `same_parameter`,
`split_closed_faces`, `unify_same_domain`, `wire_tails_real`, …).

**But**: the scripts reference their inputs through `locate_data_file`, and
there are **15,323 such references against 77 committed data files**. OCCT ships
its test data in a separate repository which is not in this checkout. So this is
19,090 *procedures* without *inputs* — valuable as a taxonomy of what OCCT
itself considers a distinct failure mode, near-worthless as a fixture corpus.
The 77 files that are here (`data/occ/*.brep` ×44, `data/stl` ×10, `data/step`
×2 — `linkrods.step`, `screw.step` — `data/iges` ×2, `data/vrml` ×1) are the
Draw-harness demo shapes, not canaries.

One small thing *is* extractable and is a genuine finding:
`tests/parse.rules` (10 lines) is OCCT's own **verdict-assignment rule set** for
its harness — the regex→outcome table that decides whether a run passed:

```
SKIPPED  /Tcl Exception: .*[fF]ile .* could not be found/   data file is missing
IGNORE   /Tcl Exception: [*][*] Exception [*][*]/           duplicate report
IGNORE   /Relative error of mass computation :/             diagnostic of *props*
FAILED   /\b[Ee]xception\b/ ; /\b[Ee][Rr][Rr][Oo][Rr]\b/ ; /\b[Ff][Aa][Ii][Ll][0-9]\b/ ;
         /\b[Ff][Aa][Ii][Ll][Ee][Dd]\b/ ; /\b[Ff][Aa][Ii][Ll][Uu][Rr][Ee]\b/
FAILED   /Process killed by CPU limit/                      Killed by CPU limit
FAILED   /Process killed by elapsed limit/                  Killed by elapsed time limit
```

Two things are notable for a harness with the same job. First, the
four-outcome vocabulary is `SKIPPED / IGNORE / FAILED / (pass)` — a missing
input is *not* a failure, and a known-noisy diagnostic is explicitly
whitelisted. Second, **`FAIL[0-9]` is a failure pattern**: OCCT's own harness
greps for the `ShapeExtend_FAILn` flags from finding 1 leaking into output.
Third and most useful: **a timeout is classified as FAILED, with CPU-limit and
wall-clock-limit as separate causes.** A kernel that hangs is a kernel that
failed, and OCCT distinguishes "spinning" from "waiting". `DrawResources/CheckCommands.tcl:458,463`
shows the paired before/after `checkshape` protocol these tests use.

### ALREADY COVERED

- The 30 reviewer-tag quirks in `agents/generation/occt_quirks_oce.py` (verified
  by reading it: `fix-intersecting-edges-no-cut`,
  `nonadjacent-intersection-vertex-tol`, `pcurve-self-intersection-loops`,
  `pcurve-shift-between-singularities`, `shared-pcurve-inplace-transform-aliasing`,
  `split-edge-in-singularity`, `self-intersection-appears-after-bend`,
  `degenerated-edge-no-pcurve-removal`, `closed-edge-must-not-become-degenerated`,
  `degenerated-flag-still-needs-pcurve-check`, `missing-natural-bound-sphere-torus`,
  `makeface-isdegenerated-error`, `small-face-precision-autocorrect`,
  `missing-seam-on-periodic-faces`, `composeshell-edge-tol-for-coincidence`,
  `intersection-tool-always-modifies`, `rational-bspline-closure-by-sampling`,
  `projection-point-densing-near-end`, `adjust-periodic-needs-pconfusion`,
  `projection-ignore-singular-points`, `degenerate-projection-minimal-pcurve`,
  `step-transfer-bind-early-against-cycles`, `step-shape-aspect-sdr-duplicates`,
  `step-edge-explicit-range-after-makeedge`, `iges-spline-transform-scope`,
  `iges-offset-surface-bspline-conversion`, `surface-walker-local-resolution`,
  `loft-conic-sections-param-range`, `seam-requires-two-pcurves-same-surface`,
  `fillet-corner-orientation-composition`). All are drawn from `.cxx` reviewer
  tags; none overlaps findings 1-6, which come from headers, resource files and
  message catalogues.
- The ShapeFix flag recipe and escalating tolerance ladder —
  `eval/reliability/brep_repair.py` (`default_ladder`, `_apply_face_recipe`,
  `_sew_shape`). Note this is credited to Brepler, not to OCCT's own
  `XSTEPResource` defaults; finding 4 is the vendor's version of the same
  configuration and disagrees with it in places worth reconciling.
- Topology traversal / shape identity — `domain/geometry/topology/explorer.py`.

### VERDICT

**mine-further.** Six findings, all outside the 30 already mined, and findings
1-3 are the same shape of gap: the harness *calls* OCCT's diagnostic machinery
and then discards everything it returns except one bit. That is the single
highest-leverage correction available in this repo for a verifier-first project.
Finding 6 is a deliberate negative on the largest-looking target in the tree.

---

## OCP-master (7,431 files)

### LICENSE

`LICENSE` is the full Apache-2.0 text — **Apache-2.0, verified**. (The mission
brief did not state a prior verdict for OCP; recording it now.)

### WHAT IT IS

CadQuery's OCCT Python bindings. Critically, **the generated bindings are not in
this checkout** — they are produced at build time by `pywrap all ocp.toml`. What
*is* here is 7,412 vendored upstream OCCT C++ headers (`opencascade/*.hxx`,
`*.lxx`) plus **19 hand-written files**, of which one (`ocp.toml`, 1,533 lines)
carries essentially all the content.

### READ

Read fully: `ocp.toml` (1,533 lines), `OCP_specific.inc` (73), `README.md`,
`templates/CMakeLists.j2`, `conda/meta.yaml`. Header-read: `pystreambuf.h`.
Exhaustive `find`: **no `patches/` directory and no `.patch`/`.diff` file
anywhere**, and no `Modules.yaml`.

### SKIMMED-NOT-READ

The 7,412 vendored OCCT headers were not read (they are the *input* to the
generator, and are the same upstream material covered above at a newer version).
`extern/msvc-wine/`, 6 CI YAMLs, `dump_symbols.py` — build machinery.
**Read 5 files fully, 1 header-read, of 7,431 — of which 7,412 are vendored
upstream headers and 19 are OCP's own.**

### FINDINGS

**1. The un-bindable OCCT surface — an exclusion table. NONE-verified.**

`ocp.toml:348-364` (file-level `exclude`) plus ~90 per-module
`exclude_classes` / `exclude_methods` / `exclude_typedefs` /
`exclude_constructors` blocks at `ocp.toml:429-1533`. Where a reason is given it
is concrete:

- `ocp.toml:681-684` — the whole `ChFi3d_*` free-function family
  (`ChFi3d_ApproxByC2`, `ChFi3d_SearchPivot`, `ChFi3d_EnlargeFace`,
  `ChFi3d_InPeriod`, `ChFi3d_Boite`, `ChFi3d_SetPointTolerance`) excluded:
  `error: invalid use of incomplete type 'class Geom_BSplineCurve'`.
  **The fillet/chamfer helper entry points are simply absent from OCP.**
- `ocp.toml:698` + `:354-361` — `BRepMesh_Triangle`, `BRepMesh_Delaun`,
  `BRepMesh_DefaultRangeSplitter`, `BRepMesh_Context`, every
  `BRepMesh_*MeshAlgo`, every `*RangeSplitter` excluded
  (`#constructor with int&[3]`). **The entire pluggable meshing internals of
  OCCT are unreachable from OCP** — you get `BRepMesh_IncrementalMesh` and
  nothing below it.
- `ocp.toml:263` — the whole `ShapePersistent` module commented out:
  `#Need to implement inner typedefs parsing to wrap this properly`.
- `ocp.toml:104, 129, 197, 208` — `ApproxInt`, `IntWalk`, `ChFiKPart`,
  `TColQuantity` modules commented out, no reason given.
- `ocp.toml:810` — `Select3D_IndexedMapOfEntity`, `Graphic3d_Vec3d`,
  `Select3D_BndBox3d`: `#couldn't deduce template parameter 'Hasher'`.
  `:814` — `Select3D_SensitiveCircle` ctors 0,1: `#missing vftable`.
- `ocp.toml:390` — **Windows only**: `exclude_classes = ["Handle_*"]`. A
  platform-conditional binding difference.
- `ocp.toml:915, 921-925` — `TDF_Label::FindAttribute` excluded and *replaced*
  by a hand-written lambda, because the native signature takes an out-`Handle`
  pybind cannot fill. OCAF attribute lookup has OCP-specific semantics.

**Why we want it.** A static oracle for "will this OCCT symbol exist at runtime
in a CadQuery/OCP stack?" — a class of failure that currently only surfaces as
an `AttributeError` deep inside a generated part. The `BRepMesh_*` exclusion is
directly load-bearing for a harness that wants to control tessellation, and the
`ChFi3d_*` exclusion for one that wants to reason about fillet failure.

**HOW verified NONE.** `grep -riE 'exclusion|unbindable|not_bound|pywrap'` over
`src/harnesscad/` returns only unrelated hits; `Standard_OutOfRange`,
`Standard_DomainError`, `ClassNotWrapped` all return zero.

**2. OCP targets OCCT 7.9.3.** `ocp.toml:427` — `__version__ = "7.9.3.1"`.

Cheap but load-bearing: the corpus now holds **three different OCCT
generations** — this repo's own sources are 8.0.0 (above), OCP binds 7.9.3, and
pythonocc's tests record a bug against 7.7.0. Any quirk recorded without a
version is under-specified. NONE-verified: no OCP↔OCCT version map in the harness.

**3. Exception translation is a single root. Partial.**

`OCP_specific.inc:13-28` (`register_occ_exception`) + `ocp.toml:367`
(`exceptions = ["Standard_Failure"]`). Only `Standard_Failure` is registered,
carrying `e.GetMessageString()` across. Consequence: OCCT errors reach Python
through OCP as **one exception family with a message string**, not as the
differentiated `IndexError`/`ValueError`/`RuntimeError` mapping pythonocc gives
(see pythonocc P2). **A harness that parses kernel errors must branch on which
binding it is running.** Harness has `eval/reliability/error_contract.py:118,274`
normalising `StdFail_NotDone`, but no OCP-vs-pythonocc taxonomy split.

**4. Hand-injected shape identity primitives — and a `__bool__` trap. Partial.**

`ocp.toml:544-566` — OCP injects `TopoDS_Shape.__bool__` (= `!IsNull()`),
`__hash__` (= `std::hash<TopoDS_Shape>`), `_address`, `_from_address`.
`ocp.toml:582-586` — `BRepBuilderAPI_Command.__bool__` (= `IsDone()`).

`bool(shape)` means **not-null**, *not* **non-empty**. `if not shape:` on an
empty-but-valid compound is silently False. The harness's
`domain/geometry/topology/explorer.py:1-38` documents the
`(TShape, orientation, location)` identity triple and the hash-bucket + `IsSame`
scheme, but not this trap.

### ALREADY COVERED

Shape identity / hash-bucket + `IsSame` — `domain/geometry/topology/explorer.py`.

### NOT FINDINGS (stated as volume)

`OCP_specific.inc:31-70` (holder types, `nodelete` deleter), `pystreambuf.h`
(vendored cctbx stream adapter, BSD), `templates/`, `dump_symbols.py`,
`extern/msvc-wine/`, CI YAMLs, `conda/meta.yaml`. **No GC/ownership documentation
exists in OCP** beyond the holder declarations, and **no documented
segfault/crash workarounds** — that material lives entirely in pythonocc's
tests. The 7,412 vendored headers are duplicate upstream and are covered by the
OCCT 8.0.0 read above.

### VERDICT

**mine-further, narrowly.** One substantial finding (the exclusion table), three
small ones. 99.7% of the repo is vendored or generated and correctly yields
nothing.

---

## pythonocc-core-master (1,075 files)

### LICENSE

`LICENSE` is LGPLv3 — **LGPL-3.0, verified.** Manifest-only. Nothing vendored,
no source text reproduced; behavioural facts cited by path and line only.

### WHAT IT IS

The other OCCT Python binding (SWIG-generated rather than pybind11). 948 of the
1,075 files are `src/SWIG_files/` machine output. The value is concentrated in
two hand-written trees: `src/Extend/` (4 modules, ~85KB of practical recipes)
and `test/` (~20 Python files that function as a regression log of
binding-level crashes).

### READ

Read fully: all four `src/Extend/` modules (`DataExchange.py`, `ShapeFactory.py`,
`TopologyUtils.py`, `__init__.py`), and the ~20 Python files in `test/`.
Surveyed not deep-read: `src/Display/` (30 files) and
`src/Addons`/`Tesselator`/`MeshDataSource` (16) — viewer and tessellator glue,
out of scope for a headless verifier. **Read ~24 files fully, ~46 header-read,
of 1,075 — of which 948 are machine-generated SWIG wrappers/stubs.**

Excluded per brief: the STEP/BREP fixtures already in
`eval/corpus/fixtures/step_canaries.py`.

### FINDINGS

**P1. The `Interface_Static` table is smaller than hoped — but its *semantics*
are the real prize. Partial.**

The hoped-for large domain table **does not exist here**, and that is worth
recording so it is not chased again. Only three parameters are ever set in the
whole repo:

| Parameter | Values | Where |
|---|---|---|
| `write.step.schema` | `AP203`, `AP214IS`, `AP242DIS` (validated set, `DataExchange.py:172-175`); also `AP214` at `test_core_wrapper_features.py:1069` | `DataExchange.py:182` |
| `write.step.product.name` | arbitrary string | `test_core_wrapper_features.py:1070` |
| `write.stepcaf.subshapes.name` | `1` | `test_core_wrapper_features.py:1068` |

Verified **absent from the entire repo**: `read.step.product.mode`,
`write.surfacecurve.mode`, `write.iges.brep.mode`,
`read.iges.bspline.continuity`, `write.step.unit`, `xstep.cascade.unit`,
`read.precision.val`/`.mode`. STEP *read* (`DataExchange.py:88-149`) and both
IGES functions (`:472-551`) set **no statics at all**.

Two behavioural facts around it are high-value:

- **Silent-typo failure.** `SetCVal`/`SetIVal` return *false* for a non-existent
  parameter name rather than raising; `CVal` returns `""` and `IVal` returns `0`
  for unknown names (`src/SWIG_files/wrapper/Interface.i:11386, 11454, 11624,
  11643`). A misspelled knob is undetectable without checking the return value.
- **Ordering trap.** `test_core_wrapper_features.py:352` records that
  `STEPControl_Writer()` **must be instantiated before** `Interface_Static.SetCVal`
  has any effect — the static registry is populated by the writer's construction.
  Calling `SetCVal` first is a silent no-op.

**Why we want it.** The harness already calls
`SetCVal_s("write.step.schema", …)` at `io/backends/cadquery.py:1574`, and
`io/backends/freecad.py:146` documents FreeCAD's failure to set it. Neither
checks the return value, and neither records the ordering constraint — so the
harness's own schema-setting call is **currently unverified** and could be a
silent no-op depending on construction order. NONE-verified for both facts.

**P2. Degenerate-input exception taxonomy with concrete causes. NONE-verified.**

`test/test_core_exception.py` (49 lines, read fully):

| Cause | OCCT exception | Python type |
|---|---|---|
| `gp_Dir.Coord(-1)` — index outside 1..3 (`:35`) | `Standard_OutOfRange` | `IndexError` |
| `BRepBuilderAPI_Sewing().FreeEdge(-1)` (`:38`) | `Standard_OutOfRange` | `IndexError` |
| **Degenerate edge** — `MakeEdge(p, p)`, identical endpoints, `.Edge()` on a not-done builder (`:43`) | `StdFail_NotDone` | `RuntimeError` |
| **Zero-extent box** — `MakeBox(0,0,0).Shape()` (`:48`) | `Standard_DomainError` | `ValueError` |
| Invalid downcast `Geom_BSplineCurve.DownCast(<Geom_Line>)` (`test_core_wrapper_features.py:662`) | — | `SystemError`, **only on Python < 3.12** (explicit guard at `:661-663`) |
| Deliberately unwrapped class/method (`:913, 938`) | — | `ClassNotWrappedError` / `MethodNotWrappedError` |

**Why we want it.** Zero-size box and coincident-point edge are exactly what an
LLM emits, and they raise **different Python types** — so a single
`except RuntimeError` catches half of them and lets the other half escape.
`StdFail_NotDone` alone is known to the harness
(`eval/reliability/error_contract.py:118`); `Standard_OutOfRange`,
`Standard_DomainError` and the Python-3.12 downcast divergence are NONE-verified.

**P3. Object-lifetime crash traps — the strongest material in the repo.
NONE-verified.**

Each of these is a Python-level pattern a generated CAD script emits innocently,
and each historically segfaulted. All are statically lintable.

- `test_core_wrapper_features.py:613-631` (`test_memory_handle_getobject`, refs
  pythonocc-core **#292**, generator **PR #24**): a handle from an API function
  governed the object's lifetime alone; retaining only a reference to the object
  destroyed it prematurely and **crashed the process**. Two asserted forms "used
  to crash on pythonocc-0.16.5": `GC_MakeSegment(a,b).Value()` stored then
  queried, and fully-anonymous `GC_MakeSegment(a,b).Value().EndPoint()`.
- `test_core_wrapper_features.py:1142-1169` (issue **#1277**, titled "const byref
  should be wrapped as copy constructors **to prevent memory issues**"): the trap
  shape is `adaptor.Surface().BSpline().Bounds()` — a chain of const-ref
  temporaries.
- `test_core_wrapper_features.py:1172-1182` (issue **#1218**):
  `BRepAdaptor_Curve(local_edge).Curve().Curve()` returned from a function —
  **both the edge and the adaptor are GC'd on return** and the resulting
  `Geom_Curve` must survive. The canonical "geometry outliving its builder" trap.
- `test_core_extend_topology.py:160-177` (`test_edges_out_of_scope`,
  `test_wires_out_of_scope`): sub-shapes extracted from a
  `TopologyExplorer`/`WireExplorer` that is then dropped must still be non-null.
- `test_core_wrapper_features.py:783-797` (issue **#600**, PR **#614**): a null
  `TopoDS_Shape` from the TopoDS transformer must surface as Python **`None`**,
  not a null shape — so `is None` and `IsNull()` are *both* live null idioms in
  the same API.
- `test_core_wrapper_features.py:577-598`: subclassing `TopoDS_Edge` requires
  manually copying `TShape()`, `Location()`, `Orientation()`; before those three
  calls `self.IsNull()` is true.

**HOW verified NONE.** No hit anywhere in `src/harnesscad/` for handle lifetime,
`GetObject`, or premature destruction.

**P4. Version-specific behaviour. NONE-verified.**

- `test_core_geometry.py:487-491` — the 3-arg `GeomFill_Pipe(spline, TC1, TC2)`
  + `Perform` path is **disabled with the comment "bug with occt-770"**. The
  4-arg form with an explicit `GeomFill_*` trihedron mode (`:504`) is used
  instead. A dated OCCT-7.7.0 bug with a stated workaround.
- `test_core_geometry.py:695-711` (issue **#1057**) — the
  `BRepAdaptor_Curve(...).Curve().Curve()` chain is commented out with the note
  that it **"only works on linux"**. A live, unresolved platform divergence.
- `test_core_wrapper_features.py:1047-1052` — since pythonocc-core **7.7.1**,
  free-function statics (`gp_OX()`) emit `DeprecationWarning`; `gp.OX()` is
  supported.
- `test_core_wrapper_features.py:809-810, 832-833` — "since OpenCASCADE 7.x"
  some objects gained `DumpJson`/`InitFromJson`.
- `test_core_wrapper_features.py:240-260` (issue **#257**) — `gp_Vec` division
  broken on py3k with SWIG < 3.0.8.
- `test_display_sideeffects.py:9-12, 47-51` — module skipped on Linux; explicit
  guard against initialising more than one display (a viewer-singleton hazard).

**P5. Concrete numeric recipes from `src/Extend/`. Partial.**

- STL write (`DataExchange.py:379-385, 413-415`): `linear_deflection=0.9`,
  `angular_deflection=0.5`, via
  `BRepMesh_IncrementalMesh(shape, lin, isRelative=False, ang, isInParallel=True)`.
- **PLY / OBJ / glTF writers** (`:708, :746, :834`): `BRepMesh_IncrementalMesh(shape, True)`
  — the second positional arg **is the deflection**, and Python `True` == `1.0`.
  So all three mesh exports silently tessellate at **deflection 1.0**. Preceded
  by `breptools.Clean(shape)`.
- Wire vs edge discretisation defaults **disagree**: `TopologyUtils.py:778`
  `deflection=0.5` for wires, `:816` `deflection=0.2` for edges — and a wire
  passes its own 0.5 down to each edge (`:809`), overriding the edge default.
  Algorithms: `UniformAbscissa` / `QuasiUniformDeflection` (default) /
  `UniformDeflection` (`:72-76`).
- Bounding boxes (`ShapeFactory.py:246-267, 296-302, 353-370`): AABB uses
  `tol=1e-6`, `SetGap(tol)`, `AddOptimal(..., use_triangulation=True,
  use_shapetolerance=True)`; OBB uses `is_shape_tolerance_used=**False**` —
  **the two box routines disagree on whether shape tolerance counts**, which
  changes the reported extents.
- OBJ export unit fudge (`DataExchange.py:759-766`):
  `GetCasCadeLengthUnit() * 0.001` set as *both* input and output length unit to
  mimic m→mm; input CS `posYfwd_posZup`, output `negZfwd_posYup`.
- SVG export negates X (`DataExchange.py:583`) to flip handedness.

Harness has its own declared deflections (`io/backends/build123d.py:1413`,
`cadquery.py:1526`, `freecad.py:130-140`, `freecad_driver.py:59-63,767` — the
last already records that `exportStl` hard-codes 0.01). The
**`IncrementalMesh(shape, True)` == deflection-1.0 trap** and the **AABB/OBB
shape-tolerance disagreement** are NONE-verified.

**P6. Read-status handling is a one-branch collapse — an anti-pattern worth
recording. Harness is already better here, but not complete.**

`DataExchange.py` checks **only** `!= IFSelect_RetDone` (`:114, :188, :229,
:500, :548, :813, :849`). `RetVoid`, `RetError`, `RetFail`, `RetStop` are never
distinguished. Three concrete holes: `read_step_file_with_names_colors`
(`:229-231`) **silently returns an empty dict** on read failure rather than
raising; IGES `TransferRoots()`'s return is **ignored** (`:508`, asymmetric with
STEP at `:121`); PLY and OBJ `Perform()` returns are ignored entirely (`:727,
:770`) — **those two exports have no failure detection at all**. Also
`ShapeFactory.py:82-85`'s `assert_shape_not_null` tests Python `None`, **not**
`IsNull()`, so an OCCT null shape passes the guard.

The harness is ahead here — `io/ingest/import_brep.py:214`, `metadata.py:118`
and `_step_check_worker.py:64` all check `IFSelect_RetDone`, and STEP checking is
already subprocess-isolated (commit `9fc246b`). But the harness likewise never
branches on the non-Done values (`grep RetVoid|RetFail|RetStop` → zero hits), so
the four-way discrimination is NONE-verified.

### ALREADY COVERED

**P7. Topology traversal — already ported, nothing to add.**
`TopologyUtils.py:171-184, 224-265, 376-413` (hash-bucket + `IsSame` dedup, hash
is a *bucket key only*, `IsSame` ignores orientation, `ignore_orientation=True`
default collapsing a cube's 24 oriented edges to 12, eager materialisation of
`TopExp_Explorer` results, `BRepTools_WireExplorer` single-pass with explicit
`_reinitialize()`, ancestor queries via
`TopTools_IndexedDataMapOfShapeListOfShape`) is reproduced in
`domain/geometry/topology/explorer.py`, whose docstring (`:1-38`) already states
the 24→12 fact and the orientation-blind `IsSame` predicate. Two upstream defects
worth *not* porting: `TopologyUtils.py:720-726` `number_of_solids_from_shell`
passes `TopAbs_FACE, TopAbs_SOLID` where `:718` uses `TopAbs_SHELL`
(copy-paste bug), and `_map_shapes_and_ancestors:384` yields a sentinel `None`.

Also already ahead: the harness's own escalating ShapeFix ladder (commit
`b611abe`).

### EXPLICIT "NOTHING HERE" RESULTS

- **No shape healing anywhere in `src/Extend/`.** Zero `ShapeFix_*`,
  `ShapeUpgrade_*`, `ShapeAnalysis_*` imports across all four modules. The only
  repair-adjacent code is optional `BRepBuilderAPI_Sewing` on STL read
  (`DataExchange.py:455-458`) **with no tolerance argument**. The hoped-for
  "shape healing invocation recipes" do not exist in this repo.
- **No `test_core_bugs*.py` in this checkout** — verified by full directory
  listing; its role is absorbed into `test_core_wrapper_features.py`.
- **Zero OCCT Mantis bug IDs** and zero mailing-list references anywhere in
  `test/`. Every referenced ID is a *pythonocc-core GitHub issue*, not a kernel
  bug id — so this repo cannot be cross-referenced against
  `occt_quirks_oce.py`'s OCC-bug citations.
- **No FIXME / HACK / crash / segfault / hang / workaround comment anywhere in
  `src/Extend/`**, and no OCCT version number there. No hang or deadlock
  documented anywhere in `test/`.
- `test_core_extend_shapefactory.py:56` is a vacuous assertion
  (`box_volume == approx(box_volume, …)` — a value compared to itself), so the
  box-volume check tests nothing. Flagged so it is not mistaken for a fixture.

### VERDICT

**mine-further.** P3 (six lifetime crash patterns) and P2 (degenerate-input
exception types) are the strongest items; P1's two semantics facts retroactively
undermine an existing harness call site. The 948 SWIG files correctly yield
nothing.

---

## cadquery-master (218) / cadquery-plugins-main (83) / cadquery-contrib-master (59) / CQ-editor-master (79)

### LICENSE — one correction

- `cadquery-master/LICENSE` — **Apache-2.0** ("This library is free software…
  under the terms of the Apache Public License, v 2.0"). Confirmed.
- `cadquery-plugins-main/LICENSE` — **Apache-2.0**, full text. Confirmed.
- `cadquery-contrib-master/LICENSE` — **MIT**, "Copyright (c) 2018 Dave Cowden".
  **The mission brief and the prior inventory both list this as Apache-2.0. That
  is wrong** — the file is the standard MIT grant, read in full.
- `CQ-editor-master/LICENSE` — **Apache-2.0**, full text. Confirmed.

### WHAT IT IS

CadQuery is the fluent Python API over OCP/OCCT the harness's primary backend
already drives. `cadquery-plugins` and `cadquery-contrib` are community add-on
collections; CQ-editor is a Qt IDE.

### READ

This section is the thinnest of the three in this report and is labelled as such.
Read fully or in large targeted sections: `cadquery/selectors.py` (the grammar
region, lines 600-820), the tolerance and mesh-parameter regions of
`cadquery/occ_impl/shapes.py` (~6,300 lines; the constant and default sites read,
not the whole file), the exporter directory listing, `tests/` listing and
`tests/testdata/` in full, and the plugin directory listing. **Read ~4 files in
substantial part, directory- and header-read ~20, of 218 + 83 + 59 + 79.** The
bulk of `shapes.py`, `sketch.py`, `cq.py` and the eight exporter modules were not
read line by line.

### FINDINGS

**1. The string-selector grammar. ALREADY COVERED — and this was the highest
priority target, so recording the negative matters.**

`cadquery/selectors.py:600-820` is a real pyparsing grammar, not ad-hoc string
handling: `:609-618` numeric literals (`point`, `plusmin`, `integer`, `floatn`,
brackets, comma); `:635` `type_op = Literal("%")`; `:644-646` the index sublanguage
`ix_number` with suppressed square brackets; `:659-660` the two directional
alternatives (`direction_op + direction + Optional(index)` and
`center_nth_op + direction + Optional(index)`); and `:777-815` an
`infixNotation` table giving the full operator precedence and associativity —
`and` (binary, LEFT), `or` (binary, LEFT), the delta/exception op (binary, LEFT),
`not` (unary, RIGHT).

**HOW verified ALREADY COVERED.** Registry search for `selector` returns three
directly relevant modules:
`domain.geometry.topology.selector_dsl` ("Deterministic parser + evaluator for
the CadQuery string-selector DSL"), `domain.geometry.topology.selector_grammar`
("**Grammar-faithful** compiler from CadQuery selector strings to the object
algebra"), and `domain.geometry.topology.selector_algebra` ("Programmatic
CadQuery selector algebra: composable `Selector` objects"). Three layers —
parser, grammar-faithful compiler, and algebra — already mined. Nothing to add.

**2. Tolerance constants and the mesh-parameter defaults. Mostly already covered.**

`cadquery/occ_impl/shapes.py:341` — the module-level **`TOLERANCE = 1e-6`**, the
single global geometric epsilon. The per-operation defaults are notably
*heterogeneous*, which is the actual finding:

| Default | Site | What it guards |
|---|---|---|
| `tolerance = 1e-3`, `angularTolerance = 0.1` | `:490-491` (with the docstring "Default is 1e-3, which is a good starting point for a range of cases") and again `:1580, :1589, :1836` | tessellation / meshing — the same pair replicad uses (report 02) |
| `1e-6` | `:2077, :2203, :2549, :2570, :2734, :5075, :5926, :5947, :5996, :6127, :6178, :6239` | the dominant "resolution" / `tol` default across curve and surface work |
| `1e-9` | `:2958, :3341` | wire assembly from edges and point-list construction — **three orders tighter than the general default** |
| `1e-4` | `:1470` | |
| `1e-2` | `:3713` | |
| `1e-3` | `:1642, :2809, :3831` (`toArcs`) | approximation/conversion |
| `BuildCurves3d_s(w, 1e-6, MaxSegment=2000)` | `:3136`, commented **"NB: preliminary values"** | an author-flagged uncertain constant |

**Why we want it.** The 1e-9 vs 1e-6 vs 1e-3 spread is a tolerance *ladder*
expressed as scattered defaults rather than a policy, and `:3136`'s "preliminary
values" comment is an explicit admission that one of them is unvalidated.

**HOW verified mostly-covered.** `io/backends/cadquery.py:1528-1530` already
records the export signature and — importantly — the trap that
**`angularTolerance` is in RADIANS**, and the backend carries its own
`ANGULAR_DEFLECTION` (`:1567`), as does `io/backends/build123d.py:1429`. The
**global `TOLERANCE = 1e-6`** and the **1e-9 wire-assembly outlier** are the two
values not reflected in the harness; both NONE-verified, and both are minor.

**3. Committed fixtures — thin, and mostly 2D. Reported honestly as a near-miss.**

`tests/testdata/` holds **nine files**: `1001.dxf`, `gear.dxf`, `genshi.dxf`,
`MC 12x31.dxf`, `rational_spline.dxf`, `spline.dxf`, `three_layers.dxf`,
`OpenSans-Regular.ttf`, and one solid — `red_cube_blue_cylinder.step`.

That is a **DXF-and-fonts corpus, not a B-rep canary corpus**. Seven of the nine
are DXF, exercising the 2D importer's spline/layer/rational-curve handling; the
single STEP file is named for a colour-and-assembly test, not a malformed-geometry
test. There is **no known-bad STEP, no malformed STL, no non-manifold solid**.
Anyone expecting a fixture trove here should not look.

The 21 `tests/test_*.py` files were not read, so any exception-asserting tests
they contain are **"I did not find it", not "the harness lacks it"** — an honest
gap in this pass and the most likely place further value sits.

**4. cadquery-plugins — ten plugins, judged.**

`plugins/`: `apply_to_each_face`, `cq_cache`, `fragment`, `freecad_import`,
`gear_generator`, `heatserts`, `localselectors`, `more_selectors`, `sampleplugin`,
`teardrop`. Directory-read only.

Three are selector extensions (`localselectors`, `more_selectors`, and arguably
`apply_to_each_face`) and therefore fall under finding 1's already-covered
selector work. `gear_generator` and `teardrop` are parametric-geometry
generators; `heatserts` is a fastener-pocket table; `cq_cache` is memoisation;
`freecad_import` and `sampleplugin` are glue. **`gear_generator` and `heatserts`
are the two worth a real read** — a gear generator is a domain algorithm and a
heatsert table is a domain table — but I did not read either, so this is a
**lead, not a finding.** Note the harness already has gear formulas mined from
modeling-app (report 02), so `gear_generator` may well be redundant.

**5. cadquery-contrib and CQ-editor — nothing here.**

`cadquery-contrib` (59 files, MIT) is a collection of example scripts and
notebooks. Directory- and header-read; no kernel constants, no error taxonomy, no
fixtures. **nothing-here.**

`CQ-editor` (79 files, Apache-2.0) is a PyQt IDE — editor widget, object tree,
debugger, traceback pane. Directory- and header-read. Per the brief's exclusion
of UI glue: **nothing-here**, and this is a confident negative rather than an
unread one, because a Qt IDE's file inventory is diagnostic on its own.

### ALREADY COVERED

The selector DSL, in three harness modules (finding 1). The export
tolerance/angular-tolerance signature and the radians trap
(`io/backends/cadquery.py:1528-1530`). The three quirks already in
`agents/generation/occt_quirks.py` — SetFuzzyValue-on-every-boolean,
0-degree-revolve-is-360, the 1e99 infinite-face sentinel — none of which recurred
in what was read here.

### VERDICT

cadquery-master — **mine-further, but at low priority and with a specific
target**: the 21 unread `tests/test_*.py` files, for exception-asserting tests
and fillet/shell failure handling. Everything I *did* read was either already
covered (the selector grammar, the export defaults) or minor (two stray tolerance
values).
cadquery-plugins-main — **unresolved lead**: read `gear_generator` and
`heatserts`; the rest is selectors (covered) or glue.
cadquery-contrib-master — **nothing-here** (and its licence is MIT, not Apache).
CQ-editor-master — **nothing-here**.

**Honest caveat on this section**: it is materially shallower than the OCE and
pythonocc sections above. The highest-priority target (the selector grammar)
turned out to be already covered, which lowered the value of pressing further,
but the unread test suite is a real remaining gap and is flagged rather than
papered over.

---

## Cross-repo notes

**Three OCCT generations in one corpus.** `oce-oce-patches` is OCCT **8.0.0**;
OCP binds **7.9.3**; pythonocc's tests record a bug against **7.7.0**. Quirks
recorded without a version are under-specified, and `occt_quirks_oce.py`'s name
attributes its 30 rows to a kernel (OCE) that is not in this tree at all.

**The recurring shape of the gap.** Findings OCE-1, OCE-2, OCE-3 and pythonocc-P6
are the same defect seen four times: OCCT reports a structured diagnosis
(`ShapeExtend` status bitset, `BRepCheck_Status`, `BOPAlgo` alert-with-shape,
`IFSelect_ReturnStatus`) and every consumer in reach — including the harness —
reduces it to a boolean. For a project whose stated bar is "what can be measured
or refused honestly", recovering those four channels is worth more than any
individual quirk row.

**Two bindings, two error taxonomies.** OCP collapses everything to
`Standard_Failure` + message string (OCP-3); pythonocc maps to
`IndexError`/`ValueError`/`RuntimeError` per OCCT exception type (pythonocc-P2).
Any harness error contract must branch on the binding, and
`eval/reliability/error_contract.py` currently does not.
