# Deep read — the tail: UI apps, bindings, converters, duplicates, reconstruction

Repo set C, the low-yield tail: the 87 repos left after 38 committed reports.
Sources under `resources/cad_repos/` (gitignored, double-nested `X-main/X-main/`).

Genuine read, not a grep sweep. File counts are actual `find -type f`. Volume is
stated honestly per repo, including which clusters were fanned out to parallel
read-only sub-agents and re-verified here. Every gap claim was checked against
`registry.index()` (1579 modules) and targeted greps of `src/harnesscad/`.

**"nothing-here", "duplicate-of-X", "already-covered" and "manifest-only" are the
expected, valued outcomes for most of this batch, and most repos get one.** Where
a lead or a prior/sub-agent claim was contradicted by verification it is called
out — two are (CAD-Coder "new haul" was already vendored; the cadquery
`gear_generator`/`heatserts` lead resolves to already-covered).

Bottom line up front: **zero genuinely new fixture corpora** in this tail. The
real yield is (a) a handful of committed expected-REJECTION cases, (b) three
manifest-only reference corpora (BRepNet triples, BrickGPT fixtures, CADBench
grader), (c) two portable algorithms the harness may lack (a provider-signature
parity checker; a GDS DRC rule-id taxonomy in a different domain), and (d) one
duplicate and one mis-scraped non-CAD repo to drop.

---

## zip (115 files) — DUPLICATE

### LICENSE
N/A (the payload has no LICENSE — see CodeToCAD below).

### WHAT IT IS / VERDICT
`zip/CodeToCAD-develop/CodeToCAD-develop/` is a **literal copy of `CodeToCAD-develop`**.
The directory is misnamed (`zip`), not a real repo. Evidence: `diff` of the two
recursive file listings is empty except that the `CodeToCAD-develop` copy carries
committed `__pycache__/*.pyc` build artifacts the `zip` copy lacks; `cmp` on
`pyproject.toml`, `README.md`, `codetocad/__init__.py` reports byte-identical.

**duplicate-of CodeToCAD-develop.** Vendor nothing; treat as one repo.

---

## CodeToCAD-develop (138 files; ~78 non-pycache)

### LICENSE
**No LICENSE file present → manifest-only, vendor nothing.** (Verified: no
`LICENSE`/`COPYING` at repo root.)

### WHAT IT IS
Python text-to-CAD DSL with a core-interface / provider-adapter split
(`core/cad` abstract interface vs `integrations/build123d`, `integrations/open3d`
implementations). The harness already carries a CodeToCAD-derived
`domain.numeric.unit_expressions` (a `LengthExp` evaluator) and a
`domain.geometry.kinematics.gear_coupling` ("CodeToCAD gear constraint").

### READ (sub-agent, re-verified)
`tests/test_core.py`, `tests/test_signature_validation.py`,
`core/dimensions/length_expression.py`, `core/dimensions/angle.py`,
`core/enums/{cardinal_directions,axis,plane,preset_material}.py`, README; plus
`src/harnesscad/domain/numeric/unit_expressions.py` for the comparison.

### SKIMMED-NOT-READ
The ~50 provider-adapter modules under `integrations/`; the blender bootstrap.

### FINDINGS
1. **`tests/test_signature_validation.py` — an AST provider-signature-parity
   validator. Candidate algorithm the harness lacks.** Walks
   `integrations/*/cad/*.py` against `core/cad/*.py`, extracts top-level function
   signatures via `ast.unparse`, and asserts param-name / type-annotation /
   default / return-annotation parity between the abstract interface and every
   provider. A concrete "multi-backend contract drift" checker. Grep of
   `src/harnesscad/` and the index found no equivalent AST signature-parity check
   across the harness's own backends (`io/backends/{cadquery,onshape,stub}.py`).
   No-license → re-implement the idea, do not vendor.
2. `core/enums/cardinal_directions.py` (213 lines) — a named-position-on-bbox
   selector vocabulary (`TOP_FRONT_LEFT`, `LEFT_CENTER`, … each → a `Point` on the
   bounding box, right-handed +X-right/+Y-back/+Z-up). No `cardinal`/`TOP_FRONT`
   vocabulary in the index. Minor selector-naming reference, not a fixture.

### ALREADY COVERED
`core/dimensions/angle.py` (`Angle("90deg + 0.5rad") → 2.0708 rad`) and
`length_expression.py` → `domain.numeric.unit_expressions`, whose recursive-descent
`parse_angle` (deg/rad/grad) is **strictly superior** to CodeToCAD's
`eval(expr, {"__builtins__": None}, {})`. Recorded negative: both CodeToCAD
dimension classes use `eval` and `LengthExp.__pow__` is buggy
(`pow(self.value, other, mod)` passes a `LengthExp`, not `.value`) — a known-bad
pattern the harness already avoids.

### VERDICT
**manifest-only** — mine the one idea (AST signature-parity validator); everything
else is covered or unlicensed.

---

## action-convert-directory-main (7 files)

### LICENSE
`LICENSE` — **MIT**, "Copyright (c) 2022 KittyCAD". Verified.

### WHAT IT IS / READ
A composite GitHub Action that shells `zoo file convert` over a directory. CI glue.
Read `action.yml`, `README.md`, `test-files/` listing (all 7 files).

### FINDINGS
Essentially none. Only datum worth recording: the Zoo conversion format taxonomy
in `action.yml` — accepted inputs `{dae, fbx, obj, step, stl}`, outputs
`{dae, fbx, fbxb, obj, step, stl}`, default host `https://api.zoo.dev`. Already
subsumed by the far richer Zoo format-option matrix in report 02.
`test-files/{test-obj.obj,test-stl.stl}` are trivial known-good geometry with no
committed expected output (the Action only checks that files appear).

### VERDICT
**nothing-here** (CI glue).

---

## kittycad.ts-main (456) / kittycad.rs-main (106) / kittycad.go-main (66) / cli-main (126)

### LICENSE
All **MIT** (verified in report 02).

### VERDICT
**already-covered** — these four are read end-to-end in
`audit/deep_read/02_zoo_js_langs.md` (the `kittycad.py/.rs/.ts/.go / cli-main / Zoo`
section). Its findings stand: the `ErrorCode` retry taxonomy, the
`error_geometry_mismatch` state machine, the 14 unit enums, the format-option
matrix, the four disagreeing retry ladders, the CLI refusal predicates and
`gear.kcl` physics golden corpus, `expectedToFail.ts`, and the websocket filters.
Nothing to add here; not re-read to avoid duplicating that pass.

---

## scad-hs-master (15 files)

### LICENSE
`LICENSE` present; header reads "All rights reserved" (curated-code-cad's README
calls it BSD-3-Clause — the two disagree). Treated conservatively → **vendor
NOTHING** either way.

### WHAT IT IS / READ
Haskell → OpenSCAD source emitter (pretty-printer over a GADT `Model d` indexed by
a 2D/3D `Dimension`). Read `src/Graphics/Scad/Types.hs` in full; README is empty.

### FINDINGS (knowledge only, vendor nothing)
1. **Type-level 2D/3D discipline**: `data Dimension = Two | Three` + `type family V d`
   (V2 for 2D, V3 for 3D) makes "translate a 2D model by a 3D vector" a *type
   error*. A DSL correctness invariant expressible as expected-rejections in a SCAD
   grader.
2. **Radian→degree emission quirk**: internal `newtype Radian`, but the `Pretty`
   instance emits `180*r/pi` — API takes radians, SCAD output is degrees.
3. `difference(x, Union' ys)` is special-cased to `difference(){ x; ys… }`
   (semantic-preserving flattening). `Square`/`Rectangle` both emit `square(...)`;
   `Cube`/`Box` both emit `cube(...)`; 2D `Mirror (V2 x y)` promotes to
   `mirror([x,y,0])`.

### VERDICT
**vendor-nothing** — record quirks #1–#3 as SCAD-emission corner-case knowledge.

---

## scad-clj-master (11 files)

### LICENSE
`LICENSE` — **EPL-1.0** (Eclipse Public License). Verified.

### WHAT IT IS / READ
Clojure → OpenSCAD DSL (emits `(:offset {...} …)` s-expr IR, then SCAD). Read
`README.md` and `test/scad_clj/core_test.clj` in full; noted
`src/scad_clj/{model,scad,geometry,text}.clj`.

### FINDINGS
1. **Two documented semantic divergences from OpenSCAD** (README): "All angles are
   in radians" and "All primitive forms are centered at the origin." A scad-clj
   model differs from equivalent raw OpenSCAD by default centering and angle units
   — a DSL-dialect quirk relevant to any SCAD-generation grader.
2. `core_test.clj` — polymorphic `offset` dispatch (numeric → `{:r n}`; map →
   `{:r :delta :chamfer}`); an arg-overloading normalization, no rejection cases.

### ALREADY COVERED
Harness has `domain/programs/ast/openscad.py` + `scad_data_ir.py`. Only the
radians + origin-centered-primitives dialect note is additive.

### VERDICT
**already-covered** (dialect note aside).

---

## cadhub-main (456 files)

### LICENSE
`LICENSE` — **GPL-3.0** → vendor NOTHING; port ideas only if re-implemented.

### WHAT IT IS
CadHub — a model-gallery + live IDE web app for code-CAD (OpenSCAD, CadQuery,
JSCAD, Curv); RedwoodJS front end + Dockerized AWS-Lambda render backend that
shells out to the real CAD binaries.

### READ (sub-agent, re-verified)
README; `app/web/src/helpers/cadPackages/{common.ts,openScadController.ts,
cadQueryController.ts,curvController.ts,jsCad/jscadWorker.ts}`;
`app/api/src/docker/{common/utils.ts,openscad/runScad.ts,openscad/openscad.ts,
cadquery/runCQ.ts,curv/runCurv.ts}`. The ~30 `*.test.tsx` are React/Storybook
snapshot tests — no CAD.

### SKIMMED-NOT-READ
The RedwoodJS app (~400 files of web/UI/auth/GraphQL). Excluded as UI glue.

### FINDINGS (thin — engine error-handling, no fixtures)
1. `app/api/src/docker/common/utils.ts` — `runCommand(cmd, timeout,
   shouldRejectStdErr)`: stdout/stderr-reject ambiguity, hard-timeout reject;
   errors mapped to HTTP 400 `{error, fullPath}`.
2. `app/api/src/docker/openscad/runScad.ts` — invokes `openscad-nightly` under
   xvfb; `cleanOpenScadError()` regex-scrubs `/tmp/<hash>/main.scad` → `main.scad`
   (parser-message path normalization); timeouts 15 s preview / 60 s stl.
3. `app/web/src/helpers/cadPackages/common.ts` — `HealthyResponse | ErrorResponse`
   status union, 400/502 → message mapping; JSCAD in-browser compile-error path.
4. Expected-REJECTION (hardcoded, not a test): `app/api/src/docker/cadquery/runCQ.ts:33`
   returns `{error: 'python execution currently disabled...'}` unconditionally.
Not findings: `app/web/public/{hinge.stl,pumpjack.stl}` and `*/initialCode.{scad,py,curv,jscad.js}`
are demo/starter assets, no expected-output pairing.

### VERDICT
**mine-further (thin, ideas-only under GPL)** — the compile-error/timeout
normalization pattern; no fixture corpus.

---

## gaudi-frontend-main (44) / structural-frontend-main (61)

### LICENSE
Both **NO-LICENSE**.

### WHAT THEY ARE / READ
React (CRA) chat UIs for text-to-building generators with a Three.js GLB/STL
viewer. `gaudi-frontend` pairs with `gaudi-backend` (below). Read file trees,
READMEs; only test present in each is the default CRA `App.test.js`.
`src/assets/temp/*.stl` + `generated_model.glb` are demo output assets (no
expected-output pairing).

### VERDICT
**nothing-here** (pure React/CSS/shader glue; confident negative — a CRA tree is
diagnostic on its own).

---

## gaudi-backend-main (8 files)

### LICENSE
**NO-LICENSE** → vendor nothing.

### WHAT IT IS / READ
Flask backend: text prompt → LLM-generated parametric Blender "plate" script →
headless Blender → GLB, with an LLM error-fix retry loop. Read `app.py` (480
lines) in full; `template.py` present; `initial_model.glb` output asset.

### FINDINGS (modest)
1. `app.py` `render_script()` (L436-466) — a real, if crude, generate→render→
   detect-error→regenerate loop: naive Blender error detection
   (`if "Error" in full_output or "Traceback"`, extract matching lines), then
   `generate_initial_script`/`fix_claude`/`fix_script` feed the error back to the
   model. A substring heuristic, no taxonomy.
2. **Operational note (not a harness finding): a live Anthropic API key is
   hardcoded at `app.py:23`.** Flagged for the maintainer; nothing to vendor.

### VERDICT
**mine-further (lite, ideas-only, no license)** — only the render-error-detect +
regenerate pattern is measurable, and the harness already has richer refine loops
(`domain.editing.refine_loop`, `agents.generation.dual_loop`).

---

## Studio-OSS-main (119 files)

### LICENSE
`LICENSE` — **GPL-3.0** → vendor NOTHING; port ideas only.

### WHAT IT IS
Next.js text-to-CAD studio (parse-intent → generate build123d code → compile →
critique/score). Not pure UI — real server-side geometry.

### READ (sub-agent, re-verified)
`app/api/compile/route.ts` (470 lines, full); trees of `app/api/*` and
`components/parametric/*`.

### FINDINGS
1. `app/api/compile/route.ts` — executes build123d, exports STL, and extracts a
   rich BREP metrics schema for scoring: bbox/dimensions, volume, surface_area,
   face/edge/vertex counts, **face_type & edge_type distributions**
   (PLANE/CYLINDER/CONE/SPHERE/TORUS/BSPLINE), `is_valid` (OpenCascade
   `BRepCheck_Analyzer`), aspect_ratio, compactness, center_of_mass, symmetry_hint;
   an HTTP taxonomy (400/422/501); and a **fillet/chamfer auto-heal retry** (strips
   fillet/chamfer lines on `BRep_API: command not done` and re-runs).

### ALREADY COVERED
The metrics are largely present as harness modules: `eval.bench.geometry.step_file_metrics`,
`eval.bench.vision.face_segmentation`, and the face/edge-type surface-classification
work in `domain.geometry.topology.edge_convexity`. The **fillet auto-heal on
`BRep_API: command not done`** is the one concrete repair-retry rule worth checking
against `agents.generation.occt_quirks`; it is a GPL source, so re-implement.

### VERDICT
**mine-further (ideas-only under GPL)** — the fillet auto-heal repair rule and the
consolidated BREP-metrics-as-score schema.

---

## Vibe_Layout-main (48 files)

### LICENSE
`LICENSE` — **MIT**, "(c) 2026 JJH and Vibe_Layout contributors". Verified.

### WHAT IT IS
Prompt→GDS (KLayout) layout agent for nano-fab devices that forces each request
through executable validation harnesses before accepting a layout. A genuine
verification harness — but in the **GDS / 2D-planar-layout / nanofab domain**, not
mechanical B-rep. (Its `CLAUDE.md` agent-contract is irrelevant to this audit and
was disregarded.)

### READ (sub-agent, re-verified)
`src/klayout_harness/feedback.py` (443 lines, full); README; noted
`semantic.py`, `cad.py`, per-device `tests/test_*.py`.

### FINDINGS
1. `src/klayout_harness/feedback.py` — `ValidationFinding`/`ValidationReport`
   dataclasses with an explicit **rule_id error taxonomy**: `geometry.empty`,
   `cell.missing`, `layer.missing`, `drc.min_width`, `drc.min_spacing`,
   `drc.minimum_resolution`, `dbu.exact_mapping`, `geometry.positive_area`,
   `geometry.bbox`, `hierarchy.instance`, `geometry.electrode_count`,
   `nanogap.sweep`, `srr.inner_size`, … Includes **µm→DBU unit-mapping checks**
   (`maps_exactly_to_dbu`), GDS readback via `klayout.db`, and expected-shape-count
   assertions per device (e.g. Hall bar == 13 shapes).
2. `src/klayout_harness/semantic.py` + `tests/test_{hall_bar,nanogap_array,
   micro_channel,srr_feedline}.py` — per-device expected-geometry assertions.

### ALREADY COVERED
The harness's error taxonomies are all 3D/kernel-flavoured
(`eval.verifiers.kernel_preflight`, `agents.generation.feedback_taxonomy`,
`core.observe.FailureTaxonomy`). No GDS/DRC (`min_width`/`min_spacing`/DBU) rule-id
taxonomy exists — a genuinely absent *domain*, not a duplicated one.

### VERDICT
**mine-further** — the DRC rule-id taxonomy + µm→DBU exact-mapping check are a
clean, MIT-licensed, verifier-first pattern; but note the domain mismatch (2D
GDS/EDA), so it is an adjacent extension, not a core mechanical-CAD fixture.

---

## spatialhero-main (37 files)

### LICENSE
**NO-LICENSE → manifest-only, vendor nothing.**

### WHAT IT IS
An RLHF/reward-model pipeline that fine-tunes LLMs to emit CadQuery code, scored by
a multi-modal verifier (code validity / dimensional accuracy / topology / visual).
Not UI — a Python verify+reward pipeline.

### READ (sub-agent, re-verified)
`core/verifier.py` (380 lines, full); `tests/fixtures/sample_codes.py` (full);
README; tree.

### FINDINGS
1. `tests/fixtures/sample_codes.py` — **paired valid/invalid CadQuery fixtures with
   named expected-REJECTION cases**: `VALID_SIMPLE_BOX`, `VALID_CHAIR`,
   `INVALID_SYNTAX`, `MISSING_RESULT`, `INVALID_IMPORT`, `RUNTIME_ERROR`
   (div-by-zero), and a **security case `DANGEROUS_CODE` = `os.system("rm -rf /")`**.
   A small ready-made adversarial CadQuery corpus.
2. `core/verifier.py` — `CodeVerifier` (syntax via `ast.parse` / execution /
   geometry), flags `eval`/`exec`/`compile`/`__import__` and file ops;
   `GeometricConstraintChecker.check_physical_plausibility` (aspect-ratio, fill-ratio,
   SA/volume thresholds) and `check_constraints` (max_width/volume/min_faces/
   must_be_closed); default dimensional tolerance 0.05.

### ALREADY COVERED
The **verifier logic is covered**: `domain.programs.validate.code_safety`
(`assert_cad_code_safe`, per-kernel import allowlist, blocks `os.system(...)`),
`eval.bench.sequence.code_validity` (static safety + VSR), and
`domain.programs.validate.fluent_subset_policy`. What is NOT in the harness is a
**committed adversarial CadQuery snippet corpus** (grep of `src/harnesscad/` finds
the safety *logic* but no `rm -rf`/`DANGEROUS` fixture strings).

### VERDICT
**mine-further (manifest-only)** — register `sample_codes.py`'s good/bad snippets
(esp. the security case) as a code-safety regression manifest; the checker is
already ported; no license → vendor nothing.

---

## solidtype-main (500 files)

### LICENSE
**NO-LICENSE → manifest-only, vendor nothing.**

### WHAT IT IS
SolidType — a from-scratch, history-capable parametric CAD app with its OWN
geometry kernel (`packages/core`), plus an OpenCascade.js path and a TanStack/AI
web app. Emphatically not pure UI.

### READ (sub-agent, re-verified)
`packages/core/src/topo/validate.ts` (882 lines, full);
`packages/core/src/num/tolerance.ts` (full);
`packages/app/tests/integration/ai-evals.test.ts`; core `src/` + `tests/` trees.

### SKIMMED-NOT-READ
The TanStack web app (majority of 500 files) — UI, excluded.

### FINDINGS
1. `packages/core/src/topo/validate.ts` — B-rep validation with a **24-value
   `ValidationIssueKind` taxonomy** (nonManifoldEdge, brokenLoopCycle, twinMismatch,
   zeroLengthEdge/Area, sliverFace, boundaryEdge, crack, loopNotClosed,
   duplicateVertex, faceShellMismatch, …), severity levels, manifold/degenerate/
   sliver checks (isoperimetric ratio), `ValidationReport`.
2. `packages/core/src/num/tolerance.ts` — centralized tolerance/unit model
   (length 1e-6, angle 1e-8 for mm CAD), `snap`, `eqLength`/`eqAngle` (2π wrap),
   tolerant `lt/lte/gt/gte`; plus `src/num/predicates.ts` (robust predicates) with
   `tests/num/{predicates,tolerance,rootFinding,integer-geometry}.test.ts`.
3. `packages/core/tests/` — a real kernel test suite incl.
   `model/goldenMesh.test.ts` + `tests/fixtures/goldenMeshUtils.ts` (golden-mesh
   regression harness), `topo/{validate,heal,sameParameter}.test.ts`,
   `boolean/planar/*`, `export/stl.test.ts`.

### ALREADY COVERED
Substantially. The harness has half-edge manifold invariants
(`domain.geometry.mesh.halfedge.MeshIssue`), loop validity
(`domain.geometry.sketch.loop_validity`, `domain.programs.validate.openecad_validity`),
mesh repair (`repair_toolkit.is_manifold/is_closed`), robust predicates
(`domain.numeric.exact_predicates`: orient2d/3d/incircle/insphere), and a full
tolerance stack (report 02's kcl ladder; `domain.geometry.parametric.chord_tolerance`).
The **delta** is solidtype's *named 24-value half-edge topology taxonomy* — the
`twinMismatch`/`sliverFace`/`crack`/`brokenLoopCycle` labels together as one enum —
which the harness spreads across modules rather than naming; a reference for a
consolidated topology error-taxonomy.

### VERDICT
**mine-further (manifest-only)** — the consolidated 24-value topology taxonomy as
a reference; the underlying checks and tolerance model are already covered; no
license → vendor nothing.

---

## text-to-cad-blender-addon-main (19 files)

### LICENSE
`LICENSE` — **MIT**, "Copyright (c) 2024 Zoo". Verified.

### WHAT IT IS / READ
Blender addon: prompt → Zoo `text-to-cad` API → download → import into Blender.
Mostly bpy/UI glue. Read `src/text_to_cad.py` (346 lines, full); tree; requirements.

### FINDINGS (minor, all already-covered by report 02's Zoo work)
`OutputFormat` enum mapping 6 formats → importers (fbx/glb/gltf/obj/ply/stl);
`call_zoo_api()` async operation-status polling
(`while status not in ["completed","failed"]`), returns `result["error"]` on
failure; a **base64-padding workaround** for Zoo's Base64data
(`outputs.strip("=") + "==="`); missing-token error path.

### VERDICT
**nothing-here** — the format enum and async-status handling are covered by
report 02's kittycad/CLI pass; the base64 padding quirk is a client detail, no
fixtures.

---

## diff-viewer-extension-main (86 files)

### LICENSE
**NO-LICENSE** → vendor nothing.

### WHAT IT IS / READ
Zoo Chrome extension that renders 3D CAD file diffs inline on GitHub via the
KittyCAD conversion API. Read `tests/fixtures.ts` (Playwright), `src/chrome/web.ts`
(partial), `src/chrome/diff.ts`, `src/utils/three.ts`.

### FINDINGS (borderline)
`src/chrome/diff.ts:10` — `extensionToSrcFormat` taxonomy (fbx, gltf, obj, ply,
sldprt, stp→step, step, stl; **`dae` explicitly disabled**, comment "Disabled in
new format api"), `isFilenameSupported()`. A small format-support map, subsumed by
report 02. `tests/*-snapshots/*.png` are visual-regression screenshots — no
committed source geometry (files are fetched from GitHub at test time).

### VERDICT
**nothing-here**.

---

## AlphaCAD-main (570 files; ~30 real source)

### LICENSE
**NO-LICENSE** → vendor nothing.

### WHAT IT IS / READ
A text-to-3D-LEGO-brick demo (BrickGPT wrapper): Flask + React + Three.js gallery
on a 20³ brick grid. Most of the 570 files are `.history/` snapshots + `__pycache__`.
Read `summit-demo/utils.py` (678 lines, full); skimmed `summit-demo/*`,
`web/src/App.tsx`. Confirmed zero `.step/.stp/.stl/.obj/.brep/.scad/.kcl`.

### FINDINGS (weak)
`summit-demo/utils.py` `validate_brickgpt_prompt()` (L118-169) — a runtime keyword
heuristic rejecting out-of-scope prompts (curves/spheres/oversize/non-brick), flips
`valid=False` at >2 warnings; `post_json()` retry-with-backoff. Runtime
string-matching, not a committed reject corpus.

### VERDICT
**nothing-here** (BrickGPT's real verifiers are covered under BrickGPT below).

---

## Code2World-main (552 files) — NOT CAD

### LICENSE
NO-LICENSE (moot).

### WHAT IT IS
**Not CAD.** A VLM "GUI World Model" (arXiv 2602.09856) that predicts the next
Android screen via HTML code generation, built on Google's AndroidWorld benchmark
(AndroidWorld agents/task_evals + miniwob HTML apps + an Android emulator server).
Verified: README, dir tree, and a grep of all 149 `.py` for CAD/geometry terms —
the only CAD-keyword hit is "salmon fillets" in `task_evals/single/recipe.py`.

### VERDICT
**nothing-here** — mis-scraped into the corpus on a "renderable code generation"
name collision. **Recommend dropping from the CAD corpus entirely.**

---

## AutoBrep-main (25 files)

### LICENSE
`LICENSE` — **MIT**, "Copyright (c) 2025 Autodesk AI Lab". Verified (vendorable).

### WHAT IT IS / READ
Autoregressive B-rep generation model (VAE + FSQ + AR transformer). Read the full
listing; `configs/sample.json`, `configs/autobrep.yaml`,
`core/src/autobrep/inference/post_process.py` (head).

### FINDINGS
None meeting the bar. No committed geometry anywhere. `post_process.py` has
`detect_shared_edge`/`detect_shared_vertex`/`joint_optimize` (B-rep sewing) and
`configs/sample.json` exposes sewing tolerances (`vertex_threshold: 0.002`,
`sewing_tolerance: 0.002`, `z_threshold`) — generation-time params with no paired
expected output. Everything else is nn/AR-sampler/training code (in the IGNORE
set). Harness sewing already covered by `domain.geometry.topology.sew`.

### VERDICT
**nothing-here**.

---

## BRepNet-master (443 files)

### LICENSE
`LICENSE` — **CC-BY-NC-SA 4.0 (NON-COMMERCIAL) → manifest-only, vendor nothing.**

### WHAT IT IS
B-rep face segmentation (Fusion Gallery); convolutional kernels over topology.

### READ (sub-agent, re-verified)
`pipeline/face_index_validator.py`, `pipeline/segmentation_file_crosschecker.py`,
`feature_lists/all.json`, `kernels/winged_edge.json`,
`example_files/feature_standardization/s2.0.0_step_all_features.json`, a
`_topology.json`/`_labels.json` pair; counted fixtures.

### FINDINGS (all manifest-only)
1. `tests/test_data/equivalent_dataloaders/` — **94 triples** of
   `*_topology.json` + `*_features.json` + `*_labels.json` (B-rep topology graph +
   per-entity feature vectors + per-face segment labels). Genuine known-good
   geometry↔label pairings, measurable for a segmentation/topology-consistency
   verifier.
2. `example_files/step_examples/` (25 `.stp`) + `tests/test_data/*.stp` (3) — real
   STEP solids.
3. `pipeline/face_index_validator.py` — a real validity checker (face-count match +
   face-order-vs-color + bbox cross-check between STEP and mesh; rejects files whose
   labels don't correspond). `pipeline/segmentation_file_crosschecker.py` — checks
   `#faces(STEP) == #segment-labels`, a clean expected-rejection.
4. `feature_lists/all.json` + 7 `no_*.json` — the face/edge/coedge **feature schema**
   enum (Plane/Cylinder/Cone/…, Concave/Convex/Smooth edges). `kernels/*.json` (7) —
   topological-walk kernel definitions (winged-edge / asymmetric neighborhoods).

### ALREADY COVERED
The harness imported the STEP half as `eval.corpus.fixtures.brepnet_steps`
("known-good/known-bad STEP sets, ingest_canaries, manifest") — the step_canaries
the brief referenced. That module tracks **STEP cases only**; the 94
topology/feature/label triples, the feature-schema enum, and the walk-kernel defs
are **not** reflected (0 index hits for `feature_list` / `winged`-kernel).

### VERDICT
**mine-further (manifest-only)** — register the 94 triples + feature schema + kernel
defs as reference manifests; NonCommercial → vendor nothing.

---

## BrickGPT-main (91 files)

### LICENSE
`LICENSE` — **MIT**, "Copyright (c) 2025 Ava Pun" (CMU). Verified (vendorable).

### WHAT IT IS / READ (sub-agent, re-verified)
LLM LEGO-brick generation with physics/stability post-checks. Read
`tests/test_brick_structure.py`, `data/brick_library.json`,
`stability_analysis/stability_analysis.py` (head), `mesh2brick/tests/car.txt`;
listed obj/txt fixtures.

### FINDINGS
1. `tests/test_brick_structure.py` — parametrized **known-good/known-bad literals**:
   collision pairs (`'2x6 (0,0,0)\n2x6 (1,0,0)'` → collides) and floating pairs
   (`… (2,0,1)` → floating), plus round-trip equality across txt/json/ldr. Reusable
   expected-rejection cases.
2. `src/mesh2brick/tests/{car,chair,ship}.obj` + matching `.txt` — mesh→brick
   expected-output fixtures (input `.obj`, expected brick layout `.txt`), exercised
   by `mesh2brick_test.py`.
3. `src/brickgpt/data/brick_library.json` — brick catalog with mass/dimensions/partID.

### ALREADY COVERED
The **checkers are already ported**: `domain.geometry.assembly.brick_connectivity`
(floating_bricks/grounded/connection_area), `brick_structure` (bricks_overlap),
`brick_assembly` (is_buildable/is_assembly_stable), a stability LP
(`solve_lp`/analyze_stability), and `domain.fabrication.legolization`
(is_valid_placement/physics_aware_rollback/rejection_sample). (This is the
"verifiers already mined" note in the brief.) **Not covered**: the committed fixture
*files* — the car/chair/ship obj↔txt pairs and the parametrized collision/floating
literals as a regression corpus.

### VERDICT
**mine-further (thin)** — vendor only the obj↔txt fixtures + parametrized good/bad
literals as a regression corpus; the verifiers are done.

---

## CAD-Coder-main (462 files) — CORRECTION

### LICENSE
`LICENSE` — **Apache-2.0**. Verified.

### WHAT IT IS
LLaVA-based image→CadQuery-code generation (Doris, Alam, Nobari & Ahmed, 2025,
arXiv:2505.14646).

### READ (sub-agent, re-verified against harness)
`inference/cadquery_test_data_subset100.jsonl`, `scripts/compute_iou.py`,
`inference/test100_gt_steps/` (100 `.step`, counted).

### FINDINGS / CORRECTION
A sub-agent flagged this as the "best new haul" (100 CadQuery-code / GT-STEP pairs
+ an inertia-align volumetric-IoU grader in `compute_iou.py`). **Verification
contradicts that — it is already vendored.** The harness has:
- `eval.corpus.fixtures.cad_coder_heldout` ("CAD-Coder's 100-part held-out set: GT
  STEP + reference CadQuery code pairs"; `HeldoutPair`, `manifest`, `pairs`) with a
  committed `eval/corpus/fixtures/cad_coder/MANIFEST.json` (sha256-referenced).
  This is the "heldout already → fixture" the brief noted; the "rest" the sub-agent
  found IS that same subset100 + its 100 GT STEPs.
- `domain.geometry.transforms.principal_axes` — the exact CAD-Coder
  correspondence-free inertia / principal-axis alignment, the **four eigenvector-sign
  rotation family**, and `voxel_iou_score` — i.e. `compute_iou.py`'s align-and-IoU
  grader, already ported (`principal_axes.py` cites arXiv:2505.14646 directly).
- `data.dataengine.reward.group_relative_advantage` — CAD-Coder's GRPO reward.

### VERDICT
**already-covered.** (A textbook case of the brief's warning not to inherit
sub-agent/inventory claims without checking: the "un-mined rest" was already
mined.)

---

## CAD-Editor-main (71 files)

### LICENSE
`LICENSE` — **MIT**, "Copyright (c) Microsoft Corporation". Verified.

### WHAT IT IS / READ
LLM text-based CAD editing (locate-then-infill over CAD op-sequences). Read README,
`utils/eval_cad.py` (head), `utils/geometry/{arc,circle,line,curve,obj_parser}.py`.

### FINDINGS
None meeting the bar. `utils/eval_cad.py` is standard generative-set metrics
(Chamfer + COV/MMD/JSD via NearestNeighbors over `.ply`) — already covered
(`eval.bench.geometry.chamfer` + the generative-metric modules). The JSON-OBJ
sketch parser has no committed fixtures; the actual data ships as an uncommitted
`data/processed.zip`. No fixtures on disk.

### VERDICT
**nothing-here** (metrics already covered; no committed geometry).

---

## Sketch2CAD-master (34 files)

### LICENSE
`LICENSE` — **MIT**, "Copyright (c) Microsoft Corporation". Verified.

### WHAT IT IS / READ
SIGGRAPH-Asia 2020 sequential CAD-by-sketching; TensorFlow operator regressors.
Read README, full listing.

### FINDINGS
None. Pure network train/deploy: per-operator train/test scripts
(extrusion/bevel/sweep/addSub regressors, opt classifier), `network.py`,
`loader.py`, TF freeze / lmdb→tfrecord tooling. No committed geometry, no graders,
no schemas, no expected-output pairs (all IGNORE-set).

### VERDICT
**nothing-here**.

---

## StepForge-main (97 files)

### LICENSE
`LICENSE` — **Apache-2.0**. Verified.

### WHAT IT IS / READ (sub-agent, re-verified against harness)
LLM caption→STEP generation with a regex STEP-reserialization data pipeline + RL
reward. Read `data/step_parser.py`, `scripts/validate_step_files.py`,
`reward/scd_reward.py`, `tests/test_roundtrip.py`, `scripts/analyze_step_quality.py`.

### FINDINGS (all already-covered)
`data/step_parser.py` + `dfs_reserializer.py` + `step_restructurer.py` carry a
STEP-text corner-case taxonomy (C5–C10): C9 `#`-in-string-literals masked before
ref-extraction (`PRODUCT('Bracket #42 rev B')`), `''`-escaped quotes; C6 complex
entities `#3=( A() B() );`; C7 hard-fail on dangling ref; C8 preserve `ENDSEC;`
before `DATA;`. `scripts/analyze_step_quality.py` is a structural checker
(`has_terminator`, `dangling_refs`, `dropped_complex`, `empty_output`,
`entity_count`). `reward/scd_reward.py` is a Scaled-Chamfer reward with
reward-hacking guards (`_MIN_TRIANGLES`, `_MIN_UNIQUE_POINTS`).

### ALREADY COVERED
Heavily. The harness has `io.formats.step_reserialize` + `io.ingest.step_reserialize`
(DFS reserialization / renumber / normalize_reals — StepForge's exact pipeline), a
`io/formats/step.py` parser handling `''`-escaped strings, complex `(A()B())`
instances and comment stripping, and `io/formats/step_graph.py` with
`dangling_references()`/acyclicity/roots. SCD/Chamfer reward covered. C5–C10 all
matched. No committed geometry to vendor (tests read external dirs via env var).

### VERDICT
**nothing-here** — parser/reward already implemented; no fixtures. (Marginal: the
`analyze_step_quality.py` named enum is a tidy reference, but the checks exist.)

---

## Graph-CAD-main (20 files)

### LICENSE
**NO-LICENSE → manifest-only, vendor nothing.**

### WHAT IT IS / READ (sub-agent, re-verified)
A Blender-Python text-to-CAD benchmark (CADBench) graded by a VLM judge. Read
`CADBench.jsonl` (2 of 700 rows), `evaluate_and_report.py` (head), prompt listing.

### FINDINGS (manifest-only)
1. `CADBench.jsonl` — **700** rubric-annotated tasks:
   `{id, name, instruction, criteria{Object Attributes / Spatial / Instruction}, …}`.
   The `criteria` tree is the task/rubric schema.
2. `evaluate_and_report.py` — the grader: renders outputs and scores each criterion
   via a VLM (OpenAI-compatible) judge across three dimensions (Attr/Spat/Inst).
   Note: LLM-as-judge, **non-deterministic** — not a hard geometric check.
3. `prompt_sft/{bpy,graph,mcp}_prompt.txt` — three prompt templates.

### ALREADY COVERED
`eval.bench.imports.graphcad_cadbench` ("700 rubric-annotated text-to-CAD tasks;
GraphCadTask, RubricRow, rubric, manifest") — the 700-task manifest the brief
referenced; the rubric schema is captured. The VLM grader's `DIMENSION_MAP` and the
three prompt templates are not imported.

### VERDICT
**manifest-only** — 700 tasks already registered; only the grader-dimension
structure + prompt templates remain as reference; no license → vendor nothing.

---

## curated-code-cad-main (10 files)

### LICENSE
`LICENSE` — **MIT**, "Copyright (c) 2020 Kurt Hutten". Verified.

### WHAT IT IS / READ
A code-CAD landscape catalog (README) + one "birdhouse" part authored in 8 DSLs.
Read README, `birdhouse/OpenSCAD.scad`, `birdhouse/JSCAD.js` (full); confirmed the
other 6 (build123d, CadQuery, CascadeStudio.js, DeclaraCAD.enaml, FreeCad.py,
sdfx.go) exist.

### FINDINGS
A cross-DSL semantic-equivalence fixture set: the *same* parametric birdhouse
(identical params `width=120, height=85, holeR=28, thickness=2, hookHeight=10`) in
8 languages. **Limiting caveat**: `OpenSCAD.scad` depends on non-vendored
`roundanything/polyround.scad` (`$fn=50`), so the variants are **not
self-contained/runnable**, and there are **no committed golden STL/STEP** — any
equivalence check must be geometric, not exact.

### ALREADY COVERED
The birdhouse is already `eval.corpus.fixtures.birdhouse_nversion`. The remaining 7
variants are low-value (no goldens, external deps).

### VERDICT
**already-covered** (fixture concept present; the extra variants add little).

---

## OpenCAD-Examples-main (9 files)

### LICENSE
**NO-LICENSE → manifest-only, vendor nothing.**

### WHAT IT IS / READ
6 example scripts for a fluent in-process Python CAD API
(`Sketch().rect().circle(subtract=True)… → Part().extrude().fillet()`) + one
agent-codegen demo. Read all 6 example `.py`, `agents/generate_mounting_bracket_code.py`,
both READMEs.

### FINDINGS
1. The harness's `discipline_examples` was already built from these; nothing new
   geometric beyond it. **No committed expected outputs** (`.step`/tree JSON are
   produced at runtime), **no rejection cases** — inputs without goldens.
2. Minor: the agent example carries a `FeatureTree`/`FeatureNode` DAG schema
   (`operation`, `parameters`, `depends_on`, `status`, `shape_id`) — low value,
   harness has richer op-log/session state.
3. The upstream example *sources* are clean — the "bug-ridden" note in the brief
   pertains to the harness's derived copy, not upstream. The fluent API does have
   ambiguous signatures (`Part().cylinder(14, 10)` unlabeled radius/height;
   `.fillet(edges="top")` string selector) worth noting as fragile-to-generate.

### VERDICT
**manifest-only** — nothing new beyond `discipline_examples`.

---

## AutoCAD-main (6 files)

### LICENSE
**NO-LICENSE** → vendor nothing.

### WHAT IT IS / READ
A `pywin32` COM-automation wrapper for desktop AutoCAD (2D drafting: layers,
blocks, attributes, primitives, dimensions). Read `README.md` in full;
`AutoCAD.py` + `example_house.py` present (2 source files, rest is `.pyc`/gitignore).

### FINDINGS
None. It is a Windows-only COM API surface with no committed geometry, no expected
outputs, no rejection cases, and no text-to-CAD component. Not a generator, not a
verifier.

### VERDICT
**nothing-here**.

---

## cadquery-plugins-main (83 files) — DELTA to report 01

### LICENSE
`LICENSE` — **Apache-2.0**. Confirmed (matches report 01).

### STATUS
Covered in `audit/deep_read/01_occt_kernels.md` (§4), which left an **unresolved
lead**: "read `gear_generator` and `heatserts`." **Lead resolved here — both are
already covered:**
- `plugins/heatserts/heatserts.py` — a 4-row heat-set insert table
  (`heatsert_dims`: M6/M5/M4/M3 → diam/depth/bolt_diam). The harness's
  `domain.standards.heatsert_bores` is a **richer** heat-set bore schedule
  (`insert_dims`, `bore_depth`, `bore_volume`, `melt_displacement`, `fits_in_wall`,
  `select_for_bolt`, chamfer). Subsumed.
- `plugins/gear_generator/gear_generator/helpers.py` — `involute()` and
  `spherical_involute()` (bevel gears). Both are in the harness:
  `domain.geometry.kinematics.involute_gear` (`involute_point`) and
  `domain.geometry.kinematics.bevel_gear` ("Bevel-gear cone geometry and the
  spherical involute"; `spherical_involute`). Subsumed.

### FINDINGS (DELTA — concrete expected-REJECTION fixtures the prior directory-read
missed)
1. `tests/test_teardrop.py::test_teardrop_clip_value_illegal` — **parametrized
   expected-`ValueError`**: `teardrop(rad, 0, clipval)` rejects `clipval = 2*rad`
   ("argument must be less than …") and `clipval = -rad` ("argument must be greater
   than …"); the valid clip at `4.1` carries a geometric golden (top vertex z≈4.1,
   mirror-symmetry `|x1+x2| < 0.001`, `x1 ≈ -5.65685`).
2. `tests/test_cq_cache.py::test_workplane_typeerror` — **expected `TypeError`**: the
   `cq_cache` memoizer refuses un-hashable `cq.Workplane` args (positional and
   kwarg) — a caching-layer input-validation reject.

### VERDICT
**already-covered** overall (gear/heatsert lead now closed); the teardrop
illegal-clip and cq_cache TypeError are two small **expected-rejection fixtures**
worth capturing if not already in the corpus.

---

## cadquery-contrib-master (59 files) — DELTA to report 01

### LICENSE
`LICENSE` — **MIT**, "Copyright (c) 2018 Dave Cowden". Confirmed (report 01 already
corrected this from a mistaken "Apache").

### STATUS
Covered in report 01 (§5) as nothing-here for the examples/notebooks. **DELTA: an
`mcp-server/` subtree that likely postdates that pass.**

### FINDINGS
`mcp-server/cadquery_mcp_server.py` + `test_cadquery_mcp_server.py` — an MCP server
exposing 4 tools (`render`→SVG, `inspect`→geometry, `get_parameters`, `export`) with
a **3-way execution error taxonomy carrying exact expected strings**:
`"No shape produced"` (empty result), `"Syntax error"` (bad Python),
`"Execution error"` (runtime raise, tested via `raise ValueError('test error')`);
plus **geometry-inspection goldens** (bbox `"10.0000"` for a 10 mm box; volume/area/
center-of-mass/topology counts) and an export-format list
`{STEP, STL, SVG, DXF, AMF, 3MF, VRML, BREP}`.

### ALREADY COVERED
The *taxonomy* is covered conceptually: `agents.generation.feedback_taxonomy`
(CADCodeVerify error types), `core.observe.FailureTaxonomy`, and
`agents.generation.dual_loop` (inner "execution errors" loop);
`domain/programs/ast/cadquery.py` handles CadQuery AST. Not covered: the **exact
render/execution error-triad strings + the inspection golden contract** as a
ready-made test oracle.

### VERDICT
**already-covered** except the `mcp-server/` subtree → **mine-further** on that one:
its error-triad + inspection goldens are a concrete, MIT-licensed test oracle.

---

## Summary — set C dispositions (30 repos)

| Repo | Files | License | Verdict |
|---|---|---|---|
| zip | 115 | (payload) | **duplicate-of CodeToCAD-develop** |
| CodeToCAD-develop | 138 | none | manifest-only (1 idea: AST sig-parity check) |
| action-convert-directory-main | 7 | MIT | nothing-here |
| kittycad.ts / .rs / .go / cli-main | 456/106/66/126 | MIT | already-covered (report 02) |
| scad-hs-master | 15 | "all rights reserved" | vendor-nothing (quirks only) |
| scad-clj-master | 11 | EPL-1.0 | already-covered (dialect note) |
| cadhub-main | 456 | GPL-3.0 | mine-further (thin, ideas-only) |
| gaudi-frontend / structural-frontend | 44/61 | none | nothing-here |
| gaudi-backend-main | 8 | none | mine-further (lite; leaked API key flagged) |
| Studio-OSS-main | 119 | GPL-3.0 | mine-further (ideas-only: fillet auto-heal) |
| Vibe_Layout-main | 48 | MIT | mine-further (GDS DRC taxonomy; other domain) |
| spatialhero-main | 37 | none | mine-further (manifest-only: adversarial CQ corpus) |
| solidtype-main | 500 | none | mine-further (manifest-only: 24-value topo enum) |
| text-to-cad-blender-addon | 19 | MIT | nothing-here |
| diff-viewer-extension | 86 | none | nothing-here |
| AlphaCAD-main | 570 | none | nothing-here |
| Code2World-main | 552 | none | **nothing-here — NOT CAD, drop from corpus** |
| AutoBrep-main | 25 | MIT | nothing-here |
| BRepNet-master | 443 | CC-BY-NC-SA | mine-further (manifest-only: 94 triples + schema) |
| BrickGPT-main | 91 | MIT | mine-further (thin: obj↔txt + reject literals) |
| CAD-Coder-main | 462 | Apache-2.0 | **already-covered (correction: was called new)** |
| CAD-Editor-main | 71 | MIT | nothing-here |
| Sketch2CAD-master | 34 | MIT | nothing-here |
| StepForge-main | 97 | Apache-2.0 | nothing-here (parser/reward already covered) |
| Graph-CAD-main | 20 | none | manifest-only (700 tasks covered) |
| curated-code-cad-main | 10 | MIT | already-covered (birdhouse fixture) |
| OpenCAD-Examples-main | 9 | none | manifest-only (discipline_examples) |
| AutoCAD-main | 6 | none | nothing-here (COM 2D-drafting wrapper) |
| cadquery-plugins-main | 83 | Apache-2.0 | already-covered (lead closed) + 2 reject fixtures |
| cadquery-contrib-master | 59 | MIT | already-covered + mcp-server error-triad delta |

**Highest-value actionable items in this tail** (all small, none a new corpus):
1. **spatialhero** `sample_codes.py` — adversarial CadQuery good/bad snippets incl.
   `os.system("rm -rf /")`, as a code-safety regression manifest (checker exists).
2. **BRepNet** 94 topology/feature/label triples + feature schema + walk-kernels
   (manifest-only, NonCommercial) — beyond the existing step_canaries.
3. **BrickGPT** obj↔txt fixtures + parametrized collision/floating literals
   (MIT-vendorable regression corpus; verifiers already ported).
4. **cadquery-plugins** teardrop-illegal-clip `ValueError` + `cq_cache` `TypeError`;
   **cadquery-contrib** `mcp-server` render/execution error-triad + inspection
   goldens (concrete expected-reject/oracle strings).
5. **CodeToCAD** AST provider-signature-parity validator; **Vibe_Layout** GDS DRC
   rule-id taxonomy (adjacent EDA domain).

**Verification corrections recorded** (per brief: do not inherit dismissals or
claims): CAD-Coder was NOT an un-mined new haul — its 100-pair heldout set
(`cad_coder_heldout` + `MANIFEST.json`) and its inertia-align/voxel-IoU grader
(`principal_axes`) are already vendored. The report-01 cadquery `gear_generator` /
`heatserts` lead resolves to already-covered (`involute_gear` / `bevel_gear` /
`heatsert_bores`). Code2World is not a CAD repo at all.
