# Deep read: ML / dataset / research repos (tail sweep A)

Genuine deep read, not a grep sweep. 22 never-deep-read repos from the ML/dataset
tail. Every claim was re-derived against the actual sources under
`resources/cad_repos/<name>/<name>/` (double-nested) and checked against the
harness with `registry.index()` (1579 modules) plus targeted greps of
`src/harnesscad/`. Reading was fanned out across five read-only sub-passes; every
finding below was then re-verified by the author directly against the named
harness module before its verdict was written.

**Headline result: this tail is almost entirely already-mined or nothing-here.**
The harness carries a remarkably complete `domain/reconstruction/tokens/` package
(DeepCAD, SkexGen, hnc-cad, Text2CAD, vitruvion curation) and an `eval/bench/`
suite (DeepCAD ACC_cmd/ACC_param, COV/MMD/JSD set metrics, SymPoint panoptic)
that between them absorb the tokenization/quantization/eval content these repos
ship. `quantization_ranges.py` covers DeepCAD+SkexGen ranges; the token package
covers the PAD families, code layouts and the hnc 25-frame rotation codebook.
Volume is reported honestly per repo. **"Nothing here" is a result and most repos
below get it.** The few genuine residual gaps are small and are called out with
their harness-verified basis.

Licenses were read from each repo's actual LICENSE file. Verdicts: MIT/Apache =
vendorable-with-attribution; CC-BY-NC / GPL / research-only = facts-with-citation;
no-LICENSE = manifest-only, vendor nothing.

---

## DeepCAD-master (51 files)

LICENSE: **MIT**, (c) 2022 Rundi Wu (`LICENSE`). Vendorable with attribution.
WHAT IT IS: the reference DeepCAD autoencoder + its official evaluation scripts.
The single most-mined upstream in the harness.
READ: `evaluation/evaluate_ae_cd.py`, `evaluate_ae_acc.py`, `evaluate_gen_torch.py`,
`collect_gen_pc.py`, `run_eval_gen.sh` (full); `cadlib/macro.py`; dataset/ + eval/
listings; README download section.
SKIMMED-NOT-READ: model/ + trainer/ architecture (~20 files, hyperparameters only).

FINDINGS (ranked):
1. **DeepCAD generation-metric numeric constants** — `evaluation/evaluate_gen_torch.py`:
   JSD grid `resolution=28` (L90), unit-cube occupancy bound `epsilon=10e-4` /
   `bound=1+epsilon` (L109), `N_POINTS=2000`, `random.seed(1234)`; and
   `evaluate_ae_cd.py`: 10%-each-tail trimmed-mean CD (`valid_dists[int(n*0.1):-int(n*0.1)]`,
   L124), out-of-bound renormalise when `max(abs(pc))>2` (L66), default `--n_points 2000`.
   HARNESS: the *formulas* (coverage/MMD, JSD) are covered deterministically in
   `eval/bench/generative/brep_set_metrics.py` (`coverage_mmd`, `jsd`), but the
   DeepCAD-specific numeric constants (res=28, epsilon=10e-4, trimmed-mean, >2
   renorm) are NOT pinned there — verified by reading that module (it is
   representation-agnostic, no constants). Minor gap; stdlib-portable if wanted.
2. DeepCAD ACC_cmd / ACC_param with `TOLERANCE=3` and the EXT-last-2 / ARC-slot-3
   strict-equality carve-outs — **ALREADY COVERED**, exactly, in
   `eval/bench/sequence/autoencoder_accuracy.py` (docstring cites
   `evaluate_ae_acc.py`; `TOLERANCE=3`, `slot_hits` strict carve-outs, SOL/EOS
   exclusion). Do not re-mine.
FIXTURES: none. All geometry is behind `data.tar` download (README:40); only
committed binary is `teaser.png`.
ALREADY COVERED: `quantization_ranges.py` (macro.py constants),
`autoencoder_accuracy.py` (ACC metric), `brep_set_metrics.py` (gen formulas).
VERDICT: **already-covered** (bar the minor gen-metric-constants gap above).

---

## SkexGen-main (39 files)

LICENSE: **MIT**, (c) 2022 Xiang Xu (`LICENSE`). Vendorable with attribution.
WHAT IT IS: disentangled-codebook CAD sequence generator.
READ: `dataset.py` (head), `utils/invalid.py`, `utils/eval_cad.py`, `utils/normalize.py`,
`utils/parse.py`, `utils/geometry/curve.py`; grep-read `model/decoder.py`,
`utils/converter.py`, `utils/deduplicate.py`.
SKIMMED-NOT-READ: model/ transformer definitions.

FINDINGS (ranked):
1. **Second-stage mesh normalisation constants** — `utils/normalize.py:40-44,169`:
   `NormalizeSE(cube_size=5.0, norm_factor=0.98, extrude_size=1.0, sketch_size=1.0)`
   (`MR=5.0, F=0.98, ER=1.0, SR=1.0`); sketch bbox rescaled to `F*SR`; SIGALRM
   60-second per-shape timeout (L178). HARNESS: not found (grep for `cube_size`/`0.98`/
   `NormalizeSE` hit only unrelated modules). This is a distinct normalisation stage
   from the token *ranges* already in `quantization_ranges.py`. Minor gap.
2. **OpenCASCADE converter error taxonomy** — `utils/converter.py`: named exception
   strings ("face builder not done", "wire builder not done", "face check failed",
   "unknown curve type", "non-zero z", "start/end point same location",
   "extrude refers to different sketch plane") + `BRepCheck_Analyzer(...).IsValid()`.
   HARNESS: not found (grep empty). NOT stdlib (OCC-dependent), so low value for a
   stdlib-portable verifier — record as facts-with-citation only.
3. PAD family (`PIX_PAD=4, CMD_PAD=3, COORD_PAD=4, EXT_PAD=1, EXTRA_PAD=1, R_PAD=2`),
   `0=End-of-SE` convention, and vocab-size formulas — **ALREADY COVERED** verbatim in
   `domain/reconstruction/tokens/skexgen_quantize.py` (L54-59 + `PIX_OFFSET`, offset
   formulas). Do not re-mine.
4. Invalidity-rate metric (`100*len(invalid)/len(dataset)` via CADparser try/except),
   eval_cad COV/MMD/JSD (res=28, epsilon=10e-4) — same as DeepCAD; formulas covered
   in `brep_set_metrics.py`.
FIXTURES: none committed (data downloaded).
ALREADY COVERED: `tokens/skexgen_quantize.py`, `tokens/skexgen_code_layout.py`,
`tokens/skexgen_extrude.py`, `quantization_ranges.py` (SkexGen ranges + truncation).
VERDICT: **already-covered** (bar NormalizeSE constants + OCC error taxonomy).

---

## GenCAD-main (125 files)

LICENSE: **NONE** — no LICENSE/COPYING file, no readme mention (verified). Treat as
all-rights-reserved -> **manifest-only, vendor nothing**.
WHAT IT IS: image-conditioned CAD generation; `cadlib/` is a DeepCAD fork.
READ: `cadlib/macro.py` (full); `diff` of all `cadlib/*.py` vs DeepCAD;
`data/filtered_data.json` + `image_ids.json` (heads); `config/configAE.py` (grep);
model/ + ckpt/ listings.

FINDINGS:
1. `cadlib/macro.py, curves.py, extrude.py, sketch.py, math_utils.py` are **byte-identical**
   to DeepCAD (empty diff). Only `cadlib/visualize.py` differs, and only by a debug
   STL-dump path. No new constant or vocabulary. HARNESS: DeepCAD cadlib already in
   `quantization_ranges.py` — so nothing new, and the no-license status forbids
   vendoring regardless.
2. Committed fixtures exist (unlike DeepCAD): `data/filtered_data.json` (4.2 MB
   train/val/test id split), `data/image_ids.json` (14 MB id->view-index map),
   `data/test_images/*.png` (input + generated render pairs). Real in-repo files, but
   they are ID lists + rendered PNGs, not geometry, and the **no-license** status
   means manifest-only anyway.
ALREADY COVERED: DeepCAD cadlib via `quantization_ranges.py`.
VERDICT: **already-covered / manifest-only** (verbatim DeepCAD fork, unlicensed).

---

## cadrille-master (11 files)

LICENSE: **Apache-2.0**. Vendorable with attribution.
WHAT IT IS: multimodal (text/pc/img) CAD-code generator (Qwen2-VL backbone) with a
small self-contained evaluate.py.
READ: `evaluate.py`, `test.py`, `data/README.md`, `data/cadrecode2mesh.py` (full).

FINDINGS:
1. **evaluate.py metric recipe** — CD via `cKDTree` k=1 both directions, returns
   *squared symmetric sum* `mean(sq(gt_d)) + mean(sq(pred_d))` (L31-36),
   `--n-points 8192`; volumetric IoU over `mesh.split()` components
   `inter/(vg+vp-inter)` (L39-54); normalisation center->unit-extent->`[0.5,0.5,0.5]`
   (L100-105); invalidity ratio `ir=(ir_cd+i)/len*100` reported after dropping the i
   worst (L175); median-CD reported x1000; each predicted `.py` `exec()`'d in a
   separate Process with a **3-second timeout** (L77-88). HARNESS: CD/IoU formulas
   are covered abstractly in `brep_set_metrics.py` + shape_metrics; the specific
   cadrille constants (8192 pts, squared-sum CD, x1000 median, 3s exec timeout,
   best-of-N argmin/max) are not pinned. Low value (one more numeric variant of a
   covered metric family).
FIXTURES: none. `data/` holds only a conversion script + README documenting
HuggingFace/git-lfs downloads; no `.stl`/`.py` pairs checked in.
VERDICT: **nothing-here** (metric formulas already covered; only numeric variants).

---

## hnc-cad-main (69 files)

LICENSE: **GPL-3.0, research-only, NON-COMMERCIAL** — `LICENSE` forbids commercial
use and applies GPL v3 (`LICENSE_GPL`) for research. Copyleft + non-commercial ->
**facts-with-citation only, vendor nothing.**
WHAT IT IS: hierarchical neural coding (VQ codebooks) for controllable CAD gen.
READ: `codebook/config.py`, `codebook/dataset.py`, `data_process/utils.py`.

FINDINGS:
1. `quantize(data, n_bits=8, ...)` is the **same scale-clip-truncate codec as SkexGen**
   (`2**n_bits-1`, clip, `.astype('int32')`); `BIT=6`; ranges `[-1,1]`
   (`SKETCH_R=EXTRUDE_R=CUBOID_RANGE=BBOX_RANGE=1`). Behaviour identical to
   `quantization_ranges.skexgen_quantize` — **ALREADY COVERED**.
2. Command token ids (loop-style line/arc/circle = 1/2/3; CAD-style = 4/5/6;
   topology-end tokens 1/2/3; pad `-1`), set-op ids (add/cut/intersect 1/2/3 or
   0/1/2), max lengths (`MAX_SOLID=5, MAX_PROFILE=20, MAX_LOOP=60`), codebook dims
   (`SOLID=10000, PROFILE=LOOP=5000`), `EOS=(-2,-2)`, `LOOP_PARAM_PAD=2`, mask ratios
   0.3-0.7, and the **25-entry ROT rotation table** — **ALREADY COVERED** by the
   dedicated `domain/reconstruction/tokens/hnc_codebooks.py`, `hnc_vector_codec.py`,
   `hnc_spl_tree.py` and `hnc_rotation_codebook.py` (the last cites the exact 25-frame
   ROT table and the `.all(axis=1).sum()==1` exact-match assert). Verified by reading
   `hnc_rotation_codebook.py`.
FIXTURES: none (only a 3 MB UID split list, not geometry).
ALREADY COVERED: `tokens/hnc_*` (four modules), `quantization_ranges.py`.
VERDICT: **already-covered** (and GPL/non-commercial forbids vendoring anyway).

---

## vitruvion-main (181 files)

LICENSE: **NONE for vitruvion's own code** — no top-level LICENSE, no README mention;
only the vendored `sketchgraphs/onshape/LICENSE` is MIT (PTC). vitruvion's own code
is all-rights-reserved -> **facts-with-citation only.**
WHAT IT IS: image->sketch primitive/constraint transformer over SketchGraphs.
READ: `img2cad/data_utils.py`, `primitives_data.py`, `dataset.py`, `constraint_data.py`,
`conf/model/transformer.yaml`; tests/ listing.

FINDINGS:
1. **Quantization + token vocab** — `data_utils.py:381-409`: `MIN_VAL=-0.5, MAX_VAL=0.5`,
   default `num_position_bins=64`, `max_token_length=130`; entity param counts
   `Arc:6, Circle:3, Line:4, Point:2`; primitive `Token` IntEnum
   `Pad0 Start1 Stop2 Arc3 Circle4 Line5 Point6` (coords offset by 7); constraint
   `Token` IntEnum `Pad0 Start1 Stop2 Coincident3 Concentric4 Equal5 Fix6 Horizontal7
   Midpoint8 Normal9 Offset10 Parallel11 Perpendicular12 Quadrant13 Tangent14
   Vertical15`; image `input_size=128`, primitive-noise std 0.15, augmentation
   `shift=12/128, rot=8, shear=8, scale=0.2`. HARNESS: the vitruvion **data pipeline**
   is already modelled — `data/dataengine/curation/sketch_filters.py` (Vitruvion
   FilterConfiguration + token-dedup) and `data/datagen/primitive_noise.py` (Vitruvion
   truncated-normal noise). The specific bin/token-enum constants above are not a
   distinct module, but they are (a) essentially SketchGraphs primitives + a
   `[-0.5,0.5]/64` bin scheme, and (b) **unlicensed** so facts-only. Low value.
FIXTURES: `img2cad/tests/testdata/images/` = 29 tiny committed PNG sketch renderings
(~0.3-1.4 KB). No committed `.npy`/`.json` sequence geometry (download-only). Marginal
and unlicensed.
ALREADY COVERED: `curation/sketch_filters.py`, `datagen/primitive_noise.py`.
VERDICT: **already-covered / facts-only** (unlicensed; pipeline already mined).

---

## SketchGraphs-master (90 files)

LICENSE: **MIT**, (c) 2020 Seff/Ovadia/Zhou/Adams (`LICENSE`); onshape sub-client also
MIT (PTC). Vendorable with attribution.
WHAT IT IS: the canonical Onshape sketch dataset library — the upstream schema behind
vitruvion, SketchConcept and the harness's own onshape ingest.
READ: `sketchgraphs/data/_constraint.py`, `_entity.py`, `sequence.py`,
`constraint_checks.py`, `pipeline/graph_model/quantization.py`,
`pipeline/numerical_parameters.py`, `dof.py`, `tests/` + `tests/conftest.py` (full).

FINDINGS (ranked):
1. **Full Onshape `ConstraintType` IntEnum (0-29 + Subnode=101)** — `_constraint.py:12-44`:
   Coincident0 Projected1 Mirror2 Distance3 Horizontal4 Parallel5 Vertical6 Tangent7
   Length8 Perpendicular9 Midpoint10 Equal11 Diameter12 Offset13 Radius14 Concentric15
   Fix16 Angle17 Circular_Pattern18 Pierce19 Linear_Pattern20 Centerline_Dimension21
   Intersected22 Silhoutted23 Quadrant24 Normal25 Minor_Diameter26 Major_Diameter27
   Rho28 Unknown29. Plus supporting enums DirectionValue/HalfSpaceValue/AlignmentValue/
   ConstraintParameterType. HARNESS: **PARTIAL gap.** `io/formats/onshape_json.py` has
   the `EntityType` (0-9) + `SubnodeType` (101-103) enums **already** (verified — Point0
   ..Unknown9, SN_Start101..SN_Center103, matches `_entity.py` exactly), but it does NOT
   define `ConstraintType`. `domain/reconstruction/sketch/constraint_taxonomy.py` models
   ~17 constraints by *name* with DOF-removed + SketchGraphs frequency-% (coincident
   42.17%, projected 9.71%, ...) but has no integer ids and deliberately omits the ~12
   rarely-modelled types (Circular_Pattern, Pierce, Linear_Pattern, Centerline_Dimension,
   Intersected, Silhoutted, Quadrant, Minor/Major_Diameter, Rho, Normal). So the
   *integer-id-keyed full 30-type table* is genuinely absent. MIT -> vendorable.
   Medium-low value (the harness intentionally models only the frequent set).
2. `constraint_checks.py` — numpy geometric constraint *checker* (is a horizontal/
   parallel/perpendicular/coincident/tangent/concentric/... constraint satisfied).
   HARNESS: **ALREADY COVERED and superseded** by `domain/geometry/sketch/constraint_satisfaction.py`,
   which is stdlib-only (no numpy) and covers a strict superset (adds length, distance,
   diameter, radius, angle, minor/major-radius, fix, midpoint, mirror). Do not re-mine.
3. `dof.py` NODE_DOF / EDGE_DOF_REMOVED heuristic tables — already referenced by the
   harness (see 05 report; `reconstruction/sketch/dof_mask.py` imports the onshape
   EntityType). Covered.
4. Quantization is **data-driven** (`quantization.py`: KMeans / uniform / cdf schemes
   fitted from statistics with a `num_points` arg), NOT fixed ranges — so there are no
   DeepCAD-style constants to mine here.
FIXTURES: `tests/testdata/sample_json.tar.xz` (763 KB real Onshape sketch JSON, loaded
by `conftest.py` via `Sketch.from_fs_json`) + `sg_t16.stats.pkl.gz` (109 KB pickled
stats). Genuine committed geometry, MIT-licensed — but a compressed tar of raw Onshape
JSON, not directly harness-shaped; usable as a provenance-clean sketch corpus if a
loader is written. Medium value.
ALREADY COVERED: EntityType/SubnodeType (`io/formats/onshape_json.py`),
constraint checking (`sketch/constraint_satisfaction.py`), DOF (`sketch/dof_mask.py`),
frequency/DOF taxonomy (`reconstruction/sketch/constraint_taxonomy.py`).
VERDICT: **mine-further (narrow)** — the integer-keyed full 30-type ConstraintType
enum + the MIT `sample_json.tar.xz` fixture are the only clean, uncovered items.

---

## SketchConcept-main (20 files)

LICENSE: **NONE** — no LICENSE file anywhere, no README mention. All-rights-reserved
-> **manifest-only, vendor nothing.**
WHAT IT IS: learned "concept library" abstraction over SketchGraphs sequences.
READ: `data.py`, `make_graph.py`, `experiment_directory/sample_exp/specs.json`.

FINDINGS:
1. A **remapped unified primitive+constraint vocabulary** (Line0 Point1 Circle2 Arc3;
   Coincident4..Normal18; ids 19/20/21 = start/new/end) and a quantization scheme
   `length_quantization=20 (range [0,2])`, `angle_quantization=30 (range [0,2pi])`,
   `coordinate_quantization=80 (range [-1,1])`, plus a novel `num_library=1000`
   concept-library size and a per-type argument grammar (construction_flag/coordinate/
   length/angle/pointer). HARNESS: the quantization scheme + library size are not
   modelled, but the repo is **unlicensed**, so this is facts-with-citation only, and
   the vocabulary is a re-id of SketchGraphs (already covered structurally). Low value.
FIXTURES: none — `Data/` holds only a 0-byte placeholder `.txt`.
VERDICT: **nothing-here / manifest-only** (unlicensed; vocab is a SketchGraphs re-id).

---

## UV-Net-main (22 files)

LICENSE: **MIT**, (c) 2021 Autodesk (`LICENSE`) — permissive, NOT the assumed
research license. Vendorable with attribution.
WHAT IT IS: B-rep face/edge UV-grid CNN encoder.
READ: `process/solid_to_graph.py`, `uvnet/encoders.py` (full).

FINDINGS:
1. UV-grid sampling schema: defaults `curv_u_samples=surf_u_samples=surf_v_samples=10`
   (10 edge-1D, 10x10 face-2D); face channels = 7 (`3 point + 3 normal + 1 trimming
   mask`), edge channels = 6 (`3 point + 3 tangent`); trimming mask =
   `visibility_status in {0 inside, 2 boundary}`. HARNESS: **ALREADY COVERED** — the
   `uxvx7` face grid + `ux6` edge grid schema is documented in
   `domain/reconstruction/brep/uvnet_face_adjacency.py` (verified: "num_u x num_v x 7"
   face feature, "num_u x 6" edge feature) with backing `geometry.uvnet_uv_grid` /
   `uvnet_u_grid` modules. Only the literal default sample-count `10` is not pinned as
   a constant. Negligible gap.
FIXTURES: none (only 3 doc PNGs).
ALREADY COVERED: `reconstruction/brep/uvnet_face_adjacency.py` (+ uvnet_uv_grid/u_grid).
VERDICT: **already-covered**.

---

## UVStyle-Net-master (64 files)

LICENSE: **CC BY-NC-SA 4.0** — non-commercial/research-only. Facts-with-citation only.
WHAT IT IS: B-rep style-similarity network (built on UV-Net).
READ: `chamfer.py`, `chamfer_distance/chamfer_distance.py` (full).

FINDINGS:
1. `chamfer.py` — symmetric squared-distance Chamfer via the `||x||^2+||y||^2-2xy^T`
   expansion, `0.5*(min-over-1 + min-over-2)`, uses only first 3 coords, no threshold
   (a training loss). Pure PyTorch/CPU-portable but **not** numpy/stdlib. The
   `chamfer_distance/` variant JIT-compiles a C++/CUDA extension (not portable).
   HARNESS: Chamfer is already provided stdlib-only in
   `agents/generation/shape_metrics.py` (`chamfer` = bidirectional mean NN). Nothing
   portable to add, and the license bars vendoring.
FIXTURES: none (download-only).
VERDICT: **nothing-here** (Chamfer already covered stdlib-only; non-commercial).

---

## SketchGraphs (see above) / JoinABLe-main (46 files)

LICENSE: **MIT**, (c) 2022 Autodesk (`LICENSE`) — permissive. Vendorable with attribution.
WHAT IT IS: B-rep joint (assembly-mate) axis prediction from two parts.
READ: `datasets/joint_graph_dataset.py`, `joint/joint_axis.py`, `utils/metrics.py`,
`args/args_search.py` (full); `geometry/brep.py`, README (grep).

FINDINGS (ranked):
1. **16-entry B-rep entity-type enum** (`joint_graph_dataset.py:28-45`): surfaces 0-7
   = Plane, Cylinder, Cone, Sphere, Torus, **EllipticalCylinder, EllipticalCone, Nurbs**;
   curves 8-15 = Line3D, Arc3D, Circle3D, Ellipse3D, **EllipticalArc3D, InfiniteLine3D,
   NurbsCurve3D, Degenerate3D**. HARNESS: **PARTIAL gap.** `reconstruction/brep/brep_clip_tokens.py`
   has `SURFACE_VOCAB=(plane,cylinder,cone,sphere,torus,nurbs)` (6) and
   `CURVE_VOCAB=(line,circle,arc,ellipse,bspline)` (5) — coarser. JoinABLe's extra
   granularity (elliptical cylinder/cone, elliptical-arc, infinite-line, degenerate) is
   NOT represented as an enum. MIT -> vendorable. Medium-low value.
2. **Edge convexity taxonomy** (`convexity_type_map`, L65-72): None0 Convex1 Concave2
   Smooth3 Non-manifold4 Degenerate5; and **joint label_map** (L74-82): Non-joint0
   Joint1 Ambiguous2 JointEquivalent3 AmbiguousEquivalent4 Hole5 HoleEquivalent6.
   HARNESS: `domain/geometry/topology/edge_convexity.py` computes convexity *geometrically*
   (sign of dihedral) but has **no** discrete 6-category id table (verified — it bins
   into a histogram by computed sign, not the None/Convex/Concave/Smooth/Non-manifold/
   Degenerate enum). Small genuine gap; the convexity enum is a useful verifier
   vocabulary. MIT -> vendorable.
3. **Axis-hit thresholds + eval** (`joint_axis.py:39-68`, `utils/metrics.py`): a joint-axis
   prediction is a "hit" iff angle `<10.0deg` AND distance `<1e-2`; precision@k over
   k-sweep `[1,2,3,4,5,10,20,30,40,50,60,70,80,90,100]`; reference accuracy 79.53% vs
   human 80% (README); UV-grid encoding `grid_size=10, grid_channels=7, grid_total=700`,
   area/length rel_tol `0.00015`. HARNESS: the axis-hit angle/distance thresholds and
   k-sweep are not pinned (grep of `joint_axis`/`axis_hit` found kinematics modules, not
   this eval). Low-medium value; concrete measurable thresholds, MIT.
FIXTURES: none (5 `.ckpt` pretrained weights only; dataset downloaded).
ALREADY COVERED: coarse surface/curve vocab (`brep_clip_tokens.py`), geometric convexity
(`topology/edge_convexity.py`).
VERDICT: **mine-further (narrow)** — the 16-type entity enum, the convexity/label id
tables, and the 10deg/1e-2 axis-hit thresholds are clean MIT items the harness lacks.

---

## SymPoint-main (74 files)

LICENSE: **Non-commercial research license** (IDEA, `LICENSE.txt`). Facts-with-citation only.
WHAT IT IS: SVG floor-plan panoptic symbol spotting (FloorPlanCAD).
READ: `parse_svg.py`, `svgnet/data/svg.py`, `svgnet/evaluation/point_wise_eval.py`.

FINDINGS:
1. **SVG serialization schema + primitive vocab** (`parse_svg.py`): `LABEL_NUM=35`;
   `COMMANDS=['Line','Arc','circle','ellipse']` (cmd ids 0-3); each primitive ->4 sample
   points (path at t=0,1/3,2/3,1; circle/ellipse at 4 quadrant angles); circle length
   `2*pi*r`, ellipse `2*pi*b + 4*(a-b)`; output JSON keys `commands,args,lengths,
   semanticIds,instanceIds,width,height,boxes,rgb,layerIds,widths`. HARNESS: the SVG
   primitive vocab is partially present (`domain/drawings/*`, `reconstruction/ortho/svg.py`)
   but this exact 4-command / 4-sample-point serialization is not a distinct module.
   Non-commercial -> facts-only. Low value.
2. `SVG_CATEGORIES` — 35-class FloorPlanCAD symbol taxonomy (doors 1-6, windows 7-10,
   furniture 11-27, stairs 28, equipment 29-30, stuff 31-35) with id/name/isthing/color.
   HARNESS: the *class-name table* is NOT present (verified — `domain/spec/intent_categories.py`
   has no door/window symbol classes). But it is a non-commercial dataset taxonomy;
   record as facts-with-citation. Low value.
3. Panoptic eval (`PointWiseEval(num_classes=35, ignore_label=35)`, mIoU + PQ, log(1+length)
   point weighting) — **ALREADY COVERED** in `eval/bench/vision/point_weighted_panoptic.py`
   (verified: docstring cites SymPoint ECCV 2024, `num_classes=35`, `DEFAULT_IGNORE_LABEL=35`,
   `round(log(1+length))` weighting) and `length_weighted_panoptic.py` /
   `instance_segmentation.py`. Do not re-mine.
FIXTURES: none (SVG data via download_data.py).
ALREADY COVERED: SymPoint panoptic protocol (`eval/bench/vision/*panoptic*`).
VERDICT: **already-covered** (eval protocol mined; class-name table is non-commercial
facts-only, low value).

---

## CADTransformer-main (24 files)

LICENSE: **MIT**, (c) 2023 VITA (`LICENSE`). Vendorable with attribution.
WHAT IT IS: FloorPlanCAD panel/symbol recognition from SVG (graph + image).
READ: `config/anno_config.py`, `preprocess/preprocess_svg.py`, `dataset.py`.

FINDINGS:
1. **FloorPlanCAD 35-class annotation config** (`config/anno_config.py`): `anno_list_all`
   (35 name->id, single door=1 .. railing=35), `anno_list_noBG` (1-30), `anno_list_door_wind`
   (7-class), `super_class_dict` (door{1-6}, window{7-10}, stairs{28}, home_appliance
   {18,20,24}, furniture{...}, equipment{29,30}, countable{1-30}, uncountable{31-35}),
   `RemapDict.mapping` (36-entry remap), `bandwidth_dict` mean-shift tiers
   (`super_tiny=10, tiny=15, small=20, middle=30, large=50, super_large=80`), `color_pallete`.
   Plus `dataset.py` constants `img_size=700, filter_num=64, max_prim=12000`, sentinel
   `[-999,-999]`. HARNESS: same conclusion as SymPoint — the FloorPlanCAD panoptic *eval*
   is covered (`eval/bench/vision/*`), but the class-name/id table + mean-shift bandwidth
   tiers are not. This one is **MIT** (so the class table IS vendorable, unlike SymPoint's
   copy), but it is a domain-specific 2D-drawing symbol taxonomy of low relevance to the
   3D text-to-CAD verifier core. Low value.
2. `preprocess_svg.py::square_distance` — stdlib-ish pairwise-distance helper; trivial,
   already available (`shape_metrics`, parametric closest_point). Not a finding.
FIXTURES: none (FloorPlanCAD downloaded).
VERDICT: **nothing-here** (eval covered; class taxonomy is MIT-vendorable but
low-relevance 2D-symbol domain).

---

## faceformer-master (46 files)

LICENSE: **MIT**, (c) 2019 Manycore Tech. WHAT IT IS: speech-driven facial-animation
transformer (VOCASET). READ: dataset/ + assets/ listings; `dataset/dataset_gen_logs/
filtered_id_list.json`.
FINDINGS: **none.** No CAD, no committed geometry (only a teaser gif + a speaker-id list).
VERDICT: **nothing-here** (not CAD).

---

## MeshTransformer-main (79 files)

LICENSE: **MIT**, (Microsoft). WHAT IT IS: METRO human/hand mesh recovery from images.
READ: `metro/` listing; `samples/` listing.
FINDINGS: **none for CAD.** Committed assets are JPEG photos (FreiHAND/3DPW); metrics are
human-pose (PA-MPJPE/MPJPE) with no checked-in reference values and no CAD relevance.
VERDICT: **nothing-here**.

---

## mesh-transformer-jax-master (58 files)

LICENSE: **Apache-2.0**. WHAT IT IS: GPT-J-6B language model in JAX (named "mesh" for the
TPU mesh, NOT geometry). READ: `data/*.index` listing, README.
FINDINGS: **none.** `data/*.index` are newline lists of tfrecord filenames; tokenizer is
standard GPT-2 BPE. No geometry, no CAD.
VERDICT: **nothing-here** (misleadingly named; a plain LLM repo).

---

## PartCrafter-main (98 files)

LICENSE: **MIT**, (c) 2025 Yuchen Lin. WHAT IT IS: part-level 3D generation (TripoSG-based).
READ: `datasets/object_part_configs.json`, `example_configs.json`, `configs/mp16_nt1024.yaml`.
FINDINGS:
1. Part-schema + mesh-eval constants — `configs/mp16_nt1024.yaml`: `min_num_parts=1,
   max_num_parts=16, max_iou_mean=0.2, max_iou_max=0.2`, val metric block
   `cd_num_samples=204800, cd_metric="l2", f1_score_threshold=0.1, default_cd=1e6,
   default_f1=0.0`; `example_configs.json` = real Objaverse GLB ids with part counts 1-7
   + IoU values (`object_part_configs.json` is placeholder-filled, not a usable fixture).
   HARNESS: CD/F1/IoU metric families already in `brep_set_metrics.py`/`shape_metrics`;
   these are just another set of numeric thresholds for mesh-part generation, not a CAD
   command/schema. Low value; part-count bounds are project-specific.
VERDICT: **nothing-here** (metric-constant variant; no CAD schema or usable fixture).

---

## BlendCLIP-main (82 files)

LICENSE: **MIT** (+ BSD-3 for vendored Salesforce/LAVIS). WHAT IT IS: CLIP for 3D point
clouds. READ: `data/labels.json`, `data/templates.json`.
FINDINGS: **none for CAD.** `labels.json` = generic object-category names (ModelNet40 etc.),
`templates.json` = CLIP prompt strings ("a point cloud model of {}."). Not CAD vocab, not
geometry.
VERDICT: **nothing-here**.

---

## Text2CAD-main (99 files)

LICENSE: **CC BY-NC-SA 4.0** (Text2CAD Community License) — non-commercial/research-only.
Facts-with-citation only; do NOT vendor code or the STEP fixtures for commercial use.
WHAT IT IS: text->CAD-sequence VLM; the harness's primary text-to-CAD upstream.
READ: `CadSeqProc/utility/macro.py`, `Evaluation/eval_seq.py`, `prompt.json`,
`Sample_Prompts_Images_Steps/README.md` + steps/ listing.

FINDINGS (ranked):
1. **Committed STEP fixtures** — `Sample_Prompts_Images_Steps/steps/` = **10 real STEP
   files** (baseplate, box, clamp_ring, flange, Handwheel, manifold, pulley, shaft,
   topplate, U_profil; verified ISO-10303-21 AP214, 9-42 KB) + 8 PNG renders. Real
   committed B-rep geometry with prompt->part mapping. HARNESS: genuine fixtures, but
   **CC-BY-NC-SA** — usable as citation/manifest reference, not vendorable into a
   permissive corpus. Medium value as a provenance-noted eval set.
2. `prompt.json` (== `jobs_worked_de.json`) = **22 German dimensioned mechanical-part
   prompts** with explicit GD&T (e.g. "Rechteckplatte 120x80x8 mm, Bohrung d=6 mm,
   counterbore d=12 mm Tiefe 4 mm"). These read as harness-authored job fixtures, not
   upstream templates. Usable committed prompt fixtures (note: not a system-prompt —
   the VLM `prompt_file` is a placeholder).
3. `CadSeqProc/utility/macro.py` token scheme: `N_BIT=8` (256 bins), `MAX_CAD_SEQUENCE_LENGTH=272`,
   hierarchical END tokens (PADDING/START/END_SKETCH/END_FACE/END_LOOP/END_CURVE/
   END_EXTRUSION), `CAD_CLASS_INFO one_hot_size=267`, `NORM_FACTOR=0.75`. A superset
   re-serialization of DeepCAD. HARNESS: **ALREADY COVERED** — `domain/reconstruction/
   tokens/text2cad_tokens.py` + `text2cad_vector_codec.py` reproduce the exact Table-3
   vocabulary (id map 0..266, `COORD_OFFSET=11`, curve parameterization). Verified by
   reading `text2cad_tokens.py`. Do not re-mine.
4. `Evaluation/eval_seq.py`: per-primitive F1/P/R (Line/Arc/Circle/Extrusion x100) over
   4 abstraction levels, CD median/mean over cd>0, invalidity `count(cd<0)*100/len`.
   HARNESS: sequence/token accuracy covered in `eval/bench/sequence/*`; per-primitive
   F1 over 4 difficulty levels is a Text2CAD-specific reporting layer, low incremental value.
ALREADY COVERED: `tokens/text2cad_tokens.py`, `tokens/text2cad_vector_codec.py`,
`eval/bench/sequence/*`.
VERDICT: **mine-further (narrow) / facts-only** — the 10 committed STEP fixtures + 22
dimensioned prompts are the real assets, but CC-BY-NC-SA restricts them to
provenance-noted / manifest use. Token scheme already fully covered.

---

## Text23D-master (86 files)

LICENSE: **MIT**, (c) 2026 Quanfei Zhang. WHAT IT IS: text->3D web app with CadQuery +
FreeCAD runners. READ: `cad-runner/run_cadquery.py`, `cad-runner/examples/cube_with_hole.py`,
`freecad-runner/examples/cube_with_hole.py`.
FINDINGS:
1. Generated-script *contract* templates: CadQuery script must expose `build_model()`,
   exports `model.step`+`preview.glb`, default assembly color `cq.Color(0.64,0.68,0.72,1.0)`;
   FreeCAD equivalent `build_model(doc)` returning object list. Example templates
   (`box().faces(">Z").workplane().hole(d)`, `Part::Box`+`Part::Cut`). HARNESS: the
   harness already carries far richer build123d/CadQuery skillpacks and a `gen_step()`
   contract (`data/skillpacks/text_to_cad.json`); these two tiny examples add nothing.
VERDICT: **nothing-here** (trivial example templates already exceeded by harness skillpacks).

---

## CanonicalVAE-main (101 files)

LICENSE: **MIT** (inherited from taming-transformers/VQGAN, (c) 2020 Esser/Rombach/Ommer).
WHAT IT IS: point-cloud VQ-VAE for ShapeNet (car/chair/plane) — NOT text-to-CAD.
READ: `assets/` listing; `src/canonicalvq/models/` listing; README.
FINDINGS: **none.** Committed assets are 15 ShapeNet reconstruction GIFs (not CAD geometry).
No CAD command vocab, no CAD metrics. Generic VQ/mingpt code.
VERDICT: **nothing-here** (point-cloud VQ-VAE, not CAD).

---

## neuralCAD-Edit-main (91 files)

LICENSE: **MIT**, (c) 2026 Autodesk (`LICENSE`). Vendorable with attribution.
WHAT IT IS: natural-language CAD-editing benchmark (LLM -> CadQuery `my_cad_function`).
READ: `src/harnesses/cadquery_script.py`, `src/utils/evals_feature_geometric.py`,
`src/scripts/benchmark_evals/edit.py`, `config/edit_192_external.json`,
`example_data/.../settings.json`.
FINDINGS (ranked):
1. **Geometric-eval implementations** (`evals_feature_geometric.py`): "chamfer similarity"
   = `1.0/(chamfer_dist+1e-8)` (symmetric mean-NN *Euclidean*, not squared; empty->100.0;
   10000 samples; optional CPD pre-align with 8 random rotations); voxel IoU
   `voxel_size=min(diag)/100`, index-set intersection/union, 120s subprocess timeout;
   plus DINOv2 + CLIP cosine similarity, and a VLM 1-7 grading rubric
   (`score-instruction-understanding`, `score-quality`, over 6-view images+video).
   HARNESS: Chamfer/IoU covered stdlib in `shape_metrics`; the `1/(cd+eps)` *similarity*
   framing, the voxel-IoU divisor=100 recipe, and the committed **1-7 VLM rubric** are
   not present. The rubric + difficulty split (easy/medium/hard keyed to
   `chamfer_similarity`) are reusable, MIT. Low-medium value.
2. Edit taxonomy: **no formal edit grammar/DSL** — edits are free-form NL producing a
   CadQuery function; only an easy/medium/hard difficulty split exists. HARNESS: the
   harness has richer edit schemas (`domain/editing/sketch_edit_schema.py`,
   `domain/editing/brep.py`). Nothing to add.
FIXTURES: one committed example (`example_data/.../brep_end/.../tmp.step`+`tmp.stl`+9
JPG views+`settings.json` recording token/cost counts) + many benchmark mp4/gif. One
STEP is a thin fixture; the rest is media.
VERDICT: **nothing-here (near-miss)** — the MIT 1-7 VLM edit-grading rubric + voxel-IoU
recipe are the only mildly novel items; edit taxonomy is free-form and already exceeded.

---

## Cross-repo summary

**Verified already-covered (do not re-mine):** DeepCAD ranges + ACC_cmd/ACC_param
(`quantization_ranges.py`, `autoencoder_accuracy.py`); SkexGen ranges + PAD family + code
layout (`quantization_ranges.py`, `tokens/skexgen_*`); hnc-cad codec + 25-frame ROT table
(`tokens/hnc_*`); Text2CAD Table-3 token vocab (`tokens/text2cad_*`); vitruvion data
pipeline (`curation/sketch_filters.py`, `datagen/primitive_noise.py`); SketchGraphs
EntityType/SubnodeType + DOF + constraint checking (`io/formats/onshape_json.py`,
`sketch/dof_mask.py`, `sketch/constraint_satisfaction.py`, `reconstruction/sketch/
constraint_taxonomy.py`); UV-Net 7ch/6ch grid schema (`reconstruction/brep/
uvnet_face_adjacency.py`); SymPoint/FloorPlanCAD panoptic eval (`eval/bench/vision/
*panoptic*`); COV/MMD/JSD set metrics (`eval/bench/generative/brep_set_metrics.py`).

**Genuine residual gaps, ranked by value (all narrow):**
1. SketchGraphs **full Onshape ConstraintType integer-id enum (0-29+101)** — MIT,
   vendorable; harness has only the ~17 frequent names + EntityType. (`_constraint.py:12-44`)
2. JoinABLe **16-type B-rep entity enum** + **convexity id table (None/Convex/Concave/
   Smooth/Non-manifold/Degenerate)** + **axis-hit thresholds (10deg / 1e-2)** — MIT,
   vendorable; harness surface/curve vocab is coarser and has no discrete convexity enum.
3. Text2CAD **10 committed STEP fixtures + 22 dimensioned prompts** — real geometry, but
   **CC-BY-NC-SA** so provenance-noted/manifest use only.
4. Minor numeric-constant gaps: DeepCAD gen-metric constants (JSD res=28, epsilon=10e-4,
   10%-trimmed CD, >2 renorm); SkexGen `NormalizeSE(5.0,0.98,1.0,1.0)`; neuralCAD-Edit
   1-7 VLM edit-grading rubric + voxel-IoU divisor=100 (MIT). SkexGen OCC converter error
   taxonomy (not stdlib).

**Nothing-here (verified):** cadrille, GenCAD (verbatim DeepCAD, unlicensed),
SketchConcept (unlicensed re-id), UVStyle-Net (non-commercial; Chamfer covered),
faceformer, MeshTransformer, mesh-transformer-jax, PartCrafter, BlendCLIP, Text23D,
CanonicalVAE, CADTransformer (eval covered; 2D-symbol taxonomy low-relevance).

**License ledger:** MIT — DeepCAD, SkexGen, UV-Net, JoinABLe, SketchGraphs, faceformer,
MeshTransformer, PartCrafter, BlendCLIP, Text23D, neuralCAD-Edit, CADTransformer,
CanonicalVAE. Apache-2.0 — cadrille, mesh-transformer-jax. CC-BY-NC-SA (non-commercial)
— Text2CAD, UVStyle-Net. GPL-3.0/non-commercial — hnc-cad. Research-only — SymPoint.
**No LICENSE (manifest-only)** — GenCAD, SketchConcept, vitruvion (own code; only the
vendored onshape sub-client is MIT).
