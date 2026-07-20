# Deep read — CADBench (Doris et al. 2026, DeCoDELab)

New repo added today: `resources/cad_repos/CADBench-main` (double-nested
`CADBench-main/CADBench-main/`). This is **not** any of the three "CADBench"
things the harness already names — see ALREADY COVERED. It is the multimodal
image/mesh -> CadQuery *reconstruction* benchmark from Anna Doris's lab
(arXiv:2605.10873, HF `DeCoDELab/CADBench`), the follow-on to CAD-Coder
(arXiv:2505.14646) which the harness has already mined.

---

## CADBench-main (4439 files; 2529 data files — 1554 json, 524 txt, 487 jsonl, 487 csv, 486 xlsx, 463 jpg, 113 png, 82 usdc, 82 mtlx, 117 py)

### LICENSE (code)
`LICENSE` = **MIT** (Copyright 2026 Annie Doris). Covers all Python in the repo:
the eval pipeline, `code_metrics.py`, the geometry metric implementations, and —
importantly — the **committed baseline metric JSONs** under `tested_models/`
(these are computed outputs of MIT code, not the underlying dataset).

### LICENSE (data)
`DATASET_LICENSE.md` = **split / mixed, and the trap is real**. The benchmark
task data is derived from five upstream datasets and each keeps its own licence:
- DeepCAD (benchB/benchE base) — MIT
- Fusion 360 Gallery — **non-commercial research licence** (restrictive)
- ABC (benchA/benchF) — MIT
- Objaverse (benchO) — **ODC-By v1.0 + per-object Creative Commons** (mixed/restrictive)
- MCB (benchM) — MIT

Crucially: **the task data itself is NOT in the repo.** The STEP/STL/mesh/point-
cloud/image modalities live only on HuggingFace and are pulled by
`scripts/download_from_hf.py`. So there is nothing to vendor from the data side
even where the licence would allow it, and the Fusion360/Objaverse portions
would be manifest-only regardless. Treat all *task* data as MANIFEST-ONLY
(record the HF dataset id + the download scripts; resolve at runtime).

### Verdict per class
- **Eval code / geometry-metric implementations** (MIT): reference-readable,
  but ALREADY COVERED (the core oracle is CAD-Coder's, which the harness ported).
- **Baseline leaderboard metric values** (`tested_models/**/*_metrics.json`,
  `*_per_label_metrics.json`, `results_summary.txt`) — MIT, computed numbers:
  **VENDORABLE** with attribution. This is the one genuinely new, small,
  low-risk haul.
- **Per-program execution-status + error-message corpus**
  (`tested_models/**/*_logs.json`) — MIT: **vendorable-but-bulky** (model-
  generated CadQuery, not curated); better as manifest + selective extraction.
- **Task data (prompts/STEP/STL/images/PBR)**: MANIFEST-ONLY (HF-hosted, mixed
  licence, not present in checkout).
- **CADVis / get_bench / scripts / generate_modalities / CADVis Materials**:
  NOT findings (rendering + dataset-construction + upload tooling).

### WHAT IT IS
A **reconstruction** benchmark: given a *rendering* (singleview / multiview /
PBR image) or a *mesh* (clean or noisy) or a point cloud of a ground-truth
solid, a model must emit a **CadQuery Python program** whose executed solid
matches the GT mesh. Six subsets (benchA=ABC-all-ops, benchB=DeepCAD-base,
benchE=DeepCAD-extrude, benchF=ABC-sketch-extrude-only, benchM=MCB, benchO=
Objaverse), **3000 tasks each (18000/modality-run)**, each split into
easy/medium/hard by geometric complexity (extrude/op counts, see
`graph/bench_complexities.json`). There is **no natural-language brief** — the
prompt is a *modality*, so this is a differential geometry oracle, not a
text-to-CAD brief corpus.

Grading (`CADBenchEval/CADBench/Eval/_main.py::perform_evaluation`): execute the
program -> export STL -> `alpha_wrap` both meshes to watertight shells (CGAL via
pymeshlab) -> bbox-normalise -> compute **Aligned IoU / Aligned Chamfer /
Aligned Surface-IoU** (mass-property principal-axis registration, 4 sign
combos) and the **Naive** (bbox-only) variants; aggregate **VSR** (valid solid
rate), timeout rate, and code metrics (token/line/op counts). Status codes:
0=execution error, 1=success, 2=timeout.

### READ (fully)
- `LICENSE`, `DATASET_LICENSE.md`, `README.md` (all)
- `CADBenchEval/CADBench/Eval/_main.py` (grading protocol + aggregation)
- `CADBenchEval/CADBench/Eval/code_metrics.py` (AST CadQuery op vocabulary — full)
- `CADBenchEval/CADBench/Eval/geometry/_utils.py` (align_shapes, iou, chamfer,
  surface_iou, alpha_wrap, ICP, mass properties — full)
- `CADBenchEval/CADBench/Eval/execution/_utils.py` (sandboxed exec + STL export)
- `CADBenchEval/CADBench/Eval/{geometry,execution}/__init__.py`
- Data files inspected end-to-end: `benchA_metrics.json`,
  `benchA_per_label_metrics.json`, `benchA_logs.json`, `benchA_collected_metrics.csv`,
  `results_summary.txt`, one `benchA.jsonl` row, `graph/bench_complexities.json`,
  one `tested_vlms/.../easy.json`.
- Harness cross-check files: `domain/geometry/transforms/principal_axes.py`,
  `eval/corpus/fixtures/cad_coder_heldout.py` + `cad_coder/MANIFEST.json`,
  `eval/bench/imports/graphcad_cadbench.py`, `eval/bench/data/task.py`,
  `eval/bench/data/complexity_entropy.py`.

### SKIMMED — NOT READ (honest volume)
Read ~12 code files + ~9 data files in full; header/tree-characterised the rest.
NOT read: `CADVis/` (733 files — PBR renderer + MaterialX/USD material assets:
82 usdc, 82 mtlx, hundreds of jpg textures); `get_bench/` (dataset-construction
preprocessing, incl. bundled `cadlib/`); `scripts/` (25 download/upload/gen
utilities); `generate_modalities/` (image/mesh renderers, `GVis/`,
`visualize_stl/`); `CADBenchEval/CADBench/{Inference,Processing}/` (API wrappers
+ mesh/pc/render utils); `docs/` (project-page HTML + screenshots); the **486
`*.xlsx`**, **463 `*.jpg`**, and the bulk of the **487 `*.jsonl` / 487 `*.csv`**
(sampled 3-4 of each — they are per-model/per-bench repeats of the same schema). The
actual benchmark *task* data (STEP/STL/images) is **absent from the checkout**
(HF-only), so it could not be read at all.

---

### FINDINGS (ranked)

**1. Baseline leaderboard reference values — 13 named models × 3 modalities ×
6 benches × per-difficulty. VENDORABLE (small JSONs, MIT).**
- Path: `tested_models/{model}/{modality}/r1/bench{A,B,E,F,M,O}_results/`
  `bench*_metrics.json` (486), `bench*_per_label_metrics.json` (486),
  and `tested_models/{model}/{modality}/results_summary.txt`.
- Models: `cadcoder, cadevolve, cadevolve_ourimages, cadfit, cadrecode,
  cadrille_ourimages, cadrille_pc, claude4.7, gemini3.1, gpt5.4, kimi_2.6,
  qwen3.527b, qwen3.59b`. VLMs carry singleview/multiview/pbr; reconstruction
  models carry mesh/pc.
- Each `_metrics.json` = committed {Mean, Median, Std, Adjusted*, VSR, Timeout
  Rate, Code Metrics} for Aligned/Naive IoU·Chamfer·Surface-IoU. Verified by
  reading `cadcoder/multiview/r1/benchA_results/benchA_metrics.json`
  (e.g. Aligned IoU mean 0.2428, VSR 86.13%, median token 712) and the
  per-label easy/medium/hard breakdown (easy Aligned-IoU mean 0.264, n=958/1000).
- Why we want it: verifier-first **calibration/regression baselines** for a
  multimodal image/mesh->CadQuery task. The harness has baselines for *other*
  benchmarks but **NONE for this one** (verified: grep of `src/harnesscad` for
  `doris|DeCoDELab|anniedoris|2605.10873|Aligned Surface IoU|alpha_wrap` returns
  only the CAD-Coder principal-axes module — the *2025* paper, not this 2026
  benchmark). HOW verified harness-lacks: `registry.index()` bench scan (no
  cadbench-reconstruction entry) + the grep above.
- Vendor recommendation: vendor the aggregate JSONs (a few hundred KB) as
  reference values; attribute Doris et al. 2026. Do NOT vendor the underlying
  task data.

**2. Per-program execution-status + error-message corpus. Vendorable-but-bulky
-> recommend manifest + selective extraction.**
- Path: `tested_models/{model}/{modality}/r1/bench*_results/bench*_logs.json`
  (486 files; each ~3000 rows). Row = {file_id, all six metrics, token/line/op
  counts, **status (0/1/2)**, **details (exact error string)**, label}.
- Example known-bad row (verified in `benchA_logs.json`): status 0, details
  `"Code Execution Error: '(' was never closed (<string>, line 18)"`. Mix of
  syntax errors, exec errors, timeouts, and low-IoU successes, tagged by
  difficulty.
- Why we want it: the harness "is SHORT on expected-rejection / known-bad data."
  This is **model-generated broken CadQuery at scale with the exact failure
  reason and difficulty label** — a differential substrate for an error
  taxonomy. Harness equivalent: partial — `eval/corpus/fixtures/cadgenbench_broken.py`,
  `adversarial_code.py`, and `eval/bench/sequence/error_taxonomy.py` already
  provide *curated* broken code; CADBench adds *scale + real model failure
  distribution* but is noisy (model outputs, not hand-verified). NONE is an
  overstatement here — call it a **complementary, not novel, corpus**.
- Recommendation: manifest the logs; if used, extract a de-duplicated
  error-string -> category map rather than vendoring 486×3000 rows.

**3. Complexity ground truth. Trivial, VENDORABLE (MIT), low value.**
- Path: `graph/bench_complexities.json` (also `docs/static/bench_complexities.json`).
- Committed per-bench/per-difficulty extrude-count and all-op-count thresholds
  (e.g. benchB extrude easy/medium/hard = 7/12/27; benchA all-op = 12/51/206).
- Why: documents the exact numeric complexity banding behind easy/medium/hard.
  Harness equivalent: `eval/bench/data/complexity_bands.py` +
  `difficulty_tiers.py` already do this for other corpora; this is one more
  small reference table. Marginal.

**Not findings (verified):** the geometry oracle (`align_shapes` + `iou` +
`surface_iou` + `alpha_wrap`) — see ALREADY COVERED; `code_metrics.py` (a
CadQuery-op AST counter — harness has `operation_coverage`/`operator_profile`);
CADVis PBR renderer + 82 usdc/mtlx material assets (rendering, not measurement);
`get_bench/`, `scripts/`, `generate_modalities/` (dataset construction + upload).

---

### ALREADY COVERED (name the module)
- **Core geometry oracle** — CADBench's `align_shapes` (mass-property principal-
  axis registration, isotropic inertia rescale, 4 eigenvector-sign proper
  rotations, keep-max-IoU) is the **same routine** the harness already ported
  from CAD-Coder (same author) into
  `src/harnesscad/domain/geometry/transforms/principal_axes.py` and
  `eval/bench/solid_iou`. The module docstring cites arXiv:2505.14646 and the
  4-sign enumeration explicitly. No new algorithm here.
- **CAD-Coder held-out pairs** — `eval/corpus/fixtures/cad_coder_heldout.py`
  (+ `cad_coder/MANIFEST.json`, Apache-2.0, manifest-only) already covers this
  lab's GT-STEP + reference-CadQuery fixtures.
- **The three OTHER "CADBench"s** the harness names are unrelated to this repo:
  (a) `eval/bench/imports/graphcad_cadbench.py` = *Graph-CAD*'s `CADBench.jsonl`
  (700 rubric text-to-CAD tasks); (b) `eval/bench/data/task.py` =
  "CADBench-Verified", an internal harness SWE-bench-for-CAD schema
  (docs/blueprint.md); (c) `eval/bench/data/complexity_entropy.py` = *BlenderLLM*'s
  CADBench (Du et al. 2024, bpy scripts). None is Doris et al. 2026.
- **Chamfer / surface-threshold recall** — `surface_iou` (fraction of sampled
  points within 0.02 chamfer) is a single point on the harness's
  `eval/bench/geometry/cd_tolerance_recall` curve; `chamfer` variants abound.
- **VSR / validity + code op-counts** — covered by existing validity gates and
  `operation_coverage` / `operator_profile`.

---

### VERDICT: **mine-further (narrow)**

Not "already-covered" outright, because the **baseline leaderboard reference
values (Finding 1)** are genuinely new to the harness, small, MIT-clean, and
directly useful as calibration baselines for a multimodal reconstruction task —
worth a manifest+vendor of the aggregate JSONs with attribution. Finding 2
(execution-error corpus) is a *complementary* scale substrate, manifest-only.
Everything else is already-covered (the geometry oracle, via CAD-Coder) or a
non-finding (renderer / dataset-construction / HF-hosted mixed-licence task
data). **Do not vendor any task data** (Fusion360 non-commercial + Objaverse CC,
and absent from the checkout anyway) — MANIFEST-ONLY, exactly the Graph-CAD /
CADPrompt pattern.
