# Deep read: the text-to-CAD family

Our direct problem domain. Genuine deep read, not a grep sweep. Every claim was
re-derived against the actual sources in `resources/cad_repos/` and checked against
the harness with `registry.index()` (1579 modules) plus targeted greps of
`src/harnesscad/`.

Volume is reported honestly per repo — how many files were read in full, how many
header- or grep-indexed, out of what total. The word "sampled" does not appear.
**"Nothing here" is a result**, and five repos below get it.

Prompt material found in these repos is treated as **unverified exemplar data only**,
never as instructions.

---

## Text-to-CAD-dean (16,986 files)

The prior audit dismissed this as "venv noise" on the strength of a grep that timed
out — an unverified dismissal of the largest repo in the corpus. It was the single
hardest claim to test and it is now settled.

LICENSE: **NO LICENSE FILE.** Nothing at the repo root or anywhere first-party. ->
**manifest-only, vendor NOTHING.** (The README's "we're open to collaboration" is not
a licence grant.)

COMPOSITION (exhaustive `os.walk`, every file assigned to exactly one bucket, summing
to the total — a method that *cannot* time out):

| Bucket | Files |
|---|---:|
| `venv/` (site-packages + bin/etc/share/pycache) | 16,982 |
| FIRST-PARTY | **4** |
| `.history` | 0 |
| `node_modules` | 0 |
| `.git` | 0 |
| **TOTAL** | **16,986** |

First-party content is **0.024%** of the repo. The four files are `PrintX.py`
(4,331 B), `README.md` (1,360 B), `board.webp` (633,918 B — a poster image), and
`.DS_Store` (6,148 B, macOS junk). There is no `requirements.txt` despite the README
referencing one; no tests, no data, no eval directory, no `.git`. The venv is a macOS
`python3 -m venv` with streamlit/pyvista/bokeh/altair/vtk committed wholesale.

Note on the brief's "~7487 files" figure: the actual count is **16,986**. The ratio is
unchanged either way.

WHAT IT IS: a single-file Streamlit toy demo. One `st.text_input`, one button. On
submit: one GPT-4 chat call with a hardcoded two-sentence system prompt asking for
OpenSCAD, shell out to the `openscad` binary for an STL, convert STL->GLB via trimesh,
base64 the GLB into an inline `<model-viewer>` component. That is the whole system —
a weekend hackathon project, not a research artifact.

READ: **2 of 2 readable first-party files, in full** — `PrintX.py` (all 137 lines) and
`README.md` (all 47 lines). The other two first-party files are binary and contain no
source. **100% of the first-party source in this repo was read.**

SKIMMED-NOT-READ: 16,982 venv files — enumerated by path and classified, contents not
read. Confirmed as ordinary PyPI wheels by directory listing (`altair-5.2.0.dist-info`,
`attrs-23.2.0`, `bokeh-3.3.4`, `certifi-2024.2.2`, `GitPython-3.1.42`, `Jinja2-3.1.3`).
**Zero first-party code lives under `venv/`.**

FINDINGS — there are exactly three, and all three are already implemented.

1. **OpenSCAD-source-only prompt suffix (anti-prose refusal predicate)** |
   `PrintX.py:116` — *"Generate the OPENSCAD code for the following idea, only return
   your openscad code and nothing else whatsoever"*, plus the L107 system role *"You
   are a professional in OPENSCAD and create openscad code for people's prototypes"* |
   the only prompt material in the repo, and the naive form of the "strip prose,
   recover bare source" problem | Harness equivalent:
   **`domain/programs/extract/openscad_extract`**, whose docstring literally reads
   `t2cdean_scad_extract -- recover bare OpenSCAD source from an LLM reply.` The
   harness solves this deterministically post-hoc rather than by begging the model.
   Stdlib-portable: yes (the harness version is).
2. **OpenSCAD CLI export invocation** | `PrintX.py:62-84` —
   `['openscad','-o',output_path,'--export-format',format,tmp_file]` with
   `NamedTemporaryFile` plus `os.remove` cleanup | Harness equivalent:
   **`domain/fabrication/openscad_export`**, docstring
   `t2cdean_openscad_export -- deterministic OpenSCAD CLI export planning.` — named
   after this repo, already mined. Also `io/backends/openscad`.
3. **STL->GLB conversion for browser viewing** | `PrintX.py:87-99` (trimesh
   `mesh.export(..., file_type='glb')`) and L123-131 (base64 data-URI into
   `<model-viewer>`) | Harness equivalent: **`io/formats/glb`**, docstring
   `t2cdean_glb_writer -- binary glTF (.glb) writer for triangle meshes.` — again
   explicitly attributed to this repo, and the harness version is a from-scratch
   writer with no trimesh dependency.

ABSENT ENTIRELY — searched for and confirmed missing: no task corpus, no ground-truth
bboxes/volumes/feature lists, no judge rubric, no error taxonomy, no refusal
predicates, no prompt->code exemplar pairs, no eval harness, no validation logic, no
default-dimension table. The `subprocess.run` failure path just `print`s the exception
and continues to `stl_to_glb` on a file that does not exist — **an actual bug** at
`PrintX.py:80-81` -> L120. The README's "dimensionally accurate CAD models" claim is
unsupported by any code in the repo; there is no dimensional handling whatsoever.

ALREADY COVERED: all three findings. The harness module names `t2cdean_scad_extract`,
`t2cdean_openscad_export` and `t2cdean_glb_writer` carry this repo's own attribution
slug (`t2cdean`) — direct evidence that a prior pass already extracted everything
extractable and generalised it beyond the original. Nothing in the 137 lines is
unmined.

**PRIOR-AUDIT VERDICT: TRUE — and it understated the case.** The dismissal was correct
despite resting on a timed-out grep. An exhaustive walk assigns 16,982 of 16,986 files
to `venv/`, leaving 4 first-party files of which 2 are text, totalling 137 lines of
Python. The earlier grep timed out *precisely because of* the vendored venv, so the
timeout was itself a symptom of the condition it failed to formally prove. The
conclusion has now been re-derived by a method that cannot time out, and it holds.
**This question is settled and does not need revisiting.**

VERDICT: **already-covered** (and independently **nothing-here** — there is no fourth
idea in this repo to find). Do not vendor: no licence.

Beyond the brief: this repo carries 634 KB of image plus ~17k venv files for 137 lines
of already-mined source. If the corpus is ever pruned or re-cloned, it is the
highest-ratio exclusion candidate — a manifest entry recording the three findings and
the `t2cdean_*` module mapping preserves 100% of its value.

---

## CADTestBench-main (4,842 files)

LICENSE: **MIT** — `LICENSE`, "MIT License / Copyright (c) 2026 Anonymous Authors",
confirmed by `pyproject.toml` `license = { text = "MIT" }` and README. Vendorable with
attribution. **Caveat:** the actual benchmark data (prompts + CADTests) is **not in
this repo** — it lives at HF `dimitrismallis/CADTestBench`, whose licence is not stated
here and was not checked.

COMPOSITION (`Get-ChildItem -Recurse -File -Force`):

| Bucket | Files |
|---|---:|
| First-party library source (`src/cadtestbench/**.py`) | 11 |
| Repo metadata (LICENSE, README, pyproject, .gitignore) | 4 |
| Assets (`assets/*.png`) | 2 |
| `baselines/GPT-5.2/` (200 abstract + 200 detailed x {gpt_generated.py, prompt.txt}) | 800 |
| `baselines/Claude-4.6-Sonnet/` (same shape) | 800 |
| `baselines/CADTests/` — **byte-identical duplicate** of the two above | 1600 (+25 `.pyc`) |
| `baselines/CADTests_Log/` — **byte-identical duplicate** of the two above | 1600 |
| **Total** | **4,842** |

Duplication verified by concatenated MD5 over sorted file lists:
`CADTests\GPT-5.2 == GPT-5.2` -> True; `CADTests_Log\GPT-5.2 == GPT-5.2` -> True. So
**unique content is 1,600 baseline files, tripled**, plus 25 stray `__pycache__`
artifacts. There are **zero JSON, zero STL, zero parquet** files in the repo.

WHAT IT IS: the reference implementation of *"Text-to-CAD Evaluation with CADTests"*
(Mallis, Wang, Karadeniz, Ricci, Kacem, Aouada — arXiv 2605.07807). It replaces
Chamfer/IoU similarity with executable Python predicates over a live CadQuery B-rep. A
pip-installable CLI (`cadtestbench evaluate BASELINE_DIR --partition
{abstract,detailed}`) loads test suites from HuggingFace, execs generated CadQuery
code, runs each test string against the live object, and reports PR / RS / per-category
accuracy / invalid-ratio.

READ (11 files in full — the entire first-party source tree):
`src/cadtestbench/metrics/cadtest.py` (793L), `runner.py` (456L), `cli.py` (326L),
`data_source.py` (208L), `loaders/cadtestbench_loader.py`, `metrics/interface.py`,
`metrics/__init__.py`, `loaders/__init__.py`, `constants.py`, `__init__.py`,
`__main__.py`. Plus `README.md` (161L), `LICENSE`, `pyproject.toml`. Plus 8 individual
`prompt.txt` files and 1 `gpt_generated.py` read verbatim.

SKIMMED-NOT-READ: 4,825 baseline files. The 800 generated `.py` bodies and the
remaining 792 `prompt.txt` bodies were not read individually; they were characterised
by aggregate measurement (file counts, directory structure, MD5 dedup, prompt-length
distribution, regex scans for numeric constraints and refusal vocabulary). 2 PNGs and
25 `.pyc` not opened.

FINDINGS

1. **GROUND TRUTH IS NOT IN THIS REPO — the headline finding.**
   `src/cadtestbench/data_source.py:102-104`
   (`datasets.load_dataset(self.source, name=table_name, split=self.partition)`),
   `src/cadtestbench/cli.py:19` (`_DEFAULT_HUB_DATASET = "dimitrismallis/CADTestBench"`).
   The CADTests — the machine-checkable part — are Parquet tables (`samples/`,
   `cadtests/`) fetched from HuggingFace at runtime. Nothing in the 4,842 files
   contains a single `cadtest_code` string, expected bbox, expected volume, or expected
   feature count. There is no `data/hf/` directory. **The benchmark corpus is a network
   dependency, not committed data.** HOW verified: full read of `data_source.py`; an
   extension census showing zero `.json`/`.parquet`; `Get-ChildItem` of every baselines
   leaf directory returning exactly `{gpt_generated.py, prompt.txt}`.

2. **Machine-checkability: YES in principle, six-category taxonomy — but four of six
   taxonomy strings MISMATCH the harness.** `src/cadtestbench/runner.py:53-60`:
   ```
   _CATEGORY_ABBREV = {"topology_checks":"topo", "solid_shell_validity":"solid",
                       "dimensions_ratios":"dim", "volumetric_checks":"vol",
                       "spatial_arrangement":"space", "geometry_types":"geom"}
   ```
   This is the **authoritative on-the-wire category vocabulary** emitted by the
   reference implementation. The harness's `CATEGORIES` is
   `("solid_shell_validity", "topology", "geometric_types", "dimensions_ratios",
   "volumetric", "spatial")` at
   `src/harnesscad/eval/bench/protocols/test_assertions.py:37-45` — **four of six
   strings differ**: `topology` vs `topology_checks`, `geometric_types` vs
   `geometry_types`, `volumetric` vs `volumetric_checks`, `spatial` vs
   `spatial_arrangement`. Any harness code consuming a real CADTestBench parquet row's
   `cadtest_type` would bucket **everything** into `uncategorized` and per-category
   accuracy would silently collapse.
   HOW verified: read both files; `grep -rn
   "topology_checks\|volumetric_checks\|spatial_arrangement\|geometry_types" src/` ->
   **zero hits**. Stdlib-portable: yes (a 6-entry alias dict).

   **This resolves the campaign's "4 of 6 category IDs diverge" lead.** The lead was
   real and is confirmed here, in CADTestBench — not, as the brief's phrasing implied,
   spread across the small text-to-CAD repos, where an independent search found no
   category-ID lists at all (see the cross-repo section below).

3. **Judge rubrics / refusal predicates / geometric-similarity metrics: NONE EXIST
   HERE.** The design is explicitly anti-rubric — README L24-29 argues test-based eval
   replaces Chamfer Distance and learned evaluators ("no inference cost from learned
   evaluators"). There is **no LLM judge, no rubric text, no scoring scale, no
   refusal/rejection predicate, no expected-rejection tasks**. Scoring is binary
   conjunction only. HOW verified: read all 11 source files (no judge/rubric symbol
   anywhere); `Select-String` for
   `impossible|cannot|invalid|not possible|ambiguous|refuse` across all 400 GPT-5.2
   `prompt.txt` -> **0 matching files**. Every prompt is a satisfiable build request.

4. **Prompt corpus: 200 samples x 2 partitions; the detailed partition is 100%
   numerically constrained.**
   `baselines/GPT-5.2/{Abstract,Detailed}/generated_models/<8-digit-id>/prompt.txt`.
   200 sample directories per partition, **identical id sets across both partitions and
   both models** (Compare-Object -> null). Abstract prompts mean 289 chars (min 50, max
   1024); detailed mean 525 (min 112, max 1205). Regex `\d+\.\d+`: **200/200 detailed
   prompts carry explicit decimal dimensions; 1/200 abstract**. Example (`00019015`):
   *"cylinder with a height of 0.468766 units and a diameter of 1.31858 units...
   rectangular prism with a length of 1.5, width 0.1875, height 0.075... center of the
   prism directly above the center of the cylinder."* This is de-facto machine-checkable
   ground truth **embedded in prose** — it must be parsed out, and the repo does not
   provide the parsed form.
   **Provenance risk:** the 200 programs are sourced from **CADPrompt** (README L33,
   `github.com/Kamel773/CAD_Code_Generation`) — a third-party upstream **this MIT file
   does not cover**. Harness equivalent: NONE for this prompt corpus
   (`grep -rln "CADPrompt" src/` -> 0 hits).

5. **Known-good/known-bad models: 800 committed generations, but UNLABELED.**
   `baselines/{GPT-5.2,Claude-4.6-Sonnet}/{Abstract,Detailed}/generated_models/*/gpt_generated.py`
   — 800 real LLM CadQuery outputs. These are genuine hard positives and negatives —
   e.g. `00000007` Abstract shows a **nested-function bug** (`create_cad` defined inside
   `create_cad`) plus a hardcoded absolute export path, and uses radius 10.0 / height
   20.0 where the detailed prompt specifies 0.75 / 0.20923. But **no pass/fail labels,
   no scores, no eval output is committed** — results live only on an external Vercel
   viewer. So they are unlabeled fixtures, not a labelled corpus. Harness equivalent:
   NONE; low value without labels.

6. **Metric definitions PR / RS / Acc / IR.** `src/cadtestbench/metrics/cadtest.py:164-251`
   (`_compute_rs_scoring`), L713-792 (`aggregate_results`). RS = groups_passed /
   groups_total x100 where a requirement group passes iff **every** filtered test in it
   passes; PR = samples_passed_all / total x100; IR = invalid_samples / total;
   per-category accuracy = passed/total per `cadtest_type`. Invalid samples are counted
   as failures, never excluded (L737-741).
   Harness equivalent: **`eval/bench/protocols/test_suite_quality`**
   (`src/harnesscad/eval/bench/protocols/test_suite_quality.py:8-31`) — documents
   PR/RS/Acc/IR with identical semantics **plus** Valid/Sound/MScore mutation analysis
   the reference repo does not implement. **The harness is a superset.**

7. **Execution-harness mechanics.** `metrics/cadtest.py:74-90` (`_extract_model_var_name`,
   an AST walk for `cq.exporters.export`), L386-390 (export-line stripping plus
   `compile(..., optimize=2)`), L59-66 (`ASSERTION_EXEC_PREAMBLE` with
   `check(condition, pass_msg, fail_msg)`), L427-478 (`create_code_cadtest_block`
   replay-script generator) — the four tricks that make string-form tests runnable.
   Harness equivalent: **`eval/bench/protocols/test_execution`**, whose docstring names
   all four (`extract_model_var_name`, `strip_export_calls`, `execute_cadtest`,
   `run_cadtest_block`) plus `build_replay_script`, and explicitly credits *"the
   reference `cadtestbench.metrics.cadtest` module"*. **Already ported.**

8. **Minor uncovered mechanics.** `runner.py:36-50` `filesystem_run_label_slug`
   (Windows-forbidden-char slugging); `runner.py:328-381` `_format_sample_line` (the
   `[001/200] 00000007 PASS 12/12 0.19s topo:6/6 ... RS 6/6` console line);
   `metrics/__init__.py:24-56` `_discover_metrics` (pkgutil auto-registration with a
   duplicate-name guard); `cadtest.py:254-299` `_resolve_model_code_path` (a 7-tier
   candidate search). Small, real, portable; low strategic value.

ALREADY COVERED — five modules mined from this exact paper, all deep rather than stubs:
`eval/bench/data/cad_model_schema` (B-rep `m=(F,E,V)` plus similarity-transform
`transformed` for the pose/scale passing set); `eval/bench/protocols/test_assertions`
(357L — `CadTest` predicate, six categories, 13 assertion factories);
`eval/bench/protocols/test_execution` (341L, explicitly credited);
`eval/bench/protocols/test_suite_runner` (152L — conjunction `T(m)=AND_i T_i(m)`,
requirement groups, per-sample PR/RS, invalid-as-failure, `run_passing_set`
invariance); `eval/bench/protocols/test_suite_quality` (187L — PR/RS/Acc/IR **plus**
mutation analysis). Adjacent: `eval/verifiers/requirements`,
`eval/bench/protocols/chamfer_bbox_judged`, `domain/programs/extract/cadquery_clean`.
In several respects the harness implementation is **ahead of** the reference repo —
mutation analysis and the passing-set invariance check are described in the paper but
absent from this code drop.

VERDICT: **already-covered**, with exactly two things worth acting on:
1. **The category-string mismatch (finding 2)** — a real interoperability bug. Fix is a
   6-entry alias map; no vendoring required.
2. **The 400 prompts (finding 4)** are the only substantive data here and are legally
   the *weakest* item — they derive from CADPrompt, which this MIT LICENSE does not
   cover. Given that a missing LICENSE already cost Graph-CAD its 700 tasks, treat these
   as **manifest-only, do not vendor**, pending a CADPrompt licence check.

The 4,800-file baselines tree is 3x redundant, unlabeled, and contributes no ground
truth. There is no judge rubric, no error taxonomy beyond the six categories, no
refusal predicate, and no geometric-similarity metric in this repository — by explicit
design.

---

## text-to-cad-main (1,733 files; no node_modules and no .git in this copy, so 0 excluded)

LICENSE: **MIT** confirmed from the actual file — `LICENSE:1` `MIT License`, L3
`Copyright (c) 2026 earthtojake`.

COMPOSITION (`find . -type f` by top-level directory): `skills/` 448 (the skillpack —
cad 60, cad-viewer 207, implicit-cad 49, dxf 43, sdf 28, srdf 20, urdf 18, bambu-labs
7, gcode 6, step-parts 5, sendcutsend 5); `plugins/` 441 — a **vendored duplicate**,
`plugins/cad/skills/*` being a byte-level re-bundle of `skills/` + `packages/cadpy` +
`packages/implicitjs` + `viewer/` (3 identical copies of the 34-file cadpy tree alone);
`viewer/` 373 (Vite/React + 32 built `dist/assets`); `packages/` 229 (cadjs 144,
implicitjs 43, cadpy 36, cadpy_metadata 5) — **the only real first-party engine code**;
`tests/` 45; `scripts/` 44; `backend/` 42; `docs/` 39 (38 are the viewer docs site);
`frontend/` 23; `benchmarks/` 20 (10 `.md` + 10 `.gif`); root 11; `.github/` 8;
`.githooks/` 5; `assets/` 3; plugin manifests 2. By extension: 688 `.js`, 450 `.py`,
149 `.md`, 94 `.mjs`, 55 `.sh`, 38 `.json`.

WHAT IT IS: two things stapled together. (1) A thin FastAPI demo backend (~6.5k LOC
incl. tests) that pipes a prompt to Gemini, regex-extracts BadCAD Python, `exec()`s it,
and writes STL. (2) The actual product: a Claude/Codex plugin skillpack (`skills/`)
backed by a build123d-based Python engine (`packages/cadpy`, ~13k LOC) and a JS
implicit/SDF + rendering stack (`packages/implicitjs`, `packages/cadjs`). Everything in
`plugins/` is a redistribution bundle of (2).

READ (in full, 13 files, ~2,900 lines): `backend/core/exceptions.py` (162),
`backend/core/models.py` (163), `backend/core/config.py` (97),
`backend/services/badcad_executor.py` (273), `backend/utils/code_extraction.py` (171),
`backend/utils/stl_fallback.py` (262), `packages/cadpy/src/cadpy/validators.py` (119),
`packages/cadpy/src/cadpy/generation_status.py` (179),
`packages/cadpy/src/cadpy/analysis.py` (683, via full dump),
`packages/implicitjs/src/lib/implicitCad/meshQuality.js` (243),
`skills/cad/references/repair-loop.md`, `skills/sdf/references/llm-guardrails.md`,
`skills/sendcutsend/SKILL.md` (122), `skills/gcode/references/gcode-validation.md` (42),
`benchmarks/03-l-bracket.md` (42).

SKIMMED-NOT-READ: ~1,700 files. Specifically: `packages/cadpy`'s remaining 26 modules
(~12,000 lines — read only their `wc -l`, symbol lists, and targeted greps for
`raise`/tolerance/unit constants; 66 `raise` sites in `generation.py`, all CLI-argument
shape validation); all 144 `packages/cadjs` files (grep only); the other 42
`packages/implicitjs` files; all 373 `viewer/` files; all 441 `plugins/` files
(confirmed vendored, not opened); `backend/tests` 14 files (line counts only);
`tests/python` 29 files (line counts only — `tests/python/skills/cad/cadpy/test_generation.py`
alone is 2,691 lines of CLI-argument-shape assertions); `skills/sdf/references/validation.md`
and `skills/urdf/references/validation.md` first 80/60 lines each, not to end; 9 of the
10 benchmark briefs (headers plus line counts only).

FINDINGS

1. **Implicit-mesh quality analyzer with derived, geometry-scaled tolerances** —
   `packages/implicitjs/src/lib/implicitCad/meshQuality.js:110-243`.
   `analyzeImplicitMeshQuality(mesh, options)` returns manifold-edge classification
   (boundary=1 / manifold=2 / non-manifold>2 via quantized vertex keys, L161-204),
   degenerate-triangle count, non-finite position/normal counts, worst normal
   alignment, and **SDF-probed orientation checking** (L166-190: sample the SDF at
   `center +/- normal*eps`; `positive < negative` implies inverted winding;
   `|delta| <= tol` implies ambiguous), returning `invertedRatio`, `boundaryRatio`,
   `nonManifoldRatio`.
   The **tolerance derivation** (L117-120) is the load-bearing and rare part — every
   tolerance is a max of an absolute floor and a term scaled to the marching grid step
   and bbox diagonal:
   ```js
   const edgeTolerance = Math.max(finiteNumber(options.edgeTolerance, 0), minStep * 1e-4, bounds.diagonal * 1e-8, 1e-7);
   const areaTolerance = Math.max(finiteNumber(options.areaTolerance, 0), minStep * minStep * 1e-10, 1e-12);
   const orientationEpsilon = Math.max(finiteNumber(options.orientationEpsilon, 0), minStep * 0.2, 1e-5);
   ```
   `orientationEpsilon = minStep * 0.2` (probe a fifth of a voxel off-surface) and
   `areaTolerance ~ minStep^2 * 1e-10` are exactly the constants-with-a-reason this
   audit hunts. Also the hard-coded `alignment < -0.35` poor-normal threshold (L157) and
   the `minStep` fallback `bounds.diagonal / 64` (L116).
   Harness equivalent: **PARTIAL.** `eval/bench/geometry/mesh_topology` and
   `eval/bench/geometry/mesh_quality` cover manifold-edge counting and quality scoring;
   `domain/geometry/mesh/halfedge` carries manifoldness invariants. **NOT in the 1579
   modules:** the **SDF-probe winding-orientation test** (verify a mesh's face
   orientation against the field that generated it) and the **grid-step-derived
   tolerance ladder**. Verified by grepping the dump for `manifold|watertight|mesh_quality`
   (24 hits, all listed above) — none mentions orientation-vs-SDF or step-scaled
   tolerance. Stdlib-portable: **yes**, trivially — pure arithmetic over flat
   position/normal arrays plus a callable SDF.

2. **G-code static-validation rule set plus bounds policy** —
   `skills/gcode/references/gcode-validation.md:7-32`. An explicit fail/warn split with
   reasons. Fails: empty file; no `G0/G1/G2/G3`; no extrusion moves; no temperature
   commands; absolute `X/Y/Z` exceeding profile bounds. Warns: unknown commands;
   relative positioning (L18-19, *"bounds checking is skipped while relative mode is
   active"*). Bounds from `machine.bed_size_mm[0/1]` and `machine.z_height_mm`, with an
   override `machine.motion_bounds_mm` gated by a stated reason — L32: *"only from a
   real printer/profile source, not as a way to silence unknown G-code."* L24 gives the
   reason for the `G90`/`G91` policy: avoid false hard failures on relative blocks while
   still catching off-bed absolute moves. L36: `ok: true` explicitly does not mean
   safe-to-print.
   Harness equivalent: **NONE.** `grep -ic "gcode\|g-code"` over the 1579-module dump =
   **0**. `domain/fabrication/printability_verdict` and `feature_minima` cover FDM
   *geometry* printability, not toolpath validation. No slicer/nozzle/toolpath module
   exists. Stdlib-portable: yes — line-oriented text parsing plus a bounds dict.

3. **SDFormat (Gazebo) generation-time validator check catalogue** —
   `skills/sdf/references/validation.md`. A three-severity diagnostic model (`error`
   blocks write / `warning` unless `--strict` / `info`) with an enumerated check list:
   root/version shape; name uniqueness per scope; **pose checks** (6 finite values for
   `euler_rpy`, 7 for `quat_xyzw`, quaternion approximately normalized, `degrees="true"`
   is a warning, omitted `relative_to` is a warning, `::` nested-reference syntax);
   frame `attached_to` cycle detection; the exact SDF 1.12 joint-type list
   (`continuous, revolute, gearbox, revolute2, prismatic, ball, screw, universal,
   fixed`); `world` allowed as parent but never as child; axis finite/nonzero/normalized;
   `axis2` only where the type supports it; lower <= upper; continuous joints with fake
   finite limits => warning.
   A ready-made unit- and frame-aware refusal predicate set for a spec format, with
   severity attached to each predicate.
   Harness equivalent: **NONE for SDFormat.** `grep -ic "sdformat\|gazebo"` = **0**. The
   harness's 22 `domain/geometry/sdf/*` modules are *signed-distance-field* geometry, an
   entirely different thing — do not confuse them. `domain/spec/urdf` and
   `domain/spec/srdf` cover the sibling formats but not SDFormat. The companion
   `skills/urdf/references/validation.md` (units: `xyz` metres, `rpy` radians, revolute
   limits radians, prismatic limits metres; exactly one root link; connected and acyclic;
   `links - 1` joints) **is** covered by `domain/spec/urdf`. Stdlib-portable: yes
   (`xml.etree.ElementTree` plus arithmetic).

4. **LLM spatial-inference guardrail taxonomy** — `skills/sdf/references/llm-guardrails.md`.
   An explicit two-column split of what an agent may infer vs what it must never infer
   silently. The never-infer list is the useful half: exact link poses / frame
   transforms / joint origins; positive joint-axis direction from appearance; mesh
   units / scale / coordinate convention; COM and inertia tensor from rendered shape;
   plugin filenames / params / topics; whether a plugin is runtime vs
   visualization-only; whether collision geometry is physics-stable; whether external
   URIs resolve. Paired with a five-source provenance requirement for every
   spatial/physical value, a placeholder policy with named unacceptable cases (*"silently
   flipping a joint axis to match an expected screenshot"*), and a 7-row spatial-reasoning
   evidence table.
   This is a refusal *policy* keyed to a named failure mode (hallucinated units and axis
   signs), and the "do not hide guessed values in raw XML" code-style contrast (named
   constants plus a source comment vs inline `"0.18 0 .12 0 -11.5 0"` hiding
   degrees-vs-radians) is directly reusable as a generation constraint.
   Harness equivalent: **PARTIAL/NONE.** `agents/generation/feedback_taxonomy` and
   `domain/programs/review/taxonomy` are *error* taxonomies (post-hoc);
   `domain/spec/clarify_scaling` covers one specific scaling failure mode. **No module
   encodes a pre-generation "must-not-infer" predicate list** — grep for `guardrail`
   returns 0 hits. Stdlib-portable: yes (a rule table, not code).

5. **`assert_*` geometry-assertion vocabulary over a selector manifest** —
   `packages/cadpy/src/cadpy/validators.py:40-100`. `assert_close(actual, expected,
   tol=1e-6, label)` raising
   `AssertionError(f"{label} mismatch: expected {expected:.6f}, got {actual:.6f} (tol={tol:.6f})")`
   (L40-47), plus `assert_bbox_coordinate`, `assert_bbox_span`, `assert_selector_count`
   over `shape|face|edge|occurrence`. Companion constants: `AXIS_ALIGNMENT_THRESHOLD =
   0.985` in `analysis.py:13` (a ~10-degree cone for "this face normal is axis-aligned"),
   and `coordinate_tolerance = 1e-3` with `min_area_ratio = 0.05` for grouping coplanar
   faces into "major planes" (`analysis.py:434-488`, `reporting.py:17-19`).
   Harness equivalent: **COVERED.** `domain/geometry/topology/entity_selector`,
   `region_selectors`, `selector_algebra`, `selector_dsl`, `selector_grammar` and
   `domain/programs/expressions/cad_ref_selectors` are a far richer selector stack. The
   bare `assert_close` wrapper adds nothing; only the three numeric constants are
   marginally interesting.

6. **Failure-class -> cause -> fix repair table** — `skills/cad/references/repair-loop.md`.
   Nine named failure classes (source import/syntax; invalid-or-missing geometry;
   fillet/chamfer failure; wrong scale or bbox; missing feature; selector fragility;
   positioning/joint mismatch; viewer startup; snapshot failure), each with likely-causes
   and fixes. E.g. fillet failure => *"radius/length exceeds local geometry"* => *"reduce
   radius/length, filter edges more narrowly, apply fillets later in the model."*
   Selector fragility => *"arbitrary index selection; topology changed after fillet or
   boolean"* => *"select by axis, plane, position, normal."*
   Harness equivalent: **COVERED.** `agents/agent/code_repair_rules` is described
   verbatim as *"Deterministic error-to-hint-to-autofix rules for generated CAD code"*;
   plus `agents/generation/error_patterns` (CADSmith KB2), `domain/programs/review/syntax_repair`,
   `domain/programs/validate/diagnostics`, `agents/memory/error_notebook`.

NON-FINDINGS worth stating explicitly, because they were looked at and are empty:
- `backend/core/exceptions.py` is a **10-class HTTP error taxonomy**
  (`BADCAD_EXECUTION_ERROR`, `AI_GENERATION_ERROR`, `USER_LIMIT_EXCEEDED`,
  `MODEL_NOT_FOUND`, `INVALID_INPUT`, `STORAGE_ERROR`, `CONFIGURATION_ERROR`,
  `DEPENDENCY_ERROR`, `AUTHENTICATION_ERROR`, `AUTHORIZATION_ERROR`) — every one a web
  API status code, **zero geometry semantics**. Not a CAD error taxonomy.
- `backend/utils/code_extraction.py:114-141` `validate_badcad_code()` is the only
  "validation" in the backend: three string checks (`"from badcad import" in code`,
  `"model =" in code`) plus `compile(code, "<string>", "exec")`. Its sibling
  `extract_badcad_code` (L40-72) uses a hand-written code-vs-prose heuristic
  (`text_patterns = ["create","generate","the","this",...]`) that is genuinely bad — it
  misclassifies any comment-heavy or docstring-bearing output. Nothing to mine.
- `backend/utils/stl_fallback.py:110-141` is a **default-shape lookup table**
  (`("gear","cog","teeth","sprocket")` -> 12-tooth gear at r=12/r=8;
  `("ring","washer","hole","donut","torus")` -> r=10/r=5 extrude 5; default box
  15x15x8) — arbitrary demo numbers with no stated reason, emitted when the AI is down.
  **Not a defensible default-dimension table.**
- `backend/services/badcad_executor.py:229-243` `sandbox_execution()` is an **empty
  placeholder** — the comment admits resource limits, network restrictions and FS limits
  are all unimplemented, while L135 does a bare `exec(badcad_code, exec_globals,
  exec_locals)` with real `__builtins__`, `os` and `sys` injected. Anti-pattern; do not
  mine.
- **No unit-conversion logic anywhere in `packages/cadpy`.** Grep for
  `UNIT|millimet|unit_scale|INSUNITS|inch` over all 36 cadpy modules returns **4 hits**:
  `step_export.py:59,70` (a one-line OCP call
  `XCAFDoc_DocumentTool.SetLengthUnit_s(doc, 1/UNITS_PER_METER[Unit.MM])`),
  `step_scene.py:491` (a comment), `threemf.py:432` (`"unit": "millimeter"` literal).
  Everything is hard-assumed mm. No conversion table.
- **No known-bad-input corpus, no judge/eval rubric, no LLM-scored evaluation anywhere.**
- `backend/tests/` (14 files, 2,300 lines) and `tests/python/` (29 files, 13,694 lines)
  are implementation unit tests against fixtures, not a portable task corpus.
- The 66 `raise` sites in `packages/cadpy/src/cadpy/generation.py` are **all CLI
  argument validation** (*"must use POSIX '/' separators"*, *"--output can only be used
  with exactly one target"*). No geometric refusal predicates.
- `packages/cadpy/src/cadpy/generation_status.py` is a heartbeat lockfile tracker — pure
  infrastructure.
- `plugins/` (441 files, 25% of the repo) is a **vendored republication**; nothing unique.

ALREADY COVERED: the five known-mined items — skillpack (`skills/`, 448 files),
`cad_defaults`, `plugin_manifest`, `printer_profiles`, and the 10 briefs
(`benchmarks/01-10*.md`, each with a `## Test Cases` table of machine-checkable expected
results including explicit negative checks). Plus, newly confirmed as duplicated:
selector/reference machinery -> `domain/geometry/topology/{entity_selector,region_selectors,selector_algebra,selector_dsl,selector_grammar}`,
`domain/programs/expressions/cad_ref_selectors`; the repair loop ->
`agents/agent/code_repair_rules`, `agents/generation/error_patterns`,
`domain/programs/review/{syntax_repair,taxonomy,detect,correct}`; manifold/mesh-topology
metrics -> `eval/bench/geometry/{mesh_quality,mesh_topology}`,
`domain/geometry/mesh/{halfedge,repair_toolkit,intersection_repair}`; URDF/SRDF
validation -> `domain/spec/urdf`, `domain/spec/srdf`; SendCutSend DFM review ->
`eval/verifiers/dfm`, `eval/bench/protocols/dfm_scoring`, `domain/drawings/dfm_review`,
`domain/fabrication/registry`; 3MF handling -> `io/formats/threemf`.

Note on SendCutSend: the SKILL is deliberately **rule-free** — it fetches SendCutSend's
live catalog/specs JSON at runtime and hard-refuses to hardcode thresholds (*"Treat
SendCutSend's ordering guide, catalog JSON, and specs JSON as evidence feeds, not stable
APIs"*, L14). **There is no DFM constant table in this repo to extract** — only
field-path references like `bending_specs.min_flange_length_before_bend`. Its one
genuinely portable contribution is the local-vs-aggregate rule at L60: measure minimum
*local* flange depth at every sample point along each bend, because notches, slots and
split tabs create local free edges that aggregate source values miss. Also worth having
is its 3-label restraint vocabulary `pass / fail / need more info`, where missing
evidence is never silently a pass.

VERDICT: **mine-further — narrowly.** Four items, in priority order: (1) the SDF-probe
mesh-orientation test plus grid-step-derived tolerance ladder from `meshQuality.js:110-243`;
(2) the G-code static validator rule set (a confirmed 0-module gap); (3) the SDFormat
validator check catalogue (also a confirmed 0-module gap — not to be confused with the
harness's 22 signed-distance-field modules); (4) the SDF LLM-guardrail must-not-infer
list as a pre-generation constraint table. All four are stdlib-portable.

**Correcting the prior run:** the claim that "the unmined territory is `backend/` and
`packages/`" is **half wrong**. `backend/` is 42 files of FastAPI demo scaffolding whose
only geometry-adjacent code is three string checks and an unsafe `exec()` — **nothing
there**. `packages/cadpy` (13k lines) is real engineering but build123d/OCP-bound
orchestration and selector plumbing the harness already surpasses; its only portable
content is three tolerance constants. The genuinely unmined material is in
`packages/implicitjs/src/lib/implicitCad/meshQuality.js` and in three skill *reference*
documents the skillpack mining evidently indexed but did not extract as rules.

---

## text-to-cad-better (261 files, non-.git; inner dir is `text-to-cad-main/`)

LICENSE: **MIT** (Copyright (c) 2026 Thompson Labs), `LICENSE:1-3` -> vendorable with
attribution. Note the harness's existing importer cites this same upstream as "MIT (c)
2026 earthtojake" — same skill pack, different copyright line; worth reconciling before
re-vendoring.

WHAT IT IS: an agent skill pack (`.agents/skills/{cad,urdf,robot-motion}`) for
STEP-first build123d CAD — SKILL.md plus 9 reference docs plus a
`scripts/{step,inspect,render,dxf}` CLI (~11.3k lines Python) plus a Vite/React "CAD
Explorer" plus 10 benchmark briefs.

READ (in full): `.agents/skills/cad/SKILL.md`,
`references/inspection-and-validation.md`, `references/repair-loop.md`,
`references/natural-language-specs.md`, `assets/design-brief-template.md`,
`references/positioning.md` (first 70 of 242 lines), `scripts/common/validators.py`,
`scripts/common/catalog.py` (first 120 of 532).

SKIMMED-NOT-READ: the remaining ~10,000 lines of `scripts/common/*.py`
(`step_scene.py` 1827, `generation.py` 1385, `assembly_composition.py` 1245, `glb.py`
745, `threemf.py` 539); all 40+ Explorer `.jsx` files; the entire `urdf` and
`robot-motion` skills.

FINDINGS

1. **Error taxonomy — 9 named failure classes with cause->fix pairs** |
   `.agents/skills/cad/references/repair-loop.md:14-160` (source-import/syntax, invalid
   geometry, fillet/chamfer, wrong scale/bbox, missing feature, selector fragility,
   positioning/joint, Explorer startup, render) | the cleanest CAD repair taxonomy in
   this family | Harness: **partially covered** — `domain/programs/review/taxonomy`
   (CADReview's eight error scenarios) and `agents/agent/code_repair_rules` exist but
   derive from different sources; **"selector fragility" and the fillet-radius-vs-edge
   class were not found ported** (`grep -riE "selector fragility|fillet.*chamfer failure"
   src/harnesscad` -> 0 hits). Stdlib-portable: yes (prose -> table).
2. **Refusal predicate** | `references/inspection-and-validation.md:150-156` — *"Do not
   claim: structural safety / process certification / tolerance compliance /
   manufacturability beyond geometric plausibility unless the relevant analysis was
   explicitly performed"*, plus the scope-exclusion list at `SKILL.md:17` (no CAM
   toolpaths, FEA conclusions, BIM) | a directly encodable abstention rule | Harness:
   the *framing* is already quoted verbatim in `domain/standards/cad_defaults.py:9-10`,
   but **as a docstring caveat, not as a predicate**.
3. **Clarification policy — an ask/don't-ask predicate** |
   `references/natural-language-specs.md:83-99` — ask only when no dimensions are given,
   mating geometry is unspecified, the request is safety/load/pressure/medical/compliance
   -bound, or a required source file is absent; do **not** ask for clearance standards,
   cosmetic fillet radii, or origin choice | a machine-checkable gate on when an agent
   may block | Harness: **NONE found** — `grep -rl "clarification" src/harnesscad` and
   the registry dump show no ask-vs-assume predicate module. Stdlib-portable: yes.
4. **Default-dimension table** | `SKILL.md:31-33` (M3/M4/M5 clearance 3.4/4.5/5.5 mm;
   enclosure wall 2.0-3.0; cosmetic fillet 1.0-3.0) | **ALREADY PORTED**, verbatim and
   cited, in `domain/standards/cad_defaults.py:12-16`.
5. **Assertion helpers for measured ground truth** | `scripts/common/validators.py:44-100`
   (`assert_bbox_span`, `assert_bbox_coordinate`, `assert_selector_count` over
   face/edge/shape/occurrence counts, tol=1e-6) | Harness: covered by
   `eval/verifiers/validity_gate` plus `eval/quality/geometry/solid_usability`.
   Stdlib-portable: no as written (imports `inspect_refs.analysis`), but trivially
   reimplementable.
6. **NEGATIVE FINDING — the 10 benchmark briefs are not in this checkout.** All ten
   `.assets/benchmarks/*.md` are **git-LFS pointer stubs** (verified: every file's line
   1 is `version https://git-lfs.github.com/spec/v1`, sizes ~2 KB). The real briefs,
   with their machine-checkable "Test Cases" tables, were obtained from a *different*
   checkout and are already vendored at `src/harnesscad/eval/bench/imports/textcad/`.
   **Do not re-mine from here — this copy has no content.**

ALREADY COVERED: `domain/spec/design_brief` (*"Natural-language CAD brief IR with
deterministic default resolution"*) is the port of `natural-language-specs.md`;
`domain/standards/cad_defaults` is the port of the defaults table;
`eval/bench/imports/cadam_textcad_briefs` is the port of the benchmarks.

VERDICT: **mine-further** — but only items 1-3 (the repair-taxonomy classes, the refusal
predicate, the clarification predicate). The geometry/CLI code is a build123d-coupled
11k-line CLI with no reusable ground truth.

---

## text-to-cad-ui-main (112 files)

LICENSE: **MIT** (Copyright (c) 2023 The KittyCAD Authors) -> vendorable with attribution.

WHAT IT IS: Zoo/KittyCAD's SvelteKit front-end for their hosted text-to-CAD API — auth,
billing dialogs, infinite scroll, a GLTF viewer, Playwright e2e.

READ (in full): `src/lib/consts.ts`, `src/components/ExamplePrompts.svelte`,
`src/components/PromptGuide.svelte`. Enumerated all 74 files under `src/`, `tests/`,
`static/`.

SKIMMED-NOT-READ: ~2,300 lines of Svelte components and the 3 Playwright specs.

FINDINGS

1. **15 example prompts** | `src/lib/consts.ts:11-27` | real-user-shaped briefs (a
   320 mm vented brake rotor with 5x M12 on a 114.3 mm PCD; a 12 ft I-beam with full
   imperial section dimensions; a femur bone plate) — decent exemplar prompts, **but no
   expected output, no ground truth, no rejections** | Harness:
   `eval/bench/imports/zoo_kcl_manifest` already imports 100 human-described Zoo parts,
   a strictly larger and better-grounded corpus.
2. **Prompt-writing rubric, 3 rules** | `src/components/PromptGuide.svelte:3-15` —
   geometric-not-nebulous, be explicit about hole placement and diameter, single objects
   beat assemblies | weak; UI copy, not a rubric.

ALREADY COVERED: `domain/spec/zoo_catalog`, `zoo_cli_catalog`, `zoo_ml_feedback`,
`io/adapters/zoo_api`, `io/backends/zoo`, `io/formats/kcl`, `domain/spec/kcl_grammar` —
the Zoo surface is thoroughly mined already.

VERDICT: **already-covered.** Everything else is UI glue and billing.

---

## Text-to-CadQuery-main (30 files)

LICENSE: **NO LICENSE FILE** (verified: the tree contains only `data_annotation/`,
`inference/`, `README.md`, `train/`) -> **manifest-only, vendor NOTHING.**
Facts-with-citation only.

WHAT IT IS: NeurIPS-submission code fine-tuning six small LLMs to emit CadQuery from
Text2CAD/DeepCAD prose — 6 train scripts, 6 inference scripts, a Gemini annotation
pipeline, a Gemini image judge, Blender rendering, a Chamfer-distance notebook.

READ (in full): all 6 `train/*.py` (via targeted greps for every model ID, data path,
prompt template and output dir); all 6 `inference/step1_generate_CadQuery/*.py`;
`data_annotation/gemini_pipeline.py:1-80`;
`inference/step4_gemini_eval/eval_tuned_model.py` (all 160 lines); all 4 READMEs; the
schema of all 5,198 lines of `test_filtered.jsonl`.

SKIMMED-NOT-READ: the 6 `step2_clean_run_CadQuery/*.ipynb` cleaning notebooks;
`step3_rendering/custom_main.py` (~400 lines, vendored from objaverse-xl);
`step5_compute_metrics/compute_CD.ipynb`.

FINDINGS

1. **Binary LLM-judge rubric with a deliberate asymmetric bias toward Yes** |
   `inference/step4_gemini_eval/eval_tuned_model.py:57-75` — *"Since the image is a
   single-angle rendering, some features may not be visible. If it is reasonably
   possible that the model matches, answer `Yes`. Only respond `No` if you are very
   certain"* — **a judge tuned to under-report failures**, which is a citable
   *anti-pattern* for a harness judge | Harness: `eval/bench/protocols/criteria` (typed
   per-sample criteria) is the counter-design. Stdlib-portable: yes (prompt text).
2. **BUG — `train_gpt2_medium.py` trains gpt2-large.** `train/train_gpt2_medium.py:10`
   is `model_id = "openai-community/gpt2-large"` — identical to `train_gpt2_large.py:10`
   — and it writes to `output_dir="./checkpoints-gpt2large"` (L40), the same path as the
   large run. `train/README.md` claims *"train_gpt2_medium.py — Finetunes GPT-2 Medium"*.
   **The published "GPT-2 Medium" results row cannot have come from this script as
   committed.**
3. **BUG — `train_qwen-3B.py` reads different data files.** L16-17 load
   `data_train_save_file.jsonl` / `data_val_save_file.jsonl`; the other 5 train scripts
   and `train/README.md` all say `data_train.jsonl` / `data_val.jsonl`.
4. **Broken import** | `data_annotation/gemini_pipeline.py:6` is
   `from prompts import PROMPT`, but no `prompts.py` exists anywhere in
   `data_annotation/`. **The annotation pipeline cannot run as shipped.**
5. **Hardcoded empty API keys** | `gemini_pipeline.py:11` and `eval_tuned_model.py:34`
   both `genai.Client(api_key="")`.
6. Prompt-template train/inference consistency **checks out** for all 6 models (bare
   `input` for CodeGPT and both GPT-2s; `<start_of_turn>` for Gemma; `<s>[INST]` for
   Mistral; `### Instruction:` for Qwen) — no divergence there.

ALREADY COVERED: `data/dataengine/annotation/prompt_levels` (Text2CAD taxonomy),
`eval/bench/geometry/scaled_chamfer_reward`.

VERDICT: **nothing-here** for vendoring (no licence) — mine only as cited facts, and
really only finding 1 (the judge anti-pattern).

---

## CADAM-master (363 files)

LICENSE: **GPL-3.0 CONFIRMED** — `LICENSE:1-2` reads
`GNU GENERAL PUBLIC LICENSE / Version 3, 29 June 2007`. The prior report is correct. ->
**MANIFEST-ONLY. Vendor NOTHING.**

WHAT IT IS: a commercial hosted text-to-OpenSCAD SaaS (adam.new/cadam) — React/Vite
front-end, Supabase back-end, Vercel AI SDK chat, token-cost accounting — plus a
`benchmarks/` showcase.

READ (in full): `benchmarks/README.md`, `src/server/aiChat.ts:500-520` (token
accounting), the file tree of `src/server/`, `shared/`, `scripts/`.

SKIMMED-NOT-READ: ~330 files of React UI, Supabase migrations, and 1,822 lines of
benchmark `.scad`.

FINDINGS (facts-with-citation only; nothing to copy)

1. **13 known-good OpenSCAD reference models with a difficulty ladder** |
   `benchmarks/README.md` plus `benchmarks/01..13-*.scad` (1,822 lines total, **real
   content, not LFS stubs** — verified, `benchmarks/03-hex-bolt-and-nut.scad:1-2` opens
   with `include <BOSL2/std.scad>`). The ladder runs twisted vase -> V8 engine; the
   README tabulates each model's parametric surface ("22 dims, 8 colors") | Harness:
   **already handled correctly** — `eval/bench/imports/cadam_textcad_briefs.py:15-21`
   records SHA-256 plus byte counts in `cadam/MANIFEST.json`, copies nothing, and
   degrades cleanly when the checkout is absent. GPL policy already enforced.
2. `shared/parametricParts.ts` plus `shared/parseParameters.ts` (each with a `.test.ts`)
   parse the OpenSCAD Customizer parameter syntax. Referenceable as an existence proof
   only.

ALREADY COVERED: item 1, fully.

VERDICT: **already-covered** (and correctly licence-fenced).

---

## cadsmith-main (24 files)

LICENSE: **MIT** (Copyright (c) 2026 cadsmith contributors) -> vendorable with attribution.

WHAT IT IS: a small, unusually clean generate -> execute -> **measure** -> self-correct
CadQuery agent. 1,514 lines total including tests.

READ (in full): **all 9** `src/cadsmith/*.py`, **all 6** `tests/*.py`, both
`examples/*.py`. **Nothing skimmed — this repo was read end to end.**

SKIMMED-NOT-READ: none.

FINDINGS

1. **Measured-geometry rejection taxonomy** | `src/cadsmith/validator.py:104-182` — six
   rejections with model-readable reason strings: no-solid; volume < `min_volume`
   (1.0 mm^3); degenerate bbox (a dimension <= 0); bbox > `max_dim` (10000 mm, *"likely
   a units mistake"*); `isValid()` false; free-edge count > 0 | Harness: **ALREADY
   PORTED** — `eval/quality/geometry/solid_usability.py:1-33` names cadsmith explicitly
   and reproduces all six checks (`no_solid`, `below_min_volume`, `degenerate_bbox`,
   `units_mistake`, `malformed_brep`, `not_watertight`) as a stdlib decision layer.
2. **Free-edge watertightness via an OCCT ancestor map** | `validator.py:63-88` —
   `TopExp.MapShapesAndAncestors_s(EDGE, FACE)`, skipping `BRep_Tool.Degenerated_s`
   edges (cone apexes). **The degenerate-edge exclusion is the subtle part.** | Harness:
   covered by `domain/geometry/topology/explorer`, `sew.py`, `euler_poincare.py`.
3. **Restricted-exec sandbox** | `src/cadsmith/executor.py:18-33, 90-105` — a 5-module
   import allowlist (`math`, `cmath`, `cadquery`, `cadquery.selectors`, `cadquery.func`,
   `numpy`) plus a ~40-name builtin allowlist plus a guarded `__import__` | Harness:
   `domain/programs/validate/fluent_subset_policy` is the only allowlist-style module
   found; `eval/reliability/executor` is the sandboxed orchestrator. Reasonably covered.
   The repo itself disclaims it as *"not a security sandbox against adversarial code"*
   (`executor.py:4-6`).
4. **6-prompt eval set plus scoring** | `examples/eval.py:22-29` (20 mm cube; 60x40x10
   plate with four M3 corner holes; hex nut 10 AF with 6 mm bore; 30 mm pulley with 5 mm
   bore and v-groove; 6 mm cable clip; 40 mm L-bracket 4 mm thick with 2 holes per leg)
   reporting success rate, **first-try rate**, and average attempts-on-success | tiny,
   but the attempts-to-success metric is the right shape | Harness:
   `eval/reliability/repair_metrics` covers feasibility and repair-success rates.
5. **Known-good/known-bad geometry test pairs** | `tests/test_validator.py:11-50` — a box
   passes with volume exactly 1000.0; `rect(20,20)` un-extruded fails;
   `box(50000,10,10)` trips the units guard; box plus `hole(6)` stays watertight |
   **genuine machine-checkable ground truth, 5 cases.** Harness: not present as literal
   regression cases.
6. **Negative-example list embedded in the system prompt** | `src/cadsmith/prompts.py:26-31`
   — un-extruded sketch, fillet radius >= adjacent edge, wrong face selection leaving an
   open shell, metres/inches confusion.

ALREADY COVERED: items 1 and 2, by name, in `solid_usability.py`.

VERDICT: **already-covered** for the validator; **mine-further** only for item 5 (the 4
concrete pass/fail geometry fixtures), if the harness wants literal regression cases.

---

## ScadLM-main (34 files)

LICENSE: **MIT** (Copyright (c) 2024 Krish Shah) -> vendorable with attribution.

WHAT IT IS: a GPT-4-Vision -> OpenSCAD loop with a render-and-look-at-it feedback step,
plus a Next.js UI reusing `three-cad-viewer`.

READ (in full): `backend/prompts.py` (all, including the appended conflict region),
`dataset/scad_model_dataset - Sheet1.csv` (parsed all 21 rows), the file tree.

SKIMMED-NOT-READ: `backend/{generate,server,util}.py`, the ~1,900-line
`testing/open-source-llms.ipynb`, the Next.js `ui/` (a near-duplicate of CQAsk's).

FINDINGS

1. **Visual-feedback rubric** | `backend/prompts.py:feedback_prompt` — renders the
   model, shows it back, and asks *"Does it look correct? If yes, respond YES. If no,
   identify what is wrong, then generate new code"* with an explicit checklist (*"check
   if parts align correctly, and that they are connected properly"*) | Harness:
   `agents/generation/three_view` plus `escalation` (CADSmith) cover this shape more
   rigorously.
2. **21-row description->OpenSCAD corpus** | `dataset/scad_model_dataset - Sheet1.csv`
   (2,544 physical lines, **21 CSV rows**, 0 empty code cells) — nut, parametric cube,
   drawer chest, Lego block, planetary gearset, pirate ship, helical gears, bike, Batman
   symbol, tic-tac-toe | Thingiverse-flavoured, **no ground truth, no expected
   dimensions** | Harness: strictly weaker than `eval/bench/imports/zoo_kcl_manifest`
   (100 parts) and the textcad briefs.
3. **BROKEN FILE — a committed, unresolved git merge conflict.** `backend/prompts.py`
   ends with `<<<<<<< HEAD:backend/prompts.py` ... `>>>>>>> ae537af (added image
   check):generate.py`, splicing ~120 lines of `generate.py` (with `render_scad`,
   `encode_image`, the vision loop) into the tail of `prompts.py`. **The file is not
   valid Python as committed.**
4. Prompt hygiene note: `work_harder_prompt = "Do not be lazy. Give your full best
   effort..."` is appended to every prompt — cargo-cult, no evidence offered.

ALREADY COVERED: the OpenSCAD cheatsheet in `prompts.py` duplicates what `domain/spec`
already models.

VERDICT: **nothing-here.**

---

## CQAsk-main (34 files)

LICENSE: **MIT** (Copyright (c) 2023 Open Orion LLC), `LICENSE.md:1-3` -> vendorable
with attribution.

WHAT IT IS: a 2023 GPT-3.5 -> CadQuery demo — one backend prompt, a FastAPI wrapper, a
Next.js viewer.

READ (in full): `backend/codex.py` (all ~110 lines),
`backend/utils/{download,json,tessellate}.py`, the file tree.

SKIMMED-NOT-READ: `backend/api.py`, the `ui/` Next.js app (a duplicate of ScadLM's), the
vendored single-file `three-cad-viewer.esm.js`.

FINDINGS

1. **Curated CadQuery API surface in-prompt** | `backend/codex.py:12-70` — ~30
   `cq.Workplane` sketch methods with signatures, plus the 6 `cq_gears` classes
   (`BevelGear`, `CrossedHelicalGear`, `RackGear`, `RingGear`, `Worm`, `SpurGear`) with
   full parameter lists, plus `parafoil` NACA airfoil classes | Harness:
   `agents/generation/api_reference` already does curated-API-surface-in-prompt.
2. **Output-contract rule** | `codex.py:14-15` — *"VERY IMPORTANT to name the output
   variable `obj` and do not use `show_object` or any show functions"* — the same class
   of rule as cadsmith's `result` convention.
3. **Anti-pattern worth citing** | `codex.py:96-101` — writes the LLM's raw output to
   disk and `exec`s it via `importlib.util.spec_from_file_location` with **no sandbox,
   no allowlist, no validation whatsoever**. The precise contrast to cadsmith's
   `executor.py`.

ALREADY COVERED: `agents/generation/api_reference`; `domain/library/gear_train` covers
the gear parameter math.

VERDICT: **nothing-here.**

---

## BlenderLLM-main (24 files)

LICENSE: **Apache-2.0** -> vendorable with attribution plus NOTICE.

WHAT IT IS: inference-side code for the BlenderLLM paper (Du et al., 2024) — prompt ->
`bpy` script -> headless Blender -> `.obj` plus 8 rendered views.

READ (in full): `scripts/config.py`, `scripts/geometry_utils.py`, `modeling.py`,
`README.md`. Enumerated all 24 files.

SKIMMED-NOT-READ: `chat.py`, `scripts/infer.py`, `scripts/blender_runner.py`.

FINDINGS

1. **8-pose camera rig plus a 5-level brightness table** | `scripts/config.py:3-12`
   (8 Euler triples) and L15-40 (5 named levels x 3 light groups x 8 values) | Harness:
   **ALREADY PORTED** — `domain/geometry/views/camera_rig` is described as *"BlenderLLM
   object-framing camera rig (Du et al., 2024)"*.
2. **Object-framing math** | `scripts/geometry_utils.py:14-36` — bbox from `.obj` verts,
   then 8 camera positions at `delta_max * 2.5/sqrt(2)` in X/Y and `delta_max * 2.5` in
   Z. Note the **axis swap at L32-34** (`y` is built from `center[2]`, `z` from
   `center[1]`), i.e. an OBJ Y-up -> Blender Z-up conversion done **silently inline** —
   a real portability trap if reimplemented.
3. **Dataset facts** (README only; the data is not in the repo): 12k samples, 2k
   human-annotated plus 10k GPT-4o; **16 object types, 8 instruction tones**; complexity
   via Unit Number / Parameter Density / Entropy | Harness: **ALREADY PORTED** —
   `eval/bench/data/complexity_entropy` (*"CADBench task-complexity metrics for
   BlenderLLM"*).

ALREADY COVERED: items 1 and 3, by name. Plus `domain/programs/validate/bpy_script`
(*"Static analysis of BlenderLLM `bpy` scripts"*).

VERDICT: **already-covered.**

---

## CAD-GPT-main (45 files)

LICENSE: **LGPL-2.1** — `LICENSE:1-2` reads
`GNU LESSER GENERAL PUBLIC LICENSE / Version 2.1, February 1999` -> **manifest-only,
vendor NOTHING.** Also relevant: `paper_code/common/*/gears.scad` is a vendored
third-party OpenSCAD gear library present in **4 copies with 3 distinct hashes**
(`bevel` and `helix_gear` are identical; `herringbon` and `spur_gear` each differ).

WHAT IT IS: a 3-prompt agent (parse -> design -> script) turning a gear-drive request
into OpenSCAD via a gear library, with 4 worked gear-pair examples.

READ (in full): `agent.py` (all 113 lines), all 3 `paper_code/prompts/*.py`,
`parameters_format.json`, `spur_gear/gears_parameters.json`,
`bevel/gear_parameters.json`, the module list of `gears.scad`.

SKIMMED-NOT-READ: the 4 `paper_code/notebooks/*.ipynb`, the 4 `gears_design.md`, the
~270-line `gears.scad` body.

FINDINGS

1. **Default-parameter table: the preferred gear-module series — and it contains a
   verifiable error** | `paper_code/prompts/design_prompt.py:35` (priority) and L37
   (fallback). `14` appears in **both** lists. L35 is presented as the preferred series
   but reads `1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 25, 32, 40, 50`
   — **ISO 54 Series 1 does not contain 14** (it goes ...12, 16, 20...); 14 belongs to
   Series 2, where L37 also lists it. So the "priority" series is contaminated with a
   Series-2 value and the two tiers overlap | Harness: **ALREADY COVERED** —
   `domain/geometry/kinematics/gear_modules` (*"Standard gear-module series selection,
   deterministic, stdlib-only"*) and `domain/library/gear_train` (*"mined from CAD-GPT"*).
2. **Constrained-vocabulary parse contract** | `prompts/parsing_prompt.py:20-27` —
   `gear_type` in {`spur`, `helix`, `herringbone`, `bevel`}; `source` in {`driven`,
   `driving`}; plus **conditional zeroing rules** (spur implies `helix_angle`=0 and
   `pitch_angle`=0; spur/helix/herringbone imply `pitch_angle`=0). Genuine
   machine-checkable validity predicates over the parameter dict.
3. **Vocabulary divergence, three ways** | the enum says `helix`
   (`parsing_prompt.py:22`), the design examples say "helical gear"
   (`design_prompt.py:23-24`), and the directory is `helix_gear/`. `herringbone` in the
   enum vs the directory `herringbon/` (a typo). The schema key is `"gear "` — **with a
   trailing space** — in `parameters_format.json:2`. Instance files disagree with each
   other too: `spur_gear/gears_parameters.json` uses key `gear_1`,
   `bevel/gear_parameters.json` uses `gear 1` (space); filenames alternate
   `gear_parameters.json` / `gears_parameters.json`.
4. **Type->function dispatch table** | `prompts/scripting_prompt.py:32-36` maps the 4
   gear types to `gear` / `gear_helix` / `gear_herringbone` / `gear_bevel` — all 4
   verified present at `spur_gear/gears.scad:175-206`. The library actually exposes
   **6** gear modules (adding `Rack`/`rack` at L260-265 and `gear2D` at L243); the prompt
   exposes only 4. Also `agent.py:56` references `function_response` before assignment
   when `finish_reason != "stop"` and there is no `function_call` — a latent
   `UnboundLocalError`.

ALREADY COVERED: items 1 and 4's math, by `domain/library/gear_train` plus
`kinematics/gear_modules`.

VERDICT: **already-covered** (and licence-blocked from vendoring anyway).

---

## CAD2Program-gh-pages (23 files)

LICENSE: **CC BY-SA 4.0** (`LICENSE:1` — `Attribution-ShareAlike 4.0 International`) ->
share-alike, **manifest-only, vendor NOTHING.**

WHAT IT IS: a GitHub Pages academic project page. One `index.html` plus Bulma CSS,
FontAwesome, carousel JS, 6 images and a webm.

READ (in full): the file tree; `index.html` structure.

SKIMMED-NOT-READ: the ~19 vendored Bulma/FontAwesome/carousel asset files.

FINDINGS: **none.** Zero code, zero data, zero ground truth. It is a paper landing page.

ALREADY COVERED: the paper's substance is already mined from elsewhere —
`domain/reconstruction/tokens/cad2program`, `domain/reconstruction/translate/shape_program`,
`domain/reconstruction/evaluate/primitive_match_metrics`, `domain/drawings/canvas_layout`,
`domain/drawings/view_lifting`.

VERDICT: **nothing-here.**

---

# Cross-cutting summary — text-to-CAD family

## Verdicts at a glance

| Repo | Files | License | Verdict |
|---|---:|---|---|
| Text-to-CAD-dean | 16,986 | **NONE** | already-covered / nothing-here (settled) |
| CADTestBench-main | 4,842 | MIT (data: unclear) | already-covered + 1 real bug |
| text-to-cad-main | 1,733 | MIT | **mine-further** (4 items) |
| text-to-cad-better | 261 | MIT | **mine-further** (3 items) |
| CADAM-master | 363 | **GPL-3** | already-covered (manifest-only) |
| cadsmith-main | 24 | MIT | already-covered + 4 fixtures |
| BlenderLLM-main | 24 | Apache-2.0 | already-covered |
| text-to-cad-ui-main | 112 | MIT | already-covered |
| CAD-GPT-main | 45 | **LGPL-2.1** | already-covered (manifest-only) |
| Text-to-CadQuery-main | 30 | **NONE** | nothing-here (facts only) |
| ScadLM-main | 34 | MIT | **nothing-here** |
| CQAsk-main | 34 | MIT | **nothing-here** |
| CAD2Program-gh-pages | 23 | CC BY-SA 4.0 | **nothing-here** |

## The two campaign leads, resolved

**1. "Text-to-CAD-dean is just venv noise" — TRUE, and now proven.** 16,982 of 16,986
files are `venv/`. Four first-party files, two of them text, 137 lines of Python. All
three extractable ideas are already in the harness under `t2cdean_*` module names. The
prior audit reached the right answer by a method that could not establish it (a grep
that timed out *because of* the venv); the answer has now been re-derived by exhaustive
`os.walk` classification, which cannot time out. **Settled; do not revisit.**

**2. "4 of 6 category IDs diverge from the real dataset" — CONFIRMED, and localized to
CADTestBench.** `runner.py:53-60` `_CATEGORY_ABBREV` is the authoritative on-the-wire
vocabulary; the harness's `test_assertions.py:37-45` `CATEGORIES` differs in exactly
four of six strings:

| CADTestBench (authoritative) | Harness | Match |
|---|---|---|
| `solid_shell_validity` | `solid_shell_validity` | yes |
| `dimensions_ratios` | `dimensions_ratios` | yes |
| `topology_checks` | `topology` | **NO** |
| `geometry_types` | `geometric_types` | **NO** |
| `volumetric_checks` | `volumetric` | **NO** |
| `spatial_arrangement` | `spatial` | **NO** |

Consequence: any harness code pointed at genuine CADTestBench parquet rows would bucket
every test into `uncategorized` and per-category accuracy would silently collapse. Fix
is a 6-entry alias map; no vendoring required.

An independent search of the ten small repos for the same pattern found **no category-ID
lists at all** (`grep -rniE "categor"` across every source and data file in all ten
returned 3 files, all false positives: a Svelte time-bucket loop variable, a comment
about *token* categories, and README prose). So the lead belonged to CADTestBench alone.
A separate, smaller divergence does exist in Text-to-CadQuery — **2 of 6 model IDs**
diverge from its README (`train_gpt2_medium.py` actually trains `gpt2-large`, and
`train_qwen-3B.py` reads different data files) — which is a real correctness finding in
its own right but is not the "4 of 6" claim.

## Genuinely unmined, ranked

1. **SDF-probe mesh-orientation test plus grid-step-derived tolerance ladder** —
   `text-to-cad-main`, `packages/implicitjs/src/lib/implicitCad/meshQuality.js:110-243`.
   Verify a mesh's face orientation against the field that generated it; every tolerance
   scaled to the marching grid step and bbox diagonal.
2. **G-code static-validation rule set** — `text-to-cad-main`,
   `skills/gcode/references/gcode-validation.md:7-32`. A confirmed **0-module** gap
   (`grep -ic "gcode"` over 1579 modules = 0).
3. **SDFormat validator check catalogue** — `text-to-cad-main`,
   `skills/sdf/references/validation.md`. Also a confirmed 0-module gap; **not** to be
   confused with the harness's 22 signed-distance-field modules.
4. **The clarification ask/don't-ask predicate** — `text-to-cad-better`,
   `references/natural-language-specs.md:83-99`. No harness module gates when an agent
   may block.
5. **The "must-not-infer" guardrail list** — `text-to-cad-main`,
   `skills/sdf/references/llm-guardrails.md`. A *pre-generation* constraint table; the
   harness's taxonomies are all post-hoc.
6. **CADTestBench category alias map** — a 6-entry dict fixing a live interoperability
   bug.
7. **Two repair-taxonomy classes** — "selector fragility" and fillet-radius-vs-edge, from
   `text-to-cad-better/references/repair-loop.md`, not found in the harness's taxonomies.
8. **cadsmith's 4 literal pass/fail geometry fixtures** — `tests/test_validator.py:11-50`.

## Licence fences to respect

- **Vendor NOTHING:** Text-to-CAD-dean (no licence), Text-to-CadQuery (no licence),
  CADAM (GPL-3), CAD-GPT (LGPL-2.1), CAD2Program (CC BY-SA 4.0).
- **CADTestBench's 400 prompts derive from CADPrompt**, which its MIT LICENSE does not
  cover. Manifest-only pending a CADPrompt licence check — the same failure mode that
  already cost Graph-CAD its 700 tasks.
- **text-to-cad-better's copyright line ("Thompson Labs") differs from text-to-cad-main's
  ("earthtojake")** for what appears to be the same skill pack. Reconcile before
  re-vendoring either.

## Confirmed absences (stated plainly, because they save future work)

- **CADTestBench commits no ground truth.** All CADTests are a HuggingFace network
  dependency; zero JSON, zero parquet, zero STL in 4,842 files. Its 4,800-file baselines
  tree is **3x byte-identical redundancy** and its 800 model outputs are **unlabeled**.
- **CADTestBench has no judge rubric, no error taxonomy beyond six categories, no refusal
  predicate, and no geometric-similarity metric** — by explicit design; the paper argues
  against exactly those.
- **No repo in this family contains an expected-rejection task.** A scan of all 400
  CADTestBench prompts for `impossible|cannot|invalid|not possible|ambiguous|refuse`
  matched **zero files**. Every prompt everywhere is a satisfiable build request. If the
  harness wants refusal ground truth, it will have to author it.
- **text-to-cad-main has no unit-conversion logic at all** (4 grep hits across 36 cadpy
  modules, all trivial); everything is hard-assumed mm.
- **text-to-cad-better's 10 benchmark briefs are git-LFS pointer stubs in this checkout**
  — already vendored from a different checkout at `eval/bench/imports/textcad/`.
- **ScadLM ships a file with an unresolved merge conflict** (`backend/prompts.py`) and
  **Text-to-CadQuery ships a broken import** (`gemini_pipeline.py:6`); neither runs as
  committed.
