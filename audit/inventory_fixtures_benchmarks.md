# Deep inventory: missed test fixtures and benchmark material in resources/cad_repos

Read-only sweep (2026-07-16). Ground truth verified first: `src/harnesscad/` contains
NO committed geometry or task data files at all (only `eval/gates/*_baseline.json` and
Rust build artifacts). Every corpus in the harness is procedurally generated -- the
hard corpus from `eval/hardcorpus/generate.py` factories, the fleet-audit known-good/
known-bad corpora from hand-written synthetic op streams, defect injection on synthetic
meshes. So every real-part fixture below has harness equivalent = NONE unless noted.

All paths under `resources/` (repos double-nest, e.g. `cad_repos/X-main/X-main/`).

## Top-10 highest-value imports (ranked)

1. **Graph-CAD `CADBench.jsonl`** -- 700 rubric-annotated tasks (id, instruction,
   per-task rubric criteria tree: shape/color/proportion/spatial). Largest
   judge-calibration + brief corpus available; one JSONL loader.
2. **modeling-app (Zoo) `public/kcl-samples/` + `manifest.json`** -- 103 realistic,
   human-described, categorized parts (ball-bearing, bone-plate, axial-fan,
   car-wheel-assembly). Transforms the hard corpus from ~10 procedural families to
   real-world diversity; KCL sources are reference-solution existence proofs.
3. **BRepNet `tests/test_data/`** -- `simple_solids/*.step` (18 curated known-good) +
   `issues_16/*.step` (11 real ABC parts that broke their pipeline = known-bad) +
   `example_files/step_examples/*.stp` (~27 ABC parts). The only curated
   known-good/known-bad STEP pair-set in the tree; feeds fleet_audit-style precision
   corpora and STEP-ingest canaries.
4. **cadgenbench `tests/fixtures/geometry/`** -- rotation/translation twins of the same
   part (l_bracket +/-45deg, tapered box) + `open_shell.step` (explicit invalid-solid
   oracle). Exactly the real-file regression data `invariance.py`/`canonical_pose.py`
   lack.
5. **manifold `test/models/`** -- `self_intersectA/B.obj`, `openscad-nonmanifold-crash.obj`,
   boolean pairs + `polygon_corpus.txt` triangulation fuzz corpus. Real crash-reproducer
   meshes = strongest adversarial inputs for defect_injection verifier scoring.
6. **AgentSCAD `benchmarks/{simple,medium,hard}/*.json`** (14) -- machine-checkable
   tasks: prompt + difficulty + required_features + expected_bbox + tolerances.
   Closest existing format to `hardcorpus.contract_grader`.
7. **CAD-Coder `inference/`** -- 100 GT STEPs + images + reference CadQuery code JSONL.
   Ready-made 100-part held-out geometry-similarity eval set.
8. **CADTestBench `baselines/`** -- Claude-4.6-Sonnet + GPT-5.2 baseline outputs across
   Abstract/Detailed prompt styles. Seeds eval/leaderboard with real comparators
   (task suite itself is an HF download -- external dependency).
9. **IntentForge `src/benchmark/{prompts,expected}/`** -- prompt -> expected-feature-state
   AND prompt -> expected-REJECTION oracle pairs. The only expected-rejection data
   found; plugs the refusal/underspecification gap ambiguous.py tests without labels.
10. **CADAM `benchmarks/01..10` + text-to-cad `benchmarks/01..10`** -- 20 graded briefs
    (twisted vase -> herringbone planetary gearbox) each with reference .scad solution
    + render. Extends the hard corpus into genuinely hard territory.

Honorable mentions: curated-code-cad `birdhouse/` (same part in 8 languages -- natural
cross-backend differential oracle), cad-cae-copilot `analytical_fea/reference.json`
(closed-form FEA oracles: cantilever/buckling/modal) + honesty-scoring leaderboard
rubric, CADCLAW BOM good/bad sextet (1 good + 5 labeled-wrong), Text2CAD German
dimensioned prompts + worked STEPs (multilingual canaries), Roshera-CAD 9 scenario
specs incl. 08-saddle-honesty trap, mrCAD `test_design_pool.jsonl` (real human 2D
designs), cad-judge 3-tier abstraction prompts, cadquery DXF/STEP import edge-cases,
Sim-Correct `opencad_forearm_artifact.json` (OpenCAD artifact conformance canary),
ezpz proptest-regression seeds, ImplicitCAD `tests/golden/` (29 program->golden pairs).

## Skips (verified)

BikeBench metrics already ported (`eval/bench/bikebench_metrics.py`) -- only its prompt
battery/baselines optionally importable. SKIP-irrelevant: BrickGPT golden pairs (brick
domain), RapCAD/.curv conformance cases (language-specific syntax; mine the categories
only), solvespace .slvs binaries, cadquery-plugins FCStd, CascadeStudio/spatialhero
inline fixtures, SketchGraphs tar/pickle sample (mrCAD pool is better), hnc-cad split
ids, muse name list, OpenCAD-main golden-loop (inline expectations, already mined).

## Coverage note (honest)

Fully inspected: every fixtures/golden/testdata/benchmark(s) dir at depth <=4, all
geometry files in test/fixture/example paths (first ~120 listed), all
benchmark-manifest JSON/YAML/JSONL hits, targeted reads of ~35 repos.

Partially inspected: CADTestBench baselines (structure sampled), cad-cae-copilot
(benchmarks/ read, not all benchmark_runs/), manifold (data listed, C++ expectations
not extracted); geometry listing truncated ~120 entries.

Known likely-missed troves for follow-up: **pythonocc-core-master** and
**ruststep-master** (both typically ship committed STEP/BREP parse fixtures) -- run
`find <repo> -iname "*.step" -o -iname "*.brep"` on those two. ML repos
(DeepCAD/SkexGen/GenCAD/cadrille/vitruvion/UV-Net/...) ship download scripts, not
committed fixtures (spot-checked). libfive/OpenJSCAD nested test dirs not enumerated.
