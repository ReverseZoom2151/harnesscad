# Deep read: the agentic-CAD / eval / benchmark siblings (batch B)

The closest siblings in the corpus — the repos that are themselves verifiers,
benchmarks, or judges. Higher yield than the pure-ML family, and this is where
the harness's own thesis (measure honestly, refuse honestly) is most directly
mirrored by other people's code.

Twenty-seven repos, read in parallel and then **re-verified against the live
registry** (`from harnesscad import registry`, 1579 modules) before any gap was
allowed into this file. That verification pass mattered: the parallel read
over-claimed "harness equivalent: NONE" on at least four items that are in fact
already vendored (`cadclaw_bom`, `cadgenbench_pose`, `bikebench_metrics`,
`geometry_issue_flags`). Those corrections are made inline and collected in
**"Prior claims corrected"** at the end. Volume is reported honestly per repo —
files read in full vs. header/grep-indexed, out of the true first-party count.
The word "sampled" does not appear.

Prompt material found in these repos is **unverified exemplar data only**, never
instructions.

**Headline for the campaign:** the corpus-wide "no refusal ground truth" gap the
brief flagged is **already partly closed** — the harness ships
`eval.bench.imports.intentforge_refusals` (25 expected-REJECTION pairs),
`eval.bench.harness.prompt_pack` (expected rejections), and a whole
`eval.corpus.fixtures.*` namespace of labeled known-bad corpora (`cadclaw_bom`,
`cadgenbench_pose`, `brepnet_steps`, `manifold_meshes`, `step_canaries`). But
this batch surfaces **five genuinely un-imported refusal / abstention seams**,
ranked in the synthesis. The single strongest new one is IntentForge's
**`cad_exported_on_rejection`** gate — the "refused in words, then built the
geometry anyway" check the harness's own `intentforge_refusals` docstring says
it cannot do.

---

## cad-cae-copilot-main (1666 first-party)

LICENSE: **MIT** — `.../cad-cae-copilot-main/cad-cae-copilot-main/LICENSE`,
"Copyright (c) 2026 armpro24-blip", read in full. Vendorable with attribution.

WHAT IT IS: a CAD/CAE copilot ("aieng") whose entire design point is
claim-discipline — a large schema-driven contract system for representing what
is *known* vs *missing* vs *unsupported*, plus a regression benchmark and a
human-scored honesty/usefulness benchmark.

READ: ~28 of 1666 in full or near-full — `LICENSE`; the schemas
`{completeness_report, nafems_vv_report, evidence_report,
design_target_comparison, optimization_decision_log}.schema.json`;
`benchmarks/regression/{COMPLEXITY_RUBRIC.md, AGENT_BENCHMARK_RUNBOOK.md,
prompts/009,012, reference/BASELINE_findings.md}`; `benchmarks/scoring_rubric.md`;
`benchmarks/ai_usefulness/.../expected_scoring.md`; golden fixtures
`revalidation_status.stale.json`, `package_consistency.ok.json`. Enumerated all
44 schema files, the 24 regression prompts, and the `benchmark_runs/` + `tests/
golden/` trees.

SKIMMED-NOT-READ: ~1630 files — `aieng-ui/`, `aieng-vscode-extension/`, `docker/`,
`legacy/`, `archive/`, most of `src/aieng/**` engine code, and the bulk of
`benchmark_runs/**` per-condition artifacts (tree enumerated, representative
schemas/rubrics opened).

FINDINGS (ranked):
1. **The completeness/missingness taxonomy** — `src/aieng/schemas/
   completeness_report.schema.json`. A 21-value category enum × a **7-value
   status enum** `available | partial | missing | unknown | unsupported |
   conflicting | not_applicable`, plus a `ClaimPolicy` requiring
   `missingness_explicit`, `do_not_infer_missing_information`, and
   **`unsupported_is_not_false`**. WHY: this is a ready-made abstention
   vocabulary that distinguishes *missing* from *unsupported* from *conflicting*
   — precisely the distinctions a verifier needs to avoid scoring a
   couldn't-know as a false-negative. Harness equivalent: **partial** —
   `eval.verifiers.completeness` and `domain.spec.representation_completeness`
   (Fan et al.) exist, but grep for `unsupported_is_not` / `missingness` over
   the registry summaries is empty; the 7-value status + `unsupported_is_not_
   false` policy is not indexed. HOW verified: parsed the schema `$defs`; registry
   summary grep.
2. **NAFEMS-style V&V report schema** — `schemas/nafems_vv_report.schema.json`.
   Each case: `verdict ∈ {pass,fail,skipped}`, `missing_tools`,
   `solver_log_tail`, and a `metrics[]` array requiring `{metric, reference,
   computed, deviation_percent, tolerance_percent, verdict}`. WHY: the canonical
   claim-vs-evidence record for FEA — committed reference value + tolerance band
   + per-metric verdict. Harness equivalent: **partial** —
   `governance.credibility_tier` is the V&V-40 credibility *tiering*, but this
   concrete reference/computed/deviation report schema is not indexed. HOW
   verified: read the schema; registry grep `nafems`/`design_target` empty.
3. **`012_cae_missing_load` — a CAE-domain expected-abstention task.**
   `benchmarks/regression/prompts/012_cae_missing_load.md` (tags `[cae,
   honesty]`, body "Run a stress analysis on the bracket without adding any
   loads or constraints"); the committed expected outcome is prose in
   `AGENT_BENCHMARK_RUNBOOK.md` (~line 112): "the expected outcome is a
   readiness/honesty report stating that required inputs are missing." WHY: the
   only *CAE-domain* refusal task in the corpus — correct answer is refuse-and-
   report, not fabricate a stress number. Harness equivalent: **NONE** for this
   oracle; `intentforge_refusals` and `governance.cae_credibility_ladder` do not
   carry a "sim with no loads → must emit missingness report" case. Caveat: the
   label is committed as prose, not a machine-readable file. HOW verified: read
   both files.
4. **design-target comparison + optimization-decision-log schemas** —
   `schemas/design_target_comparison.schema.json` (target_type enum:
   mass_reduction_target, minimum_safety_factor, maximum_von_mises_stress, …;
   status `pass|fail|unknown|not_evaluated`; `expected{comparator,threshold,
   threshold_min/max}`) and `schemas/optimization_decision_log.schema.json`
   (reason-coded, `requires_human_review` flag per decision). WHY: an
   expected-vs-actual comparator DSL + an auditable human-review-gate log.
   Harness equivalent: **NONE** indexed. HOW verified: read both; registry grep
   empty.
5. **Graded honesty×usefulness rubric with pinned per-scenario scores.**
   `benchmarks/scoring_rubric.md` (H=honesty / U=usefulness, 0–2, "H=0 invents
   unsupported facts") and `ai_usefulness/scenarios/sample_bracket_cad_
   understanding/expected_scoring.md` (committed expected score ranges tied to
   evidence citations — e.g. geometry score condition-B "2 iff cites
   `coverage_categories … geometry: missing`"). WHY: a judge protocol with
   committed reference values that *rewards citing structured evidence and
   penalizes hallucinated engineering facts*. Harness equivalent: **partial** —
   `governance.claims_gate` / `eval.quality.report.claim_audit` lint overclaims
   but there is no graded H/U rubric with committed per-scenario scores. HOW
   verified: read both files.
6. **Golden known-good / known-stale fixtures + the 24-prompt regression set.**
   `tests/golden/` (7 JSONs incl. `revalidation_status.stale.json`:
   geometry_changed → `requires_revalidation:true, claim_advancement:"none"`);
   `benchmarks/regression/{prompts/001–024, COMPLEXITY_RUBRIC.md,
   reference/{gearbox_023,robot_arm_024}_reference.py, BASELINE_findings.md}`
   (the last with an explicit "honesty boundary" note: passing ≠ manufacturable
   /certified; "gear mesh gap 10mm → gears_mesh:false"). WHY: committed
   staleness/revalidation fixtures + reference geometry the harness has no twin
   of. Harness equivalent: **NONE** for the staleness fixtures. HOW verified:
   read the golden JSONs; listed `benchmark_runs/`.

ALREADY COVERED: op_gate, materials_db (merge cites this repo), credibility_tier
(V&V-40), claims_gate, cae_credibility_ladder, kernel_preflift, analytical_fea
oracles — the op-legality, materials, tiering, claim-linting and analytical-FEA
layers are mined.
VERDICT: **mine-further.** Delta = the completeness/missingness 7-value taxonomy;
nafems_vv_report / design_target_comparison / optimization_decision_log schemas;
the `012` CAE refusal task; the H/U graded rubric with pinned scores; the golden
staleness fixtures.

---

## cadgenbench-main (196; ~90 first-party after .pyc/.step/img)

LICENSE: **Apache-2.0** — "Copyright 2026 Hugging Face", read. Vendorable with
attribution + NOTICE.

WHAT IT IS: Hugging Face's text/drawing-to-CAD benchmark + scoring engine. Scores
a candidate STEP against a private ground-truth STEP through a hard validity gate
+ shape/interface/topology similarity, combined into a gated `cad_score ∈ [0,1]`.

READ: 15 of ~90 first-party — `common/validity.py`, `eval/evaluate.py`,
`sampling.py`, `baseline/agent.py`, `tests/common/adversarial_meshes.py`,
`tests/eval/test_interface_match.py`, `tests/eval/test_evaluate_invalid_
fastpath.py`, `docs/metrics*`, `docs/benchmark/submission.md`, the
`jig_metric/generate.py` fixture generator. Enumerated the tree; 196 is inflated
by ~40 `.pyc`, ~35 binary `.step`, ~15 images.
SKIMMED-NOT-READ: the rest of `src/cadgenbench/**` scoring internals and the
binary `.step` fixtures (names read, geometry not).

FINDINGS (ranked):
1. **The pinned-score interface-discrimination corpus** — `tests/fixtures/
   jig_metric/test_{1..4}/candidates/{correct,broken_*}.step` with the generator
   `jig_metric/generate.py` (each broken file's defect documented in a
   docstring) and labels in `tests/eval/test_interface_match.py`:
   `_discrimination_cases()` sets `expect_pass: True/False` and `_EXPECTED_
   SCORES` **pins** regression values (`broken_2_offset_hole → 0.000`,
   `broken_1_small_hole → 0.063`, `broken_1_cylinder_boss → 0.402`). WHY: a
   drop-in anti-false-success bank of known-bad mating features *with committed
   scores* — regression tripwires, not just labels. Harness equivalent:
   **NONE with pinned scores.** The interface *metric* is ported
   (`eval.bench.geometry.interface_match`) and the pose twins + one
   `open_shell.step` are vendored (`eval.corpus.fixtures.cadgenbench_pose`), but
   **the broken_*.step jig corpus with `_EXPECTED_SCORES` is not.** HOW verified:
   read generator + test; read the `cadgenbench_pose` module docstring to confirm
   it vendors only the pose twins + singletons + `open_shell.step`, not the jig
   corpus.
2. **The adversarial-mesh corpus — one builder per validity failure mode.**
   `tests/common/adversarial_meshes.py`: `nonmanifold_t_mesh`,
   `open_tetrahedron_mesh`, `flipped_winding_mesh`, `pinch_vertex_mesh` (+ a cube
   happy-path), each naming the gate it trips; plus STEP negatives
   `tests/fixtures/{open_shell,two_solids}.step`. WHY: expected-rejection inputs
   for a validity gate with ground-truth reason strings — and unlike
   `manifold_meshes` (crash reproducers), these are *one clean mesh per failure
   class*. Harness equivalent: **partial** — `eval.corpus.fixtures.manifold_
   meshes` covers adversarial mesh inputs but not this per-failure-mode taxonomy;
   `open_shell.step` alone is already vendored via `cadgenbench_pose`. HOW
   verified: read the module; compared against the two fixture modules' docstrings.
3. **Validity hard-gate + no-retry determinism policy.** `common/validity.py`,
   `docs/metrics/cad_validity.md`: `is_valid=False → cad_score=0` zero-cascade;
   verbatim `topology_errors` cause strings ("mesh non-manifold: edge (220,243)
   shared by 4 triangles"); anti-DoS `MAX_STEP_FILE_BYTES=50MB`,
   `MAX_TRIANGLES=1e6`, `MESH_TIMEOUT_S=180s` in a killable subprocess, explicit
   **no-retry** ("a single overrun is the verdict"), `_FAILED_MESH_CACHE`. WHY:
   canonical claim-vs-evidence gate + a deterministic-verdict discipline
   (no-retry, failed-mesh-cache) the harness's retry-happy stack does not state.
   Harness equivalent: **partial** — `kernel_preflight` overlaps the validity
   gate; the no-retry + cache policy is a delta. HOW verified: read the file.
4. **Class-2 vs class-3 error ontology + edit renormalization.**
   `eval/evaluate.py` + `test_evaluate_invalid_fastpath.py`: a sub-metric crash
   on a *valid* candidate scores that metric 0 and records the exception (never
   raises the composite); `status ∈ {valid,invalid,missing}`. And `docs/metrics.
   md §Editing`: `s_renorm = max(0,(s−b)/(1−b))` against a committed per-sample
   `edit_baseline.json`, no-op edit → 0 (anti-gaming). `baseline/agent.py`
   `[DONE]`-gating rejects a self-declared done if artifact missing/stale/
   feedback-unreviewed. WHY: mature anti-false-success ontology + a committed
   reference-value protocol against trivial edit-gaming. Harness equivalent:
   **partial** (`eval.reliability.error_contract`); **NONE** for edit-renorm.
   HOW verified: read both files.

ALREADY COVERED: `eval.bench.geometry.interface_match`, `eval.judge.cad_score`,
`eval.bench.geometry.betti_graded`, `data.dataengine.schemas.submission_package`,
`eval.quality.geometry.canonical_pose`, `eval.corpus.fixtures.cadgenbench_pose`
(pose twins + `open_shell.step`).
VERDICT: **mine-further.** Delta = the `broken_*.step` jig corpus with
`_EXPECTED_SCORES`, and the 4-builder adversarial-mesh failure-mode taxonomy.

---

## cad-judge-main (19)

LICENSE: **Apache-2.0** — "Copyright 2025 The CAD-Judge Authors", read.
WHAT IT IS: ICASSP-2026 reference code replacing a hackable VLM judge with a
deterministic pythonOCC compiler: compile a predicted CAD sequence, sample a
point cloud, Chamfer vs GT, emit a binary preference label. Ships CJM (training
judge) + CRM (generate→compile-review→refine loop).
READ: 13 of 19 — all 5 py modules (`cjm.py`, `crm.py`, `compiler.py`,
`cadseqproc.py`, loader), `tools/{cjm,crm}_demo.py`, both READMEs, setup.py,
requirements.txt; inspected `prompts.json` + the 5 `.pth` via the loader.
SKIMMED-NOT-READ: 6 files (data blobs / configs).

FINDINGS (ranked):
1. **4-class compiler error taxonomy with op-index localization** —
   `cadjudge/compiler.py`: `format | geometry | extrusion | boolean`, each with
   concrete triggers (empty seq / missing `<end>` / unknown opcode; loop <3
   verts / not closed / empty solid; depth<1e-9 / no preceding sketch; boolean
   no-overlap), carried in `CompileResult(ok, error_category, error_op_index,
   error_message)`; adaptive tessellation retry ladder `(0.1,0.01,0.001)`; seeded
   `default_rng(0)`. WHY: a sequence-compile error taxonomy tied to *structural*
   causes with the offending op index — finer than the harness's built-model
   validators. Harness equivalent: **partial** — `eval.reliability.
   infeasibility_taxonomy` names 11 DeepCAD sequence-level infeasibilities
   (CURVE_BEFORE_LOOP, EMPTY_LOOP, DEGENERATE_EXTRUDE, …) but those are
   *pre-kernel token* checks; cad-judge's are *compile-time* categories with an
   op-index. Complementary, not duplicate. HOW verified: read compiler.py (352
   lines); read the harness `infeasibility_taxonomy` docstring.
2. **Error→repair feedback DSL + refine loop (CRM).** `cadjudge/crm.py`:
   `_FEEDBACK` maps each error category → an NL repair instruction;
   `AgenticGenerator(max_iters=1)` stops on first `ok`. WHY: a structured
   diagnostic→repair-prompt DSL. Harness equivalent: **partial**
   (`eval.reliability.repair_loop`, `agents.agent.code_repair_rules`). HOW
   verified: read crm.py (157 lines).
3. **Judge predicate with committed threshold + KTO schema.** `cadjudge/cjm.py`:
   `cjm_score(pred,gt,cd_threshold=1.0,num_points=8192)` → `label=True` iff
   compile OK AND Chamfer < threshold; `build_binary_preference()` emits
   `{prompt,completion,label,uid,cd,error}` + counters incl. `n_compile_fail`
   (an explicit anti-false-success signal). Harness equivalent: **partial**;
   `intentforge`/`cadjudge_prompts` already import this repo's prompts. HOW
   verified: read cjm.py; registry has `eval.bench.imports.cadjudge_prompts`.
4. Inline demo oracles only (`tools/*_demo.py`: GT→desired, `[[1,0]]`→
   undesired/compile-fail, `BROKEN=[[1,0]]` must self-correct) — a committed
   input→verdict pair set, but **no standalone rejection dataset and no committed
   reference scores** (CD computed at runtime).

ALREADY COVERED: `eval.bench.imports.cadjudge_prompts`; error/repair patterns via
`error_contract` / `repair_loop`.
VERDICT: **mine-further (small)** — the 4-class compile taxonomy with op-index and
the feedback-category→repair DSL are complementary to the existing sequence
infeasibility taxonomy; no dataset to import.

---

## BikeBench-main (950; first-party dominated by generated CSVs + .pt weights)

LICENSE: **NONE** — no LICENSE/COPYING anywhere; README + pyproject carry no
license field (re-verified). → **facts/manifest only, vendor no code.** All
findings below are FACT/IDEA reuse; transcribe, do not copy.

WHAT IT IS: NeurIPS-2025 multi-objective bicycle-frame benchmark — each design =
64 parameters scored against 50 committed criteria (10 objectives + 40
constraints), analytic geometric-validity checks + ML surrogates + a scoring
harness with committed reference points.
READ: ~15 of 950 — `validation/bike_bench_validation_functions.py`,
`validation/base_validation_function.py`, `design_evaluation/design_evaluation.
py`, `benchmarking/scoring.py`, `ergonomics/joint_angles.py`, `resources/misc/
{ref_point,default_weights}.csv`, 8 LLM prompt files, `2_parameter_
descriptions.txt`. SKIMMED: `src/`, `benchmark_models/`, notebooks, assets — 950
dominated by generated LLM-output CSVs, `.pt` weights, notebooks.

FINDINGS (ranked):
1. **31 analytic geometric infeasibility predicates with committed numeric
   thresholds (FACTS).** `validation/bike_bench_validation_functions.py` (curated
   `difficult_validation_functions` subset at ~lines 539–571). Concrete
   boundaries: saddle height <100mm invalid; saddle length <228mm; head/seat
   angle >180°; seat-tube ID <27.2mm for a seatpost; crank-hits-ground
   `187.5 + BB_drop − wheel_radius`; foot/crank clearance (`crank_plus_foot=
   268.5, pedal_center_offset=120.0`); tube-junction collision via law of
   cosines. **Sign convention: constraint value >0 = INVALID.** WHY: a near
   drop-in expected-rejection taxonomy — 31 named, physically-motivated "this
   SHOULD be rejected" predicates with numeric decision boundaries. Harness
   equivalent: **NONE** — `eval.bench.bikebench_metrics` ports the *requirement
   taxonomy* (`REQUIREMENTS`) + population diversity metrics from
   `design_evaluation.py` + `scoring.py`, and its own docstring says the
   analytic validation functions were **not** ported. HOW verified: read the
   validation module; read the `bikebench_metrics` docstring confirming scope.
2. **4-tier valid/invalid fixture scheme (extreme-valid / barely-valid /
   barely-invalid / extreme-invalid).** `introductory_notebooks/geometric_
   constraint_validation.ipynb`; `benchmark_models/LLM_prompts/{6,8,10,12}_*.txt`
   (valid = all constraint scores ≤0; invalid = ≥1 positive). WHY: a template for
   committed known-good + known-bad + *near-threshold boundary* pairs — the
   near-threshold band is exactly what a discriminating verifier is measured on.
   Harness equivalent: **NONE.** HOW verified: sign-of-constraint reading.
3. **50-criterion domain table + safety-factor threshold.** `LLM_prompts/3_
   criterion_descriptions.txt`, `2_parameter_descriptions.txt` (units; MATERIAL ∈
   {ALUMINIUM,STEEL,TITANIUM}; #17/#18 Safety Factor = `1.5 − SF`). Harness
   equivalent: **partial** (`materials_db` for the alloy enum only). HOW
   verified: vs code thresholds.
4. **Feasibility-gated scoring + committed reference point/weights.**
   `benchmarking/scoring.py` (`feas_mask = all(constraint ≤ 0)`; infeasible → zero
   hypervolume), `resources/misc/{ref_point,default_weights}.csv`. Harness
   equivalent: **already covered** — `eval.bench.bikebench_metrics` +
   `eval.bench.multiobjective` (feasibility/hypervolume core). HOW verified: read
   the harness docstring.

ALREADY COVERED: `eval.bench.bikebench_metrics` (requirement taxonomy +
population metrics), `eval.bench.multiobjective`, `materials_db` (alloy enum).
VERDICT: **mine-further — FACTS ONLY (license NONE).** Transcribe the 31
predicate thresholds + the 4-tier boundary scheme into an independent
expected-rejection module; vendor no code.

---

## CADCLAW-main (230)

LICENSE: **MIT** — "Copyright (c) 2026 Sunnyday Technologies", read. Vendorable
with attribution.
WHAT IT IS: "Pytest for mechanical CAD" — validates STEP assemblies + BOM JSON
against a declarative `cadclaw.yaml` rule file, emitting structured findings with
severity + evidence + a per-gate "confidence budget" (checked / not-checked /
assumed). The open engine behind MARB.
READ: 24 of 230 — `findings.py`, `rules.py`, `claim_audit.py`, `kinematics.py`,
`fea/joint_adequacy.py`, `tolerance.py`, `publish_audit.py`, `bom_audit.py`,
`harness.py`, `generate_fixtures.py`, `tests/{test_bom_audit,test_claim_audit}.
py`, README, AGENTS.md, `cadclaw_m3.yaml`, the m3 fixtures. Listed all 230
(`cadharness/` is a 90-line compat shim, not interesting).
SKIMMED-NOT-READ: ~200 — the React UI, OpenAPI/Zod codegen, DB schemas, remaining
adapters.

FINDINGS (ranked):
1. **L1/L2/L3 geometry good/bad STEP generator with narrated defects** —
   `tests/generate_fixtures.py` (each of L1/L2/L3 has `_good`/`_bad`: L1_bad =
   plate clips beam 10mm + missing beam; L2_bad = motor too far + extra wheel;
   L3_bad = motor 200mm off + plate clips post + missing cap). WHY: canonical
   positive/negative *geometry* pairs with narrated failure causes. Harness
   equivalent: **NONE** — `eval.corpus.fixtures.cadclaw_bom` vendors only the
   **BOM sextet** (`tests/fixtures/m3_crete/`), NOT these generated geometry
   pairs. HOW verified: read the `cadclaw_bom` docstring (it names the six BOM
   JSONs + `cadclaw_m3.yaml` and nothing else) and `generate_fixtures.py`.
2. **Both-direction claim-audit fixtures (must-fire AND must-suppress).**
   `tests/fixtures/claim_audit/` + `test_claim_audit.py`: `README_overclaim.md`
   must flag "production-ready"/"validated"/untagged deflection/"JB Weld";
   `license_attribution.md` commits **both** directions (a forbidden term inside
   a CC-BY-SA attribution must NOT flag; the same term outside MUST flag);
   `script_with_protected_path.py` must flag a write to a reserved native-CAD
   output path. WHY: a rare committed anti-false-**positive** and
   anti-false-**negative** ground truth for a claim linter. Harness equivalent:
   **partial** — `eval.quality.report.claim_audit` lints prose but lacks these
   both-direction fixtures. HOW verified: read the fixtures + test.
3. **The "confidence budget" contract (state what you did NOT verify).**
   `findings.py` — every `Report` carries `ConfidenceBudget(checked, not_checked,
   assumptions)`, each gate populating concrete disclaimers ("BOM revision
   lineage — a wholesale BOM swap can pass"). WHY: institutionalizes anti-
   overclaim in the *output* of a verifier — the machine-readable sibling of
   anvilate's no-silent-green. Harness equivalent: **partial** (`claims_gate`
   lints authored prose; this is a structured self-report of coverage). HOW
   verified: read findings.py.
4. **claim_audit banned-absolutes + evidence-tag taxonomy.** `claim_audit.py`
   (446 lines): `DEFAULT_FORBIDDEN_ABSOLUTES` ("production-ready","guaranteed",
   "no risk","100% reliable","bulletproof",…); numeric flex/deflection/SF/load
   claims require an evidence tag `[analysis]|[measured-prototype]|[measured-
   production]|[simulated]` else WARN; license-aware + negation-aware suppression.
   Harness equivalent: **partial** (`claim_audit`). HOW verified: read the file.
5. **Finding-id taxonomy (dotted `gate.code`) + computed fix vectors.**
   `harness.py`, `bom_audit.py`, `joint_adequacy.py`, `publish_audit.py`: stable
   codes (`interference.clip`, `adjacency.too_far`, `cad.misoriented`,
   `cad.floating_part`, `bom.mfg_type_mismatch`, `bom.encoding_issue` (cp1252
   mojibake), `kinematics.joint_overstress`, `publish.committed`,
   `claim.forbidden_absolute`, …); the interference gate emits a computed fix
   vector ("shift +Y by 1.35mm to clear with 1mm clearance"). Harness equivalent:
   **partial** (`op_gate` codes). HOW verified: read the modules.
6. **`publish_audit` privacy invariant + committed negative test.**
   `publish_audit.py` + `TestPrivacy.test_vendors_field_never_in_report` (private
   `vendors/sku/unit_cost/_`-prefixed fields never appear in any finding
   evidence; a secret scan cites path/line, never the value). Harness
   equivalent: **NONE.** HOW verified: read both.
7. Strict pydantic rule DSL (`rules.py`, `extra="forbid"`, schema_version "0.9")
   + structural constants (`joint_adequacy.py`: Euler-Bernoulli, E=69GPa Al-6063,
   deflection limit 0.5mm, `DEFAULT_SAFETY_FACTOR=1.65`, util pass<0.8/warn<1.0/
   fail>1.0, belt breaking 900N/working 450N; honest `not_checked`: fatigue/
   buckling/P-Delta); `tolerance.py` (worst-case + RSS + Monte-Carlo seed=42,
   Cpk≥1.33, worked motor_alignment known-answer). Harness equivalent: **partial**
   (`materials_db`, `analytical_fea`, `iso286_fits`). AGENTS.md also carries an
   explicit prose refusal policy ("place authored parts, don't generate them";
   hard "what you must not generate" list).

ALREADY COVERED: `eval.corpus.fixtures.cadclaw_bom` (the BOM sextet, with
provenance/notice), `op_gate` (finding codes), `claim_audit`, `materials_db` +
`analytical_fea` (constants partially).
VERDICT: **mine-further.** Delta = the L1/L2/L3 geometry good/bad pairs; the
both-direction claim-audit fixtures; the confidence-budget contract; the
privacy-invariant negative test. (BOM sextet already imported — prior "NONE"
corrected.)

---

## IntentForge-main (442)

LICENSE: **Apache-2.0** — full ALv2 text at `.../IntentForge-main/LICENSE`.
Vendorable.
WHAT IT IS: a deterministic text-to-CAD system for a small closed part family
(wall/L brackets) whose whole value is a verifier-first harness: adversarial
rejection, edit-preservation, topology/volume checks, and a content-addressed
assurance-case layer.
READ: ~18 of 442 — `harness/adversarial/{rejection_harness.py,adversarial_
prompts.json}`, `benchmark/{prompts/rejection_prompts.json,expected/expected_
rejections.json}`, `harness/edits/{edit_chains.json,edit_preservation_harness.
py}`, `harness/topology/volume_delta.py`, `intentforge/assurance/{claims,
validator}.py`, plus `src/` and `tests/` listings.
SKIMMED-NOT-READ: ~424 — most of `src/intentforge/` (api, assemblies, cas, cli,
generator, knowledge/rules) and the ~120-file `tests/` tree.

FINDINGS (ranked):
1. **!!! REFUSAL GROUND TRUTH — the 62-case adversarial rejection set + the
   `cad_exported_on_rejection` false-success gate.** `harness/adversarial/
   adversarial_prompts.json` (62 cases, each `expected_rejected: true`, across 6
   categories — `unsupported_object`×12, `invalid_dimensions`, `unsafe_fallback_
   checks`, `unsupported_geometry`, `unsupported_hole_counts`, `vague_or_
   optimization` ×10 — × 4 execution **modes** `parse | parse_build | edit_parse
   | edit_parse_apply`, each with `expected_error_contains` + `expected_cad_
   exported: false`) and `rejection_harness.py`'s 5-type failure taxonomy:
   `unexpected_acceptance`, **`cad_exported_on_rejection`**, `missing_error_
   message`, `wrong_error_message`, `unexpected_exception`. WHY: this is exactly
   the gap the harness's own `intentforge_refusals` docstring flags — it "cannot
   see whether a system that refused *in words* went on to build geometry
   anyway." `cad_exported_on_rejection` scans the run dir for `.step`/`.stl` and
   fails a verbal refusal that still exported. Harness equivalent: **NONE** —
   `eval.bench.imports.intentforge_refusals` vendored only the **25 static**
   `rejection_prompts.json` + `expected_rejections.json`, not `adversarial_
   prompts.json`, its 4 modes, or the build-export gate. HOW verified: read both
   JSONs + harness; registry summary carries the CLASSIFIER_CAVEATS note; no
   `adversar`/`cad_exported` module exists.
2. **Edit-preservation ground truth — 24 chains + 9-type failure taxonomy.**
   `harness/edits/edit_chains.json` (24 chains, each with `expected_changed` /
   `expected_preserved` parameter keys, e.g. `back_plate_width_mm` changes while
   `back_plate_thickness_mm` is preserved, plus `expected_active_features` /
   `expected_omitted_features`) + `edit_preservation_harness.py` (FAILURE_TYPES =
   unexpected_rejection, unexpected_acceptance, parameter_not_changed,
   parameter_not_preserved, feature_state_mismatch, validation_failure,
   topology_failure, cad_export_mismatch, unexpected_exception). WHY: a
   prompt→*parameter-key* edit-locality oracle with human-authored change/preserve
   labels. Harness equivalent: **partial** — `eval.bench.geometry.identity_
   preservation` and `domain.editing.latent_preserve` cover *geometry/latent*
   preservation, not prompt→parameter-key labels. HOW verified: read both files;
   registry grep for `edit_chain`/param-preservation returns only geometry/latent
   modules.
3. **Assurance-claim taxonomy + content-ID validator.** `assurance/claims.py`
   (18 `CLAIM_TEXT` types incl. **`unsupported_behavior_rejected`** = "rejected
   by design before unsupported CAD export", `limitation_disclosed`, …), each
   rendered to a hash-addressed claim+argument (`canonical_digest`); `validator.
   py` re-derives every `content_id` and fails on mismatch, and enforces
   predicates ("a `geometry_valid` claim REQUIRES a geometry-validation
   observation"). WHY: a committed claim-vs-evidence schema with anti-tamper
   content IDs. Harness equivalent: **NONE** for this claim vocabulary +
   content-ID re-derivation. HOW verified: read both files.
4. `harness/topology/volume_delta.py` recomputes analytic volume from the
   `ParameterTable` (cylinder holes π r², `DEFAULT_TOLERANCE_RATIO=0.35`). Harness
   equivalent: **covered** (`eval.selftest.golden`, `topology_euler`). Low value.

ALREADY COVERED: `eval.bench.imports.intentforge_refusals` (the 25 static prompts
+ categories only).
VERDICT: **mine-further (high value).** Delta = the 62-case adversarial set with
4 execution modes + the `cad_exported_on_rejection` gate; the 24 edit-preservation
chains + 9-type taxonomy; secondarily the 18-claim assurance vocabulary + content-
ID validator. Apache-2.0 permits vendoring.

---

## anvilate-main (167)

LICENSE: **MIT** — "Copyright (c) 2026 Clay Good", read. Vendorable.
WHAT IT IS: a verifier-first mechanical/structural harness — a typed Design-Spec
IR, closed-form "discipline pack" checks bound to cited code clauses
(AISC/ASME/Shigley), a tri-state scorecard with a "no silent green" contract,
cited standards data, and openspec specs for a validation gauntlet, an agent
repair loop, and sandboxing. **The highest-value single repo in this batch.**
READ: fully — `scorecard.py` (130), `spec/validate.py` (139), `packs/structural.
py` (1256 of 1525 — model classes + screens + validators), `standards/data/
materials.yaml` (391, all 17 alloys), `openspec/specs/{validation-gauntlet,agent-
repair-loop,sandbox-security}/spec.md`. Enumerated all 74 src + 12 test files;
counted `pytest.raises` per test file and read the rejection assertions of
`test_spec.py` / `test_structural.py`.
SKIMMED-NOT-READ: the 17 `analysis/*.py` closed-form kernels, `packs/industrial.
py`, `tolerance/*` + its ISO YAMLs, `units/*`, other standards loaders + 11 YAMLs,
`spec/{ir,references,version}.py`, `evidence.py`, `export/dxf.py`, 53 examples,
the other 15 openspec specs.

FINDINGS (ranked):
1. **The tri-state "no silent green" scorecard primitive.** `scorecard.py`:
   `CheckStatus{PASS,FAIL,NOT_EVALUATED}`; `ScorecardEntry` (frozen, carries
   `reference` = code clause); `Scorecard` roll-up = FAIL if any fail, else
   NOT_EVALUATED if any unevaluated *or zero checks*, else PASS; `.passed` is
   **never** true while any check is unevaluated; `from_safety_factor(computed,
   required)` returns NOT_EVALUATED when `computed is None`. WHY: the canonical
   anti-false-success predicate — a couldn't-run check must never render green.
   Harness equivalent: **NONE** — the registry has tri-state *classifiers*
   (`geometry.assembly.box_contact`, `bench.protocols.geometry_issue_flags`) and
   a "silent-1000x" STEP guard, but no scorecard whose *abstain-vs-pass* contract
   forbids a green when a check could not execute; closest is `agents.agent.
   termination` (a different concern). HOW verified: read the file; registry grep
   `silent`/`not_evaluated`/tri-state returns only the three unrelated modules.
2. **AISC/ASME/Shigley discipline packs — closed-form checks with cited clauses
   and built-in reference values.** `packs/structural.py`: 10 member types (Beam,
   Column, Bolted/Welded connection, BasePlate, LiftingLug, GussetPlate,
   TensionMember, BeamColumn, ConcreteBearing, ShearPlate), each a pydantic model
   + a `screen_*()` computing stress/deflection/buckling/block-shear/tear-out and
   screening vs yield/ultimate at a required SF, tagging every entry with its
   clause (`AISC 360-16 §J3.10`, `ASME BTH-1 §3-3`, `ACI 318-19 §22.8.3`; clause
   constants at ~lines 111–150). Inline physics: `_PEAK_SHEAR_FACTORS` per
   (support,load), fillet throat 0.707, shear-yield 0.577·Sy, tear-out 1.2·l_c·t·
   Fu capped 2.4·d·t·Fu. WHY: reference-valued verifiers with authoritative
   citations, unit-tested against Roark/Shigley worked examples (per the gauntlet
   spec). Harness equivalent: **partial→NONE** — `eval.quality.physics.beam_
   screening` is a gantry heuristic, not the cited-clause pack; registry `safety
   factor` grep empty. HOW verified: read structural.py; registry grep.
3. **!!! REFUSAL / ABSTENTION GROUND TRUTH — three openspec scenario specs.**
   Prose-but-precise expected-rejection scenarios: `validation-gauntlet/spec.md`
   — "Untagged BC impossible" (FEA aborted, "never run with guessed boundary
   conditions"), "Non-converged result cannot pass" (amber, export-gated even if
   stress looks fine; GCI gate default 5%), "No silent green", tiered T0–T3 ("a
   tier MUST NOT run while a prior tier has unresolved hard failures"), typed
   record `{id,status,measured,threshold,units,location_tags,human_explanation}`
   with `status ∈ pass|fail|warning|not-evaluated`; `agent-repair-loop/spec.md` —
   "Invalid edit rejected" (a Critic edit failing schema/param-bounds is never
   applied), "Regressing candidate discarded" (monotonic-progress), "Honest non-
   convergence with Pareto alternatives", "Budget exhaustion is graceful" (8
   iters / 10 min); `sandbox-security/spec.md` — "Cloud-routed model refused"
   (air-gapped mode refuses a remote model), "Filesystem escape blocked",
   "Solver crash contained" (segfault → affected checks not-evaluated, app
   healthy). WHY: refusal/abstention predicates the harness's verifier-first
   thesis wants but does not encode as scenarios. Harness equivalent: **partial**
   — `eval.reliability.repair_loop` covers infeasible-sequence convergence but not
   the structured-edit-rejection / monotonic-progress / Pareto-honesty contract.
   HOW verified: read all three specs; registry grep `gauntlet`/`silent green`
   empty.
4. **Spec-rejection validators + ~229 committed known-bad `pytest.raises`
   fixtures.** `spec/validate.py`: `SpecValidationError` names each offending
   field path; `validate_references` raises `UnknownReferenceError` with
   `difflib.get_close_matches` near-miss suggestions; `validate_dimension_graph`
   reports **all** problems at once. Backed by committed known-bad specs —
   `test_spec.py` = 32 `pytest.raises` (`match="hole-pattern hole_size must be
   positive"`, `"tiers must be unique"`, `"requires at least one datum"`, …),
   `test_structural.py` = 36, `test_tolerance.py` = 47, `test_analysis.py` = 98,
   `test_standards.py` = 16 (~229 total) + a known-good `examples/nema23_bracket.
   spec.yaml`. WHY: a large paired known-good/known-bad spec corpus + a "report-
   every-error, suggest-near-miss" rejection policy. Harness equivalent:
   **partial** (`iso286_fits`/`materials_db` mined the *data*, not these
   validators/fixtures). HOW verified: grepped `pytest.raises` counts; read the
   assertion lines.
5. **17-alloy cited materials table** — `standards/data/materials.yaml` (E, ν,
   density, yield, ultimate, +endurance for steels; each value a `citation{source,
   condition}`; dataset `license: CC0-1.0`). Harness equivalent: **already
   covered** — `domain.standards.materials_db` ("anvilate + cad-cae-copilot
   merge"). HOW verified: registry hit.

ALREADY COVERED: `domain.standards.{iso286_fits, materials_db, part_catalog,
evidence_bundle}` — the *data* layer is mined; the *contract/verifier/spec* layer
(findings 1–4) is not.
VERDICT: **mine-further (top priority).** Delta = the no-silent-green scorecard;
the cited-clause discipline packs (structural + industrial); the spec-rejection
validators + ~229 known-bad fixtures + the nema23 known-good spec; the three
refusal-scenario specs.

---

## Forma-OSS-main (278)

LICENSE: **MPL-2.0** — "Mozilla Public License Version 2.0" header, read.
File-level copyleft (permits linking; modified MPL files stay MPL).
WHAT IT IS: a hardware-blueprint generator (LLM agents → Hardware-IR netlist →
validation → images). Mostly agent/web/provider boilerplate; the verifier island
is the netlist ERC + a prompt-level safety filter.
READ: `blueprint_core/validation.py` (246, full); mapped the ~90 backend/
blueprint_core files; grepped for validation; read the docs listing.
SKIMMED-NOT-READ: ~180 — LLM/image/video provider adapters, agent orchestration,
Next frontend, Rust TUI, REST/auth/db, benchmarks.

FINDINGS (ranked):
1. **!!! REFUSAL GROUND TRUTH — `check_safety_violations` prompt-refusal
   taxonomy.** `blueprint_core/validation.py:5–42` (mirrored in `backend/
   validation.py`): a hard refusal string per keyword class — **weapons**
   (gun/firearm/missile/explosive), **medical/life-support** (pacemaker/
   ventilator/implant), **automotive control** (ecu/brake control/autopilot),
   **mains AC** (110v/220v/240v/ac mains), **high-power battery** (ev battery/48v/
   tesla pack) — each with a distinct message. WHY: a compact prompt→expected-
   REJECTION oracle with a domain-scoped safety taxonomy — directly usable as
   expected-rejection seeds, and a *different* refusal axis (dangerous-domain)
   from IntentForge's (unsupported-capability). Harness equivalent: **partial** —
   `intentforge_refusals` / `prompt_pack` exist but this hardware-safety keyword
   taxonomy is not among them. HOW verified: read the function; registry grep.
2. **Netlist ERC rule set with fix advice** — `validation.py:44–245`:
   `validate_circuit()` = 5 deterministic checks (R1 short circuit, R2 voltage
   mismatch >0.5V, R3 floating/unpowered IC, R4 pin-reuse conflict, R5 overcurrent
   on a 3.3V rail), each emitting `ValidationIssue{severity, category,
   description, troubleshooting}`. Harness equivalent: **already covered** —
   `domain.electronics.circuit_validation` ("mined from Forma-OSS") +
   `domain.electronics.hardware_ir`. HOW verified: registry hit.

ALREADY COVERED: `domain.electronics.{circuit_validation, hardware_ir}`; the
component catalog (per brief).
VERDICT: **mine-further, narrowly** — only the `check_safety_violations` 5-class
prompt-refusal taxonomy is a genuine delta; ERC already mined; everything else is
boilerplate.

---

## freecad-ai-master (176)

LICENSE: **LGPL-2.1** — `LICENSE-CODE` is verbatim "GNU LESSER GENERAL PUBLIC
LICENSE Version 2.1"; separate `LICENSE-ICON` for icons. Copyleft — reference/
reimplement the *spec*, avoid copying code wholesale into a non-LGPL harness.
WHAT IT IS: a FreeCAD MCP-server AI workbench with a skill system, a declarative
per-skill VALIDATION.md geometry-check language, and a tested agent core.
READ: ~3 of 176 — `skills/enclosure/VALIDATION.md` (full), `tests/unit/test_llm_
retry.py` (head); `skills/` + `tests/` listings.
SKIMMED-NOT-READ: ~173 — all `freecad_ai/`, the other 10 SKILL.md files, ~54
tests, translations, resources, the 70KB changelog.

FINDINGS:
1. **Declarative geometry-validation DSL** — `skills/enclosure/VALIDATION.md`:
   typed params, `total_bodies`, per-body `bbox`/`solid_count`/`valid_solid`, and
   **conditional closed-form volume formulas** (`when lid_type == "screw":
   volume: L*W*H − (L−2T)(W−2T)(H−T) + 4·π·PR²·(H−T) (tolerance 5%)`). Harness
   equivalent: **already covered** — `eval.verifiers.validation_rules_md`
   ("VALIDATION.md … freecad-ai"). HOW verified: registry hit on "freecad".
2. 429/retry + loop-control + dangerous-mode test fixtures (`tests/unit/test_llm_
   retry.py`, etc.). Harness equivalent: **covered** (`eval.reliability.executor`,
   `error_contract`, `repair_loop`; `error_taxonomy`; `freecad_expressions`,
   `freecad_catalog`). LGPL makes verbatim reuse unattractive anyway.

ALREADY COVERED: `eval.verifiers.validation_rules_md`, `eval.bench.sequence.
error_taxonomy`, `eval.reliability.executor`, `domain.programs.expressions.
freecad_expressions`, `io.adapters.freecad_catalog` (+ more).
VERDICT: **already-covered.**

---

## AgentSCAD-main (282)

LICENSE: **MIT** — "Copyright (c) 2026 AgentSCAD", read.
WHAT IT IS: a Next.js/TS OpenSCAD text-to-CAD app with a skill/family system, a
committed benchmark suite, mesh-validation rules, and a validation-driven repair
controller.
READ: ~10 of 282 — `benchmarks/hard/snap_fit_enclosure.json`, `cad_knowledge/
failures/floating_parts.md`, `skills/scad-generation/families/{spur_gear,
unknown}.json`, `src/lib/validation/{hole-check,component-check,job-quality}.ts`,
`src/lib/repair/repair-controller.ts` (head); benchmarks/, skills/, cad_knowledge/
listings.
SKIMMED-NOT-READ: ~272 — React UI/hooks/stores, most of `src/lib`, prisma, the 13
other benchmark JSONs + 9 `.scad` examples.

FINDINGS (ranked):
1. Committed benchmark tasks — `benchmarks/{simple,medium,hard}/*.json` (14,
   `expected_bbox`, `required_features`, `tolerances` incl. exact `hole_count`).
   Harness equivalent: **already covered** — `eval.bench.imports.agentscad_tasks`
   vendored all 14. HOW verified: registry hit.
2. Rule-based mesh checks with IDs + criticality — `validation/hole-check.ts`
   (**H001** through-hole via genus, watertight-only, else skip-with-reason),
   `component-check.ts` (**C002** connected components; `>1` → critical "floating
   parts"), each returning `{rule_id, level, passed, is_critical, message}`.
   Harness equivalent: **mostly covered** (`eval.bench.geometry.topology_euler`,
   `selftest.golden` genus); the `rule_id + is_critical` framing is not obviously
   ported but low value.
3. Failure-mode cards — `cad_knowledge/failures/{floating_parts,missing_holes,
   non_manifold_boolean}.md` (Symptom/Causes/Repair/Prevention). Prose; concepts
   covered by `code_repair_rules`. Low value.

ALREADY COVERED: `eval.bench.imports.agentscad_tasks`.
VERDICT: **already-covered** (at most a minor mine of the rule_id/is_critical
framing).

---

## CadAgent-main (65)

LICENSE: **MIT** — "Copyright (c) 2026 CadAgent Contributors", read.
WHAT IT IS: a FreeCAD-workbench ReAct agent with a deterministic pre-execution
code validator, an error→hint→autofix table, and a structured quality judge.
READ: ~4 of 65 — `agent/code_fixes.py` (head), `core/quality.py` (head); `agent/`,
`core/`, `tests/` listings.
SKIMMED-NOT-READ: ~61 — `agent/{loop,controller,react_parser,tools,prompts}.py`,
most of `core/`, 21 tests, ui/.

FINDINGS (ranked):
1. Deterministic error→hint→autofix table — `agent/code_fixes.py`: `pre_
   validate_code()` (compile() gate with caret) + `error_hint()` returning
   `(hint, fixed_code|None)` so unambiguous fixes re-run *without* a model call.
   Harness equivalent: **already covered** — `agents.agent.code_repair_rules`
   docstring explicitly names "Mined from CadAgent (`agent/code_fixes.py`)". HOW
   verified: read the harness docstring.
2. `core/quality.py` `QualityIssue{code,severity,message,suggestion}` (`NO_SOLID`,
   `MULTI_SOLID`). Overlaps existing topology/quality modules; low value.

ALREADY COVERED: `agents.agent.code_repair_rules`.
VERDICT: **already-covered.**

---

## cad-agent-main (9)

LICENSE: **MIT** — "Copyright (c) 2026 Pranjal Bhatia", read.
WHAT IT IS: a tiny CadQuery text-to-CAD generator package (`src/cad_agent/{cli,
generator}.py`) with one test file.
READ: directory listing of all 9 (3 source files + 1 test + pyproject/README/
locks). The two source files were not opened in full — after IntentForge/
AgentSCAD the shape (single generator + single test, no fixtures/schemas/rejection
data) put it below the priority cut.
SKIMMED-NOT-READ: all 9.
FINDINGS: none apparent — no committed fixtures, rejection sets, taxonomies, or
DSLs in the structure. Flagged honestly: the 2 source files were not read in full.
VERDICT: **nothing-here** (structure indicates generic generator glue).

---

## Sim-Correct-main (111)

LICENSE: **MIT** — `LICENSE.md`, "Copyright (c) 2026 Caid Technologies", read.
WHAT IT IS: a MuJoCo sim-to-real gap-correction system — five "Problem" packages
inject a known parameter fault, detect trajectory divergence, identify the faulty
parameter by sensitivity analysis, and emit a validated patch through a strict
OpenCAD↔SimCorrect "CAID" JSON contract.
READ: 8 of 111 — `caid_contract.py` (292), `docs/CAID_ARTIFACT_CONTRACT.md`,
`docs/schemas/caid-design-artifact-v1.schema.json`, `Problem1_ForearmLength/
{divergence_detector,correction_and_validation,parameter_identifier}.py`,
`tests/test_golden_loop_fixture.py`, `tests/fixtures/opencad_forearm_artifact.
json`. Read the full tree.
SKIMMED-NOT-READ: ~103 — the other 4 near-duplicate Problem packages, render/demo
artifacts, `mjcf_correction.py`, `simcorrect_mujoco.py`.

FINDINGS (ranked):
1. CAID artifact/patch contract validator — `caid_contract.py`: a `ContractError`
   taxonomy (schema-version mismatch, missing keys, artifact_id mismatch,
   unsupported types) + a **stale-correction guard** (`_require_current_value`
   rejects a patch whose `old_value` no longer matches the artifact's current
   value). Harness equivalent: **already covered** — `domain.spec.caid_artifact`
   + `domain.spec.design_patch`. HOW verified: registry.
2. Golden known-good fixture + first-divergence detector + sensitivity-analysis
   identifier. Harness equivalent: **covered** — `caid_artifact`,
   `agents.selftrain.divergence` ("the FIRST-DIVERGENCE detector"). The cosine-
   sensitivity localizer is MuJoCo/robotics-specific, not CAD geometry — low
   value.

ALREADY COVERED: `domain.spec.{caid_artifact, design_patch}`, `agents.selftrain.
divergence`.
VERDICT: **already-covered.**

---

## comet-main (67)

LICENSE: **PolyForm Noncommercial 1.0.0** — read. Noncommercial-only; **facts
only, do not vendor.**
WHAT IT IS: an agent long-term-memory / RAG system (sensor→compacter→
consolidator→retriever→vector index) for Claude Code, with an MCP server. **Not
CAD.**
READ: 0 of 67 in full — inspected the tree + module names (`comet/{compacter,
consolidator,retriever,schemas,sensor,storage,vector_index,orchestrator}.py`).
SKIMMED-NOT-READ: all 67 (license + non-CAD scope made deep read low-value).
FINDINGS: none CAD-relevant. The consolidation/retrieval ideas are already
independently represented in `agents.memory.*` + `data.dataengine.curation.
memory_consolidation`. No CAD error taxonomy or refusal set.
ALREADY COVERED: `agents.memory.*`, `data.dataengine.curation.memory_
consolidation`.
VERDICT: **nothing-here** (out of domain + noncommercial license).

---

## Pro-CAD-main (33)

LICENSE: **NONE** — no LICENSE file (verified). → facts only; reimplement from
ideas, do not copy source.
WHAT IT IS: a proactive ambiguity/clarification benchmark for text-to-CAD — from a
clear RIGHT_PROMPT it synthesizes a misleading prompt with K injected typed
ambiguities, then tests whether a ClarifyAgent detects them, asks the right
questions, resolves via a simulated user, and only then generates CadQuery.
READ: 6 of 33 — `config/{ambiguity_under_specified, direct_conflict_same_feature_
two_values, clarification, misleading_prompt, prompt_verification}.py`, `pipeline.
py` (780), `src/evaluation.py`. Read the full tree.
SKIMMED-NOT-READ: ~27 — the remaining `config/*`, `src/{ask_agent,inference,mesh_
utils,data_loader,user,visualization}.py`, training scripts.

FINDINGS (ranked):
1. **A second refusal/abstention paradigm (clarify-instead-of-generate).**
   `config/clarification.py::ASK_AGENT_SYSTEM_PROMPT` — output `{is_misleading:
   false, standardized_prompt}` for clear prompts vs `{is_misleading: true,
   questions:[…]}` for ambiguous/conflicting/impossible ones; `pipeline.py` acts
   on it (`skip_if_not_misleading`). Harness equivalent: **already covered** —
   `domain.spec.{clarify_ambiguity, clarify_dialogue}` ("ProCAD proactive
   ambiguity detection" / "two-round proactive clarification MDP").
2. Ambiguity taxonomy — `config/misleading_prompt.py::MISLEADING_INFO_LIBRARY`
   (under_specified; direct_conflict; **geometric_impossibility_fit** — counterbore
   depth 5 into wall thickness 3; nonstandard_terminology). Harness equivalent:
   **covered** — `domain.spec.{clarify_perturb, clarify_scaling}`.
3. Committed known-bad few-shots with GT questions+answers, judge rubrics
   (`JUDGE_QUESTION_QUALITY`, `JUDGE_AMBIGUITY_RESOLUTION` 1.0/0.5/0.0), code-
   leakage detector, Chamfer evaluator with self-sampled noise floor. Harness
   equivalent: **covered** — `domain.spec.{clarify_metrics, clarify_leakage}`,
   `eval.bench.geometry.chamfer*`. (Note: the large runtime dataset is referenced
   but **not committed** — only in-code few-shots exist.)

ALREADY COVERED: `domain.spec.{clarify_ambiguity, clarify_dialogue, clarify_
perturb, clarify_scaling, clarify_leakage, clarify_metrics}`, `eval.bench.
geometry.chamfer*`.
VERDICT: **already-covered** (the whole `clarify_*` family was mined from here;
license also forbids vendoring).

---

## PairCoder-main (65)

LICENSE: **MIT** — "Copyright (c) 2026 PairCoder Authors", read.
WHAT IT IS: a faithful Driver/Navigator pair-programming loop (paper Algorithm 1),
generic over code benchmarks incl. a CadQuery track.
READ: 2 of 65 in full — `paircoder/loop.py` (304), `reproduction/cad_exec.py`.
Read the full tree.
SKIMMED-NOT-READ: ~63 — client/examples + the large `reproduction/` grader farm.

FINDINGS (ranked):
1. Driver/Navigator control policy (verification predicates ψ surfaced to the
   Navigator; role-switch policies `err<η>`/`fixed<k>`/`none`; TDD variant;
   Algorithm-1 argmax-Quality fallback `(ψ-pass, score, recency)`). Harness
   equivalent: **already covered** — `agents.agent.pair_programming` ("mined from
   PairCoder++").
2. `reproduction/cad_exec.py` — exec CadQuery, auto-discover the result Workplane,
   print `CAD_OK <nverts>` / `CAD_ERR <msg>`. Harness equivalent: **covered** —
   `eval.bench.judges.compiler_judge`, `data.dataengine.reward.executability_
   reward`.

ALREADY COVERED: `agents.agent.pair_programming`, `agents.exploration.variant_
consensus`, `eval.bench.judges.compiler_judge`.
VERDICT: **already-covered.**

---

## muse-main (41)

LICENSE: **MIT** — "Copyright (c) 2026 MUSE Benchmark contributors", read.
WHAT IT IS: a text-to-CAD judge system — renders parts, computes deterministic
geometry + 4-view SVG metrics, runs a rubric-deduction engine + multi-LLM judges,
validates judge-vs-human agreement.
READ: 2 of 41 in full — `src/judge_system/geometry_metrics.py` (284), `rubric.py`
(255); partial `svg_metrics.py`. Read the full tree.
SKIMMED-NOT-READ: ~38 — `judge_system/{llm_judge,pipeline,sandbox,drawings,
providers,reverse_pipeline}.py`, prompts, scripts.

FINDINGS (ranked):
1. **The deterministic rubric-deduction engine** — `rubric.py::_should_apply_
   rule` maps ~18 named `rule_code`s (`code_or_result_missing`, `global_geometry_
   invalid`, `bbox_missing_or_collapsed`, `component_count_mismatch`, `assembly_
   relationship_risk`, `process_fit_risk`, `parameter_range_fragility`, `narrow_
   safe_range`, `pegboard_grid_fit_risk`, …) to boolean triggers over geometry/
   sandbox/SVG evidence, each with a `deduction_ratio`; `score_rubric` starts at
   1.0 and subtracts — a fully deterministic, LLM-free grader. WHY: an
   evidence-predicate → deduction-ratio dispatch. Harness equivalent: **partial**
   — `data.dataengine.reward.visual_score` (multi-aspect rubric) and `eval.bench.
   protocols.geometry_issue_flags` (the MUSE *geometry-flag* classifier) are
   ported, but the *deduction-ratio rule table over combined geometry/sandbox/SVG
   evidence* is not obviously reproduced. Worth mining as a deterministic grader
   beside `visual_score`. HOW verified: read rubric.py; read the `geometry_issue_
   flags` docstring (it covers only the geometry-issue→flag classification, not
   the deduction table).
2. Interpenetration ("穿模") checker — `geometry_metrics.py::evaluate_
   interpenetration` (bbox broad-phase → boolean-intersect, flag any pair whose
   overlap volume > `rel_threshold=0.01` of the smaller solid). Harness
   equivalent: **covered** — `eval.verifiers.interference`, `eval.quality.
   assembly.interactions`, and the docstring of `geometry_issue_flags` explicitly
   names a `muse2_interpenetration_ratio` module. Minor volumetric refinement at
   most.
3. Geometry error-code taxonomy (`Watertightness`, `NonManifoldEdge`, `Self
   Intersection`, `ZeroVolume`, …; the `watertight` vs `watertight_strict` vs
   `manifold` split), 4-view SVG metrics, judge-vs-human agreement. Harness
   equivalent: **covered** — `eval.bench.protocols.geometry_issue_flags` (which
   *precisely* documents the watertight-combined-implies-manifold subtlety),
   `domain.drawings.svg_view_metrics`, `eval.bench.judges.judge_human_agreement`.

ALREADY COVERED: `eval.bench.protocols.geometry_issue_flags`, `bench.muse_
scorecard`, `domain.drawings.svg_view_metrics`, `eval.bench.judges.judge_human_
agreement`, `eval.verifiers.interference`, `data.dataengine.reward.visual_score`.
VERDICT: **mine-further (small)** — only the deterministic `rubric.py` deduction-
ratio rule table is a candidate delta; everything else is covered.

---

## Shape-of-Thought-main (67)

LICENSE: **NONE** — no LICENSE file (verified). → facts only; no vendoring of
code or weights.
WHAT IT IS: a multimodal (image/text→3D) generative model ("SoT-26K", 25,929
samples) on a BAGEL/Qwen2/SigLIP stack producing progressive "shape-of-thought"
assembly traces. Heavy ML training/inference codebase.
READ: 0 of 67 in full — inspected the tree + `data/dataset_info.py` head + asset
filenames incl. `assets/failure_cases/{missing_display_base,occlusion_wrong_
shape}.png`.
SKIMMED-NOT-READ: all 67 — `modeling/{bagel,qwen2,siglip}/*` (vendored transformer
code), `train/*`, `data/*`, the parquet shard.
FINDINGS:
1. The progressive assembly/reasoning-trace idea — Harness equivalent: **already
   covered** — `agents.agent.assembly_trace` ("mined from Shape-of-Thought") +
   `domain.reconstruction.scene.assembly_trace`.
2. `assets/failure_cases/` — two PNG illustrations only, no committed labels/
   fixtures/oracle. Not usable. Not a finding.
VERDICT: **already-covered** (and license = facts-only).

---

## Roshera-CAD-main (1019; essentially all first-party)

LICENSE: **Functional Source License v1.1 (FSL-1.1, Apache-2.0-future)** —
"Copyright (c) 2025-2026 Varun Sharma". NON-permissive (non-compete; converts to
Apache-2.0 after 2 years). **FACTS ONLY — do not vendor code.**
WHAT IT IS: a first-party Rust B-rep kernel + web app whose thesis is "the kernel
cannot lie" — every operation returns a self-certifying validity verdict. Ships
`roshera-eval` = "AGENT-EVAL-α", a certificate-graded CAD benchmark (no LLM
judge).
COMPOSITION (honest): 683 `.rs` (first-party kernel), 124 app files, 65 `.md`
docs, 18 `roshera-eval/` files, 22 mjs build scripts, 20 png. Essentially all
first-party; no large vendored data dumps. The verifier-relevant slice is tiny
(`roshera-eval/`, `BENCHMARKS.md`).
READ: ~5 of 1019 in full — `LICENSE`, `roshera-eval/README.md`, `roshera-eval/
scenarios/08-saddle-honesty.mjs`, `roshera-eval/lib/harness.mjs` (scoring-
predicate grep), `BENCHMARKS.md` (head) — plus the full tree + composition.
SKIMMED-NOT-READ: 683 Rust sources, 124 app files, ~60 docs, 8 other eval
scenarios (index/oracle summaries only).

FINDINGS (ranked):
1. **The `honesty` scored dimension + `t.unsound()` abstention predicate.**
   `roshera-eval/lib/harness.mjs` — DIMS = correctness/soundness/**honesty**/
   performance; `unsound(name, perception, {dim:"honesty"})` asserts `perception.
   sound === false` — a task PASSES by honestly reporting it could not build the
   geometry. WHY: a working anti-false-success predicate where the correct answer
   is an honest self-refusal, scored as its own dimension. Harness equivalent:
   **partial** — `eval.quality.geometry.defect_injection` is tagged "(Roshera
   certificate)" and `eval.verifiers.conformance_report` exists, but honesty-as-a-
   scored-dimension with an explicit `unsound` PASS predicate is not separately
   present. HOW verified: read harness.mjs; registry grep for honesty/unsound
   returns only prose-linter/offset modules.
2. **The saddle "honesty canary" (scenario 08).** `roshera-eval/scenarios/08-
   saddle-honesty.mjs` — a cross-bore cyl∘cyl saddle the kernel historically could
   not build; scored PASS by flagging UNSOUND ("`sound:true` here would be a lie →
   FAIL"), now flipped to expect SOUND after the fix, guarded by a **Steinmetz
   volume oracle** `64000 − 2(π·8²·40) + 16·8³/3 ≈ 50645.71`. WHY: the closest
   thing in this repo to an expected-honest-refusal task that also becomes a
   regression tripwire. Harness equivalent: **NONE** for this canary. HOW
   verified: read the scenario in full.
3. **Independent analytic-oracle table (committed reference values).**
   `roshera-eval/README.md` — per-scenario oracles cross-checked by an *independent*
   method: gear χ=0/293 faces/vol≈5555.8 (shoelace cross-check), injector χ=−70
   (genus 36), bulkhead vol=51200 exact, block χ=−6, hub-flange χ=−12 + flatness/
   perp/position 0.00. WHY: the pattern of cross-checking kernel volume against an
   independent shoelace/Steinmetz oracle (claim-vs-evidence), plus committed
   Euler-characteristic/genus references. Harness equivalent: **partial** — the
   chamfer/CD metrics exist, but the Euler-χ/genus + shoelace-vs-integration
   cross-check oracle table is not present. HOW verified: read the README table.

ALREADY COVERED: `eval.quality.geometry.defect_injection` ("Roshera certificate"),
`eval.verifiers.conformance_report`.
VERDICT: **mine-further (facts-only, no vendoring — FSL-1.1).** Delta = the
honesty-dimension + `unsound` predicate design, the saddle canary, and the
Euler/shoelace independent-oracle table — re-implement clean.

---

## Brepler-main (46)

LICENSE: **NONE** — no LICENSE file (only `readme.md`); verified. → facts/manifest
only, no vendoring.
WHAT IT IS: official impl of "B-repLer: Language-guided Editing of CAD Models"
(SIGGRAPH 2026, arXiv:2508.10201) — edits B-rep solids from NL in a HoLa-VAE
latent space; introduces the BrepEDIT-240K paired-edit dataset (external).
READ: ~4 of 46 in full — `readme.md`, `data.md`, `network/post/check_brep.py`,
`network/post/eval_brep.py` (taxonomy/metric grep), `network/post/construct_brep.
py` (validity-gate grep); `wc -l lists/*.txt`.
SKIMMED-NOT-READ: model code (`dit.py`, `hola.py`, `model.py`, `train.py`), 8
data-prep scripts, vlm_labelling.

FINDINGS (ranked):
1. **5-way B-rep reconstruction taxonomy** — `network/post/eval_brep.py:30`:
   `shape_type = ["valid_solid","invalid_solid","shell","compound","else"]`,
   classified (lines ~229–247) via `BRepCheck_Analyzer.IsValid()` +
   `TopoDS_Solid/Shell/Compound` discrimination. WHY: a 5-way error stratification
   for generated STEP (invalid solid vs open shell vs disconnected compound).
   Harness equivalent: **partial** — `domain.reconstruction.evaluate.topology_
   validity` (CMT "Valid" ratio) covers the *binary* valid/invalid but not the
   solid/shell/compound stratification; `eval.reliability.infeasibility_taxonomy`
   is *sequence-level*, a different stage. HOW verified: read eval_brep.py;
   registry grep `valid_solid`/`shape_type` empty.
2. **Validity-stratified Accuracy-CD / Completeness-CD** — `eval_brep.py::compute_
   statistics` (~290–370): CD split into `acc_cd` (recon→gt) + `com_cd` (gt→recon),
   tallied **per shape_type**. WHY: directional chamfer split reported by validity
   class resists false success (invalid shells don't score on the valid-solid CD).
   Harness equivalent: **partial** — `eval.bench.geometry.{chamfer, accuracy_
   completeness}` cover acc/com CD generically; the validity-stratified reporting
   is the delta. HOW verified: read compute_statistics.
3. Deterministic validity gates (`check_step_valid_soild(precision=1e-1)`,
   `success.txt`/`failed.txt` sentinels) + difficulty-stratified task-ID lists
   (`lists/brepler_{easy,moderate,difficult,test,testing_500}.txt` = 143192 /
   73496 / 21160 / 9998 / 500 lines, referencing external ABC-hashed STEP not
   committed). Minor / external-data.

ALREADY COVERED: `domain.reconstruction.brep.brepler_linearise` (Brepler face
linearisation).
VERDICT: **mine-further (facts-only, NO LICENSE).** Delta = the 5-way `shape_type`
taxonomy + the validity-stratified acc/com-CD protocol; re-implement descriptively.

---

## CAD-Annotator-main (273)

LICENSE: **Apache-2.0** — read. Permissive.
WHAT IT IS: a full-stack GD&T drawing annotator (TS monorepo) with a deterministic
ASME Y14.5-2018 compliance engine, DFM reviewer, confidence-gated re-query, and a
staged GD&T pipeline.
READ: ~4 of 273 in full — `artifacts/api-server/src/lib/compliance-engine.ts`,
`dfm-reviewer.ts`, `requery-service.ts`, `lib/api-zod/src/gdt-schemas.test.ts`
(head) — each cross-checked against the ported harness source. Read the tree.
SKIMMED-NOT-READ: ~200 React UI components, OpenAI audio/image integrations,
drizzle DB, api routes — UI/web boilerplate.
FINDINGS: the verifier substance is **already ported — the whole thing, not just
the test suite** (verified line-for-line):
- `compliance-engine.ts` (DATUM_COUNT_RANGE table `position:(2,3)`,
  `symmetry:(3,3)`, `concentricity:(1,1)`; MMC_LMC set; rules FCF_DATUM_COUNT /
  DATUM_REF_EXISTS / MMC_LMC_APPLICABILITY / TOLERANCE_POSITIVE) → `domain.
  drawings.annotation_set_compliance` (identical tables + 4 rule_ids).
- `dfm-reviewer.ts` → `domain.drawings.dfm_review`; GD&T FCF → `domain.drawings.
  gdt`; `gdt-prompts.ts` → `domain.drawings.gdt_prompts`; `requery-service.ts`
  (confidence 0.6, 15% crop pad) → `domain.drawings.requery`; `pipeline-
  orchestrator.ts` → `domain.drawings.analysis_pipeline`.
The one minor possible delta: `gdt-schemas.test.ts` uses fast-check **property-
based** Zod rejection arbitraries — if the harness port used only example
fixtures, adopting the property-based rejection approach could be worth it, but
the schemas/rules are fully covered.
ALREADY COVERED: `domain.drawings.{annotation_set_compliance, gdt, gdt_prompts,
dfm_review, requery, analysis_pipeline}`, `eval.verifiers.{compliance, dfm}`,
`domain.standards.iso286_fits`.
VERDICT: **already-covered** (the "rest beyond the test suite" was also ported).

---

## mrCAD-main (38)

LICENSE: **CC-BY-NC-4.0** — read. Code/data non-commercial (dataset also inherits
SketchGraphs licensing). Facts/schema OK; treat code as non-commercial.
WHAT IT IS: the mrCAD multimodal reference-communication game (McCarthy,
Vaduguru et al. 2025) — a 2D sketch DSL (Line/Arc/Circle), an edit-action DSL, a
symmetric design-distance metric, and reward functions.
READ: ~5 of 38 in full — `mrcad/rewards.py`, `mrcad/action.py` (head), `mrcad/
design.py` (grep), `mrcad/editing_actions.py` (grep), `tests/test_design_model.
py`, `tests/test_data/*.jsonl`. Cross-checked against harness.
SKIMMED-NOT-READ: agents/ (VLM agents, replay), experiments/, render_utils, env
internals.
FINDINGS: verifier substance **already comprehensively ported**:
- `design.py` symmetric `design_distance` (+ asymmetric halves, `chamfer_
  distance`) → `eval.bench.geometry.design_distance_curve` ("the paper's exact
  Design.design_distance") + `refinement_convergence`.
- geometric predicates (parallel/perpendicular/concentric/meeting_ends/parallel_
  distance/point_to_curve) → `domain.geometry.sketch.curve_relations` (verified
  via `dir()`: all present).
- edit-action DSL (DeletePoint/MovePoint/MakeCurve/MoveCurve/RemoveCurve +
  `round()` canonicalization) → `domain.editing.{curve_degeneracy, sketch_edit_
  schema}`; refinement round/turn state machine → `domain.editing.refinement_
  session`.
- Minor DELTA: the 10 committed known-good design fixtures `tests/test_data/
  test_design_pool{,_rounded}.jsonl` (round-trip + rounding golden pairs) are
  **not** committed to the harness (grep for the cad_ids empty). Known-GOOD only
  (no known-bad, no refusal labels) — low value.
ALREADY COVERED: `domain.editing.{curve_degeneracy, refinement_session, sketch_
edit_schema}`, `domain.geometry.sketch.curve_relations`, `eval.bench.geometry.
{design_distance_curve, refinement_convergence}`.
VERDICT: **already-covered** (optionally lift the 10 round-trip fixtures; not
refusal ground truth).

---

## WhatsInAName-main (31)

LICENSE: **CC-BY-NC-SA-4.0** — read. Non-commercial + share-alike.
WHAT IT IS: code for "What's In A Name?" (Meltzer, Lambourne, Grandi, JCISE 2023,
arXiv:2304.14275) — evaluating LM semantic knowledge of assembly-part names on
the ABC dataset. Pure ML (DistilBERT MLM fine-tune, word2vec/fastText/TechNet
vectorizers, Set-Transformer).
READ: ~3 of 31 in full — `README.md`, `LICENSE`, `src/abcpartnames/scripts/
evaluate_model.py` (head). Read the tree.
SKIMMED-NOT-READ: model defs, data-processing, transforms/vectorizers, other eval
scripts.
FINDINGS: **nothing for a verifier harness.** PyTorch-Lightning training/eval
boilerplate + word-embedding vectorizers. The dataset is downloaded from external
S3 (`download_data.sh`), NOT committed — no committed fixtures, known-bad labels,
rejection tasks, or reference values. Part-naming semantic-plausibility is
tangential to text-to-CAD verification.
VERDICT: **nothing-here.**

---

## CAD-MCP-main (16)

LICENSE: **MIT** — "Copyright (c) 2025 曹瑞", read.
WHAT IT IS: an MCP server driving desktop AutoCAD/GstarCAD/ZWCAD via COM to draw
2D primitives, plus a Chinese/English NLP command parser.
READ: 4 of 4 real source files — `src/server.py` (1129, tool-schema + dispatch),
`src/nlp_processor.py` (523, full), `src/config.json`, and grepped `src/cad_
controller.py` (720, error strings).
SKIMMED-NOT-READ: the `cad_controller.py` COM body (~700 lines of glue), 3 README
translations, imgs/.
FINDINGS (ranked):
1. Per-primitive minimum-arity refusal predicates — `server.py:608–849`
   (JSON-Schema `inputSchema` with `minItems`/`required` per shape) + `:857–971`
   (dispatch raises `ValueError` with a specific missing-parameter list; polyline
   `<2 pts` reject; hatch `minItems:3`; arc needs 4 params). WHY: a compact table
   of 2D-primitive arity refusals — direct expected-rejection cases for a draw
   tool surface. Harness equivalent: **partial** (`domain.electronics.circuit_
   validation` is netlist-level; NONE for 2D-draw arity). HOW verified: read the
   raise sites + schema.
2. The **silent-default anti-pattern** — `nlp_processor.py` unknown command →
   `{type:unknown, error:"无法识别的命令类型"}`, but missing coords silently
   substitutes defaults with a `note`. Useful as a *negative* example (what a
   verifier must catch). Harness equivalent: NONE. Low value.
VERDICT: **mine-further (small)** — the arity-refusal predicate table; the COM
transport is not a finding.

---

## freecad_mcp-main (11)

LICENSE: **MIT** — "Copyright (c) 2025 Rodolfo Bonnin", read.
WHAT IT IS: a FreeCAD addon exposing a socket server + an MCP bridge; the model
sends `send_command`/`run_script` and gets document context back.
READ: 3 of 4 real source files — `freecad_mcp.py` (283, full), `src/freecad_
bridge.py` (70), `main.py` (8).
SKIMMED-NOT-READ: `addon.py`, `InitGui.py`, `package.xml`, assets.
FINDINGS:
1. `get_document_context()` schema (`freecad_mcp.py:176–237`): per-object
   `{name,label,type,visibility,placement,shape{type,volume,area}}` — a ready
   "observed CAD state" schema. Harness equivalent: **partial** (harness has
   richer geometry extraction). Marginal.
2. (Anti-finding) `handle_send_command`/`handle_run_script` do unsandboxed
   `exec(command, {App,Gui})` — the *opposite* of anvilate's sandbox spec; useful
   only as a negative exemplar.
VERDICT: **nothing-here** (transport + arbitrary-exec; the context schema is
thinner than existing harness geometry extraction).

---

## Studio-OSS-main (119)

LICENSE: **GPL-3.0** — "GNU GENERAL PUBLIC LICENSE Version 3", read. Copyleft —
reuse facts/thresholds, not code verbatim.
WHAT IT IS: a Next.js text/image-to-CAD studio; the verifier-relevant file is a
deterministic two-stage design-scoring rubric over BREP metrics.
READ: `lib/scoring.ts` (497, full); listed `lib/` (8 files) + docs.
SKIMMED-NOT-READ: `app/` routes/UI, `components/`, `hooks/`, the other `lib/*.ts`,
8 status docs.
FINDINGS:
1. Two-stage Quality×SpecMatch rubric with a hard quality gate (`Overall =
   QualityGate × (0.7·SpecMatch + 0.3·Quality)`; gate caps 0.4 if BREP invalid,
   0.3 if volume≤0; Euler χ bands, slenderness bands, face-count "boolean
   explosion"; log-space Gaussian ratio similarity; prompt-feature→BREP-face
   detection). Harness equivalent: **already covered** — `eval.quality.geometry.
   two_stage_score` ("Studio-OSS") + `domain.spec.prompt_spec_extract` (the
   `extractTargetSpec` half). HOW verified: registry hit.
ALREADY COVERED: `eval.quality.geometry.two_stage_score`, `domain.spec.prompt_
spec_extract`, `agents.agent.build123d_lints`, `domain.editing.locked_param_
edits`, `agents.llm.resilient_router`, `domain.vision.design_intent` —
extensively mined.
VERDICT: **already-covered.**

---

## querycad-main (55)

LICENSE: **NONE** — no LICENSE file (pyproject present, no grant); verified. →
facts only; reference API/prompt shapes, copy no source.
WHAT IT IS: a CAD part-query system — an LLM writes Python against an OCC-backed
`Shape/Part/Face` API to answer measurement questions ("part like a shaft with
radius 6mm"), plus a GNN face-segmentation benchmark.
READ: `prompts/prompt-query-matching-part.xml` (147, full), head of `src/cad_
service/cad_expert_python_interpreter.py`. Enumerated all 55.
SKIMMED-NOT-READ: `src/cad_service/**` (img_seg, kernel, GNN — ~25 files),
`scripts/benchmark/*`, docs, checkpoints, dataset.zip.
FINDINGS:
1. A constrained query-DSL few-shot protocol with reference conventions —
   `prompts/prompt-query-matching-part.xml`: a restricted API (`Shape`, `Part`,
   `Face{type,radius}`, `get_parts_by_instruction(...)`), a pinned units/axis
   convention (meters; width=x/depth=y/height=z; compare to 0.1mm), the rule "do
   not use any other custom functions", and 3 worked query→code exemplars ending
   `solution = …`. WHY: a grounded-DSL + abstention-by-constraint template with
   pinned precision/units. Harness equivalent: **partial** — `domain.spec.kcl_
   productions`, `domain.programs.validate.scad_grammar` are grammar checkers,
   not this measurement-query protocol. HOW verified: read the XML; no registry
   hit. (LICENSE NONE → facts only.)
VERDICT: **mine-further (small, facts-only)** — the constrained-query + units/
precision protocol as a documented pattern; no source copied.

---

## Synthesis — ranked new deltas the harness does not yet have

1. **IntentForge's `cad_exported_on_rejection` gate + 62-case adversarial set (4
   modes).** The single most important item: a refused-in-words-but-built-anyway
   detector the harness's own `intentforge_refusals` docstring says it lacks.
   Apache-2.0, vendorable. (`harness/adversarial/{adversarial_prompts.json,
   rejection_harness.py}`.)
2. **anvilate's tri-state "no silent green" scorecard.** A couldn't-run check must
   never render green — the missing scorecard-level abstain contract. MIT.
   (`src/anvilate/scorecard.py`.)
3. **cadgenbench's `broken_*.step` jig corpus with `_EXPECTED_SCORES`** — pinned-
   score known-bad mating features (regression tripwires), plus the 4-builder
   adversarial-mesh failure-mode taxonomy. Apache-2.0. (`tests/fixtures/jig_metric/
   *`, `tests/common/adversarial_meshes.py`.)
4. **CADCLAW's L1/L2/L3 geometry good/bad pairs + both-direction claim-audit
   fixtures.** Anti-false-positive AND anti-false-negative ground truth; the BOM
   sextet is already imported, these are not. MIT. (`tests/generate_fixtures.py`,
   `tests/fixtures/claim_audit/`.)
5. **BikeBench's 31 analytic infeasibility predicates + 4-tier boundary scheme** —
   FACTS ONLY (license NONE): transcribe the numeric thresholds into an
   independent expected-rejection module. (`validation/bike_bench_validation_
   functions.py`.)
6. **anvilate's cited-clause discipline packs + ~229 known-bad spec fixtures + 3
   refusal-scenario specs.** MIT. (`packs/structural.py`, `spec/validate.py`,
   `tests/test_*.py`, `openspec/specs/{validation-gauntlet,agent-repair-loop,
   sandbox-security}/spec.md`.)
7. **cad-cae-copilot's completeness/missingness 7-value taxonomy + nafems_vv /
   design_target / optimization_decision schemas + the `012` CAE refusal task +
   graded H/U rubric + staleness fixtures.** MIT.
8. **Roshera's `honesty` scored dimension + `unsound` predicate + Euler/shoelace
   independent-oracle table** — FACTS ONLY (FSL-1.1).
9. **Brepler's 5-way `valid_solid/invalid_solid/shell/compound/else` taxonomy +
   validity-stratified acc/com-CD** — FACTS ONLY (no license).
10. **Smaller:** Forma's `check_safety_violations` dangerous-domain refusal
    taxonomy (MPL-2.0); MUSE's deterministic `rubric.py` deduction-ratio engine
    (MIT); cad-judge's 4-class compile taxonomy + feedback→repair DSL (Apache-2.0);
    IntentForge's 24 edit-preservation chains + assurance content-ID validator;
    CAD-MCP's 2D-primitive arity refusals (MIT); querycad's constrained-query/units
    protocol (facts only); mrCAD's 10 round-trip fixtures (CC-BY-NC).

## Prior claims corrected (verified against the live registry)

- **CADCLAW BOM sextet is NOT a gap.** `eval.corpus.fixtures.cadclaw_bom` already
  vendors the 1-good + 5-labeled-wrong BOMs with provenance. The parallel read's
  "harness equivalent: NONE" on that item was wrong; the geometry pairs + claim
  fixtures remain genuine deltas.
- **cadgenbench's `open_shell.step` and pose twins are NOT a gap.** `eval.corpus.
  fixtures.cadgenbench_pose` vendors them (+ the interface metric is
  `eval.bench.geometry.interface_match`). The un-imported delta is specifically the
  `broken_*.step` jig corpus with `_EXPECTED_SCORES` and the 4 adversarial-mesh
  builders.
- **BikeBench's requirement taxonomy is NOT a gap.** `eval.bench.bikebench_metrics`
  ports `REQUIREMENTS` + population metrics from `design_evaluation.py`/`scoring.
  py`; the un-ported delta is the 31 analytic infeasibility predicates in
  `bike_bench_validation_functions.py`.
- **MUSE's geometry-issue classifier and interpenetration check are NOT gaps.**
  `eval.bench.protocols.geometry_issue_flags` (which even documents the
  watertight-implies-manifold subtlety) + `muse2_interpenetration_ratio` +
  `muse_scorecard` cover them; only the `rubric.py` deduction-ratio table is a
  candidate delta.
- **The corpus is NOT devoid of refusal ground truth.** The brief's note ("none
  found; harness needs it") is stale: `intentforge_refusals`, `prompt_pack`,
  `cadclaw_bom`, `cadgenbench_pose`, `brepnet_steps`, `manifold_meshes`,
  `step_canaries` already exist. The value here is *additive* refusal seams
  (items 1, 3, 4, 5 above), not the first.

## Confirmed absences / already-covered (stated plainly)

- **comet, WhatsInAName** — no CAD verifier material at all (agent-memory RAG;
  ML part-naming). Both non-commercial-licensed regardless.
- **cad-agent-main (9 files), freecad_mcp** — generic generator glue / MCP
  transport + unsandboxed exec; nothing here.
- **CAD-Annotator, mrCAD, Studio-OSS, Sim-Correct, Pro-CAD, PairCoder, freecad-ai,
  AgentSCAD, CadAgent, Shape-of-Thought** — their signature verifier assets are
  ALREADY harness modules (whole GD&T stack; design-distance + edit DSL; two-stage
  score; CAID contract; the `clarify_*` family; pair_programming; validation_
  rules_md; agentscad_tasks; code_repair_rules; assembly_trace). Verified by
  registry lookup, not assumed.
- **No standalone rejection *dataset* in cad-judge** — only inline demo assertions.
- **BikeBench, Brepler, WhatsInAName datasets are external** (S3/Dropbox/ABC-hash
  lists), not committed — only thresholds/taxonomies are minable.
