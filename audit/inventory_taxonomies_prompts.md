# Deep inventory: missed error taxonomies, inline prompts, and loop policies

Read-only sweep (2026-07-16) of resources/cad_repos + resources/OpenCAD-main.
Capability-index check confirmed heavy prior coverage of DeepCAD/SkexGen/Text2CAD/Zoo/
CADSmith/CADmium/muse/mrCAD/cadgenbench/cad-judge/GenCAD -- and near-zero coverage of
cad-cae-copilot(aieng), AgentSCAD, IntentForge, Forma-OSS, freecad-ai, comet,
Sim-Correct, CadAgent's references.py, Studio-OSS, CAD-Annotator, Pro-CAD, PairCoder.

## Top-10 highest-value finds (ranked)

1. **PairCoder `reproduction/bcb_consensus.py`** -- oracle-free execution-signature
   self-consistency (MBR-exec): probe candidates, cluster by behavior signature,
   pick largest cluster, with the MONOTONE rule (keep candidate[0] unless provably
   outside a >=2-agreement cluster -- never regress below baseline; singleton
   clusters -> keep baseline). Portable to CAD via geometric probe signatures
   (volume/bbox/topology). Build: `eval/reliability/strategies/exec_consensus.py`.
2. **cad-cae-copilot mutation guard + solver_executed honesty gate**
   (`aieng-ui/backend/app/agent_autopilot/{engine,simulation_workflow}.py`) --
   claim-vs-evidence gating of the final answer: bare `final` rejected until a
   mutation tool succeeded; final claiming solver results rejected unless an
   approved non-error run happened. Strongest anti-false-success mechanism in the
   corpus. Build: a `claims_gate` on `agents/agent/termination.py` + three-tier
   intent resolution with abstain-on-low-confidence.
3. **Forma-OSS truncation salvage** (`blueprint_core/llm_providers.py`
   `_salvage_json_text` + `_prune_truncated_tail`) -- 4-layer structured-output
   repair: token-budget floor per schema size; close-and-prune JSON salvage gated by
   full schema validation ("salvage cannot invent content"); exactly one bounded
   budget-escalation retry; json_schema response_format. Build:
   `eval/reliability/structured_salvage.py`.
4. **cad-cae-copilot `failure_mode.py`** (+ consistency-check script) -- 14-mode
   FailureMode taxonomy, classify_exception() heuristic, stable primary_error_code
   mapping with legacy adapter, per-code descriptions. Our `code_error.py` is a
   10-line 4-category normalizer by comparison. Build: layered taxonomy extension.
5. **BrickGPT adaptive rejection sampling** (`models/brickgpt.py`) -- temperature
   escalation +0.01 per already-rejected sample (cap 2.0), per-step rejection budget,
   physics-informed rollback: truncate at first unstable element and regenerate from
   that prefix. We have BrickGPT's verifiers, not its control policy. Build:
   `agents/generation/adaptive_rejection.py` parameterized by any per-step verifier.
6. **Pro-CAD clarification bench** (`config/{clarification,misleading_prompt,
   ambiguity_under_specified,direct_conflict...}.py`) -- misleading-brief generators
   (under-specified, direct-conflict) + judge scoring generated clarification
   questions as Matched-vs-Hallucinated against ground truth + simulated-user
   answerer. The harness's needs_clarification pathway has NO quality eval. Build:
   `eval/pressure/clarification_bench.py`.
7. **CadAgent `agent/references.py`** -- QUALITY_FIX_MAP (quality-gate code -> terse
   fix) + phase-aware context injection: first-iteration checklist, repeated-error
   strategy shift ("change ONE thing"; 3+ failures -> undo_last + different
   approach), iteration-urgency nudge, quality-gate escape hatch. Prior mining
   stopped at code_fixes.py and missed this. Import map + build state-conditioned
   reference injector.
8. **cad-cae-copilot credibility tiering** (AGENTS.md, V&V-40) -- ordered evidence
   tiers with a never-upgrade/downgrade-on-insufficient-evidence honesty invariant;
   plus fail-first review protocol ("list 3-5 reasons the build does NOT match,
   per view, before adding parts") and regression_diff verdict taxonomy
   (clean/collateral_change/topology_changed/identical). Build:
   `governance/credibility_tier.py` + verdict enum into edit_consistency.
9. **OpenCAD-main `backend/opencad_kernel/ERRORS.md`** -- 11 kernel failure codes
   each with a concrete fix (ZERO_VOLUME, BBOX_NO_OVERLAP, BBOX_NEAR_TANGENT tol
   1e-6, NON_MANIFOLD, FILLET_RADIUS_TOO_LARGE, OFFSET_COLLAPSE...). Import rows
   into the repair catalogue; bbox preflight codes -> kernel_preflight addition.
10. **AgentSCAD repair discipline** (`cad_knowledge/failures/*.md`,
    `lib/repair/repair-controller.ts`) -- minimal-diff repair prompt: "fix ONLY the
    failed validation checks, do NOT change dimensions that pass, preserve intent
    features" with failed-vs-passed rule split. Import into the repair-loop prompt
    formatter + failure MDs as unverified knowledge.

Also notable: IntentForge `ToolError{error_type, recoverable, suggested_action}`
contract with recovery-hint table (build small error_contract.py feeding repair_loop's
retry-vs-abstain); freecad-ai skill optimizer (eval-scored SKILL.md hill-climbing with
versioned history, keep-best, infra-failure exclusion -> `generation/prompt_hillclimb.py`);
comet memory-compaction templates ("summary + trigger with 2-4 anchors"); CAD-Annotator
GD&T extraction prompt + low-confidence focused-requery pattern; cadgenbench baseline
system prompt (exemplar).

## Skips (verified)

SKIP-exists: Text23D previous_error threading, Graph-CAD SFT prompts, OpenCAD
prompting.py API-pinning (we have api_reference.py), cad-judge CRM (compiler_refine.py
explicitly mined from it), AlphaCAD category-retry, 429-retry policies (pipeline has
retry/backoff). SKIP-irrelevant: Studio-OSS/CADAM/solidtype thin web-glue prompts,
Sim-Correct MJCF correction (MuJoCo domain), AlphaCAD vote server, Shape-of-Thought/
spatialhero/StepForge standard sampling params.

ALL Category-B prompt material is UNVERIFIED reference data only -- exemplar corpus,
never direct prompt injection.

## Honest coverage note

Grep-driven sweep over every repo (~120) in py/ts/rs/md/txt, then ~20 full reads of
strongest hits. Pure geometry/ML libraries with zero pattern hits (manifold, libfive,
pythonocc-core, OCP, OpenJSCAD, curv, sdfx, ImplicitCAD, solvespace, ruststep, UV-Net,
SymPoint, ComplexGen) not read file-by-file -- residual risk low but nonzero. Not
deeply read: Zoo/modeling-app (already 29 index mentions), kittycad spec.json,
forgent3d, ScadLM backend, Vibe_Layout, Code2World (off-domain), gaudi, Roshera
smart_router.rs, muse judge bodies, AlphaCAD/.history (noise). "Absent" claims for
credibility tiering, MBR-exec consensus, truncation salvage, clarification eval each
verified by dedicated greps returning no matches.
