# pressure v2 -- STATUS

v2 apparatus is complete and tested. No model has been run and no number from the
deleted lineup has been published. This file records the state of the CODE, not a
result.

## What v2 is

The v2 apparatus is the pressure A/B rebuilt so that it measures the harness's
claim instead of the grader. It carries six code fixes over v1, each with a
per-module test in `tests/eval/pressure/test_pressure_v2.py`:

1. The model-facing channel (`prompts.format_typed`) routes through
   `eval/verifiers/soundness.model_facing`, NOT through the severity filter
   `BLOCKING_SEVERITIES`. Severity and soundness are orthogonal: a HEURISTIC
   warning may BLOCK a build but must never INSTRUCT the model. On a known-good
   part the model is told nothing.

2. The grader routes every graded attempt through `io/gate.py`. A gate refusal is
   a loss, so an op stream that produced a wrong-sized part can no longer be
   scored solved by the experiment that exists to catch it.

3. A volumetric-IoU SHAPE metric (`shape.py`) scores each candidate against the
   brief's own reference op stream, in world coordinates. bbox + volume + probes
   are envelope families and are many-to-one; shape catches a feature that landed
   in the wrong place. Shape is reported ALONGSIDE the envelope verdict
   (`solved` survives; `solved_shape` sits beside it), never instead of it.

4. The shell briefs' inside probes were moved off the outer face (x=0) into the
   middle of the wall. See point below.

5. A statistics module (`stats.py`): Wilson 95% CI, exact McNemar for matched
   pairs, pass@k (imported from `eval/bench/sequence/pass_at_k.py`) and pass^k.
   Pure functions, unit-tested with fixed inputs, no model.

6. The mesher is pinned explicitly (`session.py`): `marching_cubes`, what v1 ran,
   against a repo default that has since flipped to dual contouring. Resolution
   is forced to 96 (not v1's 48) because the backend's new wall-resolution guard
   makes `trap_shell_too_thick` unsolvable by every answer at 48; that confound
   is disclosed.

## The v1 corpus bug (disclosed, not silently repaired)

v1's shell briefs probed their "inside" point EXACTLY ON the outer face
(`shell_box_3mm` probed (0, 20, 10) on a block whose x starts at 0). The signed
distance there is 0, so those probes could only be satisfied by a part whose
material was pushed OUTWARD past x=0 -- which is exactly what the broken two-sided
F-rep shell did. The benchmark encoded the bug as the correct answer: every shell
solve in v1 is a part with the wrong outside dimensions.

The fix lives in `src/harnesscad/eval/pressure/briefs.py` (the tier-5 block
comment and the per-brief `note` fields), where the v2 corpus moves each inside
probe to the middle of the wall the brief asked for. The frozen v1 corpus in
`assets/pressure/` is NOT touched. This disclosure is the record of that bug.

## The run (not yet executed)

The run will be executed on the FRONTIER lineup -- qwen3.6:27b, qwen3.6:35b,
ornith:9b, ornith:35b -- not the deleted six-model set. The lineup is the default
in `runner.DEFAULT_MODELS` and is overridable with `--model`.

The honest headline will be the three arms measured against each other,
compute-matched (compare on `model_calls`):

* blind -- raw kernel errors, the bare-resampling baseline;
* soundness-tiered harness -- the v2 typed channel;
* oracle Best-of-N -- the mandatory selection baseline.

Each reported with Wilson CIs and McNemar on the matched pairs. The headline is
NOT a v1-vs-v2 before/after: v2 changed the system under test (soundness tiering,
the gate) AND the ruler (resolution 48 -> 96), so a v1 number and a v2 number are
not interchangeable, and the report says so.

## Obsolete data

The partial results from the deleted lineup have been moved to
`obsolete_deleted_lineup/` and marked OBSOLETE-DO-NOT-PUBLISH. `merge.py` refuses
any shard whose models are not the frontier set, so the dead data cannot be
published by accident.
