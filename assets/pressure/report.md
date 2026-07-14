# Pressure test: do typed diagnostics beat blind resampling?

**They do not. The harness loses by 8.3 points, and it loses hardest on the best model.**

6 models x 12 briefs x 2 loops x 3 attempts. Seed `20260713`, temperature 0.0,
backend `frep`. Raw cells in `results.json`.

Both arms get the same model, prompt, seed and attempt budget. They differ in one
thing: what comes back when an attempt fails.

* **blind**: `apply(verify_level="core")`, feed back the raw exception.
* **harness**: `apply(verify_level="full")`, feed back the typed diagnostics.

## Headline

| model | blind | harness | delta | blind attempts | harness attempts |
|---|---:|---:|---:|---:|---:|
| qwen2.5-coder:1.5b | 16.7% | 16.7% | +0.0 | 1.25 | 1.67 |
| qwen2.5-coder:3b | 25.0% | 25.0% | +0.0 | 1.25 | 1.92 |
| qwen2.5-coder:7b | 33.3% | 25.0% | -8.3 | 1.17 | 1.67 |
| qwen2.5-coder:14b | **66.7%** | 41.7% | **-25.0** | 1.08 | 1.67 |
| mistral:7b | 33.3% | 25.0% | -8.3 | 1.67 | 1.83 |
| codellama:7b | 25.0% | 16.7% | -8.3 | 1.33 | 2.08 |
| **pooled** | **33.3%** (24/72) | **25.0%** (18/72) | **-8.3** | 1.29 | 1.81 |

Worse on solve rate, worse on attempts (40% more model calls), better on nothing.
Tied on the two weakest models; loses on the four strongest.

## Why: a lever amplifies whichever way it is pushed

All eight of the harness arm's net losses are **regressions**: an attempt the
grader accepted, which the arm then changed into a wrong one *because the fleet
told it to*. The blind arm has zero regressions.

The effect scales with model strength, monotonically:

| qwen | delta | regressions caused by typed feedback |
|---|---:|---:|
| 1.5b | +0.0 | 0 |
| 3b | +0.0 | 1 |
| 7b | -8.3 | 2 |
| 14b | **-25.0** | **3** |

A weak model cannot act on a typed diagnostic at all. Told "reduce the radius
below 4", the 1.5b deleted the fillet op entirely. It cannot be poisoned because
it is not listening.

A strong model acts precisely. The 14b reads `hole diameter 30 mm >= plate/stock
wall 8 mm` and changes exactly one field, `30 -> 7.5`, leaving the other four
holes untouched. That is flawless instruction-following applied to a false
statement.

**The value of a typed diagnostic is bounded above by its truth, and the tighter a
model's instruction-following, the tighter that bound binds.** A blind loop is
un-poisonable because it is deaf. A typed loop is a lever, and a lever amplifies
whichever way it is pushed.

This reproduces across families: all three 7b-class models (qwen, mistral,
codellama) lose exactly 8.3 points, each on the same brief. It is the verifier's
bug, not the model's.

## What the harness genuinely wins

On `trap_shell_too_thick` the blind arm halts after **one** attempt on every
model. The core verifiers pass, the F-rep backend silently inflates a 60x40x5
plate into a 44,941 mm3 blob, and the loop never knows anything went wrong. The
harness arm is told `Shell thickness 9 leaves no cavity (smallest extent 5).
(Reduce below 2.5.)` and the 3b, 7b and 14b all fix it. The 14b bisects: 3 ->
2.5 -> 2.

**That is a detection capability the blind loop structurally cannot have.** It is
worth 3 briefs. The unaudited rules cost 8.

A counter-example worth knowing: on `trap_shell_too_thin`, blind's *uninformative*
`RuntimeError: empty solid` provoked a jump from 0.2 to 1 and solved it, while the
typed `empty-solid` + `invalid-brep` made the 14b creep 0.2 -> 0.3 -> 0.4 and
fail. Typed is not automatically better even when typed is correct.

## Fleet holes: four real bugs

Each was reproduced against the backend by hand, independently of the run.

**1. `precheck.py`: hole diameter compared against plate thickness.**
Orthogonal dimensions. An 80 mm disc, 8 mm thick, with a 30 mm bore is a washer;
it builds correctly (volume 34,217, bbox 80x80x8) and the fleet raises
`infeasible-plan`. It fired 40 times and caused every regression. The correct
check is diameter against the *in-plane* extent, not Z.

**2. `shell` grows the part, and nothing notices.** The F-rep shell is two-sided
(`|f| - t/2`), so it dilates outward by t/2. A 60x40x20 box shelled at t=3 comes
out with bbox `[63.0, 43.0, 23.0]` and **no diagnostics at all**. No verifier
asserts `bbox_after(shell) <= bbox_before`. Every `shell_box_3mm` solve in this
run is a part with wrong outside dimensions, and the brief carries `bbox=None`, so
the corpus is blind to it too. Fleet and corpus share the blind spot.

**3. A sketch may hold multiple disjoint profiles and be extruded twice,
silently.** qwen-1.5b put a rectangle and an overlapping circle in one sketch and
extruded it at 6 *and* at 30, unioned: volume 99,438 mm3 against a brief wanting
~20,000, `is_valid=True`, no diagnostics. Two verifiers are missing: "sketch has
more than one profile" and "sketch consumed by more than one extrude".

**4. `preflight-RADIUS_TOO_LARGE` is itself unsound**, and it contaminates the
experiment *in the harness's favour*. A 50x30x6 plate with `fillet r=3.1` is
valid, watertight and correctly bounded, and the rule fires anyway; meanwhile
`r == half-extent`, the true boundary, fires nothing. The `trap_fillet_*` briefs
encode the harness's own wrong ceiling as ground truth, so the grader rewards
obeying the harness. It still failed to win either brief. **The -8.3 headline is,
if anything, generous to the harness.**

Not bugs, stated so the raw `fleet_missed` column is not misread: a model emitting
no boolean, using wrong coordinates, or cutting one hole instead of five produces
a valid solid, and no verifier can know the brief. The bug count is four.

## Counterfactual, as a debugging aid and not as the result

Excluding the three briefs the broken hole rule touches: blind 16/54 (29.6%) vs
harness 18/54 (33.3%), **+3.7**. Two briefs; noise at n=54. A harness with a
correct hole rule would win by nothing much.

**The honest headline stands at -8.3, because the broken rule is what the harness
actually ships.**

## Scope, stated plainly

* **Models (6 of 10):** the qwen2.5-coder ladder 1.5b/3b/7b/14b (the controlled
  size comparison) plus mistral:7b and codellama:7b (cross-family). Skipped:
  deepseek-coder-v2:16b, llama3.1:8b, starcoder2:7b, granite-code:8b. A real
  limitation, worth closing if the result were marginal. It is not.
* **Briefs (12 of 28):** a brief that every model solves in one attempt measures
  nothing. Kept 7 plain and all 5 traps. Every kept brief yields a *typed*
  diagnostic when wrong, so the harness is given every chance.
* **Attempts:** 3.

## No thumb on the scale

No bug was found in the experiment harness itself, so nothing there was changed.

**The broken hole rule was NOT fixed before reporting, even though fixing it flips
the headline from -8.3 to +3.7.** It is a bug in the product under test, and
repairing the thing under test to improve its score is the definition of a rigged
result. Hole 4's contamination favours the harness and was likewise left in place.

No brief, prompt, seed, temperature or attempt budget was changed after seeing any
result.

## Reproduce

```bash
export OLLAMA_MODELS="D:\ollama\models"
harnesscad pressure \
  --model qwen2.5-coder:1.5b --model qwen2.5-coder:3b \
  --model qwen2.5-coder:7b   --model qwen2.5-coder:14b \
  --model mistral:7b         --model codellama:7b \
  --loop both --max-attempts 3 --seed 20260713 --temperature 0.0 \
  --briefs plate_hole_four,strip_hole_row,l_bracket,step_block,flange_round,flange_square,shell_box_3mm,trap_shell_too_thick,trap_shell_too_thin,trap_fillet_too_big,trap_fillet_thin_plate,trap_hole_oversize \
  --cache .pressure_cache --out assets/pressure/results.json
```

~9 minutes elapsed across 3 concurrent processes.

## The finding

Typed diagnostics did not beat blind resampling.

The harness's real structural win is that the blind loop **cannot detect** an
infeasible shell: it halts after one attempt and ships an inflated blob without
ever knowing. That is worth 3 briefs. The harness's unaudited feasibility rules
cost it 8.

**A verifier fleet is a trust system, and its throughput is set by its worst rule,
not its best one.**
