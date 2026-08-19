# Pressure test v2: did soundness tiering stop the harness from losing?

**The regression mechanism is gone. The harness no longer loses. It does not yet win.**

v1 recorded that typed diagnostics made a capable model *worse*: the harness arm
lost by 8.3 points, and every one of its eight net losses was a **regression** --
an attempt the model had already got right, edited into a wrong one because a
heuristic rule fired falsely and the model obeyed it. The fix was soundness
tiering: only PROVEN (a theorem) or MEASURED (an observed fact) diagnostics may
instruct a model; a mere HEURISTIC may block a build but never rewrite an answer.

This is the re-run of the same A/B with the tiered fleet.

## Result

| | blind | harness | delta | regressions | wins |
|---|---|---|---|---|---|
| **v1** (3 models, 72 attempts) | 33.3% (24/72) | 25.0% (18/72) | **-8.3** | **8** | 0 |
| **v2** (2 models, 56 attempts) | 60.7% (34/56) | 60.7% (34/56) | **+0.0** | **0** | **0** |

Per model in v2, both arms solve exactly the same briefs:

| model | blind | harness | delta | regressions | wins |
|---|---|---|---|---|---|
| `ornith:9b` | 19/28 | 19/28 | +0.0 | 0 | 0 |
| `qwen3.6:27b` | 15/28 | 15/28 | +0.0 | 0 | 0 |

## What this does and does not show

**The models are not the same as v1.** v1 ran `qwen2.5-coder:7b`, `mistral:7b`
and `codellama:7b`; v2 ran `ornith:9b` and `qwen3.6:27b`. So the *solve-rate*
comparison across versions is confounded -- the jump from 33.3% to 60.7% on the
blind arm is a model improvement, not evidence about the harness.

**The regression count is not confounded.** It is a within-run, within-model
comparison: for each (model, brief), did the harness arm lose a brief the blind
arm solved? In v1 that happened 8 times and the blind arm never did the reverse.
In v2 it happens **zero** times, on either model. That is the model-independent
evidence that the false-instruction channel is closed.

**The arms genuinely diverged** -- this result is not a plumbing artifact where
the harness silently ran blind. All 28 briefs differ in trajectory for both
models; the fleet raised 8 actionable diagnostics on `ornith:9b` and 2 on
`qwen3.6:27b`; and `qwen3.6:27b` needed a retry on 22 of 28 briefs (75.7% of its
attempts failed to parse). There was ample opportunity for a diagnostic to change
an outcome. None changed one for the worse.

**There are also zero wins.** Soundness tiering removed a harm; it has not been
shown to add a benefit. On this corpus and these two models the typed channel is
outcome-neutral. Claiming more than that would repeat v1's mistake in the
opposite direction.

## Honest residuals

* Two models, both local (`ollama`), 28 briefs. Not a frontier evaluation.
* Zero wins may mean the tiered channel is too quiet on this corpus, or that
  these briefs are not ones where a sound diagnostic changes the answer. The
  experiment does not distinguish those.
* `fleet MISSES` (geometry wrong, fleet silent) are still 3 and 1 -- verifier-fleet
  bugs, recorded rather than hidden.

Raw data: `results_v2.json`. The v1 run is preserved unchanged in `results.json`
and `report.md`.
