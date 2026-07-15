"""The two loops. They are the same loop.

Read this file adversarially -- it is where an experiment like this gets rigged,
so everything that could differ between the arms is listed here and shown not to:

    same model            (the caller passes ONE client to both)
    same seed             (carried on the client, not the loop)
    same temperature      (ditto)
    same system prompt    (prompts.SYSTEM_PROMPT, one constant)
    same user prompt      (prompts.user_prompt(brief.text), one function)
    same attempt budget   (max_attempts, one argument)
    same parser           (model.extract_ops, one function)
    same parse-error text (prompts.format_parse_error, given the arm's name but
                           returning an arm-independent string -- a malformed
                           JSON array is not a geometry failure and neither a
                           kernel nor a fleet has anything special to say)
    same grader           (metrics.grade, always at verify_level="full")
    same stopping rule    (stop when the arm's OWN channel reports success)

    DIFFERENT: verify_level on the session the arm applies into
               ("core" for blind, "full" for harness)
    DIFFERENT: the feedback formatter (prompts.FEEDBACK[arm])

That is the entire delta, and it is exactly the delta the harness claims to be
worth something.

One consequence deserves to be said out loud rather than buried, because it does
half the work of any result this produces: THE STOPPING RULE IS PART OF THE
TREATMENT. The blind arm halts when the core verifiers are happy. On a trap
brief the F-rep backend builds an infeasible shell without complaint, so the
blind arm halts after one attempt believing it has won, and it never gets a
second attempt -- not because we denied it one, but because a bare kernel gives
it no reason to take one. That is not an artefact of the experiment; it IS the
thing the harness is selling. It does mean the blind arm's loss on traps is a
loss of *detection*, not of *repair ability*, and the report says so.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Dict, List, Optional

from harnesscad.eval.pressure import prompts
from harnesscad.eval.pressure.briefs import Brief
from harnesscad.eval.pressure.metrics import AttemptRecord, BriefResult, grade
from harnesscad.eval.pressure.model import Client, extract_ops, ops_to_dicts

BLIND = "blind"
HARNESS = "harness"

#: v2. The arm the book MANDATES and v1 never ran (H2 section 8.5.4: "Always
#: compare your RL method against Best-of-N with the same compute budget"). Draw
#: N candidates, score each with the differential oracle + the output gate, keep
#: the best. No feedback channel at all, so no poisoning surface.
ORACLE_BON = "oracle_bon"

#: v2. The FREE control (H2 section 13.2.2). It reads the SAME N candidates the
#: BoN arm drew -- zero extra model calls -- and picks by majority vote over the
#: canonicalised op stream instead of by the oracle. It is the answer to "does
#: any selection beat any feedback", with the selector's quality held out.
SELF_CONSISTENCY = "self_consistency"

#: The two v1 arms, in v1 order. `LOOPS` is what `--loop both` still means.
LOOPS = (BLIND, HARNESS)

#: The arms that draw a fixed number of INDEPENDENT samples and then select one,
#: rather than iterating with feedback. They share their samples (see
#: `run_sampling`), so running both costs exactly what running one costs.
SELECTION_LOOPS = (ORACLE_BON, SELF_CONSISTENCY)

ALL_LOOPS = LOOPS + SELECTION_LOOPS

#: The verify level each iterative arm APPLIES at. This is the mechanism behind
#: the feedback difference: a core session simply does not run the fleet, so the
#: blind arm has nothing typed to be told even in principle.
VERIFY_LEVEL = {BLIND: "core", HARNESS: "full"}

DEFAULT_MAX_ATTEMPTS = 4

# --------------------------------------------------------------------------- #
# COMPUTE MATCHING, AND THE ONE THING THAT CANNOT BE MATCHED
# --------------------------------------------------------------------------- #
# The iterative arms get a budget of `max_attempts` MODEL CALLS and may stop
# early. The selection arms spend EXACTLY `max_attempts` model calls, always.
# So the selection arms are matched on the CEILING and spend MORE on the mean --
# and every table in the v2 report carries `mean model calls` beside the solve
# rate, and a solve-rate-per-model-call column, so the reader can see it. (v1
# never reported this either: v1's harness arm used 1.81 calls to blind's 1.29 --
# 40% MORE -- and still lost, which makes v1 worse than its own headline and is
# stated nowhere in v1's report.)
#
# TEMPERATURE. Best-of-N REQUIRES sample diversity, and at temperature 0.0 there
# is none: greedy decoding is a function of the prompt, so N samples of one
# prompt are N copies of one sample and the arm degenerates to blind@1. This was
# verified against ollama before the run -- qwen2.5-coder:3b at T=0.0 returns
# BYTE-IDENTICAL text for seeds 20260713, 20260714 and 20260715, and returns
# three different op streams for the same seeds at T=0.8.
#
# So the selection arms CANNOT be run at T=0.0. They are run at
# `SAMPLING_TEMPERATURE`, with one seed per candidate. This is a REAL and
# UNAVOIDABLE confound between the iterative arms and the selection arms, it is
# not a choice made to flatter anybody, and it is stated in the report at the top
# rather than in a footnote. It is a property of what Best-of-N IS.
SAMPLING_TEMPERATURE = 0.8


def sampling_seeds(seed: int, n: int) -> List[int]:
    """One seed per candidate. Deterministic, and a superset relation in N: the
    first k seeds of an N-sample run are the seeds of a k-sample run, so BoN@k
    for k < N can be read off the same draws with no extra model calls."""
    return [int(seed) + i for i in range(int(n))]


def _apply(ops: List[dict], verify_level: str) -> Dict[str, Any]:
    """Apply an op stream in a fresh session at `verify_level`; never raises."""
    from harnesscad.eval.pressure.session import frep_server

    server = frep_server(verify_level)    # PINNED mesher -- see session.py
    try:
        return server.applyOps(ops)
    except Exception as exc:
        # A malformed-but-parseable op (bad field types) can still blow up the
        # server. That IS a kernel-style exception, so hand it back as one.
        return {
            "ok": False,
            "applied": 0,
            "digest": "",
            "diagnostics": [{"severity": "error", "code": "kernel-exception",
                             "message": str(exc), "where": None}],
            "rejected": None,
        }


def run_brief(client: Client, brief: Brief, loop: str, seed: int,
              max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> BriefResult:
    """Run one (model x brief x loop) cell and return its record."""
    if loop not in LOOPS:
        raise ValueError(f"unknown loop {loop!r}; expected one of {LOOPS!r}")

    verify_level = VERIFY_LEVEL[loop]
    feedback_fn: Callable[[dict], Optional[str]] = prompts.FEEDBACK[loop]

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": prompts.user_prompt(brief.text)},
    ]

    records: List[AttemptRecord] = []
    invalid_ops = 0
    fleet_caught = 0
    fleet_missed = 0
    attempts_to_solve: Optional[int] = None
    solved = False
    last_grade = None
    t_cell = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        raw = client.complete(messages, attempt)
        parsed = extract_ops(raw)

        if not parsed.ok:
            invalid_ops += 1
            fb = prompts.format_parse_error(parsed.error or "", loop)
            records.append(AttemptRecord(
                attempt=attempt, raw=raw, parse_ok=False,
                parse_error=parsed.error, ops=[], grade=None, feedback=fb,
                seconds=time.perf_counter() - t0,
            ))
            if attempt == max_attempts:
                break
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": fb},
            ]
            continue

        ops = ops_to_dicts(parsed)

        # (1) the arm's OWN view of the model -- this is the only thing the arm
        #     is allowed to see, and it drives both the feedback and the stop.
        arm_result = _apply(ops, verify_level)
        fb = feedback_fn(arm_result)

        # (2) the referee's view -- never shown to the arm, always at full fleet.
        g = grade(brief, ops)
        last_grade = g
        if g.fleet_caught:
            fleet_caught += 1
        if g.fleet_missed:
            fleet_missed += 1

        records.append(AttemptRecord(
            attempt=attempt, raw=raw, parse_ok=True, parse_error=None,
            ops=ops, grade=g.to_dict(), feedback=fb,
            seconds=time.perf_counter() - t0,
        ))

        if g.solved and attempts_to_solve is None:
            # Record the FIRST attempt whose geometry was actually right, even if
            # the arm goes on to ruin it -- otherwise "attempts to solve" would
            # be measuring the arm's stopping rule rather than its repair.
            attempts_to_solve = attempt
            solved = True

        if fb is None:
            # The arm's channel reports nothing left to fix; it stops here.
            # Whether it was RIGHT to stop is the grader's business, not its own.
            solved = g.solved
            attempts_to_solve = attempt if g.solved else attempts_to_solve
            break

        if attempt == max_attempts:
            break

        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": fb},
        ]

    # The final verdict is the state of the LAST plan the arm stood behind.
    final = records[-1] if records else None
    final_solved = bool(final and final.grade and final.grade.get("solved"))
    final_shape = bool(final and final.grade and final.grade.get("solved_shape"))
    final_iou = ((final.grade.get("shape") or {}).get("iou")
                 if (final and final.grade) else None)
    final_reasons = list(final.grade["reasons"]) if (final and final.grade) else \
        ([final.parse_error] if (final and final.parse_error) else ["no attempts"])
    final_diags = list(final.grade["fleet_actionable"]) if (final and final.grade) else []

    return BriefResult(
        model=client.name,
        loop=loop,
        brief=brief.id,
        category=brief.category,
        trap=brief.trap,
        seed=seed,
        solved=final_solved,
        solved_shape=final_shape,
        shape_iou=final_iou,
        model_calls=len(records),
        attempts_used=len(records),
        attempts_to_solve=(attempts_to_solve if final_solved else None),
        invalid_ops=invalid_ops,
        fleet_caught=fleet_caught,
        fleet_missed=fleet_missed,
        final_reasons=final_reasons,
        final_diagnostics=final_diags,
        seconds=time.perf_counter() - t_cell,
        records=[r.to_dict() for r in records],
    )


# --------------------------------------------------------------------------- #
# v2: the selection arms
# --------------------------------------------------------------------------- #
def _canonical(ops: List[dict]) -> str:
    """A stable fingerprint of an op stream, for the majority vote.

    Numbers are rounded to 3 decimals so that 2.9999999 and 3.0 are one answer,
    and keys are sorted so that field ORDER is not mistaken for disagreement.
    Two streams with the same fingerprint build the same solid.
    """
    import json

    def norm(v):
        if isinstance(v, float):
            return round(v, 3)
        if isinstance(v, list):
            return [norm(x) for x in v]
        if isinstance(v, dict):
            return {k: norm(v[k]) for k in sorted(v)}
        return v

    return json.dumps([norm(o) for o in ops], sort_keys=True)


def run_sampling(client: Client, brief: Brief, seed: int,
                 n: int = DEFAULT_MAX_ATTEMPTS,
                 temperature: float = SAMPLING_TEMPERATURE,
                 ) -> Dict[str, BriefResult]:
    """Draw N independent candidates ONCE, and return BOTH selection arms.

    The two arms see the SAME N samples. Self-consistency is therefore free: it
    costs zero extra model calls, exactly as the book says (H2 section 13.2.2),
    and it is a clean control, because the only thing that differs between it and
    oracle-BoN is the SELECTOR.

    The candidates are drawn with no feedback of any kind -- the prompt is the
    same two messages the other arms open with, and nothing is ever appended. An
    arm with no feedback channel has no poisoning surface, which is the whole
    hypothesis under test.
    """
    from harnesscad.eval.pressure import oracle as oracle_mod

    t_cell = time.perf_counter()
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": prompts.user_prompt(brief.text)},
    ]

    records: List[AttemptRecord] = []
    candidates: List[List[dict]] = []
    invalid_ops = 0
    seeds = sampling_seeds(seed, n)

    for i, s in enumerate(seeds, start=1):
        t0 = time.perf_counter()
        raw = client.complete(messages, i, seed=s, temperature=temperature)
        parsed = extract_ops(raw)
        ops = ops_to_dicts(parsed) if parsed.ok else []
        if not parsed.ok:
            invalid_ops += 1
        records.append(AttemptRecord(
            attempt=i, raw=raw, parse_ok=parsed.ok, parse_error=parsed.error,
            ops=ops, grade=None, feedback=None,
            seconds=time.perf_counter() - t0,
        ))
        candidates.append(ops)

    # ---- grade EVERY draw -------------------------------------------------- #
    # Not to select with -- the grader is never shown to any arm -- but because N
    # independent draws of the same brief is the ONLY place in this experiment
    # where pass@k and pass^k are honestly computable. The iterative arms do not
    # produce independent draws (attempt 2 is conditioned on attempt 1), so a
    # pass@k over them would be a different quantity wearing the same name.
    draw_grades = [grade(brief, ops) if ops else None for ops in candidates]
    for i, g in enumerate(draw_grades):
        if g is not None:
            records[i] = dataclasses.replace(records[i], grade=g.to_dict())
    n_correct = sum(1 for g in draw_grades if g and g.solved)
    n_correct_shape = sum(1 for g in draw_grades if g and g.solved_shape)

    # ---- the selectors ---------------------------------------------------- #
    best_bon, scores = oracle_mod.rank(candidates, name=brief.id)

    counts: Dict[str, List[int]] = {}
    for i, ops in enumerate(candidates):
        if ops:
            counts.setdefault(_canonical(ops), []).append(i)
    if counts:
        # Most-voted stream; ties break on the earliest sample, so the arm is a
        # deterministic function of the draws.
        winner = max(counts.values(), key=lambda idxs: (len(idxs), -idxs[0]))
        best_sc = winner[0]
        votes = len(winner)
    else:
        best_sc = 0
        votes = 0

    shared = {
        # THE number that tells you whether the selector is any good: how many of
        # the N draws were actually correct. A selector can only be judged
        # against what it had to choose from. If n_correct is 0, no selector on
        # earth wins the cell; if n_correct is N, none can lose it. The interval
        # in between is the only place selection means anything.
        "n_correct": n_correct,
        "n_correct_shape": n_correct_shape,
        "draw_solved": [bool(g and g.solved) for g in draw_grades],
    }
    out: Dict[str, BriefResult] = {}
    for arm, pick, extra in (
            (ORACLE_BON, best_bon,
             {"selector": "differential oracle (6 engines) + output gate",
              "scores": [s.to_dict() for s in scores]}),
            (SELF_CONSISTENCY, best_sc,
             {"selector": "majority vote over canonicalised op streams",
              "votes": votes, "distinct": len(counts)}),
    ):
        g = draw_grades[pick] if draw_grades else None
        recs = [r.to_dict() for r in records]
        out[arm] = BriefResult(
            model=client.name,
            loop=arm,
            brief=brief.id,
            category=brief.category,
            trap=brief.trap,
            seed=seed,
            solved=bool(g and g.solved),
            solved_shape=bool(g and g.solved_shape),
            shape_iou=(g.shape.get("iou") if g else None),
            model_calls=len(records),
            attempts_used=len(records),
            # A selection arm has no repair sequence, so "attempts to solve" is
            # not a quantity it has. Do not invent one.
            attempts_to_solve=None,
            invalid_ops=invalid_ops,
            fleet_caught=int(bool(g and g.fleet_caught)),
            fleet_missed=int(bool(g and g.fleet_missed)),
            final_reasons=(list(g.reasons) if g else ["no parseable candidate"]),
            final_diagnostics=(list(g.fleet_actionable) if g else []),
            seconds=time.perf_counter() - t_cell,
            records=recs,
            selection=dict(**extra, **shared, chosen=pick, n=len(candidates),
                           seeds=seeds, temperature=temperature,
                           picked_a_correct_one=bool(g and g.solved),
                           a_correct_one_existed=bool(n_correct)),
        )
    return out
