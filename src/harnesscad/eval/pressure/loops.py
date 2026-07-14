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

import time
from typing import Any, Callable, Dict, List, Optional

from harnesscad.eval.pressure import prompts
from harnesscad.eval.pressure.briefs import Brief
from harnesscad.eval.pressure.metrics import AttemptRecord, BriefResult, grade
from harnesscad.eval.pressure.model import Client, extract_ops, ops_to_dicts

BLIND = "blind"
HARNESS = "harness"
LOOPS = (BLIND, HARNESS)

#: The verify level each arm APPLIES at. This is the mechanism behind the
#: feedback difference: a core session simply does not run the fleet, so the
#: blind arm has nothing typed to be told even in principle.
VERIFY_LEVEL = {BLIND: "core", HARNESS: "full"}

DEFAULT_MAX_ATTEMPTS = 4


def _apply(ops: List[dict], verify_level: str) -> Dict[str, Any]:
    """Apply an op stream in a fresh session at `verify_level`; never raises."""
    from harnesscad.io.surfaces.server import CISPServer

    server = CISPServer(backend="frep", verify_level=verify_level)
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
