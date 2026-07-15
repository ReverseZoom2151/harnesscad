"""The prompt both arms share, and the two feedback formatters that differ.

There is exactly ONE system prompt and ONE user prompt. Both arms send them
byte-for-byte identically, at the same temperature, with the same seed and the
same attempt budget. The single independent variable in this experiment is the
function that turns a failed attempt into the next user turn:

    format_blind(...)   what a bare kernel gives you: the raw failure text, no
                        code, no location, no advice -- and, when the kernel did
                        not fail at all, nothing (which is itself the finding).
    format_typed(...)   what the harness claims to give you: severity + CODE +
                        message + WHERE, one line each, straight from the fleet.

If those two strings were ever built from different information about the *same*
model state, the experiment would be measuring the grader instead of the claim.
They are not: both are rendered from one ``ApplyOpsResult`` of one op stream --
``format_blind`` is simply shown a core-level result and ``format_typed`` a
full-fleet one, which is precisely the difference the harness is selling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from harnesscad.eval.verifiers import soundness

# --------------------------------------------------------------------------- #
# WHICH DIAGNOSTICS COUNT -- read this before trusting any number in the report
# --------------------------------------------------------------------------- #
# The full fleet emits ~23 diagnostics on EVERY model, most of which say nothing
# about whether the model got the brief right:
#
#   missing-metadata        fires 3x on every model ever built -- CISP has no op
#                           that can attach a name, a unit or a material, so the
#                           model literally cannot act on it.
#   under-constrained       fires on every sketch that is not fully dimensioned;
#                           the corpus never asks for a constrained sketch.
#   *-skipped               "I could not run" (no assembly, no load case, ...).
#   *-not-yet-measurable    "this backend does not expose the query I need".
#   non-preferred-dimension an ISO style lint: it fires on a CORRECT 40 mm plate
#                           because 40 is not on the preferred-number series.
#   implausible-solid       a fill-ratio heuristic that fires on every solid box.
#
# A correction loop fed those would churn forever on a part that is already
# right, so the typed channel carries only diagnostics that assert the plan is
# INFEASIBLE or INVALID:
#
#   * anything at ERROR severity (empty-solid, invalid-brep, infeasible-plan,
#     bad-value, bad-ref, ...), and
#   * WARNINGs from the kernel preflight (code prefix "preflight-"), which are
#     the fleet's only warnings that mean "the kernel will not build this"
#     (RADIUS_TOO_LARGE, THICKNESS_TOO_LARGE).
#
# BE CLEAR ABOUT WHAT THIS DOES: it is a curation of the harness's output, and it
# is a curation that FAVOURS THE HARNESS ARM. The unfiltered fleet would hand the
# harness arm a stream of un-actionable style lints and it would score worse than
# it does here. This experiment gives the harness its best possible showing, so
# that a negative result cannot be blamed on a strawman configuration. The same
# rule defines "the fleet caught something" in `metrics`, so the catch counts and
# the feedback are the same set by construction.
UNACTIONABLE_CODES = frozenset({
    "missing-metadata",
    "under-constrained",
    "non-preferred-dimension",
    "implausible-solid",
})

_SKIP_SUFFIXES = ("-skipped", "-unmeasurable", "-not-yet-measurable")

#: Warning-severity codes that nonetheless mean "infeasible", so they belong in
#: the typed channel.
FEASIBILITY_WARNING_PREFIX = "preflight-"


def is_actionable(diag: dict) -> bool:
    """Did the FLEET claim something is wrong? An ACCOUNTING/BLOCKING filter.

    THIS IS NOT THE MODEL-FACING FILTER, and it must never be used as one again.
    It selects on SEVERITY (plus a denylist of codes that fire on every part),
    which answers "should this block the build" -- the one question severity
    actually answers. It does NOT answer "may this instruct the model", which is
    a question about TRUTH, and that is :func:`model_facing`'s job.

    It survives because `metrics.fleet_caught` / `fleet_missed` need to count what
    the fleet SAID, independently of what the model was TOLD -- and keeping those
    two numbers separate is how the cost of a false instruction becomes visible
    at all.
    """
    code = str(diag.get("code", ""))
    if code in UNACTIONABLE_CODES:
        return False
    if any(code.endswith(s) for s in _SKIP_SUFFIXES):
        return False
    severity = str(diag.get("severity", ""))
    if severity == "error":
        return True
    if severity == "warning" and code.startswith(FEASIBILITY_WARNING_PREFIX):
        return True
    return False


# --------------------------------------------------------------------------- #
# THE v2 FIX: SOUNDNESS TIERING (the whole reason v1's harness arm lost)
# --------------------------------------------------------------------------- #
# v1's typed channel filtered on `is_actionable` alone -- i.e. on SEVERITY. It
# never asked whether the rule that fired was TRUE. `eval/verifiers/soundness.py`
# was written afterwards, in direct response to the -8.3, and it answers exactly
# that question: PROVEN (a theorem), MEASURED (an observed fact), or HEURISTIC (a
# guess). Only PROVEN and MEASURED are allowed to instruct a model, because a
# typed diagnostic is an INSTRUCTION and an instruction gets obeyed -- including
# when it is wrong. Every one of v1's eight net losses was a regression caused by
# obeying a HEURISTIC rule that happened to be false.
#
# v1 could not prove that fix, because `format_typed` never imported `soundness`.
# The harness had repaired its bug and, as wired, COULD NOT DEMONSTRATE IT. That
# is the single most important plumbing change in v2, and it is disclosed as a
# change to the SYSTEM UNDER TEST, not to the experiment: the v2 harness arm is a
# DIFFERENT arm from v1's, and the two numbers are not interchangeable.
#
# The two filters compose, and they are not the same filter:
#   is_actionable  -- "is this diagnostic about feasibility at all, or is it a
#                     style lint that fires on every correct part?"  (NOISE)
#   soundness      -- "can this rule be trusted to be RIGHT?"        (TRUTH)
# SEVERITY AND SOUNDNESS ARE ORTHOGONAL AXES, AND v1 CONFLATED THEM.
#
#   severity  -- how bad is this IF TRUE?     -> may it BLOCK the build
#   soundness -- how likely is it to BE TRUE? -> may it INSTRUCT the model
#
# Those are different powers with different risks. A false BLOCK costs a retry.
# A false INSTRUCTION destroys a correct answer. v1's typed channel was selected
# by severity alone -- and `is_actionable`, below, is a severity filter with a
# denylist bolted on. Selecting the model's instructions with it is not an
# implementation of the precision policy; it is a BYPASS of it. The red team
# measured what that bypass cost: the fleet was handing a false instruction on
# 69% of provably-correct parts, almost all of it through the WARNING channel,
# which reaches the model identically to ERROR.
#
# So the model-facing channel is now `soundness.model_facing` and NOTHING ELSE.
# It requires BOTH conditions -- a trusted TIER (PROVEN or MEASURED) and a
# severity that claims something is actually wrong (ERROR or WARNING). A MEASURED
# INFO is a true statement with nothing to act on ("assembly-skipped", "40 is not
# on the ISO series", "the sketch is unpinned"), and putting a true statement
# with nothing to act on into a retry prompt can only invite an edit to a part
# that was already right.
def model_facing(diagnostics: Sequence[dict]) -> List[dict]:
    """The diagnostics allowed into the model's retry prompt. THE gate.

    Delegates, wholly, to :func:`harnesscad.eval.verifiers.soundness.model_facing`
    -- the policy lives with the tier table, not in the experiment, or the
    experiment would be measuring its own copy of the rules instead of the ones
    the harness ships.

    ``soundness.tier_of`` FAILS CLOSED: an unrecognised code is HEURISTIC, never
    trusted. Nothing is silenced -- every diagnostic is still produced, still
    returned in the ApplyOpsResult, still graded, still written to the results
    file. It is narrowed in the ONE channel where being wrong destroys work.
    """
    return list(soundness.model_facing(diagnostics or []))


SYSTEM_PROMPT = """\
You are a CAD planner. You turn a design brief into a JSON array of CISP \
operations. You output NOTHING but the JSON array.

The operations, with their exact field names:

  {"op":"new_sketch","plane":"XY"}
      Creates a sketch. Sketches are auto-named sk1, sk2, ... in order.
  {"op":"add_rectangle","sketch":"sk1","x":0,"y":0,"w":60,"h":40}
      x,y is the BOTTOM-LEFT CORNER of the rectangle, not its centre.
  {"op":"add_circle","sketch":"sk1","cx":0,"cy":0,"r":15}
      r is the RADIUS. A 30 mm diameter disc has r = 15.
  {"op":"extrude","sketch":"sk1","distance":5}
      Extrudes a sketch's profile into a solid. Solids are auto-named f1, f2, ...
      in the order the features are created.
  {"op":"hole","face_or_sketch":"solid","x":30,"y":20,"diameter":8,"through":true}
      Cuts a round hole through the existing solid at sketch-plane coordinates
      (x, y). diameter is a DIAMETER. The hole must lie inside the material.
  {"op":"fillet","edges":[],"radius":3}
      Rounds the solid's edges. An empty "edges" list means all edges.
  {"op":"chamfer","edges":[],"distance":2}
      Chamfers the solid's edges. An empty "edges" list means all edges.
  {"op":"shell","faces":[],"thickness":3}
      Hollows the solid out, leaving a wall of "thickness".
  {"op":"boolean","kind":"union","target":"f1","tool":"f2"}
      kind is "union", "cut" or "intersect". Combines two existing solids.
      "cut" removes the tool from the target.

Rules:
  - Output a JSON array. No prose, no markdown fences, no comments.
  - Build the base solid first (new_sketch -> add_rectangle/add_circle ->
    extrude), then apply features (hole / fillet / chamfer / shell / boolean).
  - All dimensions are millimetres.
  - GEOMETRY MUST BE FEASIBLE. A wall cannot be thicker than the stock it is cut
    from; a fillet radius cannot exceed half the smallest dimension of the solid.
    If the brief asks for a value that is not physically buildable, emit the
    closest value that IS buildable rather than the impossible one.\
"""


def user_prompt(brief_text: str) -> str:
    return (
        f"Design brief:\n{brief_text}\n\n"
        "Emit the JSON array of CISP operations that builds this part."
    )


# --------------------------------------------------------------------------- #
# the independent variable: the two feedback channels
# --------------------------------------------------------------------------- #
def format_parse_error(error: str, arm: str) -> str:
    """The one failure both arms see identically: the response was not valid ops.

    A parse failure happens before any geometry exists, so neither a kernel nor a
    verifier fleet has anything to say about it. Reporting it the same way in
    both arms keeps the comparison about *geometry* repair, which is what the
    claim is actually about.
    """
    return (
        f"Your previous output could not be parsed into operations:\n"
        f"  {error}\n\n"
        "Emit ONLY a JSON array of CISP operation objects. Try again."
    )


def format_blind(result: dict) -> Optional[str]:
    """The BLIND channel: what a bare geometry kernel hands back.

    A kernel tells you it threw and it tells you the exception text. It does not
    tell you which op, it does not give you a stable code, and -- crucially --
    when it did NOT throw it tells you nothing at all, because as far as it is
    concerned the build succeeded. Returning ``None`` means "the kernel is happy;
    this arm has no reason to retry", and that is a faithful model of blind
    resampling, not a handicap invented for this experiment.
    """
    if result.get("ok"):
        return None
    msgs: List[str] = []
    for d in result.get("diagnostics") or []:
        if d.get("severity") != "error":
            continue
        msgs.append(str(d.get("message", "")))
    rejected = result.get("rejected")
    if not msgs:
        msgs.append("the operation failed")
    # Rendered as a kernel would: a traceback-ish blob of prose, no codes, no
    # op index, no remedy.
    body = "; ".join(msgs)
    trace = (
        "Traceback (most recent call last):\n"
        '  File "kernel.cpp", line 0, in BRepBuilderAPI::Build\n'
        f"RuntimeError: {body}"
    )
    if rejected:
        trace += "\n(the build was aborted; no solid was produced)"
    return (
        f"The build failed:\n{trace}\n\n"
        "Emit a corrected JSON array of CISP operations."
    )


def format_typed(result: dict) -> Optional[str]:
    """The HARNESS channel: the fleet's SOUND typed diagnostics, verbatim.

    Every diagnostic that is both actionable and sound (PROVEN or MEASURED -- see
    :func:`model_facing`) is rendered as ``[severity] code: message (at where)``
    -- a stable name, a location, and a sentence that says what is wrong in the
    units of the brief. Returning ``None`` means the fleet found nothing it can
    stand behind, and the arm stops.

    v1 filtered on severity alone and handed the model every HEURISTIC guess the
    fleet made. That is the arm that lost by 8.3 points.
    """
    blocking = model_facing(result.get("diagnostics") or [])
    if not blocking:
        return None
    lines = []
    for d in blocking:
        where = f" (at {d['where']})" if d.get("where") else ""
        lines.append(f"  [{d['severity']}] {d['code']}: {d['message']}{where}")
    return (
        "The harness verified your plan and found these problems:\n"
        + "\n".join(lines)
        + "\n\nFix exactly these problems and emit the corrected JSON array of "
          "CISP operations. Keep everything the brief asked for that was not "
          "flagged."
    )


#: name -> formatter. ``loops`` picks one of these and nothing else changes.
FEEDBACK = {
    "blind": format_blind,
    "harness": format_typed,
}
