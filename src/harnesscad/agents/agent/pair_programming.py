"""Driver/Navigator pair-programming loop controller, mined from PairCoder++
(ACL 2026 Findings / arXiv:2607.01883).

PairCoder grounds code-artifact review in the toolchain: a *Driver* writes the
program, a *Navigator* reviews it against verification evidence (does it parse /
compile / execute / render?) and either accepts (``[NOERROR]``) or asks for a
concrete fix; roles switch when errors persist, and if no round is accepted the
loop returns the argmax-Quality candidate seen.

Everything about that loop except the two LLM calls is deterministic control
flow -- role-switching policy, prompt construction, candidate bookkeeping, and
quality selection. This module extracts that control flow as a reusable
controller: the caller injects a ``generate`` callable (the Driver) and a
``review`` callable (the Navigator), optionally a ``check`` verifier, and the
controller runs the paper's Algorithm 1 over them. No model calls live here, so
the loop is unit-testable with plain Python stand-ins.

Design note vs the harness: this is a *sibling* to :mod:`harnesscad.core.harness`
that structures the reviewer as a second persona with error-triggered role
switching. The verifier fleet supplies the ``check`` evidence in real use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

__all__ = [
    "DRIVER_PROMPT",
    "NAVIGATOR_PROMPT",
    "SwitchPolicy",
    "should_switch",
    "build_navigator_prompt",
    "build_driver_fix_prompt",
    "Round",
    "PairResult",
    "run_pair_loop",
    "select_best",
]

DRIVER_PROMPT = (
    "You are the Driver in pair programming. Write the full artifact code only, "
    "no extra prose."
)
NAVIGATOR_PROMPT = (
    "You are the Navigator in pair programming. Review the artifact against the "
    "verification evidence and return [NOERROR] if and only if you cannot cite a "
    "concrete, definite error; otherwise state the specific fix."
)

#: The Navigator's accept token.
ACCEPT_TOKEN = "[NOERROR]"


@dataclass(frozen=True)
class SwitchPolicy:
    """A parsed role-switch policy (Sec. 4.5 ablation)."""

    kind: str  # "err" | "fixed" | "none"
    param: int = 1

    @staticmethod
    def parse(spec: str) -> "SwitchPolicy":
        """Parse ``err<eta>`` / ``fixed<k>`` / ``none`` (PairCoder's PAIRCODER_SWITCH)."""
        spec = (spec or "err1").strip().lower()
        if spec == "none":
            return SwitchPolicy("none", 0)
        if spec.startswith("err"):
            return SwitchPolicy("err", int(spec[3:] or "1"))
        if spec.startswith("fixed"):
            return SwitchPolicy("fixed", int(spec[5:] or "1"))
        raise ValueError(f"unknown switch policy {spec!r}")


def should_switch(policy: SwitchPolicy, consecutive_revises: int, round_index: int) -> bool:
    """Whether to switch seats after a REVISE, per *policy*.

    * ``err<eta>``  -- switch after ``eta`` consecutive REVISE signals,
    * ``fixed<k>``  -- switch every ``k`` rounds regardless of signal,
    * ``none``      -- never switch.
    """
    if policy.kind == "none":
        return False
    if policy.kind == "err":
        return consecutive_revises >= max(1, policy.param)
    if policy.kind == "fixed":
        return (round_index + 1) % max(1, policy.param) == 0
    raise ValueError(f"unknown policy kind {policy.kind!r}")


def build_navigator_prompt(question: str, artifact: str, evidence: str = "") -> str:
    """Construct the Navigator's review prompt (deterministic string assembly)."""
    ev = f"\nVerification evidence:{evidence}" if evidence else ""
    return (
        f"{NAVIGATOR_PROMPT}\n"
        f"Re-read every requirement, trace concrete inputs, and check interfaces. "
        f"If the evidence PASSED and you cannot cite a concrete error, return "
        f"{ACCEPT_TOKEN}.\n"
        f"Task:\n{question}\n"
        f"Artifact:\n{artifact}{ev}"
    )


def build_driver_fix_prompt(question: str, artifact: str, review: str, evidence: str = "") -> str:
    """Construct the Driver's fix prompt from the Navigator's review + evidence."""
    ev = f"\nEvidence:{evidence}" if evidence else ""
    return (
        f"{DRIVER_PROMPT}\n"
        f"Fix the artifact per the reviewer feedback. Return the full artifact only.\n"
        f"Task:\n{question}\n"
        f"Artifact to fix:\n{artifact}\n"
        f"Reviewer feedback:\n{review}{ev}"
    )


@dataclass(frozen=True)
class Round:
    """Telemetry for one Driver/Navigator round."""

    artifact: str
    check_ok: Optional[bool]
    score: Optional[float]
    review: str
    accepted: bool
    switched: bool


@dataclass(frozen=True)
class PairResult:
    """Result of the pair loop: the chosen artifact and per-round telemetry."""

    artifact: str
    accepted: bool
    iters: int
    rounds: Tuple[Round, ...] = field(default=())


def select_best(rounds: List[Round]) -> int:
    """Index of the argmax-Quality round: ``(check_ok, score, recency)``.

    Quality prefers a passing check, then a higher continuous score, then the
    later round (recency), exactly as PairCoder's Algorithm 1 line 19.
    """
    if not rounds:
        raise ValueError("no rounds to select from")
    best_i = 0
    best_key: Tuple[int, float, int] = (-1, float("-inf"), -1)
    for i, r in enumerate(rounds):
        key = (1 if r.check_ok else 0, r.score if r.score is not None else 0.0, i)
        if key > best_key:
            best_key, best_i = key, i
    return best_i


def run_pair_loop(
    question: str,
    generate: Callable[[str], str],
    review: Callable[[str], str],
    *,
    check: Optional[Callable[[str], Tuple[bool, str, float]]] = None,
    max_iters: int = 4,
    policy: str = "err1",
) -> PairResult:
    """Run the deterministic pair-programming loop over injected callables.

    ``generate(prompt) -> artifact`` is the Driver; ``review(prompt) -> text``
    is the Navigator (accepts by including :data:`ACCEPT_TOKEN`);
    ``check(artifact) -> (ok, evidence, score)`` is the optional toolchain
    verifier whose evidence is shown to the Navigator. Roles are abstract seats:
    a switch flips which persona prompt each callable is handed, matching the
    paper's "the finder of the bug takes the keyboard".

    Returns the accepted artifact, or -- if none is accepted within
    ``max_iters`` -- the argmax-Quality candidate. Deterministic given
    deterministic callables.
    """
    pol = SwitchPolicy.parse(policy)
    artifact = generate(f"{DRIVER_PROMPT}\nTask:\n{question}")
    rounds: List[Round] = []
    consecutive = 0
    accepted = False
    iters = 0
    while iters < max_iters:
        ok: Optional[bool] = None
        score: Optional[float] = None
        evidence = ""
        if check is not None:
            ok, ev, score = check(artifact)
            evidence = (" PASSED." if ok else f" FAILED: {ev}") + (f" ({ev})" if ok and ev else "")
        review_text = review(build_navigator_prompt(question, artifact, evidence))
        did_accept = ACCEPT_TOKEN in (review_text or "")
        switched = False
        if not did_accept:
            consecutive += 1
            switched = should_switch(pol, consecutive, iters)
            if switched:
                consecutive = 0
        rounds.append(Round(artifact, ok, score, review_text or "", did_accept, switched))
        if did_accept:
            accepted = True
            break
        artifact = generate(build_driver_fix_prompt(question, artifact, review_text or "", evidence))
        iters += 1

    if accepted:
        return PairResult(artifact, True, iters, tuple(rounds))
    # No acceptance: append the final (unreviewed) artifact as a candidate and
    # pick the argmax-Quality one, scoring it via check if available.
    final_ok, final_score = None, None
    if check is not None:
        final_ok, _, final_score = check(artifact)
    tail = Round(artifact, final_ok, final_score, "", False, False)
    candidates = rounds + [tail]
    best = select_best(candidates)
    return PairResult(candidates[best].artifact, False, iters, tuple(rounds))
