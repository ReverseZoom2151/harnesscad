"""Locked-parameter edit-script protocol and iteration gate (Studio-OSS).

Mined from **Studio-OSS** (``lib/types.ts`` EditScript/EditAction,
``app/api/iterate/route.ts`` applyEditScript and the iteration loop). Studio's
optimizer step lets a model *propose* parameter edits but never lets it
*apply* them: the proposal is a typed edit script, and a deterministic
applicator enforces the constraints the model might ignore --

  * **locked parameters are inviolable**: an edit touching a parameter the
    user locked is silently dropped, whatever the justification says;
  * **unknown parameters are dropped**: an edit may only touch parameters
    that already exist (the planner is a SET_PARAM optimizer, not a modeler);
  * **bounds are clamped**: a proposed value outside a parameter's
    ``min``/``max`` is clamped into range rather than trusted;
  * **every edit carries a justification** string -- edits are audit records,
    not silent mutations;
  * the applicator returns a *new* parameter set; the input is untouched.

On top sits the **iteration gate** from the loop: apply the script, rescore,
and accept only if the score actually improved -- otherwise keep the previous
state and report "no improvement". The scorer is injected (for the harness
that is :mod:`harnesscad.eval.quality.geometry.two_stage_score` or any other
scorer); this module owns only the deterministic protocol.

Complements :mod:`harnesscad.domain.editing.sketch_edit_schema` (mrCAD's
curve-level edit grammar): this protocol operates at the named-parameter
level of a parametric design with user-held locks.

stdlib-only, deterministic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "EDIT_TYPES",
    "EditAction",
    "EditScript",
    "ApplyOutcome",
    "apply_edit_script",
    "IterationResult",
    "iterate_once",
    "main",
]

EDIT_TYPES = ("SET_PARAM", "ADD_NODE", "REMOVE_NODE", "CONNECT", "DISCONNECT")


@dataclass(frozen=True)
class EditAction:
    """One proposed edit. Only SET_PARAM is applied by this module; the other
    types are carried for protocols that grow node-level editing later."""

    type: str
    key: Optional[str] = None
    value: Optional[Any] = None
    node: Optional[str] = None
    from_id: Optional[str] = None
    to_id: Optional[str] = None
    justification: str = ""

    def __post_init__(self) -> None:
        if self.type not in EDIT_TYPES:
            raise ValueError(f"unknown edit type: {self.type!r}")
        if self.type == "SET_PARAM" and not self.key:
            raise ValueError("SET_PARAM requires a key")


@dataclass(frozen=True)
class EditScript:
    """A model-proposed batch of edits for one iteration."""

    iteration: int
    edits: Tuple[EditAction, ...]
    why: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EditScript":
        edits = []
        for e in payload.get("edits", []) or []:
            edits.append(EditAction(
                type=str(e.get("type", "")),
                key=e.get("key"),
                value=e.get("value"),
                node=e.get("node"),
                from_id=e.get("from"),
                to_id=e.get("to"),
                justification=str(e.get("justification", "")),
            ))
        return cls(
            iteration=int(payload.get("iteration", 0)),
            edits=tuple(edits),
            why=tuple(str(w) for w in payload.get("why", []) or ()),
        )


@dataclass(frozen=True)
class ApplyOutcome:
    """Result of the constrained application of an edit script."""

    parameters: Dict[str, Dict[str, Any]]      # new parameter set
    applied: Tuple[str, ...]                   # "key: old -> new" records
    rejected: Tuple[str, ...]                  # named rejections
    clamped: Tuple[str, ...]                   # bounds-clamp records


def _coerce_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def apply_edit_script(
    parameters: Mapping[str, Mapping[str, Any]],
    script: EditScript,
) -> ApplyOutcome:
    """Apply SET_PARAM edits under lock / existence / bounds constraints.

    ``parameters`` maps names to records carrying at least ``value`` and
    optionally ``min``, ``max`` and ``locked``. The input mapping is never
    mutated.
    """
    updated: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in parameters.items()}
    applied: List[str] = []
    rejected: List[str] = []
    clamped: List[str] = []

    for edit in script.edits:
        if edit.type != "SET_PARAM":
            rejected.append(f"{edit.type}: only SET_PARAM is applied by this protocol")
            continue
        key = edit.key or ""
        record = updated.get(key)
        if record is None:
            rejected.append(f"{key}: unknown parameter")
            continue
        if record.get("locked"):
            rejected.append(f"{key}: locked by the user")
            continue
        value = _coerce_number(edit.value)
        if value is None:
            rejected.append(f"{key}: non-numeric value {edit.value!r}")
            continue
        lo = record.get("min")
        hi = record.get("max")
        original = value
        if lo is not None and value < float(lo):
            value = float(lo)
        if hi is not None and value > float(hi):
            value = float(hi)
        if value != original:
            clamped.append(f"{key}: {original:g} clamped to {value:g} "
                           f"(bounds [{lo}, {hi}])")
        old = record.get("value")
        record["value"] = value
        applied.append(f"{key}: {old} -> {value:g}"
                       + (f" ({edit.justification})" if edit.justification else ""))

    return ApplyOutcome(parameters=updated, applied=tuple(applied),
                        rejected=tuple(rejected), clamped=tuple(clamped))


# --------------------------------------------------------------------------- #
# Iteration gate
# --------------------------------------------------------------------------- #
Scorer = Callable[[Mapping[str, Mapping[str, Any]]], float]


@dataclass(frozen=True)
class IterationResult:
    accepted: bool
    score_before: float
    score_after: float
    parameters: Dict[str, Dict[str, Any]]      # accepted state (new or previous)
    outcome: ApplyOutcome

    @property
    def improved(self) -> bool:
        return self.score_after > self.score_before


def iterate_once(
    parameters: Mapping[str, Mapping[str, Any]],
    script: EditScript,
    scorer: Scorer,
) -> IterationResult:
    """Apply, rescore, and accept only on strict improvement.

    On no improvement the previous parameter state is kept -- the loop never
    walks downhill on its own score.
    """
    score_before = scorer(parameters)
    outcome = apply_edit_script(parameters, script)
    score_after = scorer(outcome.parameters)
    accepted = score_after > score_before
    kept = outcome.parameters if accepted else {k: dict(v) for k, v in parameters.items()}
    return IterationResult(accepted=accepted, score_before=score_before,
                           score_after=score_after, parameters=kept,
                           outcome=outcome)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.editing.locked_param_edits",
        description="Locked-parameter edit-script protocol + iteration gate "
                    "(Studio-OSS).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="apply a script that hits every constraint (lock, "
                             "unknown key, clamp) and run the accept/reject "
                             "iteration gate both ways.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    params = {
        "height": {"value": 100.0, "min": 50.0, "max": 500.0, "locked": False},
        "radius": {"value": 30.0, "min": 10.0, "max": 200.0, "locked": True},
        "wall": {"value": 2.0, "min": 1.0, "max": 6.0, "locked": False},
    }
    script = EditScript.from_dict({
        "iteration": 1,
        "edits": [
            {"type": "SET_PARAM", "key": "height", "value": 200,
             "justification": "proportion score low"},
            {"type": "SET_PARAM", "key": "radius", "value": 60,
             "justification": "should be ignored: locked"},
            {"type": "SET_PARAM", "key": "wall", "value": 9,
             "justification": "should clamp to 6"},
            {"type": "SET_PARAM", "key": "ghost", "value": 1,
             "justification": "unknown parameter"},
        ],
        "why": ["raise height toward the 200mm target"],
    })

    outcome = apply_edit_script(params, script)
    assert outcome.parameters["height"]["value"] == 200.0
    assert outcome.parameters["radius"]["value"] == 30.0
    assert outcome.parameters["wall"]["value"] == 6.0
    assert params["height"]["value"] == 100.0, "input must not mutate"
    assert len(outcome.rejected) == 2 and len(outcome.clamped) == 1
    for line in outcome.applied:
        print(f"  [applied]  {line}")
    for line in outcome.clamped:
        print(f"  [clamped]  {line}")
    for line in outcome.rejected:
        print(f"  [rejected] {line}")

    def scorer(p: Mapping[str, Mapping[str, Any]]) -> float:
        # Best when height/radius ratio is near the 200/30 target.
        ratio = float(p["height"]["value"]) / float(p["radius"]["value"])
        return max(0.0, 1.0 - abs(ratio - 200.0 / 30.0) / 10.0)

    up = iterate_once(params, script, scorer)
    assert up.accepted and up.parameters["height"]["value"] == 200.0
    print(f"[selfcheck] improving edit accepted: "
          f"{up.score_before:.2f} -> {up.score_after:.2f}")

    downgrade = EditScript(iteration=2, edits=(
        EditAction("SET_PARAM", key="height", value=50.0,
                   justification="worse"),))
    down = iterate_once(up.parameters, downgrade, scorer)
    assert not down.accepted
    assert down.parameters["height"]["value"] == 200.0, "previous state kept"
    print(f"[selfcheck] regressing edit rejected: "
          f"{down.score_before:.2f} -> {down.score_after:.2f} (state kept)")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
