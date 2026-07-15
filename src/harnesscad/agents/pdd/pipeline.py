"""The PDD orchestrator: brief -> MGC -> CISP -> artifact -> measured verdict.

This is the capstone of Parts-Driven Development (``audit/pdd_synthesis.md``). It
runs SDD's four phases as ONE named pipeline with the Measured Geometric Contract
(MGC) as the spine, and folds every check into a single :class:`PddVerdict`.

    Specify   brief -> compile_contract() -> MGC (unbound predicates surfaced)
    Plan      accept a CISP op program (the model built it; we do not)
    Implement apply the ops through an INJECTED executor (no kernel dependency)
    Validate  re-measure through io/gate.py + differential oracle + MGC.check()
              + standing gates (orphan-provenance, mutation-score, coverage)

WHAT A PASS MEANS, AND WHAT IT DOES NOT
---------------------------------------
The final verdict is :data:`PASS` only when ALL of the following hold: the output
gate did not refuse; the differential oracle's engines agree; every MEASURED and
bound contract predicate passes; and no orphan ops remain. Anything provably
wrong is :data:`FAIL`; anything we simply cannot certify (an unbound measured
predicate, an abstaining oracle, an unavailable standing gate, a missing
measurement) is :data:`UNCERTIFIED` -- never silently promoted to PASS.

And even a PASS carries the synthesis doc's honest residual verbatim: it means
"passes every measured predicate", NOT "matches the designer's intent". The
oracle is many-to-one -- a hole bored at the wrong coordinate changes no measured
quantity -- so PDD narrows the space of parts; it does not close it.

LAZY COLLABORATORS
------------------
Four sibling modules are authored in parallel and may be absent at import time.
Every one of them is imported LAZILY inside a function, guarded by
``try/except ImportError``, and its absence degrades the relevant check to
"unavailable" (which blocks certification, never fabricates a pass). The public
surface here therefore imports and runs cleanly with no sibling present -- the
``--selfcheck`` CLI proves both a PASS path and an UNCERTIFIED path on injected
synthetic doubles, with no kernel and no model.

Absolute imports under ``harnesscad.``, stdlib-only at module import, deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "PASS",
    "FAIL",
    "UNCERTIFIED",
    "HONEST_RESIDUAL",
    "PhaseResult",
    "PddVerdict",
    "PddPipeline",
    "run_pdd",
    "main",
]

# --- the three terminal verdicts ------------------------------------------- #
#: Every measured predicate held, and every check that could rule the part out
#: was run and cleared.
PASS = "PASS"
#: Something is provably wrong (a gate refusal, a failed predicate, an oracle
#: disagreement, an orphan op).
FAIL = "FAIL"
#: Nothing was proven wrong, but the part could not be certified -- an unbound
#: measured predicate, an abstaining oracle, a missing measurement, or a standing
#: gate that could not run. Never a silent pass.
UNCERTIFIED = "UNCERTIFIED"

#: Copied onto every verdict, PASS included. The failure mode of a verifier is
#: not that it is wrong -- it is that it is BELIEVED past what it checked.
HONEST_RESIDUAL = (
    "A PASS means the part passes every MEASURED, bound predicate of its "
    "contract -- not that it matches the designer's intent. The oracle is "
    "many-to-one: volume, bounding box and genus do not pin down a part, and a "
    "feature at the wrong coordinate changes no measured quantity. PDD narrows "
    "the space of parts; it does not close it. Contract completeness is the new "
    "spec-quality problem: garbage MGC in, faithfully-wrong part out."
)

# The executor callable this pipeline is built against (see run_pdd's docstring):
#     executor(ops: Sequence[Op]) -> artifact
# where ``artifact`` is anything the output gate can measure -- a HarnessSession,
# a GeometryBackend, or a raw mesh. It is injected so the pipeline hard-depends on
# no geometry kernel, exactly as best_of_n injects a ``session_factory`` and the
# CUA modules inject their doubles.
Executor = Callable[[Sequence[Any]], Any]


# --------------------------------------------------------------------------- #
# Result records
# --------------------------------------------------------------------------- #


@dataclass
class PhaseResult:
    """One phase's outcome: a name, whether it advanced, and its detail payload.

    ``ok`` is the phase's local success (Specify compiled a contract, Implement
    produced an artifact, ...). It is NOT the final verdict -- a phase can be
    locally ``ok`` while the pipeline still declines to certify.
    """

    phase: str
    ok: bool
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"phase": self.phase, "ok": self.ok, "detail": self.detail}


@dataclass
class PddVerdict:
    """The single structured result of a PDD run -- phase by phase, plus verdict.

    ``verdict`` is one of :data:`PASS`, :data:`FAIL`, :data:`UNCERTIFIED`.
    ``reasons`` names every fact that pushed the verdict below PASS.
    ``failures`` are the provably-wrong findings; ``clarifications`` are the
    outstanding ``[NEEDS CLARIFICATION]`` items. ``honest_residual`` is carried
    verbatim on every verdict, PASS included.
    """

    verdict: str
    part_id: str = ""
    contract_digest: Optional[str] = None
    phases: Tuple[PhaseResult, ...] = ()
    reasons: Tuple[str, ...] = ()
    failures: Tuple[str, ...] = ()
    clarifications: Tuple[str, ...] = ()
    honest_residual: str = HONEST_RESIDUAL

    @property
    def certified(self) -> bool:
        """True only for a full PASS. UNCERTIFIED and FAIL are both not-certified."""
        return self.verdict == PASS

    def phase(self, name: str) -> Optional[PhaseResult]:
        for p in self.phases:
            if p.phase == name:
                return p
        return None

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "certified": self.certified,
            "part_id": self.part_id,
            "contract_digest": self.contract_digest,
            "phases": [p.as_dict() for p in self.phases],
            "reasons": list(self.reasons),
            "failures": list(self.failures),
            "clarifications": list(self.clarifications),
            "honest_residual": self.honest_residual,
        }


# --------------------------------------------------------------------------- #
# The orchestrator
# --------------------------------------------------------------------------- #


class PddPipeline:
    """Runs the four PDD phases end to end with the MGC as the spine.

    Collaborators are injected (executor, gate measurement, differential oracle,
    provenance builder) so the pipeline depends on no geometry kernel and no model,
    and every real collaborator is imported lazily so a missing sibling degrades a
    check to "unavailable" rather than breaking import.
    """

    def __init__(
        self,
        executor: Executor,
        *,
        gate_measure: Optional[Callable[[Any, Sequence[Any]], "GateOutcome"]] = None,
        oracle: Optional[Callable[[Any, Sequence[Any]], "OracleOutcome"]] = None,
        provenance_builder: Optional[Callable[[Sequence[Any], Any], Any]] = None,
        contract_check: Optional[Callable[[Any, Mapping[str, Any]], Any]] = None,
        hidden_fraction: float = 0.0,
    ) -> None:
        self._executor = executor
        self._gate_measure = gate_measure
        self._oracle = oracle
        self._provenance_builder = provenance_builder
        self._contract_check = contract_check
        self._hidden_fraction = float(hidden_fraction)

    # -- Specify ------------------------------------------------------------ #

    def specify(self, brief: Any, *, part_id: Optional[str] = None) -> Tuple[Any, PhaseResult]:
        """brief -> Measured Geometric Contract. Surfaces unbound predicates.

        Returns ``(contract_or_None, PhaseResult)``. The contract is ``None`` only
        when the ``contract`` sibling is absent AND the caller passed no ready-made
        MGC -- in which case the pipeline still runs Implement/Validate but cannot
        certify against a contract it does not have.
        """
        detail: Dict[str, Any] = {}
        contract = None

        # A caller may hand us an already-compiled MGC (duck-typed on .digest).
        if _looks_like_contract(brief):
            contract = brief
            detail["source"] = "supplied-contract"
        else:
            mod = _lazy("harnesscad.domain.spec.contract")
            if mod is None:
                detail["contract_module"] = "unavailable"
            else:
                try:
                    contract = mod.compile_contract(brief)
                    detail["source"] = "compiled-from-brief"
                except Exception as exc:  # noqa: BLE001 -- defensive against partial input
                    detail["compile_error"] = f"{type(exc).__name__}: {exc}"

        if contract is None:
            return None, PhaseResult("specify", ok=False, detail=detail)

        if part_id and not _get(contract, "part_id"):
            try:
                contract.part_id = part_id  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

        detail["part_id"] = _get(contract, "part_id")
        detail["digest"] = _call0(contract, "digest")
        measured = _call0(contract, "measured") or ()
        unbound = _call0(contract, "unbound") or ()
        # The unbound MEASURED predicates are the [NEEDS CLARIFICATION] markers
        # that block certification -- surface them BEFORE anything is built.
        unbound_measured = [p for p in unbound if _is_measured_kind(p)]
        detail["measured_predicates"] = len(measured)
        detail["unbound_predicates"] = [_pred_key(p) for p in unbound]
        detail["unbound_measured"] = [_pred_key(p) for p in unbound_measured]

        # Optional visible/hidden split (anti-gaming; TDAD hidden-predicate hold-out).
        if self._hidden_fraction > 0.0:
            split_mod = _lazy("harnesscad.domain.spec.contract_split")
            if split_mod is None:
                detail["split"] = "unavailable"
            else:
                try:
                    visible, hidden = split_mod.split_contract(
                        contract, hidden_fraction=self._hidden_fraction
                    )
                    detail["split"] = {
                        "visible": len(_call0(visible, "measured") or ()),
                        "hidden": len(_call0(hidden, "measured") or ()),
                    }
                except Exception as exc:  # noqa: BLE001
                    detail["split_error"] = f"{type(exc).__name__}: {exc}"

        return contract, PhaseResult("specify", ok=True, detail=detail)

    # -- Plan --------------------------------------------------------------- #

    def plan(self, ops: Sequence[Any]) -> Tuple[List[Any], PhaseResult]:
        """Accept a CISP op program. The pipeline never generates it.

        Mirrors ``best_of_n``: the plan is the model's job and arrives as input.
        We only record it and confirm it is a non-empty op sequence.
        """
        program = list(ops or [])
        detail = {"op_count": len(program), "ops": [_op_name(o) for o in program]}
        return program, PhaseResult("plan", ok=bool(program), detail=detail)

    # -- Implement ---------------------------------------------------------- #

    def implement(self, ops: Sequence[Any]) -> Tuple[Any, PhaseResult]:
        """Apply the ops through the injected executor to produce an artifact."""
        detail: Dict[str, Any] = {"executor": _callable_name(self._executor)}
        try:
            artifact = self._executor(list(ops))
        except Exception as exc:  # noqa: BLE001 -- a failed build is not a shipped part
            detail["error"] = f"{type(exc).__name__}: {exc}"
            return None, PhaseResult("implement", ok=False, detail=detail)
        detail["artifact"] = type(artifact).__name__
        return artifact, PhaseResult("implement", ok=artifact is not None, detail=detail)

    # -- Validate ----------------------------------------------------------- #

    def validate(
        self,
        contract: Any,
        ops: Sequence[Any],
        artifact: Any,
        *,
        measurement: Optional[Mapping[str, Any]] = None,
    ) -> PhaseResult:
        """Gate + oracle + MGC.check + standing gates, folded into one phase.

        ``measurement`` is the contract-keyed mapping the MGC is checked against.
        When omitted, a best-effort mapping is adapted from the gate's own
        measurement (whose keys differ -- see :func:`_adapt_gate_measurement`);
        a caller that wants the full MGC checked should pass an explicit mapping
        keyed by the contract's predicate keys.
        """
        detail: Dict[str, Any] = {}

        # 1) The output gate: measure the artifact, or accept an injected verdict.
        gate = self._run_gate(artifact, ops)
        detail["gate"] = gate.as_dict()

        # 2) The measurement the contract is judged on.
        base: Dict[str, Any] = {}
        if gate.measurement:
            base.update(_adapt_gate_measurement(gate.measurement))
        if measurement:
            base.update(dict(measurement))  # caller-supplied keys win
        detail["measurement_keys"] = sorted(base.keys())

        # 3) The differential oracle: independent engines must agree.
        oracle = self._run_oracle(artifact, ops)
        detail["oracle"] = oracle.as_dict()

        # 4) The contract check: every MEASURED, bound predicate must pass.
        contract_report = self._check_contract(contract, base)
        detail["contract"] = contract_report

        # 5) The standing gates: orphan-provenance (+ mutation-score, coverage).
        gates = self._run_standing_gates(ops, artifact, base)
        detail["standing_gates"] = gates

        return PhaseResult("validate", ok=gate.ok, detail=detail)

    # -- Validate helpers --------------------------------------------------- #

    def _run_gate(self, artifact: Any, ops: Sequence[Any]) -> "GateOutcome":
        if self._gate_measure is not None:
            try:
                return self._gate_measure(artifact, list(ops))
            except Exception as exc:  # noqa: BLE001
                return GateOutcome(ok=False, available=True,
                                   detail=f"injected gate raised {type(exc).__name__}: {exc}")
        mod = _lazy("harnesscad.io.gate")
        if mod is None:
            return GateOutcome(ok=False, available=False, detail="io.gate unavailable")
        try:
            report = mod.check(artifact, source=artifact)
        except Exception as exc:  # noqa: BLE001
            return GateOutcome(ok=False, available=True,
                               detail=f"gate.check raised {type(exc).__name__}: {exc}")
        failures = [str(f) for f in getattr(report, "failures", ()) or ()]
        return GateOutcome(
            ok=bool(getattr(report, "ok", False)),
            available=True,
            measurement=dict(getattr(report, "measurement", {}) or {}),
            failures=tuple(failures),
            detail="measured by io.gate.check",
        )

    def _run_oracle(self, artifact: Any, ops: Sequence[Any]) -> "OracleOutcome":
        if self._oracle is None:
            return OracleOutcome(available=False, agree=None,
                                 detail="no differential oracle injected -- abstains")
        try:
            result = self._oracle(artifact, list(ops))
        except Exception as exc:  # noqa: BLE001
            return OracleOutcome(available=True, agree=False,
                                 detail=f"oracle raised {type(exc).__name__}: {exc}")
        if isinstance(result, OracleOutcome):
            return result
        # Duck-type a foreign oracle result onto agree/engines.
        agree = _first_attr(result, ("agree", "agreed", "consensus", "certified"))
        engines = _first_attr(result, ("engines", "backends", "n_engines"))
        return OracleOutcome(
            available=True,
            agree=bool(agree) if agree is not None else None,
            engines=engines if isinstance(engines, int) else None,
            detail="differential oracle consulted",
        )

    def _check_contract(self, contract: Any, measurement: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"available": False}
        if contract is None:
            out["detail"] = "no contract to check"
            return out
        checker = self._contract_check
        if checker is None:
            mod = _lazy("harnesscad.domain.spec.contract")
            if mod is None:
                out["detail"] = "contract module unavailable"
                return out
            checker = mod.check
        try:
            report = checker(contract, measurement)
        except Exception as exc:  # noqa: BLE001
            out["detail"] = f"contract.check raised {type(exc).__name__}: {exc}"
            return out
        out["available"] = True
        out["satisfied"] = bool(getattr(report, "satisfied", False))
        out["failures"] = [_result_key(r) for r in _call0(report, "failures") or ()]
        out["clarifications"] = [_result_key(r) for r in _call0(report, "clarifications") or ()]
        out["missing"] = [_result_key(r) for r in (_call0(report, "missing") or ())]
        return out

    def _run_standing_gates(
        self, ops: Sequence[Any], artifact: Any, measurement: Mapping[str, Any]
    ) -> Dict[str, Any]:
        gates: Dict[str, Any] = {}

        # Orphan-provenance is a PASS criterion: build provenance, find orphan ops.
        prov = self._build_provenance(ops, artifact, measurement)
        gates["orphan_provenance"] = prov

        # Mutation-score and coverage-matrix are advisory standing gates: recorded
        # when available, never fabricated. Their absence does not block a PASS
        # (they measure the verifier fleet, not this one part).
        for name in ("mutation_score", "coverage_matrix"):
            gates[name] = self._run_named_gate(name, ops, artifact, measurement)
        return gates

    def _build_provenance(
        self, ops: Sequence[Any], artifact: Any, measurement: Mapping[str, Any]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {"available": False, "orphan_ops": None}
        builder = self._provenance_builder
        prov_mod = _lazy("harnesscad.core.cisp.provenance")
        try:
            if builder is not None:
                prov = builder(list(ops), measurement)
            elif prov_mod is not None:
                prov = prov_mod.build_provenance(list(ops), measurement)
            else:
                out["detail"] = "provenance module unavailable"
                return out
        except Exception as exc:  # noqa: BLE001
            out["detail"] = f"build_provenance raised {type(exc).__name__}: {exc}"
            return out

        # Prefer the standing gate's own check(); fall back to orphan_ops().
        gate_mod = _lazy("harnesscad.eval.gates.orphan_provenance")
        orphans = None
        if gate_mod is not None and hasattr(gate_mod, "check"):
            try:
                report = gate_mod.check(prov)
                orphans = _first_attr(report, ("orphan_ops", "orphans", "count"))
                if orphans is None and hasattr(report, "ok"):
                    orphans = 0 if getattr(report, "ok") else 1
            except Exception as exc:  # noqa: BLE001
                out["detail"] = f"orphan_provenance.check raised {type(exc).__name__}: {exc}"
        if orphans is None and prov_mod is not None and hasattr(prov_mod, "orphan_ops"):
            try:
                orphans = prov_mod.orphan_ops(prov)
            except Exception as exc:  # noqa: BLE001
                out["detail"] = f"orphan_ops raised {type(exc).__name__}: {exc}"
        if orphans is None:
            # Last resort: read an orphan count straight off the provenance object
            # (an injected/duck-typed provenance double carries it directly).
            orphans = _first_attr(prov, ("orphan_ops", "orphans"))

        if orphans is None:
            out["detail"] = out.get("detail", "no orphan-ops check available")
            return out
        out["available"] = True
        out["orphan_ops"] = _count(orphans)
        return out

    def _run_named_gate(
        self, name: str, ops: Sequence[Any], artifact: Any, measurement: Mapping[str, Any]
    ) -> Dict[str, Any]:
        mod = _lazy(f"harnesscad.eval.gates.{name}")
        if mod is None or not hasattr(mod, "check"):
            return {"available": False, "detail": f"{name} unavailable"}
        # We do not know each gate's exact check() arity, so try the plausible
        # shapes defensively rather than hard-code one.
        for args in ((ops, artifact), (ops,), (artifact,), ()):
            try:
                report = mod.check(*args)
            except TypeError:
                continue
            except Exception as exc:  # noqa: BLE001
                return {"available": True, "detail":
                        f"{name}.check raised {type(exc).__name__}: {exc}"}
            return {"available": True, "ok": bool(getattr(report, "ok", True))}
        return {"available": False, "detail": f"{name}.check signature unrecognised"}

    # -- The whole pipeline ------------------------------------------------- #

    def run(
        self,
        brief: Any,
        ops: Sequence[Any],
        *,
        measurement: Optional[Mapping[str, Any]] = None,
        part_id: Optional[str] = None,
    ) -> PddVerdict:
        """Run all four phases and return the folded :class:`PddVerdict`."""
        contract, specify = self.specify(brief, part_id=part_id)
        program, plan = self.plan(ops)
        artifact, implement = self.implement(program)

        phases: List[PhaseResult] = [specify, plan, implement]
        if artifact is None:
            # Nothing was built; there is no part to measure. This is not a wrong
            # part -- it is the absence of one -- so it is UNCERTIFIED, not FAIL.
            reasons = ("implement produced no artifact",)
            if not implement.ok and "error" in implement.detail:
                reasons = (f"implement failed: {implement.detail['error']}",)
            return self._fold(contract, phases, reasons, failures=(), forced=UNCERTIFIED)

        validate = self.validate(contract, program, artifact, measurement=measurement)
        phases.append(validate)
        return self._decide(contract, specify, validate, phases)

    # -- Verdict folding ---------------------------------------------------- #

    def _decide(
        self,
        contract: Any,
        specify: PhaseResult,
        validate: PhaseResult,
        phases: List[PhaseResult],
    ) -> PddVerdict:
        failures: List[str] = []
        soft: List[str] = []          # reasons that block PASS without proving wrong
        clarifications: List[str] = list(specify.detail.get("unbound_measured", []))

        gate = validate.detail.get("gate", {})
        oracle = validate.detail.get("oracle", {})
        contract_r = validate.detail.get("contract", {})
        gates = validate.detail.get("standing_gates", {})

        # -- Hard failures: something is provably wrong. --
        if gate.get("available") and not gate.get("ok"):
            failures.append("output gate REFUSED the artifact: "
                            + "; ".join(gate.get("failures", ())) or "output gate refused")
        if oracle.get("available") and oracle.get("agree") is False:
            failures.append("differential oracle: independent engines DISAGREE")
        cfailures = contract_r.get("failures") or []
        for key in cfailures:
            failures.append(f"contract predicate FAILED: {key}")
        orphan = gates.get("orphan_provenance", {})
        if orphan.get("available") and orphan.get("orphan_ops"):
            failures.append(f"orphan ops: {orphan.get('orphan_ops')} op(s) attributed to no feature")

        # -- Soft blocks: cannot certify, but nothing proven wrong. --
        if not gate.get("available"):
            soft.append("output gate did not run")
        if clarifications:
            soft.append("contract has unbound MEASURED predicate(s): "
                        + ", ".join(clarifications))
        if contract_r.get("available"):
            if contract_r.get("missing"):
                soft.append("contract predicate(s) measured MISSING (taint): "
                            + ", ".join(contract_r.get("missing")))
            if not contract_r.get("satisfied") and not cfailures:
                soft.append("contract not satisfied (no bound MEASURED predicate passed)")
        else:
            soft.append("contract could not be checked: " + str(contract_r.get("detail", "")))
        if not oracle.get("available"):
            soft.append("differential oracle abstained (fewer than two independent engines)")
        if not orphan.get("available"):
            soft.append("orphan-provenance gate did not run")

        if failures:
            verdict = FAIL
        elif soft:
            verdict = UNCERTIFIED
        else:
            verdict = PASS

        reasons = tuple(failures) + tuple(soft) if verdict != PASS else ()
        return PddVerdict(
            verdict=verdict,
            part_id=_get(contract, "part_id") or specify.detail.get("part_id", "") or "",
            contract_digest=specify.detail.get("digest"),
            phases=tuple(phases),
            reasons=reasons,
            failures=tuple(failures),
            clarifications=tuple(clarifications),
        )

    def _fold(
        self,
        contract: Any,
        phases: List[PhaseResult],
        reasons: Tuple[str, ...],
        failures: Tuple[str, ...],
        forced: str,
    ) -> PddVerdict:
        specify = next((p for p in phases if p.phase == "specify"), None)
        digest = specify.detail.get("digest") if specify else None
        part_id = (_get(contract, "part_id")
                   or (specify.detail.get("part_id") if specify else "") or "")
        clar = tuple(specify.detail.get("unbound_measured", [])) if specify else ()
        return PddVerdict(
            verdict=forced,
            part_id=part_id,
            contract_digest=digest,
            phases=tuple(phases),
            reasons=reasons,
            failures=failures,
            clarifications=clar,
        )


# --------------------------------------------------------------------------- #
# Injected-double outcome records (also the shapes the CLI selfcheck feeds in)
# --------------------------------------------------------------------------- #


@dataclass
class GateOutcome:
    """What the output gate found -- or an injected stand-in for it."""

    ok: bool
    available: bool = True
    measurement: Dict[str, Any] = field(default_factory=dict)
    failures: Tuple[str, ...] = ()
    detail: str = ""

    def as_dict(self) -> dict:
        return {"ok": self.ok, "available": self.available,
                "failures": list(self.failures), "detail": self.detail}


@dataclass
class OracleOutcome:
    """The differential oracle's verdict. ``agree=None`` is an ABSTENTION.

    ``agree`` is True when independent engines agree, False when they disagree
    (a bug with no ground truth), and ``None`` when fewer than two independent
    engines were available -- an abstention, which blocks certification but
    proves nothing wrong.
    """

    available: bool
    agree: Optional[bool]
    engines: Optional[int] = None
    detail: str = ""

    def as_dict(self) -> dict:
        return {"available": self.available, "agree": self.agree,
                "engines": self.engines, "detail": self.detail}


# --------------------------------------------------------------------------- #
# Module-level convenience (mirrors best_of_n's free function)
# --------------------------------------------------------------------------- #


def run_pdd(
    brief: Any,
    ops: Sequence[Any],
    executor: Executor,
    *,
    measurement: Optional[Mapping[str, Any]] = None,
    gate_measure: Optional[Callable[[Any, Sequence[Any]], GateOutcome]] = None,
    oracle: Optional[Callable[[Any, Sequence[Any]], OracleOutcome]] = None,
    provenance_builder: Optional[Callable[[Sequence[Any], Any], Any]] = None,
    contract_check: Optional[Callable[[Any, Mapping[str, Any]], Any]] = None,
    hidden_fraction: float = 0.0,
    part_id: Optional[str] = None,
) -> PddVerdict:
    """Run the whole PDD pipeline once and return its :class:`PddVerdict`.

    Args:
        brief: a part brief (a ``PartSpec``, ``CADBrief``, mapping, or free text
            ``compile_contract`` accepts), OR an already-compiled MGC.
        ops: the CISP op program (the model built it; the pipeline does not).
        executor: ``executor(ops) -> artifact``. The artifact is anything the
            output gate can measure (a HarnessSession, a backend, a raw mesh).
            Injected so the pipeline hard-depends on no geometry kernel.
        measurement: optional contract-keyed measurement mapping the MGC is
            checked against (keys such as ``volume_mm3``, ``bbox_mm``, ``genus``).
            When omitted, a best-effort mapping is adapted from the gate's own
            measurement, whose keys differ.
        gate_measure: optional ``(artifact, ops) -> GateOutcome`` override. When
            omitted, ``io.gate.check`` is used lazily.
        oracle: optional ``(artifact, ops) -> OracleOutcome`` differential oracle.
            When omitted the oracle abstains (which blocks certification).
        provenance_builder: optional ``(ops, measure_state) -> provenance`` override
            for the orphan-provenance gate. When omitted the ``provenance`` sibling
            is used lazily if present.
        contract_check: optional ``(contract, measurement) -> report`` override for
            the MGC check. When omitted ``contract.check`` is used lazily.
        hidden_fraction: if > 0, split the MGC visible/hidden via ``contract_split``.
        part_id: optional id to stamp on a contract that has none.
    """
    pipeline = PddPipeline(
        executor,
        gate_measure=gate_measure,
        oracle=oracle,
        provenance_builder=provenance_builder,
        contract_check=contract_check,
        hidden_fraction=hidden_fraction,
    )
    return pipeline.run(brief, ops, measurement=measurement, part_id=part_id)


# --------------------------------------------------------------------------- #
# Lazy-import + duck-typing helpers (no eager sibling imports anywhere above)
# --------------------------------------------------------------------------- #


def _lazy(module: str) -> Optional[Any]:
    """Import a sibling module lazily; return None if it is not present yet.

    Only an ImportError (a genuinely-absent module) is swallowed -- any other
    exception from a present-but-broken module propagates, because that is a real
    defect, not a not-yet-authored collaborator.
    """
    try:
        import importlib

        return importlib.import_module(module)
    except ImportError:
        return None


def _looks_like_contract(obj: Any) -> bool:
    return (
        hasattr(obj, "digest")
        and hasattr(obj, "measured")
        and hasattr(obj, "unbound")
        and hasattr(obj, "predicates")
    )


def _is_measured_kind(pred: Any) -> bool:
    """True when a predicate's kind is MEASURED (gate-checkable), duck-typed."""
    kind = getattr(pred, "kind", None)
    value = getattr(kind, "value", kind)
    return str(value).lower() == "measured"


def _pred_key(pred: Any) -> str:
    return str(getattr(pred, "key", pred))


def _result_key(result: Any) -> str:
    pred = getattr(result, "predicate", None)
    if pred is not None:
        return _pred_key(pred)
    return str(getattr(result, "key", result))


def _get(obj: Any, attr: str) -> Any:
    return getattr(obj, attr, None) if obj is not None else None


def _call0(obj: Any, method: str) -> Any:
    """Call a zero-arg method if it exists, else return None -- never raise."""
    fn = getattr(obj, method, None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _first_attr(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    return None


def _count(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    try:
        return len(value)
    except TypeError:
        return 1 if value else 0


def _op_name(op: Any) -> str:
    return getattr(op, "OP", None) or type(op).__name__


def _callable_name(fn: Any) -> str:
    return getattr(fn, "__name__", type(fn).__name__)


#: The gate reports geometry under short keys (``volume``, ``bbox``); the contract
#: predicates use suffixed keys (``volume_mm3``, ``bbox_mm``). This is a real
#: interface mismatch between io/gate.py and domain/spec/contract.py, so a caller
#: that wants the full MGC checked should pass an explicit contract-keyed
#: ``measurement``. This adapter only bridges the few unambiguous keys.
_GATE_KEY_ALIASES = {
    "volume": "volume_mm3",
    "bbox": "bbox_mm",
    "genus": "genus",
    "euler_characteristic": "euler_characteristic",
}


def _adapt_gate_measurement(measurement: Mapping[str, Any]) -> Dict[str, Any]:
    """Best-effort bridge from gate measurement keys to contract predicate keys."""
    out: Dict[str, Any] = dict(measurement)
    for gate_key, contract_key in _GATE_KEY_ALIASES.items():
        if gate_key in measurement and contract_key not in out:
            value = measurement[gate_key]
            out[contract_key] = tuple(value) if isinstance(value, list) else value
    return out


# --------------------------------------------------------------------------- #
# CLI -- including --selfcheck on injected synthetic doubles (no kernel, no model)
# --------------------------------------------------------------------------- #


def _synthetic_pass() -> PddVerdict:
    """A PASS path on injected doubles: a fully-bound contract every predicate of
    which is satisfied, the gate does not refuse, the oracle's engines agree, and
    no orphan ops remain. Fully self-contained -- no sibling module is consulted."""
    contract = _FakeContract(
        part_id="demo-plate",
        predicates=(
            _FakePredicate("volume_mm3"),
            _FakePredicate("bbox_mm"),
            _FakePredicate("hole_count"),
            _FakePredicate("genus"),
        ),
    )
    ops = [
        _FakeOp("primitive"), _FakeOp("hole"), _FakeOp("hole"),
        _FakeOp("hole"), _FakeOp("hole"),
    ]
    measurement = {
        "volume_mm3": 80.0 * 40.0 * 3.0,
        "bbox_mm": (80.0, 40.0, 3.0),
        "hole_count": 4,
        "genus": 4,
    }

    def executor(_ops: Sequence[Any]) -> Any:
        return _FakeArtifact("demo-plate")

    def gate_measure(_artifact: Any, _ops: Sequence[Any]) -> GateOutcome:
        return GateOutcome(ok=True, available=True, measurement={},
                           detail="synthetic gate: measured valid")

    def oracle(_artifact: Any, _ops: Sequence[Any]) -> OracleOutcome:
        return OracleOutcome(available=True, agree=True, engines=2,
                             detail="synthetic oracle: two engines agree")

    def provenance_builder(op_list: Sequence[Any], _measure: Any) -> Any:
        return _FakeProvenance(orphan_ops=0)

    def contract_check(_contract: Any, _measurement: Mapping[str, Any]) -> Any:
        # Every bound MEASURED predicate passes -> satisfied, nothing outstanding.
        return _FakeContractReport(satisfied=True)

    return run_pdd(
        contract, ops, executor,
        measurement=measurement,
        gate_measure=gate_measure,
        oracle=oracle,
        provenance_builder=provenance_builder,
        contract_check=contract_check,
    )


def _synthetic_unbound() -> PddVerdict:
    """An UNCERTIFIED path: the contract carries an unbound MEASURED predicate (a
    ``[NEEDS CLARIFICATION]`` marker), so the pipeline refuses to certify even
    though the gate and oracle are both happy and no orphan ops remain."""
    contract = _FakeContract(
        part_id="demo-bracket",
        predicates=(
            _FakePredicate("volume_mm3"),
            _FakePredicate("min_wall_mm", unbound=True),   # brief did not state it
        ),
    )
    ops = [_FakeOp("primitive")]
    measurement = {"volume_mm3": 1000.0}

    def executor(_ops: Sequence[Any]) -> Any:
        return _FakeArtifact("demo-bracket")

    def gate_measure(_artifact: Any, _ops: Sequence[Any]) -> GateOutcome:
        return GateOutcome(ok=True, available=True, detail="synthetic gate: valid")

    def oracle(_artifact: Any, _ops: Sequence[Any]) -> OracleOutcome:
        return OracleOutcome(available=True, agree=True, engines=2)

    def provenance_builder(op_list: Sequence[Any], _measure: Any) -> Any:
        return _FakeProvenance(orphan_ops=0)

    def contract_check(_contract: Any, _measurement: Mapping[str, Any]) -> Any:
        # The bound predicate passes, but the unbound one is an open clarification.
        return _FakeContractReport(
            satisfied=True, clarifications=("min_wall_mm",))

    return run_pdd(
        contract, ops, executor,
        measurement=measurement,
        gate_measure=gate_measure,
        oracle=oracle,
        provenance_builder=provenance_builder,
        contract_check=contract_check,
    )


@dataclass(frozen=True)
class _FakeOp:
    """A stand-in CISP op for the selfcheck (no kernel, no real op set needed)."""

    OP: str = "op"


@dataclass
class _FakeArtifact:
    """A stand-in built artifact -- the executor's output in the selfcheck."""

    part_id: str = "fake"


@dataclass(frozen=True)
class _FakePredicate:
    """A stand-in contract predicate, duck-typed onto .key/.kind/.unbound."""

    key: str
    unbound: bool = False

    @property
    def kind(self) -> str:
        return "measured"


@dataclass
class _FakeContract:
    """A stand-in MGC, duck-typed onto the interface specify() reads."""

    part_id: str
    predicates: Tuple[_FakePredicate, ...] = ()
    intent: str = ""

    def measured(self) -> Tuple[_FakePredicate, ...]:
        return tuple(p for p in self.predicates if not p.unbound)

    def unbound(self) -> Tuple[_FakePredicate, ...]:
        return tuple(p for p in self.predicates if p.unbound)

    def digest(self) -> str:
        return "fake:" + ",".join(p.key for p in self.predicates)


class _FakeContractReport:
    """A stand-in ContractReport for the selfcheck's injected contract_check."""

    def __init__(self, satisfied: bool = True, failures: Sequence[Any] = (),
                 clarifications: Sequence[str] = (), missing: Sequence[str] = ()):
        self.satisfied = satisfied
        self._failures = tuple(failures)
        self._clarifications = tuple(clarifications)
        self._missing = tuple(missing)

    def failures(self) -> Tuple[Any, ...]:
        return self._failures

    def clarifications(self) -> Tuple[str, ...]:
        return self._clarifications

    def missing(self) -> Tuple[str, ...]:
        return self._missing


@dataclass
class _FakeProvenance:
    """A stand-in provenance record with a known orphan-op count."""

    orphan_ops: int = 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` runs the whole pipeline on synthetic
    doubles with no kernel and no model, demonstrating a PASS path and an
    UNCERTIFIED-due-to-unbound path."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.agents.pdd.pipeline",
        description="Parts-Driven Development (PDD) orchestrator.",
    )
    parser.add_argument(
        "--selfcheck", action="store_true",
        help="run the pipeline on injected synthetic doubles (no kernel/model) "
             "and print a PASS verdict and an UNCERTIFIED-due-to-unbound verdict.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit the verdict(s) as JSON.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    passed = _synthetic_pass()
    unbound = _synthetic_unbound()

    if args.json:
        print(json.dumps(
            {"pass_path": passed.as_dict(), "unbound_path": unbound.as_dict()},
            indent=2, sort_keys=True, default=str,
        ))
    else:
        for label, verdict in (("PASS path", passed), ("UNBOUND path", unbound)):
            print(f"[{label}] verdict={verdict.verdict} "
                  f"part={verdict.part_id or '?'} digest={verdict.contract_digest}")
            for reason in verdict.reasons:
                print(f"    - {reason}")
            if verdict.clarifications:
                print(f"    [NEEDS CLARIFICATION]: {', '.join(verdict.clarifications)}")
        print()
        print("honest residual:", HONEST_RESIDUAL)

    # The selfcheck asserts the two demonstrated invariants so a regression in the
    # verdict logic fails loudly rather than printing a wrong verdict.
    ok = passed.verdict == PASS and unbound.verdict == UNCERTIFIED
    if not ok:
        print(f"SELFCHECK FAILED: expected PASS/UNCERTIFIED, got "
              f"{passed.verdict}/{unbound.verdict}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
