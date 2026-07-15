"""THE VERIFIER-FLEET MUTATION SCORE GATE. A verifier fleet you never try to
fool is a test suite you never run: it can be arbitrarily weak and you will
never know.

WHAT IT ENFORCES
----------------
TDAD (Rehan, Fiverr Labs, arXiv 2603.08806) names the anti-gaming metric for
spec-driven generation: **semantic mutation testing**. Inject plausible-faulty
variants into a known-good artefact and measure whether the suite catches them;
the caught fraction is the **Mutation Score (MS)**. A low MS is proof the suite
has a blind spot -- it says nothing about the code under test, it indicts the
tests themselves.

This gate lifts that discipline onto HarnessCAD's verifier fleet / differential
oracle. It takes a set of **provably-good parts**, injects the harness's known
defect classes (reusing the existing injectors in
:mod:`harnesscad.eval.quality.geometry.defect_injection` -- the same taxonomy
Roshera's certificate benchmark uses), runs the whole fleet, and scores

    MS = (injected defects KILLED by at least one verifier) / (ACTIVATING defects)

A defect is **killed** when some verifier in the fleet judges the clean part
SOUND but the injected part UNSOUND -- exactly TDAD's "a test that passed on the
original fails on the mutant". A low MS is the CAD analogue of a weak test
suite: the oracle has a blind spot and cannot be trusted to gate generation.

The gate FAILS when MS drops below a floor (default ``0.9``): the fleet is
missing too many lies to be a credible verifier.

NON-ACTIVATING DEFECTS ARE EXCLUDED FROM THE DENOMINATOR
-------------------------------------------------------
TDAD (and mutation testing generally) excludes **non-activating** mutants -- a
mutation that does not actually change program behaviour cannot be caught, so
counting it would silently deflate MS and let a real blind spot hide behind
dead mutations. The geometric analogue is a defect that does not change the
geometry: the injector refused (raised ``DefectError``) or produced an artefact
equal to the base. Those cases are recorded as ``non_activating`` and kept OUT
of the denominator. Only defects that genuinely perturb the part are scored --
if nothing changed, there was no lie to catch.

DEFENSIVE DEGRADATION
---------------------
The real verifier fleet needs kernels (OCCT / Manifold / truck / frep) that may
be absent in a bare environment. :func:`main` tries to assemble the real fleet;
if it cannot, it degrades to the synthetic ``--selfcheck`` fixture (synthetic
verifiers + synthetic defects, no kernel and no model) and says so clearly, so
the gate is always runnable and its degraded status is never silent.

:func:`check` itself is pure: it takes the parts, the fleet, and the injectors
as arguments and computes the report with no I/O.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence

# Reuse the existing defect-injection machinery (do NOT reimplement it).
from harnesscad.eval.quality.geometry.defect_injection import (
    DEFECT_CLASSES,
    DefectError,
    INJECTORS,
    Mesh,
    topology_verifier,
    unit_cube_mesh,
    unit_tetrahedron,
)

__all__ = [
    "DEFAULT_MS_FLOOR",
    "Verifier",
    "Injector",
    "CaseResult",
    "MutationReport",
    "check",
    "self_check_report",
    "format_text",
    "main",
]

#: TDAD-style anti-gaming floor: below this the fleet is too blind to gate on.
DEFAULT_MS_FLOOR: float = 0.9

#: A verifier judges an artefact SOUND (True) or UNSOUND (False).
Verifier = Callable[[Any], bool]
#: An injector maps a clean artefact to a defective one (or raises DefectError).
Injector = Callable[[Any], Any]


@dataclass(frozen=True)
class CaseResult:
    """The outcome of one (part, defect class) mutation."""

    part: str
    defect: str
    activating: bool
    killed: bool
    #: Indices of the verifiers that killed this defect (passed base, failed mutant).
    killers: Sequence[int] = ()
    #: Why a non-activating case was excluded ("refused" or "no-change"), else "".
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "part": self.part,
            "defect": self.defect,
            "activating": self.activating,
            "killed": self.killed,
            "killers": list(self.killers),
            "reason": self.reason,
        }


@dataclass
class MutationReport:
    """Fleet-wide Mutation Score over a set of (part, defect) pairs."""

    floor: float = DEFAULT_MS_FLOOR
    fleet_size: int = 0
    cases: List[CaseResult] = field(default_factory=list)
    #: Parts no verifier in the fleet judged sound -- a fixture fault, not a kill.
    base_unsound_parts: List[str] = field(default_factory=list)
    degraded: bool = False
    note: str = ""

    @property
    def activating(self) -> int:
        """Defects that genuinely changed the geometry -- the MS denominator."""
        return sum(1 for c in self.cases if c.activating)

    @property
    def non_activating(self) -> int:
        """Defects excluded from the denominator (refused or no-change)."""
        return sum(1 for c in self.cases if not c.activating)

    @property
    def killed(self) -> int:
        return sum(1 for c in self.cases if c.activating and c.killed)

    @property
    def mutation_score(self) -> float:
        """MS = killed / activating; 0.0 when nothing activated (uncertifiable)."""
        denom = self.activating
        return (self.killed / denom) if denom else 0.0

    @property
    def survivors(self) -> List[CaseResult]:
        """Activating defects NO verifier caught -- the fleet's blind spots."""
        return [c for c in self.cases if c.activating and not c.killed]

    @property
    def ok(self) -> bool:
        # A fleet that judges a provably-good part unsound cannot be trusted, and
        # a run with nothing to measure cannot certify anything.
        if self.base_unsound_parts:
            return False
        if self.activating == 0:
            return False
        return self.mutation_score >= self.floor

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "mutation_score": self.mutation_score,
            "floor": self.floor,
            "fleet_size": self.fleet_size,
            "activating": self.activating,
            "non_activating": self.non_activating,
            "killed": self.killed,
            "survivors": [c.to_dict() for c in self.survivors],
            "base_unsound_parts": list(self.base_unsound_parts),
            "degraded": self.degraded,
            "note": self.note,
            "cases": [c.to_dict() for c in self.cases],
        }


def _is_activating(base: Any, mutant: Any) -> bool:
    """A defect activates only if it actually changed the artefact.

    Equality is by value (the ``Mesh`` dataclass is frozen, so ``!=`` is a
    structural compare). If a type does not support equality we conservatively
    treat the mutation as activating -- a defect we cannot prove inert is scored.
    """
    try:
        return bool(base != mutant)
    except Exception:
        return True


def check(
    parts: Mapping[str, Any],
    fleet: Sequence[Verifier],
    injectors: Mapping[str, Injector] = INJECTORS,
    classes: Sequence[str] = DEFECT_CLASSES,
    floor: float = DEFAULT_MS_FLOOR,
) -> MutationReport:
    """Score a verifier fleet by semantic mutation over ``parts``.

    For each provably-good ``part`` and each defect ``class`` we inject the
    defect (via the shared injectors) and ask the fleet to catch it. A defect is
    **killed** when at least one verifier judges the base part SOUND and the
    injected part UNSOUND. Non-activating defects (injector refused, or the
    geometry did not change) are recorded and excluded from the denominator,
    exactly as TDAD excludes non-activating mutants.

    Pure: no I/O, no kernel, no model. ``fleet`` and ``injectors`` are supplied
    by the caller so the same routine scores the real fleet and the self-check.
    """
    report = MutationReport(floor=floor, fleet_size=len(fleet))

    for part_name in sorted(parts):
        base = parts[part_name]
        # Which verifiers accept the clean part? Only those can legitimately kill
        # a mutant (a verifier that rejects everything catches nothing real).
        base_sound = [i for i, v in enumerate(fleet) if _verdict(v, base)]
        if not base_sound:
            report.base_unsound_parts.append(part_name)

        for cls in classes:
            injector = injectors.get(cls)
            if injector is None:
                raise DefectError("unknown defect class %r" % (cls,))
            try:
                mutant = injector(base)
            except DefectError:
                report.cases.append(
                    CaseResult(part_name, cls, activating=False, killed=False,
                               reason="refused"))
                continue
            if not _is_activating(base, mutant):
                report.cases.append(
                    CaseResult(part_name, cls, activating=False, killed=False,
                               reason="no-change"))
                continue
            killers = [i for i in base_sound if not _verdict(fleet[i], mutant)]
            report.cases.append(
                CaseResult(part_name, cls, activating=True,
                           killed=bool(killers), killers=tuple(killers)))
    return report


def _verdict(verifier: Verifier, artefact: Any) -> bool:
    """Run one verifier defensively: a verifier that throws did not judge SOUND."""
    try:
        return bool(verifier(artefact))
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Synthetic self-check (no kernel, no model)                                   #
# --------------------------------------------------------------------------- #
def _blind_verifier(_artefact: Any) -> bool:
    """A deliberately weak verifier that calls everything SOUND.

    It stands in for a fleet member with a blind spot: on its own it kills
    nothing, so the fixture proves the fleet must not rely on any single member.
    """
    return True


def _noop_injector(mesh: Mesh) -> Mesh:
    """A synthetic NON-ACTIVATING defect: it returns the part unchanged.

    Included in the self-check to exercise the denominator exclusion -- it must
    land in ``non_activating`` (reason ``no-change``) and never in MS.
    """
    return mesh


def self_check_report(floor: float = DEFAULT_MS_FLOOR) -> MutationReport:
    """Build and score the synthetic fixture: synthetic verifiers + defects.

    Parts are two provably-good meshes; the fleet pairs the real
    ``topology_verifier`` (which catches every class) with a blind verifier
    (which catches none), so the fleet's union still kills all four activating
    classes -> MS == 1.0. A synthetic non-activating ``noop`` defect is added to
    prove it is excluded from the denominator rather than counted as a miss.
    """
    parts = {
        "unit_tetrahedron": unit_tetrahedron(),
        "unit_cube": unit_cube_mesh(),
    }
    fleet: Sequence[Verifier] = (topology_verifier, _blind_verifier)
    injectors = dict(INJECTORS)
    injectors["noop_non_activating"] = _noop_injector
    classes = tuple(DEFECT_CLASSES) + ("noop_non_activating",)

    report = check(parts, fleet, injectors=injectors, classes=classes, floor=floor)
    report.degraded = True
    report.note = ("synthetic self-check (no kernel, no model): "
                   "topology_verifier + blind verifier over injected defects")
    return report


def _load_real_fleet() -> Optional[Sequence[Verifier]]:
    """Try to assemble the real verifier fleet / differential oracle.

    The real fleet is backed by independent geometry kernels that may be absent
    in a bare environment. This returns ``None`` (never raises) when the fleet
    cannot be built, so :func:`main` can degrade to the synthetic self-check and
    report the degradation instead of crashing.
    """
    try:  # pragma: no cover - exercised only where the real oracle is wired.
        from harnesscad.eval.quality.geometry.oracle_fleet import (  # type: ignore
            load_fleet,
        )
    except Exception:
        return None
    try:  # pragma: no cover
        fleet = list(load_fleet())
    except Exception:
        return None
    return fleet or None


def _load_real_parts() -> Optional[Mapping[str, Any]]:
    """Try to load provably-good reference parts for the real fleet.

    Returns ``None`` when unavailable so :func:`main` degrades cleanly.
    """
    try:  # pragma: no cover - exercised only where the corpus is wired.
        from harnesscad.eval.quality.geometry.oracle_fleet import (  # type: ignore
            load_known_good_parts,
        )
    except Exception:
        return None
    try:  # pragma: no cover
        parts = dict(load_known_good_parts())
    except Exception:
        return None
    return parts or None


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def format_text(report: MutationReport) -> str:
    lines: List[str] = []
    lines.append("VERIFIER-FLEET MUTATION SCORE GATE")
    lines.append("=" * 72)
    if report.degraded:
        lines.append("[degraded] %s" % (report.note or "real fleet unavailable"))
    lines.append("fleet size: %d verifier(s)" % report.fleet_size)
    lines.append("mutation score: %.4f  (floor %.4f)"
                 % (report.mutation_score, report.floor))
    lines.append("  killed:         %d" % report.killed)
    lines.append("  activating:     %d   (the MS denominator)" % report.activating)
    lines.append("  non-activating: %d   (excluded: refused or no-change)"
                 % report.non_activating)
    lines.append("")
    if report.base_unsound_parts:
        lines.append("BASE FAULT: no verifier judged these provably-good parts "
                     "sound -- the fixture, not the mutant, is wrong:")
        for name in report.base_unsound_parts:
            lines.append("  [base-unsound] %s" % name)
        lines.append("")
    if report.ok:
        lines.append("PASS: the fleet kills at least %.0f%% of activating defects."
                     % (report.floor * 100.0))
    else:
        lines.append("FAIL:")
        if report.activating == 0:
            lines.append("  [no-signal] nothing activated; MS is uncertifiable.")
        for c in report.survivors:
            lines.append("  [survivor] %s / %s escaped the whole fleet -- a blind "
                         "spot in the oracle." % (c.part, c.defect))
        lines.append("")
        lines.append("A low mutation score indicts the VERIFIERS, not the parts: "
                     "the oracle has a blind spot and cannot be trusted to gate "
                     "generation. Add a verifier that catches the survivors.")
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the synthetic fixture (no kernel, no model).")
    parser.add_argument("--floor", type=float, default=DEFAULT_MS_FLOOR,
                        help="minimum mutation score to pass (default %.2f)."
                        % DEFAULT_MS_FLOOR)
    parser.add_argument("--json", action="store_true", dest="as_json")


def run(args: argparse.Namespace) -> int:
    floor = getattr(args, "floor", DEFAULT_MS_FLOOR)
    if getattr(args, "selfcheck", False):
        report = self_check_report(floor=floor)
    else:
        fleet = _load_real_fleet()
        parts = _load_real_parts()
        if fleet is None or parts is None:
            # Degrade to the synthetic self-check rather than crash, and say so.
            report = self_check_report(floor=floor)
            report.note = ("real verifier fleet / kernel unavailable; "
                           "degraded to synthetic self-check")
        else:
            report = check(parts, fleet, floor=floor)

    if getattr(args, "as_json", False):
        import json
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_text(report))
    return 0 if report.ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mutation_score",
        description="Fail the build if the verifier fleet kills too few injected "
                    "defects (TDAD semantic mutation testing over the oracle).")
    add_arguments(parser)
    return run(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
