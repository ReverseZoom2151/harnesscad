"""Run the red team: certify the part, then show it to the whole fleet.

THE MEASUREMENT
---------------
For every attack that the oracle CERTIFIES as a correct, buildable, gate-accepted
part, run every verifier in the fleet over it and record every ERROR. Each one is
a FALSE POSITIVE: a rule rejecting a part that is provably fine.

    false-positive rate = (certified parts with >= 1 ERROR) / (certified parts)

That number is the fleet's PRECISION, measured adversarially -- against a corpus
the harness did not write, aimed deliberately at the boundary of every rule -- and
not on a corpus we wrote for ourselves, where a rule's precision is mostly a
measure of how carefully we avoided its blind spot.

WARNINGS ARE COUNTED, SEPARATELY, AND THEY MATTER
-------------------------------------------------
The pressure loop feeds a model back everything of severity ERROR *or* WARNING
(``pressure/metrics.BLOCKING_SEVERITIES``). So a WARNING on a good part is not
cosmetic: it is a false instruction handed to a model that will execute it. The
14b read a false diagnostic, changed exactly one field, and destroyed a correct
answer. A typed diagnostic is an INSTRUCTION, and the value of an instruction is
bounded above by its truth.

The headline count is ERRORs (the loop REJECTS on those). WARNINGs on certified
parts are reported underneath as false alarms, because a reader who only sees the
headline will underestimate what the loop actually says to the model.

THE DISCIPLINE
--------------
This module REPORTS. It does not fix. ``eval/verifiers/`` belongs to another
agent, and a red team that repaired the thing it was auditing would be producing a
number about itself.

COST. Every attack builds a real solid (the F-rep sampler marches a grid, ~1-3 s)
and runs the gate. The full sweep is a report a human asks for. The test suite
runs a handful, and the full sweep is opt-in behind ``HARNESSCAD_REDTEAM_FULL=1``
-- skipped LOUDLY, with a reason, never silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.eval.redteam.attacks import Attack, SEED, generate
from harnesscad.eval.redteam.oracle import Certificate, certify
from harnesscad.eval.selftest.probe import plan_opdag, resolve
from harnesscad.eval.verifiers import registry as fleet_registry
from harnesscad.eval.verifiers.verify import Severity

__all__ = ["FalsePositive", "RedTeamReport", "run", "format_text"]


@dataclass
class FalsePositive:
    """(verifier, op stream, why the part is actually fine). The deliverable."""

    verifier: str
    tier: str
    severity: str                      # "error" | "warning"
    codes: List[str]
    messages: List[str]
    attack: str
    family: str
    ops: List[str]                     # canonical JSON: replayable by hand
    why_the_part_is_fine: str
    proof: str                         # the oracle's certificate, in words

    def to_dict(self) -> dict:
        return {"verifier": self.verifier, "tier": self.tier,
                "severity": self.severity, "codes": self.codes,
                "messages": self.messages, "attack": self.attack,
                "family": self.family, "ops": self.ops,
                "why_the_part_is_fine": self.why_the_part_is_fine,
                "proof": self.proof}


@dataclass
class RedTeamReport:
    seed: int = SEED
    backend: str = "frep"
    attacks: int = 0
    certified: int = 0
    uncertified: List[Dict[str, str]] = field(default_factory=list)
    #: ERROR on a certified-good part. The loop REJECTS on these.
    false_positives: List[FalsePositive] = field(default_factory=list)
    #: WARNING on a certified-good part. The loop still SPEAKS these to the model.
    false_alarms: List[FalsePositive] = field(default_factory=list)
    #: verifiers that crashed on a good part (a bug, and it hides as a warning)
    crashed: List[Tuple[str, str]] = field(default_factory=list)
    skipped: str = ""

    @property
    def rejected_parts(self) -> List[str]:
        return sorted({fp.attack for fp in self.false_positives})

    @property
    def false_positive_rate(self) -> float:
        if not self.certified:
            return 0.0
        return len(self.rejected_parts) / float(self.certified)

    @property
    def warned_parts(self) -> List[str]:
        return sorted({fp.attack for fp in self.false_alarms})

    @property
    def by_verifier(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for fp in self.false_positives:
            out[fp.verifier] = out.get(fp.verifier, 0) + 1
        return dict(sorted(out.items()))

    @property
    def ok(self) -> bool:
        return not self.false_positives

    def to_dict(self) -> dict:
        return {
            "oracle": "redteam", "ok": self.ok, "seed": self.seed,
            "backend": self.backend, "skipped": self.skipped,
            "attacks": self.attacks, "certified": self.certified,
            "uncertified": self.uncertified,
            "parts_rejected": self.rejected_parts,
            "false_positive_rate": self.false_positive_rate,
            "false_positives_by_verifier": self.by_verifier,
            "false_positives": [fp.to_dict() for fp in self.false_positives],
            "false_alarms": [fp.to_dict() for fp in self.false_alarms],
            "crashed": [{"verifier": v, "attack": a} for v, a in self.crashed],
        }


def _fleet_on(attack: Attack, engine: Any,
              fleet: Sequence[Any]) -> Dict[str, Tuple[List[Any], bool, str]]:
    """{verifier: (diagnostics, crashed, tier)} for one part.

    The fleet is shown the WHOLE PLAN (``probe.plan_opdag``), including any op a
    backend declined: the LINT tier reads the op stream, and judging it on evidence
    it was never given would manufacture a clean bill of health.

    ``verifiers=[v]`` is load-bearing. Without it ``run_all`` re-DISCOVERS the
    fleet and scores nothing for the verifier we asked about.
    """
    state = fleet_registry.model_state(engine, plan_opdag(attack.ops))
    out: Dict[str, Tuple[List[Any], bool, str]] = {}
    for v in fleet:
        name = getattr(v, "name", type(v).__name__)
        tier = getattr(v, "tier", "lint")
        diags = fleet_registry.run_all(state, tiers=fleet_registry.TIERS,
                                       only=[name], verifiers=[v])
        crashed = any(d.code == "verifier-error" for d in diags)
        out[name] = ([d for d in diags if d.code != "verifier-error"], crashed, tier)
    return out


def run(seed: int = SEED,
        backend: str = "frep",
        families: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        fleet: Optional[Sequence[Any]] = None) -> RedTeamReport:
    """Certify every attack, then show the certified ones to the whole fleet."""
    r = RedTeamReport(seed=seed, backend=backend)
    engine, skip = resolve(backend)
    if engine is None:
        r.skipped = skip
        return r
    the_fleet = list(fleet) if fleet is not None else fleet_registry.discover()

    the_attacks = generate(seed=seed, families=families, limit=limit)
    r.attacks = len(the_attacks)

    for attack in the_attacks:
        cert: Certificate = certify(attack, backend=backend)
        if not cert.certified:
            r.uncertified.append({"attack": attack.name, "family": attack.family,
                                  "why": cert.reason})
            continue
        r.certified += 1

        # A FRESH engine per part: the fleet reads the backend's current solid, and
        # a stale one would attribute one part's geometry to another.
        build_engine, _ = resolve(backend)
        from harnesscad.core.loop import HarnessSession
        try:
            HarnessSession(build_engine, verify_level="core").apply_ops(
                list(attack.ops))
        except Exception:                                      # noqa: BLE001
            pass

        fired = _fleet_on(attack, build_engine, the_fleet)
        for name, (diags, crashed, tier) in fired.items():
            if crashed:
                r.crashed.append((name, attack.name))
            errors = [d for d in diags if d.severity is Severity.ERROR]
            warns = [d for d in diags if d.severity is Severity.WARNING]
            if errors:
                r.false_positives.append(FalsePositive(
                    verifier=name, tier=tier, severity="error",
                    codes=sorted({d.code for d in errors}),
                    messages=[d.message for d in errors],
                    attack=attack.name, family=attack.family,
                    ops=list(attack.ops_json()),
                    why_the_part_is_fine=attack.why_fine,
                    proof=cert.reason))
            if warns:
                r.false_alarms.append(FalsePositive(
                    verifier=name, tier=tier, severity="warning",
                    codes=sorted({d.code for d in warns}),
                    messages=[d.message for d in warns],
                    attack=attack.name, family=attack.family,
                    ops=list(attack.ops_json()),
                    why_the_part_is_fine=attack.why_fine,
                    proof=cert.reason))
    return r


def format_text(report: RedTeamReport) -> str:
    lines: List[str] = []
    lines.append("RED TEAM -- adversarial hunt for FALSE POSITIVES in the fleet")
    lines.append("=" * 78)
    if report.skipped:
        lines.append("skipped: " + report.skipped)
        return "\n".join(lines)
    lines.append("seed %d, engine %s. %d attacks generated, %d CERTIFIED as "
                 "provably-correct parts." % (report.seed, report.backend,
                                              report.attacks, report.certified))
    lines.append("")
    lines.append("HEADLINE")
    lines.append("  parts rejected by at least one verifier : %d / %d"
                 % (len(report.rejected_parts), report.certified))
    lines.append("  ADVERSARIAL FALSE-POSITIVE RATE         : %.0f%%"
                 % (100.0 * report.false_positive_rate))
    lines.append("  (a false positive is a verifier raising an ERROR on a part "
                 "proven correct by")
    lines.append("   arithmetic, built by an engine at that exact volume, and "
                 "re-measured and")
    lines.append("   accepted by io/gate.py. There is no appeal: the part is good.)")
    lines.append("")
    if report.by_verifier:
        lines.append("%-24s %8s" % ("verifier", "FPs"))
        lines.append("-" * 34)
        for name, n in report.by_verifier.items():
            lines.append("%-24s %8d" % (name, n))
        lines.append("")
    if report.false_positives:
        lines.append("FALSE POSITIVES -- (verifier, op stream, why the part is fine)")
        lines.append("-" * 78)
        for fp in report.false_positives:
            lines.append("  %s [%s] on %s" % (fp.verifier, ",".join(fp.codes),
                                              fp.attack))
            lines.append("      said : %s" % (fp.messages[0][:150]))
            lines.append("      FINE : %s" % fp.why_the_part_is_fine)
            lines.append("      proof: %s" % fp.proof)
            lines.append("      ops  : %s" % " ".join(fp.ops))
            lines.append("")
    else:
        lines.append("NO FALSE POSITIVES. Every certified-good part passed every "
                     "verifier.")
        lines.append("")
    if report.false_alarms:
        lines.append("FALSE ALARMS -- WARNINGs on provably-good parts")
        lines.append("-" * 78)
        lines.append("Not rejections, but the pressure loop SPEAKS warnings to the "
                     "model (metrics.BLOCKING_SEVERITIES), so each of these is a "
                     "false instruction a capable model will execute precisely.")
        seen = set()
        for fp in report.false_alarms:
            key = (fp.verifier, tuple(fp.codes))
            if key in seen:
                continue
            seen.add(key)
            n = sum(1 for x in report.false_alarms
                    if x.verifier == fp.verifier and tuple(x.codes) == key[1])
            lines.append("  %-20s %-28s on %d part(s), e.g. %s"
                         % (fp.verifier, ",".join(fp.codes), n, fp.attack))
            lines.append("      %s" % fp.messages[0][:150])
        lines.append("")
    if report.crashed:
        lines.append("VERIFIERS THAT CRASHED on a good part (%d)" % len(report.crashed))
        for v, a in sorted(set(report.crashed)):
            lines.append("  %-24s on %s" % (v, a))
        lines.append("")
    if report.uncertified:
        lines.append("UNCERTIFIED (%d) -- attacks the oracle could NOT prove good, "
                     "so they accuse nobody" % len(report.uncertified))
        lines.append("-" * 78)
        for u in report.uncertified[:12]:
            lines.append("  %-34s %s" % (u["attack"], u["why"][:110]))
        if len(report.uncertified) > 12:
            lines.append("  ... and %d more" % (len(report.uncertified) - 12))
    return "\n".join(lines)
