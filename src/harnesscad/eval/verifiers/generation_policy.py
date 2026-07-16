"""Placement-vs-generation policy gate: place authored parts, don't generate.

Source: ``resources/cad_repos/CADCLAW-main`` (``AGENTS.md``, the repo's
"load-bearing policy" for agents). CADCLAW's field-tested rule (M3-CRETE,
2026-04-26): an agent asked to add a part should *place an authored STEP*,
not synthesise parametric geometry -- because the constraints that survive
contact with a real assembly (clearance, shaft offsets, mating interfaces)
live in the user's CAD model, not in any spec. Generated bolt patterns
against assumed positions were uniformly wrong. The policy therefore
whitelists a small set of genuinely-parametric categories (extrusion stock,
v-wheels, belt segments, and fastener stand-ins only when the project
declares it wants them) and names the canonical violations: plates with hole
patterns, brackets, bolt-circle helpers, and stand-in geometry nobody asked
for.

The earlier CADCLAW mining pass (repo 10) took the numeric verifiers
(tolerance stack, beam screening, clearance shift, exploded view, claim
audit) and left this agent policy unmined. It is a deterministic plan-level
gate in exactly the harness's sense: audit the *op stream's provenance*
before the kernel runs.

* :class:`PartPlan` -- one planned part: name, category, provenance
  (``authored_step`` with its path, ``generated``, or ``unknown``), and
  feature flags (hole pattern, requested-by-user).
* :class:`PolicyConfig` -- the whitelist, extendable per project, with the
  fastener opt-in flag.
* :func:`audit` -- findings per part (``forbidden-generated``,
  ``unsolicited-standin``, ``missing-provenance``, ``missing-step-path``)
  plus the confidence-budget summary CADCLAW requires of every report:
  what was checked, what was assumed, what could not be checked.

Stdlib only, deterministic, absolute imports. ``--selfcheck`` replays the
policy against the M3-CRETE failure pattern.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "GENERATABLE_DEFAULT",
    "FORBIDDEN_GENERATED",
    "PartPlan",
    "PolicyConfig",
    "Finding",
    "PolicyReport",
    "audit",
    "main",
]

#: Categories CADCLAW's policy allows an agent to generate by default:
#: genuinely parametric (fixed cross-section or fixed geometry, one variable).
GENERATABLE_DEFAULT: Tuple[str, ...] = (
    "extrusion_stock",   # C-beam / V-slot bars cut to length ("cbeam")
    "v_wheel",           # fixed-geometry rolling elements
    "belt_segment",      # already special-cased by belt_heuristic
)

#: Categories the policy explicitly names as the canonical wrong moves when
#: generated instead of placed from an authored STEP.
FORBIDDEN_GENERATED: Tuple[str, ...] = (
    "plate",             # esp. plates with hole patterns
    "bracket",
    "mount",
    "gusset",
    "motor_adapter",
    "spacer",            # spacers with bolt patterns
    "idler_holder",
)

#: Fastener stand-ins are generatable ONLY when the project declares it.
FASTENER_CATEGORY = "fastener_standin"

PROVENANCES = ("authored_step", "generated", "unknown")


@dataclass(frozen=True)
class PartPlan:
    """One part the plan intends to put into the assembly."""
    name: str
    category: str                       # e.g. "plate", "extrusion_stock"
    provenance: str = "unknown"         # "authored_step" | "generated" | "unknown"
    step_path: str = ""                 # required when provenance is authored_step
    has_hole_pattern: bool = False
    requested_by_user: bool = True      # stand-ins nobody asked for are flagged


@dataclass
class PolicyConfig:
    """The project's placement policy."""
    allow_generated_fasteners: bool = False
    extra_generatable: Tuple[str, ...] = ()
    extra_forbidden: Tuple[str, ...] = ()

    def generatable(self) -> Tuple[str, ...]:
        cats = GENERATABLE_DEFAULT + tuple(self.extra_generatable)
        if self.allow_generated_fasteners:
            cats = cats + (FASTENER_CATEGORY,)
        return cats

    def forbidden(self) -> Tuple[str, ...]:
        return FORBIDDEN_GENERATED + tuple(self.extra_forbidden)


@dataclass(frozen=True)
class Finding:
    """One policy finding, always with the fix the policy prescribes."""
    part: str
    kind: str        # "forbidden-generated" | "unsolicited-standin" |
                     # "missing-provenance" | "missing-step-path" |
                     # "undeclared-fastener"
    severity: str    # "error" | "warning"
    message: str
    fix: str

    def to_dict(self) -> dict:
        return {"part": self.part, "kind": self.kind, "severity": self.severity,
                "message": self.message, "fix": self.fix}


@dataclass
class PolicyReport:
    """Findings plus CADCLAW's confidence budget: checked / assumed / unchecked."""
    findings: List[Finding] = field(default_factory=list)
    checked: List[str] = field(default_factory=list)
    assumed: List[str] = field(default_factory=list)
    not_checked: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    def to_dict(self) -> dict:
        return {"ok": self.ok,
                "findings": [f.to_dict() for f in self.findings],
                "checked": list(self.checked),
                "assumed": list(self.assumed),
                "not_checked": list(self.not_checked)}


def _audit_part(part: PartPlan, config: PolicyConfig) -> List[Finding]:
    findings: List[Finding] = []
    if part.provenance not in PROVENANCES:
        findings.append(Finding(
            part=part.name, kind="missing-provenance", severity="error",
            message=f"unknown provenance '{part.provenance}'",
            fix="declare authored_step (with path) or generated"))
        return findings

    if part.provenance == "unknown":
        findings.append(Finding(
            part=part.name, kind="missing-provenance", severity="error",
            message="cannot tell whether this part is authored or a stand-in",
            fix="ask the user: where is the authored STEP for this part?"))
        return findings

    if part.provenance == "authored_step":
        if not part.step_path:
            findings.append(Finding(
                part=part.name, kind="missing-step-path", severity="error",
                message="placed part does not name the authored STEP it came from",
                fix="record the source STEP path with the placement"))
        return findings

    # provenance == "generated"
    category = part.category.strip().lower()
    if category in config.forbidden():
        detail = " with a hole pattern" if part.has_hole_pattern else ""
        findings.append(Finding(
            part=part.name, kind="forbidden-generated", severity="error",
            message=(f"generated '{category}'{detail}: hole positions and "
                     "interfaces depend on assembly context, not spec data"),
            fix="author it in the native CAD package and place it via STEP"))
    elif category == FASTENER_CATEGORY and not config.allow_generated_fasteners:
        findings.append(Finding(
            part=part.name, kind="undeclared-fastener", severity="error",
            message=("generated fastener body, but the project has not "
                     "declared it wants generated fastener stand-ins"),
            fix="set allow_generated_fasteners=True or let the BOM audit "
                "account for fasteners from text"))
    elif category not in config.generatable():
        findings.append(Finding(
            part=part.name, kind="forbidden-generated", severity="error",
            message=(f"category '{category}' is outside the generatable "
                     "whitelist; default to placement, not generation"),
            fix="ask the user whether to author it before building anything "
                "parametric"))
    if not part.requested_by_user:
        findings.append(Finding(
            part=part.name, kind="unsolicited-standin", severity="error",
            message="stand-in geometry the user did not ask for "
                    "(the parity gate exists to catch these)",
            fix="remove it, or get the user's explicit go-ahead"))
    return findings


def audit(parts: Sequence[PartPlan],
          config: Optional[PolicyConfig] = None) -> PolicyReport:
    """Audit a planned part list against the placement policy."""
    cfg = config or PolicyConfig()
    report = PolicyReport()
    for part in parts:
        report.findings.extend(_audit_part(part, cfg))
        report.checked.append(
            f"{part.name}: provenance={part.provenance}, category={part.category}")
    report.assumed.append(
        "category labels are taken from the plan as stated, not re-derived "
        "from geometry")
    report.not_checked.append(
        "whether an authored STEP's contents actually match its label "
        "(that is the parity gate's job)")
    report.not_checked.append(
        "hole positions inside authored parts (assembly-context data the "
        "policy defers to the user's CAD model)")
    return report


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------

def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            failures.append(message)

    # The M3-CRETE failure pattern: generated plates with hole patterns.
    m3_crete = [
        PartPlan(name="y_mount_plate", category="plate", provenance="generated",
                 has_hole_pattern=True),
        PartPlan(name="z_mount_plate", category="plate", provenance="generated",
                 has_hole_pattern=True),
        PartPlan(name="cbeam_500", category="extrusion_stock",
                 provenance="generated"),
    ]
    report = audit(m3_crete)
    check(not report.ok, "generated hole-pattern plates fail the gate")
    plate_findings = [f for f in report.findings
                      if f.kind == "forbidden-generated"]
    check(len(plate_findings) == 2, "both plates flagged, the cbeam not")
    check(all("STEP" in f.fix for f in plate_findings),
          "the fix prescribes authored-STEP placement")

    # The intended workflow passes: authored parts placed with named STEPs,
    # parametric stock generated.
    good = [
        PartPlan(name="y_mount_plate", category="plate",
                 provenance="authored_step", step_path="cad/y_mount.step"),
        PartPlan(name="cbeam_500", category="extrusion_stock",
                 provenance="generated"),
        PartPlan(name="v_wheel", category="v_wheel", provenance="generated"),
    ]
    good_report = audit(good)
    check(good_report.ok, "authored placement + parametric stock passes: "
          + "; ".join(f.message for f in good_report.findings))

    # Placement without a named source STEP is an honesty violation.
    unnamed = audit([PartPlan(name="gusset", category="gusset",
                              provenance="authored_step")])
    check(any(f.kind == "missing-step-path" for f in unnamed.findings),
          "placement must name its authored STEP")

    # Unknown provenance means: ask, don't build.
    unknown = audit([PartPlan(name="mystery", category="bracket")])
    check(any(f.kind == "missing-provenance" and "ask the user" in f.fix
              for f in unknown.findings), "unknown provenance asks the user")

    # Fastener stand-ins are opt-in.
    fast = [PartPlan(name="m5_bolt", category="fastener_standin",
                     provenance="generated")]
    check(not audit(fast).ok, "undeclared fastener stand-in fails")
    check(audit(fast, PolicyConfig(allow_generated_fasteners=True)).ok,
          "declared fastener stand-in passes")

    # Stand-ins nobody asked for are flagged even in a generatable category.
    standin = audit([PartPlan(name="approx_wheel", category="v_wheel",
                              provenance="generated", requested_by_user=False)])
    check(any(f.kind == "unsolicited-standin" for f in standin.findings),
          "unsolicited stand-in flagged")

    # Whitelist is extendable per project.
    custom = PolicyConfig(extra_generatable=("belt_clip",))
    check(audit([PartPlan(name="clip", category="belt_clip",
                          provenance="generated")], custom).ok,
          "project can extend the whitelist")

    # Confidence budget is always present.
    check(report.checked and report.assumed and report.not_checked,
          "confidence budget populated")

    if failures:
        for f in failures:
            print(f"selfcheck FAIL: {f}")
        return 1
    print("generation_policy selfcheck: OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Placement-vs-generation policy gate (CADCLAW)")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
