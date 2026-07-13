"""The GOVERNANCE surface -- the gates, the evidence and the audit trail.

``governance/`` carried three kinds of thing, all unreachable:

*   **security** -- the trust boundaries. A file arriving from outside, a prompt
    arriving from an untrusted tier, an image with sensitive regions in it: each
    has a gate that says admit / redact / refuse, and writes an audit event.
*   **research** -- the statistics that decide whether a change is REAL: effect
    size, inter-rater agreement, paired ablations, a model-promotion gate, a
    resource profile. These exist so "the new prompt is better" has to be earned.
*   **audit** -- the closure check over the corpus register: does every claimed
    idea actually resolve to something on disk?

This module dispatches into all three, because they are the same move: a
DECISION with a stated reason, recorded. Nothing here is advisory-by-vibes; every
route returns the reasons it decided the way it did.

    ingest_gate(path)            -> admit / refuse a file, and redact its metadata
    prompt_gate(text, trust)     -> admit / refuse an untrusted prompt
    privacy_gate(regions)        -> release / hold an image
    effect(a, b)                 -> is the difference real, and how big
    agreement(x, y)              -> do two raters actually agree (kappa)
    promotion(...)               -> may this candidate replace the baseline
    closure(register)            -> does the corpus register resolve

RESOURCE PROFILING IS INJECTED, NOT MEASURED HERE. :func:`profile` takes the
sampler; the harness does not ship one, because a memory sampler that lies is
worse than no number at all.

Adapters only: the governance modules are never modified. Deterministic,
stdlib-only, no network.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "GovernanceError",
    "DEFAULT_TOOLS",
    "ingest_gate",
    "redact",
    "prompt_gate",
    "tool_gate",
    "privacy_gate",
    "effect",
    "agreement",
    "ablation",
    "role_ablation",
    "judge_ablation",
    "promotion",
    "profile",
    "evidence_gate",
    "closure",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_GOV = "harnesscad.governance."


class GovernanceError(ValueError):
    """Base class for every governance-surface failure."""


#: The harness's own tool surface -- the default allow-list for the trust gate.
#: These are exactly the CLI subcommands: a caller cannot be authorised for a
#: tool the harness does not have.
DEFAULT_TOOLS: Tuple[str, ...] = (
    "apply", "bench", "build", "capabilities", "catalog", "dataset", "demo",
    "export", "fabricate", "formats", "govern", "ingest", "procedural",
    "program", "reconstruct", "report", "spec",
)


def _tool_policy(policy: Optional[Mapping[str, Any]]):
    """The ToolPolicy, defaulting its allow-list to the harness's own tools."""
    from harnesscad.governance.security.tool_gate import ToolPolicy

    kw = dict(policy or {})
    kw.setdefault("allowed_tools", frozenset(DEFAULT_TOOLS))
    if not isinstance(kw["allowed_tools"], frozenset):
        kw["allowed_tools"] = frozenset(kw["allowed_tools"])
    if "minimum_trust" in kw and not hasattr(kw["minimum_trust"], "name"):
        kw["minimum_trust"] = _tier(kw["minimum_trust"])
    return ToolPolicy(**kw)


def _tier(name: str):
    """'untrusted' / 'user' / 'project' / 'system' -> the TrustTier enum member."""
    from harnesscad.governance.security.tool_gate import TrustTier

    try:
        return TrustTier[str(name).upper()]
    except KeyError:
        raise GovernanceError(
            "unknown trust tier %r; known: %s"
            % (name, ", ".join(t.name.lower() for t in TrustTier))) from None


# --------------------------------------------------------------------------- #
# Security gates
# --------------------------------------------------------------------------- #
def ingest_gate(path: str, metadata: Optional[Mapping[str, Any]] = None,
                policy: Optional[Mapping[str, Any]] = None,
                root: Optional[str] = None) -> dict:
    """Admit / refuse a file arriving from outside, and log the decision.

    ``policy`` overrides the default :class:`DataPolicy` fields (allowed
    extensions, size ceiling, execution mode, network, PII/secret redaction).
    """
    from harnesscad.governance.security.policy import DataPolicy, SecureIngestGate

    pol = DataPolicy(**dict(policy)) if policy else DataPolicy()
    gate = SecureIngestGate(pol)
    decision = gate.inspect(path, dict(metadata or {}), root=root)
    return {
        "allowed": bool(decision.allowed),
        "code": decision.code,
        "reason": decision.reason,
        "content_sha256": decision.content_sha256,
        "redacted_metadata": dict(decision.redacted_metadata or {}),
        "events": [dict(e) for e in gate.audit_log()],
    }


def redact(metadata: Mapping[str, Any],
           policy: Optional[Mapping[str, Any]] = None) -> dict:
    """Strip the PII / secrets a policy forbids from a metadata blob."""
    from harnesscad.governance.security.policy import DataPolicy, redact_metadata

    pol = DataPolicy(**dict(policy)) if policy else DataPolicy()
    return redact_metadata(dict(metadata), pol)


def prompt_gate(text: str, trust: str = "untrusted",
                policy: Optional[Mapping[str, Any]] = None) -> dict:
    """Admit / refuse a prompt from a given TRUST TIER, with the risks it carries."""
    from harnesscad.governance.security.tool_gate import ToolTrustGate, prompt_risks

    gate = ToolTrustGate(_tool_policy(policy))
    decision = gate.inspect_prompt(text, _tier(trust))
    out = decision.to_dict()
    out["allowed"] = bool(decision.allowed)
    out["risks"] = prompt_risks(text)
    return out


def tool_gate(tool: str, trust: str = "untrusted",
              arguments: Optional[Mapping[str, Any]] = None,
              policy: Optional[Mapping[str, Any]] = None) -> dict:
    """May a caller at this trust tier invoke this tool, with these arguments?"""
    from harnesscad.governance.security.tool_gate import ToolTrustGate

    gate = ToolTrustGate(_tool_policy(policy))
    decision = gate.authorize_tool(tool, _tier(trust), dict(arguments or {}))
    out = decision.to_dict()
    out["allowed"] = bool(decision.allowed)
    return out


def privacy_gate(regions: Sequence[Mapping[str, Any]],
                 manually_verified: bool = False) -> dict:
    """Release / hold an image, given the sensitive regions somebody else DETECTED.

    The detector is not here: region detection needs a trained model. This gate
    reads its output and decides, which is the part that must be deterministic.
    """
    from harnesscad.governance.security.image_privacy import PrivacyRegion, release_gate

    rs = [PrivacyRegion(**dict(r)) for r in regions]
    decision = release_gate(rs, manually_verified=bool(manually_verified))
    return {"releasable": bool(decision.releasable),
            "reasons": list(decision.reasons)}


# --------------------------------------------------------------------------- #
# Research: is the difference real?
# --------------------------------------------------------------------------- #
def effect(a: Sequence[float], b: Sequence[float]) -> dict:
    """Effect size + uncertainty between two samples. A p-value is not a verdict."""
    from harnesscad.governance.research.statistics import (
        compare_samples, effect_magnitude,
    )

    r = compare_samples(list(a), list(b))
    return {
        "mean_a": r.mean_a, "mean_b": r.mean_b,
        "difference": r.difference, "cohen_d": r.cohen_d,
        "magnitude": effect_magnitude(r.cohen_d),
        "ci95_low": r.ci95_low, "ci95_high": r.ci95_high,
    }


def agreement(first: Iterable[Any], second: Iterable[Any]) -> dict:
    """Cohen's kappa: do two raters agree beyond what chance would give them?"""
    from harnesscad.governance.research.agreement import cohen_kappa

    r = cohen_kappa(list(first), list(second))
    return {"kappa": r.kappa, "observed": r.observed,
            "expected": r.expected, "n": r.n,
            "labels": list(r.labels),
            "confusion": {str(k): v for k, v in sorted(r.confusion.items())}}


def ablation(rows: Sequence[Mapping[str, Any]], metric: str) -> Any:
    """A paired, stratified ablation summary over per-case rows."""
    from harnesscad.governance.research.ablation_matrix import compare_ablation

    return compare_ablation([dict(r) for r in rows], metric=metric)


def role_ablation(baseline: Mapping[str, float],
                  without_role: Mapping[str, float], role: str) -> dict:
    """Did removing this agent role HURT? (If not, the role is not earning its keep.)"""
    from harnesscad.governance.research.role_ablation import compare_role_ablation

    r = compare_role_ablation(dict(baseline), dict(without_role), role)
    return {"removed_role": r.removed_role, "harmful": bool(r.harmful),
            "deltas": dict(r.deltas)}


def judge_ablation(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Candidate-controlled ablation of a preference-judge pipeline."""
    from harnesscad.governance.research.judge_ablation import judge_ablation as _ja

    return _ja([dict(r) for r in rows])


def promotion(baseline_quality: float, candidate_quality: float,
              candidate_peak_memory: int, memory_ceiling: int,
              evidence_count: int, minimum_improvement: float = 0.0,
              minimum_evidence: int = 1) -> dict:
    """May this candidate model REPLACE the baseline? Quality, memory and evidence."""
    from harnesscad.governance.research.model_promotion import promotion_gate

    d = promotion_gate(baseline_quality=float(baseline_quality),
                       candidate_quality=float(candidate_quality),
                       candidate_peak_memory=int(candidate_peak_memory),
                       memory_ceiling=int(memory_ceiling),
                       minimum_improvement=float(minimum_improvement),
                       evidence_count=int(evidence_count),
                       minimum_evidence=int(minimum_evidence))
    return {"promoted": bool(d.promoted), "improvement": d.improvement,
            "reasons": list(d.reasons)}


def profile(call, sampler) -> Any:
    """Profile a call with an INJECTED sampler. The harness ships no sampler.

    A resource number is only worth reporting if you can say how it was measured;
    the provenance rides along with the profile.
    """
    from harnesscad.governance.research.resource_profile import profile as _profile

    return _profile(call, sampler)


def evidence_gate():
    """A fresh :class:`ResearchGovernance` -- stage evidence, claim, check, decide."""
    from harnesscad.governance.research.evidence_gate import ResearchGovernance

    return ResearchGovernance()


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def closure(register: Mapping[str, Any], repo_root: str, corpus_root: str) -> dict:
    """Does every idea claimed in the corpus register actually resolve on disk?"""
    from harnesscad.governance.audit.closure import validate_register

    report = validate_register(dict(register), repo_root=repo_root,
                               corpus_root=corpus_root)
    return report.to_dict()


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    return {e.dotted: e for e in capability_registry.index()
            if e.dotted.startswith(_GOV)}


def _available(dotted: str) -> bool:
    return dotted in _index()


_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("security", "ingest_gate", _GOV + "security.policy",
     "admit/refuse an incoming file; redact its metadata; write the audit event"),
    ("security", "prompt_gate", _GOV + "security.tool_gate",
     "trust-boundary enforcement over prompts and tool calls"),
    ("security", "privacy_gate", _GOV + "security.image_privacy",
     "release/hold an image given externally DETECTED sensitive regions"),
    ("research", "effect", _GOV + "research.statistics",
     "effect size + uncertainty between two samples"),
    ("research", "agreement", _GOV + "research.agreement",
     "Cohen's kappa: agreement beyond chance"),
    ("research", "ablation", _GOV + "research.ablation_matrix",
     "paired, stratified ablation summary"),
    ("research", "role_ablation", _GOV + "research.role_ablation",
     "did removing an agent role actually hurt?"),
    ("research", "judge_ablation", _GOV + "research.judge_ablation",
     "candidate-controlled ablation of a preference-judge pipeline"),
    ("research", "promotion", _GOV + "research.model_promotion",
     "may a candidate model replace the baseline? quality/memory/evidence"),
    ("research", "profile", _GOV + "research.resource_profile",
     "profile a call with an INJECTED sampler (provenance rides along)"),
    ("research", "evidence_gate", _GOV + "research.evidence_gate",
     "stage evidence -> claim -> check -> gate decision, with rollback"),
    ("audit", "closure", _GOV + "audit.closure",
     "does the corpus register resolve to real files?"),
)


def routed_modules() -> Tuple[str, ...]:
    return tuple(sorted({m for _g, _n, m, _d in _ROUTES if _available(m)}))


def discover() -> List[dict]:
    return [{"group": g, "route": n, "module": m, "doc": d,
             "present": _available(m)}
            for (g, n, m, d) in _ROUTES]


def unadapted() -> List[Tuple[str, str]]:
    routed = set(routed_modules())
    return [(d, "no route yet") for d in sorted(_index())
            if d not in routed and not d.endswith(".registry")]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every governance route")
    parser.add_argument("--prompt", default=None,
                        help="run a prompt through the trust gate")
    parser.add_argument("--trust", default="untrusted",
                        help="the trust tier of --prompt / --tool")
    parser.add_argument("--tool", default=None,
                        help="authorise a tool call at --trust")
    parser.add_argument("--ingest", default=None,
                        help="run a file path through the ingest policy gate")
    parser.add_argument("--effect", default=None, metavar="JSON",
                        help='two samples: {"a": [...], "b": [...]}')
    parser.add_argument("--unadapted", action="store_true",
                        help="list governance modules with no route")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "prompt", None):
        d = prompt_gate(args.prompt, getattr(args, "trust", "untrusted"))
        print(json.dumps(d, indent=2, sort_keys=True, default=str))
        return 0 if d["allowed"] else 1

    if getattr(args, "tool", None):
        d = tool_gate(args.tool, getattr(args, "trust", "untrusted"))
        print(json.dumps(d, indent=2, sort_keys=True, default=str))
        return 0 if d.get("allowed") else 1

    if getattr(args, "ingest", None):
        d = ingest_gate(args.ingest)
        print(json.dumps(d, indent=2, sort_keys=True, default=str))
        return 0 if d["allowed"] else 1

    if getattr(args, "effect", None):
        payload = json.loads(args.effect)
        print(json.dumps(effect(payload["a"], payload["b"]),
                         indent=2, sort_keys=True))
        return 0

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-9s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad govern",
        description="governance surface: security gates, research evidence, audit")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
