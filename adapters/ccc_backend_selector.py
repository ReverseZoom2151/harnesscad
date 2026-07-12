"""Deterministic backend selection over the code-CAD ecosystem catalogue.

Two jobs, both grounded in ``adapters/ccc_codecad_ecosystem`` (mined from the
curated-code-cad list) and cross-checked against what this harness can actually
execute (``adapters/cadhub_language_registry``):

1. ``rank(Requirement)`` -- given a task's needs (a paradigm, a kernel class, an
   output format, a host language, whether internal fillets are required, whether
   it must run in the harness), score every catalogued system with an explainable,
   deterministic rubric and return a ranked list with reasons. This is the
   "which code-CAD system should I target?" question a text-to-CAD planner has to
   answer BEFORE it picks a code generator.

2. ``coverage_report()`` -- what fraction of the ecosystem the harness's own
   backends cover, and which paradigms / kernels / export formats have no
   supported backend at all. This is the honest gap list, not a wish list.

The rubric encodes the curated list's own editorial claims (see
``ccc_codecad_ecosystem.CSG_CAVEAT`` and ``REPRESENTATION_NOTES``): B-rep is
preferred for parts that need internal fillets or exact exchange (STEP), mesh
kernels are fine for 3d printing, implicit/SDF kernels are the pick for organic
blends and generative art, and transpilers carry no kernel of their own.

Deterministic: pure functions over a frozen table; ties broken by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from adapters import cadhub_language_registry as langreg
from adapters import ccc_codecad_ecosystem as eco

# ---------------------------------------------------------------------------
# What the harness can actually drive
# ---------------------------------------------------------------------------

# Languages the harness can execute end-to-end (CadHub language registry).
def executable_systems() -> List[str]:
    """Catalogue names for the languages the harness's runner registry knows."""
    return sorted(n for n in langreg.language_names() if eco.has(n))


# Systems the harness can only EMIT source for (no in-harness evaluation):
# programs/solidpy_scad_emit.py (SolidPython -> .scad),
# formats/angelcad_xcsg_xml.py + formats/angelcad_amf_codec.py (AngelCAD),
# backends/ocp_occt_api_catalog.py (OCCT/OCP API surface, no kernel bundled).
EMIT_ONLY_SYSTEMS: Tuple[str, ...] = ("angelcad", "opencascade", "solidpython")

SUPPORT_EXECUTE = "execute"
SUPPORT_EMIT = "emit"
SUPPORT_NONE = "none"


def support_level(name: str) -> str:
    """How well the harness supports a catalogued system."""
    spec = eco.get(name)
    if spec.name in executable_systems():
        return SUPPORT_EXECUTE
    if spec.name in EMIT_ONLY_SYSTEMS:
        return SUPPORT_EMIT
    return SUPPORT_NONE


# ---------------------------------------------------------------------------
# Requirements + rubric
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Requirement:
    """What a modelling task needs from a code-CAD system."""

    paradigm: Optional[str] = None  # eco.PARA_*
    kernel: Optional[str] = None  # eco.K_*
    host_language: Optional[str] = None  # "python", "javascript", ...
    export_format: Optional[str] = None  # "step", "stl", ...
    import_format: Optional[str] = None
    internal_fillets: bool = False  # the list's canonical CSG pain point
    exact_exchange: bool = False  # STEP/IGES handoff to downstream MCAD
    organic_blends: bool = False  # smooth/implicit blends, generative art
    browser: bool = False  # must have an online editor
    must_be_executable: bool = False  # must run inside this harness
    scripting_only: bool = True  # exclude node/block editors


@dataclass(frozen=True)
class Candidate:
    """A scored system with the reasons behind its score."""

    name: str
    score: int
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    support: str = SUPPORT_NONE


# Hard filters return None (excluded); soft signals add to the score.
def _score(spec: eco.SystemSpec, req: Requirement) -> Optional[Candidate]:
    reasons: List[str] = []
    score = 0

    if req.scripting_only and not spec.scripting:
        return None
    if req.paradigm is not None:
        if not spec.has_paradigm(req.paradigm):
            return None
        reasons.append("paradigm:%s" % req.paradigm)
        score += 20
    if req.kernel is not None:
        if spec.kernel != req.kernel:
            return None
        reasons.append("kernel:%s" % req.kernel)
        score += 15
    if req.host_language is not None:
        if spec.host_language != req.host_language.strip().lower():
            return None
        reasons.append("host:%s" % spec.host_language)
        score += 15
    if req.export_format is not None:
        if not spec.exports(req.export_format):
            return None
        reasons.append("exports:%s" % req.export_format.lower())
        score += 20
    if req.import_format is not None:
        if not spec.imports(req.import_format):
            return None
        reasons.append("imports:%s" % req.import_format.lower())
        score += 10
    if req.browser:
        if not spec.online_editor:
            return None
        reasons.append("online-editor")
        score += 5

    support = support_level(spec.name)
    if req.must_be_executable and support != SUPPORT_EXECUTE:
        return None

    # Soft, editorial signals from the curated list.
    if req.internal_fillets:
        if spec.has_paradigm(eco.PARA_BREP):
            score += 25
            reasons.append("brep: internal fillets are the CSG paradigm's weak point")
        elif spec.has_paradigm(eco.PARA_SDF):
            score += 10
            reasons.append("sdf: blends/fillets are natural, but meshed output")
        else:
            score -= 15
            reasons.append("csg/mesh-only: internal fillets are difficult")
    if req.exact_exchange:
        if spec.exports("step"):
            score += 25
            reasons.append("exports STEP for exact downstream exchange")
        else:
            score -= 20
            reasons.append("no STEP export recorded")
    if req.organic_blends:
        if spec.representation == "implicit":
            score += 20
            reasons.append("implicit kernel: smooth blends are native")
        else:
            score -= 5
            reasons.append("non-implicit representation")

    if spec.name in eco.RECOMMENDED_BREP_SYSTEMS:
        score += 10
        reasons.append("curated list's explicit B-rep recommendation")
    if spec.maturity == eco.MAT_MATURE:
        score += 6
        reasons.append("mature")
    elif spec.maturity == eco.MAT_EARLY:
        score -= 6
        reasons.append("early-stage")
    elif spec.maturity == eco.MAT_UNMAINTAINED:
        score -= 20
        reasons.append("unmaintained")
    if spec.kernel_free:
        score -= 5
        reasons.append("no kernel of its own; needs the target toolchain")

    if support == SUPPORT_EXECUTE:
        score += 12
        reasons.append("harness can execute it")
    elif support == SUPPORT_EMIT:
        score += 4
        reasons.append("harness can emit source only")

    return Candidate(name=spec.name, score=score, reasons=tuple(reasons), support=support)


def rank(req: Requirement, limit: Optional[int] = None) -> List[Candidate]:
    """All systems satisfying the hard constraints, best first.

    Ties are broken by name so the ordering is fully deterministic.
    """
    scored: List[Candidate] = []
    for spec in eco.all_systems():
        cand = _score(spec, req)
        if cand is not None:
            scored.append(cand)
    scored.sort(key=lambda c: (-c.score, c.name))
    if limit is not None:
        return scored[:limit]
    return scored


def best(req: Requirement) -> Optional[Candidate]:
    """The single best system for a requirement, or None if nothing qualifies."""
    ranked = rank(req, limit=1)
    return ranked[0] if ranked else None


def explain(name: str, req: Requirement) -> Candidate:
    """Score one named system against a requirement (score 0 / reason if excluded)."""
    cand = _score(eco.get(name), req)
    if cand is None:
        return Candidate(
            name=eco.get(name).name,
            score=0,
            reasons=("excluded: fails a hard constraint",),
            support=support_level(name),
        )
    return cand


# ---------------------------------------------------------------------------
# Coverage / gap report vs the harness's own backends
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageReport:
    """What the harness covers of the code-CAD ecosystem, and what it misses."""

    total_systems: int
    executable: Tuple[str, ...]
    emit_only: Tuple[str, ...]
    unsupported: Tuple[str, ...]
    covered_paradigms: Tuple[str, ...]
    missing_paradigms: Tuple[str, ...]
    covered_kernels: Tuple[str, ...]
    missing_kernels: Tuple[str, ...]
    covered_formats_out: Tuple[str, ...]
    missing_formats_out: Tuple[str, ...]
    covered_host_languages: Tuple[str, ...]

    def as_row(self) -> Dict[str, object]:
        return {
            "total_systems": self.total_systems,
            "executable": list(self.executable),
            "emit_only": list(self.emit_only),
            "unsupported_count": len(self.unsupported),
            "covered_paradigms": list(self.covered_paradigms),
            "missing_paradigms": list(self.missing_paradigms),
            "covered_kernels": list(self.covered_kernels),
            "missing_kernels": list(self.missing_kernels),
            "covered_formats_out": list(self.covered_formats_out),
            "missing_formats_out": list(self.missing_formats_out),
        }


def _union(names: Sequence[str], attr: str) -> List[str]:
    out = set()
    for name in names:
        value = getattr(eco.get(name), attr)
        out.update(value)
    return sorted(out)


def coverage_report(supported: Optional[Sequence[str]] = None) -> CoverageReport:
    """Coverage of the catalogue by the harness's supported systems.

    ``supported`` defaults to the executable set (the CadHub language registry);
    pass an explicit list to model a hypothetical backend addition.
    """
    execu = tuple(executable_systems() if supported is None else sorted(supported))
    emit = tuple(n for n in EMIT_ONLY_SYSTEMS if n not in execu)
    every = eco.system_names()
    unsupported = tuple(n for n in every if n not in execu and n not in emit)

    reachable = list(execu) + list(emit)
    covered_paradigms = _union(reachable, "paradigms")
    covered_kernels = sorted({eco.get(n).kernel for n in reachable})
    covered_formats = _union(reachable, "formats_out")
    covered_hosts = sorted({eco.get(n).host_language for n in reachable})

    all_kernels = sorted({s.kernel for s in eco.all_systems()} - {eco.UNKNOWN})
    all_formats = eco.known_formats_out()

    return CoverageReport(
        total_systems=len(every),
        executable=execu,
        emit_only=emit,
        unsupported=unsupported,
        covered_paradigms=tuple(covered_paradigms),
        missing_paradigms=tuple(p for p in eco.PARADIGMS if p not in covered_paradigms),
        covered_kernels=tuple(k for k in covered_kernels if k != eco.UNKNOWN),
        missing_kernels=tuple(k for k in all_kernels if k not in covered_kernels),
        covered_formats_out=tuple(covered_formats),
        missing_formats_out=tuple(f for f in all_formats if f not in covered_formats),
        covered_host_languages=tuple(covered_hosts),
    )


def gap_recommendations(supported: Optional[Sequence[str]] = None) -> List[Tuple[str, str]]:
    """(system, gap-it-closes) pairs: the cheapest additions that close a gap.

    Deterministic: for each missing paradigm and each missing kernel, name the
    unsupported system that provides it, preferring mature systems then name
    order; each system is recommended once, for the first gap it closes.
    """
    report = coverage_report(supported)
    taken: Dict[str, str] = {}
    order = {eco.MAT_MATURE: 0, eco.MAT_ACTIVE: 1, eco.MAT_EARLY: 2, eco.MAT_UNMAINTAINED: 3}

    def _pick(candidates: Sequence[str]) -> Optional[str]:
        pool = [c for c in candidates if c in report.unsupported and c not in taken]
        if not pool:
            return None
        pool.sort(key=lambda n: (order.get(eco.get(n).maturity, 9), n))
        return pool[0]

    for paradigm in report.missing_paradigms:
        pick = _pick(eco.by_paradigm(paradigm))
        if pick:
            taken[pick] = "paradigm:" + paradigm
    for kernel in report.missing_kernels:
        pick = _pick(eco.by_kernel(kernel))
        if pick:
            taken[pick] = "kernel:" + kernel
    for fmt in report.missing_formats_out:
        pick = _pick(eco.exporters_of(fmt))
        if pick:
            taken[pick] = "format:" + fmt

    return sorted(taken.items())
