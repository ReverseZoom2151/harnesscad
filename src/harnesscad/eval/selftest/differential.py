"""Differential oracle — six engines, one op stream, no ground truth needed.

The harness owns six independent geometry engines: a stub, a sampled SDF, two
OCCT B-rep kernels (CadQuery, FreeCAD), a CGAL mesher (OpenSCAD) and a mesh
kernel (Blender). They were built as interchangeable *products*. They are also,
for free, the strongest oracle in the repository, and nothing had ever used them
as one.

Run the same plan on all of them. Where they disagree, at least one is wrong --
and you did not need to know the right answer to find that out. It cost nothing,
it needed no model, no gold, no label, and it catches on the first part what ~200
benchmark modules and 23 verifiers missed for the life of the project:

    a 60 x 40 x 20 box, shelled at 3 mm, must still be 60 x 40 x 20.

What this module will and will not claim
----------------------------------------
It reports the SPREAD and names the clusters. The largest cluster is called the
"consensus" because a human reading a table needs somewhere to start -- but
consensus is a SIGNAL, NOT TRUTH. Five backends can share a bug. The golden
corpus (:mod:`harnesscad.eval.selftest.golden`) is what adjudicates; this module
only ever says "these engines do not agree, and here is by how much".

Expected error is not disagreement
----------------------------------
The F-rep backend samples a field on a grid, so its volume lands ~1-2% off and
its bbox a fraction of a cell short. Charging that as a disagreement would drown
the real ones. Two backends AGREE when they are within the LOOSER of their two
tolerances (see :mod:`harnesscad.eval.selftest.probe`, where every tolerance is
derived from what the engine physically is). A STRUCTURAL disagreement -- a bbox
that is wrong by millimetres, a different genus, a volume that moved the wrong
way -- is never inside anybody's tolerance and is always reported.

A backend that REFUSES the plan is not a disagreement either; it is a capability
gap, reported in its own column. An engine that cannot shell is honest. An engine
that shells the part into a bigger part is not.

``shell`` is now SPECIFIED -- it hollows inward, an empty ``faces`` list means a
sealed void, and the outer surface does not move. So every engine is expected to
AGREE on it, and any spread reported here is a real finding, not the op's meaning
being underdetermined.

And a caveat this module will not hide: the signature it compares (volume, bbox,
genus, watertightness) is MANY-TO-ONE. Engines agreeing does not make them right;
they can share a bug, and a part with its holes in the wrong places matches every
number. Agreement is evidence, not proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op
from harnesscad.eval.selftest import golden
from harnesscad.eval.selftest.probe import (BackendFactory, EXACTNESS_ORDER,
                                            GEOMETRIC_BACKENDS, Observation,
                                            available, bbox_delta, observe,
                                            resolve, tolerance,
                                            volume_rel_delta)

__all__ = ["Disagreement", "CaseReport", "DifferentialReport", "STREAMS",
           "agree", "compare", "run", "format_text"]


#: The streams the oracle is run over: every golden part (they are the parts an
#: engineer recognises) plus the shell/boolean stressors that separate the
#: engines. NOTE: no analytic value is read here -- only the op streams.
def _streams() -> List[Tuple[str, Tuple[Op, ...]]]:
    return [(p.name, p.ops) for p in golden.PARTS]


STREAMS: List[Tuple[str, Tuple[Op, ...]]] = _streams()


@dataclass
class Disagreement:
    """Two clusters of engines returned different geometry for the same plan."""

    metric: str                     # volume | bbox | genus | watertight
    backend: str                    # the engine outside the consensus
    consensus: List[str] = field(default_factory=list)
    consensus_value: object = None
    value: object = None
    delta: str = ""
    structural: bool = False        # beyond ANY plausible sampling error

    def to_dict(self) -> dict:
        return {"metric": self.metric, "backend": self.backend,
                "consensus": self.consensus, "consensus_value": self.consensus_value,
                "value": self.value, "delta": self.delta,
                "structural": self.structural}


@dataclass
class CaseReport:
    name: str
    observations: List[Observation] = field(default_factory=list)
    consensus: List[str] = field(default_factory=list)
    clusters: List[List[str]] = field(default_factory=list)
    disagreements: List[Disagreement] = field(default_factory=list)
    refused: Dict[str, str] = field(default_factory=dict)   # backend -> why
    crashed: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.disagreements

    def volume_spread(self) -> Optional[float]:
        vols = [o.volume for o in self.observations if o.geometric]
        if len(vols) < 2 or max(vols) <= 0:
            return None
        return (max(vols) - min(vols)) / max(vols)

    def to_dict(self) -> dict:
        return {"name": self.name,
                "consensus": self.consensus,
                "clusters": self.clusters,
                "volume_spread": self.volume_spread(),
                "refused": self.refused,
                "crashed": self.crashed,
                "disagreements": [d.to_dict() for d in self.disagreements],
                "observations": [o.to_dict() for o in self.observations]}


@dataclass
class DifferentialReport:
    cases: List[CaseReport] = field(default_factory=list)
    backends: List[str] = field(default_factory=list)
    skipped_backends: Dict[str, str] = field(default_factory=dict)

    @property
    def disagreements(self) -> List[Tuple[str, Disagreement]]:
        return [(c.name, d) for c in self.cases for d in c.disagreements]

    @property
    def crashes(self) -> List[Tuple[str, str, str]]:
        """(part, engine, error). An engine that BLEW UP is a finding, and it must
        not be able to hide inside a "0 disagreements" headline just because a
        corpse has no bbox to disagree with."""
        return [(c.name, b, err) for c in self.cases
                for b, err in sorted(c.crashed.items())]

    @property
    def refusals(self) -> List[Tuple[str, str, str]]:
        """(part, engine, why). A capability gap, reported separately from a bug."""
        return [(c.name, b, why) for c in self.cases
                for b, why in sorted(c.refused.items())]

    @property
    def findings(self) -> int:
        return len(self.disagreements) + len(self.crashes)

    @property
    def ok(self) -> bool:
        return not self.disagreements and not self.crashes

    def by_backend(self) -> Dict[str, int]:
        counts = {b: 0 for b in self.backends}
        for _, d in self.disagreements:
            counts[d.backend] = counts.get(d.backend, 0) + 1
        for _, b, _ in self.crashes:
            counts[b] = counts.get(b, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "oracle": "differential",
            "ok": self.ok,
            "findings": self.findings,
            "crashes": [{"part": p, "backend": b, "error": e}
                        for p, b, e in self.crashes],
            "refusals": [{"part": p, "backend": b, "why": w}
                         for p, b, w in self.refusals],
            "backends": self.backends,
            "skipped_backends": self.skipped_backends,
            "cases": [c.to_dict() for c in self.cases],
            "disagreements_by_backend": self.by_backend(),
            "note": ("consensus is the largest agreeing cluster; it is a SIGNAL, "
                     "NOT TRUTH. Use the golden corpus to adjudicate."),
        }


# --- agreement -------------------------------------------------------------

def agree(a: Observation, b: Observation) -> bool:
    """Do two engines agree, within the LOOSER of their two tolerances?

    Both a volume and a bbox must match. Genus is compared separately: two
    engines can agree on size and still build a different topology, and that is a
    disagreement worth its own line rather than one hidden inside this predicate.
    """
    if not (a.geometric and b.geometric):
        return False
    return (volume_rel_delta(a.volume, b.volume) <= _vol_tol(a, b)
            and bbox_delta(a.bbox, b.bbox) <= _bbox_tol(a, b))


def _vol_tol(a: Observation, b: Observation) -> float:
    """The LOOSER of the two engines' derived volume tolerances for this part."""
    extent = max(a.extent, b.extent)
    thin = min(x for x in (a.min_extent, b.min_extent) if x > 0.0) \
        if max(a.min_extent, b.min_extent) > 0.0 else 0.0
    return max(tolerance(a.backend).volume_tol(extent, thin),
               tolerance(b.backend).volume_tol(extent, thin))


def _bbox_tol(a: Observation, b: Observation) -> float:
    extent = max(a.extent, b.extent)
    return max(tolerance(a.backend).bbox_tol(extent),
               tolerance(b.backend).bbox_tol(extent))


def _cluster(obs: Sequence[Observation]) -> List[List[str]]:
    """Group engines into maximal agreeing clusters (a simple union-find)."""
    geo = [o for o in obs if o.geometric]
    parent = {o.backend: o.backend for o in geo}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(geo):
        for b in geo[i + 1:]:
            if agree(a, b):
                ra, rb = find(a.backend), find(b.backend)
                if ra != rb:
                    parent[rb] = ra
    groups: Dict[str, List[str]] = {}
    for o in geo:
        groups.setdefault(find(o.backend), []).append(o.backend)
    clusters = [sorted(v, key=lambda n: EXACTNESS_ORDER.index(n)
                       if n in EXACTNESS_ORDER else 99)
                for v in groups.values()]
    # Deterministic order: biggest cluster first, then by its most exact member.
    clusters.sort(key=lambda c: (-len(c),
                                 EXACTNESS_ORDER.index(c[0])
                                 if c[0] in EXACTNESS_ORDER else 99))
    return clusters


def _structural(metric: str, a: Observation, b: Observation) -> bool:
    """Is this difference too big to be anybody's sampling error?

    A structural difference is one that survives being given TEN TIMES the looser
    engine's own tolerance -- i.e. no grid, no polygonisation and no rounding can
    explain it. A wrong bbox, a wrong genus and a volume that changed sign are all
    structural by construction.
    """
    if metric in ("genus", "watertight"):
        return True
    if metric == "volume":
        return volume_rel_delta(a.volume, b.volume) > 10 * _vol_tol(a, b)
    if metric == "bbox":
        return bbox_delta(a.bbox, b.bbox) > 10 * _bbox_tol(a, b)
    return False


def compare(name: str, ops: Sequence[Op],
            backends: Optional[Sequence[str]] = None,
            factory: Optional[BackendFactory] = None) -> CaseReport:
    """Run one op stream on every available engine and report the spread."""
    wanted = tuple(backends) if backends is not None else GEOMETRIC_BACKENDS
    case = CaseReport(name)
    obs: List[Observation] = []
    for b in wanted:
        o = observe(b, ops, factory=factory)
        obs.append(o)
        if not o.available:
            continue
        if o.error:
            case.crashed[b] = o.error
        elif not o.ok:
            case.refused[b] = "rejected %s (%s)" % (o.rejected or "?",
                                                    ",".join(o.codes) or "no code")
    case.observations = obs
    case.clusters = _cluster(obs)
    if not case.clusters:
        return case
    case.consensus = case.clusters[0]
    ref = next(o for o in obs if o.backend == case.consensus[0])

    for o in obs:
        if not o.geometric or o.backend in case.consensus:
            continue
        if volume_rel_delta(o.volume, ref.volume) > _vol_tol(o, ref):
            case.disagreements.append(Disagreement(
                "volume", o.backend, list(case.consensus),
                round(ref.volume, 4), round(o.volume, 4),
                "%+.2f%%" % (100.0 * (o.volume - ref.volume) / max(ref.volume, 1e-9)),
                _structural("volume", o, ref)))
        if bbox_delta(o.bbox, ref.bbox) > _bbox_tol(o, ref):
            case.disagreements.append(Disagreement(
                "bbox", o.backend, list(case.consensus),
                [round(v, 3) for v in ref.bbox], [round(v, 3) for v in o.bbox],
                "max axis delta %+.3f" % max(
                    (x - y for x, y in zip(o.bbox, ref.bbox)), key=abs),
                _structural("bbox", o, ref)))

    # Genus / watertightness are compared against the consensus independently of
    # the size clusters: same size, different topology is still a bug.
    ref_genus = _majority_genus(obs, case.consensus)
    for o in obs:
        if not o.geometric or o.genus is None:
            continue
        if ref_genus is not None and o.genus != ref_genus:
            case.disagreements.append(Disagreement(
                "genus", o.backend, list(case.consensus), ref_genus, o.genus,
                "topology differs", True))
        if o.watertight is False:
            case.disagreements.append(Disagreement(
                "watertight", o.backend, list(case.consensus), True, False,
                "not a closed solid", True))
    return case


def _majority_genus(obs: Sequence[Observation],
                    consensus: Sequence[str]) -> Optional[int]:
    """The genus the consensus cluster reports, if any of them report one."""
    votes: Dict[int, int] = {}
    for o in obs:
        if o.genus is None or not o.geometric:
            continue
        weight = 2 if o.backend in consensus else 1
        votes[o.genus] = votes.get(o.genus, 0) + weight
    if not votes:
        return None
    best = max(votes.values())
    winners = sorted(g for g, v in votes.items() if v == best)
    return winners[0] if len(winners) == 1 else None


def run(backends: Optional[Sequence[str]] = None,
        streams: Optional[Sequence[Tuple[str, Sequence[Op]]]] = None,
        factory: Optional[BackendFactory] = None) -> DifferentialReport:
    wanted = tuple(backends) if backends is not None else GEOMETRIC_BACKENDS
    live = available(wanted, factory)
    report = DifferentialReport(backends=list(live))
    for name in wanted:
        if name not in live:
            report.skipped_backends[name] = resolve(name, factory)[1]
    if len(live) < 2:
        return report  # nothing to differentiate against; the suite stays green
    for name, ops in (streams if streams is not None else STREAMS):
        report.cases.append(compare(name, ops, backends=live, factory=factory))
    return report


def format_text(report: DifferentialReport) -> str:
    lines: List[str] = []
    lines.append("DIFFERENTIAL -- the same plan on every engine")
    lines.append("=" * 72)
    if len(report.backends) < 2:
        lines.append("fewer than two geometric backends are available here; "
                     "differential testing needs at least two.")
        for name, why in sorted(report.skipped_backends.items()):
            lines.append("  skipped %-9s %s" % (name, why))
        return "\n".join(lines)
    lines.append("engines: " + ", ".join(report.backends))
    for name, why in sorted(report.skipped_backends.items()):
        lines.append("  skipped %-9s %s" % (name, why))
    lines.append("")
    counts = report.by_backend()
    crashed = {}
    for _, b, _ in report.crashes:
        crashed[b] = crashed.get(b, 0) + 1
    refused = {}
    for _, b, _ in report.refusals:
        refused[b] = refused.get(b, 0) + 1
    lines.append("%-10s %8s %8s %9s" % ("engine", "findings", "crashes", "refusals"))
    lines.append("-" * 39)
    for b in report.backends:
        lines.append("%-10s %8d %8d %9d"
                     % (b, counts.get(b, 0), crashed.get(b, 0), refused.get(b, 0)))
    lines.append("(a refusal is a CAPABILITY GAP, not a bug: the engine declined "
                 "rather than\n building the part wrong. A crash IS a finding.)")
    lines.append("")
    for case in report.cases:
        if case.ok and not case.refused and not case.crashed:
            continue
        lines.append("%s" % case.name)
        lines.append("  %-9s %14s %-26s %-7s %s"
                     % ("engine", "volume", "bbox", "genus", "status"))
        for o in case.observations:
            if not o.available:
                continue
            if o.error:
                lines.append("  %-9s %14s %-26s %-7s CRASH %s"
                             % (o.backend, "-", "-", "-", o.error[:40]))
                continue
            bbox = ("[%s]" % ", ".join("%.2f" % v for v in o.bbox)) if o.bbox else "-"
            vol = "%14.2f" % o.volume if o.volume is not None else "%14s" % "-"
            status = "ok" if o.ok else ("REFUSED " + ",".join(o.codes))
            if o.backend not in case.consensus and o.geometric:
                status = "OUTLIER " + status
            lines.append("  %-9s %s %-26s %-7s %s"
                         % (o.backend, vol, bbox,
                            "-" if o.genus is None else str(o.genus), status))
        for d in case.disagreements:
            lines.append("    %-10s %-9s consensus %s -> got %s  (%s)%s"
                         % (d.metric.upper(), d.backend, d.consensus_value,
                            d.value, d.delta,
                            "  STRUCTURAL" if d.structural else ""))
        lines.append("")
    if report.ok:
        lines.append("no disagreements and no crashes across %d parts x %d engines."
                     % (len(report.cases), len(report.backends)))
        lines.append("NOTE: agreement is not proof. The signature compared (volume, "
                     "bbox, genus,\nwatertightness) is MANY-TO-ONE -- engines can "
                     "share a bug, and a part with its\nholes in the wrong places "
                     "matches every number here.")
    else:
        n_struct = sum(1 for _, d in report.disagreements if d.structural)
        lines.append("%d disagreements (%d STRUCTURAL) and %d crashes across %d "
                     "parts."
                     % (len(report.disagreements), n_struct, len(report.crashes),
                        len(report.cases)))
        for part, engine, err in report.crashes:
            lines.append("    CRASH      %-9s on %s: %s" % (engine, part, err[:60]))
        lines.append("consensus is the largest agreeing cluster -- a signal, not "
                     "truth. Run --golden to adjudicate.")
    return "\n".join(lines)
