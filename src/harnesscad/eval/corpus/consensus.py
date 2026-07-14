"""The differential oracle as an INDEPENDENT GRADER.

Six engines: a sampled SDF, two OCCT B-rep kernels, a CGAL mesher, a mesh kernel,
a stub. They were built as interchangeable products. Used as a grader they are
something our brief-writer can never be: independent of our brief-writer. A part
that CadQuery, FreeCAD, OpenSCAD, Blender and the F-rep sampler all measure the
same way is right for reasons that have nothing to do with anybody's opinion here.

WHAT THIS IS FOR, EXACTLY
-------------------------
Two jobs, and it is worth being precise about which is which.

1.  **Cross-checking the brief's own arithmetic.** A brief whose closed form says
    22296 mm3 and whose op stream, run on four independent kernels, measures
    22296 mm3 is a brief whose arithmetic and whose op stream agree. If they
    disagree, either the formula is wrong or the reference stream does not build
    the part the formula describes, and EITHER WAY the brief is broken -- and it
    is broken in a way no amount of staring at our own code would reveal, because
    our own code is what wrote both halves. :func:`corroborate` does this.

2.  **Adjudicating a candidate answer when the analytic truth runs out.** For a
    part with no closed form, the engines' consensus is the best grader available.

AND THE CAVEAT, STATED PLAINLY
------------------------------
CONSENSUS IS NOT TRUTH. Five engines can share a bug -- and in this repository
they nearly did: the F-rep shell dilated every part it touched, and the reason we
know is that the OTHER engines did not. Where a closed form exists it ADJUDICATES
and the consensus is merely corroboration. Where it does not, consensus is the
strongest evidence available and is still evidence, not proof.

The heavy engines fork processes. This module therefore defaults to whatever is
installed and SAYS SO in the report; on a bare machine that is the F-rep sampler
alone, "consensus" degenerates to one voice, and :meth:`Corroboration.independent`
goes False so nobody can read a single-engine run as agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Op
from harnesscad.eval.corpus.spec import Brief
from harnesscad.eval.selftest.probe import (GEOMETRIC_BACKENDS, Observation,
                                            available, observe, tolerance)

__all__ = ["Corroboration", "corroborate", "corroborate_all"]


@dataclass
class Corroboration:
    """Do the engines agree with the brief's ARITHMETIC, and with each other?"""

    brief: str
    engines: List[str] = field(default_factory=list)
    volumes: Dict[str, Optional[float]] = field(default_factory=dict)
    bboxes: Dict[str, Optional[List[float]]] = field(default_factory=dict)
    #: engines whose measurement matches the closed form within THEIR OWN tolerance
    agree: List[str] = field(default_factory=list)
    #: engines that measured something else. Each one is a finding.
    disagree: List[str] = field(default_factory=list)
    #: engines that declined the plan (a capability gap, not a bug)
    refused: Dict[str, str] = field(default_factory=dict)
    crashed: Dict[str, str] = field(default_factory=dict)
    detail: List[str] = field(default_factory=list)

    @property
    def independent(self) -> bool:
        """At least two engines measured it. Below that, 'consensus' is one voice."""
        return len(self.agree) + len(self.disagree) >= 2

    @property
    def ok(self) -> bool:
        return not self.disagree and bool(self.agree)

    def to_dict(self) -> dict:
        return {"brief": self.brief, "engines": self.engines,
                "independent": self.independent, "ok": self.ok,
                "agree": self.agree, "disagree": self.disagree,
                "refused": self.refused, "crashed": self.crashed,
                "volumes": self.volumes, "bboxes": self.bboxes,
                "detail": self.detail}


def corroborate(brief: Brief,
                backends: Optional[Sequence[str]] = None) -> Corroboration:
    """Run the brief's REFERENCE stream on every available engine.

    Compare each engine's measurement against the brief's CLOSED FORM -- not
    against each other, and not against a blessed engine. The closed form is the
    only party in the room that did not come out of this repository.
    """
    wanted = tuple(backends) if backends is not None else GEOMETRIC_BACKENDS
    live = available(wanted)
    c = Corroboration(brief=brief.id, engines=list(live))
    for name in live:
        obs: Observation = observe(name, list(brief.reference), verify_level="core")
        if obs.error:
            c.crashed[name] = obs.error
            continue
        if not obs.ok:
            c.refused[name] = "rejected %s (%s)" % (obs.rejected or "?",
                                                    ",".join(obs.codes) or "no code")
            continue
        if not obs.geometric:
            c.refused[name] = "produced no measurable solid"
            continue
        c.volumes[name] = obs.volume
        c.bboxes[name] = [float(v) for v in (obs.bbox or ())]

        tol = tolerance(name)
        vtol = tol.volume_tol(brief.extent, brief.feature)
        btol = tol.bbox_tol(brief.extent)
        rel = abs((obs.volume or 0.0) - brief.volume) / max(brief.volume, 1e-9)
        bad: List[str] = []
        if rel > vtol:
            bad.append("volume %.2f vs closed form %.2f (%+.2f%%, tol %.2f%%)"
                       % (obs.volume or 0.0, brief.volume,
                          100.0 * ((obs.volume or 0.0) - brief.volume) / brief.volume,
                          100.0 * vtol))
        for axis, want, got in zip("xyz", brief.bbox, obs.bbox or (0, 0, 0)):
            if abs(got - want) > btol:
                bad.append("bbox %s %.3f vs %g (tol %.3f)" % (axis, got, want, btol))
        if brief.genus is not None and obs.genus is not None \
                and obs.genus != brief.genus:
            bad.append("genus %d vs %d" % (obs.genus, brief.genus))
        if bad:
            c.disagree.append(name)
            c.detail.append("%s: %s" % (name, "; ".join(bad)))
        else:
            c.agree.append(name)
    return c


def corroborate_all(briefs: Sequence[Brief],
                    backends: Optional[Sequence[str]] = None
                    ) -> List[Corroboration]:
    return [corroborate(b, backends) for b in briefs]


def format_text(results: Sequence[Corroboration]) -> str:
    lines: List[str] = []
    lines.append("CORROBORATION -- every brief's reference stream, on every engine")
    lines.append("=" * 76)
    engines = sorted({e for r in results for e in r.engines})
    lines.append("engines available here: %s" % (", ".join(engines) or "none"))
    if len(engines) < 2:
        lines.append("FEWER THAN TWO ENGINES. 'Consensus' is one voice; this run "
                     "corroborates the arithmetic against a single engine and "
                     "nothing more. It cannot detect a bug that engine has.")
    lines.append("")
    bad = [r for r in results if r.disagree]
    lines.append("%d briefs; %d have an engine that disagrees with the closed form."
                 % (len(results), len(bad)))
    for r in bad:
        lines.append("  %-28s disagree: %s" % (r.brief, ", ".join(r.disagree)))
        for d in r.detail:
            lines.append("      %s" % d)
    if not bad:
        lines.append("Every available engine reproduced every brief's closed form.")
    return "\n".join(lines)
