"""The corpus runner. THE PROPERTY GATE COMES FIRST, AND IT IS A GATE.

ORDER OF OPERATIONS, AND WHY IT IS THIS ORDER
---------------------------------------------
1.  **The metamorphic properties**
    (:mod:`harnesscad.eval.selftest.properties`), run over THIS CORPUS'S OWN op
    streams. The audit asked for properties.py to be made the PRIMARY gate, and
    this is what that means operationally: **if a law is broken, no score is
    reported.** Not "reported with a warning" -- withheld. A score measured on an
    engine that grows a part when you hollow it, or that returns a different
    volume for the same ops twice, is not a weak measurement, it is not a
    measurement, and printing a number next to it invites somebody to quote the
    number.

    The laws are the most contamination-resistant oracle in the repository because
    ``scale_is_cubic`` relates TWO RUNS OF THE SAME ENGINE: multiply every length
    in the plan by k, and the volume must go up by k^3. That holds for an engine
    whose absolute numbers are all wrong, and it needs no ground truth, no brief,
    no reference and no opinion. Nothing anybody wrote here can flatter it.

2.  **The reference self-test.** Every brief graded against its OWN reference op
    stream. A corpus whose reference solution fails its own grader is measuring
    the engine's bugs and billing them to the model. The pressure corpus failed
    exactly this on two shell briefs -- and shipped, because nobody ran it.

3.  **Corroboration** (:mod:`harnesscad.eval.corpus.consensus`), when more than
    one geometry engine is installed: every brief's closed form checked against
    every available kernel. On a bare machine this degenerates to one engine and
    the report SAYS SO rather than calling one voice a consensus.

COST. Every step here drives a real engine; the F-rep sampler marches a grid
(~1-3 s a part) and the B-rep kernels fork processes. This is a report a human
asks for, not a unit test. The test suite runs a handful of briefs on ``frep``
alone, and the full multi-engine sweep is opt-in behind ``HARNESSCAD_CORPUS_FULL=1``
-- skipped LOUDLY, with a reason, never silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from harnesscad.eval.corpus import consensus as consensus_mod
from harnesscad.eval.corpus import dev as dev_split
from harnesscad.eval.corpus import measurement as measurement_mod
from harnesscad.eval.corpus.grade import Score, grade_reference
from harnesscad.eval.corpus.spec import Brief, Source
from harnesscad.eval.selftest import properties

__all__ = ["CorpusReport", "run", "property_gate", "format_text"]


@dataclass
class CorpusReport:
    backend: str = "frep"
    briefs: int = 0
    #: THE GATE. False => the scores below are withheld and must not be quoted.
    properties_ok: bool = True
    property_violations: List[dict] = field(default_factory=list)
    property_checks: int = 0
    scores: List[Score] = field(default_factory=list)
    #: The FULL MEASUREMENT VECTOR per brief (envelope residuals + IoU + probes),
    #: from ``eval.corpus.measurement``. Reported alongside the envelope scores; it
    #: is the many-to-one closer the audit (gap #5) asked the grader to call.
    vectors: List[dict] = field(default_factory=list)
    corroboration: List[dict] = field(default_factory=list)
    engines: List[str] = field(default_factory=list)
    by_source: Dict[str, int] = field(default_factory=dict)

    @property
    def gated(self) -> bool:
        """Is the score suppressed because a law was broken?"""
        return not self.properties_ok

    @property
    def scored(self) -> List[Score]:
        """The briefs this engine can actually measure. See ``grade.resolvable``."""
        return [s for s in self.scores if s.scored]

    @property
    def unmeasurable(self) -> List[Score]:
        return [s for s in self.scores if s.unmeasurable]

    @property
    def reference_pass(self) -> int:
        return sum(1 for s in self.scored if s.solved)

    @property
    def reference_shape_pass(self) -> int:
        return sum(1 for s in self.scored if s.solved_shape)

    @property
    def ok(self) -> bool:
        return (self.properties_ok
                and self.reference_pass == len(self.scored)
                and not any(c.get("disagree") for c in self.corroboration))

    def to_dict(self) -> dict:
        return {"oracle": "corpus", "ok": self.ok, "backend": self.backend,
                "briefs": self.briefs, "by_source": self.by_source,
                "properties_ok": self.properties_ok,
                "property_checks": self.property_checks,
                "property_violations": self.property_violations,
                "score_withheld": self.gated,
                "measurable": len(self.scored),
                "unmeasurable": [{"brief": s.brief, "why": s.unmeasurable_why}
                                 for s in self.unmeasurable],
                "reference_pass": None if self.gated else self.reference_pass,
                "reference_shape_pass": (None if self.gated
                                         else self.reference_shape_pass),
                "scores": [] if self.gated else [s.to_dict() for s in self.scores],
                "measurement_vectors": [] if self.gated else self.vectors,
                "engines": self.engines,
                "corroboration": self.corroboration}


def property_gate(briefs: Sequence[Brief], backend: str = "frep",
                  seed: int = 20260714) -> properties.PropertyReport:
    """Run the metamorphic laws over the corpus's OWN streams. THE PRIMARY GATE."""
    corpus = [(b.id, list(b.reference)) for b in briefs]
    return properties.run(backends=(backend,), seed=seed, corpus=corpus)


def run(briefs: Optional[Sequence[Brief]] = None,
        backend: str = "frep",
        corroborate: bool = False,
        with_shape: bool = True) -> CorpusReport:
    """The dev corpus, gated by the laws.

    ``corroborate`` drives every installed geometry engine and forks processes for
    the external ones. Off by default; the CLI and the opt-in sweep turn it on.
    """
    the_briefs = list(briefs) if briefs is not None else list(dev_split.BRIEFS)
    r = CorpusReport(backend=backend, briefs=len(the_briefs))
    for b in the_briefs:
        r.by_source[b.source] = r.by_source.get(b.source, 0) + 1

    # 1. THE GATE.
    prop = property_gate(the_briefs, backend=backend)
    r.properties_ok = prop.ok
    r.property_checks = prop.checked
    r.property_violations = [v.to_dict() for v in prop.violations]
    if not r.properties_ok:
        # No score. A number measured on an engine that breaks a law of geometry
        # is a number somebody will quote.
        return r

    # 2. the reference self-test.
    from harnesscad.eval.corpus.grade import grade
    for b in the_briefs:
        s = grade(b, list(b.reference), backend=backend, with_shape=with_shape)
        r.scores.append(s)
        if with_shape:
            # The FULL measurement vector, composed from the score already computed
            # (no rebuild). This is the corpus grader calling the many-to-one closer.
            r.vectors.append(measurement_mod.from_score(b, s).to_dict())

    # 3. corroboration.
    if corroborate:
        results = consensus_mod.corroborate_all(the_briefs)
        r.engines = results[0].engines if results else []
        r.corroboration = [c.to_dict() for c in results]
    return r


def format_text(report: CorpusReport) -> str:
    lines: List[str] = []
    lines.append("CORPUS -- briefs the harness did not write")
    lines.append("=" * 76)
    lines.append("%d briefs on %s. Ground truth by source: %s"
                 % (report.briefs, report.backend,
                    ", ".join("%s=%d" % (k, v)
                              for k, v in sorted(report.by_source.items()))))
    lines.append("")
    lines.append("PROPERTY GATE (the primary gate: laws, not opinions)")
    lines.append("  %d checks over the corpus's own streams -> %s"
                 % (report.property_checks,
                    "PASS" if report.properties_ok
                    else "FAIL (%d violations)" % len(report.property_violations)))
    if not report.properties_ok:
        for v in report.property_violations[:6]:
            lines.append("    %s on %s [%s]: %s"
                         % (v["property"], v["backend"], v["stream"], v["detail"]))
        lines.append("")
        lines.append("SCORE WITHHELD. An engine that breaks a law of geometry "
                     "cannot be scored on a corpus; the number would be quoted.")
        return "\n".join(lines)
    lines.append("")
    lines.append("REFERENCE SELF-TEST (every brief against its own solution)")
    lines.append("  envelope: %d / %d    envelope+shape: %d / %d   (of the %d "
                 "briefs this engine can measure)"
                 % (report.reference_pass, len(report.scored),
                    report.reference_shape_pass, len(report.scored),
                    len(report.scored)))
    if report.unmeasurable:
        lines.append("")
        lines.append("  NOT MEASURABLE ON %s -- these are GOOD PARTS the engine "
                     "cannot resolve." % report.backend.upper())
        lines.append("  They are excluded from the score, not failed. Score them "
                     "on a B-rep kernel.")
        for s in report.unmeasurable:
            lines.append("    %s" % s.brief)
            lines.append("      %s" % s.unmeasurable_why)
    bad = [s for s in report.scored if not s.solved]
    for s in bad:
        lines.append("  BROKEN BRIEF %s" % s.brief)
        for reason in s.reasons:
            lines.append("      %s" % reason)
    if not bad:
        lines.append("  every brief's own reference solution passes its own grader.")
    lines.append("")
    lines.append("SHAPE (IoU vs the reference solid; reported ALONGSIDE, not "
                 "instead of, the envelope)")
    lines.append("  %-34s %12s %8s" % ("brief", "envelope", "IoU"))
    for s in report.scores:
        verdict = ("unmeasurable" if s.unmeasurable
                   else ("pass" if s.solved else "FAIL"))
        lines.append("  %-34s %12s %8s"
                     % (s.brief, verdict,
                        "-" if s.iou is None else "%.3f" % s.iou))
    if report.corroboration:
        lines.append("")
        lines.append("CORROBORATION -- the closed form against every installed engine")
        lines.append("  engines: %s" % (", ".join(report.engines) or "none"))
        if len(report.engines) < 2:
            lines.append("  FEWER THAN TWO ENGINES: 'consensus' is one voice here. "
                         "This corroborates the arithmetic against a single engine "
                         "and cannot see a bug that engine has.")
        for c in report.corroboration:
            if not c.get("disagree"):
                continue
            lines.append("  %-30s disagrees: %s"
                         % (c["brief"], ", ".join(c["disagree"])))
            for d in c.get("detail", []):
                lines.append("      %s" % d)
    return "\n".join(lines)
