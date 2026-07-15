"""The whole corpus in one page: the census, the headline table, and the caveats.

Run it::

    python -m harnesscad.eval.hardcorpus.report

It scores every reference solution against BOTH oracles (proving the corpus is not
broken), prints the discriminative table (the headline), the constraint briefs and
which constraints are genuinely checked versus dropped, the underspecification set,
and -- last and on purpose -- an honest statement of what this corpus still cannot
decontaminate. No model is run: the frontier models are still downloading, and this
page is the thing that must be right before they arrive.
"""

from __future__ import annotations

from typing import List

from harnesscad.eval.hardcorpus import ambiguous as amb
from harnesscad.eval.hardcorpus import constraints as con
from harnesscad.eval.hardcorpus import dev
from harnesscad.eval.hardcorpus import discriminative as disc
from harnesscad.eval.hardcorpus import generate as gen
from harnesscad.eval.hardcorpus import oracle

__all__ = ["census", "discriminative_table", "constraint_report",
           "ambiguous_report", "self_test", "decontamination_limits", "main"]


def census() -> str:
    c = dev.counts()
    lines = ["CORPUS CENSUS", "=" * 60,
             "total briefs (dev): %d" % c["total_briefs"], ""]
    lines.append("generated L3 parts: %d families x 2 prompt styles = %d briefs"
                 % (len(gen.FACTORIES), c["generated"]))
    for fam, lvl in c["generated_by_family"].items():
        lines.append("    %-20s %s" % (fam, lvl))
    lines.append("discriminative near-misses: %d" % c["near_misses"])
    lines.append("constraint-satisfaction briefs: %d" % c["constraint_briefs"])
    lines.append("underspecification briefs: %d" % c["ambiguous_briefs"])
    lines.append("")
    lines.append("dropped ops (reference will not build): %s"
                 % ", ".join(gen.DROPPED_OPS) or "none")
    return "\n".join(lines)


def self_test() -> str:
    """Every reference solution must build and pass its own oracle. Per brief."""
    lines = ["SELF-TEST -- every reference passes its own oracle", "=" * 60]
    fails = 0
    for b in dev.GENERATED:
        s = oracle.grade_reference(b)
        if not s.solved:
            fails += 1
            lines.append("  BROKEN %s: %s" % (b.id, s.reasons[:1]))
    for nm in dev.NEAR_MISSES:
        v = disc.grade_case(nm)
        if not v.controls_hold:
            fails += 1
            lines.append("  BROKEN CONTROL %s" % nm.id)
    for cb in dev.CONSTRAINTS:
        r = con.grade(cb, cb.reference)
        if not (r.built and r.satisfied):
            fails += 1
            lines.append("  BROKEN %s: %s" % (cb.id, r.reason or "unsatisfied"))
    lines.append("references checked: %d generated + %d controls + %d constraint; "
                 "broken: %d"
                 % (len(dev.GENERATED), len(dev.NEAR_MISSES),
                    len(dev.CONSTRAINTS), fails))
    lines.append("(a brief whose own reference does not pass is the v1 "
                 "contamination bug; this must read 0)")
    return "\n".join(lines)


def discriminative_table() -> str:
    return disc.table()


def constraint_report() -> str:
    lines = ["CONSTRAINT SATISFACTION -- checked vs dropped", "=" * 60]
    lines.append("SHIPPED (each an exact measurement on the candidate's own solid):")
    for cb in con.BRIEFS:
        r = con.grade(cb, cb.reference)
        lines.append("  %s [%s]: reference satisfies=%s"
                     % (cb.id, cb.material, r.satisfied))
        for res in r.results:
            lines.append("      %-10s %-4s  %.3f %s (limit %.3f) -- %s"
                         % (res.name, "OK" if res.satisfied else "FAIL",
                            res.measured, res.units, res.limit, res.soundness))
    lines.append("")
    lines.append("IoU is inapplicable here BY CONSTRUCTION: two DIFFERENT parts that")
    lines.append("both satisfy con_bracket_m8_200n score IoU 0.42 against each other.")
    lines.append("")
    lines.append("DROPPED (a constraint we cannot check is decoration):")
    for name, why in con.DROPPED_CONSTRAINTS.items():
        lines.append("  %-16s %s" % (name, why))
    return "\n".join(lines)


def ambiguous_report() -> str:
    lines = ["UNDERSPECIFICATION -- the answer is a QUESTION", "=" * 60]
    for b in amb.BRIEFS:
        lines.append("  %-26s missing: %s" % (b.id, ", ".join(b.missing)))
    lines.append("")
    lines.append("scored: did the response ASK about the missing dimension, or")
    lines.append("HALLUCINATE a value it was never given? classifier caveats:")
    for c in amb.CLASSIFIER_CAVEATS:
        lines.append("  - %s" % c)
    return "\n".join(lines)


def decontamination_limits() -> str:
    return "\n".join([
        "WHAT THIS CORPUS STILL CANNOT DECONTAMINATE", "=" * 60,
        "1. ONE KERNEL GRADES AND (implicitly) ANCHORS. The oracle measures on the",
        "   cadquery/OCCT B-rep. The closed forms are independent of it (arithmetic),",
        "   so a systematic OCCT volume bug would be caught by the volume family",
        "   disagreeing with the closed form -- but a bug in OCCT's POINT",
        "   CLASSIFIER would corrupt the probe family with no independent check here.",
        "   A second kernel voting on membership (a differential probe) is the",
        "   mitigation and it is not yet built. Stated, not hidden.",
        "2. THE PROMPTS ARE TEMPLATED. Two phrasings per part is not the diversity of",
        "   human language; a model that learned our two templates would score high",
        "   for the wrong reason. The generator can mint unlimited phrasings, but the",
        "   ones shipped are ours, and a held-out SEED does not decontaminate a",
        "   held-out STYLE.",
        "3. THE FAMILIES ARE OURS. The corpus samples the op families we chose; a",
        "   model weak on a family we did not think to generate is not measured. This",
        "   is the same bound Text2CAD-Bench and MUSE have, moved but not removed.",
        "4. THE AMBIGUOUS SCORER IS A HEURISTIC, not a model of intent -- it reads",
        "   interrogative form plus a keyword, and reports 'hedged' when it cannot",
        "   tell. It cannot yet distinguish a good-faith ask from a rote one.",
        "5. NO MODEL HAS BEEN RUN. Every 'the field passes this' figure is measured on",
        "   the metric, not on a model emitting the near-miss; the claim is that the",
        "   GRADER is fooled, which is exactly what a benchmark's grader must not be.",
    ])


def main() -> int:
    blocks = [census(), self_test(), discriminative_table(), constraint_report(),
              ambiguous_report(),
              "WHAT THIS MEASURES THAT Text2CAD-Bench AND MUSE STRUCTURALLY CANNOT\n"
              + "=" * 60 + "\n"
              "- Text2CAD-Bench grades IoU + Chamfer; both are ENVELOPE/shape metrics\n"
              "  and are many-to-one. The discriminative table is five wrong parts\n"
              "  they score CORRECT. Point membership is not in their instrument.\n"
              "- MUSE's geometric stage is watertight + manifold + volume + genus;\n"
              "  the shell_face near-miss holds ALL FOUR exactly equal to the correct\n"
              "  part. Their design-intent stage is a VLM judge at r = 0.713 -- a\n"
              "  model's opinion of a picture; ours is an exact kernel query.\n"
              "- Neither benchmarks CONSTRAINT SATISFACTION where no reference shape\n"
              "  exists (IoU is undefined), nor UNDERSPECIFICATION where the correct\n"
              "  output is a question, not geometry.",
              decontamination_limits()]
    print("\n\n".join(blocks))
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
