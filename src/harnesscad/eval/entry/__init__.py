"""Runnable faces for the metric / preference modules the audit found orphaned.

WHY THIS PACKAGE EXISTS
-----------------------
``audit/book_hitchhiker_rl_eval.md`` and ``audit/book_agentic_design_patterns.md``
found six modules that were "built, unit-tested, and imported by nothing but their
own test": ``judge_calibration``, ``judge_human_agreement``, ``metric_correlation``,
``pass_at_k`` (Chapter 14 evaluation metrics), ``ablation_matrix`` (paired A/B
summaries) and ``dpo_pairs`` (oracle-labelled preference construction). The audit's
one-line diagnosis of the whole repository applies to each: *"HarnessCAD's problem
is not that it lacks capability. It is that the capability is not wired to the
loop."*

A pure function with no caller is a fossil. This package gives each of the six a
``python -m harnesscad.eval.entry.<name>`` entry point: read a JSON document
(``--input FILE`` or stdin), call the underlying module, print a JSON result. Every
wrapper carries a ``--selfcheck`` that runs the computation on a tiny in-code
fixture, so the wiring can be exercised end to end WITHOUT any model call, any data
file or any geometry engine -- which is exactly what a CI reachability check needs.

The wrappers add no analysis of their own. They import the orphan, marshal input,
and marshal output; the numbers come from the module the audit named, unchanged.
"""

from __future__ import annotations

__all__ = [
    "judge_calibration",
    "judge_human_agreement",
    "metric_correlation",
    "pass_at_k",
    "ablation_matrix",
    "dpo_pairs",
]
