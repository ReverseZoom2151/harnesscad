"""Retrieval-quality metrics for the RAG grounding layer (blueprint sec.16.8).

A retriever that no one measures drifts silently: a fusion tweak or an embedding
swap that *feels* better can quietly regress the ranking that grounds every
generation. This module is the METRIC, not an experiment -- it computes
**Recall@K** and **MRR** over a set of ``(query, relevant-doc-ids)`` cases and
returns them; it does not run over any real corpus or call a model.

  - **Recall@K** -- did any / how many of the relevant docs land in the top K?
    Reported per K, so you can watch the head of the ranking, not just its tail.
  - **MRR** (Mean Reciprocal Rank) -- ``1/rank`` of the first relevant hit,
    averaged over cases. Rewards putting *a* right answer high, which is what a
    grounding retriever must do before a context window fills.

The retriever is any object with ``retrieve(query, k) -> [hit, ...]`` (the
``HybridRetriever`` contract); each hit is mapped to a doc-id by ``id_of``
(default: the chunk's ``source``), so relevant-doc-ids are matched at document
granularity. ``--selfcheck`` builds a tiny SYNTHETIC corpus and asserts the
metrics behave (perfect ranking -> recall 1.0, MRR 1.0; an empty retriever ->
0.0). It is deterministic, kernel-free, and runs nothing over real data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Per-ranking metrics (pure functions over ordered id lists)
# ---------------------------------------------------------------------------
def recall_at_k(retrieved_ids: Sequence[str],
                relevant_ids: Sequence[str],
                k: int) -> float:
    """Fraction of relevant docs that appear in the top-``k`` retrieved ids.

    Returns ``0.0`` when there are no relevant ids (an undefined recall is not a
    perfect one). Duplicate retrieved ids are collapsed so a repeated hit cannot
    inflate the count.
    """
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    if k <= 0:
        return 0.0
    topk = set(retrieved_ids[:k])
    return len(topk & rel) / len(rel)


def reciprocal_rank(retrieved_ids: Sequence[str],
                    relevant_ids: Sequence[str]) -> float:
    """``1/rank`` of the FIRST relevant hit (1-indexed), or ``0.0`` if none."""
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    for i, rid in enumerate(retrieved_ids):
        if rid in rel:
            return 1.0 / (i + 1)
    return 0.0


def mrr(rankings: Sequence[Tuple[Sequence[str], Sequence[str]]]) -> float:
    """Mean Reciprocal Rank over ``(retrieved_ids, relevant_ids)`` pairs.

    Empty input is ``0.0`` (there is nothing to average).
    """
    pairs = list(rankings)
    if not pairs:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in pairs) / len(pairs)


# ---------------------------------------------------------------------------
# Eval cases + report
# ---------------------------------------------------------------------------
@dataclass
class EvalCase:
    """One labelled query: the text and the ids of the docs that should surface."""

    query: str
    relevant_ids: List[str] = field(default_factory=list)


@dataclass
class RetrievalReport:
    """Aggregate retrieval quality over a set of cases."""

    n: int
    recall_at_k: Dict[int, float]
    mrr: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "recall_at_k": {str(k): round(v, 6) for k, v in
                            sorted(self.recall_at_k.items())},
            "mrr": round(self.mrr, 6),
        }

    def text(self) -> str:
        lines = [f"cases: {self.n}"]
        for k in sorted(self.recall_at_k):
            lines.append(f"recall@{k}: {self.recall_at_k[k]:.3f}")
        lines.append(f"mrr: {self.mrr:.3f}")
        return "\n".join(lines)


def _default_id_of(hit: Any) -> str:
    """Map a retrieval hit to a doc-id: prefer ``source``, fall back to ``id``.

    Works for ``Retrieved`` (``.source``) and bare ``Chunk`` objects alike; a raw
    string id passes straight through.
    """
    if isinstance(hit, str):
        return hit
    src = getattr(hit, "source", None)
    if src is not None:
        return str(src)
    hid = getattr(hit, "id", None)
    if hid is not None:
        return str(hid)
    return str(hit)


def evaluate(retriever: Any,
             cases: Sequence[EvalCase],
             k_values: Sequence[int] = (1, 3, 5),
             id_of: Optional[Callable[[Any], str]] = None) -> RetrievalReport:
    """Run ``retriever`` over ``cases`` and return a :class:`RetrievalReport`.

    ``retriever`` needs only a ``retrieve(query, k)`` method. Each query is run
    once at ``k = max(k_values)`` and the single ranking is reused for every K,
    so ranking cost does not multiply with the number of cut-offs. ``id_of``
    maps a hit to the doc-id space of ``relevant_ids`` (default: chunk source).
    """
    id_of = id_of or _default_id_of
    ks = sorted({int(k) for k in k_values if int(k) > 0})
    if not ks:
        ks = [1]
    top = max(ks)

    rr_pairs: List[Tuple[List[str], List[str]]] = []
    recall_sums: Dict[int, float] = {k: 0.0 for k in ks}
    cases = list(cases)
    for case in cases:
        hits = retriever.retrieve(case.query, top)
        ids = [id_of(h) for h in hits]
        rr_pairs.append((ids, list(case.relevant_ids)))
        for k in ks:
            recall_sums[k] += recall_at_k(ids, case.relevant_ids, k)

    n = len(cases)
    recall = {k: (recall_sums[k] / n if n else 0.0) for k in ks}
    return RetrievalReport(n=n, recall_at_k=recall, mrr=mrr(rr_pairs))


# ---------------------------------------------------------------------------
# Self-check: synthetic corpus, no real data, no kernel
# ---------------------------------------------------------------------------
# ``(text, source)`` pairs -- the order ``build_from_docs`` expects; the source
# label is the doc-id the cases below reference.
_SYNTHETIC_DOCS: Tuple[Tuple[str, str], ...] = (
    ("# Fasteners\nM6 socket head cap screw torque and preload for steel joints. "
     "Tightening torque tables for coarse thread bolts.", "bolt"),
    ("# Bearings\nDeep groove ball bearing radial load rating and fit tolerance "
     "for a rotating shaft. Lubrication interval for sealed bearings.", "bearing"),
    ("# Seals\nFlat gasket compression set and flange sealing pressure for a "
     "bolted pipe joint. O-ring groove dimensions for face seals.", "gasket"),
    ("# Joining\nFillet weld throat size and leg length for a structural steel "
     "bracket. Weld symbol interpretation on engineering drawings.", "weld"),
)

_SYNTHETIC_CASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("torque for an M6 socket head cap screw", ("bolt",)),
    ("radial load rating of a ball bearing", ("bearing",)),
    ("flange sealing pressure for a gasket joint", ("gasket",)),
    ("fillet weld leg length on a steel bracket", ("weld",)),
)


def _build_synthetic_retriever(**kw):
    """Build a HybridRetriever over the synthetic corpus (imported lazily)."""
    from harnesscad.agents.rag.retriever import build_from_docs

    return build_from_docs(list(_SYNTHETIC_DOCS), **kw)


class _EmptyRetriever:
    """A retriever that returns nothing -- the recall/MRR floor case."""

    def retrieve(self, query: str, k: int = 5) -> List[Any]:
        return []


def selfcheck(verbose: bool = True) -> bool:
    """Validate the metrics on a synthetic corpus. Returns True on success.

    No real corpus, no model, no kernel: it exercises the pure metric functions
    and the ``evaluate`` loop against a retriever whose right answers are known
    by construction. Kept as a METRIC self-test, not an experiment.
    """
    ok = True
    cases = [EvalCase(q, list(rel)) for q, rel in _SYNTHETIC_CASES]

    # 1) Pure-function invariants.
    checks: List[Tuple[str, bool]] = []
    checks.append(("recall@k hit",
                   recall_at_k(["a", "b", "c"], ["c"], 3) == 1.0))
    checks.append(("recall@k miss",
                   recall_at_k(["a", "b"], ["c"], 2) == 0.0))
    checks.append(("recall@k partial",
                   abs(recall_at_k(["a", "b"], ["a", "c"], 2) - 0.5) < 1e-9))
    checks.append(("recall@k empty relevant",
                   recall_at_k(["a"], [], 1) == 0.0))
    checks.append(("rr rank-1", reciprocal_rank(["c", "a"], ["c"]) == 1.0))
    checks.append(("rr rank-2",
                   abs(reciprocal_rank(["a", "c"], ["c"]) - 0.5) < 1e-9))
    checks.append(("rr none", reciprocal_rank(["a", "b"], ["c"]) == 0.0))
    checks.append(("mrr empty", mrr([]) == 0.0))

    # 2) End-to-end over the synthetic retriever: each query's gold doc is the
    #    lexically-obvious match, so a working retriever ranks it first.
    rep = evaluate(_build_synthetic_retriever(), cases, k_values=(1, 3, 5))
    checks.append(("synthetic n", rep.n == len(cases)))
    checks.append(("synthetic recall@5 == 1.0",
                   abs(rep.recall_at_k[5] - 1.0) < 1e-9))
    checks.append(("synthetic recall@1 == 1.0",
                   abs(rep.recall_at_k[1] - 1.0) < 1e-9))
    checks.append(("synthetic mrr == 1.0", abs(rep.mrr - 1.0) < 1e-9))

    # 3) Floor case: an empty retriever scores zero everywhere.
    floor = evaluate(_EmptyRetriever(), cases, k_values=(1, 5))
    checks.append(("empty recall@5 == 0.0", floor.recall_at_k[5] == 0.0))
    checks.append(("empty mrr == 0.0", floor.mrr == 0.0))

    for label, passed in checks:
        ok = ok and passed
        if verbose:
            print(f"[{'ok' if passed else 'FAIL'}] {label}")
    if verbose:
        print("--- synthetic report ---")
        print(rep.text())
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.agents.rag.retrieval_eval",
        description="Retrieval-quality metrics (Recall@K, MRR) for the RAG layer.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the synthetic metric self-check (no real data)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.selfcheck:
        return 0 if selfcheck(verbose=True) else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
