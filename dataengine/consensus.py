"""Consensus / QC labeling (Scale-AI data-engine playbook).

The solver-in-the-loop (datagen/pipeline.py) settles *objective* labels: does the
part build? But the corpus also needs *subjective* semantic labels the verifier
can't decide — "is this a bracket or a gusset?", "is this design manufacturable
by CNC?" — and those come from N independent LLM/verifier/human votes. Quality
control then means: only accept a label the annotators actually agree on, and
report the disagreement instead of silently taking a coin-flip.

:func:`consensus_label` aggregates the votes into a :class:`ConsensusResult`:
the majority label, the inter-annotator agreement (both the winning-label share
and the pairwise-agreement rate), and an accept/reject decision gated on a
``threshold`` (default 0.66 — two-thirds). A split vote falls below threshold and
is rejected with ``label=None`` so it can be re-queued rather than trusted. An
optional ``gold`` answer adds a gold-standard spot-check (did the majority match
the known-good label?).

Absolute imports, stdlib only, deterministic (ties break by sorted label order).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


def _label_of(vote: Any) -> Any:
    """Extract the label from a vote: a scalar, a ``{"label"/"vote": ...}`` dict,
    or a ``(annotator, label)`` pair."""
    if isinstance(vote, dict):
        for key in ("label", "vote", "answer", "value"):
            if key in vote:
                return vote[key]
        raise ValueError(f"vote dict has no label key: {vote!r}")
    if isinstance(vote, tuple) and len(vote) == 2:
        return vote[1]
    return vote


@dataclass
class ConsensusResult:
    """Aggregated verdict over N independent votes."""

    label: Optional[Any]            # accepted label, or None when below threshold
    accepted: bool
    agreement: float                # share of votes for the winning label
    pairwise_agreement: float       # fraction of annotator pairs that agree
    majority: Optional[Any]         # plurality label (even when rejected)
    n_votes: int
    threshold: float
    distribution: Dict[str, int] = field(default_factory=dict)
    disagreement: bool = False
    minority: Dict[str, int] = field(default_factory=dict)
    gold: Optional[Any] = None
    gold_matches: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "accepted": self.accepted,
            "agreement": self.agreement,
            "pairwise_agreement": self.pairwise_agreement,
            "majority": self.majority,
            "n_votes": self.n_votes,
            "threshold": self.threshold,
            "distribution": self.distribution,
            "disagreement": self.disagreement,
            "minority": self.minority,
            "gold": self.gold,
            "gold_matches": self.gold_matches,
        }


def consensus_label(votes: Iterable[Any],
                    *,
                    threshold: float = 0.66,
                    gold: Optional[Any] = None) -> ConsensusResult:
    """Fold N independent votes into an accept/reject consensus decision.

    Accepts a semantic ``label`` only when its share of the votes reaches
    ``threshold`` (default two-thirds); otherwise the split is reported and
    ``label`` is ``None`` (re-queue it). ``agreement`` is the winning-label share;
    ``pairwise_agreement`` is the classic inter-annotator agreement (fraction of
    the C(N,2) annotator pairs that voted the same). When ``gold`` is given, the
    majority is spot-checked against it (``gold_matches``).
    """
    vote_list = [_label_of(v) for v in votes]
    n = len(vote_list)
    if n == 0:
        return ConsensusResult(
            label=None, accepted=False, agreement=0.0, pairwise_agreement=0.0,
            majority=None, n_votes=0, threshold=threshold,
            distribution={}, disagreement=False, minority={},
            gold=gold, gold_matches=None,
        )

    # Counter keyed by string form so the distribution is JSON-serialisable, but
    # keep the original label value for the returned majority/label.
    counts: Counter = Counter()
    repr_value: Dict[str, Any] = {}
    for v in vote_list:
        key = str(v)
        counts[key] += 1
        repr_value.setdefault(key, v)

    # Winning label: highest count, ties broken by sorted key for determinism.
    top_count = max(counts.values())
    winners = sorted(k for k, c in counts.items() if c == top_count)
    win_key = winners[0]
    majority = repr_value[win_key]
    tie = len(winners) > 1

    agreement = top_count / n

    # Pairwise agreement: same-vote pairs / all pairs.
    total_pairs = n * (n - 1) // 2
    same_pairs = sum(c * (c - 1) // 2 for c in counts.values())
    pairwise = (same_pairs / total_pairs) if total_pairs else 1.0

    accepted = (agreement >= threshold) and not tie
    minority = {k: c for k, c in counts.items() if k != win_key}
    disagreement = len(counts) > 1

    gold_matches = None
    if gold is not None:
        gold_matches = (majority == gold)

    return ConsensusResult(
        label=majority if accepted else None,
        accepted=accepted,
        agreement=agreement,
        pairwise_agreement=pairwise,
        majority=majority,
        n_votes=n,
        threshold=threshold,
        distribution=dict(sorted(counts.items())),
        disagreement=disagreement,
        minority=dict(sorted(minority.items())),
        gold=gold,
        gold_matches=gold_matches,
    )
