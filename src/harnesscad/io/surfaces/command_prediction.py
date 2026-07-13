"""Bayesian workflow-graph next-command prediction.

Paper 179 -- "Toward AI-driven Multimodal Interfaces for Industrial CAD
Modeling" (Choi, Jang, Hyun, CHI '25) -- lists "Bayesian Command
Prediction" (Table 1) and "Bayesian workflow inference" (section 4.1) as a
core AI-enhanced-automation capability, citing the authors' own work
"Advancing 3D CAD with Workflow Graph-Driven Bayesian Command Inferences"
(ref [9]).  The paper does not publish the algorithm; this module supplies a
fully *deterministic* implementation of the idea that needs no trained
neural model:

* A first-order **workflow graph** counts observed command transitions.
* A **Bayesian predictor** turns those counts into a posterior over the next
  command using a symmetric Dirichlet (add-alpha / Laplace) prior over the
  known vocabulary, so unseen contexts fall back gracefully to the prior.

All outputs are deterministic: ranking ties break lexicographically.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

_START = "START"  # sentinel context for sequence-initial commands


class WorkflowGraph:
    """Counts of first-order command transitions across many sessions.

    A single sentinel start-state models sequence-initial commands so that
    "what is the first command usually issued" is a normal prediction.
    """

    def __init__(self) -> None:
        self._trans: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._vocab: set[str] = set()

    def add_sequence(self, commands: Sequence[str]) -> None:
        prev = _START
        for cmd in commands:
            if not cmd:
                raise ValueError("command names must be non-empty")
            self._vocab.add(cmd)
            self._trans[prev][cmd] += 1
            prev = cmd

    def add_sequences(self, sequences: Iterable[Sequence[str]]) -> None:
        for seq in sequences:
            self.add_sequence(seq)

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return tuple(sorted(self._vocab))

    def transition_count(self, context: str, nxt: str) -> int:
        return self._trans.get(context, {}).get(nxt, 0)

    def outgoing(self, context: str) -> Mapping[str, int]:
        return dict(self._trans.get(context, {}))

    def context_total(self, context: str) -> int:
        return sum(self._trans.get(context, {}).values())

    def edges(self) -> tuple[tuple[str, str, int], ...]:
        out: list[tuple[str, str, int]] = []
        for ctx in sorted(self._trans):
            for nxt in sorted(self._trans[ctx]):
                out.append((ctx, nxt, self._trans[ctx][nxt]))
        return tuple(out)


@dataclass(frozen=True)
class Prediction:
    command: str
    probability: float


class BayesianCommandPredictor:
    """Posterior next-command distribution over a :class:`WorkflowGraph`.

    With a symmetric Dirichlet prior of concentration ``alpha`` over a
    vocabulary of size ``V`` and observed transition counts ``n(ctx, c)``
    with total ``N(ctx)``, the posterior predictive probability is::

        P(c | ctx) = (n(ctx, c) + alpha) / (N(ctx) + alpha * V)

    An unseen context (``N(ctx) = 0``) yields the uniform prior, exactly the
    graceful fallback the paper wants for novel workflows.
    """

    def __init__(self, graph: WorkflowGraph, *, alpha: float = 1.0) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        self.graph = graph
        self.alpha = alpha

    def _vocab(self) -> tuple[str, ...]:
        return self.graph.vocabulary

    def probability(self, context: str, command: str) -> float:
        vocab = self._vocab()
        v = len(vocab)
        if v == 0:
            return 0.0
        n = self.graph.transition_count(context, command)
        total = self.graph.context_total(context)
        # An unknown candidate command still gets prior mass proportionally,
        # but is not part of the normalised vocabulary; report 0 for it.
        if command not in self.graph._vocab:  # noqa: SLF001 (intentional)
            return 0.0
        return (n + self.alpha) / (total + self.alpha * v)

    def distribution(self, context: str) -> tuple[Prediction, ...]:
        vocab = self._vocab()
        preds = [
            Prediction(cmd, self.probability(context, cmd)) for cmd in vocab
        ]
        # Highest probability first; deterministic lexical tie-break.
        preds.sort(key=lambda p: (-p.probability, p.command))
        return tuple(preds)

    def predict_next(
        self, history: Sequence[str], *, top_k: Optional[int] = None
    ) -> tuple[Prediction, ...]:
        context = history[-1] if history else _START
        dist = self.distribution(context)
        return dist if top_k is None else dist[:top_k]

    def most_likely(self, history: Sequence[str]) -> Optional[str]:
        preds = self.predict_next(history, top_k=1)
        return preds[0].command if preds else None

    def sequence_log_likelihood(self, commands: Sequence[str]) -> float:
        """Natural-log likelihood of a full command sequence under the model."""
        total = 0.0
        prev = _START
        for cmd in commands:
            p = self.probability(prev, cmd)
            total += math.log(p) if p > 0.0 else float("-inf")
            prev = cmd
        return total

    def perplexity(self, commands: Sequence[str]) -> float:
        """Per-command perplexity; ``inf`` if any step has zero probability."""
        if not commands:
            return 1.0
        ll = self.sequence_log_likelihood(commands)
        if ll == float("-inf"):
            return float("inf")
        return math.exp(-ll / len(commands))
