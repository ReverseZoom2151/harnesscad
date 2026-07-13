"""Benchmark task / scoring harness for E3D-Bench (Fig. 1, Sec. 3).

E3D-Bench standardises evaluation across five tasks and many datasets, then (for
its summary chart) reports every metric "averaged per scene, normalized, and
converted to a higher-is-better scale for consistent comparison".  This module
implements that deterministic aggregation layer:

* ``MetricSpec`` declares a metric name and whether lower is better (AbsRel,
  Acc, Comp, ATE, RPE...) or higher is better (delta inlier ratio, F-score,
  PSNR, SSIM, NC).
* ``normalize_higher_is_better`` min-max normalises a metric's values across the
  compared models into ``[0, 1]`` on a higher-is-better scale.
* ``BenchmarkHarness`` collects raw per-model, per-scene metric values for a set
  of tasks and produces a normalized leaderboard: per-scene scores are averaged,
  then averaged across scenes and tasks.

No wall clock, no randomness -- purely a function of the supplied numbers.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


class MetricSpec:
    """Declares a metric's name and optimisation direction."""

    __slots__ = ("name", "lower_is_better")

    def __init__(self, name: str, lower_is_better: bool = True) -> None:
        self.name = name
        self.lower_is_better = lower_is_better

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        d = "lower" if self.lower_is_better else "higher"
        return "MetricSpec(%r, %s-is-better)" % (self.name, d)


def normalize_higher_is_better(values: Sequence[float],
                               lower_is_better: bool) -> List[float]:
    """Min-max normalise ``values`` to [0, 1], 1 = best.

    A lower-is-better metric is flipped so its smallest raw value maps to 1.
    When every value is equal (zero spread) all scores are 1.0 (all tie at the
    best).  Deterministic and order-preserving.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    out = []
    for v in values:
        if span == 0.0:
            out.append(1.0)
            continue
        frac = (v - lo) / span  # 0 at min, 1 at max
        out.append(1.0 - frac if lower_is_better else frac)
    return out


class BenchmarkHarness:
    """Collects raw metrics and produces a normalized leaderboard."""

    def __init__(self, metrics: Sequence[MetricSpec]) -> None:
        if not metrics:
            raise ValueError("at least one metric required")
        self._metrics: List[MetricSpec] = list(metrics)
        self._specs: Dict[str, MetricSpec] = {m.name: m for m in metrics}
        # results[scene][model][metric] = raw value
        self._results: Dict[str, Dict[str, Dict[str, float]]] = {}

    @property
    def metric_names(self) -> List[str]:
        return [m.name for m in self._metrics]

    def add_result(self, scene: str, model: str,
                   values: Dict[str, float]) -> None:
        """Record one model's raw metric values for one scene."""
        unknown = set(values) - set(self._specs)
        if unknown:
            raise ValueError("unknown metrics: %s" % sorted(unknown))
        self._results.setdefault(scene, {})[model] = dict(values)

    def models(self) -> List[str]:
        seen: List[str] = []
        for scene in self._results.values():
            for model in scene:
                if model not in seen:
                    seen.append(model)
        return sorted(seen)

    def scenes(self) -> List[str]:
        return sorted(self._results)

    def scene_scores(self, scene: str) -> Dict[str, float]:
        """Normalized per-model score for one scene (mean over metrics, [0,1])."""
        if scene not in self._results:
            raise KeyError(scene)
        entry = self._results[scene]
        models = sorted(entry)
        # accumulate normalized score per model across metrics
        acc = {m: 0.0 for m in models}
        counts = {m: 0 for m in models}
        for spec in self._metrics:
            present = [m for m in models if spec.name in entry[m]]
            if not present:
                continue
            raw = [entry[m][spec.name] for m in present]
            norm = normalize_higher_is_better(raw, spec.lower_is_better)
            for m, s in zip(present, norm):
                acc[m] += s
                counts[m] += 1
        return {m: (acc[m] / counts[m] if counts[m] else 0.0) for m in models}

    def leaderboard(self) -> List[tuple]:
        """(model, mean_score) sorted best-first, averaged over all scenes.

        Ties broken by model name for determinism.
        """
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for scene in self.scenes():
            for model, score in self.scene_scores(scene).items():
                totals[model] = totals.get(model, 0.0) + score
                counts[model] = counts.get(model, 0) + 1
        board = [(m, totals[m] / counts[m]) for m in totals]
        board.sort(key=lambda kv: (-kv[1], kv[0]))
        return board

    def best_model(self) -> str:
        board = self.leaderboard()
        if not board:
            raise ValueError("no results recorded")
        return board[0][0]
