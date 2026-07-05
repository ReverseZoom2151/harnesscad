"""Geometry anomaly detector — a pre-verify "smell" pass.

The rule-based verifiers (``verify.py``, ``checks_dfm``, ``checks_geometry`` ...)
catch geometry that violates an *explicit* rule: an over-constrained sketch, an
invalid B-rep, an aspect ratio past a hard DFM limit. This module catches the
other half: geometry that breaks no stated rule yet is *statistically unusual*
relative to a corpus of known-good parts. It is an unsupervised outlier detector
over cheap geometric features, and it is purely advisory — it never blocks.

Design (stdlib only, deterministic):

  * :func:`feature_vector` turns a backend (or a raw metrics dict) into a small
    dict of scale-robust geometric features that ARE computable today — bbox
    aspect ratios, surface-area-to-volume ratio, log-scaled volume/area, and the
    available face/edge/vertex/feature/entity counts. It degrades gracefully:
    a field that a backend cannot answer simply produces no feature.

  * :class:`AnomalyModel` fits a per-feature baseline (mean/std AND median/IQR)
    over a list of feature vectors from known-good parts, then :meth:`score`\\ s a
    new vector, flagging every feature that lands beyond a z-score / IQR
    threshold. The baseline round-trips to JSON (:meth:`to_dict` / :meth:`save`).

  * :class:`IsolationLite` is a small, self-contained isolation-forest-style
    ensemble over the same vectors (random axis-aligned splits; short average
    isolation path == outlier). It is the multivariate option; the per-feature
    baseline is the interpretable default.

  * :class:`AnomalyCheck` is a ``verify.Verifier`` (name ``'anomaly'``) that
    scores the current part against a fitted baseline and emits a WARNING
    ``'geometry-anomaly'`` naming the outlier features. It INFO-skips when the
    model is unfit or the backend cannot be measured. It never emits an ERROR.

Upgrade path: :class:`AnomalyModel` / :class:`IsolationLite` are drop-in
placeholders for a real ``sklearn.ensemble.IsolationForest`` or a small
autoencoder trained on the same feature vectors (reconstruction error as the
anomaly score) — swap the object behind the same ``fit`` / ``score`` interface
without touching :class:`AnomalyCheck` or its call sites.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from verify import Diagnostic, Severity, VerifyReport

# --- tuning knobs (module-level so they are easy to find / override) --------
DEFAULT_Z_THRESHOLD = 3.5      # |z| beyond this flags a feature (mean/std mode)
DEFAULT_IQR_K = 3.0            # value outside [q1 - k*iqr, q3 + k*iqr] flags it
_EPS = 1e-12
_BIG = 1e6                     # finite stand-in for "infinitely far" (zero spread)
_MIN_SAMPLES = 2              # a feature needs >=2 baseline values to judge spread


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _finite(x: Any) -> Optional[float]:
    """Coerce to a finite float, or None (so it is simply omitted)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _merge_source(backend_or_metrics: Any) -> Dict[str, Any]:
    """Collect a single flat dict of raw measurements from a backend or dict.

    A backend is anything exposing ``query``; we ask it for the geometry
    queries a real kernel answers (``metrics`` / ``measure`` / ``summary``) and
    merge whatever it returns. The stub answers only ``summary`` (no measure /
    metrics), so callers naturally get fewer features and degrade gracefully.
    """
    if hasattr(backend_or_metrics, "query"):
        src: Dict[str, Any] = {}
        for q in ("summary", "measure", "metrics"):
            try:
                part = backend_or_metrics.query(q)
            except Exception:  # noqa: BLE001 - a backend query must never break us
                part = None
            if isinstance(part, dict):
                src.update(part)
        return src
    if isinstance(backend_or_metrics, dict):
        return dict(backend_or_metrics)
    raise TypeError(
        "feature_vector expects a backend (with .query) or a metrics dict, "
        f"got {type(backend_or_metrics).__name__}")


def feature_vector(backend_or_metrics: Any) -> Dict[str, float]:
    """Extract per-part geometric features that are computable *now*.

    Accepts a :class:`~backends.base.GeometryBackend` (queried for
    ``metrics``/``measure``/``summary``) or a raw metrics dict. Only features
    whose inputs are present and finite are emitted, so a backend that cannot
    answer a query (e.g. the stub, which has no real geometry) yields a smaller
    vector rather than zeros or errors.

    Features (all scale-robust so parts of different absolute size compare):
      * ``aspect_ratio``  — longest bbox side / shortest positive side
      * ``elongation``    — longest side / median side
      * ``flatness``      — median side / shortest positive side
      * ``sa_to_vol``     — surface area / volume (shape complexity per volume)
      * ``log_volume`` / ``log_surface_area`` — log10 of raw magnitudes
      * ``faces`` / ``edges`` / ``vertices`` — B-rep counts, when reported
      * ``feature_count`` / ``entity_count`` / ``sketch_count`` — from summary
    """
    src = _merge_source(backend_or_metrics)
    feats: Dict[str, float] = {}

    # -- bounding-box shape ratios -----------------------------------------
    bbox = src.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 3:
        dims = [_finite(d) for d in bbox]
        if all(d is not None for d in dims):
            ordered = sorted(dims)                       # [min, mid, max]
            lo, mid, hi = ordered[0], ordered[1], ordered[2]
            positive = [d for d in ordered if d > _EPS]
            if positive:
                pmin = positive[0]
                feats["aspect_ratio"] = hi / pmin
                if mid > _EPS:
                    feats["flatness"] = mid / pmin
                    feats["elongation"] = hi / mid

    # -- surface-area / volume ---------------------------------------------
    volume = _finite(src.get("volume"))
    area = _finite(src.get("surface_area"))
    if volume is not None and volume > _EPS:
        feats["log_volume"] = math.log10(volume)
        if area is not None and area > _EPS:
            feats["sa_to_vol"] = area / volume
    if area is not None and area > _EPS:
        feats["log_surface_area"] = math.log10(area)

    # -- discrete counts (present only when the backend reports them) ------
    for key in ("faces", "edges", "vertices",
                "feature_count", "entity_count", "sketch_count"):
        v = _finite(src.get(key))
        if v is not None:
            feats[key] = v

    return feats


# ---------------------------------------------------------------------------
# Per-feature statistical baseline
# ---------------------------------------------------------------------------
@dataclass
class AnomalyScore:
    """Result of scoring one feature vector against a baseline.

    ``score`` is the largest per-feature deviation (in z units for the
    ``'zscore'`` method, in IQR-distance units for ``'iqr'``); ``outlier_features``
    names the flagged features worst-first; ``is_outlier`` is True when any
    feature is flagged. ``details`` maps every judged feature to its deviation.
    """

    score: float
    outlier_features: List[str] = field(default_factory=list)
    is_outlier: bool = False
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "outlier_features": list(self.outlier_features),
            "is_outlier": self.is_outlier,
            "details": dict(self.details),
        }


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _quartiles(values: Sequence[float]) -> tuple:
    """(q1, q3) via linear interpolation on the sorted sample."""
    s = sorted(values)
    n = len(s)

    def _q(p: float) -> float:
        if n == 1:
            return s[0]
        pos = p * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return s[lo]
        frac = pos - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    return _q(0.25), _q(0.75)


class AnomalyModel:
    """A per-feature baseline over a corpus of known-good feature vectors.

    ``method`` selects how a feature's deviation is measured:
      * ``'zscore'`` — ``|x - mean| / std`` against ``z_threshold``;
      * ``'iqr'``    — distance outside ``[q1 - k*iqr, q3 + k*iqr]`` in IQR units
        against a threshold of 1.0 (robust to outliers in the baseline itself).

    Both statistics are always stored, so the persisted baseline can be scored
    either way after loading.
    """

    def __init__(self, method: str = "zscore",
                 z_threshold: float = DEFAULT_Z_THRESHOLD,
                 iqr_k: float = DEFAULT_IQR_K) -> None:
        if method not in ("zscore", "iqr"):
            raise ValueError(f"unknown method '{method}' (use 'zscore' or 'iqr')")
        self.method = method
        self.z_threshold = float(z_threshold)
        self.iqr_k = float(iqr_k)
        # feature -> {n, mean, std, median, q1, q3, iqr, min, max}
        self.baseline: Dict[str, Dict[str, float]] = {}
        self.n_samples = 0

    # -- fitting ------------------------------------------------------------
    def fit(self, vectors: Sequence[Dict[str, float]]) -> "AnomalyModel":
        """Build the baseline from feature vectors of known-good parts."""
        columns: Dict[str, List[float]] = {}
        for vec in vectors:
            for key, val in vec.items():
                f = _finite(val)
                if f is not None:
                    columns.setdefault(key, []).append(f)

        baseline: Dict[str, Dict[str, float]] = {}
        for key, vals in columns.items():
            n = len(vals)
            mean = sum(vals) / n
            if n >= 2:
                var = sum((v - mean) ** 2 for v in vals) / (n - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            q1, q3 = _quartiles(vals)
            baseline[key] = {
                "n": float(n),
                "mean": mean,
                "std": std,
                "median": _median(vals),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "min": min(vals),
                "max": max(vals),
            }
        self.baseline = baseline
        self.n_samples = len(vectors)
        return self

    @property
    def is_fit(self) -> bool:
        return bool(self.baseline)

    # -- scoring ------------------------------------------------------------
    def _deviation(self, key: str, value: float) -> float:
        """Deviation of ``value`` for feature ``key`` in this method's units."""
        stat = self.baseline[key]
        if self.method == "zscore":
            std = stat["std"]
            if std <= _EPS:
                return 0.0 if abs(value - stat["mean"]) <= _EPS else _BIG
            return abs(value - stat["mean"]) / std
        # iqr: how far outside the whisker fence, in IQR units (0 == inside)
        iqr = stat["iqr"]
        if iqr <= _EPS:
            # Degenerate spread: fall back to "differs from the median at all".
            return 0.0 if abs(value - stat["median"]) <= _EPS else _BIG
        lo = stat["q1"] - self.iqr_k * iqr
        hi = stat["q3"] + self.iqr_k * iqr
        if value < lo:
            return (lo - value) / iqr
        if value > hi:
            return (value - hi) / iqr
        return 0.0

    def _threshold(self) -> float:
        return self.z_threshold if self.method == "zscore" else 1.0

    def score(self, vector: Dict[str, float]) -> AnomalyScore:
        """Score one feature vector; flag features beyond the threshold."""
        thresh = self._threshold()
        details: Dict[str, float] = {}
        flagged: List[tuple] = []
        worst = 0.0
        for key, raw in vector.items():
            val = _finite(raw)
            stat = self.baseline.get(key)
            if val is None or stat is None or stat["n"] < _MIN_SAMPLES:
                continue  # nothing to compare against -> degrade gracefully
            dev = self._deviation(key, val)
            details[key] = dev
            worst = max(worst, dev)
            if dev > thresh:
                flagged.append((dev, key))
        flagged.sort(key=lambda t: (-t[0], t[1]))
        return AnomalyScore(
            score=worst,
            outlier_features=[k for _, k in flagged],
            is_outlier=bool(flagged),
            details=details,
        )

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "kind": "per_feature_baseline",
            "method": self.method,
            "z_threshold": self.z_threshold,
            "iqr_k": self.iqr_k,
            "n_samples": self.n_samples,
            "baseline": self.baseline,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnomalyModel":
        model = cls(
            method=d.get("method", "zscore"),
            z_threshold=d.get("z_threshold", DEFAULT_Z_THRESHOLD),
            iqr_k=d.get("iqr_k", DEFAULT_IQR_K),
        )
        model.baseline = {
            k: {sk: float(sv) for sk, sv in stat.items()}
            for k, stat in d.get("baseline", {}).items()
        }
        model.n_samples = int(d.get("n_samples", 0))
        return model

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "AnomalyModel":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# IsolationLite — a self-contained isolation-forest-style ensemble
# ---------------------------------------------------------------------------
def _avg_path_length(n: int) -> float:
    """Expected path length of an unsuccessful BST search over n points.

    The normalisation constant c(n) from the Isolation Forest paper; used to
    turn an average isolation depth into a bounded anomaly score.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    harmonic = math.log(n - 1) + 0.5772156649015329  # + Euler-Mascheroni
    return 2.0 * harmonic - (2.0 * (n - 1) / n)


class _ITree:
    """One isolation tree over a fixed feature ordering."""

    __slots__ = ("split_feature", "split_value", "left", "right", "size")

    def __init__(self) -> None:
        self.split_feature: Optional[int] = None
        self.split_value: float = 0.0
        self.left: Optional["_ITree"] = None
        self.right: Optional["_ITree"] = None
        self.size: int = 0


class IsolationLite:
    """A tiny isolation-forest-style multivariate outlier detector.

    Builds ``n_trees`` random trees over the fitted rows; each tree repeatedly
    picks a random feature and a random split within the current node's range.
    Points that isolate in a short average path are outliers. Deterministic for
    a given ``seed``. Score is in [0, 1); ~0.5 is the ensemble's expected depth,
    higher means more anomalous. This is the drop-in placeholder for
    ``sklearn.ensemble.IsolationForest``.
    """

    def __init__(self, n_trees: int = 64, sample_size: int = 256,
                 seed: int = 0, threshold: float = 0.62) -> None:
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.seed = seed
        self.threshold = threshold
        self.features: List[str] = []
        self.trees: List[_ITree] = []
        self._c = 1.0

    def _row(self, vec: Dict[str, float]) -> List[float]:
        return [float(vec.get(f, 0.0)) for f in self.features]

    def _grow(self, rows: List[List[float]], rnd: random.Random,
              depth: int, max_depth: int) -> _ITree:
        node = _ITree()
        node.size = len(rows)
        if depth >= max_depth or len(rows) <= 1:
            return node
        # Pick a feature that actually varies in this subset, if any.
        dims = list(range(len(self.features)))
        rnd.shuffle(dims)
        chosen = None
        for f in dims:
            col = [r[f] for r in rows]
            lo, hi = min(col), max(col)
            if hi - lo > _EPS:
                chosen = (f, lo, hi)
                break
        if chosen is None:
            return node  # all points identical -> external node
        f, lo, hi = chosen
        node.split_feature = f
        node.split_value = rnd.uniform(lo, hi)
        left = [r for r in rows if r[f] < node.split_value]
        right = [r for r in rows if r[f] >= node.split_value]
        node.left = self._grow(left, rnd, depth + 1, max_depth)
        node.right = self._grow(right, rnd, depth + 1, max_depth)
        return node

    def fit(self, vectors: Sequence[Dict[str, float]]) -> "IsolationLite":
        # Fixed, sorted feature ordering across all trees -> deterministic.
        keys = set()
        for vec in vectors:
            keys.update(k for k, v in vec.items() if _finite(v) is not None)
        self.features = sorted(keys)
        rows_all = [self._row(v) for v in vectors]
        n = len(rows_all)
        self._c = _avg_path_length(min(self.sample_size, n)) or 1.0
        rnd = random.Random(self.seed)
        max_depth = max(1, int(math.ceil(math.log2(max(2, self.sample_size)))))
        self.trees = []
        for _ in range(self.n_trees):
            if n > self.sample_size:
                sample = rnd.sample(rows_all, self.sample_size)
            else:
                sample = list(rows_all)
            self.trees.append(self._grow(sample, rnd, 0, max_depth))
        return self

    @property
    def is_fit(self) -> bool:
        return bool(self.trees)

    def _path_length(self, row: List[float], node: _ITree, depth: int) -> float:
        if node.split_feature is None or node.left is None:
            return depth + _avg_path_length(node.size)
        if row[node.split_feature] < node.split_value:
            return self._path_length(row, node.left, depth + 1)
        return self._path_length(row, node.right, depth + 1)

    def anomaly_score(self, vector: Dict[str, float]) -> float:
        """Bounded anomaly score in [0, 1); higher == more anomalous."""
        if not self.trees:
            return 0.0
        row = self._row(vector)
        avg = sum(self._path_length(row, t, 0) for t in self.trees) / len(self.trees)
        return 2.0 ** (-avg / self._c)

    def score(self, vector: Dict[str, float]) -> AnomalyScore:
        """Score matching :meth:`AnomalyModel.score`'s shape (no per-feature
        attribution — an isolation forest is inherently multivariate)."""
        s = self.anomaly_score(vector)
        return AnomalyScore(
            score=s,
            outlier_features=[],
            is_outlier=s > self.threshold,
            details={"isolation_score": s},
        )

    def is_outlier(self, vector: Dict[str, float]) -> bool:
        return self.anomaly_score(vector) > self.threshold


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
class AnomalyCheck:
    """Advisory verifier: flag statistically-unusual geometry (a smell pass).

    Scores the current part against a fitted :class:`AnomalyModel` (or any object
    with a compatible ``is_fit`` / ``score`` interface, e.g. :class:`IsolationLite`
    wrapped to expose ``is_fit``). Emits a WARNING ``'geometry-anomaly'`` naming
    the outlier features when the part is an outlier. INFO-skips (never blocks)
    when the model is unfit or the backend yields no measurable features. It is
    advisory-only and NEVER emits an ERROR.
    """

    name = "anomaly"

    def __init__(self, model: Optional[AnomalyModel] = None) -> None:
        self.model = model

    def check(self, backend, opdag) -> VerifyReport:
        model = self.model
        if model is None or not getattr(model, "is_fit", False):
            return VerifyReport([Diagnostic(
                Severity.INFO, "anomaly-skipped",
                "no fitted anomaly baseline; skipping anomaly smell pass")])

        try:
            vector = feature_vector(backend)
        except Exception:  # noqa: BLE001 - detection must never break the loop
            vector = {}

        if not vector:
            return VerifyReport([Diagnostic(
                Severity.INFO, "anomaly-unmeasurable",
                "backend reported no measurable geometry features; "
                "skipping anomaly smell pass")])

        result = model.score(vector)
        if not result.is_outlier:
            return VerifyReport([Diagnostic(
                Severity.INFO, "anomaly-clear",
                f"geometry within learned baseline (score={result.score:.3g})")])

        if result.outlier_features:
            named = ", ".join(result.outlier_features)
            msg = (f"geometry is statistically unusual (score={result.score:.3g}); "
                   f"outlier features: {named}")
        else:
            msg = (f"geometry is statistically unusual (score={result.score:.3g})")
        return VerifyReport([Diagnostic(
            Severity.WARNING, "geometry-anomaly", msg)])


def with_anomaly(verifiers: Sequence, model: Optional[AnomalyModel] = None) -> list:
    """Return a new verifier list with an :class:`AnomalyCheck` appended.

    Mirrors ``checks_dfm.with_dfm``: does not mutate the input list, so it is a
    non-invasive way to add the smell pass to a default verifier set.
    """
    return list(verifiers) + [AnomalyCheck(model)]
