"""Dataset-distribution auditing (Scale-AI data-engine playbook).

The named #1 risk for a synthetic data engine is the **synthetic-vs-real
distribution gap** (datagen/pipeline.py CAVEAT): a 100% build-yield measures
*validity*, not *realism*, and a bootstrap generator over-produces whatever its
templates make cheap. Before a corpus is trained on it, someone has to look at
its shape — which op tags dominate, which op transitions (n-grams) never occur,
which feature types and part families are covered — and flag the regions that
are over- or under-represented relative to an intended (real-world) target.

:func:`audit_distribution` folds an arbitrary bag of synthetic artefacts (the
:class:`~datagen.pipeline.Sample` objects the generators emit, or the
:class:`~dataengine.trajectory.Trajectory` records the flywheel logs, or raw op
dicts) into a :class:`DistributionReport`: four histograms (op-tag frequency,
op bigrams, feature-type counts, part-family coverage) plus an over/under flag
against an optional ``target`` distribution. When no target is supplied the
expectation defaults to *uniform* over the observed tags, so a lopsided corpus
still gets flagged. Divergence from target is summarised two ways — a
KL divergence and a chi-square-lite statistic (the "lite" is that we use the
target as the expected law directly, no continuity correction).

Absolute imports, stdlib only, deterministic (histogram key order is sorted; no
wall clock, no randomness).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- op taxonomy ------------------------------------------------------------
# Mirrors cisp/ops.py: which tags are sketch primitives, constraints, features.
SKETCH_OPS = frozenset({
    "new_sketch", "add_point", "add_line", "add_circle", "add_rectangle",
})
CONSTRAINT_OPS = frozenset({"constrain"})
FEATURE_OPS = frozenset({
    "extrude", "fillet", "boolean", "revolve", "chamfer", "hole", "shell",
    "draft", "loft", "sweep", "linear_pattern", "circular_pattern", "mirror",
})


def _category(tag: str) -> str:
    if tag in SKETCH_OPS:
        return "sketch"
    if tag in CONSTRAINT_OPS:
        return "constraint"
    if tag in FEATURE_OPS:
        return "feature"
    return "other"


# =====================================================================
# Extraction — turn heterogeneous inputs into (op-tag stream, family)
# =====================================================================

def _tag_of(entry: Any) -> Optional[str]:
    """The op tag of a single op entry (a dict, an Op instance, or a tool_call)."""
    if isinstance(entry, dict):
        for key in ("op", "kind", "type"):
            v = entry.get(key)
            if v:
                return str(v)
        return None
    # An Op dataclass carries its tag on the OP class attribute.
    tag = getattr(entry, "OP", None)
    if tag:
        return str(tag)
    return None


def op_tags(item: Any) -> List[str]:
    """The ordered op-tag stream of one artefact.

    Accepts a :class:`~datagen.pipeline.Sample` (``.ops`` = list of op dicts), a
    :class:`~dataengine.trajectory.Trajectory` (``.steps`` -> ``action.tool_call``),
    a plain dict with ``ops``/``steps``/``op`` keys, or a bare list of op
    entries. Divergent (rolled-back / rejected) trajectory steps are skipped so
    the audit reflects the ops that actually stuck.
    """
    # Trajectory-like: has steps with actions.
    steps = getattr(item, "steps", None)
    if steps is not None:
        tags: List[str] = []
        for s in steps:
            if getattr(s, "divergent", False):
                continue
            action = getattr(s, "action", None)
            call = getattr(action, "tool_call", None) if action is not None else None
            t = _tag_of(call) if call is not None else None
            if t:
                tags.append(t)
        return tags

    # Sample-like / object with .ops.
    ops = getattr(item, "ops", None)
    if ops is not None:
        return [t for t in (_tag_of(o) for o in ops) if t]

    if isinstance(item, dict):
        if "steps" in item:
            out: List[str] = []
            for s in item["steps"]:
                if isinstance(s, dict) and s.get("outcome") not in (None, "applied"):
                    continue
                call = None
                if isinstance(s, dict):
                    action = s.get("action") or {}
                    call = action.get("tool_call") if isinstance(action, dict) else None
                t = _tag_of(call) if call is not None else None
                if t:
                    out.append(t)
            return out
        if "ops" in item:
            return [t for t in (_tag_of(o) for o in item["ops"]) if t]
        if _tag_of(item):
            return [_tag_of(item)]  # a single bare op dict
        return []

    if isinstance(item, (list, tuple)):
        return [t for t in (_tag_of(o) for o in item) if t]

    return []


def family_of(item: Any) -> Optional[str]:
    """The part-family label of one artefact (generator name / declared family)."""
    gen = getattr(item, "generator", None)
    if gen:
        return str(gen)
    params = getattr(item, "params", None)
    if isinstance(params, dict) and params.get("generator"):
        return str(params["generator"])
    meta = getattr(item, "metadata", None)
    if isinstance(meta, dict):
        for key in ("part_family", "family", "generator"):
            if meta.get(key):
                return str(meta[key])
    if isinstance(item, dict):
        for key in ("generator", "family", "part_family"):
            if item.get(key):
                return str(item[key])
        p = item.get("params")
        if isinstance(p, dict) and p.get("generator"):
            return str(p["generator"])
        m = item.get("metadata")
        if isinstance(m, dict):
            for key in ("part_family", "family", "generator"):
                if m.get(key):
                    return str(m[key])
    return None


def _bigrams(tags: List[str]) -> List[Tuple[str, str]]:
    return [(tags[i], tags[i + 1]) for i in range(len(tags) - 1)]


# =====================================================================
# Report
# =====================================================================

@dataclass
class DistributionReport:
    """The audited shape of a synthetic corpus + over/under-representation flags."""

    n_items: int
    n_ops: int
    op_tag_freq: Dict[str, int] = field(default_factory=dict)
    op_ngram_freq: Dict[str, int] = field(default_factory=dict)   # "a>b" bigram -> count
    feature_types: Dict[str, int] = field(default_factory=dict)
    categories: Dict[str, int] = field(default_factory=dict)
    families: Dict[str, int] = field(default_factory=dict)
    over_represented: List[dict] = field(default_factory=list)
    under_represented: List[dict] = field(default_factory=list)
    divergence: Dict[str, float] = field(default_factory=dict)
    target: Optional[Dict[str, float]] = None

    @property
    def coverage(self) -> int:
        """Number of distinct part families present in the corpus."""
        return len(self.families)

    def op_tag_share(self) -> Dict[str, float]:
        total = sum(self.op_tag_freq.values())
        if not total:
            return {}
        return {k: v / total for k, v in self.op_tag_freq.items()}

    def to_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "n_ops": self.n_ops,
            "coverage": self.coverage,
            "op_tag_freq": dict(sorted(self.op_tag_freq.items())),
            "op_ngram_freq": dict(sorted(self.op_ngram_freq.items())),
            "feature_types": dict(sorted(self.feature_types.items())),
            "categories": dict(sorted(self.categories.items())),
            "families": dict(sorted(self.families.items())),
            "over_represented": self.over_represented,
            "under_represented": self.under_represented,
            "divergence": self.divergence,
            "target": self.target,
        }

    def render(self) -> str:
        lines = [
            "distribution audit",
            f"  items={self.n_items} ops={self.n_ops} family_coverage={self.coverage}",
            "  op-tag frequency:",
        ]
        share = self.op_tag_share()
        for tag, cnt in sorted(self.op_tag_freq.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"    {tag:<16} {cnt:>4}  ({share.get(tag, 0.0):.1%})")
        lines.append("  families:")
        for fam, cnt in sorted(self.families.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"    {fam:<16} {cnt:>4}")
        if self.over_represented:
            lines.append("  OVER-represented vs target:")
            for f in self.over_represented:
                lines.append(f"    {f['tag']:<16} obs={f['observed']:.1%} "
                             f"target={f['target']:.1%} ratio={f['ratio']:.2f}x")
        if self.under_represented:
            lines.append("  UNDER-represented vs target:")
            for f in self.under_represented:
                lines.append(f"    {f['tag']:<16} obs={f['observed']:.1%} "
                             f"target={f['target']:.1%} ratio={f['ratio']:.2f}x")
        if self.divergence:
            lines.append(f"  divergence: KL={self.divergence.get('kl', 0.0):.4f} "
                         f"chi_square={self.divergence.get('chi_square', 0.0):.4f}")
        return "\n".join(lines)


# =====================================================================
# audit_distribution
# =====================================================================

def audit_distribution(samples_or_trajectories: Iterable[Any],
                       target: Optional[Dict[str, float]] = None,
                       tolerance: float = 0.5) -> DistributionReport:
    """Audit a synthetic corpus's distribution and flag skew vs an optional target.

    ``samples_or_trajectories`` is any iterable of Samples, Trajectories, dicts, or
    raw op lists. ``target`` maps op tag -> desired weight (need not be
    normalised); when ``None`` the expected law defaults to *uniform* over the
    observed tags so a lopsided corpus is still flagged. A tag is flagged
    over-represented when its observed share exceeds its target share by more than
    ``tolerance`` (a fraction), under-represented when it falls short by more than
    ``tolerance`` (or is absent). KL and a chi-square-lite statistic summarise the
    total divergence.
    """
    items = list(samples_or_trajectories)

    tag_counts: Counter = Counter()
    ngram_counts: Counter = Counter()
    feature_counts: Counter = Counter()
    category_counts: Counter = Counter()
    family_counts: Counter = Counter()

    for it in items:
        tags = op_tags(it)
        tag_counts.update(tags)
        for a, b in _bigrams(tags):
            ngram_counts[f"{a}>{b}"] += 1
        for t in tags:
            category_counts[_category(t)] += 1
            if t in FEATURE_OPS:
                feature_counts[t] += 1
        fam = family_of(it)
        if fam is not None:
            family_counts[fam] += 1

    n_ops = sum(tag_counts.values())

    # --- over/under representation vs target (default uniform) --------------
    over: List[dict] = []
    under: List[dict] = []
    divergence: Dict[str, float] = {}
    resolved_target: Optional[Dict[str, float]] = None

    if n_ops > 0:
        observed_p = {t: c / n_ops for t, c in tag_counts.items()}
        if target:
            tw = sum(v for v in target.values() if v > 0)
            target_p = {t: (w / tw) for t, w in target.items() if tw > 0}
        else:
            # Uniform expectation over the observed tag vocabulary.
            distinct = sorted(tag_counts)
            target_p = {t: 1.0 / len(distinct) for t in distinct} if distinct else {}
        resolved_target = dict(sorted(target_p.items()))

        all_tags = sorted(set(observed_p) | set(target_p))
        for t in all_tags:
            obs = observed_p.get(t, 0.0)
            tgt = target_p.get(t, 0.0)
            ratio = (obs / tgt) if tgt > 0 else float("inf")
            flag = {"tag": t, "observed": obs, "target": tgt, "ratio": ratio}
            if tgt <= 0.0:
                # Present in corpus but absent from target -> over-represented.
                if obs > 0:
                    over.append(flag)
            elif obs > tgt * (1.0 + tolerance):
                over.append(flag)
            elif obs < tgt * (1.0 - tolerance):
                under.append(flag)

        kl = 0.0
        chi = 0.0
        for t in all_tags:
            obs = observed_p.get(t, 0.0)
            tgt = target_p.get(t, 0.0)
            if tgt > 0.0:
                if obs > 0.0:
                    kl += obs * math.log(obs / tgt)
                exp_count = tgt * n_ops
                obs_count = tag_counts.get(t, 0)
                chi += (obs_count - exp_count) ** 2 / exp_count
        divergence = {"kl": kl, "chi_square": chi}

    return DistributionReport(
        n_items=len(items),
        n_ops=n_ops,
        op_tag_freq=dict(tag_counts),
        op_ngram_freq=dict(ngram_counts),
        feature_types=dict(feature_counts),
        categories=dict(category_counts),
        families=dict(family_counts),
        over_represented=over,
        under_represented=under,
        divergence=divergence,
        target=resolved_target,
    )
