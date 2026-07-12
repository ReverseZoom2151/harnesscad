"""CADBench scoring protocol as used by Graph-CAD's ``evaluate_and_report.py``.

Graph-CAD scores generated CAD on CADBench, whose per-sample config is a
three-level checklist: *dimension* ("Object Attributes", "Spatial Understanding
and Structure", "User Instruction Understanding and Execution") -> *parameter*
("Shape accuracy", "Size", ...) -> a list of binary requirement strings. A judge
marks each requirement 0/1; everything after that is arithmetic, and the exact
arithmetic is what makes the published table reproducible:

* a parameter scores the **mean of its requirements**;
* a dimension scores the **mean of its parameter means** -- so a parameter with
  five requirements does not outweigh one with a single requirement;
* ``Avg`` is the unweighted mean of the three dimension scores;
* a corpus score divides by **every** sample, not just the successfully judged
  ones, so a sample whose code failed to execute contributes 0 rather than
  vanishing from the denominator;
* ``E_syntax`` (lower is better) is the percentage of samples that produced no
  exported geometry at all;
* results are reported per split -- CADBench-Sim (``Simulative``) versus
  CADBench-Wild (``Wild``) -- as well as overall.

This module implements that protocol deterministically. The judge itself (a
vision-language model) is external: judgements are supplied as plain 0/1 data,
so the aggregation can be unit-tested and reused for any binary-checklist
benchmark.

Note this is a *sample-level, checklist-shaped* protocol, complementary to the
typed criterion aggregation in :mod:`bench.criteria` (which scores one model
against typed criteria via subdimension means).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DIMENSIONS",
    "DIMENSION_KEYS",
    "SampleConfig",
    "SampleScore",
    "SplitReport",
    "score_parameter",
    "score_dimension",
    "score_sample",
    "report",
]

#: CADBench's three dimensions, in the order the paper's tables use them.
DIMENSIONS: Tuple[str, ...] = (
    "Object Attributes",
    "Spatial Understanding and Structure",
    "User Instruction Understanding and Execution",
)

#: Short names used in the results table.
DIMENSION_KEYS: Mapping[str, str] = {
    "Object Attributes": "Attr",
    "Spatial Understanding and Structure": "Spat",
    "User Instruction Understanding and Execution": "Inst",
}


@dataclass(frozen=True)
class SampleConfig:
    """One CADBench record: its checklist and its split."""

    sample_id: str
    criteria: Mapping[str, Mapping[str, Sequence[str]]]
    split: str = "Unknown"

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id is required")
        for dimension, parameters in self.criteria.items():
            if dimension not in DIMENSIONS:
                raise ValueError(f"unknown dimension: {dimension!r}")
            for parameter, requirements in parameters.items():
                if not requirements:
                    raise ValueError(
                        f"{self.sample_id}/{dimension}/{parameter} has no requirements"
                    )

    def requirement_count(self) -> int:
        return sum(
            len(requirements)
            for parameters in self.criteria.values()
            for requirements in parameters.values()
        )


@dataclass(frozen=True)
class SampleScore:
    """A judged sample: per-dimension scores plus the unweighted average."""

    sample_id: str
    split: str
    dimension_scores: Mapping[str, float]
    average: float
    has_geometry: bool = True
    judged: bool = True


def score_parameter(judgements: Sequence[int]) -> float:
    """Mean of a parameter's binary requirement judgements."""
    values = []
    for value in judgements:
        score = int(value)
        if score not in (0, 1):
            raise ValueError(f"judgements must be 0 or 1, got {value!r}")
        values.append(score)
    if not values:
        raise ValueError("a parameter needs at least one judgement")
    return sum(values) / len(values)


def score_dimension(
    parameters: Mapping[str, Sequence[str]],
    judgements: Mapping[str, Sequence[int]],
) -> float:
    """Mean of the parameter means -- parameters are weighted equally."""
    if not parameters:
        return 0.0
    means: List[float] = []
    for parameter, requirements in parameters.items():
        if parameter not in judgements:
            raise KeyError(f"missing judgements for parameter {parameter!r}")
        values = judgements[parameter]
        if len(values) != len(requirements):
            raise ValueError(
                f"{parameter}: expected {len(requirements)} judgements, got {len(values)}"
            )
        means.append(score_parameter(values))
    return sum(means) / len(means)


def score_sample(
    config: SampleConfig,
    judgements: Optional[Mapping[str, Mapping[str, Sequence[int]]]],
    has_geometry: bool = True,
) -> SampleScore:
    """Score one sample; ``judgements=None`` marks an unjudged (failed) sample.

    A failed sample keeps its place in the corpus with all-zero scores, which is
    what makes the corpus denominator the *total* sample count.
    """
    if judgements is None:
        return SampleScore(
            sample_id=config.sample_id,
            split=config.split,
            dimension_scores={key: 0.0 for key in DIMENSION_KEYS.values()},
            average=0.0,
            has_geometry=has_geometry,
            judged=False,
        )

    scores: Dict[str, float] = {}
    for dimension in DIMENSIONS:
        key = DIMENSION_KEYS[dimension]
        parameters = config.criteria.get(dimension, {})
        if not parameters:
            scores[key] = 0.0
            continue
        scores[key] = score_dimension(parameters, judgements.get(dimension, {}))

    average = sum(scores[DIMENSION_KEYS[name]] for name in DIMENSIONS) / len(DIMENSIONS)
    return SampleScore(
        sample_id=config.sample_id,
        split=config.split,
        dimension_scores=scores,
        average=average,
        has_geometry=has_geometry,
        judged=True,
    )


@dataclass(frozen=True)
class SplitReport:
    """The published table row for one split."""

    split: str
    total: int
    judged: int
    attr: float
    spat: float
    inst: float
    average: float
    syntax_error_rate: float
    extras: Mapping[str, float] = field(default_factory=dict)

    def as_row(self) -> Dict[str, float]:
        return {
            "Attr": self.attr,
            "Spat": self.spat,
            "Inst": self.inst,
            "Avg": self.average,
            "Esyntax": self.syntax_error_rate,
        }


def _split_report(split: str, scores: Sequence[SampleScore]) -> SplitReport:
    total = len(scores)
    if total == 0:
        return SplitReport(split, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    totals = {key: 0.0 for key in DIMENSION_KEYS.values()}
    average_total = 0.0
    missing = 0
    for score in scores:
        for key, value in score.dimension_scores.items():
            totals[key] += value
        average_total += score.average
        if not score.has_geometry:
            missing += 1
    return SplitReport(
        split=split,
        total=total,
        judged=sum(1 for score in scores if score.judged),
        attr=totals["Attr"] / total,
        spat=totals["Spat"] / total,
        inst=totals["Inst"] / total,
        average=average_total / total,
        syntax_error_rate=100.0 * missing / total,
    )


def report(scores: Iterable[SampleScore]) -> Dict[str, SplitReport]:
    """Aggregate sample scores into an ``overall`` row plus one row per split.

    Denominators are the sample counts of each group, so unjudged and
    geometry-less samples drag the score down exactly as they do in the paper.
    """
    items = list(scores)
    grouped: Dict[str, List[SampleScore]] = {}
    for score in items:
        grouped.setdefault(score.split, []).append(score)

    result: Dict[str, SplitReport] = {"overall": _split_report("overall", items)}
    for split in sorted(grouped):
        result[split] = _split_report(split, grouped[split])
    return result
