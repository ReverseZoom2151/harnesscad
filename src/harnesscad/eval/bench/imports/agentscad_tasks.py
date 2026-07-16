"""AgentSCAD's 14 machine-checkable text-to-CAD tasks, contract-graded.

Source: AgentSCAD (resources/cad_repos/AgentSCAD-main/AgentSCAD-main),
``benchmarks/{simple,medium,hard}/*.json``. Each task carries a prompt, a
difficulty tier, an ``expected_bbox`` in mm, ``required_features`` (grading
language for a checklist judge) and ``tolerances`` (a per-task bbox band in mm
plus an exact expected hole count). It is the closest external format to the
harness's own ``eval/hardcorpus/contract_grader.py``: the expectations are
MEASURABLE, so each task compiles into a Measured Geometric Contract.

LICENSE: MIT (Copyright (c) 2026 AgentSCAD) -- redistribution permitted, so
all 14 task files are VENDORED under ``agentscad/`` next to this module, with
provenance (source path, SHA-256, bytes) in ``agentscad/MANIFEST.json`` and
attribution in ``agentscad/LICENSE-NOTICE``. The resources checkout is the
fallback, never a requirement.

Mapping onto harness shapes:

* :func:`briefs` -- each task as an
  :class:`~harnesscad.eval.bench.imports.ImportedBrief` with the stated bbox
  (volume/genus stay ``None``: AgentSCAD does not state them, and this package
  never invents numbers).
* :func:`contract_for_task` -- the task's contract-style oracle: three bound
  ``bbox_{x,y,z}_mm`` predicates using THE TASK'S OWN tolerance band (2-5 mm,
  wider than the oracle's 1e-2 default -- these prompts under-determine exact
  extents, and the task authors said so numerically), one bound exact
  ``hole_count`` predicate, and unbound ``volume_mm3`` / ``genus`` markers for
  the measurables the task does not state. Graded with the same
  ``harnesscad.domain.spec.contract.check`` the contract grader uses.

The ``required_features`` sentences are carried as checklist grading language
(same role as CADBench rubric rows), not as predicates: "flat L-shape" is a
judge's sentence, not a measurement.

Stdlib + the pure contract module only; no kernel, no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from harnesscad.domain.spec import contract as _contract
from harnesscad.eval.bench.imports import ImportedBrief, Manifest, load_manifest

__all__ = [
    "SOURCE_REPO",
    "LICENSE",
    "EXPECTED_TASKS",
    "EXPECTED_PER_TIER",
    "AgentScadTask",
    "manifest",
    "load",
    "briefs",
    "contract_for_task",
    "main",
]

_SOURCE = "agentscad"
SOURCE_REPO = "AgentSCAD"
LICENSE = "MIT"

EXPECTED_TASKS = 14
EXPECTED_PER_TIER: Dict[str, int] = {"simple": 5, "medium": 6, "hard": 3}

_TIERS: Tuple[str, ...] = ("simple", "medium", "hard")
_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class AgentScadTask:
    """One AgentSCAD benchmark task, exactly as its authors stated it."""

    id: str
    prompt: str
    difficulty: str
    expected_part_type: str
    required_features: Tuple[str, ...]
    expected_bbox: Tuple[float, float, float]
    #: the task's own bbox tolerance band in mm (2-5 mm across the suite).
    tol_bbox_mm: float
    #: exact expected hole count (0 is a real expectation: "no holes").
    expected_hole_count: Optional[int]


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def load() -> List[AgentScadTask]:
    """All 14 tasks from the vendored copies (resources/ as fallback)."""
    m = manifest()
    tasks: List[AgentScadTask] = []
    for e in m.by_role("task"):
        path = m.resolve(e)
        if path is None:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        bbox = raw.get("expected_bbox") or []
        tol = raw.get("tolerances") or {}
        hole_count = tol.get("hole_count")
        tasks.append(AgentScadTask(
            id=str(raw["id"]),
            prompt=str(raw["prompt"]),
            difficulty=str(raw.get("difficulty", "")),
            expected_part_type=str(raw.get("expected_part_type", "")),
            required_features=tuple(str(f) for f in raw.get("required_features", [])),
            expected_bbox=tuple(float(v) for v in bbox),
            tol_bbox_mm=float(tol.get("bbox_mm", 0.0)),
            expected_hole_count=int(hole_count) if hole_count is not None else None,
        ))
    tasks.sort(key=lambda t: (_TIERS.index(t.difficulty)
                              if t.difficulty in _TIERS else len(_TIERS), t.id))
    return tasks


def briefs() -> List[ImportedBrief]:
    return [
        ImportedBrief(
            id="agentscad_%s" % t.id,
            source_repo=SOURCE_REPO,
            license=LICENSE,
            text=t.prompt,
            difficulty=t.difficulty,
            bbox=t.expected_bbox,
            categories=(t.expected_part_type,) if t.expected_part_type else (),
            tags=("machine_checkable",) + t.required_features,
            note="AgentSCAD task; contract_for_task() compiles its stated "
                 "expectations into a Measured Geometric Contract",
        )
        for t in load()
    ]


def contract_for_task(task: AgentScadTask) -> "_contract.MeasuredGeometricContract":
    """The task's stated expectations as a Measured Geometric Contract.

    Bound predicates carry ONLY numbers the task states: per-axis bbox with
    the task's own tolerance band, and the exact hole count. Volume and genus
    are unbound ``[NEEDS CLARIFICATION]`` markers -- AgentSCAD does not state
    them and the anti-guess rule says an unstated measurable is never guessed
    (a hole count is NOT promoted to a genus: the task does not say the holes
    are through-holes on every part).
    """
    predicates: List["_contract.Predicate"] = []
    for axis, want in zip(_AXES, task.expected_bbox):
        predicates.append(_contract.Predicate(
            key="bbox_%s_mm" % axis,
            target=float(want),
            tolerance=float(task.tol_bbox_mm),
            kind=_contract.PredicateKind.MEASURED,
            note="expected_bbox %s axis, task's own +/-%g mm band"
                 % (axis, task.tol_bbox_mm),
        ))
    if task.expected_hole_count is not None:
        predicates.append(_contract.Predicate(
            key="hole_count",
            target=int(task.expected_hole_count),
            kind=_contract.PredicateKind.MEASURED,
            note="exact expected hole count from the task's tolerances",
        ))
    for key, why in (("volume_mm3", "task states no volume"),
                     ("genus", "task states hole count, not through-hole topology")):
        predicates.append(_contract.Predicate(
            key=key,
            target=None,
            tolerance=0.0,
            kind=_contract.PredicateKind.MEASURED,
            unbound=True,
            note=why,
        ))
    return _contract.MeasuredGeometricContract(
        part_id="agentscad_%s" % task.id,
        predicates=tuple(predicates),
        intent=task.prompt,
    )


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "MIT", m.license
    problems = m.verify_vendored()
    assert not problems, "vendored data drifted: %s" % problems

    tasks = load()
    assert len(tasks) == EXPECTED_TASKS, (
        "expected %d tasks, loaded %d" % (EXPECTED_TASKS, len(tasks)))
    per_tier = {tier: sum(1 for t in tasks if t.difficulty == tier)
                for tier in _TIERS}
    assert per_tier == EXPECTED_PER_TIER, per_tier
    ids = {t.id for t in tasks}
    assert len(ids) == len(tasks), "duplicate task ids"
    for t in tasks:
        assert t.prompt.strip(), "task %s has an empty prompt" % t.id
        assert len(t.expected_bbox) == 3, "task %s bbox is not a triplet" % t.id
        assert all(v > 0 for v in t.expected_bbox), t.id
        assert t.tol_bbox_mm > 0, "task %s has no bbox tolerance" % t.id
        assert t.expected_hole_count is not None and t.expected_hole_count >= 0
        assert t.required_features, "task %s has no required features" % t.id

        mgc = contract_for_task(t)
        assert len(mgc.measured()) == 4, (
            "task %s: expected 4 bound predicates" % t.id)
        assert len(mgc.unbound()) == 2, t.id
        assert mgc.digest() == contract_for_task(t).digest(), (
            "task %s: contract digest not deterministic" % t.id)
        # A measurement copied off the task's own expectations must satisfy
        # its contract; the same measurement off-band on one axis must not.
        good = {"bbox_x_mm": t.expected_bbox[0], "bbox_y_mm": t.expected_bbox[1],
                "bbox_z_mm": t.expected_bbox[2], "hole_count": t.expected_hole_count}
        assert _contract.check(mgc, good).satisfied, t.id
        bad = dict(good, bbox_x_mm=t.expected_bbox[0] + t.tol_bbox_mm + 1.0)
        assert not _contract.check(mgc, bad).satisfied, t.id

    assert len(briefs()) == EXPECTED_TASKS
    print("SELFCHECK OK: %d tasks (%s), all vendored SHAs verified, all "
          "contracts compile and discriminate"
          % (len(tasks), ", ".join("%s=%d" % kv for kv in per_tier.items())))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="AgentSCAD machine-checkable task loader (14 tasks, MIT, "
                    "vendored) with per-task Measured Geometric Contracts.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate vendored hashes, counts, and that every "
                             "task's contract compiles and discriminates.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _selfcheck()
    except AssertionError as exc:
        print("SELFCHECK FAILED: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
