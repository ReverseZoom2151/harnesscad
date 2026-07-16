"""Graph-CAD's CADBench: 700 rubric-annotated text-to-CAD tasks as harness shapes.

Source: Graph-CAD (resources/cad_repos/Graph-CAD-main/Graph-CAD-main),
``CADBench.jsonl``. Each line is one task: an ``instruction`` (the prompt) and
a ``criteria`` tree -- sections ("Object Attributes", "Spatial Understanding
and Structure", "User Instruction Understanding and Execution") containing
dimensions ("Shape accuracy", "Size", "Proportion", ...) containing individual
checkable criterion sentences. It is the largest judge-calibration + brief
corpus in the resources tree.

LICENSE: the Graph-CAD repository ships NO LICENSE file (verified 2026-07-16:
no LICENSE/COPYING anywhere in the checkout, no license grant in README.md).
No license means no redistribution right, so NOTHING is vendored.
``graphcad/MANIFEST.json`` records the file's resources-relative path, SHA-256
and byte count; this loader resolves it against ``resources/`` at run time and
DEGRADES CLEANLY (an empty task list plus a stated reason) when the checkout
is absent. Attribution: Graph-CAD authors.

Mapping onto harness shapes:

* :func:`briefs` -- each task as an :class:`~harnesscad.eval.bench.imports.ImportedBrief`
  (``text`` = the instruction; CADBench states no bbox/volume/genus, so those
  stay ``None`` and the contract grader's unbound-predicate path applies).
* :func:`rubric` -- a task's criteria tree flattened into the checklist shape
  the deterministic judges consume: ordered rows of (section, dimension,
  criterion), each row independently satisfiable. This is the calibration
  corpus for ``eval/judge``-style graders: 700 tasks x ~10 rows of
  human-authored, per-task grading language.

Stdlib only, deterministic, no kernel, no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from harnesscad.eval.bench.imports import ImportedBrief, Manifest, load_manifest

__all__ = [
    "SOURCE_REPO",
    "LICENSE",
    "EXPECTED_TASKS",
    "RubricRow",
    "GraphCadTask",
    "manifest",
    "corpus_path",
    "load",
    "briefs",
    "rubric",
    "main",
]

_SOURCE = "graphcad"
SOURCE_REPO = "Graph-CAD"
LICENSE = "UNLICENSED"

#: Manifested line count of CADBench.jsonl; the selfcheck re-counts it.
EXPECTED_TASKS = 700


@dataclass(frozen=True)
class RubricRow:
    """One judge-checkable criterion sentence, with its position in the tree."""

    section: str      # e.g. "Object Attributes"
    dimension: str    # e.g. "Shape accuracy"
    criterion: str    # the human-authored sentence a judge marks pass/fail

    def to_dict(self) -> dict:
        return {"section": self.section, "dimension": self.dimension,
                "criterion": self.criterion, "satisfied": None}


@dataclass(frozen=True)
class GraphCadTask:
    """One CADBench line: prompt + its per-task grading rubric."""

    id: str
    name: str
    instruction: str
    rows: Tuple[RubricRow, ...]
    color_brightness: str = ""
    type: str = ""

    @property
    def sections(self) -> Tuple[str, ...]:
        seen: List[str] = []
        for r in self.rows:
            if r.section not in seen:
                seen.append(r.section)
        return tuple(seen)


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def corpus_path() -> Optional[Path]:
    """``CADBench.jsonl`` resolved from resources/, or ``None`` when absent."""
    m = manifest()
    e = m.by_name("CADBench.jsonl")
    return m.resolve(e) if e is not None else None


def _flatten_criteria(criteria: dict) -> Tuple[RubricRow, ...]:
    rows: List[RubricRow] = []
    for section, dims in criteria.items():
        if not isinstance(dims, dict):
            continue
        for dimension, sentences in dims.items():
            if not isinstance(sentences, list):
                continue
            for sentence in sentences:
                text = str(sentence).strip()
                if text:
                    rows.append(RubricRow(str(section), str(dimension), text))
    return tuple(rows)


def load() -> List[GraphCadTask]:
    """Every CADBench task, or ``[]`` when resources/ is not checked out.

    Callers must treat an empty list as "corpus not present", never as
    "corpus passed" -- the same rule every manifest-mode loader follows.
    """
    path = corpus_path()
    if path is None:
        return []
    tasks: List[GraphCadTask] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            tasks.append(GraphCadTask(
                id=str(raw.get("id", "")),
                name=str(raw.get("name", "")),
                instruction=str(raw.get("instruction", "")),
                rows=_flatten_criteria(raw.get("criteria") or {}),
                color_brightness=str(raw.get("color brightness", "")),
                type=str(raw.get("type", "")),
            ))
    return tasks


def briefs() -> List[ImportedBrief]:
    """Each task as an ImportedBrief (no bbox/volume/genus: CADBench states none)."""
    out: List[ImportedBrief] = []
    for t in load():
        if not t.id or not t.instruction.strip():
            continue
        out.append(ImportedBrief(
            id="graphcad_%s" % t.id,
            source_repo=SOURCE_REPO,
            license=LICENSE,
            text=t.instruction,
            categories=(t.type,) if t.type else (),
            tags=("rubric_annotated",),
            note="CADBench task %r with a %d-row grading rubric; see rubric()"
                 % (t.name, len(t.rows)),
        ))
    return out


def rubric(task: GraphCadTask) -> Dict[str, object]:
    """A task's criteria tree in the judge checklist shape.

    ``rows`` is an ordered list of independently satisfiable checks with
    ``satisfied`` left ``None`` -- a grader fills the verdicts in; this module
    never does (it runs no judge and no model).
    """
    return {
        "task_id": task.id,
        "name": task.name,
        "instruction": task.instruction,
        "rows": [r.to_dict() for r in task.rows],
    }


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "UNLICENSED", m.license
    # Policy check, executable: an unlicensed repo may vendor NOTHING.
    vendored = [e.name for e in m.entries if e.vendored]
    assert not vendored, "policy violation: vendored files %s" % vendored
    for e in m.entries:
        assert e.resource, "entry %s has no resources path" % e.name
        assert len(e.sha256) == 64, "entry %s has no sha256" % e.name

    tasks = load()
    if not tasks:
        print("SELFCHECK OK: manifest valid (%d entries); resources/ absent, "
              "corpus degrades to empty as designed" % len(m.entries))
        return 0

    assert len(tasks) == EXPECTED_TASKS, (
        "expected %d tasks, loaded %d" % (EXPECTED_TASKS, len(tasks)))
    ids = {t.id for t in tasks}
    assert len(ids) == len(tasks), "duplicate task ids in CADBench.jsonl"
    for t in tasks:
        assert t.instruction.strip(), "task %s has an empty instruction" % t.id
        assert t.rows, "task %s has an empty rubric" % t.id
        r = rubric(t)
        assert len(r["rows"]) == len(t.rows)

    bs = briefs()
    assert len(bs) == EXPECTED_TASKS, "briefs dropped tasks: %d" % len(bs)
    n_rows = sum(len(t.rows) for t in tasks)
    sections = sorted({row.section for t in tasks for row in t.rows})
    print("SELFCHECK OK: %d tasks, %d rubric rows, sections=%s"
          % (len(tasks), n_rows, sections))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Graph-CAD CADBench loader (700 rubric-annotated tasks; "
                    "UNLICENSED source: manifest + resources-path, nothing "
                    "vendored).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate the manifest, counts and rubric shapes; "
                             "degrades cleanly when resources/ is absent.")
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
