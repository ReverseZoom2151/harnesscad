"""CADSpot leaderboard -- a scoreboard over the CAD GUI-grounding benchmark.

:mod:`harnesscad.eval.grounding.cadspot` scores ONE predictor and returns a
:class:`~harnesscad.eval.grounding.cadspot.Report`: an accuracy per region plus an
overall. This module collects MANY such reports -- one per submission -- and turns
them into a ranked table, per region and overall.

WHY THE RANKING IS PER REGION, NOT A SINGLE NUMBER
--------------------------------------------------
ScreenSpot publishes one accuracy because its surface is uniform: everything on the
page has an accessibility box. A CAD session does not work like that. The toolbar,
the dialog and the feature tree are chrome with exact boxes (a re-run of what
OS-Atlas already did for Windows), and the 3D viewport is the region that cannot be
scraped and that no other grounding dataset has at all. Averaging those into one
score would let a model that is strong on free chrome and blind in the viewport
outrank a model that can actually ground where the geometry is. So the board ranks
each region on its own, and marks the viewport for what it is.

THE VIEWPORT COLUMN IS SCORED ON THE HONEST METRIC WHEN IT EXISTS
-----------------------------------------------------------------
For chrome, ``point_in_bbox`` is exactly right: a button IS a rectangle. For the
viewport it is a lenient PROXY (an entity's projected box includes pixels that
belong to whatever occludes it), and the metric the viewport deserves is
``selects_expected`` -- put the predicted pixel to the application's own picker and
ask what it selected. That number needs a live FreeCAD, so a submission scored
offline will not carry it. The board ranks the viewport on ``selects_expected``
where a submission has it and falls back to the proxy where it does not, and it
LABELS which one each row used, because a proxy score and an adjudicated score are
not the same claim and must never share a column silently.

NO MODEL IS RUN HERE
--------------------
A submission is a finished :class:`Report` (or its JSON). :func:`from_predictors`
is a convenience that scores callables -- the baselines (centre, random,
projection-oracle) and any predictor already in hand -- by delegating to
:func:`cadspot.evaluate`. It runs whatever callable it is handed; it never trains
or invokes a model of its own.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.eval.grounding import cadspot

__all__ = [
    "REGIONS", "PROXY_METRIC", "ADJUDICATED_METRIC", "Entry", "RankRow",
    "region_ranking", "overall_ranking", "viewport_ranking", "Board",
    "from_predictors", "from_reports", "load_entries", "render", "main",
]

#: The four surfaces, ranked in the order a reader should read them: the chrome the
#: field can already ground, then the viewport that is the contribution.
REGIONS: Tuple[str, ...] = cadspot.REGIONS

#: The lenient proxy every submission can carry offline. ScreenSpot's whole metric.
PROXY_METRIC = "point_in_bbox"

#: The honest viewport metric; present only when a live app adjudicated the run.
ADJUDICATED_METRIC = "selects_expected"


@dataclass
class Entry:
    """One submission, normalised to the numbers a board needs and nothing else.

    Built from a :class:`cadspot.Report` or its serialised ``to_dict`` form, so the
    board can rank a run scored in this process or one loaded from a file the same
    way. ``regions`` maps a region name to that region's score dict (``n``,
    ``point_in_bbox``, ``selects_expected`` when adjudicated, ...).
    """

    name: str
    overall: Dict[str, Any] = field(default_factory=dict)
    regions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_report(cls, report: "cadspot.Report") -> "Entry":
        d = report.to_dict()
        return cls(name=d.get("predictor", "") or "predictor",
                   overall=dict(d.get("overall", {})),
                   regions={k: dict(v) for k, v in (d.get("regions") or {}).items()})

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Entry":
        return cls(name=d.get("predictor", d.get("name", "")) or "predictor",
                   overall=dict(d.get("overall", {})),
                   regions={k: dict(v) for k, v in (d.get("regions") or {}).items()})

    def region_metric(self, region: str) -> Tuple[Optional[float], str, int]:
        """The score to rank ``region`` on, plus which metric it is and its ``n``.

        For the viewport this prefers the adjudicated metric and falls back to the
        proxy; for chrome the proxy IS the exact metric, so it is used directly.
        Returns ``(value, metric_name, n)`` with ``value = None`` when the region
        is absent from this submission.
        """
        r = self.regions.get(region)
        if not r:
            return (None, "", 0)
        n = int(r.get("n", 0) or 0)
        if region == "viewport" and r.get(ADJUDICATED_METRIC) is not None:
            return (float(r[ADJUDICATED_METRIC]), ADJUDICATED_METRIC, n)
        val = r.get(PROXY_METRIC)
        return (None if val is None else float(val), PROXY_METRIC, n)

    def to_dict(self) -> dict:
        return {"name": self.name, "overall": dict(self.overall),
                "regions": {k: dict(v) for k, v in self.regions.items()}}


@dataclass
class RankRow:
    """One row of a ranked table."""

    rank: int
    name: str
    value: Optional[float]
    metric: str
    n: int

    def to_dict(self) -> dict:
        return {"rank": self.rank, "name": self.name, "value": self.value,
                "metric": self.metric, "n": self.n}


def _sorted_rows(scored: Sequence[Tuple[str, Optional[float], str, int]]
                 ) -> List[RankRow]:
    """Rank descending by value; a missing score sorts last; ties break by name.

    Deterministic: scored highest first, an absent score always last, and equal
    scores read alphabetically so a board is byte-identical run to run.
    """
    ordered = sorted(
        scored,
        key=lambda it: (-(1 if it[1] is not None else 0), -(it[1] or 0.0), it[0]))
    return [RankRow(rank=i + 1, name=n, value=v, metric=m, n=cnt)
            for i, (n, v, m, cnt) in enumerate(ordered)]


def region_ranking(entries: Sequence[Entry], region: str) -> List[RankRow]:
    """Rank submissions on one region, using that region's honest metric."""
    scored: List[Tuple[str, Optional[float], str, int]] = []
    for e in entries:
        val, metric, n = e.region_metric(region)
        scored.append((e.name, val, metric or PROXY_METRIC, n))
    return _sorted_rows(scored)


def viewport_ranking(entries: Sequence[Entry]) -> List[RankRow]:
    """The contribution's own board -- the viewport, ranked on ``selects_expected``
    where a live app scored it and on the labelled proxy where it did not."""
    return region_ranking(entries, "viewport")


def overall_ranking(entries: Sequence[Entry]) -> List[RankRow]:
    """Rank on the overall ``point_in_bbox`` across every region a submission has."""
    scored: List[Tuple[str, Optional[float], str, int]] = []
    for e in entries:
        val = e.overall.get(PROXY_METRIC)
        n = int(e.overall.get("n", 0) or 0)
        scored.append((e.name, None if val is None else float(val),
                       PROXY_METRIC, n))
    return _sorted_rows(scored)


@dataclass
class Board:
    """A whole leaderboard: the collected entries and every ranking off them."""

    entries: List[Entry] = field(default_factory=list)

    def add(self, entry: Entry) -> "Board":
        self.entries.append(entry)
        return self

    def add_report(self, report: "cadspot.Report") -> "Board":
        return self.add(Entry.from_report(report))

    def overall(self) -> List[RankRow]:
        return overall_ranking(self.entries)

    def by_region(self) -> Dict[str, List[RankRow]]:
        return {r: region_ranking(self.entries, r) for r in REGIONS}

    def to_dict(self) -> dict:
        return {"overall": [row.to_dict() for row in self.overall()],
                "regions": {r: [row.to_dict() for row in rows]
                            for r, rows in self.by_region().items()},
                "entries": [e.to_dict() for e in self.entries]}

    def render(self) -> str:
        return render(self.entries)


def from_reports(reports: Sequence["cadspot.Report"]) -> Board:
    """Build a board from reports already scored in this process."""
    board = Board()
    for r in reports:
        board.add_report(r)
    return board


def from_predictors(targets: Sequence["cadspot.Target"],
                    predictors: Sequence[Tuple[str, "cadspot.Predictor"]],
                    root: str = "",
                    adjudicator: Optional["cadspot.Adjudicator"] = None) -> Board:
    """Score named predictors against ``targets`` and rank them.

    A convenience over :func:`cadspot.evaluate` for the offline baselines and any
    predictor already in hand. ``adjudicator``, if given, is the live-FreeCAD
    picker that yields the honest ``selects_expected`` for the viewport. This runs
    whatever callables it is handed; it starts no model of its own.
    """
    board = Board()
    for name, fn in predictors:
        report = cadspot.evaluate(targets, fn, name=name, root=root,
                                  adjudicator=adjudicator)
        board.add_report(report)
    return board


def load_entries(*paths: str) -> List[Entry]:
    """Load submissions from JSON report files (each a ``Report.to_dict``).

    A file may hold a single report object or a JSON list of them. Names come from
    the ``predictor`` field; a file basename is used only when that field is empty.
    """
    entries: List[Entry] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        items = payload if isinstance(payload, list) else [payload]
        stem = os.path.splitext(os.path.basename(path))[0]
        for item in items:
            entry = Entry.from_dict(item)
            if not entry.name or entry.name == "predictor":
                entry.name = stem
            entries.append(entry)
    return entries


def _fmt(value: Optional[float]) -> str:
    return "   -  " if value is None else "%6.4f" % value


def render(entries: Sequence[Entry]) -> str:
    """The full leaderboard as text: overall, then one table per region."""
    lines: List[str] = []
    lines.append("CADSPOT LEADERBOARD -- CAD GUI grounding, split by surface")
    lines.append("=" * 72)
    lines.append("Chrome (toolbar/dialog/tree) is scored on point_in_bbox, which is")
    lines.append("exact there. The viewport is scored on selects_expected where a live")
    lines.append("app adjudicated it and on the labelled point_in_bbox PROXY otherwise.")
    lines.append("")

    lines.append("OVERALL (point_in_bbox, all regions pooled)")
    lines.append("-" * 72)
    lines.append("%-4s %-28s %8s %8s  %s" % ("#", "submission", "score", "n", "metric"))
    for row in overall_ranking(entries):
        lines.append("%-4d %-28s %8s %8d  %s"
                     % (row.rank, row.name[:28], _fmt(row.value), row.n, row.metric))
    lines.append("")

    for region in REGIONS:
        tag = "  <-- the contribution; no other grounding board has it" \
            if region == "viewport" else ""
        lines.append("REGION: %s%s" % (region, tag))
        lines.append("-" * 72)
        lines.append("%-4s %-28s %8s %8s  %s"
                     % ("#", "submission", "score", "n", "metric"))
        rows = region_ranking(entries, region)
        if not any(r.n for r in rows):
            lines.append("     (no submission carries this region)")
        else:
            for row in rows:
                lines.append("%-4d %-28s %8s %8d  %s"
                             % (row.rank, row.name[:28], _fmt(row.value),
                                row.n, row.metric))
        lines.append("")
    lines.append("A viewport row marked point_in_bbox is a PROXY: it was scored")
    lines.append("offline and its rank is provisional until a live app adjudicates it.")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Rank CADSpot grounding submissions (report JSON files).")
    ap.add_argument("reports", nargs="*",
                    help="Report.to_dict JSON files, one or more per submission.")
    ap.add_argument("--json", action="store_true",
                    help="emit the board as JSON instead of a text table.")
    args = ap.parse_args(list(argv) if argv is not None else None)
    if not args.reports:
        print("no report files given; nothing to rank.")
        return 2
    entries = load_entries(*args.reports)
    board = Board(entries=list(entries))
    if args.json:
        print(json.dumps(board.to_dict(), indent=2, sort_keys=True))
    else:
        print(board.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
