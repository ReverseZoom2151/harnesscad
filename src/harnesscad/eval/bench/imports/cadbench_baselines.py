"""CADBench's committed baseline leaderboard values, as typed comparator rows.

Source: CADBench (resources/cad_repos/CADBench-main/CADBench-main),
``tested_models/{model}/{modality}/r1/bench{A,B,E,F,M,O}_results/``. CADBench is
the multimodal image/mesh -> CadQuery *reconstruction* benchmark of Doris et al.
2026 (arXiv:2605.10873, HF ``DeCoDELab/CADBench``). It ships, committed to the
repo, the AGGREGATE scores of ~13 named models across three modality families,
six benches and per-difficulty -- the exact numbers behind its public
leaderboard. The harness had baselines for *other* benchmarks but NONE for this
multimodal reconstruction task; this loader imports those reference values so a
harness run CAN be placed beside them.

WHAT IS VENDORED, AND WHY IT IS SAFE (MIT, not dataset content)
--------------------------------------------------------------
CADBench's ``LICENSE`` is MIT (Copyright (c) 2026 Annie Doris). The two files
vendored per (model, modality, bench) are computed OUTPUTS of that MIT code:

* ``bench*_metrics.json``           -- the overall {Mean, Median, Std, Adjusted*,
  VSR, Timeout Rate, Code Metrics} block for Aligned/Naive IoU, Chamfer and
  Surface-IoU;
* ``bench*_per_label_metrics.json`` -- the same statistics split easy/medium/hard
  (``success_only`` over executed solids, ``adjusted`` over all tasks).

Both are pure aggregate statistics (mean/median/std/count) with NO per-object
provenance -- no file ids, no STEP/STL/image/mesh content. They are eval numbers,
not benchmark task data, so the MIT license covers redistribution. The benchmark
TASK DATA (DeepCAD / Fusion360 / ABC / Objaverse / MCB derived modalities) is
NOT vendored: it is HuggingFace-hosted, absent from the checkout, and under a
mixed / non-commercial license (repo ``DATASET_LICENSE.md``). Only the canonical
first run (``r1``, or the runless ``cadfit`` variant) is vendored -- the primary
leaderboard run; repeat runs r2/r3 are left in resources only.

Discipline is the sibling-loader idiom (``agentscad_tasks``, ``cadjudge_prompts``):
a ``cadbench/MANIFEST.json`` records every vendored file's resources-relative
source path, SHA-256, byte count and role; ``cadbench/LICENSE-NOTICE`` names the
license and attribution; ``--selfcheck`` verifies the SHAs and degrades to empty
when neither the vendored copy nor ``resources/`` is present.

Mapping onto harness shapes
---------------------------
* :func:`rows` -- every baseline as a typed :class:`BaselineRow`
  ``(model, modality, bench, difficulty) -> metric dict``. ``metrics.json``
  yields one ``difficulty="overall"`` row; ``per_label`` yields easy/medium/hard.
* :func:`to_standing` -- one row adapted to the harness's own
  :class:`~harnesscad.eval.leaderboard.hardcorpus_board.Standing` row type, so a
  run can be ranked against these external baselines (see the mapping note on
  that function). This is OPT-IN: importing this module changes no existing
  leaderboard behaviour.
* :func:`to_scorecard_row` -- the same row as the ``{model, cd, ir, iou}`` dict
  that ``eval.bench.protocols.tiered_leaderboard.rank_leaderboard`` consumes (a
  metric-faithful CD/IR/IoU view; that module is dict-based, not row-typed).

Stdlib only, deterministic, ASCII, no kernel, no model, no task data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from harnesscad.eval.bench.imports import Manifest, load_manifest

__all__ = [
    "SOURCE_REPO",
    "LICENSE",
    "ATTRIBUTION",
    "BENCHES",
    "DIFFICULTIES",
    "BENCH_TASK_COUNT",
    "EXPECTED_METRICS_FILES",
    "EXPECTED_PER_LABEL_FILES",
    "EXPECTED_COMBOS",
    "BaselineRow",
    "manifest",
    "rows",
    "to_standing",
    "to_standings",
    "to_scorecard_row",
    "main",
]

_SOURCE = "cadbench"
SOURCE_REPO = "CADBench"
LICENSE = "MIT"
ATTRIBUTION = ("CADBench / Doris et al., arXiv:2605.10873 / DeCoDELab "
               "(MIT License, Copyright (c) 2026 Annie Doris)")

#: The six benchmark subsets, in the repo's letter order.
BENCHES: Tuple[str, ...] = ("benchA", "benchB", "benchE", "benchF", "benchM",
                            "benchO")

#: The four benches banded easy/medium/hard by geometric complexity (extrude /
#: op counts). benchM (MCB) and benchO (Objaverse) are NOT banded -- their
#: ``per_label`` file carries a single unlabelled whole-bench block, so they
#: contribute an ``overall`` row only and no per-difficulty rows are invented.
BANDED_BENCHES: Tuple[str, ...] = ("benchA", "benchB", "benchE", "benchF")

#: Difficulty keys a row can carry. ``overall`` comes from ``metrics.json``;
#: easy/medium/hard from a BANDED bench's ``per_label`` file.
DIFFICULTIES: Tuple[str, ...] = ("overall", "easy", "medium", "hard")
_PER_LABEL_DIFF: Tuple[str, ...] = ("easy", "medium", "hard")

#: Documented tasks per bench (README / paper: "3000 tasks each"). Used only as
#: the sample count for an ``overall`` row, whose ``metrics.json`` does not store
#: a count; per-difficulty rows carry their own ``adjusted`` count instead.
BENCH_TASK_COUNT = 3000

EXPECTED_METRICS_FILES = 186        # 31 (model, modality) combos x 6 benches
EXPECTED_PER_LABEL_FILES = 186      # one per (model, modality, bench)
EXPECTED_COMBOS = 31                # distinct (model, modality) baselines
#: banded per_label files (A/B/E/F): 31 combos x 4 benches -> easy/medium/hard.
EXPECTED_BANDED_PER_LABEL = 124
#: overall (186) + banded per_label (124) x {easy, medium, hard}.
EXPECTED_ROWS = EXPECTED_METRICS_FILES + EXPECTED_BANDED_PER_LABEL * 3

_RUN_RE = re.compile(r"^r\d+$")
#: metric-name -> which primary mean lives where in each schema.
_PRIMARY = "Aligned IoU"
_CHAMFER = "Aligned Chamfer Distance"


@dataclass(frozen=True)
class BaselineRow:
    """One CADBench baseline: ``(model, modality, bench, difficulty)`` + metrics.

    ``metrics`` is the raw aggregate block exactly as CADBench committed it (the
    schema differs between an ``overall`` row and a per-difficulty row, so the
    accessors below normalise the two). ``run`` is the source run ("r1", or ""
    for the runless ``cadfit`` variant). ``modality`` is "" when the model's
    path names no modality dir (the ``*_ourimages`` models, whose modality is
    already encoded in the model name).
    """

    model: str
    modality: str
    bench: str
    difficulty: str
    run: str
    role: str                 # "metrics" (overall) or "per_label"
    metrics: Mapping[str, Any]

    @property
    def key(self) -> Tuple[str, str, str, str]:
        return (self.model, self.modality, self.bench, self.difficulty)

    @property
    def label(self) -> str:
        """A stable, human-readable id for this baseline row."""
        mod = self.modality or "-"
        return "cadbench:%s/%s:%s:%s" % (self.model, mod, self.bench,
                                         self.difficulty)

    # -- metric accessors, schema-normalised ------------------------------- #
    def _pair(self, metric: str, stat: str, variant: str) -> Optional[float]:
        m = self.metrics
        if self.difficulty == "overall":
            block = m.get(stat)  # e.g. m["Mean"]["Aligned IoU"]
            if isinstance(block, Mapping) and metric in block:
                return _as_float(block[metric])
            return None
        node = m.get(metric)     # e.g. m["Aligned IoU"]["success_only"]["mean"]
        if isinstance(node, Mapping):
            sub = node.get(variant)
            if isinstance(sub, Mapping):
                return _as_float(sub.get(stat.lower()))
        return None

    def aligned_iou_mean(self) -> Optional[float]:
        """Mean Aligned IoU over executed solids (the ``success_only`` set)."""
        return self._pair(_PRIMARY, "Mean" if self.difficulty == "overall"
                          else "mean", "success_only")

    def aligned_chamfer_mean(self) -> Optional[float]:
        """Mean Aligned Chamfer Distance over executed solids (lower better)."""
        return self._pair(_CHAMFER, "Mean" if self.difficulty == "overall"
                          else "mean", "success_only")

    def vsr(self) -> Optional[float]:
        """Valid-solid rate in percent (0-100): the field's validity metric.

        ``overall`` reads the committed ``VSR``; a per-difficulty row derives it
        from the executed vs. total counts (``success_only`` / ``adjusted``) of
        the primary metric, which is exactly how CADBench defines it.
        """
        if self.difficulty == "overall":
            return _as_float(self.metrics.get("VSR"))
        succ = self._count(_PRIMARY, "success_only")
        tot = self._count(_PRIMARY, "adjusted")
        if succ is None or not tot:
            return None
        return 100.0 * succ / tot

    def _count(self, metric: str, variant: str) -> Optional[int]:
        node = self.metrics.get(metric)
        if isinstance(node, Mapping):
            sub = node.get(variant)
            if isinstance(sub, Mapping) and sub.get("count") is not None:
                return int(sub["count"])
        return None

    def sample_count(self) -> int:
        """Number of tasks behind this row.

        Per-difficulty: the ``adjusted`` count (all tasks at that difficulty).
        Overall: the documented per-bench task count (:data:`BENCH_TASK_COUNT`),
        since ``metrics.json`` stores no count of its own.
        """
        if self.difficulty == "overall":
            return BENCH_TASK_COUNT
        cnt = self._count(_PRIMARY, "adjusted")
        return int(cnt) if cnt is not None else BENCH_TASK_COUNT


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def _parse_resource(resource: str) -> Optional[Tuple[str, str, str, str]]:
    """(model, modality, run, bench) from a resources-relative metrics path.

    ``.../tested_models/{model}/[modality/][run/]bench{X}_results/bench{X}_*.json``
    -- ``run`` is any ``r\\d+`` component (else ""), ``modality`` is the joined
    remaining middle components (else ""), ``bench`` is the ``bench{X}`` prefix.
    """
    parts = resource.replace("\\", "/").split("/")
    try:
        anchor = parts.index("tested_models")
    except ValueError:
        return None
    tail = parts[anchor + 1:]
    if len(tail) < 3:
        return None
    model = tail[0]
    fname = tail[-1]
    m = re.match(r"^(bench[A-Z])_", fname)
    if not m:
        return None
    bench = m.group(1)
    middle = tail[1:-2]  # drop model and (benchX_results, benchX_*.json)
    run = next((p for p in middle if _RUN_RE.match(p)), "")
    modality = "/".join(p for p in middle if not _RUN_RE.match(p))
    return (model, modality, run, bench)


def rows() -> List[BaselineRow]:
    """Every baseline row from the vendored copies (resources/ as fallback).

    Degrades to ``[]`` when neither the vendored data nor ``resources/`` is
    present (the documented not-present contract): an empty list ALWAYS means
    "not present", never "no baselines". Deterministic order:
    (model, modality, bench, difficulty).
    """
    try:
        m = manifest()
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return []
    out: List[BaselineRow] = []
    for e in m.entries:
        if e.role not in ("metrics", "per_label"):
            continue
        parsed = _parse_resource(e.resource or "")
        if parsed is None:
            continue
        model, modality, run, bench = parsed
        path = m.resolve(e)
        if path is None:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        if e.role == "metrics":
            out.append(BaselineRow(model, modality, bench, "overall", run,
                                   e.role, raw))
        else:
            for diff in _PER_LABEL_DIFF:
                block = raw.get(diff)
                if isinstance(block, dict):
                    out.append(BaselineRow(model, modality, bench, diff, run,
                                           e.role, block))
    out.sort(key=lambda r: (r.model, r.modality, BENCHES.index(r.bench)
                            if r.bench in BENCHES else len(BENCHES),
                            DIFFICULTIES.index(r.difficulty)
                            if r.difficulty in DIFFICULTIES else len(DIFFICULTIES)))
    return out


# --------------------------------------------------------------------------- #
# Leaderboard adapters.
#
# CHOSEN ROW TYPE: eval.leaderboard.hardcorpus_board.Standing.
#
# The hard-corpus board is the harness's GENERAL submission scoreboard and the
# only one of the three whose row is a typed class fed by external numbers: it
# contrasts what "the field publishes" (its ``weak_rate`` -- a valid solid +
# IoU) against what the harness's measured oracle solves. A CADBench baseline IS
# a field-published number, so it maps natively onto the weak/IoU columns:
#   * weak_passed <- round(VSR% * n)     (VSR is the valid-solid rate -- exactly
#                                          the board's "valid solid" weak metric)
#   * mean_iou    <- Aligned IoU mean    (the field's geometry headline)
#   * built       <- the valid count      (a solid was produced)
# The measured-oracle lens is GENUINELY ABSENT for an external model -- we never
# ran our point-membership oracle on it -- so ``oracle_solved`` stays 0 and the
# board shows its oracle column as 0.000: honest "field number, oracle not run".
# A harness run carrying a real oracle_rate therefore ranks above these on the
# board's oracle bar, which is the correct relationship; the baselines' real
# signal stays visible in the weak/IoU columns the board prints beside it.
#
# cadspot_board.Entry is region / point_in_bbox shaped (GUI grounding) -- no fit
# for aggregate reconstruction metrics. tiered_leaderboard's CD/IR/IoU map
# cleanly but it is dict-based with no typed row, so it is offered as the
# secondary ``to_scorecard_row`` adapter, not the primary one. No leaderboard
# module is edited: both adapters import/produce those modules' shapes read-only.
# --------------------------------------------------------------------------- #

def to_standing(row: BaselineRow):
    """Adapt one baseline to a hard-corpus ``Standing`` (field-side comparator).

    See the block comment above for the column mapping and why ``oracle_solved``
    is 0 (CADBench provides no measured-oracle lens). Imports ``Standing``
    lazily and read-only, so this module has no import-time leaderboard
    dependency and changes no existing board behaviour.
    """
    from harnesscad.eval.leaderboard.hardcorpus_board import Standing

    n = int(row.sample_count())
    vsr = row.vsr()
    valid = int(round((vsr / 100.0) * n)) if vsr is not None else 0
    valid = max(0, min(valid, n))
    return Standing(
        name=row.label,
        n=n,
        built=valid,
        oracle_solved=0,        # external baseline: our oracle was never run
        weak_passed=valid,      # VSR -> the field's valid-solid weak metric
        field_fooled=0,
        mean_iou=row.aligned_iou_mean(),
        failed={})


def to_standings(baselines: Optional[List[BaselineRow]] = None,
                 difficulty: str = "overall") -> List:
    """``Standing`` rows for every baseline at one difficulty (default overall).

    Convenience for dropping the whole external field beside a run on the
    hard-corpus board. ``difficulty=""`` keeps every difficulty.
    """
    src = rows() if baselines is None else baselines
    return [to_standing(r) for r in src
            if not difficulty or r.difficulty == difficulty]


def to_scorecard_row(row: BaselineRow) -> Dict[str, Any]:
    """The metric-faithful ``{model, cd, ir, iou}`` view for tiered_leaderboard.

    Maps CADBench's aligned metrics onto Text2CAD-Bench's CD/IR/IoU vocabulary
    (``eval.bench.protocols.tiered_leaderboard.rank_leaderboard``): IR = 100 -
    VSR (invalidity rate), CD = Aligned Chamfer mean, IoU = Aligned IoU mean.
    That module ranks plain dicts, so this returns a dict rather than a typed row.
    """
    vsr = row.vsr()
    return {
        "model": row.label,
        "cd": row.aligned_chamfer_mean(),
        "ir": None if vsr is None else max(0.0, 100.0 - vsr),
        "iou": row.aligned_iou_mean(),
    }


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "MIT", m.license
    assert m.source_repo == SOURCE_REPO, m.source_repo
    problems = m.verify_vendored()
    assert not problems, "vendored data drifted: %s" % problems[:5]

    metrics_entries = m.by_role("metrics")
    per_label_entries = m.by_role("per_label")
    assert len(metrics_entries) == EXPECTED_METRICS_FILES, len(metrics_entries)
    assert len(per_label_entries) == EXPECTED_PER_LABEL_FILES, len(per_label_entries)
    # Every vendored file is a JSON metrics output, never task/dataset content.
    for e in m.entries:
        assert (e.vendored or "").endswith(".json"), e.name
        assert e.role in ("metrics", "per_label"), e.role
        assert "tested_models" in (e.resource or ""), e.resource

    all_rows = rows()
    assert len(all_rows) == EXPECTED_ROWS, (
        "expected %d rows, got %d" % (EXPECTED_ROWS, len(all_rows)))
    keys = {r.key for r in all_rows}
    assert len(keys) == len(all_rows), "duplicate (model,modality,bench,difficulty)"

    combos = {(r.model, r.modality) for r in all_rows}
    assert len(combos) == EXPECTED_COMBOS, (
        "expected %d model/modality combos, got %d" % (EXPECTED_COMBOS, len(combos)))
    # Every bench carries an overall row; only the banded benches carry
    # easy/medium/hard, and the unbanded ones carry NO per-difficulty row.
    for bench in BENCHES:
        assert any(r.bench == bench and r.difficulty == "overall"
                   for r in all_rows), "missing overall for %s" % bench
    banded = {r.bench for r in all_rows if r.difficulty in _PER_LABEL_DIFF}
    assert banded == set(BANDED_BENCHES), (
        "banded benches mismatch: %s" % sorted(banded))

    # Every row parses into a typed record with sane, present primary metrics.
    for r in all_rows:
        assert r.bench in BENCHES, r.bench
        assert r.difficulty in DIFFICULTIES, r.difficulty
        iou = r.aligned_iou_mean()
        assert iou is not None and 0.0 <= iou <= 1.0, (r.label, iou)
        vsr = r.vsr()
        assert vsr is not None and 0.0 <= vsr <= 100.0, (r.label, vsr)
        assert r.sample_count() > 0, r.label
        cd = r.aligned_chamfer_mean()
        assert cd is not None and cd >= 0.0, (r.label, cd)

    # Leaderboard adapter: a valid Standing for a sample spanning both schemas.
    from harnesscad.eval.leaderboard.hardcorpus_board import Standing, ranking
    sample = ([r for r in all_rows if r.difficulty == "overall"][:8]
              + [r for r in all_rows if r.difficulty == "hard"][:8])
    standings = [to_standing(r) for r in sample]
    assert all(isinstance(s, Standing) for s in standings)
    for s, r in zip(standings, sample):
        assert s.n > 0 and 0 <= s.weak_passed <= s.n, s.name
        assert 0.0 <= s.weak_rate <= 1.0, (s.name, s.weak_rate)
        assert s.oracle_rate == 0.0, s.name  # external baseline: oracle not run
        assert s.mean_iou == r.aligned_iou_mean()
    # A run with a real oracle_rate must outrank the field-only baselines.
    run = Standing(name="harness-run", n=100, oracle_solved=40, weak_passed=60)
    ranked = ranking([run] + standings)
    assert ranked[0].name == "harness-run", "oracle run should top field baselines"

    # tiered_leaderboard dict adapter is metric-faithful and rankable.
    from harnesscad.eval.bench.protocols.tiered_leaderboard import rank_leaderboard
    sc = [to_scorecard_row(r) for r in sample]
    ranked_iou = rank_leaderboard(sc, metric="iou")
    assert len(ranked_iou) == len(sc) and ranked_iou[0]["rank"] == 1

    # Determinism.
    assert [r.key for r in rows()] == [r.key for r in all_rows]

    print("SELFCHECK OK: %d baseline rows (%d overall + %d per-difficulty) from "
          "%d model/modality combos x %d benches, all vendored SHAs verified; "
          "hard-corpus Standing + tiered_leaderboard adapters both valid"
          % (len(all_rows),
             sum(1 for r in all_rows if r.difficulty == "overall"),
             sum(1 for r in all_rows if r.difficulty != "overall"),
             len(combos), len(BENCHES)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CADBench baseline leaderboard loader (~13 models x "
                    "modalities x 6 benches x difficulty, MIT, vendored) with "
                    "a hard-corpus Standing adapter.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate vendored hashes, that every baseline "
                             "parses into a typed row, and that the leaderboard "
                             "adapter yields valid Standings.")
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
