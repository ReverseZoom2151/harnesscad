"""Provider/model benchmark protocol with crash-safe incremental flush.

Source: Forma-OSS (resources/cad_repos/Forma-OSS-main, benchmarks/
benchmark_models.py). That tool benchmarks configured LLM provider/model
selectors in two modes -- a CONFIG-ONLY mode that validates each selector
without sending a single provider call (so the bench runs safely with no API
keys), and a LIVE mode that times real calls -- and streams every completed
job to durable JSONL and CSV files the moment it finishes (its
``BenchmarkJobSink`` with per-record flush), so a crash mid-run loses nothing.
What is ported here is exactly that protocol: the two-mode attempt record, the
selector "provider/model" key, the round/iteration structure, and the
append-per-record JSONL/CSV job log with a stable CSV header (the reference's
``model-job-results-*`` files).

Gap it fills: eval/bench has task-suite runners (bench/harness/runner.py) and
paper-metric protocols, but no provider/model-level benchmark harness -- no
way to sweep a list of model selectors, validate them keylessly, time live
attempts, and persist each result incrementally. This module supplies that
slot with the provider call INJECTED (an ``attempt(selector) -> dict``
callable), so the harness stays stdlib-only and provider-agnostic.

What it complements: :mod:`harnesscad.eval.bench.harness.runner` (which
benchmarks a SOLVER over TASKS; this benchmarks PROVIDERS over ROUNDS) and the
eval/bench registry conventions (deterministic, dependency-free).

Determinism: config-only mode touches NO clock at all -- durations are exactly
0.0 and ``completed_at`` is empty -- so its output is byte-stable. Live mode
times the injected callable with ``time.monotonic`` only (never wall clock),
and records ``completed_at`` as a monotonic offset from the run start
("t+<seconds>s"), so even the live record carries no wall-clock timestamp.
No randomness anywhere.

Stdlib only. main(argv) + --selfcheck.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

__all__ = [
    "AttemptRecord",
    "JsonlFlushWriter",
    "CsvFlushWriter",
    "validate_selector",
    "run_benchmark",
    "CSV_COLUMNS",
    "MODES",
    "STATUSES",
    "main",
]

MODES = ("config-only", "live")
STATUSES = ("ok", "error", "skipped")

# Stable CSV header, in the spirit of the reference's JOB_CSV_COLUMNS: one
# column per AttemptRecord field, ``extra`` serialized as JSON.
CSV_COLUMNS = (
    "selector",
    "round_index",
    "status",
    "duration_seconds",
    "completed_at",
    "error",
    "mode",
    "extra",
)

# An attempt callable: given "provider/model", perform one live call and
# return a result dict (stored in AttemptRecord.extra). Raising marks the
# attempt as status "error".
AttemptFn = Callable[[str], dict]


@dataclass(frozen=True)
class AttemptRecord:
    """One completed benchmark attempt (the reference's per-job record)."""

    selector: str            # "provider/model"
    round_index: int
    status: str              # ok | error | skipped
    duration_seconds: float  # 0.0 in config-only mode, monotonic delta live
    completed_at: str        # "" config-only; "t+<sec>s" monotonic offset live
    error: Optional[str] = None
    mode: str = "config-only"  # config-only | live
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_csv_row(self) -> Dict[str, object]:
        row = self.to_dict()
        row["extra"] = json.dumps(self.extra, sort_keys=True)
        row["error"] = self.error if self.error is not None else ""
        return row


def validate_selector(selector: str) -> Optional[str]:
    """Config-only selector/policy validation (no provider call).

    A valid selector is "provider/model" with both halves non-empty (the
    reference's ``LLMSelector`` key form). Returns None when valid, else the
    error message.
    """
    if not isinstance(selector, str) or not selector.strip():
        return "selector must be a non-empty string"
    parts = selector.split("/", 1)
    if len(parts) != 2:
        return "selector %r is not of the form provider/model" % selector
    provider, model = parts[0].strip(), parts[1].strip()
    if not provider:
        return "selector %r has an empty provider" % selector
    if not model:
        return "selector %r has an empty model" % selector
    return None


# --------------------------------------------------------------------------- #
# incremental flush writers (crash loses nothing: open/append/close per record)
# --------------------------------------------------------------------------- #
class JsonlFlushWriter:
    """Append each record to a JSONL file immediately, one open/close per write."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.count = 0

    def write(self, record: AttemptRecord) -> None:
        line = json.dumps(record.to_dict(), sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        self.count += 1


class CsvFlushWriter:
    """Append each record to a CSV with a stable header, per-record open/close.

    The header row is written exactly once (when the file is missing or
    empty), matching the reference's model-job-results CSV discipline.
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.count = 0

    def _needs_header(self) -> bool:
        try:
            return os.path.getsize(self.path) == 0
        except OSError:
            return True

    def write(self, record: AttemptRecord) -> None:
        needs_header = self._needs_header()
        with open(self.path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
            if needs_header:
                writer.writeheader()
            writer.writerow(record.to_csv_row())
            handle.flush()
        self.count += 1


# --------------------------------------------------------------------------- #
# the benchmark
# --------------------------------------------------------------------------- #
def _run_one(
    selector: str,
    round_index: int,
    mode: str,
    attempt: Optional[AttemptFn],
    run_start_monotonic: Optional[float],
) -> AttemptRecord:
    validation_error = validate_selector(selector)

    if mode == "config-only":
        # No provider call, no clock: durations are exactly 0.0 and the
        # record is byte-stable (safe with no keys configured).
        if validation_error:
            return AttemptRecord(
                selector=selector, round_index=round_index, status="error",
                duration_seconds=0.0, completed_at="",
                error=validation_error, mode=mode,
            )
        return AttemptRecord(
            selector=selector, round_index=round_index, status="skipped",
            duration_seconds=0.0, completed_at="",
            error=None, mode=mode,
            extra={"note": "config-only: selector validated, no call made"},
        )

    # live mode
    if validation_error:
        return AttemptRecord(
            selector=selector, round_index=round_index, status="error",
            duration_seconds=0.0, completed_at="",
            error=validation_error, mode=mode,
        )
    started = time.monotonic()
    try:
        result = attempt(selector)  # type: ignore[misc]
        status, error = "ok", None
        extra = dict(result) if isinstance(result, dict) else {"result": result}
    except Exception as exc:  # a failing provider is data, not a crash
        status, error, extra = "error", "%s: %s" % (type(exc).__name__, exc), {}
    ended = time.monotonic()
    offset = ended - (run_start_monotonic if run_start_monotonic is not None
                      else started)
    return AttemptRecord(
        selector=selector, round_index=round_index, status=status,
        duration_seconds=round(ended - started, 6),
        completed_at="t+%.6fs" % offset,
        error=error, mode=mode, extra=extra,
    )


def run_benchmark(
    selectors: Sequence[str],
    iterations: int,
    attempt: Optional[AttemptFn] = None,
    jsonl_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    mode: str = "config-only",
) -> dict:
    """Run ``iterations`` rounds over ``selectors``; flush each attempt at once.

    In "config-only" mode every valid selector is recorded as status
    "skipped" with duration 0.0 (validation only, zero provider calls). In
    "live" mode the injected ``attempt`` callable is invoked and timed with
    ``time.monotonic``. Returns a summary dict with per-selector counts and
    the mean duration over ok attempts.
    """
    if mode not in MODES:
        raise ValueError("mode must be one of %s, got %r" % (MODES, mode))
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if mode == "live" and attempt is None:
        raise ValueError("live mode requires an attempt callable")
    if not selectors:
        raise ValueError("at least one selector is required")

    writers: List[object] = []
    if jsonl_path:
        writers.append(JsonlFlushWriter(jsonl_path))
    if csv_path:
        writers.append(CsvFlushWriter(csv_path))

    run_start = time.monotonic() if mode == "live" else None
    records: List[AttemptRecord] = []
    for round_index in range(1, iterations + 1):
        for selector in selectors:
            record = _run_one(selector, round_index, mode, attempt, run_start)
            records.append(record)
            for writer in writers:
                writer.write(record)  # type: ignore[attr-defined]

    per_selector: Dict[str, dict] = {}
    for selector in selectors:
        mine = [r for r in records if r.selector == selector]
        ok_durations = [r.duration_seconds for r in mine if r.status == "ok"]
        per_selector[selector] = {
            "attempts": len(mine),
            "ok": sum(1 for r in mine if r.status == "ok"),
            "error": sum(1 for r in mine if r.status == "error"),
            "skipped": sum(1 for r in mine if r.status == "skipped"),
            "mean_ok_duration_seconds": (
                round(sum(ok_durations) / len(ok_durations), 6)
                if ok_durations else 0.0
            ),
            "errors": sorted({r.error for r in mine if r.error})[:5],
        }

    return {
        "mode": mode,
        "iterations": iterations,
        "selectors": list(selectors),
        "total_attempts": len(records),
        "ok": sum(1 for r in records if r.status == "ok"),
        "error": sum(1 for r in records if r.status == "error"),
        "skipped": sum(1 for r in records if r.status == "skipped"),
        "per_selector": per_selector,
        "jsonl_path": jsonl_path,
        "csv_path": csv_path,
    }


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #
def _count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _selfcheck() -> int:
    tmp_dir = tempfile.mkdtemp(prefix="harnesscad-provider-bench-")
    jsonl_path = os.path.join(tmp_dir, "job-results.jsonl")
    csv_path = os.path.join(tmp_dir, "job-results.csv")
    selectors = ["fakeprov/fake-model-a", "fakeprov/fake-model-b"]

    # 1) config-only: two selectors x two iterations, no keys, no clock.
    summary = run_benchmark(
        selectors, iterations=2, jsonl_path=jsonl_path, csv_path=csv_path,
        mode="config-only",
    )
    print("config-only summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    assert summary["total_attempts"] == 4
    assert summary["skipped"] == 4 and summary["ok"] == 0
    for stats in summary["per_selector"].values():
        assert stats["attempts"] == 2 and stats["skipped"] == 2
        assert stats["mean_ok_duration_seconds"] == 0.0
    jsonl_lines = _count_lines(jsonl_path)
    csv_lines = _count_lines(csv_path)
    print("flushed: jsonl=%d lines, csv=%d lines (incl. header)"
          % (jsonl_lines, csv_lines))
    assert jsonl_lines == 4
    assert csv_lines == 5  # header + 4 records
    # Every JSONL line round-trips and reports duration 0.0 / empty timestamp.
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assert row["status"] == "skipped"
            assert row["duration_seconds"] == 0.0
            assert row["completed_at"] == ""
    # CSV header is stable.
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
        assert tuple(header) == CSV_COLUMNS, header

    # 2) an invalid selector is an error, still with no call and no clock.
    bad = run_benchmark(["not-a-selector"], iterations=1, mode="config-only")
    assert bad["error"] == 1 and bad["skipped"] == 0

    # 3) fake live run with a stub attempt callable.
    calls: List[str] = []

    def stub_attempt(selector: str) -> dict:
        calls.append(selector)
        if selector.endswith("-b"):
            raise RuntimeError("stub provider refused")
        return {"response_summary": "stub ok for %s" % selector}

    live_jsonl = os.path.join(tmp_dir, "live-results.jsonl")
    live_csv = os.path.join(tmp_dir, "live-results.csv")
    live = run_benchmark(
        selectors, iterations=2, attempt=stub_attempt,
        jsonl_path=live_jsonl, csv_path=live_csv, mode="live",
    )
    print("live summary:")
    print(json.dumps(live, indent=2, sort_keys=True))
    assert len(calls) == 4
    assert live["ok"] == 2 and live["error"] == 2
    a_stats = live["per_selector"]["fakeprov/fake-model-a"]
    b_stats = live["per_selector"]["fakeprov/fake-model-b"]
    assert a_stats["ok"] == 2 and b_stats["error"] == 2
    assert b_stats["errors"] and "stub provider refused" in b_stats["errors"][0]
    assert _count_lines(live_jsonl) == 4
    assert _count_lines(live_csv) == 5
    print("flushed: live jsonl=%d lines, live csv=%d lines (incl. header)"
          % (_count_lines(live_jsonl), _count_lines(live_csv)))

    print("SELFCHECK OK: config-only keyless run, invalid-selector error, "
          "stubbed live run, incremental JSONL/CSV flush")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Provider/model benchmark protocol (Forma-OSS port): "
        "config-only vs live modes with crash-safe JSONL/CSV flush."
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="run a config-only sweep and a stubbed live sweep into temp "
        "files and assert the summaries and flushed line counts.",
    )
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
