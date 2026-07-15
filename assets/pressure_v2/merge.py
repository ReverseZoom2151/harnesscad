"""Merge the shards of the v2 FRONTIER run into one results.json and render it.

The grid is split by MODEL across processes (the same way v1 was run), so the
shards are disjoint by construction: no cell appears twice. This asserts that
rather than assuming it.

    python assets/pressure_v2/merge.py part_qwen3_6_27b.json part_qwen3_6_35b.json ...

THE DELETED LINEUP. A previous agent ran the v2 CODE against a now-DELETED
six-model set (qwen2.5-coder and friends). Those partial shards live in
``obsolete_deleted_lineup/`` and are NOT inputs to this script. They are kept for
forensic reference only and must never be turned into a published number. This
merger refuses any shard whose models are not the frontier lineup, so it cannot
be pointed at the dead data by accident.
"""

from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from harnesscad.eval.pressure import report as report_mod   # noqa: E402
from harnesscad.eval.pressure.runner import DEFAULT_MODELS   # noqa: E402

FRONTIER = frozenset(DEFAULT_MODELS)
OUT = os.path.join(HERE, "results.json")


def _shard_paths(argv):
    """Shards named on the command line, else every part*.json in this dir.

    ``obsolete_deleted_lineup/`` is never globbed: it holds the dead lineup.
    """
    if argv:
        return [os.path.abspath(p) for p in argv]
    return sorted(glob.glob(os.path.join(HERE, "part*.json")))


def main(argv) -> int:
    shards = _shard_paths(argv)
    if not shards:
        raise SystemExit(
            "no shards to merge. Run the frontier lineup first:\n"
            "  harnesscad pressure --loop all --out assets/pressure_v2/part1.json\n"
            "(the four frontier tags are the default; the deleted six-model set "
            "in obsolete_deleted_lineup/ is NOT an input)")

    results = []
    meta = {}
    models = []
    seen = set()
    for path in shards:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        shard_models = payload["meta"]["models"]
        stray = [m for m in shard_models if m not in FRONTIER]
        if stray:
            raise SystemExit(
                f"REFUSED: {os.path.basename(path)} carries non-frontier models "
                f"{stray}. This is the deleted lineup -- its results must not be "
                f"published. The frontier lineup is {sorted(FRONTIER)}.")
        for r in payload["results"]:
            cell = (r["model"], r["brief"], r["loop"])
            if cell in seen:
                raise SystemExit(f"duplicate cell {cell} in {path}")
            seen.add(cell)
            results.append(r)
        meta = dict(payload["meta"])
        models.extend(m for m in shard_models if m not in models)

    meta["models"] = models
    meta["shards"] = [os.path.basename(p) for p in shards]
    meta["mesher"] = "marching_cubes"
    meta["resolution"] = 96
    payload = {"meta": meta, "results": results}
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, sort_keys=True, indent=1)
        fh.write("\n")

    print(report_mod.render_v2(payload))
    print()
    print(f"wrote {OUT}: {len(results)} cells, {len(models)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
