"""Cross-variant consensus and per-voxel ensemble confidence (AlphaCAD).

Source: ``AlphaCAD-main`` (``summit-demo/vote_server.py`` comparative-analysis
block). When several candidate voxel models are generated for one prompt,
AlphaCAD compares them to surface *agreement*: which cells every variant places,
and how confident each individual brick is given how many variants share its
exact cell. This is a deterministic set/counting analysis over a small ensemble
of candidate designs -- complementary to the harness's single-model quality
metrics, and directly reusable for any "generate N variants, rank by agreement"
exploration loop.

All functions are pure, stdlib only, and deterministic (dict/set ordering is
made explicit via sorting where output order matters).

A "model" is any dict with a ``bricks`` list of ``{'id','x','y','z'}`` cells.
"""

from __future__ import annotations

from dataclasses import dataclass


def _cells(model: dict):
    return [(b["x"], b["y"], b["z"]) for b in model.get("bricks", [])]


def base_positions(model: dict) -> set[tuple[int, int, int]]:
    """Set of ground-layer cells ``(x, y, 0)`` occupied by the model."""
    return {(b["x"], b["y"], 0) for b in model.get("bricks", []) if b["z"] == 0}


def consensus_base(models: list[dict]) -> set[tuple[int, int, int]]:
    """Ground-layer cells occupied by *every* model (set intersection).

    An empty ensemble yields the empty set.
    """
    if not models:
        return set()
    result = base_positions(models[0])
    for m in models[1:]:
        result &= base_positions(m)
    return result


def position_frequency(models: list[dict]) -> dict[tuple[int, int, int], int]:
    """Map each occupied cell to the number of models that place a brick there."""
    freq: dict[tuple[int, int, int], int] = {}
    for m in models:
        for cell in _cells(m):
            freq[cell] = freq.get(cell, 0) + 1
    return freq


def consensus_brick_ids(model: dict, consensus_cells: set) -> list[int]:
    """IDs of ``model``'s bricks whose cell is in ``consensus_cells`` (sorted)."""
    out = [b["id"] for b in model.get("bricks", [])
           if (b["x"], b["y"], b["z"]) in consensus_cells
           or (b["x"], b["y"], 0) in consensus_cells and b["z"] == 0]
    return sorted(set(out))


@dataclass(frozen=True)
class BrickConfidence:
    brick_id: int
    frequency: int
    confidence: int


def ensemble_brick_confidence(model: dict, freq: dict,
                              n_models: int,
                              violations: set[int] | None = None) -> list[BrickConfidence]:
    """Per-brick confidence from how many models share its cell.

    Confidence tiers (matching AlphaCAD's demo): a cell shared by all models
    scores 100, by a strict majority-or-more (>= 2 given 3) scores 66, unique
    scores 33. A brick flagged as a support *violation* is penalised by 40.
    Generalised: ``freq >= n_models`` -> 100, ``freq >= 2`` -> 66, else 33.
    """
    violations = violations or set()
    out: list[BrickConfidence] = []
    for b in model.get("bricks", []):
        cell = (b["x"], b["y"], b["z"])
        f = freq.get(cell, 1)
        if f >= max(1, n_models):
            base = 100
        elif f >= 2:
            base = 66
        else:
            base = 33
        if b["id"] in violations:
            base = max(0, base - 40)
        out.append(BrickConfidence(b["id"], f, base))
    return out


def analyze_variants(models: list[dict]) -> dict:
    """Full ensemble report: consensus cells, per-model consensus ids, freq map.

    Returns a dict with ``consensus`` (sorted list of ground cells),
    ``frequency`` (cell->count), and ``per_model`` (list aligned with ``models``,
    each with ``consensus_ids`` and ``brick_confidence`` list of
    ``(id, confidence)``).
    """
    n = len(models)
    consensus = consensus_base(models)
    freq = position_frequency(models)
    per_model = []
    for m in models:
        ids = consensus_brick_ids(m, consensus)
        confs = ensemble_brick_confidence(m, freq, n)
        per_model.append({
            "consensus_ids": ids,
            "brick_confidence": [(c.brick_id, c.confidence) for c in confs],
        })
    return {
        "consensus": sorted(consensus),
        "frequency": freq,
        "per_model": per_model,
    }
