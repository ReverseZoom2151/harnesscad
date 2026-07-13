"""Detect command-family shortcuts hidden by aggregate geometry improvement."""

from __future__ import annotations

from collections import Counter


def reward_hacking_audit(actual_commands, expected_commands, *,
                         candidate_distance, baseline_distance=None):
    actual = Counter(item["type"] for item in actual_commands)
    expected = Counter(item["type"] for item in expected_commands)
    deltas = {name: actual[name]-expected[name] for name in sorted(actual | expected)}
    arc_collapse = expected["arc"] > actual["arc"] and (
        actual["line"] > expected["line"] or actual["circle"] > expected["circle"])
    complexity_collapse = sum(actual.values()) < sum(expected.values())
    geometry_improved = (baseline_distance is not None
                         and candidate_distance < baseline_distance)
    flags = []
    if arc_collapse: flags.append("arc-substitution")
    if complexity_collapse: flags.append("complexity-collapse")
    if geometry_improved and flags: flags.append("geometry-semantic-conflict")
    return {"flags": tuple(flags), "family_deltas": deltas,
            "candidate_distance": candidate_distance,
            "geometry_improved": geometry_improved}
