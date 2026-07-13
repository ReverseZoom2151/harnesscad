"""Continuous primitive matching and subreference-aware constraint F1."""

from __future__ import annotations

import math


def _f1(matched, actual, expected):
    p = matched / actual if actual else (1.0 if not expected else 0.0)
    r = matched / expected if expected else (1.0 if not actual else 0.0)
    return {"precision": p, "recall": r,
            "f1": 2*p*r/(p+r) if p+r else 0.0, "matched": matched}


def sketch_f1(actual_primitives, expected_primitives, actual_constraints=(),
              expected_constraints=(), *, tolerance=1e-3):
    unmatched = set(range(len(expected_primitives)))
    mapping = {}
    for ai, actual in enumerate(actual_primitives):
        candidates = []
        for ei in unmatched:
            expected = expected_primitives[ei]
            if actual["type"] != expected["type"]:
                continue
            av, ev = tuple(actual.get("params", ())), tuple(expected.get("params", ()))
            if len(av) == len(ev) and math.dist(av, ev) <= tolerance:
                candidates.append(ei)
        if candidates:
            ei = min(candidates)
            mapping[actual["id"]] = expected_primitives[ei]["id"]
            unmatched.remove(ei)
    def signature(item, remap=False):
        refs = tuple(mapping.get(ref, ref) for ref in item.get("primitives", ())) if remap \
            else tuple(item.get("primitives", ()))
        return item["type"], refs, tuple(item.get("subreferences", ()))
    expected_signatures = {signature(item) for item in expected_constraints}
    matched_constraints = sum(
        signature(item, True) in expected_signatures
        and all(ref in mapping for ref in item.get("primitives", ()))
        for item in actual_constraints)
    return {
        "primitive": _f1(len(mapping), len(actual_primitives), len(expected_primitives)),
        "constraint": _f1(matched_constraints, len(actual_constraints),
                          len(expected_constraints)),
        "mapping": mapping,
    }
