"""Per-family tolerant command F1 for lines, arcs, circles and extrusions."""

from __future__ import annotations

import math


def command_metrics(actual, expected, *, tolerance=1e-3,
                    families=("line", "arc", "circle", "extrude")):
    result = {}
    for family in families:
        left = [item for item in actual if item["type"] == family]
        right = [item for item in expected if item["type"] == family]
        unmatched, matched = set(range(len(right))), 0
        for item in left:
            choices = [index for index in unmatched
                       if len(item.get("params", ())) == len(right[index].get("params", ()))
                       and math.dist(tuple(item.get("params", ())),
                                     tuple(right[index].get("params", ()))) <= tolerance]
            if choices:
                unmatched.remove(min(choices)); matched += 1
        precision = matched/len(left) if left else (1.0 if not right else 0.0)
        recall = matched/len(right) if right else (1.0 if not left else 0.0)
        result[family] = {"matched": matched, "actual": len(left), "expected": len(right),
                          "precision": precision, "recall": recall,
                          "f1": 2*precision*recall/(precision+recall)
                          if precision+recall else 0.0}
    result["macro_f1"] = sum(result[name]["f1"] for name in families)/len(families)
    return result
