"""Conservative CAD abstraction proposals accepted only by morphology proof."""

from __future__ import annotations


def propose_abstraction(ops):
    values = tuple(ops)
    if len(values) == 5 and all(item.get("op") == "add_line" for item in values[:4]) \
            and values[4].get("op") == "extrude":
        points = [(item["x1"], item["y1"], item["x2"], item["y2"]) for item in values[:4]]
        xs = {value for point in points for value in (point[0], point[2])}
        ys = {value for point in points for value in (point[1], point[3])}
        if len(xs) == len(ys) == 2:
            return {"kind": "box", "width": max(xs)-min(xs),
                    "height": max(ys)-min(ys),
                    "depth": values[4]["distance"]}
    return None


def accept_abstraction(original, proposal, compile_candidate, judge, *, tolerance):
    if proposal is None:
        return {"accepted": False, "reason": "no-proposal"}
    result = judge(compile_candidate(original), compile_candidate(proposal))
    accepted = bool(result.get("valid") and result.get("distance") is not None
                    and result["distance"] <= tolerance)
    return {"accepted": accepted, "reason": "" if accepted else "not-equivalent",
            "evidence": result}
