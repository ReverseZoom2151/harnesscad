"""Static validity audit for tool-use trajectories."""

from __future__ import annotations


def audit_tool_trajectory(calls, tools, *, final_verified=False):
    issues = []
    available = set()
    used_results = set()
    for index, call in enumerate(calls):
        name = call.get("tool")
        tool = tools.get(name)
        if tool is None:
            issues.append((index, "unknown-tool"))
            continue
        try:
            tool.validate_args(call.get("arguments", {}))
        except Exception:
            issues.append((index, "invalid-arguments"))
        missing = set(call.get("prerequisites", ())) - available
        if missing:
            issues.append((index, "missing-prerequisite"))
        referenced = set(call.get("uses", ()))
        if not referenced <= available:
            issues.append((index, "unknown-result-reference"))
        used_results.update(referenced)
        if call.get("result_id"):
            available.add(call["result_id"])
        if call.get("state_digest") != call.get("current_digest"):
            issues.append((index, "stale-state"))
    unused = tuple(sorted(available - used_results))
    if not final_verified:
        issues.append((len(calls), "unverified-terminal"))
    return {"valid": not issues, "issues": tuple(issues), "unused_results": unused}
