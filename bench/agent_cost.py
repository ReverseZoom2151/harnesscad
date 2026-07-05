"""Task-level token, call, latency and price aggregation."""

from __future__ import annotations


def agent_cost(calls, pricing):
    totals = {"input_tokens": 0, "output_tokens": 0, "tool_tokens": 0,
              "calls": 0, "latency_seconds": 0.0, "cost": 0.0}
    for call in calls:
        model = call["model"]
        rate = pricing[model]
        inp, out, tools = (int(call.get(key, 0))
                           for key in ("input_tokens", "output_tokens", "tool_tokens"))
        totals["input_tokens"] += inp
        totals["output_tokens"] += out
        totals["tool_tokens"] += tools
        totals["calls"] += 1
        totals["latency_seconds"] += float(call.get("latency_seconds", 0))
        totals["cost"] += (inp + tools) / 1000 * rate["input_per_1k"] \
            + out / 1000 * rate["output_per_1k"]
    totals["pricing_models"] = tuple(sorted({call["model"] for call in calls}))
    return totals
