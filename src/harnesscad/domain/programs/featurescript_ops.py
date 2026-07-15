"""FeatureScript extended operation set and query-reference validation (Pyatov et
al., 2026, "CADFS: A Big CAD Program Dataset and Framework for Computer-Aided
Design with Large Language Models").

Prior generative-CAD datasets are limited to *sketch + extrude*, because a flat
design history lets an operation refer only to the immediately-preceding one.
CADFS adopts a FeatureScript-based representation that (a) exposes ~15 modeling
operations and (b) lets any operation target entities produced by *any* earlier
operation via a query -- ``makeQuery(operation_id, entity_role)`` -- so fillet,
loft, pattern, etc. can act on edges/faces created much earlier. That query
mechanism is the paper's central representational advance and is deterministic:

* :data:`FEATURESCRIPT_OPS` -- the extended operation vocabulary with, per op, the
  entity roles it *produces* and whether it *requires a query reference*.
* :func:`validate_program` -- checks each operation's params and that every query
  reference resolves to an earlier operation and a role that operation actually
  produces (the "operation identifier scopes the query" rule). Unlike a flat
  sketch-extrude history, references may point arbitrarily far back.

Deterministic, stdlib-only. Operations are plain dicts.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

__all__ = ["FEATURESCRIPT_OPS", "FeatureScriptError", "validate_program", "op_names"]


class FeatureScriptError(ValueError):
    """Raised on an invalid FeatureScript program."""


# op -> {"produces": roles it emits, "needs_query": requires >=1 entity reference,
#        "required_params": param names that must be present}
FEATURESCRIPT_OPS: Mapping[str, Mapping[str, object]] = {
    "sketch":    {"produces": ("face", "edge", "vertex"), "needs_query": False, "required_params": ()},
    "extrude":   {"produces": ("solid", "face", "edge"),  "needs_query": True,  "required_params": ("distance",)},
    "revolve":   {"produces": ("solid", "face", "edge"),  "needs_query": True,  "required_params": ("angle",)},
    "loft":      {"produces": ("solid", "face", "edge"),  "needs_query": True,  "required_params": ()},
    "sweep":     {"produces": ("solid", "face", "edge"),  "needs_query": True,  "required_params": ()},
    "fillet":    {"produces": ("face", "edge"),           "needs_query": True,  "required_params": ("radius",)},
    "chamfer":   {"produces": ("face", "edge"),           "needs_query": True,  "required_params": ("distance",)},
    "shell":     {"produces": ("solid", "face"),          "needs_query": True,  "required_params": ("thickness",)},
    "draft":     {"produces": ("face",),                  "needs_query": True,  "required_params": ("angle",)},
    "hole":      {"produces": ("face", "edge"),           "needs_query": True,  "required_params": ("diameter",)},
    "mirror":    {"produces": ("solid",),                 "needs_query": True,  "required_params": ()},
    "pattern":   {"produces": ("solid",),                 "needs_query": True,  "required_params": ("count",)},
    "boolean":   {"produces": ("solid",),                 "needs_query": True,  "required_params": ("kind",)},
    "remove":    {"produces": (),                         "needs_query": True,  "required_params": ()},
    "thicken":   {"produces": ("solid",),                 "needs_query": True,  "required_params": ("thickness",)},
}

_BOOLEAN_KINDS = frozenset({"union", "subtract", "intersect"})


def op_names() -> List[str]:
    """Return the FeatureScript operation vocabulary (sorted)."""
    return sorted(FEATURESCRIPT_OPS)


def _check_params(idx: int, op: str, params: Mapping[str, object]) -> None:
    spec = FEATURESCRIPT_OPS[op]
    for req in spec["required_params"]:
        if req not in params:
            raise FeatureScriptError(f"op {idx} ({op}) missing required param {req!r}")
    if op == "boolean" and params.get("kind") not in _BOOLEAN_KINDS:
        raise FeatureScriptError(
            f"op {idx} boolean kind must be one of {sorted(_BOOLEAN_KINDS)}"
        )
    if op == "pattern":
        count = params.get("count")
        if not isinstance(count, int) or count < 2:
            raise FeatureScriptError(f"op {idx} pattern count must be an int >= 2")


def validate_program(ops: Sequence[Mapping[str, object]]) -> None:
    """Validate a FeatureScript program (raises :class:`FeatureScriptError`).

    Each op is ``{"op": name, "params": {...}, "queries": [{"op_index": i,
    "role": r}, ...]}``. Rules:

    * ``op`` must be in :data:`FEATURESCRIPT_OPS`;
    * required params present (and boolean/pattern param constraints hold);
    * ops with ``needs_query`` must supply >=1 query, and every query must point
      to a strictly-earlier op index and a role that op *produces*.
    """
    for idx, entry in enumerate(ops):
        op = entry.get("op")
        if op not in FEATURESCRIPT_OPS:
            raise FeatureScriptError(f"op {idx}: unknown operation {op!r}")
        params = entry.get("params", {})
        _check_params(idx, op, params)
        queries = entry.get("queries", []) or []
        spec = FEATURESCRIPT_OPS[op]
        if spec["needs_query"] and not queries:
            raise FeatureScriptError(f"op {idx} ({op}) requires an entity query")
        for q in queries:
            j = q.get("op_index")
            role = q.get("role")
            if not isinstance(j, int) or not (0 <= j < idx):
                raise FeatureScriptError(
                    f"op {idx} ({op}) query must reference an earlier op index, got {j!r}"
                )
            target_op = ops[j]["op"]
            produced = FEATURESCRIPT_OPS[target_op]["produces"]
            if role not in produced:
                raise FeatureScriptError(
                    f"op {idx} ({op}) queries role {role!r} not produced by op {j} ({target_op})"
                )
