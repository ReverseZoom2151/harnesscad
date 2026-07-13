"""Reviewable semantic parameter schemas from typed operation fields."""

from __future__ import annotations


def expose_parameters(ops, *, ranges=None, labels=None):
    ranges, labels = ranges or {}, labels or {}
    fields = []
    for index, op in enumerate(ops):
        for key, value in sorted(op.items()):
            if key == "op" or not isinstance(value, (int, float)):
                continue
            field_id = f"{index}.{key}"
            bounds = ranges.get(field_id)
            fields.append({"id": field_id, "op": op["op"], "parameter": key,
                           "label": labels.get(field_id, key.replace("_", " ").title()),
                           "value": value, "minimum": bounds[0] if bounds else None,
                           "maximum": bounds[1] if bounds else None})
    return {"schema": "harnesscad.parameters.v1", "fields": tuple(fields),
            "executable": False}
