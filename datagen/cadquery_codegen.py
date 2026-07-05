"""Restricted deterministic CISP-dict to CadQuery source emitter."""

from __future__ import annotations


def emit_cadquery(ops):
    lines = ["import cadquery as cq", "solid = None"]
    sketches = {}
    for index, op in enumerate(ops):
        tag = op.get("op")
        if tag == "new_sketch":
            name = f"sketch_{index}"
            sketches[op.get("name", name)] = name
            lines.append(f'{name} = cq.Workplane("{op.get("plane", "XY")}")')
        elif tag == "add_rectangle":
            sketch = sketches.get(op.get("sketch"))
            if not sketch:
                raise ValueError("rectangle references unknown sketch")
            lines.append(f"{sketch} = {sketch}.rect({float(op['w'])!r}, {float(op['h'])!r})")
        elif tag == "add_circle":
            sketch = sketches.get(op.get("sketch"))
            if not sketch:
                raise ValueError("circle references unknown sketch")
            lines.append(f"{sketch} = {sketch}.center({float(op.get('cx', 0))!r}, "
                         f"{float(op.get('cy', 0))!r}).circle({float(op['r'])!r})")
        elif tag == "extrude":
            sketch = sketches.get(op.get("sketch"))
            if not sketch:
                raise ValueError("extrude references unknown sketch")
            lines.append(f"solid = {sketch}.extrude({float(op['distance'])!r})")
        elif tag == "fillet":
            lines.append(f"solid = solid.edges().fillet({float(op['radius'])!r})")
        elif tag == "chamfer":
            lines.append(f"solid = solid.edges().chamfer({float(op['distance'])!r})")
        else:
            raise ValueError(f"unsupported op: {tag}")
    if lines[-1] == "solid = None":
        raise ValueError("op stream produced no solid")
    return "\n".join(lines) + "\n"
