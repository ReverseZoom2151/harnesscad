"""Canonical constructive scripts from already-recorded execution events."""
from __future__ import annotations
import ast

GEOMETRY_KINDS=frozenset({"sketch","extrude","revolve","loft","sweep","fillet",
                          "chamfer","shell","boolean","pattern","transform"})

def slice_trace(parameters, events, *, imports=("import cadquery as cq",)):
    binds=[f"{name} = {value!r}" for name,value in parameters]
    calls=[]
    for event in events:
        if event.get("kind") in GEOMETRY_KINDS and event.get("contributes", True):
            statement=str(event["statement"]).strip()
            ast.parse(statement)
            calls.append(statement)
    used_math=any("math." in line for line in binds+calls)
    header=sorted(set(imports) | ({"import math"} if used_math else set()))
    body=header+binds+calls
    if not calls or not calls[-1].lstrip().startswith("result ="):
        output=next((e.get("output") for e in reversed(events) if e.get("output")), None)
        if not output: raise ValueError("trace has no final output")
        body.append(f"result = {output}")
    return "\n".join(body)+"\n"

def verify_slice(source, execute, equivalent):
    ast.parse(source)
    try: result=execute(source)
    except Exception as exc:
        return {"accepted":False,"reason":f"execution:{type(exc).__name__}"}
    evidence=dict(equivalent(result))
    return {"accepted":bool(evidence.get("equivalent")),
            "reason":"" if evidence.get("equivalent") else "not-equivalent",
            "evidence":evidence}
