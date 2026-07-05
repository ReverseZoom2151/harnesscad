"""Executable contracts for representative natural-language CAD tasks."""
from __future__ import annotations

def evaluate_case(intent, *, required_operations=(), required_capabilities=()):
    operations=set(intent.operations)
    missing=tuple(sorted(set(required_operations)-operations))
    capabilities=tuple(sorted(required_capabilities))
    return {"resolved":not intent.needs_clarification,"missing_operations":missing,
            "capability_route":capabilities,
            "intent_coverage":1-len(missing)/len(required_operations)
            if required_operations else 1.0}

def paper_casebook():
    return (
        {"id":"box-sphere-union","operations":("union","bake"),"capabilities":("geometry",)},
        {"id":"iterative-pavilion","operations":("loft",),"capabilities":("geometry","feedback")},
        {"id":"building-sun-study","operations":("simulate",),
         "capabilities":("geometry","sun-analysis")},
    )
