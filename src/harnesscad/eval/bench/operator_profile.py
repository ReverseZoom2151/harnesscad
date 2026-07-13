"""CAD-operation, sequence-depth and face-count distribution profiles."""
from __future__ import annotations
from collections import Counter

def operator_profile(records, *, reference=None):
    values=tuple(records); occurrences=Counter()
    for row in values: occurrences.update(set(row.get("operations",())))
    rates={op:count/len(values) for op,count in sorted(occurrences.items())} if values else {}
    ref=dict(reference or {})
    return {"count":len(values),"operation_rates":rates,
            "operation_delta":{op:rates.get(op,0)-ref.get(op,0)
                               for op in sorted(set(rates)|set(ref))},
            "sequence_lengths":tuple(sorted(len(row.get("operations",())) for row in values)),
            "face_counts":tuple(sorted(int(row.get("face_count",0)) for row in values))}
