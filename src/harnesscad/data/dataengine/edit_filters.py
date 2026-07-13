"""Auditable filters for synthesized edit triplets."""
from __future__ import annotations
import re
NOOP=re.compile(r"\b(no (?:change|transformation|edit)|unchanged)\b",re.I)

def filter_edit(*,instruction,mask_count,changed_spans,se_count,
                max_instructions=3,max_masks=5,max_changed_spans=3,max_se=3):
    reasons=[]
    clauses=sum(bool(x.strip()) for x in re.split(r"[.;]",instruction))
    if clauses>max_instructions:reasons.append("too-many-instructions")
    if mask_count>max_masks:reasons.append("too-many-masks")
    if changed_spans>max_changed_spans:reasons.append("too-many-changes")
    if se_count>max_se:reasons.append("sequence-too-complex")
    if NOOP.search(instruction):reasons.append("no-op")
    return tuple(reasons)
