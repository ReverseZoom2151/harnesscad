"""Injected multimodal describe→difference→compression caption workflow."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class EditCaption:
    original_visual: str; edited_visual: str; original_sequence: str
    edited_sequence: str; difference: str; instruction: str

def caption_edit(original,edited,*,visual_describer,sequence_describer,differ,compress):
    ov=visual_describer(original); ev=visual_describer(edited)
    os=sequence_describer(original); es=sequence_describer(edited)
    change=differ((ov,os),(ev,es)); instruction=compress(change)
    if not instruction.strip():raise ValueError("empty editing instruction")
    return EditCaption(ov,ev,os,es,change,instruction.strip())
