"""Report constraint-label flips under primitive perturbation/quantization."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class LabelStability:
    baseline: tuple; variants: tuple[tuple[str,tuple],...]; flips: tuple[str,...]
def constraint_label_stability(primitives,transforms,classify):
    baseline=tuple(classify(primitives));variants=[];flips=[]
    for name,transform in sorted(transforms.items()):
        labels=tuple(classify(transform(primitives)));variants.append((name,labels))
        if labels!=baseline:flips.append(name)
    return LabelStability(baseline,tuple(variants),tuple(flips))
