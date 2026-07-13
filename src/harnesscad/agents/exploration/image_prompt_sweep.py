"""Provider-neutral fixed-configuration image-prompt sweeps."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PromptRun:
    weight: float; seed: int; index: int; output: object
def sweep(generator,*,text,image,weights,seeds,outputs_per_setting,config):
    if outputs_per_setting<1:raise ValueError("outputs_per_setting must be positive")
    out=[]
    for weight in sorted(set(weights)):
        if not 0<=weight<=1:raise ValueError("weights must be in [0,1]")
        for seed in sorted(set(seeds)):
            for index in range(outputs_per_setting):
                out.append(PromptRun(weight,seed,index,generator(
                    text=text,image=image,image_weight=weight,seed=seed,index=index,config=dict(config))))
    return tuple(out)
