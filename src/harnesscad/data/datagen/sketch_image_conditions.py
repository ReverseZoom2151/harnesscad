"""Precise, hand-drawn, and affine/noisy capture manifests."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SketchImageCondition:
    name: str; image: object; seed: int; lineage: tuple[str,...]
def image_conditions(primitives,*,render_precise,simulate_hand,affine_noise,seed):
    precise=render_precise(primitives)
    hand=simulate_hand(precise,seed)
    noisy=affine_noise(hand,seed)
    return (SketchImageCondition("precise",precise,seed,("render",)),
            SketchImageCondition("hand_drawn",hand,seed,("render","hand")),
            SketchImageCondition("noisy_hand_drawn",noisy,seed,("render","hand","affine_noise")))
