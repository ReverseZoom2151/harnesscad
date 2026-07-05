"""Seeded point corruption and modality-complementarity evaluation."""
from __future__ import annotations
import random

def gaussian_noise(points,variance,seed):
    if variance<0:raise ValueError("variance must be non-negative")
    rng=random.Random(seed);sigma=variance**.5
    return tuple(tuple(v+rng.gauss(0,sigma) for v in p) for p in points)
def eliminate_points(points,fraction,seed):
    if not 0<=fraction<=1:raise ValueError("fraction must be in [0,1]")
    values=list(points);random.Random(seed).shuffle(values)
    return tuple(values[round(len(values)*fraction):])
def degradation_curve(levels,corrupt,evaluate):
    return tuple((level,float(evaluate(corrupt(level)))) for level in levels)
def complementarity_delta(base_score,combined_score,higher_is_better=True):
    return (combined_score-base_score)*(1 if higher_is_better else -1)
