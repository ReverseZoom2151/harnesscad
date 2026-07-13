"""Deterministic spatial contact/intersection heatmaps."""
from collections import Counter
def contact_heatmap(points,*,bin_size):
    if bin_size<=0:raise ValueError("bin size must be positive")
    bins=Counter(tuple(round(float(v)/bin_size) for v in point) for point in points)
    return dict(sorted(bins.items()))
