"""Class and spatial-density audit for scan labels."""
from collections import Counter
from dataclasses import dataclass

@dataclass(frozen=True)
class ScanAudit:
    counts: dict[str, int]
    prevalence: dict[str, float]
    occupied_bins: int
    inverse_frequency: dict[str, float]

def audit(points, labels, bin_size=1.0):
    if bin_size <= 0 or len(points) != len(labels): raise ValueError("invalid audit input")
    counts = Counter(labels); total = len(labels)
    bins = {tuple(int(v//bin_size) for v in p) for p in points}
    return ScanAudit(dict(counts), {k:v/total for k,v in counts.items()} if total else {},
                     len(bins), {k: total/v for k,v in counts.items()})
