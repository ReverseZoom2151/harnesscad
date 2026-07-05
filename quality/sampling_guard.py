"""Coverage and multi-resolution diagnostics against sparse-sampling reward hacks."""
from __future__ import annotations

def sampling_diagnostics(samples, *, thin_regions=(), interior_regions=(),
                         coarse_cd=None, fine_cd=None, relative_tolerance=.25):
    points=tuple(samples); issues=[]
    def covered(region): return any(region(p) for p in points)
    if any(not covered(r) for r in thin_regions):issues.append("thin-region-uncovered")
    if any(not covered(r) for r in interior_regions):issues.append("interior-uncovered")
    if coarse_cd is not None and fine_cd is not None:
        denom=max(abs(fine_cd),1e-12)
        if abs(coarse_cd-fine_cd)/denom>relative_tolerance:issues.append("multires-disagreement")
    return tuple(issues)
