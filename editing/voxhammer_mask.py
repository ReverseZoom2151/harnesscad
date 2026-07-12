"""VoxHammer 3D edit-region mask scheme (deterministic region selection).

Implements the deterministic, training-free part of VoxHammer's masking: given
the set of active voxels/tokens of a structured 3D latent (SLAT) and a
user-specified 3D editing region, select the voxels inside the region (to be
edited) and the complementary preserved set ``Omega_keep`` (kept fixed). Masks
come in two flavours matching the paper:

- binary masks ``Mss in {0,1}`` used for latent/KV replacement (eqs. 4-7);
- soft masks ``f_Mss in [0,1]`` obtained by dilation + Gaussian falloff to
  mitigate visible seams at the mask boundary.

A voxel coordinate is a tuple of three ints. All operations are deterministic
and stdlib-only (no wall clock, no unseeded randomness).
"""
from __future__ import annotations

from math import exp, sqrt


def _as_coord(c):
    t = tuple(int(x) for x in c)
    if len(t) != 3:
        raise ValueError("voxel coordinate must have 3 components")
    return t


def _euclid(a, b):
    return sqrt(sum((float(u) - float(v)) ** 2 for u, v in zip(a, b)))


def edit_voxels_in_box(coords, lo, hi):
    """Select active voxels lying inside the inclusive axis-aligned box [lo, hi]."""
    lo = _as_coord(lo)
    hi = _as_coord(hi)
    if any(h < l for l, h in zip(lo, hi)):
        raise ValueError("hi must be componentwise >= lo")
    out = set()
    for c in coords:
        c = _as_coord(c)
        if all(lo[i] <= c[i] <= hi[i] for i in range(3)):
            out.add(c)
    return frozenset(out)


def edit_voxels_in_sphere(coords, center, radius):
    """Select active voxels within Euclidean ``radius`` of ``center``."""
    center = _as_coord(center)
    r = float(radius)
    if r < 0:
        raise ValueError("radius must be non-negative")
    return frozenset(c for c in (_as_coord(x) for x in coords) if _euclid(c, center) <= r)


def preserved_voxels(coords, edit_voxels):
    """Omega_keep: the active voxels that are NOT in the edit region."""
    edit = frozenset(_as_coord(c) for c in edit_voxels)
    return frozenset(c for c in (_as_coord(x) for x in coords) if c not in edit)


def binary_mask(coords, edit_voxels):
    """Binary edit mask Mss: 1.0 for edit voxels, 0.0 for preserved voxels.

    Convention matches the replacement equations where a value of 1 selects the
    freshly denoised (edited) feature and 0 selects the cached inverted feature.
    """
    edit = frozenset(_as_coord(c) for c in edit_voxels)
    return {c: (1.0 if c in edit else 0.0) for c in (_as_coord(x) for x in coords)}


def dilate(edit_voxels, radius=1, connectivity=6):
    """Grow the edit set by ``radius`` steps under 6- or 26-connectivity."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if connectivity not in (6, 26):
        raise ValueError("connectivity must be 6 or 26")
    cur = set(_as_coord(c) for c in edit_voxels)
    for _ in range(int(radius)):
        nxt = set(cur)
        for (x, y, z) in cur:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        if connectivity == 6 and (abs(dx) + abs(dy) + abs(dz)) != 1:
                            continue
                        nxt.add((x + dx, y + dy, z + dz))
        cur = nxt
    return frozenset(cur)


def soft_mask(coords, edit_voxels, dilation=1, sigma=1.0):
    """Soft edit mask f_Mss in [0,1] via dilation plateau + Gaussian falloff.

    For each active voxel let ``d`` be its Euclidean distance to the nearest edit
    voxel. Voxels within ``dilation`` of the edit region form a plateau of weight
    1.0; beyond that the weight decays as ``exp(-(d-dilation)^2 / (2 sigma^2))``.
    This smooth transition suppresses the visible seams described in the paper.
    """
    if dilation < 0:
        raise ValueError("dilation must be non-negative")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    edit = [_as_coord(c) for c in edit_voxels]
    edit_set = frozenset(edit)
    out = {}
    for c in (_as_coord(x) for x in coords):
        if c in edit_set:
            out[c] = 1.0
            continue
        if not edit:
            out[c] = 0.0
            continue
        d = min(_euclid(c, e) for e in edit)
        if d <= dilation:
            out[c] = 1.0
        else:
            out[c] = exp(-((d - dilation) ** 2) / (2.0 * sigma * sigma))
    return out
