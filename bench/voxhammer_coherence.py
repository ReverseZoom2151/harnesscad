"""VoxHammer boundary coherence metrics for 3D local editing.

Deterministic checks that (a) the edited region integrates smoothly with the
preserved region across their shared boundary, and (b) preserved-region features
are faithfully retained relative to the source (the goal of unedited-region
preservation the paper measures with masked PSNR/Chamfer). These operate on the
structured latents directly, keyed by voxel coordinate, and are stdlib-only.
"""
from __future__ import annotations

from math import log10, sqrt


def _l2(a, b):
    return sqrt(sum((float(u) - float(v)) ** 2 for u, v in zip(a, b)))


def _neighbors(c, connectivity):
    x, y, z = c
    if connectivity == 6:
        offs = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    elif connectivity == 26:
        offs = tuple(
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        )
    else:
        raise ValueError("connectivity must be 6 or 26")
    return tuple((x + dx, y + dy, z + dz) for dx, dy, dz in offs)


def boundary_pairs(coords, edit_set, connectivity=6):
    """Adjacent (edit, keep) voxel pairs straddling the edit boundary.

    Returns a sorted tuple of ``(edit_coord, keep_coord)`` pairs where the two
    voxels are both active and neighbours under the given connectivity.
    """
    active = frozenset(tuple(int(v) for v in c) for c in coords)
    edit = frozenset(tuple(int(v) for v in c) for c in edit_set)
    pairs = set()
    for e in edit:
        if e not in active:
            continue
        for nb in _neighbors(e, connectivity):
            if nb in active and nb not in edit:
                pairs.add((e, nb))
    return tuple(sorted(pairs))


def boundary_discontinuity(latents, coords, edit_set, connectivity=6):
    """Mean L2 latent difference across the edit/keep boundary (lower=smoother).

    Returns ``None`` if there is no boundary. This is the coherence check: a low
    value means the edited region blends smoothly into the preserved region.
    """
    pairs = boundary_pairs(coords, edit_set, connectivity)
    if not pairs:
        return None
    total = 0.0
    for e, k in pairs:
        total += _l2(latents[e], latents[k])
    return total / len(pairs)


def preservation_mse(edited_latents, source_latents, keep_set):
    """Mean squared latent error over the preserved region (lower=better)."""
    keep = [c for c in keep_set if c in edited_latents and c in source_latents]
    if not keep:
        return None
    total = 0.0
    count = 0
    for c in keep:
        for u, v in zip(edited_latents[c], source_latents[c]):
            total += (float(u) - float(v)) ** 2
            count += 1
    return total / count if count else None


def preservation_max_error(edited_latents, source_latents, keep_set):
    """Worst-case per-voxel L2 error over the preserved region."""
    keep = [c for c in keep_set if c in edited_latents and c in source_latents]
    if not keep:
        return None
    return max(_l2(edited_latents[c], source_latents[c]) for c in keep)


def preservation_psnr(edited_latents, source_latents, keep_set, peak=1.0):
    """Masked PSNR-style score over the preserved region (higher=better).

    Returns ``float('inf')`` for a perfect match. Analogue of the masked PSNR
    used in the paper, computed on latents instead of rendered pixels.
    """
    mse = preservation_mse(edited_latents, source_latents, keep_set)
    if mse is None:
        return None
    if mse == 0.0:
        return float("inf")
    return 10.0 * log10((float(peak) ** 2) / mse)


def coherence_report(edited_latents, source_latents, coords, edit_set,
                     keep_set, connectivity=6, peak=1.0):
    """Bundle the boundary and preservation metrics into one dict."""
    return {
        "boundary_discontinuity": boundary_discontinuity(
            edited_latents, coords, edit_set, connectivity),
        "preservation_mse": preservation_mse(edited_latents, source_latents, keep_set),
        "preservation_max_error": preservation_max_error(
            edited_latents, source_latents, keep_set),
        "preservation_psnr": preservation_psnr(
            edited_latents, source_latents, keep_set, peak),
        "n_boundary_pairs": len(boundary_pairs(coords, edit_set, connectivity)),
    }
