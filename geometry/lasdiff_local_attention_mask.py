"""Deterministic geometry of LAS-Diffusion's view-aware local attention.

From "Locally Attentional SDF Diffusion for Controllable 3D Shape Generation"
(Zheng et al., ACM TOG 2023), Section 3.5. LAS-Diffusion conditions the coarse
occupancy-diffusion U-Net on a 2D sketch through *view-aware local attention*:
each 3D voxel attends only to the image patches near its projected pixel. The
attention *weights* are learned, but the **neighbourhood construction and the
attention mask are deterministic geometry**, which is what this module builds.

Pipeline (Section 3.5 / Fig. 3):

  1. **Patch grid.** A ViT tokenises a ``W x W`` image into non-overlapping
     square patches of side ``patch_width`` (default 224 / 14 -> 16 x 16 patches).
     :func:`patch_centers` returns each patch's pixel centre.
  2. **Voxel projection.** Each voxel centre ``V`` is projected to pixel ``p`` by
     a known pinhole projection (:func:`pinhole_project`).
  3. **Local neighbourhood** ``N_V`` (Eq. before Eq. 4): patch ``P_j`` belongs to
     ``N_V`` iff ``dist(p, centre(P_j)) < d_delta``. The paper's default is
     ``d_delta = 4 * patch_width``.
  4. **Attention mask** ``M`` (Eq. 4): a boolean voxel x patch matrix. We also
     provide the two ablation variants -- *view-agnostic* (mask is all-True; every
     voxel attends to every patch) and *global* (a single global token every
     voxel attends to).

Because the mask depends only on projected distances, small view perturbations
leave most patch sets unchanged (the paper's robustness argument);
:func:`neighborhood_stability` quantifies that overlap. This module is separate
from ``geometry.gaussiancad_camera`` (full extrinsic/intrinsic CAD camera math);
here the projection is a minimal self-contained pinhole. Stdlib-only.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Set, Tuple

Vec3 = Tuple[float, float, float]
Patch = Tuple[int, int]  # (row, col)


def patch_grid_dim(image_width: int = 224, patch_width: int = 14) -> int:
    """Number of patches per side; the ViT tiling must divide evenly."""
    if image_width <= 0 or patch_width <= 0:
        raise ValueError("image_width and patch_width must be positive")
    if image_width % patch_width != 0:
        raise ValueError("image_width must be a multiple of patch_width")
    return image_width // patch_width


def patch_centers(image_width: int = 224, patch_width: int = 14) -> Dict[Patch, Tuple[float, float]]:
    """Pixel centre ``(x, y)`` of every ViT patch, keyed by ``(row, col)``."""
    n = patch_grid_dim(image_width, patch_width)
    centers: Dict[Patch, Tuple[float, float]] = {}
    half = patch_width / 2.0
    for row in range(n):
        for col in range(n):
            cx = col * patch_width + half
            cy = row * patch_width + half
            centers[(row, col)] = (cx, cy)
    return centers


def pinhole_project(point: Vec3, focal: float, cx: float, cy: float,
                    cam_z: float = 3.0) -> Tuple[float, float]:
    """Minimal pinhole projection of a voxel centre to a pixel.

    The camera looks down ``+z`` from distance ``cam_z``; a point at depth
    ``cam_z - z`` in front of the camera maps to ``(focal * x / depth + cx, ...)``.
    Returns pixel ``(px, py)``. Raises if the point is at/behind the camera.
    """
    x, y, z = point
    depth = cam_z - z
    if depth <= 0.0:
        raise ValueError("point is at or behind the camera plane")
    px = focal * x / depth + cx
    py = focal * y / depth + cy
    return (px, py)


def local_neighborhood(pixel: Tuple[float, float],
                       centers: Mapping[Patch, Tuple[float, float]],
                       d_delta: float) -> Set[Patch]:
    """Patches whose centre lies within ``d_delta`` of ``pixel`` (the set N_V)."""
    if d_delta <= 0.0:
        raise ValueError("d_delta must be positive")
    px, py = pixel
    out: Set[Patch] = set()
    for patch, (cx, cy) in centers.items():
        if math.hypot(px - cx, py - cy) < d_delta:
            out.add(patch)
    return out


def default_d_delta(patch_width: int = 14, factor: float = 4.0) -> float:
    """The paper's default neighbourhood radius ``d_delta = 4 * patch_width``."""
    return factor * patch_width


def attention_mask(voxel_pixels: Mapping[str, Tuple[float, float]],
                   centers: Mapping[Patch, Tuple[float, float]],
                   d_delta: float,
                   mode: str = "local") -> Dict[str, Set[Patch]]:
    """Boolean voxel->patch adjacency (the mask ``M`` of Eq. 4).

    ``mode``:
      * ``"local"``  -- view-aware local attention: ``N_V`` per voxel.
      * ``"view_agnostic"`` -- every voxel attends to *all* patches (mask None in
        the paper); returned as the full patch set for each voxel.
    """
    all_patches = set(centers.keys())
    if mode == "view_agnostic":
        return {vid: set(all_patches) for vid in voxel_pixels}
    if mode == "local":
        return {vid: local_neighborhood(p, centers, d_delta)
                for vid, p in voxel_pixels.items()}
    raise ValueError("mode must be 'local' or 'view_agnostic'")


def mask_matrix(mask: Mapping[str, Set[Patch]],
                voxel_order: Sequence[str],
                patch_order: Sequence[Patch]) -> List[List[bool]]:
    """Materialise the adjacency into a dense boolean matrix (rows = voxels)."""
    rows: List[List[bool]] = []
    for vid in voxel_order:
        allowed = mask.get(vid, set())
        rows.append([patch in allowed for patch in patch_order])
    return rows


def neighborhood_stability(a: Set[Patch], b: Set[Patch]) -> float:
    """Jaccard overlap of two neighbourhood sets (view-perturbation robustness).

    Returns 1.0 when both are empty.
    """
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 1.0
