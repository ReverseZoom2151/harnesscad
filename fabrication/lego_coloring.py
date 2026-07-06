"""Uniform brick-color assignment against a LEGO(R) color palette.

Distilled from Pun, Deng, Liu et al., *Generating Physically Stable and
Buildable LEGO Designs from Text* (LEGOGPT), Section 4.3 ("Uniform Brick Color
Assignment", Eqns 5-7).

The paper's texturing pipeline needs a *learned* mesh texturer (FlashTex) that
is out of scope here, but the **color-quantization** half is fully
deterministic and LEGO-specific, and no generic ``brick_*`` module covers it:

* each occupied voxel's color ``C(v)`` is the mean of its visible-face colors
  (Eqn 7);
* each brick's color ``C(b)`` is the mean over its constituent voxels;
* the brick is then snapped to the *closest color in the standard LEGO color
  set* so it can be built from real parts.

Everything is stdlib-only and deterministic.  Colors are plain ``(r, g, b)``
integer triples in ``[0, 255]``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from fabrication.lego_brick_library import Brick

RGB = Tuple[int, int, int]
Cell = Tuple[int, int, int]

# A representative subset of the standard LEGO solid-color palette.
LEGO_PALETTE: Dict[str, RGB] = {
    "white": (244, 244, 244),
    "black": (27, 42, 52),
    "brick_yellow": (222, 198, 156),
    "bright_red": (196, 40, 27),
    "bright_blue": (13, 105, 172),
    "bright_yellow": (245, 205, 47),
    "dark_green": (40, 127, 70),
    "bright_green": (75, 159, 74),
    "bright_orange": (218, 133, 64),
    "medium_stone_grey": (163, 162, 164),
    "dark_stone_grey": (99, 95, 97),
    "reddish_brown": (105, 64, 39),
    "bright_purple": (200, 112, 160),
    "medium_azure": (54, 174, 191),
    "sand_green": (112, 142, 124),
    "dark_red": (114, 0, 18),
}


def _mean_rgb(colors: Sequence[RGB]) -> RGB:
    n = len(colors)
    if n == 0:
        raise ValueError("cannot average an empty color list")
    r = sum(c[0] for c in colors)
    g = sum(c[1] for c in colors)
    b = sum(c[2] for c in colors)
    # Round-half-up for determinism independent of banker's rounding.
    return ((r * 2 + n) // (2 * n), (g * 2 + n) // (2 * n), (b * 2 + n) // (2 * n))


def voxel_color(face_colors: Sequence[RGB]) -> RGB:
    """C(v): mean of a voxel's visible-face colors (Eqn 7)."""
    return _mean_rgb(face_colors)


def brick_color(voxel_colors: Sequence[RGB]) -> RGB:
    """C(b): mean color over a brick's constituent voxels."""
    return _mean_rgb(voxel_colors)


def _dist2(a: RGB, b: RGB) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def nearest_lego_color(rgb: RGB,
                       palette: Mapping[str, RGB] = LEGO_PALETTE) -> str:
    """Name of the closest palette color by squared Euclidean distance.

    Ties are broken by palette-name order for determinism.
    """
    best_name = None
    best_d = None
    for name in sorted(palette):
        d = _dist2(rgb, palette[name])
        if best_d is None or d < best_d:
            best_d = d
            best_name = name
    assert best_name is not None
    return best_name


def assign_brick_colors(
    bricks: Sequence[Brick],
    voxel_face_colors: Mapping[Cell, Sequence[RGB]],
    palette: Mapping[str, RGB] = LEGO_PALETTE,
) -> List[str]:
    """Assign each brick the closest LEGO color name.

    ``voxel_face_colors`` maps an occupied cell to its list of visible-face
    colors.  Cells with no entry (fully occluded, no visible faces) are ignored
    when averaging; a brick with no visible voxels at all is snapped from mid
    grey.
    """
    out: List[str] = []
    for b in bricks:
        vox_cols: List[RGB] = []
        for cell in b.cells():
            faces = voxel_face_colors.get(cell)
            if faces:
                vox_cols.append(voxel_color(faces))
        if vox_cols:
            col = brick_color(vox_cols)
        else:
            col = (128, 128, 128)
        out.append(nearest_lego_color(col, palette))
    return out
