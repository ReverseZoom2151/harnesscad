"""cadrille multi-view image adapter: 2x2 orthogonal-view grid composition.

For image-based CAD reconstruction cadrille renders four views with fixed
orthogonal camera directions and concatenates them into a 2x2 grid, forming one
combined image (268x268 px in the paper) that is passed through the Qwen vision
encoder as a single input. This module implements the deterministic grid tiling.

Images are represented as row-major 2D arrays (``list`` of rows, each row a list
of pixels); a pixel may be a scalar or an RGB tuple — the tiler is agnostic to
the cell type. The four views must share identical dimensions.

Canonical view order (top-left, top-right, bottom-left, bottom-right):
    front, right, top, back.
"""

from __future__ import annotations

NUM_VIEWS = 4
COMBINED_SIZE = 268
VIEW_SIZE = COMBINED_SIZE // 2  # 134
VISION_TOKENS = 400  # tokens Qwen emits for the combined image

VIEW_ORDER = ("front", "right", "top", "back")


def _dims(image):
    rows = len(image)
    cols = len(image[0]) if rows else 0
    for r in image:
        if len(r) != cols:
            raise ValueError("image rows must be equal length")
    return rows, cols


def compose_grid(views):
    """Tile four equal-size views into one 2x2 grid image.

    ``views`` is a sequence of exactly four 2D arrays with identical dimensions,
    ordered ``[top_left, top_right, bottom_left, bottom_right]``. Returns the
    combined ``(2H) x (2W)`` array.
    """
    if len(views) != NUM_VIEWS:
        raise ValueError(f"expected {NUM_VIEWS} views, got {len(views)}")
    dims = [_dims(v) for v in views]
    if len(set(dims)) != 1:
        raise ValueError("all views must share identical dimensions")
    h, w = dims[0]
    tl, tr, bl, br = views
    combined = []
    for y in range(h):
        combined.append(list(tl[y]) + list(tr[y]))
    for y in range(h):
        combined.append(list(bl[y]) + list(br[y]))
    return combined


def compose_named(view_map):
    """Compose from a ``{view_name: image}`` mapping using ``VIEW_ORDER``."""
    missing = [name for name in VIEW_ORDER if name not in view_map]
    if missing:
        raise ValueError(f"missing views: {missing}")
    return compose_grid([view_map[name] for name in VIEW_ORDER])


def grid_layout():
    """Return the pixel-space placement of each named view in the grid."""
    positions = [(0, 0), (0, VIEW_SIZE), (VIEW_SIZE, 0), (VIEW_SIZE, VIEW_SIZE)]
    return {
        name: {"row": r, "col": c, "size": VIEW_SIZE}
        for name, (r, c) in zip(VIEW_ORDER, positions)
    }
