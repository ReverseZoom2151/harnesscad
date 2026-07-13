"""Stable Morton ordering for quadtree surface patches."""

from __future__ import annotations


def morton2(x: int, y: int, *, bits: int | None = None) -> int:
    if x < 0 or y < 0:
        raise ValueError("coordinates must be non-negative")
    width = bits if bits is not None else max(1, x.bit_length(), y.bit_length())
    if width < x.bit_length() or width < y.bit_length():
        raise ValueError("bits cannot represent coordinates")
    key = 0
    for bit in range(width):
        key |= ((x >> bit) & 1) << (2 * bit + 1)
        key |= ((y >> bit) & 1) << (2 * bit)
    return key


def patch_order(patches):
    """Sort mappings/objects by depth, Morton key and triangle index."""
    def field(item, name, default=0):
        return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)
    return tuple(sorted(patches, key=lambda item: (
        field(item, "depth"), morton2(field(item, "x"), field(item, "y"),
                                      bits=field(item, "depth") or None),
        field(item, "triangle"), repr(item),
    )))
