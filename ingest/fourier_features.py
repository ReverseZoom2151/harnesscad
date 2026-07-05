"""Declared Fourier coordinate features concatenated with optional normals."""

from __future__ import annotations

import math


def fourier_features(point, frequencies=(1., 2., 4.), *, include_coordinates=True):
    if len(point) not in {3, 6}:
        raise ValueError("point must be XYZ or XYZ+normal")
    coords, result = point[:3], list(point[:3] if include_coordinates else ())
    for frequency in frequencies:
        if frequency <= 0:
            raise ValueError("frequencies must be positive")
        for value in coords:
            result.extend((math.sin(2*math.pi*frequency*value),
                           math.cos(2*math.pi*frequency*value)))
    result.extend(point[3:])
    return tuple(result)
