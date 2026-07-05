"""Analytic plane/triangle cross-sections and tolerance-aware segment stitching."""

from __future__ import annotations

import math


def _signed(point, origin, normal):
    return sum((point[i] - origin[i]) * normal[i] for i in range(3))


def triangle_plane_segment(triangle, origin, normal, tolerance=1e-9):
    distances = [_signed(point, origin, normal) for point in triangle]
    points = []
    for i, a in enumerate(triangle):
        b, da, db = triangle[(i + 1) % 3], distances[i], distances[(i + 1) % 3]
        if abs(da) <= tolerance:
            points.append(tuple(a))
        if da * db < -(tolerance ** 2):
            t = da / (da - db)
            points.append(tuple(a[j] + t * (b[j] - a[j]) for j in range(3)))
    unique = []
    for point in points:
        if not any(math.dist(point, item) <= tolerance for item in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    return tuple(sorted(unique)[:2])


def stitch_segments(segments, tolerance=1e-6):
    work = [list(segment) for segment in segments if segment]
    polylines = []
    while work:
        line = work.pop(0)
        changed = True
        while changed:
            changed = False
            for index, segment in enumerate(work):
                for reverse_line, reverse_segment in ((False, False), (False, True),
                                                      (True, False), (True, True)):
                    endpoint = line[0] if reverse_line else line[-1]
                    candidate = segment[-1] if reverse_segment else segment[0]
                    if math.dist(endpoint, candidate) <= tolerance:
                        if reverse_line:
                            line.reverse()
                        if reverse_segment:
                            segment.reverse()
                        line.extend(segment[1:])
                        work.pop(index)
                        changed = True
                        break
                if changed:
                    break
        polylines.append(tuple(line))
    return tuple(sorted(polylines))


def cross_section(triangles, origin, normal, tolerance=1e-6):
    norm = math.sqrt(sum(float(x) ** 2 for x in normal))
    if norm <= tolerance:
        raise ValueError("plane normal must be non-zero")
    unit = tuple(float(x) / norm for x in normal)
    segments = [triangle_plane_segment(triangle, origin, unit, tolerance)
                for triangle in triangles]
    return stitch_segments([item for item in segments if item], tolerance)
