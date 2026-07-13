"""Deterministic hierarchical numeric aggregation for B-rep evidence."""

from __future__ import annotations

from statistics import fmean


def aggregate(vectors):
    values = tuple(tuple(float(x) for x in vector) for vector in vectors)
    if not values:
        return ()
    if len({len(value) for value in values}) != 1:
        raise ValueError("vectors must have equal length")
    return tuple(fmean(column) for column in zip(*values)) + tuple(
        max(column) for column in zip(*values))


def hierarchy_descriptors(hierarchy, geometry):
    """Return stable edge/loop/face descriptors from ID->numeric-vector geometry."""
    vertices = {item.id: tuple(geometry.get(item.id, item.point))
                for item in hierarchy.vertices}
    edges = {}
    for item in hierarchy.edges:
        edges[item.id] = tuple(geometry.get(item.id, item.geometry)) + \
            vertices[item.start] + vertices[item.end]
    coedges = {item.id: edges[item.edge] + (1.0 if item.forward else -1.0,)
               for item in hierarchy.coedges}
    loops = {item.id: aggregate(coedges[key] for key in item.coedges)
             for item in hierarchy.loops}
    faces = {}
    loop_map = {item.id: item for item in hierarchy.loops}
    for item in hierarchy.faces:
        outer = [loops[key] for key in item.loops if loop_map[key].outer]
        inner = [loops[key] for key in item.loops if not loop_map[key].outer]
        faces[item.id] = tuple(geometry.get(item.id, item.geometry)) + \
            tuple(outer[0] if outer else ()) + aggregate(inner)
    neighbors = hierarchy.face_neighbors()
    return {
        "edges": edges, "loops": loops,
        "faces": {key: faces[key] + aggregate(faces[n] for n in neighbors[key])
                  for key in sorted(faces)},
    }
