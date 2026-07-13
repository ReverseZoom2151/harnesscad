"""Mesh segment, manifold-edge, intersection, and enclosure metrics."""
from __future__ import annotations
from collections import Counter
from math import dist

def segment_error(reference_segments,predicted_segments):
    return abs(predicted_segments-reference_segments)/reference_segments if reference_segments else None
def dangling_edge_length(vertices,faces):
    edges=Counter()
    for face in faces:
        for a,b in zip(face,face[1:]+face[:1]):edges[tuple(sorted((a,b)))]+=1
    return sum(dist(vertices[a],vertices[b]) for (a,b),count in edges.items() if count==1)
def self_intersection_ratio(face_count,intersected_faces):
    return len(set(intersected_faces))/face_count if face_count else 0.0
def flux_enclosure_error(triangles):
    """Triangles carry (unit_normal_xyz, area)."""
    return abs(sum((n[0]+n[1]+n[2])*area for n,area in triangles))
