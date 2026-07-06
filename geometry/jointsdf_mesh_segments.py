"""Mesh label propagation and connected-component part segmentation.

The joint SDF paper extracts a mesh with Marching Cubes, evaluates the
segmentation head at each vertex, and then assigns *each face the majority label
of its three vertices*.  Given per-vertex labels this face-labelling step is a
deterministic mesh operation, implemented here.

Downstream, a face-labelled mesh is decomposed into *instances* by splitting each
label into its adjacency-connected components: two faces sharing an edge and the
same label belong to the same part.  This connected-component segmentation is the
deterministic clustering the paper relies on to turn a per-face label field into
discrete parts (and to count predicted parts for the part-count metric).
"""

from __future__ import annotations


def majority_face_labels(faces, vertex_labels):
    """Label each triangle by the majority label of its three vertices.

    ``faces`` is a list of ``(i, j, k)`` vertex-index triples.  On a 3-way tie
    (all vertices distinct labels) the smallest-``repr`` label is chosen, keeping
    the result deterministic.  Returns a list of per-face labels.
    """
    out = []
    for f in faces:
        if len(f) != 3:
            raise ValueError("faces must be triangles (i, j, k)")
        counts = {}
        for v in f:
            lab = vertex_labels[v]
            counts[lab] = counts.get(lab, 0) + 1
        best = max(counts.items(), key=lambda kv: (kv[1], _rank(kv[0])))
        out.append(best[0])
    return out


def _rank(label):
    # Larger tuple wins in max(); invert repr so the *smallest* repr is chosen.
    return tuple(-ord(ch) for ch in repr(label))


def _edge_adjacency(faces):
    """Map face index -> set of face indices sharing at least one edge."""
    edge_faces = {}
    for fi, f in enumerate(faces):
        a, b, c = f
        for e in ((a, b), (b, c), (a, c)):
            key = (min(e), max(e))
            edge_faces.setdefault(key, []).append(fi)
    adj = {i: set() for i in range(len(faces))}
    for _, flist in edge_faces.items():
        for i in flist:
            for j in flist:
                if i != j:
                    adj[i].add(j)
    return adj


def connected_components(faces, face_labels):
    """Split faces into label-homogeneous edge-connected components.

    Returns a list ``comp`` where ``comp[fi]`` is the integer component id of
    face ``fi``.  Component ids are assigned in ascending face order so the
    labelling is deterministic.  Two faces get the same id iff they are connected
    through a path of same-label edge-adjacent faces.
    """
    if len(faces) != len(face_labels):
        raise ValueError("faces and face_labels length mismatch")
    adj = _edge_adjacency(faces)
    comp = [-1] * len(faces)
    next_id = 0
    for start in range(len(faces)):
        if comp[start] != -1:
            continue
        comp[start] = next_id
        stack = [start]
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if comp[nb] == -1 and face_labels[nb] == face_labels[cur]:
                    comp[nb] = next_id
                    stack.append(nb)
        next_id += 1
    return comp


def part_count(face_labels):
    """Number of distinct labels (the paper's predicted / GT part count)."""
    return len(set(face_labels))


def component_count(faces, face_labels):
    """Number of connected components (over-segmentation-aware part count)."""
    comp = connected_components(faces, face_labels)
    return len(set(comp))
