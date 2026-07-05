"""Coordinate-tolerant reconstruction precision, recall and F1."""

from __future__ import annotations


def _score(actual, expected):
    actual, expected = set(actual), set(expected)
    matched = len(actual & expected)
    precision = matched / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = matched / len(expected) if expected else (1.0 if not actual else 0.0)
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "matched": matched, "actual": len(actual), "expected": len(expected)}


def edge_prf(actual, expected, tolerance: float = 1e-6):
    def signature(edge):
        q = lambda p: tuple(round(value / tolerance) for value in p)
        return tuple(sorted((q(edge.start), q(edge.end))))
    return _score(map(signature, actual), map(signature, expected))


def face_prf(actual, expected):
    """Compare faces topologically by their (unordered) boundary edge sets."""
    def signature(face):
        loops = (face.outer, *face.inner) if hasattr(face, "outer") else (face,)
        return tuple(sorted(tuple(sorted(loop.edge_indices)) for loop in loops))
    return _score(map(signature, actual), map(signature, expected))
