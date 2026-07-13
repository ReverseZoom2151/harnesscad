"""Sheaf-inspired shared attribute space for Img2CAD conditional factorization.

The Img2CAD paper motivates its attribute predictor with a sheaf-theoretic view:
over a base space of *labelled CAD structures*, the fibres are the continuous
attribute-parameter spaces, and parts that share a semantic label (e.g. the four
"leg" parts of a chair, or "backrest" across different chairs) have locally
consistent attributes. TrAssembler exploits this by sharing information across
commands that share a semantic part name, which lets it learn from limited data.

This module is the deterministic, network-free analogue of that shared space. It
aggregates ground-truth attribute vectors keyed by ``(semantic_label,
command_type, position_within_part)`` across a collection of CAD models, then

* predicts a baseline attribute assignment for a new discrete structure by reading
  the shared mean of each matching key (a deterministic Stage-2 baseline), and
* regularizes an existing attribute prediction by blending it toward the shared
  mean (the "consistency across the attribute space" the paper reports as
  essential to performance).

The learned flow-matching TrAssembler and GMFlow remain out of scope.
"""

from __future__ import annotations

from harnesscad.domain.reconstruction.img2cadrev_schema import attribute_dim
from harnesscad.domain.reconstruction.img2cadrev_factorization import (
    factorize,
    validate_structure,
    structure_command_count,
)


def _keys_for_structure(structure):
    """Yield (part_index, command_index, key) over a discrete structure.

    ``key`` is ``(label, command_type, position_within_part)`` -- the coordinate
    in the shared attribute space.
    """
    for part in structure:
        label = part["label"]
        for pos, ctype in enumerate(part["command_types"]):
            yield label, ctype, (label, ctype, pos)


class SharedAttributePrior:
    """Elementwise mean of attribute vectors grouped by shared semantic key."""

    def __init__(self):
        # key -> [running_sum_vector, count]
        self._sums: dict[tuple, list] = {}

    def add_model(self, model):
        """Accumulate one CAD model's ground-truth attributes into the prior."""
        structure, attributes = factorize(model)
        idx = 0
        for _label, _ctype, key in _keys_for_structure(structure):
            vec = attributes[idx]
            idx += 1
            if key not in self._sums:
                self._sums[key] = [list(vec), 1]
            else:
                acc = self._sums[key]
                for i, v in enumerate(vec):
                    acc[0][i] += v
                acc[1] += 1

    def fit(self, models):
        """Accumulate a collection of models; returns self for chaining."""
        for model in models:
            self.add_model(model)
        return self

    def count(self, key) -> int:
        entry = self._sums.get(key)
        return entry[1] if entry else 0

    def mean(self, key):
        """Shared mean attribute vector for ``key``, or None if unseen."""
        entry = self._sums.get(key)
        if not entry:
            return None
        total, n = entry
        return [s / n for s in total]

    def keys(self):
        return list(self._sums.keys())

    def predict(self, structure):
        """Deterministic Stage-2 baseline: fill attributes from shared means.

        For every command in ``structure`` whose key was seen during ``fit`` the
        prediction is the shared mean; unseen keys get a zero vector of the correct
        arity. Returns a flat, model-ordered attribute list (assemble-compatible).
        """
        validate_structure(structure)
        out = []
        for _label, ctype, key in _keys_for_structure(structure):
            mean = self.mean(key)
            if mean is None:
                out.append([0.0] * attribute_dim(ctype))
            else:
                out.append(list(mean))
        return out

    def coverage(self, structure) -> float:
        """Fraction of a structure's commands with a matching shared key."""
        total = structure_command_count(structure)
        if total == 0:
            return 1.0
        hit = sum(1 for _l, _c, key in _keys_for_structure(structure)
                  if self.count(key) > 0)
        return hit / total

    def regularize(self, structure, attributes, weight: float = 0.5):
        """Blend a prediction toward the shared mean: (1-w)*pred + w*mean.

        ``weight`` in [0, 1]; 0 returns the prediction unchanged, 1 replaces it
        with the shared mean where available. Unseen keys are left untouched.
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        validate_structure(structure)
        expected = structure_command_count(structure)
        if len(attributes) != expected:
            raise ValueError(
                f"structure needs {expected} attribute vectors, "
                f"got {len(attributes)}"
            )
        out = []
        for idx, (_label, _ctype, key) in enumerate(_keys_for_structure(structure)):
            pred = [float(a) for a in attributes[idx]]
            mean = self.mean(key)
            if mean is None:
                out.append(pred)
            else:
                out.append([(1.0 - weight) * p + weight * m
                            for p, m in zip(pred, mean)])
        return out
