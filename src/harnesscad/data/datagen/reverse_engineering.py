"""Provenance-rich synthetic reverse-engineering sample orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class ReverseEngineeringSample:
    id: str
    recipe: object
    ops: tuple[dict, ...]
    code: str
    point_cloud: tuple[tuple[float, ...], ...]
    shape_digest: str
    provenance: dict


def build_reverse_sample(sample_id, recipe, build_ops, verify, emit_code,
                         mesh, sample_points, *, seed=0, point_count=256,
                         provenance=None):
    ops = tuple(build_ops(recipe))
    shape = verify(ops)
    if shape is None:
        raise ValueError("recipe did not produce a verified shape")
    code = emit_code(ops)
    triangles = mesh(shape)
    cloud = tuple(sample_points(triangles, point_count, seed))
    digest = hashlib.sha256(json.dumps(
        {"ops": ops, "code": code}, sort_keys=True, default=repr).encode()).hexdigest()
    return ReverseEngineeringSample(sample_id, recipe, ops, code, cloud, digest,
                                    dict(provenance or {}))
