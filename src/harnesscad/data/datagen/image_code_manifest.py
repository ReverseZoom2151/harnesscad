"""Immutable provenance for paired render/CAD-code training examples."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class ImageCodeManifest:
    id: str
    shape_digest: str
    code_digest: str
    image_digest: str
    view: str
    renderer: str
    split: str
    source: str
    material: str = ""

    @classmethod
    def create(cls, id, *, shape: bytes, code: str, image: bytes, view,
               renderer, split, source, material=""):
        digest = lambda value: hashlib.sha256(value).hexdigest()
        return cls(id, digest(shape), digest(code.encode()), digest(image),
                   view, renderer, split, source, material)


def audit_manifests(manifests):
    issues = []
    by_shape = {}
    for item in manifests:
        if item.split not in {"train", "validation", "test"}:
            issues.append((item.id, "invalid-split"))
        by_shape.setdefault(item.shape_digest, set()).add(item.split)
    issues.extend((digest, "cross-split-shape-leakage")
                  for digest, splits in sorted(by_shape.items()) if len(splits) > 1)
    return tuple(issues)
