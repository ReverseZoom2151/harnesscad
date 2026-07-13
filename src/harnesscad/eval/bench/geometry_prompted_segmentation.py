"""Dataset manifests for geometry-prompted, all-instance segmentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometrySegmentationCase:
    id: str
    image_digest: str
    mesh_digest: str
    prompt_id: str
    instance_mask_digests: tuple[str, ...]
    split: str
    unseen_geometry: bool = False
    appearance: str = ""
    clutter: str = ""


def audit_cases(cases):
    issues, mesh_splits = [], {}
    for case in cases:
        if not case.instance_mask_digests:
            issues.append((case.id, "no-instances"))
        mesh_splits.setdefault(case.mesh_digest, set()).add(case.split)
    issues.extend((mesh, "mesh-split-leakage") for mesh, splits in sorted(mesh_splits.items())
                  if len(splits) > 1)
    return tuple(issues)
