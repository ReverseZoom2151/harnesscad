"""Digest-bound multi-view geometry prompt bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class PromptView:
    id: str
    camera: tuple[float, ...]
    rgb: bytes
    mask: tuple[tuple[bool, ...], ...]
    points: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class GeometryPrompt:
    mesh_digest: str
    views: tuple[PromptView, ...]
    version: str = "geometry-prompt-v1"

    def validate(self):
        issues = []
        if len({view.id for view in self.views}) != len(self.views):
            issues.append("duplicate-view-id")
        if len({view.camera for view in self.views}) != len(self.views):
            issues.append("duplicate-camera")
        for view in self.views:
            height = len(view.mask)
            width = len(view.mask[0]) if height else 0
            if not width or any(len(row) != width for row in view.mask):
                issues.append(f"{view.id}:invalid-mask")
                continue
            if not any(any(row) for row in view.mask):
                issues.append(f"{view.id}:empty-mask")
            for x, y in view.points:
                if not (0 <= y < height and 0 <= x < width and view.mask[y][x]):
                    issues.append(f"{view.id}:point-outside-mask")
        return tuple(issues)

    @property
    def digest(self):
        payload = {"mesh": self.mesh_digest, "version": self.version, "views": [
            {"id": view.id, "camera": view.camera,
             "rgb": hashlib.sha256(view.rgb).hexdigest(),
             "mask": view.mask, "points": view.points} for view in self.views]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
