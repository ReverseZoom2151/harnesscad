"""Digest-bound multimodal CAD observations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping


@dataclass(frozen=True)
class CADObservation:
    state_digest: str
    geometry: Mapping[str, object]
    renders: Mapping[str, bytes]
    entity_ids: tuple[str, ...] = ()

    def canonical_json(self) -> str:
        payload = {
            "state_digest": self.state_digest,
            "geometry": self.geometry,
            "renders": {key: hashlib.sha256(value).hexdigest()
                        for key, value in sorted(self.renders.items())},
            "entity_ids": sorted(self.entity_ids),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def require_current(self, current_digest: str) -> None:
        if current_digest != self.state_digest:
            raise RuntimeError("stale CAD observation")
