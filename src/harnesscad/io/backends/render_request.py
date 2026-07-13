"""Render-request canonicalisation and cache keys (deterministic, stdlib-only).

Ported from CadHub's ``openScadController.ts`` / ``curvController.ts``. The
interesting trick there is a one-liner with a comment: before the render request
is sent, every camera float is rounded to one decimal *"to give our caching a
chance to sometimes work"*. A raw orbit camera emits a new float every frame, so
an un-rounded request is never cache-identical to the previous one; quantising
the camera turns a continuous stream of requests into a small set of repeated
ones, and the expensive render (OpenSCAD/curv in a container) is served from
cache. The same reasoning applies to the viewport size (multiplied by the device
pixel ratio and rounded) and to which settings are part of the key at all --
curv ignores camera and parameters entirely, so including them would fragment
its cache for no reason.

The harness had a plan-level cache key for OpenSCAD CLI exports
(``fabrication/t2cdean_openscad_export.plan_cache_key``: source + format +
defines) but nothing that models a *preview* request -- camera, viewport,
parameters -- nor the quantisation that makes such a key hit. This module adds
both, per language, driven by ``adapters.cadhub_language_registry``.

Deterministic: fixed rounding, sorted JSON, sha256. No clock, no I/O.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from harnesscad.io.adapters.language_registry import PARAMS_NONE, get as get_language

# Which settings each language's render actually depends on. Anything else is
# dropped from the cache key so it cannot fragment the cache.
_CAMERA_LANGUAGES = frozenset({"openscad"})
_SIZE_LANGUAGES = frozenset({"openscad", "curv"})

CAMERA_DECIMALS = 1


@dataclass(frozen=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_list(self) -> list:
        return [self.x, self.y, self.z]


@dataclass(frozen=True)
class Camera:
    position: Vec3 = field(default_factory=Vec3)
    rotation: Vec3 = field(default_factory=Vec3)
    dist: float = 200.0


@dataclass(frozen=True)
class Viewport:
    width: int
    height: int


def round_to(value: float, decimals: int = CAMERA_DECIMALS) -> float:
    """Half-up rounding to ``decimals`` places, mirroring CadHub's ``round1dec``.

    ``Math.round((n + EPSILON) * 10) / 10`` in JS -- half-up, not Python's
    banker's rounding, so 0.25 -> 0.3 (not 0.2). Negative values round away from
    zero symmetrically, keeping the quantisation grid uniform.
    """
    factor = 10 ** decimals
    scaled = value * factor
    sign = -1.0 if scaled < 0 else 1.0
    rounded = int(abs(scaled) + 0.5) * sign
    out = rounded / factor
    return int(out) if float(out).is_integer() else out


def quantize_camera(camera: Camera, decimals: int = CAMERA_DECIMALS) -> Camera:
    """Snap every camera float to the caching grid."""

    def snap(vec: Vec3) -> Vec3:
        return Vec3(
            round_to(vec.x, decimals),
            round_to(vec.y, decimals),
            round_to(vec.z, decimals),
        )

    return Camera(
        position=snap(camera.position),
        rotation=snap(camera.rotation),
        dist=round_to(camera.dist, decimals),
    )


def device_size(viewport: Viewport, pixel_ratio: float = 1.0) -> Tuple[int, int]:
    """Physical render size: CSS size * device pixel ratio, rounded to ints."""
    if viewport.width <= 0 or viewport.height <= 0:
        raise ValueError("viewport must be positive")
    if pixel_ratio <= 0:
        raise ValueError("pixel_ratio must be positive")
    return (
        int(round_to(viewport.width * pixel_ratio, 0)),
        int(round_to(viewport.height * pixel_ratio, 0)),
    )


def camera_args(camera: Camera) -> str:
    """OpenSCAD's ``--camera=px,py,pz,rx,ry,rz,dist`` argument value."""
    snapped = quantize_camera(camera)
    values = snapped.position.as_list() + snapped.rotation.as_list() + [snapped.dist]
    return ",".join(_fmt(v) for v in values)


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def canonical_request(
    language: str,
    source: str,
    *,
    camera: Optional[Camera] = None,
    viewport: Optional[Viewport] = None,
    pixel_ratio: float = 1.0,
    parameters: Optional[Mapping[str, Any]] = None,
    artifact: str = "image",
    view_all: bool = False,
) -> Dict[str, Any]:
    """The cache-normalised render request for ``language``.

    Settings the language cannot consume are dropped (curv has no parameters and
    takes no camera; jscad renders in-process at any size), floats are quantised,
    and mappings are key-sorted -- so two requests that will render identically
    produce identical dicts.
    """
    spec = get_language(language)
    request: Dict[str, Any] = {
        "language": spec.name,
        "artifact": artifact,
        "source": source,
    }
    if spec.name in _SIZE_LANGUAGES and viewport is not None:
        width, height = device_size(viewport, pixel_ratio)
        request["size"] = {"x": width, "y": height}
    if spec.name in _CAMERA_LANGUAGES and camera is not None:
        snapped = quantize_camera(camera)
        request["camera"] = {
            "position": snapped.position.as_list(),
            "rotation": snapped.rotation.as_list(),
            "dist": snapped.dist,
        }
        request["viewAll"] = bool(view_all)
    if spec.params != PARAMS_NONE and parameters:
        request["parameters"] = {k: parameters[k] for k in sorted(parameters)}
    return request


def cache_key(request: Mapping[str, Any]) -> str:
    """sha256 over the canonical JSON encoding of a canonical request."""
    blob = json.dumps(request, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def request_key(language: str, source: str, **kwargs: Any) -> str:
    """Convenience: canonicalise then hash."""
    return cache_key(canonical_request(language, source, **kwargs))


def same_render(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Do two canonical requests denote the same render?"""
    return cache_key(a) == cache_key(b)
