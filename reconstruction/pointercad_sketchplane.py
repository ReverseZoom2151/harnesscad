"""Pointer-based sketch-plane coordinate-system construction (Pointer-CAD Sec. 10.2).

Where DeepCAD encodes a sketch plane as a 3-angle + 3-translation regression,
Pointer-CAD *selects* the plane with a face pointer and then builds a deterministic
local frame ``UVW`` on it (Sec. 10.2, Table 14). This reformulates plane placement
from continuous rotation regression into a discrete selection plus a fixed geometric
recipe, "reducing the search space and mitigating misalignment" (Sec. 4).

The construction, given the selected face's normal and a ``<dr>`` direction label:

  1. **W'** = the face normal, sign-chosen so it has a positive dot product with the
     world direction ``n`` named by ``<dr>`` (Table 14 primary direction).
  2. **U'** = the *auxiliary* direction ``d`` (Table 14) projected onto the plane and
     normalised.
  3. **V'** = ``W' x U'`` (right-hand rule), completing an orthonormal basis.
  4. A counter-clockwise in-plane rotation by angle ``alpha`` about ``W`` yields the
     final ``UVW``; an optional isotropic scale may be applied to mitigate
     quantisation error.

Pure stdlib ``math`` on length-3 tuples. This is the deterministic geometry only;
the face pointer itself comes from :mod:`reconstruction.pointercad_pointer`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]

# Table 14: <dr> symbol -> (primary direction, auxiliary direction).
DIRECTION_MAP: dict[str, tuple[Vec3, Vec3]] = {
    "X+": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "X-": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "Y+": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "Y-": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    "Z+": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    "Z-": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
}


class SketchPlaneError(ValueError):
    pass


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Vec3) -> Vec3:
    n = _norm(a)
    if n == 0.0:
        raise SketchPlaneError("cannot normalise a zero vector")
    return (a[0] / n, a[1] / n, a[2] / n)


@dataclass(frozen=True)
class SketchFrame:
    """An orthonormal local frame on a sketch plane. ``w`` is the plane normal."""
    u: Vec3
    v: Vec3
    w: Vec3
    scale: float = 1.0

    def to_world(self, x: float, y: float) -> Vec3:
        """Lift a 2D sketch point ``(x, y)`` (in ``u/v``) to world space (no origin)."""
        sx, sy = x * self.scale, y * self.scale
        return (self.u[0] * sx + self.v[0] * sy,
                self.u[1] * sx + self.v[1] * sy,
                self.u[2] * sx + self.v[2] * sy)


def direction_vectors(symbol: str) -> tuple[Vec3, Vec3]:
    """Return ``(primary, auxiliary)`` world vectors for a ``<dr>`` symbol."""
    if symbol not in DIRECTION_MAP:
        raise SketchPlaneError(f"unknown direction symbol {symbol!r}")
    return DIRECTION_MAP[symbol]


def orient_normal(face_normal: Vec3, symbol: str) -> Vec3:
    """Sign-fix ``face_normal`` to have a positive dot with the ``<dr>`` primary dir."""
    primary, _ = direction_vectors(symbol)
    w = _unit(face_normal)
    if _dot(w, primary) < 0.0:
        w = _scale(w, -1.0)
    return w


def build_frame(
    face_normal: Vec3,
    direction_symbol: str,
    rotation_deg: float = 0.0,
    scale: float = 1.0,
) -> SketchFrame:
    """Construct the ``UVW`` sketch frame (Sec. 10.2 steps 1-4).

    ``face_normal`` is the selected face's normal; ``direction_symbol`` is the
    ``<dr>`` label; ``rotation_deg`` is the counter-clockwise in-plane roll about
    ``W``; ``scale`` is the optional isotropic scale.
    """
    if scale <= 0.0:
        raise SketchPlaneError("scale must be positive")
    w = orient_normal(face_normal, direction_symbol)
    _, aux = direction_vectors(direction_symbol)

    # Project auxiliary direction onto the plane (remove its W component).
    aux_in_plane = _sub(aux, _scale(w, _dot(aux, w)))
    if _norm(aux_in_plane) < 1e-12:
        raise SketchPlaneError("auxiliary direction is parallel to the face normal")
    u_prime = _unit(aux_in_plane)
    v_prime = _unit(_cross(w, u_prime))  # right-hand rule: W x U'

    # Counter-clockwise in-plane rotation by alpha about W.
    a = math.radians(rotation_deg)
    ca, sa = math.cos(a), math.sin(a)
    u = (u_prime[0] * ca + v_prime[0] * sa,
         u_prime[1] * ca + v_prime[1] * sa,
         u_prime[2] * ca + v_prime[2] * sa)
    v = (-u_prime[0] * sa + v_prime[0] * ca,
         -u_prime[1] * sa + v_prime[1] * ca,
         -u_prime[2] * sa + v_prime[2] * ca)
    return SketchFrame(u=_unit(u), v=_unit(v), w=w, scale=scale)


def is_orthonormal(frame: SketchFrame, tol: float = 1e-9) -> bool:
    """Check the frame's axes are mutually orthogonal, unit, and right-handed."""
    axes = (frame.u, frame.v, frame.w)
    for a in axes:
        if abs(_norm(a) - 1.0) > tol:
            return False
    if abs(_dot(frame.u, frame.v)) > tol:
        return False
    if abs(_dot(frame.u, frame.w)) > tol:
        return False
    if abs(_dot(frame.v, frame.w)) > tol:
        return False
    # Right-handed: u x v == w.
    c = _cross(frame.u, frame.v)
    return all(abs(c[i] - frame.w[i]) <= tol for i in range(3))
