"""Port- and mate-based assembly interfaces with deterministic closed-form transforms.

This module implements the *interface-centric* assembly abstraction shared by three
recent text-to-CAD assembly papers, distilling only their deterministic geometric
core (no model calls):

* **ASSEMCAD** (Dong et al., 2026), Sec. 4 -- an assembly is a set of typed parts
  connected by *typed mates* between *geometry-backed ports*. Each port is a local
  coordinate frame with a semantic type; a mate is admissible only when the two
  ports it joins are *type-compatible* (their Definition 3). The library "computes
  each mate as a closed-form rigid-body transform" (their Eq. 2)::

      T = L_b . R_flip . R_theta . L_c^{-1}

  where ``L_b`` / ``L_c`` are the base/incoming port frames in SE(3), ``R_flip`` is a
  180-degree rotation about the local x-axis (so the two z-axes become anti-parallel
  for face-to-face contact) and ``R_theta`` is an optional twist about the shared
  z-axis. "This single matrix multiplication produces byte-identical results across
  runs" -- exactly the determinism this harness wants.

* **ArtiCAD** (Shui et al., 2026), Sec. 4.1 -- the *Connector Contract*
  ``c = (n, o, z, x, l)``: a named attachment point carrying a local frame and a
  semantic label. "By fixing connectors early, assembly reduces to deterministic
  frame alignment." A ``Port`` here IS that connector.

* **ASSEMCAD** Sec. 4.4 / **ArtiCAD** Sec. 3 -- each mate type removes a fixed number
  of relative degrees of freedom (their axiom C-02: a coaxial constraint "removes two
  translational and two rotational DOF, leaving one rotational and one translational
  DOF"). :data:`MATE_DOF_REMOVED` tabulates this, consumed by
  :mod:`harnesscad.domain.geometry.assembly.mobility`.

The transform math is self-contained (row-major 4x4 tuples, stdlib only). It is a
sibling of :mod:`harnesscad.domain.geometry.kinematics.joint_transform` (which aligns
two *axes* for a JoinABLe joint); this module aligns two *typed port frames* and adds
the port-type compatibility gate and the DOF bookkeeping that the joint module lacks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

__all__ = [
    "PORT_TYPES",
    "MATE_TYPES",
    "MATE_DOF_REMOVED",
    "CONTACT_MATES",
    "PORT_MATE_COMPATIBILITY",
    "Port",
    "Mate",
    "port_frame",
    "invert_frame",
    "compose",
    "flip_x_frame",
    "twist_z_frame",
    "mate_transform",
    "ports_compatible",
    "mate_is_valid",
    "transform_point",
]

# Twelve port types (ASSEMCAD Sec. 4.2) and seven mate types (their Table 2).
PORT_TYPES: tuple[str, ...] = (
    "flat_face", "bore", "shaft_seat", "boss", "cavity", "rim",
    "thread_male", "thread_female", "gear_teeth", "snap", "axis", "generic",
)

MATE_TYPES: tuple[str, ...] = (
    "face_to_face", "coaxial", "coaxial_face", "gear_mesh",
    "press_fit", "thread_engage", "snap_to_face",
)

# Relative DOF removed by each mate type (of the free 6 between two rigid bodies).
# ASSEMCAD axiom C-02 / F-01: coaxial removes 2 translational + 2 rotational = 4,
# leaving one slide + one spin. face_to_face fixes the two in-plane offsets are free
# but the normal gap, in-plane rotation are pinned per the mate's option set; we use
# the standard mechanical-mate DOF counts.
MATE_DOF_REMOVED: Mapping[str, int] = {
    "face_to_face": 3,    # coincident plane: 1 translation (normal) + 2 rotations
    "coaxial": 4,         # collinear axes: 2 translation + 2 rotation
    "coaxial_face": 5,    # coaxial + axial seat: leaves 1 rotation (revolute)
    "gear_mesh": 5,       # meshed spur pair: coupled single rotation
    "press_fit": 6,       # interference fit: fully constrained
    "thread_engage": 5,   # helical pair: 1 coupled screw DOF
    "snap_to_face": 6,    # snap seat: fully constrained
}

# Mate types whose geometry intentionally interferes (ASSEMCAD Eq. 3): a positive
# intersection volume here is *expected*, not a clash.
CONTACT_MATES: frozenset[str] = frozenset({"gear_mesh", "press_fit", "thread_engage"})

# Port-mate compatibility relation (ASSEMCAD Definition 3): the unordered set of port
# types each mate may join. A mate is admissible only if both its endpoints' types
# fall in the mate's set.
PORT_MATE_COMPATIBILITY: Mapping[str, frozenset[str]] = {
    "face_to_face": frozenset({"flat_face", "rim", "boss"}),
    "coaxial": frozenset({"bore", "shaft_seat", "axis"}),
    "coaxial_face": frozenset({"bore", "shaft_seat", "axis", "flat_face"}),
    "gear_mesh": frozenset({"gear_teeth", "axis"}),
    "press_fit": frozenset({"bore", "shaft_seat", "boss"}),
    "thread_engage": frozenset({"thread_male", "thread_female"}),
    "snap_to_face": frozenset({"snap", "flat_face", "rim"}),
}


def _almost_zero(x: float) -> bool:
    return abs(x) < 1e-12


def _normalize(v: Sequence[float]) -> tuple[float, float, float]:
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@dataclass(frozen=True)
class Port:
    """A typed attachment point: ArtiCAD's connector ``(n, o, z, x, l)``.

    ``z`` is the primary axis (rotation / slide / normal direction); ``x`` an
    orthogonal reference; the frame's y-axis is ``z x x`` (right-handed). ``kind``
    must be one of :data:`PORT_TYPES`.
    """

    name: str
    origin: tuple[float, float, float]
    z_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    x_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    kind: str = "generic"
    label: str = ""

    def __post_init__(self):
        if not self.name:
            raise ValueError("port requires a name")
        if self.kind not in PORT_TYPES:
            raise ValueError(f"unknown port type {self.kind!r}")
        z = _normalize(self.z_axis)
        x = self.x_axis
        # Gram-Schmidt: make x orthogonal to z; if x collapses onto z, pick a fallback.
        proj = _dot(x, z)
        x_ortho = (x[0] - proj * z[0], x[1] - proj * z[1], x[2] - proj * z[2])
        if math.sqrt(sum(c * c for c in x_ortho)) < 1e-9:
            fallback = (1.0, 0.0, 0.0) if abs(z[0]) < 0.9 else (0.0, 1.0, 0.0)
            proj = _dot(fallback, z)
            x_ortho = tuple(fallback[i] - proj * z[i] for i in range(3))
        object.__setattr__(self, "z_axis", z)
        object.__setattr__(self, "x_axis", _normalize(x_ortho))


@dataclass(frozen=True)
class Mate:
    """A typed relation between two ports, optionally twisted about the shared axis."""

    kind: str
    base_port: str
    incoming_port: str
    twist_rad: float = 0.0
    axioms: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.kind not in MATE_TYPES:
            raise ValueError(f"unknown mate type {self.kind!r}")
        if self.base_port == self.incoming_port:
            raise ValueError("a mate must join two distinct ports")


def port_frame(port: Port) -> tuple[float, ...]:
    """Return the port's SE(3) frame as a row-major 4x4 tuple (columns = x, y, z)."""
    z = port.z_axis
    x = port.x_axis
    y = _cross(z, x)
    o = port.origin
    return (
        x[0], y[0], z[0], o[0],
        x[1], y[1], z[1], o[1],
        x[2], y[2], z[2], o[2],
        0.0, 0.0, 0.0, 1.0,
    )


def compose(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    """Matrix product ``a @ b`` of two row-major 4x4 tuples."""
    out = [0.0] * 16
    for r in range(4):
        for c in range(4):
            out[r * 4 + c] = sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4))
    return tuple(out)


def invert_frame(m: Sequence[float]) -> tuple[float, ...]:
    """Inverse of a rigid frame (rotation transpose + rotated translation)."""
    r = [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]]
    t = (m[3], m[7], m[11])
    # R^-1 = R^T, t' = -R^T t
    rt = [[r[0][0], r[1][0], r[2][0]],
          [r[0][1], r[1][1], r[2][1]],
          [r[0][2], r[1][2], r[2][2]]]
    tp = tuple(-sum(rt[i][k] * t[k] for k in range(3)) for i in range(3))
    return (
        rt[0][0], rt[0][1], rt[0][2], tp[0],
        rt[1][0], rt[1][1], rt[1][2], tp[1],
        rt[2][0], rt[2][1], rt[2][2], tp[2],
        0.0, 0.0, 0.0, 1.0,
    )


def flip_x_frame() -> tuple[float, ...]:
    """R_flip: 180-degree rotation about the local x-axis (z becomes anti-parallel)."""
    return (
        1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def twist_z_frame(angle_rad: float) -> tuple[float, ...]:
    """R_theta: rotation about the local z-axis by ``angle_rad``."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return (
        c, -s, 0.0, 0.0,
        s, c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def mate_transform(base: Port, incoming: Port, twist_rad: float = 0.0,
                   flip: bool = True) -> tuple[float, ...]:
    """Closed-form rigid transform placing ``incoming`` onto ``base`` (ASSEMCAD Eq. 2).

    ``T = L_b . R_flip . R_theta . L_c^{-1}``. Applying ``T`` to the incoming part's
    local geometry seats its port frame against the (already world-placed) base port.
    Deterministic: pure matrix products, no solver iteration.
    """
    lb = port_frame(base)
    lc_inv = invert_frame(port_frame(incoming))
    inner = compose(twist_z_frame(twist_rad), lc_inv)
    if flip:
        inner = compose(flip_x_frame(), inner)
    return compose(lb, inner)


def transform_point(m: Sequence[float], p: Sequence[float]) -> tuple[float, float, float]:
    """Apply a 4x4 row-major transform to a 3D point."""
    return (
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
        m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
        m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
    )


def ports_compatible(mate_kind: str, a: Port, b: Port) -> bool:
    """Port-mate compatibility (ASSEMCAD Definition 3)."""
    if mate_kind not in PORT_MATE_COMPATIBILITY:
        raise ValueError(f"unknown mate type {mate_kind!r}")
    allowed = PORT_MATE_COMPATIBILITY[mate_kind]
    return a.kind in allowed and b.kind in allowed


def mate_is_valid(mate: Mate, ports: Mapping[str, Port]) -> tuple[bool, str]:
    """Check a mate against a port table: endpoints resolve and are type-compatible.

    Returns ``(ok, reason)``; ``reason`` is empty on success.
    """
    if mate.base_port not in ports:
        return False, f"unresolved base port {mate.base_port!r}"
    if mate.incoming_port not in ports:
        return False, f"unresolved incoming port {mate.incoming_port!r}"
    a, b = ports[mate.base_port], ports[mate.incoming_port]
    if not ports_compatible(mate.kind, a, b):
        return False, (
            f"port types ({a.kind}, {b.kind}) incompatible with mate {mate.kind!r}"
        )
    return True, ""
