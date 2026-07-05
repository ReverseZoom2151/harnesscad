"""Assembly / mate solver — the third verifier family the blueprint reserves.

``verify.py`` carries the TODO ``# - (later) assembly solver (mates / DOF /
collision)`` next to the constraint solver and the B-rep validity check. This
module fills the *mates / DOF* half (its sibling :mod:`checks_interference`
fills *collision*).

Where the sketch constraint solver reasons about a single sketch's degrees of
freedom, an assembly solver reasons about the *rigid-body* degrees of freedom of
a set of placed parts coupled by mates (joints). Each part is a free rigid body
with 6 DOF (3 translation + 3 rotation); each mate removes some of them:

  =============  =====  =========================================================
  mate           DOF    physical joint
  =============  =====  =========================================================
  rigid           6     fully welded / bonded (no relative motion)
  revolute        5     hinge — one rotation left
  slider          5     prismatic — one translation left
  cylindrical     4     rotate *and* slide about one axis
  planar          3     free to slide in a plane and spin about its normal
  =============  =====  =========================================================

The residual mobility of the whole assembly is the Grübler-style count

    residual_dof = 6 * n_parts - 6 * n_grounded - sum(mate DOF removed)

(``n_grounded`` defaults to 0, so the bare formula matches the blueprint note;
a caller may pin parts to make the count reflect motion relative to a frame).

  * ``residual_dof < 0``  -> **over-constrained**  (ERROR): mates fight each
    other; the assembly is generally unsolvable / has redundant constraints.
  * ``residual_dof > 0``  -> **under-constrained** (WARNING): the assembly (or a
    sub-chain) still floats; fine for a mechanism, suspicious for a fixture.
  * ``residual_dof == 0`` -> fully constrained (INFO).

On top of the mobility count, each mate carries an optional *satisfaction
residual*: if the mate names a coincident point on each part (``point_a`` /
``point_b`` in the parts' local frames) and the model carries the parts'
placement ``transforms``, we place both points in world space and measure their
gap. A gap beyond the mate tolerance means the current placement does **not**
satisfy the mate -> ERROR ``unsatisfied-mate``.

Standalone by design, exactly like :class:`checks_geometry.BRepValidityCheck`
and :class:`checks_dfm.DFMCheck`: this is NOT wired into
:func:`verify.default_verifiers` (that would be a circular import, and the
assembly stage is opt-in anyway). A caller adds it explicitly via
:func:`with_assembly`. The verifier reads ``query('assembly')`` and degrades
gracefully — an INFO skip, never a crash — when the backend does not answer it
(the stub returns ``{}``), mirroring :mod:`contract` / :mod:`checks_dfm`.

The same analysis is available *without* a backend: build an
:class:`AssemblyModel` (from code or :meth:`AssemblyModel.from_dict`) and hand
it to :meth:`AssemblyCheck.check_model` (or call
:func:`assembly_diagnostics`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from verify import Diagnostic, Severity, VerifyReport


# --------------------------------------------------------------------------- #
# Mate degree-of-freedom table
# --------------------------------------------------------------------------- #
#: DOF *removed* by each mate type (6 = fully fixed, 0 = free).
MATE_DOF: Dict[str, int] = {
    "rigid": 6,        # welded / bonded — no relative motion
    "fixed": 6,        # alias for rigid
    "revolute": 5,     # hinge — 1 rotational DOF remains
    "hinge": 5,        # alias for revolute
    "slider": 5,       # prismatic — 1 translational DOF remains
    "prismatic": 5,    # alias for slider
    "cylindrical": 4,  # rotate + slide about one axis — 2 DOF remain
    "planar": 3,       # slide in a plane + spin about its normal — 3 DOF remain
}

#: DOF of a single free rigid body.
BODY_DOF = 6


def mate_dof(kind: str) -> Optional[int]:
    """DOF removed by a mate of type ``kind``; ``None`` if the kind is unknown."""
    return MATE_DOF.get(str(kind).strip().lower())


# --------------------------------------------------------------------------- #
# Data model (usable standalone, without any backend)
# --------------------------------------------------------------------------- #
Vec3 = Tuple[float, float, float]


@dataclass
class Mate:
    """A joint coupling two parts (or one part to ground when ``b is None``).

    ``point_a`` / ``point_b`` are optional coincident anchor points, expressed
    in the *local* frame of parts ``a`` and ``b`` respectively. When both are
    given and the model carries the relevant placement transforms, the mate's
    satisfaction residual is the world-space gap between the two anchors.
    A precomputed ``residual`` (already in world units) overrides that.
    """

    kind: str
    a: str
    b: Optional[str] = None
    point_a: Optional[Vec3] = None
    point_b: Optional[Vec3] = None
    tol: float = 1e-6
    residual: Optional[float] = None
    name: str = ""

    def label(self) -> str:
        base = self.name or f"{self.kind}({self.a}"
        if not self.name:
            base += f"->{self.b})" if self.b is not None else "->ground)"
        return base

    def to_dict(self) -> dict:
        d: dict = {"kind": self.kind, "a": self.a}
        if self.b is not None:
            d["b"] = self.b
        if self.point_a is not None:
            d["point_a"] = list(self.point_a)
        if self.point_b is not None:
            d["point_b"] = list(self.point_b)
        if self.tol != 1e-6:
            d["tol"] = self.tol
        if self.residual is not None:
            d["residual"] = self.residual
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Mate":
        return cls(
            kind=str(d.get("kind", d.get("type", ""))),
            a=str(d.get("a", d.get("part_a", d.get("from", "")))),
            b=(None if d.get("b", d.get("part_b", d.get("to"))) is None
               else str(d.get("b", d.get("part_b", d.get("to"))))),
            point_a=_as_vec3(d.get("point_a")),
            point_b=_as_vec3(d.get("point_b")),
            tol=float(d.get("tol", 1e-6)),
            residual=(None if d.get("residual") is None
                      else float(d["residual"])),
            name=str(d.get("name", "")),
        )


@dataclass
class AssemblyModel:
    """A set of placed parts coupled by mates — the assembly solver's input.

    * ``parts``       — ordered list of part ids.
    * ``mates``       — the joints coupling them.
    * ``transforms``  — optional placement per part id, each
      ``{"translate": [x,y,z], "rotate_deg": [rx,ry,rz]}`` (either key optional).
      Used only for mate-satisfaction residuals.
    * ``grounded``    — part ids pinned to the world frame (each removes 6 DOF).
    """

    parts: List[str] = field(default_factory=list)
    mates: List[Mate] = field(default_factory=list)
    transforms: Dict[str, dict] = field(default_factory=dict)
    grounded: List[str] = field(default_factory=list)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_dict(cls, d: dict) -> "AssemblyModel":
        """Build a model from a plain ``{parts, mates, transforms, grounded}``
        dict — exactly the shape ``query('assembly')`` returns."""
        d = d or {}
        parts: List[str] = []
        for p in d.get("parts", []):
            if isinstance(p, dict):
                parts.append(str(p.get("id", p.get("name", ""))))
            else:
                parts.append(str(p))
        mates = [Mate.from_dict(m) for m in d.get("mates", []) if isinstance(m, dict)]
        transforms = dict(d.get("transforms", {}) or {})
        grounded = [str(g) for g in (d.get("grounded", []) or [])]
        return cls(parts=parts, mates=mates, transforms=transforms,
                   grounded=grounded)

    # -- degree-of-freedom accounting --------------------------------------- #
    def n_parts(self) -> int:
        return len(self.parts)

    def removed_dof(self) -> int:
        """Sum of DOF removed by every recognised mate (unknown mates ignored)."""
        total = 0
        for m in self.mates:
            dof = mate_dof(m.kind)
            if dof is not None:
                total += dof
        return total

    def residual_dof(self) -> int:
        """Grübler-style residual mobility of the whole assembly."""
        grounded = {g for g in self.grounded if g in set(self.parts)}
        return (BODY_DOF * self.n_parts()
                - BODY_DOF * len(grounded)
                - self.removed_dof())

    def classify(self) -> str:
        """``'over'`` / ``'under'`` / ``'well'`` / ``'trivial'``."""
        if self.n_parts() < 2 and not self.mates:
            return "trivial"
        r = self.residual_dof()
        if r < 0:
            return "over"
        if r > 0:
            return "under"
        return "well"

    # -- mate satisfaction -------------------------------------------------- #
    def mate_residual(self, mate: Mate) -> Optional[float]:
        """World-space gap of a mate's coincident anchors, or ``None`` when the
        mate does not declare enough to evaluate (best-effort)."""
        if mate.residual is not None:
            return abs(float(mate.residual))
        if mate.point_a is None or mate.point_b is None:
            return None
        if mate.b is None:
            # part-to-ground: point_b is already a world anchor.
            wa = _place(mate.point_a, self.transforms.get(mate.a))
            wb = tuple(float(v) for v in mate.point_b)
        else:
            wa = _place(mate.point_a, self.transforms.get(mate.a))
            wb = _place(mate.point_b, self.transforms.get(mate.b))
        return _dist(wa, wb)


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class AssemblyCheck:
    """A :class:`verify.Verifier` (``name='assembly'``) for the mate/DOF solver.

    ``check(backend, opdag)`` reads ``query('assembly')`` and returns a
    :class:`verify.VerifyReport`:

      * INFO  ``assembly-skipped``     — backend has no ``'assembly'`` query
        (e.g. the stub) or it is empty; nothing else runs.
      * INFO  ``assembly-trivial``     — fewer than two parts and no mates.
      * INFO  ``assembly-dof``         — the computed residual DOF (always, when
        evaluated).
      * ERROR ``over-constrained``     — residual DOF < 0.
      * WARNING ``under-constrained``  — residual DOF > 0.
      * WARNING ``unknown-mate``       — a mate kind not in :data:`MATE_DOF`
        (excluded from the DOF sum).
      * WARNING ``bad-part-ref``       — a mate references an unknown part.
      * ERROR ``unsatisfied-mate``     — a mate's anchors do not coincide under
        the current placement (gap > tolerance).

    Only ``over-constrained`` and ``unsatisfied-mate`` are ERRORs, so an
    under-constrained mechanism does not flip ``report.ok`` to False.
    """

    name = "assembly"

    def check(self, backend, opdag) -> VerifyReport:
        raw = _query(backend, "assembly")
        if not raw:
            return VerifyReport([_info(
                "assembly-skipped",
                "assembly DOF/mate checks skipped: backend has no 'assembly' "
                "query (only an assembly-aware backend exposes parts + mates).")])
        model = AssemblyModel.from_dict(raw)
        return self.check_model(model)

    def check_model(self, model: AssemblyModel) -> VerifyReport:
        """Run the solver on an :class:`AssemblyModel` directly (no backend)."""
        return VerifyReport(assembly_diagnostics(model))


def assembly_diagnostics(model: AssemblyModel) -> List[Diagnostic]:
    """The full mate/DOF analysis as a flat diagnostic list (backend-free)."""
    diags: List[Diagnostic] = []
    part_set = set(model.parts)

    # Trivial assemblies carry no meaningful mobility story.
    if model.n_parts() < 2 and not model.mates:
        diags.append(_info(
            "assembly-trivial",
            f"assembly has {model.n_parts()} part(s) and no mates: nothing to "
            "solve."))
        return diags

    # Per-mate validation (unknown kinds / dangling part refs).
    for m in model.mates:
        if mate_dof(m.kind) is None:
            diags.append(_warn(
                "unknown-mate",
                f"mate {m.label()} has unknown kind '{m.kind}' "
                f"(known: {', '.join(sorted(MATE_DOF))}); excluded from the DOF "
                "count.", m.name or None))
        for ref in (m.a, m.b):
            if ref is not None and ref not in part_set:
                diags.append(_warn(
                    "bad-part-ref",
                    f"mate {m.label()} references unknown part '{ref}'.", ref))

    # Whole-assembly mobility count.
    residual = model.residual_dof()
    diags.append(_info(
        "assembly-dof",
        f"residual assembly DOF = {residual} "
        f"(6 x {model.n_parts()} parts"
        + (f" - 6 x {len([g for g in model.grounded if g in part_set])} grounded"
           if model.grounded else "")
        + f" - {model.removed_dof()} removed by mates)."))

    cls = model.classify()
    if cls == "over":
        diags.append(_err(
            "over-constrained",
            f"assembly is over-constrained (residual DOF = {residual} < 0): "
            "mates remove more freedom than the parts have — redundant or "
            "conflicting mates make the assembly generally unsolvable."))
    elif cls == "under":
        diags.append(_warn(
            "under-constrained",
            f"assembly is under-constrained (residual DOF = {residual} > 0): "
            "at least one part or sub-chain still floats. Expected for a "
            "mechanism; add mates (or ground a part) to fully fix a fixture."))

    # Per-mate satisfaction residuals (best-effort, needs anchors + transforms).
    for m in model.mates:
        r = model.mate_residual(m)
        if r is None:
            continue
        if r > m.tol:
            diags.append(_err(
                "unsatisfied-mate",
                f"mate {m.label()} is not satisfied: anchor gap {r:.6g} exceeds "
                f"tolerance {m.tol:g} under the current placement.",
                m.name or None))

    return diags


# --------------------------------------------------------------------------- #
# Wiring helper
# --------------------------------------------------------------------------- #
def with_assembly(verifiers) -> List:
    """Return a new verifier list with an :class:`AssemblyCheck` appended.

    Mirrors :func:`checks_dfm.with_dfm` — a caller opts the assembly solver into
    the default set without editing ``verify.py``::

        from verify import default_verifiers
        from checks_assembly import with_assembly
        verifiers = with_assembly(default_verifiers())
    """
    return list(verifiers) + [AssemblyCheck()]


# --------------------------------------------------------------------------- #
# Geometry helpers (stdlib only)
# --------------------------------------------------------------------------- #
def _as_vec3(v) -> Optional[Vec3]:
    if v is None:
        return None
    seq = list(v)
    if len(seq) < 3:
        seq = seq + [0.0] * (3 - len(seq))
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _rotate(point: Vec3, axis: str, degrees: float) -> Vec3:
    if not degrees:
        return point
    r = math.radians(degrees)
    c, s = math.cos(r), math.sin(r)
    x, y, z = point
    if axis == "x":
        return (x, y * c - z * s, y * s + z * c)
    if axis == "y":
        return (x * c + z * s, y, -x * s + z * c)
    return (x * c - y * s, x * s + y * c, z)


def _place(point: Vec3, transform: Optional[dict]) -> Vec3:
    """Apply a ``{translate, rotate_deg}`` placement to a local point."""
    p = (float(point[0]), float(point[1]), float(point[2]))
    if not transform:
        return p
    rot = transform.get("rotate_deg", transform.get("rotate", [0.0, 0.0, 0.0]))
    rot = _as_vec3(rot) or (0.0, 0.0, 0.0)
    p = _rotate(p, "x", rot[0])
    p = _rotate(p, "y", rot[1])
    p = _rotate(p, "z", rot[2])
    tr = transform.get("translate", transform.get("translate_mm",
                                                   [0.0, 0.0, 0.0]))
    tr = _as_vec3(tr) or (0.0, 0.0, 0.0)
    return (p[0] + tr[0], p[1] + tr[1], p[2] + tr[2])


def _dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# --------------------------------------------------------------------------- #
# Graceful-degradation helpers (mirror contract.py / checks_dfm.py)
# --------------------------------------------------------------------------- #
def _query(backend, q: str) -> Optional[dict]:
    """Read a backend query, returning None when the backend does not answer it
    (backends return {} for unknown queries) so callers can INFO-skip."""
    try:
        result = backend.query(q)
    except Exception:  # noqa: BLE001 - an unsupported query must degrade, not crash
        return None
    return result or None


def _err(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.ERROR, code, msg, where)


def _warn(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.WARNING, code, msg, where)


def _info(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.INFO, code, msg, where)
