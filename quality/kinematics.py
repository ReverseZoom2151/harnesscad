"""Mechanism kinematics / motion-behaviour validator — the behavioural
counterpart to the static assembly DOF graph.

Where :mod:`checks_assembly` answers *"can these parts be placed at all?"*
(a static, Gruebler-style mobility count that flags over-/under-constraint),
this module answers *"does the assembled mechanism move the way it is meant
to?"*. Any mechanical system abstracts to a **motion-constraint graph**: parts
are rigid bodies, mates are kinematic joints, and each joint permits a small set
of relative motions (rotations / translations, and — critically — *directions*).
Functional intent is a statement about that graph:

  * a ratchet's pawl joint is a *one-way* revolute — reverse rotation is
    forbidden;
  * a landing-gear actuator is a slider with a hard travel limit;
  * a four-bar linkage must have exactly **one** degree of freedom (mobility 1)
    so a single input crank drives a determinate output.

:class:`MotionSpec` captures that intent (permitted DOF, forbidden directions,
travel / angle limits). :class:`MechanismGraph` builds the motion-constraint
graph from an :class:`checks_assembly.AssemblyModel` and computes mobility via
the Kutzbach–Gruebler criterion

    M = d * (n_links - 1) - sum_over_joints( d - f_joint )

where ``d`` is the working DOF of the space (6 spatial, 3 planar), ``n_links``
counts the parts (one link is the fixed frame — that is the ``-1``), and
``f_joint`` is the joint's freedom (a revolute = 1, a cylindrical = 2, …).
:class:`KinematicsCheck` is a standalone :class:`verify.Verifier` that reads
``query('assembly')`` and validates a :class:`MotionSpec` against the graph.

This is a **mobility-and-screw heuristic**, not a physics engine: it reasons
about the rigid-body mobility count and the per-joint *permitted / forbidden*
motion sets (a screw-theory-style twist sign per joint). It does not integrate
equations of motion or detect motion that only manifests through a full loop
closure; it catches the functional-intent violations that are visible in the
constraint graph itself — a locked mechanism that must move, a free mechanism
that must be fixed, and a joint that permits a direction the intent forbids.

Standalone by design, exactly like :class:`checks_assembly.AssemblyCheck`: it is
NOT wired into :func:`verify.default_verifiers`. A caller adds it explicitly via
:func:`with_kinematics`, or runs it backend-free through
:meth:`KinematicsCheck.check_mechanism` / :func:`kinematics_diagnostics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

from verifiers.assembly import BODY_DOF, MATE_DOF, AssemblyModel, Mate
from verifiers.verify import Diagnostic, Severity, VerifyReport


# --------------------------------------------------------------------------- #
# Motion vocabulary — directional twists a joint may permit
# --------------------------------------------------------------------------- #
#: Direction tokens. A joint's *allowed* set is the relative motions it permits;
#: a :class:`MotionSpec` forbids a subset of these per joint.
ROT_POS = "rot_pos"      # forward / CCW rotation (a ratchet's driven direction)
ROT_NEG = "rot_neg"      # reverse / CW rotation (what a ratchet forbids)
TRANS_POS = "trans_pos"  # extend / advance along the slide axis
TRANS_NEG = "trans_neg"  # retract along the slide axis

ALL_DIRECTIONS: FrozenSet[str] = frozenset(
    {ROT_POS, ROT_NEG, TRANS_POS, TRANS_NEG})

_ROT: FrozenSet[str] = frozenset({ROT_POS, ROT_NEG})
_TRANS: FrozenSet[str] = frozenset({TRANS_POS, TRANS_NEG})

#: Human aliases accepted in ``from_dict`` for the direction tokens.
_DIRECTION_ALIASES: Dict[str, str] = {
    "reverse": ROT_NEG, "backward": ROT_NEG, "cw": ROT_NEG, "-": ROT_NEG,
    "forward": ROT_POS, "ccw": ROT_POS, "+": ROT_POS,
    "retract": TRANS_NEG, "back": TRANS_NEG,
    "extend": TRANS_POS, "advance": TRANS_POS,
}


def normalize_direction(token: str) -> str:
    """Map a direction token (or a friendly alias) to its canonical form."""
    t = str(token).strip().lower()
    if t in ALL_DIRECTIONS:
        return t
    return _DIRECTION_ALIASES.get(t, t)


# --------------------------------------------------------------------------- #
# Per-joint kinematics table — freedom + permitted directions
# --------------------------------------------------------------------------- #
#: Directions each joint kind permits by default. Standard mates are symmetric
#: (a revolute turns both ways); the *one-way* variants (ratchet family) are the
#: asymmetric joints that encode functional intent directly in the graph.
JOINT_MOTION: Dict[str, FrozenSet[str]] = {
    "rigid": frozenset(),          # welded — no relative motion
    "fixed": frozenset(),          # alias for rigid
    "revolute": _ROT,              # hinge — turns both ways
    "hinge": _ROT,                 # alias for revolute
    "slider": _TRANS,              # prismatic — slides both ways
    "prismatic": _TRANS,           # alias for slider
    "cylindrical": _ROT | _TRANS,  # rotate + slide, both ways
    "planar": _ROT | _TRANS,       # slide in plane + spin
    "ratchet": frozenset({ROT_POS}),          # one-way revolute (pawl)
    "one_way": frozenset({ROT_POS}),          # alias for ratchet
    "ratchet_linear": frozenset({TRANS_POS}),  # one-way slider (backstop)
}

#: DOF *removed* by the extra one-way joint kinds (revolute-/slider-like). The
#: standard kinds reuse :data:`checks_assembly.MATE_DOF` so the two modules keep
#: one mate vocabulary.
_EXTRA_REMOVED: Dict[str, int] = {
    "ratchet": 5, "one_way": 5, "ratchet_linear": 5,
}


def joint_removed_dof(kind: str) -> Optional[int]:
    """DOF removed by a joint of type ``kind`` (spatial). ``None`` if unknown."""
    k = str(kind).strip().lower()
    if k in MATE_DOF:
        return MATE_DOF[k]
    return _EXTRA_REMOVED.get(k)


def joint_freedom(kind: str) -> Optional[int]:
    """Spatial freedom of a joint (``6 - removed``); ``None`` if unknown."""
    removed = joint_removed_dof(kind)
    return None if removed is None else BODY_DOF - removed


def joint_directions(kind: str) -> FrozenSet[str]:
    """Relative-motion directions a joint permits (empty for unknown/rigid)."""
    return JOINT_MOTION.get(str(kind).strip().lower(), frozenset())


# --------------------------------------------------------------------------- #
# The motion-constraint graph node
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KinematicJoint:
    """One joint in the motion-constraint graph (a mate seen kinematically)."""

    kind: str
    a: str
    b: Optional[str]
    name: str = ""
    removed_dof: Optional[int] = None      # spatial DOF removed (None = unknown)
    allowed: FrozenSet[str] = frozenset()  # permitted relative-motion directions

    def key(self) -> str:
        """Stable identifier a :class:`MotionSpec` uses to address this joint."""
        if self.name:
            return self.name
        tgt = self.b if self.b is not None else "ground"
        return f"{self.kind}({self.a}->{tgt})"

    @property
    def freedom(self) -> Optional[int]:
        if self.removed_dof is None:
            return None
        return BODY_DOF - self.removed_dof

    @classmethod
    def from_mate(cls, mate: Mate) -> "KinematicJoint":
        return cls(
            kind=mate.kind,
            a=mate.a,
            b=mate.b,
            name=mate.name,
            removed_dof=joint_removed_dof(mate.kind),
            allowed=joint_directions(mate.kind),
        )


# --------------------------------------------------------------------------- #
# Intended-behaviour specification
# --------------------------------------------------------------------------- #
@dataclass
class JointIntent:
    """Intended behaviour of a single joint.

    * ``forbidden``  — directions the joint must NOT permit (a ratchet forbids
      ``rot_neg``). Any forbidden direction the joint actually allows is a hard
      functional violation.
    * ``permitted``  — if given, the *only* directions allowed; a joint that
      permits anything outside this set violates the intent (a stricter,
      whitelist form of ``forbidden``).
    * ``min_angle`` / ``max_angle`` — rotational travel limits (deg), advisory.
    * ``min_travel`` / ``max_travel`` — translational travel limits, advisory.
    """

    forbidden: Set[str] = field(default_factory=set)
    permitted: Optional[Set[str]] = None
    min_angle: Optional[float] = None
    max_angle: Optional[float] = None
    min_travel: Optional[float] = None
    max_travel: Optional[float] = None

    def has_limits(self) -> bool:
        return any(v is not None for v in (
            self.min_angle, self.max_angle, self.min_travel, self.max_travel))

    def violations(self, allowed: FrozenSet[str]) -> Set[str]:
        """Directions the joint permits that this intent forbids."""
        bad = {d for d in allowed if d in self.forbidden}
        if self.permitted is not None:
            bad |= {d for d in allowed if d not in self.permitted}
        return bad

    def to_dict(self) -> dict:
        d: dict = {}
        if self.forbidden:
            d["forbidden"] = sorted(self.forbidden)
        if self.permitted is not None:
            d["permitted"] = sorted(self.permitted)
        for key in ("min_angle", "max_angle", "min_travel", "max_travel"):
            v = getattr(self, key)
            if v is not None:
                d[key] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JointIntent":
        d = d or {}
        forbidden = {normalize_direction(t) for t in d.get("forbidden", [])}
        # convenience: one_way=True / one_way="reverse" forbids reverse rotation.
        ow = d.get("one_way")
        if ow:
            forbidden.add(ROT_NEG if ow is True else normalize_direction(ow))
        permitted = d.get("permitted")
        permitted_set = (None if permitted is None
                         else {normalize_direction(t) for t in permitted})
        return cls(
            forbidden=forbidden,
            permitted=permitted_set,
            min_angle=_opt_float(d.get("min_angle")),
            max_angle=_opt_float(d.get("max_angle")),
            min_travel=_opt_float(d.get("min_travel")),
            max_travel=_opt_float(d.get("max_travel")),
        )


@dataclass
class MotionSpec:
    """The intended motion behaviour of a whole mechanism.

    * ``name``               — a label for diagnostics.
    * ``expected_mobility``  — the intended DOF count (a four-bar = 1). A
      mismatch flags a locked mechanism that should move, or a free one that
      should be constrained.
    * ``planar``             — evaluate mobility in the plane (``d = 3``) rather
      than in space (``d = 6``); most textbook linkages are planar.
    * ``joints``             — per-joint intent, addressed by joint key (mate
      name, else ``kind(a->b)``).
    """

    name: str = ""
    expected_mobility: Optional[int] = None
    planar: bool = False
    joints: Dict[str, JointIntent] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {}
        if self.name:
            d["name"] = self.name
        if self.expected_mobility is not None:
            d["expected_mobility"] = self.expected_mobility
        if self.planar:
            d["planar"] = True
        if self.joints:
            d["joints"] = {k: ji.to_dict() for k, ji in self.joints.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MotionSpec":
        d = d or {}
        em = d.get("expected_mobility", d.get("mobility"))
        joints_raw = d.get("joints", {}) or {}
        joints = {str(k): JointIntent.from_dict(v)
                  for k, v in joints_raw.items()}
        return cls(
            name=str(d.get("name", "")),
            expected_mobility=(None if em is None else int(em)),
            planar=bool(d.get("planar", False)),
            joints=joints,
        )


# --------------------------------------------------------------------------- #
# The motion-constraint graph
# --------------------------------------------------------------------------- #
class MechanismGraph:
    """The motion-constraint graph of an assembly.

    Nodes are parts (rigid links); edges are :class:`KinematicJoint` s. Mobility
    is the Kutzbach–Gruebler count; per-joint allowed motions come from the
    joint kind. Build from an :class:`checks_assembly.AssemblyModel`.
    """

    def __init__(self, model: AssemblyModel, planar: bool = False) -> None:
        self.model = model
        self.planar = planar
        self.dof_space = 3 if planar else BODY_DOF
        self.joints: List[KinematicJoint] = [
            KinematicJoint.from_mate(m) for m in model.mates]

    @classmethod
    def from_model(cls, model: AssemblyModel, planar: bool = False
                   ) -> "MechanismGraph":
        return cls(model, planar=planar)

    def n_links(self) -> int:
        """Number of rigid links (parts). One is the fixed frame (Kutzbach)."""
        return self.model.n_parts()

    def joint_by_key(self, key: str) -> Optional[KinematicJoint]:
        for j in self.joints:
            if j.key() == key or j.name == key:
                return j
        return None

    def constraints_removed(self) -> int:
        """Total constraints imposed by recognised joints, in the working space.

        Each joint removes ``d - f`` DOF (``d`` the space DOF, ``f`` the joint
        freedom, clamped to ``[0, d]``). Unknown joints impose nothing (and are
        surfaced separately), mirroring :meth:`AssemblyModel.removed_dof`.
        """
        d = self.dof_space
        total = 0
        for j in self.joints:
            f = j.freedom
            if f is None:
                continue
            total += d - min(max(f, 0), d)
        return total

    def mobility(self) -> int:
        """Kutzbach–Gruebler mobility ``M = d(n-1) - sum(d - f)``."""
        n = self.n_links()
        if n <= 0:
            return 0
        return self.dof_space * (n - 1) - self.constraints_removed()

    def dof_summary(self) -> dict:
        """A structured snapshot of the mobility computation and per-joint DOF."""
        return {
            "planar": self.planar,
            "dof_space": self.dof_space,
            "n_links": self.n_links(),
            "n_joints": len(self.joints),
            "constraints_removed": self.constraints_removed(),
            "mobility": self.mobility(),
            "joints": [
                {
                    "key": j.key(),
                    "kind": j.kind,
                    "freedom": j.freedom,
                    "allowed": sorted(j.allowed),
                }
                for j in self.joints
            ],
        }


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class KinematicsCheck:
    """A :class:`verify.Verifier` (``name='kinematics'``) for motion behaviour.

    ``check(backend, opdag)`` reads ``query('assembly')``, builds a
    :class:`MechanismGraph`, and validates the configured :class:`MotionSpec`:

      * INFO  ``kinematics-skipped``   — no ``'assembly'`` query (e.g. the stub)
        or it is empty; nothing else runs.
      * INFO  ``kinematics-trivial``   — fewer than two parts and no joints.
      * INFO  ``mobility``             — the computed Kutzbach mobility (always).
      * WARNING ``unknown-joint``      — a joint kind with no known freedom
        (excluded from the mobility count).
      * ERROR ``mechanism-locked``     — the spec expects motion
        (``expected_mobility >= 1``) but mobility <= 0: the mechanism cannot
        perform its intended motion (a hard functional-intent violation).
      * WARNING ``mobility-mismatch``  — mobility differs from the intended count
        but the mechanism is not locked (advisory): a free mechanism that should
        be constrained, or a different DOF than expected.
      * ERROR ``forbidden-motion``     — a joint permits a direction the intent
        forbids (a ratchet joint that still allows reverse rotation).
      * WARNING ``unknown-joint-ref``  — the spec addresses a joint the graph
        does not contain.
      * INFO  ``motion-limit``         — a joint declares travel/angle limits
        (carried through; not enforceable by this static heuristic).

    Only ``mechanism-locked`` and ``forbidden-motion`` are ERRORs, so an
    advisory mobility mismatch does not flip ``report.ok`` to False.
    """

    name = "kinematics"

    def __init__(self, motion_spec: Optional[MotionSpec] = None) -> None:
        self.motion_spec = motion_spec

    def check(self, backend, opdag) -> VerifyReport:
        raw = _query(backend, "assembly")
        if not raw:
            return VerifyReport([_info(
                "kinematics-skipped",
                "motion-behaviour checks skipped: backend has no 'assembly' "
                "query (only an assembly-aware backend exposes parts + mates).")])
        model = AssemblyModel.from_dict(raw)
        return self.check_mechanism(model, self.motion_spec)

    def check_mechanism(self, model: AssemblyModel,
                        spec: Optional[MotionSpec] = None) -> VerifyReport:
        """Validate a mechanism against a :class:`MotionSpec` (no backend)."""
        return VerifyReport(kinematics_diagnostics(model, spec))


def kinematics_diagnostics(model: AssemblyModel,
                           spec: Optional[MotionSpec] = None
                           ) -> List[Diagnostic]:
    """The full motion-behaviour analysis as a flat diagnostic list."""
    diags: List[Diagnostic] = []

    if model.n_parts() < 2 and not model.mates:
        diags.append(_info(
            "kinematics-trivial",
            f"mechanism has {model.n_parts()} part(s) and no joints: no motion "
            "to validate."))
        return diags

    planar = bool(spec.planar) if spec is not None else False
    graph = MechanismGraph(model, planar=planar)

    # Joints whose freedom is unknown are excluded from the mobility count.
    for j in graph.joints:
        if j.freedom is None:
            diags.append(_warn(
                "unknown-joint",
                f"joint {j.key()} has unrecognised kind '{j.kind}'; excluded "
                "from the mobility count and its permitted motion is unknown.",
                j.name or None))

    mobility = graph.mobility()
    space = "planar" if graph.planar else "spatial"
    diags.append(_info(
        "mobility",
        f"Kutzbach {space} mobility M = {mobility} "
        f"({graph.dof_space} x ({graph.n_links()} links - 1) "
        f"- {graph.constraints_removed()} joint constraints)."))

    if spec is None:
        return diags

    # -- mobility (functional DOF) intent -------------------------------------
    expected = spec.expected_mobility
    if expected is not None and mobility != expected:
        if expected >= 1 and mobility <= 0:
            diags.append(_err(
                "mechanism-locked",
                f"mechanism '{spec.name or 'unnamed'}' is meant to move "
                f"(expected mobility {expected}) but the motion-constraint graph "
                f"is locked (M = {mobility} <= 0): joints remove too much "
                "freedom for the intended motion."))
        elif expected == 0 and mobility > 0:
            diags.append(_warn(
                "mobility-mismatch",
                f"mechanism '{spec.name or 'unnamed'}' is meant to be fully "
                f"constrained (expected mobility 0) but still has M = {mobility} "
                "free DOF: at least one link floats."))
        else:
            diags.append(_warn(
                "mobility-mismatch",
                f"mechanism '{spec.name or 'unnamed'}' mobility M = {mobility} "
                f"does not match the intended {expected} "
                "(too many or too few independent inputs)."))

    # -- per-joint permitted / forbidden motion -------------------------------
    for key, intent in spec.joints.items():
        joint = graph.joint_by_key(key)
        if joint is None:
            diags.append(_warn(
                "unknown-joint-ref",
                f"motion spec addresses joint '{key}' which the mechanism does "
                "not contain (name it via the mate's 'name', or use "
                "'kind(a->b)').", key))
            continue
        bad = intent.violations(joint.allowed)
        if bad:
            diags.append(_err(
                "forbidden-motion",
                f"joint '{key}' ({joint.kind}) permits {sorted(bad)} which the "
                f"intended behaviour forbids: a one-way / limited joint is "
                f"modelled as a freely reversible '{joint.kind}'. Use a one-way "
                "joint kind (e.g. 'ratchet') to encode the intent.", key))
        if intent.has_limits():
            diags.append(_info(
                "motion-limit",
                f"joint '{key}' declares travel/angle limits; carried through "
                "but not enforceable by the static mobility heuristic (needs a "
                "motion simulation).", key))

    return diags


# --------------------------------------------------------------------------- #
# Wiring helper
# --------------------------------------------------------------------------- #
def with_kinematics(verifiers, motion_spec: Optional[MotionSpec] = None) -> List:
    """Return a new verifier list with a :class:`KinematicsCheck` appended.

    Mirrors :func:`checks_assembly.with_assembly`::

        from verifiers.verify import default_verifiers
        from quality.kinematics import with_kinematics, MotionSpec
        verifiers = with_kinematics(default_verifiers(), MotionSpec(...))
    """
    return list(verifiers) + [KinematicsCheck(motion_spec)]


# --------------------------------------------------------------------------- #
# Small helpers (mirror checks_assembly)
# --------------------------------------------------------------------------- #
def _opt_float(v) -> Optional[float]:
    return None if v is None else float(v)


def _query(backend, q: str) -> Optional[dict]:
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
