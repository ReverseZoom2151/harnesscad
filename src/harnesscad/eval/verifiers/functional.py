"""Functional-behaviour acceptance oracle — check that the built mechanism
*does what it is meant to do*.

The static assembly solver (:mod:`checks_assembly`) asks "can these parts be
placed?"; the kinematics validator (:mod:`quality.kinematics`) computes the
motion-constraint graph and its Kutzbach mobility. This module closes the loop
with an *acceptance oracle*: a designer declares the intended function
(:class:`FunctionalSpec` — required DOF/mobility and per-joint motions, e.g.
"1-DOF rotary", "one-way ratchet"), and :class:`FunctionalCheck` computes the
built assembly's kinematics and asserts it MATCHES that declared intent.

The spec is therefore *both* the specification and the test oracle: any
divergence between built behaviour and declared intent — the mechanism is locked
when it must move, has the wrong number of degrees of freedom, or permits a
motion the intent forbids (a ratchet that still turns backwards) — is a single,
legible ERROR ``function-mismatch``. When the built behaviour matches the intent
the oracle passes cleanly (``function-verified``).

It reuses :class:`quality.kinematics.MechanismGraph` for the mobility count and
:class:`quality.kinematics.JointIntent` for the per-joint permitted/forbidden
motion logic, so there is one motion vocabulary across the harness.

Standalone by design, exactly like :class:`checks_assembly.AssemblyCheck`: NOT
wired into :func:`verify.default_verifiers`. A caller adds it via
:func:`with_functional`, or runs it backend-free through
:meth:`FunctionalCheck.check_mechanism`. It reads ``query('assembly')`` and
INFO-skips gracefully when there is no assembly or no declared spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from harnesscad.eval.verifiers.assembly import AssemblyModel
from harnesscad.eval.verifiers.verify import Diagnostic, Severity, VerifyReport
from harnesscad.eval.quality.kinematics import JointIntent, MechanismGraph


# --------------------------------------------------------------------------- #
# Declared intended behaviour
# --------------------------------------------------------------------------- #
_DOF_PHRASE_RE = re.compile(r"(-?\d+)\s*-?\s*dof", re.IGNORECASE)


@dataclass
class FunctionalSpec:
    """A declaration of a mechanism's intended function.

    * ``name``               — a label for diagnostics.
    * ``required_mobility``  — the intended number of degrees of freedom (a
      1-DOF rotary hinge / a four-bar = 1; a fully-fixed bracket = 0). A built
      mobility that differs is a ``function-mismatch``.
    * ``planar``             — evaluate mobility in the plane (``d = 3``) rather
      than in space (``d = 6``); most textbook linkages are planar.
    * ``motions``            — per-joint intended motion, addressed by joint key
      (the mate's ``name``, else ``kind(a->b)``), as
      :class:`quality.kinematics.JointIntent` s (forbidden / permitted
      directions). A joint that permits a forbidden direction is a
      ``function-mismatch`` (a ratchet modelled as a free revolute).
    * ``behavior``           — an optional free-text tag (e.g. "1-DOF rotary",
      "one-way ratchet") kept for provenance; a leading ``N-DOF`` in it seeds
      ``required_mobility`` when that field is not set explicitly.
    """

    name: str = ""
    required_mobility: Optional[int] = None
    planar: bool = False
    motions: Dict[str, JointIntent] = field(default_factory=dict)
    behavior: str = ""

    def describe(self) -> str:
        """A short human phrase for the declared intent."""
        if self.behavior:
            return self.behavior
        bits: List[str] = []
        if self.required_mobility is not None:
            bits.append(f"{self.required_mobility}-DOF")
        bits.append("planar" if self.planar else "spatial")
        if self.motions:
            bits.append(f"{len(self.motions)} constrained joint(s)")
        return " ".join(bits)

    def to_dict(self) -> dict:
        d: dict = {}
        if self.name:
            d["name"] = self.name
        if self.required_mobility is not None:
            d["required_mobility"] = self.required_mobility
        if self.planar:
            d["planar"] = True
        if self.motions:
            d["motions"] = {k: ji.to_dict() for k, ji in self.motions.items()}
        if self.behavior:
            d["behavior"] = self.behavior
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FunctionalSpec":
        d = d or {}
        behavior = str(d.get("behavior", ""))
        req = d.get("required_mobility", d.get("mobility", d.get("dof")))
        if req is None and behavior:
            m = _DOF_PHRASE_RE.search(behavior)
            if m:
                req = int(m.group(1))
        # motions accepts either 'motions' or 'joints'.
        raw = d.get("motions", d.get("joints", {})) or {}
        motions = {str(k): JointIntent.from_dict(v) for k, v in raw.items()}
        return cls(
            name=str(d.get("name", "")),
            required_mobility=(None if req is None else int(req)),
            planar=bool(d.get("planar", False)),
            motions=motions,
            behavior=behavior,
        )


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class FunctionalCheck:
    """A :class:`verify.Verifier` (``name='functional'``) acceptance oracle.

    ``check(backend, opdag)`` reads ``query('assembly')``, builds a
    :class:`quality.kinematics.MechanismGraph`, and asserts it matches the
    declared :class:`FunctionalSpec`:

      * INFO  ``functional-skipped``   — no ``'assembly'`` query (e.g. the stub),
        an empty assembly, or no :class:`FunctionalSpec` declared.
      * INFO  ``functional-trivial``   — fewer than two parts and no joints.
      * INFO  ``built-mobility``       — the computed Kutzbach mobility (always,
        when a mechanism is evaluated).
      * WARNING ``unknown-joint``      — a joint kind with no known freedom
        (excluded from the mobility count).
      * WARNING ``unknown-joint-ref``  — the spec addresses a joint the built
        mechanism does not contain.
      * ERROR ``function-mismatch``    — the built behaviour diverges from the
        declared intent: locked-when-should-move, wrong DOF, or a forbidden
        motion the built joint still permits.
      * INFO  ``function-verified``    — built behaviour matches the declared
        intent (no mismatch).

    Only ``function-mismatch`` is an ERROR, so an advisory unknown-joint note
    does not by itself flip ``report.ok`` to False.
    """

    name = "functional"

    def __init__(self, spec: Optional[FunctionalSpec] = None) -> None:
        self.spec = spec

    def check(self, backend, opdag=None) -> VerifyReport:
        if self.spec is None:
            return VerifyReport([_info(
                "functional-skipped",
                "functional acceptance skipped: no FunctionalSpec declared "
                "(nothing to check the built behaviour against).")])
        raw = _query(backend, "assembly")
        if not raw:
            return VerifyReport([_info(
                "functional-skipped",
                "functional acceptance skipped: backend has no 'assembly' query "
                "(only an assembly-aware backend exposes parts + mates).")])
        model = AssemblyModel.from_dict(raw)
        return self.check_mechanism(model, self.spec)

    def check_mechanism(self, model: AssemblyModel,
                        spec: Optional[FunctionalSpec] = None) -> VerifyReport:
        """Validate a mechanism against a :class:`FunctionalSpec` (no backend)."""
        spec = spec if spec is not None else self.spec
        return VerifyReport(functional_diagnostics(model, spec))


def functional_diagnostics(model: AssemblyModel,
                           spec: Optional[FunctionalSpec]
                           ) -> List[Diagnostic]:
    """The full functional-acceptance analysis as a flat diagnostic list."""
    if spec is None:
        return [_info(
            "functional-skipped",
            "functional acceptance skipped: no FunctionalSpec declared.")]

    if model.n_parts() < 2 and not model.mates:
        return [_info(
            "functional-trivial",
            f"mechanism has {model.n_parts()} part(s) and no joints: no "
            "function to accept.")]

    diags: List[Diagnostic] = []
    graph = MechanismGraph(model, planar=spec.planar)

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
        "built-mobility",
        f"built {space} mobility M = {mobility} "
        f"({graph.dof_space} x ({graph.n_links()} links - 1) "
        f"- {graph.constraints_removed()} joint constraints)."))

    mismatched = False

    # -- required DOF / mobility -------------------------------------------- #
    expected = spec.required_mobility
    label = spec.name or spec.describe() or "unnamed"
    if expected is not None and mobility != expected:
        mismatched = True
        if expected >= 1 and mobility <= 0:
            diags.append(_err(
                "function-mismatch",
                f"function '{label}' is meant to move (required mobility "
                f"{expected}) but the built mechanism is locked (M = {mobility} "
                "<= 0): the joints remove too much freedom for the intended "
                "motion."))
        elif expected == 0 and mobility > 0:
            diags.append(_err(
                "function-mismatch",
                f"function '{label}' is meant to be fully constrained (required "
                f"mobility 0) but the built mechanism still has M = {mobility} "
                "free DOF: at least one link floats."))
        else:
            diags.append(_err(
                "function-mismatch",
                f"function '{label}' requires mobility {expected} but the built "
                f"mechanism has M = {mobility} (wrong number of degrees of "
                "freedom for the intended function)."))

    # -- per-joint permitted / forbidden motion ----------------------------- #
    for key, intent in spec.motions.items():
        joint = graph.joint_by_key(key)
        if joint is None:
            diags.append(_warn(
                "unknown-joint-ref",
                f"functional spec addresses joint '{key}' which the built "
                "mechanism does not contain (name it via the mate's 'name', or "
                "use 'kind(a->b)').", key))
            continue
        bad = intent.violations(joint.allowed)
        if bad:
            mismatched = True
            diags.append(_err(
                "function-mismatch",
                f"joint '{key}' ({joint.kind}) permits {sorted(bad)} which the "
                f"intended function forbids: a one-way / limited joint is built "
                f"as a freely reversible '{joint.kind}'. Use a one-way joint "
                "kind (e.g. 'ratchet') to realise the intent.", key))

    if not mismatched:
        diags.append(_info(
            "function-verified",
            f"built behaviour matches the declared function '{label}': "
            f"mobility M = {mobility}"
            + (f" (required {expected})" if expected is not None else "")
            + (f", {len(spec.motions)} joint-motion intent(s) satisfied"
               if spec.motions else "")
            + "."))

    return diags


# --------------------------------------------------------------------------- #
# Wiring helper
# --------------------------------------------------------------------------- #
def with_functional(verifiers, spec: Optional[FunctionalSpec] = None) -> List:
    """Return a new verifier list with a :class:`FunctionalCheck` appended.

    Mirrors :func:`checks_assembly.with_assembly`::

        from harnesscad.eval.verifiers.verify import default_verifiers
        from harnesscad.eval.verifiers.functional import with_functional, FunctionalSpec
        verifiers = with_functional(default_verifiers(), FunctionalSpec(...))
    """
    return list(verifiers) + [FunctionalCheck(spec)]


# --------------------------------------------------------------------------- #
# Small helpers (mirror checks_assembly)
# --------------------------------------------------------------------------- #
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
