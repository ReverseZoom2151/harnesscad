"""Proving a part is FINE, before anybody is accused of rejecting it.

A false-positive report is worth exactly as much as the proof behind it. "Verifier
X rejected op stream Y" is not a bug report -- it is a bug report only once
somebody has established that Y was a good part, and the whole reason the pressure
experiment's finding survives scrutiny is that each of its four fleet bugs was
reproduced against the backend BY HAND before it was written down.

So an attack is only ever promoted to a FALSE POSITIVE when all three of these
hold, and the report says which:

1.  **ARITHMETIC.** The part has a volume in closed form, computed from its own
    dimensions by a formula written down in ``attacks.py`` next to it. This is the
    only party in the room that did not come out of this repository.

2.  **AN ENGINE BUILDS IT, AT THAT VOLUME.** The op stream is run at
    ``verify_level="core"`` (never "full" -- consulting the fleet to decide whether
    the fleet is wrong is a circle) and the measured volume must land on the closed
    form within that engine's OWN physical tolerance
    (``selftest.probe.tolerance``). A part that arithmetic says is 22296 mm3 and
    that the engine builds at 22296 mm3 is a part that exists.

3.  **THE OUTPUT GATE ACCEPTS IT** (``io/gate.py``). This is a MEASURED verdict,
    not an asserted one: the gate re-tessellates the solid and independently
    re-measures it -- watertight, 2-manifold, positively oriented, no
    self-intersections, no degenerate triangles.

WHAT THIS ORACLE DOES NOT ESTABLISH, STATED BEFORE ANYBODY OVERCLAIMS
--------------------------------------------------------------------
``io/gate.py`` says so itself, and it is repeated here because it bounds every
finding below it: **the gate does not prove a feature is in the right PLACE.** A
hole bored at x = 40 instead of x = 20 passes the gate, and passes the closed-form
volume check too (it is the same hole). That is fine for THIS oracle's purpose --
we are not grading a model's answer, we are establishing that a part is
manufacturable and well-formed, and a hole in the wrong place is still a
well-formed part. But it means the certificate says "this is a real, sound,
buildable solid of the volume claimed", and NOT "this is the part somebody asked
for". A verifier that rejects it is rejecting a real part, and that is the entire
claim being made.

When the oracle cannot certify -- the engine will not build it, the volume misses,
the gate refuses -- the attack is DROPPED and counted as uncertified. It is never
silently promoted. A red team that inflates its own hit count is worth less than
no red team.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from harnesscad.eval.redteam.attacks import Attack
from harnesscad.eval.selftest.probe import resolve, tolerance

__all__ = ["Certificate", "certify"]


@dataclass
class Certificate:
    """The proof, or the reason there is none."""

    attack: str
    backend: str = "frep"
    certified: bool = False
    reason: str = ""                         # why NOT, when not certified
    measured_volume: Optional[float] = None
    measured_bbox: Optional[List[float]] = None
    closed_form: float = 0.0
    volume_error: Optional[float] = None     # relative
    gate_ok: Optional[bool] = None
    gate_failures: List[str] = field(default_factory=list)
    #: The engine that measured it. Kept so a finding can be replayed.
    watertight: Optional[bool] = None

    def to_dict(self) -> dict:
        return {"attack": self.attack, "backend": self.backend,
                "certified": self.certified, "reason": self.reason,
                "closed_form": self.closed_form,
                "measured_volume": self.measured_volume,
                "measured_bbox": self.measured_bbox,
                "volume_error": self.volume_error,
                "gate_ok": self.gate_ok, "gate_failures": self.gate_failures,
                "watertight": self.watertight}


#: A sampled engine needs at least this many grid cells across the part's thinnest
#: feature before its measurement of that part means anything (Nyquist). Below it,
#: the field cannot represent the feature and the engine quietly builds a smaller,
#: different part -- see ``eval/corpus/grade.py``. An attack the engine cannot
#: resolve is UNCERTIFIED, not a finding: we would be unable to say the part it
#: actually built was fine.
CELLS_PER_FEATURE = 2.0


def certify(attack: Attack, backend: str = "frep") -> Certificate:
    """Prove the attack builds a correct part, or say why we cannot."""
    from harnesscad.core.loop import HarnessSession
    from harnesscad.io import gate

    c = Certificate(attack=attack.name, backend=backend,
                    closed_form=float(attack.volume))

    tol = tolerance(backend)
    if tol.cells > 0:
        cell = tol.cell(max(attack.bbox))
        if attack.min_feature < CELLS_PER_FEATURE * cell:
            c.reason = ("%s samples a field on %d cells across the largest extent "
                        "(%.1f mm) -> a %.2f mm cell, and this part's thinnest "
                        "feature is %.2f mm (%.1f cells). Below %g cells the "
                        "engine cannot represent the feature and builds a "
                        "different part. UNCERTIFIED: we cannot claim the part it "
                        "built is fine."
                        % (backend, tol.cells, max(attack.bbox), cell,
                           attack.min_feature, attack.min_feature / cell,
                           CELLS_PER_FEATURE))
            return c

    engine, skip = resolve(backend)
    if engine is None:
        c.reason = "engine %r unavailable: %s" % (backend, skip)
        return c

    # verify_level="core", NOT "full". Asking the fleet whether the part is good,
    # in order to decide whether the fleet is wrong about the part, is a circle.
    try:
        session = HarnessSession(engine, verify_level="core")
        result = session.apply_ops(list(attack.ops))
    except Exception as exc:                                   # noqa: BLE001
        c.reason = "the engine raised %s: %s" % (type(exc).__name__, exc)
        return c
    if not getattr(result, "ok", False):
        rej = getattr(result, "rejected", None)
        c.reason = ("the engine refused the plan (%s). A part no engine will build "
                    "cannot be certified fine."
                    % (rej.get("op") if isinstance(rej, dict) else "?"))
        return c

    try:
        m = engine.query("measure") or {}
        v = engine.query("validity") or {}
    except Exception as exc:                                   # noqa: BLE001
        c.reason = "measurement failed: %s" % exc
        return c
    vol = m.get("volume")
    bbox = m.get("bbox")
    if not isinstance(vol, (int, float)) or not bbox:
        c.reason = "the engine produced no measurable solid"
        return c
    c.measured_volume = float(vol)
    c.measured_bbox = [float(x) for x in bbox]
    c.watertight = v.get("watertight")

    # 1 + 2: the closed form, within the engine's OWN physical tolerance.
    vtol = tol.volume_tol(max(attack.bbox), attack.min_feature)
    err = abs(float(vol) - attack.volume) / max(attack.volume, 1e-9)
    c.volume_error = err
    if err > vtol:
        c.reason = ("the engine built it at %.1f mm3 but the closed form says "
                    "%.1f mm3 (%+.1f%%, tolerance %.1f%%). Either the arithmetic "
                    "or the engine is wrong, and until we know which, this part "
                    "cannot be used to accuse a verifier of anything."
                    % (vol, attack.volume,
                       100.0 * (vol - attack.volume) / attack.volume, 100.0 * vtol))
        return c

    # 3: the output gate. A MEASURED verdict -- it re-tessellates and re-measures.
    try:
        report = gate.check(engine, source=session)
        c.gate_ok = bool(report.ok)
        c.gate_failures = ["%s: %s" % (f.check, f.detail) for f in report.failures]
    except Exception as exc:                                   # noqa: BLE001
        c.gate_ok = False
        c.gate_failures = ["gate raised %s: %s" % (type(exc).__name__, exc)]
    if not c.gate_ok:
        c.reason = ("the output gate REFUSED this artifact (%s). The gate measures "
                    "the written solid independently; a part it refuses is not a "
                    "part we may call fine." % "; ".join(c.gate_failures))
        return c

    c.certified = True
    c.reason = ("closed form %.1f mm3, engine measured %.1f mm3 (%+.2f%%, within "
                "its %.1f%% physical tolerance), watertight, and the output gate "
                "re-measured the written solid and accepted it."
                % (attack.volume, vol,
                   100.0 * (vol - attack.volume) / attack.volume, 100.0 * vtol))
    return c
