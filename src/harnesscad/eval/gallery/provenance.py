"""WHO MADE THIS PART? The only question a gallery must never fudge.

A picture of a part proves nothing on its own. What it proves depends entirely on
WHERE THE OPS CAME FROM, and there are exactly three possibilities:

``TIER_A`` -- MODEL-GENERATED.
    A natural-language brief went to a language model, the model emitted the CISP
    op stream, the harness built it, and the fleet + :mod:`harnesscad.io.gate`
    verified it. This is the only tier that is evidence about the AGENT, and it
    is therefore the only tier that supports a claim like "the system turns text
    into CAD". Every Tier-A entry publishes the model, the brief VERBATIM, the op
    stream the model actually emitted, the attempt count, and the verdict.

``TIER_B`` -- HAND-AUTHORED OP STREAM.
    A human wrote the CISP ops; the harness built them on a real backend and
    verified them exactly as it would a model's. This is evidence about the
    BACKENDS and the op vocabulary at their ceiling -- a 22-op shelled housing
    with bosses says something true and useful about the kernels. It says
    NOTHING about what a model can do, and an unlabelled gallery that mixes it
    with Tier A is making a claim it has not earned.

``PROTOCOL_BYPASS`` -- NOT AN OP STREAM AT ALL.
    Vertices and faces produced by calling the geometry services directly, with
    no CISP op anywhere. The gyroid lattice, the smooth-blend fusion, the helical
    bolt thread, the swept duct, the coil spring, the cam and the spiral flexure
    are all of these. They are the most visually impressive geometry in the
    repository and **no agent can ever ask for one**, because the op vocabulary
    has no verb that names them.

That last line is the finding, and it is a bigger one than any picture.

The bypass is a finding, not a feature
--------------------------------------
It is tempting to show the gyroid and say "only an F-rep backend can build this;
a B-rep kernel structurally cannot". That is TRUE, and it is worth showing. But
the sentence that matters more is the one underneath it:

    The gyroid is not reachable through the CISP protocol. There is no
    ``tpms`` op, no ``sweep`` any backend realises, no ``loft``, no ``draft``,
    no implicit-surface op of any kind. The harness's most impressive geometry
    is exactly the geometry its own agents cannot request.

So these parts do not get to sit in the gallery next to a model's bracket and
borrow its credibility. They are listed here, by name, as
:data:`PROTOCOL_BYPASS_PARTS`, with the op the vocabulary would need in order to
reach them -- which is a to-do list for the op set, and a far more useful
artifact than another render.

Nothing here is shape-specific
------------------------------
This module classifies PROVENANCE. It does not know what a gyroid is, it does not
special-case any part, and no verifier, gate or oracle in the repo is allowed to
either: they operate on the op stream and the resulting geometry, full stop. A
part that fails the gate is a bug or a real limit of the engine, and it is
REPORTED -- never accommodated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "TIER_A",
    "TIER_B",
    "PROTOCOL_BYPASS",
    "TIERS",
    "CAPTIONS",
    "PROTOCOL_BYPASS_PARTS",
    "MISSING_OPS",
    "classify",
    "caption_for",
    "bypass_report",
]

TIER_A = "model-generated"
TIER_B = "hand-authored-ops"
PROTOCOL_BYPASS = "protocol-bypass"

TIERS: Tuple[str, ...] = (TIER_A, TIER_B, PROTOCOL_BYPASS)

#: The caption each tier MUST carry wherever it is shown. Not a footnote.
CAPTIONS: Dict[str, str] = {
    TIER_A: ("MODEL-GENERATED: a language model was given the brief below and "
             "emitted this CISP op stream itself. Built and gate-verified by the "
             "harness. This is evidence about the agent."),
    TIER_B: ("HAND-AUTHORED OP STREAM: a human wrote these CISP ops; the harness "
             "built and gate-verified them on a real backend. This demonstrates "
             "the BACKENDS and the op vocabulary -- NOT the agent. No model was "
             "involved."),
    PROTOCOL_BYPASS: ("PROTOCOL BYPASS -- NOT A GALLERY PART: this geometry was "
                      "produced by calling the geometry services directly. It is "
                      "not a CISP op stream, no agent can request it, and it is "
                      "listed only as a finding about the op vocabulary."),
}


@dataclass(frozen=True)
class BypassPart:
    """A capability the harness HAS and the op protocol cannot NAME.

    ``needs_op`` is the verb the CISP vocabulary would have to grow before any
    agent could ask for this part. That list is the actual deliverable here.
    """

    name: str
    capability: str
    needs_op: str
    why: str

    def to_dict(self) -> dict:
        return {"name": self.name, "capability": self.capability,
                "needs_op": self.needs_op, "why": self.why}


#: Every gallery part that is NOT an op stream. Derived from the catalogue's own
#: ``kind == "mesh"`` flag (see :func:`classify`), and spelled out here with the
#: op each one would need, because "we cannot express this" is only useful if it
#: says what is missing.
PROTOCOL_BYPASS_PARTS: Tuple[BypassPart, ...] = (
    BypassPart(
        name="gyroid-lattice",
        capability="domain.geometry.sdf.tpms",
        needs_op="an implicit-surface / TPMS op (none exists)",
        why="A triply-periodic minimal surface has no B-rep, no sketch and no "
            "feature tree. It exists only as a field. NO op in the CISP set "
            "names a field, so the lattice is unreachable from the protocol -- "
            "by an agent OR by a hand-written op stream.",
    ),
    BypassPart(
        name="blend-smooth-union",
        capability="domain.geometry.sdf.combinators",
        needs_op="a smooth_union / blend op (Boolean has kind=union|cut|intersect only)",
        why="A field-level smooth-min is not a boolean. `Boolean.kind` admits "
            "union, cut and intersect -- there is no blend radius anywhere in "
            "the op set, so the operator is unreachable.",
    ),
    BypassPart(
        name="bolt-m10-iso",
        capability="domain.geometry.features.screw_thread",
        needs_op="sweep (declared in the op set; realised by NO backend)",
        why="`Sweep` EXISTS as an op and every backend returns not-yet-supported "
            "for it. So the op vocabulary names this one and the engines still "
            "cannot build it -- a gap between the protocol and its "
            "implementations, which is worse than a missing verb.",
    ),
    BypassPart(
        name="sweep-taper-duct",
        capability="domain.geometry.features.sweep",
        needs_op="sweep + loft (both declared; both refused by every backend)",
        why="As the bolt: `Sweep` and `Loft` are in the vocabulary and no backend "
            "realises either. The Tier-2 corpus keeps `loft-duct` under "
            "measurement so this gap is re-proved on every run.",
    ),
    BypassPart(
        name="coil-spring",
        capability="domain.geometry.features.sweep",
        needs_op="sweep along a helical path (refused by every backend)",
        why="A spring is a swept section on a helix. There is no helix path "
            "primitive in the sketch vocabulary and no backend realises Sweep.",
    ),
    BypassPart(
        name="cam-three-arc",
        capability="domain.geometry.sdf.cam_profile",
        needs_op="an SDF-profile op, or a spline/arc sketch entity",
        why="The cam profile is a signed distance function. The sketch vocabulary "
            "has point/line/circle/rectangle -- no arc, no spline -- so the "
            "profile cannot even be drawn, let alone extruded.",
    ),
    BypassPart(
        name="spiral-flexure",
        capability="domain.geometry.sdf.spiral",
        needs_op="an SDF-profile op, or a spline sketch entity",
        why="Thickening a transcendental curve by an offset is one line in a "
            "field and has no expression in the op set at all.",
    ),
)

#: The ops the vocabulary is MISSING, and the ops it DECLARES but no engine
#: realises. The second list is the more damning one.
MISSING_OPS: Dict[str, List[str]] = {
    "absent_from_the_vocabulary": [
        "implicit-surface / TPMS (no op names a field)",
        "smooth blend (Boolean.kind is union|cut|intersect only)",
        "arc and spline sketch entities (only point/line/circle/rectangle exist)",
        "helix / path primitives for a sweep to follow",
    ],
    "declared_but_realised_by_no_backend": [
        "sweep",
        "loft",
        "draft",
    ],
}


def classify(part) -> str:
    """Which tier is this gallery :class:`~harnesscad.eval.gallery.parts.Part`?

    Read off the catalogue's own ``kind``: an ``"ops"`` part IS a CISP op stream
    and is built through a real backend (Tier B unless a model wrote it); a
    ``"mesh"`` part is vertices and faces straight from a geometry service and
    never touched the protocol at all.

    A model-generated part does not come from the static catalogue -- it comes
    from a showcase :class:`~harnesscad.eval.showcase.loop.RunRecord` -- so
    :data:`TIER_A` is assigned by :func:`classify_run`, not here.
    """
    kind = getattr(part, "kind", None)
    if kind == "mesh":
        return PROTOCOL_BYPASS
    if kind == "ops":
        return TIER_B
    raise ValueError("cannot classify a part with kind=%r" % (kind,))


def classify_run(_run) -> str:
    """A showcase RunRecord is model-generated by construction."""
    return TIER_A


def caption_for(tier: str) -> str:
    try:
        return CAPTIONS[tier]
    except KeyError:
        raise KeyError("unknown provenance tier %r (known: %s)"
                       % (tier, ", ".join(TIERS))) from None


def bypass_report() -> dict:
    """The op-vocabulary finding, as data.

    This is the artifact to read instead of looking at the gyroid render and
    concluding the harness can make gyroids on request. It cannot. Nothing can
    ask it to.
    """
    return {
        "headline": ("the harness's most impressive geometry is exactly the "
                     "geometry its own agents cannot request: %d catalogued "
                     "parts are not CISP op streams and are unreachable through "
                     "the protocol." % len(PROTOCOL_BYPASS_PARTS)),
        "parts": [p.to_dict() for p in PROTOCOL_BYPASS_PARTS],
        "missing_ops": MISSING_OPS,
        "caption": CAPTIONS[PROTOCOL_BYPASS],
    }
