"""OCCT kernel-quirk catalog as DATA, plus cheap feasibility predicates.

The scar tissue every OCCT-facing CAD codebase accumulates, collected here as
a machine-readable catalog so generation and repair can consult it BEFORE an
operation hangs, silently no-ops, or produces an invalid solid. Each entry
names the quirk, the trigger that provokes it, and the proven workaround.

  * cadquery (cadquery-master): SetFuzzyValue on every boolean builder
    (``occ_impl/shapes.py``); a 0-degree revolve must be rewritten as 360
    (``cq.py``, "Compensate for OCCT not assuming that a 0 degree revolve
    means a 360 degree revolve"); infinite faces report a center near the
    (1e99, 1e99) sentinel.
  * Roshera-CAD (Roshera-CAD-main ``roshera-mcp/src/tools/modify.ts``): the
    cyl-cyl saddle boolean from ADJACENT INTERSECTING ring holes is a known
    open kernel bug -- refuse loudly when chord spacing
    ``2*ring_r*sin(pi/count) <= 2*hole_r`` instead of hanging. Exposed here
    as the callable :func:`ring_holes_feasible`.
  * Zoo (Zoo-main, KCL docs): revolving a profile that TOUCHES the rotation
    axis is buggy; workaround is revolve ONE feature and circular-pattern it.
  * cadquery-plugins gear_generator (``plugins/gear_generator/main.py``):
    ``makeLoft`` over many rotated sections yields an invalid solid "for
    unknown reasons"; recovery is rebuilding via ``Shell.makeShell`` from the
    valid faces then ``Solid.makeSolid``.
  * cadgenbench (cadgenbench-main, metric mesh policy + the no-OCCT test):
    OCCT booleans HANG on interface-overlay geometry, so metric booleans use
    manifold-style mesh booleans only; sub-epsilon overlap is numerical
    noise, not intersection.
  * OpenCAD (resources/OpenCAD-main ``backend/opencad_kernel/ERRORS.md``):
    ``BBOX_NEAR_TANGENT`` -- bounding-box overlap below 1e-6 is treated as
    unstable and the boolean is REFUSED. Exposed as
    :func:`overlap_is_near_tangent`.

Pure stdlib, deterministic; no kernel, no model. The catalog is data for the
generation/repair loops (and prompt builders) to consult, never a kernel
shim itself.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

#: OpenCAD's near-tangent instability threshold (ERRORS.md, BBOX_NEAR_TANGENT).
NEAR_TANGENT_TOLERANCE = 1e-6

#: cadquery's infinite-face center sentinel magnitude (occ_impl).
INFINITE_FACE_SENTINEL = 1e99


@dataclass(frozen=True)
class Quirk:
    """One catalogued kernel quirk: what breaks, when, and what to do instead."""

    id: str
    operation: str          # the op family the quirk bites (boolean/revolve/...)
    quirk: str
    trigger: str
    workaround: str
    source: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "operation": self.operation,
            "quirk": self.quirk,
            "trigger": self.trigger,
            "workaround": self.workaround,
            "source": self.source,
        }


OCCT_QUIRKS: Tuple[Quirk, ...] = (
    Quirk(
        id="boolean-fuzzy-value",
        operation="boolean",
        quirk="Exact-tolerance booleans fail or produce slivers on "
              "nearly-coincident faces.",
        trigger="Any cut/fuse/intersect where inputs share faces, edges, or "
                "near-tangent geometry.",
        workaround="Call SetFuzzyValue(tol) on EVERY boolean builder before "
                   "Build(). Do NOT assume the binding does it for you: "
                   "cadquery makes fuzzy OPT-IN and defaults it OFF. The "
                   "Shape methods take tol=None and guard `if tol:` "
                   "(shapes.py:1417-1464); the free functions take tol=0.0, "
                   "and 0.0 is falsy, so _set_builder_options skips it too "
                   "(shapes.py:6764-6771). Exact-tolerance booleans are the "
                   "default on both paths.",
        source="cadquery-master cadquery/occ_impl/shapes.py:1417-1464 "
               "(Shape.cut/fuse/intersect) and :6764-6771 "
               "(_set_builder_options). Corrected 2026-07-19: this row "
               "previously claimed cadquery sets a fuzzy value on every "
               "boolean 'without exception', which is false and inverted the "
               "risk -- the knob exists and is off unless you pass it.",
    ),
    Quirk(
        id="revolve-zero-degrees",
        operation="revolve",
        quirk="OCCT does NOT treat a 0-degree revolve as full revolution; it "
              "produces a degenerate result.",
        trigger="revolve(angle=0) or any angle that reduces to 0 modulo 360.",
        workaround="Rewrite angle 0 as 360 before calling the kernel "
                   "(angle %= 360; angle = 360 if angle == 0 else angle).",
        source="cadquery-master cadquery/cq.py revolve() ('Compensate for "
               "OCCT not assuming that a 0 degree revolve means a 360 "
               "degree revolve')",
    ),
    Quirk(
        id="infinite-face-center-sentinel",
        operation="face-query",
        quirk="Infinite (unbounded) faces report a face center near "
              "(1e99, 1e99) instead of failing.",
        trigger="Querying .Center() / centroid of a face built from an "
                "unbounded surface (e.g. a half-space plane).",
        workaround="Treat any face-center coordinate with magnitude >= 1e99 "
                   "as the infinite-face sentinel and exclude the face from "
                   "centroid/area logic.",
        source="cadquery-master occ_impl (infinite-face center sentinel "
               "1e99)",
    ),
    Quirk(
        id="saddle-boolean-adjacent-holes",
        operation="boolean",
        quirk="The cyl-cyl saddle boolean produced by ADJACENT INTERSECTING "
              "ring holes is a known open kernel bug; the boolean hangs or "
              "corrupts the shell.",
        trigger="A circular pattern of `count` holes of radius hole_r on a "
                "ring of radius ring_r where adjacent chord spacing "
                "2*ring_r*sin(pi/count) <= 2*hole_r.",
        workaround="Refuse loudly BEFORE calling the kernel (see "
                   "ring_holes_feasible); tell the caller to reduce "
                   "count/hole_r or grow ring_r.",
        source="Roshera-CAD-main roshera-mcp/src/tools/modify.ts (ring-hole "
               "saddle-boolean refusal guard)",
    ),
    Quirk(
        id="revolve-touching-axis",
        operation="revolve",
        quirk="Revolving a profile that touches the rotation axis is buggy: "
              "self-intersection at the axis yields invalid or missing "
              "geometry.",
        trigger="Any revolve whose sketch profile has a vertex or edge ON "
                "the rotation axis.",
        workaround="Revolve ONE feature offset from the axis and reproduce "
                   "the rest with a circular pattern instead of a single "
                   "axis-touching revolve.",
        source="Zoo-main (KCL revolve-touching-axis workaround: revolve one "
               "+ pattern)",
    ),
    Quirk(
        id="loft-invalid-solid",
        operation="loft",
        quirk="makeLoft over many rotated wire sections can return a solid "
              "that is INVALID for unknown reasons even though every input "
              "wire is valid.",
        trigger="Lofting a stack of closely-spaced rotated profiles (e.g. "
                "helical gear tooth sections).",
        workaround="Rebuild via the shell route: keep the loft's valid "
                   "FACES, Shell.makeShell(faces) with the closing faces, "
                   "then Solid.makeSolid(shell).",
        source="cadquery-plugins-main plugins/gear_generator/gear_generator/"
               "main.py ('There is a bug here, the solid isnt valid for "
               "unknown reasons' -> shell rebuild)",
    ),
    Quirk(
        id="no-occt-booleans-for-metrics",
        operation="boolean",
        quirk="OCCT booleans HANG (not fail) on interface-overlay geometry "
              "-- two solids sharing a coincident interface region.",
        trigger="Metric computations (intersection volume, overlap checks) "
                "over generated geometry with coincident faces.",
        workaround="Policy: never use OCCT booleans for metrics; use "
                   "mesh/manifold booleans only, and classify sub-epsilon "
                   "overlap as numerical noise, not intersection. Enforce "
                   "with a test that greps metric code for OCCT imports.",
        source="cadgenbench-main (manifold-only metric boolean policy + "
               "tests/eval/test_interface_viz_no_occt.py)",
    ),
    Quirk(
        id="bbox-near-tangent-refusal",
        operation="boolean",
        quirk="A boolean whose inputs overlap by less than 1e-6 (bounding-"
              "box preflight) is numerically unstable: results flip between "
              "empty, sliver, and hang across runs.",
        trigger="union/intersection preflight where bbox overlap exists but "
                "is below 1e-6.",
        workaround="Refuse with BBOX_NEAR_TANGENT and ask for more overlap "
                   "(or an explicit tolerance policy) instead of executing.",
        source="resources/OpenCAD-main backend/opencad_kernel/ERRORS.md "
               "(BBOX_NEAR_TANGENT, tol 1e-6)",
    ),
)


# --------------------------------------------------------------------------- #
# lookups + predicates
# --------------------------------------------------------------------------- #

def quirks_for_operation(operation: str) -> List[Quirk]:
    """All catalogued quirks for an operation family (e.g. ``"boolean"``)."""
    op = operation.strip().lower()
    return [q for q in OCCT_QUIRKS if q.operation == op]


def quirk_by_id(quirk_id: str) -> Optional[Quirk]:
    for q in OCCT_QUIRKS:
        if q.id == quirk_id:
            return q
    return None


@dataclass(frozen=True)
class Feasibility:
    """Outcome of a pre-refusal predicate: proceed or refuse with a reason."""

    ok: bool
    reason: str = ""
    quirk_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "quirk_id": self.quirk_id}


def ring_holes_feasible(count: int, ring_r: float, hole_r: float) -> Feasibility:
    """Roshera's saddle-boolean refusal formula as a callable predicate.

    Adjacent holes on a ring intersect when their chord spacing
    ``2*ring_r*sin(pi/count)`` is at most ``2*hole_r``; the resulting
    cyl-cyl saddle boolean is a known open kernel bug, so the guard refuses
    BEFORE the kernel is invoked (source refuses with the same message
    shape). ``count < 2`` is trivially feasible.
    """
    if count < 2:
        return Feasibility(ok=True)
    if ring_r <= 0 or hole_r <= 0:
        return Feasibility(
            ok=False, quirk_id="saddle-boolean-adjacent-holes",
            reason=f"REFUSED: non-positive ring_r={ring_r} or hole_r={hole_r}")
    spacing = 2.0 * ring_r * math.sin(math.pi / count)
    if spacing <= 2.0 * hole_r:
        return Feasibility(
            ok=False,
            quirk_id="saddle-boolean-adjacent-holes",
            reason=(f"REFUSED: {count} holes of r={hole_r} on a ring of "
                    f"r={ring_r} are spaced {spacing:.3f} <= 2*r="
                    f"{2.0 * hole_r:.3f} (adjacent holes intersect; cyl-cyl "
                    f"saddle boolean is a known open kernel bug). Reduce "
                    f"count/hole_r or grow ring_r."),
        )
    return Feasibility(ok=True)


def overlap_is_near_tangent(overlap: float,
                            tolerance: float = NEAR_TANGENT_TOLERANCE) -> bool:
    """OpenCAD's BBOX_NEAR_TANGENT preflight: overlap exists but is below
    tolerance, so the boolean is unstable and should be refused."""
    return 0.0 < overlap < tolerance


def is_infinite_face_center(coord: Sequence[float]) -> bool:
    """True when a face-center coordinate carries cadquery's infinite-face
    sentinel (any component with magnitude >= 1e99)."""
    return any(abs(c) >= INFINITE_FACE_SENTINEL for c in coord)


def normalize_revolve_angle(angle_degrees: float) -> float:
    """cadquery's 0-degree revolve compensation, verbatim policy.

    ``angle %= 360``; a result of 0 means FULL revolution, because OCCT does
    not assume that itself.
    """
    angle = angle_degrees % 360.0
    return 360.0 if angle == 0 else angle


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="OCCT kernel-quirk catalog + pre-refusal predicates "
                    "(cadquery, Roshera-CAD, Zoo, gear_generator, "
                    "cadgenbench, OpenCAD).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="assert catalog integrity and exercise every "
                             "predicate on synthetic inputs.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # 1. Catalog integrity: unique ids, every field populated, source named.
    ids = [q.id for q in OCCT_QUIRKS]
    assert len(ids) == len(set(ids)) == 8, ids
    for q in OCCT_QUIRKS:
        for f in (q.quirk, q.trigger, q.workaround, q.source, q.operation):
            assert f and isinstance(f, str), q.id
    print(f"[selfcheck] catalog: {len(OCCT_QUIRKS)} quirks, unique ids, "
          f"all attributed")

    # 2. Lookups.
    assert len(quirks_for_operation("boolean")) == 4
    assert len(quirks_for_operation("revolve")) == 2
    assert quirk_by_id("loft-invalid-solid") is not None
    assert quirk_by_id("nope") is None
    print("[selfcheck] operation/id lookups")

    # 3. Saddle-boolean predicate: 6 holes r=10 on ring r=25 -> spacing 25.0
    #    > 20 feasible; 8 holes r=10 on ring r=25 -> spacing ~19.13 <= 20
    #    refused with the formula in the message.
    ok = ring_holes_feasible(6, 25.0, 10.0)
    assert ok.ok, ok.to_dict()
    bad = ring_holes_feasible(8, 25.0, 10.0)
    assert not bad.ok and "REFUSED" in bad.reason
    assert bad.quirk_id == "saddle-boolean-adjacent-holes"
    expected = 2 * 25.0 * math.sin(math.pi / 8)
    assert f"{expected:.3f}" in bad.reason
    assert ring_holes_feasible(1, 25.0, 10.0).ok  # single hole trivially ok
    assert not ring_holes_feasible(4, -1.0, 5.0).ok
    print(f"[selfcheck] ring_holes_feasible: spacing {expected:.3f} <= 20 "
          f"refused")

    # 4. Near-tangent overlap preflight (OpenCAD tol 1e-6).
    assert overlap_is_near_tangent(1e-7)
    assert not overlap_is_near_tangent(0.0)       # no overlap: different code
    assert not overlap_is_near_tangent(1e-3)      # healthy overlap
    print("[selfcheck] overlap_is_near_tangent at tol 1e-6")

    # 5. Infinite-face sentinel and revolve normalization.
    assert is_infinite_face_center((1e99, 1e99, 0.0))
    assert not is_infinite_face_center((100.0, -50.0, 3.0))
    assert normalize_revolve_angle(0.0) == 360.0
    assert normalize_revolve_angle(720.0) == 360.0
    assert normalize_revolve_angle(90.0) == 90.0
    print("[selfcheck] infinite-face sentinel + 0-degree revolve rewrite")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
