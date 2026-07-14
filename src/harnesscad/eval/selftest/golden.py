"""Golden corpus — parts whose volume, bbox and genus are known in CLOSED FORM.

The differential oracle can tell you that two backends disagree. It cannot tell
you WHICH one is wrong. This can: every part here declares its exact geometry
from first principles, so a backend that misses it is the backend with the bug.

Nothing here is a fixture recorded from a previous run. A recorded fixture only
proves the code still does what it did the day the bug was introduced; the shell
that grows a box would have been recorded, blessed, and defended by its own test.
Every number below is derived, and the derivation is written down next to it.

The four non-obvious derivations, stated once:

**All-edge fillet (a rounded box).** Filleting all 12 edges of an ``a x b x c``
box at radius ``r`` is exactly the Minkowski sum of the ball of radius ``r`` with
the shrunken box ``A x B x C`` (``A = a - 2r``). Steiner's formula gives::

    V = A*B*C + 2r(AB + BC + CA) + pi*r^2*(A + B + C) + (4/3)*pi*r^3

(volume + surface*r + integral-mean-curvature*r^2 + the eight corner octants,
which sum to one sphere). The bbox is unchanged: a fillet removes material.

**All-edge chamfer.** Each of the 12 edges loses a triangular prism of
cross-section ``d^2/2`` over its length less the two corner setbacks, and each of
the 8 corners loses a further ``(5/6)d^3``::

    V = a*b*c - (d^2/2) * sum(L_i - 2d) - 8*(5/6)*d^3

**Shell.** ``shell(faces, thickness)`` hollows INWARD: the outer surface does not
move. An EMPTY ``faces`` list is a CLOSED hollow -- a sealed void, no face removed
-- so an ``a x b x c`` box walled to ``t`` is::

    V = a*b*c - (a-2t)(b-2t)(c-2t)          (60x40x20 at t=3 -> exactly 22296)

and the bbox is UNCHANGED. Both halves of that matter, and only one of them is
easy: **a bbox check cannot prove a shell.** An inward shell can preserve the
envelope exactly and still leave the wall 42% too thin -- ``t/sqrt(3)``, from an
uncorrected corner normal, which is a real bug that lived INSIDE the part where no
envelope check could ever see it. That is why every shell part here asserts the
exact analytic VOLUME, which pins the wall, and not merely the box around it.

:func:`open_top_shell_box_volume` is the rival reading (one face removed). It is
kept only so the report can NAME what a non-conforming engine is doing instead of
just calling its number wrong.

**Genus.** A through hole adds one handle, so N through holes give genus N. N
disjoint bodies give Euler 2N and genus ``(2 - 2N)/2 = 1 - N`` (negative: it is a
component count in disguise, but it is exactly comparable across backends). A
part whose genus a backend does not report is not scored on genus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import (AddCircle, AddRectangle, Boolean, Chamfer,
                                      Extrude, Fillet, Hole, LinearPattern,
                                      NewSketch, Op, Revolve, Shell)
from harnesscad.eval.selftest.probe import (BackendFactory, GEOMETRIC_BACKENDS,
                                            Observation, available, observe,
                                            tolerance)

__all__ = ["GoldenPart", "Deviation", "GoldenResult", "GoldenReport", "PARTS",
           "check_part", "run", "format_text"]

PI = math.pi


# --- closed forms ----------------------------------------------------------

def rounded_box_volume(a: float, b: float, c: float, r: float) -> float:
    """Volume of an ``a x b x c`` box with every edge filleted at radius r."""
    A, B, C = a - 2 * r, b - 2 * r, c - 2 * r
    if min(A, B, C) < 0:
        raise ValueError("fillet radius exceeds half the smallest extent")
    return (A * B * C
            + 2 * r * (A * B + B * C + C * A)
            + PI * r * r * (A + B + C)
            + (4.0 / 3.0) * PI * r ** 3)


def chamfered_box_volume(a: float, b: float, c: float, d: float) -> float:
    """Volume of an ``a x b x c`` box with every edge chamfered at setback d."""
    edge_len = 4 * (a - 2 * d) + 4 * (b - 2 * d) + 4 * (c - 2 * d)
    return a * b * c - (d * d / 2.0) * edge_len - 8 * (5.0 / 6.0) * d ** 3


def shelled_box_volume(a: float, b: float, c: float, t: float) -> float:
    """The NORMATIVE shell: hollowed inward to wall ``t``, sealed (no face removed).

    This is the spec: ``shell(faces=(), thickness=t)`` is a closed hollow. The
    volume pins the WALL, which a bbox check cannot: a shell whose wall came out
    at t/sqrt(3) has a perfect envelope and is 42% too thin.
    """
    return a * b * c - (a - 2 * t) * (b - 2 * t) * (c - 2 * t)


def open_top_shell_box_volume(a: float, b: float, c: float, t: float) -> float:
    """The RIVAL reading -- one face removed. NOT the spec; kept to name an outlier."""
    return a * b * c - (a - 2 * t) * (b - 2 * t) * (c - t)


def annulus_volume(r_outer: float, r_inner: float, h: float) -> float:
    return PI * (r_outer ** 2 - r_inner ** 2) * h


# --- the part type ---------------------------------------------------------

@dataclass(frozen=True)
class GoldenPart:
    """One part, its op stream, and its geometry derived from first principles."""

    name: str
    ops: Tuple[Op, ...]
    volume: float
    bbox: Tuple[float, float, float]
    genus: Optional[int] = None
    note: str = ""
    #: A DIFFERENT, defensible reading of the same op (today: only ``shell``).
    #: A backend landing here is not producing noise -- it is implementing the
    #: other semantics, and the report should say so rather than cry "wrong".
    rival_volume: Optional[float] = None
    rival_name: str = ""

    @property
    def extent(self) -> float:
        return max(self.bbox)


def _sk(plane: str = "XY") -> Op:
    return NewSketch(plane)


# --- the corpus ------------------------------------------------------------
# 20 parts. Planar prisms (exact on every backend), curved parts (which separate
# a B-rep kernel from a mesher), holes (genus), and the four operations the
# harness gets wrong: shell, fillet, chamfer, boolean.

PARTS: Tuple[GoldenPart, ...] = (
    # -- prisms: no curvature, so EVERY backend must be exact -----------------
    GoldenPart(
        "plate_60x40x10",
        (_sk(), AddRectangle("sk1", 0, 0, 60, 40), Extrude("sk1", 10.0)),
        volume=60 * 40 * 10, bbox=(60.0, 40.0, 10.0), genus=0,
        note="a x b x c"),
    GoldenPart(
        "plate_thin_100x50x3",
        (_sk(), AddRectangle("sk1", 0, 0, 100, 50), Extrude("sk1", 3.0)),
        volume=100 * 50 * 3, bbox=(100.0, 50.0, 3.0), genus=0,
        note="a x b x c (thin: the extent that bounds a shell/fillet)"),
    GoldenPart(
        "cube_20",
        (_sk(), AddRectangle("sk1", 0, 0, 20, 20), Extrude("sk1", 20.0)),
        volume=8000.0, bbox=(20.0, 20.0, 20.0), genus=0,
        note="a^3"),

    # -- curved: exact for a B-rep, polygonised for a mesh, sampled for a field
    GoldenPart(
        "cylinder_d30_h30",
        (_sk(), AddCircle("sk1", 0, 0, 15.0), Extrude("sk1", 30.0)),
        volume=PI * 15 ** 2 * 30, bbox=(30.0, 30.0, 30.0), genus=0,
        note="pi r^2 h"),
    GoldenPart(
        "pin_d10_h50",
        (_sk(), AddCircle("sk1", 0, 0, 5.0), Extrude("sk1", 50.0)),
        volume=PI * 25 * 50, bbox=(10.0, 10.0, 50.0), genus=0,
        note="pi r^2 h (slender: the frep grid's worst case)"),
    GoldenPart(
        "disc_d80_h8",
        (_sk(), AddCircle("sk1", 0, 0, 40.0), Extrude("sk1", 8.0)),
        volume=PI * 1600 * 8, bbox=(80.0, 80.0, 8.0), genus=0,
        note="pi r^2 h (the washer's blank)"),

    # -- through holes: genus is the assertion --------------------------------
    GoldenPart(
        "washer_d80_bore30_h8",
        (_sk(), AddCircle("sk1", 0, 0, 40.0), Extrude("sk1", 8.0),
         Hole("sk1", 0.0, 0.0, 30.0, None, True, "simple")),
        volume=annulus_volume(40.0, 15.0, 8.0), bbox=(80.0, 80.0, 8.0), genus=1,
        note="pi (R^2 - r^2) h -- the part the fleet rejects"),
    GoldenPart(
        "tube_d40_bore24_h50",
        (_sk(), AddCircle("sk1", 0, 0, 20.0), Extrude("sk1", 50.0),
         Hole("sk1", 0.0, 0.0, 24.0, None, True, "simple")),
        volume=annulus_volume(20.0, 12.0, 50.0), bbox=(40.0, 40.0, 50.0), genus=1,
        note="pi (R^2 - r^2) h"),
    GoldenPart(
        "plate_one_hole",
        (_sk(), AddRectangle("sk1", 0, 0, 80, 60), Extrude("sk1", 5.0),
         Hole("sk1", 40.0, 30.0, 10.0, None, True, "simple")),
        volume=80 * 60 * 5 - PI * 25 * 5, bbox=(80.0, 60.0, 5.0), genus=1,
        note="abc - pi r^2 c"),
    GoldenPart(
        "plate_four_holes",
        (_sk(), AddRectangle("sk1", 0, 0, 80, 60), Extrude("sk1", 5.0),
         Hole("sk1", 10.0, 10.0, 6.0, None, True, "simple"),
         Hole("sk1", 70.0, 10.0, 6.0, None, True, "simple"),
         Hole("sk1", 10.0, 50.0, 6.0, None, True, "simple"),
         Hole("sk1", 70.0, 50.0, 6.0, None, True, "simple")),
        volume=80 * 60 * 5 - 4 * PI * 9 * 5, bbox=(80.0, 60.0, 5.0), genus=4,
        note="abc - N pi r^2 c; N through holes => genus N"),
    GoldenPart(
        "strip_five_holes",
        (_sk(), AddRectangle("sk1", 0, 0, 120, 30), Extrude("sk1", 6.0),
         Hole("sk1", 20.0, 15.0, 8.0, None, True, "simple"),
         Hole("sk1", 40.0, 15.0, 8.0, None, True, "simple"),
         Hole("sk1", 60.0, 15.0, 8.0, None, True, "simple"),
         Hole("sk1", 80.0, 15.0, 8.0, None, True, "simple"),
         Hole("sk1", 100.0, 15.0, 8.0, None, True, "simple")),
        volume=120 * 30 * 6 - 5 * PI * 16 * 6, bbox=(120.0, 30.0, 6.0), genus=5,
        note="a bolt-hole row"),
    GoldenPart(
        "flange_bolt_circle",
        (_sk(), AddCircle("sk1", 0, 0, 50.0), Extrude("sk1", 10.0),
         Hole("sk1", 0.0, 0.0, 40.0, None, True, "simple"),
         Hole("sk1", 35.0, 0.0, 9.0, None, True, "simple"),
         Hole("sk1", -35.0, 0.0, 9.0, None, True, "simple"),
         Hole("sk1", 0.0, 35.0, 9.0, None, True, "simple"),
         Hole("sk1", 0.0, -35.0, 9.0, None, True, "simple")),
        volume=PI * 10 * (2500 - 400 - 4 * 20.25), bbox=(100.0, 100.0, 10.0),
        genus=5, note="a bore plus a four-bolt circle"),

    # -- shell: the bbox must NOT change --------------------------------------
    GoldenPart(
        "shelled_box_60x40x20_t3",
        (_sk(), AddRectangle("sk1", 0, 0, 60, 40), Extrude("sk1", 20.0),
         Shell((), 3.0)),
        volume=shelled_box_volume(60, 40, 20, 3), bbox=(60.0, 40.0, 20.0),
        genus=None,
        note="THE README PART. Closed hollow: 48000 - 54*34*14 = 22296 exactly; "
             "bbox UNCHANGED. The volume is what pins the WALL -- an envelope "
             "check passes a wall that is t/sqrt(3) thick.",
        rival_volume=open_top_shell_box_volume(60, 40, 20, 3),
        rival_name="open-top shell (one face removed -- NOT the spec)"),
    GoldenPart(
        "shelled_box_40x40x40_t2",
        (_sk(), AddRectangle("sk1", 0, 0, 40, 40), Extrude("sk1", 40.0),
         Shell((), 2.0)),
        volume=shelled_box_volume(40, 40, 40, 2), bbox=(40.0, 40.0, 40.0),
        genus=None, note="a thin-walled enclosure, closed hollow; bbox UNCHANGED",
        rival_volume=open_top_shell_box_volume(40, 40, 40, 2),
        rival_name="open-top shell (one face removed -- NOT the spec)"),
    GoldenPart(
        "shelled_box_30x30x30_t5",
        (_sk(), AddRectangle("sk1", 0, 0, 30, 30), Extrude("sk1", 30.0),
         Shell((), 5.0)),
        volume=shelled_box_volume(30, 30, 30, 5), bbox=(30.0, 30.0, 30.0),
        genus=None, note="a thick wall, closed hollow; bbox UNCHANGED",
        rival_volume=open_top_shell_box_volume(30, 30, 30, 5),
        rival_name="open-top shell (one face removed -- NOT the spec)"),

    # -- fillet / chamfer: the exact removed volume ---------------------------
    GoldenPart(
        "filleted_block_50x30x6_r2",
        (_sk(), AddRectangle("sk1", 0, 0, 50, 30), Extrude("sk1", 6.0),
         Fillet((), 2.0)),
        volume=rounded_box_volume(50, 30, 6, 2), bbox=(50.0, 30.0, 6.0), genus=0,
        note="Steiner: r=2 on a 6 mm plate is VALID (r < c/2)"),
    GoldenPart(
        "filleted_block_40x40x20_r5",
        (_sk(), AddRectangle("sk1", 0, 0, 40, 40), Extrude("sk1", 20.0),
         Fillet((), 5.0)),
        volume=rounded_box_volume(40, 40, 20, 5), bbox=(40.0, 40.0, 20.0),
        genus=0, note="Steiner"),
    GoldenPart(
        "chamfered_block_50x30x6_d1",
        (_sk(), AddRectangle("sk1", 0, 0, 50, 30), Extrude("sk1", 6.0),
         Chamfer((), 1.0)),
        volume=chamfered_box_volume(50, 30, 6, 1), bbox=(50.0, 30.0, 6.0),
        genus=0, note="12 edge prisms + 8 corners"),

    # -- revolve / pattern / boolean ------------------------------------------
    GoldenPart(
        "revolved_ring_r10_r15_h20",
        (_sk(), AddRectangle("sk1", 10, 0, 5, 20),
         Revolve("sk1", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0), 360.0)),
        volume=annulus_volume(15.0, 10.0, 20.0), bbox=(30.0, 20.0, 30.0), genus=1,
        note="a profile offset from the axis sweeps an annulus"),
    GoldenPart(
        "revolved_cylinder_r10_h20",
        (_sk(), AddRectangle("sk1", 0, 0, 10, 20),
         Revolve("sk1", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0), 360.0)),
        volume=PI * 100 * 20, bbox=(20.0, 20.0, 20.0), genus=0,
        note="a profile touching the axis sweeps a solid cylinder"),
    GoldenPart(
        "linear_pattern_3x_block",
        (_sk(), AddRectangle("sk1", 0, 0, 10, 10), Extrude("sk1", 5.0),
         LinearPattern("f1", (1.0, 0.0, 0.0), 3, 20.0)),
        volume=3 * 10 * 10 * 5, bbox=(50.0, 10.0, 5.0), genus=-2,
        note="N disjoint bodies => Euler 2N => genus 1-N"),
    GoldenPart(
        "boolean_cut_notch",
        (_sk(), AddRectangle("sk1", 0, 0, 40, 40), Extrude("sk1", 10.0),
         _sk(), AddRectangle("sk2", 0, 0, 10, 10), Extrude("sk2", 10.0),
         Boolean("cut", "f1", "f2")),
        volume=40 * 40 * 10 - 10 * 10 * 10, bbox=(40.0, 40.0, 10.0), genus=0,
        note="a cut REMOVES volume: 16000 - 1000"),
)


# --- checking --------------------------------------------------------------

@dataclass
class Deviation:
    """One backend missing one declared value on one part."""

    part: str
    backend: str
    metric: str            # volume | bbox | genus | watertight | build
    expected: object
    actual: object
    tol: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict:
        return {"part": self.part, "backend": self.backend, "metric": self.metric,
                "expected": self.expected, "actual": self.actual,
                "tol": self.tol, "detail": self.detail}


@dataclass
class GoldenResult:
    part: str
    backend: str
    ok: bool
    skipped: bool = False
    skip_reason: str = ""
    observation: Optional[Observation] = None
    deviations: List[Deviation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"part": self.part, "backend": self.backend, "ok": self.ok,
                "skipped": self.skipped, "skip_reason": self.skip_reason,
                "observation": self.observation.to_dict() if self.observation else None,
                "deviations": [d.to_dict() for d in self.deviations]}


@dataclass
class GoldenReport:
    results: List[GoldenResult] = field(default_factory=list)
    backends: List[str] = field(default_factory=list)
    skipped_backends: Dict[str, str] = field(default_factory=dict)

    @property
    def deviations(self) -> List[Deviation]:
        return [d for r in self.results for d in r.deviations]

    @property
    def ok(self) -> bool:
        return not self.deviations

    def by_backend(self) -> Dict[str, int]:
        counts = {b: 0 for b in self.backends}
        for d in self.deviations:
            counts[d.backend] = counts.get(d.backend, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "oracle": "golden",
            "ok": self.ok,
            "backends": self.backends,
            "skipped_backends": self.skipped_backends,
            "parts": len(PARTS),
            "deviations": [d.to_dict() for d in self.deviations],
            "deviations_by_backend": self.by_backend(),
            "results": [r.to_dict() for r in self.results],
        }


def check_part(part: GoldenPart, backend: str,
               factory: Optional[BackendFactory] = None) -> GoldenResult:
    """Run one part on one backend and compare against the closed form."""
    obs = observe(backend, part.ops, factory=factory)
    if not obs.available:
        return GoldenResult(part.name, backend, ok=True, skipped=True,
                            skip_reason=obs.skip_reason, observation=obs)
    tol = tolerance(backend)
    devs: List[Deviation] = []

    if obs.error:
        devs.append(Deviation(part.name, backend, "build", "a solid", "an exception",
                              detail=obs.error))
        return GoldenResult(part.name, backend, ok=False, observation=obs,
                            deviations=devs)
    if not obs.ok:
        # The op was REFUSED, so the backend still holds the pre-op solid. Its
        # volume is now a measurement of a DIFFERENT part and comparing it would
        # report the same gap twice. A refusal is a capability gap, and it is
        # reported as exactly one finding: this one.
        devs.append(Deviation(
            part.name, backend, "unsupported", "the plan applies",
            "refused at " + (obs.rejected or "?"),
            detail="codes=" + (",".join(obs.codes) or "none")
                   + "; the engine cannot build this part (it did not build it "
                     "WRONG -- it declined, which is the honest failure)"))
        return GoldenResult(part.name, backend, ok=False, observation=obs,
                            deviations=devs)
    if obs.volume is None or obs.bbox is None:
        devs.append(Deviation(part.name, backend, "build", "a measurable solid",
                              "no measurement",
                              detail="backend answered query('measure') with nothing"))
        return GoldenResult(part.name, backend, ok=False, observation=obs,
                            deviations=devs)

    vtol = tol.volume_tol(part.extent, min(part.bbox)) * part.volume
    if abs(obs.volume - part.volume) > vtol:
        detail = "off by %.2f%%" % (100.0 * (obs.volume - part.volume)
                                    / part.volume)
        if (part.rival_volume is not None
                and abs(obs.volume - part.rival_volume)
                <= tol.volume_tol(part.extent, min(part.bbox)) * part.rival_volume):
            detail += ("; this is EXACTLY the %s (%.1f) -- a semantic split in the "
                       "op, not noise" % (part.rival_name, part.rival_volume))
        devs.append(Deviation(
            part.name, backend, "volume", round(part.volume, 4),
            round(obs.volume, 4), round(vtol, 4), detail=detail))

    btol = tol.bbox_tol(part.extent)
    for axis, exp, act in zip("xyz", part.bbox, obs.bbox):
        if abs(act - exp) > btol:
            devs.append(Deviation(
                part.name, backend, "bbox", exp, round(act, 4), round(btol, 4),
                detail="%s axis off by %+.3f" % (axis, act - exp)))

    if part.genus is not None and obs.genus is not None and obs.genus != part.genus:
        devs.append(Deviation(part.name, backend, "genus", part.genus, obs.genus,
                              detail="wrong topology, not just wrong size"))

    if obs.watertight is False:
        devs.append(Deviation(part.name, backend, "watertight", True, False,
                              detail="a valid part must be closed"))

    return GoldenResult(part.name, backend, ok=not devs, observation=obs,
                        deviations=devs)


def run(backends: Optional[Sequence[str]] = None,
        parts: Optional[Sequence[GoldenPart]] = None,
        factory: Optional[BackendFactory] = None) -> GoldenReport:
    """Every part against every AVAILABLE geometric backend."""
    wanted = tuple(backends) if backends is not None else GEOMETRIC_BACKENDS
    live = available(wanted, factory)
    report = GoldenReport(backends=list(live))
    for name in wanted:
        if name not in live:
            from harnesscad.eval.selftest.probe import resolve
            report.skipped_backends[name] = resolve(name, factory)[1]
    for part in (parts if parts is not None else PARTS):
        for name in live:
            report.results.append(check_part(part, name, factory))
    return report


def format_text(report: GoldenReport) -> str:
    lines: List[str] = []
    lines.append("GOLDEN CORPUS -- analytic ground truth")
    lines.append("=" * 72)
    lines.append("%d parts x %d backends" % (len(PARTS), len(report.backends)))
    if report.skipped_backends:
        for name, why in sorted(report.skipped_backends.items()):
            lines.append("  skipped %-9s %s" % (name, why))
    lines.append("")
    counts = report.by_backend()
    lines.append("%-10s %8s" % ("backend", "misses"))
    lines.append("-" * 20)
    for name in report.backends:
        lines.append("%-10s %8d" % (name, counts.get(name, 0)))
    lines.append("")
    if not report.deviations:
        lines.append("no deviations: every backend matched every closed form.")
        return "\n".join(lines)
    lines.append("DEVIATIONS (%d)" % len(report.deviations))
    lines.append("-" * 72)
    for d in report.deviations:
        lines.append("  %-28s %-9s %-10s expected %-14s got %-14s (tol %s)"
                     % (d.part, d.backend, d.metric, d.expected, d.actual, d.tol))
        if d.detail:
            lines.append("      %s" % d.detail)
    return "\n".join(lines)
