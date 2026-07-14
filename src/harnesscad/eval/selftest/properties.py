"""Property / metamorphic oracle — invariants that hold for ANY part, ANY engine.

The golden corpus needs a part someone thought of. This does not. It states the
laws geometry obeys, generates a seeded random corpus of op streams, and hunts for
a counterexample. It is the only oracle here that can find a bug in a part nobody
has ever drawn.

The laws, and why each one is here:

``shell_does_not_grow``
    Hollowing a part cannot make it bigger. This one sentence, checked once,
    would have caught the bug that shipped in the README: a 60x40x20 box shelled
    at 3 mm came out 63x43x23 -- watertight, valid, beautiful, and 3 mm oversize
    on every face. Twenty-three verifiers and ~200 benchmark modules never asked.

``shell_does_not_shrink``
    And it cannot make it SMALLER either. The outer surface does not move, in
    either direction -- ``shell`` hollows inward, full stop. This law was added
    after its twin above found nothing and this one found a real bug: the F-rep
    shell of an 80x30x5 box at t=1 comes back 78.1 x 28.2 x 3.5, eroded on every
    face, 75% under volume, watertight and silent. The wall (1 mm) is smaller than
    the sampling grid's cell (80/48 = 1.67 mm), so the field cannot represent it
    and the engine quietly builds a different, smaller part instead of refusing.
    An engine that cannot build the part must SAY SO.

``shell_wall_is_thickness_t``
    And a bbox check CANNOT PROVE A SHELL. An inward shell can hold the envelope
    to the micron and still leave the wall at ``t/sqrt(3)`` -- 42% too thin, from
    an uncorrected corner normal -- because the error is on the INSIDE, where an
    envelope check has no vision at all. The corpus's box-shell streams are
    generated here, so their exact closed-hollow volume
    ``abc - (a-2t)(b-2t)(c-2t)`` is known, and that volume pins the wall. This law
    is the reason the previous one is not enough.

``cut_does_not_add`` / ``union_does_not_remove``
    A boolean cut removes material; a union adds it. Monotonic, no exceptions.

``scale_is_cubic`` (METAMORPHIC)
    There is no scale op, so the relation is stated on the INPUT: multiply every
    length in the plan by k and the volume must go up by k^3 and the bbox by k.
    This needs no ground truth at all -- it relates two runs of the SAME engine,
    so it holds even for an engine whose absolute numbers are all wrong. It is
    the check that survives when everything else is uncalibrated.

``extrude_gives_height``
    An extrude of distance d makes a part d tall. The most basic op in the set.

``valid_is_closed``
    A part the harness calls valid must be watertight and 2-manifold. If the
    harness will hand it to a slicer, it must be a solid.

``replay_is_identical``
    The CISP contract's central promise: the same ops twice give the same digest.
    Break this and every cached result, every checkpoint and every regression test
    in the repo is quietly meaningless.

Every violation is reported WITH THE OP STREAM that produced it, so it can be
replayed by hand. The corpus is seeded: the same seed gives the same 200 streams
on every machine, forever.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import (AddCircle, AddRectangle, Boolean, Chamfer,
                                      Extrude, Fillet, Hole, LinearPattern,
                                      NewSketch, Op, Shell, canonical_json)
from harnesscad.eval.selftest.probe import (BackendFactory, Observation,
                                            bbox_delta, observe, observe_steps,
                                            resolve, scale_ops, tolerance,
                                            volume_rel_delta)

__all__ = ["Violation", "PropertyReport", "generate_corpus", "check_stream",
           "run", "format_text", "PROPERTIES"]

#: Named for the report; the check functions live in ``check_stream``.
PROPERTIES: Tuple[str, ...] = (
    "shell_does_not_grow",
    "shell_does_not_shrink",
    "shell_wall_is_thickness_t",
    "cut_does_not_add",
    "union_does_not_remove",
    "scale_is_cubic",
    "extrude_gives_height",
    "valid_is_closed",
    "replay_is_identical",
)


@dataclass
class Violation:
    """One law, broken once, by one engine, on one op stream."""

    prop: str
    backend: str
    stream: str                       # the seeded stream's name
    detail: str
    ops: List[str] = field(default_factory=list)   # canonical JSON, replayable

    def to_dict(self) -> dict:
        return {"property": self.prop, "backend": self.backend,
                "stream": self.stream, "detail": self.detail, "ops": self.ops}


@dataclass
class PropertyReport:
    backends: List[str] = field(default_factory=list)
    skipped_backends: Dict[str, str] = field(default_factory=dict)
    seed: int = 0
    streams: int = 0
    checked: int = 0                                  # property checks performed
    violations: List[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def by_property(self) -> Dict[str, int]:
        out = {p: 0 for p in PROPERTIES}
        for v in self.violations:
            out[v.prop] = out.get(v.prop, 0) + 1
        return out

    def by_backend(self) -> Dict[str, int]:
        out = {b: 0 for b in self.backends}
        for v in self.violations:
            out[v.backend] = out.get(v.backend, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {"oracle": "properties", "ok": self.ok, "seed": self.seed,
                "backends": self.backends,
                "skipped_backends": self.skipped_backends,
                "streams": self.streams, "checks": self.checked,
                "violations_by_property": self.by_property(),
                "violations_by_backend": self.by_backend(),
                "violations": [v.to_dict() for v in self.violations]}


# --- the seeded corpus -----------------------------------------------------

def generate_corpus(count: int = 200, seed: int = 20260714
                    ) -> List[Tuple[str, List[Op]]]:
    """``count`` op streams from one seed. Same seed, same streams, forever.

    Every stream is FEASIBLE by construction (a fillet below half the thinnest
    extent, a shell that leaves a cavity, a hole that leaves material). That is
    deliberate: a property violation on an infeasible plan tells you nothing --
    the engine was entitled to produce nonsense. A violation on a plan that is
    obviously buildable is a bug.
    """
    rng = random.Random(seed)
    streams: List[Tuple[str, List[Op]]] = []
    for i in range(count):
        kind = rng.choice(["box", "cylinder", "plate_holes", "shell", "fillet",
                           "chamfer", "cut", "union", "pattern"])
        w = float(rng.randrange(20, 120, 5))
        h = float(rng.randrange(20, 80, 5))
        d = float(rng.randrange(5, 40, 5))
        ops: List[Op] = [NewSketch("XY")]
        if kind == "box":
            ops += [AddRectangle("sk1", 0, 0, w, h), Extrude("sk1", d)]
        elif kind == "cylinder":
            r = float(rng.randrange(5, 40, 5))
            ops += [AddCircle("sk1", 0, 0, r), Extrude("sk1", d)]
        elif kind == "plate_holes":
            n = rng.randint(1, 4)
            hole_d = float(rng.randrange(2, 10, 2))
            ops += [AddRectangle("sk1", 0, 0, w, h), Extrude("sk1", d)]
            margin = hole_d
            for k in range(n):
                x = margin + (w - 2 * margin) * (k + 1) / (n + 1)
                ops.append(Hole("sk1", x, h / 2.0, hole_d, None, True, "simple"))
        elif kind == "shell":
            # t < half the thinnest extent, so a cavity always survives.
            thin = min(w, h, d)
            t = max(1.0, round(rng.uniform(0.1, 0.35) * thin, 1))
            ops += [AddRectangle("sk1", 0, 0, w, h), Extrude("sk1", d), Shell((), t)]
        elif kind == "fillet":
            thin = min(w, h, d)
            r = max(0.5, round(rng.uniform(0.1, 0.4) * thin, 1))
            ops += [AddRectangle("sk1", 0, 0, w, h), Extrude("sk1", d), Fillet((), r)]
        elif kind == "chamfer":
            thin = min(w, h, d)
            c = max(0.5, round(rng.uniform(0.1, 0.4) * thin, 1))
            ops += [AddRectangle("sk1", 0, 0, w, h), Extrude("sk1", d),
                    Chamfer((), c)]
        elif kind == "cut":
            cw = round(w / 3.0, 1)
            ch = round(h / 3.0, 1)
            ops += [AddRectangle("sk1", 0, 0, w, h), Extrude("sk1", d),
                    NewSketch("XY"), AddRectangle("sk2", 0, 0, cw, ch),
                    Extrude("sk2", d), Boolean("cut", "f1", "f2")]
        elif kind == "union":
            ops += [AddRectangle("sk1", 0, 0, w, h), Extrude("sk1", d),
                    NewSketch("XY"),
                    AddRectangle("sk2", w / 2.0, h / 2.0, w, h),
                    Extrude("sk2", d), Boolean("union", "f1", "f2")]
        else:  # pattern
            n = rng.randint(2, 4)
            ops += [AddRectangle("sk1", 0, 0, w / 4.0, h / 4.0), Extrude("sk1", d),
                    LinearPattern("f1", (1.0, 0.0, 0.0), n, w)]
        streams.append(("%s_%03d" % (kind, i), ops))
    return streams


# --- the laws --------------------------------------------------------------

def _op_index(ops: Sequence[Op], cls: type) -> Optional[int]:
    for i, op in enumerate(ops):
        if isinstance(op, cls):
            return i
    return None


def check_stream(name: str, ops: Sequence[Op], backend: str,
                 factory: Optional[BackendFactory] = None,
                 scale_k: float = 2.0) -> Tuple[List[Violation], int]:
    """Check every law that applies to this stream on this engine.

    Returns (violations, number of checks performed). Pure with respect to the
    harness: it only ever reads.
    """
    sig = [canonical_json(op) for op in ops]
    viol: List[Violation] = []
    checks = 0
    tol = tolerance(backend)

    def fail(prop: str, detail: str) -> None:
        viol.append(Violation(prop, backend, name, detail, list(sig)))

    steps, applied = observe_steps(backend, ops, factory=factory)
    if not steps or not steps[0].available:
        return [], 0
    final = steps[-1]
    if final.error:
        fail("valid_is_closed", "the engine raised: %s" % final.error)
        return viol, 1
    if not final.ok:
        # The engine refused an op it was given a feasible plan for. Not a law
        # violation -- a capability gap -- and the remaining laws have no state to
        # be checked against, so we stop here without inventing failures.
        return viol, 0

    extent = final.extent
    btol = tol.bbox_tol(extent) + 1e-6
    vtol = tol.volume_tol(extent, final.min_extent)

    # -- shell must not grow the part -------------------------------------
    i = _op_index(ops, Shell)
    if i is not None and i < len(steps) and i > 0:
        before, after = steps[i - 1], steps[i]
        if before.geometric and after.geometric and after.ok:
            checks += 1
            grew = [(ax, b, a) for ax, b, a in zip("xyz", before.bbox, after.bbox)
                    if a > b + btol]
            if grew:
                fail("shell_does_not_grow",
                     "bbox %s -> %s: a shell HOLLOWS, it cannot dilate (%s)"
                     % ([round(v, 2) for v in before.bbox],
                        [round(v, 2) for v in after.bbox],
                        ", ".join("%s +%.2f" % (ax, a - b) for ax, b, a in grew)))
            checks += 1
            if after.volume > before.volume * (1.0 + vtol):
                fail("shell_does_not_grow",
                     "volume %.1f -> %.1f: hollowing ADDED %.1f mm3"
                     % (before.volume, after.volume,
                        after.volume - before.volume))
            checks += 1
            shrank = [(ax, b, a) for ax, b, a in zip("xyz", before.bbox, after.bbox)
                      if a < b - btol]
            if shrank:
                fail("shell_does_not_shrink",
                     "bbox %s -> %s: a shell hollows INWARD -- the outer surface "
                     "must not move at all (%s). The engine built a smaller part "
                     "and did not say so."
                     % ([round(v, 2) for v in before.bbox],
                        [round(v, 2) for v in after.bbox],
                        ", ".join("%s %.2f" % (ax, a - b) for ax, b, a in shrank)))

    # -- the wall must actually be t thick ---------------------------------
    # Only decidable where we KNOW the part: a box, extruded, then shelled. Those
    # streams are generated in this module, so the closed form is available and the
    # law can be stated exactly instead of gestured at.
    if (i is not None and final.geometric and final.ok
            and len(ops) == 4
            and isinstance(ops[1], AddRectangle) and isinstance(ops[2], Extrude)):
        a, b = float(ops[1].w), float(ops[1].h)
        c, t = float(ops[2].distance), float(ops[3].thickness)
        if min(a, b, c) > 2 * t:
            checks += 1
            want = a * b * c - (a - 2 * t) * (b - 2 * t) * (c - 2 * t)
            if volume_rel_delta(final.volume, want) > vtol:
                thin = a * b * c - (a - 2 * t / 3 ** 0.5) * (b - 2 * t / 3 ** 0.5) \
                    * (c - 2 * t / 3 ** 0.5)
                hint = ""
                if volume_rel_delta(final.volume, thin) <= vtol:
                    hint = (" -- and this is EXACTLY a t/sqrt(3) wall (%.1f), the "
                            "uncorrected-corner-normal bug, which the bbox check "
                            "cannot see" % thin)
                fail("shell_wall_is_thickness_t",
                     "shell(t=%.1f) of a %gx%gx%g box: volume %.1f, closed hollow "
                     "says %.1f (off by %+.1f%%)%s"
                     % (t, a, b, c, final.volume, want,
                        100.0 * (final.volume - want) / want, hint))

    # -- boolean monotonicity ----------------------------------------------
    i = _op_index(ops, Boolean)
    if i is not None and i < len(steps) and i > 0:
        bop = ops[i]
        before, after = steps[i - 1], steps[i]
        if before.geometric and after.geometric and after.ok:
            checks += 1
            if bop.kind == "cut" and after.volume > before.volume * (1.0 + vtol):
                fail("cut_does_not_add",
                     "cut took volume %.1f -> %.1f (it went UP)"
                     % (before.volume, after.volume))
            # NOTE: 'before' is the compound of BOTH operand bodies, so a union
            # must not come out SMALLER than the larger operand. The compound's
            # volume double-counts any overlap, so the only sound bound is the
            # lower one.
            if bop.kind == "union" and after.volume < before.volume * (1.0 - vtol) \
                    and after.volume < before.volume / 2.0:
                fail("union_does_not_remove",
                     "union took volume %.1f -> %.1f, less than half the operands' "
                     "total: material vanished" % (before.volume, after.volume))

    # -- extrude of distance d gives a part d tall -------------------------
    i = _op_index(ops, Extrude)
    if i is not None and i < len(steps) and steps[i].geometric and steps[i].ok:
        checks += 1
        got = steps[i].bbox[2]
        want = float(ops[i].distance)
        if abs(got - want) > tol.bbox_tol(max(want, steps[i].extent)) + 1e-6:
            fail("extrude_gives_height",
                 "extrude(distance=%.2f) built a part %.3f tall" % (want, got))

    # -- a valid part is closed and 2-manifold ------------------------------
    if final.geometric:
        checks += 1
        if final.watertight is False or final.manifold is False:
            fail("valid_is_closed",
                 "the harness accepted a part that is watertight=%s manifold=%s"
                 % (final.watertight, final.manifold))

    # -- metamorphic: scale the PLAN by k, the volume must go up by k^3 ------
    if final.geometric and final.volume > 0:
        scaled = observe(backend, scale_ops(ops, scale_k), factory=factory)
        if scaled.geometric and scaled.ok:
            checks += 1
            want_v = final.volume * scale_k ** 3
            # Tolerance compounds: both runs carry the engine's own error.
            if volume_rel_delta(scaled.volume, want_v) > 2 * vtol + 0.01:
                fail("scale_is_cubic",
                     "scaling every length by %g took volume %.1f -> %.1f; k^3 "
                     "says %.1f (off by %+.1f%%)"
                     % (scale_k, final.volume, scaled.volume, want_v,
                        100.0 * (scaled.volume - want_v) / want_v))
            checks += 1
            want_b = tuple(v * scale_k for v in final.bbox)
            if bbox_delta(scaled.bbox, want_b) > 2 * tol.bbox_tol(
                    max(want_b)) + 1e-6:
                fail("scale_is_cubic",
                     "scaling every length by %g took bbox %s -> %s; k says %s"
                     % (scale_k, [round(v, 2) for v in final.bbox],
                        [round(v, 2) for v in scaled.bbox],
                        [round(v, 2) for v in want_b]))

    # -- determinism: the same ops twice give the same digest ---------------
    again = observe(backend, ops, factory=factory)
    if again.available and not again.error:
        checks += 1
        if again.digest != final.digest:
            fail("replay_is_identical",
                 "replaying the same ops gave digest %s, was %s"
                 % (again.digest[:16], final.digest[:16]))
        if again.geometric and final.geometric:
            checks += 1
            if volume_rel_delta(again.volume, final.volume) > 1e-9:
                fail("replay_is_identical",
                     "replaying the same ops gave volume %.6f, was %.6f"
                     % (again.volume, final.volume))
    return viol, checks


def run(backends: Sequence[str] = ("frep",),
        count: int = 200,
        seed: int = 20260714,
        factory: Optional[BackendFactory] = None,
        corpus: Optional[Sequence[Tuple[str, Sequence[Op]]]] = None
        ) -> PropertyReport:
    """Check every law on every stream on every requested engine.

    The default engine is ``frep``: it is always available (no dependency at all),
    so this oracle runs on any machine. Ask for more engines and it will use them.
    """
    report = PropertyReport(seed=seed)
    the_corpus = list(corpus) if corpus is not None else generate_corpus(count, seed)
    report.streams = len(the_corpus)
    for name in backends:
        engine, skip = resolve(name, factory)
        if engine is None:
            report.skipped_backends[name] = skip
            continue
        report.backends.append(name)
    for name in report.backends:
        for stream_name, ops in the_corpus:
            viol, checks = check_stream(stream_name, ops, name, factory=factory)
            report.violations.extend(viol)
            report.checked += checks
    return report


def format_text(report: PropertyReport) -> str:
    lines: List[str] = []
    lines.append("PROPERTIES -- laws that hold for any part, on any engine")
    lines.append("=" * 76)
    lines.append("%d seeded streams (seed %d) x %s -> %d property checks"
                 % (report.streams, report.seed,
                    ", ".join(report.backends) or "no engine", report.checked))
    for name, why in sorted(report.skipped_backends.items()):
        lines.append("  skipped %-9s %s" % (name, why))
    lines.append("")
    lines.append("%-24s %10s" % ("law", "violations"))
    lines.append("-" * 36)
    counts = report.by_property()
    for p in PROPERTIES:
        lines.append("%-24s %10d" % (p, counts.get(p, 0)))
    lines.append("")
    if report.ok:
        lines.append("no violations.")
        return "\n".join(lines)
    lines.append("VIOLATIONS (%d)" % len(report.violations))
    lines.append("-" * 76)
    # Group by (law, engine, detail shape) so 40 instances of one bug read as one
    # bug with 40 witnesses, not as 40 findings.
    seen: Dict[Tuple[str, str], List[Violation]] = {}
    for v in report.violations:
        seen.setdefault((v.prop, v.backend), []).append(v)
    for (prop, backend), group in sorted(seen.items()):
        lines.append("  %s on %s -- %d stream(s)" % (prop, backend, len(group)))
        for v in group[:3]:
            lines.append("      %s: %s" % (v.stream, v.detail))
            lines.append("      ops: %s" % " ".join(v.ops))
        if len(group) > 3:
            lines.append("      ... and %d more" % (len(group) - 3))
    return "\n".join(lines)
