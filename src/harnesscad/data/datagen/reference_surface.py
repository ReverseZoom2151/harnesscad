"""Reference-surface program generation for design-procedure data augmentation.

Chen, Shu, Hong, Taber, Li & Klenk, *Learning From Design Procedure To Generate
CAD Programs for Data Augmentation* (NeurIPS 2025 Workshop, Deep Learning for
Code in the Agentic Era).

The paper's central data-augmentation idea is a *reference surface program*: a
short parametric script that defines a free-form **B-Spline** surface, which is
prepended to an LLM prompt so the generated CAD object conforms to (and thereby
inherits the organic curvature of) that surface. The surface is later removed,
but its curvature has already rippled into the object's faces and edges.

The *learned* half of that pipeline (prompting o3 to write the conforming
bracket) is external. This module implements the deterministic half: the
generation of the reference-surface control-point nets themselves and their
emission as CadQuery-style Python scripts, exactly the four families the paper
lists (Sec. 4.1, Appendix B):

    Gaussian surface   z = H * exp(-(x^2 + y^2) / (span/3)^2)
    Saddle surface     z = curv * (x^2 - y^2)
    Wave surface       z = amp * sin(freq * x)
    Ripple surface     z = amp * cos(freq * r),  r = hypot(x, y)

"To further increase shape diversity, we vary the parameters of each reference
surface script" (Sec. 3.2) -- ``vary_parameters`` produces a deterministic
parameter sweep (e.g. saddle curvature shallow -> deep) for exactly that.

All geometry is plain Python (no CadQuery import); scripts are emitted as text.
Determinism: randomness (only used by ``vary_parameters``) flows through a
supplied ``random.Random`` or integer seed; stdlib only; no wall clock.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Sequence, Tuple

Point3 = Tuple[float, float, float]
Net = List[List[Point3]]


# ---------------------------------------------------------------------------
# Surface height fields (the four B-Spline families from the paper)
# ---------------------------------------------------------------------------

def gaussian_height(x: float, y: float, span: float, height: float) -> float:
    """Gaussian bump height (Appendix B Example 2): ``H * exp(-r2)``."""
    denom = (span / 3.0) ** 2
    r2 = (x * x + y * y) / denom
    return height * math.exp(-r2)


def saddle_height(x: float, y: float, curv: float) -> float:
    """Hyperbolic-paraboloid (saddle) height (Appendix B Example 1)."""
    return curv * (x * x - y * y)


def wave_height(x: float, y: float, amp: float, freq: float) -> float:
    """Single-direction sinusoidal wave height ``amp * sin(freq * x)``."""
    return amp * math.sin(freq * x)


def ripple_height(x: float, y: float, amp: float, freq: float) -> float:
    """Concentric ripple height ``amp * cos(freq * r)``, ``r = hypot(x, y)``."""
    return amp * math.cos(freq * math.hypot(x, y))


# Registry: kind -> (height_fn(x, y, **params), required-param-keys, defaults).
_SURFACES: Dict[str, Tuple[Callable, Tuple[str, ...], Dict[str, float]]] = {
    "gaussian": (
        lambda x, y, span, height, **_: gaussian_height(x, y, span, height),
        ("span", "height"),
        {"span": 100.0, "height": 7.0},
    ),
    "saddle": (
        lambda x, y, curv, **_: saddle_height(x, y, curv),
        ("curv",),
        {"curv": 0.004},
    ),
    "wave": (
        lambda x, y, amp, freq, **_: wave_height(x, y, amp, freq),
        ("amp", "freq"),
        {"amp": 10.0, "freq": 0.05},
    ),
    "ripple": (
        lambda x, y, amp, freq, **_: ripple_height(x, y, amp, freq),
        ("amp", "freq"),
        {"amp": 8.0, "freq": 0.06},
    ),
}

SURFACE_TYPES: Tuple[str, ...] = tuple(_SURFACES)


def default_params(kind: str) -> Dict[str, float]:
    """Return a fresh copy of the paper-default parameters for ``kind``."""
    _require_kind(kind)
    d = dict(_SURFACES[kind][2])
    d["span"] = d.get("span", 100.0)
    return d


def _require_kind(kind: str) -> None:
    if kind not in _SURFACES:
        raise ValueError(
            "unknown surface kind %r (known: %s)"
            % (kind, ", ".join(SURFACE_TYPES))
        )


# ---------------------------------------------------------------------------
# Control-point net generation
# ---------------------------------------------------------------------------

def make_net(kind: str, params: Dict[str, float] = None,
             resolution: int = 8, span: float = 100.0) -> Net:
    """Build the ``resolution x resolution`` control-point net of a surface.

    Mirrors the double loop in the paper's Appendix B scripts: ``u, v`` sweep
    ``[0, 1]``, ``x, y`` span ``[-span/2, span/2]`` and ``z`` is the height
    field. Returns a list of rows, each a list of ``(x, y, z)`` points.
    """
    _require_kind(kind)
    if resolution < 2:
        raise ValueError("resolution must be >= 2")
    height_fn = _SURFACES[kind][0]
    p = dict(_SURFACES[kind][2])
    if params:
        p.update(params)
    p.setdefault("span", span)
    sp = float(p["span"])
    net: Net = []
    for i in range(resolution):
        u = i / (resolution - 1)
        x = (u - 0.5) * sp
        row: List[Point3] = []
        for j in range(resolution):
            v = j / (resolution - 1)
            y = (v - 0.5) * sp
            z = height_fn(x, y, **p)
            row.append((x, y, z))
        net.append(row)
    return net


def net_z_range(net: Net) -> Tuple[float, float]:
    """Return ``(min_z, max_z)`` over all points of a net."""
    zs = [pt[2] for row in net for pt in row]
    return (min(zs), max(zs))


def is_curved(net: Net, tol: float = 1e-9) -> bool:
    """True if the net is non-planar (has real z-variation => B-Spline surface).

    A flat (constant-z) net is a degenerate reference surface that would induce
    no organic curvature; the paper's whole premise is that reference surfaces
    are genuinely curved B-Splines.
    """
    lo, hi = net_z_range(net)
    return (hi - lo) > tol


# ---------------------------------------------------------------------------
# CadQuery-style script emission (the "reference surface program" text)
# ---------------------------------------------------------------------------

def emit_script(kind: str, params: Dict[str, float] = None,
                resolution: int = 100, span: float = 100.0) -> str:
    """Emit a CadQuery-style Python reference-surface program as text.

    The emitted script matches the structure of Appendix B: a ``U x V`` loop
    that appends ``cq.Vector(x, y, z)`` rows and calls
    ``cq.Face.makeSplineApprox(net)``. It is deterministic text -- no CadQuery
    is imported or executed here.
    """
    _require_kind(kind)
    p = dict(_SURFACES[kind][2])
    if params:
        p.update(params)
    p.setdefault("span", span)
    sp = float(p["span"])
    lines: List[str] = []
    lines.append("# %s.py" % kind)
    lines.append("import cadquery as cq, math")
    lines.append("U, V, SPAN = %d, %d, %g" % (resolution, resolution, sp))
    lines.append("net = []")
    lines.append("for i in range(U):")
    lines.append("    u = i / (U - 1); x = (u - 0.5) * SPAN")
    lines.append("    row = []")
    lines.append("    for j in range(V):")
    lines.append("        v = j / (V - 1); y = (v - 0.5) * SPAN")
    lines.append("        z = %s" % _height_expr(kind, p))
    lines.append("        row.append(cq.Vector(x, y, z))")
    lines.append("    net.append(row)")
    lines.append("surf = cq.Face.makeSplineApprox(net)")
    lines.append('cq.exporters.export(surf, "%s.step")' % kind)
    return "\n".join(lines)


def _height_expr(kind: str, p: Dict[str, float]) -> str:
    """Return the ``z = ...`` right-hand-side expression string for ``kind``."""
    if kind == "gaussian":
        return ("%g * math.exp(-((x**2 + y**2) / ((SPAN/3)**2)))"
                % float(p["height"]))
    if kind == "saddle":
        return "%g * (x**2 - y**2)" % float(p["curv"])
    if kind == "wave":
        return "%g * math.sin(%g * x)" % (float(p["amp"]), float(p["freq"]))
    if kind == "ripple":
        return ("%g * math.cos(%g * math.hypot(x, y))"
                % (float(p["amp"]), float(p["freq"])))
    raise ValueError("unknown kind %r" % kind)  # pragma: no cover


def script_line_count(script: str) -> int:
    """Number of non-empty lines in an emitted script."""
    return sum(1 for ln in script.splitlines() if ln.strip())


# ---------------------------------------------------------------------------
# Parameter variation (Sec. 3.2: "vary the parameters ... shallow -> deep")
# ---------------------------------------------------------------------------

def _rng(seed) -> random.Random:
    if isinstance(seed, random.Random):
        return seed
    return random.Random(seed)


# Per-kind sweepable parameter ranges (lo, hi) for diversity.
_SWEEP_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "gaussian": {"height": (3.0, 15.0)},
    "saddle": {"curv": (0.001, 0.010)},
    "wave": {"amp": (4.0, 16.0), "freq": (0.02, 0.10)},
    "ripple": {"amp": (3.0, 12.0), "freq": (0.03, 0.12)},
}


def sweep_parameters(kind: str, n: int) -> List[Dict[str, float]]:
    """Deterministic *linear* parameter sweep of length ``n`` for ``kind``.

    The first sweepable key is stepped uniformly from its low to high bound
    ("shallow to deep" in the paper's saddle example); any other sweepable keys
    are held at their midpoint. No randomness -- the sweep is fully reproducible.
    """
    _require_kind(kind)
    if n < 1:
        raise ValueError("n must be >= 1")
    ranges = _SWEEP_RANGES.get(kind, {})
    out: List[Dict[str, float]] = []
    keys = list(ranges)
    for i in range(n):
        p = dict(_SURFACES[kind][2])
        p.setdefault("span", 100.0)
        frac = 0.0 if n == 1 else i / (n - 1)
        for k_idx, key in enumerate(keys):
            lo, hi = ranges[key]
            if k_idx == 0:
                p[key] = lo + (hi - lo) * frac
            else:
                p[key] = (lo + hi) / 2.0
        out.append(p)
    return out


def vary_parameters(kind: str, seed, n: int) -> List[Dict[str, float]]:
    """Deterministic *random* parameter variation of length ``n`` for ``kind``.

    Each sweepable key is drawn uniformly from its range using ``random.Random``,
    giving a diverse but reproducible family of reference surfaces.
    """
    _require_kind(kind)
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = _rng(seed)
    ranges = _SWEEP_RANGES.get(kind, {})
    out: List[Dict[str, float]] = []
    for _ in range(n):
        p = dict(_SURFACES[kind][2])
        p.setdefault("span", 100.0)
        for key, (lo, hi) in ranges.items():
            p[key] = rng.uniform(lo, hi)
        out.append(p)
    return out


def surface_family(seed, per_kind: int,
                   kinds: Sequence[str] = SURFACE_TYPES) -> List[dict]:
    """Build a diverse family of reference surfaces across ``kinds``.

    Returns a list of dicts ``{"kind", "params", "script", "curved"}``. Each
    kind contributes ``per_kind`` parameter-varied surfaces, seeded from a stable
    per-kind sub-seed so the full family is deterministic in ``(seed, per_kind,
    kinds)``.
    """
    if per_kind < 1:
        raise ValueError("per_kind must be >= 1")
    out: List[dict] = []
    for idx, kind in enumerate(kinds):
        _require_kind(kind)
        sub = _rng(seed).randint(0, 2 ** 31 - 1) ^ (idx * 0x9E3779B1)
        for params in vary_parameters(kind, sub & 0x7FFFFFFF, per_kind):
            net = make_net(kind, params, resolution=6)
            out.append({
                "kind": kind,
                "params": params,
                "script": emit_script(kind, params, resolution=100),
                "curved": is_curved(net),
            })
    return out
