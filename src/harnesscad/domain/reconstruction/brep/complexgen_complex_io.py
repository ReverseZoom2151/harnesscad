"""Reader / writer for the ComplexGen ``.complex`` chain-complex file format.

ComplexGen serialises a B-Rep chain complex as a plain-text ``.complex`` file
(``load_complex_file`` / ``export_visualization_file`` in
``PostProcess/complex_extraction.py``, described in
``docs/complex_extraction_complex_description.md``).  The layout is

  1. ``nv ne nf``                                       -- counts;
  2. ``nv`` lines: ``x y z``                            -- corner positions;
  3. ``ne`` lines: ``<curve_type> <closed_prob> x y z x y z ...``
     with a fixed number of samples per curve (34 in the release);
  4. ``nf`` lines: ``<patch_type> x y z x y z ...``
     with a fixed number of samples per patch (20x20 = 400, 100 in the loader);
  5. ``ne`` lines: the curve-to-corner incidence row (``nv`` numbers);
  6. ``nf`` lines: the patch-to-curve incidence row (``ne`` numbers).

Curve types are ``Circle | BSpline | Line | Ellipse``, patch types
``Cylinder | Torus | BSpline | Plane | Cone | Sphere`` -- the exact vocabularies
(and their integer ids) of the reference implementation.

The parser is tolerant of a leading comment/header line and of extra blank lines;
it infers the sample counts from the row widths instead of hard-coding 34/100, so
files written with a different sampling density round-trip too.  Values are read
as floats and the incidence matrices are returned both raw (probabilities) and
thresholded into a :class:`reconstruction.complexgen_chain_complex.ChainComplex`.
"""

from __future__ import annotations

from dataclasses import dataclass

from harnesscad.domain.reconstruction.brep.complexgen_chain_complex import (
    ChainComplex, Curve, Patch, make_complex, threshold_incidence)

CURVE_TYPES = ("Circle", "BSpline", "Line", "Ellipse")
PATCH_TYPES = ("Cylinder", "Torus", "BSpline", "Plane", "Cone", "Sphere")

CURVE_TYPE_ID = {name: i for i, name in enumerate(CURVE_TYPES)}
PATCH_TYPE_ID = {name: i for i, name in enumerate(PATCH_TYPES)}


@dataclass(frozen=True)
class ComplexFile:
    """The full content of a ``.complex`` file."""
    corners: tuple[tuple[float, float, float], ...]
    curve_types: tuple[str, ...]
    curve_closed_prob: tuple[float, ...]
    curve_points: tuple[tuple[tuple[float, float, float], ...], ...]
    patch_types: tuple[str, ...]
    patch_points: tuple[tuple[tuple[float, float, float], ...], ...]
    curve_corner: tuple[tuple[float, ...], ...]
    patch_curve: tuple[tuple[float, ...], ...]

    def to_chain_complex(self, threshold: float = 0.5,
                         closed_threshold: float = 0.5) -> ChainComplex:
        curves = [Curve(pts, self.curve_closed_prob[i] > closed_threshold)
                  for i, pts in enumerate(self.curve_points)]
        patches = [Patch(pts) for pts in self.patch_points]
        return make_complex(self.corners, curves, patches,
                            threshold_incidence(self.curve_corner, threshold),
                            threshold_incidence(self.patch_curve, threshold))


def _triples(values, name: str):
    if len(values) % 3 != 0:
        raise ValueError(f"{name}: coordinate count {len(values)} is not a multiple of 3")
    return tuple(tuple(values[i:i + 3]) for i in range(0, len(values), 3))


def parse_complex(text: str) -> ComplexFile:
    """Parse the text of a ``.complex`` file."""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise ValueError("empty complex file")

    header = lines[0].split()
    if len(header) < 3:
        raise ValueError("first line must be 'nv ne nf'")
    try:
        nv, ne, nf = (int(header[0]), int(header[1]), int(header[2]))
    except ValueError as exc:
        raise ValueError("first line must be 'nv ne nf'") from exc
    # zero-width incidence rows (nv == 0 or ne == 0) are not written out at all,
    # because a blank line carries no information in this format.
    ev_rows = ne if nv else 0
    fe_rows = nf if ne else 0
    expected = 1 + nv + ne + nf + ev_rows + fe_rows
    if len(lines) < expected:
        raise ValueError(f"expected {expected} content lines, found {len(lines)}")

    cursor = 1
    corners = []
    for _ in range(nv):
        parts = lines[cursor].split()
        cursor += 1
        if len(parts) < 3:
            raise ValueError("corner line needs 3 coordinates")
        corners.append(tuple(float(p) for p in parts[:3]))

    curve_types, curve_closed, curve_points = [], [], []
    for _ in range(ne):
        parts = lines[cursor].split()
        cursor += 1
        ctype = parts[0]
        if ctype not in CURVE_TYPE_ID:
            raise ValueError(f"unknown curve type {ctype!r}")
        curve_types.append(ctype)
        curve_closed.append(float(parts[1]))
        curve_points.append(_triples([float(p) for p in parts[2:]], "curve"))

    patch_types, patch_points = [], []
    for _ in range(nf):
        parts = lines[cursor].split()
        cursor += 1
        ptype = parts[0]
        if ptype not in PATCH_TYPE_ID:
            raise ValueError(f"unknown patch type {ptype!r}")
        patch_types.append(ptype)
        patch_points.append(_triples([float(p) for p in parts[1:]], "patch"))

    curve_corner = []
    for _ in range(ne):
        if not nv:
            curve_corner.append(())
            continue
        row = [float(p) for p in lines[cursor].split()]
        cursor += 1
        if len(row) != nv:
            raise ValueError(f"curve-corner row must have {nv} entries, got {len(row)}")
        curve_corner.append(tuple(row))

    patch_curve = []
    for _ in range(nf):
        if not ne:
            patch_curve.append(())
            continue
        row = [float(p) for p in lines[cursor].split()]
        cursor += 1
        if len(row) != ne:
            raise ValueError(f"patch-curve row must have {ne} entries, got {len(row)}")
        patch_curve.append(tuple(row))

    return ComplexFile(
        corners=tuple(corners),
        curve_types=tuple(curve_types),
        curve_closed_prob=tuple(curve_closed),
        curve_points=tuple(curve_points),
        patch_types=tuple(patch_types),
        patch_points=tuple(patch_points),
        curve_corner=tuple(curve_corner),
        patch_curve=tuple(patch_curve),
    )


def _fmt(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def serialize_complex(cf: ComplexFile, precision: int = 6) -> str:
    """Serialise back to the ``.complex`` text format (round-trips :func:`parse_complex`)."""
    nv, ne, nf = len(cf.corners), len(cf.curve_points), len(cf.patch_points)
    out = [f"{nv} {ne} {nf}"]
    for c in cf.corners:
        out.append(" ".join(_fmt(x, precision) for x in c))
    for i in range(ne):
        coords = [x for p in cf.curve_points[i] for x in p]
        out.append(" ".join([cf.curve_types[i], _fmt(cf.curve_closed_prob[i], precision)]
                            + [_fmt(x, precision) for x in coords]))
    for k in range(nf):
        coords = [x for p in cf.patch_points[k] for x in p]
        out.append(" ".join([cf.patch_types[k]] + [_fmt(x, precision) for x in coords]))
    for row in cf.curve_corner:
        if row:
            out.append(" ".join(_fmt(v, precision) for v in row))
    for row in cf.patch_curve:
        if row:
            out.append(" ".join(_fmt(v, precision) for v in row))
    return "\n".join(out) + "\n"


def from_chain_complex(cx: ChainComplex,
                       curve_types=None, patch_types=None) -> ComplexFile:
    """Build a :class:`ComplexFile` from a definite :class:`ChainComplex`.

    Types default to ``BSpline`` (the generic type in both vocabularies).
    """
    if curve_types is None:
        curve_types = ["BSpline"] * cx.n_curves
    if patch_types is None:
        patch_types = ["BSpline"] * cx.n_patches
    if len(curve_types) != cx.n_curves or len(patch_types) != cx.n_patches:
        raise ValueError("type lists must match the number of cells")
    for t in curve_types:
        if t not in CURVE_TYPE_ID:
            raise ValueError(f"unknown curve type {t!r}")
    for t in patch_types:
        if t not in PATCH_TYPE_ID:
            raise ValueError(f"unknown patch type {t!r}")
    return ComplexFile(
        corners=cx.corners,
        curve_types=tuple(curve_types),
        curve_closed_prob=tuple(1.0 if c.closed else 0.0 for c in cx.curves),
        curve_points=tuple(c.points for c in cx.curves),
        patch_types=tuple(patch_types),
        patch_points=tuple(p.points for p in cx.patches),
        curve_corner=tuple(tuple(float(v) for v in row) for row in cx.curve_corner),
        patch_curve=tuple(tuple(float(v) for v in row) for row in cx.patch_curve),
    )


def load_complex(path: str) -> ComplexFile:
    with open(path, "r", encoding="utf-8") as handle:
        return parse_complex(handle.read())


def save_complex(path: str, cf: ComplexFile, precision: int = 6) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(serialize_complex(cf, precision))
