"""import_brep — load a reference solid (STEP / IGES / STL) via OCCT.

The *inverse* of the CadQuery backend's export path (see
``backends.cadquery_backend.export``): where the backend turns an op stream into
a B-rep and writes STEP/IGES/STL, this reads such a file back into a *measurable
reference solid* — a fixed geometry we can score a generated model against
(reference-match verification), decompile into a best-effort feature tree, or
ingest into the RAG store as a retrievable precedent.

OCCT (via cadquery / cadquery-ocp) is imported LAZILY and every entry point is
GUARDED: with no kernel installed, a missing file, or an unsupported/broken
file, :func:`import_solid` returns a clean, measurable-less
:class:`ImportedPart` carrying a human-readable ``note`` — it NEVER raises. So
the whole ingest layer imports and runs on a machine with no geometry kernel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# path extension -> logical format tag
_EXT_FMT = {
    ".step": "step", ".stp": "step",
    ".iges": "iges", ".igs": "iges",
    ".stl": "stl",
}


@dataclass
class ImportedPart:
    """A loaded reference solid (or a clear record of why it could not load).

    - ``path``     : the source file path.
    - ``fmt``      : detected format tag ("step" | "iges" | "stl" | "unknown").
    - ``shape``    : the loaded OCCT/cq shape when available, else ``None``.
    - ``metrics``  : mass properties (volume/surface_area/bbox/center_of_mass/
                     counts) when measurable, else ``{}``.
    - ``bbox``     : ``[dx, dy, dz]`` bounding-box extents (``[0,0,0]`` if none).
    - ``available``: True only when a real, measured solid was loaded.
    - ``note``     : why the part is unavailable / a status message.
    """

    path: str
    fmt: str = "unknown"
    shape: object = None
    metrics: dict = field(default_factory=dict)
    bbox: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    available: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.available

    @property
    def volume(self) -> float:
        v = self.metrics.get("volume") if self.metrics else None
        return float(v) if isinstance(v, (int, float)) else 0.0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "fmt": self.fmt,
            "available": self.available,
            "metrics": dict(self.metrics),
            "bbox": list(self.bbox),
            "note": self.note,
        }


def _cq():
    """Lazy import of cadquery so this module loads without OCCT installed."""
    import cadquery  # noqa: WPS433 (deliberately local / lazy)
    return cadquery


def _step_units(path: str) -> tuple:
    """The file's DECLARED length unit, read kernel-free. ``(units_dict, note)``.

    This does NOT rescale anything, and that is deliberate. OCCT's STEP reader
    already resolves the file's declared unit and hands back millimetres: a
    part-21 file declaring ``SI_UNIT($,.METRE.)`` for a 10 m box measures 10000
    on ``ImportedPart.bbox``, not 10. Multiplying by ``scale_to_mm`` here would
    apply the same 1000x a SECOND time -- it would CAUSE the silent-1000x bug
    this module's units awareness exists to catch, not fix it.

    What was actually missing is that the rescale was INVISIBLE: a metre-declared
    file quietly became a 10000 mm part with nothing saying why. So this reads
    the declaration independently (io/formats/step_units.py, no kernel) and
    reports it, giving the importer a cross-check against OCCT's own conversion.

    Never raises: an unparseable/absent declaration is simply no units.
    """
    try:
        from harnesscad.io.formats.step_units import extract_step_units
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            units = extract_step_units(fh.read())
    except Exception:  # noqa: BLE001 - units are advisory, never a read failure
        return {}, ""
    info = units.to_dict()
    # A millimetre file (scale 1.0) is the assumed case and stays silent, so an
    # mm import's `note` is exactly what it always was. Only an explicitly
    # non-mm declaration -- the case that silently moved the numbers -- speaks.
    if abs(units.scale_to_mm - 1.0) <= 1e-12:
        return info, ""
    note = (f"declared unit {units.unit_name} (x{units.scale_to_mm:g} to mm); "
            "measurements below are millimetres after the kernel's own unit "
            "conversion")
    extra = "; ".join(str(n) for n in units.notes)
    return info, f"{note} [{extra}]" if extra else note


def detect_format(path: str) -> str:
    """Format tag from the file extension ("unknown" if unrecognised)."""
    _, ext = os.path.splitext(path or "")
    return _EXT_FMT.get(ext.lower(), "unknown")


def _unavailable(path: str, fmt: str, note: str) -> ImportedPart:
    return ImportedPart(path=path, fmt=fmt, note=note)


def import_solid(path: str) -> ImportedPart:
    """Load ``path`` as a measurable reference solid, degrading gracefully.

    Returns an :class:`ImportedPart`. When the file is missing, the format is
    unsupported, OCCT is not installed, or the kernel fails to read the file, the
    result has ``available=False`` and a descriptive ``note`` — never an
    exception.
    """
    fmt = detect_format(path)
    if not path or not os.path.exists(path):
        return _unavailable(path, fmt, f"file not found: {path!r}")
    if not os.path.isfile(path):
        return _unavailable(path, fmt, f"not a file: {path!r}")
    if fmt == "unknown":
        return _unavailable(
            path, fmt,
            "unsupported format (expected .step/.stp/.iges/.igs/.stl)")

    try:
        cq = _cq()
    except Exception as exc:  # noqa: BLE001 - no kernel -> clean unavailable
        return _unavailable(
            path, fmt,
            f"cadquery/OCCT unavailable ({type(exc).__name__}); "
            "install the 'cadquery' extra to import real solids")

    try:
        shape = _load_shape(cq, path, fmt)
    except Exception as exc:  # noqa: BLE001 - kernel read failure -> unavailable
        return _unavailable(
            path, fmt, f"failed to read {fmt} file: {type(exc).__name__}: {exc}")
    if shape is None:
        return _unavailable(path, fmt, f"{fmt} file contained no readable solid")

    metrics = _measure_shape(shape)
    bbox = list(metrics.get("bbox", [0.0, 0.0, 0.0]))
    # Latched BEFORE the units annotation below: `units` is metadata, not a
    # measurement, and must never make an unmeasurable part look available.
    measured = bool(metrics)
    note = "" if measured else "loaded but unmeasurable"
    if fmt == "step":
        units_info, units_note = _step_units(path)
        if units_info:
            metrics["units"] = units_info
        if units_note:
            note = f"{note}; {units_note}" if note else units_note
    return ImportedPart(
        path=path, fmt=fmt, shape=shape, metrics=metrics, bbox=bbox,
        available=measured, note=note)


# --------------------------------------------------------------------------- #
# Kernel-side loaders (only reached when cadquery/OCCT is importable)
# --------------------------------------------------------------------------- #
def _load_shape(cq, path: str, fmt: str):
    """Return a cq ``Shape`` for the file, or ``None``. May raise (guarded)."""
    if fmt == "step":
        wp = cq.importers.importStep(path)
        return _single_shape(cq, wp)
    if fmt == "stl":
        return _import_stl(cq, path)
    if fmt == "iges":
        return _import_iges(cq, path)
    return None


def _single_shape(cq, wp):
    """Collapse a cq Workplane's solids into a single Shape/Compound."""
    try:
        solids = wp.solids().vals()
    except Exception:  # noqa: BLE001
        solids = wp.vals() if hasattr(wp, "vals") else []
    if not solids:
        vals = wp.vals() if hasattr(wp, "vals") else []
        solids = [v for v in vals if v is not None]
    if not solids:
        return None
    if len(solids) == 1:
        return solids[0]
    return cq.Compound.makeCompound(solids)


def _import_iges(cq, path: str):
    """IGES via the OCCT reader (cq.importers has no IGES entry point)."""
    from OCP.IGESControl import IGESControl_Reader
    from OCP.IFSelect import IFSelect_ReturnStatus
    reader = IGESControl_Reader()
    status = reader.ReadFile(path)
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise ValueError("IGES reader reported failure")
    reader.TransferRoots()
    wrapped = reader.OneShape()
    return cq.Shape(wrapped)


def _import_stl(cq, path: str):
    """STL (a mesh) via the OCCT RWStl reader -> a shell Shape."""
    from OCP.RWStl import RWStl
    from OCP.TopoDS import TopoDS_Face
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    poly = RWStl.ReadFile_s(path)
    if poly is None:
        return None
    builder = BRep_Builder()
    face = TopoDS_Face()
    builder.MakeFace(face, poly)
    return cq.Shape(face)


def _measure_shape(shape) -> dict:
    """Mass properties + bbox + topology counts for a cq/OCP shape.

    Mirrors ``CadQueryBackend._metrics``. Returns ``{}`` on any failure so the
    caller records the part as unavailable rather than crashing.
    """
    wrapped = getattr(shape, "wrapped", shape)
    out: dict = {}
    try:
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        vprops = GProp_GProps()
        BRepGProp.VolumeProperties_s(wrapped, vprops)
        volume = float(vprops.Mass())
        com = vprops.CentreOfMass()
        sprops = GProp_GProps()
        BRepGProp.SurfaceProperties_s(wrapped, sprops)
        surface_area = float(sprops.Mass())
        out.update({
            "volume": volume,
            "surface_area": surface_area,
            "center_of_mass": [float(com.X()), float(com.Y()), float(com.Z())],
        })
    except Exception:  # noqa: BLE001
        pass
    try:
        bb = shape.BoundingBox()
        out["bbox"] = [float(bb.xlen), float(bb.ylen), float(bb.zlen)]
    except Exception:  # noqa: BLE001
        try:
            from OCP.Bnd import Bnd_Box
            from OCP.BRepBndLib import BRepBndLib
            box = Bnd_Box()
            BRepBndLib.Add_s(wrapped, box)
            xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
            out["bbox"] = [float(xmax - xmin), float(ymax - ymin),
                           float(zmax - zmin)]
        except Exception:  # noqa: BLE001
            pass
    try:
        out["faces"] = len(shape.Faces())
        out["edges"] = len(shape.Edges())
        out["solids"] = len(shape.Solids())
    except Exception:  # noqa: BLE001
        pass
    return out


# --------------------------------------------------------------------------- #
# RAG precedent ingestion
# --------------------------------------------------------------------------- #
def precedent_text(part: ImportedPart, metadata=None) -> str:
    """A retrievable text description of an imported part for the RAG store.

    Works with or without a real kernel: it summarises whatever metrics/metadata
    are available so an imported STEP becomes a searchable precedent even when
    only its name/format is known.
    """
    lines = [f"# Imported part: {os.path.basename(part.path) or part.path}",
             f"format: {part.fmt}",
             f"available: {part.available}"]
    m = part.metrics or {}
    if m.get("volume") is not None:
        lines.append(f"volume: {float(m['volume']):.6g}")
    if m.get("surface_area") is not None:
        lines.append(f"surface_area: {float(m['surface_area']):.6g}")
    if part.bbox and any(part.bbox):
        lines.append("bbox: " + " x ".join(f"{float(v):.6g}" for v in part.bbox))
    for key in ("faces", "edges", "solids"):
        if m.get(key) is not None:
            lines.append(f"{key}: {m[key]}")
    if metadata is not None:
        name = getattr(metadata, "name", "") or ""
        if name:
            lines.append(f"name: {name}")
        material = getattr(metadata, "material", None)
        if material:
            lines.append(f"material: {material}")
        bom = getattr(metadata, "bom_lines", None) or []
        for bl in bom:
            lines.append("bom: " + ", ".join(
                f"{k}={v}" for k, v in bl.items()))
    if part.note:
        lines.append(f"note: {part.note}")
    return "\n".join(lines)


def index_precedent(retriever, path: str, metadata=None) -> ImportedPart:
    """Import ``path`` and add its precedent text to a RAG ``retriever``.

    ``retriever`` is any object exposing ``add_document(text, source)`` (e.g.
    :class:`rag.HybridRetriever`). Returns the :class:`ImportedPart` (imported
    even when the kernel is absent, so a precedent is always indexable). Never
    raises for a missing kernel; propagation is left to the retriever only.
    """
    part = import_solid(path)
    text = precedent_text(part, metadata)
    retriever.add_document(text, source=os.path.basename(path) or path)
    return part
