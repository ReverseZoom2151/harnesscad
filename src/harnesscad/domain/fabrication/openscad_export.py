"""Deterministic OpenSCAD CLI export planning.

A typical app materialises geometry by shelling out to the
OpenSCAD binary::

    openscad -o <out> --export-format <fmt> <tmp>.scad

and it commonly does so with three bugs worth fixing rather than copying:

1.  the temp file name is a ``uuid4()`` -- so an identical prompt re-renders and
    re-writes a new artefact every time, and nothing can be cached;
2.  the requested ``--export-format`` is never checked against the output file's
    extension, and OpenSCAD happily writes an ``.stl`` that is really SVG;
3.  ``subprocess.run(check=True)`` only catches a non-zero exit -- but the single
    most common OpenSCAD failure is exit code **0** with the message
    ``Current top level object is empty``, i.e. a syntactically valid script that
    produced no geometry.  The app then converts an empty STL and shows a blank
    viewer.

This module is the deterministic half of that hop: it *plans and validates* the
invocation and *classifies* the result.  It never spawns a process (the harness
stays stdlib-only and testable without OpenSCAD installed) -- ``plan.argv`` is
handed to whatever executor the caller owns.

The content-addressed name is the useful trick: ``uuid5`` over a SHA-256 of the
source makes the artefact path a pure function of the script, so identical
scripts map to identical files and a cache hit is a file-exists check.
"""

from __future__ import annotations

import hashlib
import posixpath
import uuid
from typing import Any, Dict, List, Mapping, Sequence, Tuple

# OpenSCAD's --export-format values, mapped to the canonical file extension and
# the dimensionality of the geometry each one can carry.
FORMAT_EXTENSIONS: Dict[str, str] = {
    "stl": ".stl",
    "binstl": ".stl",
    "asciistl": ".stl",
    "off": ".off",
    "amf": ".amf",
    "3mf": ".3mf",
    "obj": ".obj",
    "wrl": ".wrl",
    "dxf": ".dxf",
    "svg": ".svg",
    "pdf": ".pdf",
    "csg": ".csg",
    "png": ".png",
}

# 3D formats need a 3D top-level object; 2D formats need a 2D one. Mixing them
# is the "Current top level object is not a 2D object" failure.
FORMATS_3D = frozenset({"stl", "binstl", "asciistl", "off", "amf", "3mf", "obj", "wrl"})
FORMATS_2D = frozenset({"dxf", "svg", "pdf"})
FORMATS_OTHER = frozenset({"csg", "png"})

# Namespace for content-addressed artefact names (stable across runs/machines).
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_DNS


class OpenScadExportError(ValueError):
    """Raised when an export request is inconsistent or unsupported."""


class ExportPlan:
    """A validated, fully-determined OpenSCAD invocation."""

    __slots__ = ("source", "export_format", "scad_path", "output_path", "argv", "digest")

    def __init__(
        self,
        source: str,
        export_format: str,
        scad_path: str,
        output_path: str,
        argv: List[str],
        digest: str,
    ) -> None:
        self.source = source
        self.export_format = export_format
        self.scad_path = scad_path
        self.output_path = output_path
        self.argv = argv
        self.digest = digest

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExportPlan):
            return NotImplemented
        return self.argv == other.argv and self.source == other.source

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "ExportPlan(%s -> %s)" % (self.export_format, self.output_path)


def source_digest(source: str) -> str:
    """SHA-256 of the OpenSCAD source, normalised for line endings."""
    normalised = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def artifact_name(source: str) -> str:
    """Content-addressed stem: identical source always yields the same name.

    Replaces a ``uuid.uuid4()`` stem, which made every render a cache miss.
    """
    return str(uuid.uuid5(_NAMESPACE, source_digest(source)))


def extension_for(export_format: str) -> str:
    fmt = export_format.lower()
    try:
        return FORMAT_EXTENSIONS[fmt]
    except KeyError:
        raise OpenScadExportError(
            "unknown OpenSCAD export format %r (known: %s)"
            % (export_format, ", ".join(sorted(FORMAT_EXTENSIONS)))
        ) from None


def format_dimension(export_format: str) -> str:
    """``'3d'``, ``'2d'`` or ``'other'`` for a known format."""
    fmt = export_format.lower()
    extension_for(fmt)  # validates
    if fmt in FORMATS_3D:
        return "3d"
    if fmt in FORMATS_2D:
        return "2d"
    return "other"


def scad_literal(value: Any) -> str:
    """Render a Python value as an OpenSCAD literal for a ``-D`` override.

    OpenSCAD booleans are lowercase, strings are double-quoted with escapes, and
    sequences become vectors -- passing Python's ``repr`` (``True``, ``'x'``)
    silently produces an undefined variable instead of an error.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return '"%s"' % escaped
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(scad_literal(v) for v in value) + "]"
    raise OpenScadExportError("cannot express %r as an OpenSCAD literal" % (value,))


def define_args(defines: Mapping[str, Any]) -> List[str]:
    """``-D name=literal`` pairs, emitted in sorted order for determinism."""
    args: List[str] = []
    for key in sorted(defines):
        if not key.isidentifier():
            raise OpenScadExportError("invalid OpenSCAD variable name %r" % key)
        args.extend(["-D", "%s=%s" % (key, scad_literal(defines[key]))])
    return args


def plan_export(
    source: str,
    export_format: str = "stl",
    out_dir: str = ".",
    defines: Mapping[str, Any] | None = None,
    executable: str = "openscad",
) -> ExportPlan:
    """Build the validated argv for rendering ``source`` to ``export_format``.

    The output and temp-scad paths are content-addressed, so re-planning the same
    source is idempotent and the artefact can be cached by existence.
    """
    if not source.strip():
        raise OpenScadExportError("empty OpenSCAD source")
    fmt = export_format.lower()
    ext = extension_for(fmt)
    stem = artifact_name(source)
    scad_path = posixpath.join(out_dir, stem + ".scad")
    output_path = posixpath.join(out_dir, stem + ext)

    argv = [executable, "-o", output_path, "--export-format", fmt]
    argv.extend(define_args(defines or {}))
    argv.append(scad_path)
    return ExportPlan(
        source=source,
        export_format=fmt,
        scad_path=scad_path,
        output_path=output_path,
        argv=argv,
        digest=source_digest(source),
    )


def check_output_extension(output_path: str, export_format: str) -> None:
    """Raise unless ``output_path``'s extension matches ``export_format``.

    OpenSCAD does not complain when told to write SVG bytes into a ``.stl``; the
    downstream mesh loader is what explodes, far from the cause.
    """
    ext = extension_for(export_format)
    lowered = output_path.lower()
    if not lowered.endswith(ext):
        raise OpenScadExportError(
            "output %r does not match export format %r (expected %s)"
            % (output_path, export_format, ext)
        )


# --- result classification -------------------------------------------------

# OpenSCAD reports "no geometry" on stdout/stderr while still exiting 0.
_EMPTY_MARKERS = (
    "current top level object is empty",
    "no top level geometry to render",
)
_DIMENSION_MARKERS = (
    "current top level object is not a 2d object",
    "current top level object is not a 3d object",
)

STATUS_OK = "ok"
STATUS_EMPTY = "empty_geometry"
STATUS_WRONG_DIMENSION = "wrong_dimension"
STATUS_ERROR = "error"


def classify_result(returncode: int, stderr: str) -> Tuple[str, List[str]]:
    """Classify an OpenSCAD run into ``(status, messages)``.

    The zero-exit-but-empty case is promoted to a failure, because an empty STL
    is what reaches the user as a blank 3D viewer.
    """
    text = stderr or ""
    lowered = text.lower()
    messages = [
        line.strip()
        for line in text.splitlines()
        if line.strip().upper().startswith(("ERROR", "WARNING"))
    ]
    if any(marker in lowered for marker in _DIMENSION_MARKERS):
        return STATUS_WRONG_DIMENSION, messages
    if any(marker in lowered for marker in _EMPTY_MARKERS):
        return STATUS_EMPTY, messages
    if returncode != 0:
        return STATUS_ERROR, messages
    if any(m.upper().startswith("ERROR") for m in messages):
        return STATUS_ERROR, messages
    return STATUS_OK, messages


def is_success(returncode: int, stderr: str) -> bool:
    """True only when geometry was actually produced."""
    status, _ = classify_result(returncode, stderr)
    return status == STATUS_OK


def warnings_only(stderr: str) -> List[str]:
    """The WARNING lines, which are advisory (e.g. ``$fn`` too small)."""
    return [
        line.strip()
        for line in (stderr or "").splitlines()
        if line.strip().upper().startswith("WARNING")
    ]


def summarize(plan: ExportPlan, returncode: int, stderr: str) -> Dict[str, Any]:
    """A deterministic, JSON-serialisable record of one export attempt."""
    status, messages = classify_result(returncode, stderr)
    return {
        "digest": plan.digest,
        "format": plan.export_format,
        "output": plan.output_path,
        "status": status,
        "returncode": returncode,
        "messages": messages,
        "warnings": warnings_only(stderr),
    }


def plan_cache_key(source: str, export_format: str, defines: Mapping[str, Any] | None = None) -> str:
    """Stable cache key over (source, format, defines) -- the tuple that fully
    determines the artefact bytes."""
    parts: List[str] = [source_digest(source), export_format.lower()]
    for key in sorted(defines or {}):
        parts.append("%s=%s" % (key, scad_literal((defines or {})[key])))
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def sorted_formats(dimension: str) -> Sequence[str]:
    """Known formats for ``'3d'`` / ``'2d'`` / ``'other'``, sorted."""
    table = {"3d": FORMATS_3D, "2d": FORMATS_2D, "other": FORMATS_OTHER}
    if dimension not in table:
        raise OpenScadExportError("unknown dimension %r" % dimension)
    return sorted(table[dimension])
