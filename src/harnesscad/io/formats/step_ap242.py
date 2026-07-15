"""STEP AP242 header/schema writer over the existing part-21 substrate.

The harness's :mod:`harnesscad.io.formats.step` codec reads and writes ISO
10303-21 part-21 text, and its geometry corpus is AP203 / AP214 (the config- and
automotive-design schemas). **AP242** -- ``Managed Model Based 3D Engineering`` --
is the modern superset that unifies AP203 and AP214 and, crucially, carries
**PMI** (Product and Manufacturing Information: semantic GD&T, datums, dimensional
and geometric tolerances, annotations, saved views). It is the schema CAD vendors
now default to for a model-based-definition hand-off.

A full AP242 MIM writer is very large (thousands of entity types). This module
does the tractable, honest part: it emits a **correct AP242 HEADER** -- in
particular the mandatory ``FILE_SCHEMA`` identifier with the AP242 MIM long-form
name and its object identifier -- and **reuses the existing part-21 geometry
serialisation** from :mod:`harnesscad.io.formats.step` for the DATA section. The
result is a part-21 file that declares itself AP242 and whose geometry is the same
B-rep the STEP codec already round-trips.

WHAT IS NOT EMITTED (declared, not hidden): this writer carries geometry and the
AP242 *header*, not PMI. :func:`pmi_gaps` enumerates the semantic parts a full
AP242 MBD exporter would add and this one does not. See it before relying on an
AP242 file from here for downstream tolerance analysis.

AXIS + UNITS. STEP is a **right-handed, +Z-up** modelling space, and units are
declared IN THE DATA SECTION via ``*_UNIT`` / ``GEOMETRIC_REPRESENTATION_CONTEXT``
entities -- not in the header. This module does not invent or rewrite those
entities: it passes the source model's DATA through unchanged, so whatever unit
context the geometry carries is preserved exactly. The header's schema identifier
is asserted to be the AP242 name on write.

Depends only on :mod:`harnesscad.io.formats.step` (pure, deterministic).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from harnesscad.io.formats import step as step_codec
from harnesscad.io.formats.step import StepFile, Typed

__all__ = [
    "Ap242Error",
    "AP242_SCHEMA",
    "AXIS",
    "ap242_header",
    "to_ap242",
    "serialize_ap242",
    "write_ap242",
    "parse_ap242",
    "is_ap242",
    "pmi_gaps",
]

#: The AP242 MIM long-form schema identifier (edition 2 / DIS), with its object
#: identifier, exactly as it must appear in FILE_SCHEMA.
AP242_SCHEMA = "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }"

#: STEP's modelling space. Recorded for symmetry with the mesh codecs; STEP itself
#: fixes the handedness and declares units in the DATA section (see module docs).
AXIS = "right-handed,+Z-up"

_IMPL_LEVEL = "2;1"


class Ap242Error(ValueError):
    """Raised when an AP242 file/model cannot be produced or recognised."""


def ap242_header(
    name: str = "",
    author: str = "",
    organization: str = "",
    preprocessor: str = "harnesscad.io.formats.step_ap242",
    originating_system: str = "harnesscad",
    time_stamp: str = "",
    description: str = "AP242 managed model based 3D engineering",
) -> List[Typed]:
    """Build the three mandatory AP242 HEADER records as part-21 ``Typed`` values.

    The timestamp defaults to empty so output is deterministic (no wall clock).
    The FILE_SCHEMA identifier is the AP242 MIM long-form name.
    """
    file_description = Typed("FILE_DESCRIPTION", (
        [description],
        _IMPL_LEVEL,
    ))
    file_name = Typed("FILE_NAME", (
        name,
        time_stamp,
        [author] if author else [""],
        [organization] if organization else [""],
        preprocessor,
        originating_system,
        "",
    ))
    file_schema = Typed("FILE_SCHEMA", (
        [AP242_SCHEMA],
    ))
    return [file_description, file_name, file_schema]


def to_ap242(source: StepFile, **header_kwargs) -> StepFile:
    """Return a new :class:`StepFile` with the AP242 header and ``source``'s DATA.

    The entity graph (the DATA section) is carried over unchanged -- including its
    unit context -- and only the HEADER's schema declaration is (re)written to
    AP242. ``header_kwargs`` are forwarded to :func:`ap242_header`.
    """
    if not isinstance(source, StepFile):
        raise Ap242Error(
            "to_ap242 needs a StepFile, got %r" % type(source).__name__)
    out = StepFile()
    out.header = ap242_header(**header_kwargs)
    # Preserve the DATA section exactly (order + entities).
    for ent_id in source.order:
        out.add(source.entities[ent_id])
    return out


def serialize_ap242(source: StepFile, **header_kwargs) -> str:
    """Serialise ``source`` as AP242 part-21 text (header retagged, geometry reused)."""
    step = to_ap242(source, **header_kwargs)
    # Assert the schema really is AP242 before it leaves.
    schema_ids = _schema_identifiers(step)
    assert any(sid.startswith("AP242_MANAGED_MODEL_BASED_3D_ENGINEERING")
               for sid in schema_ids), \
        "AP242 header must declare the AP242 MIM schema, got %r" % (schema_ids,)
    return step_codec.serialize(step)


def write_ap242(path: str, source: StepFile, **header_kwargs) -> str:
    """Write an AP242 part-21 file. Returns the text written."""
    text = serialize_ap242(source, **header_kwargs)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def parse_ap242(text: str) -> StepFile:
    """Parse AP242 part-21 text into a :class:`StepFile` (via the STEP parser).

    AP242 files are ordinary part-21, so the shared parser reads them; this raises
    :class:`Ap242Error` if the file does not declare the AP242 schema, so a caller
    asking specifically for AP242 is not silently handed an AP203/214 file.
    """
    step = step_codec.parse(text)
    if not is_ap242(step):
        raise Ap242Error(
            "file does not declare the AP242 schema (FILE_SCHEMA = %r)"
            % (_schema_identifiers(step),))
    return step


def _schema_identifiers(step: StepFile) -> Tuple[str, ...]:
    for rec in step.header:
        if isinstance(rec, Typed) and rec.keyword.upper() == "FILE_SCHEMA":
            if rec.params and isinstance(rec.params[0], (list, tuple)):
                return tuple(s for s in rec.params[0] if isinstance(s, str))
    return ()


def is_ap242(step: StepFile) -> bool:
    """True if the file's FILE_SCHEMA names the AP242 MIM schema."""
    return any(sid.startswith("AP242_MANAGED_MODEL_BASED_3D_ENGINEERING")
               for sid in _schema_identifiers(step))


def pmi_gaps() -> Tuple[str, ...]:
    """The AP242 PMI/MBD parts this writer does NOT yet emit.

    Naming the gap is the point: an AP242 file from here is geometry + a correct
    schema declaration, not a semantic model-based-definition. A full exporter
    would additionally emit each of these.
    """
    return (
        "semantic GD&T: geometric_tolerance and dimensional_location entities",
        "datum systems: datum, datum_feature, datum_reference_compartment",
        "annotation planes and saved views (draughting_model / camera_model)",
        "presentation: annotation_occurrence, tessellated/polyline annotation",
        "product structure: product_definition_formation and MBD associativity",
        "surface finish, material and property PMI attached to faces",
    )
