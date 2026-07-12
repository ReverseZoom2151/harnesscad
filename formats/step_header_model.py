"""Structured accessors for the ISO 10303-21 HEADER section.

A part-21 file's ``HEADER`` section is not free-form: ISO 10303-21 clause 8.2
mandates exactly three records, in order -- ``FILE_DESCRIPTION``, ``FILE_NAME``
and ``FILE_SCHEMA`` -- each with a fixed attribute layout (see ruststep's
``header.rs``, which hard-codes these three entities because "there is a schema
for the HEADER section, but we do not generate this structure from it").

:mod:`formats.stepllm_parser` parses the header into an *unstructured* list of
:class:`~formats.stepllm_parser.Typed` records; it never exposes the schema
identifier, author, or timestamp by name.  This module lifts that raw list into
typed dataclasses with named fields, so callers can ask "which STEP schema is
this file in?" (the crucial ``FILE_SCHEMA`` identifier used to pick the right
EXPRESS schema for validation) without re-walking the record tuples.

Pure/deterministic; depends only on :mod:`formats.stepllm_parser`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from formats.stepllm_parser import StepFile, Typed


class HeaderError(ValueError):
    """Raised when the HEADER section does not match the part-21 layout."""


@dataclass(frozen=True)
class FileDescription:
    description: tuple = ()          # LIST OF STRING
    implementation_level: str = ""


@dataclass(frozen=True)
class FileName:
    name: str = ""
    time_stamp: str = ""
    author: tuple = ()
    organization: tuple = ()
    preprocessor_version: str = ""
    originating_system: str = ""
    authorization: str = ""


@dataclass(frozen=True)
class FileSchema:
    schema_identifiers: tuple = ()  # LIST OF schema_name


@dataclass
class StepHeader:
    file_description: FileDescription
    file_name: FileName
    file_schema: FileSchema
    extra: list = field(default_factory=list)   # records beyond the mandatory 3

    @property
    def schema_name(self) -> str:
        """The primary schema identifier (e.g. ``AUTOMOTIVE_DESIGN {...}``)."""

        ids = self.file_schema.schema_identifiers
        return ids[0] if ids else ""


def _as_str(value) -> str:
    if isinstance(value, str):
        return value
    raise HeaderError(f"expected a string, got {value!r}")


def _as_str_list(value) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise HeaderError(f"expected a list of strings, got {value!r}")
    return tuple(_as_str(v) for v in value)


def _find(records, keyword: str) -> Typed | None:
    for rec in records:
        if isinstance(rec, Typed) and rec.keyword.upper() == keyword:
            return rec
    return None


def _require_arity(rec: Typed, n: int) -> None:
    if len(rec.params) != n:
        raise HeaderError(
            f"{rec.keyword} expects {n} attributes, got {len(rec.params)}")


def parse_header(records) -> StepHeader:
    """Build a :class:`StepHeader` from a list of header :class:`Typed` records.

    Accepts ``StepFile.header`` directly, or a :class:`StepFile` (its header is
    used).  Records are matched by keyword so trailing optional header entities
    (``FILE_POPULATION`` etc.) are tolerated and returned in ``extra``.
    """

    if isinstance(records, StepFile):
        records = records.header

    fd = _find(records, "FILE_DESCRIPTION")
    fn = _find(records, "FILE_NAME")
    fs = _find(records, "FILE_SCHEMA")
    if fd is None or fn is None or fs is None:
        missing = [k for k, v in (
            ("FILE_DESCRIPTION", fd), ("FILE_NAME", fn), ("FILE_SCHEMA", fs))
            if v is None]
        raise HeaderError(f"missing mandatory header records: {missing}")

    _require_arity(fd, 2)
    file_description = FileDescription(
        description=_as_str_list(fd.params[0]),
        implementation_level=_as_str(fd.params[1]),
    )

    _require_arity(fn, 7)
    file_name = FileName(
        name=_as_str(fn.params[0]),
        time_stamp=_as_str(fn.params[1]),
        author=_as_str_list(fn.params[2]),
        organization=_as_str_list(fn.params[3]),
        preprocessor_version=_as_str(fn.params[4]),
        originating_system=_as_str(fn.params[5]),
        authorization=_as_str(fn.params[6]),
    )

    _require_arity(fs, 1)
    file_schema = FileSchema(
        schema_identifiers=_as_str_list(fs.params[0]),
    )

    mandatory = {"FILE_DESCRIPTION", "FILE_NAME", "FILE_SCHEMA"}
    extra = [r for r in records
             if isinstance(r, Typed) and r.keyword.upper() not in mandatory]

    return StepHeader(
        file_description=file_description,
        file_name=file_name,
        file_schema=file_schema,
        extra=extra,
    )
