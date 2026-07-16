"""DXF $INSUNITS resolution: the 2D half of the silent-scale trap.

``io/formats/step_units.py`` closed the STEP unit gap (a metre file read as
millimetres is silently 1000x wrong, and every downstream check verifies the
wrong number happily). DXF has the identical trap through a different door: the
drawing's length unit lives in ONE header variable, ``$INSUNITS``, as a bare
integer code. Read the drawing without resolving it and a 10-inch bracket
becomes a 10-millimetre one -- same failure, worse odds, because DXF is the
format hand-drawn 2D profiles actually arrive in.

WHY A DATA MODULE AND NOT AN IMPORTER HOOK
The harness has no DXF importer. ``io/formats/dxf.py`` defines a neutral
``DxfDocument`` contract whose ``units`` field is a free-text string
(``"mm"``/``"cm"``/``"m"``/``"in"``/``"ft"``), and ``DxfParser`` is a Protocol
with no implementation; nothing in the tree parses DXF bytes (no ezdxf, no
reader). So there is no existing reader whose behaviour could be doubled up on.

That distinction is load-bearing, and the STEP analogue is why. When
``step_units`` was wired into the importer, OCCT turned out to ALREADY resolve
the declared unit and hand back millimetres -- applying ``scale_to_mm`` on top
would have CAUSED the very 1000x bug it was meant to prevent (see
``io/ingest/import_brep.py::_step_units``). The rule that came out of that:
never apply a scale until you have verified what the reader in front of you
already did. Here the verification is trivial and was performed -- there is no
reader -- so :func:`resolve_insunits` returns a factor and applies nothing. A
future DXF reader must re-verify against ITS OWN behaviour before multiplying.

WHAT THE CODE TABLE IS FOR
``$INSUNITS = 0`` means UNITLESS: the drawing declares no unit at all. It is not
millimetres, and quietly treating it as millimetres is the same silence-is-
success bug in miniature. :class:`DxfUnits` therefore reports ``scale_to_mm =
None`` for code 0 with an explicit note, leaving the caller to decide rather
than deciding wrong for it.

Attribution: kerf (kerf-main
``packages/kerf-imports/src/kerf_imports/dxf/reader.py``) ships the
``$INSUNITS`` code->unit table this is modelled on (its ``_INSUNITS_MAP``, read
from header group code 70, defaulting to mm). kerf is MIT-licensed (Copyright
(c) 2026 Imran Paruk). Nothing is copied: kerf's map covers 8 of the codes
(0,1,2,4,5,6,8,10) and carries no scale factors; the table below is the full
0..20 enumeration from the DXF header-variable specification with mm factors
computed here, and the unitless handling is the opposite of kerf's mm default.

Pure stdlib, deterministic, no kernel.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

#: ``$INSUNITS`` code -> (name, millimetres per unit). ``None`` means the code
#: declares no resolvable length. Codes 0..20 per the DXF header-variable spec.
#: Exact by definition where the definition is exact (inch = 25.4 mm exactly);
#: the astronomical tail uses IAU/SI defined values.
INSUNITS: Dict[int, Tuple[str, Optional[float]]] = {
    0:  ("unitless", None),
    1:  ("inches", 25.4),
    2:  ("feet", 304.8),
    3:  ("miles", 1609344.0),
    4:  ("millimeters", 1.0),
    5:  ("centimeters", 10.0),
    6:  ("meters", 1000.0),
    7:  ("kilometers", 1.0e6),
    8:  ("microinches", 2.54e-5),
    9:  ("mils", 0.0254),
    10: ("yards", 914.4),
    11: ("angstroms", 1.0e-7),
    12: ("nanometers", 1.0e-6),
    13: ("microns", 1.0e-3),
    14: ("decimeters", 100.0),
    15: ("decameters", 10000.0),
    16: ("hectometers", 100000.0),
    17: ("gigameters", 1.0e12),
    18: ("astronomical units", 1.495978707e14),
    19: ("light years", 9.4607304725808e18),
    20: ("parsecs", 3.0856775814913673e19),
}

#: ``$MEASUREMENT``: which linetype/hatch pattern file the drawing uses. It is
#: NOT the drawing's unit and must never be used as one -- a metric drawing can
#: still be authored in inches. Kept only so a reader can report the pair.
MEASUREMENT = {0: "english", 1: "metric"}

#: ``$INSUNITS`` name -> the vocabulary ``io/formats/dxf.py::DxfDocument.units``
#: accepts. Only the codes that contract can express appear here.
_DOCUMENT_UNITS = {
    "millimeters": "mm", "centimeters": "cm", "meters": "m",
    "inches": "in", "feet": "ft",
}

#: DXF header group code carrying $INSUNITS' value (a 16-bit int).
INSUNITS_GROUP_CODE = 70


@dataclass(frozen=True)
class DxfUnits:
    """A resolved ``$INSUNITS`` declaration.

    ``scale_to_mm`` multiplies a raw DXF coordinate into millimetres, or is
    ``None`` when the drawing declares no unit (code 0, or absent). ``None`` is
    a real answer -- "unknown" -- and is why this is not a bare float.
    """

    code: Optional[int]
    name: str
    scale_to_mm: Optional[float] = None
    declared: bool = True
    notes: tuple = ()

    @property
    def resolved(self) -> bool:
        """True when the drawing declares a usable length unit."""
        return self.scale_to_mm is not None

    @property
    def document_units(self) -> Optional[str]:
        """The ``DxfDocument.units`` string, or ``None`` if not expressible."""
        return _DOCUMENT_UNITS.get(self.name)

    def to_dict(self) -> dict:
        return {
            "code": self.code, "name": self.name,
            "scale_to_mm": self.scale_to_mm, "declared": self.declared,
            "resolved": self.resolved, "document_units": self.document_units,
            "notes": list(self.notes),
        }


def resolve_insunits(code: Optional[int]) -> DxfUnits:
    """Resolve an ``$INSUNITS`` code to a scale-to-millimetres factor.

    Never raises and never guesses: an absent declaration, code 0, and an
    unknown code are three different unresolved answers, each with its own note.
    """
    if code is None:
        return DxfUnits(
            code=None, name="unitless", scale_to_mm=None, declared=False,
            notes=("no $INSUNITS header variable; drawing declares no unit -- "
                   "do NOT assume millimetres, ask for the intended unit",))
    try:
        code = int(code)
    except (TypeError, ValueError):
        return DxfUnits(
            code=None, name="unknown", scale_to_mm=None, declared=False,
            notes=(f"$INSUNITS value {code!r} is not an integer",))
    entry = INSUNITS.get(code)
    if entry is None:
        return DxfUnits(
            code=code, name="unknown", scale_to_mm=None,
            notes=(f"$INSUNITS code {code} is outside the known 0..20 table; "
                   "unit unresolved",))
    name, scale = entry
    if scale is None:
        return DxfUnits(
            code=code, name=name, scale_to_mm=None,
            notes=("$INSUNITS=0 (unitless): the drawing explicitly declares no "
                   "unit -- treating it as millimetres is a silent-scale bug",))
    return DxfUnits(code=code, name=name, scale_to_mm=scale)


def parse_insunits(text: str) -> Optional[int]:
    """The ``$INSUNITS`` code from ASCII DXF text, or ``None`` if not declared.

    A deliberately narrow scanner: DXF is a flat stream of (group code, value)
    line pairs, so finding one header variable needs no full parser. It walks
    pairs rather than regex-matching, because the literal ``$INSUNITS`` can
    legitimately appear as an entity's text content -- only an occurrence at
    group code 9 (the header-variable-name code) is the declaration.
    """
    lines = text.splitlines()
    i = 0
    in_header = False
    while i + 1 < len(lines):
        code_raw, value = lines[i].strip(), lines[i + 1].strip()
        i += 2
        if code_raw == "0" and value == "SECTION":
            continue
        if code_raw == "2" and value == "HEADER":
            in_header = True
            continue
        if code_raw == "0" and value == "ENDSEC":
            if in_header:
                return None
            continue
        if not in_header or code_raw != "9" or value != "$INSUNITS":
            continue
        # The variable's value follows as the next pair, at group code 70.
        if i + 1 < len(lines):
            vcode, vval = lines[i].strip(), lines[i + 1].strip()
            if vcode == str(INSUNITS_GROUP_CODE):
                try:
                    return int(vval)
                except ValueError:
                    return None
        return None
    return None


def units_from_dxf_text(text: str) -> DxfUnits:
    """Resolve the drawing's unit straight from ASCII DXF text."""
    return resolve_insunits(parse_insunits(text))


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def _dxf(insunits: Optional[int], extra: str = "") -> str:
    body = "  0\nSECTION\n  2\nHEADER\n"
    body += "  9\n$ACADVER\n  1\nAC1027\n"
    if insunits is not None:
        body += f"  9\n$INSUNITS\n {INSUNITS_GROUP_CODE}\n{insunits:6d}\n"
    body += "  9\n$MEASUREMENT\n 70\n     1\n"
    body += "  0\nENDSEC\n"
    body += extra
    body += "  0\nEOF\n"
    return body


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="DXF $INSUNITS code table -> scale-to-mm (the 2D analogue "
                    "of step_units; table modelled on kerf's dxf reader).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="assert the code table's factors, the "
                             "unitless/absent/unknown split, and header "
                             "scanning.")
    parser.add_argument("--table", action="store_true",
                        help="print the full $INSUNITS code table")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.table:
        for code in sorted(INSUNITS):
            name, scale = INSUNITS[code]
            factor = "unresolved" if scale is None else f"{scale:g} mm"
            print(f"{code:3d}  {name:<20s} {factor}")
        return 0
    if not args.selfcheck:
        parser.print_help()
        return 0

    # 1. The whole table resolves, exactly once, with sane factors.
    assert sorted(INSUNITS) == list(range(21)), "table is not 0..20"
    for code, (name, scale) in INSUNITS.items():
        u = resolve_insunits(code)
        assert u.code == code and u.name == name
        assert u.scale_to_mm == scale
        assert u.resolved is (scale is not None)
        assert scale is None or scale > 0.0
    print(f"[selfcheck] $INSUNITS table: {len(INSUNITS)} codes (0..20) resolve")

    # 2. The factors that matter -- exact by definition.
    assert resolve_insunits(1).scale_to_mm == 25.4          # inch, exact
    assert resolve_insunits(2).scale_to_mm == 304.8         # foot = 12 in
    assert resolve_insunits(4).scale_to_mm == 1.0           # mm
    assert resolve_insunits(6).scale_to_mm == 1000.0        # the 1000x case
    assert resolve_insunits(10).scale_to_mm == 914.4        # yard = 36 in
    assert abs(resolve_insunits(9).scale_to_mm - 25.4 / 1000.0) < 1e-15  # mil
    assert abs(resolve_insunits(8).scale_to_mm - 25.4e-6) < 1e-18  # microinch
    # Internal consistency: the imperial chain is one definition, not five.
    inch = resolve_insunits(1).scale_to_mm
    assert abs(resolve_insunits(2).scale_to_mm - 12 * inch) < 1e-12
    assert abs(resolve_insunits(10).scale_to_mm - 36 * inch) < 1e-12
    assert abs(resolve_insunits(3).scale_to_mm - 63360 * inch) < 1e-6
    # ... and the metric chain is decimal.
    assert abs(resolve_insunits(5).scale_to_mm - 10.0) < 1e-12
    assert abs(resolve_insunits(14).scale_to_mm - 100.0) < 1e-12
    assert abs(resolve_insunits(13).scale_to_mm - 1e-3) < 1e-15
    print("[selfcheck] inch/foot/yard/mile chain consistent (x25.4 exact); "
          "metre = 1000 mm (the silent-1000x code)")

    # 3. Unitless is NOT millimetres -- the trap this table exists to refuse.
    u = resolve_insunits(0)
    assert u.name == "unitless" and u.scale_to_mm is None and not u.resolved
    assert u.declared is True  # it IS declared -- declared as nothing
    assert any("silent-scale" in n for n in u.notes)
    # Absent is a DIFFERENT unresolved: never declared at all.
    absent = resolve_insunits(None)
    assert absent.scale_to_mm is None and absent.declared is False
    assert absent.code is None
    assert absent.declared != u.declared, "absent and unitless collapsed"
    # Unknown code is a third.
    unknown = resolve_insunits(99)
    assert unknown.scale_to_mm is None and unknown.code == 99
    assert unknown.name == "unknown"
    print("[selfcheck] unitless(0) vs absent vs unknown(99): three distinct "
          "unresolved answers, none of them 'mm'")

    # 4. Header scanning off real DXF text.
    assert parse_insunits(_dxf(1)) == 1
    assert units_from_dxf_text(_dxf(1)).scale_to_mm == 25.4
    assert units_from_dxf_text(_dxf(6)).scale_to_mm == 1000.0
    assert parse_insunits(_dxf(None)) is None
    assert units_from_dxf_text(_dxf(None)).declared is False
    print("[selfcheck] $INSUNITS scanned from DXF header (inch, metre, absent)")

    # 5. A $INSUNITS-looking string in an ENTITY must not be mistaken for the
    #    header declaration -- group code 9 is the only place it counts.
    decoy = _dxf(None, extra=(
        "  0\nSECTION\n  2\nENTITIES\n"
        "  0\nTEXT\n  1\n$INSUNITS\n 70\n     1\n"
        "  0\nENDSEC\n"))
    assert parse_insunits(decoy) is None, "entity text mistaken for header var"
    print("[selfcheck] entity text '$INSUNITS' ignored (only group code 9 "
          "declares)")

    # 6. Bridge to the DxfDocument contract, and its honest limits.
    from harnesscad.io.formats.dxf import DxfDocument
    assert resolve_insunits(4).document_units == "mm"
    assert resolve_insunits(1).document_units == "in"
    assert resolve_insunits(2).document_units == "ft"
    # A microinch drawing is real but the contract cannot say so: None, not a lie.
    assert resolve_insunits(8).document_units is None
    assert resolve_insunits(0).document_units is None
    doc = DxfDocument(units=resolve_insunits(1).document_units,
                      layers=(), entities={})
    assert doc.units == "in"
    print("[selfcheck] resolves into DxfDocument.units; inexpressible units "
          "return None rather than a wrong string")

    # 7. Nothing is applied. This module reports a factor; it never scales, and
    #    no reader exists whose conversion it could double.
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
