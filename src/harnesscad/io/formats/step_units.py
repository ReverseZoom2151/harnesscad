"""STEP unit-scale + global-tolerance extraction: the silent-1000x guard.

The classic text-to-CAD ingest failure: a STEP file authored in METRES read as
if it were MILLIMETRES (or vice versa) produces geometry silently wrong by
1000x, and every downstream volume/bbox check happily verifies the wrong
number. The harness's STEP stack had ZERO ``SI_UNIT`` handling; this module
closes the gap by parsing the three unit-bearing constructs from a STEP
file's DATA section and returning ONE scale-to-millimetres factor plus the
file's declared global tolerance:

  * ``SI_UNIT(.MILLI., .METRE.)``            -- SI prefix + base unit;
  * ``CONVERSION_BASED_UNIT('INCH', #m)``    -- named unit defined as a
    ``LENGTH_MEASURE_WITH_UNIT`` multiple of another unit (chained, cycles
    guarded);
  * ``UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(t), #u, ...)`` -- the
    file's ``distance_accuracy_value`` global tolerance, converted to mm.

Policy sources:
  * cadquery importer (cadquery-master ``occ_impl/importers/__init__.py``):
    sets ``xstep.cascade.unit`` so OCCT rescales from the file's declared
    unit to a fixed target -- i.e. ALWAYS resolve the declared unit, never
    assume; this module is the kernel-free analogue producing the explicit
    factor.
  * kerf (kerf-main ``io/step_reader.py``): resolves
    ``UNCERTAINTY_MEASURE_WITH_UNIT`` from the DATA section (its reader
    treats it as a first-class entity rather than noise).

Reuses (imports, does not rewrite) the harness's part-21 parser
``harnesscad.io.formats.step``. Pure stdlib, deterministic; no kernel.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Union

from harnesscad.io.formats.step import (
    Entity,
    Enum,
    Real,
    Ref,
    StepFile,
    Typed,
    parse,
)

#: SI prefix -> multiplier (ISO 10303-41 si_prefix enumeration).
SI_PREFIX_FACTORS = {
    "EXA": 1e18, "PETA": 1e15, "TERA": 1e12, "GIGA": 1e9, "MEGA": 1e6,
    "KILO": 1e3, "HECTO": 1e2, "DECA": 1e1,
    "DECI": 1e-1, "CENTI": 1e-2, "MILLI": 1e-3, "MICRO": 1e-6,
    "NANO": 1e-9, "PICO": 1e-12, "FEMTO": 1e-15, "ATTO": 1e-18,
}

#: How many millimetres one unprefixed base length unit is.
_BASE_LENGTH_MM = {"METRE": 1000.0, "METER": 1000.0}


class UnitError(ValueError):
    """Raised when a unit construct is present but unresolvable."""


@dataclass(frozen=True)
class StepUnits:
    """The file's resolved length unit and global tolerance.

    ``scale_to_mm`` multiplies a raw coordinate from the file into
    millimetres. ``tolerance_mm`` is the ``distance_accuracy_value`` global
    tolerance converted to mm (None when the file declares none).
    """

    scale_to_mm: float
    unit_name: str
    unit_entity_id: Optional[int] = None
    tolerance_mm: Optional[float] = None
    tolerance_raw: Optional[float] = None
    notes: tuple = ()

    def to_dict(self) -> dict:
        return {
            "scale_to_mm": self.scale_to_mm,
            "unit_name": self.unit_name,
            "unit_entity_id": self.unit_entity_id,
            "tolerance_mm": self.tolerance_mm,
            "tolerance_raw": self.tolerance_raw,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# unit resolution
# --------------------------------------------------------------------------- #

def _typed_parts(entity: Entity) -> List[Typed]:
    """All Typed parts of an entity: the parts of a complex instance, or the
    entity itself viewed as a single Typed."""
    if entity.keyword is None:
        return [p for p in entity.params if isinstance(p, Typed)]
    return [Typed(entity.keyword, tuple(entity.params))]


def _find_part(entity: Entity, keyword: str) -> Optional[Typed]:
    for part in _typed_parts(entity):
        if part.keyword.upper() == keyword:
            return part
    return None


def is_length_unit(entity: Entity) -> bool:
    """True when the entity declares a LENGTH unit.

    A complex instance carries an explicit ``LENGTH_UNIT()`` part; a bare
    ``SI_UNIT`` entity is a length unit when its base name is METRE.
    """
    if _find_part(entity, "LENGTH_UNIT") is not None:
        return True
    si = _find_part(entity, "SI_UNIT")
    if si is not None:
        name = _si_unit_name(si)
        return name in _BASE_LENGTH_MM
    return False


def _si_unit_name(si: Typed) -> str:
    names = [p.name.upper() for p in si.params if isinstance(p, Enum)]
    return names[-1] if names else ""


def _si_unit_prefix(si: Typed) -> Optional[str]:
    names = [p.name.upper() for p in si.params if isinstance(p, Enum)]
    return names[0] if len(names) >= 2 else None


def _real_value(value: object) -> Optional[float]:
    if isinstance(value, Real):
        return value.value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Typed):  # LENGTH_MEASURE(0.01) wrapper
        for p in value.params:
            v = _real_value(p)
            if v is not None:
                return v
    return None


def resolve_unit_scale_mm(
    step: StepFile,
    unit: Union[Entity, Ref, int],
    _seen: Optional[Set[int]] = None,
) -> float:
    """How many millimetres one of ``unit`` is; follows conversion chains.

    Handles ``SI_UNIT`` (prefix * base) and ``CONVERSION_BASED_UNIT``
    (factor * referenced unit, recursively, with a cycle guard). Raises
    :class:`UnitError` when the construct is malformed or unresolvable.
    """
    if isinstance(unit, (Ref, int)):
        entity = step.get(unit)
        if entity is None:
            raise UnitError(f"unit reference #{unit if isinstance(unit, int) else unit.id} "
                            f"does not resolve")
    else:
        entity = unit
    seen = _seen or set()
    if entity.id in seen:
        raise UnitError(f"unit conversion cycle at #{entity.id}")
    seen = seen | {entity.id}

    si = _find_part(entity, "SI_UNIT")
    if si is not None:
        name = _si_unit_name(si)
        base_mm = _BASE_LENGTH_MM.get(name)
        if base_mm is None:
            raise UnitError(f"#{entity.id}: SI_UNIT base {name!r} is not a "
                            f"length unit")
        prefix = _si_unit_prefix(si)
        factor = 1.0
        if prefix is not None:
            if prefix not in SI_PREFIX_FACTORS:
                raise UnitError(f"#{entity.id}: unknown SI prefix {prefix!r}")
            factor = SI_PREFIX_FACTORS[prefix]
        return factor * base_mm

    cbu = _find_part(entity, "CONVERSION_BASED_UNIT")
    if cbu is not None:
        # CONVERSION_BASED_UNIT(name, #measure_with_unit)
        measure_ref = next((p for p in cbu.params if isinstance(p, Ref)), None)
        if measure_ref is None:
            raise UnitError(f"#{entity.id}: CONVERSION_BASED_UNIT without a "
                            f"measure reference")
        measure = step.get(measure_ref)
        if measure is None:
            raise UnitError(f"#{entity.id}: conversion measure "
                            f"#{measure_ref.id} does not resolve")
        # LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(v), #unit)
        value: Optional[float] = None
        next_unit: Optional[Ref] = None
        for p in measure.params:
            if value is None:
                v = _real_value(p)
                if v is not None:
                    value = v
                    continue
            if isinstance(p, Ref) and next_unit is None:
                next_unit = p
        if value is None or next_unit is None:
            raise UnitError(f"#{measure.id}: malformed "
                            f"LENGTH_MEASURE_WITH_UNIT")
        return value * resolve_unit_scale_mm(step, next_unit, seen)

    raise UnitError(f"#{entity.id}: not a resolvable unit entity")


def _unit_display_name(step: StepFile, entity: Entity) -> str:
    cbu = _find_part(entity, "CONVERSION_BASED_UNIT")
    if cbu is not None:
        for p in cbu.params:
            if isinstance(p, str):
                return p
    si = _find_part(entity, "SI_UNIT")
    if si is not None:
        prefix = _si_unit_prefix(si)
        name = _si_unit_name(si)
        return f"{prefix or ''}{'.' if prefix else ''}{name}".strip(".")
    return f"#{entity.id}"


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def extract_step_units(source: Union[str, StepFile]) -> StepUnits:
    """Extract the global length scale and tolerance from a STEP file.

    ``source`` is either raw part-21 text (parsed with the harness's
    ``io/formats/step.py`` parser) or an already-parsed :class:`StepFile`.

    Resolution order:
      1. the length unit referenced by an ``UNCERTAINTY_MEASURE_WITH_UNIT``
         (the geometric context's own declaration, the strongest signal);
      2. else the first length-unit entity in DATA order.
    Missing units fall back to scale 1.0 (millimetre assumption) with a note
    saying so -- the caller can escalate that note to a verifier warning.
    """
    step = parse(source) if isinstance(source, str) else source
    notes: List[str] = []

    # Global tolerance: UNCERTAINTY_MEASURE_WITH_UNIT, simple or complex part.
    tol_raw: Optional[float] = None
    tol_unit_ref: Optional[Ref] = None
    for ent_id in step.order:
        entity = step.entities[ent_id]
        umwu = _find_part(entity, "UNCERTAINTY_MEASURE_WITH_UNIT")
        if umwu is None:
            continue
        value: Optional[float] = None
        unit_ref: Optional[Ref] = None
        for p in umwu.params:
            if value is None:
                v = _real_value(p)
                if v is not None:
                    value = v
                    continue
            if isinstance(p, Ref) and unit_ref is None:
                unit_ref = p
        if value is not None:
            tol_raw, tol_unit_ref = value, unit_ref
            break

    # Length unit: prefer the tolerance's own unit, else first in DATA order.
    unit_entity: Optional[Entity] = None
    if tol_unit_ref is not None:
        candidate = step.get(tol_unit_ref)
        if candidate is not None and is_length_unit(candidate):
            unit_entity = candidate
    if unit_entity is None:
        for ent_id in step.order:
            if is_length_unit(step.entities[ent_id]):
                unit_entity = step.entities[ent_id]
                break

    if unit_entity is None:
        notes.append("no length unit declared; assuming millimetres "
                     "(scale 1.0) -- verify against expected part size")
        scale = 1.0
        unit_name = "assumed MILLI.METRE"
        unit_id = None
    else:
        scale = resolve_unit_scale_mm(step, unit_entity)
        unit_name = _unit_display_name(step, unit_entity)
        unit_id = unit_entity.id

    tol_mm: Optional[float] = None
    if tol_raw is not None:
        tol_scale = scale
        if tol_unit_ref is not None:
            tol_entity = step.get(tol_unit_ref)
            if tol_entity is not None and is_length_unit(tol_entity):
                tol_scale = resolve_unit_scale_mm(step, tol_entity)
        tol_mm = tol_raw * tol_scale
    else:
        notes.append("no UNCERTAINTY_MEASURE_WITH_UNIT declared")

    return StepUnits(
        scale_to_mm=scale,
        unit_name=unit_name,
        unit_entity_id=unit_id,
        tolerance_mm=tol_mm,
        tolerance_raw=tol_raw,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

_STEP_TEMPLATE = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('selfcheck'),'2;1');
FILE_NAME('t.step','2026-07-16',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
{body}
ENDSEC;
END-ISO-10303-21;
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="STEP SI_UNIT / CONVERSION_BASED_UNIT / "
                    "UNCERTAINTY_MEASURE_WITH_UNIT extraction to a "
                    "scale-to-mm factor + global tolerance (cadquery "
                    "importer policy + kerf step_reader).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="parse synthetic STEP snippets in mm, m, and "
                             "inch and assert the resolved scales.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # 1. Millimetre file with a global tolerance (the common AP214 shape).
    mm = _STEP_TEMPLATE.format(body=(
        "#10=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));\n"
        "#11=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-2),#10,"
        "'distance_accuracy_value','confusion accuracy');"
    ))
    u = extract_step_units(mm)
    assert abs(u.scale_to_mm - 1.0) < 1e-12, u.to_dict()
    assert u.tolerance_mm is not None and abs(u.tolerance_mm - 0.01) < 1e-12
    assert u.unit_entity_id == 10
    print(f"[selfcheck] mm file: scale={u.scale_to_mm}, "
          f"tol={u.tolerance_mm} mm")

    # 2. Metre file: the silent-1000x case this module exists to catch.
    m = _STEP_TEMPLATE.format(body=(
        "#10=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.));\n"
        "#11=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-5),#10,"
        "'distance_accuracy_value','');"
    ))
    u = extract_step_units(m)
    assert abs(u.scale_to_mm - 1000.0) < 1e-9, u.to_dict()
    assert u.tolerance_mm is not None and abs(u.tolerance_mm - 0.01) < 1e-12
    print(f"[selfcheck] metre file: scale={u.scale_to_mm} (1000x guarded)")

    # 3. Inch via CONVERSION_BASED_UNIT chained onto a mm SI_UNIT.
    inch = _STEP_TEMPLATE.format(body=(
        "#10=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));\n"
        "#11=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);\n"
        "#12=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#10);\n"
        "#13=(CONVERSION_BASED_UNIT('INCH',#12)LENGTH_UNIT()NAMED_UNIT(#11));\n"
        "#14=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(3.94E-4),#13,"
        "'distance_accuracy_value','');"
    ))
    u = extract_step_units(inch)
    assert abs(u.scale_to_mm - 25.4) < 1e-9, u.to_dict()
    assert u.unit_name == "INCH"
    assert u.tolerance_mm is not None and abs(u.tolerance_mm - 3.94e-4 * 25.4) < 1e-9
    print(f"[selfcheck] inch file: scale={u.scale_to_mm} via "
          f"CONVERSION_BASED_UNIT chain")

    # 4. Centimetre prefix and micro prefix.
    cm = _STEP_TEMPLATE.format(
        body="#10=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.CENTI.,.METRE.));")
    assert abs(extract_step_units(cm).scale_to_mm - 10.0) < 1e-9
    um = _STEP_TEMPLATE.format(
        body="#10=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MICRO.,.METRE.));")
    assert abs(extract_step_units(um).scale_to_mm - 1e-3) < 1e-15
    print("[selfcheck] centi/micro prefixes resolved")

    # 5. No unit declared -> mm assumption with an explicit note.
    bare = _STEP_TEMPLATE.format(
        body="#1=CARTESIAN_POINT('',(0.,0.,0.));")
    u = extract_step_units(bare)
    assert u.scale_to_mm == 1.0 and u.unit_entity_id is None
    assert any("assuming millimetres" in n for n in u.notes)
    print("[selfcheck] missing unit falls back to mm WITH a note")

    # 6. Cycle guard.
    cyc = _STEP_TEMPLATE.format(body=(
        "#11=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);\n"
        "#12=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(2.),#13);\n"
        "#13=(CONVERSION_BASED_UNIT('LOOP',#12)LENGTH_UNIT()NAMED_UNIT(#11));"
    ))
    try:
        extract_step_units(cyc)
        raise AssertionError("cycle not detected")
    except UnitError as exc:
        assert "cycle" in str(exc)
    print("[selfcheck] conversion cycle raises UnitError")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
