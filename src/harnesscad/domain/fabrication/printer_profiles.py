"""Printer wrapper-profile schema and static G-code bounds validation.

Derived from text-to-cad (MIT, Copyright (c) 2026 earthtojake).

What this gives the harness
---------------------------

A *wrapper profile* is the machine-readable contract between a slicer backend
and the harness: it names the backend and its native config files, and states
the machine's printable envelope (bed size, Z height, optional explicit motion
bounds) and filament temperatures. Two capabilities follow from it:

* :func:`load_profile` validates a profile mapping into a typed, frozen
  :class:`PrinterProfileSpec`, rejecting a malformed envelope up front rather
  than letting a bad bound silently pass a G-code file.
* :func:`validate_gcode` runs STATIC checks over a G-code body:
  the file must contain movement commands, extrusion moves and temperature
  commands, and every absolute ``X``/``Y``/``Z`` move must lie inside the
  profile's motion bounds. Unknown commands and relative-positioning blocks
  warn rather than fail.

Bounds policy: motion bounds default
to ``X=0..bed_size_mm[0]``, ``Y=0..bed_size_mm[1]``, ``Z=0..z_height_mm``.
A profile may override any axis via ``machine.motion_bounds_mm`` -- intended
only for real printers with safe off-bed wipe/purge positions, never as a way
to silence unexpected motion. Validation assumes absolute positioning until
``G91`` appears and resumes bounds checks after ``G90``; bounds are skipped
while relative mode is active. ``ok`` means the file passed these static checks
only -- it does NOT mean the G-code is safe to print on real hardware.

Relation to the rest of the printability stack
----------------------------------------------

:mod:`harnesscad.domain.fabrication.printability_verdict` judges a measured
model against a :class:`~harnesscad.domain.fabrication.printability_verdict.PrinterProfile`
(build volume, margin, min wall). This module carries the *machine-side* half
of the same idea -- the envelope as a real slicer profile states it -- and
:func:`to_printability_profile` converts one into that module's profile, so a
build envelope declared once for slicing is the same envelope the fit check
consults. :mod:`harnesscad.domain.fabrication.feature_minima` remains the
feature-level layer; nothing here is modified by or modifies those modules.

This module validates G-code *text*; it never runs a slicer and never touches
the filesystem. Where the original loader additionally requires
``native_config`` to be an existing absolute file, this version validates the
field's shape only (a non-empty string), keeping the loader deterministic and
side-effect free -- resolving a path against a real disk is the caller's job.

stdlib-only (``argparse``, ``dataclasses``, ``re``), deterministic, no network.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from harnesscad.domain.fabrication.printability_verdict import PrinterProfile

__all__ = [
    "ProfileError",
    "PREFERRED_BACKEND_ORDER",
    "SUPPORTED_GCODE_COMMANDS",
    "PrinterProfileSpec",
    "GCodeReport",
    "load_profile",
    "parse_axis_bounds",
    "strip_gcode_comment",
    "parse_command",
    "parse_numeric_tokens",
    "validate_gcode",
    "to_printability_profile",
    "main",
]


class ProfileError(ValueError):
    """A profile mapping is malformed, or a G-code body cannot be validated."""


# Accepted slicer backends, in preference order.
PREFERRED_BACKEND_ORDER: Tuple[str, ...] = ("orcaslicer", "prusa-slicer", "curaengine")

# The G-code commands the validator recognises.
# Anything else warns (never fails) -- an unknown command is a review prompt.
SUPPORTED_GCODE_COMMANDS: frozenset = frozenset(
    {
        "G0", "G1", "G2", "G3", "G4", "G21", "G28", "G29", "G90", "G91", "G92",
        "M18", "M73", "M82", "M83", "M84", "M104", "M106", "M107", "M109",
        "M117", "M118", "M140", "M190", "M201", "M203", "M204", "M205",
        "M220", "M221", "M400", "M500", "M501", "M900",
    }
)

# The movement and temperature command groups the required-checks list keys on.
_MOVEMENT_COMMANDS = frozenset({"G0", "G1", "G2", "G3"})
_TEMPERATURE_COMMANDS = frozenset({"M104", "M109", "M140", "M190"})

_AXES = ("x", "y", "z")

_RELATIVE_WARNING = (
    "Relative positioning found; XYZ bounds validation is skipped while "
    "relative mode is active."
)

_SOURCE = "text-to-cad plugins/cad/skills/gcode (MIT)"


@dataclass(frozen=True)
class PrinterProfileSpec:
    """A validated wrapper profile: backend, native config, envelope, filament.

    ``motion_bounds_mm`` maps each of ``"x"``/``"y"``/``"z"`` to a
    ``(min, max)`` pair in millimetres -- defaulted from the bed size and Z
    height, or taken from an explicit ``machine.motion_bounds_mm`` override.
    """

    backend: str
    native_config: str
    native_settings: Tuple[str, ...]
    native_filaments: Tuple[str, ...]
    machine_name: str
    bed_size_mm: Tuple[float, float]
    z_height_mm: float
    motion_bounds_mm: Dict[str, Tuple[float, float]]
    filament_type: str
    nozzle_temp_c: float
    bed_temp_c: float
    source: str = _SOURCE

    def __str__(self) -> str:
        return "%s (%s): %gx%gx%g mm, %s" % (
            self.machine_name,
            self.backend,
            self.bed_size_mm[0],
            self.bed_size_mm[1],
            self.z_height_mm,
            self.filament_type,
        )


@dataclass(frozen=True)
class GCodeReport:
    """The outcome of static G-code validation against a profile.

    ``ok`` is True when no error was raised. It means the body passed these
    static checks only -- not that the G-code is safe to print on hardware.
    """

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "stats": dict(self.stats),
        }


def _require_number(value: Any, label: str) -> float:
    """Coerce a required numeric profile field, rejecting bools and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError("Profile field %s must be a number." % label)
    return float(value)


def _require_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ProfileError("Profile field %s is required." % label)
    return text


def parse_axis_bounds(value: Any, label: str) -> Tuple[float, float]:
    """Parse an explicit ``[min, max]`` axis bound, rejecting an empty range."""
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise ProfileError("Profile field %s must be [min, max]." % label)
    lower = float(value[0])
    upper = float(value[1])
    if lower >= upper:
        raise ProfileError("Profile field %s must have min less than max." % label)
    return (lower, upper)


def _optional_path_list(data: Mapping[str, Any], field_name: str) -> Tuple[str, ...]:
    """Parse an optional list-of-paths field; present-but-empty is an error."""
    if field_name not in data:
        return ()
    raw = data.get(field_name)
    if not isinstance(raw, (list, tuple)):
        raise ProfileError("Profile field %s must be a list." % field_name)
    if not raw:
        raise ProfileError("Profile field %s cannot be empty when provided." % field_name)
    return tuple(_require_text(item, field_name) for item in raw)


def load_profile(data: Mapping[str, Any]) -> PrinterProfileSpec:
    """Validate a wrapper-profile mapping into a :class:`PrinterProfileSpec`.

    ``data`` is the parsed profile document (typically read from JSON;
    this version takes the mapping so the loader stays pure). Raises
    :class:`ProfileError` naming the offending field for an unknown backend, a
    missing ``native_config``/``machine.name``/``filament.type``, a malformed
    ``machine.bed_size_mm``, a non-numeric temperature or Z height, or an
    invalid ``machine.motion_bounds_mm`` override.
    """
    if not isinstance(data, Mapping):
        raise ProfileError("Profile must be an object.")

    backend = str(data.get("backend") or "").strip().lower()
    if backend not in PREFERRED_BACKEND_ORDER:
        raise ProfileError(
            "Profile backend must be one of: %s." % ", ".join(PREFERRED_BACKEND_ORDER)
        )

    native_config = _require_text(data.get("native_config"), "native_config")
    native_settings = _optional_path_list(data, "native_settings") or (native_config,)
    native_filaments = _optional_path_list(data, "native_filaments")

    machine = data.get("machine")
    if not isinstance(machine, Mapping):
        raise ProfileError("Profile field machine must be an object.")
    machine_name = _require_text(machine.get("name"), "machine.name")

    bed_size = machine.get("bed_size_mm")
    if (
        not isinstance(bed_size, (list, tuple))
        or len(bed_size) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in bed_size)
    ):
        raise ProfileError("Profile field machine.bed_size_mm must be [width, depth].")
    bed_x = _require_number(bed_size[0], "machine.bed_size_mm[0]")
    bed_y = _require_number(bed_size[1], "machine.bed_size_mm[1]")
    z_height = _require_number(machine.get("z_height_mm"), "machine.z_height_mm")
    for label, extent in (("bed_size_mm[0]", bed_x), ("bed_size_mm[1]", bed_y),
                          ("z_height_mm", z_height)):
        if extent <= 0.0:
            raise ProfileError(
                "Profile field machine.%s must be greater than 0." % label
            )

    # Default bounds are the printable envelope; an override is per-axis.
    motion_bounds: Dict[str, Tuple[float, float]] = {
        "x": (0.0, bed_x),
        "y": (0.0, bed_y),
        "z": (0.0, z_height),
    }
    if "motion_bounds_mm" in machine:
        raw_bounds = machine.get("motion_bounds_mm")
        if not isinstance(raw_bounds, Mapping):
            raise ProfileError("Profile field machine.motion_bounds_mm must be an object.")
        for axis in _AXES:
            if axis in raw_bounds:
                motion_bounds[axis] = parse_axis_bounds(
                    raw_bounds[axis], "machine.motion_bounds_mm.%s" % axis
                )

    filament = data.get("filament")
    if not isinstance(filament, Mapping):
        raise ProfileError("Profile field filament must be an object.")
    filament_type = _require_text(filament.get("type"), "filament.type")
    nozzle_temp = _require_number(filament.get("nozzle_temp_c"), "filament.nozzle_temp_c")
    bed_temp = _require_number(filament.get("bed_temp_c"), "filament.bed_temp_c")

    return PrinterProfileSpec(
        backend=backend,
        native_config=native_config,
        native_settings=native_settings,
        native_filaments=native_filaments,
        machine_name=machine_name,
        bed_size_mm=(bed_x, bed_y),
        z_height_mm=z_height,
        motion_bounds_mm=motion_bounds,
        filament_type=filament_type,
        nozzle_temp_c=nozzle_temp,
        bed_temp_c=bed_temp,
    )


# ---------------------------------------------------------------------------
# G-code lexing helpers
# ---------------------------------------------------------------------------

_COMMAND_RE = re.compile(r"[GMT]\d+(?:\.\d+)?\Z")
_TOKEN_RE = re.compile(r"([A-Z])([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
_TOOL_RE = re.compile(r"T\d+\Z")


def strip_gcode_comment(line: str) -> str:
    """Drop a trailing ``;`` comment and surrounding whitespace."""
    return line.split(";", 1)[0].strip()


def parse_command(line: str) -> str:
    """The leading command word of a line (``"G1"``), or ``""`` if there is none.

    A sub-numbered command (``G1.1``) reduces to its integer form (``G1``).
    """
    stripped = strip_gcode_comment(line)
    if not stripped:
        return ""
    first = stripped.split(None, 1)[0].upper()
    if _COMMAND_RE.match(first):
        if "." in first:
            first = first.split(".", 1)[0]
        return first
    return first


def parse_numeric_tokens(line: str) -> Dict[str, float]:
    """The ``letter -> value`` numeric tokens on a line (``X10 Y-2`` -> X/Y)."""
    stripped = strip_gcode_comment(line).upper()
    tokens: Dict[str, float] = {}
    for key, value in _TOKEN_RE.findall(stripped):
        try:
            tokens[key] = float(value)
        except ValueError:  # pragma: no cover - regex guarantees a float
            continue
    return tokens


def validate_gcode(text: str, profile: PrinterProfileSpec) -> GCodeReport:
    """Statically validate a G-code body against ``profile``'s motion bounds.

    Applies the required-checks list. Fails when the body is
    empty, has no ``G0``/``G1``/``G2``/``G3`` movement, no extrusion move, no
    temperature command, or an absolute ``X``/``Y``/``Z`` move outside the
    profile's motion bounds. Warns on unknown commands and on relative
    positioning (bounds are skipped while ``G91`` is active, resuming at
    ``G90``). Never rewrites or deletes a command.
    """
    errors: List[str] = []
    warnings: List[str] = []
    unknown_commands: Dict[str, int] = {}
    lines = text.splitlines()
    stats: Dict[str, int] = {
        "lines": len(lines),
        "non_comment_lines": 0,
        "movement_commands": 0,
        "extrusion_moves": 0,
        "temperature_commands": 0,
    }

    if not text.strip():
        errors.append("G-code file is empty.")

    absolute_positioning = True
    x_min, x_max = profile.motion_bounds_mm["x"]
    y_min, y_max = profile.motion_bounds_mm["y"]
    z_min, z_max = profile.motion_bounds_mm["z"]
    ranges = {
        "X": (x_min, x_max),
        "Y": (y_min, y_max),
        "Z": (z_min, z_max),
    }

    for line_number, line in enumerate(lines, start=1):
        command = parse_command(line)
        if not command:
            continue
        stats["non_comment_lines"] += 1
        tokens = parse_numeric_tokens(line)

        if command not in SUPPORTED_GCODE_COMMANDS and not _TOOL_RE.match(command):
            unknown_commands.setdefault(command, line_number)

        if command == "G90":
            absolute_positioning = True
        elif command == "G91":
            absolute_positioning = False
            if _RELATIVE_WARNING not in warnings:
                warnings.append(_RELATIVE_WARNING)

        if command in _MOVEMENT_COMMANDS:
            stats["movement_commands"] += 1
            if command == "G1" and "E" in tokens:
                stats["extrusion_moves"] += 1
            if absolute_positioning:
                for axis in ("X", "Y", "Z"):
                    if axis not in tokens:
                        continue
                    low, high = ranges[axis]
                    value = tokens[axis]
                    if not low <= value <= high:
                        errors.append(
                            "Line %d: %s=%g is outside %s motion range %g..%g mm."
                            % (line_number, axis, value, axis, low, high)
                        )
        if command in _TEMPERATURE_COMMANDS:
            stats["temperature_commands"] += 1

    if stats["movement_commands"] == 0:
        errors.append("No G0/G1/G2/G3 movement commands found.")
    if stats["extrusion_moves"] == 0:
        errors.append("No extrusion moves found.")
    if stats["temperature_commands"] == 0:
        errors.append("No nozzle or bed temperature commands found.")
    if unknown_commands:
        sample = ", ".join(
            "%s line %d" % (cmd, line) for cmd, line in sorted(unknown_commands.items())[:12]
        )
        warnings.append(
            "Unknown or unsupported commands were left unchanged: %s." % sample
        )

    return GCodeReport(ok=not errors, errors=errors, warnings=warnings, stats=stats)


def to_printability_profile(
    spec: PrinterProfileSpec,
    margin_mm: float = 2.0,
    min_wall_mm: float = 0.8,
    min_feature_mm: float = 0.4,
    support_free_angle_deg: float = 45.0,
) -> PrinterProfile:
    """Convert a wrapper profile into a :mod:`printability_verdict` profile.

    The build volume is taken from the profile's *motion bounds* (their extent
    per axis), so an explicit ``machine.motion_bounds_mm`` override is honoured
    rather than silently overridden by the raw bed size. The remaining
    printability thresholds have no counterpart in a slicer wrapper profile and
    keep their :class:`~harnesscad.domain.fabrication.printability_verdict.PrinterProfile`
    defaults unless given.

    This is the composition point: a build envelope declared once for slicing
    becomes the envelope
    :func:`~harnesscad.domain.fabrication.printability_verdict.check_fit`
    measures against, so a model that would not fit is caught before a slicer
    is ever invoked.
    """
    extents = tuple(
        spec.motion_bounds_mm[axis][1] - spec.motion_bounds_mm[axis][0] for axis in _AXES
    )
    return PrinterProfile(
        build_volume_mm=(extents[0], extents[1], extents[2]),
        margin_mm=margin_mm,
        min_wall_mm=min_wall_mm,
        min_feature_mm=min_feature_mm,
        support_free_angle_deg=support_free_angle_deg,
    )


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------

_EXAMPLE_PROFILE: Dict[str, Any] = {
    "backend": "orcaslicer",
    "native_config": "/profiles/machine.json",
    "machine": {"name": "Example 256", "bed_size_mm": [256, 256], "z_height_mm": 256},
    "filament": {"type": "PLA", "nozzle_temp_c": 210, "bed_temp_c": 60},
}


def _selfcheck() -> None:
    # --- Loader: defaults derived from the envelope. ---
    spec = load_profile(_EXAMPLE_PROFILE)
    assert spec.backend == "orcaslicer"
    assert spec.machine_name == "Example 256"
    assert spec.bed_size_mm == (256.0, 256.0)
    assert spec.z_height_mm == 256.0
    assert spec.motion_bounds_mm == {
        "x": (0.0, 256.0),
        "y": (0.0, 256.0),
        "z": (0.0, 256.0),
    }
    # native_settings defaults to native_config when absent.
    assert spec.native_settings == ("/profiles/machine.json",)
    assert spec.native_filaments == ()
    assert spec.filament_type == "PLA"
    assert spec.nozzle_temp_c == 210.0

    # --- Loader: explicit motion bounds override only the named axes. ---
    override = dict(_EXAMPLE_PROFILE)
    override["machine"] = dict(_EXAMPLE_PROFILE["machine"])
    override["machine"]["motion_bounds_mm"] = {"x": [-5, 260]}
    bounded = load_profile(override)
    assert bounded.motion_bounds_mm["x"] == (-5.0, 260.0)
    assert bounded.motion_bounds_mm["y"] == (0.0, 256.0)

    # --- Loader: malformed profiles are rejected, by field. ---
    def _rejects(mutate: Dict[str, Any]) -> str:
        bad = dict(_EXAMPLE_PROFILE)
        bad.update(mutate)
        try:
            load_profile(bad)
        except ProfileError as exc:
            return str(exc)
        raise AssertionError("expected ProfileError for %r" % (mutate,))

    assert "backend" in _rejects({"backend": "slic3r"})
    assert "native_config" in _rejects({"native_config": ""})
    assert "machine" in _rejects({"machine": None})
    assert "filament" in _rejects({"filament": []})
    assert "bed_size_mm" in _rejects({"machine": {"name": "m", "bed_size_mm": [1, 2, 3],
                                                  "z_height_mm": 10}})
    # A bool is not a number, even though bool is an int subclass.
    assert "z_height_mm" in _rejects(
        {"machine": {"name": "m", "bed_size_mm": [10, 10], "z_height_mm": True}}
    )
    # An inverted axis range is an error, not a silently swapped one.
    try:
        parse_axis_bounds([10, 10], "test")
        raise AssertionError("expected ProfileError for an empty range")
    except ProfileError as exc:
        assert "min less than max" in str(exc)

    # --- Lexer. ---
    assert strip_gcode_comment("G1 X10 ; move") == "G1 X10"
    assert parse_command("; pure comment") == ""
    assert parse_command("G1.1 X5") == "G1"
    assert parse_command("  g28  ") == "G28"
    assert parse_numeric_tokens("G1 X10.5 Y-2 E.3") == {"G": 1.0, "X": 10.5, "Y": -2.0, "E": 0.3}

    # --- Validator: a well-formed body inside the envelope passes. ---
    good = "\n".join(
        [
            "; header",
            "G21",
            "G90",
            "M104 S210",
            "M140 S60",
            "G28",
            "G1 X10 Y10 Z0.2 E1",
            "G1 X200 Y200 Z10 E5",
        ]
    )
    report = validate_gcode(good, spec)
    assert report.ok, report.errors
    assert report.errors == []
    assert report.stats["movement_commands"] == 2
    assert report.stats["extrusion_moves"] == 2
    assert report.stats["temperature_commands"] == 2

    # --- Validator: an out-of-bounds absolute move is an error, per axis. ---
    over = validate_gcode(good + "\nG1 X300 Y10 Z1 E6", spec)
    assert not over.ok
    assert len(over.errors) == 1
    assert "X=300" in over.errors[0] and "0..256" in over.errors[0]

    # A negative move is out of the default 0-based bound too.
    assert not validate_gcode(good + "\nG1 X-1 Y10 E6", spec).ok
    # ...but is in bounds for the profile whose X range starts at -5.
    assert validate_gcode(good + "\nG1 X-1 Y10 E6", bounded).ok

    # --- Validator: required-checks list. ---
    empty = validate_gcode("", spec)
    assert not empty.ok
    assert "G-code file is empty." in empty.errors
    no_move = validate_gcode("M104 S210\nM140 S60", spec)
    assert not no_move.ok
    assert "No G0/G1/G2/G3 movement commands found." in no_move.errors
    assert "No extrusion moves found." in no_move.errors
    no_temp = validate_gcode("G90\nG1 X10 Y10 E1", spec)
    assert not no_temp.ok
    assert "No nozzle or bed temperature commands found." in no_temp.errors
    # A G0 travel move is movement but not extrusion.
    travel = validate_gcode("G90\nM104 S210\nG0 X10 Y10", spec)
    assert "No extrusion moves found." in travel.errors

    # --- Validator: relative positioning warns and suspends bounds checks. ---
    relative = "\n".join(
        ["G90", "M104 S210", "G1 X10 Y10 E1", "G91", "G1 X9999 Y9999 E2"]
    )
    rel_report = validate_gcode(relative, spec)
    assert rel_report.ok, rel_report.errors
    assert _RELATIVE_WARNING in rel_report.warnings
    # G90 resumes absolute bounds checking.
    resumed = validate_gcode(relative + "\nG90\nG1 X9999 Y10 E3", spec)
    assert not resumed.ok
    assert "X=9999" in resumed.errors[0]

    # --- Validator: unknown commands warn, never fail. ---
    unknown = validate_gcode(good + "\nM1234 S1", spec)
    assert unknown.ok
    assert any("M1234 line" in w for w in unknown.warnings)
    # A tool change is recognised and does not warn.
    tool = validate_gcode(good + "\nT0", spec)
    assert tool.ok
    assert not any("T0" in w for w in tool.warnings)

    # --- Bridge into printability_verdict. ---
    pp = to_printability_profile(spec)
    assert isinstance(pp, PrinterProfile)
    assert pp.build_volume_mm == (256.0, 256.0, 256.0)
    assert pp.margin_mm == 2.0
    # The override's extent, not the raw bed size, drives the build volume.
    assert to_printability_profile(bounded).build_volume_mm == (265.0, 256.0, 256.0)

    # The bridged envelope really does drive the fit check.
    from harnesscad.domain.fabrication.printability_verdict import check_fit

    assert check_fit((100.0, 100.0, 100.0), pp).fits
    assert not check_fit((300.0, 10.0, 10.0), pp).fits
    # 254 mm exceeds the usable 256 - 2*2 = 252 mm.
    assert not check_fit((254.0, 10.0, 10.0), pp).fits

    # --- Report serialisation is plain data. ---
    payload = report.as_dict()
    assert payload["ok"] is True
    assert payload["stats"]["lines"] == 8


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="printer_profiles",
        description="Printer wrapper-profile schema and G-code bounds validation "
        "(text-to-cad, MIT).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="assert schema, validator and bridge behaviour, then exit 0",
    )
    args = parser.parse_args(argv)
    if args.selfcheck:
        _selfcheck()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
