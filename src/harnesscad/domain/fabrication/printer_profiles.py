"""Printer wrapper-profile schema and static G-code bounds validation.

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
the filesystem. ``native_config`` is validated for shape only (a non-empty
string), keeping the loader deterministic and side-effect free -- resolving a
path against a real disk is the caller's job.

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

# The G-code commands the validator recognises, grouped by what they do so a
# new command lands next to its peers. Anything outside the union warns (never
# fails) -- an unknown command is a prompt to review, not a defect.
_MOVEMENT_COMMANDS = frozenset({"G0", "G1", "G2", "G3"})
_TEMPERATURE_COMMANDS = frozenset({"M104", "M109", "M140", "M190"})
_POSITIONING_COMMANDS = frozenset({"G4", "G21", "G28", "G29", "G90", "G91", "G92",
                                   "M82", "M83"})
_MACHINE_COMMANDS = frozenset({"M18", "M73", "M84", "M106", "M107", "M117", "M118",
                               "M201", "M203", "M204", "M205", "M220", "M221",
                               "M400", "M500", "M501", "M900"})

SUPPORTED_GCODE_COMMANDS: frozenset = (
    _MOVEMENT_COMMANDS | _TEMPERATURE_COMMANDS | _POSITIONING_COMMANDS
    | _MACHINE_COMMANDS
)

#: Command that restores absolute positioning, and the one that leaves it.
_ABSOLUTE_MODE_COMMAND = "G90"
_RELATIVE_MODE_COMMAND = "G91"

_AXES = ("x", "y", "z")
_AXIS_LETTERS = ("X", "Y", "Z")

#: How many distinct unknown commands the warning names before it stops.
_UNKNOWN_SAMPLE_LIMIT = 12

_RELATIVE_WARNING = (
    "Relative positioning found; XYZ bounds validation is skipped while "
    "relative mode is active."
)

_SOURCE = "harnesscad slicer wrapper profile"


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


# ---------------------------------------------------------------------------
# Typed field readers
#
# Each reader takes a raw value plus the dotted label it will be blamed under,
# so every rejection names the exact field the caller has to go and fix.
# ---------------------------------------------------------------------------

def _is_real_number(value: Any) -> bool:
    """True for an int/float that is not a bool (``bool`` subclasses ``int``)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_number(value: Any, label: str) -> float:
    """Coerce a required numeric profile field, rejecting bools and non-numbers."""
    if not _is_real_number(value):
        raise ProfileError("Profile field %s must be a number." % label)
    return float(value)


def _require_positive(value: float, label: str) -> float:
    if value <= 0.0:
        raise ProfileError("Profile field %s must be greater than 0." % label)
    return value


def _require_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ProfileError("Profile field %s is required." % label)
    return text


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileError("Profile field %s must be an object." % label)
    return value


def _require_number_pair(value: Any, message: str) -> Tuple[float, float]:
    """A two-element list/tuple of real numbers, or ``ProfileError(message)``."""
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(_is_real_number(item) for item in value)):
        raise ProfileError(message)
    return (float(value[0]), float(value[1]))


def parse_axis_bounds(value: Any, label: str) -> Tuple[float, float]:
    """Parse an explicit ``[min, max]`` axis bound, rejecting an empty range."""
    lower, upper = _require_number_pair(
        value, "Profile field %s must be [min, max]." % label)
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


def _read_backend(data: Mapping[str, Any]) -> str:
    backend = str(data.get("backend") or "").strip().lower()
    if backend not in PREFERRED_BACKEND_ORDER:
        raise ProfileError(
            "Profile backend must be one of: %s." % ", ".join(PREFERRED_BACKEND_ORDER)
        )
    return backend


def _read_envelope(
    machine: Mapping[str, Any]
) -> Tuple[Tuple[float, float], float, Dict[str, Tuple[float, float]]]:
    """The machine's ``(bed_size, z_height, motion_bounds)``.

    The printable envelope is the default bound on every axis; an explicit
    ``motion_bounds_mm`` replaces only the axes it actually names, so declaring
    an off-bed purge position on X does not quietly widen Y and Z.
    """
    bed_x, bed_y = _require_number_pair(
        machine.get("bed_size_mm"),
        "Profile field machine.bed_size_mm must be [width, depth].")
    z_height = _require_number(machine.get("z_height_mm"), "machine.z_height_mm")

    _require_positive(bed_x, "machine.bed_size_mm[0]")
    _require_positive(bed_y, "machine.bed_size_mm[1]")
    _require_positive(z_height, "machine.z_height_mm")

    bounds: Dict[str, Tuple[float, float]] = {
        "x": (0.0, bed_x),
        "y": (0.0, bed_y),
        "z": (0.0, z_height),
    }
    if "motion_bounds_mm" in machine:
        overrides = _require_mapping(machine.get("motion_bounds_mm"),
                                     "machine.motion_bounds_mm")
        for axis in _AXES:
            if axis in overrides:
                bounds[axis] = parse_axis_bounds(
                    overrides[axis], "machine.motion_bounds_mm.%s" % axis)
    return (bed_x, bed_y), z_height, bounds


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

    backend = _read_backend(data)
    native_config = _require_text(data.get("native_config"), "native_config")

    machine = _require_mapping(data.get("machine"), "machine")
    bed_size, z_height, motion_bounds = _read_envelope(machine)

    filament = _require_mapping(data.get("filament"), "filament")

    return PrinterProfileSpec(
        backend=backend,
        native_config=native_config,
        # A profile that names no settings files is understood to configure the
        # slicer with its single native config.
        native_settings=_optional_path_list(data, "native_settings") or (native_config,),
        native_filaments=_optional_path_list(data, "native_filaments"),
        machine_name=_require_text(machine.get("name"), "machine.name"),
        bed_size_mm=bed_size,
        z_height_mm=z_height,
        motion_bounds_mm=motion_bounds,
        filament_type=_require_text(filament.get("type"), "filament.type"),
        nozzle_temp_c=_require_number(filament.get("nozzle_temp_c"),
                                      "filament.nozzle_temp_c"),
        bed_temp_c=_require_number(filament.get("bed_temp_c"), "filament.bed_temp_c"),
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
    body = strip_gcode_comment(line)
    if not body:
        return ""
    word = body.split(None, 1)[0].upper()
    if _COMMAND_RE.match(word):
        return word.split(".", 1)[0]
    return word


def parse_numeric_tokens(line: str) -> Dict[str, float]:
    """The ``letter -> value`` numeric tokens on a line (``X10 Y-2`` -> X/Y)."""
    body = strip_gcode_comment(line).upper()
    tokens: Dict[str, float] = {}
    for letter, number in _TOKEN_RE.findall(body):
        try:
            tokens[letter] = float(number)
        except ValueError:  # pragma: no cover - regex guarantees a float
            continue
    return tokens


def _is_recognised(command: str) -> bool:
    """A known command, or any tool change (``T0``, ``T1``, ...)."""
    return command in SUPPORTED_GCODE_COMMANDS or bool(_TOOL_RE.match(command))


# ---------------------------------------------------------------------------
# Static validation
# ---------------------------------------------------------------------------

class _GCodeScan:
    """Accumulates the state one pass over a G-code body needs.

    Kept as an object rather than a pile of locals so each concern -- counting,
    positioning mode, bounds, unknown commands -- is one short method, and so
    the driver loop reads as "for each line, absorb it".
    """

    def __init__(self, profile: PrinterProfileSpec, line_count: int) -> None:
        self._ranges = {
            letter: profile.motion_bounds_mm[axis]
            for letter, axis in zip(_AXIS_LETTERS, _AXES)
        }
        self._absolute = True
        self._unknown: Dict[str, int] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, int] = {
            "lines": line_count,
            "non_comment_lines": 0,
            "movement_commands": 0,
            "extrusion_moves": 0,
            "temperature_commands": 0,
        }

    def _note_positioning(self, command: str) -> None:
        if command == _ABSOLUTE_MODE_COMMAND:
            self._absolute = True
        elif command == _RELATIVE_MODE_COMMAND:
            self._absolute = False
            if _RELATIVE_WARNING not in self.warnings:
                self.warnings.append(_RELATIVE_WARNING)

    def _check_bounds(self, line_number: int, tokens: Mapping[str, float]) -> None:
        """Blame every axis of this move that leaves the machine's envelope.

        Only meaningful under absolute positioning: a relative move's target
        depends on where the head already is, which a static pass cannot know,
        so those moves are skipped rather than guessed at.
        """
        if not self._absolute:
            return
        for letter in _AXIS_LETTERS:
            if letter not in tokens:
                continue
            low, high = self._ranges[letter]
            value = tokens[letter]
            if not low <= value <= high:
                self.errors.append(
                    "Line %d: %s=%g is outside %s motion range %g..%g mm."
                    % (line_number, letter, value, letter, low, high)
                )

    def absorb(self, line_number: int, line: str) -> None:
        command = parse_command(line)
        if not command:
            return
        self.stats["non_comment_lines"] += 1
        tokens = parse_numeric_tokens(line)

        if not _is_recognised(command):
            self._unknown.setdefault(command, line_number)

        self._note_positioning(command)

        if command in _MOVEMENT_COMMANDS:
            self.stats["movement_commands"] += 1
            if command == "G1" and "E" in tokens:
                self.stats["extrusion_moves"] += 1
            self._check_bounds(line_number, tokens)

        if command in _TEMPERATURE_COMMANDS:
            self.stats["temperature_commands"] += 1

    def finish(self) -> None:
        """Apply the whole-file requirements once every line has been seen."""
        if self.stats["movement_commands"] == 0:
            self.errors.append("No G0/G1/G2/G3 movement commands found.")
        if self.stats["extrusion_moves"] == 0:
            self.errors.append("No extrusion moves found.")
        if self.stats["temperature_commands"] == 0:
            self.errors.append("No nozzle or bed temperature commands found.")
        if self._unknown:
            sample = ", ".join(
                "%s line %d" % (command, line_number)
                for command, line_number in sorted(self._unknown.items())[:_UNKNOWN_SAMPLE_LIMIT]
            )
            self.warnings.append(
                "Unknown or unsupported commands were left unchanged: %s." % sample
            )


def validate_gcode(text: str, profile: PrinterProfileSpec) -> GCodeReport:
    """Statically validate a G-code body against ``profile``'s motion bounds.

    Fails when the body is empty, has no ``G0``/``G1``/``G2``/``G3`` movement,
    no extrusion move, no temperature command, or an absolute ``X``/``Y``/``Z``
    move outside the profile's motion bounds. Warns on unknown commands and on
    relative positioning (bounds are skipped while ``G91`` is active, resuming
    at ``G90``). Never rewrites or deletes a command.
    """
    lines = text.splitlines()
    scan = _GCodeScan(profile, len(lines))

    if not text.strip():
        scan.errors.append("G-code file is empty.")

    for line_number, line in enumerate(lines, start=1):
        scan.absorb(line_number, line)
    scan.finish()

    return GCodeReport(ok=not scan.errors, errors=scan.errors,
                       warnings=scan.warnings, stats=scan.stats)


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
    low_x, high_x = spec.motion_bounds_mm["x"]
    low_y, high_y = spec.motion_bounds_mm["y"]
    low_z, high_z = spec.motion_bounds_mm["z"]
    return PrinterProfile(
        build_volume_mm=(high_x - low_x, high_y - low_y, high_z - low_z),
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

    # --- Validator: whole-file requirements. ---
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
        description="Printer wrapper-profile schema and static G-code bounds validation.",
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
