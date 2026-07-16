"""Named RC-servo dimension database and Gridfinity envelope constants (sdfx).

Reimplementation of the dimension lookup tables from deadsy/sdfx
(MIT licence, (c) 2017-2019 Jason T. Harris):

* ``obj/servo.go`` -- ``initServoLookup``: a named lookup covering the well
  known servo size categories (nano through giant) with the Hitec reference
  models (HS-40, HS-55, HS-85BB, HS-225BB, HS-311, HS-805BB, HS-1005SGT) plus
  the Annimos DS3218.  Each entry carries the full ``ServoParms`` field set:
  body size, mounting lug size, mounting hole layout, lug z-offset, shaft
  x-offset, shaft length/radius and mounting hole radius.
* ``obj/gridfinity.go`` -- the Gridfinity (https://gridfinity.xyz/) envelope
  constants: the 42.0 mm base pitch, the 7.0 mm height unit, the female /
  male / lip profile step heights and corner rounds, and the magnet / bolt
  hole geometry.  Only the constants are ported here, not the SDF geometry
  generators (``GfBase`` / ``GfBody``).

This module is the *standards data* layer, structured exactly like its sibling
:mod:`standards.thread_database` (which ports sdfx ``sdf/screw.go`` the same
way): NamedTuple records, a module-level ``_DB`` built by small builder
functions, ``*_lookup`` / ``*_names`` accessors and derived-geometry helpers.

All dimensions are millimetres, as in the Go source.  Pure stdlib,
deterministic.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, NamedTuple, Optional, Tuple

__all__ = [
    "ServoParameters",
    "servo_lookup",
    "servo_names",
    "servo_mount_hole_positions",
    "servo_shaft_xy",
    "GridfinityEnvelope",
    "GRIDFINITY",
    "gridfinity_base_footprint",
    "gridfinity_body_footprint",
    "gridfinity_body_height",
    "gridfinity_grid_centers",
    "gridfinity_hole_offset_from_center",
    "main",
]


# ---------------------------------------------------------------------------
# Servo database (obj/servo.go)
# ---------------------------------------------------------------------------

class ServoParameters(NamedTuple):
    """Resolved dimensions for a named servo, in millimetres.

    Mirrors sdfx ``obj.ServoParms``.  ``body`` and ``mount`` are (x, y, z)
    sizes; ``hole`` is the (x, y) mounting hole layout spacing (center to
    center, 0 for a single hole pair on the x-axis).
    """

    name: str
    body: Tuple[float, float, float]   # body size (x, y, z)
    mount: Tuple[float, float, float]  # mounting lugs size (x, y, z)
    hole: Tuple[float, float]          # mounting hole layout spacing (x, y)
    mount_offset: float  # z-offset of mounting lugs (from base of servo)
    shaft_offset: float  # x-offset of drive shaft (mount hole center to shaft)
    shaft_length: float
    shaft_radius: float
    hole_radius: float


_DB: Dict[str, ServoParameters] = {}


def _servo_add(names: Tuple[str, ...],
               body: Tuple[float, float, float],
               mount: Tuple[float, float, float],
               hole: Tuple[float, float],
               mount_offset: float,
               shaft_offset: float,
               shaft_length: float,
               shaft_radius: float,
               hole_radius: float) -> None:
    """Register one servo record under each of its names (model + size class)."""
    for name in names:
        _DB[name] = ServoParameters(
            name=name,
            body=body,
            mount=mount,
            hole=hole,
            mount_offset=mount_offset,
            shaft_offset=shaft_offset,
            shaft_length=shaft_length,
            shaft_radius=shaft_radius,
            hole_radius=hole_radius,
        )


def _build() -> None:
    # Values verbatim from obj/servo.go initServoLookup (mm).
    _servo_add(
        ("hitec_hs_40", "nano"),
        body=(20.0, 8.7, 20.3),
        mount=(28.0, 8.7, 1.0),
        hole=(24.0, 0.0),
        mount_offset=12.0,
        shaft_offset=6.4,
        shaft_length=2.8,
        shaft_radius=1.4,
        hole_radius=1.0,
    )
    _servo_add(
        ("hitec_hs_55", "submicro"),
        body=(22.6, 11.5, 24.5),
        mount=(32.6, 10.4, 1.0),
        hole=(28.5, 0.0),
        mount_offset=16.6,
        shaft_offset=9.0,
        shaft_length=2.5,
        shaft_radius=1.25,
        hole_radius=0.95,
    )
    _servo_add(
        ("hitec_hs_85bb", "micro"),
        body=(29.1, 13.0, 30.4),
        mount=(40.0, 12.0, 2.0),
        hole=(35.6, 0.0),
        mount_offset=19.0,
        shaft_offset=9.8,
        shaft_length=3.8,
        shaft_radius=1.9,
        hole_radius=2.25,
    )
    _servo_add(
        ("hitec_hs_225bb", "mini"),
        body=(32.3, 16.8, 33.0),
        mount=(44.3, 16.0, 2.2),
        hole=(39.6, 7.9),
        mount_offset=23.5,
        shaft_offset=12.2,
        shaft_length=3.3,
        shaft_radius=1.65,
        hole_radius=2.25,
    )
    _servo_add(
        ("hitec_hs_311", "standard"),
        body=(40.2, 20.2, 38.3),
        mount=(52.9, 20.2, 2.5),
        hole=(47.6, 10.1),
        mount_offset=26.5,
        shaft_offset=13.85,
        shaft_length=3.5,
        shaft_radius=1.75,
        hole_radius=2.15,
    )
    _servo_add(
        ("annimos_ds3218",),
        body=(40.0, 20.0, 41.5),
        mount=(54.2, 18.5, 3.0),
        hole=(49.5, 10.0),
        mount_offset=28.0,
        shaft_offset=14.75,
        shaft_length=4.2,
        shaft_radius=2.1,
        hole_radius=2.15,
    )
    _servo_add(
        ("hitec_hs_805bb", "large"),
        body=(65.9, 29.9, 59.3),
        mount=(82.9, 29.9, 4.0),
        hole=(74.9, 17.8),
        mount_offset=42.0,
        shaft_offset=18.9,
        shaft_length=5.4,
        shaft_radius=2.7,
        hole_radius=2.8,
    )
    _servo_add(
        ("hitec_hs_1005sgt", "giant"),
        body=(64.0, 33.0, 73.3),
        mount=(88.0, 33.0, 4.0),
        hole=(76.0, 21.0),
        mount_offset=53.3,
        shaft_offset=20.6,
        shaft_length=7.6,
        shaft_radius=3.8,
        hole_radius=3.0,
    )


_build()


def servo_lookup(name: str) -> ServoParameters:
    """Look up a servo's dimensions by name, or raise ``KeyError``.

    Names are the sdfx names: manufacturer models (``hitec_hs_55``) and size
    class aliases (``nano``, ``submicro``, ``micro``, ``mini``, ``standard``,
    ``large``, ``giant``).
    """
    try:
        return _DB[name]
    except KeyError:
        raise KeyError('servo "%s" not found' % name)


def servo_names(prefix: Optional[str] = None) -> List[str]:
    """Sorted list of known servo names, optionally filtered by prefix."""
    names = sorted(_DB.keys())
    if prefix is not None:
        names = [n for n in names if n.startswith(prefix)]
    return names


def servo_mount_hole_positions(k: ServoParameters) -> List[Tuple[float, float]]:
    """The four mounting hole (x, y) centers, with the drive shaft at origin.

    Reproduces the hole layout of sdfx ``Servo2D``: holes sit at
    (+-hole_x/2, +-hole_y/2) about the mount plate center, and the whole
    pattern is shifted so the drive shaft is at the origin
    (x shift = hole_x/2 - shaft_offset).  For two-hole servos (hole_y == 0)
    duplicate y positions collapse and only two distinct points are returned.
    """
    x_ofs = 0.5 * k.hole[0]
    y_ofs = 0.5 * k.hole[1]
    shift = 0.5 * k.hole[0] - k.shaft_offset
    raw = [
        (x_ofs, y_ofs), (-x_ofs, y_ofs),
        (x_ofs, -y_ofs), (-x_ofs, -y_ofs),
    ]
    seen: List[Tuple[float, float]] = []
    for (x, y) in raw:
        p = (x + shift, y)
        if p not in seen:
            seen.append(p)
    return seen


def servo_shaft_xy(k: ServoParameters) -> Tuple[float, float]:
    """Drive shaft (x, y) position relative to the body center.

    From sdfx ``Servo3D``: the shaft is offset from the body center by
    -(hole_x/2 - shaft_offset) along x, and centered in y.
    """
    return (-(0.5 * k.hole[0] - k.shaft_offset), 0.0)


# ---------------------------------------------------------------------------
# Gridfinity envelope constants (obj/gridfinity.go)
# ---------------------------------------------------------------------------

class GridfinityEnvelope(NamedTuple):
    """Gridfinity envelope constants, in millimetres.

    Field names track the ``gf*`` constants in sdfx ``obj/gridfinity.go``.
    The female profile is the socket in a baseplate; the male profile is the
    plug under a bin; the lip is the stacking recess in a bin top.  Each
    profile is a three-step (h0 45-degree chamfer, h1 straight, h2 45-degree
    chamfer) rounded-square section.
    """

    female_size: float    # base grid pitch (socket square size)
    female_round: float   # socket corner radius
    female_h0: float      # socket upper chamfer height
    female_h1: float      # socket straight wall height
    female_h2: float      # socket lower chamfer height
    female_height: float  # total socket profile height

    male_size: float      # bin base plug square size
    male_round: float     # plug corner radius
    male_h0: float        # plug upper chamfer height
    male_h1: float        # plug straight wall height
    male_h2: float        # plug lower chamfer height
    male_height: float    # total plug profile height

    lip_round: float      # stacking lip corner radius
    lip_h0: float         # lip upper chamfer height
    lip_h1: float         # lip straight wall height
    lip_h2: float         # lip lower chamfer height
    lip_height: float     # total lip profile height

    height_size: float    # gridfinity height unit (bin height quantum)

    hole_offset: float    # magnet/bolt hole inset from the plug chamfer edge
    hole_minor: float     # bolt through-hole radius (M3 clearance, r = 1.5)
    hole_major: float     # magnet pocket radius (6 mm magnet, r = 3.25)
    hole_height: float    # magnet pocket depth

    floor: float          # floor thickness for an empty bin (not in spec)
    base_height: float    # extra base height for magnet mounts (not in spec)


GRIDFINITY = GridfinityEnvelope(
    female_size=42.0,
    female_round=0.5 * 8.0,
    female_h0=2.15,
    female_h1=1.8,
    female_h2=0.7,
    female_height=2.15 + 1.8 + 0.7,
    male_size=41.5,
    male_round=0.5 * 7.5,
    male_h0=2.15,
    male_h1=1.8,
    male_h2=0.8,
    male_height=2.15 + 1.8 + 0.8,
    lip_round=0.5 * 7.5,
    lip_h0=1.9,
    lip_h1=1.8,
    lip_h2=0.7,
    lip_height=1.9 + 1.8 + 0.7,
    height_size=7.0,
    hole_offset=4.8,
    hole_minor=0.5 * 3.0,
    hole_major=0.5 * 6.5,
    hole_height=2.0,
    floor=1.0,
    base_height=4.0,
)


def gridfinity_base_footprint(nx: int, ny: int) -> Tuple[float, float]:
    """Outer (x, y) size in mm of an nx-by-ny baseplate (n * 42.0)."""
    if nx < 1 or ny < 1:
        raise ValueError("gridfinity base size must be at least 1x1")
    return (nx * GRIDFINITY.female_size, ny * GRIDFINITY.female_size)


def gridfinity_body_footprint(nx: int, ny: int) -> Tuple[float, float]:
    """Outer (x, y) size in mm of an nx-by-ny bin body.

    From sdfx ``GfBody``: n * female_size minus the female/male clearance
    (42.0 - 41.5 = 0.5 mm total, i.e. 0.25 mm per side).
    """
    if nx < 1 or ny < 1:
        raise ValueError("gridfinity body size must be at least 1x1")
    clearance = GRIDFINITY.female_size - GRIDFINITY.male_size
    return (
        nx * GRIDFINITY.female_size - clearance,
        ny * GRIDFINITY.female_size - clearance,
    )


def gridfinity_body_height(nz: int) -> float:
    """Bin body extrusion height in mm for nz gridfinity height units.

    From sdfx ``GfBody``: nz * height_size + lip_height - male_height (the
    base plugs add male_height back underneath).
    """
    if nz < 1:
        raise ValueError("gridfinity body height must be at least 1 unit")
    return nz * GRIDFINITY.height_size + GRIDFINITY.lip_height - GRIDFINITY.male_height


def gridfinity_grid_centers(nx: int, ny: int) -> List[Tuple[float, float]]:
    """Cell center (x, y) positions of an nx-by-ny grid about the origin.

    Reproduces sdfx ``gfGrid``: cells on a female_size (42.0 mm) pitch,
    pattern centered at the origin, x-major then y ordering.
    """
    if nx < 1 or ny < 1:
        raise ValueError("gridfinity grid size must be at least 1x1")
    pitch = GRIDFINITY.female_size
    x_ofs = -0.5 * (nx - 1) * pitch
    y_ofs = -0.5 * (ny - 1) * pitch
    centers: List[Tuple[float, float]] = []
    for i in range(nx):
        for j in range(ny):
            centers.append((x_ofs + i * pitch, y_ofs + j * pitch))
    return centers


def gridfinity_hole_offset_from_center(g: GridfinityEnvelope = GRIDFINITY) -> float:
    """Magnet/bolt hole distance from the cell center along x and y.

    From sdfx ``gfHoles``: 0.5 * male_size - (male_h0 + male_h2 + hole_offset).
    The four holes sit at (+-ofs, +-ofs) in each cell.
    """
    return 0.5 * g.male_size - (g.male_h0 + g.male_h2 + g.hole_offset)


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _selfcheck() -> None:
    # Servo entries straight from obj/servo.go.
    hs55 = servo_lookup("hitec_hs_55")
    assert hs55.body == (22.6, 11.5, 24.5)
    assert hs55.mount == (32.6, 10.4, 1.0)
    assert hs55.hole == (28.5, 0.0)
    assert hs55.mount_offset == 16.6
    assert hs55.shaft_offset == 9.0
    assert hs55.shaft_length == 2.5
    assert hs55.shaft_radius == 1.25
    assert hs55.hole_radius == 0.95

    hs40 = servo_lookup("hitec_hs_40")
    assert hs40.body == (20.0, 8.7, 20.3)
    assert hs40.hole_radius == 1.0

    hs85 = servo_lookup("hitec_hs_85bb")
    assert hs85.mount == (40.0, 12.0, 2.0)
    assert hs85.shaft_offset == 9.8

    giant = servo_lookup("giant")
    assert giant.body == (64.0, 33.0, 73.3)
    assert giant.shaft_radius == 3.8

    # Aliases resolve to the same dimensions as their reference model.
    assert servo_lookup("submicro")[1:] == hs55[1:]
    assert servo_lookup("nano")[1:] == hs40[1:]
    assert servo_lookup("standard")[1:] == servo_lookup("hitec_hs_311")[1:]

    # 15 names: 8 records, 7 of which carry a size-class alias.
    names = servo_names()
    assert len(names) == 15, names
    assert names == sorted(names)
    assert servo_names("hitec_") == [
        "hitec_hs_1005sgt", "hitec_hs_225bb", "hitec_hs_311",
        "hitec_hs_40", "hitec_hs_55", "hitec_hs_805bb", "hitec_hs_85bb",
    ]
    for n in names:
        assert servo_lookup(n).name == n

    # Unknown name raises KeyError.
    try:
        servo_lookup("no_such_servo")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown servo")

    # Derived servo geometry.
    holes55 = servo_mount_hole_positions(hs55)
    assert len(holes55) == 2  # two-hole layout (hole_y == 0)
    xs = sorted(p[0] for p in holes55)
    assert abs((xs[1] - xs[0]) - 28.5) < 1e-12  # hole spacing preserved
    # shaft at origin: nearest hole at -shaft_offset, far hole at hole_x - shaft_offset
    assert abs(xs[0] - (-9.0)) < 1e-12
    assert abs(xs[1] - (28.5 - 9.0)) < 1e-12
    holes311 = servo_mount_hole_positions(servo_lookup("hitec_hs_311"))
    assert len(holes311) == 4  # four-hole layout
    assert servo_shaft_xy(hs55) == (-(0.5 * 28.5 - 9.0), 0.0)

    # Gridfinity constants straight from obj/gridfinity.go.
    assert GRIDFINITY.female_size == 42.0
    assert GRIDFINITY.male_size == 41.5
    assert GRIDFINITY.height_size == 7.0
    assert GRIDFINITY.female_round == 4.0
    assert GRIDFINITY.male_round == 3.75
    assert abs(GRIDFINITY.female_height - 4.65) < 1e-12
    assert abs(GRIDFINITY.male_height - 4.75) < 1e-12
    assert abs(GRIDFINITY.lip_height - 4.4) < 1e-12
    assert GRIDFINITY.hole_minor == 1.5
    assert GRIDFINITY.hole_major == 3.25
    assert GRIDFINITY.hole_height == 2.0
    assert GRIDFINITY.hole_offset == 4.8

    # Derived gridfinity geometry.
    assert gridfinity_base_footprint(2, 3) == (84.0, 126.0)
    assert gridfinity_body_footprint(1, 1) == (41.5, 41.5)
    assert abs(gridfinity_body_footprint(2, 1)[0] - 83.5) < 1e-12
    assert abs(gridfinity_body_height(3) - (21.0 + 4.4 - 4.75)) < 1e-12
    centers = gridfinity_grid_centers(2, 2)
    assert len(centers) == 4
    assert centers[0] == (-21.0, -21.0)
    assert centers[-1] == (21.0, 21.0)
    assert gridfinity_grid_centers(1, 1) == [(0.0, 0.0)]
    ofs = gridfinity_hole_offset_from_center()
    assert abs(ofs - (0.5 * 41.5 - (2.15 + 0.8 + 4.8))) < 1e-12

    for bad in ((0, 1), (1, 0)):
        try:
            gridfinity_base_footprint(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for %r" % (bad,))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Named servo dimension database and Gridfinity envelope "
                    "constants (ported from deadsy/sdfx obj/servo.go and "
                    "obj/gridfinity.go).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run internal consistency checks and exit")
    parser.add_argument("--list", action="store_true",
                        help="list known servo names")
    parser.add_argument("--servo", metavar="NAME",
                        help="print the dimensions of a named servo")
    args = parser.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        print("servo_database selfcheck OK: %d servo names, "
              "gridfinity pitch %.1f mm" % (len(servo_names()),
                                            GRIDFINITY.female_size))
        return 0

    if args.list:
        for n in servo_names():
            print(n)
        return 0

    if args.servo:
        k = servo_lookup(args.servo)
        for field, value in zip(k._fields, k):
            print("%-13s %r" % (field, value))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
