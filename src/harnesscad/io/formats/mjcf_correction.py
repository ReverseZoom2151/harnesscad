"""MJCF (MuJoCo XML) parameter-correction writer with an audit trail.

Mined from **SimCorrect** (``mjcf_correction.py``), the correction engine its
fault pipeline drives after identification: a fluent ``Part`` helper that
rewrites a named body's inertial mass or a named joint's zero reference inside
an MJCF file, records every change as a :class:`CorrectionRecord`, and -- when
no source model is available -- emits a standalone correction-record XML so the
patch is still an auditable artifact rather than a mutation that happened
somewhere.

Why the harness wants it: MJCF is the simulation-facing sibling of the CAD
formats already under ``io/formats``. A text-to-CAD part that becomes a robot
body ends up as an MJCF ``<body>``/``<joint>``, and a simulation-driven
correction loop (see ``eval/quality/physics/fault_identification``) must land
its fix in that file deterministically, with the old value preserved in the
record. The write is surgical: only the targeted attribute changes, everything
else in the document round-trips through ``xml.etree`` untouched.

stdlib-only (``xml.etree``, ``dataclasses``), deterministic aside from
timestamps, no MuJoCo import anywhere.
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

__all__ = ["CorrectionRecord", "MjcfPart", "main"]


@dataclass(frozen=True)
class CorrectionRecord:
    """One applied correction: which part, which field, old -> new."""

    part_name: str
    field: str
    old_value: object
    new_value: object
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def __repr__(self) -> str:
        return (f"CorrectionRecord(part={self.part_name!r}, field={self.field!r}, "
                f"{self.old_value} -> {self.new_value})")


class MjcfPart:
    """Fluent MJCF corrector for body mass and joint zero-reference fields.

    ``MjcfPart("grip", "model.xml").set_mass(0.160).export("corrected.xml")``
    rewrites ``<body name="grip"><inertial mass=...>`` in place (creating the
    ``<inertial>`` element if the body lacks one) and records the change.
    Without a source file, ``export`` writes a standalone
    ``<harnesscad_mjcf_correction>`` record document instead.
    """

    def __init__(self, name: str, xml_source: Optional[Union[str, Path]] = None):
        if not name:
            raise ValueError("part name must be non-empty")
        self.name = name
        self.xml_source = str(xml_source) if xml_source is not None else None
        self._mass: Optional[float] = None
        self._ref: Optional[float] = None
        self._corrections: List[CorrectionRecord] = []
        self._tree: Optional[ET.ElementTree] = None
        self._root: Optional[ET.Element] = None
        if self.xml_source is not None:
            path = Path(self.xml_source)
            if not path.exists():
                raise FileNotFoundError(f"MJCF source not found: {path}")
            self._tree = ET.parse(str(path))
            self._root = self._tree.getroot()

    # ----------------------------------------------------------------- #
    # Fluent setters
    # ----------------------------------------------------------------- #
    def set_mass(self, mass_kg: float) -> "MjcfPart":
        """Correct ``<body name=...><inertial mass=...>`` (kg)."""
        if mass_kg <= 0:
            raise ValueError("mass must be positive")
        self._mass = float(mass_kg)
        return self

    def set_ref(self, ref_rad: float) -> "MjcfPart":
        """Correct ``<joint name=... ref=...>`` (radians)."""
        self._ref = float(ref_rad)
        return self

    # ----------------------------------------------------------------- #
    # Export
    # ----------------------------------------------------------------- #
    def export(self, output_path: Union[str, Path]) -> "MjcfPart":
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._tree is not None and self._root is not None:
            self._apply_to_tree()
            self._tree.write(str(path), encoding="unicode", xml_declaration=False)
        else:
            self._write_record_xml(path)
        return self

    @property
    def corrections(self) -> List[CorrectionRecord]:
        return list(self._corrections)

    def report(self) -> str:
        lines = [f"MJCF correction - part: {self.name!r}"]
        if not self._corrections:
            lines.append("  No corrections recorded.")
        for record in self._corrections:
            lines.append(f"  {record.field}: {record.old_value} -> {record.new_value}")
        return "\n".join(lines)

    # ----------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------- #
    def _apply_to_tree(self) -> None:
        assert self._root is not None
        if self._mass is not None:
            self._apply_mass()
        if self._ref is not None:
            self._apply_ref()

    def _apply_mass(self) -> None:
        assert self._root is not None and self._mass is not None
        for body in self._root.iter("body"):
            if body.get("name") != self.name:
                continue
            inertial = body.find("inertial")
            if inertial is None:
                inertial = ET.SubElement(body, "inertial")
            old = inertial.get("mass", "unknown")
            inertial.set("mass", f"{self._mass:.6f}")
            self._corrections.append(
                CorrectionRecord(self.name, "inertial.mass", old, self._mass))
            return
        raise KeyError(f"no <body name={self.name!r}> in MJCF source")

    def _apply_ref(self) -> None:
        assert self._root is not None and self._ref is not None
        for joint in self._root.iter("joint"):
            if joint.get("name") != self.name:
                continue
            old = joint.get("ref", "0.0")
            joint.set("ref", f"{self._ref:.6f}")
            self._corrections.append(
                CorrectionRecord(self.name, "joint.ref", old, self._ref))
            return
        raise KeyError(f"no <joint name={self.name!r}> in MJCF source")

    def _write_record_xml(self, path: Path) -> None:
        root = ET.Element("harnesscad_mjcf_correction")
        root.set("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        part_el = ET.SubElement(root, "part")
        part_el.set("name", self.name)
        if self._mass is not None:
            el = ET.SubElement(part_el, "correction")
            el.set("field", "inertial.mass")
            el.set("value", f"{self._mass:.6f}")
            el.set("unit", "kg")
            self._corrections.append(
                CorrectionRecord(self.name, "inertial.mass", "unknown", self._mass))
        if self._ref is not None:
            el = ET.SubElement(part_el, "correction")
            el.set("field", "joint.ref")
            el.set("value", f"{self._ref:.6f}")
            el.set("unit", "rad")
            self._corrections.append(
                CorrectionRecord(self.name, "joint.ref", "unknown", self._ref))
        ET.ElementTree(root).write(str(path), encoding="unicode", xml_declaration=True)

    def __repr__(self) -> str:
        return f"MjcfPart({self.name!r}, mass={self._mass}, ref={self._ref})"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_DEMO_MJCF = """<mujoco model="demo_arm">
  <worldbody>
    <body name="link1" pos="0 0 0.1">
      <joint name="joint1" type="hinge" axis="0 0 1" ref="0.14"/>
      <body name="grip" pos="0.25 0 0">
        <inertial mass="0.100" pos="0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.io.formats.mjcf_correction",
        description="MJCF parameter-correction writer (SimCorrect).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="apply a mass and a joint-ref correction to a "
                             "temporary demo MJCF and verify the rewrite.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "demo.xml"
        src.write_text(_DEMO_MJCF, encoding="utf-8")
        out_mass = Path(tmp) / "grip_corrected.xml"
        part = MjcfPart("grip", src).set_mass(0.160).export(out_mass)
        print(part.report())
        corrected = ET.parse(str(out_mass)).getroot()
        masses = [b.find("inertial").get("mass")  # type: ignore[union-attr]
                  for b in corrected.iter("body") if b.get("name") == "grip"]
        assert masses == ["0.160000"], masses

        out_ref = Path(tmp) / "joint1_corrected.xml"
        joint = MjcfPart("joint1", src).set_ref(0.0).export(out_ref)
        print(joint.report())
        ref_root = ET.parse(str(out_ref)).getroot()
        refs = [j.get("ref") for j in ref_root.iter("joint")
                if j.get("name") == "joint1"]
        assert refs == ["0.000000"], refs

        out_record = Path(tmp) / "record_only.xml"
        MjcfPart("grip").set_mass(0.160).export(out_record)
        record = ET.parse(str(out_record)).getroot()
        assert record.tag == "harnesscad_mjcf_correction"
        print("[selfcheck] standalone correction record written")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
