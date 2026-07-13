"""CADTalk machine-made program synthesis (Sec. 4.1 / 9.3).

To build the large CADTalk-Cube / CADTalk-Ellip tracks, the authors convert
shape-abstraction results (labelled cuboid or ellipsoid primitives from PartNet
shapes) into OpenSCAD programs, "each such primitive forming a code block
associated with a semantic label". The translation is trivial and deterministic
(Sec. 9.3, "Program Translation"):

  * a cuboid (center + size, optional rotation) -> a native ``cube`` primitive
    wrapped in ``translate`` / ``rotate``;
  * an ellipsoid (semi-axis lengths + rotation + translation) -> a unit
    ``sphere`` wrapped in ``scale`` (OpenSCAD has no native ellipsoid).

Because a labelled shape often has several primitives sharing one part label, and
"consecutive blocks with the same labels can then be naturally grouped" (Sec.
4.1), this module emits one commented code block per *run* of consecutive
same-label primitives (a ``union { ... }`` when the run has more than one
primitive). It returns both the program text and the ground-truth ``block_id ->
label`` map, i.e. a ready-to-score CADTalk dataset entry.

Pure stdlib; deterministic; nothing is executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]


def _fmt(x: float) -> str:
    """Format a number compactly and deterministically (no locale, no -0.0)."""
    if x == 0:
        x = 0.0
    if float(x).is_integer():
        return str(int(x))
    return repr(round(float(x), 6))


def _vec(v: Sequence[float]) -> str:
    return "[" + ", ".join(_fmt(c) for c in v) + "]"


@dataclass
class Cuboid:
    """An axis-aligned (optionally rotated) box primitive with a part label."""

    label: str
    size: Vec3
    center: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec3 = (0.0, 0.0, 0.0)

    def to_scad(self) -> str:
        s = f"cube({_vec(self.size)}, center=true);"
        if any(self.rotation):
            s = f"rotate({_vec(self.rotation)}) " + s
        if any(self.center):
            s = f"translate({_vec(self.center)}) " + s
        return s


@dataclass
class Ellipsoid:
    """An ellipsoid primitive (unit sphere scaled by semi-axes) with a label."""

    label: str
    semi_axes: Vec3
    center: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec3 = (0.0, 0.0, 0.0)
    facets: int = 24

    def to_scad(self) -> str:
        s = f"scale({_vec(self.semi_axes)}) sphere(1, $fn={int(self.facets)});"
        if any(self.rotation):
            s = f"rotate({_vec(self.rotation)}) " + s
        if any(self.center):
            s = f"translate({_vec(self.center)}) " + s
        return s


Primitive = object  # Cuboid | Ellipsoid (structural: has .label and .to_scad()).


@dataclass
class ProgramEntry:
    """A synthesized CADTalk program plus its ground-truth annotations."""

    source: str
    block_labels: Dict[int, str] = field(default_factory=dict)
    #: For each block id, the indices (into the input primitive list) it covers.
    block_primitives: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def num_blocks(self) -> int:
        return len(self.block_labels)


def _runs(primitives: Sequence[Primitive]) -> List[Tuple[str, List[int]]]:
    """Group *consecutive* primitives sharing a label into runs (label, idxs)."""
    runs: List[Tuple[str, List[int]]] = []
    for idx, p in enumerate(primitives):
        lab = getattr(p, "label")
        if runs and runs[-1][0] == lab:
            runs[-1][1].append(idx)
        else:
            runs.append((lab, [idx]))
    return runs


def synthesize(
    primitives: Sequence[Primitive],
    group_consecutive: bool = True,
    category: Optional[str] = None,
) -> ProgramEntry:
    """Translate labelled primitives into a commented OpenSCAD program.

    With ``group_consecutive`` (default), each run of consecutive same-label
    primitives becomes one commented block (a ``union`` if the run has >1
    primitive); otherwise every primitive is its own block. Returns a
    :class:`ProgramEntry` carrying the source and the ground-truth
    ``block_id -> label`` and ``block_id -> primitive indices`` maps."""
    groups = _runs(primitives) if group_consecutive else [
        (getattr(p, "label"), [i]) for i, p in enumerate(primitives)
    ]
    lines: List[str] = []
    if category:
        lines.append(f"// category: {category}")
    block_labels: Dict[int, str] = {}
    block_primitives: Dict[int, List[int]] = {}
    for bid, (label, idxs) in enumerate(groups):
        block_labels[bid] = label
        block_primitives[bid] = list(idxs)
        lines.append(f"// [{bid}] {label}")
        if len(idxs) == 1:
            lines.append(primitives[idxs[0]].to_scad())
        else:
            lines.append("union() {")
            for i in idxs:
                lines.append("    " + primitives[i].to_scad())
            lines.append("}")
        lines.append("")
    source = "\n".join(lines).rstrip() + "\n"
    return ProgramEntry(source=source, block_labels=block_labels,
                        block_primitives=block_primitives)
