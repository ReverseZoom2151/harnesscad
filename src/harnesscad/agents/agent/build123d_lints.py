"""Deterministic Build123d code lints and fillet auto-heal (Studio-OSS).

This code-validation pass
``app/api/generate/route.ts`` and the auto-heal path in
``app/api/compile/route.ts``). Studio wraps its model-generated Build123d code
in two deterministic safety layers that this module ports:

**Pre-execution lints** for the failure modes its 48-hour run kept hitting:

  * ``SUBTRACT_BEFORE_BASE`` -- a boolean subtraction appears before any 3D
    base solid is assigned ("subtract from nothing");
  * ``SUBTRACT_FROM_2D`` -- ``Circle()`` / ``BuildSketch()`` / ``make_face()``
    used together with subtraction: 2D minus 3D crashes the kernel with
    "Dimensions of objects to subtract from are inconsistent";
  * ``KEYWORD_PRIMITIVE_ARGS`` -- ``Cylinder(r=5, h=10)`` style keyword
    arguments that Build123d primitives reject (``align=`` is the allowed
    exception);
  * ``POS_ROT_KEYWORDS`` -- ``Pos(X=...)`` / ``Rot(Z=...)`` keyword axes;
  * ``SCALE_METHOD`` -- ``.scale()`` calls (multiply dimensions instead);
  * ``NO_RESULT_VARIABLE`` -- the final shape must land in ``result``;
  * ``BUILDER_MODE`` -- ``with BuildPart()`` builder mode where algebra mode
    was required;
  * ``UNGUARDED_FILLET`` -- a fillet/chamfer call outside try/except.

**Fillet auto-heal**: when a compile fails with a fillet/chamfer error, strip
every fillet/chamfer line and re-run -- the user still gets a solid when the
rest of the model is valid. Also the preventive variant: wrap each unguarded
``x = fillet(...)`` assignment in try/except falling back to the unfilleted
shape, exactly the defensive pattern Studio's prompt demands.

This complements :mod:`harnesscad.agents.agent.code_repair_rules` (CadQuery /
FreeCAD dialect rules); Build123d is a distinct dialect with distinct failure
modes. stdlib-only (``re``), deterministic.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Build123dLint",
    "lint_build123d",
    "is_fillet_failure",
    "strip_fillets",
    "guard_fillets",
    "main",
]

_PRIMITIVES = ("Box", "Cylinder", "Sphere", "Cone", "Torus")
_PRIM_RE = "|".join(_PRIMITIVES)
_2D_BASES = ("Circle", "BuildSketch", "make_face", "RectangleRounded")

_FILLET_ERROR_MARKERS = (
    "fillet", "chamfer", "Failed creating a fillet", "BRep_API: command not done",
)


@dataclass(frozen=True)
class Build123dLint:
    code: str
    line: int              # 1-based; 0 when the finding is code-wide
    message: str

    def __str__(self) -> str:
        where = f" @line {self.line}" if self.line else ""
        return f"{self.code}{where}: {self.message}"


def _code_lines(source: str) -> List[Tuple[int, str]]:
    """(1-based line number, text) for non-comment lines."""
    out = []
    for i, raw in enumerate(source.split("\n"), start=1):
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            out.append((i, raw))
    return out


def lint_build123d(source: str) -> List[Build123dLint]:
    """All deterministic findings for a Build123d source string."""
    findings: List[Build123dLint] = []
    lines = _code_lines(source)

    # Subtraction ordering and 2D-base subtraction.
    subtract_re = re.compile(r"\s-\s")
    base_assign_re = re.compile(rf"^\s*\w+\s*=\s*(?:{_PRIM_RE})\s*\(")
    first_subtract = None
    first_base = None
    for lineno, text in lines:
        if first_subtract is None and subtract_re.search(text) \
                and "=" in text.split("#")[0]:
            first_subtract = lineno
        if first_base is None and base_assign_re.match(text):
            first_base = lineno
    has_subtraction = first_subtract is not None
    if has_subtraction and (first_base is None or first_subtract < first_base):
        findings.append(Build123dLint(
            "SUBTRACT_BEFORE_BASE", first_subtract or 0,
            "boolean subtraction before any 3D base solid is assigned; create "
            "the base primitive (Cylinder/Box/...) first"))

    if has_subtraction:
        for lineno, text in lines:
            for name in _2D_BASES:
                if re.search(rf"\b{name}\s*\(", text):
                    findings.append(Build123dLint(
                        "SUBTRACT_FROM_2D", lineno,
                        f"{name}() is 2D and cannot participate in 3D boolean "
                        f"subtraction; use a 3D primitive (e.g. Cylinder(r, "
                        "thickness) instead of Circle(r))"))
                    break

    # Keyword arguments on primitives.
    for lineno, text in lines:
        m = re.search(rf"\b({_PRIM_RE})\s*\(([^)]*)\)", text)
        if m:
            args = m.group(2)
            kw = re.search(r"\b(?!align\b)([A-Za-z_]\w*)\s*=", args)
            if kw:
                findings.append(Build123dLint(
                    "KEYWORD_PRIMITIVE_ARGS", lineno,
                    f"{m.group(1)}() rejects keyword argument "
                    f"'{kw.group(1)}='; use positional arguments "
                    "(align= is the only allowed keyword)"))
        if re.search(r"\b(?:Pos|Rot)\s*\([^)]*\b[XYZ]\s*=", text):
            findings.append(Build123dLint(
                "POS_ROT_KEYWORDS", lineno,
                "Pos()/Rot() take positional axes only: Pos(x, y, z)"))
        if re.search(r"\.scale\s*\(", text):
            findings.append(Build123dLint(
                "SCALE_METHOD", lineno,
                ".scale() is not supported; multiply dimensions in the "
                "constructor instead"))

    # result variable.
    if not re.search(r"^\s*result\s*=", source, re.MULTILINE):
        findings.append(Build123dLint(
            "NO_RESULT_VARIABLE", 0,
            "the final shape must be assigned to a variable named 'result'"))

    # Builder mode.
    for lineno, text in lines:
        if re.search(r"with\s+Build(?:Part|Sketch|Line)\s*\(", text):
            findings.append(Build123dLint(
                "BUILDER_MODE", lineno,
                "builder mode detected; algebra mode was required"))
            break

    # Unguarded fillet/chamfer.
    guarded = _guarded_line_numbers(source)
    for lineno, text in lines:
        if re.search(r"\b(?:fillet|chamfer)\s*\(", text) and lineno not in guarded:
            findings.append(Build123dLint(
                "UNGUARDED_FILLET", lineno,
                "fillet/chamfer outside try/except; radii too large for small "
                "edges crash the kernel -- guard with a fallback to the "
                "unfilleted shape"))

    return findings


def _guarded_line_numbers(source: str) -> set:
    """Line numbers inside any try block (coarse indentation-based scan)."""
    guarded = set()
    lines = source.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("try:"):
            indent = len(lines[i]) - len(lines[i].lstrip())
            j = i + 1
            while j < len(lines):
                line = lines[j]
                if line.strip() and (len(line) - len(line.lstrip())) <= indent \
                        and not line.strip().startswith(("except", "finally", "else")):
                    break
                guarded.add(j + 1)
                j += 1
            i = j
        else:
            i += 1
    return guarded


# --------------------------------------------------------------------------- #
# Fillet auto-heal
# --------------------------------------------------------------------------- #
def is_fillet_failure(error_message: str) -> bool:
    """Does a compile error message look like a fillet/chamfer failure."""
    lower = error_message.lower()
    return any(marker.lower() in lower for marker in _FILLET_ERROR_MARKERS)


def strip_fillets(source: str) -> Tuple[str, int]:
    """Remove every fillet/chamfer line (Studio's auto-heal retry).

    Returns ``(healed_source, removed_line_count)``. Lines whose sole purpose
    is the fillet call are dropped; the rest of the model is untouched.
    """
    kept: List[str] = []
    removed = 0
    for line in source.split("\n"):
        lowered = line.strip().lower()
        if not lowered.startswith("#") and (
                "fillet(" in lowered or "chamfer(" in lowered):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


_FILLET_ASSIGN_RE = re.compile(
    r"^(?P<indent>\s*)(?P<target>\w+)\s*=\s*(?:fillet|chamfer)\s*\("
    r"\s*(?P<base>\w+)\.edges\(\)")


def guard_fillets(source: str) -> Tuple[str, int]:
    """Wrap each unguarded ``x = fillet(base.edges(), r)`` in try/except.

    The except branch falls back to the unfilleted base shape -- the
    defensive pattern Studio's code-generation prompt mandates. Returns
    ``(guarded_source, wrapped_count)``.
    """
    guarded_lines = _guarded_line_numbers(source)
    out: List[str] = []
    wrapped = 0
    for lineno, line in enumerate(source.split("\n"), start=1):
        m = _FILLET_ASSIGN_RE.match(line)
        if m and lineno not in guarded_lines:
            indent = m.group("indent")
            target = m.group("target")
            base = m.group("base")
            out.append(f"{indent}try:")
            out.append(f"{indent}    {line.strip()}")
            out.append(f"{indent}except Exception:")
            out.append(f"{indent}    {target} = {base}")
            wrapped += 1
        else:
            out.append(line)
    return "\n".join(out), wrapped


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_BAD_CODE = """from build123d import *
base = Circle(12.5)
base = base - Box(30, 2, 10)
disc = Cylinder(r=12.5, h=2.5)
disc = disc.scale(2)
disc = Pos(X=1, Y=2, Z=0) * disc
final = fillet(disc.edges(), 2)
"""

_GOOD_CODE = """from build123d import *
disc = Cylinder(12.5, 2.5)
disc = disc - Pos(0, 0, 0) * Box(30, 2, 10)
try:
    disc = fillet(disc.edges(), 0.5)
except Exception:
    pass
result = disc
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.agents.agent.build123d_lints",
        description="Deterministic Build123d lints and fillet auto-heal "
                    "(Studio-OSS).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="lint a known-bad and a known-good snippet, then "
                             "exercise strip/guard fillet healing.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    findings = lint_build123d(_BAD_CODE)
    codes = {f.code for f in findings}
    for f in findings:
        print(f"  [lint] {f}")
    for expected in ("SUBTRACT_BEFORE_BASE", "SUBTRACT_FROM_2D",
                     "KEYWORD_PRIMITIVE_ARGS", "POS_ROT_KEYWORDS",
                     "SCALE_METHOD", "NO_RESULT_VARIABLE", "UNGUARDED_FILLET"):
        assert expected in codes, (expected, codes)
    print(f"[selfcheck] bad snippet: {len(findings)} findings")

    clean = lint_build123d(_GOOD_CODE)
    assert not clean, [str(f) for f in clean]
    print("[selfcheck] good snippet: 0 findings")

    assert is_fillet_failure("BRep_API: command not done")
    healed, removed = strip_fillets(_BAD_CODE)
    assert removed == 1 and "fillet(" not in healed
    print(f"[selfcheck] strip_fillets removed {removed} line(s)")

    src = "box = Box(10, 10, 5)\nbox = fillet(box.edges(), 2)\nresult = box\n"
    guarded, wrapped = guard_fillets(src)
    assert wrapped == 1 and "except Exception:" in guarded
    assert not any(f.code == "UNGUARDED_FILLET" for f in lint_build123d(guarded))
    print(f"[selfcheck] guard_fillets wrapped {wrapped} call(s)")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
