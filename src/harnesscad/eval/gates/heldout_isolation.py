"""THE HELD-OUT ISOLATION GATE. A private test set stops being private the moment
a second module can read it.

WHAT IT ENFORCES
----------------
Two held-out brief splits exist so the harness can tell "it got better" from "it
learnt this corpus": ``eval.corpus.heldout`` (scored only through
``eval.corpus.score``) and ``eval.hardcorpus.heldout`` (scored only through
``eval.hardcorpus.score``). Each split names in its own docstring the one scorer
allowed to import it. This gate makes that mechanical: it walks every ``.py`` file
under ``src/`` and ``tests/`` and FAILS if any module other than the sanctioned
scorer actually imports a held-out split.

WHY A GATE AS WELL AS THE UNIT TEST
-----------------------------------
``tests/eval/corpus/test_holdout_isolation.py`` already asserts this for the corpus
split, and it runs inside the sharded test job. This module lifts the same check to
a first-class, standing CI gate -- one ``python -m`` invocation, next to
``precision_floor`` and ``warning_channel`` -- so the held-out discipline is
enforced by the same surface as every other release gate and cannot be lost if the
test tree is resharded or a scorer is renamed. It also covers BOTH splits (corpus
and hardcorpus) from one place. The two checks are deliberately redundant: a
contamination guard nobody would notice going missing is a guard already gone.

It reasons about IMPORTS, not mentions. These packages explain the discipline at
length in prose that names the module; a check that flagged prose would be switched
off within the week, so only a real ``import`` statement (plain, relative, ``from``,
or ``importlib``/``__import__``) trips it. This module does not itself import either
held-out split -- it reads source as text -- so running the gate cannot be the leak.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

__all__ = ["Split", "SPLITS", "Leak", "GateReport", "scan", "check", "main"]

#: .../src/harnesscad/eval/gates/heldout_isolation.py -> repo root is five up.
_ROOT = pathlib.Path(__file__).resolve().parents[4]
_SRC = _ROOT / "src" / "harnesscad"
_TESTS = _ROOT / "tests"


@dataclass(frozen=True)
class Split:
    """One held-out module and the single scorer permitted to reach it."""

    name: str                 # dotted module of the held-out split
    scorer: str               # dotted module of the only sanctioned importer
    rel_scorer: pathlib.Path  # path of the scorer, relative to src/harnesscad

    @property
    def leaf(self) -> str:
        return self.name.rsplit(".", 1)[-1]

    def imports_re(self) -> "re.Pattern[str]":
        pkg = self.name.rsplit(".", 1)[0]           # e.g. harnesscad.eval.corpus
        leaf = re.escape(self.leaf)                 # e.g. heldout
        pkg_re = re.escape(pkg)
        tail = re.escape("." + self.leaf)
        return re.compile(
            r"(?mx)"
            r"^\s*from\s+" + pkg_re + tail + r"\s+import\b"
            r"|^\s*import\s+" + pkg_re + tail + r"\b"
            r"|^\s*from\s+" + pkg_re + r"\s+import\s+(?:[^\n]*[,(]\s*)?" + leaf + r"\b"
            r"|(?:importlib\.import_module|__import__)\s*\(\s*[\"'][^\"']*"
            + re.escape(pkg.split(".")[-1] + "." + self.leaf))


#: Every held-out split in the repository and its sanctioned scorer.
SPLITS: Sequence[Split] = (
    Split("harnesscad.eval.corpus.heldout", "harnesscad.eval.corpus.score",
          pathlib.Path("eval") / "corpus" / "score.py"),
    Split("harnesscad.eval.hardcorpus.heldout", "harnesscad.eval.hardcorpus.score",
          pathlib.Path("eval") / "hardcorpus" / "score.py"),
)


@dataclass(frozen=True)
class Leak:
    """A module that imports a held-out split and is not its scorer."""

    split: str
    offender: str   # path relative to the repo root

    def to_dict(self) -> dict:
        return {"split": self.split, "offender": self.offender}


@dataclass
class GateReport:
    leaks: List[Leak] = field(default_factory=list)
    #: split -> the scorer that is REQUIRED to import it, and whether it does.
    scorers: Dict[str, bool] = field(default_factory=dict)
    checked_files: int = 0

    @property
    def ok(self) -> bool:
        return not self.leaks and all(self.scorers.values())

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checked_files": self.checked_files,
                "leaks": [l.to_dict() for l in self.leaks],
                "scorers_import_their_split": self.scorers}


#: The unittest guards of the same discipline. They MUST name every import form as
#: a fixture and in prose to prove the check would fire, so they are not leaks --
#: exactly as ``tests/eval/corpus/test_holdout_isolation.py`` allow-lists itself.
_GUARD_TEST_NAMES = frozenset({"test_holdout_isolation.py"})


def _allowed_paths(split: Split) -> set:
    """Paths permitted to name the split's import: its scorer and the split itself."""
    leaf_path = _SRC / split.rel_scorer.parent / (split.leaf + ".py")
    return {
        (_SRC / split.rel_scorer).resolve(),
        leaf_path.resolve(),
        pathlib.Path(__file__).resolve(),   # this gate reasons ABOUT the imports
    }


def scan(roots: Sequence[pathlib.Path] = (_SRC, _TESTS)) -> GateReport:
    """Walk the trees and collect every illicit import of a held-out split."""
    report = GateReport()
    files: List[pathlib.Path] = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.rglob("*.py")))

    for split in SPLITS:
        pattern = split.imports_re()
        allowed = _allowed_paths(split)
        scorer_ok = False
        scorer_path = (_SRC / split.rel_scorer).resolve()
        for path in files:
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not pattern.search(text):
                continue
            if path.resolve() == scorer_path:
                scorer_ok = True
                continue
            if path.resolve() in allowed or path.name in _GUARD_TEST_NAMES:
                continue
            report.leaks.append(Leak(split.name, str(path.relative_to(_ROOT))))
        report.scorers[split.name] = scorer_ok
    report.checked_files = sum(1 for _ in files if "__pycache__" not in _.parts)
    return report


def check() -> GateReport:
    return scan()


def format_text(report: GateReport) -> str:
    lines: List[str] = []
    lines.append("HELD-OUT ISOLATION GATE")
    lines.append("=" * 72)
    lines.append("scanned %d source/test files for imports of a held-out split"
                 % report.checked_files)
    lines.append("")
    for split, ok in sorted(report.scorers.items()):
        lines.append("  %-40s scorer imports it: %s"
                     % (split, "yes" if ok else "NO -- the guarded path is DEAD"))
    lines.append("")
    if report.ok:
        lines.append("PASS: every held-out split is reachable only through its "
                     "sanctioned scorer.")
    else:
        lines.append("FAIL:")
        for leak in report.leaks:
            lines.append("  [leak] %s is imported by %s -- it is no longer held out."
                         % (leak.split, leak.offender))
        for split, ok in sorted(report.scorers.items()):
            if not ok:
                lines.append("  [dead-guard] nothing imports %s through its scorer; "
                             "the path this gate protects does not exist." % split)
        lines.append("")
        lines.append("A held-out set stops being held out the moment a second "
                     "module can read it. Score through the sanctioned scorer; "
                     "tune on the dev split.")
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", dest="as_json")


def run(args: argparse.Namespace) -> int:
    report = check()
    if getattr(args, "as_json", False):
        import json
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_text(report))
    return 0 if report.ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="heldout_isolation",
        description="Fail the build if any held-out brief split is imported by "
                    "anything but its sanctioned scorer.")
    add_arguments(parser)
    return run(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
