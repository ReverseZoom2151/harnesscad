"""THE ENFORCEMENT. The hard corpus's held-out split is held out because this says so.

A discipline that lives only in a docstring lasts until the first person in a hurry.
This scans every ``.py`` file in the source and test trees and FAILS if any module
other than ``harnesscad.eval.hardcorpus.score`` imports
``harnesscad.eval.hardcorpus.heldout``. It mirrors
``tests/eval/corpus/test_holdout_isolation.py`` exactly, for the same reason: the
pressure experiment had one corpus, one hand, and no held-out set, so no instrument
was left that could tell "the harness got better" from "the harness learned this
corpus".
"""

from __future__ import annotations

import pathlib
import re
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src" / "harnesscad"
_TESTS = _ROOT / "tests"

#: The one module allowed to reach the held-out briefs.
SCORER = _SRC / "eval" / "hardcorpus" / "score.py"

_ALLOWED = {
    SCORER.resolve(),
    (_SRC / "eval" / "hardcorpus" / "heldout.py").resolve(),
    pathlib.Path(__file__).resolve(),
}

_IMPORTS = re.compile(
    r"""(?mx)
      ^\s*from\s+harnesscad\.eval\.hardcorpus\.heldout\s+import\b
    | ^\s*import\s+harnesscad\.eval\.hardcorpus\.heldout\b
    | ^\s*from\s+harnesscad\.eval\.hardcorpus\s+import\s+(?:[^\n]*[,(]\s*)?heldout\b
    | ^\s*from\s+\.\s*import\s+(?:[^\n]*[,(]\s*)?heldout\b
    | ^\s*from\s+\.heldout\s+import\b
    | (?:importlib\.import_module|__import__)\s*\(\s*["'][^"']*hardcorpus\.heldout
    """)


class TestHeldOutIsolation(unittest.TestCase):

    def _offenders(self, root: pathlib.Path):
        bad = []
        for path in sorted(root.rglob("*.py")):
            if path.resolve() in _ALLOWED or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if _IMPORTS.search(text):
                bad.append(str(path.relative_to(_ROOT)))
        return bad

    def test_only_the_scorer_imports_the_held_out_split(self):
        offenders = self._offenders(_SRC) + self._offenders(_TESTS)
        self.assertEqual(
            offenders, [],
            "\n\nTHE HELD-OUT SPLIT HAS BEEN IMPORTED BY SOMETHING THAT IS NOT THE "
            "SCORER.\n\n%s\n\nScore it through harnesscad.eval.hardcorpus.score. "
            "Tune on the dev split (seed 1)." % "\n".join(offenders))

    def test_the_check_would_actually_catch_a_leak(self):
        leaks = [
            "from harnesscad.eval.hardcorpus.heldout import BRIEFS\n",
            "import harnesscad.eval.hardcorpus.heldout\n",
            "from harnesscad.eval.hardcorpus import heldout\n",
            "from harnesscad.eval.hardcorpus import dev, heldout\n",
            "from . import heldout\n",
            "from .heldout import BRIEFS\n",
            "m = importlib.import_module('harnesscad.eval.hardcorpus.heldout')\n",
        ]
        for leak in leaks:
            self.assertRegex(leak, _IMPORTS, "this leak would go undetected")
        prose = ("    See :mod:`harnesscad.eval.hardcorpus.heldout` -- the held-out\n"
                 "    split, which only the scorer may import.\n")
        self.assertIsNone(_IMPORTS.search(prose))

    def test_the_scorer_exists_and_does_import_it(self):
        self.assertTrue(SCORER.is_file())
        self.assertRegex(SCORER.read_text(encoding="utf-8"),
                         r"from harnesscad\.eval\.hardcorpus import .*heldout")

    def test_the_scorer_does_not_hand_back_the_briefs(self):
        from harnesscad.eval.hardcorpus import score

        self.assertFalse(hasattr(score, "BRIEFS"))
        for name in dir(score):
            if name.startswith("_"):
                continue
            self.assertNotIsInstance(getattr(score, name), tuple,
                                     "score.%s is a module-scope tuple; if that is "
                                     "the held-out briefs, it is a leak" % name)


if __name__ == "__main__":
    unittest.main()
