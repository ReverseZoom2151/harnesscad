"""THE ENFORCEMENT. The held-out split is held out because this test says so.

A discipline that lives in a docstring is a discipline that lasts until the first
person in a hurry. This scans every ``.py`` file in the source tree and in the
test tree and FAILS if any module other than ``harnesscad.eval.corpus.score``
names ``harnesscad.eval.corpus.heldout``.

That is the mitigation the audit prescribed for benchmark contamination (14.8.1)
and for Goodhart's Law (14.8.3) -- "maintain a private test set" -- and which the
pressure experiment did not have. One corpus, 28 briefs, one hand, no held-out
set, and therefore no instrument left that could tell "the harness got better"
from "the harness learned this corpus".
"""

from __future__ import annotations

import pathlib
import re
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src" / "harnesscad"
_TESTS = _ROOT / "tests"

#: The one module allowed to reach the held-out briefs.
SCORER = _SRC / "eval" / "corpus" / "score.py"

#: The held-out module itself, and this test (which must name it to check for it).
_ALLOWED = {
    SCORER.resolve(),
    (_SRC / "eval" / "corpus" / "heldout.py").resolve(),
    pathlib.Path(__file__).resolve(),
}

#: The corpus package directory. A RELATIVE import (``from . import heldout``)
#: names ``heldout`` in the importing file's OWN package, so it is only a leak of
#: THIS held-out split when the file lives here. The same text in
#: ``tests/eval/hardcorpus/`` refers to the hardcorpus split, not this one.
_CORPUS_PKG = (_SRC / "eval" / "corpus").resolve()

#: The ABSOLUTE, corpus-qualified ways of importing the held-out split. These name
#: ``harnesscad.eval.corpus.heldout`` explicitly, so they are a leak from anywhere
#: and are checked in every file.
_IMPORTS_ABS = re.compile(
    r"""(?mx)
      ^\s*from\s+harnesscad\.eval\.corpus\.heldout\s+import\b
    | ^\s*import\s+harnesscad\.eval\.corpus\.heldout\b
    | ^\s*from\s+harnesscad\.eval\.corpus\s+import\s+(?:[^\n]*[,(]\s*)?heldout\b
    | (?:importlib\.import_module|__import__)\s*\(\s*["'][^"']*\bcorpus\.heldout
    """)

#: The full check -- absolute forms plus the RELATIVE forms. The relative forms are
#: package-agnostic, so they are only applied to files inside the corpus package
#: (see ``_offenders``); this regex, matching either, is what the leak self-test
#: below asserts against. Prose that merely names the module is not a leak, and a
#: check that flagged it would be turned off within the week.
_IMPORTS = re.compile(
    r"""(?mx)
      ^\s*from\s+harnesscad\.eval\.corpus\.heldout\s+import\b
    | ^\s*import\s+harnesscad\.eval\.corpus\.heldout\b
    | ^\s*from\s+harnesscad\.eval\.corpus\s+import\s+(?:[^\n]*[,(]\s*)?heldout\b
    | ^\s*from\s+\.\s*import\s+(?:[^\n]*[,(]\s*)?heldout\b
    | ^\s*from\s+\.heldout\s+import\b
    | (?:importlib\.import_module|__import__)\s*\(\s*["'][^"']*\bcorpus\.heldout
    """)


class TestHeldOutIsolation(unittest.TestCase):

    def _offenders(self, root: pathlib.Path):
        bad = []
        for path in sorted(root.rglob("*.py")):
            resolved = path.resolve()
            if resolved in _ALLOWED or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            # A relative ``from . import heldout`` only leaks THIS split when the
            # file is in the corpus package; elsewhere it names another package's
            # heldout (the hardcorpus isolation test carries such a string as a
            # fixture). So the full check runs inside the package, the absolute-only
            # check everywhere else.
            in_corpus_pkg = _CORPUS_PKG in resolved.parents
            pattern = _IMPORTS if in_corpus_pkg else _IMPORTS_ABS
            if pattern.search(text):
                bad.append(str(path.relative_to(_ROOT)))
        return bad

    def test_only_the_scorer_imports_the_held_out_split(self):
        offenders = self._offenders(_SRC) + self._offenders(_TESTS)
        self.assertEqual(
            offenders, [],
            "\n\nTHE HELD-OUT SPLIT HAS BEEN IMPORTED BY SOMETHING THAT IS NOT "
            "THE SCORER.\n\n%s\n\n"
            "A held-out set stops being held out the moment a second module can "
            "read it, because the next debugging session will read it, and the "
            "session after that will fix the code against it -- and then there is "
            "no instrument left that can tell 'the harness got better' from 'the "
            "harness learned this corpus'.\n\n"
            "Score it through harnesscad.eval.corpus.score. Tune on "
            "harnesscad.eval.corpus.dev.\n" % "\n".join(offenders))

    def test_the_check_would_actually_catch_a_leak(self):
        """A guard nobody has ever seen fire is a guard nobody knows is broken.
        These are the ways somebody would really do it."""
        leaks = [
            "from harnesscad.eval.corpus.heldout import BRIEFS\n",
            "import harnesscad.eval.corpus.heldout\n",
            "from harnesscad.eval.corpus import heldout\n",
            "from harnesscad.eval.corpus import dev, heldout\n",
            "from . import heldout\n",
            "from .heldout import BRIEFS\n",
            "m = importlib.import_module('harnesscad.eval.corpus.heldout')\n",
        ]
        for leak in leaks:
            self.assertRegex(leak, _IMPORTS, "this leak would go undetected")
        # ...and prose about the module must NOT trip it, or the check gets
        # switched off the first week.
        prose = ("    See :mod:`harnesscad.eval.corpus.heldout` -- the held-out\n"
                 "    split, which only the scorer may import.\n")
        self.assertIsNone(_IMPORTS.search(prose))

    def test_the_scorer_exists_and_does_import_it(self):
        # The enforcement is worthless if the path it guards is dead.
        self.assertTrue(SCORER.is_file())
        self.assertRegex(SCORER.read_text(encoding="utf-8"),
                         r"from harnesscad\.eval\.corpus import heldout")

    def test_the_scorer_does_not_hand_back_the_briefs(self):
        """It returns SCORES. A module that returned the briefs would be a leak
        with an extra step in it."""
        from harnesscad.eval.corpus import score

        self.assertFalse(hasattr(score, "BRIEFS"))
        for name in dir(score):
            if name.startswith("_"):
                continue
            obj = getattr(score, name)
            self.assertNotIsInstance(
                obj, tuple,
                "score.%s is a tuple at module scope; if that is the held-out "
                "brief list, it is a leak" % name)


if __name__ == "__main__":
    unittest.main()
