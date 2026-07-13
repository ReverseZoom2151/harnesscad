"""Guard against test files that the canonical runner silently ignores.

The project runs `python -m unittest`, which collects only `unittest.TestCase`
subclasses. A file of bare module-level `def test_*` functions (pytest style)
imports cleanly, reports `NO TESTS RAN`, and its assertions never execute. Seven
such files sat in this suite undetected; this test makes the next one fail loudly.

Bare `assert` in a non-collected file is doubly unsafe: `python -O` strips it, so
even a pytest run would be vacuous.
"""

import ast
import pathlib
import unittest

TESTS_DIR = pathlib.Path(__file__).resolve().parent


def _test_modules():
    # rglob, not glob: the suite mirrors src/ across ~99 sub-packages, so a
    # flat glob would silently check 2 files instead of ~892 -- exactly the
    # blind spot this guard exists to prevent.
    return sorted(p for p in TESTS_DIR.rglob("test_*.py") if p.name != pathlib.Path(__file__).name)


def _defines_testcase(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", None)
            if name == "TestCase":
                return True
    return False


def _module_level_test_functions(tree):
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


class SuiteCollectableTest(unittest.TestCase):
    def test_every_test_file_defines_a_testcase(self):
        offenders = []
        for path in _test_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if not _defines_testcase(tree):
                offenders.append(path.name)
        self.assertEqual(
            offenders,
            [],
            "test files with no unittest.TestCase are never collected by "
            "`python -m unittest` and their assertions never run: %s" % (offenders,),
        )

    def test_no_bare_module_level_test_functions(self):
        offenders = {}
        for path in _test_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            funcs = _module_level_test_functions(tree)
            if funcs:
                offenders[path.name] = funcs
        self.assertEqual(
            offenders,
            {},
            "module-level `def test_*` functions are pytest-style and are not "
            "collected by the canonical runner; move them onto a TestCase: %s"
            % (offenders,),
        )


if __name__ == "__main__":
    unittest.main()
