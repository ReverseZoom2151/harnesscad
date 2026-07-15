"""The trainer must import on a core-only machine and refuse loudly, not crash.

The whole point of the ``train`` subpackage being a subpackage is that the core
suite runs without torch. These tests assert exactly that: the package imports, the
capability probe is honest, and ``require()`` raises an actionable error rather than
an ImportError deep in a call stack when the stack is absent.
"""

from __future__ import annotations

import unittest

from harnesscad.agents.selftrain import train


class TestGuard(unittest.TestCase):

    def test_package_imports_without_torch(self):
        # Importing the package must never require the heavy stack. The booleans
        # tell the truth about what is installed.
        self.assertIsInstance(train.HAS_TORCH, bool)
        self.assertIsInstance(train.MISSING, list)

    def test_require_matches_probe(self):
        if train.MISSING:
            with self.assertRaises(RuntimeError):
                train.require()
            self.assertIn("harnesscad[train]", train.why_unavailable())
        else:
            train.require()  # must not raise when everything is present
            self.assertEqual(train.why_unavailable(), "")

    def test_submodules_import_without_gpu(self):
        # The trainer modules must be importable for their pure-python helpers even
        # when torch is missing; only the train_* entry points touch the GPU.
        from harnesscad.agents.selftrain.train import sft, kto, evaluate, generate, data
        self.assertTrue(hasattr(sft, "train_sft"))
        self.assertTrue(hasattr(kto, "train_kto"))
        self.assertTrue(hasattr(evaluate, "grade_solver"))
        self.assertTrue(hasattr(generate, "sample_and_certify"))
        self.assertTrue(hasattr(data, "sft_dataset"))


if __name__ == "__main__":
    unittest.main()
