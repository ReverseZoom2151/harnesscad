import unittest

from programs.cqeditor_module_sandbox import (
    newly_added_keys,
    module_sandbox,
    prune_new_modules,
)


class TestNewlyAddedKeys(unittest.TestCase):
    def test_added(self):
        self.assertEqual(
            newly_added_keys({"a": 1}, {"a": 1, "b": 2, "c": 3}), ["b", "c"]
        )

    def test_none_added(self):
        self.assertEqual(newly_added_keys({"a": 1}, {"a": 9}), [])

    def test_sorted_deterministic(self):
        self.assertEqual(
            newly_added_keys(set(), {"z": 1, "a": 1, "m": 1}), ["a", "m", "z"]
        )

    def test_removed_keys_ignored(self):
        self.assertEqual(newly_added_keys({"a": 1, "b": 2}, {"a": 1}), [])


class TestModuleSandbox(unittest.TestCase):
    def test_unloads_added(self):
        reg = {"base": object()}
        with module_sandbox(reg):
            reg["new1"] = object()
            reg["new2"] = object()
        self.assertEqual(set(reg.keys()), {"base"})

    def test_preserves_preexisting(self):
        base_val = object()
        reg = {"base": base_val}
        with module_sandbox(reg):
            reg["tmp"] = object()
        self.assertIs(reg["base"], base_val)

    def test_preserves_replaced_preexisting(self):
        reg = {"base": 1}
        with module_sandbox(reg):
            reg["base"] = 2  # replaced, not added
            reg["tmp"] = 3
        self.assertEqual(reg, {"base": 2})

    def test_yields_removed_list(self):
        reg = {"base": 1}
        with module_sandbox(reg) as added:
            reg["x"] = 1
            reg["y"] = 1
        self.assertEqual(added, ["x", "y"])

    def test_cleanup_on_exception(self):
        reg = {"base": 1}
        with self.assertRaises(RuntimeError):
            with module_sandbox(reg):
                reg["leak"] = 1
                raise RuntimeError("boom")
        self.assertEqual(set(reg.keys()), {"base"})


class TestPruneNewModules(unittest.TestCase):
    def test_prunes_and_reports(self):
        reg = {"a": 1, "b": 2, "c": 3}
        removed = prune_new_modules(reg, {"a"})
        self.assertEqual(removed, ["b", "c"])
        self.assertEqual(reg, {"a": 1})

    def test_nothing_to_prune(self):
        reg = {"a": 1}
        self.assertEqual(prune_new_modules(reg, {"a", "b"}), [])
        self.assertEqual(reg, {"a": 1})


if __name__ == "__main__":
    unittest.main()
