"""Tests for the coding-agent plugin surface.

Two things are under test. First, that generation still tracks the real CLI --
the manifest's whole claim is that it cannot advertise a verb that does not
exist. Second, and the reason these tests were written, that generated and
hand-authored files coexist: the generator writes one skill directory and one
command file per verb, and a hand-authored skill in a sibling directory must
survive regeneration byte for byte. Before the marker guard existed the write
was unconditional, so a hand-authored file on a generated path was destroyed
silently; that case is pinned here as a refusal.

The checked-in tree under ``plugins/`` is also verified against a fresh
generation, so drift between the repo and the CLI fails a test rather than
waiting to be noticed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harnesscad.io.surfaces import plugin_manifest as pm

REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = REPO_ROOT / "plugins" / pm.PLUGIN_NAME


class TestIntrospection(unittest.TestCase):
    def test_specs_come_from_the_live_parser(self) -> None:
        specs = pm.command_specs()
        names = [s.name for s in specs]
        self.assertEqual(names, sorted(names))
        self.assertGreaterEqual(len(specs), 10)
        for verb in ("apply", "build", "export", "render", "pdd", "selftest"):
            self.assertIn(verb, names)

    def test_a_verb_the_cli_lacks_is_not_advertised(self) -> None:
        # 'verify' is a flag on apply (--verify core|full), not a verb. If it
        # ever appears here the manifest has stopped mirroring the parser.
        self.assertNotIn("verify", [s.name for s in pm.command_specs()])

    def test_apply_arguments_are_introspected_not_hardcoded(self) -> None:
        spec = next(s for s in pm.command_specs() if s.name == "apply")
        self.assertTrue(any(a.positional and a.name == "ops" for a in spec.args))
        backend = next(a for a in spec.args if a.flags == ["--backend"])
        self.assertIn("cadquery", backend.choices)
        self.assertIn("frep", backend.choices)


class TestGeneratedMarkdown(unittest.TestCase):
    def test_command_file_has_frontmatter_and_marker(self) -> None:
        spec = next(s for s in pm.command_specs() if s.name == "apply")
        md = pm.command_markdown(spec)
        self.assertTrue(md.startswith("---\nname: apply\n"))
        self.assertIn("## Usage", md)
        self.assertIn(pm.GENERATED_MARKER, md)

    def test_skill_file_has_frontmatter_and_marker(self) -> None:
        md = pm.skill_markdown(pm.command_specs())
        self.assertTrue(md.startswith("---\nname: %s\n" % pm.PLUGIN_NAME))
        self.assertIn("## Use this skill when", md)
        self.assertIn(pm.GENERATED_MARKER, md)

    def test_generated_skill_points_at_the_hand_authored_ones(self) -> None:
        # The generated index says what exists; the authored skills say when
        # to use it. If the index stops mentioning them they are undiscoverable.
        self.assertIn("Task-shaped skills", pm.skill_markdown(pm.command_specs()))


class TestWriteSet(unittest.TestCase):
    def test_generated_relpaths_predicts_the_write_set(self) -> None:
        specs = pm.command_specs()
        with tempfile.TemporaryDirectory() as tmp:
            written = pm.write_plugin_tree(Path(tmp), version="0.0.0-test")
            rels = sorted(str(p.relative_to(tmp)).replace("\\", "/")
                          for p in written)
        self.assertEqual(rels, pm.generated_relpaths(specs))

    def test_regeneration_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = pm.write_plugin_tree(Path(tmp), version="0.0.0-test")
            before = {p: p.read_bytes() for p in first}
            second = pm.write_plugin_tree(Path(tmp), version="0.0.0-test")
            self.assertEqual(sorted(second), sorted(first))
            for path in second:
                self.assertEqual(path.read_bytes(), before[path], str(path))


class TestCoexistence(unittest.TestCase):
    """Hand-authored skills must survive generation. This is the whole point."""

    def _authored(self, root: Path, name: str) -> Path:
        skill_dir = root / f"plugins/{pm.PLUGIN_NAME}/skills/{name}"
        skill_dir.mkdir(parents=True, exist_ok=True)
        path = skill_dir / "SKILL.md"
        path.write_text(
            "---\nname: %s\ndescription: hand written\n---\n\nbody\n" % name,
            encoding="utf-8",
        )
        return path

    def test_sibling_skill_survives_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pm.write_plugin_tree(root, version="0.0.0-test")
            authored = self._authored(root, "cad-repair")
            original = authored.read_bytes()
            pm.write_plugin_tree(root, version="0.0.0-test")
            self.assertEqual(authored.read_bytes(), original)

    def test_sibling_skill_is_not_in_the_write_set(self) -> None:
        rels = pm.generated_relpaths()
        skill_paths = [r for r in rels if "/skills/" in r]
        self.assertEqual(
            skill_paths,
            [f"plugins/{pm.PLUGIN_NAME}/skills/{pm.PLUGIN_NAME}/SKILL.md"],
        )

    def test_hand_authored_skills_lists_only_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pm.write_plugin_tree(root, version="0.0.0-test")
            self.assertEqual(pm.hand_authored_skills(root), [])
            self._authored(root, "cad-pdd")
            self._authored(root, "cad-op-streams")
            self.assertEqual(pm.hand_authored_skills(root),
                             ["cad-op-streams", "cad-pdd"])

    def test_unmarked_file_on_a_generated_path_is_refused_not_clobbered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            victim = root / f"plugins/{pm.PLUGIN_NAME}/skills/{pm.PLUGIN_NAME}/SKILL.md"
            victim.parent.mkdir(parents=True, exist_ok=True)
            victim.write_text("---\nname: mine\n---\n\nprecious\n", encoding="utf-8")
            with self.assertRaises(pm.GeneratedFileConflict):
                pm.write_plugin_tree(root, version="0.0.0-test")
            self.assertIn("precious", victim.read_text(encoding="utf-8"))


class TestCheckedInTree(unittest.TestCase):
    def test_repo_tree_matches_a_fresh_generation(self) -> None:
        specs = pm.command_specs()
        with tempfile.TemporaryDirectory() as tmp:
            written = pm.write_plugin_tree(Path(tmp))
            for path in written:
                rel = str(path.relative_to(tmp)).replace("\\", "/")
                checked_in = REPO_ROOT / rel
                self.assertTrue(checked_in.is_file(), f"{rel} is not checked in")
                self.assertEqual(
                    checked_in.read_text(encoding="utf-8"),
                    path.read_text(encoding="utf-8"),
                    f"{rel} has drifted from the CLI; regenerate with "
                    f"`python -m harnesscad.io.surfaces.plugin_manifest --out .`",
                )
        manifest = json.loads(
            (PLUGIN_ROOT / ".claude-plugin" / "plugin.json").read_text(
                encoding="utf-8"))
        self.assertEqual(manifest["metadata"]["command_count"], len(specs))

    def test_no_stale_command_files(self) -> None:
        # Generation never deletes, so a verb removed from the CLI would leave
        # its command file behind advertising something that no longer exists.
        on_disk = sorted(p.stem for p in (PLUGIN_ROOT / "commands").glob("*.md"))
        self.assertEqual(on_disk, sorted(s.name for s in pm.command_specs()))


class TestAuthoredSkills(unittest.TestCase):
    """The hand-authored suite itself: well-formed and actually installed."""

    EXPECTED = [
        "cad-brief-to-part",
        "cad-gate-verdicts",
        "cad-op-streams",
        "cad-pdd",
        "cad-repair",
    ]

    def test_the_suite_is_present(self) -> None:
        self.assertEqual(pm.hand_authored_skills(REPO_ROOT), self.EXPECTED)

    def test_each_skill_has_valid_frontmatter(self) -> None:
        for name in self.EXPECTED:
            path = PLUGIN_ROOT / "skills" / name / "SKILL.md"
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\n"), name)
            _, frontmatter, _ = text.split("---\n", 2)
            fields = dict(
                line.split(":", 1) for line in frontmatter.strip().splitlines()
            )
            self.assertEqual(fields["name"].strip(), name)
            # The description is the only thing always in context, so it is
            # what decides whether the skill ever fires. An empty or generic
            # one makes the body unreachable.
            description = fields["description"].strip()
            self.assertGreater(len(description), 80, name)
            # A description that only says what the skill does never fires.
            # Each of ours must name the situations that should trigger it.
            self.assertRegex(description, r"Use (when|after)", name)

    def test_no_skill_carries_the_generated_marker(self) -> None:
        # A hand-authored file carrying the marker would be silently
        # overwritable, which is exactly the guarantee these skills rely on.
        for name in self.EXPECTED:
            text = (PLUGIN_ROOT / "skills" / name / "SKILL.md").read_text(
                encoding="utf-8")
            self.assertNotIn(pm.GENERATED_MARKER, text, name)

    def test_bodies_stay_within_the_progressive_disclosure_budget(self) -> None:
        # A SKILL.md body is loaded whole on invoke; detail belongs in
        # references/, which is read only when the body says to.
        for name in self.EXPECTED:
            path = PLUGIN_ROOT / "skills" / name / "SKILL.md"
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertLess(len(lines), 200, f"{name} is {len(lines)} lines")

    def test_references_are_reachable_from_their_body(self) -> None:
        for name in self.EXPECTED:
            refs = PLUGIN_ROOT / "skills" / name / "references"
            if not refs.is_dir():
                continue
            body = (PLUGIN_ROOT / "skills" / name / "SKILL.md").read_text(
                encoding="utf-8")
            for ref in sorted(refs.glob("*.md")):
                self.assertIn(ref.name, body,
                              f"{name}/references/{ref.name} is never referenced")
                self.assertLess(len(ref.read_text(encoding="utf-8").splitlines()),
                                300, ref.name)

    def test_op_vocabulary_reference_matches_the_real_registry(self) -> None:
        # The reference is generated from the parser's own registry; if an op
        # is added or renamed and the file is not regenerated, a skill would
        # be teaching a vocabulary the harness does not accept.
        from harnesscad.core.cisp.ops import _REGISTRY

        text = (PLUGIN_ROOT / "skills" / "cad-op-streams" / "references"
                / "op-vocabulary.md").read_text(encoding="utf-8")
        for tag in _REGISTRY:
            self.assertIn("`%s`" % tag, text, tag)
        self.assertIn("Total: %d ops." % len(_REGISTRY), text)


if __name__ == "__main__":
    unittest.main()
