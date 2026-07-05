"""Tests for the end-to-end build pipeline (`pipeline.build`) and the CLI `build`.

Everything runs with a MockLLM (canned CISP-op JSON for a simple plate) + a
StubBackend, so there is NO API key and NO network involved. When cadquery is
importable the cadquery path is exercised too; otherwise it transparently falls
back to the stub.
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import cli
import pipeline
from pipeline import build, BuildError
from llm.base import CompletionResult, ToolCall

from tests.test_llm import MockLLM, plate_ops_json


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:
        return False


class TestBuildWithStub(unittest.TestCase):
    def test_build_stub_reaches_ok(self):
        result = build("make a 20x10x5 plate",
                       llm=MockLLM([plate_ops_json()]), backend="stub")
        self.assertTrue(result["ok"])
        self.assertGreater(result["applied"], 0)
        self.assertEqual(result["applied"], 7)
        self.assertTrue(result["digest"])
        self.assertEqual(result["backend"], "stub")
        # An ok build carries no ERROR-severity diagnostics (warnings are fine).
        errors = [d for d in result["diagnostics"] if d.get("severity") == "error"]
        self.assertEqual(errors, [])

    def test_build_stub_summary_is_sensible(self):
        result = build("make a 20x10x5 plate",
                       llm=MockLLM([plate_ops_json()]), backend="stub")
        summary = result["summary"]
        self.assertTrue(summary["solid_present"])
        self.assertEqual(summary["feature_count"], 1)
        self.assertEqual(summary["sketch_count"], 1)

    def test_build_stub_exports_step_text(self):
        result = build("make a 20x10x5 plate",
                       llm=MockLLM([plate_ops_json()]), backend="stub")
        self.assertIsInstance(result["step"], str)
        self.assertIn("stub-step", result["step"])

    def test_build_result_has_all_keys(self):
        result = build("make a plate", llm=MockLLM([plate_ops_json()]), backend="stub")
        for key in ("ok", "applied", "digest", "diagnostics", "summary",
                    "step", "backend", "backend_note"):
            self.assertIn(key, result)

    def test_build_recovers_from_bad_first_plan(self):
        # First plan references a missing sketch -> block-and-correct; second is good.
        bad = json.dumps([{"op": "extrude", "sketch": "nope", "distance": 5.0}])
        result = build("make a plate",
                       llm=MockLLM([bad, plate_ops_json()]), backend="stub", max_iters=5)
        self.assertTrue(result["ok"])

    def test_build_reports_failure_when_never_converges(self):
        bad = json.dumps([{"op": "extrude", "sketch": "nope", "distance": 5.0}])
        result = build("make a plate",
                       llm=MockLLM([bad] * 10), backend="stub", max_iters=2)
        self.assertFalse(result["ok"])
        # A failing build still exports nothing.
        self.assertIsNone(result["step"])

    def test_build_accepts_tool_call_llm(self):
        tc = ToolCall("emit_ops", plate_ops_json())
        result = build("make a plate",
                       llm=MockLLM([CompletionResult(tool_calls=[tc])]), backend="stub")
        self.assertTrue(result["ok"])


class TestBuildBackendSelection(unittest.TestCase):
    def test_build_default_backend_gives_sensible_summary(self):
        # backend defaults to cadquery; falls back to stub when unavailable.
        result = build("make a 20x10x5 plate", llm=MockLLM([plate_ops_json()]))
        self.assertTrue(result["ok"])
        self.assertIn(result["backend"], ("cadquery", "stub"))
        self.assertTrue(result["summary"]["solid_present"])
        if result["backend"] == "cadquery":
            # Real geometry: STEP text is a genuine ISO-10303 file.
            self.assertIsInstance(result["step"], str)
            self.assertIn("ISO-10303", result["step"])

    @unittest.skipUnless(_cadquery_available(), "cadquery not installed")
    def test_build_cadquery_produces_real_step(self):
        result = build("make a 20x10x5 plate",
                       llm=MockLLM([plate_ops_json()]), backend="cadquery")
        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "cadquery")
        self.assertIn("ISO-10303", result["step"])


class TestBuildErrors(unittest.TestCase):
    def test_no_llm_and_no_key_raises_builderror(self):
        saved = {k: os.environ.pop(k, None) for k in pipeline._API_KEY_ENV_VARS}
        try:
            with self.assertRaises(BuildError) as ctx:
                build("make a plate", backend="stub")
            msg = str(ctx.exception)
            self.assertIn("ANTHROPIC_API_KEY", msg)
            self.assertIn("OPENAI_API_KEY", msg)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_no_llm_but_key_present_builds_lazy_client(self):
        # With a key set but no llm, a lazy client is constructed (never called
        # here because we do not run the planner against it).
        os.environ["ANTHROPIC_API_KEY"] = "test-not-a-real-key"
        try:
            llm = pipeline._resolve_llm(None, None)
            self.assertIsInstance(llm, pipeline._LazyLiteLLM)
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)


class TestCliBuild(unittest.TestCase):
    def _run_with_mock(self, argv):
        """Invoke cli.main with pipeline.build monkeypatched to inject a MockLLM."""
        real_build = pipeline.build

        def patched(brief, **kwargs):
            kwargs.pop("model", None)
            kwargs["llm"] = MockLLM([plate_ops_json()])
            return real_build(brief, **kwargs)

        pipeline.build = patched
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                code = cli.main(argv)
        finally:
            pipeline.build = real_build
        return code, buf.getvalue()

    def test_cli_build_stub_exits_zero_and_ok(self):
        code, out = self._run_with_mock(["build", "make a 20x10x5 plate", "--backend", "stub"])
        self.assertEqual(code, 0)
        self.assertIn("ok:       True", out)
        self.assertIn("digest:", out)
        self.assertIn("solid_present", out)

    def test_cli_build_writes_step_to_out(self):
        tmpdir = tempfile.mkdtemp()
        out_path = os.path.join(tmpdir, "part.step")
        code, out = self._run_with_mock(
            ["build", "make a plate", "--backend", "stub", "--out", out_path])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(out_path))
        with open(out_path, "r", encoding="utf-8") as fh:
            self.assertTrue(fh.read().strip())
        self.assertIn(f"wrote:    {out_path}", out)

    def test_cli_build_missing_key_exits_two(self):
        # No llm injected and no API key -> BuildError -> exit code 2.
        saved = {k: os.environ.pop(k, None) for k in pipeline._API_KEY_ENV_VARS}
        try:
            code = cli.main(["build", "make a plate", "--backend", "stub"])
            self.assertEqual(code, 2)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
