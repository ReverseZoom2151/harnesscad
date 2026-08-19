"""Any model, any provider -- and the honesty that has to come with it.

The model seam went through litellm from the start, so the only thing pinning
this repo's experiments to local models was a hardcoded ``ollama/`` prefix. These
tests pin the routing rule, the back-compat guarantee for the six call sites that
construct the client, and -- most importantly -- the two things that could break
quietly:

* the completion CACHE KEY must not move for a local tag, or every cached
  completion in every existing sweep is silently orphaned;
* ollama's ``api_base`` must never be sent to a hosted provider, which would
  either 404 or silently retarget the call.

No network, no ollama: ``litellm`` is replaced by a recording double, so these
tests assert the exact kwargs the seam would send.
"""

from __future__ import annotations

import sys
import types
import unittest

from harnesscad.eval.pressure import model as m
from harnesscad.eval.pressure.cache import cache_key


class _FakeResponse:
    def __init__(self, text):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=text))]


class _FakeLiteLLM(types.ModuleType):
    """Records the kwargs of the last completion() call."""

    def __init__(self):
        super().__init__("litellm")
        self.calls = []
        self.suppress_debug_info = False
        self.drop_params = False

    def completion(self, **kwargs):
        self.calls.append(dict(kwargs))
        return _FakeResponse("[]")


class _WithFakeLiteLLM(unittest.TestCase):
    def setUp(self):
        self.fake = _FakeLiteLLM()
        self._saved = sys.modules.get("litellm")
        sys.modules["litellm"] = self.fake

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("litellm", None)
        else:
            sys.modules["litellm"] = self._saved

    def _complete(self, name, **kw):
        m.ModelClient(name, **kw).complete([{"role": "user", "content": "hi"}], attempt=0)
        return self.fake.calls[-1]


class ProviderRouting(unittest.TestCase):
    def test_a_bare_tag_is_an_ollama_model(self):
        for tag in ("ornith:9b", "qwen3.6:27b", "llama3"):
            self.assertEqual(m.qualified_model(tag), "ollama/" + tag)
            self.assertTrue(m.is_local(tag))

    def test_a_qualified_name_passes_through_untouched(self):
        for name in ("anthropic/claude-opus-4-1", "openai/gpt-5.2",
                     "gemini/gemini-2.5-pro", "deepseek/deepseek-chat"):
            self.assertEqual(m.qualified_model(name), name)
            self.assertFalse(m.is_local(name))

    def test_a_nested_provider_path_survives(self):
        # OpenRouter names carry a second slash; the rule must not mangle them.
        name = "openrouter/meta-llama/llama-3.1-70b-instruct"
        self.assertEqual(m.qualified_model(name), name)
        self.assertFalse(m.is_local(name))

    def test_a_namespaced_local_tag_can_be_forced(self):
        # The one ambiguous case, and its documented escape hatch.
        self.assertTrue(m.is_local("ollama/library/llama3"))
        self.assertEqual(m.qualified_model("ollama/library/llama3"),
                         "ollama/library/llama3")

    def test_the_rule_reads_the_head_not_the_tag(self):
        # A colon-bearing tag has no slash BEFORE the colon, so it stays local.
        self.assertFalse(m.is_provider_qualified("qwen3.6:27b"))
        self.assertTrue(m.is_provider_qualified("openai/gpt-5.2"))


class SeedHonesty(unittest.TestCase):
    """Determinism is claimed by this repo, so where it is lost must be visible."""

    def test_local_and_openai_honour_a_seed(self):
        self.assertTrue(m.seed_is_honoured("ornith:9b"))
        self.assertTrue(m.seed_is_honoured("openai/gpt-5.2"))

    def test_providers_that_ignore_seed_say_so(self):
        for name in ("anthropic/claude-opus-4-1", "gemini/gemini-2.5-pro"):
            self.assertFalse(
                m.seed_is_honoured(name),
                "a provider that ignores `seed` must not be reported as reproducible")


class BackCompat(unittest.TestCase):
    def test_ollama_client_is_still_constructible(self):
        # Six modules construct OllamaClient; none of them were changed.
        self.assertIs(m.OllamaClient, m.ModelClient)
        c = m.OllamaClient("ornith:9b", seed=7, temperature=0.0)
        self.assertEqual(c.name, "ornith:9b")
        self.assertTrue(c.is_local)

    def test_the_cache_key_of_a_local_tag_did_not_move(self):
        # THE regression that would be invisible: if the client started keying on
        # the litellm-qualified name, every completion in every existing sweep is
        # orphaned and the models get silently re-billed. The two names DO key
        # differently, so the guarantee is precisely that `name` stays bare.
        msgs = [{"role": "user", "content": "a 60x40x5 plate"}]
        self.assertNotEqual(
            cache_key("ornith:9b", 1, 0.0, 0, msgs),
            cache_key("ollama/ornith:9b", 1, 0.0, 0, msgs),
            "sanity: the bare and qualified names are not interchangeable keys")
        self.assertEqual(
            m.ModelClient("ornith:9b").name, "ornith:9b",
            "CachedClient keys on client.name; it must stay the bare tag")


class CallShape(_WithFakeLiteLLM):
    def test_a_local_call_carries_the_ollama_api_base(self):
        kw = self._complete("ornith:9b", api_base="http://localhost:11434")
        self.assertEqual(kw["model"], "ollama/ornith:9b")
        self.assertEqual(kw["api_base"], "http://localhost:11434")

    def test_a_hosted_call_never_carries_an_api_base(self):
        kw = self._complete("anthropic/claude-opus-4-1")
        self.assertEqual(kw["model"], "anthropic/claude-opus-4-1")
        self.assertNotIn(
            "api_base", kw,
            "sending ollama's localhost base to a hosted provider would 404 or "
            "silently retarget the call")

    def test_unsupported_params_are_dropped_only_for_hosted_providers(self):
        self._complete("ornith:9b")
        self.assertFalse(self.fake.drop_params,
                         "a local call must still fail loudly on a bad param")
        self._complete("anthropic/claude-opus-4-1")
        self.assertTrue(self.fake.drop_params)

    def test_seed_and_temperature_are_always_sent(self):
        kw = self._complete("openai/gpt-5.2", seed=42, temperature=0.0)
        self.assertEqual(kw["seed"], 42)
        self.assertEqual(kw["temperature"], 0.0)

    def test_per_call_overrides_win(self):
        c = m.ModelClient("openai/gpt-5.2", seed=1, temperature=0.0)
        c.complete([{"role": "user", "content": "x"}], attempt=0, seed=9, temperature=0.7)
        kw = self.fake.calls[-1]
        self.assertEqual(kw["seed"], 9)
        self.assertEqual(kw["temperature"], 0.7)


if __name__ == "__main__":
    unittest.main()
