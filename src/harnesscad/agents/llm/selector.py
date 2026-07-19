"""Runtime model selector and allowlist policy for LLM provider/model choice.

The LLMSelector dataclass and parse/split helpers use a frozen data model, and
allowlist enforcement rejects a provider outside LLM_ALLOWED_PROVIDERS while
requiring the default or fallback model when no per-provider allowlist exists.

Gap filled: harnesscad.agents.llm.base defines the vendor-neutral LLM protocol
and harnesscad.agents.llm.litellm_backend talks to providers, but nothing in
the harness governs WHICH provider/model a run may select at runtime -- any
string a caller (or a model-suggested config) passes through goes straight to
the backend. This module is that missing policy gate: a deterministic,
injectable SelectorPolicy that resolves user requests against defaults and
allowlists before any client is constructed. It complements (and never
duplicates) harnesscad.agents.llm.base: base carries the conversation
vocabulary, this module only decides the (provider, model) pair.

Deterministic: resolve() never reads the environment or the clock; the single
env touchpoint is the explicit SelectorPolicy.from_env() constructor.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "LLMSelector",
    "parse_llm_selector",
    "split_llm_selector",
    "SelectorPolicy",
]


# --- selector value object ----------------------------------------------------
@dataclass(frozen=True)
class LLMSelector:
    """An immutable (provider, model) pair, printable as 'provider/model'."""

    provider: str
    model: str

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"

    def as_tuple(self) -> Tuple[str, str]:
        return self.provider, self.model


def parse_llm_selector(value: Optional[str]) -> Optional[LLMSelector]:
    """Parse a runtime selector formatted as provider/model."""
    if value is None:
        return None

    provider, separator, model = value.strip().partition("/")
    if not separator or not provider.strip() or not model.strip():
        raise ValueError(
            "LLM selector must look like provider/model, for example openai/gpt-5.5."
        )

    return LLMSelector(provider=provider.strip(), model=model.strip())


def split_llm_selector(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Split a 'provider/model' string into its parts, (None, None) for None."""
    selector = parse_llm_selector(value)
    if selector is None:
        return None, None
    return selector.provider, selector.model


# --- policy ---------------------------------------------------------------------
def _env_model_var(provider: str) -> str:
    return "%s_ALLOWED_MODELS" % provider.upper().replace("-", "_")


@dataclass
class SelectorPolicy:
    """Defaults plus allowlists governing runtime provider/model selection.

    Rules:
      - a missing provider/model falls back to the defaults;
      - when allowed_providers is set, any other provider is rejected with a
        ValueError naming the allowlist variable;
      - when a provider has an entry in allowed_models_by_provider, only those
        models pass; without an explicit allowlist only the default model (and
        fallback_model, if set) is allowed;
      - when strict is False and fallback_model is set, a disallowed model
        resolves to fallback_model instead of raising.
    """

    default_provider: str
    default_model: str
    allowed_providers: Optional[List[str]] = None
    allowed_models_by_provider: Dict[str, List[str]] = field(default_factory=dict)
    strict: bool = True
    fallback_model: Optional[str] = None

    def _allowed_models(self, provider: str) -> List[str]:
        explicit = self.allowed_models_by_provider.get(provider)
        if explicit is not None:
            return list(explicit)
        # The reference's rule: without an explicit allowlist only the
        # default (and fallback) model may be selected.
        implicit = [self.default_model]
        if self.fallback_model and self.fallback_model not in implicit:
            implicit.append(self.fallback_model)
        return implicit

    def resolve(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> LLMSelector:
        """Resolve a (possibly partial) request into an allowed LLMSelector."""
        requested_provider = provider.strip() if isinstance(provider, str) else ""
        resolved_provider = requested_provider or self.default_provider

        if self.allowed_providers is not None and resolved_provider not in self.allowed_providers:
            raise ValueError(
                "Provider '%s' is not allowed for runtime selection. "
                "Set LLM_ALLOWED_PROVIDERS to include it." % resolved_provider
            )

        requested_model = model.strip() if isinstance(model, str) else ""
        resolved_model = requested_model or self.default_model

        allowed_models = self._allowed_models(resolved_provider)
        if resolved_model not in allowed_models:
            if not self.strict and self.fallback_model:
                resolved_model = self.fallback_model
            else:
                raise ValueError(
                    "Model '%s' is not allowed for provider '%s'. "
                    "Set %s to include it."
                    % (resolved_model, resolved_provider, _env_model_var(resolved_provider))
                )

        return LLMSelector(provider=resolved_provider, model=resolved_model)

    @classmethod
    def from_env(cls, environ: Optional[Dict[str, str]] = None) -> "SelectorPolicy":
        """Build a policy from environment variables (the only env touchpoint).

        Reads LLM_PROVIDER, LLM_MODEL, LLM_ALLOWED_PROVIDERS (comma-separated),
        STRICT_LLM, LLM_FALLBACK_MODEL, plus {PROVIDER}_ALLOWED_MODELS
        (comma-separated) for the default provider and every allowed provider.
        Pass `environ` explicitly (e.g. a plain dict) to keep tests hermetic.
        """
        env = os.environ if environ is None else environ

        default_provider = (env.get("LLM_PROVIDER") or "simulation").strip()
        default_model = (env.get("LLM_MODEL") or "simulation").strip()

        allowed_providers: Optional[List[str]] = None
        raw_providers = env.get("LLM_ALLOWED_PROVIDERS")
        if raw_providers is not None:
            allowed_providers = [
                part.strip() for part in raw_providers.split(",") if part.strip()
            ]

        strict = (env.get("STRICT_LLM") or "").strip().lower() in {"1", "true", "yes", "on"}
        fallback_model = (env.get("LLM_FALLBACK_MODEL") or "").strip() or None

        allowed_models_by_provider: Dict[str, List[str]] = {}
        candidates = list(allowed_providers or [])
        if default_provider not in candidates:
            candidates.append(default_provider)
        for candidate in candidates:
            raw_models = env.get(_env_model_var(candidate))
            if raw_models is not None:
                allowed_models_by_provider[candidate] = [
                    part.strip() for part in raw_models.split(",") if part.strip()
                ]

        return cls(
            default_provider=default_provider,
            default_model=default_model,
            allowed_providers=allowed_providers,
            allowed_models_by_provider=allowed_models_by_provider,
            strict=strict,
            fallback_model=fallback_model,
        )


# --- CLI --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` exercises parsing, defaulting, allowlist
    rejection, fallback behavior and the from_env constructor with asserts."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.agents.llm.selector",
        description="Runtime LLM selector parsing and allowlist policy "
        "(provider/model resolution with defaults, allowlists, fallback).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="run deterministic selector/policy checks and exit 0 on success.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    failures: List[str] = []

    def check(label: str, condition: bool) -> None:
        if not condition:
            failures.append(label)

    # 1. Selector parsing round-trip.
    sel = parse_llm_selector("openai/gpt-5.5")
    check(
        "parse selector",
        sel == LLMSelector("openai", "gpt-5.5")
        and sel.key == "openai/gpt-5.5"
        and sel.as_tuple() == ("openai", "gpt-5.5"),
    )
    check("parse None", parse_llm_selector(None) is None)
    check("split selector", split_llm_selector(" a / b ".replace(" ", "")) == ("a", "b"))
    check("split None", split_llm_selector(None) == (None, None))

    # 2. Malformed selectors raise with the reference's message shape.
    for bad in ("openai", "/model", "provider/", "  /  "):
        try:
            parse_llm_selector(bad)
        except ValueError as exc:
            check(
                "error message for %r" % bad,
                str(exc)
                == "LLM selector must look like provider/model, for example openai/gpt-5.5.",
            )
        else:
            failures.append("no error for %r" % bad)

    # 3. Policy defaulting.
    policy = SelectorPolicy(
        default_provider="anthropic",
        default_model="claude-sonnet-4-5",
        allowed_providers=["anthropic", "openai"],
        allowed_models_by_provider={"openai": ["gpt-5.5", "gpt-5.5-mini"]},
        strict=True,
    )
    check(
        "defaults resolve",
        policy.resolve() == LLMSelector("anthropic", "claude-sonnet-4-5"),
    )
    check(
        "explicit allowlisted model",
        policy.resolve("openai", "gpt-5.5-mini") == LLMSelector("openai", "gpt-5.5-mini"),
    )

    # 4. Provider allowlist rejection names the allowlist.
    try:
        policy.resolve("gemini", "gemini-pro")
    except ValueError as exc:
        check(
            "provider rejection message",
            "Provider 'gemini' is not allowed" in str(exc)
            and "LLM_ALLOWED_PROVIDERS" in str(exc),
        )
    else:
        failures.append("provider rejection did not raise")

    # 5. Model rejection: no explicit allowlist means only default/fallback.
    try:
        policy.resolve("anthropic", "claude-haiku-4-5")
    except ValueError as exc:
        check(
            "model rejection message",
            "Model 'claude-haiku-4-5' is not allowed for provider 'anthropic'" in str(exc)
            and "ANTHROPIC_ALLOWED_MODELS" in str(exc),
        )
    else:
        failures.append("model rejection did not raise")

    # 6. Model rejection against an explicit allowlist.
    try:
        policy.resolve("openai", "gpt-4o")
    except ValueError:
        pass
    else:
        failures.append("explicit-allowlist rejection did not raise")

    # 7. Non-strict policy applies the fallback model.
    lenient = SelectorPolicy(
        default_provider="anthropic",
        default_model="claude-sonnet-4-5",
        strict=False,
        fallback_model="claude-haiku-4-5",
    )
    check(
        "fallback applied when not strict",
        lenient.resolve(model="claude-opus-4")
        == LLMSelector("anthropic", "claude-haiku-4-5"),
    )
    # Fallback model itself is implicitly allowed.
    check(
        "fallback model implicitly allowed",
        lenient.resolve(model="claude-haiku-4-5")
        == LLMSelector("anthropic", "claude-haiku-4-5"),
    )

    # 8. from_env on an injected dict (hermetic; no ambient env reads).
    env = {
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-5.5",
        "LLM_ALLOWED_PROVIDERS": "openai, anthropic",
        "STRICT_LLM": "true",
        "OPENAI_ALLOWED_MODELS": "gpt-5.5,gpt-5.5-mini",
    }
    env_policy = SelectorPolicy.from_env(env)
    check(
        "from_env fields",
        env_policy.default_provider == "openai"
        and env_policy.default_model == "gpt-5.5"
        and env_policy.allowed_providers == ["openai", "anthropic"]
        and env_policy.strict is True
        and env_policy.allowed_models_by_provider == {"openai": ["gpt-5.5", "gpt-5.5-mini"]},
    )
    check(
        "from_env resolves",
        env_policy.resolve(None, "gpt-5.5-mini") == LLMSelector("openai", "gpt-5.5-mini"),
    )
    try:
        env_policy.resolve("anthropic", "some-unlisted-model")
    except ValueError:
        pass
    else:
        failures.append("from_env allowlist rejection did not raise")

    if failures:
        print("SELFCHECK FAILED: %s" % ", ".join(failures), file=sys.stderr)
        return 1
    print("PASS: selector selfcheck (parsing, defaults, allowlists, fallback, from_env)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
