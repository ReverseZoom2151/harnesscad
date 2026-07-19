"""Multi-provider resilient routing with failure cooldown.

A provider cascade tries each configured backend in order until
one answers, with the two behaviours that make the cascade *stable* rather
than merely retrying:

  * **failure cooldown** -- a provider that fails with a billing / quota /
    auth / rate-limit error is benched for a cooldown window (60 s in
    the configured cooldown), so a dead API is not hammered on every subsequent call; other
    failures (transient network, empty response) are retried next call;
  * **error classification** -- only errors matching the exhaustion markers
    ("credit balance", "quota", "rate_limit", 401/403/429) trigger the bench;
  * **preferred-first ordering** -- a caller can promote one provider to the
    front without changing the standing order;
  * **provider attribution** -- the result carries which provider answered,
    so downstream traces can display and account per provider.

The router is deliberately transport-agnostic: providers are injected as
plain callables, the clock is injected for determinism, and nothing here
imports a vendor SDK -- the same seam discipline as
:mod:`harnesscad.agents.llm.base`. A ``LiteLLMClient`` or any ``LLM``
implementation can be wrapped as a provider callable in one line.

stdlib-only, deterministic given injected clock and providers.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ProviderError",
    "AllProvidersFailedError",
    "RoutedResult",
    "is_exhaustion_error",
    "ResilientRouter",
    "main",
]

ProviderFn = Callable[..., Any]
Clock = Callable[[], float]

DEFAULT_COOLDOWN_S = 60.0

_EXHAUSTION_MARKERS = (
    "credit balance", "insufficient", "quota", "rate_limit", "rate limit",
    "401", "403", "429",
)


class ProviderError(RuntimeError):
    """A provider call failed; message is inspected for exhaustion markers."""


class AllProvidersFailedError(RuntimeError):
    """Every configured provider failed or was benched."""

    def __init__(self, purpose: str, errors: Mapping[str, str]):
        self.errors = dict(errors)
        detail = "; ".join(f"{name}: {msg[:80]}" for name, msg in errors.items())
        super().__init__(f"all providers failed for {purpose}: {detail}")


def is_exhaustion_error(message: str) -> bool:
    """Should this failure bench the provider for the cooldown window."""
    lower = message.lower()
    if any(marker in lower for marker in _EXHAUSTION_MARKERS if not marker.isdigit()):
        return True
    return bool(re.search(r"\b(?:401|403|429)\b", message))


@dataclass(frozen=True)
class RoutedResult:
    """A successful answer plus which provider produced it."""

    value: Any
    provider: str
    attempts: Tuple[str, ...]        # providers tried, in order, incl. winner
    skipped: Tuple[str, ...]         # providers skipped while benched


@dataclass
class _ProviderState:
    fn: ProviderFn
    benched_at: Optional[float] = None
    failures: int = 0
    successes: int = 0


class ResilientRouter:
    """Ordered provider cascade with per-provider failure cooldown.

    Providers are registered in priority order. ``call`` walks the order,
    skipping benched providers, and returns the first non-empty answer as a
    :class:`RoutedResult`. A provider returning ``None`` or an empty string is
    treated as a failure. Exhaustion-class failures bench the provider until
    ``cooldown_s`` elapses on the injected clock.
    """

    def __init__(
        self,
        providers: Sequence[Tuple[str, ProviderFn]],
        *,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        clock: Clock = time.monotonic,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        names = [name for name, _ in providers]
        if len(set(names)) != len(names):
            raise ValueError("provider names must be unique")
        self._order: List[str] = names
        self._states: Dict[str, _ProviderState] = {
            name: _ProviderState(fn) for name, fn in providers
        }
        self.cooldown_s = cooldown_s
        self._clock = clock

    # ----------------------------------------------------------------- #
    # Introspection
    # ----------------------------------------------------------------- #
    def is_available(self, name: str) -> bool:
        state = self._states[name]
        if state.benched_at is None:
            return True
        if self._clock() - state.benched_at > self.cooldown_s:
            state.benched_at = None
            return True
        return False

    def stats(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "benched": not self.is_available(name),
                "failures": state.failures,
                "successes": state.successes,
            }
            for name, state in self._states.items()
        }

    # ----------------------------------------------------------------- #
    # Routing
    # ----------------------------------------------------------------- #
    def call(
        self,
        *args: Any,
        preferred: Optional[str] = None,
        purpose: str = "unknown",
        **kwargs: Any,
    ) -> RoutedResult:
        order = list(self._order)
        if preferred is not None:
            if preferred not in self._states:
                raise KeyError(f"unknown provider: {preferred}")
            order = [preferred] + [n for n in order if n != preferred]

        attempts: List[str] = []
        skipped: List[str] = []
        errors: Dict[str, str] = {}

        for name in order:
            if not self.is_available(name):
                skipped.append(name)
                errors.setdefault(name, "benched (cooldown)")
                continue
            state = self._states[name]
            attempts.append(name)
            try:
                value = state.fn(*args, **kwargs)
                if value is None or value == "":
                    raise ProviderError("empty response")
                state.successes += 1
                return RoutedResult(value=value, provider=name,
                                    attempts=tuple(attempts),
                                    skipped=tuple(skipped))
            except Exception as exc:  # provider callables may raise anything
                message = str(exc)
                errors[name] = message
                state.failures += 1
                if is_exhaustion_error(message):
                    state.benched_at = self._clock()
        raise AllProvidersFailedError(purpose, errors)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.agents.llm.resilient_router",
        description="Provider cascade with failure cooldown (Studio-OSS).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the cascade against injected fake providers: "
                             "fallback, cooldown bench, bench expiry, and the "
                             "all-failed error.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    now = [0.0]

    def clock() -> float:
        return now[0]

    calls = {"a": 0, "b": 0}

    def provider_a(prompt: str) -> str:
        calls["a"] += 1
        raise ProviderError("429 rate_limit: credit balance too low")

    def provider_b(prompt: str) -> str:
        calls["b"] += 1
        return f"b:{prompt}"

    router = ResilientRouter([("a", provider_a), ("b", provider_b)],
                             cooldown_s=60.0, clock=clock)

    result = router.call("hello", purpose="demo")
    assert result.provider == "b" and result.value == "b:hello"
    assert result.attempts == ("a", "b")
    print(f"[selfcheck] fallback: answered by '{result.provider}' "
          f"after attempts {list(result.attempts)}")

    result = router.call("again", purpose="demo")
    assert result.skipped == ("a",) and calls["a"] == 1, (result, calls)
    print("[selfcheck] cooldown: provider 'a' benched, not re-hammered")

    now[0] = 61.0
    router.call("later", purpose="demo")
    assert calls["a"] == 2
    print("[selfcheck] bench expiry: provider 'a' retried after cooldown")

    def provider_c(prompt: str) -> str:
        raise ProviderError("connection reset")

    lonely = ResilientRouter([("c", provider_c)], clock=clock)
    try:
        lonely.call("x", purpose="doomed")
    except AllProvidersFailedError as exc:
        assert "doomed" in str(exc)
        print(f"[selfcheck] all-failed: {exc}")
    else:
        raise AssertionError("expected AllProvidersFailedError")

    assert is_exhaustion_error("HTTP 429 Too Many Requests")
    assert not is_exhaustion_error("connection reset by peer")
    print("[selfcheck] error classification OK")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
