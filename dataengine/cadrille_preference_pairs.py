"""cadrille DPO preference-pair construction from K sampled programs (2026).

For offline DPO fine-tuning cadrille samples ``K`` (=5) Python codes per input
from the SFT policy. At each training step, for a given input, two of the K
outputs are picked at random; the one with the *larger* reward ``R(tau)`` becomes
the preferred ``tau_w`` and the other the non-preferred ``tau_l``. Pairs whose
two members carry equal reward provide no preference signal and are dropped.

This is distinct from the repository's ``dataengine.export.to_dpo``, which emits
a single best/worst pair per prompt group. Here we build the *random-pair*
sampling scheme over the K candidates that cadrille actually uses, driven by a
seeded ``random.Random`` for determinism.
"""

from __future__ import annotations

import random

DEFAULT_K = 5


def preference_pair(a: dict, b: dict):
    """Order two ``{'code', 'reward'}`` samples into (chosen, rejected).

    Returns ``None`` when the rewards tie (no preference signal).
    """
    ra, rb = float(a["reward"]), float(b["reward"])
    if ra == rb:
        return None
    return (a, b) if ra > rb else (b, a)


def all_preference_pairs(samples):
    """Every unordered pair of the K samples, ordered by reward (ties dropped).

    Deterministic: iterates index pairs ``i < j`` in order.
    """
    items = list(samples)
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ordered = preference_pair(items[i], items[j])
            if ordered is not None:
                pairs.append(ordered)
    return pairs


def sample_preference_pairs(samples, count: int, seed: int = 0):
    """Draw ``count`` random preference pairs from the K samples.

    Mirrors the cadrille training loop that, per step, randomly selects two of
    the K outputs and labels the higher-reward one preferred. ``random.Random``
    is seeded for reproducibility. Tie draws are skipped (they carry no signal)
    but still consume the requested number of attempts, so the output length may
    be < ``count`` when rewards collide.
    """
    items = list(samples)
    if len(items) < 2:
        raise ValueError("need at least two samples to form a pair")
    if count < 0:
        raise ValueError("count must be non-negative")
    rng = random.Random(seed)
    pairs = []
    for _ in range(count):
        i, j = rng.sample(range(len(items)), 2)
        ordered = preference_pair(items[i], items[j])
        if ordered is not None:
            pairs.append(ordered)
    return pairs


def to_dpo_records(pairs, prompt: str = ""):
    """Serialise ordered pairs into DPO rows ``{prompt, chosen, rejected}``."""
    rows = []
    for chosen, rejected in pairs:
        rows.append({
            "prompt": prompt,
            "chosen": chosen["code"],
            "chosen_reward": float(chosen["reward"]),
            "rejected": rejected["code"],
            "rejected_reward": float(rejected["reward"]),
        })
    return rows
