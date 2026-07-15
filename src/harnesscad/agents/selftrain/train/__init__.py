"""The trainer subpackage -- the gradient the rest of ``selftrain`` refused to spend.

``recipe.py`` in the parent package is a DECLARATION: it names the models, the
QLoRA rank, the epochs and the cost, and it imports no torch. This subpackage is
the part that actually runs it. It is separated for exactly one reason: the parent
package, and the whole ``harnesscad`` core, is stdlib-only and must stay importable
and testable on a machine with no CUDA, no torch and no bitsandbytes. So every
module here import-guards its heavy dependencies and every test SKIPS -- loudly,
with a reason -- when they are absent.

The optional dependency set lives behind the ``harnesscad[train]`` extra
(``pyproject.toml``). :data:`HAS_TORCH` and :func:`require` are the seam the tests
and the CLI check before they do anything that needs a GPU.

READ ``selftrain/ledger.py`` BEFORE ANYTHING HERE. The single most important fact
about this trainer is WHAT IT IS TRAINED ON: op streams certified by the
conjunction of gate + envelope + shape, never the verifier fleet, with the
many-to-one and size-blindness holes recorded on every record. A trainer that
forgets that is a trainer optimising a reward it does not understand.
"""

from __future__ import annotations

from typing import List, Tuple

__all__ = ["HAS_TORCH", "HAS_TRL", "HAS_BNB", "MISSING", "require", "why_unavailable"]


def _probe() -> Tuple[bool, bool, bool, List[str]]:
    missing: List[str] = []
    try:
        import torch  # noqa: F401
        has_torch = True
    except Exception:  # noqa: BLE001
        has_torch = False
        missing.append("torch")
    try:
        import transformers  # noqa: F401
        import peft  # noqa: F401
        import trl  # noqa: F401
        has_trl = True
    except Exception:  # noqa: BLE001
        has_trl = False
        missing.append("transformers+peft+trl")
    try:
        import bitsandbytes  # noqa: F401
        has_bnb = True
    except Exception:  # noqa: BLE001
        has_bnb = False
        missing.append("bitsandbytes")
    return has_torch, has_trl, has_bnb, missing


HAS_TORCH, HAS_TRL, HAS_BNB, MISSING = _probe()


def why_unavailable() -> str:
    if not MISSING:
        return ""
    return ("the training stack is not installed (missing: %s). Install it with "
            "`pip install harnesscad[train]`. The core suite runs without it; the "
            "trainer does not." % ", ".join(MISSING))


def require() -> None:
    """Raise a clear, actionable error if the training stack is absent."""
    if MISSING:
        raise RuntimeError(why_unavailable())
