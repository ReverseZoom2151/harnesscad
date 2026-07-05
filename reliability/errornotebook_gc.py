"""errornotebook_gc — grammar-constraint (GC) verifier for corrected CoTs.

Paper sec. 2.4 ("Verifying Corrected Reasoning: GC Filtering"). Before a
corrected reasoning trajectory is admitted to the Error Notebook, it is passed
through a *deterministic* grammar-constraint verifier that checks structural
and semantic well-formedness. Only trajectories that pass are used to build the
notebook, which measurably improves exemplar quality (up to +4.5 accuracy pts).

The verifier searches for a line beginning ``Final Answer:``, extracts the
predicted filenames, and accepts iff:

  (1) such a line exists,
  (2) at least one filename is provided, and
  (3) every predicted filename appears in the allowed set P.

Two variants (sec. 2.4):

  - **sGC** (strict): requires the explicit ``Final Answer:`` marker.
  - **rGC** (relaxed): additionally accepts a trajectory that is otherwise
    valid but omits the marker — the filenames are then read off the last
    non-empty line.

Also included: :func:`build_corrected_trajectory`, the deterministic assembly
of R_corr = R_prev^sub (+) TR (+) R_g from its components (Eq. 3) — the string
plumbing around a correction, independent of the (external) VLM that authors
the natural-language reflection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

_FINAL = re.compile(r"^\s*final\s+answer\s*:\s*(.*)$", re.IGNORECASE)
_SPLIT = re.compile(r"[;,]")

DEFAULT_TRANSITION = "But wait, let's pause and examine this more carefully."


def _split_filenames(payload: str) -> List[str]:
    return [p.strip() for p in _SPLIT.split(payload) if p.strip()]


def extract_final_answer(trajectory: str) -> Optional[List[str]]:
    """Return the filenames on the last ``Final Answer:`` line, or None if absent."""
    found: Optional[List[str]] = None
    for line in trajectory.splitlines():
        m = _FINAL.match(line)
        if m is not None:
            found = _split_filenames(m.group(1))
    return found


def _last_nonempty_line(trajectory: str) -> str:
    for line in reversed(trajectory.splitlines()):
        if line.strip():
            return line.strip()
    return ""


@dataclass
class GCResult:
    """Outcome of a grammar-constraint check."""

    accepted: bool
    reason: str
    predicted: Tuple[str, ...] = ()
    marker_present: bool = False
    variant: str = "sGC"

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "predicted": list(self.predicted),
            "marker_present": self.marker_present,
            "variant": self.variant,
        }


def gc_check(
    trajectory: str,
    allowed: Iterable[str],
    variant: str = "sGC",
) -> GCResult:
    """Grammar-constraint verify one corrected trajectory against allowed set P.

    ``variant`` is ``"sGC"`` (strict; marker required) or ``"rGC"`` (relaxed;
    marker optional — falls back to the last non-empty line).
    """
    if variant not in ("sGC", "rGC"):
        raise ValueError("variant must be 'sGC' or 'rGC'")
    allowed_set: Set[str] = {str(a).strip() for a in allowed}

    predicted = extract_final_answer(trajectory)
    marker_present = predicted is not None

    if not marker_present:
        if variant == "sGC":
            return GCResult(False, "missing 'Final Answer:' marker",
                            (), False, variant)
        # rGC: try the last non-empty line as an implicit answer line
        predicted = _split_filenames(_last_nonempty_line(trajectory))

    predicted = predicted or []
    if len(predicted) == 0:
        return GCResult(False, "no filenames in final answer",
                        (), marker_present, variant)

    out_of_vocab = [p for p in predicted if p not in allowed_set]
    if out_of_vocab:
        return GCResult(
            False,
            "filenames not in allowed set: " + ";".join(sorted(out_of_vocab)),
            tuple(predicted), marker_present, variant,
        )

    return GCResult(True, "ok", tuple(predicted), marker_present, variant)


def gc_filter(
    trajectories: Sequence[str],
    allowed: Iterable[str],
    variant: str = "sGC",
) -> List[int]:
    """Indices of the trajectories that pass the GC check (build a clean notebook)."""
    allowed_set = {str(a).strip() for a in allowed}
    return [
        i for i, t in enumerate(trajectories)
        if gc_check(t, allowed_set, variant=variant).accepted
    ]


def build_corrected_trajectory(
    steps_up_to_first_error: Sequence[str],
    corrected_steps: Sequence[str],
    ground_truth: Sequence[str],
    transition: str = DEFAULT_TRANSITION,
) -> str:
    """Assemble R_corr = R_prev^sub (+) TR (+) R_g, ending in a Final Answer line (Eq. 3).

    Deterministic string plumbing: the caller supplies the retained correct
    prefix (through the first error), the corrected continuation, and the
    ground-truth filenames; this concatenates them with the transition phrase
    and appends a well-formed ``Final Answer:`` line so the result passes sGC.
    """
    parts: List[str] = [s.rstrip() for s in steps_up_to_first_error if s.strip()]
    parts.append(transition.rstrip())
    parts.extend(s.rstrip() for s in corrected_steps if s.strip())
    answer = ";".join(
        dict.fromkeys(str(g).strip() for g in ground_truth if str(g).strip())
    )
    parts.append("Final Answer: " + answer)
    return "\n".join(parts)
