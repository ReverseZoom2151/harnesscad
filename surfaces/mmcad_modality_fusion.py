"""Deterministic modality fusion for multimodal CAD command entry.

Paper 179 -- "Toward AI-driven Multimodal Interfaces for Industrial CAD
Modeling" (Choi, Jang, Hyun, CHI '25) -- describes voice, gesture, and
sketch inputs as complementary modalities for issuing CAD operations
(Table 1: Voice-based Modeling Commands, Gesture and Motion Controls,
Sketch-to-3D Conversion).  The paper is a position/HCI paper and does not
specify a fusion algorithm; this module supplies a *deterministic*,
learned-model-free scheme that combines several single-modality signals
into one CAD intent.

The scheme is complementary + competitive fusion:

* Competitive: when several modalities each name an operation, one wins by
  ``confidence * modality_weight`` with fully deterministic tie-breaks.
* Complementary: slots (parameters such as ``depth`` or ``profile``) are
  merged across modalities, each slot taken from the strongest signal that
  supplies it, with recorded provenance.

No wall clock and no randomness: signal ordering is an explicit integer
``order`` field supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Sequence


class ModalityKind(str, Enum):
    """The input channels named by the paper, plus classic keyboard."""

    KEYBOARD = "keyboard"
    VOICE = "voice"
    SKETCH = "sketch"
    GESTURE = "gesture"


# Fixed precedence used for deterministic tie-breaking and default weights.
# Keyboard is the accustomed industrial channel (paper section 3.1) and is
# trusted most on ties; the AI-driven channels follow.
_PRECEDENCE: tuple[ModalityKind, ...] = (
    ModalityKind.KEYBOARD,
    ModalityKind.VOICE,
    ModalityKind.SKETCH,
    ModalityKind.GESTURE,
)

_DEFAULT_WEIGHTS: dict[ModalityKind, float] = {
    ModalityKind.KEYBOARD: 1.0,
    ModalityKind.VOICE: 0.9,
    ModalityKind.SKETCH: 0.85,
    ModalityKind.GESTURE: 0.7,
}


def _precedence_rank(kind: ModalityKind) -> int:
    return _PRECEDENCE.index(kind)


@dataclass(frozen=True)
class ModalitySignal:
    """One interpreted single-modality input.

    ``operation`` may be ``None`` when a modality only contributes slots
    (e.g. a gesture that supplies a magnitude but names no operation).
    ``slots`` are parameter name -> value pairs.  ``confidence`` is in
    ``[0, 1]``.  ``order`` is a caller-supplied deterministic sequence
    index (lower arrives earlier).
    """

    kind: ModalityKind
    operation: Optional[str] = None
    slots: Mapping[str, object] = field(default_factory=dict)
    confidence: float = 1.0
    order: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if self.operation is not None and not self.operation:
            raise ValueError("operation must be non-empty or None")


@dataclass(frozen=True)
class FusedIntent:
    """The single deterministic CAD intent produced by fusion."""

    operation: Optional[str]
    slots: Mapping[str, object]
    provenance: Mapping[str, ModalityKind]
    confidence: float
    contributors: tuple[ModalityKind, ...]
    ambiguous: bool
    conflicts: tuple[tuple[str, str], ...]

    @property
    def needs_clarification(self) -> bool:
        return self.ambiguous or self.operation is None


class ModalityFuser:
    """Fuse several :class:`ModalitySignal` into one :class:`FusedIntent`.

    Parameters
    ----------
    confidence_floor:
        Signals at or below this confidence are discarded before fusion.
    conflict_margin:
        When the two strongest *distinct* operation proposals score within
        this relative margin of each other, the result is flagged
        ``ambiguous`` (a clarification should be requested).
    weights:
        Optional override of the per-modality trust weights.
    """

    def __init__(
        self,
        *,
        confidence_floor: float = 0.0,
        conflict_margin: float = 0.15,
        weights: Optional[Mapping[ModalityKind, float]] = None,
    ) -> None:
        if not 0.0 <= confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0,1]")
        if conflict_margin < 0.0:
            raise ValueError("conflict_margin must be non-negative")
        self.confidence_floor = confidence_floor
        self.conflict_margin = conflict_margin
        self.weights = dict(_DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def _weight(self, kind: ModalityKind) -> float:
        return self.weights.get(kind, 0.5)

    def _score(self, signal: ModalitySignal) -> float:
        return signal.confidence * self._weight(signal.kind)

    def _sort_key(self, signal: ModalitySignal):
        # Higher score first; then precedence (keyboard first); then earlier
        # order; then lexical operation name. Fully deterministic.
        return (
            -self._score(signal),
            _precedence_rank(signal.kind),
            signal.order,
            signal.operation or "",
        )

    def fuse(self, signals: Sequence[ModalitySignal]) -> FusedIntent:
        kept = [s for s in signals if s.confidence > self.confidence_floor]
        if not kept:
            return FusedIntent(
                operation=None, slots={}, provenance={}, confidence=0.0,
                contributors=(), ambiguous=False, conflicts=(),
            )

        ordered = sorted(kept, key=self._sort_key)

        # --- Operation selection (competitive) ---------------------------
        op_signals = [s for s in ordered if s.operation is not None]
        operation: Optional[str] = None
        conflicts: list[tuple[str, str]] = []
        ambiguous = False
        if op_signals:
            operation = op_signals[0].operation
            best_score = self._score(op_signals[0])
            # Best score for each distinct competing operation.
            distinct: dict[str, float] = {}
            for s in op_signals:
                distinct.setdefault(s.operation, self._score(s))
            rivals = sorted(
                (score for op, score in distinct.items() if op != operation),
                reverse=True,
            )
            if rivals and best_score > 0.0:
                top_rival = rivals[0]
                if (best_score - top_rival) <= self.conflict_margin * best_score:
                    ambiguous = True
            # Report every distinct losing operation as a conflict pair.
            for op in sorted(o for o in distinct if o != operation):
                conflicts.append((operation, op))

        # --- Slot merge (complementary) ----------------------------------
        slots: dict[str, object] = {}
        provenance: dict[str, ModalityKind] = {}
        for s in ordered:  # already strongest-first
            for key, value in s.slots.items():
                if key not in slots:
                    slots[key] = value
                    provenance[key] = s.kind

        # --- Fused confidence --------------------------------------------
        # Weighted mean of contributing scores, normalised by weight sum, so
        # a chorus of agreeing weak signals does not exceed a strong one.
        contributors = tuple(s.kind for s in ordered)
        total_w = sum(self._weight(s.kind) for s in ordered)
        conf = (
            sum(self._score(s) for s in ordered) / total_w
            if total_w > 0.0 else 0.0
        )

        return FusedIntent(
            operation=operation,
            slots=slots,
            provenance=provenance,
            confidence=conf,
            contributors=contributors,
            ambiguous=ambiguous,
            conflicts=tuple(conflicts),
        )
