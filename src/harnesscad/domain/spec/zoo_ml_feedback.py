"""Zoo ML text-to-CAD *response + feedback* model and acceptance metric.

Mined from **Zoo** (tryAGI/Zoo, the .NET SDK generated from Zoo/KittyCAD's
OpenAPI). The generated ML client surfaces the *result* side of Zoo's text-to-CAD
service that the harness had not modelled: ``TextToCadResponse`` (prompt, model
version, output format, status, per-format ``outputs``, and a user
``feedback``), the ``MlFeedback`` thumbs-up/down vocabulary, and
``ListTextToCadPartsForUser`` -- a paginated history of a user's generated parts
sorted by creation time (``CreatedAtSortMode``).

This is deliberately distinct from the harness's existing Zoo pieces:

* :mod:`harnesscad.domain.spec.zoo_catalog` -- engine ops, KCL stdlib, formats;
* :mod:`harnesscad.domain.spec.zoo_cli_catalog` -- the CLI verb surface;
* :mod:`harnesscad.io.adapters.zoo_api` -- the *request/poll* side (submit a
  prompt, poll the async operation).

Those cover "how to ask" and "what formats exist". This covers "what came back
and was it any good" -- the response record, the terminal status semantics, and
the feedback signal **reframed as a checkable acceptance metric**: over a set of
completed generations with recorded feedback, what fraction did users accept
(thumbs up). That turns Zoo's ``MlFeedback`` from a UI affordance into an offline
quality gate a text-to-CAD harness can score itself with.

Everything here is inert data plus pure functions; no network, no wall clock.
stdlib-only, absolute imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ApiCallStatus",
    "TERMINAL_STATUSES",
    "is_terminal",
    "is_success",
    "MlFeedback",
    "FEEDBACK_VALUES",
    "CreatedAtSortMode",
    "TextToCadResponse",
    "AcceptanceStats",
    "acceptance_stats",
    "sort_by_created_at",
    "paginate",
]


class ApiCallStatus:
    """Zoo async API-call status vocabulary (``Zoo.ApiCallStatus``)."""

    QUEUED = "queued"
    UPLOADED = "uploaded"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATUSES: Tuple[str, ...] = (ApiCallStatus.COMPLETED, ApiCallStatus.FAILED)


def is_terminal(status: str) -> bool:
    """True when the operation has reached a terminal status."""
    return status in TERMINAL_STATUSES


def is_success(status: str) -> bool:
    """True only for ``completed`` (a failed op is terminal but not a success)."""
    return status == ApiCallStatus.COMPLETED


class MlFeedback:
    """Zoo ML feedback vocabulary (``Zoo.MlFeedback``)."""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


FEEDBACK_VALUES: Tuple[str, ...] = (
    MlFeedback.THUMBS_UP,
    MlFeedback.THUMBS_DOWN,
    MlFeedback.ACCEPTED,
    MlFeedback.REJECTED,
)

# Positive feedback = the user kept the generation.
_POSITIVE_FEEDBACK = frozenset({MlFeedback.THUMBS_UP, MlFeedback.ACCEPTED})
_NEGATIVE_FEEDBACK = frozenset({MlFeedback.THUMBS_DOWN, MlFeedback.REJECTED})


class CreatedAtSortMode:
    """Zoo history sort mode (``Zoo.CreatedAtSortMode``)."""

    ASC = "created_at_ascending"
    DESC = "created_at_descending"


@dataclass(frozen=True)
class TextToCadResponse:
    """A Zoo ML text-to-CAD response record (``Zoo.TextToCadResponse``).

    ``outputs`` maps an output filename (e.g. ``"source.step"``) to its bytes /
    base64 payload -- here we keep only the keys, since payloads are opaque.
    ``feedback`` is one of :data:`FEEDBACK_VALUES` or ``None`` (not yet rated).
    ``created_at`` is an ISO-8601 string used only for ordering.
    """

    id: str
    prompt: str
    status: str = ApiCallStatus.QUEUED
    output_format: str = "step"
    model_version: str = ""
    created_at: str = ""
    output_keys: Tuple[str, ...] = ()
    feedback: Optional[str] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return is_success(self.status)

    @property
    def rated(self) -> bool:
        return self.feedback in FEEDBACK_VALUES

    @property
    def accepted(self) -> bool:
        return self.feedback in _POSITIVE_FEEDBACK


@dataclass(frozen=True)
class AcceptanceStats:
    """Aggregate acceptance metric over a set of responses."""

    total: int
    completed: int
    rated: int
    accepted: int
    rejected: int

    @property
    def acceptance_rate(self) -> float:
        """Accepted / rated. Zero when nothing was rated (no silent 1.0)."""
        return self.accepted / self.rated if self.rated else 0.0

    @property
    def completion_rate(self) -> float:
        """Completed / total. Zero when the set is empty."""
        return self.completed / self.total if self.total else 0.0


def acceptance_stats(responses: Sequence[TextToCadResponse]) -> AcceptanceStats:
    """Reframe Zoo's ``MlFeedback`` as a checkable acceptance metric.

    Counts only rated, completed generations toward acceptance; a failed or
    unrated response never counts as accepted. This is the offline quality gate a
    text-to-CAD harness scores itself with.
    """
    total = len(responses)
    completed = sum(1 for r in responses if r.succeeded)
    rated = sum(1 for r in responses if r.succeeded and r.rated)
    accepted = sum(1 for r in responses if r.succeeded and r.accepted)
    rejected = sum(
        1 for r in responses if r.succeeded and r.feedback in _NEGATIVE_FEEDBACK
    )
    return AcceptanceStats(
        total=total,
        completed=completed,
        rated=rated,
        accepted=accepted,
        rejected=rejected,
    )


def sort_by_created_at(
    responses: Sequence[TextToCadResponse], mode: str = CreatedAtSortMode.DESC
) -> List[TextToCadResponse]:
    """Order a user's parts by creation time (``ListTextToCadPartsForUser``)."""
    reverse = mode != CreatedAtSortMode.ASC
    return sorted(responses, key=lambda r: r.created_at, reverse=reverse)


def paginate(
    responses: Sequence[TextToCadResponse],
    *,
    page_size: int,
    page: int = 0,
) -> List[TextToCadResponse]:
    """A single page of a results-page listing (0-based ``page``)."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    start = page * page_size
    return list(responses[start : start + page_size])
