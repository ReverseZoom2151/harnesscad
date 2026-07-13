"""The CADGenBench submission contract: candidate discovery, meta, manifest.

A submission is deliberately minimal and tool-agnostic: one directory per
sample, each holding a single ``output.step`` (or ``output.stp``) candidate, plus
a small ``meta.json`` at the root. No description, no metadata, no sub-volumes -
the grader owns all of that. Any kernel that exports STEP can play.

Three rules worth transferring verbatim:

1. **A sample directory with no candidate is preserved, not dropped.** It is
   recorded as ``status: "missing"`` and scores zero. Silently omitting it would
   let a submission raise its average by skipping the samples it could not
   build - the single most important anti-gaming rule in the contract.
2. **Candidate name is fixed, extension is not.** Exactly one of
   :data:`CANDIDATE_NAMES` is accepted, in priority order; a zero-byte file does
   not count as a candidate.
3. **Publication consent is explicit.** ``agree_to_publish`` defaults to false
   and a submission is not accepted until it is true. Consent is never inferred.

This module validates and manifests a submission tree; it does not need the
files' contents, so it works equally on a real directory or on an in-memory
listing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Accepted candidate filenames, in priority order.
CANDIDATE_NAMES: Tuple[str, ...] = ("output.step", "output.stp")

REQUIRED_META_KEYS: Tuple[str, ...] = (
    "submitter_name",
    "submission_name",
    "agent_url",
    "notes",
    "agree_to_publish",
)

MAX_NOTES_CHARS = 500

STATUS_PRESENT = "present"
STATUS_MISSING = "missing"


@dataclass(frozen=True)
class SampleEntry:
    """One sample folder in the submission."""

    name: str
    candidate: Optional[str] = None

    @property
    def status(self) -> str:
        return STATUS_PRESENT if self.candidate else STATUS_MISSING

    def to_dict(self) -> dict:
        return {"name": self.name, "candidate": self.candidate, "status": self.status}


@dataclass(frozen=True)
class SubmissionReport:
    accepted: bool
    errors: List[str] = field(default_factory=list)
    manifest: Dict[str, object] = field(default_factory=dict)


def pick_candidate(filenames: Iterable[str], sizes: Optional[Mapping[str, int]] = None) -> Optional[str]:
    """Return the accepted candidate filename in a sample folder, or ``None``.

    Priority order is :data:`CANDIDATE_NAMES`. A file present but empty (size 0)
    is not a candidate: an empty export is a failed build, not a submission.
    """
    names = set(filenames)
    for accepted in CANDIDATE_NAMES:
        if accepted in names:
            if sizes is not None and sizes.get(accepted, 1) <= 0:
                continue
            return accepted
    return None


def discover_samples(
    tree: Mapping[str, Sequence[str]],
    sizes: Optional[Mapping[str, Mapping[str, int]]] = None,
) -> List[SampleEntry]:
    """Turn ``{sample_name: [filenames]}`` into sample entries, sorted by name.

    Sample folders without a candidate are **kept** (status ``missing``).
    """
    entries = []
    for name in sorted(tree):
        sample_sizes = (sizes or {}).get(name)
        entries.append(
            SampleEntry(name=name, candidate=pick_candidate(tree[name], sample_sizes))
        )
    return entries


def scan_directory(run_dir: str | Path) -> List[SampleEntry]:
    """Scan a real run directory (``<run>/<sample>/output.step``)."""
    root = Path(run_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"run dir not found: {root}")
    tree: Dict[str, List[str]] = {}
    sizes: Dict[str, Dict[str, int]] = {}
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [f.name for f in child.iterdir() if f.is_file()]
        tree[child.name] = files
        sizes[child.name] = {
            f.name: f.stat().st_size for f in child.iterdir() if f.is_file()
        }
    return discover_samples(tree, sizes)


def default_meta(submitter_name: str, submission_name: str) -> Dict[str, object]:
    """A meta stub. ``agree_to_publish`` starts false: consent is never assumed."""
    return {
        "submitter_name": submitter_name,
        "submission_name": submission_name,
        "agent_url": None,
        "notes": None,
        "agree_to_publish": False,
    }


def validate_meta(meta: Mapping[str, object]) -> List[str]:
    """Return the reasons ``meta.json`` is unacceptable ([] = acceptable)."""
    errors: List[str] = []
    for key in REQUIRED_META_KEYS:
        if key not in meta:
            errors.append(f"meta.json missing required key: {key}")
    if not str(meta.get("submitter_name") or "").strip():
        errors.append("meta.json submitter_name must be non-empty")
    if not str(meta.get("submission_name") or "").strip():
        errors.append("meta.json submission_name must be non-empty")
    notes = meta.get("notes")
    if notes is not None and len(str(notes)) > MAX_NOTES_CHARS:
        errors.append(f"meta.json notes exceeds {MAX_NOTES_CHARS} characters")
    if meta.get("agree_to_publish") is not True:
        errors.append(
            "meta.json agree_to_publish is not true: the submission is not "
            "accepted until publication is explicitly consented to"
        )
    return errors


def build_manifest(
    entries: Sequence[SampleEntry], meta: Mapping[str, object]
) -> Dict[str, object]:
    """Assemble the submission manifest (what the grader will iterate over)."""
    n_present = sum(1 for e in entries if e.candidate)
    return {
        "meta": dict(meta),
        "n_samples": len(entries),
        "n_with_candidate": n_present,
        "n_missing": len(entries) - n_present,
        "samples": [e.to_dict() for e in entries],
    }


def validate_submission(
    entries: Sequence[SampleEntry],
    meta: Mapping[str, object],
    *,
    expected_samples: Optional[Iterable[str]] = None,
) -> SubmissionReport:
    """Validate a whole submission: meta, sample folders, expected coverage.

    An *unknown* sample folder is an error (the grader has no ground truth for
    it). A *missing* expected sample is **not**: it is simply scored zero, which
    is the contract's whole point.
    """
    errors = list(validate_meta(meta))
    if not entries:
        errors.append("submission contains no sample folders")

    if expected_samples is not None:
        expected = set(expected_samples)
        found = {e.name for e in entries}
        for unknown in sorted(found - expected):
            errors.append(f"unknown sample folder: {unknown}")

    manifest = build_manifest(entries, meta)
    if expected_samples is not None:
        expected = set(expected_samples)
        manifest["n_not_submitted"] = len(expected - {e.name for e in entries})
    return SubmissionReport(accepted=not errors, errors=errors, manifest=manifest)
