"""IntentForge's prompt -> expected-REJECTION oracle pairs: refusal canaries.

Source: IntentForge (resources/cad_repos/IntentForge-main/IntentForge-main),
``src/benchmark/prompts/rejection_prompts.json`` (19 cases),
``src/benchmark/prompts/l_bracket_rejection_prompts.json`` (6 cases) and
``src/benchmark/expected/expected_rejections.json`` (the canonical rejection
categories). Each case is a prompt the system is REQUIRED to refuse -- an
unsupported object ("make a gear"), an unsupported pattern, a vague edit --
together with the substring its rejection message must contain. This is the
only expected-rejection data found in the resources tree, and it plugs the
labelling gap ``eval/hardcorpus/ambiguous.py`` names: that module scores
ask-vs-hallucinate on UNDERSPECIFIED briefs it generates itself; these cases
score refuse-vs-comply on OUT-OF-SCOPE briefs with human-authored labels.

LICENSE: Apache License 2.0 -- redistribution permitted, so the three JSON
files are VENDORED under ``intentforge/`` (provenance in
``intentforge/MANIFEST.json``, attribution in ``intentforge/LICENSE-NOTICE``).

Scoring rule (the inversion these canaries exist for): on every geometry
benchmark a model that BUILDS something scores higher than one that refuses.
Here it is the opposite -- the correct answer to each of these prompts is a
refusal that names why, and :func:`classify` calls a response that produced
geometry a ``complied`` FAILURE. The classifier is substring-conservative and
its limits are stated in :data:`CLASSIFIER_CAVEATS`; it never guesses.

Stdlib only, deterministic, no kernel, no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from harnesscad.eval.bench.imports import Manifest, load_manifest

__all__ = [
    "SOURCE_REPO",
    "LICENSE",
    "EXPECTED_CASES",
    "CLASSIFIER_CAVEATS",
    "RefusalCase",
    "manifest",
    "load",
    "expected_rejections",
    "classify",
    "score",
    "main",
]

_SOURCE = "intentforge"
SOURCE_REPO = "IntentForge"
LICENSE = "Apache-2.0"

#: 19 general + 6 L-bracket rejection cases; the selfcheck re-counts.
EXPECTED_CASES = 25

CLASSIFIER_CAVEATS = (
    "classify() is substring-conservative: 'refused' requires the case's own "
    "expected_error_contains marker (case-insensitive) in the response; a "
    "response that refuses in OTHER words is 'refused_offlabel' (a refusal, "
    "but not the labelled one); anything else is 'complied', the failure "
    "these canaries exist to catch. It cannot see whether a system that "
    "refused in words went on to build geometry anyway -- callers that have "
    "the build result must gate on it separately."
)

#: Verdicts, from best to worst.
REFUSED = "refused"
REFUSED_OFFLABEL = "refused_offlabel"
COMPLIED = "complied"


@dataclass(frozen=True)
class RefusalCase:
    """One prompt the system must refuse, with its labelled rejection marker."""

    id: str
    prompt: str
    #: substring the rejection message must contain (the human-authored label).
    expected_error_contains: str
    #: "" for the general suite, "l_bracket" for the family-scoped suite.
    family: str
    type: str

    def to_dict(self) -> dict:
        return {"id": self.id, "prompt": self.prompt,
                "expected_error_contains": self.expected_error_contains,
                "family": self.family, "type": self.type}


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def _read_json(m: Manifest, name: str):
    e = m.by_name(name)
    path = m.resolve(e) if e is not None else None
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load() -> List[RefusalCase]:
    """All 25 expected-rejection cases from the vendored copies."""
    m = manifest()
    cases: List[RefusalCase] = []
    for name in ("rejection_prompts", "l_bracket_rejection_prompts"):
        raw = _read_json(m, name)
        if raw is None:
            continue
        for r in raw:
            cases.append(RefusalCase(
                id=str(r["id"]),
                prompt=str(r["prompt"]),
                expected_error_contains=str(r.get("expected_error_contains", "")),
                family=str(r.get("expected_family", "")),
                type=str(r.get("type", "")),
            ))
    return cases


def expected_rejections() -> Dict[str, str]:
    """The canonical rejection categories (category -> message marker)."""
    raw = _read_json(manifest(), "expected_rejections")
    return {str(k): str(v) for k, v in (raw or {}).items()}


def classify(case: RefusalCase, response: str) -> str:
    """REFUSED / REFUSED_OFFLABEL / COMPLIED for one response to one case.

    Conservative on purpose; see :data:`CLASSIFIER_CAVEATS`.
    """
    text = (response or "").lower()
    marker = case.expected_error_contains.lower()
    if marker and marker in text:
        return REFUSED
    for canonical in expected_rejections().values():
        if canonical.lower() in text:
            return REFUSED_OFFLABEL
    return COMPLIED


def score(pairs: List[Tuple[RefusalCase, str]]) -> Dict[str, object]:
    """Verdict counts over (case, response) pairs. Runs no model.

    ``refusal_rate`` counts BOTH refusal verdicts -- a refusal in other words
    still refused; the split is reported so the difference stays visible.
    """
    verdicts = [classify(c, r) for c, r in pairs]
    n = len(verdicts)
    counts = {v: verdicts.count(v) for v in (REFUSED, REFUSED_OFFLABEL, COMPLIED)}
    return {
        "n": n,
        "counts": counts,
        "refusal_rate": (counts[REFUSED] + counts[REFUSED_OFFLABEL]) / n if n else 0.0,
        "labelled_refusal_rate": counts[REFUSED] / n if n else 0.0,
        "caveats": CLASSIFIER_CAVEATS,
    }


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "Apache-2.0", m.license
    problems = m.verify_vendored()
    assert not problems, "vendored data drifted: %s" % problems

    cases = load()
    assert len(cases) == EXPECTED_CASES, (
        "expected %d cases, loaded %d" % (EXPECTED_CASES, len(cases)))
    ids = {c.id for c in cases}
    assert len(ids) == len(cases), "duplicate case ids"
    for c in cases:
        assert c.prompt.strip(), "case %s has an empty prompt" % c.id
        assert c.expected_error_contains.strip(), (
            "case %s has no expected-rejection marker" % c.id)

    canon = expected_rejections()
    assert len(canon) >= 4 and all(v.strip() for v in canon.values()), canon

    # The classifier's three verdicts, exercised on synthetic responses.
    probe = cases[0]
    assert classify(probe, "Rejected: %s." % probe.expected_error_contains) == REFUSED
    offlabel = "Rejected: %s." % list(canon.values())[0]
    got = classify(RefusalCase("x", "p", "zzz_never_present", "", ""), offlabel)
    assert got == REFUSED_OFFLABEL, got
    assert classify(probe, "Sure! Here is your gear: cube(10);") == COMPLIED

    report = score([(probe, "Rejected: %s" % probe.expected_error_contains),
                    (probe, "done, geometry built")])
    assert report["n"] == 2 and report["refusal_rate"] == 0.5, report

    families = {c.family for c in cases if c.family}
    print("SELFCHECK OK: %d expected-rejection cases (%d general, %d "
          "family-scoped %s), %d canonical categories, classifier verdicts "
          "exercised"
          % (len(cases), sum(1 for c in cases if not c.family),
             sum(1 for c in cases if c.family), sorted(families), len(canon)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="IntentForge expected-rejection canaries (25 prompts the "
                    "system must refuse; Apache-2.0, vendored).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate vendored hashes, counts, labels and "
                             "the classifier's three verdicts.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _selfcheck()
    except AssertionError as exc:
        print("SELFCHECK FAILED: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
