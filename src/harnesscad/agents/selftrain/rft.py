"""Rejection-Sampling Fine-Tuning (RFT / STaR) -- the cheapest use of a verifiable reward.

The book ranks it first (H2 sec. 8.5.2): "Generate many responses, select best
ones. SFT on the selected responses. Repeat." The whole method is a filter, and
the filter is the expensive part everywhere else in the literature. Here it is
free: :func:`selftrain.ledger.certify` is exact, deterministic, needs no human,
no reference answer and no LLM judge.

So the interesting question is not "can we build the filter" but "what does the
filter let through", and that is what :mod:`selftrain.ledger` exists to answer
before this module is ever run.

THREE THINGS THIS MODULE REFUSES TO DO
======================================
**It does not accept on ENVELOPE alone.** bbox + volume + probes are many-to-one
and a hole in an unprobed place scores perfectly. Acceptance is the conjunction
of gate + envelope + shape. That is the ``accept_full`` policy and it is the
default.

**It does not accept on the FLEET's verdict.** Not once, not as a tie-break. The
fleet's false positives are what cost the harness its own experiment.

**It does not silently drop the failures.** STaR's rationalization step (H2 lines
6122-6127) turns a failure into a training example by conditioning on the answer.
:func:`rationalized_records` does exactly that, and marks every record it produces
``source="rationalized"`` -- because a rationalized record is the model being
taught to imitate the REFERENCE, not to reason, and a corpus that mixes the two
without saying so is a corpus that has lied to its trainer.

DISTRIBUTION, HONESTLY
======================
RFT trains on our own successes. Everything the sampled models never tried, the
trained model will be *less* likely to try. Our corpus is 6 Ollama models on 12
briefs of one CISP dialect at temperature 0.0 -- so the yield is dominated by the
strongest model's idiom, the op vocabulary is whatever those models happened to
emit, and every brief is a plate/bracket/flange. Fine-tuning on it will make a
model better at *our* twelve briefs in *our* style and will teach it, as a side
effect, that a CAD plan is four ops long. :data:`DISTRIBUTION_WARNING` states this
on the dataset itself, in the manifest, so it travels with the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from harnesscad.agents.selftrain.trajectory import Trajectory

__all__ = [
    "AcceptPolicy",
    "POLICIES",
    "DISTRIBUTION_WARNING",
    "SFTRecord",
    "build",
    "rationalized_records",
    "acceptance_stats",
]


DISTRIBUTION_WARNING = (
    "This corpus is the exhaust of one experiment: 6 Ollama models, 12 briefs, "
    "one prompt, temperature 0.0, one seed. RFT on it teaches the model OUR "
    "habits -- our op vocabulary, our plan length, our three part families "
    "(plate / bracket / flange). It cannot teach anything no sampled model ever "
    "emitted, and it will make rare-but-correct constructions RARER. Treat a "
    "score gain on these 12 briefs as evidence of nothing until it is measured "
    "on a held-out corpus written by a different hand (eval/corpus/)."
)


@dataclass(frozen=True)
class AcceptPolicy:
    """A filter over certificates. Named, so the manifest can record which ran."""

    name: str
    description: str
    predicate: Callable[[Dict[str, Any]], bool]


def _full(v: Dict[str, Any]) -> bool:
    return bool(v.get("apply_ok") and v.get("gate_ok")
                and v.get("envelope_ok") and v.get("shape_ok"))


def _envelope_only(v: Dict[str, Any]) -> bool:
    return bool(v.get("apply_ok") and v.get("envelope_ok"))


def _gate_only(v: Dict[str, Any]) -> bool:
    return bool(v.get("apply_ok") and v.get("gate_ok"))


#: The policies, so the report can state what each one would have let through.
#: ``full`` is the only one anything should be trained on; the other two exist to
#: MEASURE the blindness rather than to assert it, and the difference between
#: their yields is the size of the many-to-one hole.
POLICIES: Dict[str, AcceptPolicy] = {
    "full": AcceptPolicy(
        "full",
        "gate AND envelope AND shape. The only policy fit to train on.",
        _full),
    "envelope_only": AcceptPolicy(
        "envelope_only",
        "the v1 grader: bbox + volume + probes + op assertions. MANY-TO-ONE.",
        _envelope_only),
    "gate_only": AcceptPolicy(
        "gate_only",
        "reference-free: well-formed and honours its own declared intent. Has "
        "never read the brief. This is the ONLY policy available on a brief a "
        "user typed, and its yield is the honest ceiling of self-training in "
        "production.",
        _gate_only),
}


@dataclass
class SFTRecord:
    """One (prompt, completion) pair, plus the provenance that justifies it."""

    schema: str = "selftrain/rft/1"
    trajectory_id: str = ""
    brief_id: str = ""
    model: str = ""
    source: str = "sampled"          # "sampled" | "rationalized"
    prompt: str = ""
    completion: str = ""             # the op stream, canonical JSON
    accept_policy: str = "full"
    shape_iou: Optional[float] = None
    reward_total: float = 0.0
    blind_spots: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema, "trajectory_id": self.trajectory_id,
            "brief_id": self.brief_id, "model": self.model,
            "source": self.source, "prompt": self.prompt,
            "completion": self.completion, "accept_policy": self.accept_policy,
            "shape_iou": self.shape_iou, "reward_total": self.reward_total,
            "blind_spots": list(self.blind_spots),
        }


def _completion(ops: Sequence[dict]) -> str:
    import json
    return json.dumps([dict(o) for o in ops], sort_keys=True, indent=2)


def build(trajectories: Iterable[Trajectory], *,
          policy: str = "full",
          dedup: bool = True) -> List[SFTRecord]:
    """The RFT set: every candidate the oracle certified, as an SFT pair.

    ``dedup`` collapses op streams that are byte-identical after canonicalisation
    for the same brief -- six models converging on the same four ops is ONE fact,
    and counting it six times would let the strongest model's idiom dominate the
    loss by sheer repetition. The manifest reports both numbers.
    """
    accept = POLICIES[policy].predicate
    seen = set()
    out: List[SFTRecord] = []
    for t in trajectories:
        if not t.parse_ok or not t.ops:
            continue
        if not accept(t.verdict or {}):
            continue
        completion = _completion(t.ops)
        key = (t.brief_id, completion)
        if dedup and key in seen:
            continue
        seen.add(key)
        out.append(SFTRecord(
            trajectory_id=t.trajectory_id,
            brief_id=t.brief_id,
            model=t.model,
            source="sampled",
            prompt=t.prompt or t.brief_text,
            completion=completion,
            accept_policy=policy,
            shape_iou=(t.verdict or {}).get("shape_iou"),
            reward_total=t.reward_total,
            blind_spots=list((t.verdict or {}).get("blind_spots") or []),
        ))
    return out


def rationalized_records(briefs_without_success: Sequence[Any]) -> List[SFTRecord]:
    """STaR rationalization: the brief's own reference, as an SFT pair.

    For a brief NO model ever solved, the corpus otherwise contributes nothing.
    The book's fix is to condition on the answer and train on the trace that
    reaches it. Here the "answer" is the brief's hand-written ``reference`` op
    stream -- so these records teach the model to reproduce a HUMAN's solution,
    not to reason its way to one.

    They are marked ``source="rationalized"`` and they are emitted to a SEPARATE
    file. Mixing them into the sampled set without a flag would let a trainer
    report an RFT yield that is really a count of hand-written answers.
    """
    out: List[SFTRecord] = []
    for brief in briefs_without_success:
        out.append(SFTRecord(
            trajectory_id="reference|%s" % brief.id,
            brief_id=brief.id,
            model="(hand-written reference)",
            source="rationalized",
            prompt=brief.text,
            completion=_completion(brief.reference),
            accept_policy="reference",
            shape_iou=1.0,
            reward_total=0.0,
            blind_spots=["this is a HUMAN's answer, not a model's reasoning"],
        ))
    return out


def acceptance_stats(trajectories: Sequence[Trajectory]) -> Dict[str, Any]:
    """Yield per policy, and the gap between them -- which IS the blindness."""
    parsed = [t for t in trajectories if t.parse_ok and t.ops]
    stats: Dict[str, Any] = {
        "trajectories": len(list(trajectories)),
        "parsed": len(parsed),
        "by_policy": {},
    }
    for name, policy in POLICIES.items():
        kept = [t for t in parsed if policy.predicate(t.verdict or {})]
        stats["by_policy"][name] = {
            "accepted": len(kept),
            "acceptance_rate": (len(kept) / len(parsed)) if parsed else 0.0,
            "unique_streams": len({(t.brief_id, _completion(t.ops)) for t in kept}),
            "briefs_covered": len({t.brief_id for t in kept}),
        }
    env = stats["by_policy"]["envelope_only"]["accepted"]
    full = stats["by_policy"]["full"]["accepted"]
    stats["many_to_one_gap"] = env - full
    stats["many_to_one_note"] = (
        "%d candidate(s) pass the ENVELOPE grader and fail the SHAPE metric. Each "
        "is a part with the right bounding box, the right volume, the right probe "
        "answers, and the wrong geometry. Every one of them would have been a "
        "training pair under the v1 grader." % (env - full)
    )
    return stats
