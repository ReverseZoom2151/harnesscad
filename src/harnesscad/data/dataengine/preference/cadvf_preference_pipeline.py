"""CADFusion visual-feedback preference-data construction (Section 3.3, Fig. 3).

Builds the DPO preference dataset D_VF = {(x, o_w, o_l)} that CADFusion (Wang et
al., ICML 2025) uses in its visual-feedback stage. The pipeline (Figure 3):

  (a) for each text prompt x, the post-SL policy samples K parametric sequences,
      which are rendered into visual objects;
  (b) each rendered object is scored by the multi-aspect LVM rubric
      (dataengine.cadvf_visual_score.visual_score);
  (c) the higher-scored object becomes the preferred o_w and the lower-scored
      one the less-preferred o_l.

Two filtering steps precede scoring, exactly as the paper describes ("filtering
out invalid or low-quality samples", Section 4.1):

  * INVALID filtering -- sequences that fail to render into a valid visual object
    are dropped, and their fraction contributes to the Invalidity Ratio (IR).
  * LOW-QUALITY filtering -- rendered objects whose visual grade falls below a
    floor are dropped so preference pairs are not built from two poor samples.

This is distinct from the repo's existing DPO builders. ``dataengine.export.to_dpo``
pairs best/worst on a generic scalar reward; ``dataengine.cadrille_preference_pairs``
draws *random* pairs over K samples. Neither renders-filters-then-grades with the
CADFusion multi-aspect visual signal, and neither computes the invalidity ratio.
Here the ranking signal IS the visual grade, invalid renders are quantified, and
one best-vs-worst pair is emitted per prompt (Figure 3(c)).

Deterministic, stdlib-only. Sampling of the K sequences is external; this module
consumes already-sampled candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harnesscad.data.dataengine.reward.cadvf_visual_score import DEFAULT_WEIGHTS, visual_score

# A rendered object scoring below this 0-10 floor is treated as low-quality.
DEFAULT_QUALITY_FLOOR = 1.0


@dataclass(frozen=True)
class Candidate:
    """One sampled parametric sequence and its rendered visual object.

    ``sequence`` is the parametric-sequence string. ``renderable`` is False when
    the sequence failed to render into a valid object (it then feeds the
    invalidity ratio and is excluded from preference construction). ``components``
    is the rendered object as a list of cadvf_visual_score.Component; it is
    ignored when ``renderable`` is False.
    """

    sequence: str
    renderable: bool
    components: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "components", tuple(self.components))


@dataclass
class PromptResult:
    """Outcome of the pipeline for a single prompt."""

    prompt: str
    expected_count: int
    pair: tuple = None                    # (o_w, o_l) scored dicts, or None
    scored: list = field(default_factory=list)
    n_total: int = 0
    n_invalid: int = 0
    n_low_quality: int = 0

    @property
    def invalidity_ratio(self):
        return (self.n_invalid / self.n_total) if self.n_total else 0.0


def _score_candidate(cand, expected_count, weights):
    grade = visual_score(cand.components, expected_count, weights=weights)
    return {"sequence": cand.sequence, "score": grade["score"], "grade": grade}


def build_prompt_pair(prompt, expected_count, candidates,
                      quality_floor=DEFAULT_QUALITY_FLOOR, weights=None):
    """Render-score-pair one prompt's K candidates into a PromptResult.

    Invalid (non-renderable) candidates are counted then dropped. Remaining
    objects are graded; those below ``quality_floor`` are dropped as low-quality.
    If at least two graded objects with *different* scores survive, the highest
    becomes o_w and the lowest o_l. Ties across all survivors yield no pair
    (no preference signal). ``weights`` is forwarded to the visual rubric.
    """
    cands = list(candidates)
    w = DEFAULT_WEIGHTS if weights is None else weights
    scored = []
    n_invalid = 0
    n_low = 0
    for c in cands:
        if not c.renderable:
            n_invalid += 1
            continue
        s = _score_candidate(c, expected_count, w)
        if s["score"] < quality_floor:
            n_low += 1
            continue
        scored.append(s)
    # Deterministic ordering: score desc, then sequence string.
    scored.sort(key=lambda s: (-s["score"], s["sequence"]))
    pair = None
    if len(scored) >= 2 and scored[0]["score"] > scored[-1]["score"]:
        pair = (scored[0], scored[-1])
    return PromptResult(prompt=prompt, expected_count=expected_count, pair=pair,
                        scored=scored, n_total=len(cands), n_invalid=n_invalid,
                        n_low_quality=n_low)


def build_preference_dataset(prompt_batches, quality_floor=DEFAULT_QUALITY_FLOOR,
                             weights=None):
    """Assemble D_VF over many prompts (the full VF-stage data collection).

    ``prompt_batches`` is an iterable of ``(prompt, expected_count, candidates)``.
    Returns a dict with:
      * ``pairs``   -- list of {prompt, chosen, chosen_score, rejected,
                       rejected_score} DPO rows (one per prompt that produced a
                       separable pair);
      * ``results`` -- the per-prompt PromptResult objects;
      * ``invalidity_ratio`` -- aggregate IR = total invalid / total sampled;
      * ``counts``  -- {prompts, sampled, invalid, low_quality, pairs}.
    """
    results = []
    pairs = []
    total = 0
    total_invalid = 0
    total_low = 0
    for prompt, expected_count, candidates in prompt_batches:
        res = build_prompt_pair(prompt, expected_count, candidates,
                                quality_floor=quality_floor, weights=weights)
        results.append(res)
        total += res.n_total
        total_invalid += res.n_invalid
        total_low += res.n_low_quality
        if res.pair is not None:
            ow, ol = res.pair
            pairs.append({
                "prompt": prompt,
                "chosen": ow["sequence"],
                "chosen_score": ow["score"],
                "rejected": ol["sequence"],
                "rejected_score": ol["score"],
            })
    ir = (total_invalid / total) if total else 0.0
    return {
        "pairs": pairs,
        "results": results,
        "invalidity_ratio": ir,
        "counts": {
            "prompts": len(results),
            "sampled": total,
            "invalid": total_invalid,
            "low_quality": total_low,
            "pairs": len(pairs),
        },
    }
