"""Raise the certified-candidate count by sampling a STRONGER model, honestly.

The parent corpus is 13 RFT records because it is the exhaust of an experiment whose
strongest sampled model was qwen2.5-coder:14b. ``qwen2.5-coder:32b`` has since
landed. This module samples it (and, optionally, 14b/7b) on the SAME briefs the
training set already covers -- never the held-out eval briefs -- at a spread of
temperatures and seeds, re-adjudicates every draw with :func:`ledger.certify` (the
identical conjunction labeller: gate AND envelope AND shape, NEVER the fleet), and
emits the certified ones as new RFT-``full`` records plus KTO desirable/undesirable
records.

Two disciplines, both load-bearing:

* **Only non-reasoning coder models.** A reasoning model (deepseek-r1, magistral,
  qwen3) spends its token budget inside a ``<think>`` block and, truncated,
  emits nothing -- which would be miscounted as a capability failure and would
  corrupt any Best-of-N pool. This module refuses them by name and strips a
  ``<think>`` block defensively in case one is configured in anyway.
* **Never sample an eval brief.** ``TRAIN_BRIEFS`` is exactly the 12 the corpus
  already used. Generating on the 16 held-out pressure briefs would contaminate
  the one measurement that means anything.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The 12 briefs the self-train corpus already covers. Sampling is confined here.
TRAIN_BRIEFS: Tuple[str, ...] = (
    "flange_round", "flange_square", "l_bracket", "plate_hole_four",
    "shell_box_3mm", "step_block", "strip_hole_row", "trap_fillet_thin_plate",
    "trap_fillet_too_big", "trap_hole_oversize", "trap_shell_too_thick",
    "trap_shell_too_thin",
)

#: Reasoning models are refused: their think-budget behaviour makes a bare-JSON
#: harness under-count them, which is a measurement error, not a capability one.
REASONING_DENYLIST = ("deepseek-r1", "magistral", "qwen3", "-r1", "reasoning", "think")

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def is_reasoning_model(name: str) -> bool:
    n = name.lower()
    return any(tag in n for tag in REASONING_DENYLIST)


def strip_think(text: str) -> str:
    return _THINK.sub("", text or "")


@dataclass
class GenStats:
    model: str = ""
    draws: int = 0
    parsed: int = 0
    certified: int = 0
    by_brief: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"model": self.model, "draws": self.draws, "parsed": self.parsed,
                "certified": self.certified, "by_brief": dict(sorted(self.by_brief.items()))}


def sample_and_certify(model: str, brief_ids: Sequence[str] = TRAIN_BRIEFS,
                       seeds: Sequence[int] = (1, 2, 3, 4, 5),
                       temperatures: Sequence[float] = (0.2, 0.5, 0.8),
                       max_tokens: int = 1536,
                       progress=None) -> Tuple[List[dict], List[dict], GenStats]:
    """Sample ``model`` over ``brief_ids`` and keep the certified op streams.

    Returns ``(rft_records, kto_records, stats)``. ``rft_records`` follow the
    ``selftrain/rft/1`` schema of the parent RFT file; ``kto_records`` carry the
    ``desirable`` bit (True iff certified) so every draw -- pass or fail -- becomes
    a KTO label.
    """
    if is_reasoning_model(model):
        raise ValueError("refusing reasoning model %r: its think-budget behaviour "
                         "would be miscounted by a bare-JSON harness" % model)

    from harnesscad.eval.pressure import prompts, model as pmodel, briefs as B
    from harnesscad.agents.selftrain import ledger

    stats = GenStats(model=model)
    rft: List[dict] = []
    kto: List[dict] = []
    seen_completion: set = set()

    for bid in brief_ids:
        br = B.brief_by_id(bid)
        msgs = [{"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": prompts.user_prompt(br.text)}]
        for temp in temperatures:
            for sd in seeds:
                cli = pmodel.OllamaClient(model, seed=sd, temperature=temp,
                                          max_tokens=max_tokens)
                try:
                    raw = cli.complete(msgs, attempt=1, seed=sd, temperature=temp)
                except Exception as exc:  # noqa: BLE001
                    if progress:
                        progress("  %s %s t=%.1f s=%d ERROR %s" % (model, bid, temp, sd, exc))
                    continue
                stats.draws += 1
                parsed = pmodel.extract_ops(strip_think(raw))
                if not parsed.ok:
                    kto.append(_kto_row(bid, br.text, "[]", False, model))
                    continue
                ops = pmodel.ops_to_dicts(parsed)
                stats.parsed += 1
                completion = json.dumps(ops, sort_keys=True, indent=2)
                # FAST PATH: the shape-IoU metric costs ~13s; the gate+envelope
                # pre-grade costs ~4s and is a NECESSARY condition for the full
                # conjunction. A draw that fails envelope cannot be certified, so we
                # skip the IoU entirely for it -- identical verdict, a third of the
                # cost. Only a candidate that already passes gate+envelope is worth
                # the volumetric marching.
                from harnesscad.eval.pressure import metrics as _metrics
                pre = _metrics.grade(br, list(ops), shape=False)
                if not (pre.apply_ok and pre.gate_ok and pre.solved):
                    kto.append(_kto_row(bid, br.text, completion, False, model))
                    if progress:
                        progress("  %s %-22s t=%.1f s=%d -> no(envelope)" % (
                            model, bid, temp, sd))
                    continue
                cert = ledger.certify(br, ops)
                kto.append(_kto_row(bid, br.text, completion, bool(cert.accepted),
                                    model, reward=2.0 if cert.accepted else 0.0))
                if cert.accepted:
                    key = (bid, completion)
                    if key in seen_completion:
                        continue
                    seen_completion.add(key)
                    stats.certified += 1
                    stats.by_brief[bid] = stats.by_brief.get(bid, 0) + 1
                    rft.append({
                        "schema": "selftrain/rft/1",
                        "trajectory_id": "gen|%s|%s|t%.1f|s%d" % (model, bid, temp, sd),
                        "brief_id": bid, "model": model, "source": "sampled",
                        "prompt": br.text, "completion": completion,
                        "accept_policy": "full",
                        "shape_iou": cert.shape_iou, "reward_total": 2.0,
                        "blind_spots": list(cert.blind_spots),
                    })
                if progress:
                    progress("  %s %-22s t=%.1f s=%d -> %s" % (
                        model, bid, temp, sd, "CERT" if cert.accepted else "no"))
    return rft, kto, stats


def _kto_row(brief_id: str, brief_text: str, completion: str, desirable: bool,
             model: str, reward: float = 0.0) -> dict:
    return {"schema": "selftrain/kto/1", "brief_id": brief_id, "prompt": brief_text,
            "completion": completion, "desirable": desirable, "reward": reward,
            "model": model, "label_source": "oracle (gate+envelope+shape); NEVER the fleet"}


def append_jsonl(path: str, rows: Sequence[dict]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True))
            fh.write("\n")
    return len(rows)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Sample a stronger model, certify, emit RFT/KTO.")
    ap.add_argument("--models", default="qwen2.5-coder:32b")
    ap.add_argument("--rft-out", default="assets/selftrain/rft_augmented.jsonl")
    ap.add_argument("--kto-out", default="assets/selftrain/kto_augmented.jsonl")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--temps", default="0.2,0.5,0.8")
    args = ap.parse_args(argv)

    seeds = tuple(int(s) for s in args.seeds.split(","))
    temps = tuple(float(t) for t in args.temps.split(","))
    all_stats = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        rft, kto, stats = sample_and_certify(
            model, seeds=seeds, temperatures=temps,
            progress=lambda s: sys.stderr.write(s + "\n"))
        append_jsonl(args.rft_out, rft)
        append_jsonl(args.kto_out, kto)
        all_stats.append(stats.to_dict())
        print(json.dumps(stats.to_dict(), indent=2))
    print(json.dumps({"models": all_stats}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
