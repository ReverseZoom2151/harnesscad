"""Run the whole comparison and write the report. One base model, three arms.

This is the module that produces the number the exercise exists to produce:
base@1 vs finetune@1 vs Oracle-Best-of-N@N, on the 16 held-out pressure briefs,
graded by ``ledger.certify``, with Wilson intervals and McNemar's exact test on the
paired outcomes. It loads each HF generator in turn (a 4-bit 7B and its adapter do
not co-reside comfortably with a second copy on 16 GB), scores it, frees it, and
moves on.

The honest framing, kept in the output: finetune@1 spends ONE generation per brief;
Oracle-BoN@N spends N. If finetune@1 does not beat BoN@3, the fine-tune -- which
cost real GPU-hours to make -- has been beaten by drawing three samples from the
un-tuned model and keeping the one that passes the gate. That is the book's warning
made concrete, and if it happens it is the result, not a bug.
"""

from __future__ import annotations

import gc
import json
import os
from typing import Any, Dict, List, Optional

from harnesscad.agents.selftrain.train import evaluate as E
from harnesscad.agents.selftrain.train.sft import BASE_MODEL


def _free(gen) -> None:
    try:
        import torch
        del gen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def run_experiment(*, base_model: str = BASE_MODEL,
                   adapter: Optional[str] = None,
                   bon_ns: List[int] = [3, 5],
                   bon_temperature: float = 0.8,
                   brief_ids=E.HELDOUT_BRIEFS,
                   out_path: Optional[str] = None) -> Dict[str, Any]:
    """Score base@1, finetune@1 and Oracle-BoN@N. Returns the full report dict."""
    from harnesscad.agents.selftrain.train import require
    require()

    arms: Dict[str, E.ArmResult] = {}

    # -- base@1 (greedy) --------------------------------------------------- #
    gen = E.HFGenerator(base_model, adapter=None)
    arms["base@1"] = E.grade_solver("base@1", gen.solve_greedy, brief_ids, gens_per_brief=1)
    # -- Oracle-BoN@N off the SAME base model ------------------------------ #
    for n in bon_ns:
        solver = E.best_of_n_solver(gen, n, temperature=bon_temperature)
        arms["oracle-bon@%d" % n] = E.grade_solver(
            "oracle-bon@%d" % n, solver, brief_ids, gens_per_brief=n)
    _free(gen)

    # -- finetune@1 (greedy) ----------------------------------------------- #
    if adapter and os.path.exists(adapter):
        genf = E.HFGenerator(base_model, adapter=adapter)
        arms["finetune@1"] = E.grade_solver(
            "finetune@1", genf.solve_greedy, brief_ids, gens_per_brief=1)
        _free(genf)

    # -- pairwise McNemar --------------------------------------------------- #
    comparisons = []
    names = list(arms)
    if "finetune@1" in arms:
        comparisons.append(E.compare(arms["finetune@1"], arms["base@1"]))
        for n in bon_ns:
            key = "oracle-bon@%d" % n
            if key in arms:
                comparisons.append(E.compare(arms["finetune@1"], arms[key]))
    for n in bon_ns:
        key = "oracle-bon@%d" % n
        if key in arms:
            comparisons.append(E.compare(arms[key], arms["base@1"]))

    report = {
        "base_model": base_model, "adapter": adapter,
        "brief_ids": list(brief_ids), "n_briefs": len(brief_ids),
        "grader": "ledger.certify (gate AND envelope AND shape); labeller of the "
                  "training data, NEVER the verifier fleet",
        "bon_selector": "output gate (reference-free) -- the only oracle a user's "
                        "brief affords; envelope/shape need an answer key",
        "arms": {k: v.to_dict() for k, v in arms.items()},
        "comparisons": comparisons,
        "note": ("finetune@1 costs 1 generation/brief; oracle-bon@N costs N. If "
                 "finetune@1 does not beat oracle-bon@3, the fine-tune lost to "
                 "un-tuned Best-of-N at 3x inference and no training -- the book's "
                 "warning, realised."),
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="base vs finetune vs Oracle-BoN")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--bon", default="3,5")
    ap.add_argument("--out", default="assets/selftrain/eval_report.json")
    args = ap.parse_args(argv)
    rep = run_experiment(base_model=args.base, adapter=args.adapter,
                         bon_ns=[int(x) for x in args.bon.split(",")],
                         out_path=args.out)
    # Print just the headline table.
    for name, arm in rep["arms"].items():
        print("%-16s acc=%.3f  wilson95=%s  gate=%d/%d  gens=%d" % (
            name, arm["pass@1_or_selected"], arm["wilson95"],
            arm["gate_pass"], arm["n"], arm["generations"]))
    for c in rep["comparisons"]:
        print("  %s vs %s: dRate=%+.3f  McNemar p=%.3f (a_wins=%d b_wins=%d)" % (
            c["arm_a"], c["arm_b"], c["delta_rate"], c["mcnemar_p_exact"],
            c["a_wins"], c["b_wins"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
