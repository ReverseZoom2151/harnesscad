"""The memory A/B on the CONTAMINATION-CONTROLLED corpus dev split.

WHY A SECOND CORPUS
-------------------
``eval/memory/ab.py`` runs on the pressure corpus. That corpus was written by the
system it scores (``assets/pressure/report.md``), so it is contaminated in the
same way v1 was. Contamination cuts one way for THIS experiment and it is worth
stating precisely: a contaminated corpus repeats near-duplicate briefs, which is
the single most favourable condition for a recall-based memory. If memory were
going to win anywhere, it would win there. So a negative result on the pressure
corpus is a STRONG negative, not a weak one -- but it can still be doubted, and
the honest answer to the doubt is a second, uncontaminated corpus.

``eval/corpus`` is exactly that: a dev/heldout split from independent formulae and
cited standards, with a grader (``corpus/grade.py``) whose ground truth was never
the harness's own opinion. This module runs the same ON-vs-OFF A/B against the
DEV split (17 briefs). The heldout split is deliberately NOT touched: its
isolation is enforced by a test, and a memory experiment is not a reason to spend
it.

FAITHFULNESS
------------
Unlike ``ab.py`` (which reads the pressure grader's ``gate_ok``), this module
drives the harness's OWN oracle: each candidate op stream is rebuilt on a fresh
FRep backend and measured by ``io/gate.py`` via ``gate_oracle`` -- the identical
instrument ``core/harness.AgentHarness._oracle_verdict`` uses in production. The
write gate here IS the shipping write gate. ``corpus.grade`` supplies ``solved``
for scoring only; it never reaches the prompt or the memory.

Deterministic and cached, exactly like ``ab.py``.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from harnesscad.agents.agent.planner import Planner
from harnesscad.agents.memory.harness_memory import (
    HarnessMemory,
    OracleVerdict,
    gate_oracle,
)
from harnesscad.core.loop import HarnessSession
from harnesscad.eval.corpus import dev
from harnesscad.eval.corpus.grade import grade as corpus_grade
from harnesscad.eval.corpus.spec import Brief as CorpusBrief
from harnesscad.eval.memory.ab import (
    ARM_OFF,
    ARM_ON,
    ABReport,
    ArmResult,
    BriefRun,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_SEED,
    _NoLLM,
    format_text,
)
from harnesscad.eval.pressure.cache import CompletionCache
from harnesscad.eval.pressure.model import (
    CachedClient,
    Client,
    OllamaClient,
    extract_ops,
    ops_to_dicts,
)


def _backend():
    from harnesscad.io.backends.frep import FRepBackend
    return FRepBackend()


def _parse(ops: Sequence[dict]) -> list:
    """Model-emitted op dicts -> CISP Op objects. Unparseable => dropped, so a
    malformed stream grades as invalid rather than raising."""
    from harnesscad.core.cisp.ops import parse_op

    out = []
    for d in ops:
        try:
            out.append(parse_op(d))
        except Exception:  # noqa: BLE001
            return []
    return out


def _model_facing(diagnostics: Sequence[Any]) -> List[dict]:
    """ERROR/WARNING diagnostics the planner's soundness gate would keep.

    Info-severity notes (an under-constrained sketch built exactly as asked) are
    not defects and are not spoken to the model."""
    out: List[dict] = []
    for d in diagnostics:
        dd = d.to_dict() if hasattr(d, "to_dict") else dict(d)
        sev = dd.get("severity", "error")
        sev = getattr(sev, "value", sev)
        if str(sev) in ("error", "warning"):
            out.append(dd)
    return out


def run_brief(
    client: Client,
    brief: CorpusBrief,
    planner: Planner,
    memory: Optional[HarnessMemory],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> BriefRun:
    """Plan -> apply -> gate -> (oracle-gated) remember -> grade, per attempt."""
    run = BriefRun(brief_id=brief.id, arm=(ARM_ON if memory else ARM_OFF))
    diagnostics: Optional[List[dict]] = None
    last_ops: Optional[List[dict]] = None

    for attempt in range(max_attempts):
        messages = planner.build_messages(brief.text, None, diagnostics)
        if attempt == 0 and planner.last_recalled is not None:
            run.recalled_episodes = len(planner.last_recalled.episodes)
            run.recalled_false_positives = len(planner.last_recalled.false_positives)

        raw = client.complete([m.to_dict() for m in messages], attempt)
        ops = ops_to_dicts(extract_ops(raw))
        run.attempts = attempt + 1
        last_ops = ops
        # corpus.grade and backend.apply want Op OBJECTS; the model emits dicts.
        # A dict that will not parse is an invalid op, and the attempt is graded
        # as such rather than crashing the run.
        op_objs = _parse(ops)

        # Rebuild on a fresh backend; measure with the SHIPPING oracle.
        session = HarnessSession(_backend())
        try:
            result = session.apply_ops(op_objs)
            all_diags = list(result.diagnostics)
            apply_ok = bool(result.ok)
        except Exception:  # noqa: BLE001
            all_diags, apply_ok = [], False

        verdict = gate_oracle(session, op_objs) if apply_ok else OracleVerdict(
            False, ("apply-failed",), "gate")

        if memory is not None:
            # The oracle verdict gates the write. `corpus_grade`'s `solved` (the
            # answer key) is NOT computed until after and never passed here.
            w = memory.commit(brief.text, ops, verdict,
                              fleet_diagnostics=all_diags, summary=brief.id)
            run.memory_admitted += int(w["admitted"])
            run.memory_refused += int(not w["admitted"])

        model_facing = _model_facing(all_diags)
        if apply_ok and not model_facing:
            break
        diagnostics = model_facing

    if last_ops is not None:
        g = corpus_grade(brief, _parse(last_ops))
        run.solved = bool(g.solved) and not g.unmeasurable
        run.solved_shape = bool(g.solved_shape) and not g.unmeasurable
        run.built = bool(g.built)
        run.reasons = list(g.reasons)
    return run


def run_arm(
    client: Client,
    briefs: Sequence[CorpusBrief],
    memory_on: bool,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    memory_factory: Optional[Callable[[], HarnessMemory]] = None,
) -> ArmResult:
    memory = (memory_factory or HarnessMemory)() if memory_on else None
    planner = Planner(_NoLLM(), use_tool=False, memory=memory)
    arm = ArmResult(model=getattr(client, "name", "client"),
                    arm=ARM_ON if memory_on else ARM_OFF)
    for brief in briefs:
        arm.runs.append(run_brief(client, brief, planner, memory, max_attempts))
    if memory is not None:
        arm.memory_stats = dict(memory.stats)
        arm.false_positive_counts = memory.false_positive_counts()
    return arm


def run(
    models: Sequence[str],
    seed: int = DEFAULT_SEED,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    cache_dir: str = ".memory_ab_corpus_cache",
    client_factory: Optional[Callable[[str], Client]] = None,
) -> ABReport:
    briefs = list(dev.BRIEFS)
    cache = CompletionCache(cache_dir)
    report = ABReport(seed=seed, max_attempts=max_attempts,
                      brief_order=[b.id for b in briefs])
    for name in models:
        base = (client_factory(name) if client_factory
                else OllamaClient(name, seed=seed, temperature=0.0))
        for memory_on in (False, True):
            client = CachedClient(base, cache, seed=seed, temperature=0.0)
            report.arms.append(
                run_arm(client, briefs, memory_on, max_attempts=max_attempts))
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="memory ON vs OFF A/B on the corpus dev split")
    ap.add_argument("--model", action="append")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    ap.add_argument("--cache-dir", default=".memory_ab_corpus_cache")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    models = args.model or ["qwen2.5-coder:7b"]
    report = run(models, seed=args.seed, max_attempts=args.max_attempts,
                 cache_dir=args.cache_dir)
    print("CORPUS DEV SPLIT (contamination-controlled) -- " + str(len(dev.BRIEFS))
          + " briefs")
    print(format_text(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
