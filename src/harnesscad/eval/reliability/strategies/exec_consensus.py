"""Oracle-free execution-signature self-consistency (MBR-exec) for CAD candidates.

Ported from PairCoder ``reproduction/bcb_consensus.py`` (PairCoder-main), which
re-selects among N already-generated code candidates WITHOUT a correctness
oracle: each candidate is probed by execution, its behavior signature recorded,
candidates are clustered by signature, and a representative of the LARGEST
agreeing cluster wins (self-consistency / minimum-Bayes-risk-exec). The source's
MONOTONE rule is kept verbatim in spirit:

  * candidate[0] is the baseline (the natural single-shot pick);
  * if the largest cluster contains candidate[0], keep candidate[0];
  * if the largest cluster is a singleton (no >=2 agreement anywhere), keep
    candidate[0] -- no agreement means no evidence to move;
  * candidates whose probe raised (or whose every probe raised) are ignored
    when clustering; if ALL candidates are like that, keep candidate[0].

  So the selector never regresses below baseline unless candidate[0] is
  provably outside a >=2-agreement cluster.

Adaptation to CAD: instead of executing docstring example calls and comparing
``repr`` of return values, candidates are probed by an injected ``measure``
callable returning GEOMETRIC signatures -- volume, bounding box, genus, face
count, or any other deterministic scalar/tuple probes. Two candidates agree
when their quantized geometric signatures match: same shape, same answer,
regardless of how differently the model wrote the construction.

Slots beside ``strategies/best_of_n.py``: Best-of-N needs the deterministic
verifier to rank candidates; exec-consensus needs NO verifier at all, only a
measurement channel, so it applies even when no ground truth or feasibility
signal exists.

Attribution: PairCoder (reproduction/bcb_consensus.py). Pure stdlib,
deterministic; no kernel, no model.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

Number = Union[int, float]
MeasureFn = Callable[[object], Mapping[str, object]]

#: Signature token prefix for a candidate whose measurement raised, mirroring
#: the source's "EXC:<ExceptionName>" behavior tokens.
_EXC_PREFIX = "EXC:"


# --------------------------------------------------------------------------- #
# signatures
# --------------------------------------------------------------------------- #

def quantize(value: object, rel_tol: float = 1e-4, abs_tol: float = 1e-9) -> object:
    """Quantize a probe value so nearly-equal geometry clusters together.

    Floats are snapped to a tolerance-scaled grid (relative for large values,
    absolute near zero); ints/bools/strings pass through; sequences quantize
    element-wise. Deterministic and total.
    """
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return repr(value)
        step = max(abs(value) * rel_tol, abs_tol)
        return round(value / step) * step if step > 0 else 0.0
    if isinstance(value, (list, tuple)):
        return tuple(quantize(v, rel_tol, abs_tol) for v in value)
    return str(value)


def geometric_signature(
    candidate: object,
    measure: MeasureFn,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-9,
) -> str:
    """Probe one candidate and return its behavior signature string.

    ``measure(candidate)`` must return a mapping of probe name -> value (e.g.
    ``{"volume": 1000.0, "bbox": (10, 10, 10), "genus": 0, "face_count": 6}``).
    A raising measure yields an ``EXC:<TypeName>`` signature, exactly as the
    source records exceptions instead of crashing the selector.
    """
    try:
        probes = measure(candidate)
    except Exception as exc:  # never let one bad candidate kill selection
        return _EXC_PREFIX + type(exc).__name__
    parts = []
    for key in sorted(probes):
        parts.append(f"{key}={quantize(probes[key], rel_tol, abs_tol)!r}")
    return "OK:" + ";".join(parts)


def is_exception_signature(sig: str) -> bool:
    return sig.startswith(_EXC_PREFIX)


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

@dataclass
class ConsensusResult:
    """Outcome of consensus selection over N candidates."""

    winner_index: int
    signatures: List[str] = field(default_factory=list)
    clusters: Dict[str, List[int]] = field(default_factory=dict)
    reason: str = ""

    @property
    def kept_baseline(self) -> bool:
        return self.winner_index == 0

    def to_dict(self) -> dict:
        return {
            "winner_index": self.winner_index,
            "signatures": list(self.signatures),
            "clusters": {k: list(v) for k, v in self.clusters.items()},
            "reason": self.reason,
            "kept_baseline": self.kept_baseline,
        }


def select_by_consensus(
    candidates: Sequence[object],
    measure: MeasureFn,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-9,
) -> ConsensusResult:
    """Pick the candidate index backed by the largest geometric-agreement cluster.

    Mirrors PairCoder's ``select()``: cluster candidates by execution
    signature, ignore all-exception candidates, take the largest cluster, and
    apply the MONOTONE rule (keep ``candidates[0]`` unless it is provably
    outside a >=2-agreement cluster; a singleton best cluster keeps baseline).
    Ties between equal-size clusters resolve to the cluster containing the
    lowest candidate index (deterministic; the source's dict ordering gives
    the same first-seen bias).

    Returns a :class:`ConsensusResult`; ``winner_index`` indexes ``candidates``.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")

    sigs = [geometric_signature(c, measure, rel_tol, abs_tol) for c in candidates]

    clusters: Dict[str, List[int]] = {}
    for i, sig in enumerate(sigs):
        if is_exception_signature(sig):
            continue  # source: crash/all-exception candidates are ignored
        clusters.setdefault(sig, []).append(i)

    if not clusters:
        return ConsensusResult(0, sigs, clusters,
                               "all candidates raised; keep baseline")

    # Largest cluster; tie -> the one containing the smallest index.
    best_sig = max(clusters, key=lambda s: (len(clusters[s]), -min(clusters[s])))
    best = clusters[best_sig]

    if len(best) == 1:
        return ConsensusResult(0, sigs, clusters,
                               "no >=2 agreement; keep baseline (monotone)")
    if 0 in best:
        return ConsensusResult(0, sigs, clusters,
                               "baseline inside majority cluster; keep baseline")
    return ConsensusResult(
        min(best), sigs, clusters,
        "baseline provably outside the majority cluster; "
        "switch to its lowest-index representative")


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execution-signature self-consistency selector for CAD "
                    "candidates (MBR-exec, PairCoder bcb_consensus.py port).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="run synthetic-candidate consensus scenarios and "
                             "assert the MONOTONE rule holds.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # Candidates are synthetic dicts standing in for finished CAD models; the
    # injected measure just reads their fields (no kernel).
    def measure(c):
        if c.get("broken"):
            raise RuntimeError("kernel would have failed")
        return {"volume": c["volume"], "bbox": tuple(c["bbox"]),
                "genus": c.get("genus", 0), "face_count": c.get("faces", 6)}

    box = {"volume": 1000.0, "bbox": (10.0, 10.0, 10.0)}
    box_jitter = {"volume": 1000.00004, "bbox": (10.0, 10.0, 10.000001)}
    wrong = {"volume": 500.0, "bbox": (10.0, 10.0, 5.0)}
    broken = {"broken": True}

    # 1. Majority cluster (jitter within tolerance) beats the wrong baseline.
    r = select_by_consensus([wrong, box, box_jitter, box], measure)
    assert r.winner_index == 1, r.to_dict()
    assert not r.kept_baseline
    print(f"[selfcheck] majority overrides bad baseline: winner={r.winner_index}")

    # 2. Baseline inside the majority cluster -> keep baseline.
    r = select_by_consensus([box, box_jitter, wrong], measure)
    assert r.winner_index == 0, r.to_dict()
    print("[selfcheck] baseline in majority kept (monotone)")

    # 3. All singleton clusters -> keep baseline.
    r = select_by_consensus([wrong, box, {"volume": 2.0, "bbox": (1, 1, 2)}], measure)
    assert r.winner_index == 0, r.to_dict()
    print("[selfcheck] singleton clusters keep baseline")

    # 4. Exception candidates are ignored, all-exception keeps baseline.
    r = select_by_consensus([broken, broken], measure)
    assert r.winner_index == 0 and not r.clusters, r.to_dict()
    r = select_by_consensus([broken, box, box], measure)
    assert r.winner_index == 1, r.to_dict()
    assert is_exception_signature(r.signatures[0])
    print("[selfcheck] exception candidates ignored; all-exception -> baseline")

    # 5. Determinism.
    a = select_by_consensus([wrong, box, box_jitter, box], measure).to_dict()
    b = select_by_consensus([wrong, box, box_jitter, box], measure).to_dict()
    assert a == b
    print("[selfcheck] deterministic across runs")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
