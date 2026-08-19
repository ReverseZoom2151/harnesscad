"""Curriculum ordering for training and evaluation task sequences.

Three of the four CAD-RL papers the harness tracks order their tasks
simple-to-complex, and all three measure a gain from doing so:

  * RLCAD (arXiv:2503.18549, sec 6.4) -- a 500-geometry training set ordered by
    increasing complexity, with complex models warm-started from simple ones.
    The curriculum arm beats case-by-case training on every metric reported
    (IoU 0.8757 / COV 0.8692 / MMD-CD 0.0139 / JSD 0.1111 / NC 0.8812 against
    0.8354 / 0.8544 / 0.0165 / 0.1307 / 0.8638).
  * ReCAD (arXiv:2512.06328) -- Hierarchical Primitive Learning: stage over
    P = {Loop, Face, Sketch, SketchExtrude, MultiSketchExtrude} and, within a
    stage, order by the number of curves. Ablating it raises both reconstruction
    error and failure rate. ReCAD also gates on MEASURED difficulty: a task is
    hard when the max reward over N sampled solutions falls below h = 0.8.

Layout
------
``complexity``  the metric -- a deterministic structural complexity score for a
                task's op stream (op count, distinct op types, curve count,
                constraint count, feature-tree depth, hardest op tier).
``ordering``    the API -- flat (RLCAD) and hierarchical (ReCAD) orderings, both
                total and reproducible, plus per-level stages and batches.
``difficulty``  the measurement -- a ledger of observed rewards, ReCAD's
                ``max reward < h`` hard-task selector, and a re-ordering that
                prefers what was measured over the structural proxy.
``validate``    the sanity check against the pressure corpus's hand-assigned
                ``difficulty`` column, with the ablation table.

Everything here is pure counting over op streams: no wall clock, no randomness,
no kernel evaluation, no network. The same collection always produces the same
sequence.
"""

from __future__ import annotations

from harnesscad.eval.curriculum import complexity, difficulty, ordering
from harnesscad.eval.curriculum.complexity import (
    ComplexityFeatures,
    task_features,
    task_score,
)
from harnesscad.eval.curriculum.difficulty import (
    HARD_THRESHOLD,
    DifficultyLedger,
    empirical_difficulty,
    order_by_measured,
)
from harnesscad.eval.curriculum.ordering import (
    batches,
    order_tasks,
    stages,
    structural_level,
)

__all__ = [
    "complexity",
    "difficulty",
    "ordering",
    "ComplexityFeatures",
    "task_features",
    "task_score",
    "structural_level",
    "order_tasks",
    "stages",
    "batches",
    "HARD_THRESHOLD",
    "DifficultyLedger",
    "empirical_difficulty",
    "order_by_measured",
]
