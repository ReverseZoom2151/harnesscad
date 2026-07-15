# Memory A/B: status

The memory subsystem and its A/B apparatus are **complete and tested**. The result
below is **directional, not final**, because it was measured on a model lineup that
has since been removed from the machine.

## The apparatus

`agents/memory/harness_memory.py` is an oracle-gated experience store: a trajectory
is written to memory **only if the produced part passed the measured gate**
(`io/gate.py`). A false memory is a false instruction with a longer fuse, so an
unverified trajectory is never stored. It wires into `AgentHarness` and the planner
prompt as a **parameter** (memory on/off), not a separate loop.

`eval/memory/ab.py` and `ab_corpus.py` run the controlled comparison: same briefs,
same model, same seed, memory ON vs OFF — `ab_corpus.py` on the contamination-
controlled dev split, so a gain cannot come from a near-duplicate the store happened
to have seen.

## The finding, so far

On the deleted lineup, **memory HURT**, and it hurt *more* on the contamination-
controlled split than on the raw one — the opposite of what a near-duplicate
artifact would produce. That direction replicates Agent-S, which shipped an
experience-augmented memory as its headline contribution and then **deleted it** in
the version that set SOTA, because retrieved experience was net-negative against a
capable base model.

The magnitude is **not recorded here**, deliberately: it came from models that no
longer exist on this machine, and this project does not publish numbers from a
lineup it cannot reproduce. The A/B is parameterised by model list and re-runs on
the frontier lineup (`qwen3.6`, `ornith`) to reconfirm both direction and
magnitude.

## Why this is the expected result, not a failure

The whole project exists because a mechanism can be *assumed* to help and *measured*
to hurt — that is exactly what the pressure experiment found for typed diagnostics.
Memory is a second instance of the same discipline: it is gated, wired, measured,
and if the frontier re-run confirms it is net-negative, it stays **off by default**
and the negative result is the contribution.
