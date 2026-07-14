# HarnessCAD audited against *The Hitchhiker's Guide to Agentic AI* — H1 (Foundations) and H2 (RL Methods / Agentic Training / Evaluation)

Source read in full, no sampling:

* `resources/extracted_text/chunks/H1_foundations_llm_arch.txt` — 9,401 lines. Book Part I: Ch. 1 (LLM Architecture and Optimization), Ch. 2 (Systems Foundations), Ch. 3 (Introduction to RL). Lines 1–2,670 are the book's global table of contents; content starts at line ~2,700.
* `resources/extracted_text/chunks/H2_rl_methods_agentic_training_eval.txt` — 10,510 lines. Book Parts II, III, IV: Ch. 4–12 (RL Methods for LLMs), Ch. 13 (RL for Large Reasoning Models), **Ch. 14 (LLM Evaluation)**.

System under audit: HarnessCAD @ `C:/Users/adria/Downloads/redNoise-main/harnesscad`, 1,335 Python modules under `src/harnesscad/{core,domain,io,eval,agents,data,governance}`.

The trigger for this audit is `assets/pressure/report.md`: a controlled A/B in which the harness's typed-diagnostic loop **lost to blind resampling by 8.3 points** (blind 33.3% = 24/72, harness 25.0% = 18/72), and lost *monotonically harder on stronger models* (qwen 1.5b ±0, 3b ±0, 7b −8.3, 14b −25.0). Root cause: 23 verifiers were each unit-tested for "does it FIRE on bad input" and not one was tested for "does it STAY SILENT on good input". Recall was optimised; precision was never measured.

---

## 0. Verdict counts

**116 sections audited** (H1: 30, H2: 86).

| Verdict | H1 | H2 | Total |
|---|---:|---:|---:|
| PRESENT | 3 | 8 | **11** |
| PARTIAL | 5 | 14 | **19** |
| ABSENT | 6 | 30 | **36** |
| N/A | 16 | 34 | **50** |

Every N/A is justified in-row. The N/A count is high and legitimate: HarnessCAD trains no weights, ships no GPU code, and has zero runtime dependencies (`pyproject.toml:11` — `dependencies = []`). Whole chapters of the book (tokenization internals, Flash Attention kernels, FSDP/ZeRO, NVLink topology, speculative decoding) describe a machine HarnessCAD does not operate. What is **not** N/A, and where the damage is, is **Chapter 9 (Reward Model Training)**, **Chapter 12 (Agentic Training)** and **Chapter 14 (Evaluation)** — of the 21 sections in those three chapters, **1 is PRESENT, 6 are PARTIAL, 14 are ABSENT, 0 are N/A.**

---

## 1. Coverage matrix

### H1 — Chapter 1: LLM Architecture and Optimization Methods (17 sections)

| # | Section (book's own name) | Verdict | Where in HarnessCAD (file:line) | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 1 | 1.1 How LLMs Work: An Intuitive Overview | N/A | — | Conceptual preamble; nothing to implement. | — | — |
| 2 | 1.2 Tokenization | N/A | — | HarnessCAD calls hosted/Ollama models through `agents/llm/litellm_backend.py`; it never owns a tokenizer. | — | — |
| 3 | 1.3 The Transformer Architecture | N/A | — | No model is defined or trained. | — | — |
| 4 | 1.4 Prediction Heads: What Transformers Output | N/A | — | No value head, no reward head, because no gradient ever flows. See §3 for why this is the strategic gap, not a code gap. | — | — |
| 5 | 1.5 Optimization Theory for LLM Training | N/A | — | No optimizer. | — | — |
| 6 | 1.6 Flash Attention — Algorithm and Hardware Awareness | N/A | — | Kernel-level; no training. | — | — |
| 7 | 1.7 Pretraining: Best Practices | N/A | — | Out of scope by design. | — | — |
| 8 | 1.8 Supervised Fine-Tuning (SFT) | ABSENT | `data/dataengine/selftrain/bootstrap_loop.py`, `data/pipeline.py:1154` | The book: "*SFT quality determines the RL ceiling*" (H1 §1.8.5; H2 §10.6). HarnessCAD has a data engine that can emit SFT-shaped records and a `selftrain` package, but no SFT is ever run. The differential oracle can filter correct trajectories for free. | Rejection-Sampling Fine-Tuning (RFT) on oracle-verified op streams. See §3.1. | HIGH |
| 9 | 1.9 LoRA and Parameter-Efficient Fine-Tuning | ABSENT | — | Same as above: the cheapest possible entry point to a trained CAD policy (QLoRA on a 7B, ~1 GPU) is not attempted. | Enabler for §3, not standalone. | MED |
| 10 | 1.10 Mixture of Experts (MoE) | N/A | — | Architecture choice of a model HarnessCAD does not own. | — | — |
| 11 | 1.11 Diversity in LLM Training | PARTIAL | `eval/pressure/runner.py:52-71` (`seed`, `temperature=0.0`); `agents/exploration/designspace_sampler.py`, `latin_hypercube.py`, `variation.py` | The pressure test runs at **temperature 0.0, one seed**. The book (H1 §1.11.1, H2 §7.5.1) is emphatic that diversity is what makes a group of samples informative; at T=0 every "attempt" after a failure is a near-deterministic re-roll, so the blind arm's 33.3% is a *lower bound* on what blind resampling can do. Design-space samplers exist but are not used by the pressure harness. | Re-run the A/B at T∈{0.0, 0.7} × ≥3 seeds. This is also what gives you variance (see #101). | **HIGH** |
| 12 | 1.12 Text Generation: Decoding Methods | PARTIAL | `core/grammar_fsa.py`, `core/grammar.py`, `agents/llm/structured.py` | Constrained/grammar-guided decoding (H1 §1.12.11) is exactly the right answer for op-stream generation, and the FSA exists — but the pressure loop parses free text and counts parse failures (`eval/pressure/metrics.py:227-235` `parse_ok`, `parse_error`). The book: "*Use constrained decoding whenever the consumer of the model's output is a program rather than a human.*" | Route the pressure/showcase loops through the grammar FSA so `invalid_ops` goes to zero and the A/B measures geometry, not JSON. | MED |
| 13 | 1.13 Prompt Engineering | PRESENT | `agents/agent/system_prompt.py`, `eval/pressure/prompts.py`, `agents/context/exemplar_prompt.py`, `agents/generation/prompt_evolution.py` | Roles, constraints, structured output, few-shot, prompt-evolution are all implemented. | — | — |
| 14 | 1.14 Model Compression Methods | N/A | — | No weights. | — | — |
| 15 | 1.15 Speculative Decoding Methods | N/A | — | Inference-engine concern. | — | — |
| 16 | 1.16 Hallucination Detection | PARTIAL | `eval/selftest/differential.py`; `agents/exploration/variant_consensus.py`; `eval/bench/judges/*` | The book's model-level detectors (semantic entropy, SelfCheckGPT, consistency sampling — H1 §1.16.2) are the *weak* form of what HarnessCAD has: the differential oracle is a hard consistency check across six independent engines. But it checks *engines*, never *the model's own samples*. Self-consistency over N model samples of the same brief is free and unused. | Add self-consistency (majority vote over K op streams) as a baseline arm in the pressure test. It is also a *control* — see the indictment, §2.5. | MED |
| 17 | 1.17 LLM Safety and Responsible AI | PARTIAL | `governance/security/`, `eval/reliability/guardrails.py`, `io/gate.py` | Output gating and guardrails exist. The book's over-refusal metric (H1 §1.17.5: "*Measure false-positive refusals on benign prompts (target <5%)*") is **the exact metric the harness needed and did not have** — the fleet's false-positive rate on known-good parts. That is now `eval/selftest/fleet_audit.py:260-262` (`false_positive_rate`), which is the single most valuable thing in the repo. Not wired to a gate. | Make `false_positive_rate == 0` a release gate. | **HIGH** |

### H1 — Chapter 2: Systems Foundations for LLMs (2 sections)

| # | Section | Verdict | Where | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 18 | 2.1 GPU Architecture — From Silicon to LLM Training | N/A | — | HarnessCAD is CPU/stdlib. | — | — |
| 19 | 2.2 vLLM — PagedAttention and High-Throughput Inference | N/A | — | Inference is delegated to Ollama/LiteLLM. If §3 (RL training) is ever pursued, vLLM becomes load-bearing (generation is 60–70% of RLHF wall-clock, H2 §5.5) — but it is N/A *today*. | — | — |

### H1 — Chapter 3: Introduction to Reinforcement Learning (11 sections)

| # | Section | Verdict | Where | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 20 | 3.1 The Markov Decision Process (MDP) | PARTIAL | `core/state/opdag.py`, `core/loop.py:44-55`, `agents/agent/tool_trajectory.py` | The harness *is* an MDP: state = OpDAG + backend geometry, action = an op, transition = deterministic apply, reward = the oracle. It is nowhere written down as one. The book's productivity-copilot (H2 §12.6.2) and research-agent (§12.7.2) case studies both begin by formalising the MDP; HarnessCAD never did, which is why the reward machinery it has is unconnected to the loop it runs. | Write the MDP down once (`docs/`), then wire `tool_reward` to it. | MED |
| 21 | 3.2 Core Concepts and Definitions (return, value, advantage, Bellman) | ABSENT | `data/dataengine/reward/expert_advantage.py` (formula only) | Advantage is computed by a standalone function nothing calls. | See §3. | LOW |
| 22 | 3.3 Taxonomy of RL Methods | N/A | — | Expository. | — | — |
| 23 | 3.4 Temporal Difference (TD) Learning | N/A | — | No value function; no bootstrap. | — | — |
| 24 | 3.5 Q-Learning | N/A | — | The book itself says Q-learning is intractable for LLMs (H1 §3.5, "Why Q-Learning Fails for LLMs": 32K–128K action space). | — | — |
| 25 | 3.6 Policy Gradient Methods — REINFORCE | N/A | — | No policy parameters. | — | — |
| 26 | 3.7 Actor-Critic Methods | N/A | — | No critic. | — | — |
| 27 | 3.8 Generalized Advantage Estimation (GAE) | N/A | — | Requires a value head. | — | — |
| 28 | 3.9 On-Policy vs Off-Policy — Detailed Comparison | ABSENT | — | Relevant the moment §3 is pursued: the harness's replay buffer would be `data/dataengine/selftrain/hard_negative_buffer.py`, which exists and is unused. | — | LOW |
| 29 | 3.10 Model-Based vs Model-Free | N/A | — | — | — | — |
| 30 | **3.11 Reward Shaping** | **ABSENT — and this is the section that predicted the failure** | `core/loop.py:32-40` (fleet advisory, `fleet_blocking=False`); `eval/verifiers/precheck.py` | See the indictment, §2.1. The book's warning about naive shaping is verbatim the harness's failure mode. | Potential-Based Reward Shaping, or drop the typed channel where the rule is unproven. | **CRITICAL** |

---

### H2 — Chapter 4: RL Foundations for Language Models (4 sections)

| # | Section | Verdict | Where | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 31 | 4.1 Two Paradigms for RL in LLMs (RLHF/DPO vs RLVR) | PARTIAL | `eval/selftest/differential.py`, `eval/selftest/golden.py`, `io/gate.py` | HarnessCAD sits squarely in **Paradigm 2, RLVR** — it owns a verifiable, human-label-free correctness signal — and does not know it. The book: "*the reward comes not from human preferences but from verifiable outcomes… This paradigm — RL from Verifiable Rewards (RLVR) — is now the dominant approach for building reasoning models and agentic systems.*" (H2 lines 33–41). HarnessCAD has the reward and no RL. | See §3. | **HIGH** |
| 32 | 4.2 Text Generation as an MDP | ABSENT | — | See #20. | — | LOW |
| 33 | 4.3 The RLHF Pipeline | ABSENT | — | Stages 1–2 are replaced by a verifier in RLVR (H2 lines 89–91) — which is exactly what HarnessCAD has. Stage 3 (optimisation) and Stage 4 (**Evaluation and Iteration**) are both missing. | — | HIGH |
| 34 | 4.4 Roadmap of This Part | N/A | — | Expository. | — | — |

### H2 — Chapter 5: PPO (9 sections)

| # | Section | Verdict | Where | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 35 | 5.1 Motivation and History | N/A | — | Expository. | — | — |
| 36 | 5.2 The Clipped Objective | ABSENT | `data/dataengine/reward/clipped_policy_objective.py` | The clipped surrogate is implemented as a pure function with a unit test and **zero callers**. A formula without a policy is a fossil. | — | LOW |
| 37 | 5.3 Full PPO Loss | ABSENT | — | No value loss, no entropy bonus. | — | LOW |
| 38 | 5.4 Derivation of the PPO Gradient and Update Rule | N/A | — | Derivation. | — | — |
| 39 | 5.5 Rollout Buffer and Rollouts | PARTIAL | `agents/agent/tool_trajectory.py`, `data/dataengine/selftrain/hard_negative_buffer.py`, `eval/pressure/cache.py` | HarnessCAD *does* collect trajectories (`.pressure_cache` holds every model call from the A/B). They are cached for reproducibility and then thrown away. The book calls this a Trajectory Buffer (H2 §12.2) and it is the raw material of every method in §3. | Persist pressure/showcase rollouts as a labelled trajectory corpus. **This is free — you already generated it.** | **HIGH** |
| 40 | 5.6 PPO for RLHF: The Full Loop | ABSENT | — | — | — | LOW |
| 41 | 5.7 Detailed Mechanics: Logits and Policy Updates | N/A | — | — | — | — |
| 42 | 5.8 TRL Implementation | ABSENT | `pyproject.toml:11` (`dependencies = []`) | No `trl`, no `torch`, no `transformers`. A deliberate, defensible choice for a *harness* — a fatal one for a *trainer*. | — | MED |
| 43 | 5.9 Critical Hyperparameters | N/A | — | — | — | — |

### H2 — Chapter 6: DPO (9 sections)

| # | Section | Verdict | Where | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 44 | 6.1 Motivation | N/A | — | Expository. | — | — |
| 45 | 6.2 Mathematical Derivation | N/A | — | — | — | — |
| 46 | 6.3 Gradient Analysis | N/A | — | — | — | — |
| 47 | 6.4 TRL Implementation | ABSENT | `data/dataengine/preference/dpo_pairs.py` | **Preference-pair construction exists.** The oracle can label any two op streams for the same brief (one solves, one does not) into a `(chosen, rejected)` pair *with no human*. `dpo_pairs.py` builds them. Nothing consumes them. This is the single cheapest training axis available (H2 Table 8.1: DPO = "*2 models, offline pairs, low compute, high stability*"). | See §3.2. | **HIGH** |
| 48 | 6.5 How DPO Works: Full Mechanics | N/A | — | — | — | — |
| 49 | 6.6 DPO Variants and When Each Fails | PARTIAL | — | Relevant *pre-emptively*: the book's failure mode #4 is "*Data quality: Noisy labels poison training. Unlike PPO which averages over many samples, DPO memorizes individual pairs.*" (H2 lines 1129–1130). If HarnessCAD builds DPO pairs from **fleet verdicts** rather than from the **oracle**, it will train a model on the washer bug. | Pairs must be labelled by `selftest.golden` + `selftest.differential`, **never** by the verifier fleet. | **HIGH** |
| 50 | 6.7 β Selection Guide | N/A | — | — | — | — |
| 51 | 6.8 DPO Batch Size Configuration and Scaling | N/A | — | — | — | — |
| 52 | 6.9 DPO Extensions and Variants (f-DPO, Robust DPO, TR-DPO, EXO, NCA, SLiC-HF, Iterative RPO, SimPO) | PARTIAL | — | Two are directly relevant and neither is used: **Robust DPO** (H2 §6.9.2) analytically debiases a *known label-noise rate* ε — and HarnessCAD can now *measure* its own label-noise rate, per rule, from `fleet_audit.py`. **Iterative RPO** (§6.9.7) adds an NLL term so the model can still *generate*, which matters for op streams. | If DPO is pursued, use `loss_type="robust"` with ε set from the measured fleet FP rate. | MED |

### H2 — Chapter 7: GRPO (5 sections)

| # | Section | Verdict | Where | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 53 | 7.1 Motivation | N/A | — | — | — | — |
| 54 | 7.2 Algorithm | ABSENT | `data/dataengine/reward/expert_advantage.py` | Group-relative advantage `(r − μ)/σ` is implemented; no group is ever sampled with intent to learn. | See §3.3. | **HIGH** |
| 55 | 7.3 TRL Implementation | ABSENT | — | — | — | MED |
| 56 | 7.4 Group Size Analysis | ABSENT | — | The book's **Goldilocks rule** (H2 lines 1827–1831): "*If all G responses are correct… all advantages = 0, no learning signal! … Filter prompts to 20–80% pass rate for current model.*" HarnessCAD **already did this by hand** and wrote it down: `assets/pressure/report.md:129-131` — "*a brief that every model solves in one attempt measures nothing. Kept 7 plain and all 5 traps.*" That is the Goldilocks rule, discovered independently, applied to eval, and never carried into training because there is no training. | — | MED |
| 57 | 7.5 GRPO Variants and Extensions (Diversity, DAPO, GSPO, Dr. GRPO, 2-GRPO, SAPO, TIS/MIS, VESPO, DPPO, ScaleRL/CISPO, GDPO, GOPO) | PARTIAL | `data/dataengine/reward/asymmetric_clip.py` (= DAPO Clip-Higher), `overlong_filter.py` (= DAPO Overlong Filtering), `clipped_policy_objective.py` | **Three of DAPO's five components are already implemented as formulas.** They are orphans. This is the clearest single illustration of the repo's condition: the *hard* parts of a 2025 RL recipe are written, tested and unreachable, because the *easy* part — a training loop — was never built. | — | MED |

### H2 — Chapter 8: Preference Optimization Variants (6 sections)

| # | Section | Verdict | Where | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 58 | 8.1 Online DPO | ABSENT | — | — | — | LOW |
| 59 | 8.2 KTO — Kahneman-Tversky Optimization | ABSENT | `data/dataengine/preference/kto.py`, `binary_preferences.py` | KTO needs only *unpaired binary* labels — "this op stream is good / bad" — which the differential oracle emits for free, at scale, for any op stream ever generated. `kto.py` exists. It is an orphan. The book: KTO is "*more robust than DPO to noise*" (H2 line 2952). | See §3.2. | **HIGH** |
| 60 | 8.3 IPO — Identity Preference Optimization | ABSENT | — | The book recommends IPO over DPO for "*noisy preference data*" and "*AI-judged with errors*" (H2 §8.3.4). Given a measured fleet FP rate, this is the honest default. | — | MED |
| 61 | 8.4 ORPO | ABSENT | — | — | — | LOW |
| 62 | **8.5 Best-of-N Sampling (Rejection Sampling)** | **PARTIAL — and the book hands us our own indictment here** | `agents/exploration/tournament.py`, `eval/quality/reward/pareto.py`; NOT in `eval/pressure/` | Best-of-N with the differential oracle as the reward model is the strongest thing HarnessCAD could ship *this week*, needs no training, and is **not implemented in the loop**. Worse: the book states flatly (H2 lines 3228–3230): "***Always compare your RL method against Best-of-N with the same compute budget.*** *If PPO with 64 GPU-hours doesn't beat Best-of-N with 64 GPU-hours of generation, your PPO has a bug.*" The pressure test compares harness-vs-blind at **equal attempt budget** — good instinct — but never ran the Best-of-N arm, which is the one arm that would have shown whether *any* feedback beats *more samples*. | Add a `--loop bon` arm: N samples, pick the one the oracle accepts. See §2.5. | **CRITICAL** |
| 63 | 8.6 Summary: Choosing an Alignment Method | PARTIAL | — | The book's decision tree (H2 lines 3301–3308) resolves for HarnessCAD in one step: "*1. Do you have verifiable rewards? (math/code) → GRPO*". Yes. Six engines. | — | — |

### H2 — Chapter 9: Reward Model Training (7 sections)

| # | Section | Verdict | Where | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 64 | 9.1 Bradley-Terry Model — Full Derivation | PARTIAL | `agents/exploration/elo.py`, `agents/exploration/tournament.py` | Elo/BT ranking machinery exists for design-space exploration; never used to rank *models* or *loops* in the pressure test, where it would give a principled leaderboard with CIs (see #106). | — | LOW |
| 65 | 9.2 Reward Model Architectures | N/A | — | No learned RM — by design; the oracle is exact. This is a strength (§5). | — | — |
| 66 | 9.3 Reward Model Training Tricks (centering, length bias, margin) | N/A | — | No learned RM. | — | — |
| 67 | **9.4 Process Reward Models vs Outcome Reward Models** | **PARTIAL — the machinery is built and unused** | `agents/agent/tool_reward.py:79-95` (`R = α·R_ORM + β·mean(R_step) + γ·R_format`); `tool_reward.py:51-61` (`step_execution_rewards`, `mean_step_reward`) | The per-step (process) reward is implemented, exactly as the book's PRM prescribes, and its **only importers are `agents/registry.py:239` (a dispatch table) and its own unit test**. `grep -rn "tool_reward"` returns four hits and not one of them is a loop. HarnessCAD is running **outcome-only supervision** on a 3–8-step trajectory while carrying a finished process-reward implementation. The book's PRM/ORM table (H2 lines 3455–3478) is explicit: ORM credit assignment is "*Sparse*", PRM is "*Dense*"; ORM is "*Easier to hack*". | Wire `tool_reward.aggregate_reward` into `core/loop.py` and log `R_step` per op in `core/trace.py`. See §2.7. | **CRITICAL** |
| 68 | **9.5 Rule-Based Rewards for RLVR** | **PARTIAL — and the book's pitfall list is the harness's bug list** | `eval/verifiers/registry.py:1-40`, `eval/verifiers/precheck.py`, `eval/quality/reward/executability_reward.py` | See the indictment, §2.2. The book names four "Rule-Based Reward Pitfalls" (H2 lines 3591–3598) and HarnessCAD hit two of them head-on. | — | **CRITICAL** |
| 69 | 9.6 Multi-Objective Rewards — Combination Strategies | PARTIAL | `eval/quality/reward/composite_reward.py:1-30` (weighted sum + **hard gates**) | A gated weighted-sum aggregator exists, including the "code that does not run earns nothing" gate the book endorses. Orphan — used by nothing but its test. The book warns (§7.5.11, GDPO) that a naive weighted sum lets a high-variance reward dominate; `composite_reward` does not normalise per-component. | Normalise-then-sum if it is ever used. | MED |
| 70 | 9.7 Listwise Rank-Based Rewards (Plackett-Luce, ListMLE) | ABSENT | `agents/exploration/elo.py` (pairwise only) | Low priority — HarnessCAD's rewards are *verifiable*, and the book says magnitudes are meaningful there (§7.5.12, GOPO: "*Use GRPO: When rewards are verifiable and exact… Magnitudes carry meaningful information.*"). Listwise is for noisy learned RMs. | — | LOW |

### H2 — Chapter 10: SFT Best Practices (6 sections)

| # | Section | Verdict | Where | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 71 | 10.1 Sequence Packing for Efficiency | N/A | — | No training. | — | — |
| 72 | 10.2 Chat Templates and Formatting | PARTIAL | `agents/llm/litellm_backend.py`, `agents/agent/system_prompt.py` | Templates are applied at inference. | — | LOW |
| 73 | 10.3 Completion-Only Masking | N/A | — | No training. | — | — |
| 74 | 10.4 Data Mixing Strategies for Multi-Task SFT | PARTIAL | `data/dataengine/curation/*` (13 modules: complexity tiers, curriculum, dedup, task routing) | A full curriculum/mixing engine exists and feeds no trainer. | — | LOW |
| 75 | 10.5 When SFT Hurts — Catastrophic Forgetting and Alignment Tax | N/A | — | No training. | — | — |
| 76 | **10.6 Connection to RL — SFT Quality Determines RL Ceiling** | ABSENT — but **the diagnostic is in the pressure data already** | `assets/pressure/report.md:18-24` | The book gives a go/no-go test before any RL (H2 lines 4259–4301): "*If pass@1 < 5%, RL will likely fail. If pass@k < 20%, RL will struggle.*" HarnessCAD's own numbers answer it: qwen-14b **pass@1 = 66.7%** blind. That is not "RL will struggle" territory — that is the *ideal* GRPO regime (20–80%, the Goldilocks band, #56). The pressure report is, unknowingly, a **positive RL-viability report**. | State this explicitly and use it as the case for §3. | **HIGH** |

### H2 — Chapter 11: System Architecture & Infrastructure at Scale (15 sections)

| # | Section | Verdict | Where | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 77 | 11.1 The 4-Model Memory Challenge | N/A | — | No training. | — | — |
| 78 | 11.2 Parallelism Strategies in Detail (DP/DDP, TP, SP, PP, FSDP, 3D) | N/A | — | No training. | — | — |
| 79 | 11.3 The Generation Bottleneck: Quantitative Analysis | N/A | — | — | — | — |
| 80 | 11.4 Decoupled Architecture: Production Design | N/A | — | — | — | — |
| 81 | 11.5 Weight Synchronization Strategies | N/A | — | — | — | — |
| 82 | 11.6 Memory Optimization Techniques | N/A | — | — | — | — |
| 83 | 11.7 Fault Tolerance at Scale | PARTIAL | `eval/pressure/cache.py`, `eval/pressure/runner.py:80-104` (resumable, checkpointed) | The pressure runner is resumable and cached — the right instinct at the right scale. | — | — |
| 84 | 11.8 End-to-End Latency Breakdown | PARTIAL | `eval/bench/harness/latency_speedup.py`, `agent_cost.py`, `eval/pressure/metrics.py:235` (`seconds`) | Timing is recorded; the A/B does not report cost-normalised results. The book (§8.5.4) insists methods be compared at **equal compute**, not equal attempts. The harness arm used **40% more model calls** (1.81 vs 1.29 attempts, `assets/pressure/report.md:24`) and still lost — which makes the result *worse* than the headline, and this is not stated. | Report solve-rate-per-model-call. | MED |
| 85 | 11.9 Monitoring and Observability | PARTIAL | `core/trace.py`, `core/observe.py`, `eval/reliability/repair_metrics.py` | Tracing exists. The book's RLHF watch-list (H2 lines 5140–5150) — reward, KL, length, entropy — has no analogue because there is no policy. | — | LOW |
| 86 | 11.10 Network Topology and Communication Patterns | N/A | — | — | — | — |
| 87 | 11.11 Training Throughput and Model FLOPs Utilization (MFU) | N/A | — | — | — | — |
| 88 | 11.12 Cost Analysis and Cloud Deployment | PARTIAL | `eval/bench/harness/agent_cost.py`, `resource_tradeoff.py` | Modules exist; not used in the A/B. | — | LOW |
| 89 | 11.13 Distributed Checkpointing | N/A | — | — | — | — |
| 90 | 11.14 Hardware Selection Guide | N/A | — | — | — | — |
| 91 | 11.15 Optimizer Configuration for RL Training | N/A | — | — | — | — |

### H2 — Chapter 12: LLM Agentic Training (8 sections)

| # | Section | Verdict | Where | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 92 | 12.1 Motivation: From Chatbots to Autonomous Agents | PRESENT | `core/loop.py`, `agents/agent/runner.py`, `core/harness.py` | HarnessCAD *is* the book's Figure 12.1(b): multi-step, tool-using, external-environment reward. | — | — |
| 93 | **12.2 Trajectory Buffers for LLM Agents** | PARTIAL | `agents/agent/tool_trajectory.py`; `eval/pressure/cache.py`; `data/dataengine/selftrain/hard_negative_buffer.py` | The buffer type is defined. A **hard-negative buffer** is implemented. The pressure run produced 72 × 2 × ~1.5 = **~220 real, graded, model-generated trajectories** and they live in a cache keyed for reproducibility, not in a corpus keyed for learning. | Promote `.pressure_cache` → a versioned trajectory corpus with oracle labels. Zero new compute. | **HIGH** |
| 94 | 12.3 Operational Paradigms (STaR/Reflexion; Off-Policy Exploration; RAG-over-Experiences) | PARTIAL | `agents/memory/error_notebook.py`, `agents/memory/skills.py`, `agents/rag/*`, `data/dataengine/selftrain/self_improvement.py` | Reflexion-style memory (error notebook) and Voyager-style skill libraries exist. STaR (paradigm A) — generate → filter by oracle → SFT — is the paradigm HarnessCAD is *one trainer away* from. | — | HIGH |
| 95 | 12.4 Paradigm Comparison | N/A | — | Table. | — | — |
| 96 | 12.5 Major Techniques in Agentic RL (STaR, Reflexion, ReAct, LATS, AgentQ, Voyager, RLEF, OpenHands) | PARTIAL | ReAct: `agents/agent/runner.py`. Voyager skills: `agents/memory/skills.py`. Reflexion: `agents/memory/error_notebook.py`. LATS/MCTS: **absent**. AgentQ (DPO on trajectories): **absent**. RLEF: **absent**. | **RLEF is HarnessCAD** (H2 §12.5.7): "*RL from Execution Feedback — binary reward from code/test execution*", and the book's justification is written as if about this repo: "***Zero noise:*** *Unlike human preferences, test results are deterministic… **No reward hacking:** Unlike learned reward models, a test suite can't be 'fooled'*" (H2 lines 6694–6701). HarnessCAD's test suite is six geometry kernels. It runs it. It does not learn from it. | See §3. | **HIGH** |
| 97 | 12.6 Use Case: Agentic RL for a Productivity Co-pilot | N/A | — | Worked example of another domain. Its **§12.6.9 Safety/Guardrails** and **§12.6.10 Credit Assignment** are cross-referenced below and are *not* N/A. | — | — |
| 98 | **12.6.10 Credit Assignment in Multi-App Workflows** (sub-section, audited separately because it is the mechanism the harness needs) | **ABSENT** | `agents/agent/tool_reward.py` (per-step reward exists, unused); `core/trace.py` | The book's hierarchical decomposition (H2 lines 7227–7255): sub-goal detection → sub-goal rewards → **trajectory slicing** ("*If the final task fails, identify which sub-goal failed first. Apply negative reward only to the actions within that sub-goal's span*"). This is precisely what a CAD op-stream needs — a 6-op plan that fails should not condemn ops 1–4. HarnessCAD grades the *final solid* (`eval/pressure/metrics.py:183-219`) and attributes nothing. | Log per-op `R_step` in the trace; grade the *first divergent op*, not the final part. | **HIGH** |
| 99 | 12.7 Use Case: Building a Research Agent from Scratch | N/A | — | Worked example. Its reward-hacking risk list (H2 lines 7545–7552) is cross-referenced in §2.4. | — | — |
| 100 | 12.8 State-of-the-Art RL for LLM Agents (GRPO for agents; PPO; **Fine-Grained Turn-Level Credit Assignment**; RWML; LATS) | **ABSENT** | — | §12.8.3 "**Multi-Turn Trajectory Slicing**" (H2 lines 7901–7908) is #98 again, stated as SOTA. And **RWML** (§12.8.4) is directly on point: "*To combat reward hacking, train agents to predict the semantic consequence of their actions.*" A CAD agent predicting the bbox/volume its op will produce, then being scored on the prediction, is a natural fit and is nowhere in the repo. | — | MED |

### H2 — Chapter 13: RL for Large Reasoning Models (9 sections)

| # | Section | Verdict | Where | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 101 | 13.1 Motivation and Background | N/A | — | Expository. | — | — |
| 102 | **13.2 Test-Time Scaling Methods** (CoT, Self-Consistency, ToT, GoT, Best-of-N, MCTS, Beam Search, Iterative Refinement) | **PARTIAL — and this is the cheapest unclaimed win in the repo** | `agents/generation/code_tree_control.py`, `agents/exploration/greedy_refine.py`, `agents/exploration/tournament.py`, `eval/reliability/repair_loop.py` | The pressure test evaluates exactly **one** point on the book's test-time-compute spectrum (iterative refinement with feedback — §13.2.8) against **one** degenerate baseline (blind resample). It never ran **Self-Consistency** (§13.2.2 — free, fully parallel, no reward model, "*CoT = 56.5%, Self-Consistency (N=40) = 74.4%*"), never ran **Best-of-N with the oracle as the reward model** (§13.2.5 — HarnessCAD has a near-*perfect* reward model, which is the exact regime where the book says BoN scales best: "*p = 0.3, N = 10: success = 97%*"), and never ran **ToT/MCTS**. The harness's headline claim was tested against the weakest possible alternative and still lost. | Add Self-Consistency and Oracle-BoN arms to `eval/pressure/loops.py`. | **CRITICAL** |
| 103 | 13.3 DeepSeek-R1 | PARTIAL | — | R1's core lesson (H2 lines 8608–8613, "**No Process Reward Model**"): outcome-only rewards *sufficed* because the verifier was exact, and the authors argue "*PRMs introduce their own failure modes (reward hacking at the step level)*". This is a genuine **counter-argument to #67**, and it deserves stating: HarnessCAD's outcome-only grading may be *right*. But R1's outcome reward was **exact**; HarnessCAD's *feedback channel* is not, which is a different axis and is the one that broke. | — | MED |
| 104 | 13.4 OpenAI o1/o3 Series | N/A | — | Proprietary; nothing actionable. | — | — |
| 105 | 13.5 QwQ and Qwen Reasoning Models | PARTIAL | — | §13.5.2 "**Rejection Sampling + RL Combination**" is the concrete recipe most suited to HarnessCAD: sample N, keep the oracle-correct ones, SFT, repeat. | See §3.1. | HIGH |
| 106 | 13.6 Key Methods with Mathematical Foundations (MCTS, PRMs/Math-Shepherd, ORM + majority voting, Self-Play, **RLVR**, Journey Learning, Quiet-STaR) | PARTIAL | `eval/selftest/*` = the RLVR verifier | §13.6.2 **Math-Shepherd** is the answer to "who labels the steps": *automatic* PRM annotation by Monte-Carlo rollout — "*a step s_k is labeled correct if there exists a completion from s_k that reaches the correct answer*". HarnessCAD can do this **exactly**, not statistically: roll the op stream forward from op k and ask the oracle. This is a free, exact PRM. Nothing does it. §13.6.6 **Journey Learning** — train on failed attempts and corrections, not just successes — describes the ~220 trajectories sitting in `.pressure_cache`. | — | **HIGH** |
| 107 | 13.7 Scaling Laws for Reasoning | ABSENT | — | The pressure test varied model size (1.5b→14b) and found the harness's *loss* scales with capability. That is a scaling law, discovered and correctly reported (`assets/pressure/report.md:35-42`). It is not framed as one. | — | LOW |
| 108 | 13.8 Comparison of Reasoning Models | N/A | — | Table. | — | — |
| 109 | 13.9 Summary and Open Problems | N/A | — | — | — | — |

### H2 — Chapter 14: LLM Evaluation (8 sections) — **the chapter the audit exists for**

| # | Section | Verdict | Where | Gap | Fix | Priority |
|---|---|---|---|---|---|---|
| 110 | 14.1 Evaluation Scheme Design (Intrinsic vs Extrinsic; Automatic vs Human; Reference-based vs Reference-free) | PARTIAL | `eval/bench/registry.py`, `eval/quality/registry.py`, `eval/pressure/*` | HarnessCAD has ~200 intrinsic metric modules and **one** extrinsic-ish test (`pressure`). The book: "*Intrinsic metrics are cheap and reproducible but often poorly correlated with real-world utility… A mature evaluation strategy uses intrinsic metrics for rapid iteration and extrinsic metrics for final validation.*" (H2 lines 9391–9395). HarnessCAD has never measured whether its 200 intrinsic metrics correlate with the one thing it cares about. `eval/bench/harness/metric_correlation.py` **exists** and is imported only by its own test. | Run `metric_correlation` over the pressure results: which of the 200 metrics predict `solved`? | **HIGH** |
| 111 | 14.2 Data Collection for Evaluation (annotation pipelines, **inter-annotator agreement**, guidelines, crowdsourcing vs expert) | PARTIAL | `eval/selftest/fleet_audit.py:70-172` (KNOWN_GOOD / KNOWN_BAD corpora, each with a stated `why`) | The book's five-stage pipeline (H2 lines 9472–9491) maps almost exactly onto `fleet_audit`'s corpus design — including the crucial stage 4, "*Embed gold-standard examples with known labels into the annotation queue.*" HarnessCAD's `Case.why` field is the book's "operationalised criteria" (§14.2.3). **This is done well.** What is missing: the corpora are **17 parts total** (8 good, 9 bad, `fleet_audit.py:70`, `:131`), authored by one person, with no second opinion and no κ. | Grow the corpora; get a second engineer to label independently; report Cohen's κ. | MED |
| 112 | 14.3 Synthetic Data Generation for Evaluation (**LLM-as-Judge for Calibration / ECE**, Self-Instruct, Evol-Instruct, CAI, Distillation, Arena) | PARTIAL | `eval/bench/judges/judge_calibration.py:6-20`; `eval/verifiers/vlm_judge.py`; `data/dataengine/annotation/*` | `judge_calibration.py` implements a **threshold sweep computing precision/recall/F1** — i.e. the book's calibration procedure — and its only importer is `tests/eval/bench/judges/test_paper30.py`. HarnessCAD ships a VLM judge (`vlm_judge.py`) that has **never been calibrated against a human or an oracle**. The book's ECE (§14.3.1) is not computed anywhere. | Calibrate `vlm_judge` against `selftest.golden` before it is trusted for anything. | **HIGH** |
| 113 | 14.4 Metrics for Ranking Tasks (ELO, Bradley-Terry, TrueSkill, **Win Rate with Confidence Intervals**, Chatbot Arena) | **ABSENT** | `eval/pressure/report.py:28-163` (`aggregate`, `headline`, `render_table`) | The pressure report gives point estimates and **no interval of any kind**. `grep -n "confidence\|interval\|significan\|bootstrap\|wilson"` over `eval/pressure/*.py` returns **zero hits**. The book gives the exact formula that should be there — the **Wilson score interval** (H2 lines 9846–9858) — and says explicitly why: "*preferred over the naive Wald interval because it has better coverage near p = 0 and p = 1*". At n=72 per arm, a 33.3%-vs-25.0% difference has a 95% Wilson CI of roughly [23%, 45%] vs [16%, 36%] — **the intervals overlap heavily**. See §2.6. | Wilson CIs on every rate; bootstrap CI on the delta; report both. | **CRITICAL** |
| 114 | 14.5 Metrics for Generation Tasks (BLEU, ROUGE, BERTScore, METEOR, Perplexity, **Pass@k**, Exact Match/F1) | **PARTIAL** | `eval/bench/sequence/pass_at_k.py` exists; `eval/pressure/*` does **not** import it | The book gives the **unbiased pass@k estimator** (H2 lines 10019–10058) and warns the naive estimator has high variance. HarnessCAD *has the module* and the pressure test reports a raw solve-rate over ≤3 attempts, which is neither pass@1 nor pass@3 nor pass^k — it is an unnamed quantity. See §2.6. | Report pass@1 and pass@3 with the unbiased estimator; report **pass^k** (all-k-succeed) for the reliability claim a CAD harness actually needs. | **CRITICAL** |
| 115 | **14.6 Metrics for Agentic Tasks** (Task Success Rate, **Trajectory Efficiency**, **Tool-Use Accuracy**, **Multi-Step Reasoning Accuracy / SRA**, SWE-bench, WebArena) | **PARTIAL** | TSR: `eval/pressure/metrics.py:37-58` (`Grade.solved`). Trajectory efficiency: `eval/bench/harness/tool_trajectory.py` (orphan). Tool-use accuracy: `agents/agent/tool_reward.py:148-165` `score_tool_selection` (orphan). SRA: **nothing**. | HarnessCAD reports **one** of the book's five agentic metrics. It has implementations of two more sitting unused. **Step-level Reasoning Accuracy** (H2 lines 10184–10197) — "*the fraction of reasoning steps that are correct*" — is the metric that would have exposed the false-positive problem from the *other* side: a loop whose per-step accuracy *drops* after typed feedback is a loop being poisoned, and that would have been visible without waiting for the final solid. | Report all five. `SRA` per op is computable exactly from the oracle. | **CRITICAL** |
| 116 | 14.7 LLM-as-Judge (setup, **position bias**, **multi-judge panels**, **agreement metrics**, G-Eval) | **ABSENT** | `eval/verifiers/vlm_judge.py` (uncalibrated, unbenchmarked) | The book's reliability bar for a judge (H2 lines 10380–10381): "*A judge is considered reliable if it achieves κ > 0.6 and agreement rate > 80% with human annotators on a representative sample.*" `eval/bench/judges/judge_human_agreement.py` **exists** and computes exactly this. It is imported by nothing but its test. HarnessCAD ships a judge and has never asked whether it agrees with anyone. | Either benchmark `vlm_judge` against the golden corpus, or stop shipping it. | HIGH |
| 117 | **14.8 Evaluation Pitfalls** (**Benchmark Contamination**, **Overfitting to Benchmarks**, **Goodhart's Law**, prompt sensitivity, aggregation artefacts, selection bias, eval–deployment mismatch) | **ABSENT — every one of the seven was hit** | See §2.3 and §2.4 | This is the section that should have been read before the pressure test was written. Contamination: the shell briefs probed their "inside" point *on the outer face*, so they passed only because a backend bug dilated the part — the corpus and the fleet **shared a blind spot**. Overfitting: `trap_fillet_*` briefs "*encode the harness's own wrong ceiling as ground truth, so the grader rewards obeying the harness*" (`assets/pressure/report.md:104-108`) — the benchmark was written by the system under test. Goodhart: the fleet became the target. Prompt sensitivity: one prompt, T=0, one seed. | §2 and §4. | **CRITICAL** |

**Count check: 117 rows for 116 sections** — row #98 audits H2 §12.6.10 as a standalone entry because it is the specific mechanism (credit assignment) the harness is missing, and folding it into the N/A parent (#97, a productivity-co-pilot worked example) would have hidden it. Rows #97 and #99 are N/A *as worked examples*; their reusable sub-sections (§12.6.9 guardrails, §12.6.10 credit assignment, §12.7.5 reward-hacking risks) are audited at #98, §2.4 and #117.

---

## 2. THE EVAL INDICTMENT

Nine findings. Where the book warned us and we did it anyway, the section is named and quoted.

### 2.1 The book told us that a wrong shaping signal is worse than no signal — in the section titled "The Risk of Naive Reshaping: Reward Hacking"

**H1, §3.11.1, lines 9361–9368** — under a box the author titled *"The Risk of Naive Reshaping"*:

> **If `F(s, a, s′)` is arbitrarily designed, the agent will find structural loopholes to maximize auxiliary signals while ignoring the global objective.**
> …
> **In LLMs: a model rewarded for "sounding confident" might learn to always start with "Absolutely!" regardless of accuracy.**

HarnessCAD's typed diagnostic *is* `F`. It is an auxiliary signal injected between the agent and the true objective (a correct part). It was "arbitrarily designed" in the precise sense the book means: **no proof that it preserves the optimal policy.** And the harness's failure is the book's failure, one level up — the model did not hack the reward, it *obeyed* it, because a typed diagnostic is not a reward, it is an **instruction**, and an instruction is a stronger channel than a reward. The 14b read `hole diameter 30 mm >= plate/stock wall 8 mm`, changed exactly one field, `30 → 7.5`, and left the other four holes untouched (`assets/pressure/report.md:48-51`). Flawless instruction-following applied to a false statement.

The book then gives the fix, in the very next subsection (**§3.11.2, Potential-Based Reward Shaping**, lines 9369–9400), and the guarantee it buys:

> **Policy Invariance:** The optimal policy π* under the reshaped reward R′ is identical to the optimal policy under the original reward R. **The shaping cannot introduce sub-optimal behaviors.**

HarnessCAD's shaping introduced sub-optimal behaviours in eight of 72 cells. It has no invariance property and never claimed one. **A typed diagnostic that has not been proven sound is a shaping term with no policy-invariance guarantee, and the book says in bold what happens next.**

The structural fix is *not* "write better rules". It is: **the feedback channel must be as trustworthy as the reward channel, or it must be switched off.** Concretely, in `core/loop.py:32-40`, the fleet is `advisory` by default (`fleet_blocking=False`) — which sounds conservative and is in fact the *worst* setting: an advisory diagnostic does not roll the op back (so a wrong rule cannot be caught by a failing build) but it *is* handed to the model (so a wrong rule is executed as an instruction). Advisory means **all of the harm, none of the containment.**

### 2.2 The book's "Rule-Based Reward Pitfalls" list is HarnessCAD's bug list

**H2, §9.5, lines 3591–3598:**

> **Rule-Based Reward Pitfalls**
> • **Format gaming:** models learn to produce the correct format without correct content. Always combine format and correctness rewards.
> • **Test case leakage:** if test cases are in the training data, the model memorises them.
> • **Timeout exploitation:** …
> • **Reward sparsity:** binary rewards (0/1) can be too sparse for complex tasks. Consider partial credit or intermediate rewards.

And **H2 §13.6.5 (RLVR), lines 9093–9096:**

> The key advantage of RLVR over RLHF is the **absence of reward model error**: since the reward is computed by a deterministic verifier rather than a learned model, there is no reward hacking against a flawed reward model. **The only failure mode is if the model finds solutions that pass verification but are not genuinely correct (e.g., exploiting test case weaknesses in code evaluation).**

Read that last sentence against `assets/pressure/report.md:88-94`, fleet hole #2:

> `shell` grows the part, and nothing notices. … A 60×40×20 box shelled at t=3 comes out with bbox `[63.0, 43.0, 23.0]` and **no diagnostics at all**. … **Every `shell_box_3mm` solve in this run is a part with wrong outside dimensions**, and the brief carries `bbox=None`, so the corpus is blind to it too. **Fleet and corpus share the blind spot.**

That is textbook "*solutions that pass verification but are not genuinely correct… exploiting test case weaknesses*". The book named the **only** failure mode of RLVR, and it is the one HarnessCAD shipped.

### 2.3 Benchmark contamination — the book has a section called "Benchmark Contamination" and a section called "Overfitting to Benchmarks", and we did both

**H2, §14.8.1, lines 10422–10437** and **§14.8.2, lines 10446–10455:**

> Benchmark contamination occurs when evaluation data appears in the model's training set… **Contaminated models achieve inflated scores that do not reflect true generalisation ability.**
> …
> Even without direct contamination, models can be implicitly optimised for specific benchmarks through repeated evaluation and hyperparameter tuning. This is a form of **adaptive overfitting: the benchmark leaks information into model development decisions.**

HarnessCAD's contamination is not the classic train-set kind — it is worse, because it is **structural**. From `assets/pressure/report.md:102-108`:

> `preflight-RADIUS_TOO_LARGE` **is itself unsound**, and it contaminates the experiment **in the harness's favour**. A 50×30×6 plate with `fillet r=3.1` is valid, watertight and correctly bounded, and the rule fires anyway; meanwhile `r == half-extent`, the true boundary, fires nothing. **The `trap_fillet_*` briefs encode the harness's own wrong ceiling as ground truth, so the grader rewards obeying the harness.**

The benchmark was authored by the system under test, and it encoded the system's bug as the correct answer. This is the **purest possible instance** of the book's adaptive overfitting: the benchmark did not merely leak into development — development *wrote the benchmark*. The harness's own report catches this and says so, which is to its enormous credit. But it means the mitigation the book prescribes was never in place:

> **Mitigation:** Maintain a **private test set** that is never released publicly. **Regularly refresh benchmarks with new examples.**

There is no held-out brief corpus. There is one corpus, 28 briefs, written by the same hand that wrote the rules, and there is nothing to catch the next time a rule and a brief agree with each other and are both wrong.

### 2.4 Goodhart's Law — quoted in full, because the book saw the whole thing coming

**H2, §14.8.3, lines 10457–10475:**

> Goodhart's Law states: **"When a measure becomes a target, it ceases to be a good measure."**
> In LLM evaluation, this manifests in several ways:
> • **Reward hacking:** Models trained with RLHF learn to **exploit the reward model rather than genuinely improving.**
> • **Judge gaming:** Models trained with LLM-as-judge feedback may learn **the judge's biases** rather than genuinely improving quality.
>
> **Defences Against Goodhart's Law**
> 1. **Metric diversity:** Use multiple metrics from different families; **a model that games one metric will likely not game all simultaneously.**
> 2. **Held-out evaluation:** Maintain evaluation metrics that are **not used in training or model selection.**
> 3. **Human spot-checks:** Regularly sample model outputs for human review, **independent of automated metrics.**
> 4. **Adversarial evaluation:** Actively probe for failure modes that automated metrics miss.
> 5. **Extrinsic validation:** Periodically validate intrinsic metrics against extrinsic outcomes.

HarnessCAD implements defence #2 and nothing else. The grader is genuinely blind to the arm and never shown to the model (`eval/pressure/metrics.py:11-13`, `eval/pressure/briefs.py:11-14`) — this is *correct and important* and it is why the −8.3 is believable. But:

* **#1 metric diversity:** the grader checks bbox + volume + a handful of SDF probes + op-count assertions (`metrics.py:104-177`). Four families, all *envelope* families. It has no reference-model distance — no IoU, no Chamfer, no Hausdorff — **even though `eval/bench/geometry/solid_iou.py`, `chamfer.py` and `hausdorff_iogt.py` all exist**, and even though **every brief already carries a hand-written `reference` op stream** (`eval/pressure/briefs.py:16-20`) which is used only to prove the brief is solvable and *never as a geometric target*. The grader is therefore many-to-one by construction, and the fix is sitting in the repo: rebuild `reference`, rebuild the model's ops, compute IoU. One function call.
* **#3 human spot-checks:** none. `eval/gallery/render_gallery.py` exists; nobody looked.
* **#4 adversarial evaluation:** the 5 trap briefs are the right instinct — and two of them (§2.3) were adversarial against the *model* when they should have been adversarial against the *harness*.
* **#5 extrinsic validation:** ~200 intrinsic bench metrics, zero correlation studies. `eval/bench/harness/metric_correlation.py` exists and is called by nothing.

And the book's research-agent case study (**H2 §12.7.5, lines 7545–7552**) lists "Reward Hacking Risks" with a fix for each. One of them: "*Fake results: Agent fabricates experiment outputs. **Fix: Verify code actually ran by checking execution logs against reported numbers.***" That is the output gate (`io/gate.py`). HarnessCAD built it. It is the strongest thing in the repo. It is *not applied to the pressure grader* — `grade()` at `eval/pressure/metrics.py:196` constructs a raw `CISPServer` and never routes through the gate, so an op stream that produces a dilated shell can be scored `solved=True` and never meet the one component that would have refused it.

### 2.5 The A/B was the right instinct. The book prescribes three more arms, and one of them is mandatory.

The controlled A/B — same model, same prompt, same seed, same attempt budget, differing in exactly one factor — is genuinely good experimental design and the harness deserves credit for running it and publishing the loss. But the book is explicit that this is not sufficient, and one of its instructions is unambiguous.

**H2, §8.5.4, lines 3228–3230 — a callout box titled "Best-of-N as Baseline":**

> ***Always compare your RL method against Best-of-N with the same compute budget. If PPO with 64 GPU-hours doesn't beat Best-of-N with 64 GPU-hours of generation, your PPO has a bug.***

HarnessCAD compared its method against **blind resampling with no selection** — the weakest baseline in the book's entire spectrum. The mandatory baseline, **Best-of-N *with a reward model*** (§8.5, §13.2.5), was never run — and HarnessCAD has the **best reward model in the book's taxonomy**: an exact, deterministic, six-engine differential oracle. The book's BoN scaling law (H2 lines 8300–8308):

> With a **perfect** reward model (oracle that always selects correctly): **p = 0.3, N = 10: success = 97%.**

With blind pass@1 ≈ 33% pooled, Oracle-BoN at N=3 should land near 70%. **The harness lost 25.0% to 33.3% while carrying an unused mechanism that should score ~70% on the same compute.** That is the finding this experiment should have produced.

Three arms are missing:
1. **Oracle-BoN** (§8.5) — mandatory per the book. Sample N, let the oracle pick. No feedback channel at all, no model reasoning about diagnostics, no poisoning surface.
2. **Self-Consistency** (§13.2.2) — free, parallel, no reward model. Majority-vote the op stream over N samples. This is the *control* for "does any selection beat any feedback".
3. **Blind-with-more-attempts at matched compute** (§11.8, §8.5.4) — the harness arm used **40% more model calls** (1.81 vs 1.29). Give the blind arm 1.81 attempts' worth of budget and the −8.3 gets *worse*. This is not stated anywhere and it should be.

Without arm 1, the pressure test cannot answer the question it was built to answer. It showed that **this** typed channel is worse than **no** channel. It did not show that a typed channel is worse than **the best available alternative**, and the best available alternative is already in the repo.

### 2.6 pass@k vs pass^k, variance, seeds, statistical significance at n=72

**H2, §14.4.4, lines 9843–9858** — the book gives the exact interval and says why:

> The simplest ranking metric is the win rate… A **Wilson score confidence interval** is preferred over the naive Wald interval because it has better coverage near p = 0 and p = 1.

**H2, §14.5.6, lines 10015–10033** — and the exact estimator:

> pass@k = E[1 − C(n−c, k)/C(n, k)] … This **unbiased estimator avoids the high variance of the naive estimator** (which samples exactly k solutions and checks if any pass).

**H2, §14.4.1, lines 9763–9767** — and how to get an interval when the statistic is not a simple proportion:

> **Bootstrap Confidence Intervals**… resample the battle log with replacement B = 1000 times, recompute… report the 2.5th and 97.5th percentiles.

**HarnessCAD reports none of these.** `grep -nE "confidence|interval|significan|bootstrap|wilson|p_value|binomial"` over `src/harnesscad/eval/pressure/*.py` returns zero hits outside of `sys.stderr` and an error string. `eval/pressure/report.py:28-163` computes means and deltas and prints them.

The consequences, in order of severity:

* **The headline may not be significant.** 24/72 vs 18/72. Wilson 95% CIs: blind ≈ [23.6%, 44.6%], harness ≈ [16.4%, 36.4%]. **The intervals overlap across most of their range.** A paired test is the right instrument (same briefs, same models, same seed — this is a matched design, and the report *knows* it: it identifies 8 specific regressions, `report.md:31-33`), and a paired sign/McNemar test on 8 discordant pairs (8 harness-losses, 0 harness-wins) gives p = 2⁻⁸ ≈ **0.004** — which is significant, and *stronger* than the unpaired reading. **The harness's result survives the test it never ran, and it should run it and say so**, because right now a reviewer's first move is to point at n=72 and dismiss it.
* **Zero variance estimate.** One seed (`20260713`), temperature 0.0, one prompt (`eval/pressure/runner.py:52-71`). Even a deterministic decode is not variance-free across seeds in Ollama, and the book (§14.8.4, "Prompt sensitivity") warns: "*LLM performance can vary dramatically with small changes to the evaluation prompt… **Always report the exact prompt used and consider evaluating across multiple prompt variants.***" One prompt. One seed.
* **The reported metric has no name.** "Solve rate over ≤3 attempts with feedback" is not pass@1, not pass@3, and not pass^k. `eval/bench/sequence/pass_at_k.py` exists in the repo and the pressure module does not import it.
* **pass^k is never reported at all, and it is the metric a CAD harness actually needs.** pass@k asks "did *any* of k attempts work" — the right metric for a research demo. **pass^k asks "did *all* k attempts work"** — the right metric for a tool that will hand a part to a CNC machine. A harness whose selling point is *reliability* must report the *conjunctive* metric, and it reports the disjunctive one.

### 2.7 Process supervision, credit assignment, and the reward machinery we built and never plugged in

**H2, §9.4 — the PRM vs ORM table, lines 3455–3478:**

| Property | ORM | PRM |
|---|---|---|
| Reward signal | Final answer only | Each reasoning step |
| Credit assignment | **Sparse** | **Dense** |
| **Reward hacking** | **Easier to hack** | **Harder to hack** |
| Best for | Simple tasks | **Multi-step reasoning** |

**H2, §12.8.3, "Fine-Grained Turn-Level Credit Assignment", lines 7889–7908:**

> The core challenge in agentic RL is the **sparse reward problem**. **If an agent executes 20 tool actions and finally fails a unit test, a terminal reward of 0 punishes all 20 actions equally.**
> …
> **Multi-Turn Trajectory Slicing.** Frameworks split a multi-turn agent run into individual, independent steps. A credit assignment module isolates the **exact sub-step** that broke the trajectory: 1. Replay the successful prefix (steps 1–k) 2. **Identify the first divergence point** 3. **Assign negative reward only to that specific step** 4. Assign neutral/positive rewards to correct prefix steps. **This enables surgical policy updates without degrading already-correct behavior.**

HarnessCAD grades the **final solid** (`eval/pressure/metrics.py:183-219`). A 6-op plan that produces a wrong part gets one scalar `solved=False` and a list of reasons about the *finished geometry*. Ops 1–4 may have been perfect. Nothing knows.

And the machinery to fix this **is already written**. `agents/agent/tool_reward.py:79-95`:

```python
def aggregate_reward(traj, *, orm_verdict, format_text, alpha=1.0, beta=1.0, gamma=1.0):
    """R = alpha*R_ORM + beta*mean(R_step) + gamma*R_format (Eq. 8)."""
```

with `step_execution_rewards` (`:51-53`) producing a per-step binary and `score_tool_selection` (`:148-165`) producing per-position tool-name and argument accuracy against a reference. This is the book's PRM, implemented, unit-tested, and imported by **exactly two things**: a dispatch table (`agents/registry.py:239`) and its own test (`tests/agents/agent/test_tool_reward.py:7`). **It is not in `core/loop.py`. It is not in `eval/pressure/`. It is not in `core/trace.py`.**

Two things follow, and the second is the important one:

1. HarnessCAD runs **outcome-only supervision on a multi-step trajectory** while owning a finished process-reward implementation — the exact configuration the book's own table marks "*Sparse*" and "*Easier to hack*".
2. **A per-step reward would have caught the false-positive problem before the final grade.** A loop whose *step accuracy* degrades after typed feedback is a loop being poisoned, and that is visible at op 3, not at the end. The harness had to run 6 models × 12 briefs × 2 loops × 3 attempts and hand-reproduce four backend bugs to discover what a per-step delta would have shown on the first regressed brief. **The instrument that would have detected the failure was in the repository, complete, throughout.**

A fair counter-argument, and it must be recorded: **DeepSeek-R1 explicitly rejected PRMs** (H2 §13.3.2, lines 8608–8613): "*A notable and surprising finding of R1 is that **no process reward model (PRM) is needed**… PRMs introduce their own failure modes (reward hacking at the step level).*" That is a real result and HarnessCAD's outcome-only *grading* may well be right. But R1's ORM was **exact**, and R1 had **no feedback channel** — it did not tell the model *why* it was wrong. HarnessCAD does, and that channel is the thing that broke. The R1 lesson does not exonerate the typed channel; if anything it indicts it, because R1 got its result with **no diagnostics at all** — which is precisely what the blind arm is.

### 2.8 Who verifies the verifier — and the honest ledger

The question "who verifies the verifier" has, since the post-mortem, a real answer in this repo, and the audit must say so plainly: **`eval/selftest/fleet_audit.py` is a correct, complete implementation of the thing that was missing.** Its confusion matrix (`:177-230`) is textbook — precision = TP/(TP+FP) at `:204-206`, recall at `:209-212`, F1 at `:215-219` — its two corpora carry a stated `why` per case (`:70-172`), and its `false_positive_rate` (`:260-262`) is exactly the book's "over-refusal rate" (H1 §1.17.5) transposed to geometry. It even gets the hard call right: a verifier that declines a part via `applies_to()` is scored out-of-scope, not credited with a false negative (`:284-292`), which prevents the metric from flattering rules that claim everything and catch nothing.

The ledger, honestly:

* **Fixed.** `eval/verifiers/precheck.py` no longer compares hole diameter to extrude depth. Its docstring now carries the post-mortem in the source: "*The hole diameter is an in-plane quantity and is **never** compared against the extrude depth — those are orthogonal*" (`precheck.py:24-26`, `:39-40`, logic at `:384-424`).
* **Built.** Four oracles: `differential.py` (six engines), `golden.py` (closed-form ground truth, with the Steiner-formula derivations written next to the numbers), `properties.py` (metamorphic laws — `shell_does_not_grow`, `scale_is_cubic`, `replay_is_identical`), `fleet_audit.py` (precision/recall).
* **Not enforced.** There is **no CI** (`.github/workflows` does not exist). `fleet_audit` is a CLI subcommand (`core/cli.py:603-608`, `:868-877`). Nothing prevents a rule with precision 0.4 from being merged tomorrow. The book's whole point about held-out evaluation (§14.8.3, defence #2) is that it must be **enforced**, not available.
* **Too small.** 8 known-good parts, 9 known-bad. The washer is in there *because it already burned us*. The corpus is a list of bugs we have found, which is the definition of a regression suite — and the book's §14.8.2 says a benchmark like that "*degrades over time*" and must be "*regularly refreshed with new examples*". `properties.py` is the answer (200 seeded random streams, laws instead of instances) and it should be the *primary* gate, with the corpora as a fast smoke test.

### 2.9 The single sentence

The harness's own report ends with it, and it is the correct conclusion, and it is also the book's, from **§9.5 and §3.11.1 and §14.8.3 simultaneously**:

> **A verifier fleet is a trust system, and its throughput is set by its worst rule, not its best one.**

The book's version, at H2 line 3518: RLVR "*substantially reduces reward hacking* ***(though models can still exploit format tricks, edge cases, or test memorization)***". The parenthesis is the whole harness. And what it means operationally is the thing that has to change: **a diagnostic is not free.** Its expected value is `P(true) × (value if true) − P(false) × (cost if false)`, and the pressure test measured the second term for the first time: **the unaudited rules cost 8 briefs; the one genuinely-detecting rule earned 3.** Any rule whose precision is unmeasured has an unbounded negative term and must not be in the feedback channel.

---

## 3. RL / training methods: what we could be doing and are not

The finding of this section is blunt: **HarnessCAD has the scarcest asset in applied RL — an exact, deterministic, human-label-free correctness oracle — and it has built no way to learn from it.** The book (H2 §12.5.7, RLEF) says why that asset is valuable, in a list that reads like a description of this repo:

> **Why execution feedback is ideal for RL:**
> • **Zero noise:** Unlike human preferences, test results are deterministic. Same code → same reward every time. **This eliminates reward noise that destabilizes RL training.**
> • **Infinite scale:** Can generate unlimited tasks programmatically.
> • **No reward hacking:** Unlike learned reward models, a test suite can't be "fooled".
> • **Dense signal:** Partial test passage (r = 0.6) provides richer gradient than binary pass/fail.

HarnessCAD's "test suite" is six independent geometry kernels plus a closed-form golden corpus plus seven metamorphic laws. It is *stronger* than a unit-test suite, because a unit-test suite can be wrong in the same way twice and a six-way differential cannot (cheaply).

And the go/no-go gate the book demands before spending a dollar on RL (§10.6, lines 4259–4301) **is already passed, in the harness's own published data**: "*If pass@1 < 5%, RL will likely fail. If pass@k < 20%, RL will struggle.*" qwen-2.5-coder-14b: **pass@1 = 66.7% blind** (`assets/pressure/report.md:21`). Pooled across the ladder: 33.3%. That is inside the book's Goldilocks band for GRPO (20–80%, H2 line 1830). **The pressure report is, without meaning to be, an RL-viability green light.**

What is half-built, and what it would cost:

### 3.1 Rejection-Sampling Fine-Tuning (RFT / STaR) — the cheapest, do this first

Book: H2 §8.5.2 ("*As a training method (Rejection Sampling Fine-Tuning / RFT): 1. Generate many responses, select best ones 2. SFT on the selected responses 3. Repeat*"), §12.5.1 (STaR), §13.5.2 (Qwen's RFT+RL loop).

* **What exists:** `data/dataengine/selftrain/bootstrap_loop.py`, `pseudo_label_selection.py`, `confidence_score.py`, `threshold_selection.py`, `self_improvement.py`. A curation stack (`data/dataengine/curation/`, 13 modules). And **~220 already-generated, already-graded trajectories in `.pressure_cache`** — every model call from the A/B, keyed and reproducible.
* **What is missing:** an SFT trainer. `pyproject.toml:11` — `dependencies = []`.
* **Cost, honestly:** `pip install trl peft transformers`. One 7B QLoRA run, r=16, ~1 GPU-day on a 24 GB card, or ~$30 of rented A100 time. The corpus is free — it is the exhaust of an experiment already run.
* **STaR's rationalization step (H2 lines 6122–6127) is a free extra:** for briefs the model failed, condition on the *reference op stream* (which every brief already carries, `eval/pressure/briefs.py:16-20`) and ask the model to produce a trace that arrives there. That converts every failure in the corpus into a training example.
* **Expected:** the book's STaR convergence note (line 6138): "*If p₀ = 0.3, after rationalization + SFT, p₁ ≈ 0.5. Typically converges in 3–5 iterations to p ≈ 0.7–0.9.*"

### 3.2 DPO / KTO on oracle-labelled pairs — second cheapest, and the modules are already written

Book: H2 Ch. 6, §8.2 (KTO), Table 8.1 ("DPO: 2 models, offline pairs, **low compute**, **high stability**").

* **What exists:** `data/dataengine/preference/dpo_pairs.py`, `kto.py`, `binary_preferences.py`, `binary_sampling.py`, `visual_feedback_pairs.py`. **The pair-construction layer is done.**
* **The labeller is the oracle, not the fleet.** This is the whole point and it must be stated as a rule: `(chosen, rejected)` pairs must be adjudicated by `selftest.golden` + `selftest.differential`, **never** by `eval/verifiers/registry`. If HarnessCAD builds preference pairs from fleet verdicts it will train a model to reject washers. The book's DPO failure mode #4 (H2 line 1129): "*Noisy labels poison training. **Unlike PPO which averages over many samples, DPO memorizes individual pairs.***" DPO is the *least* forgiving method for a system with a measured false-positive problem.
* **KTO is the better first bet:** it needs only *unpaired binary* labels ("this op stream is good / bad"), which the oracle emits for free on any stream ever generated — no pairing, no matched prompts. The book: KTO handles imbalance better and is "*more robust than DPO to noise*" (H2 line 2952).
* **If DPO, use Robust DPO** (§6.9.2) with `label_smoothing = ε` set from the **measured** fleet false-positive rate from `fleet_audit`. HarnessCAD is now one of very few systems that can *measure* its own ε instead of guessing it.
* **Cost:** same as 3.1. One LoRA run.

### 3.3 GRPO — the right long-term answer, and three of DAPO's five components are already implemented

Book: H2 Ch. 7; §8.6 decision tree, line 3302: "***1. Do you have verifiable rewards? (math/code) → GRPO.***"

* **What exists, as orphaned pure-Python formulas:** `data/dataengine/reward/clipped_policy_objective.py` (the PPO/GRPO clipped surrogate), `asymmetric_clip.py` (= **DAPO Clip-Higher**, §7.5.2 Component 1), `overlong_filter.py` (= **DAPO Overlong Filtering**, Component 3), `expert_advantage.py` (group-relative advantage), `curriculum_reward.py`, `iou_reward.py`, `executability_reward.py`, `geometry_semantics_reward.py`, `precision_token_loss.py`, `hard_questions.py`.
* **What is missing:** a policy, a gradient, and vLLM. Everything else is written.
* **The reward function is a two-liner given what exists:** `r = 1.0 if selftest.golden/differential accepts the built solid else 0.0`, plus the book's format reward (§13.3.2, λ_fmt = 0.1) — which is `tool_reward.format_reward` (`tool_reward.py:46-48`), already implemented.
* **Group size:** the book says G=8 (§7.4), and 2-GRPO (§7.5.5) claims G=2 matches G=16 at 8× less generation compute. Start at G=8, drop to G=2 if generation dominates.
* **The Goldilocks filter is already done by hand** (see #56): the pressure corpus was pruned to briefs with a 20–80% pass rate for exactly the reason §7.4 gives.
* **Cost, honestly:** this is the expensive one. 7B GRPO with vLLM generation, G=8, ~3,000 steps: 1–2 A100-weeks, call it $2–5k rented, plus real engineering (the book's §11 exists because this is a systems problem). **Do not start here.** Do 3.1 and 3.2 first; if RFT+KTO on the oracle does not move pass@1 by 10+ points, GRPO will not save it.

### 3.4 A free, exact Process Reward Model — the thing no one else can build

Book: H2 §13.6.2 (Math-Shepherd). The book's automatic PRM annotation is *statistical*: for partial solution `(s₁…s_k)`, sample M completions and label step k correct if **any** reaches the right answer.

**HarnessCAD can do this exactly, not statistically.** Take an op stream, truncate at op k, and ask the oracle whether a valid completion exists — or more cheaply, whether the partial geometry still satisfies the brief's monotone invariants (`properties.py`'s `cut_does_not_add`, `union_does_not_remove`, `shell_does_not_grow` are *step-local* laws). The result is a **per-op correctness label with no sampling and no noise.** This is a stronger PRM than anything in the book, it requires no model, and it is the natural home for the `tool_reward.step_execution_rewards` machinery that is already written (`tool_reward.py:51-61`).

That PRM then does three jobs at once: it is the **credit-assignment signal** (§2.7), it is the **step-accuracy eval metric** (§14.6.4, row #115), and it is the **dense reward** for 3.3.

### 3.5 What NOT to do

* **Do not train a reward model.** HarnessCAD's oracle is exact. The book's entire Ch. 9 apparatus (Bradley-Terry, ECE, length-bias correction, reward centering) exists to compensate for a *learned* reward model's error, and RLVR's headline advantage (§13.6.5) is the "*absence of reward model error*". Introducing a learned RM would be trading a perfect signal for a hackable one.
* **Do not use the verifier fleet as a reward.** Measured precision on 8 known-good parts is the only thing that could ever license this, and until `false_positive_rate == 0` is a hard gate in CI, the fleet is a poisoned signal by demonstration.
* **Do not build MCTS/LATS yet.** The book says it costs 10–50× inference FLOPs (§12.5.4) and HarnessCAD has not yet run Best-of-N, which is 1/10 the work and, per §8.5.4, is the baseline MCTS has to beat.

---

## 4. Top 10 gaps, ranked by (harm to correctness) / (effort to fix)

| # | Gap | Module | Harm | Effort | Why this rank |
|---|---|---|---|---|---|
| **1** | **The fleet's false-positive rate is measured but not enforced.** `fleet_audit` computes it; nothing gates on it; there is no CI at all (`.github/` does not exist). | `eval/selftest/fleet_audit.py:260-262`; **no** `.github/workflows/` | Total — this *is* the −8.3. An unaudited rule in the feedback channel has unbounded negative expected value (§2.1). | Hours. Add a CI job that fails the build if `fleet_fp > 0` or any verifier's precision < 1.0. | The bug is fixed, the instrument is built, and nothing stops the next one. Highest harm, near-zero effort. |
| **2** | **The mandatory baseline was never run.** No Best-of-N / Oracle-selection arm in the pressure test, despite the book's flat instruction and despite owning a near-perfect selector. | `eval/pressure/loops.py`; `eval/selftest/differential.py` | Total — the experiment cannot answer its own question (§2.5). Likely turns a −8.3 story into a **+35** story. | Days. One new loop arm: sample N, oracle-select. All parts exist. | The book: "***Always compare your RL method against Best-of-N with the same compute budget.***" We did not. The alternative we skipped is the one we're best at. |
| **3** | **No confidence intervals, no significance test, one seed, one prompt, T=0.** | `eval/pressure/report.py:28-163`; `runner.py:52-71` | High — the headline is currently dismissible at n=72, *and the harness is under-claiming*: the correct paired (McNemar) test on 8-vs-0 discordant pairs gives **p ≈ 0.004**. | Hours for Wilson CIs + McNemar; ~9 min × 3 for extra seeds. | The result is *more* defensible than reported. Fixing this costs a day and strengthens the finding. |
| **4** | **Per-step reward / credit assignment is implemented and unwired.** Outcome-only grading on a 3–8-op trajectory. | `agents/agent/tool_reward.py:79-95` (2 importers: a dispatch table and its own test); `core/loop.py`; `core/trace.py` | High — sparse credit, "easier to hack" per the book's own PRM/ORM table, **and it is the instrument that would have caught the poisoning at op 3 instead of at brief 12** (§2.7). | Days. Log `R_step` per op in the trace; report step-accuracy in `pressure/report.py`. | The detector was in the repo, complete, the whole time. |
| **5** | **The grader is many-to-one and the fix is already in the repo, unused.** No IoU/Chamfer against the reference — while every brief already carries a `reference` op stream and `eval/bench/geometry/solid_iou.py` exists. | `eval/pressure/metrics.py:104-177`; `eval/pressure/briefs.py:16-20`; `eval/bench/geometry/{solid_iou,chamfer,hausdorff_iogt}.py` | High — holes in the wrong place score perfectly. The corpus and fleet **shared** a blind spot (§2.2). | Hours. Rebuild `reference`, rebuild candidate, compute IoU. One function call. | Zero new machinery. The reference is already there and is used only to prove solvability. |
| **6** | **The output gate is not applied to the grader.** `grade()` constructs a raw `CISPServer` and bypasses `io/gate.py` — the one component that refuses a dilated shell. | `eval/pressure/metrics.py:196`; `io/gate.py` | High — an op stream that produces a wrong-sized part can score `solved=True`. This is exactly the shell bug, still reachable through the grader. | Hours. Route `grade()` through the gate. | The strongest component in the repo is not applied at the point it matters most. |
| **7** | **The LLM judge has never been calibrated, and the calibrator exists.** `vlm_judge.py` ships; `judge_calibration.py` and `judge_human_agreement.py` are orphans. | `eval/verifiers/vlm_judge.py`; `eval/bench/judges/judge_calibration.py:6-20`; `judge_human_agreement.py` | Med-High — an uncalibrated judge is a second unaudited feedback channel. **This is the same class of bug as the washer, waiting to happen.** The book's bar: κ > 0.6 and >80% agreement. | Days. Score `vlm_judge` against `selftest.golden`; publish precision/recall. | We have not learned the lesson twice: the judge is exactly the fleet, one layer up. |
| **8** | **No held-out brief corpus; the benchmark was written by the system under test.** | `eval/pressure/briefs.py` (28 briefs, 5 traps, 2 encoding the harness's own bug) | High — adaptive overfitting, §14.8.2. `trap_fillet_*` encodes a wrong ceiling as ground truth. | Days–weeks. Make `properties.py` (200 seeded random streams, laws not instances) the *primary* gate; keep the corpora as a smoke test; have a second engineer write a held-out set. | Structurally the hardest to fix and the one that will bite next. |
| **9** | **~220 graded trajectories thrown away; no SFT/RFT despite a free oracle.** | `.pressure_cache`; `data/dataengine/selftrain/bootstrap_loop.py`; `data/dataengine/preference/{dpo_pairs,kto}.py`; `pyproject.toml:11` | Med — this is opportunity cost, not incorrectness. But it is *large*: the book's RL-viability gate (pass@1 ≥ 5%) is passed at 66.7% on the 14b. | ~1 GPU-day, ~$30, plus `pip install trl peft`. | Highest reward-per-dollar in the repo. Ranked 9 only because it fixes no *existing* bug. |
| **10** | **200 intrinsic bench metrics; zero correlation with the one outcome that matters.** `metric_correlation.py` exists and is imported by nothing. | `eval/bench/harness/metric_correlation.py`; `eval/bench/**` (~200 modules) | Med — we do not know which of our metrics predict `solved`. The book (§14.1.1): intrinsic metrics are "*often poorly correlated with real-world utility*". | Hours. Run `metric_correlation` over `assets/pressure/results.json`. | Cheap, and it will probably tell us most of the 200 are decoration. |

---

## 5. Where we are genuinely ahead of the book

Honest, not flattering. Three things, and one of them is rare.

**1. The differential oracle is stronger than anything in the book's evaluation chapter, and the book does not describe it.**
Chapter 14's answer to "how do you know the answer is right without a human" is: exact match (needs a reference), execution/pass@k (needs a test suite someone wrote), or an LLM judge (needs calibration and has position bias, verbosity bias, and self-enhancement bias — §14.7.2). HarnessCAD's answer is: **run the plan on six independently-implemented geometry engines and see if they agree** (`eval/selftest/differential.py`). That is a ground-truth signal with **no human, no reference, and no model**. The closest thing in the book is §13.6.5's RLVR verifier, and a unit-test suite is *weaker*: a test suite is one implementation of the spec and can be wrong; a six-way differential across a stub, a sampled SDF, two OCCT B-rep kernels, a CGAL mesher and a mesh kernel is wrong only if all six share a bug. This is genuinely rare and it is the repo's main asset.

**2. The measured output gate, and the soundness-not-completeness invariant.**
`io/gate.py:1-50`: *"Every artifact leaving the harness is either (a) verified valid, or (b) refused with a reason. There is no third outcome. Silence is not success."* The MEASURED/DECLARED split — measure the built geometry alone, *then* replay the op log and check the geometry honours the declared intent either side of every intent-bearing op — is a construct the book does not have. Its nearest analogue is a throwaway line in the research-agent case study (§12.7.5): "*Fake results… **Fix: Verify code actually ran by checking execution logs against reported numbers.***" HarnessCAD generalised that one-line mitigation into an architectural invariant. That is ahead. (And then did not apply it to its own grader — see gap #6. Ahead is not the same as finished.)

**3. `properties.py` — metamorphic testing as an oracle, which Chapter 14 does not contain at all.**
`scale_is_cubic` relates **two runs of the same engine** and therefore needs no ground truth *and holds even for an engine whose absolute numbers are all wrong*. `replay_is_identical` guards determinism, without which every cached result and every regression test in the repo is quietly meaningless. `shell_does_not_grow` is one sentence and would have caught the bug that shipped in the README. The book has nothing like this — its evaluation chapter is entirely instance-based (corpora, references, judges), and metamorphic/property-based oracles are the one family that scales without labels and finds bugs in parts nobody has drawn. HarnessCAD's `properties.py` should be the **primary** gate, not the fourth one.

**And the honest qualifier on all three:** these are strengths of the *oracle*, not of the *harness*. The oracle is world-class and it was built **after** the harness lost its own experiment. Every one of the three is a CLI subcommand and none is enforced anywhere. The differential oracle's own docstring is the fairest summary of the repo's condition (`differential.py:19-22`):

> *"They were built as interchangeable **products**. They are also, for free, the strongest oracle in the repository, and **nothing had ever used them as one**."*

That sentence is true of `tool_reward.py`, of `judge_calibration.py`, of `metric_correlation.py`, of `pass_at_k.py`, of `dpo_pairs.py`, of `asymmetric_clip.py`, and of `composite_reward.py`. **HarnessCAD's problem is not that it lacks capability. It is that the capability is not wired to the loop, and the loop is not gated on the capability.** The book's Chapter 14 is 900 lines about the discipline of connecting those two things, and we had 1,335 modules and none of the discipline.

---

*Read-only audit. No file under `src/` or `tests/` was modified. Every verdict above is backed by a `file:line` reference or by an explicit statement that the thing does not exist.*
