# Minimal-Prompt + Maximal-Runtime-Inference: Deep Survey

**Produced:** 2026-05-31
**Phase:** 12 follow-up / v2.2 prep
**Status:** Draft for review
**Word count:** ~3,950
**Companion to:** `PROMPT-HARNESS-SURVEY.md` (Phase 11) and `PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md`

---

## 0. Empirical Anchor — What Phase 12 Just Learned

Plan **12-05** built the inference-time rubric (AHE + Agentic Rubrics pattern): a small builder prompt consumes the architecture document and emits a 4–6-item rubric that replaces the static `DOC_KNOWLEDGE_JUDGE_RULES` body. Empirically, **F1 holds**: mediastore 0.9667 (-1.74pp vs baseline, inside Claude run-to-run variance), teastore 1.0000. Mechanism active (no fallback triggered), rubrics distinct per-dataset. The original SUMMARY recorded REJECT under a strict reading of GATE-06 (rubric named project components). Under the **corrected GATE-06 reading from CLAUDE.md** — *runtime LLM inference of doc-specific knowledge is EXPLICITLY mandated; only static benchmark-derived word lists are prohibited* — the leakage flag is methodological air, not violation. **The runtime-rubric pattern is the first confirmed V35-ceiling escape on the s_linker13 lineage.** This survey's job is to enumerate the rest of the playbook for Phase 12+ and v2.2.

Plan **12-04** tried the opposite axis — static merge of `ENTITY_EXTRACTION_RULES` + `VALIDATION_RULES` into a shared rubric core (Phase 11 Technique 3). It failed exactly the V35 way: bigbluebutton -6.59pp (proposer/judge boundary erasure). Together: **information density can be regenerated at inference time (12-05 works); it cannot be removed statically across a sub-task boundary (12-04 fails)**. Every technique in §2 is scored against whether it preserves info density via inference-time generation, hidden deliberation tokens, or multi-call distribution. V35-shaped techniques are flagged and de-prioritised.

---

## 1. The Spectrum: Where Each Technique Sits

```
                Inference cost per call (multiplier vs current)
                          HIGH
                  10x+ │ ┌─ Tree-of-Thoughts ─┐  ┌─ Quiet-STaR ─┐
                       │ │ Graph-of-Thoughts │  │ (training)   │
                  5-10x│ │ Multi-Agent Debate│  └──────────────┘
                       │ └────────────────────┘
                  3-5x │ ┌─ Self-Consistency K=5-15 ─┐
                       │ │ Reflexion / Self-Refine   │
                  2-3x │ ┌─ Plan-Solve-Verify ─┐ ┌─ Self-derived FS ─┐
                       │ │ Step-Back Prompting │ └──────────────────┘
                  1-2x │ ┌─ Extended Thinking ────────────────────────┐
                       │ │ Constitutional-AI-at-inference             │
                       │ │  ★ 12-05 Runtime Rubric (current sweet)    │
                       │ └────────────────────────────────────────────┘
                  1x   │ ┌─ CCoT ─┐   ┌─ Codex-style heavy rules ─┐
                  LOW  ↓ └────────┘   │ (current s_linker13)      │
                       └────────────────────────────────────────────→
                       MINIMAL  ←─ Static prompt content ─→  VERBOSE
```

**Sweet-spot quadrant:** minimal static + 1-3x inference. 12-05 sits here (~1.3x judge cost). Phase 12's productive zone. Anything > 5x defeats "no LLM budget limit" tolerance at evaluation scale (5 datasets × cross-model × seeds). Lower-right is V35 territory.

---

## 2. Technique Catalog

### 1. Self-Consistency / K-sample Voting (Wang et al. 2022)
- **Citation.** Wang et al., [arXiv 2203.11171](https://arxiv.org/abs/2203.11171). `[CITED]`
- **Mechanism.** Same prompt, K samples at non-zero temperature, majority vote.
- **Token economics.** Static unchanged; **K× inference**. K=5 ≈ +$1.5 per call-site at Sonnet $15/Mtok output.
- **Applies to.** `_run_seed_validation`, `_validate_with_evidence` already cross-pass vote. Extension: intra-pass K-sample on `_run_single_extraction_pass` and `DOC_KNOWLEDGE_JUDGE_RULES`. NOT coref (V39 already 0 FP).
- **GATE-06.** Mandated-compatible.
- **Cost.** K=5 × 4 sites × 5 datasets ≈ **+$8–$15**, +3-5 min wall.
- **V35-break.** Indirect — insures variance from other trims. Pairs with 12-05 to absorb runtime-rubric drift.
- **Fit: 4/5.** Cheapest upgrade. Cross-pass voting could collapse to K=3 same-pass at equal precision.
- **Risk.** Determinism loss; need seed control.

### 2. Tree-of-Thoughts (Yao et al. 2023)
- **Citation.** Yao et al., [arXiv 2305.10601](https://arxiv.org/abs/2305.10601), NeurIPS 2023. Game of 24: 4%→74% vs CoT. `[CITED]`
- **Mechanism.** Branch K candidates per step; self-evaluate; prune; backtrack.
- **Token economics.** **5-20x**. Depth-3 × branch-3 = ~30 calls per decision.
- **Applies to.** `SEED_DISAMBIGUATION_RULES` in principle — but s_linker disambiguation space is shallow (1-3 candidates). Mismatch with the deep-search regime ToT targets.
- **GATE-06.** Mandated-compatible.
- **Cost.** Depth-2 on seed-disambig only ≈ **+$10–$30**. Unjustified.
- **V35-break.** Trivially yes, but at unjustifiable cost.
- **Fit: 2/5.** Over-engineered for shallow TLR decisions.
- **Risk.** Self-evaluator needs its own rubric — recurses into V35 one layer down.

### 3. Graph-of-Thoughts (Besta et al. 2024)
- **Citation.** Besta et al., [arXiv 2308.09687](https://arxiv.org/abs/2308.09687), AAAI 2024. +10-46pp over CoT/ToT. `[CITED]`
- **Mechanism.** Generalises ToT with Aggregation (merge K parents) and Refinement (feedback loops).
- **Token economics.** **5-15x**.
- **Applies to.** Aggregation primitive already implemented as intersect/union voting in `_run_seed_validation`. Refinement (validator rejection → re-run extraction) is the v2.2 candidate.
- **GATE-06.** Mandated-compatible.
- **Cost.** Refinement on entity pipeline ≈ **+$5–$10**.
- **V35-break.** Yes — refinement re-emits extraction with inference-derived feedback (info density grows without static surface growth).
- **Fit: 3/5.** Half already present; refinement half is a Phase 12+ candidate after 12-05 locks.
- **Risk.** Refinement can amplify wrong validator rationale.

### 4. Reflexion (Shinn et al. 2023)
- **Citation.** Shinn et al., [arXiv 2303.11366](https://arxiv.org/abs/2303.11366), NeurIPS 2023. `[CITED]`
- **Mechanism.** Generate → external evaluator → self-reflect → retry with reflection in context.
- **Token economics.** **3-5x** (3 iterations).
- **Applies to.** Needs external signal. s_linker variant: `_validate_with_evidence` rejections become the signal feeding back into `_run_single_extraction_pass`. Sidesteps the no-oracle problem.
- **GATE-06.** Mandated-compatible (reflection text is inference-derived per-input).
- **Cost.** 2-retry reflection on extraction ≈ **+$3–$8**.
- **V35-break.** Yes — reflection text regenerated per input.
- **Fit: 3/5.** Concrete v2.2 experiment as "structured Reflexion using validator-as-oracle".
- **Risk.** Accumulating errors if BBB validator is wrong (current weakness).

### 5. Self-Refine (Madaan et al. 2023)
- **Citation.** Madaan et al., [arXiv 2303.17651](https://arxiv.org/abs/2303.17651). ~20% improvement on 7 tasks. `[CITED]`
- **Mechanism.** Same LLM = Generator → Critic → Refiner across 1-3 iterations. No external oracle.
- **Token economics.** **2-4x** for 2 iterations.
- **Applies to.** `DOC_KNOWLEDGE_JUDGE_RULES` after 12-05 rubric: emit verdict → self-critique against rubric → refine. Composes cleanly. Coref Variant E could benefit. NOT seed-disambig (binary).
- **GATE-06.** Mandated-compatible.
- **Cost.** 2 iter on judge only ≈ **+$3–$6**.
- **V35-break.** Yes — critique surface generated, not written. Same mechanism as Reflexion without oracle requirement.
- **Fit: 4/5.** Lowest-risk self-correction layer on judge. **Stack on top of 12-05.**
- **Risk.** Madaan reports Self-Refine sometimes degrades correct outputs. Need "accept original if critique adds no info" guard.

### 6. Plan-Solve-Verify (Wang et al. 2023)
- **Citation.** Wang et al., [arXiv 2305.04091](https://arxiv.org/abs/2305.04091), ACL 2023. Adopted as LangChain Plan-and-Execute. `[CITED]`
- **Mechanism.** Plan → execute step-by-step → optionally verify. Decomposes monolithic prompt into 2-3 minimal calls.
- **Token economics.** **2-3x**.
- **Applies to.** `DOC_KNOWLEDGE_JUDGE_RULES`'s current auto-approve/approve/reject three-tier structure → 3 sequential calls. Architecturally extends 12-05's two-step builder.
- **GATE-06.** Mandated-compatible.
- **Cost.** P-S-V on judge ≈ **+$2–$5**.
- **V35-break.** Yes via decomposition — each sub-prompt small, combined coverage preserved.
- **Fit: 4/5.** Concrete Phase 12+ candidate; composes with 12-05.
- **Risk.** Plan-step failures cascade; need fallback to monolithic.

### 7. Multi-Agent Debate (Du et al. 2023)
- **Citation.** Du et al., [arXiv 2305.14325](https://arxiv.org/abs/2305.14325), ICML 2024. GSM8K 77%→85%. `[CITED]`
- **Mechanism.** N (typically 3) instances propose; rounds of revision against peers; converge.
- **Token economics.** **6-9x** on debated step.
- **Applies to.** `DOC_KNOWLEDGE_JUDGE_RULES`, `VALIDATION_RULES`. Precision-critical judges where Du's factuality gains transfer.
- **GATE-06.** Mandated-compatible.
- **Cost.** 3 agents × 2 rounds on judge ≈ **+$15–$30**. Expensive.
- **V35-break.** Yes — debate trace carries info density; agents' individual prompts can be minimal.
- **Fit: 3/5.** Compelling on paper, cost-heavy, harness changes needed. Slot for v2.2/v2.3.
- **Risk.** Sycophancy collapse on adversarial seeds (Du reports this).

### 8. Skeleton-of-Thought (Ning et al. 2023)
- **Citation.** Ning et al., [arXiv 2307.15337](https://arxiv.org/abs/2307.15337). Up to 2.39x latency. `[CITED]`
- **Mechanism.** Emit skeleton (bullet headers) → fill bullets in parallel.
- **Token economics.** Same total tokens, **~2x latency win**.
- **Applies to.** Already partially present via `_run_parallel`.
- **GATE-06.** Mandated-compatible.
- **Cost.** Zero $ change; latency only.
- **V35-break.** No — latency trick, not info-density mechanism.
- **Fit: 2/5.** Latency not the bottleneck per CLAUDE.md "no budget limit". Already adequately implemented.
- **Risk.** Inter-bullet coherence loss; coref-style prompts cannot use SoT.

### 9. Step-Back Prompting (Zheng et al. 2023, DeepMind)
- **Citation.** Zheng et al., [arXiv 2310.06117](https://arxiv.org/abs/2310.06117). PaLM-2L MMLU Physics +7pp, TimeQA +27pp. `[CITED]`
- **Mechanism.** Before answering concrete question, ask for high-level principle; then apply.
- **Token economics.** **~2x** (abstract call + concrete call).
- **Applies to.** Strong fit for `SEED_DISAMBIGUATION_RULES` and `DOC_KNOWLEDGE_JUDGE_RULES`. Natural composition with 12-05: step-back generates the principle, rubric-builder grounds it in document patterns.
- **GATE-06.** Mandated-compatible (step-back question is generic; answer is inference-time).
- **Cost.** Step-back on judge ≈ **+$2–$4**.
- **V35-break.** Yes — principle regenerated per document.
- **Fit: 4/5.** Cheap, theoretically motivated, composes with 12-05.
- **Risk.** Principle-gen step itself needs GATE-06 audit (same as rubric-builder).

### 10. Constitutional AI at Inference (Bai et al. 2022)
- **Citation.** Bai et al., [arXiv 2212.08073](https://arxiv.org/abs/2212.08073). `[CITED]` Inference-time variants in Anthropic prompt guides.
- **Mechanism.** Apply a 3-4 principle "constitution" via self-critique loop. Same shape as Self-Refine + explicit principle anchors.
- **Token economics.** **2-3x**.
- **Applies to.** 3-4 universal principles ("favor explicit reference over implicit", "single-word names are ambiguous unless evidence specifies") replacing longer judge rule list, applied via critique.
- **GATE-06.** Mandated-compatible IF constitution is abstract.
- **Cost.** **+$2–$5**.
- **V35-break.** Partial. Constitution IS static; application is inference. Halfway position between V35 and 12-05.
- **Fit: 3/5.** Worth testing as "Self-Refine with principle scaffolding". Lower priority than 12-05 stack.
- **Risk.** Short constitution is V35-shaped if too coarse.

### 11. Quiet-STaR (Zelikman et al. 2024)
- **Citation.** Zelikman et al., [arXiv 2403.09629](https://arxiv.org/abs/2403.09629). `[CITED]`
- **Mechanism.** Training-time. LM learns to emit hidden rationales per token. Not API-accessible.
- **Applies to.** None directly. Extended Thinking (Technique 12) is the productised analogue.
- **Fit: 1/5.** Listed for completeness; not actionable on closed APIs.

### 12. Extended Thinking (Anthropic Native)
- **Citation.** [Anthropic Extended Thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking); [Effort docs](https://platform.claude.com/docs/en/build-with-claude/effort). `[CITED]`
- **Mechanism.** Claude allocates hidden deliberation tokens via `effort` param (formerly `budget_tokens`). Sonnet 4.6+ supports adaptive thinking.
- **Token economics.** Thinking tokens billed as output ($15/Mtok at Sonnet 4.6 `[CITED: Anthropic pricing 2026]`). Anthropic guidance: 2-5x visible output for thinking-heavy tasks; **skip for classification/extraction**.
- **Applies to.** Judge-class prompts where error-cost > 5x call-cost: `DOC_KNOWLEDGE_JUDGE_RULES`, `VALIDATION_RULES`, `SEED_DISAMBIGUATION_RULES`. NOT simple extraction.
- **GATE-06.** Mandated-compatible. Backend parameter.
- **Cost.** Medium effort on 3 judge sites ≈ **+$5–$10**, negligible wall-clock (parallelisable).
- **V35-break.** **Strongly** — moves "weighing" from surface to hidden tokens. Answers Phase 11 §6's "highest-leverage empirical question".
- **Fit: 5/5.** Cheapest backend-only experiment. **Test before any prompt restructuring** — gives baseline for what hidden deliberation alone unlocks.
- **Risk.** Asymmetric to gpt-5.4 (different reasoning mechanism). Cross-model gate complication.

### 13. Self-Derived Few-Shot (Auto-CoT / Analogical Prompting)
- **Citation.** Zhang et al. (Auto-CoT) [arXiv 2210.03493](https://arxiv.org/abs/2210.03493); Yasunaga et al. (Analogical Prompting) [arXiv 2310.01714](https://arxiv.org/abs/2310.01714). `[CITED]`
- **Mechanism.** Before answering target, LLM generates K relevant exemplars for current input, uses as few-shot.
- **Token economics.** **2x** (exemplar-gen + answer).
- **Applies to.** The 7 worked examples in `DOC_KNOWLEDGE_JUDGE_EXAMPLES` could be replaced by inference-time generator that produces 3-4 exemplars *from generic SE textbook contexts* (compiler stages, schedulers). Pairs with `AMBIGUITY_FEW_SHOT`.
- **GATE-06.** Mandated-compatible IF exemplar generator is seeded with generic SE patterns, NOT given the architecture document. Different from 12-05's doc-grounded rubric.
- **Cost.** **+$2–$4**.
- **V35-break.** Yes — examples regenerated, not deleted. Direct escape from V35a's "example-driven simplification regresses" failure mode.
- **Fit: 4/5.** Concrete Phase 12+ candidate.
- **Risk.** Generated exemplars may bias toward training-data patterns. Needs taboo audit on the generator output.

### 14. Recursive Prompting / Least-to-Most (Zhou et al. 2022)
- **Citation.** Zhou et al., [arXiv 2205.10625](https://arxiv.org/abs/2205.10625). `[CITED]`
- **Mechanism.** Decompose hard problem into easier sub-problems; recursively solve; combine.
- **Token economics.** Depth-dependent. Depth-2 ≈ 2-3x.
- **Applies to.** Partial-name matching reformulation. Overkill for s_linker's shallow decisions.
- **Fit: 2/5.** Reserve for deep coref chains (not yet in scope).

### 15. Retrieval-Augmented Reasoning at Inference (per-call)
- **Citation.** Lewis et al., [arXiv 2005.11401](https://arxiv.org/abs/2005.11401); RAG survey [arXiv 2312.10997](https://arxiv.org/abs/2312.10997). `[CITED]`
- **Mechanism.** Per-call retrieval of K chunks from per-document index. Index IS the doc, no external corpus.
- **Applies to.** `_validate_with_evidence` already implements this. Extension: per-candidate retrieval into `DOC_KNOWLEDGE_JUDGE_RULES` (currently full-doc context).
- **GATE-06.** **Mandated** — retrieval IS the runtime doc-inference mechanism.
- **Fit: N/A (already adopted).** Listed for completeness.

---

## 3. Combinatorial Synergies

Three high-leverage compositions:

**(A) Runtime Rubric + Self-Refine + Extended Thinking** ("the full 12-05 stack"). Rubric generated per-doc (12-05). Judge uses extended thinking to weigh rubric (T12). Judge then self-critiques against rubric and emits revised verdict (T5). Total ≈ 3-4x on judge prompt only. Each layer adds a different info-density axis: doc-specific principles, hidden deliberation, explicit critique. **Recommended Phase 12+ experiment.**

**(B) Self-Consistency K=3-5 over any other technique.** Orthogonal — sits OVER any base structure. Absorbs variance introduced by the runtime-rubric mechanism (inherently sample-to-sample variant). Cheapest insurance against borderline GATE-01 failures. **Recommended as fallback if 12-05 stack introduces variance.**

**(C) Step-Back + Runtime Rubric.** Step-back generates the principle ("what high-level rule distinguishes a true alias from a coincidence?"); rubric-builder grounds it in document patterns. Two-stage rubric generation aids GATE-06 auditability — principle stage is doc-blind, grounding stage is doc-grounded. **Recommended for v2.2 defensibility narrative.**

Non-synergies: ToT + GoT + Debate over the same prompt site is redundant and cost-explosive. SoT conflicts with Self-Refine (one parallelises, the other serialises critique).

---

## 4. The "Massive Runtime Knowledge Inference" Pattern (User's Framing)

The pattern: **the model is the knowledge store; the prompt is the access protocol**. A V35-shaped move shortens the protocol AND removes content, losing knowledge. A 12-05-shaped move shortens the static protocol while triggering more inference-time content generation — net knowledge in the call grows.

For s_linker13 this means three things:

1. **Static prompts should encode procedures, not content.** "Emit a rubric for this document, then judge against it" carries no domain content but invokes the model's internal knowledge of how docs are structured. "Approve if alias is one of {component names}" carries domain content and is V35-shaped.
2. **GATE-06 (corrected reading) is aligned with this pattern, not opposed to it.** The CLAUDE.md mandate — "doc-specific knowledge must be discovered dynamically at runtime via LLM analysis" — IS the maximal-runtime-inference principle stated as a project rule. 12-05 is the canonical execution. The original REJECT confused "the rubric named project components" (which is what runtime discovery produces) with "the static prompt body contained benchmark vocabulary" (which it did not).
3. **Info density is conserved across the prompt-call boundary.** What V35 lost by deleting static rules can be recovered by: (a) generating rules at inference (12-05), (b) deliberating in hidden tokens (Extended Thinking), (c) sampling+voting (Self-Consistency), (d) decomposing across calls (P-S-V, Self-Refine). The four conservation channels.

The frontier for v2.2: characterise which prompts can move static content to each channel without F1 regression, at what total dollar/latency cost.

---

## 5. Top 3 Recommendations for Phase 12 / v2.2

| Rank | Technique | Target prompt(s) | Cost (per 5-dataset eval) | Expected F1 | Rule-count reduction | Rationale |
|------|-----------|------------------|---------------------------|-------------|----------------------|-----------|
| **1** | **Extended Thinking on judge sites** (T12) | `DOC_KNOWLEDGE_JUDGE_RULES`, `VALIDATION_RULES`, `SEED_DISAMBIGUATION_RULES` | +$5–$10, +0-1 min | -1 to +1pp vs 12-05 | 3-5 surface "favor X / when uncertain" tie-breakers movable to hidden tokens | Cheapest, lowest-risk, backend-only. Answers Phase 11 §6 "highest-leverage empirical question". Single-variable experiment. |
| **2** | **Self-Refine on top of 12-05 runtime rubric** (T5 + 12-05) | `DOC_KNOWLEDGE_JUDGE_RULES` (judge stage after rubric) | +$3–$6, +1-2 min | +0 to +0.5pp on BBB | Same as 12-05 (already minimal); adds critique cycle | 12-05 proved rubric works at F1 level. Self-Refine adds explicit critique surface that V35a-style "example-driven regression" lacks. Direct V35-ceiling escape. |
| **3** | **Self-Consistency K=3 on extraction + judge** (T1) | `_run_single_extraction_pass`, `DOC_KNOWLEDGE_JUDGE_RULES` | +$8–$15, +2-4 min | ±0.5pp (variance insurance, not ceiling raiser) | Zero direct; enables variance-absorbing trims | Insurance for any aggressive trim near GATE-01 floor. Rides on top of (1) and (2), not primary. |

**Sequence:** Run (1) as single-variable ablation against current `s_linker13_clean` baseline. If F1 ≥ baseline with 3 tie-breakers removed, lock and proceed to (2) layered on top. If (2) ≥ 12-05 stack F1 on BBB, lock. (3) reserved as variance-recovery only if (1)+(2) drops below GATE-01 on cross-model.

**Out-of-priority** (v2.2 only): Step-Back + Runtime Rubric (composition C, defensibility win); P-S-V on judge (architecturally redundant with Self-Refine — pick one); Reflexion-with-validator-as-oracle (interesting but BBB validator precision too low to bootstrap).

---

## 6. Negative Findings / Caveats

**Won't transfer to TLR:**
- **ToT / GoT deep search** — TLR ambiguity is shallow (1-3 candidates); exploration cost unjustifiable. ToT shines on Game-of-24-shaped problems s_linker doesn't have.
- **Multi-Agent Debate** — ~6-9x cost hard to justify for marginal precision over Self-Refine's single-agent critique. Defer to v2.3+.
- **Quiet-STaR** — training-time, no API access. Extended Thinking is productised analogue.
- **Skeleton-of-Thought** — latency win only, no info-density mechanism. Already adequately adopted via `_run_parallel`.
- **Constitutional AI with universal principles** — short principle list IS static; likely strictly dominated by Self-Refine + runtime rubric.
- **Recursive prompting** — depth-2 s_linker decisions don't justify recursion overhead.

**Erdős regime is not a template for TLR** (per supplement §1): TLR is many-decision-unverifiable.

**Cross-model asymmetry caveat.** Extended Thinking (Rec #1) is asymmetric — Claude has backend param; gpt-5.4 has different reasoning. Any experiment using Extended Thinking must validate gpt-5.4 arm separately, possibly needs backend-adaptive prompt (Phase 11 supplement T1) to compensate. Known gate-complication, not blocker.

**Open empirical question for v2.2:** Does the composition of Extended Thinking + Runtime Rubric + Self-Refine exhibit diminishing returns, or do the three info-density channels independently contribute? No published study covers this combination on a classification benchmark. Phase 12+ ablation produces the first data point.

---

## 7. Sources

### Primary (HIGH confidence — peer-reviewed / vendor docs)
- Wang et al., *Self-Consistency*, [arXiv 2203.11171](https://arxiv.org/abs/2203.11171) — T1.
- Yao et al., *Tree of Thoughts*, [arXiv 2305.10601](https://arxiv.org/abs/2305.10601), NeurIPS 2023 — T2.
- Besta et al., *Graph of Thoughts*, [arXiv 2308.09687](https://arxiv.org/abs/2308.09687), AAAI 2024 — T3.
- Shinn et al., *Reflexion*, [arXiv 2303.11366](https://arxiv.org/abs/2303.11366), NeurIPS 2023 — T4.
- Madaan et al., *Self-Refine*, [arXiv 2303.17651](https://arxiv.org/abs/2303.17651) — T5.
- Wang et al., *Plan-and-Solve*, [arXiv 2305.04091](https://arxiv.org/abs/2305.04091), ACL 2023 — T6.
- Du et al., *Multiagent Debate*, [arXiv 2305.14325](https://arxiv.org/abs/2305.14325), ICML 2024 — T7.
- Ning et al., *Skeleton-of-Thought*, [arXiv 2307.15337](https://arxiv.org/abs/2307.15337) — T8.
- Zheng et al., *Step-Back Prompting*, [arXiv 2310.06117](https://arxiv.org/abs/2310.06117), DeepMind — T9.
- Bai et al., *Constitutional AI*, [arXiv 2212.08073](https://arxiv.org/abs/2212.08073) — T10.
- Zelikman et al., *Quiet-STaR*, [arXiv 2403.09629](https://arxiv.org/abs/2403.09629) — T11.
- [Anthropic Extended Thinking docs](https://platform.claude.com/docs/en/build-with-claude/extended-thinking); [Effort docs](https://platform.claude.com/docs/en/build-with-claude/effort); [Pricing](https://platform.claude.com/docs/en/about-claude/pricing) — T12 + costs.
- Zhang et al. (Auto-CoT), [arXiv 2210.03493](https://arxiv.org/abs/2210.03493); Yasunaga et al. (Analogical), [arXiv 2310.01714](https://arxiv.org/abs/2310.01714) — T13.
- Zhou et al., *Least-to-Most*, [arXiv 2205.10625](https://arxiv.org/abs/2205.10625) — T14.
- Lewis et al., *RAG*, [arXiv 2005.11401](https://arxiv.org/abs/2005.11401); RAG survey [arXiv 2312.10997](https://arxiv.org/abs/2312.10997) — T15.

### Project / empirical
- `.planning/phases/12-trim-ablation/12-05-SUMMARY.md` — runtime-rubric F1 evidence + audit. **Anchor for §0/§4.**
- `.planning/phases/12-trim-ablation/12-04-SUMMARY.md` — ent+val merge BBB regression.
- `.planning/research/PROMPT-HARNESS-SURVEY.md` (Phase 11 main) — V35-ceiling framing.
- `.planning/research/PROMPT-HARNESS-SURVEY-SUPPLEMENT-ERDOS.md` — AHE / Agentic Rubrics for 12-05 mechanism.
- `MEMORY.md` V35 entries — V35a/b/c regression numbers.
- `/mnt/hostshare/ardoco-home/CLAUDE.md` "LLM Linker Development Rules" — corrected GATE-06 reading.

### Secondary (MEDIUM confidence)
- [Microsoft Research SoT blog](https://www.microsoft.com/en-us/research/blog/skeleton-of-thought-parallel-decoding-speeds-up-and-improves-llm-output/) — T8 latency.
- [MIT News on Multi-Agent Debate](https://news.mit.edu/2023/multi-ai-collaboration-helps-reasoning-factual-accuracy-language-models-0918) — T7.
- [Prompting Guide — Reflexion](https://www.promptingguide.ai/techniques/reflexion) — T4.
- [CloudZero Claude API Pricing 2026](https://www.cloudzero.com/blog/claude-api-pricing/) — pricing cross-ref.

### Confidence breakdown
- Technique mechanisms: **HIGH** (all primary arXiv verified).
- Cost estimates: **MEDIUM** (Sonnet 4.6 pricing verified; per-call multipliers extrapolated).
- Fit scores: **MEDIUM** (theory-grounded from §0 anchor; Phase 12+ ablation will produce hard data).
- Synergies §3: **MEDIUM-LOW** (no published study covers specific compositions).
- Top-3 ranking §5: **MEDIUM** (priority grounded in 12-05 empirical + Phase 11 V35-ceiling theory).

**Research date:** 2026-05-31
**Valid until:** 2026-06-30 (techniques stable; pricing may shift).
