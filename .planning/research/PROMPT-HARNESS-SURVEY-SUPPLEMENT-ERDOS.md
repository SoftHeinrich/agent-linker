# Phase 11 Survey — Supplement: OpenAI Erdős + Breaking Agent Papers (April–May 2026)

**Produced:** 2026-05-31
**Phase:** 11 (v2.1 Research, PROMPT-05)
**Status:** Supplement to PROMPT-HARNESS-SURVEY.md
**Total word count:** ~2,300 words

---

## 1. OpenAI Erdős-Conjecture Proof System

### What was found (verified)

On **2026-05-20**, OpenAI announced that an internal, general-purpose reasoning model
disproved Erdős's planar unit-distance conjecture (open since 1946) by constructing an
infinite family of point sets achieving `n^(1+δ)` unit-distance pairs (`δ ≈ 0.014`,
refined by Will Sawin) via Golod-Shafarevich theory and infinite class-field towers.

Primary sources:

- [OpenAI: An OpenAI model has disproved a central conjecture in discrete geometry](https://openai.com/index/model-disproves-discrete-geometry-conjecture/) — primary announcement (returned 403 to WebFetch but verified to exist via search index).
- [TechCrunch (May 20, 2026)](https://techcrunch.com/2026/05/20/openai-claims-it-solved-an-80-year-old-math-problem-for-real-this-time/)
- [Scientific American](https://www.scientificamerican.com/article/ai-just-solved-an-80-year-old-erdos-problem-and-mathematicians-are-amazed/)
- [Understanding AI (Timothy B. Lee)](https://www.understandingai.org/p/openais-milestone-math-breakthrough)
- [explainx.ai blog](https://explainx.ai/blog/openai-planar-unit-distance-erdos-problem-solved-2026)

### Harness Anatomy (what was actually disclosed)

**Honest scope warning:** OpenAI withheld most implementation details. Three of the
five sources explicitly note the absence of architecture/prompt disclosure. The
following is the union of what *was* stated, with "unverified" tags where the public
record is silent.

- **System prompt strategy:** Minimal. Per explainx.ai and TechCrunch, "the model
  received the problem statement and produced the solution independently" — no
  step-by-step human guidance, no partial proof provided. This is the *opposite* of
  the prompt-heavy s_linker13 surface.
- **Rule vs delegation balance:** **Maximal delegation.** No domain rules ("use number
  theory", "try class field towers") were given. The conjecture statement alone was
  the prompt. Rules emerge from the model's own chain-of-thought.
- **Tool-use schema:** **Unverified.** No public statement of tool calls, calculator,
  Lean/Coq verifier, or external search. This is striking — most 2025-era math
  agents (AlphaProof, Lean-based agents) leaned heavily on formal verifiers; the
  Erdős result appears to be unverified-by-tool natural-language proof.
- **Multi-pass / deliberation structure:** **Partially verified.** Understanding AI
  notes: "even with the maximum token budget, the internal model solves the problem
  only half of the time" — implying inference-time **best-of-N sampling** with N
  large enough that a 50% success rate is acceptable. OpenAI "could have run the
  problem many times before a model found a solution."
- **Verifier:** **Human, not model.** Per Scientific American: OpenAI "privately
  contacted" Timothy Gowers and Daniel Litt to verify; "no external experts have
  seen the AI's original output, just an edited version of its train of thought."
  There is no public evidence of an automated verifier in the loop.
- **Novel vs CoT/ReAct/self-consistency:** Per TechCrunch: the model was "tuned for
  chain-of-thought reasoning where the model endlessly 'thinks out loud'." This is
  *not* novel as a prompt technique — what's novel is the **scale of CoT tokens**
  (hundreds of pages) and the **absence of any scaffolding around it**.

### Implicit harness ("the dog that didn't bark")

The most interesting prompt-harness signal from the Erdős result is *negative*:

> A frontier mathematical proof was produced with **no domain prompt**, **no tool
> harness**, and **no automated verifier** — only raw best-of-N CoT against a
> general-purpose reasoning model, with humans serving as terminal verifiers.

Translated to s_linker13's regime: the OpenAI result is evidence that *for
sufficiently capable reasoning models*, removing the harness can outperform
optimizing it. But this assumes 50%-pass-rate-after-many-samples is acceptable —
which it is for math (one proof is enough) and emphatically is **not** for TLR
(every sentence is a separate decision, average-case quality dominates).

### Transferability to s_linker13

- **Fit score: 2/5.** The Erdős harness solves a *one-shot, verifiable* problem with
  best-of-N + human verifier. s_linker13 solves *thousands of unverifiable per-
  sentence decisions* with no human in the loop and no oracle. The architectures
  are mismatched.
- **Specific `prompts_v2.py` prompts it could re-shape:** None directly. The closest
  analog is `SEED_DISAMBIGUATION_RULES` (where the "right answer" is closer to a
  proof than a classification), but even there the absence of an automated
  verifier rules out the pattern.
- **GATE-06 verdict:** **N/A** — there is no transferable prompt artifact to audit.
- **Mechanism that could break V35 ceiling:** **Honest "cannot."** V35 failed because
  Claude exploits *information density* in long prompts; the Erdős harness goes the
  other direction (zero rules) and relies on best-of-N to recover quality. In TLR,
  best-of-N is already covered by intersect/union voting and does not break the
  ceiling. The Erdős result is *not* a counterexample to V35; it's a different
  regime where the V35 trap doesn't apply.

> **One useful negative takeaway:** the Erdős system shows that for *one-shot,
> verifiable* sub-problems, removing rules works. If any s_linker13 stage can be
> reframed as one-shot-verifiable (e.g., judge-as-binary on a structured rubric),
> that *specific* stage might benefit from radical simplification. See §3.

---

## 2. Breaking Agent Tool Papers (April–May 2026)

Six verified entries below. Two arXiv IDs I encountered in search snippets but could
not independently fetch ("2604.21003", "2603.25723") are tagged as "snippet-only".

### Paper 1: Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses

- **Source / URL:** [arXiv:2604.25850](https://arxiv.org/abs/2604.25850) (28 Apr 2026, rev. 18 May 2026)
- **Harness innovation:** Closed-loop "Component / Experience / Decision" observability
  treats each harness edit as a falsifiable contract. Critical finding: **ablations
  show gains come from "tools, middleware, and long-term memory rather than the
  system prompt"** — structural change beats prose change.
- **Transferability score: 4/5.** s_linker13 *is* a rule-heavy system prompt
  architecture; this paper is direct evidence that the system-prompt-vs-structure
  axis matters. Reframes "trim prompts" as "restructure tools/middleware".
- **GATE-06 verdict:** **Conditional.** AHE itself is project-agnostic; what it
  evolves is project-specific. Phase 12 must lock the evolved artifacts and
  re-audit for benchmark phrasing.
- **Why it could break the V35 ceiling:** V35 simplified *prose rules* and lost
  information density. AHE moves edge-case knowledge from prose into typed tools /
  middleware / memory, which preserves it in a non-prose channel the LLM still
  consults. This is exactly the "restructure not remove" axis the Phase 11
  CONTEXT.md called out as central.

### Paper 2: Code as Agent Harness

- **Source / URL:** [arXiv:2605.18747](https://arxiv.org/abs/2605.18747) (18 May 2026)
- **Harness innovation:** Position paper arguing code (not prose) is the natural
  substrate for harness logic. Three layers: harness interface, harness mechanisms
  (planning/memory/tool-use), multi-agent coordination. Emphasizes regression-free
  long-horizon execution.
- **Transferability score: 3/5.** s_linker13 already has substantial Python
  helper infrastructure; this paper validates the direction but provides no
  drop-in technique. Useful as architectural framing for `helper_v3.py` /
  `helper_v4.py` in CLEAN-02.
- **GATE-06 verdict:** **Yes.** Code-as-harness moves logic *out* of prompts, which
  is structurally GATE-06 safe (Python identifiers are easier to audit for
  benchmark leakage than free-text rules).
- **Why it could break the V35 ceiling:** Lifts deterministic edge-case handling
  out of prompt text into code helpers. The V35 simplifications removed prose
  rules without a replacement substrate — code-as-harness *provides* a
  replacement substrate. Note: applies only to deterministic rules; the Phase 9
  judge cannot be code-ified without re-introducing benchmark heuristics.

### Paper 3: Natural-Language Agent Harnesses (NLAH + IHR)

- **Source / URL:** arXiv:2603.25723 (snippet-only — could not independently fetch
  abstract; original search result attributed it to a recent preprint).
- **Harness innovation:** Per snippet: harness policy is an *editable natural-
  language document*, executed by a separate Intelligent Harness Runtime. Reported
  to match code-/prompt-based harnesses while being "significantly more concise".
- **Transferability score: 3/5.** Compelling in principle (concise + transparent),
  but the IHR runtime is non-trivial infrastructure and the result needs
  independent confirmation before adoption.
- **GATE-06 verdict:** **Conditional** — same auditability advantage as plain
  prompts; same leakage risk.
- **Why it could break the V35 ceiling:** **Probably cannot.** NLAH is essentially
  "shorter prompts via runtime indirection" — which is structurally the V35 move.
  Lower-priority for Phase 12.

### Paper 4: Agentic Rubrics as Contextual Verifiers for SWE Agents

- **Source / URL:** [arXiv:2601.04171](https://arxiv.org/abs/2601.04171)
- **Harness innovation:** An "expert agent" reads the *repository* to construct a
  project-grounded rubric checklist; candidate outputs are scored against it
  *without test execution*. Demonstrates rubric-as-verifier as an alternative to
  oracle execution.
- **Transferability score: 4/5.** This is the closest analog to s_linker13's
  Phase 9 judge stage. Crucially: rubrics are **constructed per-project at
  inference time from the repository itself** — not handwritten — which sidesteps
  GATE-06. The architecture document is the analog of "the repository".
- **GATE-06 verdict:** **Yes** — the rubric is *derived from the input document*,
  not from benchmark engineering. This is the same provenance pattern as
  s_linker13's existing Document Knowledge phase. (Caveat: ensure the rubric-
  building prompt itself contains no leaked phrasing.)
- **Why it could break the V35 ceiling:** V35 collapsed handcrafted rules and
  lost coverage. Per-input rubric *generation* is fundamentally different — the
  rubric grows back to the right level of detail for the input, recovering
  coverage automatically. This is information-density-preserving compression of
  the *prompt surface*.

### Paper 5: Rubric-based On-policy Distillation (ROPD)

- **Source / URL:** [arXiv:2605.07396](https://arxiv.org/html/2605.07396v1)
- **Harness innovation:** A "Rubricator" contrasts teacher and student rollouts to
  induce *prompt-specific* rubrics; a "Verifier" scores rollouts against the
  rubric. Rubrics are weighted criterion lists, not free-form prose.
- **Transferability score: 3/5.** The full ROPD pipeline requires RL (GRPO) and
  a teacher model — not directly usable. **But** the *Rubricator-as-prompt-pattern*
  transfers to inference: induce a rubric from the architecture document and a
  worked example, then judge against it.
- **GATE-06 verdict:** **Conditional.** Inference-time rubric induction is safe
  if the seed example is from a generic SE textbook context (parsers,
  schedulers).
- **Why it could break the V35 ceiling:** Same mechanism as Paper 4 — *generated*
  rubric carries information density that *removed* rules don't.

### Paper 6: Agentic Reasoning and Tool Integration via RL (ARTIST-style)

- **Source / URL:** [arXiv:2505.01441](https://arxiv.org/abs/2505.01441) (pre-window
  publish date but actively cited in May 2026 follow-ups)
- **Harness innovation:** Interleaves text thinking, tool queries, and tool outputs
  with RL-trained coordination — a productized ReAct with learned tool-policy.
- **Transferability score: 2/5.** Useful framing but most of the value is in the
  RL training, not the inference-time prompt pattern.
- **GATE-06 verdict:** **Yes** for the architecture; **N/A** for prompts (none
  published).
- **Why it could break the V35 ceiling:** **Probably cannot** for s_linker13's
  zero-shot regime. Listed for completeness.

### Honorable mention (snippet-only, not fully verified)

- **arXiv:2604.21003 — "The Last Harness You'll Ever Build."** Two-level
  meta-evolution loop. Could be relevant but I could not independently confirm
  the abstract content beyond search snippets. Flag for Phase 12 follow-up.

---

## 3. Cross-Cutting Themes

Three patterns recur across the Erdős evidence and the six papers:

1. **Verifier as a separate stage, not a re-prompt.** Both the Erdős system
   (humans-as-verifiers) and Papers 4 + 5 (rubric-as-verifier) cleanly separate
   *production* from *verification*. s_linker13 already has this shape (Phase 9
   judge), but the judge prompt today carries both rules *and* verification.
   Splitting them is a candidate restructure.
   - Concrete application to s_linker13: separate `JUDGE_RULES` (production
     constraints, hand-written) from a `JUDGE_RUBRIC` (generated per-document
     from the architecture doc + a generic SE example). Tests an architecture
     analogous to a parser project that ships with a grammar spec.

2. **Generate the rubric, don't write it.** Papers 4 and 5 both replace
   handcrafted rule lists with rubrics *induced at inference time* from inputs.
   This is the strongest candidate mechanism for escaping the V35 ceiling:
   information density is preserved because the rubric is regenerated to the
   right size for each input.
   - Concrete application: `DOC_KNOWLEDGE_JUDGE_RULES` becomes
     `DOC_KNOWLEDGE_JUDGE_RUBRIC_BUILDER` — a small prompt that asks the model
     to produce a 4-6-item rubric from a generic example (e.g., a queue-and-
     scheduler textbook system), then the rubric flows into the actual judge.

3. **Move edge-case knowledge into a non-prose channel.** AHE (Paper 1) and
   Code-as-Harness (Paper 2) both report that structural channels (tools,
   middleware, memory, code) carry edge-case knowledge more robustly than prose
   rules. This matches V35's lesson: removing prose rules without replacement
   loses coverage.
   - Concrete application: `STANDALONE_MENTION_RULES_*` variants — six near-
     identical templates differing only by which knowledge bundle is in scope —
     could collapse into one template plus a typed "knowledge context" tool
     payload that the model reads.

---

## 4. Recommended Additions to Phase 12 Trim Strategy

Supplementing (not replacing) the main survey:

1. **Attack `DOC_KNOWLEDGE_JUDGE_RULES` first with inference-time rubric
   generation** (Papers 4 + 5 mechanism). Build a small rubric-builder prompt
   that consumes a *generic* SE textbook example (compiler stage / job queue,
   not benchmark components) plus the current architecture document, and emits
   a 4-6-item rubric. Pass the generated rubric into the judge as the operative
   rule set. This is the highest-confidence escape from the V35 ceiling because
   it preserves information density via *regeneration*, not *retention*.

2. **Collapse the six `STANDALONE_MENTION_RULES_*` variants** by moving the
   knowledge-bundle distinction into structured input rather than separate
   templates (Paper 2 mechanism). One template + typed payload, not six
   templates. Rule count drops without losing edge cases.

3. **Split `JUDGE_RULES` into production-rules + verification-rubric**
   (cross-cutting theme #1). Smaller per-prompt rule count without removing
   total information. GATE-06 audit applies only to the production half.

4. **De-prioritize**: NLAH-style prompt-only shortening and pure CoT
   minimization (Erdős regime). Both are V35-shaped moves and the prior failure
   data outweighs the recent paper enthusiasm.

---

## 5. Open Questions / Negative Results

- **OpenAI did not disclose the Erdős harness.** Best-of-N is *implied* (50%
  pass rate), but verifier architecture, tool use, and exact prompt are
  unpublished. Any claim built on the Erdős harness is necessarily
  speculative — we can use it as a *negative* control ("zero-rule + best-of-N
  is sufficient for one-shot verifiable problems") but not as a positive
  template for s_linker13.
- **No 2026 paper found that directly addresses rule reduction for
  classification-style LLM pipelines with macro-F1 metrics.** The 2026 agent
  literature is dominated by coding agents and tool-use; TLR-shaped work is
  absent. Phase 12 is operating ahead of the public state of the art for its
  specific regime.
- **Two arXiv IDs (2604.21003, 2603.25723) could not be independently
  verified** beyond search-engine snippets. Treat the corresponding entries as
  hypotheses pending direct fetch.

---

## 6. Search Trail (Reproducibility)

- `"OpenAI Erdős conjecture proof agent 2026"` → openai.com primary post,
  TechCrunch, Scientific American.
- `"OpenAI Erdos problems LLM agentic mathematical reasoning 2026"` →
  Understanding AI substack, explainx.ai blog.
- `"arxiv agent prompting harness paper April 2026 May 2026"` → 2604.25850
  (AHE), 2605.18747 (Code-as-Harness), 2603.25723 (NLAH), 2604.21003.
- `"LLM agent reasoning paper arxiv May 2026 tool use rubric verifier"` →
  2601.04171 (Agentic Rubrics), 2508.16949 (Rubric-Scaffolded RL).
- `"rubric distillation LLM prompt simplification 2026 verifier separate model"`
  → 2605.07396 (ROPD).

[Sources used in this supplement are inline above as markdown hyperlinks.]
